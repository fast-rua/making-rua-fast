"""Where an episode's time went: profile.json and a short summary for a real-cell agent run (read-only).

    python scripts/profile_run.py ~/robot-cell/agent-runs/towel-fold-11 [...]

From the run's Codex rollouts (every thread, pruned ones too), the app-server's event stream and robotd's request
log: each model response (its request to its last item), what its tool calls were for, the robot motion during each,
images viewed, prunes. robotd's requests.jsonl stamps a request when its reply is written: it ran over
[wall - dur_s, wall]. The split of model time is an estimate: decode at DECODE_TOK_S, the rest is waiting for the
first token (upload, queue, prefill, the images in the context). DECODE_TOK_S was measured on 2026-11-07 for
GPT-6 Astra on the ultrafast tier; set it for your model.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from datetime import datetime

DECODE_TOK_S = 282.0


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _rollouts(run: str) -> list:
    return sorted(glob.glob(os.path.join(run, "codex", "home", "sessions", "**", "rollout-*.jsonl"), recursive=True),
                  key=os.path.getmtime)


def _motions(run: str, t0: float, t1: float) -> list:
    out = []
    p = os.path.join(run, "robotd", "requests.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (e.get("req") or {}).get("op") == "command" and t0 - 60 <= e.get("wall", 0) <= t1 + 60:
            out.append((e["wall"] - (e.get("dur_s") or 0.0), e["wall"]))
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _overlap(M, a, b) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in M)


def _kind(calls: list) -> str:
    txt = " ".join(c["input"] for c in calls)
    used = set()
    for c in calls:
        used |= set(re.findall(r"tools\.(\w+)", c["input"])) if c["type"] == "custom_tool_call" else {c["name"]}
    if used and used <= {"view_image"}:
        return "view images only"
    if used and used <= {"write_stdin", "wait"}:
        return "wait on a running command"
    if re.search(r"\.move\(|\.gripper\(|move_at\(|robot run", txt):
        return "move the robot"
    if re.search(r"capture\(|review\(|\.cameras\(|Overlay|View3D|viz\.show|show_cameras|marks\(", txt):
        return "capture or render views"
    if re.search(r"point\(|frames\.pkl|\.ik\(|reachable\(|dry_run", txt):
        return "look up 3D points or IK"
    if re.search(r"\.state\(\)|status\.json", txt):
        return "read robot state"
    if re.search(r"\b(cat|sed|grep|rg|head|tail|ls|find|wc)\b", txt) and "python" not in txt:
        return "read files"
    return "other"


def profile(run: str) -> dict:
    meta = json.load(open(os.path.join(run, "meta.json")))
    responses, calls_by_id = [], {}
    for path in _rollouts(run):
        lines = [json.loads(l) for l in open(path, errors="replace") if l.strip()]
        started = False
        cur, pending_in, last_done = None, None, None
        for e in lines:
            p, t = e.get("payload") or {}, _ts(e["timestamp"]) if e.get("timestamp") else None
            typ = p.get("type")
            if e.get("type") == "event_msg" and typ == "task_started":
                started, last_done = True, t
                continue
            if not started:                       # the opening context, and a pruned thread's injected history
                continue
            if e.get("type") == "response_item" and typ in ("custom_tool_call_output", "function_call_output"):
                c = calls_by_id.get(p.get("call_id"))
                if c is not None:
                    c["t_out"] = t
                    o = p.get("output")
                    c["images"] = sum(1 for x in o if isinstance(x, dict) and x.get("type") == "input_image") \
                        if isinstance(o, list) else 0
                    c["error"] = bool(re.search(r"Traceback|Error\b|Exception|exit_code\":[1-9]",
                                                json.dumps(o)[:20000]))
                pending_in = max(pending_in or 0, t)
            elif e.get("type") == "response_item" and typ in ("custom_tool_call", "function_call", "reasoning") or \
                    (e.get("type") == "response_item" and typ == "message" and p.get("role") == "assistant"):
                if cur is None:
                    cur = {"t_req": max(x for x in (last_done, pending_in) if x is not None) if (last_done or pending_in)
                           else t, "items": [], "calls": []}
                cur["items"].append(t)
                if typ in ("custom_tool_call", "function_call"):
                    c = {"type": typ, "name": p.get("name"), "input": p.get("input") or p.get("arguments") or "",
                         "t_call": t, "resp": len(responses)}
                    calls_by_id[p.get("call_id")] = c
                    cur["calls"].append(c)
            elif e.get("type") == "event_msg" and typ == "token_count" and cur is not None:
                u = ((p.get("info") or {}).get("last_token_usage")) or {}
                cur.update(t_last=max(cur["items"]), out=u.get("output_tokens", 0),
                           reason=u.get("reasoning_output_tokens", 0), input=u.get("input_tokens", 0),
                           cached=u.get("cached_input_tokens", 0))
                responses.append(cur)
                cur, last_done, pending_in = None, t, None
    if not responses:
        return {"run": os.path.basename(run), "error": "no model responses found"}
    t0 = min(r["t_req"] for r in responses)
    t1 = max(max(r["t_last"], *[c.get("t_out", 0) for c in r["calls"]] or [0]) for r in responses)
    M = _motions(run, t0, t1)
    kinds = collections.defaultdict(lambda: {"n": 0, "model_s": 0.0, "exposed_s": 0.0})
    model = first = decode = 0.0
    images = fixups = 0
    prev_err = False
    for r in responses:
        d = max(0.0, r["t_last"] - r["t_req"])
        dec = min(d, r["out"] / DECODE_TOK_S)
        model, decode, first = model + d, decode + dec, first + d - dec
        k = _kind(r["calls"]) if r["calls"] else "message only"
        exposed = d - _overlap(M, r["t_req"], r["t_last"])
        kinds[k]["n"] += 1
        kinds[k]["model_s"] += d
        kinds[k]["exposed_s"] += exposed
        if prev_err:
            fixups += 1
        prev_err = any(c.get("error") for c in r["calls"])
        images += sum(c.get("images", 0) for c in r["calls"])
    tools = [c for r in responses for c in r["calls"] if "t_out" in c]
    tool_s = sum(c["t_out"] - c["t_call"] for c in tools)
    wall = (meta.get("agent_wall_s") or (t1 - t0))
    moving = sum(b - a for a, b in M)
    moving_model = sum(_overlap(M, r["t_req"], r["t_last"]) for r in responses)
    out = {
        "run": os.path.basename(run), "wall_s": round(wall, 1), "responses": len(responses),
        "model_s": round(model, 1), "first_token_s_est": round(first, 1), "decode_s_est": round(decode, 1),
        "tool_s": round(tool_s, 1), "robot_moving_s": round(moving, 1),
        "robot_moving_while_model_s": round(moving_model, 1), "images_viewed": images,
        "round_trips_after_errors": fixups,
        "median_step_s": round(sorted(r["t_last"] - r["t_req"] for r in responses)[len(responses) // 2], 2),
        "by_kind": {k: {"n": v["n"], "model_s": round(v["model_s"], 1), "exposed_s": round(v["exposed_s"], 1)}
                    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]["exposed_s"])},
        "output_tokens": sum(r["out"] for r in responses), "reasoning_tokens": sum(r["reason"] for r in responses),
        "context_tokens_median": sorted(r["input"] for r in responses)[len(responses) // 2],
        "threads": len(_rollouts(run)), "prunes": meta.get("codex_prunes") or [],
        "reasoning_summary": (meta.get("codex") or {}).get("reasoning_summary"),
    }
    return out


def summary(p: dict) -> str:
    if "error" in p:
        return f"{p['run']}: {p['error']}"
    lines = [f"{p['run']}: {p['wall_s']:.0f} s, {p['responses']} model round trips (median {p['median_step_s']} s), "
             f"model {p['model_s']:.0f} s (first token ~{p['first_token_s_est']:.0f}, decode ~{p['decode_s_est']:.0f}), "
             f"tools {p['tool_s']:.0f} s; robot moving {p['robot_moving_s']:.0f} s, "
             f"{p['robot_moving_while_model_s']:.0f} s of it while the model worked",
             f"  images viewed {p['images_viewed']}, threads {p['threads']}, prunes {len(p['prunes'])}, "
             f"context median {p['context_tokens_median'] / 1000:.0f}k tokens, summaries {p['reasoning_summary']}"]
    for k, v in p["by_kind"].items():
        lines.append(f"  {k:28s} {v['n']:4d} trips  {v['model_s']:6.0f} s model, {v['exposed_s']:6.0f} s with the "
                     "robot still")
    return "\n".join(lines)


def main(argv=None) -> int:
    runs = (argv if argv is not None else sys.argv[1:])
    if not runs:
        print(__doc__)
        return 2
    for run in runs:
        p = profile(run)
        with open(os.path.join(run, "profile.json"), "w") as f:
            json.dump(p, f, indent=1)
        print(summary(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
