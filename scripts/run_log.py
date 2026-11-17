"""Make a real-robot agent run's folder a complete log of its episode, and write its timeline
(scripts/agent_run_real.py does this at the end of every episode; again by hand for older runs):

    python scripts/run_log.py ~/robot-cell/agent-runs/towel-fold-03 [RUN ...]
    python scripts/run_log.py --all ~/robot-cell/agent-runs      # every run whose log is missing or incomplete

RUN/robotd/: the episode's part of robotd's recording. robotd records into its own --record directory for as long
as it runs, across episodes; this copies out the episode's window (from the agent_run_real start to the end of the
episode, park and check included):
- requests.jsonl: every request to robotd in the window (every command, reply, refusal and abort, with the wall
  clock, the robot clock and the program line that sent it)
- ticks/: robotd's 250 Hz tick files that overlap the window (joints, references, commands, torques; hard links)
- code/: the program sources those requests name (by hash), and run.json, robotd's own record of its start
- WINDOW.json: the window (wall and robot clocks), what was copied, and whether it is complete. robotd writes ticks in
  60 s files, so the window's last one exists only once robotd has written it (within a minute, or when it stops):
  the launcher waits for it a little; a run left incomplete is completed by the next --all or by the next episode.

RUN/timeline.jsonl and RUN/TIMELINE.md: what happened, in time order, from
- the agent: Codex's own session record (codex/home/sessions/**/rollout-*.jsonl, a start and end time for every item):
  the prompt, the operator's messages, its messages and reasoning summaries, every command (start, duration, exit
  code, output), the images it viewed, file changes, compactions, turns and token use
- the program host (logs/program-host.jsonl): every program started and how it ended
- robotd (RUN/robotd/requests.jsonl): the commands to the arms and grippers with their results, refusals, IK queries,
  park, release, finish; and any request that failed
timeline.jsonl keeps the details (whole commands, output tails); TIMELINE.md is the same, a line each, to read.
Standard library and numpy only.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

SLACK_S = 2.0                       # the window extends this far past the episode's end (the last replies)
QUIET = {"state", "events", "health", "ping", "hello", "description"}   # polled all the time: only their errors
COLLAPSE = {"ik", "fk"}             # queries a program may send many of: consecutive ones become one line


def _utc_s(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def _link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:                                     # another filesystem
        shutil.copy2(src, dst)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ robotd's slice
def slice_robotd(run: str, meta: dict, wait_s: float = 0.0) -> dict:
    """Copy the episode's window of robotd's recording into RUN/robotd/ (idempotent). wait_s: wait this long for the
    tick file that covers the window's end."""
    rd = meta.get("robotd") or {}
    record, offset = rd.get("record"), int(rd.get("requests_offset") or 0)
    out = os.path.join(run, "robotd")
    os.makedirs(out, exist_ok=True)
    if not record or not os.path.exists(os.path.join(record, "requests.jsonl")):
        info = {"complete": False, "error": f"robotd's recording {record!r} is not there"}
        _write_json(os.path.join(out, "WINDOW.json"), info)
        return info
    w0 = _utc_s(meta.get("started_utc")) or 0.0
    w1 = (_utc_s(meta.get("ended_utc")) or time.time()) + SLACK_S
    # requests: from the episode's offset to the window's end
    lines, clock, codes = [], [], set()
    with open(os.path.join(record, "requests.jsonl"), "rb") as f:
        f.seek(offset)
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("wall", 0) > w1:
                break
            lines.append(line)
            if isinstance(d.get("t1"), (int, float)):
                clock.append(d["wall"] - d["t1"])
            for s in (d.get("req") or {}).get("src") or []:
                if len(s) > 3 and s[3]:
                    codes.add(s[3])
    with open(os.path.join(out, "requests.jsonl.tmp"), "wb") as f:
        f.writelines(lines)
    os.replace(os.path.join(out, "requests.jsonl.tmp"), os.path.join(out, "requests.jsonl"))
    for sub in ("code", "ticks"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for sha in codes:
        src = os.path.join(record, "code", f"{sha}.txt")
        if os.path.exists(src):
            _link(src, os.path.join(out, "code", f"{sha}.txt"))
    if os.path.exists(os.path.join(record, "run.json")):
        shutil.copy2(os.path.join(record, "run.json"), os.path.join(out, "run.json"))
    off = float(np.median(clock)) if clock else None              # wall - robot clock
    info = {"record": record, "requests_offset": offset, "requests": len(lines), "code_files": len(codes),
            "wall": [_iso(w0), _iso(w1)], "clock_offset_s": off,
            "t": [round(w0 - off, 3), round(w1 - off, 3)] if off is not None else None}
    if off is None:
        info.update(complete=False, ticks=[], error="no request carried the robot clock")
        _write_json(os.path.join(out, "WINDOW.json"), info)
        return info
    t0, t1 = w0 - off, w1 - off
    deadline = time.time() + wait_s
    while True:
        picked, covered = [], False
        tick_dir = os.path.join(record, "ticks")
        for fn in sorted(os.listdir(tick_dir)) if os.path.isdir(tick_dir) else []:
            p = os.path.join(tick_dir, fn)
            # a file written before the episode began cannot hold it (a --record directory reused by a later robotd
            # restarts the robot clock: the clock alone is ambiguous)
            if not fn.endswith(".npz") or os.path.getmtime(p) < w0 - 1:
                continue
            try:
                with np.load(p) as z:
                    t = z["t"]
            except Exception:                           # noqa: BLE001 -- being written
                continue
            if len(t) and t[-1] >= t0 and t[0] <= t1:
                picked.append(fn)
                covered |= bool(t[-1] >= t1)
        if covered or time.time() >= deadline:
            break
        time.sleep(5.0)
    stopped = os.path.getmtime(os.path.join(record, "requests.jsonl")) < w1 - SLACK_S
    if not covered and stopped:             # robotd stopped before the episode ended: there is nothing more to come
        covered = True
        info["note"] = "robotd stopped before the episode ended: its last tick file is the last there is"
    for fn in picked:
        _link(os.path.join(tick_dir, fn), os.path.join(out, "ticks", fn))
    info.update(ticks=picked, complete=covered, bundled_utc=_iso(time.time()))
    _write_json(os.path.join(out, "WINDOW.json"), info)
    with open(os.path.join(out, "README"), "w") as f:
        f.write(f"The episode's window of robotd's recording {record} (scripts/run_log.py): requests.jsonl from byte "
                f"{offset} of its requests.jsonl, the tick files that overlap the window (hard links), the program "
                f"sources the requests name, robotd's run.json. WINDOW.json: the window and whether it is complete.\n"
                f"Load the ticks with robotd.core.recorder.load_ticks('{out}').\n")
    return info


# ------------------------------------------------------------------------------------------------ the timeline
def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def queue_events(run: str) -> list:
    """The kernel's queue journal (robot/kernel.py _journal, workspace/.robot/events/queue.jsonl): each action queued,
    started, ended, stopped or dropped, flushes and looks, with what runs and waits after each."""
    path = os.path.join(run, "workspace", ".robot", "events", "queue.jsonl")
    ev = []
    for line in open(path) if os.path.exists(path) else []:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ev.append({"src": "queue", "kind": d.pop("event", "?"), **d})
    return ev


def agent_events(run: str) -> list:
    """The agent's items from Codex's session record(s), with their start and end times."""
    ev, prompt_seen = [], False
    for path in sorted(glob.glob(os.path.join(run, "codex", "home", "sessions", "**", "rollout-*.jsonl"),
                                 recursive=True)):
        last_usage = None
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            p = d.get("payload") or {}
            kind = p.get("type")
            if d.get("type") == "event_msg" and kind == "token_count":
                last_usage = ((p.get("info") or {}).get("total_token_usage")) or last_usage
                continue
            if d.get("type") == "event_msg" and kind in ("task_started", "task_complete"):
                e = {"wall": _utc_s(d["timestamp"]), "src": "agent",
                     "kind": "turn started" if kind == "task_started" else "turn ended"}
                if kind == "task_complete" and last_usage:
                    e["usage"] = last_usage
                ev.append(e)
                continue
            if d.get("type") == "event_msg" and "error" in (kind or ""):
                ev.append({"wall": _utc_s(d["timestamp"]), "src": "agent", "kind": "error",
                           "text": p.get("message") or json.dumps(p)[:2000]})
                continue
            if not (d.get("type") == "event_msg" and kind == "item_completed"):
                continue
            it = p.get("item") or {}
            t, s0 = it.get("type"), p.get("started_at_ms")
            e = {"wall": (s0 / 1000.0) if s0 else _utc_s(d["timestamp"]), "src": "agent"}
            if p.get("completed_at_ms") and s0:
                e["dur_s"] = round((p["completed_at_ms"] - s0) / 1000.0, 3)
            if t == "UserMessage":
                e.update(kind="operator" if prompt_seen else "prompt", text=_text(it.get("content")))
                prompt_seen = True
            elif t == "AgentMessage":
                e.update(kind="message", text=_text(it.get("content")), phase=it.get("phase"))
            elif t == "Reasoning":
                e.update(kind="reasoning", text="\n".join(it.get("summary_text") or []))
            elif t == "CommandExecution":
                cmd = it.get("command")
                cmd = cmd[-1] if isinstance(cmd, list) and cmd else cmd
                dur = it.get("duration") or {}
                out = it.get("aggregated_output") or ""
                e.update(kind="command", command=cmd, exit_code=it.get("exit_code"), status=it.get("status"),
                         dur_s=round(dur.get("secs", 0) + dur.get("nanos", 0) / 1e9, 3) if dur else e.get("dur_s"),
                         output_tail=out[-4000:], output_chars=len(out))
            elif t == "ImageView":
                e.update(kind="image", path=(it.get("path") or "").replace("file://", ""))
            elif t == "FileChange":
                e.update(kind="file change", changes=[{k: c.get(k) for k in ("path", "kind")}
                                                       for c in it.get("changes") or []])
            elif t in ("ContextCompaction", "Compaction"):
                e.update(kind="compaction")
            else:
                e.update(kind=f"item {t}", raw=json.dumps(it)[:2000])
            ev.append(e)
    return ev


def program_events(run: str) -> list:
    path = os.path.join(run, "logs", "program-host.jsonl")
    ev, names = [], {}
    for line in open(path) if os.path.exists(path) else []:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        k = d.get("event")
        if k == "start":
            argv = d.get("argv") or ["?"]                # the program file, before options like --unattended
            names[d.get("run_dir")] = os.path.basename(next((x for x in argv if x.endswith(".py")), argv[-1]))
            ev.append({"wall": d["wall"], "src": "program", "kind": "start", "program": names[d.get("run_dir")],
                       "run_dir": d.get("run_dir")})
        elif k == "exit":
            ev.append({"wall": d["wall"], "src": "program", "kind": "exit", "program": names.get(d.get("run_dir")),
                       "run_dir": d.get("run_dir"), "returncode": d.get("returncode")})
        elif k not in ("listening",):
            ev.append({"wall": d.get("wall"), "src": "program", "kind": k, **{x: y for x, y in d.items()
                                                                               if x not in ("wall", "event")}})
    return ev


def _spec_words(spec: dict) -> str:
    who = spec.get("ee") or spec.get("name") or spec.get("arms") or spec.get("joints")
    who = ",".join(who) if isinstance(who, list) and all(isinstance(x, str) for x in who) else who
    words = [str(spec.get("op")), str(who) if who else ""]
    for k in ("width", "speed", "force", "duration", "line", "relative", "free", "dry_run", "max_travel", "accel"):
        if spec.get(k) not in (None, False):
            words.append(f"{k}={spec[k]}")
    if spec.get("times"):
        words.append(f"times={spec['times']}")
    pts = spec.get("points") or spec.get("poses")
    if isinstance(pts, dict):
        words.append("points=" + "/".join(str(len(v)) for v in pts.values()))
    elif isinstance(pts, list):
        words.append(f"points={len(pts)}")
    return " ".join(w for w in words if w)


def robot_events(run: str) -> list:
    path = os.path.join(run, "robotd", "requests.jsonl")
    ev, clients = [], {}
    for line in open(path, errors="replace") if os.path.exists(path) else []:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        req, rep = d.get("req") or {}, d.get("rep") or {}
        op = req.get("op")
        if op == "hello":
            clients[d.get("client")] = req.get("client")
        if op in QUIET and rep.get("ok", True):
            continue
        src = (req.get("src") or [[None]])[0]
        # robotd logs a request when it replies (a move's reply comes when the move ends): placed at its start
        e = {"wall": d.get("wall", 0) - (d.get("dur_s") or 0), "reply_wall": d.get("wall"), "src": "robot",
             "client": clients.get(d.get("client"), d.get("client")), "op": op, "dur_s": d.get("dur_s"),
             "t0": d.get("t0"), "t1": d.get("t1"), "at": f"{os.path.basename(src[0])}:{src[1]}" if src and src[0] else None}
        if op == "command":
            spec = req.get("spec") or {}
            e["kind"] = "command"
            e["what"] = _spec_words(spec)
            e["spec"] = spec
        elif op == "finish":
            e.update(kind="finish", text=req.get("summary"))
        else:
            e["kind"] = op
            e["what"] = " ".join([str(op)] + [f"{k}={v}" for k, v in req.items() if k not in ("op", "src", "target", "seed")
                                              and not isinstance(v, (dict, list))])[:200]
        if rep.get("ok") is False or rep.get("error"):
            err = rep.get("error") or {}
            e["error"] = f"{err.get('type', 'error')}: {err.get('message', '')}" if isinstance(err, dict) else str(err)
        else:
            res = rep.get("result")
            if isinstance(res, dict) and ("status" in res or "reason" in res):
                e["result"] = {k: res.get(k) for k in ("id", "status", "reason", "duration") if k in res}
            elif op == "ik" and isinstance(res, dict):
                e["result"] = {k: res.get(k) for k in ("ok", "error_m") if k in res}
        ev.append(e)
    # consecutive IK/FK queries from one program: one line
    out = []
    for e in ev:
        prev = out[-1] if out else None
        if (prev and e["kind"] in COLLAPSE and prev["kind"] == e["kind"] and prev.get("client") == e.get("client")
                and e["wall"] - prev.get("last_wall", prev["wall"]) < 5.0):
            prev["count"] = prev.get("count", 1) + 1
            prev["last_wall"] = e["wall"]
            prev["failed"] = prev.get("failed", 0) + (not (e.get("result") or {}).get("ok", True) or "error" in e)
            continue
        if e["kind"] in COLLAPSE:
            e["failed"] = int(not (e.get("result") or {}).get("ok", True) or "error" in e)
        out.append(e)
    return out


def _clip(s, n=180) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _md_line(e: dict, t0: float) -> str | None:
    dt = e["wall"] - t0
    stamp = f"{'-' if dt < 0 else '+'}{int(abs(dt)) // 60:02d}:{abs(dt) % 60:04.1f}"
    k, src = e.get("kind"), e["src"]
    if src == "agent":
        if k == "command":
            cmd = str(e.get("command") or "")
            first = cmd.strip().splitlines()[0] if cmd.strip() else ""
            more = f" (+{cmd.count(chr(10))} lines)" if "\n" in cmd.strip() else ""
            return (f"{stamp}  agent     ran [{e.get('dur_s', 0):.1f} s, exit {e.get('exit_code')}]: "
                    f"{_clip(first, 150)}{more}")
        if k == "message":
            return f"{stamp}  agent     says: {_clip(e.get('text'), 400)}"
        if k == "reasoning":
            return f"{stamp}  agent     thinks: {_clip(e.get('text'), 300)}" if (e.get("text") or "").strip() else None
        if k in ("prompt", "operator"):
            return f"{stamp}  {'OPERATOR' if k == 'operator' else 'prompt  '}  {_clip(e.get('text'), 400)}"
        if k == "image":
            return f"{stamp}  agent     looks at {e.get('path')}"
        if k == "file change":
            return f"{stamp}  agent     edits {', '.join(str(c.get('path')) for c in e.get('changes') or [])}"
        if k == "turn ended":
            u = e.get("usage") or {}
            return (f"{stamp}  agent     turn ended (tokens so far: in {u.get('input_tokens')}, cached "
                    f"{u.get('cached_input_tokens')}, out {u.get('output_tokens')})")
        if k == "turn started":
            return f"{stamp}  agent     turn started"
        return f"{stamp}  agent     {k}: {_clip(e.get('text') or e.get('raw'), 200)}"
    if src == "queue":
        who = f"#{e['id']} {e.get('name')}" if e.get("id") is not None else ""
        extra = e.get("why") or e.get("status") or ""
        wait = f"; waiting {e['waiting']}" if e.get("waiting") else ""
        return f"{stamp}  queue     {k} {who}{(': ' + _clip(extra, 120)) if extra else ''} (running {e.get('running')}{wait})"
    if src == "program":
        if k == "start":
            return f"{stamp}  program   start {e.get('program')}  ({os.path.basename(str(e.get('run_dir')))})"
        if k == "exit":
            return f"{stamp}  program   exit  {e.get('program')}: code {e.get('returncode')}"
        return f"{stamp}  program   {k}"
    if src == "robot":
        who = f"{e.get('client')}" + (f" {e['at']}" if e.get("at") else "")
        if k == "finish":
            return f"{stamp}  robot     FINISH from {who}: {_clip(e.get('text'), 300)}"
        if k in COLLAPSE:
            n = e.get("count", 1)
            return f"{stamp}  robot     {k} x{n} from {who} ({e.get('failed', 0)} failed)"
        res = e.get("error") or ", ".join(f"{v}" for x, v in (e.get("result") or {}).items()
                                          if x in ("status", "reason")) or "ok"
        if e.get("error"):
            res = "REFUSED " + _clip(res, 220)
        return f"{stamp}  robot     {who}: {_clip(e.get('what'), 140)} -> {res} [{(e.get('dur_s') or 0):.2f} s]"
    return None


def timeline(run: str, meta: dict) -> int:
    ev = agent_events(run) + program_events(run) + robot_events(run) + queue_events(run)
    ev = [e for e in ev if e.get("wall")]
    ev.sort(key=lambda e: e["wall"])
    t0 = _utc_s(meta.get("started_utc")) or (ev[0]["wall"] if ev else time.time())
    with open(os.path.join(run, "timeline.jsonl.tmp"), "w") as f:
        for e in ev:
            f.write(json.dumps({"t": round(e["wall"] - t0, 3), "utc": _iso(e["wall"]), **e}, default=str) + "\n")
    os.replace(os.path.join(run, "timeline.jsonl.tmp"), os.path.join(run, "timeline.jsonl"))
    c = meta.get("codex") or {}
    win = {}
    try:
        win = json.load(open(os.path.join(run, "robotd", "WINDOW.json")))
    except (OSError, ValueError):
        pass
    head = [f"# {meta.get('tag')}: {meta.get('task')}",
            "",
            f"- agent: {meta.get('agent')}, {c.get('model')}, effort {c.get('reasoning_effort')}, tier "
            f"{c.get('service_tier')}, {c.get('version')}",
            f"- started {meta.get('started_utc')}, ended {meta.get('ended_utc')} ({meta.get('wall_s')} s); end: "
            f"{meta.get('end_reason')}; finished: {_clip((meta.get('finished') or {}).get('summary'), 200)}",
            f"- check: {json.dumps(meta.get('check'))}",
            f"- robotd slice: {win.get('requests')} requests, {len(win.get('ticks') or [])} tick files, complete: "
            f"{win.get('complete')}",
            f"- notes: {meta.get('notes')}",
            "",
            "Times are from the episode's start. The details of every line (whole commands, output, specs, results) "
            "are in timeline.jsonl; Codex's own record is codex/home/sessions/; the code at every step is in "
            "workspace.git; robotd's commands and 250 Hz ticks are in robotd/; the cameras are in witness/ and the "
            "video.",
            "", "```"]
    body = [x for x in (_md_line(e, t0) for e in ev) if x]
    with open(os.path.join(run, "TIMELINE.md"), "w") as f:
        f.write("\n".join(head + body + ["```", ""]))
    return len(ev)


def bundle(run: str, wait_s: float = 0.0, quiet: bool = False) -> dict:
    """robotd's slice, then the timeline. Returns WINDOW.json's content (plus the timeline's length)."""
    run = os.path.abspath(os.path.expanduser(run))
    meta = json.load(open(os.path.join(run, "meta.json")))
    info = slice_robotd(run, meta, wait_s=wait_s)
    info["timeline_events"] = timeline(run, meta)
    if not quiet:
        print(f"{os.path.basename(run)}: {info.get('requests')} robotd requests, {len(info.get('ticks') or [])} tick "
              f"files, complete {info.get('complete')}, {info['timeline_events']} timeline events"
              + (f" ({info['error']})" if info.get("error") else ""), flush=True)
    return info


def refresh(root: str, quiet: bool = False) -> None:
    """Bundle every finished run under root whose log is missing or incomplete."""
    for meta_path in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*", "meta.json"))):
        run = os.path.dirname(meta_path)
        try:
            meta = json.load(open(meta_path))
            if meta.get("state") != "done":
                continue
            win = json.load(open(os.path.join(run, "robotd", "WINDOW.json"))) \
                if os.path.exists(os.path.join(run, "robotd", "WINDOW.json")) else {}
            if win.get("complete") and os.path.exists(os.path.join(run, "TIMELINE.md")):
                continue
            if win.get("error") and "not there" in win["error"]:
                continue
            bundle(run, quiet=quiet)
        except Exception as e:                          # noqa: BLE001 -- one bad run must not stop the rest
            print(f"{run}: {e!r}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", metavar="ROOT", help="every finished run under ROOT whose log is missing or incomplete")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the tick file covering the end")
    a = ap.parse_args()
    if a.all:
        refresh(a.all)
    for run in a.runs:
        bundle(run, wait_s=a.wait)


if __name__ == "__main__":
    main()
