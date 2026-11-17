"""Export a real-robot agent run (scripts/agent_run_real.py's RUN directory) as a compact folder for analysis and for the
repository: what the agent did and saw, without the bulk (the agent's saved camera arrays, robotd's full 250 Hz ticks,
Codex's databases). Run scripts/run_log.py on the run first (the timeline and robotd's slice).

    python scripts/export_run.py ~/robot-cell/agent-runs/towel-fold-03 OUT_DIR

OUT_DIR/<tag>/:
- meta.json, verdict.json, before.json, TIMELINE.md, timeline.jsonl: the run and its timeline (scripts/run_log.py)
- AGENTS.md: what the agent was told
- codex/rollout.jsonl: Codex's own session record (every prompt, message, reasoning summary, tool call and output, with
  times; the session's opaque account ids blanked); every image embedded in it (what the model was shown) is saved
  as codex/images/NNNN.jpg and replaced by a note naming that file; codex/events.jsonl (what the monitor showed);
  codex/app-server.jsonl
  (the raw protocol, account notifications dropped); codex/config.toml; codex/egress.jsonl (the proxy's connections)
- steer/inbox.jsonl: the operator's messages
- logs/: the program host's log (every program started and how it ended), the memory watchdog's, the video recorder's
- workspace/: the agent's code (.py .md .txt .json .sh) at the episode's end, and each program run's log.txt and
  state (workspace/.robot/runs/<run>/)
- seen/: every image the agent viewed (Codex's image views), as JPEG under its own path
- witness/: the operator's top and front frames (JPEG); video-*.mp4: the timelapse
- robotd/: requests.jsonl less the polling (commands, replies, refusals, IK, park, release, finish, errors),
  WINDOW.json, run.json, code/ (the program sources the requests name), ticks_25hz.npz (t, q, q_ref, q_cmd, tau at
  every 10th tick; robotd.core.recorder's columns, joints in run.json's order)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

QUIET = {"state", "events", "health", "ping", "description"}
CODE_EXT = (".py", ".md", ".txt", ".json", ".sh", ".toml", ".yaml", ".yml")


def strip_images(x, save):
    """A JSON value with every embedded image (a data: URL) saved by save(data_url) -> its file name, and replaced by
    a note naming the file."""
    if isinstance(x, str):
        if x.startswith("data:image/") and len(x) > 1000:
            return f"(image saved as codex/images/{save(x)})"
        return x
    if isinstance(x, list):
        return [strip_images(v, save) for v in x]
    if isinstance(x, dict):
        return {k: strip_images(v, save) for k, v in x.items()}
    return x


def image_saver(folder):
    """save(data_url) -> name: the image as JPEG in folder, numbered in the order the session embedded them."""
    import base64
    import io
    os.makedirs(folder, exist_ok=True)
    n = [0]

    def save(url):
        n[0] += 1
        name = f"{n[0]:04d}.jpg"
        try:
            im = Image.open(io.BytesIO(base64.b64decode(url[url.find(",") + 1:])))
            (im.convert("RGB") if im.mode not in ("RGB", "L") else im).save(os.path.join(folder, name), quality=85)
        except Exception:                               # noqa: BLE001 -- an image that does not decode: note it
            name = f"{n[0]:04d} (did not decode)"
        return name
    return save


def copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def to_jpeg(src, dst, quality=85):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(dst, quality=quality)


def main(run, out_root):
    run = os.path.abspath(os.path.expanduser(run))
    tag = os.path.basename(run)
    out = os.path.join(os.path.abspath(out_root), tag)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in ("meta.json", "verdict.json", "before.json", "TIMELINE.md", "timeline.jsonl"):
        copy(os.path.join(run, f), os.path.join(out, f))
    copy(os.path.join(run, "workspace", "AGENTS.md"), os.path.join(out, "AGENTS.md"))
    # codex
    roll = sorted(glob.glob(os.path.join(run, "codex", "home", "sessions", "**", "rollout-*.jsonl"), recursive=True))
    os.makedirs(os.path.join(out, "codex"))
    save = image_saver(os.path.join(out, "codex", "images"))
    with open(os.path.join(out, "codex", "rollout.jsonl"), "w") as w:
        for p in roll:
            for line in open(p, errors="replace"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "session_meta":
                    pl = d.get("payload") or {}
                    for k in ("creator_user_id", "creator_account_id"):
                        if k in pl:
                            pl[k] = "(removed)"
                w.write(json.dumps(strip_images(d, save)) + "\n")
    copy(os.path.join(run, "codex", "events.jsonl"), os.path.join(out, "codex", "events.jsonl"))
    copy(os.path.join(run, "codex", "egress.jsonl"), os.path.join(out, "codex", "egress.jsonl"))
    copy(os.path.join(run, "codex", "home", "config.toml"), os.path.join(out, "codex", "config.toml"))
    raw = os.path.join(run, "codex", "app-server.jsonl")
    if os.path.exists(raw):
        with open(os.path.join(out, "codex", "app-server.jsonl"), "w") as w:
            for line in open(raw, errors="replace"):
                if '\\"method\\":\\"account/' in line:
                    continue
                w.write(line)
    copy(os.path.join(run, "steer", "inbox.jsonl"), os.path.join(out, "steer", "inbox.jsonl"))
    for p in glob.glob(os.path.join(run, "logs", "*")):
        copy(p, os.path.join(out, "logs", os.path.basename(p)))
    # the agent's code and its program runs' logs
    ws = os.path.join(run, "workspace")
    for dirpath, dirnames, files in os.walk(ws):
        rel = os.path.relpath(dirpath, ws)
        inside_runs = rel.startswith(os.path.join(".robot", "runs"))
        if "pause-" in rel:
            continue
        for f in files:
            keep = (f.endswith(CODE_EXT) and os.path.getsize(os.path.join(dirpath, f)) < 2_000_000) or \
                   (inside_runs and f in ("log.txt", "status.json", "state.json"))
            if keep and f != "AGENTS.md":
                copy(os.path.join(dirpath, f), os.path.join(out, "workspace", rel, f))
    # the images the agent viewed
    n_seen = 0
    for line in open(os.path.join(run, "timeline.jsonl")) if os.path.exists(os.path.join(run, "timeline.jsonl")) else []:
        e = json.loads(line)
        if e.get("kind") != "image":
            continue
        p = e.get("path") or ""
        src = (os.path.join(ws, os.path.relpath(p, "/workspace")) if p.startswith("/workspace/") else
               os.path.join(run, "sandbox", "tmp", os.path.relpath(p, "/tmp")) if p.startswith("/tmp/") else None)
        if src and os.path.exists(src):
            dst = os.path.join(out, "seen", os.path.splitext(p.lstrip("/"))[0] + ".jpg")
            if not os.path.exists(dst):
                try:
                    to_jpeg(src, dst)
                    n_seen += 1
                except OSError:
                    pass
    # witness frames and the video
    for p in sorted(glob.glob(os.path.join(run, "witness", "*.png"))):
        to_jpeg(p, os.path.join(out, "witness", os.path.basename(p)[:-4] + ".jpg"), quality=80)
    for p in glob.glob(os.path.join(run, "video-*.mp4")):
        copy(p, os.path.join(out, os.path.basename(p)))
    # robotd: the commands (no polling), the window, the sources, the ticks at 25 Hz
    rd = os.path.join(run, "robotd")
    for f in ("WINDOW.json", "run.json", "README"):
        copy(os.path.join(rd, f), os.path.join(out, "robotd", f))
    for p in glob.glob(os.path.join(rd, "code", "*.txt")):
        copy(p, os.path.join(out, "robotd", "code", os.path.basename(p)))
    req = os.path.join(rd, "requests.jsonl")
    if os.path.exists(req):
        with open(os.path.join(out, "robotd", "requests.jsonl"), "w") as w:
            for line in open(req, errors="replace"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ok = (d.get("rep") or {}).get("ok", True)
                if (d.get("req") or {}).get("op") in QUIET and ok:
                    continue
                w.write(line)
    ticks = sorted(glob.glob(os.path.join(rd, "ticks", "*.npz")))
    if ticks:
        win = json.load(open(os.path.join(rd, "WINDOW.json")))
        t0, t1 = (win.get("t") or [-np.inf, np.inf])
        cols = {c: [] for c in ("t", "q", "q_ref", "q_cmd", "tau")}
        for p in ticks:
            with np.load(p) as z:
                k = (z["t"] >= t0) & (z["t"] <= t1)
                for c in cols:
                    if c in z.files:
                        cols[c].append(z[c][k][::10])
        np.savez_compressed(os.path.join(out, "robotd", "ticks_25hz.npz"),
                            **{c: np.concatenate(v) for c, v in cols.items() if v})
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"{tag}: {size / 1e6:.1f} MB, {n_seen} viewed images -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
