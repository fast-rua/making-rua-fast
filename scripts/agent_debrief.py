"""Talk with an ended episode's agent: reflect with it on what happened, and hear its feedback.

    python scripts/agent_debrief.py TAG                 # e.g. towel-fold-09; then type here, or in the agent monitor
    python scripts/agent_debrief.py TAG --continue      # pick up the last debrief's conversation instead of a fresh one

The agent goes on from its episode's conversation: a fork of its Codex thread, so it remembers everything it did and
saw, and the episode's own rollout stays as it was. It runs in the episode's sandbox minus the robot:
- no robot or camera socket and no network but the model's API, so nothing it runs can reach robotd or move anything;
- /workspace: its workspace as the episode left it, through an overlay. It can read, run and change files; the changes
  go to RUN/debrief/<n>/upper and the episode's own workspace is untouched;
- /episode: the whole run record, read-only (its conversation, TIMELINE.md, the witness frames, the camera
  recordings, robotd's slice of the episode, meta.json). The camera recordings are also where robot.recording looks.

Messages come from this terminal (a line each; an empty line does nothing) and from the agent monitor's box (the run's
Agent tab), which writes RUN/steer/inbox.jsonl. The agent's replies print here, and the monitor shows the debrief after
the episode's stream. Each debrief is RUN/debrief/<n>/: events.jsonl, app-server.jsonl, meta.json, upper/ (changed
files). Ctrl-C (or --idle-h without a message) ends it.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_run_real as arr                             # noqa: E402  (sets agent_run's paths for this cell)

ar = arr.ar
from codex_driver import CodexDriver                     # noqa: E402

EPISODE = "/episode"                                     # the run record, read-only, in the sandbox
def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


NOTE = """The episode is over, and this is a debrief: the operator wants to talk with you about how it went, and to
hear what you think. The robot is not connected now: no program can reach the robot or its cameras, and nothing you
run moves anything. /workspace is your workspace as the episode left it (you can read, run and change files there; the
episode's own copy is kept apart). The whole record of the episode is read-only in /episode: your conversation
(/episode/codex/events.jsonl), TIMELINE.md (a timeline of your commands, programs and the robot's motions), the witness
frames (/episode/witness/), the camera recordings (/episode/cameras/, also through robot.recording), the robot driver's
log of the episode (/episode/robotd/) and meta.json. Answer the operator's questions; look at the record when it helps."""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tag", help="the episode's run name (a directory of agent-runs/)")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="go on with the last debrief's conversation (its thread) instead of a new fork of the episode")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default=None,
                    help="reasoning effort for the debrief (default: the episode's)")
    ap.add_argument("--codex-home", default=None, help="the Codex login (DIR/auth.json); default: the episode's")
    ap.add_argument("--idle-h", type=float, default=3.0, help="end after this long without a message")
    a = ap.parse_args()

    run = os.path.join(ar.RUNS, a.tag)
    meta = read_json(os.path.join(run, "meta.json"))
    if meta is None:
        ap.error(f"no run {a.tag} in {ar.RUNS}")
    if meta.get("state") == "running":
        ap.error(f"{a.tag} is still running: steer it from the monitor instead")
    episode_thread = meta.get("codex_thread_id")
    if not episode_thread:
        ap.error(f"{a.tag} has no Codex thread id in meta.json (not a Codex episode?)")
    if not os.path.isdir(os.path.join(run, "codex", "home", "sessions")):
        ap.error(f"{a.tag}: no Codex sessions in codex/home (the conversation was not kept)")
    login = os.path.realpath(os.path.expanduser(a.codex_home or (meta.get("codex") or {}).get("login")
                                                or os.path.dirname(ar.CODEX_AUTH)))
    ar.CODEX_AUTH = os.path.join(login, "auth.json")
    if not os.path.exists(ar.CODEX_AUTH):
        ap.error(f"no Codex login at {ar.CODEX_AUTH} (--codex-home)")

    root = os.path.join(run, "debrief")
    os.makedirs(root, exist_ok=True)
    live_path = os.path.join(root, "live.json")
    live = read_json(live_path)
    if live and time.time() - live.get("wall", 0) < 20:
        ap.error(f"a debrief of {a.tag} is already running (session {live.get('session')}, pid {live.get('pid')})")
    done = sorted(int(x) for x in os.listdir(root) if x.isdigit())
    thread, fork = episode_thread, True
    if a.cont:
        last = next((read_json(os.path.join(root, str(n), "meta.json")) for n in reversed(done)
                     if os.path.exists(os.path.join(root, str(n), "meta.json"))), None)
        if not last or not last.get("thread_id"):
            ap.error("no earlier debrief to continue")
        thread, fork = last["thread_id"], False
    n = (done[-1] + 1) if done else 1
    d = os.path.join(root, str(n))
    for sub in ("upper", "overlay-work"):
        os.makedirs(os.path.join(d, sub))

    # ---- the episode's sandbox, without the robot
    tmp = tempfile.mkdtemp(prefix=f"robotapi-debrief-{a.tag[:30]}-", dir="/tmp")
    for sub in ("sock", "egress", "host", "tmp"):          # sock stays empty: no robot, no cameras
        os.makedirs(os.path.join(tmp, sub), mode=0o700)
    password = secrets.token_urlsafe(24)
    llm = ar.llm_spec("gpt", None, f"{a.tag}-debrief-{n}")
    proxy = ar.EgressProxy(f"{tmp}/egress/proxy.sock", password, llm["egress"], os.path.join(d, "egress.jsonl"))
    host_log = os.open(os.path.join(d, "program-host.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    stop_r, stop_w = os.pipe()
    inner = [f"{ar.CODEX_DIR}/bin/codex", "app-server", "--stdio"]
    models = [m for m in (meta.get("sandbox") or {}).get("model_files", []) if os.path.exists(m)]
    cams = os.path.join(run, "cameras")
    argv = arr.sandbox_argv(run, tmp, inner, stop_r, host_log, password, models, llm, False,
                            cams if os.path.isdir(cams) else None)
    k = next(i for i in range(len(argv) - 2) if argv[i] == "--bind" and argv[i + 2] == ar.WS)
    argv[k:k + 3] = ["--overlay-src", os.path.join(run, "workspace"),          # the workspace: the episode's,
                     "--overlay", os.path.join(d, "upper"), os.path.join(d, "overlay-work"), ar.WS]   # changes kept apart
    i = argv.index("--clearenv")
    argv[i:i] = ["--ro-bind", run, EPISODE]
    ar.write_json(os.path.join(d, "bwrap.json"), [x.replace(password, "<password>") for x in argv])

    inbox = os.path.join(run, "steer", "inbox.jsonl")
    os.makedirs(os.path.dirname(inbox), exist_ok=True)
    open(inbox, "a").close()
    stderr = open(os.path.join(d, "stderr.log"), "w")
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                         pass_fds=(stop_r, host_log), start_new_session=True)
    os.close(stop_r)
    os.close(host_log)
    driver = CodexDriver(p, os.path.join(d, "events.jsonl"), os.path.join(d, "app-server.jsonl"), inbox, "", ar.WS,
                         linger_s=float("inf"))
    driver.inbox_pos = os.path.getsize(inbox)           # only messages sent from now on

    typed = []                                           # messages from this terminal (not echoed back)

    def echo(ev):
        t, item = ev.get("type"), ev.get("item") or {}
        if t == "item.completed" and item.get("type") == "agent_message":
            print(f"\n\033[1;36magent:\033[0m {item.get('text', '')}\n", flush=True)
        elif t == "item.completed" and item.get("type") == "command_execution":
            print(f"\033[2m  ran: {str(item.get('command'))[:160]}\033[0m", flush=True)
        elif t == "turn.completed":
            print("\033[2m(your turn: type a message, Enter to send)\033[0m", flush=True)
        elif t in ("error", "turn.failed"):
            print(f"\033[31m{ev.get('message') or ev.get('error')}\033[0m", flush=True)
        elif t == "operator.message":
            if ev.get("text") in typed:
                typed.remove(ev.get("text"))
            else:
                print(f"\033[1;33myou (from the monitor):\033[0m {ev.get('text')}", flush=True)
    driver.on_event = echo

    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    t0 = time.time()
    info = {"schema": "robotapi.debrief.v1", "tag": a.tag, "session": n, "started_utc": ar.utc(t0),
            "from_thread": thread, "fork": fork, "effort": a.effort, "login": login, "thread_id": None}

    def beat():                                          # the monitor's sign that the debrief is open
        while not stopping.is_set():
            ar.write_json(live_path, {"pid": os.getpid(), "session": n, "wall": time.time()})
            stopping.wait(5)
    try:
        info["thread_id"] = driver.continue_thread(thread, fork=fork, developer_instructions=NOTE if fork else None,
                                                   effort=a.effort)
        ar.write_json(os.path.join(d, "meta.json"), info)
        driver._emit({"type": "debrief.started", "session": n, "thread_id": info["thread_id"], "from_thread": thread,
                      "fork": fork})
        threading.Thread(target=beat, daemon=True).start()
        print(f"debrief {n} of {a.tag}: the agent remembers its episode; the robot is not connected.\n"
              "Type a message and press Enter (or use the monitor's box). Ctrl-C ends the debrief.", flush=True)

        def read_stdin():
            for line in sys.stdin:
                if line.strip():
                    typed.append(line.strip())
                    try:
                        driver.deliver(line.strip())
                    except Exception as e:               # noqa: BLE001
                        print(f"could not deliver: {e!r}", flush=True)
        threading.Thread(target=read_stdin, daemon=True).start()
        last = time.time()
        while not stopping.is_set() and not driver.closed.is_set() and p.poll() is None:
            driver.poll_inbox()
            if driver.turn_id:                           # the agent is answering
                last = time.time()
            msgs = driver.stats["operator_messages"]
            last = max(last, msgs[-1]["wall"] if msgs else 0, driver.idle_since or 0)
            if time.time() - last > a.idle_h * 3600:
                print("no message for a while: ending the debrief", flush=True)
                break
            time.sleep(0.5)
    finally:
        stopping.set()
        try:
            driver._emit({"type": "debrief.ended", "session": n})
        except Exception:                                # noqa: BLE001
            pass
        driver.close()
        try:
            os.write(stop_w, b"x")
            p.wait(30)
        except (OSError, subprocess.TimeoutExpired):
            os.killpg(p.pid, signal.SIGKILL)
        proxy.close()
        info.update(ended_utc=ar.utc(), stats=driver.stats)
        ar.write_json(os.path.join(d, "meta.json"), info)
        try:
            os.remove(live_path)
        except OSError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"debrief {n} ended; its record is {d}", flush=True)


if __name__ == "__main__":
    main()
