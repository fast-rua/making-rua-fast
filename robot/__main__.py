"""python -m robot: run a robot program, and work with it while it is paused.

    python -m robot run task.py [--wait S] [--break STEP | --break FILE:LINE] [--pause-each-step] [--from STEP]
                                [--only STEP] [--unattended]
    python -m robot wait [--wait S]
    python -m robot pause [--wait S]
    python -m robot stop [--wait S]
    python -m robot debug 'CMD' [--wait S]

run starts the program (robot/runner.py) in a process of its own and returns at the first of: the program pauses,
it exits, --wait seconds pass (default 60). It prints the program's output since the last report, then where and
why it paused, or how it exited. wait waits on the current run the same way. pause makes the running program pause at
the next line it runs in its own files (a robot request in progress finishes first), then waits as wait does. stop
ends the program, running or paused (SIGTERM, then SIGKILL after STOP_GRACE s), and reports its exit. debug
sends one pdb command to the paused program and prints its output; when the command resumes the program (c, n, s,
retry, goto, q, ...) it then waits as wait does. The exit status is the program's once it has exited, else 0.

Each run has a directory ./.robot/runs/<UTC time>/ with log.txt (the program's output), status.json and pause-N/
(what the cameras saw at each pause); ./.robot/current points at the latest run.

When ROBOT_PROGRAM_HOST names a program host's socket (scripts/agent/program_host.py, for an agent whose shell
commands each run in a sandbox that ends with the command), run hands the program to that host, which starts it as
the command would have and keeps it past the command's end. Whether it is alive, pause's signal and stop then go
through the host too: later commands cannot see its process.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

TAIL = 150                              # lines of new output shown at most (the rest is in log.txt)
STOP_GRACE = 5.0                        # s from stop's SIGTERM to its SIGKILL
PROGRAM_HOST = os.environ.get("ROBOT_PROGRAM_HOST")     # its socket, or None: programs start here


def _host(req: dict, fd: int | None = None) -> dict:
    """One request to the program host and its reply; fd (the run's log) goes with it."""
    data, t0 = json.dumps(req).encode(), time.time()
    while True:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        try:
            s.connect(PROGRAM_HOST)
            break
        except (FileNotFoundError, ConnectionRefusedError) as e:
            s.close()
            if time.time() - t0 > 5:            # the host may still be starting
                raise SystemExit(f"the program host ({PROGRAM_HOST}) does not answer: {e}")
            time.sleep(0.1)
    try:
        with s:
            if fd is not None:
                socket.send_fds(s, [data[:1]], [fd])
                data = data[1:]
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
            rep = b""
            while chunk := s.recv(65536):
                rep += chunk
        return json.loads(rep)
    except (OSError, ValueError) as e:
        raise SystemExit(f"the program host ({PROGRAM_HOST}) failed: {e!r}")


def _rel(path: str) -> str:
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def _base() -> str:
    return os.path.join(os.getcwd(), ".robot")


def _current() -> str:
    cur = os.path.join(_base(), "current")
    if not os.path.exists(cur):
        raise SystemExit("no run here (./.robot/current): python -m robot run task.py starts one")
    return os.path.realpath(cur)


def _status(run_dir: str) -> dict | None:
    try:
        with open(os.path.join(run_dir, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_status(run_dir: str, st: dict) -> None:
    path = os.path.join(run_dir, "status.json")
    with open(path + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(path + ".tmp", path)


def _alive(st: dict) -> bool:
    if PROGRAM_HOST:
        return _host({"op": "poll", "run_dir": st["run_dir"]})["running"]
    if st.get("host") != socket.gethostname():
        return True                     # another machine's process: cannot tell from here
    try:
        os.kill(st["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{st['pid']}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _settled(st: dict | None) -> bool:
    return st is not None and (st["state"] == "exited" or (st["state"] == "paused" and not st.get("command")))


def _log_tail(run_dir: str, n: int = 3) -> str:
    """The last lines of the run's log, to say why a program died before it could report."""
    try:
        with open(os.path.join(run_dir, "log.txt"), errors="replace") as f:
            lines = [line.rstrip() for line in f if line.strip()]
    except OSError:
        return ""
    return "".join(f"\n    {line}" for line in lines[-n:])


def _wait(run_dir: str, seconds: float, proc=None) -> dict | None:
    """The status once the program has paused (and no debug command is running), exited, or seconds have passed."""
    t0 = time.time()
    while True:
        st = _status(run_dir)
        if _settled(st):
            return st
        if proc is not None:
            dead, code = proc.poll() is not None, proc.returncode
        elif PROGRAM_HOST:
            rep = _host({"op": "poll", "run_dir": run_dir})
            dead, code = not rep["running"], rep["returncode"]
        else:
            dead, code = st is not None and not _alive(st), None
        if dead:
            time.sleep(0.2)             # it may be writing its last status now
            st = _status(run_dir)
            if _settled(st):
                return st
            st = dict(st or {}, state="exited", exit_code=code, kind="died",
                      message="the program ended without reporting why (killed, or a crash: see the log)"
                      + _log_tail(run_dir))
            _write_status(run_dir, st)
            return st
        if time.time() - t0 >= seconds:
            return st
        time.sleep(0.05)


def _report(run_dir: str, st: dict | None, waited: float) -> int:
    """Print the program's new output, then the pause block, the exit or 'still running'. Returns the CLI's status."""
    rpath = os.path.join(run_dir, "report.json")
    try:
        with open(rpath) as f:
            rep = json.load(f)
    except (OSError, ValueError):
        rep = {"log": 0, "pause": 0}
    log = os.path.join(run_dir, "log.txt")
    with open(log, "rb") as f:
        f.seek(rep["log"])
        data = f.read()
    rep["log"] += len(data)
    lines = data.decode("utf-8", "replace").splitlines()
    if len(lines) > TAIL:
        print(f"... ({len(lines) - TAIL} lines before these are in {_rel(log)})")
        lines = lines[-TAIL:]
    if lines:
        print("\n".join(lines))
    code = 0
    state = (st or {}).get("state")
    if state == "paused":
        if st["pause"] <= rep["pause"]:            # reported before: show it again
            print(st["block"])
        rep["pause"] = st["pause"]
    elif state == "exited":
        code = st.get("exit_code")
        why = f" ({st['message']})" if st.get("message") and st.get("kind") not in ("finished",) else ""
        print(f"--- exit code {code}{why}" if code is not None else f"--- the program has gone{why}")
        code = 1 if code is None else code
    else:
        print(f"--- still running after {waited:.0f} s"
              + (f" (step {st['step']})" if st and st.get("step") else "")
              + "; python -m robot wait waits again")
    with open(rpath, "w") as f:
        json.dump(rep, f)
    return code


def cmd_run(a) -> int:
    task = os.path.abspath(a.task)
    if not os.path.isfile(task):
        raise SystemExit(f"no such file: {a.task}")
    base = _base()
    cur = os.path.join(base, "current")
    if os.path.exists(cur):
        st = _status(os.path.realpath(cur))
        if st is not None and st["state"] != "exited" and _alive(st):
            how = "python -m robot debug q ends it" if st["state"] == "paused" else \
                "python -m robot wait waits for it; python -m robot stop ends it"
            pid = "" if PROGRAM_HOST else f" (pid {st['pid']})"     # the host's process: no pid for this shell
            raise SystemExit(f"run {_rel(os.path.realpath(cur))} is still {st['state']}{pid}: {how}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir, k = os.path.join(base, "runs", stamp), 1
    while os.path.exists(run_dir):
        k += 1
        run_dir = os.path.join(base, "runs", f"{stamp}-{k}")
    os.makedirs(run_dir)
    tmp = cur + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.relpath(run_dir, base), tmp)
    os.replace(tmp, cur)
    opts = [x for b in a.breaks or [] for x in ("--break", b)]
    opts += ["--pause-each-step"] * a.pause_each_step + ["--unattended"] * a.unattended
    opts += [x for flag, v in (("--from", a.start), ("--only", a.only)) if v for x in (flag, v)]
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (pkg_parent, os.environ.get("PYTHONPATH")) if p))
    # -P: the working directory is not on the runner's sys.path, so a program named inspect.py or code.py does not
    # stand in for the standard module the runner imports (the runner adds the program's directory itself)
    argv = [sys.executable, "-P", "-m", "robot.runner", run_dir, task, *opts]
    with open(os.path.join(run_dir, "log.txt"), "ab") as log:
        if PROGRAM_HOST:
            rep = _host({"op": "start", "argv": argv, "env": env, "cwd": os.getcwd(), "run_dir": run_dir},
                        log.fileno())
            if not rep.get("ok"):
                raise SystemExit(f"cannot start the program: {rep.get('error')}")
            proc = None
        else:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
    print(f"--- run {_rel(run_dir)}" + (f" (pid {proc.pid})" if proc else ""))
    return _report(run_dir, _wait(run_dir, a.wait, proc), a.wait)


def cmd_wait(a) -> int:
    run_dir = _current()
    return _report(run_dir, _wait(run_dir, a.wait), a.wait)


def cmd_pause(a) -> int:
    run_dir = _current()
    st = _status(run_dir) or {}
    if st.get("state") == "paused":
        print("the program is paused already")
        return _report(run_dir, st, 0)
    if st.get("state") != "running" or not _alive(st):
        print("the program is not running; python -m robot run starts a new run")
        return 1
    if st.get("unattended"):
        print("an --unattended run does not pause; python -m robot stop ends it")
        return 1
    if st.get("host") != socket.gethostname():
        print(f"the program runs on {st.get('host')}; run python -m robot pause there")
        return 1
    if PROGRAM_HOST:
        rep = _host({"op": "pause", "run_dir": run_dir})
        if not rep.get("ok"):
            print(f"cannot pause the program: {rep.get('error')}")
            return 1
    else:
        os.kill(st["pid"], signal.SIGUSR1)
    return _report(run_dir, _wait(run_dir, a.wait), a.wait)


def cmd_stop(a) -> int:
    run_dir = _current()
    st = _status(run_dir) or {}
    if st.get("state") not in ("running", "paused") or not _alive(st):
        print("the program is not running; python -m robot run starts a new run")
        return 1
    if st.get("host") != socket.gethostname():
        print(f"the program runs on {st.get('host')}; run python -m robot stop there")
        return 1
    if PROGRAM_HOST:
        rep = _host({"op": "stop", "run_dir": run_dir})
        if not rep.get("ok"):
            print(f"cannot stop the program: {rep.get('error')}")
            return 1
    else:
        try:
            os.kill(st["pid"], signal.SIGTERM)
            t0 = time.time()
            while _alive(st) and time.time() - t0 < STOP_GRACE:
                time.sleep(0.05)
            if _alive(st):
                os.kill(st["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    return _report(run_dir, _wait(run_dir, a.wait), a.wait)


def cmd_debug(a) -> int:
    run_dir = _current()
    command = " ".join(a.command)
    st = _status(run_dir) or {}
    if st.get("state") != "paused":
        if st.get("state") == "exited":
            print(f"the program has exited (code {st.get('exit_code')}); python -m robot run starts a new run")
        else:
            print("the program is running, not paused: python -m robot pause pauses it, python -m robot wait "
                  "waits for a pause or its end")
        return 1
    if st.get("command"):
        print(f"the program is still busy with {st['command']!r}: python -m robot wait waits until it is done")
        return 1
    if st.get("host") != socket.gethostname():
        print(f"the program runs on {st.get('host')}; run python -m robot debug there")
        return 1
    t0 = time.time()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    try:
        s.connect(st["socket"])
        s.sendall(command.encode())
        s.shutdown(socket.SHUT_WR)
    except OSError as e:
        print(f"cannot reach the program ({e})")
        return _report(run_dir, _wait(run_dir, 0), 0)
    how, tail = None, b""
    while True:
        s.settimeout(max(0.05, t0 + a.wait - time.time()))
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            sys.stdout.write(tail.decode("utf-8", "replace"))
            print(f"\n--- the command is still running after {a.wait:.0f} s; what it prints from now on goes to the "
                  "log (python -m robot wait waits until it is done)")
            return 0
        if not chunk:
            break
        data = tail + chunk
        out, nul, rest = data.partition(b"\0")
        if nul:
            how = rest
        keep = 0 if nul else 3                     # a UTF-8 character may be split between chunks
        text, tail = (out, b"") if nul or len(out) <= keep else (out[:-keep], out[-keep:])
        sys.stdout.write(text.decode("utf-8", "replace"))
        sys.stdout.flush()
        if nul:
            break
    sys.stdout.write(tail.decode("utf-8", "replace"))
    if how == b"paused":
        return 0
    # it resumed (or went away): wait as `wait` does
    return _report(run_dir, _wait(run_dir, max(0.0, t0 + a.wait - time.time())), a.wait)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot", description=__doc__.split("\n\n")[0],
                                 epilog="See docs/MANUAL.md, 'Writing a program'.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="start a program")
    run.add_argument("task")
    run.add_argument("--wait", type=float, default=60, metavar="S", help="return after S s at the latest (60)")
    run.add_argument("--break", dest="breaks", action="append", metavar="STEP|FILE:LINE",
                     help="pause before that step, or at that line")
    run.add_argument("--pause-each-step", action="store_true", help="pause after each step")
    run.add_argument("--from", dest="start", metavar="STEP", help="start at that step")
    run.add_argument("--only", metavar="STEP", help="run only that step")
    run.add_argument("--unattended", action="store_true",
                     help="no pauses: a failed assert raises AssertionError; an exception ends the program")
    wait = sub.add_parser("wait", help="wait on the current run")
    wait.add_argument("--wait", type=float, default=60, metavar="S")
    pause = sub.add_parser("pause", help="pause the running program at the next line it runs in its own files")
    pause.add_argument("--wait", type=float, default=60, metavar="S")
    stop = sub.add_parser("stop", help="end the program, running or paused")
    stop.add_argument("--wait", type=float, default=60, metavar="S")
    debug = sub.add_parser("debug", help="send one pdb command to the paused program")
    debug.add_argument("command", nargs="+")
    debug.add_argument("--wait", type=float, default=60, metavar="S")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "wait": cmd_wait, "pause": cmd_pause, "stop": cmd_stop, "debug": cmd_debug}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
