"""The program host: it starts the programs `python -m robot run` hands it and keeps them for the whole episode.

Codex runs each agent command in a sandbox of its own, with a PID namespace that ends with the command, so a program
the command started itself would die with it. sandbox_init.py starts this host in the agent's outer sandbox, before
the agent, and `python -m robot run` (robot/__main__.py) asks it to start the program when ROBOT_PROGRAM_HOST names
SOCKET. Later commands (wait, debug, pause) find the program through its run directory and, for what needs its
process (is it alive, SIGUSR1 for pause), through this host.

A program runs as an agent command does: SANDBOX... (`codex sandbox -P robot -C /workspace`: the same permission
profile and writable roots) runs `env -i -C CWD ENV... ARGV...`, i.e. the asking command's argv, environment (less
the proxy variables) and working directory, applied inside that sandbox; its stdout and stderr are the run's log,
which the command opened and passed over the socket (SCM_RIGHTS), so the host opens no path of anyone's choosing.
Only SANDBOX itself runs with the host's environment (the outer sandbox's, less the proxy variables).

Requests, one JSON object per connection, one JSON reply:
    {"op": "start", "argv": [...], "env": {...}, "cwd": DIR, "run_dir": DIR}  + the log's fd  -> {"ok", "pid"}
    {"op": "poll", "run_dir": DIR}                         -> {"ok", "running", "returncode"}
    {"op": "pause", "run_dir": DIR}                        -> {"ok"}: SIGUSR1 to the program's runner process
    {"op": "stop", "run_dir": DIR}                         -> {"ok", "signal", "returncode"} once it has ended
Stopping a program is SIGTERM to its runner, then SIGKILL to all its processes after GRACE s. On SIGTERM
(sandbox_init.py, when the agent has ended) the host stops every program still running. It writes one JSON line per
event (listening, start, exit, stop, end) on stdout, which sandbox_init.py points at a log only the launcher reads.

    python3 program_host.py SOCKET SANDBOX...

Standard library only: it runs with the system Python.
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

GRACE = 5.0
PROXY_VARS = {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
MAX_REQUEST = 1 << 20


class _Stop(Exception):
    pass


def _children() -> dict:
    """pid -> [child pids], for every process this host can see."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        out.setdefault(ppid, []).append(int(name))
    return out


def _tree(pid: int) -> list:
    """pid and its descendants, breadth first."""
    kids, out, i = _children(), [pid], 0
    while i < len(out):
        out += kids.get(out[i], [])
        i += 1
    return out


def _cmdline(pid: int) -> list:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x.decode("utf-8", "surrogateescape") for x in f.read().split(b"\0")[:-1]]
    except OSError:
        return []


class Program:
    def __init__(self, run_dir: str, argv: list, proc: subprocess.Popen):
        self.run_dir, self.argv, self.proc = run_dir, argv, proc
        self.reported = False           # its exit is in the log

    def runner(self, wait: float):
        """The runner's process (`python -m robot.runner ...`, exec'd by env inside the sandbox): the shallowest
        process from the sandbox command down whose command line is argv, looked for up to wait s (env may not have
        exec'd it yet)."""
        t0 = time.time()
        while True:
            for pid in _tree(self.proc.pid):
                if _cmdline(pid) == self.argv:
                    return pid
            if self.proc.poll() is not None or time.time() - t0 > wait:
                return None
            time.sleep(0.05)

    def term(self):
        """SIGTERM to the runner: it ends as at q, and the driver stops what it started."""
        pid = self.runner(1.0)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def finish(self, deadline: float) -> str:
        """Wait until deadline for the sandbox command to exit, else SIGKILL every process in it; how it ended."""
        try:
            self.proc.wait(max(0.0, deadline - time.time()))
            return "SIGTERM"
        except subprocess.TimeoutExpired:
            pass
        for pid in reversed(_tree(self.proc.pid)):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(10)
            return "SIGKILL"
        except subprocess.TimeoutExpired:
            return "SIGKILL, still there 10 s later"


class Host:
    def __init__(self, path: str, sandbox: list):
        self.path, self.sandbox = path, sandbox
        self.env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
        self.programs = {}
        self.lock, self.log_lock = threading.Lock(), threading.Lock()
        self.stopping = False
        if os.path.exists(path):
            os.remove(path)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(16)

    def log(self, **entry):
        with self.log_lock:
            sys.stdout.write(json.dumps({"wall": round(time.time(), 3), **entry}) + "\n")
            sys.stdout.flush()

    # ---- requests
    def serve(self):
        self.log(event="listening", socket=self.path, sandbox=self.sandbox)
        while True:
            conn, _ = self.srv.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        fds = []
        try:
            conn.settimeout(10)
            data, fds, _, _ = socket.recv_fds(conn, 65536, 4)
            while len(data) < MAX_REQUEST and (chunk := conn.recv(65536)):
                data += chunk
            req = json.loads(data)
            op = req.get("op") if isinstance(req, dict) else None
            if op == "start":
                rep = self.start(req, fds)
            elif op == "poll":
                rep = self.poll(req.get("run_dir"))
            elif op == "pause":
                rep = self.pause(req.get("run_dir"))
            elif op == "stop":
                rep = self.stop(req.get("run_dir"))
            else:
                rep = {"ok": False, "error": f"unknown request {op!r}"}
        except Exception as e:                   # noqa: BLE001 -- a bad request gets an answer, not a dead host
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            for fd in fds:
                os.close(fd)
        try:
            conn.sendall(json.dumps(rep).encode())
        except OSError:
            pass
        conn.close()

    def start(self, req: dict, fds: list) -> dict:
        argv, env, cwd, run_dir = req.get("argv"), req.get("env"), req.get("cwd"), req.get("run_dir")
        if len(fds) != 1:
            return {"ok": False, "error": "start needs the log's file descriptor"}
        if not (isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv) and "=" not in argv[0]):
            return {"ok": False, "error": "argv: a list of strings, the first without '='"}
        if not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
            return {"ok": False, "error": "env: a mapping of strings"}
        if not (isinstance(cwd, str) and isinstance(run_dir, str)):
            return {"ok": False, "error": "cwd and run_dir: strings"}
        assign = [f"{k}={v}" for k, v in env.items() if NAME.match(k) and k not in PROXY_VARS]
        cmd = [*self.sandbox, "--", "/usr/bin/env", "-i", "-C", cwd, *assign, *argv]
        with self.lock:
            if self.stopping:
                return {"ok": False, "error": "the episode is ending"}
            old = self.programs.get(run_dir)
            if old is not None and old.proc.poll() is None:
                return {"ok": False, "error": f"{run_dir} has a program running"}
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fds[0], stderr=fds[0], env=self.env,
                                    start_new_session=True)
            self.programs[run_dir] = Program(run_dir, argv, proc)
        self.log(event="start", run_dir=run_dir, pid=proc.pid, cwd=cwd, argv=argv)
        return {"ok": True, "pid": proc.pid}

    def poll(self, run_dir) -> dict:
        p = self.programs.get(run_dir)
        if p is None:
            return {"ok": True, "running": False, "returncode": None, "known": False}
        with self.lock:
            rc = p.proc.poll()
            report = rc is not None and not p.reported
            p.reported = p.reported or report
        if report:
            self.log(event="exit", run_dir=run_dir, returncode=rc)
        return {"ok": True, "running": rc is None, "returncode": rc}

    def pause(self, run_dir) -> dict:
        p = self.programs.get(run_dir)
        if p is None or p.proc.poll() is not None:
            return {"ok": False, "error": "the program is not running"}
        pid = p.runner(5.0)
        if pid is None:
            return {"ok": False, "error": "the program's runner process was not found"}
        os.kill(pid, signal.SIGUSR1)
        return {"ok": True}

    def stop(self, run_dir) -> dict:
        p = self.programs.get(run_dir)
        if p is None or p.proc.poll() is not None:
            return {"ok": False, "error": "the program is not running"}
        p.term()
        how = p.finish(time.time() + GRACE)
        p.reported = True
        self.log(event="stop", run_dir=run_dir, signal=how, returncode=p.proc.returncode, asked=True)
        return {"ok": True, "signal": how, "returncode": p.proc.returncode}

    # ---- the episode's end
    def stop_all(self):
        with self.lock:
            self.stopping = True
            programs = list(self.programs.values())
        stopped = []
        for p in programs:
            if p.proc.poll() is not None:
                self.poll(p.run_dir)
                continue
            p.term()
            stopped.append(p)
        deadline = time.time() + GRACE
        out = []
        for p in stopped:
            how = p.finish(deadline)
            self.log(event="stop", run_dir=p.run_dir, signal=how, returncode=p.proc.returncode)
            out.append({"run_dir": p.run_dir, "signal": how, "returncode": p.proc.returncode})
        self.log(event="end", started=len(programs), stopped=out)


def main():
    path, sandbox = sys.argv[1], sys.argv[2:]
    host = Host(path, sandbox)

    def on_term(signum, frame):
        raise _Stop()
    signal.signal(signal.SIGTERM, on_term)
    try:
        host.serve()
    except _Stop:
        pass
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    host.srv.close()
    host.stop_all()


if __name__ == "__main__":
    main()
