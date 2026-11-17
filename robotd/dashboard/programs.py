"""The programs the dashboard runs: Python files that drive the robot through robotd like any client.

Two sources: your programs (a directory you can add to, edit and delete from in the dashboard) and the repository's
examples (read-only; copy one to make it yours). One program runs at a time, as its own process (its own process
group) with ROBOT_ADDR pointing at the driver, so it can never hold up the control loop, and when it ends, however it
ends, the driver stops the motions it started and the arms hold. Stopping sends Ctrl-C first (the program's finally
blocks run), then SIGTERM, then SIGKILL.
"""
from __future__ import annotations

import ast
import collections
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}\.py$")
MAX_SOURCE = 256 * 1024
NEW_PROGRAM = '''"""What this program does, in one line (the dashboard shows it)."""
import robot

r = robot.connect(name="{stem}")
s = r.state()
print("connected; left TCP at", s.ee["left"].T[:3, 3].round(3))
'''


class ProgramError(ValueError):
    pass


class ProgramManager:
    def __init__(self, programs_dir: str, examples_dir: str | None, addr: str, python: str = sys.executable,
                 on_event=None):
        self.dirs = {"mine": os.path.abspath(os.path.expanduser(programs_dir))}
        if examples_dir and os.path.isdir(examples_dir):
            self.dirs["example"] = os.path.abspath(examples_dir)
        os.makedirs(self.dirs["mine"], exist_ok=True)
        self.addr, self.python = addr, python
        self.on_event = on_event or (lambda kind, text, **info: None)
        self.lock = threading.RLock()                     # re-entrant: run() logs its command line under it
        self.proc = None
        self.running = None                   # {name, origin, args, pid, started}
        self.last = None                      # {name, origin, code, ended, runtime, stopped_by}
        self.stopping = None
        self.lines = collections.deque(maxlen=5000)
        self.seq = 0

    # ------------------------------------------------------------------ files
    def _path(self, name: str, origin: str = "mine") -> str:
        if origin not in self.dirs:
            raise ProgramError(f"unknown program source {origin!r}")
        if not NAME.match(name or ""):
            raise ProgramError("a program name is letters, digits, _ and -, ending in .py (no folders)")
        return os.path.join(self.dirs[origin], name)

    def list(self) -> list:
        out = []
        for origin, d in self.dirs.items():
            for name in sorted(os.listdir(d)):
                path = os.path.join(d, name)
                if not NAME.match(name) or not os.path.isfile(path):
                    continue
                st = os.stat(path)
                out.append({"name": name, "origin": origin, "size": st.st_size, "mtime": st.st_mtime,
                            "doc": _first_doc_line(path)})
        return out

    def source(self, name: str, origin: str = "mine") -> str:
        path = self._path(name, origin)
        if not os.path.isfile(path):
            raise ProgramError(f"no program {name}")
        with open(path) as f:
            return f.read()

    def save(self, name: str, text: str | None = None) -> dict:
        path = self._path(name, "mine")
        if text is None:                                  # a new program from the template
            if os.path.exists(path):
                raise ProgramError(f"{name} exists already")
            text = NEW_PROGRAM.format(stem=name[:-3])
        if len(text.encode()) > MAX_SOURCE:
            raise ProgramError(f"too large (over {MAX_SOURCE // 1024} KB)")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        problem = _syntax_problem(text)
        self.on_event("program", f"saved {name}" + (f" (syntax: {problem})" if problem else ""))
        return {"name": name, "syntax": problem}

    def delete(self, name: str) -> dict:
        path = self._path(name, "mine")
        if self.running and self.running["name"] == name and self.running["origin"] == "mine":
            raise ProgramError(f"{name} is running: stop it first")
        os.remove(path)
        self.on_event("program", f"deleted {name}")
        return {"name": name}

    def copy(self, name: str, origin: str, new_name: str | None = None) -> dict:
        text = self.source(name, origin)
        new_name = new_name or name
        if os.path.exists(self._path(new_name, "mine")):
            raise ProgramError(f"{new_name} exists already in your programs")
        return self.save(new_name, text)

    # ------------------------------------------------------------------ running
    def run(self, name: str, origin: str = "mine", args: str = "") -> dict:
        path = self._path(name, origin)
        if not os.path.isfile(path):
            raise ProgramError(f"no program {name}")
        argv = shlex.split(args or "")
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise ProgramError(f"{self.running['name']} is running: stop it first (one program at a time)")
            env = {**os.environ, "ROBOT_ADDR": self.addr, "PYTHONUNBUFFERED": "1"}
            self.proc = subprocess.Popen([self.python, "-u", path, *argv], cwd=self.dirs["mine"], env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, bufsize=1, start_new_session=True)
            self.running = {"name": name, "origin": origin, "args": args or "", "pid": self.proc.pid,
                            "started": time.time()}
            self.stopping = None
            self._line("system", f"$ python {name} {args or ''}".rstrip())
            proc = self.proc
        for stream, pipe in (("out", proc.stdout), ("err", proc.stderr)):
            threading.Thread(target=self._read, args=(stream, pipe), daemon=True).start()
        threading.Thread(target=self._wait, args=(proc,), daemon=True).start()
        self.on_event("program", f"started {name}" + (f" {args}" if args else ""), pid=proc.pid)
        return dict(self.running)

    def stop(self, why: str = "stopped from the dashboard") -> dict:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return {"stopped": False, "why": "nothing is running"}
            if self.stopping:
                return {"stopped": True, "why": self.stopping}
            self.stopping = why
        self._line("system", f"stopping: {why}")
        threading.Thread(target=self._escalate, args=(proc,), daemon=True).start()
        return {"stopped": True, "why": why}

    def _escalate(self, proc):
        for sig, wait in ((signal.SIGINT, 3.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def _read(self, stream, pipe):
        for line in iter(pipe.readline, ""):
            self._line(stream, line.rstrip("\n"))
        pipe.close()

    def _wait(self, proc):
        code = proc.wait()
        time.sleep(0.05)                                   # let the readers drain the last lines
        with self.lock:
            run = self.running or {}
            self.last = {**run, "code": code, "ended": time.time(),
                         "runtime": time.time() - run.get("started", time.time()), "stopped_by": self.stopping}
            if proc is self.proc:
                self.running = None
        how = f"stopped ({self.stopping})" if self.stopping else f"exited with code {code}"
        self._line("system", f"[{run.get('name')} {how}]")
        self.on_event("program", f"{run.get('name')} {how}", code=code)

    def _line(self, stream, text):
        with self.lock:
            self.seq += 1
            self.lines.append({"seq": self.seq, "t": time.time(), "stream": stream, "text": text})

    # ------------------------------------------------------------------ reading
    def status(self) -> dict:
        with self.lock:
            run = dict(self.running) if self.running else None
            if run:
                run["runtime"] = time.time() - run["started"]
                run["stopping"] = self.stopping
            return {"running": run, "last": dict(self.last) if self.last else None, "out_seq": self.seq}

    def output(self, since: int = 0) -> dict:
        with self.lock:
            return {"seq": self.seq, "lines": [l for l in self.lines if l["seq"] > since][-2000:]}


def _first_doc_line(path: str) -> str:
    try:
        with open(path) as f:
            doc = ast.get_docstring(ast.parse(f.read()))
        return (doc or "").strip().splitlines()[0] if doc else ""
    except (OSError, SyntaxError, ValueError):
        return ""


def _syntax_problem(text: str) -> str | None:
    try:
        ast.parse(text)
        return None
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
