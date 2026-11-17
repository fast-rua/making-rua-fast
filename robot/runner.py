"""The program runner: what runs in the process `python -m robot run task.py` starts (robot/__main__.py).

A program is one file that defines step functions `def name(r, v, s)` and lists them in `STEPS`, with no work at
import time. The runner imports it, connects `r` (the robot) and `v` (the vision client, or None), and calls the steps
in order with one `s` (a SimpleNamespace) for the whole run.

    python -m robot.runner RUN_DIR TASK.py [--break STEP|FILE:LINE ...] [--pause-each-step] [--from STEP]
                           [--only STEP] [--unattended]

- Asserts in the task's own files (the task file and the modules it imports from its directory tree) are rewritten
  at import, the way pytest does: `assert test, msg` becomes `if not test: <pause>(msg, ...)` on the same lines, so a
  failed check pauses in that frame before anything is raised, and `c` goes on with the next line.
- Any other exception escaping a step pauses post-mortem. --break STEP, --break FILE:LINE (pdb breakpoints),
  --pause-each-step and breakpoint() pause too, and SIGUSR1 (python -m robot pause) pauses at the next line run in
  the task's own files: a robot request in progress finishes first.
- A pause saves pause-N/ (every camera's rgb and depth, K and T, r.state(), the message), prints a pause block to
  the log (stdout), updates status.json, then serves pdb over a unix socket: each connection carries one command and
  gets its output (python -m robot debug 'CMD').
- reload, retry, skip and goto are extra debugger commands. retry, skip, goto and q unwind to the step loop as
  runner-private exceptions, so they work the same at an assert, at a breakpoint and post-mortem.
"""
from __future__ import annotations

import argparse
import ast
import bdb
import ctypes
import faulthandler
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import linecache
import os
import pdb
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
import types

import robot
from robot.images import depth_mm, write_png

ASSERT_HOOK = "@robot_assert"            # the name rewritten asserts call (it cannot clash with a Python name)
_ACTIVE = None                           # the Runner of this process
_PKG = os.path.dirname(os.path.realpath(__file__)) + os.sep

try:                                     # 3.11/3.12: what a trace function's return does, for pauses called directly
    _LocalsToFast = ctypes.pythonapi.PyFrame_LocalsToFast
    _LocalsToFast.argtypes, _LocalsToFast.restype = [ctypes.py_object, ctypes.c_int], None
except AttributeError:                   # 3.13+: frame.f_locals writes through
    _LocalsToFast = None


class RunnerError(Exception):
    pass


class _Control(BaseException):
    """Unwinds to the step loop. A BaseException, so a step's own `except Exception` does not catch it."""


class _Retry(_Control):
    pass


class _Skip(_Control):
    pass


class _Goto(_Control):
    def __init__(self, name):
        super().__init__(name)
        self.name = name


class _Quit(_Control):
    pass


# ------------------------------------------------------------------------------------------------ assert rewriting
def rewrite_asserts(source: str, path: str) -> ast.Module:
    """The module's AST with each `assert test, msg` turned into `if not test: @robot_assert(msg, "test", next)`,
    on the assert's own lines (next: the line of the statement after it in its block, or None)."""
    tree = ast.parse(source, filename=path)
    following = {}
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                for a, b in zip(stmts, stmts[1:] + [None]):
                    if isinstance(a, ast.Assert):
                        following[id(a)] = None if b is None else b.lineno

    class Rewrite(ast.NodeTransformer):
        def visit_Assert(self, node):
            text = " ".join((ast.get_source_segment(source, node.test) or ast.unparse(node.test)).split())
            msg = node.msg if node.msg is not None else ast.Constant(None)
            call = ast.Call(ast.Name(ASSERT_HOOK, ast.Load()),
                            [msg, ast.Constant(text), ast.Constant(following.get(id(node)))], [])
            new = ast.If(ast.UnaryOp(ast.Not(), node.test), [ast.Expr(call)], [])
            for n in (new, new.test, new.body[0], call, call.func, *call.args[1:]):
                ast.copy_location(n, node)
            if node.msg is None:
                ast.copy_location(msg, node)
            return new
    return ast.fix_missing_locations(Rewrite().visit(tree))


def _assert_failed(msg, test, next_line):
    """A rewritten assert's check failed: pause in its frame. c returns from here, so the next line runs."""
    run = _ACTIVE
    if run is None or run.dbg is None or run.dbg.depth:      # no debugger, or inside a debugger command: raise
        raise AssertionError(msg) if msg is not None else AssertionError()
    then = f"line {next_line}" if next_line else "the code after it"
    run.dbg.pause_live(sys._getframe(1), "assert", str(msg) if msg is not None else f"assert {test}",
                       f"c goes on with {then} as if the check had passed")


# ------------------------------------------------------------------------------------------------ the task's modules
class _Loader(importlib.abc.Loader):
    def __init__(self, runner, path):
        self.runner, self.path = runner, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.runner._exec(module, self.runner._compile(self.path))

    def get_source(self, name):
        with open(self.path, encoding="utf-8") as f:
            return f.read()


class _Finder(importlib.abc.MetaPathFinder):
    """Modules found under the task's directory load through _Loader (asserts rewritten, reloadable)."""

    def __init__(self, runner):
        self.runner = runner

    def find_spec(self, name, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not (spec.origin or "").endswith(".py") or not self.runner._own(spec.origin):
            return None
        spec.loader = _Loader(self.runner, spec.origin)
        return spec


# ------------------------------------------------------------------------------------------------ the debugger's I/O
class _Channel:
    """pdb's stdin and stdout. Each connection sends one command and gets that command's output, then a NUL and
    'paused' (pdb asks for the next command) or 'resumed' (the program goes on). Output with no connection to take
    it (before the first command, or after the client gave up waiting) goes to the log."""

    def __init__(self, runner, path):
        self.runner, self.path = runner, path
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.srv.bind(path)
        except OSError:
            self.srv.close()
            raise
        self.srv.listen(4)
        self.conn = None
        self.transcript = open(os.path.join(runner.run_dir, "debug.txt"), "a", encoding="utf-8")

    def readline(self):
        self.runner.status(command=None)
        self.end(b"paused")
        while True:
            conn, _ = self.srv.accept()
            conn.settimeout(10)
            data = b""
            try:
                while chunk := conn.recv(65536):
                    data += chunk
            except OSError:
                conn.close()
                continue
            conn.settimeout(None)
            break
        line = data.decode("utf-8", "replace").rstrip("\r\n")
        self.conn = conn
        self.runner.status(command=line)
        self.transcript.write(f"(Pdb) {line}\n")
        self.transcript.flush()
        return line + "\n"

    def write(self, text):
        self.transcript.write(text)
        if self.conn is not None:
            try:
                self.conn.sendall(text.encode("utf-8", "replace"))
                return
            except OSError:
                self.conn.close()
                self.conn = None
        self.runner.log.write(text)

    def flush(self):
        self.transcript.flush()
        self.runner.log.flush()

    def end(self, how: bytes):
        if self.conn is not None:
            try:
                self.conn.sendall(b"\0" + how)
            except OSError:
                pass
            self.conn.close()
            self.conn = None

    def close(self):
        self.end(b"exited")
        self.srv.close()
        self.transcript.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------ the debugger
class Debugger(pdb.Pdb):
    """pdb with r, v and s in every command's namespace, reload/retry/skip/goto, and no prompt (one command per
    connection). Pauses come from a failed assert or breakpoint() (pause_live), a pdb breakpoint or a stepping
    command (the trace function), the start of a step (--break STEP), the end of one (pause_after) and an exception
    (post_mortem)."""

    def __init__(self, runner, channel):
        super().__init__(stdin=channel, stdout=channel, skip=[__name__], nosigint=True, readrc=False)
        self.prompt = ""
        self.runner = runner
        self.depth = 0              # > 0 while a pause is being served
        self.why = None             # (kind, message, note) of the next pause
        self.mode = "live"          # live | after (the step has returned) | post_mortem
        self.action = None          # the _Control to raise when the pause ends
        self.last = None            # the last resuming command
        self.entry = None           # code object: stop at its first line when next called (--break STEP)
        self.entry_frame = None
        self.capture = None         # code object: keep its frame when next called (--pause-each-step)
        self.captured = None
        self.pause_requested = False    # python -m robot pause: stop at the next line run in the task's own files
        self.reset()

    # ---- where the program stops
    def quiet(self):
        """Stop only at breakpoints (the state between steps, and after c)."""
        self._set_stopinfo(self.botframe, None, -1)

    def dispatch_call(self, frame, arg):
        if frame.f_code is self.capture:
            self.capture, self.captured = None, frame
            if self.entry is None and not self.breaks:
                sys.settrace(None)              # nothing more to watch in this step
                return None
        if frame.f_code is self.entry:
            self.entry, self.entry_frame = None, frame
            self._set_stopinfo(frame, None, 0)  # its first line
            return self.trace_dispatch
        if self.pause_requested and self.runner._own(frame.f_code.co_filename):
            return self.trace_dispatch          # a pause can come at its lines
        return super().dispatch_call(frame, arg)

    def dispatch_line(self, frame):
        if self.pause_requested and self.runner._own(frame.f_code.co_filename):
            self.why = ("pause", "paused on request (python -m robot pause)", "")
            self.interaction(frame, None)
            return self.trace_dispatch
        return super().dispatch_line(frame)

    def request_pause(self, frame):
        """Trace the task's own frames from frame up, so their next line pauses. The robot client's frames are not
        traced: a request in progress gets its reply first."""
        self.pause_requested = True
        while frame is not None:
            if self.runner._own(frame.f_code.co_filename):
                frame.f_trace = self.trace_dispatch
            frame = frame.f_back
        sys.settrace(self.trace_dispatch)

    def user_line(self, frame):
        if frame is self.entry_frame:
            self.entry_frame = None
            self.why = ("break", f"before step {self.runner.step}", "")
        elif self.get_breaks(self.canonic(frame.f_code.co_filename), frame.f_lineno):
            self.why = ("breakpoint", f"breakpoint at line {frame.f_lineno}", "")
        else:
            self.why = ("stopped", f"after {self.last}", "")
        super().user_line(frame)

    def user_call(self, frame, argument_list):
        self.why = ("stopped", f"calling {frame.f_code.co_name}() (after {self.last})", "")
        super().user_call(frame, argument_list)

    def user_return(self, frame, return_value):
        self.why = ("stopped", f"returning from {frame.f_code.co_name}() (after {self.last})", "")
        super().user_return(frame, return_value)

    def user_exception(self, frame, exc_info):
        exc = traceback.format_exception_only(exc_info[0], exc_info[1])[-1].strip()
        self.why = ("stopped", f"{exc} raised (after {self.last})", "")
        super().user_exception(frame, exc_info)

    def get_stack(self, f, t):
        """The stack from the step function down: the runner's frames are not the program's."""
        stack, i = super().get_stack(f, t)
        for k, (frame, _) in enumerate(stack[:-1]):
            if frame.f_code is _CALL_STEP:
                return stack[k + 1:], max(0, i - k - 1)
        return stack, i

    def pause_live(self, frame, kind, message, note=""):
        """Pause in frame, called from the program's own code (a failed assert, breakpoint())."""
        if self.depth:                          # breakpoint() inside a debugger command
            return
        sys.settrace(None)                      # commands typed at the pause do not stop at breakpoints
        self.why = (kind, message, note)
        self.interaction(frame, None)
        if self.stoplineno != -1 or self.breaks:            # n, s, until, return, breakpoints: trace from here
            f = frame
            while f is not None:
                f.f_trace = self.trace_dispatch
                if f is self.botframe:
                    break
                f = f.f_back
            sys.settrace(self.trace_dispatch)

    def pause_after(self, frame, step, nxt):
        self.why = ("step", f"step {step} done", f"c goes on with step {nxt}" if nxt else "it was the last step: "
                    "c ends the program")
        self.mode = "after"
        try:
            self.interaction(frame, None)
        finally:
            self.mode = "live"

    def post_mortem(self, exc):
        self.mode = "post_mortem"
        try:
            while True:                         # only retry, skip, goto and q leave (they raise)
                self.why = ("exception", _format_exc(exc).rstrip(),
                            "the step cannot go on from here: retry, skip, goto STEP or q (c, n and s are refused)")
                self.interaction(None, exc.__traceback__)
        finally:
            self.mode = "live"

    def interaction(self, frame, tb):
        kind, message, note = self.why or ("stopped", "", "")
        self.why = None
        self.pause_requested = False
        self.setup(frame, tb)
        if tb is not None:          # post-mortem: select the innermost frame in the program's own files, where its
            for i in range(len(self.stack) - 1, -1, -1):    # names are, not the robot client's or a library's
                if self.runner._own(self.stack[i][0].f_code.co_filename):
                    self.curindex = i
                    self.curframe = self.stack[i][0]
                    self.curframe_locals = self.curframe.f_locals
                    break
        self.depth += 1
        try:
            self.runner.paused(self, kind, message, note)
            self._cmdloop()
        finally:
            self.depth -= 1
            self.forget()
        self.runner.resumed()
        action, self.action = self.action, None
        if action is not None:
            raise action

    # ---- names: r, v and s after the selected frame's own
    def onecmd(self, line):
        g = self.curframe.f_globals if self.curframe is not None else {}
        added = [k for k in self.runner.names if k not in g]
        for k in added:
            g[k] = self.runner.names[k]
        try:
            return super().onecmd(line)
        finally:
            for k in added:
                if g.get(k) is self.runner.names[k]:
                    del g[k]

    def postcmd(self, stop, line):
        # a pause called directly (an assert) has no trace function to write `!x = ...` back into the frame
        if self.mode == "live" and self.curframe is not None and _LocalsToFast is not None:
            _LocalsToFast(self.curframe, 0)
            self.curframe_locals = self.curframe.f_locals       # the same dict, marked for the next write-back
        return stop

    def emptyline(self):
        pass

    # ---- resuming
    def _resume(self, name, arg):
        if self.mode == "post_mortem":
            self.error(f"{name}: the step raised and cannot go on; use retry, skip, goto STEP or q")
            return None
        self.last = name
        if self.mode == "after" and name != "continue":     # n or s between steps: pause at the next one's start
            self.runner.break_next = True
            name = "continue"
        return getattr(pdb.Pdb, "do_" + name)(self, arg)

    def do_continue(self, arg):
        return self._resume("continue", arg)

    def do_next(self, arg):
        return self._resume("next", arg)

    def do_step(self, arg):
        return self._resume("step", arg)

    def do_until(self, arg):
        return self._resume("until", arg)

    def do_return(self, arg):
        return self._resume("return", arg)

    def do_jump(self, arg):
        if self.mode != "live":
            self.error("jump: only in a running frame")
            return None
        return super().do_jump(arg)

    do_c = do_cont = do_continue
    do_n = do_next
    do_s = do_step
    do_unt = do_until
    do_r = do_return
    do_j = do_jump

    def _reloaded(self, event: str = "reload") -> bool:
        err = self.runner.reload()
        if err:
            self.error(err)
            return False
        self.runner.snapshot(event)
        self.message(f"reloaded {', '.join(self.runner.files())}; STEPS = [{', '.join(self.runner.step_names())}]")
        return True

    def do_reload(self, arg):
        """reload
        Reread the task file and its local helper modules from disk and rebind STEPS. Helpers looked up from now on
        run the new code, also after c; the paused function itself runs on as it was."""
        self._reloaded()

    def do_retry(self, arg):
        """retry
        Reload, then run the current step again from its start (s is kept)."""
        if not self._reloaded("retry"):
            return None
        if self.runner.step not in self.runner.step_names():
            self.error(f"step {self.runner.step} is no longer in STEPS; goto one of: "
                       f"{', '.join(self.runner.step_names())}")
            return None
        self.action = _Retry()
        return 1

    def do_skip(self, arg):
        """skip
        Go on with the step after the current one."""
        self.action = _Skip()
        return 1

    def do_goto(self, arg):
        """goto STEP
        Reload, then go on from STEP (it may be a step added since the run started)."""
        name = arg.strip()
        if not name:
            self.error(f"goto STEP, one of: {', '.join(self.runner.step_names())}")
            return None
        if not self._reloaded(f"goto {name}"):
            return None
        if name not in self.runner.step_names():
            self.error(f"no step {name!r}; STEPS = [{', '.join(self.runner.step_names())}]")
            return None
        self.action = _Goto(name)
        return 1

    def do_quit(self, arg):
        """q(uit)
        End the program. Its process exits; the robot holds."""
        self.action = _Quit()
        return 1

    do_q = do_exit = do_quit

    def do_EOF(self, arg):
        pass

    def _not_here(self, arg):
        self.error("not available over python -m robot debug (one command per call); use !STMT, retry or goto")

    do_interact = do_debug = do_run = do_restart = do_commands = _not_here


for _name in ("continue", "next", "step", "until", "return", "jump"):
    getattr(Debugger, "do_" + _name).__doc__ = getattr(pdb.Pdb, "do_" + _name).__doc__
Debugger.do_continue.__doc__ += """After a failed assert, the next line runs as if the check had passed.
        Refused after an exception (the step cannot go on): use retry, skip, goto or q.
        """


# ------------------------------------------------------------------------------------------------ the runner
def _vision():
    """The vision client, or None when no vision service is configured ($VISION_ADDR)."""
    addr = os.environ.get("VISION_ADDR")
    if not addr:
        return None
    import vision                       # the vision service's client (not part of this package)
    return vision.connect(addr)


def _rel(path: str) -> str:
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def _format_exc(exc: BaseException) -> str:
    """A traceback without the runner's and importlib's frames."""
    te = traceback.TracebackException.from_exception(exc)
    te.stack = traceback.StackSummary.from_list(
        [f for f in te.stack if f.filename != __file__ and not f.filename.startswith("<frozen")])
    return "".join(te.format())


class Runner:
    def __init__(self, args):
        self.run_dir = os.path.abspath(args.run_dir)
        os.environ["ROBOT_RUN_DIR"] = self.run_dir         # robot.viz.show saves into RUN_DIR/views
        self.task = os.path.abspath(args.task)
        self.root = os.path.realpath(os.path.dirname(self.task))
        self.attended = not args.unattended
        self.breaks = args.breaks or []
        self.pause_each = args.pause_each_step and self.attended
        self.start, self.only = args.only or args.start, args.only
        self.log = sys.stdout
        self.st = {"state": "running", "pid": os.getpid(), "host": socket.gethostname(), "task": self.task,
                   "run_dir": self.run_dir, "started": time.time(), "pause": 0, "step": None, "command": None,
                   "unattended": not self.attended}
        self.mod = None
        self.order = []                 # the task's modules in the order they finished executing (helpers first)
        self.steps = []
        self.step = None
        self.break_next = False
        self.npause = 0
        self.r = self.v = None
        self.s = types.SimpleNamespace()
        self.names = {}
        self.dbg = self.chan = None
        self.sockdir = None
        self.early_pause = False        # a pause asked for before the debugger exists
        self.owns = {}                  # file -> one of the task's own files?

    # ---- status.json
    def status(self, **kw):
        self.st.update(kw, updated=time.time())
        path = os.path.join(self.run_dir, "status.json")
        with open(path + ".tmp", "w") as f:
            json.dump(self.st, f, indent=1, default=str)
        os.replace(path + ".tmp", path)

    # ---- loading
    def _own(self, path: str) -> bool:
        own = self.owns.get(path)
        if own is None:
            real = os.path.realpath(path)
            lib = (_PKG, *(os.path.realpath(p) + os.sep for p in {sys.prefix, sys.base_prefix}))
            own = self.owns[path] = real.startswith(self.root + os.sep) and "site-packages" not in real and \
                not real.startswith(lib)
        return own

    def _compile(self, path: str):
        with open(path, "rb") as f:
            source = importlib.util.decode_source(f.read())
        tree = rewrite_asserts(source, path) if self.attended else ast.parse(source, filename=path)
        return compile(tree, path, "exec", dont_inherit=True)

    def _exec(self, module, code):
        module.__dict__[ASSERT_HOOK] = _assert_failed
        exec(code, module.__dict__)
        if module not in self.order:
            self.order.append(module)

    def load(self):
        sys.path.insert(0, os.path.dirname(self.task))
        sys.meta_path.insert(0, _Finder(self))
        name = os.path.splitext(os.path.basename(self.task))[0]
        if name in sys.modules or not name.isidentifier():
            name = "__task__"
        spec = importlib.machinery.ModuleSpec(name, _Loader(self, self.task), origin=self.task)
        spec.has_location = True
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.mod
        spec.loader.exec_module(self.mod)
        self.steps = self._steps()

    def _steps(self) -> list:
        steps = getattr(self.mod, "STEPS", None)
        if not isinstance(steps, (list, tuple)) or not steps or not all(inspect.isfunction(f) for f in steps):
            raise RunnerError(f"{_rel(self.task)} must define STEPS, a list of step functions def name(r, v, s)")
        names = [f.__name__ for f in steps]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise RunnerError(f"STEPS lists {', '.join(dup)} more than once")
        return list(steps)

    def step_names(self) -> list:
        return [f.__name__ for f in self.steps]

    def files(self) -> list:
        return [os.path.basename(m.__file__) for m in self.order]

    def snapshot(self, event: str) -> None:
        """Keep the program's files as they are now: src/<sha>.py, and a line in src/index.jsonl (the event, the step,
        the pause count, file -> sha), at the start and at every reload, retry and goto, so each version the run
        executed can be shown and diffed."""
        try:
            out = os.path.join(self.run_dir, "src")
            os.makedirs(out, exist_ok=True)
            files = {}
            for m in self.order:
                with open(m.__file__, "rb") as f:
                    data = f.read()
                sha = hashlib.sha256(data).hexdigest()[:16]
                path = os.path.join(out, f"{sha}.py")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(data)
                files[_rel(m.__file__)] = sha
            with open(os.path.join(out, "index.jsonl"), "a") as f:
                f.write(json.dumps({"wall": round(time.time(), 3), "event": event, "step": self.step,
                                    "pause": self.npause, "files": files}) + "\n")
        except OSError:
            pass                                # a full disk must not stop the program

    def reload(self) -> str | None:
        """Re-execute the task's modules from disk in their own namespaces, helpers first and the task last, so names
        the task imported from a helper are bound to the new code. Returns an error message, or None."""
        mods = list(self.order)
        try:
            codes = [self._compile(m.__file__) for m in mods]
        except (SyntaxError, OSError) as e:
            return f"reload failed, nothing changed:\n{_format_exc(e)}".rstrip()
        linecache.checkcache()
        self.order = []
        try:
            for m, code in zip(mods, codes):
                self._exec(m, code)
            self.steps = self._steps()
        except Exception as e:                  # noqa: BLE001 -- shown to the agent, who fixes it and reloads
            for m in mods:
                if m not in self.order:
                    self.order.append(m)
            return f"reload failed part way (fix it and reload again):\n{_format_exc(e)}".rstrip()
        return None

    def _index(self, name: str, fallback: int) -> int:
        names = self.step_names()
        return names.index(name) if name in names else fallback

    # ---- pauses
    def paused(self, dbg, kind, message, note):
        self.npause += 1
        n = self.npause
        frame, line = dbg.stack[dbg.curindex]
        path, func = frame.f_code.co_filename, frame.f_code.co_name
        pdir = os.path.join(self.run_dir, f"pause-{n}")
        os.makedirs(pdir, exist_ok=True)
        f = dbg.stack[-1][0] if dbg.mode == "live" else None
        while f is not None and not (f.f_code.co_filename.startswith(_PKG) and f.f_code.co_filename != __file__):
            f = f.f_back
        # stopped inside the robot client (stepping into r.move), maybe between a request and its reply: no requests;
        # after n, s, until or return: r.state() but no camera frames
        saved = self._save(pdir, frames=kind != "stopped") if f is None else "nothing (paused inside the robot client)"
        names = self.step_names()
        where = f"step {self.step} {names.index(self.step) + 1}/{len(names)}" if self.step in names else \
            f"step {self.step}"
        titles = {"assert": "assert failed", "exception": "exception", "break": "break", "breakpoint": "breakpoint",
                  "step": "after the step", "stopped": "stopped", "pause": "paused"}
        code = linecache.getline(path, line, frame.f_globals).rstrip()
        lines = [f"--- pause {n}: {titles[kind]} at {_rel(path)}:{line} in {func} ({where})",
                 f"    {line:<5d}{code.strip()}" if code else None,
                 "\n".join("    " + x for x in message.splitlines()),
                 f"    {note}" if note else None,
                 f"    saved {_rel(pdir)}/: {saved}",
                 "    python -m robot debug 'CMD', CMD one of: p EXPR, !STMT, where, up, down, l, "
                 + {"post_mortem": "", "after": "c, n (pause at the next step's start), "}.get(dbg.mode, "n, s, c, ")
                 + "reload, retry, skip, goto STEP, q"]
        block = "\n".join(x for x in lines if x is not None)
        with open(os.path.join(pdir, "message.txt"), "w") as f:
            f.write(block + "\n")
        self.log.write(block + "\n")
        self.log.flush()
        self.status(state="paused", pause=n, kind=kind, message=message.splitlines()[-1], file=path, line=line,
                    function=func, pause_dir=pdir, block=block, command=None)

    def _save(self, pdir: str, frames: bool) -> str:
        """Every camera's rgb and depth (16-bit PNG, mm), K and T (when frames), and r.state(), into pdir."""
        out = []
        if frames:
            try:
                cams = self.r.cameras()
                meta = {}
                for name, f in cams.items():
                    write_png(os.path.join(pdir, f"{name}.png"), f.rgb)
                    if f.depth is not None:
                        write_png(os.path.join(pdir, f"{name}_depth.png"), depth_mm(f.depth))
                    meta[name] = {"K": f.K.tolist(), "T": None if f.T is None else f.T.tolist(), "t": f.t,
                                  "depth_png_unit_m": 0.001}
                with open(os.path.join(pdir, "cameras.json"), "w") as fp:
                    json.dump(meta, fp, indent=1)
                out.append(f"{', '.join(cams)} (NAME.png, NAME_depth.png in mm), cameras.json (K, T)")
            except Exception as e:              # noqa: BLE001 -- a pause must not fail on its pictures
                out.append(f"no camera frames ({type(e).__name__}: {e})")
        try:
            with open(os.path.join(pdir, "state.json"), "w") as fp:
                json.dump(self.r.state().raw, fp, default=str)
            out.append("state.json")
        except Exception as e:                  # noqa: BLE001
            out.append(f"no state ({type(e).__name__}: {e})")
        return ", ".join(out)

    def resumed(self):
        self.status(state="running", kind=None, message=None, command=None)
        self.chan.end(b"resumed")

    def _pause_signal(self, signum, frame):
        """SIGUSR1, from python -m robot pause."""
        if self.dbg is None:
            if self.attended:
                self.early_pause = True
            else:
                print("pause ignored (--unattended)", file=self.log, flush=True)
        elif not self.dbg.depth:
            self.dbg.request_pause(frame)

    def _breakpoint(self, *args, **kws):
        frame = sys._getframe(1)
        if self.dbg is None:
            print(f"breakpoint() at {_rel(frame.f_code.co_filename)}:{frame.f_lineno} ignored (--unattended)",
                  file=self.log, flush=True)
            return
        self.dbg.pause_live(frame, "breakpoint", "breakpoint()")

    # ---- running
    def _call_step(self, step, entry: bool):
        """Call one step; with --pause-each-step return its frame (kept at its call event)."""
        dbg = self.dbg
        if dbg is None:
            step(self.r, self.v, self.s)
            return None
        dbg.botframe = sys._getframe()
        dbg.quiet()
        dbg.entry = step.__code__ if entry else None
        dbg.capture, dbg.captured = (step.__code__ if self.pause_each else None), None
        if dbg.breaks or dbg.entry or dbg.capture or dbg.pause_requested:
            sys.settrace(dbg.trace_dispatch)
        try:
            step(self.r, self.v, self.s)
        finally:
            sys.settrace(None)
            dbg.entry = dbg.capture = None
            dbg.quiet()                         # n or s at a step's end does not stop in the runner
        return dbg.captured

    def _loop(self):
        names = self.step_names()
        i = names.index(self.start) if self.start else 0
        while i < len(self.steps):
            step = self.steps[i]
            self.step = name = step.__name__
            entry = name in self.breaks or self.break_next
            self.break_next = False
            print(f"--- step {name} ({i + 1}/{len(self.steps)})", file=self.log, flush=True)
            self.status(step=name)
            try:
                try:
                    frame = self._call_step(step, entry)
                except Exception as e:
                    if self.dbg is None:
                        raise
                    self.dbg.post_mortem(e)
                if self.pause_each:
                    k = self._index(name, i)
                    nxt = self.steps[k + 1].__name__ if k + 1 < len(self.steps) and not self.only else None
                    self.dbg.pause_after(frame, name, nxt)
            except _Retry:
                i = self._index(name, i)
                continue
            except _Skip:
                if self.only:
                    break
                i = self._index(name, i) + 1
                continue
            except _Goto as g:
                i = self._index(g.name, i)
                continue
            if self.only:
                break
            i = self._index(name, i) + 1

    def _check_options(self):
        names = self.step_names()
        bad = [b for b in [self.start, *[b for b in self.breaks if not _is_line(b)]] if b and b not in names]
        if bad:
            raise RunnerError(f"no step {', '.join(map(repr, bad))}; STEPS = [{', '.join(names)}]")

    def _set_breakpoints(self):
        for spec in self.breaks:
            if not _is_line(spec):
                continue
            path, line = spec.rsplit(":", 1)
            for cand in (path, os.path.join(os.path.dirname(self.task), path)):
                if os.path.isfile(cand):
                    path = os.path.abspath(cand)
                    break
            before = len(bdb.Breakpoint.bpbynumber)
            self.dbg.onecmd(f"break {path}:{line}")             # pdb's own checks and message, to the log
            if len(bdb.Breakpoint.bpbynumber) == before:
                raise RunnerError(f"--break {spec}: no breakpoint set there (see the line above)")

    def _channel(self) -> _Channel:
        """The debugger's socket: debug.sock in the run directory, where every later command finds it, or in a
        directory of its own in /tmp when that path is too long for a unix socket (107 bytes) or cannot hold one."""
        path = os.path.join(self.run_dir, "debug.sock")
        if len(os.fsencode(path)) <= 107:
            try:
                return _Channel(self, path)
            except OSError:
                pass
        self.sockdir = tempfile.mkdtemp(prefix="robot-run-", dir="/tmp")
        return _Channel(self, os.path.join(self.sockdir, "debug.sock"))

    def run(self) -> int:
        global _ACTIVE
        _ACTIVE = self
        os.makedirs(self.run_dir, exist_ok=True)
        faulthandler.enable()
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
        signal.signal(signal.SIGUSR1, self._pause_signal)
        self.status()
        code, kind, message = 1, "exception", None
        try:
            try:
                self.load()
                self.snapshot("start")
                self._check_options()
            except RunnerError as e:
                code, kind, message = 2, "error", str(e)
                print(message, file=self.log, flush=True)
                return code
            addr = os.environ.get("ROBOT_ADDR", "unix:/tmp/robot.sock")
            try:
                self.r = robot.connect(name=os.path.basename(self.task))
                self.v = _vision()
            except Exception as e:              # noqa: BLE001
                message = f"cannot connect: {type(e).__name__}: {e} (ROBOT_ADDR={addr})"
                print(message, file=self.log, flush=True)
                return code
            self.names = {"r": self.r, "v": self.v, "s": self.s}
            if self.attended:
                self.chan = self._channel()
                self.dbg = Debugger(self, self.chan)
                self.dbg.pause_requested = self.early_pause
                self.status(socket=self.chan.path)
                try:
                    self._set_breakpoints()
                except RunnerError as e:
                    code, kind, message = 2, "error", str(e)
                    print(message, file=self.log, flush=True)
                    return code
            elif self.breaks or self.pause_each:
                print("--break and --pause-each-step are ignored (--unattended)", file=self.log, flush=True)
            sys.breakpointhook = self._breakpoint
            t0 = time.time()
            self._loop()
            code, kind, message = 0, "finished", f"{len(self.steps)} steps in {time.time() - t0:.1f} s"
            print(f"--- finished ({message})", file=self.log, flush=True)
            return code
        except _Quit:
            code, kind, message = 1, "quit", f"q at pause {self.npause}"
            print(f"--- ended by q at pause {self.npause}", file=self.log, flush=True)
            return code
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            kind, message = ("terminated", "SIGTERM") if code == 128 + signal.SIGTERM else ("exit", f"sys.exit({code})")
            return code
        except BaseException as e:              # noqa: BLE001 -- --unattended, or a failure loading the task
            sys.stderr.write(_format_exc(e))
            sys.stderr.flush()
            message = traceback.format_exception_only(type(e), e)[-1].strip()
            tb = e.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            if tb is not None:
                self.st.update(file=tb.tb_frame.f_code.co_filename, line=tb.tb_lineno,
                               function=tb.tb_frame.f_code.co_name)
            return code
        finally:
            sys.settrace(None)
            self.status(state="exited", exit_code=code, kind=kind, message=message, command=None, block=None)
            if self.chan is not None:
                self.chan.close()
            if self.sockdir:
                shutil.rmtree(self.sockdir, ignore_errors=True)
            if self.r is not None:
                self.r.close()


_CALL_STEP = Runner._call_step.__code__


def _is_line(spec: str) -> bool:
    return ":" in spec and spec.rsplit(":", 1)[1].isdigit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.runner", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("task")
    ap.add_argument("--break", dest="breaks", action="append", metavar="STEP|FILE:LINE")
    ap.add_argument("--pause-each-step", action="store_true")
    ap.add_argument("--from", dest="start", metavar="STEP")
    ap.add_argument("--only", metavar="STEP")
    ap.add_argument("--unattended", action="store_true")
    return Runner(ap.parse_args(argv)).run()


if __name__ == "__main__":
    import robot.runner                 # run the imported module, not this __main__ copy: rewritten asserts and
    sys.exit(robot.runner.main())       # the debugger find the runner there
