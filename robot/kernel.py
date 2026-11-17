"""robot.kernel: one long-running Python for your commands.

A fresh `python` per command pays for its start and its imports every time (numpy, OpenCV, SciPy, robot.viz: about
0.3 s on the cell, more with MuJoCo for a ghost, whose arm model also loads again each time). The kernel pays once:

    python -m robot.kernel <<'PY'
    r = robot.connect()
    print(r.state().summary())
    PY
    python -m robot.kernel task.py            # a file's code, run the same way

The first call starts the kernel in the background (for this workspace; `--status`, `--stop`); later calls reuse it.
Your code runs in ONE namespace that persists between calls: variables, functions, loaded frames and imports stay,
and robot, numpy as np, cv2, viz, watch and when are already imported. Each call ends as a program would:

- the robot connections the call opened are closed, so the motions they started stop (as when a program exits);
- it prints the `[state]` line (if it sent the arms a command) and the `[views]` line (the images it showed);
- an exception prints its traceback and the call exits 1; `exit(n)` ends the call with code n.

Killing the call (Ctrl-C, or the command's timeout) stops its code and closes its connections. Calls can run at the
same time (each in its own thread, on the same namespace); only one at a time should move the arms. Standard
library only on the client side, so a call starts in about 20 ms.

Hooks: code of yours that runs at the end of calls, so that what you would fetch in your next step comes back with
the call instead.

    def towel(r, frames):                     # r: the call's robot; frames: one grab of every camera, with depth
        f = frames['front']; h = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2HSV)
        viz.marks(f, 'towel', mask=(h[..., 0] > 85) & (h[..., 0] < 120), on_table=True, picture=False)
    kernel.after_motion(towel)                # at the end of every call that moved the arms

kernel.after_call(fn) runs fn at the end of every call. A hook runs after the call's code and its [state] line,
before its [views] line, in the call's own output (what it prints, and the images it shows, come back with the
call). The grab is taken the first time a hook reads frames, once per call. Registering a function of the same name
again replaces it; kernel.remove_hook(name) drops one, kernel.hooks() lists them. A hook that raises prints
`[hook NAME] error ...` and the call ends as it would have; one that takes over HOOK_SLOW_S says so, as every such
call pays for it. Hooks live in the kernel (a plain `python` program does not run them).

Actions: start a motion, and look at what you planned while the arms move, instead of a step to review and another
to execute.

    def grasp(r):                             # r: a robot connection of the action's own
        r.gripper('left', .07); r.move(A, ee='left', duration=3); r.move(G, ee='left', line=True, duration=3)
        r.gripper('left', 0, force=30); r.move(L, ee='left', line=True, duration=3)
    kernel.start(grasp)                       # the arms start; this returns at once
    review('left', [A, G])                    # your picture of the plan, drawn while they move, in this call's output

    # next call, while the arm moves: wrong? kernel.action().stop(); fine? let it finish:
    kernel.action().wait()                    # its output: how it ended, [state], views/after-grasp.png, the hooks

kernel.start(fn, name=None, timeout=120, views=True) runs fn in the kernel and returns its Action at once; its
connection (the actions' one, kept from one action to the next) is not closed when the call ends. Every later call
reports it (`[action grasp] running for 3.1 s`, what it printed, and as it ends `[action grasp] done in 9.8 s`, or
stopped, or its error), and the call that sees it end prints the arms' [state], shows every camera as one grid
(views/after-NAME.png; views=False: none) and runs the after-motion hooks. a.stop() halts every motion (the arms hold) and stops its code; a.wait()
waits for it (True if it ended on its own without an error); its timeout stops it too. One action at a time, and a
call that sends the arms a command stops the running action first. kernel.action() is the latest, kernel.actions()
all of them.

The push (the episode harness sets ROBOT_KERNEL_EVENTS to a directory): the report is made by the action itself as
it ends, and the harness sends it to the agent as a message (into its running turn, or as a new turn if it has ended
its own), with the after-NAME grid and the hooks' images attached. Nothing has to wait or poll: start, look at the
plan, end the turn or do something else, and the news comes. A call that waits on the action (a.wait()) gets the
report in its own output instead; other calls only say that it ended. EVENTS/actions.json names the action running.

The queue: write the next action while this one runs, and the arms go straight on to it.

    kernel.queue(next_grasp)                  # starts as the running action ends (at once if none is running)

kernel.queue(fn, name=None, timeout=120, views=True, replace_from=None) returns the Action; fn may be a list, and at
most QUEUE_MAX wait behind the running one. Every action has a number (#12), the same for as long as it exists.
replace_from=12 (or a name) takes #12 and everything behind it out of the queue and puts these in its place, in one
step: what stood behind #12 was planned on #12. If #12 has already started nothing changes and it raises.
As an action ends the next one starts first, and only then is the report made, so nothing in the report delays the
arms; the report says what started (`[queue] started: B`). An action that ended with an error or a stop starts nothing:
everything queued is dropped and the report says so (`[queue] dropped: B`), so what was planned on its success never
runs. kernel.queued() lists what waits, kernel.flush() drops it. A queued action's precondition is code at its top:
raise and it ends with an error before any motion, and whatever was queued behind it is dropped.

Check the next action from where the queue will leave the arms: kernel.planned_end() is {arm: the end of the last move
the running action has sent it so far, or what an action declared with kernel.queue(fn, ends={arm: pose or joints})},
for r.reachable(..., after=), r.ik(..., after=) and a ghost's after= (docs/MANUAL.md). It cannot see moves the running
action has yet to send, nor where a queued action without ends= goes.

With ROBOT_KERNEL_STAGGER set (the episode harness, --staggered) reports carry `[queue]` lines saying what runs (and
which command of it), and what waits; an action queued while the arms hold sends a short message as it starts; and each
time the harness writes EVENTS/look.request (the agent's turn ended while an action runs) the kernel sends a look: the
state, every camera as one grid and the queue, as of now.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import itertools
import threading
import time

SOCK = os.environ.get("ROBOT_KERNEL_SOCK") or f"/tmp/robot-kernel-{os.getuid()}.sock"
IDLE_S = float(os.environ.get("ROBOT_KERNEL_IDLE_S", 3600))
PREIMPORT = ("numpy", "cv2", "scipy.spatial.transform", "robot", "robot.viz", "robot.watch", "robot.when",
             "mujoco")
HOOK_SLOW_S = 1.0
EVENTS_ENV = "ROBOT_KERNEL_EVENTS"     # a directory: actions' end reports go there as files that the harness sends to
                                       # the agent as messages (the push); unset, the next call reports them
STAGGER_ENV = "ROBOT_KERNEL_STAGGER"   # set: an action queued while the arms hold says so as it starts (module docstring)
QUEUE_MAX = 3


# ------------------------------------------------------------------------------------------------ hooks
_HOOKS: dict = {}                                 # name -> (fn, "motion" | "call"), in the order registered


def after_motion(fn, name: str | None = None):
    """Run fn(r, frames) (or fn(r)) at the end of every kernel call that moved the arms (module docstring)."""
    _HOOKS.pop(name or fn.__name__, None)
    _HOOKS[name or fn.__name__] = (fn, "motion")
    return fn


def after_call(fn, name: str | None = None):
    """Run fn(r, frames) (or fn(r)) at the end of every kernel call (module docstring). r is None when the call
    opened no robot connection and none could be opened."""
    _HOOKS.pop(name or fn.__name__, None)
    _HOOKS[name or fn.__name__] = (fn, "call")
    return fn


def remove_hook(name: str) -> bool:
    return _HOOKS.pop(name, None) is not None


def hooks() -> dict:
    """{name: "motion" | "call"} for the hooks registered now."""
    return {k: w for k, (_, w) in _HOOKS.items()}


class _Frames(dict):
    """One grab of every camera (colour and depth), taken the first time a hook reads it."""

    def __init__(self, grab):
        super().__init__()
        self._grab, self._done = grab, False

    def _load(self):
        if not self._done:
            self._done = True
            self.update(self._grab())

    def __getitem__(self, k):
        self._load()
        return super().__getitem__(k)

    def __contains__(self, k):
        self._load()
        return super().__contains__(k)

    def __iter__(self):
        self._load()
        return super().__iter__()

    def __len__(self):
        self._load()
        return super().__len__()

    def get(self, k, default=None):
        self._load()
        return super().get(k, default)

    def keys(self):
        self._load()
        return super().keys()

    def values(self):
        self._load()
        return super().values()

    def items(self):
        self._load()
        return super().items()


def _is_open(r) -> bool:
    try:
        return r.sock.fileno() >= 0
    except Exception:                             # noqa: BLE001
        return False


def _hooks_due(moved: bool, action_ended: bool) -> list:
    return [(n, f, w) for n, (f, w) in list(_HOOKS.items()) if w == "call" or moved or action_ended]


def _run_hooks(opened: list, moved: list, action_ended: bool = False, frames=None) -> None:
    """The registered hooks due for this call, in its thread (so their output is the call's). frames: a grab already
    taken (an action's report shares its own with the hooks)."""
    import inspect
    import traceback

    import robot
    todo = _hooks_due(bool(moved), action_ended)
    if not todo:
        return
    r = next((x for x in reversed(moved or opened) if _is_open(x)), None)
    if r is None:
        try:
            r = robot.connect(name="kernel-hooks")    # one of this call's connections: closed as it ends
        except Exception:                         # noqa: BLE001 -- no driver: hooks that need none still run
            r = None

    def grab():
        return (r if r is not None else robot.connect(name="kernel-hooks")).cameras()

    frames = frames if frames is not None else _Frames(grab)
    for name, fn, when in todo:
        t0 = time.time()
        try:
            try:
                n = len(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                n = 2
            fn(r, frames) if n >= 2 else fn(r)
        except KeyboardInterrupt:                 # the call was killed: no more hooks
            break
        except BaseException as e:                # noqa: BLE001 -- a hook never changes how the call ends
            where = ""
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                where = f" (line {tb[-1].lineno} of {tb[-1].filename})"
            print(f"[hook {name}] error {type(e).__name__}: {e}{where}", flush=True)
        dt = time.time() - t0
        if dt > HOOK_SLOW_S:
            print(f"[hook {name}] took {dt:.1f} s, which every call that "
                  f"{'moves the arms' if when == 'motion' else 'runs'} pays", flush=True)


# ------------------------------------------------------------------------------------------------ actions
ACTION_TIMEOUT_S = 120.0
_ACTIONS: list = []                               # every action started in this kernel, oldest first
_QUEUE: list = []                                 # the actions waiting to start, next first
_CURRENT: list = []                               # [the action whose turn it is], from its start until the next starts
_QUEUE_LOCK = threading.Lock()                    # the queue, _CURRENT and starting an action
_IDS = itertools.count(1)


class ActionStopped(Exception):
    """Raised in an action's own code when it is stopped (by Action.stop, a call that moved the arms, or its
    timeout), so that it goes no further."""


class Action:
    """A function of yours running in the kernel while your calls go on (kernel.start): the arms move while you look
    at what you planned. Its own robot connection is not closed when the call that started it ends."""

    def __init__(self, fn, name: str, timeout: float, views: bool = True):
        self.fn, self.name, self.timeout, self.views = fn, name, float(timeout), bool(views)
        self.t0, self.t1 = time.time(), None
        self.r = None                             # its robot connection (when fn takes one)
        self.result = self.error = None
        self.stopped = None                       # why it was stopped, if it was
        self._out: list = []                      # what it printed and no call has reported yet
        self._reported = False
        self._thread = None
        self._ended = threading.Event()           # its code and motion are over (its report may still be on the way)
        self._lock = threading.Lock()
        # the push (EVENTS_ENV set): its report is built as it ends and sent as a message, or printed by a call that
        # is waiting on it
        self.push = os.environ.get(EVENTS_ENV) or None
        self.report = None                        # {"text", "images", ...} once built
        self.channel = None                       # "push" (a message) or "call" (the waiting call prints it)
        self._waiters = 0
        self.id = next(_IDS)                      # its number in the [queue] lines
        self.ncmd, self.last_cmd = 0, None        # the commands it has sent so far, and the latest one's kind
        self.ends: dict = {}                      # arm -> where the last move it sent that arm ends (planned_end)
        self.declared: dict = {}                  # arm -> where it says it leaves that arm (queue(ends=))
        self.queued = False                       # waiting in the queue
        self.queue_note: list = []                # what the queue did as it ended, for its report

    @property
    def running(self) -> bool:
        return self.t1 is None and not self.queued

    @property
    def elapsed(self) -> float:
        return (self.t1 or time.time()) - self.t0

    def status(self) -> str:
        if self.queued:
            return f"queued for {self.elapsed:.1f} s"
        if self.running:
            return f"running for {self.elapsed:.1f} s" + (f", stopping ({self.stopped})" if self.stopped else "")
        if self.error is not None:
            return f"error {type(self.error).__name__}: {self.error} after {self.elapsed:.1f} s"
        if self.stopped:
            return f"stopped after {self.elapsed:.1f} s ({self.stopped})"
        return f"done in {self.elapsed:.1f} s"

    def __repr__(self):
        return f"<action {self.name}: {self.status()}>"

    def stop(self, why: str = "stopped by your call") -> bool:
        """Stop it now: every motion stops (the arms hold where they are) and its code goes no further. False if it had
        already ended. A queued action is dropped from the queue."""
        with _QUEUE_LOCK:
            if self.queued and self in _QUEUE:
                _QUEUE.remove(self)
                _drop(self, why)
                return True
        if not self.running:
            return False
        self.stopped = self.stopped or why
        _journal("stop", self, why=why)
        if self.r is not None:                    # its own socket is busy waiting on its motion: stop from another
            try:
                import robot
                s = robot.connect(self.r.addr if hasattr(self.r, "addr") else None, name=f"stop-{self.name}")
                try:
                    s.stop()
                finally:
                    s.close()
            except Exception as e:                # noqa: BLE001
                print(f"[action {self.name}] could not reach the driver to stop it: {e}", flush=True)
        th = self._thread
        if th is not None and th.is_alive() and not getattr(self, "_closing", False):   # its own Python code too
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(ActionStopped))
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until it ends (or timeout s); True if it ended on its own, without an error. The call's output reports
        how it ended, what it printed, the arms' [state] and the after-motion hooks. (Under the harness's push you
        need not wait: the report comes as a message as the action ends. A call that does wait gets it instead.)"""
        counted = False
        if self.push:
            with self._lock:
                if self.channel is None:
                    self._waiters += 1
                    counted = True
        try:
            t_end = None if timeout is None else time.time() + timeout
            while self.queued:                    # its turn first
                if t_end is not None and time.time() >= t_end:
                    return False
                time.sleep(0.02)
            while self._thread is not None and self._thread.is_alive():
                if t_end is not None and time.time() >= t_end:
                    return False
                self._thread.join(0.1)
            return not self.running and self.error is None and not self.stopped
        finally:
            if counted:
                with self._lock:
                    self._waiters -= 1


def _action_route(a: Action):
    """The action thread's stdout/stderr: into the action, for the next call to report."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, _Route):
            stream.local.out = lambda key, s, a=a: a._out.append(s)


def _run_action(a: Action):
    try:
        _run_action_body(a)
    except ActionStopped:                         # a stop that landed in its error handling
        a.t1 = a.t1 or time.time()
    _next_after(a)                                # before its report: the report must not delay the arms
    a._ended.set()
    if a.push:
        try:
            _end_report(a)
        except BaseException as e:                # noqa: BLE001 -- never leave a waiter or the harness hanging
            with a._lock:
                a.report = a.report or {"name": a.name, "status": a.status(), "images": [],
                                        "text": f"[action {a.name}] {a.status()}\n[action {a.name}] its report "
                                                f"failed: {type(e).__name__}: {e}\n"}
                a.channel = a.channel or ("call" if a._waiters else "push")
            if a.channel == "push":
                _write_event(a, a.report)
        finally:
            with _QUEUE_LOCK:
                _write_running(a.push, _CURRENT[0] if _CURRENT else None)


def _take_out(a: Action) -> str:
    with a._lock:
        out, a._out[:] = "".join(a._out), []
    return out


def _end_report(a: Action) -> None:
    """The push: as an action ends, in its own thread, what the next look would fetch: how it ended, what it printed,
    the arms' [state], every camera as one grid (views/after-NAME.png) and the after-motion hooks (one grab shared
    with them). Then a file in the events directory, which the harness sends as a message, or, when a call is
    waiting on the action, that call's output."""
    from robot import viz
    t0 = time.time()
    buf = [f"[action {a.name}] {line}\n" for line in _take_out(a).splitlines()]
    for stream in (sys.stdout, sys.stderr):       # the report's own lines: into buf, not to a call
        if isinstance(stream, _Route):
            stream.local.out = lambda key, s: buf.append(s)
    shown: list = []
    viz._SINK.paths = shown
    try:
        print(f"[action {a.name}] {a.status()}", flush=True)
        for line in a.queue_note + (_queue_lines() if os.environ.get(STAGGER_ENV) else []):
            print(line, flush=True)
        with _REPORT_LOCK:                        # one report at a time on the reports' connection
            src = None
            try:
                src = _report_robot()
                if os.environ.get("ROBOT_EXIT_STATE", "1") != "0":
                    print("[state] " + src.state().summary(), flush=True)
            except Exception as e:                # noqa: BLE001
                _drop_report_robot()
                src = None
                print(f"[action {a.name}] no [state]: {type(e).__name__}: {e}", flush=True)
            frames = None
            if src is not None:
                frames = _Frames(src.cameras) if _hooks_due(True, True) else None   # with depth, for the hooks
                if a.views:
                    try:
                        viz.show_cameras(f"after-{a.name}", src, frames=frames)
                    except Exception as e:        # noqa: BLE001
                        print(f"[action {a.name}] no after-motion views: {type(e).__name__}: {e}", flush=True)
            _run_hooks([src] if src is not None else [], [src] if src is not None else [], True, frames=frames)
    finally:
        viz._SINK.paths = None
        _action_route(a)
    _journal("ended", a, status=a.status(), ok=a.error is None and not a.stopped, t0=round(a.t0, 3),
             t1=round(a.t1 or time.time(), 3), commands=a.ncmd)
    report = {"name": a.name, "status": a.status(), "ok": a.error is None and not a.stopped,
              "text": "".join(buf), "images": list(dict.fromkeys(shown)),
              "t0": round(a.t0, 3), "t1": round(a.t1 or time.time(), 3), "report_s": round(time.time() - t0, 2)}
    with a._lock:
        a.report = report
        a.channel = "call" if a._waiters else "push"
    if a.channel == "push":
        _write_event(a, report)


_EVENT_SEQ = [0]


def _write_event(a: Action, report: dict) -> None:
    """One file per report, written whole (a reader never sees half of one): EVENTS/<ms>-<n>-action-NAME.json."""
    d = a.push
    os.makedirs(d, exist_ok=True)
    _EVENT_SEQ[0] += 1
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in a.name)[:40] or "action"
    path = os.path.join(d, f"{int(time.time() * 1000)}-{_EVENT_SEQ[0]:04d}-action-{safe}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(dict(report, kind="action", written=round(time.time(), 3)), f)
    os.replace(path + ".tmp", path)


def _journal(event: str, a: Action | None = None, **info) -> None:
    """EVENTS/queue.jsonl: one line per queue event (queued, started, ended, dropped, replaced, flushed, look), with the
    wall clock and what runs and waits after it, so a run's queue can be drawn afterwards. (_QUEUE_LOCK may be held:
    it reads _CURRENT and _QUEUE without taking it.)"""
    d = os.environ.get(EVENTS_ENV)
    if not d:
        return
    line = {"wall": round(time.time(), 4), "event": event}
    if a is not None:
        line.update(id=a.id, name=a.name)
    line.update(info)
    line["running"] = _CURRENT[0].id if _CURRENT else None
    line["waiting"] = [b.id for b in _QUEUE]
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "queue.jsonl"), "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except OSError:
        pass


def _write_running(d: str | None, a: Action | None) -> None:
    """EVENTS/actions.json: the action running now (the harness keeps the episode open while one runs)."""
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        st = {"running": a.name if a else None, "updated": round(time.time(), 3)}
        if a is not None:
            st.update(t0=round(a.t0, 3), deadline=round(a.t0 + a.timeout + 60, 3))
        with open(os.path.join(d, "actions.json.tmp"), "w") as f:
            json.dump(st, f)
        os.replace(os.path.join(d, "actions.json.tmp"), os.path.join(d, "actions.json"))
    except OSError:
        pass


def _run_action_body(a: Action):
    import inspect
    import traceback

    import robot
    _action_route(a)
    try:
        try:
            n = len(inspect.signature(a.fn).parameters)
        except (TypeError, ValueError):
            n = 1
        if n >= 1:
            a.r = _action_robot()
            a.r._action = a
        a.result = a.fn(a.r) if n >= 1 else a.fn()
    except ActionStopped:
        pass
    except BaseException as e:                    # noqa: BLE001 -- reported to the next call
        if a.stopped and isinstance(e, robot.RobotError):
            pass                                  # a motion cut short by the stop
        else:
            a.error = e
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno} of {tb[-1].filename})" if tb else ""
            a._out.append(f"{type(e).__name__}: {e}{where}\n")
    finally:
        a._closing = True
        while True:                               # a stop's exception may land here: end cleanly all the same
            try:
                if a.r is not None:
                    a.r._action = None
                    if a.stopped or a.error is not None:    # a stop may have cut a request short: a fresh one next
                        _drop_action_robot()
                a.t1 = a.t1 or time.time()
                break
            except ActionStopped:
                continue


_ACTION_ROBOT: list = []      # the actions' robot connection, kept between actions (opening one takes about a second)


def _action_robot():
    """The actions' connection: the one kept open, or a new one (not one of any call's, so no call's end closes it)."""
    import robot
    r = _ACTION_ROBOT[0] if _ACTION_ROBOT else None
    if r is not None and _is_open(r):
        return r
    latest = list(robot._LATEST)
    r = robot.connect(name="kernel-actions")
    robot._LATEST[:] = latest                     # your calls' connection stays the one viz and [state] use
    _ACTION_ROBOT[:] = [r]
    return r


def _drop_action_robot():
    if _ACTION_ROBOT:
        try:
            _ACTION_ROBOT[0].close()
        except Exception:                         # noqa: BLE001
            pass
        _ACTION_ROBOT[:] = []


_REPORT_ROBOT: list = []      # the reports' connection: the next action may already be using the actions' one
_REPORT_LOCK = threading.Lock()


def _report_robot():
    import robot
    r = _REPORT_ROBOT[0] if _REPORT_ROBOT else None
    if r is not None and _is_open(r):
        return r
    latest = list(robot._LATEST)
    r = robot.connect(name="kernel-reports")
    robot._LATEST[:] = latest
    _REPORT_ROBOT[:] = [r]
    return r


def _drop_report_robot():
    if _REPORT_ROBOT:
        try:
            _REPORT_ROBOT[0].close()
        except Exception:                         # noqa: BLE001
            pass
        _REPORT_ROBOT[:] = []


def _open_report_robot():
    """Opening a connection takes about a second: done ahead of the first report that needs it."""
    try:
        with _REPORT_LOCK:
            _report_robot()
    except Exception:                             # noqa: BLE001 -- the report says so if it still cannot connect
        pass


def _watchdog(a: Action):
    while a.running:
        if a.elapsed > a.timeout:
            a.stop(f"its timeout of {a.timeout:g} s passed")
            return
        time.sleep(0.2)


def start(fn, name: str | None = None, timeout: float = ACTION_TIMEOUT_S, views: bool = True) -> Action:
    """Run fn(r) (or fn()) in the kernel while your calls go on, and return at once with its Action: the arms start
    on your plan, and you look at the plan (or anything else) while they move. r is the actions' robot connection
    (kept open from one action to the next). One action at a time; while it runs, a call that sends the arms a
    command stops it first. Every call reports it: `[action NAME] running for 3.2 s`, what it printed, and as it
    ends `[action NAME] done in 9.8 s` (or stopped, or its error); the call that sees it end also carries the arms'
    [state], every camera as one grid (views/after-NAME.png; views=False: none) and the after-motion hooks.
    a.stop() stops it (every motion halts; its code goes no further), a.wait() waits for it; its timeout (s) stops
    it too. Actions live in the kernel: when the episode ends, their motions stop."""
    a = Action(fn, name or getattr(fn, "__name__", "action"), timeout, views)
    with _QUEUE_LOCK:
        for b in _ACTIONS:
            if b.running:
                raise RuntimeError(f"action {b.name!r} is still {b.status()}: {b.name}.wait() or .stop() it first "
                                   "(kernel.action() is the latest)")
        if _QUEUE:
            raise RuntimeError(f"{[b.name for b in _QUEUE]} queued: kernel.flush() first, or kernel.queue this one")
        _launch(a)
    print(f"[action {a.name}] started" + (" (as it ends, its report comes to you as a message)" if a.push else ""),
          flush=True)
    return a


def _launch(a: Action) -> None:
    """Its turn: its clock and thread start now. (_QUEUE_LOCK held.)"""
    a.queued, a.t0 = False, time.time()
    _ACTIONS.append(a)
    _CURRENT[:] = [a]
    _journal("started", a)
    _write_running(a.push, a)
    a._thread = threading.Thread(target=_run_action, args=(a,), name=f"action-{a.name}", daemon=True)
    a._thread.start()
    threading.Thread(target=_watchdog, args=(a,), daemon=True).start()
    if a.push and os.environ.get(STAGGER_ENV) and not _LOOK_WATCH:
        _LOOK_WATCH.append(threading.Thread(target=_watch_looks, args=(a.push,), daemon=True))
        _LOOK_WATCH[0].start()


def _drop(a: Action, why: str) -> None:
    """It will never run. (_QUEUE_LOCK held.)"""
    a.queued, a.stopped, a.t1 = False, f"dropped: {why}", time.time()
    a._ended.set()
    _journal("dropped", a, why=why)


def _next_after(a: Action) -> None:
    """a is over: the next one starts now, or, if a did not finish, nothing planned on it does."""
    with _QUEUE_LOCK:
        if _CURRENT and _CURRENT[0] is a:
            _CURRENT[:] = []
        if not _QUEUE:
            return
        if a.error is not None or a.stopped:
            names = [b.name for b in _QUEUE]
            for b in _QUEUE:
                _drop(b, f"{a.name} did not finish")
            _QUEUE[:] = []
            a.queue_note = [f"[queue] dropped: {', '.join(names)} ({a.name} did not finish); the arms hold"]
            return
        n = _QUEUE.pop(0)
        waited = time.time() - n.t0
        _launch(n)
        a.queue_note = [f"[queue] started: {n.name} (#{n.id}; it had waited {waited:.1f} s)"]


def _queue_lines() -> list:
    """Where things stand: what runs and which of its commands, then what waits, in order."""
    with _QUEUE_LOCK:
        cur, wait = (_CURRENT[0] if _CURRENT else None), list(_QUEUE)
    if cur is None:
        return ["[queue] empty, nothing running: the arms hold until you queue the next action"]
    out = [f"[queue] running #{cur.id} {cur.name}: {cur.elapsed:.1f} s in, command {cur.ncmd}" +
           (f" ({cur.last_cmd})" if cur.last_cmd else "")]
    out += [f"[queue] waiting {i}: #{b.id} {b.name}" for i, b in enumerate(wait, 1)]
    if wait:
        out.append(f"[queue] {QUEUE_MAX - len(wait)} of {QUEUE_MAX} places free" if len(wait) < QUEUE_MAX else "[queue] full")
    else:
        out.append(f"[queue] empty behind it: as #{cur.id} {cur.name} ends the arms stop until the next action is queued")
    return out


def _queue_index(which) -> int:
    """Where #which (or the action of that name) stands in the queue. (_QUEUE_LOCK held.)"""
    for i in range(len(_QUEUE) - 1, -1, -1):
        if _QUEUE[i].id == which or _QUEUE[i].name == which:
            return i
    raise RuntimeError(f"{which!r} is not waiting in the queue (it may have started or been dropped: "
                       f"{[f'#{b.id} {b.name}' for b in _QUEUE]} wait): nothing changed")


def queue(fn, name: str | None = None, timeout: float = ACTION_TIMEOUT_S, views: bool = True, replace_from=None,
          ends=None) -> Action:
    """Run fn(r) (or fn()) as soon as the arms are free: at once if no action is running, else as the running one ends
    (module docstring). fn may be a list, run in order. replace_from=ID (or a name): that action and everything behind
    it leave the queue and these take their place, in one step. ends={arm: pose or joints} (as r.reachable's after=):
    where the (last) action leaves the arms, for planned_end. Returns the (first) Action. At most QUEUE_MAX wait."""
    fns = list(fn) if isinstance(fn, (list, tuple)) else [fn]
    if not fns:
        raise ValueError("kernel.queue: no action given")
    new = [Action(f, name if name and len(fns) == 1 else getattr(f, "__name__", "action"), timeout, views) for f in fns]
    new[-1].declared = dict(ends or {})
    replaced: list = []
    with _QUEUE_LOCK:
        keep = len(_QUEUE) if replace_from is None else _queue_index(replace_from)
        idle = not _CURRENT and not _QUEUE
        if keep + len(new) - (1 if idle else 0) > QUEUE_MAX:
            raise RuntimeError(f"the queue holds {QUEUE_MAX}: {[f'#{b.id} {b.name}' for b in _QUEUE[:keep]]} would stay and "
                               f"{len(new)} would join them. Nothing changed.")
        replaced = _QUEUE[keep:]
        for b in replaced:
            _drop(b, f"replaced by {new[0].name}")
        del _QUEUE[keep:]
        if idle:
            _launch(new[0])
        behind = _CURRENT[0].name if _CURRENT else "the running action"
        for b in new[1 if idle else 0:]:
            b.queued = True
            _QUEUE.append(b)
            _journal("queued", b, behind=_CURRENT[0].id if _CURRENT else None, timeout=b.timeout,
                     replace_from=replace_from)
    a = new[0]
    if not _REPORT_ROBOT and a.push:
        threading.Thread(target=_open_report_robot, daemon=True).start()
    if replaced:
        print(f"[queue] replaced: {', '.join(f'#{b.id} {b.name}' for b in replaced)}", flush=True)
    if idle:
        print(f"[action {a.name}] started (#{a.id})", flush=True)
    for b in new[1 if idle else 0:]:
        print(f"[action {b.name}] queued behind {behind} as #{b.id}", flush=True)
    stagger = os.environ.get(STAGGER_ENV)
    if stagger:
        for line in _queue_lines():
            print(line, flush=True)
    if idle and a.push and stagger:
        t = time.time()
        _write_event(a, {"name": a.name, "status": "started", "ok": True, "phase": "start", "images": [],
                         "text": f"[action {a.name}] started (the arms were holding)\n" + "".join(l + "\n" for l in _queue_lines()),
                         "t0": round(a.t0, 3), "t1": round(t, 3), "report_s": 0.0})
    return a


def planned_end() -> dict:
    """Where the arms will be once the running action and those queued behind it are through, as far as the kernel
    knows, for r.reachable / r.ik / a ghost's after=: per arm, the end of the last move the running action has sent
    it so far (a world pose as that move was given, or joint values), unless one of them declared its own end for it
    (queue(ends=)); the latest action that says wins. An arm none of them says anything about is left out (it is
    checked where it is commanded now). It cannot know moves the running action has yet to send, nor where a queued
    action without ends= goes: plan the next action's first move after the running one's last move is sent, or
    declare ends."""
    with _QUEUE_LOCK:
        acts = ([_CURRENT[0]] if _CURRENT else []) + list(_QUEUE)
    out: dict = {}
    for a in acts:
        out.update(a.ends)
        out.update(a.declared)
    return {k: v for k, v in out.items() if v is not None}


def _ends_of(spec, arm_joints: dict) -> dict:
    """Where a command (and its then= chain) leaves each arm it moves, as after= takes it: {arm: {"target", "tcp",
    "free"}} for a world pose, joint values for all of an arm's joints, None where it cannot tell (move_at, a
    relative or tool-frame target, some of an arm's joints)."""
    out: dict = {}
    specs = [spec]
    while specs:
        s = specs.pop(0)
        if isinstance(s, (list, tuple)):
            specs[:0] = list(s)
            continue
        if not isinstance(s, dict):
            continue
        op = s.get("op")
        if op in ("move_to", "follow", "move_at"):
            js, ee = s.get("joints"), s.get("ee")
            last = (s.get("target") if op == "move_to" else s.get("points")) if op != "move_at" else None
            known = op != "move_at" and not s.get("relative") and s.get("frame", "world") == "world"
            if js is not None:
                goal = None if last is None or not known else (last if op == "move_to" else last[-1])
                for a, names in arm_joints.items():
                    if set(names) <= set(js):
                        out[a] = None if goal is None else [float(goal[js.index(n)]) for n in names]
                    elif set(names) & set(js):
                        out[a] = None
            elif ee is not None:
                for a in ([ee] if isinstance(ee, str) else list(ee)):
                    if not known:
                        out[a] = None
                        continue
                    T = last[a] if isinstance(last, dict) else last
                    T = T if op == "move_to" else T[-1]
                    tcp, free = s.get("tcp"), s.get("free")
                    tcp = tcp.get(a) if isinstance(tcp, dict) else tcp
                    free = free.get(a) if isinstance(free, dict) else free
                    free = free[-1] if isinstance(free, (list, tuple)) else free
                    out[a] = {"target": T, "tcp": tcp, "free": free}
        specs.append(s.get("then"))
    return out


def queued() -> list:
    """The actions waiting to start, next first."""
    with _QUEUE_LOCK:
        return list(_QUEUE)


def flush(why: str = "flushed by your call", from_item=None) -> list:
    """Drop what is queued (from_item=ID or a name: that action and everything behind it); the running action goes on.
    Returns their names."""
    with _QUEUE_LOCK:
        keep = 0 if from_item is None else _queue_index(from_item)
        names = [b.name for b in _QUEUE[keep:]]
        for b in _QUEUE[keep:]:
            _drop(b, why)
        del _QUEUE[keep:]
        _journal("flushed", why=why, dropped=names)
    return names


_LOOKS = [0]
_LOOK_WATCH: list = []


def _watch_looks(events: str) -> None:
    """The harness asks for a look by writing EVENTS/look.request."""
    req = os.path.join(events, "look.request")
    while True:
        try:
            os.unlink(req)
        except OSError:
            time.sleep(0.02)
            continue
        _journal("look")
        try:
            _look()
        except BaseException as e:                # noqa: BLE001 -- the next request tries again
            print(f"robot.kernel: a look failed: {type(e).__name__}: {e}", file=sys.__stderr__, flush=True)


def _look() -> None:
    """A message for the agent while an action runs: the state, every camera as one grid and the queue, as of now."""
    with _QUEUE_LOCK:
        a = _CURRENT[0] if _CURRENT else None
    if a is None or not a.push:
        return
    from robot import viz
    t0 = time.time()
    _LOOKS[0] += 1
    n = _LOOKS[0]
    buf: list = [f"[look {n}]\n"]
    shown: list = []
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, _Route):
            stream.local.out = lambda key, s: buf.append(s)
    viz._SINK.paths = shown
    try:
        with _REPORT_LOCK:
            try:
                src = _report_robot()
                print("[state] " + src.state().summary(), flush=True)
                viz.show_cameras(f"look-{n}", src)
            except Exception as e:                # noqa: BLE001
                _drop_report_robot()
                print(f"[look {n}] no state or views: {type(e).__name__}: {e}", flush=True)
    finally:
        viz._SINK.paths = None
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, _Route):
                stream.local.out = None
    buf += [line + "\n" for line in _queue_lines()]
    _write_event(a, {"name": a.name, "status": "look", "ok": True, "phase": "look", "text": "".join(buf),
                     "images": list(dict.fromkeys(shown)), "t0": round(a.t0, 3), "t1": round(time.time(), 3),
                     "report_s": round(time.time() - t0, 2)})


def action() -> Action | None:
    """The latest action started, running or not."""
    return _ACTIONS[-1] if _ACTIONS else None


def actions() -> list:
    return list(_ACTIONS)


def _before_command(r, spec):
    """robot._BEFORE_COMMAND: an action stopped goes no further; a call's command stops the running action first."""
    a = getattr(r, "_action", None)
    if a is not None:
        if a.stopped:
            raise ActionStopped(a.stopped)
        a.ncmd += 1
        a.last_cmd = spec.get("op") if isinstance(spec, dict) else None
        try:
            a.ends.update(_ends_of(spec, r.description["arm_joints"]))
        except Exception:                         # noqa: BLE001 -- only planned_end's guess suffers
            pass
        return
    for a in _ACTIONS:
        if a.running and not a.stopped:
            a.stop("this call sent the arms a command")
            a._ended.wait(5.0)                    # its motion over (its report, if pushed, comes on its own)


def _report_actions(call_t0: float) -> tuple:
    """At the end of a call: each action's news. Returns (the actions that ended since the last report, for which the
    call prints the arms' [state], their after-motion views and runs the after-motion hooks; the images of pushed
    reports this call prints, for its [views] line)."""
    ended, images = [], []
    for a in _ACTIONS:
        if a.push and not a.running:
            if a._reported or a.channel is None:  # reported, or its report is still being made (it will come)
                continue
            a._reported = True
            if a.channel == "call":               # a call waited on it: the report is this call's output
                sys.stdout.write(a.report["text"])
                images += a.report["images"]
            else:
                print(f"[action {a.name}] {a.status()} (its report is sent to you as a message)", flush=True)
            continue
        out = _take_out(a)
        for line in out.splitlines():
            print(f"[action {a.name}] {line}", flush=True)
        if a.running:
            if a.t0 < call_t0:                    # one this call started has said so already
                print(f"[action {a.name}] {a.status()}", flush=True)
        elif not a._reported:
            a._reported = True
            ended.append(a)
            print(f"[action {a.name}] {a.status()}", flush=True)
            for line in a.queue_note:
                print(line, flush=True)
    return ended, images


# ------------------------------------------------------------------------------------------------ the client
def _send(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(len(data).to_bytes(4, "big") + data)


def _recv(sock):
    head = b""
    while len(head) < 4:
        chunk = sock.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    n, buf = int.from_bytes(head, "big"), b""
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf)


def _connect(start: bool):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCK)
        return s
    except OSError:
        s.close()
    if not start:
        return None
    log = open(SOCK + ".log", "ab")
    subprocess.Popen([sys.executable, "-m", "robot.kernel", "--serve"], stdin=subprocess.DEVNULL, stdout=log,
                     stderr=log, start_new_session=True, cwd=os.getcwd(), env=dict(os.environ))
    t0 = time.time()
    while time.time() - t0 < 30:
        time.sleep(0.05)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(SOCK)
            return s
        except OSError:
            s.close()
    raise SystemExit(f"robot.kernel: no kernel answered at {SOCK} (see {SOCK}.log)")


def run_client(code: str, name: str) -> int:
    s = _connect(start=True)
    _send(s, {"code": code, "name": name, "cwd": os.getcwd()})
    try:
        while True:
            msg = _recv(s)
            if msg is None:
                print("robot.kernel: the kernel closed the connection (it may have died: see "
                      f"{SOCK}.log)", file=sys.stderr)
                return 1
            if "out" in msg:
                sys.stdout.write(msg["out"])
                sys.stdout.flush()
            elif "err" in msg:
                sys.stderr.write(msg["err"])
                sys.stderr.flush()
            elif "exit" in msg:
                return int(msg["exit"])
    except KeyboardInterrupt:
        s.close()                                 # the kernel stops this call's code and closes its connections
        return 130


# ------------------------------------------------------------------------------------------------ the kernel
class _Route:
    """sys.stdout / sys.stderr: each call's thread writes to its own client; other threads to the kernel's log."""

    def __init__(self, base, key):
        self.base, self.key = base, key
        self.local = threading.local()

    def write(self, s):
        out = getattr(self.local, "out", None)
        if out is None:
            return self.base.write(s)
        out(self.key, s)
        return len(s)

    def flush(self):
        if getattr(self.local, "out", None) is None:
            self.base.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.base.fileno()


class Kernel:
    def __init__(self):
        self.ns = {"__name__": "__main__", "__builtins__": __builtins__}
        self.last = time.time()
        self.busy = 0
        self.lock = threading.Lock()
        self.ready = threading.Event()            # the imports are done: calls wait for it
        sys.stdout = self.out = _Route(sys.stdout, "out")
        sys.stderr = self.err = _Route(sys.stderr, "err")

    def preimport(self):
        import importlib
        for m in PREIMPORT:
            try:
                importlib.import_module(m)
            except Exception:                     # noqa: BLE001 -- not every cell has every library
                pass
        try:
            import robot
            robot.kernel = sys.modules.get("robot.kernel", sys.modules[__name__])
            robot._BEFORE_COMMAND = _before_command
        except Exception:                         # noqa: BLE001
            pass
        try:
            exec("import robot, numpy as np, cv2\nfrom robot import viz, watch, when, kernel", self.ns)
        except Exception:                         # noqa: BLE001
            pass
        self.ready.set()

    def serve(self):
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(SOCK)
        os.chmod(SOCK, 0o600)
        srv.listen(16)
        threading.Thread(target=self.preimport, daemon=True).start()
        srv.settimeout(10.0)
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if self.busy == 0 and time.time() - self.last > IDLE_S:
                    break
                continue
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        srv.close()
        os.unlink(SOCK)

    def handle(self, conn):
        req = _recv(conn)
        if req is None:
            conn.close()
            return
        if req.get("op") == "stop":
            _send(conn, {"exit": 0})
            conn.close()
            os._exit(0)
        if req.get("op") == "status":
            _send(conn, {"out": f"robot.kernel: pid {os.getpid()}, {self.busy} call(s) running\n"})
            _send(conn, {"exit": 0})
            conn.close()
            return
        with self.lock:
            self.busy += 1
        try:
            self.run(conn, req)
        finally:
            with self.lock:
                self.busy -= 1
                self.last = time.time()
            try:
                conn.close()
            except OSError:
                pass

    def run(self, conn, req):
        import ctypes

        import robot
        from robot import viz
        lock = threading.Lock()
        alive = [True]

        def emit(key, s):
            if not alive[0]:
                return
            try:
                with lock:
                    _send(conn, {key: s})
            except OSError:
                alive[0] = False

        me = threading.current_thread()
        opened = []                               # this call's robot connections, closed as it ends

        def watch_peer():                         # the client went away: stop this call's code
            try:
                conn.recv(1)
            except OSError:
                pass
            if alive[0]:
                alive[0] = False
                for r in list(opened):            # robotd stops a closed connection's motions at once, and a call
                    try:                          # waiting on one of them wakes up
                        r.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(me.ident), ctypes.py_object(KeyboardInterrupt))

        self.out.local.out = self.err.local.out = emit
        robot._KERNEL_CALL = getattr(robot, "_KERNEL_CALL", threading.local())
        robot._KERNEL_CALL.opened = opened
        views_start = len(viz._SHOWN)
        call_t0 = time.time()
        code, rc = req.get("code") or "", 0
        if req.get("cwd") and os.path.isdir(req["cwd"]) and req["cwd"] != os.getcwd():
            os.chdir(req["cwd"])
        threading.Thread(target=watch_peer, daemon=True).start()
        self.ready.wait(120)
        try:
            exec(compile(code, req.get("name") or "<kernel>", "exec"), self.ns)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except KeyboardInterrupt:
            rc = 130
        except BaseException:                     # noqa: BLE001 -- the call's error, the kernel's survival
            import traceback
            etype, value, tb = sys.exc_info()
            tb = tb.tb_next if tb is not None else tb       # leave out this frame
            emit("err", "".join(traceback.format_exception(etype, value, tb)))
            rc = 1
        finally:
            try:
                moved = [r for r in opened if getattr(r, "_moved", False)]
                ended, pushed = _report_actions(call_t0) if alive[0] else ([], [])
                src = None
                if moved or ended:
                    try:
                        src = next((x for x in reversed(moved or opened) if _is_open(x)), None) or \
                            robot.connect(name="kernel-state")
                        if os.environ.get("ROBOT_EXIT_STATE", "1") != "0":
                            print("[state] " + src.state().summary())
                    except Exception:             # noqa: BLE001
                        pass
                if alive[0] and rc != 130:
                    for a in ended:               # the look after an action: every camera, in this call's output
                        if a.views and a.r is not None and src is not None:
                            try:
                                viz.show_cameras(f"after-{a.name}", src)
                            except Exception as e:    # noqa: BLE001
                                print(f"[action {a.name}] no after-motion views: {type(e).__name__}: {e}", flush=True)
                    _run_hooks(opened, moved, bool(ended))
                new = list(dict.fromkeys(viz._SHOWN[views_start:] + pushed))
                if new and os.environ.get("ROBOT_EXIT_VIEWS", "1") != "0":
                    print("[views] " + json.dumps(new))
            finally:
                for r in opened:                  # its motions stop, as when a program exits
                    try:
                        r.close()
                    except Exception:             # noqa: BLE001
                        pass
                if robot._latest() in opened:     # a closed connection is no one's default any more
                    robot._LATEST[:] = []
                robot._KERNEL_CALL.opened = None
                self.out.local.out = self.err.local.out = None
                emit("exit", rc)
                alive[0] = False


def _main(argv) -> int:
    if argv[:1] == ["--serve"]:
        sys.modules["robot.kernel"] = sys.modules[__name__]   # `from robot import kernel` in a call: this module,
        k = Kernel()                                           # with its hooks (not a second copy)
        print(f"robot.kernel: serving {SOCK} (pid {os.getpid()})", file=sys.__stderr__, flush=True)
        k.serve()
        return 0
    if argv[:1] in (["--stop"], ["--status"]):
        s = _connect(start=False)
        if s is None:
            print("robot.kernel: none running")
            return 0
        _send(s, {"op": argv[0][2:]})
        while (m := _recv(s)) is not None and "exit" not in m:
            sys.stdout.write(m.get("out", ""))
        return 0
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if argv:
        name = argv[0]
        code = open(name).read()
    else:
        name, code = "<stdin>", sys.stdin.read()
    return run_client(code, name)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
