"""robotd: the driver process. One per robot; agents connect over a unix or TCP socket.

    python -m robotd.server --sim yam --scene pick_cube --addr unix:/tmp/robot.sock --record runs/demo
    python -m robotd.server --sim yam --scene bounce_ball --realtime ...      (simulation against the wall clock)
    python -m robotd.server --real yam --addr tcp:0.0.0.0:7410 --record runs/cell        (see backends/yam_i2rt.py)

Simulation is lockstep: the control core ticks only inside a blocking request (a command with wait=True, wait(id),
step), so the world is paused whenever the agent is thinking; requests from several clients are served one at a time.
A real robot ticks continuously in a control thread; each client has its own thread, so a stop() from a second
terminal takes effect while another client is blocked in a motion.

When a client disconnects (its program ended, crashed or was interrupted), the commands it started are stopped and
the arms hold where they are.

Operator requests (the dashboard, robotd.dashboard; not part of the agent's API): operator_stop (freeze or soft,
latched until release), release, park, health and events (the driver's event log).

--control-addr opens a second socket for the process that runs an episode (scripts/agent_run.py), placed where the
agent cannot reach it: evaluate (the task's success check, returned to that caller only; settle_s first lets the world
run that long with the arms holding), description, status and shutdown. With it, a shutdown on the agents' socket is
refused.

    python -m robotd.server --sim yam --scene pick_cube --realtime     # simulation on the wall clock, like a real robot
    python -m robotd.server --real yam --monitor                       # the real robot with its motors off: watch only
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import gc
import itertools
import json
import os
import select
import signal
import socket
import threading
import time
import traceback

import numpy as np

from robot import rpc
from robotd.core.controller import Busy, Controller, Fault, InvalidCommand
from robotd.core.recorder import Recorder
from robotd.views import ViewBoard

MAX_BLOCK_S = 600.0
PARK_SPEED = 0.3                 # rad/s: parking on exit and the operator's park
EVENT_LOG = 1000


class GcWatch:
    """Python's cyclic garbage collector, timed (gc.callbacks). A collection holds the interpreter: every thread
    waits, the control loop's too, so a long one is a stall. Keeps the collections over 2 ms: (end, s, generation,
    objects collected); counts and total time of all of them."""
    KEEP_S = 0.002

    def __init__(self):
        self._t0 = None
        self.events = collections.deque(maxlen=64)
        self.count = collections.Counter()
        self.total = 0.0
        gc.callbacks.append(self._cb)

    def _cb(self, phase, info):
        now = time.perf_counter()
        if phase == "start":
            self._t0 = now
            return
        if self._t0 is None:
            return
        d, self._t0 = now - self._t0, None
        self.total += d
        self.count[info.get("generation", -1)] += 1
        if d > self.KEEP_S:
            self.events.append((now, d, info.get("generation", -1), info.get("collected", 0)))

    def during(self, t0: float, t1: float) -> list:
        """The kept collections that overlap [t0, t1]: [(ms, generation, collected)]."""
        return [(round(d * 1000, 1), g, c) for end, d, g, c in list(self.events) if end >= t0 and end - d <= t1]


class LoopStats:
    """The real-time control loop's timing, for the dashboard and health: per tick, the period (wake to wake), the
    wait for the driver lock (held by request threads) and the tick's own work, over the last WINDOW_S, plus how many
    ticks came late since the start. A long period with short lock wait and work means the loop thread was not run
    on time: another thread held the interpreter (GIL) or the CPU."""
    WINDOW_S = 5.0
    LATE_S, STALL_S = 0.006, 0.020

    def __init__(self, period_s: float):
        n = max(10, int(self.WINDOW_S / period_s))
        self.period_s = period_s
        self.period, self.lock_wait, self.work = (collections.deque(maxlen=n) for _ in range(3))
        self.sections = collections.deque(maxlen=n)        # per tick: {phase: s} (Controller.sec)
        self.ticks = collections.deque(maxlen=n)           # per tick: (wake, lock wait, work, sections)
        self.stall_log = collections.deque(maxlen=20)      # each stall: the tick before it, which ran long or late
        self.gc = GcWatch()
        self.prev = None
        self.late = self.stalls = 0
        self.worst = 0.0
        self.last = None
        self.mutex = threading.Lock()

    def add(self, wake: float, locked: float, done: float, sections: dict | None = None) -> None:
        with self.mutex:
            if sections:
                self.sections.append(sections)
            if self.last is not None:
                p = wake - self.last
                self.period.append(p)
                self.late += p > self.LATE_S
                self.stalls += p > self.STALL_S
                self.worst = max(self.worst, p)
                if p > self.STALL_S and self.prev is not None:
                    # the period ends at this wake; the tick before it began it: its lock wait, its work (and phases),
                    # and the gap from its end to this wake (sleep, or the thread not run on time)
                    w0, l0, d0, s0 = self.prev
                    self.stall_log.append({"wake": wake, "period": p, "lock": l0 - w0, "work": d0 - l0,
                                           "gap": wake - d0, "phases": dict(s0 or {}),
                                           "gc": self.gc.during(w0, wake)})
            self.lock_wait.append(locked - wake)
            self.work.append(done - locked)
            self.ticks.append((wake, locked - wake, done - locked, sections))
            self.prev = (wake, locked, done, sections)
            self.last = wake

    def summary(self) -> dict | None:
        with self.mutex:
            p, lk, w = np.array(self.period), np.array(self.lock_wait), np.array(self.work)
            late, stalls, worst = self.late, self.stalls, self.worst
            secs = list(self.sections)
            ticks, log = list(self.ticks), list(self.stall_log)
        if len(p) < 2:
            return None
        ms = lambda x, q=None: round(float(np.max(x) if q is None else np.percentile(x, q)) * 1000, 2)
        return {"target_ms": round(self.period_s * 1000, 2), "window_s": self.WINDOW_S,
                "period_ms": {"p50": ms(p, 50), "p99": ms(p, 99), "max": ms(p)},
                "work_ms": {"mean": round(float(w.mean()) * 1000, 3), "p99": ms(w, 99), "max": ms(w)},
                "lock_ms": {"p99": ms(lk, 99), "max": ms(lk)},
                "busy_pct": round(100.0 * float(w.mean()) / self.period_s, 1),
                "late_window": int((p > self.LATE_S).sum()), "stalls_window": int((p > self.STALL_S).sum()),
                "late": int(late), "stalls": int(stalls), "worst_ms": round(worst * 1000, 1),
                "phases_ms": {k: {"mean": round(float(np.mean(v)) * 1000, 3), "p99": round(float(np.percentile(v, 99))
                                                                                    * 1000, 3),
                                  "max": round(float(np.max(v)) * 1000, 3)}
                              for k in dict.fromkeys(k for d in secs for k in d)
                              for v in [[d.get(k, 0.0) for d in secs]]},
                "slowest_tick": self._tick_json(max(ticks, key=lambda x: x[2])) if ticks else None,
                "stall_log": [{"ago_s": round(now - e["wake"], 1), "period_ms": round(e["period"] * 1000, 1),
                               "lock_ms": round(e["lock"] * 1000, 2), "work_ms": round(e["work"] * 1000, 2),
                               "gap_ms": round(e["gap"] * 1000, 2),
                               "phases_ms": {k: round(v * 1000, 2) for k, v in e["phases"].items() if v > 5e-4},
                               "gc": e["gc"]} for e in log for now in [time.perf_counter()]],
                "gc": {"collections": {str(k): v for k, v in sorted(self.gc.count.items())},
                       "total_ms": round(self.gc.total * 1000, 1),
                       "over_2ms": [(round(time.perf_counter() - end, 1), round(d * 1000, 1), g, c)
                                    for end, d, g, c in list(self.gc.events)[-10:]]}}

    @staticmethod
    def _tick_json(t) -> dict:
        wake, lock, work, sections = t
        return {"ago_s": round(time.perf_counter() - wake, 2), "lock_ms": round(lock * 1000, 2),
                "work_ms": round(work * 1000, 2),
                "phases_ms": {k: round(v * 1000, 2) for k, v in (sections or {}).items() if v > 5e-4}}


class Client:
    """One connection: the commands it started (stopped when it goes away)."""
    _ids = itertools.count(1)

    def __init__(self, sock=None):
        self.sock = sock
        self.id = next(self._ids)
        self.cmds = set()
        self.id = next(self._ids)
        self.name = None

    def gone(self) -> bool:
        """The client closed its end. It sends nothing while it waits for a reply, so readable means EOF."""
        if self.sock is None:
            return False
        try:
            readable, _, _ = select.select([self.sock], [], [], 0)
            return bool(readable) and self.sock.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True


class NotSupported(RuntimeError):
    pass


class Driver:
    def __init__(self, backend, desc, record_dir=None, evaluator=None, dev=False, task=None):
        self.backend = backend
        self.dev = dev                                  # developer ops (free-viewpoint render): never for agents
        self.desc = desc
        self.c = Controller(backend, desc)
        self.lockstep = backend.lockstep
        self.passive = bool(getattr(backend, "passive", False))     # monitor mode: the motors are off
        self.cmds = {}
        self.evaluator = evaluator
        self.finished = None
        self.public_shutdown = True                     # False when a control socket owns shutdown
        self.rec = Recorder(record_dir) if record_dir else None
        self.c.recorder = self.rec
        if self.rec is not None:
            info = getattr(backend, "scene_info", None)
            self.rec.meta({"schema": "robotd.run.v1", "backend": backend.name, "robot": desc["robot"],
                           "lockstep": backend.lockstep, "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
                           "fast_tick_s": desc["fast_tick_s"], "tick_s": desc["tick_s"], "joints": self.c.joints,
                           "arm_joints": desc["arm_joints"], "finger_joint": desc["finger_joint"],
                           "tcp_sites": desc["tcp_sites"], "task": task, "scene": info() if info else None,
                           "fence": desc.get("fence")})
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self._stop = False
        self.loop_stats = LoopStats(desc["fast_tick_s"])
        self.views = ViewBoard()                        # images programs show (robot.viz.show), for the dashboard
        self.events = collections.deque(maxlen=EVENT_LOG)
        self.event_seq = 0
        self.events_lock = threading.Lock()
        self.clients = {}
        self.home_q = self.c.q_ref.copy()                     # where the arms were at start-up: park's fallback
        self.c.listener = self.event
        w_max = desc.get("gripper_width_range_m", [0.0, 0.094])[1]
        for a in self.c.arms:                                 # a gripper at a stop gets moved off it, and a jammed
            w = self.c.gripper_state[a]["width"]              # one would push hard: say so (the guard relaxes it)
            if w > 0.985 * w_max or w < 0.015 * w_max:
                self.event("warning", f"{a} gripper starts at its {'open' if w > w_max / 2 else 'closed'} stop "
                                      f"({w * 1000:.1f} mm): it is moved off it; if it is jammed there it will be relaxed")
        # a scene's tools handed over before the episode: each gripper closes on its tool as the clock starts
        for arm, force in (getattr(getattr(backend, "cfg", None), "start_grips", None) or {}).items():
            self.c.submit({"op": "gripper", "name": arm, "width": 0.0, "force": float(force)})
        self.thread = None
        # a real robot ticks from the start; a real-time simulation starts its clock when the first program connects
        # (as in lockstep, the world does not run before anyone is there, so a scene's timed events stay in step;
        # the dashboard watching does not count)
        if not self.lockstep and not getattr(backend, "realtime", False):
            self._start_clock()

    def _start_clock(self):
        if self.thread is None:
            # Python's full garbage collection scans every tracked object and holds the interpreter while it does:
            # with the models, i2rt and the driver loaded it took 24-34 ms on the cell (2026-11-04), about once a
            # minute, stalling the control loop. Everything built so far lives as long as robotd: freeze it, so a
            # collection scans only what was made since.
            gc.collect()
            gc.freeze()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    # ------------------------------------------------------------------ time
    def _loop(self):
        """Real robots: tick forever; backend.advance() paces the loop to the control period."""
        while not self._stop:
            wake = time.perf_counter()
            with self.cv:
                locked = time.perf_counter()
                try:
                    self.c.tick(advance=False)
                except Exception as e:                       # keep holding; report on the next request
                    self.c._latch_fault(f"control loop error: {e}")
                self.cv.notify_all()
            self.loop_stats.add(wake, locked, time.perf_counter(), getattr(self.c, "sec", None))
            self.backend.advance(self.desc["fast_tick_s"])  # wait for the next period without the lock

    def advance_until(self, pred, timeout=None, client: Client | None = None):
        limit = MAX_BLOCK_S if timeout is None else float(timeout)
        gone = client.gone if client is not None else (lambda: False)
        if self.lockstep:
            t0 = self.c.t
            n = 0
            while not pred() and self.c.t - t0 < limit:
                self.c.tick()
                n += 1
                if n % 25 == 0 and gone():
                    return False
            return pred()
        deadline = time.time() + limit
        with self.cv:
            while not pred():
                if time.time() > deadline or gone():
                    return False
                self.cv.wait(0.05)
        return True

    def event(self, kind: str, text: str, **info):
        """Append to the event log (the dashboard shows it). A burst of the same entries (a servo loop's command and
        finish, alternating) collapses into one entry each, kept at the end."""
        now = time.time()
        window = 1.0 if kind in ("command", "finish") else 5.0
        with self.events_lock:
            for last in list(self.events)[-4:]:
                if last["kind"] == kind and last["text"] == text and now - last["wall"] < window:
                    self.event_seq += 1
                    last.update(info, seq=self.event_seq, wall=now, count=last.get("count", 1) + 1)
                    self.events.remove(last)
                    self.events.append(last)
                    return
            self.event_seq += 1
            self.events.append({"seq": self.event_seq, "wall": now, "kind": kind, "text": text, **info})

    def events_since(self, since: int, limit: int = 300) -> dict:
        with self.events_lock:
            new = [dict(e) for e in self.events if e["seq"] > since]
            return {"seq": self.event_seq, "events": new[-limit:]}

    def release(self, client: Client):
        """A client went away: stop what it started, then= chains and what on= started included (the arms hold)."""
        self.clients.pop(client.id, None)
        if client.name and not client.name.startswith("dashboard"):
            self.event("client", f"client {client.name} disconnected", client=client.name)
        with self.lock:
            todo = [self.cmds.get(cid) for cid in client.cmds]
            while todo:
                c = todo.pop()
                if c is None:
                    continue
                if not c.finished:
                    self.c._finish(c, "stopped", "client_disconnected")
                    self.c._hold_measured(c)
                todo += [c.next] + c.spawned
            client.cmds.clear()

    # ------------------------------------------------------------------ requests
    def handle(self, req: dict, client: Client | None = None):
        op = req.get("op")
        try:
            return self._handle(op, req, client)
        except (Fault, NotSupported):
            if op in ("command", "park") and (self.c.fault or self.c.parked or self.passive):   # tried to move
                who = client.name if client is not None and client.name else "a client"       # while it cannot
                what = req["spec"].get("op") if op == "command" else op
                why = "monitor mode, the motors are off" if self.passive else (self.c.fault or self.c.parked)["reason"]
                self.event("refused", f"{what} refused: {why}", client=who)
            raise

    def _handle(self, op, req, client):
        if self.passive and op in ("command", "park"):
            raise NotSupported("monitor mode: the motors are off; restart robotd without --monitor to move the arms")
        if op == "cameras" and self.desc.get("camera_addr"):
            raise NotSupported(f"the cameras are served by camerad at {self.desc['camera_addr']}: robot.Robot.cameras() "
                               "fetches from there (update the robot client)")
        if op == "cameras" and not self.lockstep:          # a camera grab must not hold up the control loop
            return _crop(self.backend.cameras(req.get("names"), req.get("depth", True)), req.get("roi"))
        if op == "ik":                                     # a query: nothing moves (its own kinematics, no lock)
            return self.c.solve_ik(req["ee"], req["target"], req.get("tcp"), req.get("seed"), req.get("free"),
                                   req.get("after"))
        if op == "reachable":
            return self.c.reachable(req["ee"], req["targets"], req.get("tcp"), req.get("free"), req.get("line", False),
                                    req.get("after"))
        if op == "command" and req["spec"].get("dry_run"):      # checked, never run
            return self.c.dry_run(req["spec"])
        prepared = None
        if op == "command" and not self.lockstep:          # nor an IK solve
            prepared = self.c.prepare(req["spec"])
        if op == "health":                                  # backend health reads cached values: no lock needed
            return {"backend": self.backend.health(), "name": self.backend.name, "lockstep": self.lockstep,
                    "passive": self.passive, "loop": self.loop_stats.summary(),
                    "ticks": self.c.ticks, "t": self.c.t, "clients": sorted(c for c in self.clients.values() if c),
                    "settle_bias_deg": {a: np.round(np.degrees(self.c.bias[self.c.arm_cols[a]]), 2).tolist()
                                        for a in self.c.arms}}
        if op == "events":
            return self.events_since(int(req.get("since", 0)), int(req.get("limit", 300)))
        if op == "show":                                    # robot.viz.show: an image for the dashboard
            return self.views.show(req, client.name if client is not None else None)
        if op == "views":
            return self.views.since(int(req.get("since", 0)))
        with self.lock:
            if op == "hello":
                name = None
                if client is not None:
                    name = req.get("client") or f"#{client.id}"
                    if client.name is None and not name.startswith("dashboard"):
                        self.event("client", f"client {name} connected", client=name)
                    client.name = name
                    self.clients[client.id] = name
                if not self.lockstep and self.thread is None and not (name or "").startswith("dashboard"):
                    if hasattr(self.backend, "warm_up"):
                        self.backend.warm_up()
                    self._start_clock()
                return {"description": self.desc, "lockstep": self.lockstep, "backend": self.backend.name,
                        "passive": self.passive}
            if op == "state":
                return self.c.state()
            if op == "command":
                spec = req["spec"]
                cmd = self.c.commit(prepared) if prepared is not None else self.c.submit(spec)
                self.cmds[cmd.id] = cmd
                if client is not None:
                    client.cmds.add(cmd.id)
                self._prune()
                if not spec.get("wait", True):
                    return {"id": cmd.id}
                self.advance_until(cmd.chain_finished, spec.get("timeout"), client)
                last = cmd.last()
                return last.result or {"id": last.id, "status": "running", "reason": "max_block_time"}
            if op == "wait":
                cmd = self._cmd(req["id"])
                self.advance_until(cmd.chain_finished, req.get("timeout"), client)
                last = cmd.last()
                return last.result or {"id": last.id, "status": "running", "reason": "timeout"}
            if op == "status":
                cmd = self._cmd(req["id"])
                last = cmd.last()
                return {"finished": cmd.chain_finished(), "result": last.result, "status": last.status}
            if op == "cancel":
                cmd = self._cmd(req["id"])
                for c in (cmd, cmd.last()):
                    if not c.finished:
                        self.c._finish(c, "stopped", "cancelled")
                        self.c._hold_measured(c)
                return True
            if op == "step":
                n = int(req.get("ticks", 1)) * int(round(self.desc["tick_s"] / self.desc["fast_tick_s"]))
                if self.lockstep:
                    for _ in range(n):
                        self.c.tick()
                else:
                    target = self.c.ticks + n
                    self.advance_until(lambda: self.c.ticks >= target, client=client)
                return self.c.state()
            if op == "step_until":
                t = float(req["t"])
                if self.lockstep:
                    t0 = self.c.t
                    while self.c.t < t - 1e-9 and self.c.t - t0 < MAX_BLOCK_S:
                        self.c.tick()
                else:
                    self.advance_until(lambda: self.c.t >= t - 1e-9, client=client)
                return self.c.state()
            if op == "stop":
                self.c.stop()
                return True
            if op == "zero_force":
                self.c.zero_force(req["ee"])
                return True
            if op == "recover":
                self.c.recover()
                return True
            if op == "operator_stop":
                self.c.operator_stop(soft=bool(req.get("soft", False)))
                return self.c.fault
            if op == "release":
                self.c.recover(operator=True)
                return True
            if op == "park":
                # the operator's way out, as the exit park: a guard's fault is cleared and does not stop it on the
                # way (cmd.park). So is the operator's own stop: PARK after STOP parks (an arm stopped pressing on
                # something must be able to get off it without a RELEASE that lets programs move it first)
                if self.c.fault:
                    self.c.recover(operator=bool(self.c.fault.get("operator")))
                self.c.operator_park()                       # latched: programs stopped and refused until release
                ids = []
                if (self.desc.get("separation") or {}).get("fine"):   # hands working close together: let go first, so
                    for arm in self.c.arms:                           # that they do not pull on what they hold
                        cmd = self.c.submit({"op": "gripper", "name": arm, "width": 0.09}, park=True)
                        self.cmds[cmd.id] = cmd
                        if client is not None:
                            client.cmds.add(cmd.id)
                        ids.append(cmd.id)
                for arm, goal in self._park_goals().items():
                    cmd = self.c.submit({"op": "move_to", "joints": self.desc["arm_joints"][arm], "target": goal,
                                         "speed": PARK_SPEED}, park=True)
                    self.cmds[cmd.id] = cmd
                    if client is not None:
                        client.cmds.add(cmd.id)
                    ids.append(cmd.id)
                return {"ids": ids}
            if op == "signal":
                self.c.signals[str(req["name"])] = float(req.get("value", 1.0))
                return True
            if op == "cameras":
                return _crop(self.backend.cameras(req.get("names"), req.get("depth", True)), req.get("roi"))
            if op == "finish":
                self.finished = {"summary": req.get("summary", ""), "t": self.c.t}
                if self.evaluator is not None and self.rec is not None:
                    verdict = self.evaluator(self.backend)         # privileged: written to disk, never returned
                    with open(os.path.join(self.rec.dir, "verdict.json"), "w") as f:
                        json.dump({"verdict": verdict, **self.finished}, f, default=str)
                return {"ok": True}
            if op == "render_free" and self.dev and hasattr(self.backend, "render_free"):
                return self.backend.render_free(**req.get("view", {}))
            if op == "dev_truth" and self.dev:                # privileged simulator state, for developers only
                b = self.backend
                return {"cloth": {k: v.tolist() for k, v in getattr(b, "privileged_deformables", dict)().items()},
                        "objects": {k: v.tolist() for k, v in getattr(b, "privileged_poses", dict)().items()},
                        "cloth_held": getattr(b, "cloth_held", dict)()}
            if op == "ping":
                return {"t": self.c.t}
            if op == "shutdown":
                if not self.public_shutdown:
                    raise NotSupported("shutdown is not available on this socket")
                return True
        raise InvalidCommand(f"unknown request {op!r}")

    def _park_goals(self) -> dict:
        """arm -> 6 joints: the backend's park pose, else where the robot started."""
        park = getattr(self.backend, "park_q", None)              # {arm: 6 joints} or one list for both arms
        start = getattr(self.backend, "start_q", None)
        if start is None:
            start = self.home_q
        goals = {}
        for arm in self.c.arms:
            js = self.desc["arm_joints"][arm]
            if isinstance(park, dict) and arm in park:
                goals[arm] = list(map(float, park[arm][:6]))
            elif park is not None and not isinstance(park, dict):
                goals[arm] = list(map(float, park[:6]))
            else:
                goals[arm] = [float(start[self.c.jidx[n]]) for n in js]
        return goals

    def _park(self):
        """Real robots: bring both arms to the park pose (or where they started) slowly and hold, before the
        process exits and the motors time out. A guard that latches on the way (an arm resting on a stop reads as
        pushing) does not stop it: guards stop programs, and without the park the arms drop. The operator's stop
        still does."""
        try:
            with self.lock:                                  # the control thread ticks meanwhile
                self.c.parking = True
                self.c.recover(operator=True)
            for arm, goal in self._park_goals().items():
                with self.lock:
                    cmd = self.c.submit({"op": "move_to", "joints": self.desc["arm_joints"][arm], "target": goal,
                                         "speed": PARK_SPEED})
                self.advance_until(cmd.chain_finished, 30.0)
                if cmd.status != "done":
                    print(f"robotd: the {arm} arm's park ended {cmd.status}: {(cmd.result or {}).get('reason')}",
                          flush=True)
        except Exception as e:                               # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"robotd: park failed: {e!r}", flush=True)

    def _cmd(self, cid):
        if cid not in self.cmds:
            raise InvalidCommand(f"unknown command id {cid}")
        return self.cmds[cid]

    def _prune(self):
        if len(self.cmds) > 2000:
            for k in sorted(self.cmds)[:1000]:
                self.cmds.pop(k, None)

    def close(self):
        if not self.lockstep and not self.passive:
            self._park()
        self._stop = True
        if self.thread is not None:
            self.thread.join(1.0)                            # its last tick is done before the backend lets go
        timing = getattr(self.backend, "timing", None)
        if self.rec:
            if timing and timing.get("ticks"):             # a realtime simulation: did it keep the 4 ms period?
                with open(os.path.join(self.rec.dir, "timing.json"), "w") as f:
                    json.dump(dict(timing, wall_s_per_tick=timing["physics_s"] / timing["ticks"]), f)
            self.rec.close()
        self.backend.close()


def _crop(frames: dict, roi) -> dict:
    """Cut each frame to a pixel box [u0, v0, u1, v1] (one for all cameras, or {camera: box}): less to send and to
    look through. K is shifted to match, so the cropped frame projects as before."""
    if not roi:
        return frames
    for name, f in frames.items():
        box = roi.get(name) if isinstance(roi, dict) else roi
        if box is None:
            continue
        H, W = f["rgb"].shape[:2]
        u0, v0, u1, v1 = (int(round(float(x))) for x in box)
        u0, u1 = max(0, min(u0, W - 1)), max(1, min(u1, W))
        v0, v1 = max(0, min(v0, H - 1)), max(1, min(v1, H))
        if u1 <= u0 or v1 <= v0:
            raise InvalidCommand(f"roi {box} of camera {name!r} is empty in a {W}x{H} image")
        f["rgb"] = f["rgb"][v0:v1, u0:u1].copy()
        if f.get("depth") is not None:
            f["depth"] = f["depth"][v0:v1, u0:u1].copy()
        K = np.array(f["K"], float)
        K[0, 2] -= u0
        K[1, 2] -= v0
        f["K"], f["roi"] = K, [u0, v0, u1, v1]
    return frames


def _error(e: Exception) -> dict:
    kind = {Fault: "Fault", InvalidCommand: "InvalidCommand", Busy: "Busy", NotSupported: "NotSupported",
            NotImplementedError: "NotSupported", ValueError: "InvalidCommand"}
    for cls, name in kind.items():
        if isinstance(e, cls):
            return {"type": name, "message": str(e)}
    return {"type": "RobotError", "message": f"{type(e).__name__}: {e}"}


def _serve_client(driver: Driver, conn, done: threading.Event):
    client = Client(conn)
    try:
        while not done.is_set():
            try:
                req = rpc.recv(conn)
            except (ConnectionError, OSError):
                req = None
            if req is None:
                break
            t0, rt0 = time.time(), driver.c.t
            try:
                rep = {"ok": True, "result": driver.handle(req, client)}
            except Exception as e:                             # noqa: BLE001 -- every error goes back to the agent
                rep = {"ok": False, "error": _error(e)}
                if not isinstance(e, (Fault, InvalidCommand, Busy, NotSupported)):   # a ValueError can be a bug too
                    traceback.print_exc()
            if driver.rec is not None:
                with driver.rec.lock:                          # not the driver's lock: PNG encoding takes time
                    driver.rec.request(req, rep, time.time() - t0, client.id, rt0, driver.c.t)
            try:
                rpc.send(conn, rep)
            except (ConnectionError, OSError):
                break
            if req.get("op") == "shutdown" and rep["ok"]:
                done.set()
                break
    finally:
        driver.release(client)
        with contextlib.suppress(OSError):
            conn.close()


def _serve_control(driver: Driver, conn, done: threading.Event):
    """The episode runner's connection: evaluate (the success check, for this caller only), description, status,
    shutdown."""
    try:
        while not done.is_set():
            try:
                req = rpc.recv(conn)
            except (ConnectionError, OSError):
                req = None
            if req is None:
                break
            op = req.get("op")
            try:
                if op == "evaluate":
                    settle = float(req.get("settle_s") or 0.0)
                    t_before = driver.c.t
                    if settle > 0:                         # the arms hold; cloth and stacks come to rest
                        with driver.lock:
                            driver.c.stop()
                        if driver.lockstep:
                            with driver.lock:
                                for _ in range(int(round(settle / driver.desc["fast_tick_s"]))):
                                    driver.c.tick()
                        else:
                            driver._start_clock()          # a real-time world nobody connected to has not started
                            driver.advance_until(lambda: driver.c.t >= t_before + settle)
                    with driver.lock:
                        verdict = driver.evaluator(driver.backend) if driver.evaluator is not None else None
                        result = {"verdict": verdict, "t": driver.c.t, "t_before_settle": t_before,
                                  "settle_s": settle, "finished": driver.finished}
                elif op == "description":
                    result = driver.desc
                elif op == "status":
                    result = {"t": driver.c.t, "finished": driver.finished, "clients": sorted(driver.clients.values())}
                elif op == "shutdown":
                    result = True
                else:
                    raise InvalidCommand(f"unknown control request {op!r}")
                rep = {"ok": True, "result": result}
            except Exception as e:                             # noqa: BLE001
                traceback.print_exc()
                rep = {"ok": False, "error": _error(e)}
            try:                                               # a verdict may hold numpy values (as verdict.json)
                rpc.send(conn, json.loads(json.dumps(rep, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                                     else str(o))))
            except (ConnectionError, OSError):
                break
            if op == "shutdown":
                done.set()
                break
    finally:
        with contextlib.suppress(OSError):
            conn.close()


def _listen(addr: str) -> socket.socket:
    if addr.startswith("unix:"):
        path = addr[5:]
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
    else:
        host, port = addr[4:].rsplit(":", 1)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, int(port)))
    srv.listen(16)
    srv.settimeout(0.2)
    return srv


def _accept_loop(srv: socket.socket, done: threading.Event, handler, driver: Driver):
    while not done.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        conn.settimeout(None)
        threading.Thread(target=handler, args=(driver, conn, done), daemon=True).start()


def serve(driver: Driver, addr: str, ready_file: str | None = None, control_addr: str | None = None):
    srv = _listen(addr)
    done = threading.Event()
    ctl = None
    if control_addr:
        ctl = _listen(control_addr)
        driver.public_shutdown = False
        threading.Thread(target=_accept_loop, args=(ctl, done, _serve_control, driver), daemon=True).start()
    if ready_file:
        open(ready_file, "w").write(addr)
    print(f"robotd: {driver.backend.name} on {addr} (lockstep={driver.lockstep})", flush=True)
    try:
        _accept_loop(srv, done, _serve_client, driver)
    finally:
        driver.close()
        srv.close()
        if ctl is not None:
            ctl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["yam"], help="simulated robot")
    ap.add_argument("--real", choices=["yam"], help="real robot (needs the robot's SDK and hardware)")
    ap.add_argument("--scene", default="empty", help="a task name from tasks/ (its scene and success check)")
    ap.add_argument("--seed", type=int, default=None, help="scene variant, for scenes that have several (e.g. how "
                    "the towel lies crumpled)")
    ap.add_argument("--scene-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="a keyword for the task's scene, e.g. noise=0 (recorded in run.json; for tests)")
    ap.add_argument("--addr", default="unix:/tmp/robot.sock")
    ap.add_argument("--record", default=None, help="run directory for requests.jsonl, ticks/, verdict.json")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--control-addr", default=None, help="a second socket, for the episode runner only: evaluate "
                    "(the success check) and shutdown; shutdown on --addr is then refused")
    ap.add_argument("--speed-scale", type=float, default=None, help="default speeds x this (default 1 sim, 0.5 real)")
    ap.add_argument("--fence", default=None, help="a workspace fence for the TCPs, JSON or a JSON file: "
                    "'{\"left\": {\"x\": [0.15, 0.48], \"y\": [-0.05, 0.30], \"z\": [0.03, 0.12]}, \"right\": {...}, "
                    "\"max_speed_xy\": 1.0}' (docs/MANUAL.md). Default: the scene's fence, if it sets one; none")
    ap.add_argument("--separation-fine", type=float, default=None, metavar="M",
                    help="with --separation: when the capsules say the arms are too close, measure the hands' pairs on "
                         "their meshes and keep those at least M m apart (robotd/core/separation.py)")
    ap.add_argument("--separation", type=float, default=None, metavar="M",
                    help="keep the two arms at least M m apart (capsules around their links, robotd/core/separation.py)"
                         ": a goal closer is refused, an arm moving closer stops. 0: off. Default 0.02 on the real "
                         "robot, off in simulation")
    ap.add_argument("--max-gain-scale", type=float, default=None, metavar="S",
                    help="refuse a command's gains= stiffer than the defaults x S (kp; kd x sqrt(S)). Default 1 on the "
                         "real robot (a stiffened arm pressing on something hard chatters), none in simulation")
    ap.add_argument("--settle-bias", type=float, default=None, metavar="RATE",
                    help="the settled-error correction's rate (1/s): a bias on the arm joints' commands integrates "
                         "their steady error while they are still (robotd/core/controller.py, _settle_bias). 0: off. "
                         "Default: the description's (3)")
    ap.add_argument("--realtime", action="store_true", help="simulation: run against the wall clock in a control "
                    "thread, as the real robot does (perception and thinking then cost robot time)")
    ap.add_argument("--monitor", action="store_true",
                    help="real robot: keep the motors off; read joints, temperatures and cameras only")
    ap.add_argument("--cameras", default=None, metavar="ADDR",
                    help="real robot: the cameras are served by camerad at ADDR (python -m robotd.camerad); robotd "
                         "opens none (YAM_CAMERAS is ignored) and clients fetch frames from ADDR, so camera work never "
                         "competes with the control loop for this process's interpreter")
    ap.add_argument("--dev", action="store_true", help="enable developer ops (a free-viewpoint render of the "
                    "simulation); never give an agent a driver started with this")
    ap.add_argument("--gpu", type=int, default=os.environ.get("ROBOT_GPU"),
                    help="GPU (nvidia-smi index) for simulated cameras; use one that is not serving models")
    args = ap.parse_args()
    if args.sim:
        from robotd.backends.egl import use_gpu
        use_gpu(args.gpu)
    from robotd.robots.yam_station.description import description
    if args.sim == "yam":
        from robotd.backends.mujoco_sim import MujocoYamBackend
        import tasks                               # outside the agent's sandbox: scenes and success checks
        task = tasks.get(args.scene)
        kw = tasks.scene_args(args.scene_arg)
        scene = task.scene(**kw) if args.seed is None else task.scene(seed=args.seed, **kw)
        backend, evaluator = MujocoYamBackend(scene, realtime=args.realtime), task.evaluate
    elif args.real == "yam":
        from robotd.backends.yam_i2rt import YamI2rtBackend
        if args.cameras:                           # camerad has them: this process must not open them
            os.environ.pop("YAM_CAMERAS", None)
        backend, evaluator = YamI2rtBackend.from_env(passive=args.monitor), None
    else:
        ap.error("give --sim yam or --real yam")
    scale = args.speed_scale if args.speed_scale is not None else (1.0 if args.sim else 0.5)

    def _exit(signum, frame):                     # kill / a closed terminal exit like Ctrl-C: park, then exit
        raise SystemExit(f"robotd: signal {signum}")
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGHUP, _exit)
    meta = {"name": args.scene, "seed": args.seed, "args": tasks.scene_args(args.scene_arg)} if args.sim else None
    if args.realtime or args.real:
        # the control thread shares the interpreter with client threads (JSON, camera frames): hand the GIL over
        # often, or a 5 ms default switch interval alone can make a 4 ms tick late
        import sys
        sys.setswitchinterval(0.0005)
    from robotd.robots.yam_station.model_files import with_model_files
    from robotd.core.controller import parse_fence
    desc = with_model_files(description(scale), backend.kinematics_model())
    desc["fence"] = parse_fence(args.fence) if args.fence else parse_fence(getattr(getattr(backend, "cfg", None),
                                                                                    "fence", None))
    if desc["fence"]:
        print(f"robotd: workspace fence {json.dumps(desc['fence'])}", flush=True)
    sep_m = args.separation if args.separation is not None else (0.02 if args.real else 0.0)
    if sep_m > 0:
        from robotd.core.separation import YAM, yam_fine
        desc["separation"] = {"min_m": sep_m, "fault_m": 0.0, **YAM}
        print(f"robotd: the arms kept at least {sep_m * 100:.0f} cm apart (capsules around their links)", flush=True)
        if args.separation_fine is not None:              # a Fault at measured hands touching: their gap plus the shift
            desc["separation"].update(fine=yam_fine(args.separation_fine), fault_m=sep_m - args.separation_fine)
            print(f"robotd: the hands' pairs measured on their meshes: at least {args.separation_fine * 1000:.0f} mm",
                  flush=True)
    if args.settle_bias is not None:
        desc["settle_bias"] = dict(desc.get("settle_bias") or {}, rate=args.settle_bias) if args.settle_bias > 0 else None
    sb = desc.get("settle_bias")
    print("robotd: settled-error correction " + (f"at {sb['rate']:g}/s, at most {np.degrees(sb['max_rad']):.1f} deg"
                                                 if sb else "off"), flush=True)
    gain_cap = args.max_gain_scale if args.max_gain_scale is not None else (1.0 if args.real else None)
    if gain_cap:
        desc["max_gain_scale"] = gain_cap
        print(f"robotd: command gains at most {gain_cap:g} x the defaults", flush=True)
    if args.cameras:
        if not args.real:
            ap.error("--cameras is for the real robot (the simulation renders its cameras itself)")
        desc["camera_addr"] = args.cameras
        print(f"robotd: cameras served by camerad at {args.cameras}", flush=True)
    serve(Driver(backend, desc, args.record, evaluator, dev=args.dev, task=meta), args.addr,
          args.ready_file, args.control_addr)


if __name__ == "__main__":
    main()
