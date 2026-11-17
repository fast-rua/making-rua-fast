"""robot: the agent's interface to one robot, simulated or real.

    import robot
    from robot import when, Compliance
    r = robot.connect()                                   # $ROBOT_ADDR, e.g. unix:/tmp/robot.sock
    s = r.state()                                         # joints, end effectors, grippers, time
    r.move(robot.pose([0.35, 0.3, 0.15], rpy=[3.1416, 0, 0]), ee="left")     # a goal; a list of targets = a path
    res = r.move_at({"z": -0.01}, ee="left", max_travel=0.1,                 # guarded descent: res.reason is
                    until={"contact": when.force("left", "-z") > 3})         # 'contact' or 'max_travel'
    r.gripper("left", w - 0.003, speed=0.02, force=15)    # w: the object's width; res.reason 'stalled' = holding it

Poses are 4x4 numpy matrices in the world frame. Any array-like is accepted (numpy, torch, JAX, lists); everything
returned is numpy float64. A condition that fires is a normal Result (status 'stopped', reason = its name); exceptions
are for what you did not ask for: Aborted (a motion could not go on, e.g. the arm could not track it), Fault,
InvalidCommand, Busy, NotSupported. In simulation the world is paused
whenever your code is not inside a blocking call or r.step(); on a real robot time keeps running.
"""
from __future__ import annotations

import atexit
import os
import sys
import weakref

from robot import rpc, when
from robot._lazy import Lazy
from robot.geometry import T_from, asarray, inv, pose, twist, wxyz, xyzw

np = Lazy("numpy", __name__, "np")          # loaded at first use (robot/_lazy.py)

__all__ = ["connect", "when", "Compliance", "Move", "MoveAt", "Gripper", "pose", "wxyz", "xyzw", "T_from", "inv",
           "twist", "Aborted", "Fault", "InvalidCommand", "Busy", "NotSupported", "RobotError"]


class RobotError(RuntimeError):
    pass


_PKG_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_LIB_DIRS = tuple({os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})


def _call_site(max_frames: int = 8) -> list:
    """The policy code behind a request, innermost first: frames outside this package and outside the Python
    installation (the driver records them, so a run's replay can show which line issued each command)."""
    out, f = [], sys._getframe(2)
    while f is not None and len(out) < max_frames:
        path = f.f_code.co_filename
        real = os.path.realpath(path) if not path.startswith("<") else path
        if not real.startswith(_PKG_DIR) and not real.startswith(_LIB_DIRS) and not path.startswith("<frozen"):
            out.append([real, f.f_lineno, f.f_code.co_name])
        f = f.f_back
    return out


class Fault(RobotError):
    """A guard latched (e.g. max_force): the robot holds where it is until r.recover()."""


class InvalidCommand(RobotError, ValueError):
    pass


class Aborted(RobotError):
    """A motion stopped because it could not go on: the arm could not track its path ('cannot_track'), a then=/on=
    command was refused as it fired, a gripper jammed ('gripper_overload'). .result is its Result; the arm holds."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"{result.op} aborted: {result.reason}")


class Busy(RobotError):
    pass


class NotSupported(RobotError):
    pass


_ERRORS = {"Fault": Fault, "InvalidCommand": InvalidCommand, "Busy": Busy, "NotSupported": NotSupported}


# ------------------------------------------------------------------------------------------------ the law
class Compliance(dict):
    """Cartesian compliance for one command (implicit force control on a position-controlled arm).

    frame       'tool' (moves with the TCP) | 'world' | a 4x4
    stiffness   {"z": 300, "rx": 10}: N/m, Nm/rad on those axes; others stay as stiff as the servo
    force       {"-z": 6.0}: force the tool applies along signed axes (N). The reference is pushed in by
                F*(servo compliance + 1/K), trimmed slowly toward the measured force, never beyond max_penetration:
                if the surface disappears the tool travels at most max_penetration and stops.
    max_force   estimated force above this latches a Fault (the arm holds) -- a supervision, not a stop condition
    max_deviation {"x": 0.01}: cap of the compliant offset per axis (m)
    max_penetration  m, default 0.03
    ramp        s over which force targets ramp in, default 0.3
    """

    def __init__(self, frame="tool", stiffness=None, force=None, max_force=None, max_deviation=None,
                 max_penetration=0.03, ramp=0.3):
        super().__init__(frame=frame if isinstance(frame, str) else asarray(frame).tolist(),
                         stiffness=dict(stiffness or {}), force=dict(force or {}), max_force=max_force,
                         max_deviation=dict(max_deviation or {}), max_penetration=max_penetration, ramp=ramp)


# ------------------------------------------------------------------------------------------------ command builders
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "node"):
        return x.node
    if hasattr(x, "detach") or hasattr(x, "__array__"):
        return asarray(x).tolist()
    return x


def _common(spec, compliance=None, gains=None, tcp=None, until=None, then="hold", on=None, start_at=None,
            replace=True, wait=True, timeout=None):
    spec.update(compliance=compliance, gains=gains, tcp=tcp, until=until, then=then, on=on, start_at=start_at,
                replace=replace, wait=wait, timeout=timeout)
    return _clean(spec)


def _move_to(target, joints=None, ee=None, frame="world", line=False, speed=None, accel=None, duration=None,
             free=None, **common):
    """The wire form of a move with one goal."""
    return _common({"op": "move_to", "target": target, "joints": joints, "ee": ee, "frame": frame, "line": line,
                    "speed": speed, "accel": accel, "duration": duration, "free": free}, **common)


def _follow(points, times=None, joints=None, ee=None, frame="world", relative=False, dt=None, velocities=None,
            speed=None, duration=None, free=None, **common):
    """The wire form of a move along a timed path."""
    return _common({"op": "follow", "points": points, "times": times, "joints": joints, "ee": ee, "frame": frame,
                    "relative": relative, "dt": dt, "velocities": velocities, "speed": speed,
                    "duration": duration, "free": free}, **common)


def Move(target, times=None, joints=None, ee=None, frame="world", line=False, relative=False, dt=None,
         velocities=None, speed=None, accel=None, duration=None, free=None, **common):
    """Build (without running) a move command, e.g. for then=; see Robot.move. One target and nothing timing it is a
    goal; anything else a timed path."""
    first = next(iter(target.values())) if isinstance(target, dict) else target
    one = asarray(first).ndim == (1 if joints is not None else 2)
    if one and times is None and dt is None and velocities is None and not relative:
        return _move_to(target, joints, ee, frame, line, speed, accel, duration, free, **common)
    if line or accel is not None:
        raise InvalidCommand("line= and accel= go with one target and no times: a path's legs follow its spline")
    wrap = (lambda x: [x]) if one else (lambda x: x)
    pts = {a: wrap(v) for a, v in target.items()} if isinstance(target, dict) else wrap(target)
    if one and times is not None and np.ndim(times) == 0:
        times = [times]
    if one and velocities is not None:          # the target's velocity, bare or as a list of one (as times= is)
        vwrap = (lambda v: [v] if asarray(v).ndim == 1 else v)
        velocities = {a: vwrap(v) for a, v in velocities.items()} if isinstance(velocities, dict) else vwrap(velocities)
    if one and isinstance(free, str):
        free = [free]
    # frame="tool": targets in the TCP's frame at the start, as a goal reads one
    return _follow(pts, times, joints, ee, frame, relative or frame == "tool", dt, velocities, speed, duration, free,
                   **common)


def MoveAt(velocity, joints=None, ee=None, frame="world", accel=None, max_travel=None, duration=None, **common):
    return _common({"op": "move_at", "velocity": velocity, "joints": joints, "ee": ee, "frame": frame,
                    "accel": accel, "max_travel": max_travel, "duration": duration}, **common)


def Gripper(name, width, speed=None, force=None, **common):
    return _common({"op": "gripper", "name": name, "width": width, "speed": speed, "force": force}, **common)


# ------------------------------------------------------------------------------------------------ results and state
class Snapshot:
    def __init__(self, d):
        self.t = d.get("t")
        self.T = {k: np.asarray(v) for k, v in d.get("T", {}).items()}
        self.q = d.get("q", {})
        self.wrench = {k: np.asarray(v) for k, v in d.get("wrench", {}).items()}
        self.travel = d.get("travel", {})

    def __repr__(self):
        return f"Snapshot(t={self.t:.3f})"


class Result:
    """status: done | stopped (a condition fired) | replaced | timeout | aborted; reason: why (a condition's name,
    'reached', 'stalled', 'max_travel', 'cannot_track', ...); at: state when it ended."""

    def __init__(self, d):
        self.raw = d
        self.id, self.op = d.get("id"), d.get("op")
        self.status, self.reason = d.get("status"), d.get("reason")
        self.t, self.duration = d.get("t"), d.get("duration")
        self.width = d.get("width")
        self.empty = bool(d.get("empty"))      # a close that ended at the empty-jaw width: closed on nothing
        self.at = Snapshot(d.get("at") or {})
        g = d.get("goal")                       # with free=: the full pose(s) chosen
        self.goal = None if g is None else {a: np.asarray(v) for a, v in g.items()}

    @property
    def ok(self):
        return self.status in ("done", "stopped")

    def __repr__(self):
        extra = f", width={self.width:.4f}" if self.width is not None else ""
        extra += ", closed on nothing" if self.empty else ""
        return f"Result({self.op} {self.status}: {self.reason}, {self.duration:.2f}s{extra})"


class Plan:
    """What r.move(..., dry_run=True) found, without moving: ok; else reason ('unreachable'), point (which target or
    leg), fraction (how far along that leg), T (the pose it could not reach), error_m (by how much). q_goal: the joint
    goal(s); goal: the full pose(s) reached (where free= left axes open); margin: the tightest spot on the way
    (limit_rad and limit_joint: closest joint limit; singular_m: near 0 at a straight elbow or a folded wrist)."""

    def __init__(self, d):
        self.raw = d
        self.ok = bool(d.get("ok"))
        self.reason, self.point, self.fraction = d.get("reason"), d.get("point"), d.get("fraction")
        self.T = np.asarray(d["T"]) if d.get("T") is not None else None
        self.error_m, self.arm = d.get("error_m"), d.get("arm")
        self.q_goal = d.get("q_goal")
        g = d.get("goal")
        self.goal = {a: np.asarray(v) for a, v in g.items()} if isinstance(g, dict) else g
        self.margin = d.get("margin")

    def __repr__(self):
        if self.ok:
            return f"Plan(ok, margin={self.margin})"
        return (f"Plan(fails: {self.reason} at point {self.point}, {self.fraction:.0%} along its leg"
                f"{'' if self.arm is None else ' (' + self.arm + ')'}, misses by {1000 * (self.error_m or 0):.1f} mm)")


class EE:
    def __init__(self, d, trusted=True):
        self.trusted = trusted                  # the force estimate behind contact is trusted (simulation)
        self.T = np.asarray(d["T"])
        self.v = np.asarray(d["v"])
        self.wrench = np.asarray(d["wrench"])
        self.force = self.wrench[:3]
        self.deflection = np.asarray(d["deflection"])
        self.contact = d["contact"]

    def __repr__(self):
        if not self.trusted:
            return f"EE(xyz={np.round(self.T[:3, 3], 4)})"
        return f"EE(xyz={np.round(self.T[:3, 3], 4)}, force={np.round(self.force, 2)}, contact={self.contact})"


class State:
    def __init__(self, d):
        self.raw = d
        self.t = d["t"]
        self.joints = d["joints"]
        self._i = {n: i for i, n in enumerate(self.joints)}
        self._q, self._dq = np.asarray(d["q"]), np.asarray(d["dq"])
        self._effort, self._ext = np.asarray(d["effort"]), np.asarray(d["effort_ext"])
        self.q_ref = np.asarray(d["q_ref"])
        # what the motors are commanded (q_ref plus the settle trim): q - q_cmd is each joint's tracking error, which
        # grows when a joint is pushed or blocked (raw: on the real YAM friction alone holds it within ~1 deg)
        self.q_cmd = np.asarray(d.get("q_cmd", d["q_ref"]))
        # the arms' contact flag comes from their force estimate: on the real YAM it is not trusted (it said
        # "contact" with the hand 30 cm in the air), and summary() leaves it out (r.description["arm_force_trusted"])
        self.arm_force_trusted = bool(d.get("arm_force_trusted", False))
        self.ee = {k: EE(v, self.arm_force_trusted) for k, v in d["ee"].items()}
        self.grippers = d["grippers"]
        self.fault = d.get("fault")
        self.busy = d.get("busy")

    def _pick(self, arr, names):
        if names is None:
            return arr.copy()
        return np.array([arr[self._i[n]] for n in names])

    def q(self, names=None):
        """Joint positions, in the order of `names` (e.g. a library's own joint-name list) or of self.joints."""
        return self._pick(self._q, names)

    def dq(self, names=None):
        return self._pick(self._dq, names)

    def effort(self, names=None):
        """Total motor effort (gravity compensation included)."""
        return self._pick(self._effort, names)

    def effort_ext(self, names=None):
        """External effort: measured minus the model (what contact and payload add)."""
        return self._pick(self._ext, names)

    def __repr__(self):
        return f"State(t={self.t:.3f}, ee={self.ee}, grippers={self.grippers})"

    def summary(self) -> str:
        """One line: each end effector's position (m), its gripper's width (mm) and whether it stalled on something
        or closed on nothing (the width an empty close ends at, r.description["gripper_empty_width_m"]), contact
        where the arms' force estimate is trusted (simulation); a fault if one is latched."""
        parts = []
        for name, e in self.ee.items():
            g = self.grippers.get(name) or {}
            s = f"{name} xyz=[{', '.join(f'{v:.3f}' for v in e.T[:3, 3])}]"
            if g.get("width") is not None:
                s += f" grip={1000 * float(g['width']):.1f}mm"
                if g.get("empty"):
                    s += " (closed on nothing)"
                elif g.get("stalled"):
                    s += " (stalled: holding something)"
            if e.contact and self.arm_force_trusted:
                s += " contact"
            parts.append(s)
        f = self.fault
        return "; ".join(parts) + (f"; FAULT: {f.get('reason') if isinstance(f, dict) else f}" if f else "")


class Frame:
    def __init__(self, d):
        self.rgb = d.get("rgb")
        self.depth = d.get("depth")
        self.K = np.asarray(d["K"])
        self.T = np.asarray(d["T"]) if d.get("T") is not None else None   # None: the camera has no calibrated pose
        self.pose_error = d.get("pose_error")
        self.t = d.get("t")
        self.roi = d.get("roi")             # [u0, v0, u1, v1] of the full image when cropped

    def points(self, mask=None, stride=1):
        """World points (N x 3) from depth, optionally only where mask is true."""
        if self.T is None:
            raise RobotError(f"this camera has no pose, so its points have no place in the world: {self.pose_error}")
        H, W = self.depth.shape
        if mask is None:
            v, u = np.mgrid[0:H:stride, 0:W:stride].reshape(2, -1)
        else:
            v, u = np.nonzero(mask)
        z = self.depth[v, u].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.02) & (z < 5.0)
        x = (u[ok] - self.K[0, 2]) * z[ok] / self.K[0, 0]
        y = (v[ok] - self.K[1, 2]) * z[ok] / self.K[1, 1]
        P = np.stack([x, y, z[ok], np.ones(ok.sum())])
        return (self.T @ P)[:3].T


def _after(after):
    """reachable's after= as JSON: {arm: joint values, a 4x4, or {"target", "tcp", "free"}} (None: none)."""
    if not after:
        return None
    lst = lambda v: None if v is None else asarray(v).tolist()
    return {a: {"target": lst(v["target"]), "tcp": lst(v.get("tcp")), "free": v.get("free")} if isinstance(v, dict)
            else lst(v) for a, v in after.items()}


def _checked(res: Result) -> Result:
    """A motion that ended because it could not go on raises: Fault when a guard or the operator stopped everything,
    Aborted otherwise. Ending as asked (done, a condition, replaced, timeout=) returns the Result."""
    if res.status == "aborted":
        if res.reason in ("fault", "operator_stop"):
            raise Fault(f"{res.op} aborted: a fault is latched (r.state().fault says why; r.recover() clears it)")
        raise Aborted(res)
    return res


class Motion:
    """Handle of a command started with wait=False."""

    def __init__(self, robot, cid):
        self.robot, self.id = robot, cid

    def result(self, timeout=None) -> Result:
        """Block until it ends (in simulation this advances time). Raises Aborted or Fault as a blocking call does."""
        return _checked(Result(self.robot._call("wait", id=self.id, timeout=timeout)))

    @property
    def done(self) -> bool:
        return self.robot._call("status", id=self.id)["finished"]

    def stop(self):
        return self.robot._call("cancel", id=self.id)


# ------------------------------------------------------------------------------------------------ the robot
_BEFORE_COMMAND = None  # robot.kernel sets it: fn(robot, spec) before every motion command is sent
_LATEST = []          # a weak reference to the last Robot connected, for robot.viz.show's copy to the dashboard
                      # (weak: a Robot nobody holds must still close its sockets when it is dropped)


def _latest():
    return _LATEST[0]() if _LATEST else None


_EXIT_HOOKED: list = []


def _print_exit_state():
    """As a program that moved the arms exits: `[state] ...`, the arms as it leaves them (State.summary), so the
    output of the command that ran it says where things ended without another look. ROBOT_EXIT_STATE=0: none."""
    r = _latest()
    if r is None or not getattr(r, "_moved", False) or os.environ.get("ROBOT_EXIT_STATE", "1") == "0":
        return
    try:
        print("[state] " + r.state().summary(), flush=True)
    except Exception:                                   # noqa: BLE001 -- the program has ended; nothing to add
        pass


class Robot:
    def __init__(self, addr: str, name: str | None = None):
        self.addr, self.name = addr, name
        _LATEST[:] = [weakref.ref(self)]
        self._moved = False                  # a command was sent: the program prints the arms' state as it exits
        call = globals().get("_KERNEL_CALL")   # inside a robot.kernel call: closed as the call ends
        if call is not None and getattr(call, "opened", None) is not None:
            call.opened.append(self)
        if not _EXIT_HOOKED:
            _EXIT_HOOKED.append(1)
            atexit.register(_print_exit_state)
        self.sock = rpc.connect(addr)
        self._cam = None                 # the connection to camerad, when the driver's cameras are served there
        self._sent_files = {}            # path -> (mtime_ns, size) of the text last sent: an edited file goes again
        hello = self._call("hello", program={"argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid()},
                           **({"client": name} if name else {}))
        self.description = hello["description"]
        self.lockstep = hello["lockstep"]
        self.end_effectors = list(self.description["end_effectors"])

    def _call(self, op, **kw):
        req = {"op": op, **kw}
        site = _call_site()
        if site:
            req["src"] = site
            texts = {}
            for path, _, _ in site:
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                stamp = (st.st_mtime_ns, st.st_size)
                if self._sent_files.get(path) != stamp and st.st_size < 1_000_000:
                    self._sent_files[path] = stamp
                    try:
                        texts[path] = open(path, encoding="utf-8", errors="replace").read()
                    except OSError:
                        pass
            if texts:
                req["src_text"] = texts
        rpc.send(self.sock, req)
        rep = rpc.recv(self.sock)
        if rep is None:
            raise RobotError("the driver closed the connection")
        if not rep.get("ok"):
            err = rep.get("error", {})
            raise _ERRORS.get(err.get("type"), RobotError)(err.get("message", "unknown error"))
        return rep.get("result")

    def _command(self, spec):
        hook = globals().get("_BEFORE_COMMAND")    # robot.kernel: a call that moves stops a running action, and a
        if hook is not None:                        # stopped action's code goes no further
            hook(self, spec)
        self._moved = True
        res = self._call("command", spec=spec)
        if spec.get("wait", True):
            return _checked(Result(res))
        return Motion(self, res["id"])

    # ---- reading
    def state(self) -> State:
        return State(self._call("state"))

    def T(self, ee: str) -> np.ndarray:
        """Measured TCP pose of an end effector (4x4, world)."""
        return self.state().ee[ee].T

    def ik(self, target, ee, tcp=None, seed=None, free=None, after=None) -> dict:
        """Joint angles that would put an end effector's TCP (or tcp=, a 4x4 in the TCP frame) at target (4x4, world),
        searched from seed (that arm's joint values; default: where after= puts it, else where it is commanded now)
        within the joint limits. Nothing moves. free= as for move ('yaw', 'yaw180', 'rotation'). after= as for
        reachable. Returns {"joints": names, "q": array, "ok": bool, "error_m": position miss, "T": the pose reached
        (4x4), "margin": {"limit_rad", "limit_joint", "singular_m"}}: check reach, or pick a configuration and go
        there with r.move(q, joints=...)."""
        out = self._call("ik", ee=ee, target=asarray(target).tolist(),
                         tcp=None if tcp is None else asarray(tcp).tolist(),
                         seed=None if seed is None else asarray(seed).tolist(), free=free, after=_after(after))
        out["q"], out["T"] = np.asarray(out["q"]), np.asarray(out["T"])
        return out

    def reachable(self, targets, ee, tcp=None, free=None, line=False, after=None) -> list:
        """Many candidate targets (approach angles, tilts, heights, grasps) in one call: for each, whether
        r.move(target, ee=ee, tcp=, free=, line=) would be accepted from where the arm is commanded now (the other arm
        where it is commanded, and where its running move ends). r.ik's dict plus why: None, or the refusal that move
        would get (the fence or the gripper floor, IK and its miss, the arms' separation, line=True: the straight
        line's unreachable stretch). ok only when why is None; filter on ok and rank by margin. Nothing moves.
        after={arm: where the queue will have put it}: check as the move will run, after what runs and waits ahead of
        it: joint values, a 4x4 TCP pose, or {"target": T, "tcp": .., "free": ..} as that move was given
        (kernel.planned_end() collects them). IK starts from there, and the other arm, if named, is checked for
        separation there instead of where it is now. What these checks solve robotd remembers, and the move tries
        it when its own solve (from where the arm is when it starts) misses."""
        out = self._call("reachable", ee=ee, targets=[asarray(T).tolist() for T in targets],
                         tcp=None if tcp is None else asarray(tcp).tolist(), free=free, line=bool(line),
                         after=_after(after))
        for o in out:
            o["q"], o["T"] = np.asarray(o["q"]), np.asarray(o["T"])
        return out

    def cameras(self, names=None, depth=True, roi=None) -> dict:
        """name -> Frame(rgb HxWx3 uint8, depth HxW m, K 3x3, T 4x4 world<-optical (+z forward, +y down), t: the
        time the image shows). roi=[u0, v0, u1, v1] (or {name: box}): only that pixel box comes back, with K shifted
        to match (Frame.points works as on the full image; Frame.roi says which box), which is quicker to send and
        to search when you track something small. A camera without a calibrated pose has T None and pose_error
        saying why (its points() refuse)."""
        req = {"names": names, "depth": depth}
        if roi is not None:
            req["roi"] = {k: asarray(v).tolist() for k, v in roi.items()} if isinstance(roi, dict) else \
                asarray(roi).tolist()
        cam_addr = os.environ.get("ROBOT_CAMERA_ADDR") or self.description.get("camera_addr")
        frames = self._camera_call(cam_addr, req) if cam_addr else self._call("cameras", **req)
        return {k: Frame(v) for k, v in frames.items()}

    def _camera_call(self, addr: str, req: dict) -> dict:
        """The real cell's cameras are served by camerad (robotd/camerad.py), a process of their own, so that frames
        never pass through the driver's process; its address comes with the description ($ROBOT_CAMERA_ADDR wins,
        e.g. through a tunnel on another port)."""
        try:
            if self._cam is None:
                self._cam = rpc.connect(addr)
                rpc.send(self._cam, {"op": "hello", "client": self.name})
                rpc.recv(self._cam)
            rpc.send(self._cam, {"op": "cameras", **req})
            rep = rpc.recv(self._cam)
        except OSError as e:
            self._cam = None
            raise RobotError(f"cannot reach the cameras (camerad at {addr}): {e}") from None
        if rep is None:
            self._cam = None
            raise RobotError(f"camerad at {addr} closed the connection")
        if not rep.get("ok"):
            err = rep.get("error", {})
            raise _ERRORS.get(err.get("type"), RobotError)(err.get("message", "unknown error"))
        return rep["result"]

    def _views(self, op: str, **kw):
        """The dashboard's board of shown images (robot.viz.show): at camerad when it serves the cameras, else robotd."""
        cam_addr = os.environ.get("ROBOT_CAMERA_ADDR") or self.description.get("camera_addr")
        if cam_addr:
            return self._camera_call(cam_addr, {"op": op, "client": self.name, **kw})
        return self._call(op, **kw)

    # ---- motion: move, move_at, gripper
    def move(self, target, times=None, joints=None, ee=None, frame="world", line=False, relative=False, dt=None,
             velocities=None, speed=None, accel=None, duration=None, free=None, dry_run=False, **common):
        """Go to a target, or along a path of them.
        One target, no times     go there and stop: smooth (jerk-limited) and as quick as the limits allow (speed=,
                                 accel=, duration= slow it). ee=: IK near the current joints then a joint move, or
                                 line=True for a straight TCP line.
        A list of targets        pass through them, target i at times[i] s from the start (or (i+1)*dt); with neither,
                                 each leg is timed from the speed limits (speed=: TCP m/s, or the fastest joint's
                                 rad/s; duration=: the whole path). velocities=: the speed at each target (the
                                 controlled point's m/s, or rad/s per joint), e.g. to hit something at a time.
        joints=[names]: targets are joint vectors (a planner's N x n path: r.move(qs, joints=names, dt=...)); ee="left":
        4x4 poses of the TCP (or of tcp=); ee=["left","right"] with {arm: target(s)}: one timeline for both arms.
        frame="tool": targets relative to the TCP at the start; relative=True (world): offsets from the start, rotated
        about it. Takes every common argument (until=, on=, then=, gains=, tcp=, start_at=, wait=, timeout=,
        compliance=). Raises InvalidCommand if a target is unreachable, Aborted if the arm cannot follow on the way.

        Every move continues from however the arm moves now (position, velocity and acceleration), so a new move
        replaces the running one smoothly. To servo on something that moves, re-send short timed paths in a loop:
        r.move(T, ee=arm, times=[tau], velocities=[v], wait=False), with tau about 2-3x the loop's period (0.08 s for a
        30 Hz loop): a shorter tau stops the arm between updates, a longer one reacts late.

        free= leaves axes of a pose target to the driver: 'yaw' (any turn about the tool's z axis, e.g. a top-down
        pinch), 'yaw180' (the turn as given or half a turn more: the same grasp for a parallel gripper), 'rotation'
        (position only). It picks the solution quickest to reach that keeps away from joint limits; along a path
        (free= one value or one per target) the turns that make the whole path cheapest, without wrist flips. The
        Result's goal holds the full pose(s) chosen.
        dry_run=True: nothing moves; returns a Plan: ok, or where it would fail (a straight line or a pose path is
        walked every 1 cm), the joint goal and the tightest margin on the way."""
        spec = Move(target, times, joints, ee, frame, line, relative, dt, velocities, speed, accel, duration, free,
                    **common)
        if dry_run:
            spec["dry_run"] = True
            return Plan(self._call("command", spec=spec))
        return self._command(spec)

    def move_at(self, velocity, joints=None, ee=None, frame="world", accel=None, max_travel=None, duration=None,
                **common):
        """Constant velocity until a condition, max_travel (m, default 0.30) or duration. joints= -> rad/s per joint;
        ee= -> twist {"x","y","z","rx","ry","rz"} (m/s, rad/s) in frame ('world' or 'tool')."""
        return self._command(MoveAt(velocity, joints, ee, frame, accel, max_travel, duration, **common))

    def gripper(self, name, width, speed=None, force=None, **common):
        """Open/close to `width` (m between the jaws). force= caps each pad's squeeze (N). reason 'stalled' = it
        closed on something (Result.width = what it holds); 'reached' = it got to width."""
        return self._command(Gripper(name, width, speed, force, **common))

    # ---- time and control
    def step(self, ticks: int = 1):
        """Simulation: advance ticks x 40 ms (running commands continue). Real robot: sleep that long."""
        return State(self._call("step", ticks=int(ticks)))

    def step_until(self, t: float):
        """Until the robot clock reads t (s): simulation advances to it (in 4 ms ticks), a real robot sleeps; returns
        at once if t has passed. The fixed-rate loop: t_next += period; r.step_until(t_next)."""
        return State(self._call("step_until", t=float(t)))

    def stop(self):
        """Stop every command; the arms hold where they are."""
        return self._call("stop")

    def zero_force(self, ee: str):
        """Take the current force estimate as zero (do it just before contact, after a grasp, ...)."""
        return self._call("zero_force", ee=ee)

    def recover(self):
        """Clear a latched Fault (the arms already hold)."""
        return self._call("recover")

    def signal(self, name: str, value: float = 1.0):
        """Set a number that when.signal(name) reads in the driver's fast tick (e.g. a scale reading)."""
        return self._call("signal", name=name, value=float(value))

    def finish(self, summary: str):
        """End the episode with a short summary of what you did and believe the outcome is."""
        return self._call("finish", summary=summary)

    def close(self):
        """Disconnect. The driver stops any motion this connection started (as when the program exits)."""
        for s in (self.sock, self._cam):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass


def connect(addr: str | None = None, name: str | None = None) -> Robot:
    """Connect to the robot's driver: addr or $ROBOT_ADDR ('unix:/path' or 'tcp:host:port'). name labels this
    connection in the driver's event log."""
    addr = addr or os.environ.get("ROBOT_ADDR", "unix:/tmp/robot.sock")
    return Robot(addr, name)
