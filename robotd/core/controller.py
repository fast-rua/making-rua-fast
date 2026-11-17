"""The control core: one fast tick every 4 ms.

Each tick:
1. measure: joints, TCP poses and velocities, and the wrench estimated from joint efforts;
2. step every running command's reference into joint references (Cartesian ones through differential IK);
3. apply each arm's compliance (admittance about the reference, implicit force targets);
4. evaluate stop conditions and triggers, and fire then= chains;
5. enforce guards (and the workspace fence, when one is set);
6. write PD targets and gains to the backend, and advance it.

Commands own joint groups (an arm, a gripper). A new command on an owned group replaces the old one, continuing from
the current reference. A finished command leaves its last commanded position as the reference (a press keeps
pressing until the next command), with zero reference velocity. A command runs with its own gains (or the defaults)
and a grip with its own force cap, from when it takes over until the next command on those joints.

The operator's stop (the dashboard) latches like a Fault that only the operator releases: "freeze" holds the arms
stiff where they are; "soft" drops the arm stiffness to zero and keeps damping on top of the gravity compensation, so
the arms float and yield to a push, and the grippers go limp (no stiffness at all: a gripper jammed at a stop that
holding "where it is" would keep driving, since the real backend never commands a stop, lets go).

The operator's park latches too: every program command is stopped, the arms go to the park pose, and new commands
are refused (Fault) until the operator releases it. Without the latch a running program's next command (a servo loop
sends one every few tens of ms) replaces the park at once, and the park loses its purpose.
"""
from __future__ import annotations

import collections
import itertools
import math
import threading
import time
import traceback

import numpy as np

from robotd.core import references as refs
from robotd.core.conditions import TickContext, compile_cond, compile_named
from robotd.core.estimator import MomentumObserver
from robotd.core.kinematics import FREE, Kinematics, pose_err
from robotd.core.floor import GripperFloor
from robotd.core.separation import Separation

DT = 0.004
# Rates come from the real period of each tick, not DT (as libfranka's control callback and ros2_control's
# update(time, period) get it): references are sampled at the measured time, so a late tick moves them further, and
# dividing that step by DT read as a burst of speed. On the real YAM (ticks 4-8 ms) that put 4-7 mrad tick-to-tick
# jumps into the wrist commands through the acceleration feed-forward. Bounded, so a stall cannot divide by ~0 or
# read a long gap as a crawl.
PERIOD_MIN, PERIOD_MAX = 0.25 * DT, 25 * DT
CONTACT_ON_N, CONTACT_OFF_N = 2.5, 1.5
EMPTY_GRIP_TOL_M = 0.0005        # a close that ends within this of desc["gripper_empty_width_m"], asked to go below
                                 # it, closed on nothing
JOINT_CLIP_RAD = 0.15            # guard: the commanded joint stays within this of the measured one (a blocked
                                 # arm then pushes with at most kp x 0.15, ~12 Nm on the YAM's big joints)
EFFORT_STOP_FRACTION = 0.8       # guard: an arm joint above this share of its motor limit for EFFORT_STOP_S
EFFORT_STOP_S = 0.04             #   latches a Fault (the arm is pushing against something)
GRIP_OVERLOAD = 1.5              # guard: a gripper above this x its force limit for GRIP_OVERLOAD_S is jammed or
GRIP_OVERLOAD_S = 0.3            #   blocked: it is relaxed (no stiffness) until its next command
DEFAULT_MAX_TRAVEL = 0.30        # move_at without max_travel stops here
SETTLE_POS_M = 0.004             # a move is done when the TCP is this close to its reference ...
SETTLE_SPEED = 0.05              # ... and no arm joint moves faster than this (rad/s), or 0.5 s after the reference
ACC_FILTER = 0.2                 # low-pass gain of the measured joint acceleration (per 4 ms tick)
FF_FILTER = 0.3                  # low-pass gain of the reference acceleration for the inertia feed-forward
FF_MAX_ACC = 40.0                # rad/s^2: bound of that acceleration (a stop is a step in the reference velocity)
BLEND_S = 0.25                   # a timed path that replaces a moving one blends from its velocity and acceleration
                                 # within this long (or the path's own duration if shorter)
YAW_STEP = np.pi / 6              # free='yaw': the candidate turns about the tool axis (every 30 deg)
DP_MAX_POINTS = 40               # a free path longer than this takes each point's nearest solution in turn
IK_MEMORY = 64                   # per arm: the IK solutions kept (reachable, ik, a move's goals), newest last ...
IK_MEMORY_POS_M, IK_MEMORY_ROT_RAD = 0.01, 0.1   # ... a later target this close to one (same tcp, free) tries
IK_MEMORY_SEEDS = 3              #   at most this many of their joints as seeds (nearest first) ...
IK_MEMORY_BETTER_S = 0.5         # ... and takes one over a solve that reached when it is this much quicker to go to
IK_SAME_M, IK_SAME_RAD = 1e-5, 1e-4   # after=: a target this close to a remembered one is that solution
DRY_RUN_STEP_M, DRY_RUN_STEP_RAD = 0.01, 0.05   # a dry run checks a Cartesian path every 1 cm / 3 deg
START_TOL_RAD = 0.01             # a joint path's first point this close to the current joints is its start
AUTO_RAMP = 1.875                # a path given without times: its first and last legs, from and to rest, take this
                                 # much longer than at speed (the peak over the mean speed of a smooth start or stop)
FENCE_MARGIN = 0.02              # fence: a measured TCP this far outside its box latches a Fault (the soft wrist lags
                                 # and overshoots its reference by up to ~1 cm)
FENCE_REF_TOL = 0.002            # fence: a reference TCP this far outside its box ends its commands (targets are
                                 # checked exactly; this absorbs the differential IK's trailing)
FLOOR_REF_TOL = 0.001            # fence: a reference gripper this far below gripper_floor ends its commands (tighter
                                 # than the box's: past the floor is the table)
FENCE_SPEED_FAULT = 1.5          # fence: a measured planar TCP speed above this x the limit for FENCE_SPEED_S latches
FENCE_SPEED_S = 0.02             #   a Fault (the reference itself never exceeds the limit)
# the settled-error correction (desc["settle_bias"], _settle_bias): rate (1/s) the bias integrates q_ref - q at,
# max_rad its cap per joint, speed (rad/s) above which an arm's bias is held, not integrated
SETTLE_BIAS = {"rate": 3.0, "max_rad": float(np.radians(1.5)), "speed": 0.1}
BIAS_ERR_RAD = 0.02              # settle bias: at most this error per joint is integrated (a transient is not an offset)
BIAS_WAIT_S = 0.3                # settle bias: it integrates only once the arm's reference has been still this long
                                 # (the end of a move is a transient, not an offset)
BIAS_REF_STILL = 1e-4            # rad/s: a reference slower than this is still
BIAS_MOVE_S = 0.3                # settle bias: while the reference moves it decays with this time constant. What it
                                 # holds is mostly friction (a model-fitted gravity feed-forward leaves little else),
                                 # and friction pushes the other way once the joint moves again: carried into the next
                                 # move, a 1.5 deg trim was a 1.5 deg error there (the hand 6-21 mm off as moves ended)
# settle bias: each joint stops changing (frozen until the reference moves) once it has crossed its target, or once it
# rests within this deadband (rad, by its place in the arm; ~1.5 mm at the hand on joints 1-3). Friction holds a joint
# anywhere within about its Coulomb torque / kp of a target (i2rt's YAM: 0.3 Nm / 80 ~ 0.2 deg on joints 1-3,
# 0.06 / 10 ~ 0.35 deg on the wrist), and an integrator that keeps working inside that band winds up until the joint
# breaks free and jumps past, again and again ("hunting"): freezing on the crossing leaves at most one such jump
BIAS_DEADBAND = tuple(float(np.radians(d)) for d in (0.25, 0.25, 0.25, 0.4, 0.4, 0.4))
BIAS_REST_S = 0.1                # settle bias: a joint whose position lags its low-pass (this time constant) by under
BIAS_REST_RAD = 0.0003           # this rad is at rest (~3 mrad/s: dq itself is too noisy on the real YAM)
BIAS_BLEED_S = 0.2               # settle bias: switched off, it decays with this time constant
BIAS_GAP_S = 0.1                 # settle bias: a gap this long between measurements (a reconnect) resets it
BIAS_PROBE_RAD = 0.012           # settle bias: each time it has changed this much (norm over an arm's joints) ...
BIAS_FOLLOW = 0.3                # ... the joints must have followed by this share of it, twice in a row, or the arm is
                                 # blocked (pressing on something): it bleeds off until the arm's next command
BIAS_LAG_S = 0.1                 # settle bias: the joints follow it through the PD servo with about this lag
CONTACT_METRICS = ("force", "torque", "effort", "deflection", "contact")   # a condition on these seeks contact


def _gap(d: float) -> str:
    """A separation clearance in words: 'puts the arms 1.2 cm apart', or 'puts the arms' shapes 3.4 cm into each other'."""
    return f"puts the arms {d * 100:.1f} cm apart" if d >= 0 else f"puts the arms' shapes {-d * 100:.1f} cm into each other"


class InvalidCommand(ValueError):
    pass


class Busy(RuntimeError):
    pass


class Fault(RuntimeError):
    pass


def _on_clock(times, late, points, velocities=None):
    """A timed path that starts `late` s after its start_at keeps its waypoints on the robot clock: their times count
    from now, minus the delay, and those already due are dropped (the path runs from the current reference to the
    next one still ahead; if none is, to the last within two ticks)."""
    if late <= 0:
        return times, points, velocities
    t = np.asarray(times, float) - late
    keep = t > 2 * DT
    if not keep.any():
        keep[-1], t[-1] = True, 2 * DT
    pick = lambda seq: None if seq is None else [x for x, k in zip(seq, keep) if k]
    return t[keep].tolist(), pick(list(points)), pick(None if velocities is None else list(velocities))


def _auto_times(legs, duration=None):
    """Waypoint times for a path given without them, from each leg's duration at speed: the first and last legs
    stretched by AUTO_RAMP, none under two ticks; duration= scales the whole path to that long."""
    legs = np.array(legs, float)
    legs[0] *= AUTO_RAMP
    if len(legs) > 1:
        legs[-1] *= AUTO_RAMP
    t = np.cumsum(np.maximum(legs, 2 * DT))
    if duration:
        t *= float(duration) / t[-1]
    return t.tolist()


def _rot_z(th):
    T = np.eye(4)
    c, s_ = np.cos(th), np.sin(th)
    T[:2, :2] = [[c, -s_], [s_, c]]
    return T


def _relative_world(base, P):
    """A pose given as an offset from base in the world frame: translated by it, rotated about base's point."""
    T = np.eye(4)
    T[:3, :3] = P[:3, :3] @ base[:3, :3]
    T[:3, 3] = base[:3, 3] + P[:3, 3]
    return T


def _T(x):
    a = np.asarray(x, float)
    if a.shape != (4, 4):
        raise InvalidCommand(f"a pose must be a 4x4 matrix, got shape {a.shape}")
    return a


def parse_fence(spec) -> dict | None:
    """A workspace fence, normalised: {arm: {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}, ..., "max_speed_xy": m/s,
    "margin": m, "gripper_floor": z}, from a dict, a JSON string or a JSON file's path. Each arm named gets a box for
    its TCP (an axis left out is unbounded; None for one end leaves that end open); max_speed_xy limits every fenced
    TCP's speed in the table plane (default 1.0); margin is how far a measured TCP may stray outside before a Fault
    (FENCE_MARGIN); gripper_floor (optional) is a world z that each fenced arm's whole gripper (fingers, body, wrist
    camera: robotd/core/floor.py) stays above, which a tilted gripper meets before its TCP meets the box's floor.
    Open ends stay None (the result is JSON, as r.description["fence"] carries it)."""
    import json
    import os
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        spec = json.load(open(spec)) if os.path.exists(spec) else json.loads(spec)
    out = {"max_speed_xy": float(spec.get("max_speed_xy", 1.0)), "margin": float(spec.get("margin", FENCE_MARGIN))}
    if spec.get("gripper_floor") is not None:          # world z the gripper's lowest point stays above (floor.py)
        out["gripper_floor"] = float(spec["gripper_floor"])
    for arm, box in spec.items():
        if arm in ("max_speed_xy", "margin", "gripper_floor"):
            continue
        b = {}
        for ax in "xyz":
            lo, hi = box.get(ax, [None, None])
            lo, hi = (None if lo is None else float(lo)), (None if hi is None else float(hi))
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"fence {arm} {ax}: lower {lo} is not below upper {hi}")
            b[ax] = [lo, hi]
        bad = set(box) - set("xyz")
        if bad:
            raise ValueError(f"fence {arm}: unknown keys {sorted(bad)} (x, y, z)")
        out[arm] = b
    return out


class Command:
    _ids = itertools.count(1)

    def __init__(self, spec: dict, groups: list, arms: list):
        self.id = next(self._ids)
        self.spec = spec
        self.kind = spec["op"]
        self.groups, self.arms = groups, arms
        self.status = "pending"
        self.result = None
        self.t0 = None
        self.ref = None                 # joint reference: (joints, generator)
        self.cart = {}                  # arm -> pose generator (Cartesian reference)
        self.tcp = {}                   # arm -> 4x4 controlled point in the TCP-site frame
        self.T0 = {}
        self.until, self.on, self.on_fired = {}, {}, set()
        self.next = None                # the command started by then=
        self.spawned = []               # the commands its on= started (they go with it when its client leaves)
        self.park = False               # the operator's park (set by the driver, never from a spec): see _guards
        self.track_bad = 0
        self.compliance = {}            # arm -> spec dict
        self.reason_override = None
        self.q_start = None             # a check from a planned start (reachable's after=): the joints, else the
        self.planned = ()               #   reference; and the arms whose pose after= gave

    @property
    def finished(self):
        return self.status not in ("pending", "running", "building")      # building: a placeholder (see _Build)

    def chain_finished(self):
        c = self
        while c is not None:
            if not c.finished:
                return False
            c = c.next
        return True

    def last(self):
        c = self
        while c.next is not None:
            c = c.next
        return c


class _Build:
    """then=/on= commands being built off the tick (real time): the placeholder that holds their place in the chain
    and their joint groups meanwhile, where they chain from, the specs, and what the worker made of them."""

    def __init__(self, placeholder: Command, parent: Command | None, cmd: Command, name: str, specs: list):
        self.placeholder, self.parent, self.cmd, self.name, self.specs = placeholder, parent, cmd, name, specs
        self.cmds, self.error, self.ready = None, None, False


class Controller:
    def __init__(self, backend, desc):
        self.b = backend
        self.desc = desc
        self.kin = Kinematics(backend.kinematics_model(), desc["arm_joints"], desc["tcp_sites"], backend.joints)
        # IK for commands and queries gets its own MjData: a real-time driver solves it outside the control lock,
        # while the tick uses self.kin
        self.kin_plan = Kinematics(backend.kinematics_model(), desc["arm_joints"], desc["tcp_sites"], backend.joints)
        self.plan_lock = threading.Lock()                       # (two clients may plan at once)
        self.ik_memory = {a: collections.deque(maxlen=IK_MEMORY) for a in desc["arm_joints"]}   # under plan_lock
        self.joints = list(backend.joints)
        self.jidx = {n: i for i, n in enumerate(self.joints)}
        self.arms = list(desc["arm_joints"])
        self.finger = desc["finger_joint"]                     # arm -> driven finger joint
        self.groups = {a: desc["arm_joints"][a] for a in self.arms}
        self.groups.update({f"{a}_gripper": [self.finger[a]] for a in self.arms})
        self.lo = np.array([self.kin.limits[n][0] for n in self.joints])
        self.hi = np.array([self.kin.limits[n][1] for n in self.joints])
        self.kp = np.array([desc["default_kp"][n] for n in self.joints], float)
        self.kd = np.array([desc["default_kd"][n] for n in self.joints], float)
        self.effort_limit = np.array([desc["effort_limit"][n] for n in self.joints], float)
        self.vmax = np.array([desc["max_velocity"][n] for n in self.joints], float)
        self.vcap = np.array([desc["velocity_limit"][n] for n in self.joints], float)
        self.soft_kd = np.array([desc.get("soft_kd", {}).get(n, 0.2 * desc["default_kd"][n]) for n in self.joints])
        self.arm_idx = np.array([self.jidx[n] for a in self.arms for n in self.groups[a]])
        self.listener = None                                   # f(kind, text, **info): the driver's event log
        # the arms' external torque: a momentum observer, so the estimate holds while the arm moves
        self.observer = MomentumObserver(self.kin.m, desc["arm_joints"], self.joints, getattr(backend, "dynamics", None))
        self.t_obs = None
        # joints with less friction are trusted more when joint torques become a force at the TCP
        dyn = getattr(backend, "dynamics", None) or {}
        sigma = np.asarray(dyn.get("coulomb", [0.0] * 6), float) + 0.02
        self.tau_weight = 1.0 / sigma ** 2
        self.grip_over = {a: 0 for a in self.arms}
        self.grip_relaxed = {a: False for a in self.arms}
        # what the agent is told: whether the arms' force estimate (contact) is worth reading, and the width a close
        # on nothing ends at (the backend's, unless the description says)
        desc.setdefault("arm_force_trusted", bool(getattr(backend, "arm_force_trusted", False)))
        desc.setdefault("gripper_empty_width_m", float(getattr(backend, "gripper_empty_width_m", 0.0)))
        # the settled-error correction (_settle_bias): None is off (and in monitor mode, the motors off)
        sb = desc.get("settle_bias")
        self.settle = (None if not sb or not float(sb.get("rate", 0.0)) or getattr(backend, "passive", False)
                       else {**SETTLE_BIAS, **sb})
        self.arm_cols = {a: np.array([self.jidx[n] for n in self.groups[a]]) for a in self.arms}
        self.J_meas = {}                                       # arm -> the TCP Jacobian at the measured joints
        self.bias_off = set()                                  # arms clipped or saturated at the last tick
        self.bias_probe = {}                                   # arm -> (bias, q - q_ref, probes failed in a row)
        self.bias_lag = np.zeros(len(self.joints))             # the bias as the joints follow it (BIAS_LAG_S)
        self.bias_blocked = set()                              # arms whose bias is off until their next command
        self.bias_still = {}                                   # arm -> when its reference last moved
        self.bias_sign = np.zeros(len(self.joints))            # each joint's error sign as this settle began (0: unset)
        self.bias_frozen = np.zeros(len(self.joints), bool)    # joints whose bias is done for this settle
        self.q_rest = None                                     # the measured joints, low-passed (BIAS_REST_S)
                                                               # (the joints did not follow it, or a press ended)
        m = self.b.read()
        self.t = m.t
        self.period = DT                                       # this tick's real duration (see PERIOD_MIN)
        self.ddq = np.zeros_like(m.q)
        self.q_ref = m.q.copy()
        self.dq_ref = np.zeros_like(m.q)
        self.dq_ref_prev = np.zeros_like(m.q)
        self.ddq_ref = np.zeros_like(m.q)
        self.ddq_now = np.zeros_like(m.q)                      # the reference's acceleration at the last tick
        self.feedforward = True
        self.q_cmd = m.q.copy()
        self.bias = np.zeros_like(m.q)                         # the settle bias added to each arm joint's command
        self.lo_cmd, self.hi_cmd = np.minimum(self.lo, m.q), np.maximum(self.hi, m.q)    # see _control
        self.commands: list[Command] = []
        # then=/on= commands fire inside the tick. In lockstep the world waits while they are built; in real time a
        # worker builds them (IK, free= candidates, path checks) and a later tick starts them (_start_built)
        self.builds: list[_Build] = []
        self.parking = False                    # the driver's exit park: a guard reports instead of latching (as
                                                # it does while an operator's park command runs)
        self.wrench_raw = {a: np.zeros(6) for a in self.arms}
        self.wrench_bias = {a: np.zeros(6) for a in self.arms}
        self.contact = {a: False for a in self.arms}
        self.contact_n = {a: 0 for a in self.arms}
        self.gripper_state = {a: {"width": 0.0, "effort": 0.0, "stalled": 0.0, "empty": False} for a in self.arms}
        self.signals = {}
        self.fault = None
        self.parked = None                      # the operator's park latch: {"reason", "t"} until the operator releases
        self.comp_offset = {a: np.zeros(6) for a in self.arms}
        self.comp_trim = {}
        # arm -> (pose, tcp) the running Cartesian reference asked for at the last tick: the damped differential IK
        # trails it a little, so a Cartesian command that replaces it starts there, not at the joints' pose
        self.T_des = {a: None for a in self.arms}
        self.sat_ticks = 0
        self.recorder = None
        self.ticks = 0
        self._measure(m)
        self.T_nom = dict(self.T)
        # the workspace fence (desc["fence"], parse_fence): each fenced arm's TCP box and the planar speed limit. An
        # arm's runtime checks start once its TCP is inside its box (a robot starts at rest, outside it)
        f = parse_fence(desc.get("fence"))
        end = lambda v, inf: inf if v is None else float(v)
        self.fence = {a: (np.array([end(f[a][ax][0], -np.inf) for ax in "xyz"]),
                          np.array([end(f[a][ax][1], np.inf) for ax in "xyz"])) for a in self.arms if f and a in f}
        if f:
            unknown = set(f) - set(self.arms) - {"max_speed_xy", "margin", "gripper_floor"}
            if unknown:
                raise ValueError(f"fence names unknown arms {sorted(unknown)}; arms: {self.arms}")
        self.fence_v = float(f["max_speed_xy"]) if f else np.inf
        self.fence_margin = float(f["margin"]) if f else FENCE_MARGIN
        self.fence_armed = {a: False for a in self.fence}
        self.fence_p = {}                       # arm -> its reference TCP position at the last tick
        # the fence's gripper_floor (robotd/core/floor.py): the whole gripper, not only the TCP, stays above it
        self.floor_z = f.get("gripper_floor") if f else None
        self.floor = (GripperFloor(self.kin.m, {a: desc["arm_joints"][a] for a in self.fence}, desc["tcp_sites"])
                      if self.floor_z is not None and self.fence else None)
        self.fence_low = {}                     # arm -> its reference gripper's lowest point at the last tick
        # the arms kept apart (desc["separation"], robotd/core/separation.py): off unless the description sets it
        sep = desc.get("separation")
        self.sep = (Separation(self.kin.m, sep, sep["min_m"], sep.get("fault_m", 0.0))
                    if sep and len(self.arms) == 2 else None)
        self.sep_prev = None                    # the references' clearance at the last tick
        self.sep_armed = True                   # the measured check (off after its Fault, until clear again)
        if self.sep is not None:                # (the first _measure ran before this existed)
            self.sep_meas = self.sep.at(self.kin, self._arm_q(self.m.q))
        self.fence_fast = {a: 0 for a in self.fence}

    # ================================================================== measurement
    def _arm(self, vec, arm):
        return np.array([vec[self.jidx[n]] for n in self.groups[arm]])

    def _set_arm(self, vec, arm, values):
        for n, v in zip(self.groups[arm], values):
            vec[self.jidx[n]] = v

    def _measure(self, m):
        prev = getattr(self, "m", None)
        self.m = m
        has_ff = m.tau_model is not None and np.all(np.isfinite(m.tau_model))
        # measured joint acceleration (filtered): the arm's own inertia must not read as an external force
        if prev is not None and m.t > prev.t:
            raw = (m.dq - prev.dq) / (m.t - prev.t)
            self.ddq = self.ddq + ACC_FILTER * (raw - self.ddq)
        # the arm joints: the momentum observer (a static estimate still reads Coriolis and friction torques, and
        # differentiated velocity is noisy)
        dt = DT if self.t_obs is None else float(np.clip(m.t - self.t_obs, 1e-3, 0.05))
        self.t_obs = m.t
        obs = self.observer.update(m.q, m.dq, m.tau, dt, m.tau_model if has_ff else None)
        # the fingers: the static estimate, from the observer's model at this state (M, and C qd + g)
        vadr = self.observer.vadr
        acc = np.zeros(len(self.observer.bias))
        acc[vadr] = self.ddq
        Mddq = (self.observer.M @ acc)[vadr]
        if has_ff:
            # the robot reports the feed-forward it applies (gravity): external = feed-forward + inertia - measured
            ext = np.asarray(m.tau_model, float) + Mddq - m.tau
        else:
            ext = self.observer.bias[vadr] + Mddq - m.tau
        self.tau_ext = {n: float(v) for n, v in zip(self.joints, ext)}
        for a, r in obs.items():
            for n, v in zip(self.groups[a], r):
                self.tau_ext[n] = float(v)
        self.T, self.v = {}, {}
        for a in self.arms:
            qa = self._arm(m.q, a)
            self.T[a], J = self.kin.fk_jac(a, qa)
            self.J_meas[a] = J
            self.v[a] = J[:3] @ self._arm(m.dq, a)
            tau = np.array([self.tau_ext[n] for n in self.groups[a]])
            # the force at the TCP first, as a weighted least-squares fit over all six joints (3 unknowns from 6
            # torques averages the noise down; solving force and moment together, 6 from 6, amplifies it), then the
            # moment from what the force does not explain
            Jv, Jw, w = J[:3], J[3:], self.tau_weight
            F = np.linalg.solve((Jv * w) @ Jv.T + 1e-9 * np.eye(3), (Jv * w) @ tau)
            M = np.linalg.solve((Jw * w) @ Jw.T + 1e-4 * np.eye(3), (Jw * w) @ (tau - Jv.T @ F))
            F_ext = np.concatenate([F, M])                                   # environment on the robot
            self.wrench_raw[a] = 0.85 * self.wrench_raw[a] + 0.15 * (-F_ext)  # what the tool applies
            f = np.linalg.norm(self.wrench(a)[:3])
            if self.contact[a]:
                self.contact[a] = f > CONTACT_OFF_N
            else:
                self.contact_n[a] = self.contact_n[a] + 1 if f > CONTACT_ON_N else 0
                self.contact[a] = self.contact_n[a] >= 3
            fi = self.jidx[self.finger[a]]
            self.gripper_state[a]["width"] = float(m.gripper_width[a]) if m.gripper_width else 2.0 * float(m.q[fi])
            self.gripper_state[a]["effort"] = float(m.tau[fi])
            self.gripper_state[a]["relaxed"] = self.grip_relaxed[a]
        if getattr(self, "sep", None) is not None:   # kin.d holds both arms at the measured joints (fk_jac above)
            self.sep_meas = self.sep.clearance(self.kin.d)

    def wrench(self, arm):
        return self.wrench_raw[arm] - self.wrench_bias[arm]

    def zero_force(self, arm):
        self.wrench_bias[arm] = self.wrench_raw[arm].copy()

    def snapshot(self, cmd: Command | None = None) -> dict:
        arms = cmd.arms if cmd else self.arms
        return {"t": self.t,
                "T": {a: self.T[a].tolist() for a in arms},
                "q": {n: float(self.m.q[self.jidx[n]]) for n in self.joints},
                "wrench": {a: self.wrench(a).tolist() for a in arms},
                "travel": {a: float(np.linalg.norm(self.T[a][:3, 3] - cmd.T0[a][:3, 3]))
                           for a in arms} if cmd and cmd.T0 else {}}

    def state(self) -> dict:
        m = self.m
        return {"t": self.t,
                "joints": self.joints,
                "q": m.q.tolist(), "dq": m.dq.tolist(), "effort": m.tau.tolist(),
                "effort_ext": [self.tau_ext[n] for n in self.joints],
                "q_ref": self.q_ref.tolist(), "q_cmd": self.q_cmd.tolist(), "bias": self.bias.tolist(),
                "kp": self.kp.tolist(), "kd": self.kd.tolist(),        # the gains commands set (soft overrides them)
                "ee": {a: {"T": self.T[a].tolist(), "v": self.v[a].tolist(), "wrench": self.wrench(a).tolist(),
                           "deflection": (self.T_nom[a][:3, 3] - self.T[a][:3, 3]).tolist(),
                           "contact": bool(self.contact[a])} for a in self.arms},
                "grippers": {a: dict(self.gripper_state[a]) for a in self.arms},
                "busy": {g: next((c.id for c in self.commands if g in c.groups and not c.finished), None)
                         for g in self.groups},
                "fault": self.fault, "parked": self.parked, "signals": dict(self.signals),
                "arm_force_trusted": bool(self.desc["arm_force_trusted"])}

    # ================================================================== command intake
    def submit(self, spec: dict, parent: Command | None = None, park: bool = False) -> Command:
        return self.commit(self.prepare(spec, park), parent)

    def prepare(self, spec: dict, park: bool = False) -> Command:
        """Validate and solve (IK, conditions) without touching the running commands. A realtime server runs this
        outside its lock so a slow IK never delays a control tick. park: the operator's or the exit park (the driver
        sets it, never a spec), which the fence does not stop."""
        if self.fault:
            self._refuse()
        if self.parked and not (park or self.parking):
            self._refuse_parked()
        op = spec.get("op")
        if op not in ("move_to", "follow", "move_at", "gripper", "hold"):
            raise InvalidCommand(f"unknown command {op!r}")
        groups, arms = self._groups_of(spec)
        cmd = Command(spec, groups, arms)
        cmd.park = bool(park or self.parking)
        with self.plan_lock:
            self._build(cmd)
        return cmd

    def commit(self, cmd: Command, parent: Command | None = None) -> Command:
        spec, groups = cmd.spec, cmd.groups
        if self.fault:
            self._refuse()
        if self.parked and not cmd.park:                # prepared before the operator parked: refused all the same
            self._refuse_parked()
        later = spec.get("start_at") is not None and float(spec["start_at"]) > self.t
        for other in self.commands:
            if not other.finished and set(other.groups) & set(groups):
                if not spec.get("replace", True):
                    raise Busy(f"{sorted(set(other.groups) & set(groups))} busy with command {other.id}")
                if not later or other.status == "pending":      # a running one keeps going until start_at
                    self._finish(other, "replaced", "replaced")
        self.commands = [c for c in self.commands if not c.finished]
        self.commands.append(cmd)
        self._event("command", f"{cmd.kind} {'+'.join(groups)}", id=cmd.id, op=cmd.kind, groups=list(groups))
        if parent is not None:
            parent.next = cmd
        if not later:
            self._start(cmd)
        return cmd

    def _late(self, cmd: Command) -> float:
        """How long after its start_at a command starts (it arrived after that time)."""
        s = cmd.spec
        return max(0.0, cmd.t0 - float(s["start_at"])) if s.get("start_at") is not None else 0.0

    def _take_over(self, cmd: Command):
        """A command that waited for start_at replaces what runs on its joints now, from the current reference."""
        for other in self.commands:
            if other is not cmd and other.status == "running" and set(other.groups) & set(cmd.groups):
                self._finish(other, "replaced", "replaced")
    def _refuse_parked(self):
        raise Fault(f"{self.parked['reason']}: programs are refused until the operator releases the arms (the "
                    "dashboard's RELEASE)")

    def _refuse(self):
        if self.fault.get("operator"):
            raise Fault(f"{self.fault['reason']}: the arms stay stopped until the operator releases them")
        raise Fault(f"latched fault: {self.fault['reason']}; call r.recover() first")

    def _groups_of(self, spec):
        if spec["op"] == "gripper":
            name = spec["name"]
            if name not in self.arms:
                raise InvalidCommand(f"unknown gripper {name!r}; grippers: {self.arms}")
            return [f"{name}_gripper"], []
        ee, joints = spec.get("ee"), spec.get("joints")
        if (ee is None) == (joints is None):
            raise InvalidCommand("give exactly one of joints= (names) or ee= (an end effector or a list of them)")
        if ee is not None:
            arms = [ee] if isinstance(ee, str) else list(ee)
            bad = [a for a in arms if a not in self.arms]
            if bad:
                raise InvalidCommand(f"unknown end effector {bad}; end effectors: {self.arms}")
            return list(arms), arms
        bad = [j for j in joints if j not in self.jidx]
        if bad:
            raise InvalidCommand(f"unknown joints {bad}; joints: {self.joints}")
        groups = sorted({g for g, js in self.groups.items() if set(js) & set(joints)})
        arms = [g for g in groups if g in self.arms]
        return groups, arms

    def _tcp_for(self, spec, arm):
        tcp = spec.get("tcp")
        if tcp is None:
            return None
        if isinstance(tcp, dict):
            return _T(tcp[arm]) if arm in tcp else None
        return _T(tcp)

    def _point(self, arm, tcp=None, q=None, kin=None):
        return (kin or self.kin).fk(arm, self._arm(self.q_ref if q is None else q, arm), tcp)

    def _start_pose(self, cmd: Command, arm):
        """Where a Cartesian reference starts: the pose the one it replaces asked for (same controlled point), else
        the controlled point's pose at the reference joints."""
        prev = self.T_des[arm]
        same = (prev is not None and (prev[1] is None) == (cmd.tcp[arm] is None)
                and (prev[1] is None or np.allclose(prev[1], cmd.tcp[arm])))
        if same:
            return prev[0].copy()
        return cmd.P0[arm]

    def _cart_rates(self, arm, tcp=None):
        """The controlled point's reference twist J dq and its rate J ddq + (dJ/dt) dq (world, linear on top)."""
        qa, dqa, ddqa = self._arm(self.q_ref, arm), self._arm(self.dq_ref, arm), self._arm(self.ddq_now, arm)
        J = self.kin.jac(arm, qa, tcp)
        h = 1e-4
        Jdot_dq = (self.kin.jac(arm, qa + h * dqa, tcp) @ dqa - J @ dqa) / h
        self.kin.set_arm(arm, qa)
        return J @ dqa, J @ ddqa + Jdot_dq

    def solve_ik(self, ee, target, tcp=None, seed=None, free=None, after=None) -> dict:
        """Joint angles that put an end effector's controlled point (TCP, or tcp= in the TCP frame) at target, searched
        from seed (default: where after= puts the arm, see _start_from, else the current reference) within the joint
        limits; nothing moves. free= leaves the turn about the tool axis ('yaw', 'yaw180') or the whole rotation
        ('rotation') to the solver, which picks the solution quickest to reach that keeps away from joint limits; T
        is the pose it reaches."""
        if ee not in self.arms:
            raise InvalidCommand(f"unknown end effector {ee!r}; end effectors: {self.arms}")
        names = self.groups[ee]
        if free not in FREE:
            raise InvalidCommand(f"free must be one of {FREE[1:]}, got {free!r}")
        tcp = None if tcp is None else _T(tcp)
        with self.plan_lock:
            if seed is None:
                q0 = self._arm(self._start_from(after) if after else self.q_ref, ee)
            else:
                q0 = np.asarray(seed, float).reshape(-1)
                if len(q0) != len(names):
                    raise InvalidCommand(f"seed needs {len(names)} values ({names})")
            return self._solution(ee, *self._ik_free(ee, _T(target), q0, tcp, free), tcp)

    def _solution(self, ee, q, ok, err, tcp, T=None) -> dict:
        """solve_ik's dict for joints q (kin_plan, under plan_lock)."""
        T = self.kin_plan.fk(ee, q, tcp) if T is None else T
        return {"joints": list(self.groups[ee]), "q": [float(v) for v in q], "ok": bool(ok), "error_m": float(err),
                "T": T.tolist(), "margin": self.kin_plan.margin(ee, q)}

    def _start_from(self, after) -> np.ndarray:
        """The joints a check starts from: the reference, with each arm after= names where the queue will have put it
        ({arm: its joint values, a 4x4 TCP pose, or {"target", "tcp", "free"} as a move takes them}). A pose is the
        solution remembered for that target (the one its move went to, or a check found), else solved from the
        reference (kin_plan, under plan_lock)."""
        q = self.q_ref.copy()
        for a, v in after.items():
            if a not in self.arms:
                raise InvalidCommand(f"after: unknown arm {a!r}; arms: {self.arms}")
            idx = self.arm_cols[a]
            v = dict(v) if isinstance(v, dict) else {"target": v}
            T = np.asarray(v["target"], float)
            if T.shape == (len(idx),):
                q[idx] = T
                continue
            if T.shape not in ((4, 4), (16,)):
                raise InvalidCommand(f"after[{a!r}]: {len(idx)} joint values or a 4x4 pose, got shape {T.shape}")
            T, tcp, free = _T(T), None if v.get("tcp") is None else _T(v["tcp"]), v.get("free")
            if free not in FREE:
                raise InvalidCommand(f"after[{a!r}]: free must be one of {FREE[1:]}, got {free!r}")
            same = self._remembered(a, T, tcp, free, IK_SAME_M, IK_SAME_RAD, 1)
            if same:
                q[idx] = same[0]
                continue
            sol, ok, err = self._ik_free(a, T, self._arm(self.q_ref, a), tcp, free)
            if not ok:
                raise InvalidCommand(f"after: {a}'s planned pose is unreachable (closest IK solution misses by "
                                     f"{err * 1000:.1f} mm)")
            q[idx] = sol
        return q

    def reachable(self, ee, targets, tcp=None, free=None, line=False, after=None) -> list:
        """For each of several targets, what r.move(target, ee=ee, tcp=, free=, line=) would meet from the current
        reference, nothing moving: solve_ik's dict plus why, None or the refusal the move would get (the fence or
        gripper_floor, IK, the arms' separation with the other arm where it is commanded now and where its running
        move ends, a line's unreachable stretch); ok only when why is None. after= (see _start_from): check as from
        where the queue will have put the arms instead: IK seeded there, and an arm it names checked for separation
        there (only there)."""
        if after:
            with self.plan_lock:
                q_start = self._start_from(after)
        out = []
        for T in targets:
            spec = {"op": "move_to", "ee": ee, "target": T, "tcp": tcp, "free": free, "line": bool(line)}
            cmd = Command(spec, *self._groups_of(spec))
            if after:
                cmd.q_start, cmd.planned = q_start, tuple(after)
            why = None
            try:
                with self.plan_lock:
                    self._build(cmd)
                    if line:
                        w = self._walk(ee, self._point(ee, cmd.tcp[ee], q=cmd.q_start, kin=self.kin_plan),
                                       [cmd.pose_goal[ee]], cmd.tcp[ee], q0=cmd.q_start)
                        if not w["ok"]:
                            why = (f"line: unreachable {w['fraction']:.0%} of the way (misses by "
                                   f"{w['error_m'] * 1000:.1f} mm)")
            except InvalidCommand as e:
                why = str(e)
            if why is None or getattr(cmd, "sep_goal", None):       # IK reached (a later check refused it)
                with self.plan_lock:
                    r = self._solution(ee, cmd.sep_goal[ee], True, 0.0, cmd.tcp[ee], cmd.pose_goal[ee])
            elif getattr(cmd, "ik_miss", None):                       # IK's closest miss
                with self.plan_lock:
                    r = self._solution(ee, *cmd.ik_miss, cmd.tcp[ee])
            else:                                                    # refused before IK (the fence)
                r = self.solve_ik(ee, T, tcp, None if cmd.q_start is None else self._arm(cmd.q_start, ee), free)
            r["ok"] = why is None
            r["why"] = why
            out.append(r)
        return out

    def dry_run(self, spec: dict) -> dict:
        """Check a move without running it: validate and solve it as for real (goals' IK, free axes), and walk a
        straight line or a pose path every DRY_RUN_STEP from the current reference, solving IK at each sample from
        the last. ok, or where it would fail (reason, the point and fraction of its leg, the pose, the IK miss); the
        joint goal(s); the tightest margin on the way (joint limit, singularity)."""
        cmd = self.prepare(spec)
        s, out = cmd.spec, {"ok": True}
        with self.plan_lock:
            if cmd.kind == "move_to" and hasattr(cmd, "joint_goal"):
                js, goal = cmd.joint_goal
                out["q_goal"] = {"joints": js, "q": [float(v) for v in goal]}
            elif cmd.kind == "move_to" and not s.get("line"):
                out["q_goal"] = {a: [float(v) for v in q] for a, q in cmd.ik_goal.items()}
                out["goal"] = {a: T.tolist() for a, T in cmd.pose_goal.items()}
                out["margin"] = {a: self.kin_plan.margin(a, q) for a, q in cmd.ik_goal.items()}
            elif cmd.kind == "move_to" or (cmd.kind == "follow" and not s.get("joints")):
                for a in cmd.arms:
                    base = self._point(a, cmd.tcp[a], kin=self.kin_plan)
                    poses = [cmd.pose_goal[a]] if cmd.kind == "move_to" else self._world_poses(cmd, a, base)
                    r = self._walk(a, base, poses, cmd.tcp[a])
                    out.setdefault("q_goal", {})[a] = r.pop("q_goal")
                    out.setdefault("margin", {})[a] = r.pop("margin")
                    if not r["ok"]:
                        out.update(r, arm=a)
                        break
                    out.setdefault("goal", {})[a] = poses[-1].tolist()
            elif cmd.kind == "follow":
                out["q_goal"] = {"joints": list(s["joints"]), "q": [float(v) for v in np.asarray(s["points"])[-1]]}
            else:
                out["note"] = f"{cmd.kind}: nothing to walk (checked and accepted)"
        return out

    def _world_poses(self, cmd: Command, a, base):
        """A pose path's points as world poses (relative ones from base, the controlled point at the start)."""
        s = cmd.spec
        pts = s["points"][a] if isinstance(s["points"], dict) else s["points"]
        Ps = [_T(p) for p in pts]
        if s.get("relative"):
            Ps = [base @ P for P in Ps] if s.get("frame", "world") == "tool" else [_relative_world(base, P) for P in Ps]
        return Ps

    def _walk(self, arm, T0, poses, tcp, q0=None):
        """IK along straight legs from T0 through poses, every DRY_RUN_STEP, each sample from the last solution (the
        first from q0's joints, default the reference)."""
        from scipy.spatial.transform import Rotation, Slerp
        kin = self.kin_plan
        q = self._arm(self.q_ref if q0 is None else q0, arm)
        worst = kin.margin(arm, q)
        prev = T0
        for k, T in enumerate(poses):
            d = np.linalg.norm(T[:3, 3] - prev[:3, 3])
            ang = np.linalg.norm(pose_err(T, prev)[3:])
            n = max(1, int(np.ceil(max(d / DRY_RUN_STEP_M, ang / DRY_RUN_STEP_RAD))))
            sl = Slerp([0, 1], Rotation.from_matrix(np.stack([prev[:3, :3], T[:3, :3]])))
            for i in range(1, n + 1):
                f = i / n
                Ts = np.eye(4)
                Ts[:3, :3] = sl([f]).as_matrix()[0]
                Ts[:3, 3] = prev[:3, 3] + f * (T[:3, 3] - prev[:3, 3])
                q, ok, err = kin.ik(arm, Ts, q, tcp, iters=80, restarts=0)
                if not ok:
                    return {"ok": False, "reason": "unreachable", "point": k, "fraction": round(f, 3),
                            "T": Ts.tolist(), "error_m": float(err), "q_goal": [float(v) for v in q], "margin": worst}
                m = kin.margin(arm, q)
                if m["limit_rad"] < worst["limit_rad"] or m["singular_m"] < worst["singular_m"]:
                    worst = {"limit_rad": min(m["limit_rad"], worst["limit_rad"]),
                             "limit_joint": m["limit_joint"] if m["limit_rad"] < worst["limit_rad"] else worst["limit_joint"],
                             "singular_m": min(m["singular_m"], worst["singular_m"])}
            prev = T
        return {"ok": True, "q_goal": [float(v) for v in q], "margin": worst}

    def _cost(self, arm, q_from, q, idx=None):
        """How bad a solution is to go to: the time at the joints' speed limits, plus penalties near a joint limit
        and near a singularity (s)."""
        idx = [self.jidx[n] for n in self.groups[arm]] if idx is None else idx
        t = float(np.max(np.abs(np.asarray(q) - np.asarray(q_from)) / self.vmax[idx]))
        m = self.kin_plan.margin(arm, q)
        return t + 2.0 * max(0.0, 0.2 - m["limit_rad"]) + 10.0 * max(0.0, 0.03 - m["singular_m"])

    def _ik_free(self, arm, T, q0, tcp, free, remember=True):
        """_ik_solve helped by what was solved before (ik_memory): the joints of up to IK_MEMORY_SEEDS remembered
        solutions for targets near T (same tcp and free) are seeds too. The solve from q0 stands when it reaches,
        unless one of those reaches and is IK_MEMORY_BETTER_S quicker to go to (the solve from q0 found another branch
        from a random restart: the remembered one is the solution reachable() or the ghost showed). When neither
        reaches, a polish from the closest miss. A solution that reaches is remembered unless remember is False (a
        path's points: they would push out the goals checked) (kin_plan, under plan_lock).
        Whichever is chosen, the caller's fence, gripper_floor and separation checks apply to it as to any."""
        best = self._ik_solve(arm, T, q0, tcp, free)
        cands = [c for q in self._remembered(arm, T, tcp, free) for c in self._ik_seeded(arm, T, q, tcp, free)]
        ok = [c for c in cands if c[1]]
        if ok:
            c = min(ok, key=lambda c: self._cost(arm, q0, c[0]))
            if not best[1] or self._cost(arm, q0, c[0]) < self._cost(arm, q0, best[0]) - IK_MEMORY_BETTER_S:
                best = c
        elif not best[1]:
            best = min([best] + cands, key=lambda c: c[2])
        if not best[1]:
            best = self._ik_polish(arm, T, best, tcp, free)
        if best[1] and remember:                         # the newest solution for a target replaces older ones
            T = np.asarray(T, float)
            same = lambda e: (e[2] == free and (e[1] is None) == (tcp is None) and np.abs(e[0] - T).max() < IK_SAME_M
                              and (tcp is None or np.allclose(e[1], tcp, atol=1e-9)))
            mem = collections.deque((e for e in self.ik_memory[arm] if not same(e)), maxlen=IK_MEMORY)
            mem.append((T.copy(), None if tcp is None else np.array(tcp, float), free, np.array(best[0], float)))
            self.ik_memory[arm] = mem
        return best

    def _ik_polish(self, arm, T, best, tcp, free):
        """A last try for a target IK missed: Levenberg-Marquardt from the closest miss (Kinematics.ik_polish). In
        simulation it reaches about one in five of the reachable targets near a joint limit that IK missed, for ~10 ms
        each (a solve that misses takes ~70 ms)."""
        f = free if free in ("yaw", "rotation") else None
        for th in ([0.0, np.pi] if free == "yaw180" else [0.0]):
            c = self.kin_plan.ik_polish(arm, T @ _rot_z(th), best[0], tcp, free=f)
            if c[1]:
                return c
        return best

    def _remembered(self, arm, T, tcp, free, pos=IK_MEMORY_POS_M, rot=IK_MEMORY_ROT_RAD, n=IK_MEMORY_SEEDS) -> list:
        """The joints of up to n remembered solutions for targets within pos and rot of T with the same tcp and free
        (rotation: the tool axis's turn for 'yaw', 'yaw180'; none for 'rotation'), nearest first, newest first among
        equals, no two alike."""
        T = np.asarray(T, float)
        near = []
        for k, (Tm, tm, fm, q) in enumerate(reversed(self.ik_memory[arm])):
            if fm != free or (tm is None) != (tcp is None) or (tm is not None and not np.allclose(tm, tcp, atol=1e-9)):
                continue
            dp = float(np.linalg.norm(Tm[:3, 3] - T[:3, 3]))
            if free == "rotation":
                dr = 0.0
            elif free in ("yaw", "yaw180"):
                dr = float(np.arccos(np.clip(Tm[:3, 2] @ T[:3, 2], -1.0, 1.0)))
            else:
                dr = float(np.linalg.norm(pose_err(Tm, T)[3:]))
            if dp <= pos and dr <= rot:
                near.append((dp + 0.1 * dr, k, q))
        out = []
        for _, _, q in sorted(near, key=lambda x: x[:2]):
            if not any(np.allclose(q, o, atol=1e-6) for o in out):
                out.append(q)
            if len(out) == n:
                break
        return out

    def _ik_seeded(self, arm, T, seed, tcp, free) -> list:
        """IK from one seed, no restarts: the candidates _ik_solve would weigh (for 'yaw180' both turns)."""
        kin = self.kin_plan
        if free in ("yaw", "rotation"):
            return [kin.ik(arm, T, seed, tcp, free=free, restarts=0)]
        turns = [0.0, np.pi] if free == "yaw180" else [0.0]
        return [kin.ik(arm, T @ _rot_z(th), seed, tcp, restarts=0) for th in turns]

    def _ik_solve(self, arm, T, q0, tcp, free):
        """IK with free axes: the candidate solutions (the masked solve from q0; for 'yaw' also every YAW_STEP turn
        about the tool axis, for 'yaw180' the half turn) and the cheapest of those that reach (kin_plan, under
        plan_lock)."""
        kin = self.kin_plan
        if free is None:
            return kin.ik(arm, T, q0, tcp)
        cands = []
        if free in ("yaw", "rotation"):
            cands.append(kin.ik(arm, T, q0, tcp, free=free))
        turns = np.arange(0.0, 2 * np.pi - 1e-9, YAW_STEP) if free == "yaw" else [0.0, np.pi] if free == "yaw180" else []
        for th in turns:
            Tk = T @ _rot_z(th)
            cands.append(kin.ik(arm, Tk, q0, tcp, iters=80, restarts=0 if free == "yaw" else 2))
        ok = [c for c in cands if c[1]]
        if not ok:
            return min(cands, key=lambda c: c[2])
        return min(ok, key=lambda c: self._cost(arm, q0, c[0]))

    def _resolve_path(self, arm, poses, times, tcp, free):
        """Full poses for a pose path whose points leave axes free (free: one value or one per point): the choice
        of turn at each point that makes the whole path cheapest (dynamic programming over the candidates: no wrist
        flips between points; with times, no step faster than the joints can go). A long path, or free='rotation',
        takes each point's nearest solution from the last."""
        free = list(free) if isinstance(free, (list, tuple)) else [free] * len(poses)
        if len(free) != len(poses):
            raise InvalidCommand(f"free has {len(free)} entries for {len(poses)} points")
        bad = [f for f in free if f not in FREE]
        if bad:
            raise InvalidCommand(f"free must be one of {FREE[1:]}, got {bad[0]!r}")
        kin = self.kin_plan
        idx = [self.jidx[n] for n in self.groups[arm]]
        q0 = self._arm(self.q_ref, arm)
        if "rotation" in free or len(poses) > DP_MAX_POINTS:
            out, q = [], q0
            for k, (T, f) in enumerate(zip(poses, free)):
                q, ok, err = self._ik_free(arm, T, q, tcp, f, remember=False) if f in ("yaw180",) else \
                    kin.ik(arm, T, q, tcp, free=f, restarts=2)
                if not ok:
                    raise InvalidCommand(f"{arm}: point {k} unreachable (misses by {err * 1000:.1f} mm)")
                out.append(kin.fk(arm, q, tcp))
            return out
        layers = []
        for k, (T, f) in enumerate(zip(poses, free)):
            turns = np.arange(0.0, 2 * np.pi - 1e-9, YAW_STEP) if f == "yaw" else [0.0, np.pi] if f == "yaw180" \
                else [0.0]
            layer = []
            for j, th in enumerate(turns):
                seed = q0
                if layers:                                   # warm start from the previous point's same turn
                    prev = [n for n in layers[-1] if abs(n[2] - th) < 1e-9] or layers[-1]
                    seed = prev[0][0]
                q, ok, _ = kin.ik(arm, T @ _rot_z(th), seed, tcp, iters=150, restarts=1)
                if ok:
                    m = kin.margin(arm, q)
                    pen = 2.0 * max(0.0, 0.2 - m["limit_rad"]) + 10.0 * max(0.0, 0.03 - m["singular_m"])
                    layer.append((q, T @ _rot_z(th), th, pen))
            if not layer:
                raise InvalidCommand(f"{arm}: point {k} is unreachable at every turn about the tool axis")
            layers.append(layer)
        dts = None if times is None else np.diff(np.concatenate([[0.0], np.asarray(times, float)]))
        vcap = self.vcap[idx]
        INF = float("inf")
        cost = [[INF] * len(L) for L in layers]
        back = [[-1] * len(L) for L in layers]
        for j, n in enumerate(layers[0]):
            need = np.abs(n[0] - q0)
            if dts is None or np.max(need / vcap) <= dts[0] + 1e-9:
                cost[0][j] = float(np.max(need / self.vmax[idx])) + n[3]
        for k in range(1, len(layers)):
            for j, n in enumerate(layers[k]):
                for i, p in enumerate(layers[k - 1]):
                    if cost[k - 1][i] == INF:
                        continue
                    need = np.abs(n[0] - p[0])
                    if dts is not None and np.max(need / vcap) > dts[k] + 1e-9:
                        continue
                    c = cost[k - 1][i] + float(np.max(need / self.vmax[idx])) + n[3]
                    if c < cost[k][j]:
                        cost[k][j], back[k][j] = c, i
            if all(c == INF for c in cost[k]):
                raise InvalidCommand(f"{arm}: no choice of turn reaches point {k} from point {k - 1} in time "
                                     "(stretch the times, or fix the turn)")
        j = int(np.argmin(cost[-1]))
        chosen = []
        for k in range(len(layers) - 1, -1, -1):
            chosen.append(layers[k][j][1])
            j = back[k][j]
        return chosen[::-1]

    # ================================================================== the workspace fence
    def fence_violation(self, arm, p, tol=0.0) -> str | None:
        """Which limit of arm's fence box the TCP position p breaks (by more than tol), or None."""
        if arm not in self.fence:
            return None
        lo, hi = self.fence[arm]
        for i, ax in enumerate("xyz"):
            if p[i] < lo[i] - tol:
                return f"{arm} TCP {ax} {p[i]:.3f} m < {ax}_min {lo[i]:.3f} m"
            if p[i] > hi[i] + tol:
                return f"{arm} TCP {ax} {p[i]:.3f} m > {ax}_max {hi[i]:.3f} m"
        return None

    def floor_violation(self, arm, T, tol=0.0) -> str | None:
        """Whether arm's gripper, its TCP site at pose T, reaches below the fence's gripper_floor (by more than tol)."""
        if self.floor is None or arm not in self.fence:
            return None
        low = self.floor.lowest(arm, T)
        if low < self.floor_z - tol:
            return (f"{arm} gripper's lowest point z {low:.4f} m < gripper_floor {self.floor_z:.4f} m (a tilted "
                    f"gripper's fingers and body reach below its TCP)")
        return None

    def _fence_out(self, arm, p) -> float:
        """How far the TCP position p is outside arm's box (m; 0 inside)."""
        lo, hi = self.fence[arm]
        return float(max(0.0, np.max(lo - p), np.max(p - hi)))

    def _fence_targets(self, cmd: Command):
        """Refuse a command whose targets put a fenced TCP outside its box, or that asks for a planar TCP speed above
        the fence's limit (its speed=, a timed path's legs and velocities=, a move_at's twist). A move_at stops at
        the fence (reason 'fence'). Runs in prepare, under plan_lock: positions from kin_plan, from the reference."""
        s, op = cmd.spec, cmd.kind
        arms = [a for a in cmd.arms if a in self.fence]
        if not arms or op in ("gripper", "hold"):
            return
        kin, v = self.kin_plan, self.fence_v

        def tcp_of(a, T):                       # the TCP's pose, from a controlled-point pose
            tcp = cmd.tcp.get(a)
            return T @ np.linalg.inv(tcp) if tcp is not None else np.asarray(T, float)

        def check(a, T, what):                  # T: the TCP's pose (the box bounds its point, gripper_floor all of it)
            why = self.fence_violation(a, T[:3, 3]) or self.floor_violation(a, T)
            if why:
                raise InvalidCommand(f"fence: {what}: {why}")

        def fast(what, speed):
            if speed > v * (1 + 1e-9):
                raise InvalidCommand(f"fence: {what} needs {speed:.2f} m/s in the table plane (limit "
                                     f"{v:.2f} m/s){'' if 'speed=' in what else ': stretch the times'}")

        if op in ("move_to", "follow") and not s.get("joints") and s.get("speed") is not None:
            fast("speed=", float(s["speed"]))
        if op == "move_to" and s.get("joints"):
            js, goal = cmd.joint_goal
            q = (self.q_ref if cmd.q_start is None else cmd.q_start).copy()
            q[[self.jidx[n] for n in js]] = goal
            for a in arms:
                check(a, kin.fk(a, self._arm(q, a)), "the joint target")
        elif op == "move_to":
            for a in arms:
                T = kin.fk(a, cmd.ik_goal[a]) if hasattr(cmd, "ik_goal") else tcp_of(a, cmd.pose_goal[a])
                check(a, T, "the target")
        elif op == "follow" and s.get("joints"):
            js = list(s["joints"])
            idx = [self.jidx[n] for n in js]
            q = self.q_ref.copy()
            times = s.get("times") if s.get("times") is not None else (
                None if s.get("dt") is None else [(i + 1) * float(s["dt"]) for i in range(len(s["points"]))])
            prev = {a: kin.fk(a, self._arm(q, a))[:3, 3] for a in arms}
            t_prev = 0.0
            for k, P in enumerate(np.asarray(s["points"], float)):
                q[idx] = P
                for a in arms:
                    T = kin.fk(a, self._arm(q, a))
                    p = T[:3, 3]
                    check(a, T, f"point {k}")
                    if times is not None:
                        fast(f"point {k}", np.linalg.norm((p - prev[a])[:2]) / max(float(times[k]) - t_prev, 1e-9))
                    prev[a] = p
                if times is not None:
                    t_prev = float(times[k])
        elif op == "follow":
            times = s.get("times")
            for a in arms:
                base = self._point(a, cmd.tcp[a], kin=kin)
                Ps = self._world_poses(cmd, a, base)
                tt = times if times is not None else (
                    None if s.get("dt") is None else [(i + 1) * float(s["dt"]) for i in range(len(Ps))])
                prev, t_prev = kin.fk(a, self._arm(self.q_ref, a))[:3, 3], 0.0
                for k, P in enumerate(Ps):
                    T = tcp_of(a, P)
                    p = T[:3, 3]
                    check(a, T, f"point {k}")
                    if tt is not None:
                        fast(f"point {k}", np.linalg.norm((p - prev)[:2]) / max(float(tt[k]) - t_prev, 1e-9))
                        t_prev = float(tt[k])
                    prev = p
                V = s.get("velocities")
                if V is not None:
                    V = np.asarray(V[a] if isinstance(V, dict) else V, float)
                    if s.get("relative") and s.get("frame", "world") == "tool":
                        V = V @ base[:3, :3].T
                    if len(V):
                        k = int(np.argmax(np.linalg.norm(V[:, :2], axis=1)))
                        fast(f"velocities[{k}]", float(np.linalg.norm(V[k, :2])))
        elif op == "move_at" and not s.get("joints"):
            vel = s["velocity"]
            cmd.fence_travel = {}
            for a in arms:
                tw = vel[a] if isinstance(vel, dict) and a in vel else vel
                tw = np.array([float(tw.get(k, 0.0)) for k in ("x", "y", "z", "rx", "ry", "rz")]) \
                    if isinstance(tw, dict) else np.asarray(tw, float)
                p0 = kin.fk(a, self._arm(self.q_ref, a))[:3, 3]
                if s.get("frame", "world") == "tool":
                    tw = np.concatenate([self._point(a, cmd.tcp[a], kin=kin)[:3, :3] @ tw[:3], tw[3:]])
                fast("the twist", float(np.linalg.norm(tw[:2])))
                if self._fence_out(a, p0) > FENCE_REF_TOL:
                    raise InvalidCommand(f"fence: move_at's start (move inside the fence first): "
                                         f"{self.fence_violation(a, p0)}")
                u = tw[:3]
                if np.linalg.norm(u) < 1e-9:
                    continue
                u = u / np.linalg.norm(u)
                lo, hi = self.fence[a]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ts = np.where(u > 1e-9, (hi - p0) / u, np.where(u < -1e-9, (lo - p0) / u, np.inf))
                cmd.fence_travel[a] = float(max(0.0, np.min(ts) - 0.001))

    def _fence_step(self, q_prev):
        """Each tick, after the references stepped: a fenced arm's reference TCP moves no faster than the planar limit
        in the table plane (the step is scaled down, keeping its direction, as for the joint velocity limits), and a
        reference that would leave the box stops at its edge: the arm's commands end 'aborted' (reason 'fence: ...')
        and it holds where it was."""
        parked = {a for c in self.commands if c.park and not c.finished for a in c.arms}
        if self.parking or parked:                  # a park goes where it must; the parked arm's checks start again
            for a in self.fence:                    # once it is back inside its box (a park ends outside it)
                if self.parking or a in parked:
                    self.fence_armed[a] = False
            self.fence_p.clear()
            self.fence_low.clear()
            return
        for a in self.fence:
            idx = [self.jidx[n] for n in self.groups[a]]
            qa0, qa = q_prev[idx], self.q_ref[idx]
            p0 = self.fence_p.get(a)
            if p0 is None:
                p0 = self.kin.fk(a, qa0)[:3, 3]
            moved = bool(np.any(qa != qa0))
            T_a = self.kin.fk(a, qa) if moved else None          # one FK for the check and the nominal pose
            p = T_a[:3, 3].copy() if moved else p0
            if self.fence_armed[a] and moved:
                step = float(np.linalg.norm((p - p0)[:2]))
                if step > self.fence_v * self.period:
                    k = self.fence_v * self.period / step
                    qa = qa0 + k * (qa - qa0)
                    self.q_ref[idx] = qa
                    self.dq_ref[idx] *= k
                    T_a = self.kin.fk(a, qa)
                    p = T_a[:3, 3].copy()
                out = self._fence_out(a, p)
                why = self.fence_violation(a, p) if out > max(FENCE_REF_TOL, self._fence_out(a, p0) + 1e-6) else None
                if why is None and self.floor is not None:
                    # the whole gripper above gripper_floor: a reference going lower past FLOOR_REF_TOL stops (one
                    # already below may rise)
                    low = self.floor.lowest(a, T_a)
                    low0 = self.fence_low.get(a)
                    if low0 is None:
                        low0 = self.floor.lowest(a, self.kin.fk(a, qa0))
                    if low < self.floor_z - FLOOR_REF_TOL and low < low0 - 1e-6:
                        why = (f"{a} gripper's lowest point z {low:.4f} m < gripper_floor {self.floor_z:.4f} m (a "
                               f"tilted gripper's fingers and body reach below its TCP)")
                if why:                                 # (a reference already outside may move back in)
                    for c in self.commands:
                        if c.status == "running" and a in c.arms:
                            self._finish(c, "aborted", f"fence: {why}")
                    self.q_ref[idx], self.dq_ref[idx], self.ddq_now[idx] = qa0, 0.0, 0.0
                    self.T_des[a] = None
                    qa, p = qa0, p0
                    T_a = self.kin.fk(a, qa)
                self.T_nom[a] = T_a
                if self.floor is not None:
                    self.fence_low[a] = self.floor.lowest(a, T_a)
            if not self.fence_armed[a] and self.fence_violation(a, p) is None \
                    and self.fence_violation(a, self.T[a][:3, 3]) is None:
                self.fence_armed[a] = True
                self._event("fence", f"{a} TCP inside its fence: the fence is on")
            self.fence_p[a] = p

    def _fence_guard(self):
        """A fenced TCP measured outside its box by more than the margin, or moving faster than FENCE_SPEED_FAULT x
        the planar limit for FENCE_SPEED_S, latches a Fault."""
        if self.fault or self.parking or any(c.park and not c.finished for c in self.commands):
            return
        for a in self.fence:
            if not self.fence_armed[a]:
                continue
            why = self.fence_violation(a, self.T[a][:3, 3], self.fence_margin)
            if why:
                self._latch_fault(f"fence: {why} (by more than the {self.fence_margin * 1000:.0f} mm margin)",
                                  code="fence", arm=a)
                return
            speed = float(np.linalg.norm(self.v[a][:2]))
            self.fence_fast[a] = self.fence_fast[a] + 1 if speed > FENCE_SPEED_FAULT * self.fence_v else 0
            if self.fence_fast[a] * DT >= FENCE_SPEED_S:
                self._latch_fault(f"fence: {a} TCP moving at {speed:.2f} m/s in the table plane (limit "
                                  f"{self.fence_v:.2f} m/s)", code="fence", arm=a)
                return

    # ================================================================== the arms kept apart
    def _arm_q(self, q) -> dict:
        return {n: float(q[self.jidx[n]]) for a in self.arms for n in self.groups[a] + self.groups[f"{a}_gripper"]}

    def _separation_targets(self, cmd: Command):
        """Refuse a command whose goal (a joint target, or the IK solution of a pose target) puts the arms closer than
        separation's min_m, unless it moves them apart: with the other arm at its reference, and again with it where
        the move it runs ends (it would close the gap as it gets there). A check from a planned start (cmd.q_start,
        reachable's after=) starts there, and an arm after= placed is checked there only. Runs in prepare, under
        plan_lock, on kin_plan. Paths without one goal are checked tick by tick (_separation_step)."""
        now = self._arm_q(self.q_ref if cmd.q_start is None else cmd.q_start)
        if hasattr(cmd, "joint_goal"):
            js, target = cmd.joint_goal
            mine = {n: float(v) for n, v in zip(js, target) if n in now}
        elif getattr(cmd, "sep_goal", None):
            mine = {n: float(v) for a, q in cmd.sep_goal.items() for n, v in zip(self.groups[a], q)}
        else:
            return
        others = [a for a in self.arms if a not in cmd.arms]
        cases = [(now, " (with " + " and ".join(f"{a} where after= puts it" for a in others) + ")"
                  if cmd.planned and set(others) & set(cmd.planned) else "")]
        for a, q, c in self._running_goals(cmd):
            if a not in cmd.planned:
                cases.append(({**now, **dict(zip(self.groups[a], map(float, q)))},
                              f" once {a} gets where its running {c.kind} (command {c.id}) ends"))
        for ref, when in cases:
            d_goal, shapes = self.sep.at(self.kin_plan, {**ref, **mine})
            if d_goal < self.sep.min_m and d_goal < self.sep.at(self.kin_plan, ref)[0] - 1e-6:
                raise InvalidCommand(f"separation: the goal {_gap(d_goal)} ({shapes[0]} and {shapes[1]}){when}; the "
                                     f"arms are kept at least {self.sep.min_m * 100:.0f} cm apart")

    def _running_goals(self, cmd: Command) -> list:
        """(arm, its joints, the command) for each arm cmd leaves alone that a running or waiting command takes to one
        goal (a joint target, a pose target's IK solution) and that cmd would not replace."""
        out = []
        for c in list(self.commands):
            if c is cmd or c.finished or c.status == "building" or c.park or set(c.groups) & set(cmd.groups):
                continue
            if hasattr(c, "joint_goal"):
                js, target = c.joint_goal
                goal = dict(zip(js, map(float, target)))
                out += [(a, [goal.get(n, float(self.q_ref[self.jidx[n]])) for n in self.groups[a]], c)
                        for a in c.arms if a not in cmd.arms]
            elif getattr(c, "sep_goal", None):
                out += [(a, q, c) for a, q in c.sep_goal.items() if a not in cmd.arms]
        return out

    def _separation_step(self, q_prev):
        """Each tick, after the fence: when the references would bring the arms closer than min_m, and closer than
        they were, the arms whose references moved hold where they were and their commands end 'aborted' (reason
        'separation: ...'), as the fence stops an arm at its box. Moving apart is always allowed; a park goes where
        it must."""
        parking = self.parking or any(c.park and not c.finished for c in self.commands)
        d_new, shapes = self.sep.at(self.kin, self._arm_q(self.q_ref))
        if (not parking and self.sep_prev is not None and d_new < self.sep.min_m
                and d_new < self.sep_prev - 1e-6):
            moved = [a for a in self.arms if np.any(self._arm(self.q_ref, a) != self._arm(q_prev, a))]
            why = (f"the move {_gap(d_new)} ({shapes[0]} and {shapes[1]}; the arms are kept at least "
                   f"{self.sep.min_m * 100:.0f} cm apart)")
            for a in moved:
                idx = [self.jidx[n] for n in self.groups[a]]
                for c in self.commands:
                    if c.status == "running" and a in c.arms:
                        self._finish(c, "aborted", f"separation: {why}")
                self.q_ref[idx], self.dq_ref[idx], self.ddq_now[idx] = q_prev[idx], 0.0, 0.0
                self.T_des[a] = None
                self.T_nom[a] = self.kin.fk(a, self._arm(self.q_ref, a))
            self._event("separation", f"{'+'.join(moved)} stopped: {why}", arms=moved)
            d_new = self.sep.at(self.kin, self._arm_q(self.q_ref))[0]
        self.sep_prev = d_new

    def _separation_guard(self):
        """The measured arms closer than fault_m (they lag their references, or something pushed them) latch a Fault.
        After it the check waits until they are 1 cm clear of fault_m again, so they can be moved apart."""
        d, shapes = self.sep_meas
        if not self.sep_armed:
            self.sep_armed = d > self.sep.fault_m + 0.01
            return
        if self.fault or self.parking or any(c.park and not c.finished for c in self.commands):
            return
        if d < self.sep.fault_m:
            self.sep_armed = False
            self._latch_fault(f"separation: measured, {_gap(d)} ({shapes[0]} and {shapes[1]})", code="separation")

    def _fence_rearm(self):
        """After a Fault or a recover, an arm whose TCP is outside its box may move back in (its targets must be
        inside); its runtime checks start again once it is."""
        for a in self.fence:
            if self.fence_violation(a, self.T[a][:3, 3]) is not None:
                self.fence_armed[a] = False
            self.fence_p.pop(a, None)
            self.fence_low.pop(a, None)
            self.fence_fast[a] = 0

    def _build(self, cmd: Command):
        s, op = cmd.spec, cmd.kind
        for a in cmd.arms:
            cmd.tcp[a] = self._tcp_for(s, a)
        comp = s.get("compliance")
        if comp:
            for a in cmd.arms:
                c = comp[a] if (isinstance(comp, dict) and a in comp) else comp
                if c:
                    cmd.compliance[a] = c
        if s.get("gains"):
            self._check_gains(s["gains"], cmd.groups)
        cmd.until = compile_named(s.get("until"), DT)
        for name, (cond, action) in (s.get("on") or {}).items():
            cmd.on[name] = (compile_cond(cond, DT), action)
        if op == "gripper":
            lo, hi = self.kin.limits[self.finger[s["name"]]]
            target = float(np.clip(float(s["width"]) / 2.0, lo, hi))
            cmd.gripper_target = target
        elif op == "move_to" and s.get("joints"):
            target = np.asarray(s["target"], float).reshape(-1)
            if len(target) != len(s["joints"]):
                raise InvalidCommand(f"target has {len(target)} values for {len(s['joints'])} joints")
            cmd.joint_goal = (list(s["joints"]), target)
        elif op == "move_to":
            tgt = s["target"]
            targets = {a: _T(tgt[a] if isinstance(tgt, dict) else tgt) for a in cmd.arms}
            frame = s.get("frame", "world")
            if frame != "world":
                targets = {a: self._point(a, cmd.tcp[a], q=cmd.q_start, kin=self.kin_plan) @ T if frame == "tool"
                           else T for a, T in targets.items()}
            free = s.get("free")
            if free not in FREE:
                raise InvalidCommand(f"free must be one of {FREE[1:]}, got {free!r}")
            for a, T in targets.items():                   # the fence before IK: say which limit, not "unreachable"
                p = (T @ np.linalg.inv(cmd.tcp[a]))[:3, 3] if cmd.tcp[a] is not None else T[:3, 3]
                why = None if cmd.park else self.fence_violation(a, p)
                if why:
                    raise InvalidCommand(f"fence: the target: {why}")
            cmd.ik_goal = {}
            for a, T in targets.items():
                q0 = self._arm(self.q_ref if cmd.q_start is None else cmd.q_start, a)
                q, ok, err = self._ik_free(a, T, q0, cmd.tcp[a], free)
                if not ok:
                    cmd.ik_miss = (q, ok, err)
                    raise InvalidCommand(f"{a}: target unreachable (closest IK solution misses by {err * 1000:.1f} mm)")
                cmd.ik_goal[a] = q
                if free is not None:                         # the pose the free axes settled on
                    targets[a] = self.kin_plan.fk(a, q, cmd.tcp[a])
            cmd.pose_goal = targets
            cmd.sep_goal = dict(cmd.ik_goal)
            if s.get("line"):
                del cmd.ik_goal
        elif op == "follow":
            pts = s["points"]
            if s.get("times") is None and s.get("dt") is None and isinstance(pts, dict) \
                    and len({len(pts[a]) for a in cmd.arms if a in pts}) > 1:
                raise InvalidCommand("without times= the arms share one timeline: give each the same number of poses")
            if s.get("joints"):
                self._check_joint_path(s)
            elif s.get("free") is not None:
                self._resolve_free_path(cmd)
            if not s.get("joints") and s.get("velocities") is not None:
                for a in cmd.arms:
                    V = s["velocities"][a] if isinstance(s["velocities"], dict) else s["velocities"]
                    pts = s["points"][a] if isinstance(s["points"], dict) else s["points"]
                    if np.asarray(V, float).shape != (len(pts), 3):
                        raise InvalidCommand(f"{a}: velocities must be one 3-vector (the controlled point's m/s) per "
                                             f"pose: {len(pts)} x 3, got shape {np.asarray(V, float).shape}")
        elif op == "move_at" and s.get("joints"):
            v = np.abs(np.asarray(s["velocity"], float).reshape(-1))
            cap = np.array([self.vcap[self.jidx[n]] for n in s["joints"]])
            if len(v) != len(cap):
                raise InvalidCommand(f"velocity has {len(v)} values for {len(cap)} joints")
            if np.any(v > cap):
                j = int(np.argmax(v / cap))
                raise InvalidCommand(f"{s['joints'][j]} velocity {v[j]:.2f} rad/s exceeds its limit {cap[j]:.2f} rad/s")
        if self.fence and not cmd.park:
            self._fence_targets(cmd)
        if self.sep is not None and not cmd.park:
            self._separation_targets(cmd)

    def _resolve_free_path(self, cmd: Command):
        """Turn a pose path with free axes into full world poses now (when it is sent, from the current reference):
        the spec's points become those poses, relative to nothing. (Runs in prepare, under plan_lock.)"""
        s = cmd.spec
        pts, frame = s["points"], s.get("frame", "world")
        times = s.get("times")
        if times is None and s.get("dt") is not None:
            n = len(next(iter(pts.values())) if isinstance(pts, dict) else pts)
            times = [(i + 1) * float(s["dt"]) for i in range(n)]
        out = {}
        for a in cmd.arms:
            base = self._point(a, cmd.tcp[a], kin=self.kin_plan)
            Ps = self._world_poses(cmd, a, base)
            free = s["free"][a] if isinstance(s["free"], dict) else s["free"]
            out[a] = [T.tolist() for T in self._resolve_path(a, Ps, times, cmd.tcp[a], free)]
            V = s.get("velocities")
            if V is not None and frame == "tool" and s.get("relative"):    # tool-frame velocities to world
                Va = np.asarray(V[a] if isinstance(V, dict) else V, float) @ base[:3, :3].T
                s["velocities"] = {**V, a: Va.tolist()} if isinstance(V, dict) else Va.tolist()
        s["points"] = out if isinstance(pts, dict) else out[cmd.arms[0]]
        s["relative"], s["frame"] = False, "world"
        cmd.resolved = out

    def _check_joint_path(self, s):
        """Joint paths come from libraries or the agent's own code: refuse one outside the joint limits or faster than
        the joints' velocity limits instead of running it (the message names the point, the joint and the limit, and
        says how much to stretch the times). A planner's path that starts with the current state (at t=0, or as the
        first of dt-spaced points) has that point dropped."""
        js = list(s["joints"])
        bad = [j for j in js if j not in self.jidx]
        if bad:
            raise InvalidCommand(f"unknown joints {bad}; joints: {self.joints}")
        P = np.asarray(s["points"], float)
        if P.ndim != 2 or P.shape[1] != len(js):
            raise InvalidCommand(f"points must be N x {len(js)} for joints {js}")
        idx = [self.jidx[n] for n in js]
        at_start = np.abs(P[0] - self.q_ref[idx]).max() < START_TOL_RAD
        if s.get("times") is not None and float(np.asarray(s["times"], float).reshape(-1)[0]) <= 1e-9:
            if not at_start:
                k = int(np.argmax(np.abs(P[0] - self.q_ref[idx])))
                raise InvalidCommand(f"the point at t=0 must be where the joints are now: {js[k]} is at "
                                     f"{self.q_ref[idx][k]:.3f} rad, the point says {P[0][k]:.3f}")
            s["times"] = list(np.asarray(s["times"], float).reshape(-1)[1:])
        elif s.get("dt") is not None and at_start and len(P) > 1:
            s["times"] = [i * float(s["dt"]) for i in range(1, len(P))]
            s["dt"] = None
        else:
            at_start = False
        if at_start:
            P = P[1:]
            s["points"] = P.tolist()
            if s.get("velocities") is not None:
                s["velocities"] = np.asarray(s["velocities"], float).reshape(len(P) + 1, len(js))[1:].tolist()
            if not len(P):
                raise InvalidCommand("the path has only its start point")
        lo, hi = self.lo[idx], self.hi[idx]
        out = (P < lo - 1e-6) | (P > hi + 1e-6)
        if out.any():
            k, j = np.argwhere(out)[0]
            raise InvalidCommand(f"point {k}: {js[j]} = {P[k, j]:.3f} rad is outside its limits "
                                 f"[{lo[j]:.3f}, {hi[j]:.3f}]")
        if s.get("times") is None and s.get("dt") is None:
            return                                  # timed at its start from the speed limits: never too fast
        times = np.asarray(s["times"], float) if s.get("times") is not None \
            else (np.arange(len(P)) + 1) * float(s["dt"])
        if len(times) != len(P):
            raise InvalidCommand(f"{len(P)} points but {len(times)} times")
        if times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise InvalidCommand("times must be increasing and start after 0 (seconds from the start of the command)")
        q = np.vstack([self.q_ref[idx], P])
        dt = np.diff(np.concatenate([[0.0], times]))
        speed = np.abs(np.diff(q, axis=0)) / dt[:, None]
        if s.get("velocities") is not None:
            speed = np.maximum(speed, np.abs(np.asarray(s["velocities"], float).reshape(len(P), len(js))))
        ratio = speed / self.vcap[idx]
        if ratio.max() > 1.0 + 1e-9:                # (a path planned at exactly the limit is fine)
            k, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
            raise InvalidCommand(f"point {k} needs {speed[k, j]:.2f} rad/s on {js[j]} (limit {self.vcap[idx][j]:.2f}"
                                 f" rad/s, the first point is reached from the current reference): stretch the times by"
                                 f" at least {ratio.max():.2f}x")

    def _joint_legs(self, q0, P, idx, speed=None):
        """Durations of a joint path's legs at speed: the slowest joint's, at its speed limit (speed=: the fastest
        joint's rad/s, as move_to) and four times that in rad/s^2."""
        vmax = self.vmax[idx] * (float(speed) / self.vmax[idx].max() if speed else 1.0)
        vmax = np.minimum(vmax, self.vcap[idx])
        d = np.abs(np.diff(np.vstack([q0, P]), axis=0))
        return np.max(np.maximum(d / vmax, np.sqrt(6.0 * d / (4.0 * vmax))), axis=1)

    def _pose_legs(self, P0, poses, speed=None):
        """Durations of a pose path's legs at speed: the distance at speed= (default the line speed, m/s), the turn
        at the angular speed."""
        v = float(speed or self.desc["linear_speed"])
        legs, prev = [], P0
        for T in poses:
            legs.append(max(np.linalg.norm(T[:3, 3] - prev[:3, 3]) / v,
                            np.linalg.norm(pose_err(T, prev)[3:]) / self.desc["angular_speed"]))
            prev = T
        return np.array(legs)

    def _check_gains(self, gains, groups):
        js = [n for g in groups for n in self.groups[g]]
        cap = self.desc.get("max_gain_scale")          # the real robot: no stiffer than the defaults x cap
        scale = float(gains.get("scale", 1.0))
        if cap is not None and scale > cap + 1e-9:
            raise InvalidCommand(f"gains scale {scale:g} is above this robot's limit ({cap:g} x the defaults, "
                                 "r.description['max_gain_scale']): a stiffened arm pressing on something hard chatters")
        for key in ("kp", "kd"):
            vals = gains.get(key)
            if vals is None:
                continue
            if isinstance(vals, dict):
                bad = [n for n in vals if n not in js]
                if bad:
                    raise InvalidCommand(f"gains {key}: {bad} are not joints of this command ({js})")
                named = {n: float(v) for n, v in vals.items()}
            else:
                arm_js = [n for n in js if n not in self.finger.values()]
                if len(list(vals)) != len(arm_js):
                    raise InvalidCommand(f"gains {key} needs {len(arm_js)} values ({arm_js}) or a name->value dict")
                named = dict(zip(arm_js, map(float, vals)))
            if cap is not None:
                limit = cap if key == "kp" else float(np.sqrt(cap))
                over = {n: v for n, v in named.items() if v > self.desc[f"default_{key}"][n] * limit + 1e-9}
                if over:
                    n, v = next(iter(over.items()))
                    raise InvalidCommand(f"gains {key} {v:g} on {n} is above this robot's limit "
                                         f"({self.desc[f'default_{key}'][n] * limit:g}: the default x "
                                         f"{limit:g}, r.description['max_gain_scale'])")

    def _apply_gains(self, gains, groups):
        """The gains a command runs with on the joints it moves: its own (gains=), else the defaults. They stay when
        it ends (it holds with them) until the next command on those joints."""
        js = [n for g in groups for n in self.groups[g]]
        scale = float(gains.get("scale", 1.0))
        for n in js:
            i = self.jidx[n]
            self.kp[i] = self.desc["default_kp"][n] * scale
            self.kd[i] = self.desc["default_kd"][n] * float(np.sqrt(scale))
        for key, arr in (("kp", self.kp), ("kd", self.kd)):
            if key in gains:
                vals = gains[key]
                if isinstance(vals, dict):
                    for n, v in vals.items():
                        arr[self.jidx[n]] = float(v)
                else:
                    arm_js = [n for n in js if not n.endswith(("joint7", "joint8"))]
                    vals = list(vals)
                    if len(vals) != len(arm_js):
                        raise InvalidCommand(f"gains {key} needs {len(arm_js)} values ({arm_js}) or a name->value dict")
                    for n, v in zip(arm_js, vals):
                        arr[self.jidx[n]] = float(v)

    def _start(self, cmd: Command, t0: float | None = None):
        s = cmd.spec
        cmd.status, cmd.t0 = "running", self.t if t0 is None else t0
        cmd.T0 = {a: self.T[a].copy() for a in cmd.arms}
        cmd.P0 = {a: self._point(a, cmd.tcp[a]) for a in cmd.arms}
        op = cmd.kind
        # gains and a grip's force cap belong to this command, from when it takes over (not from when it was sent: a
        # command waiting for start_at must not change the one still running)
        self._apply_gains(s.get("gains") or {}, cmd.groups)
        if op == "gripper":
            self.grip_relaxed[s["name"]], self.grip_over[s["name"]] = False, 0     # a new grip re-arms a relaxed one
            fj = self.finger[s["name"]]
            cap = self.desc["effort_limit"][fj]
            self.effort_limit[self.jidx[fj]] = cap if s.get("force") is None else min(float(s["force"]), cap)
            fi = self.jidx[fj]
            speed = float(s.get("speed") or self.desc["gripper_speed"])
            q0 = float(self.q_ref[fi])
            cmd.opening = cmd.gripper_target > float(self.m.q[fi])
            if cmd.opening:                                          # opening: from where the fingers are (a grip's
                q0 = max(q0, float(self.m.q[fi]))                    # reference sits inside what it holds)
            cmd.gref = refs.GripperRef(q0, cmd.gripper_target, speed / 2.0, DT)
            self.gripper_state[s["name"]].update(stalled=0.0, empty=False)
        elif op == "move_to" and hasattr(cmd, "joint_goal"):
            js, goal = cmd.joint_goal
            idx = [self.jidx[n] for n in js]
            vmax = self.vmax[idx] * (float(s["speed"]) / self.vmax[idx].max() if s.get("speed") else 1.0)
            vmax = np.minimum(vmax, self.vcap[idx])
            cmd.ref = (js, refs.JointGoal(self.q_ref[idx], self.dq_ref[idx], goal, vmax, vmax * 4, vmax * 40, DT,
                                          s.get("duration"), self.ddq_now[idx]))
        elif op == "move_to" and not s.get("line"):
            js = [n for a in cmd.arms for n in self.groups[a]]
            goal = np.concatenate([cmd.ik_goal[a] for a in cmd.arms])
            idx = [self.jidx[n] for n in js]
            vmax = self.vmax[idx]
            dur = s.get("duration")
            if s.get("speed") and not dur:
                dist = max(np.linalg.norm(cmd.pose_goal[a][:3, 3] - cmd.P0[a][:3, 3]) for a in cmd.arms)
                dur = dist / float(s["speed"])
            cmd.ref = (js, refs.JointGoal(self.q_ref[idx], self.dq_ref[idx], goal, vmax, vmax * 4, vmax * 40, DT, dur,
                                          self.ddq_now[idx]))
        elif op == "move_to":                                         # straight line
            speed = float(s.get("speed") or self.desc["linear_speed"])
            for a in cmd.arms:
                d = np.linalg.norm(cmd.pose_goal[a][:3, 3] - cmd.P0[a][:3, 3])
                ang = np.linalg.norm(pose_err(cmd.pose_goal[a], cmd.P0[a])[3:])
                dur = float(s.get("duration") or max(d / speed * 1.875, ang / self.desc["angular_speed"] * 1.875, 0.2))
                v0, a0 = self._cart_rates(a, cmd.tcp[a])
                cmd.cart[a] = refs.PoseBlend(refs.PoseLine(self._start_pose(cmd, a), cmd.pose_goal[a], dur), v0, a0,
                                             min(BLEND_S, dur))
        elif op == "follow":
            pts = s["points"]
            times = s.get("times")
            if s.get("joints"):
                js = list(s["joints"])
                P = np.asarray(pts, float)
                if P.ndim != 2 or P.shape[1] != len(js):
                    raise InvalidCommand(f"points must be N x {len(js)} for joints {js}")
                idx = [self.jidx[n] for n in js]
                if times is None and s.get("dt") is None:
                    times = _auto_times(self._joint_legs(self.q_ref[idx], P, idx, s.get("speed")), s.get("duration"))
                elif times is None:
                    times = [(i + 1) * float(s["dt"]) for i in range(len(P))]
                times, P, V = _on_clock(times, self._late(cmd), P, s.get("velocities"))
                path = refs.JointSpline(self.q_ref[idx], P, times, V, self.dq_ref[idx])
                cmd.ref = (js, refs.JointBlend(path, self.dq_ref[idx], self.ddq_now[idx], min(BLEND_S, times[-1])))
            else:
                frame = s.get("frame", "world")
                vels = s.get("velocities")
                paths = {}
                for a in cmd.arms:
                    poses = pts[a] if isinstance(pts, dict) else pts
                    Ps = [_T(p) for p in poses]
                    V = vels[a] if isinstance(vels, dict) else vels
                    if V is not None:
                        V = np.asarray(V, float)
                        if frame == "tool" and s.get("relative"):
                            V = V @ cmd.P0[a][:3, :3].T
                    if s.get("relative"):
                        base = cmd.P0[a]
                        if frame == "tool":
                            Ps = [base @ P for P in Ps]
                        else:
                            out = []
                            for P in Ps:
                                T = np.eye(4)
                                T[:3, :3] = P[:3, :3] @ base[:3, :3]
                                T[:3, 3] = base[:3, 3] + P[:3, 3]
                                out.append(T)
                            Ps = out
                    paths[a] = (Ps, V)
                if times is None and s.get("dt") is None:           # one timeline: each leg as long as the slower arm's
                    legs = np.max([self._pose_legs(cmd.P0[a], Ps, s.get("speed")) for a, (Ps, _) in paths.items()], 0)
                    times = _auto_times(legs, s.get("duration"))
                for a, (Ps, V) in paths.items():
                    tt = times if times is not None else [(i + 1) * float(s["dt"]) for i in range(len(Ps))]
                    if len(Ps) != len(tt):
                        raise InvalidCommand(f"{len(Ps)} poses but {len(tt)} times")
                    tt, Ps, V = _on_clock(tt, self._late(cmd), Ps, V)
                    # start at the controlled point's current reference velocity: replacing a moving path is smooth
                    # (the position spline starts at it; the blend adds the turning rate and the acceleration)
                    v0, a0 = self._cart_rates(a, cmd.tcp[a])
                    path = refs.PosePath(self._start_pose(cmd, a), Ps, tt, V, v0[:3])
                    cmd.cart[a] = refs.PoseBlend(path, v0, a0, min(BLEND_S, tt[-1]))
        elif op == "move_at":
            vel = s["velocity"]
            accel = float(s.get("accel") or 0.5)
            if s.get("joints"):
                js = list(s["joints"])
                idx = [self.jidx[n] for n in js]
                cmd.ref = (js, refs.JointVelocityRef(self.q_ref[idx], vel, accel * 4, DT, self.lo[idx], self.hi[idx],
                                                     s.get("duration"), self.dq_ref[idx]))
            else:
                frame = s.get("frame", "world")
                for a in cmd.arms:
                    tw = vel[a] if isinstance(vel, dict) and a in vel else vel
                    tw = np.array([float(tw.get(k, 0.0)) for k in ("x", "y", "z", "rx", "ry", "rz")]) \
                        if isinstance(tw, dict) else np.asarray(tw, float)
                    if frame == "tool":
                        Rt = cmd.P0[a][:3, :3]
                        tw = np.concatenate([Rt @ tw[:3], Rt @ tw[3:]])
                    travel = float(s.get("max_travel") or DEFAULT_MAX_TRAVEL)
                    fence = getattr(cmd, "fence_travel", {}).get(a)
                    cmd.cart[a] = refs.TwistRef(self._start_pose(cmd, a), tw, accel, DT,
                                                max_travel=travel if fence is None else min(travel, fence),
                                                duration=s.get("duration"), v0=self._cart_rates(a, cmd.tcp[a])[0],
                                                travel_reason="fence" if fence is not None and fence < travel
                                                else "max_travel")
        elif op == "hold":
            pass
        for a in cmd.arms:
            self.comp_offset[a][:] = 0.0
            self.comp_trim.pop(a, None)
            self.bias_blocked.discard(a)                        # the settle bias tries again (_bias_follows)

    # ================================================================== the tick
    def tick(self, advance: bool = True):
        """One control period. advance=False leaves the wait for the next period to the caller (realtime loop)."""
        self.sec = {}                          # this tick's time per phase (s): health shows where a tick goes
        self._t_lap = time.perf_counter()
        m = self.b.read()
        self._lap("read")
        if self.ticks and m.t - self.t > BIAS_GAP_S:          # a gap (the backend reconnected, a stall): the arms'
            self._bias_reset()                                 # offsets are learned again from here
        self.period = float(np.clip(m.t - self.t, PERIOD_MIN, PERIOD_MAX)) if self.ticks else DT
        self.t = m.t
        self._measure(m)
        self._lap("measure")
        if self.soft:
            self._soft_hold(m)
        else:
            self._control(m)
        self._lap("write")
        self.ticks += 1
        if self.recorder is not None:
            self.recorder.tick(self)
            self._lap("record")
        if advance:
            self.b.advance(DT)

    def _lap(self, name: str) -> None:
        """Book the time since the last lap to phase name (a few perf_counter calls a tick)."""
        t = time.perf_counter()
        self.sec[name] = self.sec.get(name, 0.0) + (t - self._t_lap)
        self._t_lap = t

    @property
    def soft(self) -> bool:
        return bool(self.fault and self.fault.get("soft"))

    def _soft_hold(self, m):
        """Soft stop: zero stiffness on the arm joints, damping on top of the backend's gravity compensation, so the
        arms float and yield to a push; no stiffness on the grippers. Every reference tracks the measured joints, so
        releasing holds everything where it is without a step."""
        self.q_ref = m.q.copy()
        self.dq_ref[:] = 0.0
        self.ddq_now[:] = 0.0
        self.T_des = {a: None for a in self.arms}
        self._bias_reset()                                     # no stiffness: nothing to correct
        self.q_cmd = self.q_ref.copy()
        kp, kd = self._gains()
        kp[self.arm_idx] = 0.0
        kd[self.arm_idx] = self.soft_kd[self.arm_idx]
        for a in self.arms:
            fi = self.jidx[self.finger[a]]
            kp[fi], kd[fi] = 0.0, 0.0
        self.b.write(self.q_cmd, self.dq_ref, kp, kd, self.effort_limit)

    def _gains(self):
        """The gains to write: the commanded ones, but none on a gripper the overload guard relaxed."""
        kp, kd = self.kp.copy(), self.kd.copy()
        for a in self.arms:
            if self.grip_relaxed[a]:
                fi = self.jidx[self.finger[a]]
                kp[fi], kd[fi] = 0.0, 0.0
        return kp, kd

    def _control(self, m):
        ctx = self._ctx()
        q_prev = self.q_ref.copy()
        # A joint whose reference is outside its model limits (it started there: an encoder zero offset at a resting
        # stop; or it was held where it was pushed) is never stepped to the limit. This tick's references and commands
        # stay within [min(lower, q_ref), max(upper, q_ref)]: it holds or moves in, and once inside, the limits apply.
        self.lo_cmd, self.hi_cmd = np.minimum(self.lo, self.q_ref), np.maximum(self.hi, self.q_ref)
        for cmd in list(self.commands):
            if cmd.status == "pending" and self.t >= float(cmd.spec.get("start_at") or 0):
                # its clock starts one tick back, as for a command sent between ticks: stepped now at DT, it moves
                # on this tick (at 0 a Cartesian path would hold still for a tick and its velocity dip to zero)
                self._take_over(cmd)
                self._start(cmd, self.t - self.period)
            if cmd.status != "running":
                continue
            try:
                self._step_reference(cmd, m)
            except Exception as e:                    # a bug in one command must not wedge the controller
                traceback.print_exc()
                self._finish(cmd, "aborted", f"internal error: {type(e).__name__}: {e}")
                self._hold_measured(cmd)
        self._lap("references")
        # guard: no reference moves a joint faster than its velocity limit (a too-fast path lags instead)
        self.q_ref = np.clip(self.q_ref, q_prev - self.vcap * self.period, q_prev + self.vcap * self.period)
        if self.fence:
            self._fence_step(q_prev)
        if self.sep is not None:
            self._separation_step(q_prev)
        self.q_cmd = self.q_ref.copy()
        for cmd in self.commands:
            if cmd.status == "running":
                for a, c in cmd.compliance.items():
                    self._comply(cmd, a, c)
        self._lap("fence")
        self._feedforward()
        self._settle_bias()
        self._lap("feedforward")
        self._guards(m)
        if self.fence:
            self._fence_guard()
        if self.sep is not None:
            self._separation_guard()
        self._lap("guards")
        ctx.T_ref = {a: self.T_nom[a] for a in self.arms}
        for cmd in list(self.commands):
            if cmd.status == "running":
                self._conditions(cmd, ctx)
        for cmd in list(self.commands):
            if cmd.status == "running":
                self._completion(cmd)
        self._lap("conditions")
        kp, kd = self._gains()
        self.b.write(self.q_cmd, self.dq_ref, kp, kd, self.effort_limit)
        if self.builds:
            self._start_built()

    def _feedforward(self):
        """The torque that accelerates the arm along its reference, M(q_ref) ddq_ref, added through the position
        servo as an offset of the command (kp dq = torque): without it a quick path lags by centimetres and rings
        after it stops (the joints are soft PD servos). The reference acceleration is low-passed and bounded: a
        command that starts or stops changes the reference velocity in one tick."""
        ddq = (self.dq_ref - self.dq_ref_prev) / self.period
        self.dq_ref_prev = self.dq_ref.copy()
        self.ddq_now = np.clip(ddq, -FF_MAX_ACC, FF_MAX_ACC)
        self.ddq_ref += FF_FILTER * (self.ddq_now - self.ddq_ref)
        arm_js = [n for a in self.arms for n in self.groups[a]]
        if not self.feedforward or not np.any(np.abs(self.ddq_ref) > 1e-3):
            return
        tau = self.kin.inertia({n: float(self.q_ref[self.jidx[n]]) for n in arm_js},
                               {n: float(self.ddq_ref[self.jidx[n]]) for n in arm_js})
        for n, t in tau.items():
            i = self.jidx[n]
            self.q_cmd[i] += t / max(self.kp[i], 1e-3)

    def _settle_bias(self):
        """The settled-error correction (desc["settle_bias"]). Nothing integrates the PD servos' error, and the error
        that i2rt's gravity feed-forward leaves is steady and depends on the pose (on the real YAM 0.7-1.4 deg on
        joints 2 and 3: the hand 7-16 mm high at the table, 8-14 mm low in the air). Once an arm's reference has been
        still for BIAS_WAIT_S, a bias on its joint commands integrates q_ref - q (at rate, within max_rad per joint),
        so the measured joints settle on the reference. Each joint's bias stops changing for the rest of that settle
        once the joint crosses its target or rests within BIAS_DEADBAND of it: friction holds a joint anywhere in a
        band about its target, and integrating inside it makes the joint stick, then jump past (a deadband and a
        hold, as control_toolbox's Pid error_deadband). While the reference moves it decays (BIAS_MOVE_S: friction
        turns around with the motion); it is held, still applied, while the gripper closes or holds something, and in
        a move_at or a move with a contact condition (pressing is not an offset). It bleeds to zero under compliance, a fault, a park, and on a joint clipped or
        saturated at the last tick. A bias the joints do not follow (BIAS_FOLLOW) is a push on something, not an
        offset: it bleeds until the arm's next command. Where the fence has a floor, it never lowers a measured
        gripper below it."""
        if self.settle is None:
            return
        rate, cap, speed = float(self.settle["rate"]), float(self.settle["max_rad"]), float(self.settle["speed"])
        bleed = math.exp(-self.period / BIAS_BLEED_S)
        parked = self.parking or self.parked is not None
        running = [c for c in self.commands if c.status == "running"]
        if self.q_rest is None:
            self.q_rest = self.m.q.copy()
        moving = np.abs(self.m.q - self.q_rest) > BIAS_REST_RAD
        self.q_rest += (self.m.q - self.q_rest) * min(1.0, self.period / BIAS_REST_S)
        for a in self.arms:
            idx = self.arm_cols[a]
            cmds = [c for c in running if a in c.arms]
            low, deep = self._below_floor(a)
            off = (parked or self.fault is not None or a in self.bias_off or a in self.bias_blocked or deep
                   or any(c.park or a in c.compliance for c in cmds))
            if off or np.abs(self.dq_ref[idx]).max() > BIAS_REF_STILL:
                self.bias_still[a] = self.t                  # a new settle: every joint may change again
                self.bias_sign[idx] = 0.0
                self.bias_frozen[idx] = False
                if not off:
                    self.bias[idx] *= math.exp(-self.period / BIAS_MOVE_S)
            if off or self.t - self.bias_still.get(a, self.t) < BIAS_WAIT_S or np.abs(self.m.dq[idx]).max() > speed \
                    or self._gripper_busy(a, running) or any(c.kind == "move_at" or self._seeks_contact(c) for c in cmds):
                self.bias_probe.pop(a, None)
                if off:
                    self.bias[idx] *= bleed
            else:
                e = self.q_ref[idx] - self.m.q[idx]
                sign = self.bias_sign[idx]
                sign[sign == 0.0] = np.sign(e[sign == 0.0])
                self.bias_sign[idx] = sign
                frozen = self.bias_frozen[idx] | (np.sign(e) == -sign) \
                    | ((np.abs(e) < BIAS_DEADBAND[:len(idx)]) & ~moving[idx])
                self.bias_frozen[idx] = frozen
                db = rate * self.period * np.clip(e, -BIAS_ERR_RAD, BIAS_ERR_RAD) * ~frozen
                if not (low and float(self.J_meas[a][2] @ db) < 0.0):         # at the floor: never further down
                    self.bias[idx] = np.clip(self.bias[idx] + db, -cap, cap)
                if not frozen.all():
                    self._bias_follows(a, idx)
            self.bias_lag[idx] += (self.bias[idx] - self.bias_lag[idx]) * min(1.0, self.period / BIAS_LAG_S)
            self.q_cmd[idx] += self.bias[idx]

    def _bias_reset(self):
        """The settle bias at zero, each joint free to change again once its reference has been still BIAS_WAIT_S."""
        self.bias[:] = 0.0
        self.bias_sign[:] = 0.0
        self.bias_frozen[:] = False
        self.bias_still = {a: self.t for a in self.arms}

    def _bias_follows(self, a, idx):
        """A free arm's joints follow a change of its bias one for one, a servo lag behind (q - q_ref grows by it); a
        blocked one's do not. Checked each time the lagged bias has moved BIAS_PROBE_RAD since the last check."""
        e, b = self.m.q[idx] - self.q_ref[idx], self.bias_lag[idx].copy()
        probe = self.bias_probe.get(a)
        if probe is None:
            self.bias_probe[a] = (b, e, 0)
            return
        d = b - probe[0]
        n2 = float(d @ d)
        if n2 < BIAS_PROBE_RAD ** 2:
            return
        failed = probe[2] + 1 if float((e - probe[1]) @ d) < BIAS_FOLLOW * n2 else 0
        self.bias_probe[a] = (b, e, failed)
        if failed >= 2:
            self.bias_blocked.add(a)
            self.bias_probe.pop(a, None)
            self._event("warning", f"{a} arm: its joints do not follow the settle bias (pressing on something?): it "
                                   "is off until the arm's next command", arm=a)

    def _below_floor(self, a) -> tuple:
        """Whether arm a's measured TCP is below its fence box's floor, or its gripper below gripper_floor (low), and
        by more than the references may go (deep)."""
        if a not in self.fence:
            return False, False
        under = float(self.fence[a][0][2] - self.T[a][2, 3])
        if self.floor is not None:
            under = max(under, self.floor_z - self.floor.lowest(a, self.T[a]))
        return under > 0.0, under > FLOOR_REF_TOL

    def _gripper_busy(self, a, running) -> bool:
        """Arm a's gripper is closing, or closed on something (what it holds or presses on is not an offset)."""
        if self.gripper_state[a].get("stalled"):
            return True
        return any(c.kind == "gripper" and c.spec["name"] == a and not getattr(c, "opening", False) for c in running)

    @staticmethod
    def _seeks_contact(cmd: Command) -> bool:
        """A command whose until= or on= reads force, torque, effort, deflection or contact (a guarded move)."""
        if not hasattr(cmd, "_contact"):
            nodes = list((cmd.spec.get("until") or {}).values()) + [c for c, _ in (cmd.spec.get("on") or {}).values()]
            found, todo = False, nodes
            while todo and not found:
                n = todo.pop()
                if isinstance(n, dict):
                    found = n.get("m") in CONTACT_METRICS
                    todo += list(n.values())
                elif isinstance(n, (list, tuple)):
                    todo += list(n)
            cmd._contact = found
        return cmd._contact

    def _ctx(self) -> TickContext:
        return TickContext(t=self.t, T=self.T, v=self.v,
                           wrench={a: self.wrench(a) for a in self.arms},
                           q={n: float(self.m.q[i]) for i, n in enumerate(self.joints)},
                           q_ref={n: float(self.q_ref[i]) for i, n in enumerate(self.joints)},
                           dq={n: float(self.m.dq[i]) for i, n in enumerate(self.joints)},
                           tau_ext=self.tau_ext, contact=self.contact,
                           grippers={a: dict(self.gripper_state[a]) for a in self.arms},
                           signals=self.signals)

    def _step_reference(self, cmd: Command, m):
        t_rel = self.t - cmd.t0
        if cmd.kind == "gripper":
            name = cmd.spec["name"]
            fi = self.jidx[self.finger[name]]
            cmd.gref.dt = self.period
            self.q_ref[fi] = cmd.gref.step(t_rel, m.q[fi], m.dq[fi])
            self.dq_ref[fi] = 0.0
            return
        if cmd.ref is not None:
            js, gen = cmd.ref
            if hasattr(gen, "dt"):                       # a generator that steps: by the real period
                gen.dt = self.period
            q, dq = gen.step(t_rel)
            idx = [self.jidx[n] for n in js]
            self.q_ref[idx] = np.clip(q, self.lo_cmd[idx], self.hi_cmd[idx])
            self.dq_ref[idx] = dq
            for a in cmd.arms:
                self.T_nom[a] = self.kin.fk(a, self._arm(self.q_ref, a))
                self.T_des[a] = None
        for a, gen in cmd.cart.items():
            if hasattr(gen, "dt"):                           # one that integrates (a twist): over the real period
                gen.dt = self.period
            T_des, v_des = gen.step(t_rel)
            self.T_des[a] = (T_des, cmd.tcp[a])
            qa = self._arm(self.q_ref, a)
            tcp = cmd.tcp[a]
            idx = [self.jidx[n] for n in self.groups[a]]
            # predict with the joints' current velocity, then correct by damped least squares: the damping then only
            # shaves the change, not the whole motion (a moving reference neither trails nor dips when handed over)
            q_pred = np.clip(qa + self._arm(self.dq_ref, a) * self.period, self.lo_cmd[idx], self.hi_cmd[idx])
            T_pred, J = self.kin.fk_jac(a, q_pred, tcp)
            dq = (q_pred - qa) + self.kin.diff_ik(a, q_pred, pose_err(T_des, T_pred), tcp, J=J)
            # too fast for a joint's limit: slow the whole step, keeping its direction (clipping joints one by one
            # bent the path: a quick sideways move drifted 8 cm upward)
            over = np.max(np.abs(dq) / (self._arm(self.vcap, a) * self.period))
            if over > 1.0:
                dq = dq / over
            q_new = np.clip(qa + dq, self.lo_cmd[idx], self.hi_cmd[idx])
            self._set_arm(self.q_ref, a, q_new)
            self._set_arm(self.dq_ref, a, (q_new - qa) / self.period)
            self.T_nom[a] = self.kin.fk(a, q_new)
            T_new = self.T_nom[a] if tcp is None else self.T_nom[a] @ tcp
            resid = np.linalg.norm(T_des[:3, 3] - T_new[:3, 3])         # (position only, as before)
            cmd.track_bad = cmd.track_bad + 1 if resid > 0.01 else 0
            if cmd.track_bad > 50:
                self._finish(cmd, "aborted", "cannot_track")
                self._hold_measured(cmd)
                return

    def _comply(self, cmd: Command, arm: str, c: dict):
        """Admittance about the nominal reference: yield like a spring of stiffness K on the requested axes, plus an
        implicit force target (the reference pushed in by F/K, never more than max_penetration)."""
        frame = c.get("frame", "tool")
        if frame == "tool":
            Rc = self.T[arm][:3, :3]
        elif frame == "world":
            Rc = np.eye(3)
        else:
            Rc = _T(frame)[:3, :3]
        F_ext = -self.wrench(arm)                                           # environment on the robot
        f = np.concatenate([Rc.T @ F_ext[:3], Rc.T @ F_ext[3:]])
        names = ["x", "y", "z", "rx", "ry", "rz"]
        K = dict(c.get("stiffness") or {})
        push = np.zeros(6)
        t_rel = self.t - cmd.t0
        ramp = min(1.0, t_rel / max(float(c.get("ramp", 0.3)), 1e-3))
        pen = float(c.get("max_penetration", 0.03))
        qa = self._arm(self.q_ref, arm)
        J = self.kin.jac(arm, qa)
        kp = np.maximum(self._arm(self.kp, arm), 1e-3)
        C = J @ np.diag(1.0 / kp) @ J.T                                     # the PD servo's own Cartesian compliance
        trim = self.comp_trim.setdefault(arm, np.zeros(6))
        for ax, F in (c.get("force") or {}).items():
            sign = -1.0 if ax.startswith("-") else 1.0
            name = ax.lstrip("+-")
            k = float(K.setdefault(name, 500.0 if name in ("x", "y", "z") else 20.0))
            i = names.index(name)
            u = Rc[:, i % 3]
            c_pd = float(u @ (C[:3, :3] if i < 3 else C[3:, 3:]) @ u)
            # implicit force target: the two springs in series (the servo and the requested stiffness) give F at
            # p = F * (c_servo + 1/K); a slow trim on the measured force corrects the model, within +-30%
            p_model = float(F) * (c_pd + 1.0 / k)
            if self.contact[arm]:
                f_meas = -sign * f[i]                                       # force applied along the signed axis
                trim[i] = float(np.clip(trim[i] + 0.8 * DT * (float(F) - f_meas) / max(float(F), 1e-6),
                                        -0.3, 0.3))
            p = float(np.clip(p_model * (1.0 + trim[i]), 0.0, pen if i < 3 else 0.5))
            push[i] = sign * p * ramp
        target = np.zeros(6)
        maxdev = c.get("max_deviation") or {}
        for name, k in K.items():
            i = names.index(name)
            o = f[i] / float(k) + push[i]
            cap = float(maxdev.get(name, 0.05 if i < 3 else 0.5))
            target[i] = np.clip(o, -cap, cap)
        off = self.comp_offset[arm]
        off += (target - off) * min(1.0, DT / 0.03)
        d_world = np.concatenate([Rc @ off[:3], Rc @ off[3:]])
        qa = self._arm(self.q_ref, arm)
        dq = self.kin.diff_ik(arm, qa, d_world)
        self._set_arm(self.q_cmd, arm, qa + dq)
        mf = c.get("max_force")
        if mf is not None and np.linalg.norm(self.wrench(arm)[:3]) > float(mf):
            self._latch_fault(f"{arm}: estimated force {np.linalg.norm(self.wrench(arm)[:3]):.1f} N > max_force {mf} N")

    def _guards(self, m):
        idx = [self.jidx[n] for a in self.arms for n in self.groups[a]]
        q_want = self.q_cmd.copy()
        self.q_cmd[idx] = np.clip(self.q_cmd[idx], m.q[idx] - JOINT_CLIP_RAD, m.q[idx] + JOINT_CLIP_RAD)
        self.q_cmd = np.clip(self.q_cmd, self.lo_cmd, self.hi_cmd)
        # the settle bias bleeds on an arm whose command was clipped or whose motor saturates (no windup)
        sat_j = np.abs(m.tau) > EFFORT_STOP_FRACTION * self.effort_limit
        self.bias_off = {a for a in self.arms if np.any(self.q_cmd[self.arm_cols[a]] != q_want[self.arm_cols[a]])
                         or np.any(sat_j[self.arm_cols[a]])}
        for a in self.arms:                       # a gripper pushing far over its limit is jammed: let it go limp
            fi = self.jidx[self.finger[a]]
            over = abs(m.tau[fi]) > GRIP_OVERLOAD * self.desc["effort_limit"][self.finger[a]]
            self.grip_over[a] = self.grip_over[a] + 1 if over and not self.grip_relaxed[a] else 0
            if self.grip_over[a] * DT >= GRIP_OVERLOAD_S:
                self._relax_gripper(a, abs(float(m.tau[fi])))
        sat = np.abs(m.tau[idx]) > EFFORT_STOP_FRACTION * self.effort_limit[idx]
        self.sat_ticks = self.sat_ticks + 1 if sat.any() else 0
        if self.sat_ticks * DT >= EFFORT_STOP_S and not self.fault:
            j = [self.joints[i] for i, s_ in zip(idx, sat) if s_]
            why = f"joint effort above {int(EFFORT_STOP_FRACTION * 100)}% of the motor limit on {j}"
            if not (self.parking or any(c.park and not c.finished for c in self.commands)):
                self._latch_fault(why)
            elif (self.sat_ticks - 1) * DT < EFFORT_STOP_S:
                # the guard stops programs; a park is the operator's way out and goes on (the clip above
                # still bounds the push to kp x JOINT_CLIP_RAD), or the arms would drop when the motors time out
                self._event("warning", f"{why}: parking on")

    def _relax_gripper(self, arm: str, effort: float):
        fi = self.jidx[self.finger[arm]]
        self.grip_relaxed[arm], self.grip_over[arm] = True, 0
        for c in self.commands:
            if not c.finished and f"{arm}_gripper" in c.groups:
                self._finish(c, "aborted", "gripper_overload")
        self.q_ref[fi], self.dq_ref[fi] = self.m.q[fi], 0.0
        self._event("fault", f"{arm} gripper pushed {effort:.0f} N for {GRIP_OVERLOAD_S:.1f} s (jammed or blocked): "
                             "relaxed, no stiffness until its next command", arm=arm, gripper_relaxed=True)

    def _conditions(self, cmd: Command, ctx: TickContext):
        ctx.t0 = cmd.t0
        ctx.T0 = cmd.T0
        ctx.progress = self._progress(cmd)
        for name, (cond, action) in cmd.on.items():
            if name not in cmd.on_fired and cond(ctx):
                cmd.on_fired.add(name)
                self._safe_action(action, cmd, False, name)
        for name, cond in cmd.until.items():
            if cond(ctx):
                self._finish(cmd, "stopped", name)
                then = cmd.spec.get("then", "hold")
                if isinstance(then, dict) and "op" not in then:
                    then = then.get(name, "hold")
                self._safe_action(then, cmd, True, name)
                return

    def _safe_action(self, action, cmd: Command, chain: bool, name: str):
        """An on=/then= command that cannot start (it is refused as it fires) stops the command that fired it, which
        ends 'aborted' with the refusal as its reason; the arm holds. It must not stop the tick: in lockstep the
        commands are built and started here (the world waits); in real time a worker builds them while a placeholder
        holds their place and joints, the arm holds its reference meanwhile, and a later tick starts them."""
        try:
            specs = self._action(action, cmd)
            parent = cmd if chain else None
            if specs and self.b.lockstep:
                for s in specs:
                    c = self.submit(s, parent=parent)
                    parent = c if chain else None
                    if not chain:
                        cmd.spawned.append(c)
            elif specs:
                self._build_later(specs, parent, cmd, name)
        except (InvalidCommand, Busy, Fault, ValueError) as e:
            self._refused(cmd, name, e)

    def _refused(self, cmd: Command, name: str, e: Exception):
        reason = f"{name}: {type(e).__name__}: {e}"
        if cmd.finished:                     # a then= that ends the chain it should have continued: say so
            cmd.status = "aborted"
            cmd.result.update(status="aborted", reason=reason)
            self._event("finish", f"{cmd.kind} {'+'.join(cmd.groups)} aborted: {reason}", id=cmd.id,
                        op=cmd.kind, status="aborted", reason=reason)
        else:
            self._finish(cmd, "aborted", reason)
        self._hold_measured(cmd)

    def _build_later(self, specs: list, parent: Command | None, cmd: Command, name: str):
        """Real time: build an action's commands on a worker thread. The placeholder is a command 'building' on their
        joint groups, chained where they go: stop(), a fault, a newer command on those joints or the client leaving
        end it as they end any command, and then the built commands are dropped."""
        if self.fault:
            self._refuse()
        groups, arms = [], []
        for s in specs:
            g, a = self._groups_of(s)
            groups += [x for x in g if x not in groups]
            arms += [x for x in a if x not in arms]
        ph = Command(specs[0], groups, arms)
        ph.status = "building"
        self.commands.append(ph)
        if parent is not None:
            parent.next = ph
        else:
            cmd.spawned.append(ph)
        job = _Build(ph, parent, cmd, name, specs)
        self.builds.append(job)

        def build():
            try:
                job.cmds = [self.prepare(dict(s)) for s in job.specs]
            except Exception as e:                    # noqa: BLE001 -- refused (or a bug): the tick reports it
                if not isinstance(e, (InvalidCommand, Busy, Fault, ValueError)):
                    traceback.print_exc()
                job.error = e
            job.ready = True
        threading.Thread(target=build, daemon=True).start()

    def _start_built(self):
        """Start the commands whose build has finished, in their placeholder's place (between ticks, as a client's
        command starts). A build refused stops the command that fired it, as a refusal at once would have."""
        for job in [j for j in self.builds if j.ready]:
            self.builds.remove(job)
            ph = job.placeholder
            if ph.status != "building":               # ended while it was built: drop what was built
                continue
            self.commands.remove(ph)
            ph.status = "replaced"
            if job.parent is not None and job.parent.next is ph:
                job.parent.next = None
            try:
                if job.error is not None:
                    raise job.error
                parent = job.parent
                for c in job.cmds:
                    c = self.commit(c, parent=parent)
                    parent = c if job.parent is not None else None
                    if job.parent is None:
                        job.cmd.spawned.append(c)
            except Exception as e:                    # noqa: BLE001
                self._refused(job.cmd, job.name, e)

    def _action(self, action, cmd: Command) -> list:
        """Do what an action does at once (hold, signal, back_off's hold) and return the specs of the commands it
        starts, in order (a list of several starts them together, each chained to the one before)."""
        if action is None or action == "hold":
            self._hold_measured(cmd)
            return []
        if isinstance(action, (list, tuple)) and action and action[0] == "back_off":
            dist = float(action[1])
            specs = []
            for a in cmd.arms:
                d = self.T_nom[a][:3, 3] - cmd.T0[a][:3, 3]
                if np.linalg.norm(d) < 1e-4:
                    d = -self.v[a]
                u = -d / (np.linalg.norm(d) + 1e-12)
                self._hold_measured(cmd)
                T1 = self.T[a].copy()
                T1[:3, 3] += u * dist
                specs.append({"op": "move_to", "ee": a, "target": T1, "line": True, "duration": max(0.25, dist / 0.04)})
            return specs
        if isinstance(action, (list, tuple)) and action and action[0] == "signal":
            self.signals[str(action[1])] = float(action[2]) if len(action) > 2 else 1.0
            return []
        if isinstance(action, list):
            return list(action)
        if isinstance(action, dict) and "op" in action:
            return [action]
        raise InvalidCommand(f"unknown action {action!r}")

    def _progress(self, cmd: Command) -> float:
        if cmd.kind == "gripper":
            return cmd.gref.progress
        ps = []
        if cmd.ref is not None:
            ps.append(cmd.ref[1].progress)
        ps += [g.progress for g in cmd.cart.values()]
        return min(ps) if ps else 0.0

    def _completion(self, cmd: Command):
        s = cmd.spec
        if s.get("timeout") is not None and self.t - cmd.t0 > float(s["timeout"]):
            self._finish(cmd, "timeout", "timeout")
            self._hold_measured(cmd)
            return
        if cmd.kind == "gripper":
            if cmd.gref.done:
                # closed on nothing: asked to go below the width an empty close ends at, the jaws reached it or ended
                # there (on the real linear_4310 2.8-3.1 mm, the hold clip's closed end, which can read as stalled)
                empty = self.desc["gripper_empty_width_m"] + EMPTY_GRIP_TOL_M
                g = self.gripper_state[s["name"]]
                g["empty"] = bool(not getattr(cmd, "opening", False) and 2.0 * cmd.gripper_target < empty
                                  and (cmd.gref.reason == "reached" or g["width"] <= empty))
                g["stalled"] = 1.0 if cmd.gref.reason == "stalled" and not g["empty"] else 0.0
                # (an opening that stalls -- something keeps the jaws from opening -- is not a grip: its reference
                # stays at the width asked for, pushing out by stiffness x gap; set inside, it closed the jaws hard)
                if cmd.gref.reason == "stalled" and not getattr(cmd, "opening", False):
                    # hold with the grip's force: the reference just inside the object by force / stiffness (as i2rt's
                    # force limiter sets the target once the jaws clog), not at the closed end it ran to, so the
                    # squeeze is the force asked for whatever the object's width, and an opening moves at once
                    fi = self.jidx[self.finger[s["name"]]]
                    self.q_ref[fi] = float(self.m.q[fi]) - self.effort_limit[fi] / max(float(self.kp[fi]), 1e-6)
                self._done(cmd, cmd.gref.reason)
            return
        gens = ([cmd.ref[1]] if cmd.ref is not None else []) + list(cmd.cart.values())
        if gens and all(g.done for g in gens):
            reason = next((getattr(g, "reason", None) for g in gens if getattr(g, "reason", None)), "reached")
            if reason in ("max_travel", "fence", "duration", "joint_limit"):
                if reason in ("max_travel", "fence"):
                    self._finish(cmd, "done", reason)
                    self._hold_measured(cmd)
                self._done(cmd, reason)
                return
            settled = all(np.linalg.norm(self.T_nom[a][:3, 3] - self.T[a][:3, 3]) < SETTLE_POS_M
                          and np.abs(self._arm(self.m.dq, a)).max() < SETTLE_SPEED for a in cmd.arms) \
                if cmd.arms else True
            if settled or self.t - cmd.t0 > getattr(cmd, "_t_done", self.t - cmd.t0) + 0.5:
                self._done(cmd, reason)
            elif not hasattr(cmd, "_t_done"):
                cmd._t_done = self.t - cmd.t0
        if cmd.kind == "hold":
            self._done(cmd, "held")

    def _done(self, cmd: Command, reason: str):
        """A command that ended as asked (arrived, travelled max_travel, ...) starts its then= command(s), as one whose
        condition fired does: then= is what comes next. A dict of them picks by the reason ('reached', 'stalled',
        'max_travel', or a condition's name); 'hold' and back_off are reactions to a condition and stay with until=."""
        self._finish(cmd, "done", reason)
        then = cmd.spec.get("then", "hold")
        if isinstance(then, dict) and "op" not in then:
            then = then.get(reason)
        commands = isinstance(then, dict) or (isinstance(then, list) and then and isinstance(then[0], dict))
        if commands or (isinstance(then, (list, tuple)) and then and then[0] == "signal"):
            self._safe_action(then, cmd, True, reason)

    def _hold_measured(self, cmd: Command):
        for g in cmd.groups:
            for n in self.groups[g]:
                i = self.jidx[n]
                if g in self.arms:
                    self.q_ref[i] = self.m.q[i]
                    self.T_des[g] = None
                self.dq_ref[i] = 0.0
                self.ddq_now[i] = 0.0
        for a in cmd.arms:
            self.T_nom[a] = self.kin.fk(a, self._arm(self.q_ref, a))
            self.comp_offset[a][:] = 0.0

    def _finish(self, cmd: Command, status: str, reason: str):
        if cmd.finished:
            return
        if cmd.status == "running":
            if status != "replaced":                  # a replacing command starts from the current velocity
                for g in cmd.groups:
                    for n in self.groups[g]:
                        self.dq_ref[self.jidx[n]] = 0.0
                        self.ddq_now[self.jidx[n]] = 0.0
                    if g in self.arms:
                        self.T_des[g] = None
            for a in cmd.compliance:
                # the reference takes over where the compliance put the command, so ending a press (or yielding)
                # is not a step; the next command moves on from there
                self._set_arm(self.q_ref, a, self._arm(self.q_cmd - self.bias, a))
                self.T_nom[a] = self.kin.fk(a, self._arm(self.q_ref, a))
                self.comp_offset[a][:] = 0.0
                if status != "replaced":                # still pressing: not an offset (the settle bias waits for
                    self.bias_blocked.add(a)            # the arm's next command)
        cmd.status = status
        cmd.result = {"id": cmd.id, "op": cmd.kind, "status": status, "reason": reason, "t": self.t,
                      "duration": self.t - (cmd.t0 if cmd.t0 is not None else self.t), "at": self.snapshot(cmd)}
        if cmd.kind == "gripper":
            cmd.result["width"] = self.gripper_state[cmd.spec["name"]]["width"]
            cmd.result["empty"] = bool(self.gripper_state[cmd.spec["name"]]["empty"])
        if cmd.spec.get("free") is not None:                    # the full pose(s) free axes settled on
            if hasattr(cmd, "resolved"):
                cmd.result["goal"] = cmd.resolved
            elif hasattr(cmd, "pose_goal"):
                cmd.result["goal"] = {a: T.tolist() for a, T in cmd.pose_goal.items()}
        if status != "replaced":                      # a servo loop replaces its command every call: not news
            self._event("finish", f"{cmd.kind} {'+'.join(cmd.groups)} {status}: {reason}", id=cmd.id, op=cmd.kind,
                        status=status, reason=reason)

    def _latch_fault(self, reason: str, code: str = "fault", **extra):
        self.fault = {"reason": reason, "t": self.t, **extra}
        for c in self.commands:
            if not c.finished:
                self._finish(c, "aborted", code)
        for a in self.arms:
            idx = [self.jidx[n] for n in self.groups[a]]
            self.q_ref[idx] = self.m.q[idx]
            self.dq_ref[idx] = 0.0
            self.ddq_now[idx] = 0.0
            self.T_des[a] = None
            self.comp_offset[a][:] = 0.0
        # held where they are with the settle bias, which bleeds from here (_settle_bias)
        self.q_cmd = np.clip(self.q_ref + self.bias, self.lo_cmd, self.hi_cmd)
        if self.fence:
            self._fence_rearm()
        self._event("fault", reason, **extra)

    def operator_stop(self, soft: bool = False):
        """The operator's stop: every command is aborted and new ones are refused until the operator releases it
        (a program's recover() cannot). Freeze holds the arms stiff at the measured joints; soft lets them float."""
        kind = "soft" if soft else "freeze"
        self._latch_fault(f"operator stop ({kind})", code="operator_stop", operator=True, soft=soft)

    def operator_park(self):
        """The operator's park, before its moves are submitted (park=True): every program command is stopped where
        the arms are, and new ones are refused until the operator releases them (a program's recover() cannot)."""
        for c in self.commands:
            if not c.finished and not c.park:
                self._finish(c, "stopped", "operator park")
                self._hold_measured(c)
        if self.parked is None:
            self.parked = {"reason": "the operator parked the arms", "t": self.t, "operator": True}
            self._event("park", "the operator parked the arms: programs are refused until RELEASE")

    def recover(self, operator: bool = False):
        """Clear a latched Fault; the arms hold stiff where they are. Only the operator clears an operator stop, and
        the operator's park latch."""
        if self.fault and self.fault.get("operator") and not operator:
            raise Fault(f"{self.fault['reason']}: only the operator can release it (the dashboard)")
        if operator and self.parked is not None:
            self.parked = None
            self._event("recover", "released: the operator's park")
        if self.fault is None:
            return
        reason = self.fault["reason"]
        for a in self.arms:
            idx = [self.jidx[n] for n in self.groups[a]]
            self.q_ref[idx] = self.m.q[idx]
            self.dq_ref[idx] = 0.0
            self.T_nom[a] = self.kin.fk(a, self._arm(self.q_ref, a))
            self.comp_offset[a][:] = 0.0
        self.fault = None
        self.sat_ticks = 0                      # the effort guard counts its EFFORT_STOP_S again from here
        if self.fence:
            self._fence_rearm()
        self._event("recover", f"released: {reason}")

    def stop(self):
        for c in self.commands:
            if not c.finished:
                self._finish(c, "stopped", "stop")
                self._hold_measured(c)

    def _event(self, kind: str, text: str, **info):
        if self.listener is not None:
            self.listener(kind, text, t=self.t, **info)
