"""Reference generators, stepped every fast tick. Joint references produce q/dq directly; Cartesian references produce
a desired pose (and velocity) of an arm's controlled point, which the controller turns into joint references by
differential IK at the fast rate.

A command that replaces a moving one starts from the current reference: its position, velocity and acceleration.
JointGoal (Ruckig) takes all three; the timed paths (JointSpline, PosePath, PoseLine) start where the reference is and
add a Blend, a short correction that starts at the velocity and acceleration the new path lacks and dies out to zero
(it never moves the path's end); the velocity references ramp from the current velocity.
"""
from __future__ import annotations

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from robotd.core.kinematics import matrix_from_rotvec, rotvec_from_matrix


class _Slerp:
    """scipy's Slerp(times, rotations)(t) for one t, on matrices: segment k (times[k] < t <= times[k+1]; the first
    for t <= times[0]) gives R_k @ exp(alpha log(R_k^T R_k+1)), alpha = (t - times[k]) / (times[k+1] - times[k])."""

    def __init__(self, times, mats):
        self.t = np.asarray(times, float)
        self.R = [np.asarray(M, float)[:3, :3] for M in mats]
        self.rv = [rotvec_from_matrix(a.T @ b) for a, b in zip(self.R[:-1], self.R[1:])]

    def __call__(self, t: float) -> np.ndarray:
        k = max(int(np.searchsorted(self.t, t)) - 1, 0)
        alpha = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        return self.R[k] @ matrix_from_rotvec(self.rv[k] * alpha)


def _quintic(s: float) -> tuple[float, float]:
    s = min(max(s, 0.0), 1.0)
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5        # rounding can put this a hair outside [0, 1] near the ends
    return min(max(p, 0.0), 1.0), 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4


class JointGoal:
    """Jerk-limited move of some joints to a goal (Ruckig, online): continuous from the current reference state."""

    def __init__(self, q0, dq0, goal, vmax, amax, jmax, dt, duration=None, ddq0=None):
        n = len(goal)
        self.otg = Ruckig(n, dt)
        self.inp, self.out = InputParameter(n), OutputParameter(n)
        self.inp.current_position = list(map(float, q0))
        self.inp.current_velocity = list(map(float, dq0))
        a0 = np.zeros(n) if ddq0 is None else np.clip(np.asarray(ddq0, float), -np.asarray(amax), np.asarray(amax))
        self.inp.current_acceleration = list(map(float, a0))
        self.inp.target_position = list(map(float, goal))
        self.inp.target_velocity = [0.0] * n
        self.inp.max_velocity = list(map(float, vmax))
        self.inp.max_acceleration = list(map(float, amax))
        self.inp.max_jerk = list(map(float, jmax))
        if duration:
            self.inp.minimum_duration = float(duration)
        self.q, self.dq = np.array(q0, float), np.array(dq0, float)
        self.done, self.progress = False, 0.0

    @property
    def dt(self) -> float:
        return self.otg.delta_time

    @dt.setter
    def dt(self, value: float) -> None:
        """The controller sets each tick's real period: Ruckig then steps the trajectory by the time that passed."""
        self.otg.delta_time = float(value)

    def step(self, t_rel: float):
        if self.done:
            return self.q, np.zeros_like(self.q)
        res = self.otg.update(self.inp, self.out)
        if res not in (Result.Working, Result.Finished):
            raise RuntimeError(f"trajectory generator failed: {res}")
        self.q, self.dq = np.array(self.out.new_position), np.array(self.out.new_velocity)
        dur = max(self.out.trajectory.duration, 1e-9)
        self.progress = min(1.0, self.out.time / dur)
        self.out.pass_to_input(self.inp)
        if res == Result.Finished:
            self.done, self.progress = True, 1.0
        return self.q, self.dq


def _blend_basis(s: float):
    """Quintic shapes on [0, 1] that start with unit slope (hv) or unit curvature (ha) and end flat at zero: value,
    first and second derivative of each."""
    s = min(max(s, 0.0), 1.0)
    hv = (s - 6 * s**3 + 8 * s**4 - 3 * s**5, 1 - 18 * s**2 + 32 * s**3 - 15 * s**4, -36 * s + 96 * s**2 - 60 * s**3)
    ha = (0.5 * s**2 - 1.5 * s**3 + 1.5 * s**4 - 0.5 * s**5, s - 4.5 * s**2 + 6 * s**3 - 2.5 * s**4,
          1 - 9 * s + 18 * s**2 - 10 * s**3)
    return hv, ha


HA_PEAK = 0.0157                     # the largest value of the unit-curvature shape ha


def accel_window(da, Tb: float, dev: float) -> float:
    """How long the acceleration part of a blend may last: Tb, or less when the missing acceleration is so large that
    blending it over Tb would bend the path by more than dev (a path that starts with a jump in acceleration, e.g.
    to a waypoint 20 ms away at speed, then takes its own acceleration within a few ticks)."""
    a = float(np.max(np.abs(da))) if np.size(da) else 0.0
    return Tb if a * Tb**2 * HA_PEAK <= dev else max(np.sqrt(dev / (a * HA_PEAK)), 1e-3)


def blend_offset(t: float, dv, da, Tv: float, Ta: float):
    """The correction at t: it starts at velocity dv (dying out by Tv) and acceleration da (by Ta), and is zero with
    its derivatives after both."""
    dv, da = np.asarray(dv, float), np.asarray(da, float)
    c, dc = np.zeros_like(dv), np.zeros_like(dv)
    if t < Tv:
        hv, _ = _blend_basis(t / Tv)
        c, dc = c + dv * Tv * hv[0], dc + dv * hv[1]
    if t < Ta:
        _, ha = _blend_basis(t / Ta)
        c, dc = c + da * Ta**2 * ha[0], dc + da * Ta * ha[1]
    return c, dc


class _Wrapped:
    """Delegates done / progress / reason to the generator it wraps."""
    gen = None

    @property
    def done(self):
        return self.gen.done

    @property
    def progress(self):
        return self.gen.progress

    @property
    def reason(self):
        return getattr(self.gen, "reason", None)


def joint_rates(gen, h: float = 1e-3):
    """A joint generator's velocity and acceleration at its start (by differences; the generator is pure in t)."""
    q = [np.asarray(gen.step(k * h)[0], float) for k in range(3)]
    gen.done, gen.progress = False, 0.0
    return (q[1] - q[0]) / h, (q[2] - 2 * q[1] + q[0]) / h**2


def pose_rates(gen, h: float = 1e-3):
    """A pose generator's twist (m/s, rad/s, world) and its derivative at its start."""
    T = [gen.step(k * h)[0] for k in range(3)]
    gen.done, gen.progress = False, 0.0
    v1 = np.concatenate([(T[1][:3, 3] - T[0][:3, 3]) / h, rotvec_from_matrix(T[1][:3, :3] @ T[0][:3, :3].T) / h])
    v2 = np.concatenate([(T[2][:3, 3] - T[1][:3, 3]) / h, rotvec_from_matrix(T[2][:3, :3] @ T[1][:3, :3].T) / h])
    return v1, (v2 - v1) / h


class JointBlend(_Wrapped):
    """A joint path plus a Blend: starts at the reference's velocity dq0 and acceleration ddq0."""

    def __init__(self, gen, dq0, ddq0, Tb, dev=0.003):
        v, a = joint_rates(gen)
        self.gen, self.dv, self.da = gen, np.asarray(dq0, float) - v, np.asarray(ddq0, float) - a
        self.Tv = max(float(Tb), 1e-3)
        self.Ta = accel_window(self.da, self.Tv, dev)

    def step(self, t: float):
        q, dq = self.gen.step(t)
        c, dc = blend_offset(t, self.dv, self.da, self.Tv, self.Ta)
        return q + c, dq + dc


class PoseBlend(_Wrapped):
    """A pose path plus a Blend: starts at the controlled point's twist v0 and its derivative a0 (world)."""

    def __init__(self, gen, v0, a0, Tb, dev=0.002):
        v, a = pose_rates(gen)
        self.gen, self.dv, self.da = gen, np.asarray(v0, float) - v, np.asarray(a0, float) - a
        self.Tv = max(float(Tb), 1e-3)
        self.Ta = accel_window(self.da, self.Tv, dev)

    def step(self, t: float):
        T, v = self.gen.step(t)
        c, dc = blend_offset(t, self.dv, self.da, self.Tv, self.Ta)
        if not np.any(c):
            return T, v
        out = T.copy()
        out[:3, 3] += c[:3]
        out[:3, :3] = matrix_from_rotvec(c[3:]) @ T[:3, :3]
        return out, v + dc


class JointSpline:
    """Timed joint path through points at times (s from the start), from the current reference q0 and its velocity
    dq0 (so a path that replaces a moving one continues smoothly). Cubic Hermite; velocities at the points are given
    or estimated (Catmull-Rom), zero at the end unless given."""

    def __init__(self, q0, points, times, velocities=None, dq0=None):
        P = np.vstack([np.asarray(q0, float)[None], np.asarray(points, float)])
        Tm = np.concatenate([[0.0], np.asarray(times, float)])
        if np.any(np.diff(Tm) <= 0):
            raise ValueError("times must be increasing and positive")
        if velocities is not None:
            V = np.vstack([np.zeros((1, P.shape[1])), np.asarray(velocities, float).reshape(len(P) - 1, -1)])
        else:
            V = np.zeros_like(P)
            for i in range(1, len(P) - 1):
                V[i] = (P[i + 1] - P[i - 1]) / (Tm[i + 1] - Tm[i - 1])
        if dq0 is not None:
            V[0] = np.asarray(dq0, float)
        self.P, self.T, self.V = P, Tm, V
        self.done, self.progress = False, 0.0

    def step(self, t: float):
        Tm, P, V = self.T, self.P, self.V
        if t >= Tm[-1]:
            self.done, self.progress = True, 1.0
            return P[-1].copy(), np.zeros(P.shape[1])
        i = int(np.searchsorted(Tm, t, side="right") - 1)
        h = Tm[i + 1] - Tm[i]
        s = (t - Tm[i]) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        q = h00 * P[i] + h10 * h * V[i] + h01 * P[i + 1] + h11 * h * V[i + 1]
        dq = (d00 * P[i] + d10 * h * V[i] + d01 * P[i + 1] + d11 * h * V[i + 1]) / h
        self.progress = t / Tm[-1]
        return q, dq


class PosePath:
    """A timed pose path for one arm's controlled point: positions by cubic Hermite, rotations by slerp per segment.
    The first pose is the controlled point's pose when the command starts, moving at v0 (m/s, world). velocities:
    the point's linear velocity at each pose (default: Catmull-Rom estimates, zero at the end)."""

    def __init__(self, T0, poses, times, velocities=None, v0=None):
        Ts = [np.asarray(T0, float)] + [np.asarray(T, float) for T in poses]
        tm = np.concatenate([[0.0], np.asarray(times, float)])
        if np.any(np.diff(tm) <= 0):
            raise ValueError("times must be increasing and positive")
        self.T_end = Ts[-1]
        self.pos = JointSpline(Ts[0][:3, 3], [T[:3, 3] for T in Ts[1:]], tm[1:], velocities, v0)
        self.slerp = _Slerp(tm, Ts)
        self.tm = tm
        self.done, self.progress = False, 0.0

    def step(self, t: float):
        tc = min(max(t, 0.0), self.tm[-1])
        p, v = self.pos.step(tc)
        T = np.eye(4)
        T[:3, :3] = self.slerp(tc)
        T[:3, 3] = p
        self.progress = tc / self.tm[-1]
        self.done = t >= self.tm[-1]
        return T, np.concatenate([v, np.zeros(3)])


class PoseLine:
    """Straight line (position) with slerped rotation from T0 to T1 in `duration` s, quintic time scaling."""

    def __init__(self, T0, T1, duration):
        self.T0, self.T1, self.dur = np.asarray(T0, float), np.asarray(T1, float), max(float(duration), 1e-3)
        self.slerp = _Slerp([0.0, 1.0], [self.T0, self.T1])
        self.done, self.progress = False, 0.0

    def step(self, t: float):
        s, ds = _quintic(t / self.dur)
        T = np.eye(4)
        T[:3, :3] = self.slerp(s)
        T[:3, 3] = self.T0[:3, 3] + s * (self.T1[:3, 3] - self.T0[:3, 3])
        v = ds / self.dur * (self.T1[:3, 3] - self.T0[:3, 3])
        self.progress = min(1.0, t / self.dur)
        self.done = t >= self.dur
        return T, np.concatenate([v, np.zeros(3)])


class TwistRef:
    """Constant twist (world frame, m/s and rad/s) for one arm's controlled point, until a duration or a travel limit
    (then done with reason 'max_travel'/'duration'; travel_reason names a travel limit set by something else, e.g.
    'fence'). It ramps from the point's current twist v0 to the target at `accel` (m/s^2; turning at
    ANGULAR_PER_LINEAR times that in rad/s^2), so a re-sent one continues."""
    ANGULAR_PER_LINEAR = 8.0

    def __init__(self, T0, twist, accel, dt, max_travel=None, duration=None, v0=None, travel_reason="max_travel"):
        self.T = np.asarray(T0, float).copy()
        self.p0 = self.T[:3, 3].copy()
        self.tw = np.asarray(twist, float)
        self.accel, self.dt = float(accel), dt
        self.max_travel, self.duration = max_travel, duration
        self.travel_reason = travel_reason
        self.v = np.zeros(6) if v0 is None else np.asarray(v0, float).copy()
        self.done, self.progress, self.reason = False, 0.0, None

    def step(self, t: float):
        if self.done:
            return self.T, np.zeros(6)
        for sl, a in ((slice(0, 3), self.accel), (slice(3, 6), self.accel * self.ANGULAR_PER_LINEAR)):
            d = self.tw[sl] - self.v[sl]
            n = np.linalg.norm(d)
            self.v[sl] += d if n <= a * self.dt else d * (a * self.dt / n)
        v = self.v.copy()
        self.T[:3, 3] = self.T[:3, 3] + v[:3] * self.dt
        if np.linalg.norm(v[3:]) > 0:
            self.T[:3, :3] = matrix_from_rotvec(v[3:] * self.dt) @ self.T[:3, :3]
        travel = np.linalg.norm(self.T[:3, 3] - self.p0)
        if self.max_travel is not None and travel >= self.max_travel:
            self.done, self.reason = True, self.travel_reason
        if self.duration is not None and t >= self.duration:
            self.done, self.reason = True, "duration"
        if self.max_travel:
            self.progress = min(1.0, travel / self.max_travel)
        return self.T, v


class JointVelocityRef:
    """Joint velocities (rad/s), ramped from the current ones dq0 at `accel` (rad/s^2, along a straight line in
    velocity space), until a joint limit or a duration."""

    def __init__(self, q0, vel, accel, dt, lo, hi, duration=None, dq0=None):
        self.q = np.asarray(q0, float).copy()
        self.vel = np.asarray(vel, float)
        self.accel, self.dt, self.lo, self.hi, self.duration = float(accel), dt, lo, hi, duration
        self.dq = np.zeros_like(self.q) if dq0 is None else np.asarray(dq0, float).copy()
        self.done, self.progress, self.reason = False, 0.0, None

    def step(self, t: float):
        if self.done:
            return self.q, np.zeros_like(self.q)
        d = self.vel - self.dq
        n = np.abs(d).max()
        self.dq = self.dq + (d if n <= self.accel * self.dt else d * (self.accel * self.dt / n))
        dq = self.dq.copy()
        q = np.clip(self.q + dq * self.dt, self.lo, self.hi)
        if np.any((q <= self.lo + 1e-9) & (dq < 0)) or np.any((q >= self.hi - 1e-9) & (dq > 0)):
            self.done, self.reason = True, "joint_limit"
        self.q = q
        if self.duration is not None and t >= self.duration:
            self.done, self.reason = True, "duration"
        return self.q, dq


class GripperRef:
    """Finger slide to a target at a speed; done 'reached' at the target, or 'stalled' when the jaws stop short on
    something. Once the slide is at the target, jaws that stay still for stall_s end it either way: within
    STALL_GAP of the target they reached it, beyond it they hold something (no gap between the two can hang)."""
    REACHED_GAP = 0.0008             # m per finger: at the target
    STALL_GAP = 0.0015               # m per finger: stopped this far short = closed on something

    def __init__(self, q0, target, speed, dt, stall_s=0.06):
        self.q, self.target = float(q0), float(target)
        self.speed, self.dt = abs(float(speed)), dt
        self.need = max(1, int(stall_s / dt))
        self.still = 0
        self.done, self.progress, self.reason = False, 0.0, None
        self.span = abs(self.target - self.q) + 1e-9

    def step(self, t: float, q_meas: float, dq_meas: float):
        if not self.done:
            step = np.clip(self.target - self.q, -self.speed * self.dt, self.speed * self.dt)
            self.q += step
            gap = abs(self.target - q_meas)
            self.progress = min(1.0, 1 - gap / self.span)
            at_target = abs(self.q - self.target) < 1e-9
            if at_target and gap < self.REACHED_GAP:
                self.done, self.reason = True, "reached"
            elif at_target and abs(dq_meas) < 0.002:
                self.still += 1
                if self.still >= self.need:
                    self.done, self.reason = True, "stalled" if gap > self.STALL_GAP else "reached"
            else:
                self.still = 0
        return self.q
