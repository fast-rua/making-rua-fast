"""External joint torque without torque sensors: a generalized-momentum observer (De Luca & Mattone, 2003).

The arm's dynamics, with the joint torque the motors report (tau_m) and what the environment adds (tau_ext):

    M(q) qdd + C(q, qd) qd + g(q) + tau_f(qd) = tau_m + tau_ext,     p = M(q) qd,
    dp/dt = tau_m + tau_ext + C(q, qd)^T qd - g(q) - tau_f(qd)

The observer integrates everything but tau_ext and compares with the momentum the model says the arm has:

    r = K (p - p0 - integral of (tau_m + C^T qd - g - tau_f + r) dt)      ->      dr/dt = K (tau_ext - r)

so r follows the external torque with a first-order lag 1/K, without differentiating velocities. The static estimate
(g - tau_m) is the same thing with qd = qdd = 0: while the arm moves it reads the torque that accelerates the links,
the Coriolis torques and the joint friction as external force; this one does not, to the accuracy of the model.

Model: M and C from the MuJoCo model of the robot (CAD inertias) plus each joint's reflected rotor inertia
(armature); g from the robot's own gravity model when the backend reports the feed-forward it applies (i2rt on the
real YAM, with its per-joint gravity factors), else from MuJoCo; joint friction tau_f = coulomb * tanh(qd / v0) +
viscous * qd. The backend supplies armature and friction (`backend.dynamics`); identifying them on the real arm is
what makes the estimate hold while it moves.
"""
from __future__ import annotations

import mujoco
import numpy as np

V0 = 0.02            # rad/s: the friction sign is smoothed over this speed


class MomentumObserver:
    def __init__(self, model: mujoco.MjModel, arm_joints: dict, all_joints: list, dynamics: dict | None = None,
                 gain: float = 40.0):
        self.m = model
        self.d = mujoco.MjData(model)
        self.gain = gain
        self.arms = list(arm_joints)
        self.jidx = {n: i for i, n in enumerate(all_joints)}               # the driver's joint order
        self.idx = {a: np.array([self.jidx[n] for n in arm_joints[a]]) for a in self.arms}
        self.dof = {a: np.array([model.jnt_dofadr[model.joint(n).id] for n in arm_joints[a]]) for a in self.arms}
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in all_joints])
        self.vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in all_joints])
        dyn = dynamics or {}
        n = len(next(iter(arm_joints.values())))
        self.armature = np.asarray(dyn.get("armature", np.zeros(n)), float)
        self.coulomb = np.asarray(dyn.get("coulomb", np.zeros(n)), float)
        self.viscous = np.asarray(dyn.get("viscous", np.zeros(n)), float)
        self.r = {a: np.zeros(n) for a in self.arms}
        self.integral = {a: np.zeros(n) for a in self.arms}
        self.p0 = None
        self.M_prev = None
        self.M = np.zeros((model.nv, model.nv))
        self.bias = np.zeros(model.nv)                 # C qd + g of the last update (full model, dof order)
        self._make_M = getattr(mujoco, "mj_makeM", mujoco.mj_crb)   # mj_crb + tendon armature (MuJoCo >= 3.3)

    def _dynamics(self, q, dq):
        """M (full), C qd and g of the model at (q, qd)."""
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        d.qvel[:] = 0
        d.qvel[self.vadr] = dq
        # what mj_fwdPosition does for M and for mj_rne, without its collision detection and constraints (50 of its
        # 51 us on the cell, every tick): positions, the composite inertias, the joint-space inertia
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self._make_M(m, d)
        mujoco.mj_fullM(m, d, self.M)                                          # MuJoCo >= 3.3 signature
        mujoco.mj_comVel(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)                                         # C qd + g
        d.qvel[:] = 0
        mujoco.mj_comVel(m, d)
        g = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, g)                                            # g
        self.bias = bias
        return self.M.copy(), bias - g, g

    def update(self, q, dq, tau_m, dt: float, g_robot=None) -> dict:
        """One tick. q, dq, tau_m (and g_robot, the robot's own gravity torque if it reports one): driver order.
        Returns arm -> external joint torque (environment on the robot), the arm's joint order."""
        q, dq, tau_m = np.asarray(q, float), np.asarray(dq, float), np.asarray(tau_m, float)
        M, Cqd, g = self._dynamics(q, dq)
        Mdot = (M - self.M_prev) / dt if self.M_prev is not None else np.zeros_like(M)
        self.M_prev = M
        out = {}
        for a in self.arms:
            i, k = self.idx[a], self.dof[a]
            Ma = M[np.ix_(k, k)] + np.diag(self.armature)
            qd = dq[i]
            p = Ma @ qd
            if self.p0 is None or a not in self.p0:
                self.p0 = dict(self.p0 or {}, **{a: p.copy()})
            CTqd = Mdot[np.ix_(k, k)] @ qd - Cqd[k]                           # C^T qd = Mdot qd - C qd
            ga = np.asarray(g_robot, float)[i] if g_robot is not None else g[k]
            tau_f = self.coulomb * np.tanh(qd / V0) + self.viscous * qd
            self.integral[a] += (tau_m[i] + CTqd - ga - tau_f + self.r[a]) * dt
            self.r[a] = self.gain * (p - self.p0[a] - self.integral[a])
            out[a] = self.r[a].copy()
        return out
