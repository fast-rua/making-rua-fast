"""Forward kinematics, Jacobians, IK and the gravity model of each arm, on a MuJoCo model of the robot.

One private MjData is used for every query, so calls never disturb a simulation. Poses are 4x4 numpy matrices in the
world frame; a `tcp` argument (4x4, in the arm's TCP-site frame) moves the controlled point, e.g. to a tool tip.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


# Rotation vector <-> matrix in plain floats, computed as scipy's Rotation does (through the unit quaternion: chosen by
# the largest of the trace and the diagonal, normalised, w >= 0; the same small-angle series). scipy's per-call
# overhead (~35 us on the cell) is most of a Cartesian command's cost in a 4 ms control tick; these take ~3 us.
def rotvec_from_matrix(Rm) -> np.ndarray:
    """scipy's Rotation.from_matrix(Rm).as_rotvec()."""
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = (map(float, row) for row in Rm[:3, :3])
    tr = m00 + m11 + m22
    dec = (m00, m11, m22, tr)
    k = max(range(4), key=dec.__getitem__)             # the first largest, as numpy's argmax
    if k == 0:
        x, y, z, w = 1.0 - tr + 2.0 * m00, m10 + m01, m20 + m02, m21 - m12
    elif k == 1:
        x, y, z, w = m01 + m10, 1.0 - tr + 2.0 * m11, m21 + m12, m02 - m20
    elif k == 2:
        x, y, z, w = m02 + m20, m12 + m21, 1.0 - tr + 2.0 * m22, m10 - m01
    else:
        x, y, z, w = m21 - m12, m02 - m20, m10 - m01, 1.0 + tr
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), w)
    if angle <= 1e-3:
        a2 = angle * angle
        scale = 2.0 + a2 / 12.0 + 7.0 * a2 * a2 / 2880.0
    else:
        scale = angle / math.sin(angle / 2.0)
    return np.array([scale * x, scale * y, scale * z])


def matrix_from_rotvec(v) -> np.ndarray:
    """scipy's Rotation.from_rotvec(v).as_matrix()."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    angle = math.sqrt(vx * vx + vy * vy + vz * vz)
    if angle <= 1e-3:
        a2 = angle * angle
        scale = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0
    else:
        scale = math.sin(angle / 2.0) / angle
    x, y, z, w = scale * vx, scale * vy, scale * vz, math.cos(angle / 2.0)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array([[x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
                     [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
                     [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2]])


def rotvec_err(R_target: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R_now to R_target."""
    return rotvec_from_matrix(R_target @ R_now.T)


def pose_err(T_target: np.ndarray, T_now: np.ndarray) -> np.ndarray:
    return np.concatenate([T_target[:3, 3] - T_now[:3, 3], rotvec_err(T_target[:3, :3], T_now[:3, :3])])


FREE = (None, "yaw", "yaw180", "rotation")


def _free_rows(free):
    """Which rows of (position; rotation in the target's frame) an IK target constrains."""
    if free in (None, "yaw180"):
        return [0, 1, 2, 3, 4, 5]
    if free == "yaw":
        return [0, 1, 2, 3, 4]
    if free == "rotation":
        return [0, 1, 2]
    raise ValueError(f"free must be one of {FREE[1:]}, got {free!r}")


class Kinematics:
    def __init__(self, model: mujoco.MjModel, arm_joints: dict, tcp_sites: dict, all_joints: list):
        self.m = model
        self.d = mujoco.MjData(model)
        self.arm_joints = arm_joints
        self.tcp_sites = tcp_sites
        self.all_joints = list(all_joints)          # the driver's joint order (arms + finger joints)
        self.qadr = {n: int(model.jnt_qposadr[model.joint(n).id]) for n in self.all_joints}
        self.dadr = {n: int(model.jnt_dofadr[model.joint(n).id]) for n in self.all_joints}
        self.limits = {n: tuple(model.jnt_range[model.joint(n).id]) for n in self.all_joints}
        self.site_id = {a: model.site(s).id for a, s in tcp_sites.items()}
        self.arm_qadr = {a: np.array([self.qadr[n] for n in js]) for a, js in arm_joints.items()}
        self.arm_cols = {a: [self.dadr[n] for n in js] for a, js in arm_joints.items()}

    # ------------------------------------------------------------------ state loading
    def set_q(self, q: dict) -> None:
        for n, v in q.items():
            self.d.qpos[self.qadr[n]] = v

    def set_arm(self, arm: str, q_arm: np.ndarray) -> None:
        self.d.qpos[self.arm_qadr[arm]] = q_arm

    # ------------------------------------------------------------------ queries
    def fk(self, arm: str, q_arm=None, tcp: np.ndarray | None = None) -> np.ndarray:
        if q_arm is not None:
            self.set_arm(arm, q_arm)
        mujoco.mj_kinematics(self.m, self.d)
        sid = self.site_id[arm]
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[sid]
        return T @ tcp if tcp is not None else T

    def jac(self, arm: str, q_arm=None, tcp: np.ndarray | None = None) -> np.ndarray:
        """6 x n Jacobian of the controlled point (linear on top), world-aligned, columns in the arm's joint order."""
        if q_arm is not None:
            self.set_arm(arm, q_arm)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        if tcp is None:
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site_id[arm])
        else:
            T = self.fk(arm, None, tcp)
            body = int(self.m.site_bodyid[self.site_id[arm]])
            mujoco.mj_jac(self.m, self.d, jp, jr, T[:3, 3], body)
        cols = self.arm_cols[arm]
        return np.vstack([jp[:, cols], jr[:, cols]])

    def fk_jac(self, arm: str, q_arm=None, tcp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(fk, jac) from one kinematics pass: what fk(arm, q_arm, tcp) and jac(arm, q_arm, tcp) return."""
        if q_arm is not None:
            self.set_arm(arm, q_arm)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        sid = self.site_id[arm]
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[sid]
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        if tcp is None:
            mujoco.mj_jacSite(self.m, self.d, jp, jr, sid)
        else:
            T = T @ tcp
            mujoco.mj_jac(self.m, self.d, jp, jr, T[:3, 3], int(self.m.site_bodyid[sid]))
        cols = self.arm_cols[arm]
        return T, np.vstack([jp[:, cols], jr[:, cols]])

    def ik(self, arm: str, T_target: np.ndarray, q0: np.ndarray, tcp=None, iters: int = 300,
           tol_pos: float = 2e-4, tol_rot: float = 2e-3, weights=(1.0, 0.6),
           restarts: int = 6, free: str | None = None) -> tuple[np.ndarray, bool, float]:
        """Damped least squares from q0 within joint limits. Returns (q, ok, position error in m).
        free='yaw': the turn about the target's own z axis (the tool's approach axis) is left to the solver;
        free='rotation': only the position counts."""
        names = self.arm_joints[arm]
        lo = np.array([self.limits[n][0] for n in names]); hi = np.array([self.limits[n][1] for n in names])
        rows = _free_rows(free)
        Rt = np.asarray(T_target, float)[:3, :3]
        M = np.eye(6)
        M[3:, 3:] = Rt.T                                # rotation error and rows in the target's frame
        best = None
        rng = np.random.default_rng(0)
        seeds = [np.asarray(q0, float), np.clip(np.asarray(q0, float) * 0.5 + 0.5 * (lo + hi) / 2, lo, hi)]
        seeds += [rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)) for _ in range(restarts)]
        W = np.diag([weights[0]] * 3 + [weights[1]] * 3)[rows][:, rows]
        for seed in seeds:
            q = np.clip(seed.copy(), lo, hi)
            lam = 0.05
            for _ in range(iters):
                T = self.fk(arm, q, tcp)
                e = (M @ pose_err(T_target, T))[rows]
                ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
                if ep < tol_pos and er < tol_rot:
                    return q, True, ep
                J = (M @ self.jac(arm, q, tcp))[rows]
                Jw, ew = W @ J, W @ e
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(len(rows)), ew)
                n = np.linalg.norm(dq)
                if n > 0.3:
                    dq *= 0.3 / n
                q = np.clip(q + dq, lo, hi)
            T = self.fk(arm, q, tcp)
            e = pose_err(T_target, T)
            if best is None or np.linalg.norm(e[:3]) < best[2]:
                best = (q, False, float(np.linalg.norm(e[:3])))
        return best

    def ik_polish(self, arm: str, T_target: np.ndarray, q0: np.ndarray, tcp=None, iters: int = 300,
                  tol_pos: float = 2e-4, tol_rot: float = 2e-3, weights=(1.0, 0.6),
                  free: str | None = None) -> tuple[np.ndarray, bool, float]:
        """Levenberg-Marquardt from q0, ik's closest miss: the damping halves after a step that lowers the weighted
        error and triples after one that does not (undone), where ik's fixed damping stalls against a joint limit or
        a near-singular wrist. Returns (q, ok, position error in m) as ik does."""
        names = self.arm_joints[arm]
        lo = np.array([self.limits[n][0] for n in names]); hi = np.array([self.limits[n][1] for n in names])
        rows = _free_rows(free)
        M = np.eye(6)
        M[3:, 3:] = np.asarray(T_target, float)[:3, :3].T
        W = np.diag([weights[0]] * 3 + [weights[1]] * 3)[rows][:, rows]
        q = np.clip(np.asarray(q0, float), lo, hi)
        e = (M @ pose_err(T_target, self.fk(arm, q, tcp)))[rows]
        cost, lam = float(np.linalg.norm(W @ e)), 0.05
        for _ in range(iters):
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                return q, True, float(np.linalg.norm(e[:3]))
            Jw = W @ (M @ self.jac(arm, q, tcp))[rows]
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(len(rows)), W @ e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            qn = np.clip(q + dq, lo, hi)
            en = (M @ pose_err(T_target, self.fk(arm, qn, tcp)))[rows]
            cn = float(np.linalg.norm(W @ en))
            if cn < cost:
                q, e, cost, lam = qn, en, cn, max(lam * 0.5, 1e-4)
            else:
                lam = min(lam * 3.0, 10.0)
        ok = bool(np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot)
        return q, ok, float(np.linalg.norm(pose_err(T_target, self.fk(arm, q, tcp))[:3]))

    def margin(self, arm: str, q_arm) -> dict:
        """How comfortable a configuration is: the smallest distance to a joint limit (rad, and which joint) and the
        smallest singular value of the position Jacobian (m per rad: near 0 at a straight elbow or a folded wrist)."""
        names = self.arm_joints[arm]
        q = np.asarray(q_arm, float)
        d = np.array([min(v - self.limits[n][0], self.limits[n][1] - v) for n, v in zip(names, q)])
        k = int(np.argmin(d))
        J = self.jac(arm, q)
        return {"limit_rad": float(d[k]), "limit_joint": names[k],
                "singular_m": float(np.linalg.svd(J[:3], compute_uv=False)[-1])}

    def diff_ik(self, arm: str, q_arm: np.ndarray, twist: np.ndarray, tcp=None, lam: float = 0.02,
                J: np.ndarray | None = None) -> np.ndarray:
        """Joint step for a desired small world twist (dx; drot), damped least squares. J: the Jacobian at q_arm if
        the caller has it already (fk_jac)."""
        if J is None:
            J = self.jac(arm, q_arm, tcp)
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), twist)

    def inertia(self, q: dict, ddq: dict) -> dict:
        """M(q) ddq: the torque each driver joint needs to accelerate the arms as ddq asks (rotor inertia included)."""
        self.set_q(q)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_crb(self.m, self.d)
        acc = np.zeros(self.m.nv)
        for n, v in ddq.items():
            acc[self.dadr[n]] = v
        out = np.zeros(self.m.nv)
        mujoco.mj_mulM(self.m, self.d, out, acc)
        return {n: float(out[self.dadr[n]]) for n in ddq}

    def bias(self, q: dict, dq: dict | None = None, ddq: dict | None = None) -> dict:
        """The torque every driver joint needs at state q: gravity, + Coriolis when dq is given, + the arm's own
        inertia M(q) ddq when ddq is given. Without the inertia term an accelerating arm reads as an external force."""
        self.d.qvel[:] = 0
        self.set_q(q)
        if dq:
            for n, v in dq.items():
                self.d.qvel[self.dadr[n]] = v
        mujoco.mj_forward(self.m, self.d)
        out = self.d.qfrc_bias.copy()
        if ddq:
            acc = np.zeros(self.m.nv)
            for n, v in ddq.items():
                acc[self.dadr[n]] = v
            Ma = np.zeros(self.m.nv)
            mujoco.mj_mulM(self.m, self.d, Ma, acc)
            out += Ma
        return {n: float(out[self.dadr[n]]) for n in self.all_joints}
