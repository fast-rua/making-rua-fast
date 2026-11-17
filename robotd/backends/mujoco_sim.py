"""MuJoCo backend for the YAM station scene: lockstep physics, PD servos with run-time gains, gravity compensation.

The privileged parts of the simulation (object poses, contacts) are never sent to the agent: they stay in this
process, readable only by the evaluator (`object_pose`, `watch_log`) and by tests (`set_object_pose`).

realtime=True runs the simulation against the wall clock, as the real robot runs: the driver ticks it from its
control thread and advance() sleeps to the next 4 ms period, so the policy's perception and thinking cost robot time.
Cameras render on one thread that owns the GL context (an EGL context is current on one thread only), so any client
thread (a program, the dashboard) can ask for frames; the scene is posed from the live state under a lock and
rendered outside it, so a render never stalls the physics.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robotd.backends.base import Backend, Measured
from robotd.robots.yam_station import scene as S

STICK_SPEED = 0.002             # rad/s: a joint with SceneConfig.joint_stiction slower than this is stuck (breakaway)


class MujocoYamBackend(Backend):
    name = "mujoco_yam_station"
    lockstep = True
    arm_force_trusted = True        # ~0.5 N from the joint torques of an exact model
    gripper_empty_width_m = 0.0     # closed on nothing the jaws meet

    def __init__(self, cfg: S.SceneConfig | None = None, initial_q: dict | None = None, realtime: bool = False):
        self.cfg = cfg or S.SceneConfig()
        self.model, self.data = S.build(self.cfg)
        m = self.model
        self.joints = []
        self.act_ids = []
        for arm in S.ARMS:
            for name in S.ARM_JOINTS[arm]:
                self.joints.append(name)
                self.act_ids.append(m.actuator(f"{name}_motor").id)
            self.joints.append(S.FINGER_JOINTS[arm][0])
            self.act_ids.append(m.actuator(f"{arm}_gripper_motor").id)
        self.act_ids = np.array(self.act_ids)
        # the gripper motors drive joint7 with joint8 coupled to it: their force is two pads' squeeze, and the driver
        # speaks per pad (gains, effort limit and the effort read back are scaled by two here)
        self._fingers = np.array([i for i, n in enumerate(self.joints) if n in {S.FINGER_JOINTS[a][0] for a in S.ARMS}])
        self._fq = {arm: [m.jnt_qposadr[m.joint(n).id] for n in S.FINGER_JOINTS[arm]] for arm in S.ARMS}
        jid = [m.joint(n).id for n in self.joints]
        self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
        self.dadr = np.array([m.jnt_dofadr[i] for i in jid])
        # gravity compensation acts on every robot dof (the follower finger too), as the robot's firmware would
        robot_joints = [m.joint(i).name for i in range(m.njnt)
                        if m.joint(i).name.startswith(("left_joint", "right_joint"))]
        self.comp_dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in robot_joints])
        # tests: a deliberately wrong compensation on the arm joints (SceneConfig.gravity_comp_scale,
        # joint_torque_bias), the fingers' left exact
        arm_js = {n for a in S.ARMS for n in S.ARM_JOINTS[a]}
        self.comp_scale = np.array([self.cfg.gravity_comp_scale if n in arm_js else 1.0 for n in robot_joints])
        self.comp_extra = np.array([float(self.cfg.joint_torque_bias.get(n, 0.0)) for n in robot_joints])
        bad = (set(self.cfg.joint_torque_bias) | set(self.cfg.joint_stiction)) - set(robot_joints)
        if bad:
            raise ValueError(f"joint_torque_bias / joint_stiction: unknown joints {sorted(bad)}")
        # tests: stick-slip (SceneConfig.joint_stiction), MuJoCo's frictionloss switched between the breakaway and
        # the sliding level by the joint's speed
        st = self.cfg.joint_stiction
        self.stick_dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in st], int)
        self.stick_fs = np.array([float(v[0]) for v in st.values()])
        self.stick_fk = np.array([float(v[1]) for v in st.values()])
        q0 = dict(self.cfg.initial_q or {})
        q0.update(initial_q or {})
        for arm in S.ARMS:
            for n, v in zip(S.ARM_JOINTS[arm], S.READY_Q[arm]):
                q0.setdefault(n, float(v))
            if S.FINGER_JOINTS[arm][0] in q0:            # the twin finger follows joint7
                q0.setdefault(S.FINGER_JOINTS[arm][1], q0[S.FINGER_JOINTS[arm][0]])
        for n, v in q0.items():
            self.data.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        # objects held where they start until let go (Obj.release_t / handover); the others start with Obj.vel
        self.holds = {}
        for o in self.cfg.objects:
            if o.fixed:
                continue
            if o.release_t is not None or o.handover:
                self.holds[o.name] = (m.equality(f"{o.name}_hold").id, o)
            else:
                self._set_velocity(o.name, o.vel)
        self._grip_pads = {a: {g: tip for tip in ("tip_left", "tip_right") for g in range(m.ngeom)    # pad -> its tip
                               if m.geom(g).name.startswith(f"{a}_{tip}_pad")} for a in S.ARMS}
        self._obj_geoms = {o.name: {m.geom(p.name).id for p in S.object_parts(o)} for o in self.cfg.objects}
        self._obj_mask = {}                                        # object -> geom id -> is one of its geoms
        for n, g in self._obj_geoms.items():
            self._obj_mask[n] = np.zeros(m.ngeom, bool)
            self._obj_mask[n][sorted(g)] = True
        # watched objects: each contact that begins (time, the other geom, position, velocity) and the path
        self.watch_log = {n: {"contacts": [], "path": []} for n in self.cfg.watch}
        self._touching = {n: set() for n in self.cfg.watch}
        # scene noise, each process on its own random stream seeded by the scene, drawn in step order (so the same
        # commands replay the same run): velocity kicks as contacts end, and gusts of wind
        self._noisy = {o.name: o.bounce_noise for o in self.cfg.objects if o.bounce_noise and not o.fixed}
        seed = self.cfg.noise_seed if self.cfg.noise_seed is not None else 0
        self._kick_rng = np.random.default_rng([seed, 1])
        self._wind_rng = np.random.default_rng([seed, 2])
        self._in_contact = {n: False for n in self._noisy}
        self._last_kick = {n: -1.0 for n in self._noisy}
        self.noise_log = {"kicks": []}
        self._touch = {}
        self._gust = np.zeros(3)
        if self.cfg.wind:
            w = self.cfg.wind
            m.opt.wind[:] = w.get("mean", (0, 0, 0))
            self._wind_mean = np.asarray(w.get("mean", (0, 0, 0)), float)
            self._wind_a = float(np.exp(-m.opt.timestep / float(w.get("tau", 1.0))))
            self._wind_b = np.array([w.get("gust", 0.0), w.get("gust", 0.0), w.get("gust_z", 0.0)], float) * \
                np.sqrt(1 - self._wind_a ** 2)
            self._wind_draws, self._wind_k = np.zeros((0, 3)), 0
        self._geom_name = [m.geom(i).name or m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]
        for c in self.cfg.cloths:
            if c.initial is not None:
                self.set_cloth_vertices(c.name, c.initial)
        # cloth held by a closed gripper (see scene.CLOTH_GRIP_SLOTS)
        self.grip_slots = {a: [i for i in range(m.neq) if m.equality(i).name.startswith(f"{a}_cloth_grip_")]
                           for a in S.ARMS}
        self.grip_held = {a: False for a in S.ARMS}
        self.grip_q0 = {a: 0.0 for a in S.ARMS}
        self._grip_cmd = {a: 0.0 for a in S.ARMS}            # the driver's own finger target, before the cloth stop
        self._grip_act = {a: m.actuator(f"{a}_gripper_motor").id for a in S.ARMS}
        self._finger_q = {a: m.jnt_qposadr[m.joint(S.FINGER_JOINTS[a][0]).id] for a in S.ARMS}
        self._finger_v = {a: m.jnt_dofadr[m.joint(S.FINGER_JOINTS[a][0]).id] for a in S.ARMS}
        self._grip_force = {a: S.GRIPPER_FORCE_LIMIT_N for a in S.ARMS}   # the grip's squeeze per pad (driver's limit)
        self._cloth_pads = {a: {m.geom(f"{a}_{tip}_cloth").id: tip for tip in ("tip_left", "tip_right")}
                            for a in S.ARMS}
        self._tip_body = {a: {tip: m.body(f"{a}_{tip}").id for tip in ("tip_left", "tip_right")} for a in S.ARMS}
        if self.cfg.cloths:
            f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{self.cfg.cloths[0].name}_flex")
            self._flex = f
            # a pinch's friction (MuJoCo takes the larger of the pair's), and each fingertip's end in its body frame
            self._cloth_mu = max(float(m.geom_friction[m.geom("left_tip_left_cloth").id, 0]),
                                 float(m.flex_friction[f, 0]))
            self._cloth_radius = float(m.flex_radius[f])
            self._fingertip = {n: p - [0.0, 0.0, S.FINGERTIP_BELOW_FACET] for n, p in S._facet_points().items()}
            self._cloth_bodies = m.flex_vertbodyid[m.flex_vertadr[f]:m.flex_vertadr[f] + m.flex_vertnum[f]].copy()
        self._ropes = {r.name: self._rope_setup(r) for r in self.cfg.ropes}
        mujoco.mj_forward(m, self.data)
        self.data.ctrl[self.act_ids] = self.data.qpos[self.qadr]
        self._renderer = None
        self._kin_model = None
        self.lock = threading.Lock()                   # physics vs camera posing (realtime)
        self._render_lock = threading.Lock()           # starting the render thread
        self._render_jobs = None
        self._render_thread = None
        self.realtime = bool(realtime)
        self.lockstep = not self.realtime
        self.timing = {"ticks": 0, "late_ticks": 0, "max_late_s": 0.0, "resyncs": 0, "physics_s": 0.0}
        self._next_tick = None
        # what the model behind the driver's force estimate needs beyond the kinematic model: the scene's own values
        self.dynamics = {"armature": list(S.ARMATURE), "coulomb": [0.02] * 6, "viscous": [0.05] * 6}

    # ---------------------------------------------------------------- control
    def read(self) -> Measured:
        d = self.data
        tau = d.actuator_force[self.act_ids] + d.qfrc_applied[self.dadr]
        tau[self._fingers] *= 0.5                                    # per pad
        width = {arm: float(d.qpos[self._fq[arm][0]] + d.qpos[self._fq[arm][1]]) for arm in S.ARMS}
        return Measured(t=float(d.time), q=d.qpos[self.qadr].copy(), dq=d.qvel[self.dadr].copy(), tau=tau.copy(),
                        gripper_width=width)

    def write(self, q_cmd, dq_cmd, kp, kd, effort_limit) -> None:
        m, d = self.model, self.data
        kp = np.maximum(np.array(kp, float), 1e-6)
        kd = np.array(kd, float)
        lim = np.array(effort_limit, float)
        kp[self._fingers] *= 2.0                                     # per pad -> the motor's
        kd[self._fingers] *= 2.0
        lim[self._fingers] *= 2.0
        m.actuator_gainprm[self.act_ids, 0] = kp
        m.actuator_biasprm[self.act_ids, 1] = -kp
        m.actuator_biasprm[self.act_ids, 2] = -kd
        m.actuator_forcerange[self.act_ids, 0] = -lim
        m.actuator_forcerange[self.act_ids, 1] = lim
        # force = kp*(ctrl - q) - kd*dq with ctrl = q_cmd + kd/kp*dq_cmd  ==  kp*(q_cmd - q) + kd*(dq_cmd - dq)
        d.ctrl[self.act_ids] = np.asarray(q_cmd, float) + kd / kp * np.asarray(dq_cmd, float)
        for k, arm in enumerate(getattr(self, "grip_held", {})):
            act = self._grip_act[arm]
            self._grip_force[arm] = float(effort_limit[self._fingers[k]])            # per pad
            self._grip_cmd[arm] = float(d.ctrl[act])
            if self.grip_held[arm]:                    # cloth fills the gap: the jaws stop where they closed on it
                d.ctrl[act] = max(d.ctrl[act], self.grip_q0[arm] - 0.0005)

    def advance(self, dt: float) -> None:
        m, d = self.model, self.data
        t_wall = time.perf_counter()
        with self.lock:
            for _ in range(max(1, int(round(dt / m.opt.timestep)))):
                d.qfrc_applied[self.comp_dofs] = d.qfrc_bias[self.comp_dofs] * self.comp_scale + self.comp_extra
                if len(self.stick_dofs):
                    m.dof_frictionloss[self.stick_dofs] = np.where(np.abs(d.qvel[self.stick_dofs]) < STICK_SPEED,
                                                                   self.stick_fs, self.stick_fk)
                for rope in self._ropes.values():
                    self._rope_couples(rope)
                if self.cfg.wind:
                    self._blow()
                mujoco.mj_step(m, d)
                if self.holds:
                    self._let_go()
                self._touch = {}                       # each object's contacts this step, computed once
                if self.watch_log:
                    self._watch()
                if self._noisy:
                    self._kick()
            if self.cfg.cloths:
                for arm in S.ARMS:
                    self._cloth_grip(arm)
        if self.realtime:
            self._pace(dt, t_wall)

    def _pace(self, dt: float, t_wall: float) -> None:
        """Wait for the next control period (wall clock), as a real robot's loop does; count late periods."""
        now = time.perf_counter()
        tm = self.timing
        tm["ticks"] += 1
        tm["physics_s"] += now - t_wall
        if self._next_tick is None:
            self._next_tick = now
        self._next_tick += dt
        late = now - self._next_tick
        if late > 0:
            tm["late_ticks"] += 1
            tm["max_late_s"] = max(tm["max_late_s"], late)
            if late > 0.1:                          # far behind (a stall): resynchronise instead of racing
                tm["resyncs"] += 1
                self._next_tick = now
        else:
            time.sleep(-late)

    def _set_velocity(self, name: str, vel) -> None:
        j = self.model.joint(f"{name}_free")
        a = self.model.jnt_dofadr[j.id]
        self.data.qvel[a:a + 6] = np.asarray(vel, float)

    def _let_go(self) -> None:
        """Release held objects: at their release time, or (a hand-over) once both pads of a gripper press on them
        (2 N each, summed over the pad's boxes)."""
        m, d = self.model, self.data
        for name, (eq, o) in list(self.holds.items()):
            go = o.release_t is not None and d.time >= o.release_t
            if o.handover and not go:
                geoms = self._obj_geoms[name]
                for arm, pads in self._grip_pads.items():
                    press, f = {"tip_left": 0.0, "tip_right": 0.0}, np.zeros(6)
                    for i in range(d.ncon):
                        c = d.contact[i]
                        g1, g2 = int(c.geom1), int(c.geom2)
                        tip = pads.get(g1 if g2 in geoms else g2 if g1 in geoms else -1)
                        if tip is not None:
                            mujoco.mj_contactForce(m, d, i, f)
                            press[tip] += f[0]
                    go = go or min(press.values()) > 2.0
            if go:
                d.eq_active[eq] = 0
                self._set_velocity(name, o.vel)
                del self.holds[name]

    def _blow(self) -> None:
        """Wind: the mean plus Ornstein-Uhlenbeck gusts (time constant tau, steady sigma per axis), one step. MuJoCo's
        fluid model takes each body's velocity relative to opt.wind. (Normals are drawn 4096 at a time, used in
        order: the same stream as one by one, a fraction of the cost.)"""
        if self._wind_k >= len(self._wind_draws):
            self._wind_draws, self._wind_k = self._wind_rng.standard_normal((4096, 3)), 0
        self._gust = self._wind_a * self._gust + self._wind_b * self._wind_draws[self._wind_k]
        self._wind_k += 1
        self.model.opt.wind[:] = self._wind_mean + self._gust

    def _kick(self) -> None:
        """An object with bounce_noise, as its contacts end (once per 20 ms), leaves with its speed scaled by
        1 + N(0, s) and its direction turned by N(0, a) deg about a random axis across it (both clipped at 3 sigma)."""
        m, d = self.model, self.data
        for name, (s_sig, a_sig) in self._noisy.items():
            on = bool(self._touching_geoms(name))
            if self._in_contact[name] and not on and d.time - self._last_kick[name] > 0.02:
                j = m.joint(f"{name}_free")
                a = m.jnt_dofadr[j.id]
                v = d.qvel[a:a + 3].copy()
                speed = float(np.linalg.norm(v))
                if speed > 1e-6:
                    ds = float(np.clip(self._kick_rng.normal(0.0, s_sig), -3 * s_sig, 3 * s_sig))
                    ang = float(np.radians(np.clip(self._kick_rng.normal(0.0, a_sig), -3 * a_sig, 3 * a_sig)))
                    phi = float(self._kick_rng.uniform(0, 2 * np.pi))
                    u = v / speed
                    e1 = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
                    e1 /= np.linalg.norm(e1)
                    axis = np.cos(phi) * e1 + np.sin(phi) * np.cross(u, e1)
                    v2 = (1 + ds) * Rotation.from_rotvec(ang * axis).apply(v)
                    d.qvel[a:a + 3] = v2
                    self.noise_log["kicks"].append({"t": round(float(d.time), 4), "object": name,
                                                    "speed": round(ds, 4), "turn_deg": round(float(np.degrees(ang)), 3),
                                                    "v_before": np.round(v, 3).tolist()})
                    self._last_kick[name] = float(d.time)
            self._in_contact[name] = on

    # ---------------------------------------------------------------- ropes (scene.Rope)
    def _rope_setup(self, r: S.Rope) -> dict:
        """A rope's link bodies, its joints' rest rotations, stiffness and damping, and its free links' dof addresses."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        bodies = np.array([m.body(f"{r.name}_{k:03d}").id for k in range(len(r.nodes()) - 1)])
        R = d.xmat[bodies].reshape(-1, 3, 3)
        rest = (np.einsum("kji,kjl->kil", R[:-1], R[1:]) if r.rest == "initial"
                else np.broadcast_to(np.eye(3), (len(bodies) - 1, 3, 3)).copy())
        lengths = np.array([2 * m.geom_size[m.body_geomadr[b], 1] for b in bodies])
        K = np.array([S.rope_stiffness(r, 0.5 * (a + b)) for a, b in zip(lengths[:-1], lengths[1:])])
        free = np.array([m.body_jntnum[b] > 0 for b in bodies])
        dof = np.array([m.jnt_dofadr[m.body_jntadr[b]] if f else -1 for b, f in zip(bodies, free)])
        # relative angular damping: a quarter of critical for the stiffest axis at the raised inertia (explicit,
        # like the couples: an interior link damped from both sides stays inside the stable bound)
        c = K.max(1) * m.opt.timestep
        return {"rope": r, "bodies": bodies, "rest": rest, "K": K, "c": c, "free": free, "dof": dof}

    def _rope_couples(self, rope: dict) -> None:
        """The rod's bending and twist couples at the current pose, and its joints' damping: at joint k the relative
        rotation R_k^T R_k+1 off its rest value, theta = log(rest^T R_k^T R_k+1) (link k+1's frame), gives the couple
        -K theta on link k+1 and the opposite on link k (world frame, as xfrc_applied)."""
        if rope["rope"].young <= 0:
            return
        m, d = self.model, self.data
        b = rope["bodies"]
        R = d.xmat[b].reshape(-1, 3, 3)
        E = np.einsum("kji,kjl->kil", rope["rest"], np.einsum("kji,kjl->kil", R[:-1], R[1:]))
        theta = Rotation.from_matrix(E).as_rotvec()
        tau = -np.einsum("kij,kj->ki", R[1:], rope["K"] * theta)               # on link k+1, world
        w = np.zeros((len(b), 3))                                              # angular velocity, world
        f = rope["free"]
        w[f] = np.einsum("kij,kj->ki", R[f], d.qvel[rope["dof"][f][:, None] + np.arange(3, 6)])
        tau -= rope["c"][:, None] * (w[1:] - w[:-1])
        out = np.zeros((len(b), 3))
        out[1:] += tau
        out[:-1] -= tau
        d.xfrc_applied[b, 3:] = out

    def rope_links(self, name: str) -> np.ndarray:
        """A rope's link poses (n x 7: x y z qw qx qy qz, world), in material order."""
        b = self._ropes[name]["bodies"]
        return np.concatenate([self.data.xpos[b], self.data.xquat[b]], 1)

    def rope_nodes(self, name: str) -> np.ndarray:
        """A rope's centre line: its link end points (n + 1 x 3, world), from the material start."""
        rope = self._ropes[name]
        b = rope["bodies"]
        m, d = self.model, self.data
        half = m.geom_size[m.body_geomadr[b], 1]
        x = d.xmat[b].reshape(-1, 3, 3)[:, :, 0]
        return np.concatenate([d.xpos[b] - half[:, None] * x, d.xpos[b[-1:]] + half[-1:, None] * x[-1:]])

    def rope_speeds(self, name: str) -> np.ndarray:
        """The speed of each link's centre (n, m/s)."""
        vel = np.zeros(6)
        out = []
        for b in self._ropes[name]["bodies"]:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            out.append(np.linalg.norm(vel[3:]))
        return np.array(out)

    def set_rope_links(self, name: str, poses) -> None:
        """Put a rope's links at the given poses (n x 7), at rest (replays and tests)."""
        m, d = self.model, self.data
        rope = self._ropes[name]
        for b, p, f in zip(rope["bodies"], np.asarray(poses, float), rope["free"]):
            if f:
                j = m.body_jntadr[b]
                d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7] = p
                d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0.0
        mujoco.mj_forward(m, d)

    def privileged_noise(self) -> dict:
        """The wind now (recording only)."""
        return {"wind": self.model.opt.wind.copy()} if self.cfg.wind else {}

    def _touching_geoms(self, name) -> set:
        """The other geoms an object's geoms touch now (the contact arrays at once: this runs every physics step)."""
        if name in self._touch:
            return self._touch[name]
        d = self.data
        out = set()
        if d.ncon:
            pair = d.contact.geom[:d.ncon]
            mine = self._obj_mask[name]
            in1, in2 = mine[pair[:, 0]], mine[pair[:, 1]]
            out = set(pair[in1 & ~in2, 1].tolist()) | set(pair[in2 & ~in1, 0].tolist())
        self._touch[name] = out
        return out

    def _watch(self) -> None:
        d = self.data
        for name, log in self.watch_log.items():
            b = d.body(name)
            now = self._touching_geoms(name)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, b.id, vel, 0)
            for g in now - self._touching[name]:
                log["contacts"].append((float(d.time), self._geom_name[g], b.xpos.copy(), vel[3:].copy()))
            self._touching[name] = now
            log["path"].append((float(d.time), *b.xpos))

    def cloth_held(self) -> dict:
        """Per arm, the cloth vertex indices welded to its gripper (developer view)."""
        out = {}
        for arm, slots in getattr(self, "grip_slots", {}).items():
            idx = {int(b): i for i, b in enumerate(self._cloth_bodies)} if self.cfg.cloths else {}
            out[arm] = [idx[int(self.model.eq_obj2id[e])] for e in slots if self.data.eq_active[e]
                        and int(self.model.eq_obj2id[e]) in idx]
        return out

    def _tip_frame(self, arm: str, tip: str):
        """A fingertip's body id and its frame at the fingertip's end (world rotation R and origin p0): x across the
        finger, y toward the other jaw for the left tip, z up the finger."""
        d = self.data
        b = self._tip_body[arm][tip]
        R = d.xmat[b].reshape(3, 3)
        return b, R, d.xpos[b] + R @ self._fingertip[f"{arm}_{tip}"]

    def _cloth_grip(self, arm: str) -> None:
        """A pinch of cloth. MuJoCo's flex contacts alone let pinched cloth slide out of the jaws (noslip does not act
        on flex), so once the closing jaws have stalled on cloth that both finger faces touch (scene.on_finger_face: not
        below the fingertip, not beside it, not in its window), the cloth vertices between the two faces are welded,
        where they are, to the nearer finger (stiff connect constraints, scene.CLOTH_GRIP_SOLIMP). A weld holds only
        what the pinch holds: a welded vertex lets go once it is no longer between the two finger faces, beside or
        below them by more than the cloth's radius; all of them let go when the jaws are told to open, or when the
        welds pull harder than the pinch's friction holds (2 mu x the squeeze per pad). The jaws, still closing, then
        pinch again wherever the cloth has slid to."""
        m, d = self.model, self.data
        q, v = d.qpos[self._finger_q[arm]], d.qvel[self._finger_v[arm]]
        target = self._grip_cmd[arm]
        rad = self._cloth_radius
        _, R, p0 = self._tip_frame(arm, "tip_left")           # left tip frame at its fingertip: y toward the other jaw
        if self.grip_held[arm]:
            slots = self.grip_slots[arm]
            release = target > self.grip_q0[arm] + 0.002     # told to open past the width it closed on
            if not release:
                rows = np.nonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
                f2 = np.zeros(m.neq)
                np.add.at(f2, d.efc_id[rows], d.efc_force[rows] ** 2)
                release = float(np.sqrt(f2[slots]).sum()) > 2.0 * self._cloth_mu * self._grip_force[arm]
            if not release:
                held = [e for e in slots if d.eq_active[e]]
                loc = (d.xpos[m.eq_obj2id[held]] - p0) @ R
                out = ~((loc[:, 1] > -rad) & (loc[:, 1] < 2 * q + rad) & S.on_finger_face(loc[:, 0], loc[:, 2], rad))
                for e in np.asarray(held)[out]:
                    d.eq_active[e] = 0
                release = bool(out.all())
            if release:
                for e in slots:
                    d.eq_active[e] = 0
                self.grip_held[arm] = False
            return
        # stalled while closing, on something: closed on nothing the jaws meet at 0, on one layer of cloth they stop
        # at 4-6 mm; below 2.4 mm nothing is between the pads (closed jaws pressed into cloth must not pick it up)
        if not (target < q - 0.001 and abs(v) < 0.002 and q > 0.0012):
            return
        geom, flex, pos = d.contact.geom[:d.ncon], d.contact.flex[:d.ncon], d.contact.pos[:d.ncon]
        for g, tip in self._cloth_pads[arm].items():
            _, Rt, pt = self._tip_frame(arm, tip)
            rows = np.nonzero(((geom[:, 0] == g) & (flex[:, 1] == self._flex)) |
                              ((geom[:, 1] == g) & (flex[:, 0] == self._flex)))[0]
            loc = (pos[rows] - pt) @ Rt
            if not S.on_finger_face(loc[:, 0], loc[:, 2]).any():
                return                                        # not pinched: the cloth must touch both finger faces
        P = d.xpos[self._cloth_bodies]
        loc = (P - p0) @ R
        idx = np.nonzero((loc[:, 1] > -rad) & (loc[:, 1] < 2 * q + rad) & S.on_finger_face(loc[:, 0], loc[:, 2]))[0]
        if len(idx) == 0:
            return
        centre = p0 + R @ np.array([0.0, q, 0.008])
        idx = idx[np.argsort(np.linalg.norm(P[idx] - centre, axis=1))][:len(self.grip_slots[arm])]
        for e, i in zip(self.grip_slots[arm], idx):
            b, Rt, _ = self._tip_frame(arm, "tip_left" if loc[i, 1] < q else "tip_right")
            m.eq_obj1id[e] = b
            m.eq_obj2id[e] = self._cloth_bodies[i]
            m.eq_data[e, :3] = Rt.T @ (P[i] - d.xpos[b])
            m.eq_data[e, 3:6] = 0.0
            d.eq_active[e] = 1
        self.grip_held[arm] = True
        self.grip_q0[arm] = float(q)

    # ---------------------------------------------------------------- sensing
    def kinematics_model(self):
        if self._kin_model is None:
            self._kin_model = S.kinematic_model()
        return self._kin_model

    def cameras(self, names=None, depth: bool = True) -> dict:
        names = list(names or S.CAMERAS)
        return self._on_render_thread(self._render_cameras, names, depth)

    def _on_render_thread(self, fn, *args):
        """Run fn on the one thread that owns the renderer's EGL context (current on one thread only)."""
        with self._render_lock:
            if self._render_thread is None:
                self._render_jobs = queue.Queue()
                self._render_thread = threading.Thread(target=self._render_loop, daemon=True, name="mujoco-render")
                self._render_thread.start()
        reply = queue.Queue(maxsize=1)
        self._render_jobs.put((fn, args, reply))
        out = reply.get()
        if isinstance(out, BaseException):
            raise out
        return out

    def _render_loop(self):
        while True:
            job = self._render_jobs.get()
            if job is None:
                break
            fn, args, reply = job
            try:
                reply.put(fn(*args))
            except Exception as e:                      # noqa: BLE001 -- goes back to the caller
                reply.put(e)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _rend(self):
        if self._renderer is None:
            self._opt = mujoco.MjvOption()
            self._opt.sitegroup[:] = 0                     # sites are markers, not things a camera sees
            # one renderer (one EGL context) for colour and depth: two contexts in one process gave intermittent
            # garbage frames
            self._renderer = mujoco.Renderer(self.model, self.cfg.camera_height, self.cfg.camera_width)
        return self._renderer

    def _render_cameras(self, names, depth):
        m, d = self.model, self.data
        H, W = self.cfg.camera_height, self.cfg.camera_width
        rend = self._rend()
        out = {}
        for name in names:
            cid = m.camera(name).id
            with self.lock:                                # pose the scene from one physics state
                rend.update_scene(d, camera=cid, scene_option=self._opt)
                t = float(d.time)
                T = np.eye(4)
                T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo camera -> optical
                T[:3, 3] = d.cam_xpos[cid]
            frame = {"rgb": rend.render().copy(), "t": t}
            if depth:                                      # the same posed scene, rendered as depth
                rend.enable_depth_rendering()
                frame["depth"] = rend.render().astype(np.float32)
                rend.disable_depth_rendering()
            fy = 0.5 * H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
            frame["K"] = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
            frame["T"] = T
            out[name] = frame
        return out

    def warm_up(self) -> None:
        """Make the renderer (an EGL context, ~0.3 s) now: a real-time driver does it before its clock starts, so the
        first camera frame does not stall a running world."""
        self.cameras([next(iter(S.CAMERAS))], depth=True)

    def render_free(self, lookat=(0.3, 0.0, 0.15), distance=1.0, azimuth=180.0, elevation=-25.0) -> dict:
        """Developer view from a free camera (the driver serves it only with --dev)."""
        return self._on_render_thread(self._render_free, lookat, distance, azimuth, elevation)

    def _render_free(self, lookat, distance, azimuth, elevation) -> dict:
        rend = self._rend()
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        with self.lock:
            rend.update_scene(self.data, camera=cam, scene_option=self._opt)
        return {"rgb": rend.render().copy()}

    def health(self) -> dict:
        return {"sim": True, "realtime": self.realtime, "sim_time": float(self.data.time)}

    # ---------------------------------------------------------------- privileged (evaluator and tests only)
    def object_pose(self, name: str) -> np.ndarray:
        b = self.data.body(name)
        T = np.eye(4)
        T[:3, :3] = b.xmat.reshape(3, 3)
        T[:3, 3] = b.xpos
        return T

    def privileged_poses(self) -> dict:
        """Every scene object's pose (x y z qw qx qy qz), for the recording only."""
        out = {}
        for o in self.cfg.objects:
            b = self.data.body(o.name)
            out[o.name] = np.concatenate([b.xpos, b.xquat])
        return out

    def cloth_vertices(self, name: str) -> np.ndarray:
        """A cloth's vertex positions (N x 3, world), in the order of its Cloth.points."""
        f = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, f"{name}_flex")
        if f < 0:
            raise KeyError(f"no cloth {name!r}")
        a, n = self.model.flex_vertadr[f], self.model.flex_vertnum[f]
        return self.data.flexvert_xpos[a:a + n].copy()

    def set_cloth_vertices(self, name: str, positions) -> None:
        """Move a cloth's vertices (N x 3, world) without changing its rest shape: each vertex is a body with three
        slide joints whose position is the offset from its rest place."""
        m, d = self.model, self.data
        f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{name}_flex")
        P = np.asarray(positions, float)
        for i in range(m.flex_vertnum[f]):
            b = m.flex_vertbodyid[m.flex_vertadr[f] + i]
            rest = m.body_pos[b]
            for k in range(m.body_jntnum[b]):
                j = m.body_jntadr[b] + k
                axis = m.jnt_axis[j]
                d.qpos[m.jnt_qposadr[j]] = float(axis @ (P[i] - rest))
                d.qvel[m.jnt_dofadr[j]] = 0.0
        mujoco.mj_forward(m, d)

    def privileged_deformables(self) -> dict:
        """Cloth vertices (N x 3) and rope link poses (n x 7), by name."""
        return {**{c.name: self.cloth_vertices(c.name) for c in self.cfg.cloths},
                **{r.name: self.rope_links(r.name) for r in self.cfg.ropes}}

    def scene_info(self) -> dict:
        """What a replay viewer needs to draw the scene (recording only)."""
        return {"objects": [{"name": o.name, "shape": o.shape, "size": list(o.size), "rgba": list(o.rgba),
                             "fixed": bool(o.fixed),
                             "parts": [{"name": p.name, "shape": p.shape, "size": list(p.size), "pos": list(p.pos),
                                        "quat": list(p.quat), "rgba": list(p.rgba)} for p in o.parts]}
                            for o in self.cfg.objects],
                "cloths": [{"name": c.name, "triangles": np.asarray(c.triangles).tolist(), "rgba": list(c.rgba)}
                           for c in self.cfg.cloths],
                "ropes": [{"name": r.name, "radius": r.radius, "rgba": list(r.rgba),
                           "link_lengths": (2 * self.model.geom_size[self.model.body_geomadr[self._ropes[r.name]["bodies"]],
                                                                    1]).round(6).tolist()} for r in self.cfg.ropes],
                "noise": self.cfg.noise_info or None,
                "table_top_z": self.cfg.table_top_z, "table_x": list(self.cfg.table_x), "station_to_world_y": S.STATION_TO_WORLD_Y,
                "urdf": S.STATION_URDF, "mjcf": S.STATION_XML, "cameras": dict(S.CAMERAS)}

    def set_object_pose(self, name: str, pos, quat=(1, 0, 0, 0), vel=None) -> None:
        j = self.model.joint(f"{name}_free")
        a = self.model.jnt_qposadr[j.id]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[self.model.jnt_dofadr[j.id]:self.model.jnt_dofadr[j.id] + 6] = 0 if vel is None else vel
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._render_thread is not None:
            self._render_jobs.put(None)
            self._render_thread.join(timeout=5)
