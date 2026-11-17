"""Real bimanual I2RT YAM through the i2rt SDK (v1.4.0): command_joint_state(pos, vel, kp, kd) per arm at 250 Hz.

The motors run the PD (MIT mode) and i2rt adds its gravity feed-forward, so this backend only forwards the driver's
targets and gains. Force is estimated as (i2rt's feed-forward) - (measured joint torque), which in static contact is
the external torque (the YAM has no wrist force sensor).

Environment (from_env):
  YAM_CHANNELS   "left=can_left,right=can_right"   CAN interfaces per arm (as in the cell config)
  YAM_SIM        1 -> i2rt's SimRobot (kinematic: joints teleport to the command; for interface tests only)
  monitor        (robotd --monitor) the motors stay off: joints, temperatures and cameras are read, nothing moves
                 (backends/yam_passive.py)
  YAM_CALIBRATION  the cell's calibration file (ORA's ora.cell-calibration.v1, written by the `ora cell` tools; only
                 read here). Supplies, per arm: gripper limits, encoder zero offsets, base corrections, the recorded
                 rest pose (the park pose), and the fixed cameras' poses and wrist-camera mount corrections.
  YAM_GRIPPER_LIMITS "closed,open" (both arms) or "left=closed,open;right=closed,open" motor positions; overrides
                 the calibration's. With neither, i2rt calibrates each gripper by driving the jaws to both stops.
  YAM_PARK_Q     "q1,...,q6" (both arms) or "left=q1,...;right=q1,...": where the arms go when the driver exits;
                 overrides the calibration's rest pose (default: where the arms were at start-up)
  YAM_CAMERAS    "top=SERIAL,left_wrist=SERIAL,right_wrist=SERIAL" RealSense serials (optional)
  YAM_CALIBRATION_IGNORE "key,..." top-level entries of the calibration to leave out, e.g. wrist_camera_corrections
                 (2026-11-10: the wrist cameras' fingers fit their depth to 1 mm on the CAD mount, 6.5 mm off with the
                 stored correction, which looks like it absorbed joint and link errors of the hand-eye fit)
  YAM_GRAVITY_FACTORS "f1,...,f6" (both arms) or "left=f1,...;right=f1,...": i2rt's gravity_comp_factor per arm joint,
                 overriding its arm config's (measured on the cell by a two-sided hold sweep, 2026-11-10: joint 2
                 needs 1.2, joint 4 0.5 where i2rt's yam config has 1.1 and 1.2; each left a ~1 deg steady offset)
  YAM_DYNAMICS   a json {"armature": [6], "coulomb": [6], "viscous": [6]} for the force estimate's model (identified on
                 the arm); default: i2rt's Coulomb friction for the YAM, the scene's armature, no viscous friction
  ORA_CELL_LOCK  the cell lock file shared with ORA's cell kit (default /tmp/ora-cell.lock)

Joint values in and out of this backend are model angles: encoder reading + the calibration's zero offset.
The grip force of gripper(force=N) is applied through i2rt's gripper force limiter.

Safety notes for a lab robot:
- The cell lock is taken first; the driver refuses to start while another program holds the cell.
- The gripper is never held at its mechanical stops (normalized command kept within 0.03..0.97): holding at a stop
  overheats the motor.
- Motors time out 400 ms after the last command and the arms drop: on exit the driver first moves both arms to the
  park pose and holds there; stop the driver only with the arms supported or parked.
"""
from __future__ import annotations

import fcntl
import functools
import json
import os
import subprocess
import time

import numpy as np

from robotd.backends.base import Backend, Measured
from robotd.robots.yam_station import scene as S

GRIPPER_HOLD = (0.03, 0.97)
PERIOD_S = 0.004
CALIBRATION_SCHEMAS = ("ora.cell-calibration.v1", "ora.yam-cell-calibration.v1")
CAMERA_ALIASES = {"top": ("top", "overhead")}       # API camera name -> calibration roles to look for


def load_calibration(path: str | None) -> dict:
    """ORA's cell calibration, as written by the `ora cell` tools (never written here)."""
    if not path:
        return {}
    with open(os.path.expanduser(path)) as f:
        cal = json.load(f)
    if cal.get("schema") not in CALIBRATION_SCHEMAS:
        raise ValueError(f"{path} is not an ora.cell-calibration.v1 file (schema {cal.get('schema')!r})")
    for key in filter(None, os.environ.get("YAM_CALIBRATION_IGNORE", "").split(",")):
        if cal.pop(key.strip(), None) is not None:
            print(f"robotd: calibration {key.strip()} ignored (YAM_CALIBRATION_IGNORE)", flush=True)
    return cal


def kinematics_model_for(calibration: dict):
    """The station model with the cell's measured base corrections (corrected base = correction @ model, in the
    world frame, which is ORA's yam_station world). Shared by the backend and camerad."""
    import mujoco
    m = S.kinematic_model()
    corrections = calibration.get("arm_base_corrections") or {}
    if corrections:
        # as ORA applies them: each arm's base moves to correction @ CAD in the world, every other arm staying at
        # its CAD pose (the right base is a child of the left one in the model)
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        body = {a: m.body(f"{a}_base").id for a in S.ARMS}
        wanted = {}
        for a, b in body.items():
            cad = np.eye(4)
            cad[:3, :3] = d.xmat[b].reshape(3, 3)
            cad[:3, 3] = d.xpos[b]
            wanted[a] = (_rigid(corrections[a]) if a in corrections else np.eye(4)) @ cad
        for a, b in body.items():
            parent = int(m.body_parentid[b])
            above = next((o for o, ob in body.items() if ob == parent), None)
            if above is not None:
                parent_T = wanted[above]
            else:
                parent_T = np.eye(4)
                parent_T[:3, :3] = d.xmat[parent].reshape(3, 3)
                parent_T[:3, 3] = d.xpos[parent]
            local = np.linalg.inv(parent_T) @ wanted[a]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, local[:3, :3].ravel())
            m.body_pos[b], m.body_quat[b] = local[:3, 3], q
    return m


def camera_pose(name: str, calibration: dict, model, joints) -> np.ndarray:
    """World <- optical (OpenCV: +z forward, +y down): fixed cameras from the cell calibration's cam2world_cv;
    wrist cameras through the arm's kinematics and the calibration's mount correction. model: the kinematic model or
    a callable giving it; joints: {joint name: model angle} or a callable giving it (only wrist cameras need either).
    Raises NotImplementedError for a camera it cannot place."""
    cams = calibration.get("cameras") or {}
    for role in CAMERA_ALIASES.get(name, (name,)):
        if role in cams and "cam2world_cv" in cams[role]:
            return np.asarray(cams[role]["cam2world_cv"], float)
    arm = next((a for a in S.ARMS if name == f"{a}_wrist"), None)
    if arm is None:
        raise NotImplementedError(f"no pose for camera {name!r}: the calibration has no cam2world_cv for it "
                                  f"(roles there: {sorted(cams)}); calibrate it with the ora cell tools")
    import mujoco
    m = model() if callable(model) else model
    q = joints() if callable(joints) else joints
    d = mujoco.MjData(m)
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    b = d.body(S.CAMERAS[name])
    T = np.eye(4)
    T[:3, :3] = b.xmat.reshape(3, 3)
    T[:3, 3] = b.xpos
    corr = (calibration.get("wrist_camera_corrections") or {}).get(arm)
    return T if corr is None else T @ _rigid(corr)


def _rigid(value: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(value.get("rotvec_rad", [0, 0, 0])).as_matrix()
    T[:3, 3] = value.get("translation_m", [0, 0, 0])
    return T


def joint_offsets(cal: dict) -> dict:
    """Per arm, each arm joint's encoder zero error: model angle = reading + offset."""
    out = {}
    for arm in S.ARMS:
        v = (cal.get("joint_zero_offsets") or {}).get(arm)
        v = v.get("rad") if isinstance(v, dict) else v
        off = np.zeros(6)
        if v is not None:
            v = np.asarray(v, float).reshape(-1)[:6]
            off[:len(v)] = v
        out[arm] = off
    return out


def _dynamics(path: str | None) -> dict:
    """Armature and joint friction for the force estimate: an identified file, else i2rt's friction table."""
    if path:
        with open(os.path.expanduser(path)) as f:
            return json.load(f)
    try:
        from i2rt.robots.utils import ArmType, _load_arm_config
        coulomb = [float(v) for v in _load_arm_config(ArmType.YAM).coulomb_friction]
    except Exception:                              # noqa: BLE001
        coulomb = [0.3, 0.3, 0.3, 0.06, 0.06, 0.06]
    return {"armature": list(S.ARMATURE), "coulomb": coulomb, "viscous": [0.0] * 6}


GRIPPER_WRAP_MARGIN = 0.05            # of the gripper's stroke: how far past its calibrated limits a reading may lie


def _unwrap_gripper(robot, arm: str) -> str | None:
    """A gripper whose motor reads a whole turn away from its calibrated range: shift the range by that turn.

    The DM motor reports its angle within one turn when i2rt starts, and i2rt unwraps from there. A calibrated range
    that reaches past one turn (the left YAM's: closed 6.38 rad, open 1.11 rad) is then read a turn off whenever the
    driver starts with the gripper where it does (closed, at ~6.2 rad, read as ~-0.06): the normalized reading lies far
    past "open", every command drives the jaws into their closed stop (the cell on 2026-11-05: ~600 N, grinding, the
    overload guard tripping at each grip). The stroke is shorter than a turn, so at most one shift fits the range.
    Returns what it did (None: nothing to do); raises if the reading fits the range at no whole turn."""
    lim = getattr(robot, "_gripper_limits", None)
    gi = getattr(robot, "_gripper_index", None)
    js = getattr(robot, "_joint_state", None)
    if lim is None or gi is None or js is None:
        return None
    lim = np.asarray(lim, float)                         # motor rad, [closed, open]
    raw = float(lim[0] + float(js.pos[gi]) * (lim[1] - lim[0]))
    lo, hi = float(lim.min()), float(lim.max())
    margin = GRIPPER_WRAP_MARGIN * (hi - lo)
    if lo - margin <= raw <= hi + margin:
        return None
    for k in (1, -1, 2, -2):
        if lo - margin <= raw + 2 * np.pi * k <= hi + margin:
            from i2rt.robots.utils import JointMapper
            new = lim - 2 * np.pi * k
            # under i2rt's command lock, all at once: its loop clips the gripper's command to _gripper_limits (the
            # old range turns the held startup angle into a push into the closed stop), and its force limiter
            # carries the last clipped command on. The held command itself is a motor angle: it stays.
            with robot._command_lock:
                robot._gripper_limits = new
                robot.remapper = JointMapper(index_range_map={gi: new}, total_dofs=len(robot.motor_chain))
                robot._last_gripper_command_qpos = raw
                limiter = getattr(robot, "_gripper_force_limiter", None)
                if limiter is not None:
                    limiter._is_clogged = False
                    limiter._gripper_adjusted_qpos = None
            g = (raw - new[0]) / (new[1] - new[0])
            return (f"{arm} gripper's motor reads {raw:.2f} rad, {k:+d} turn(s) from its calibrated range "
                    f"[{lim[0]:.2f}, {lim[1]:.2f}]: its limits are {np.round(new, 2).tolist()} for this run "
                    f"(opening {g * 2 * S.FINGER_OPEN_M * 1000:.0f} mm)")
    raise RuntimeError(f"{arm} gripper's motor reads {raw:.2f} rad, outside its calibrated range "
                       f"[{lim[0]:.2f}, {lim[1]:.2f}] by no whole turn: check the gripper (its calibration, or a jam) "
                       "before driving it")


def _gripper_force_per_torque() -> float:
    """Jaw force (N) per gripper-motor torque (Nm), the map i2rt's force limiter uses for the linear 4310 gripper:
    motor stroke / jaw stroke (6.57 rad over 0.096 m, so ~68 N per Nm)."""
    from i2rt.robots.utils import ArmType, GripperType, _load_gripper_config
    p = _load_gripper_config(GripperType.LINEAR_4310.value, ArmType.YAM).limiter_params
    return float(p["motor_stroke"]) / float(p["gripper_stroke"])


def _floats(v):
    return None if v is None else [float(x) for x in np.asarray(v, float).reshape(-1)]


def _bus_stats(channel: str) -> dict:
    """A CAN interface's state (ERROR-ACTIVE is the healthy one; BUS-OFF is not) and its packet counters."""
    out = {}
    base = f"/sys/class/net/{channel}/statistics"
    for k in ("rx_packets", "tx_packets", "rx_errors", "tx_errors"):
        try:
            out[k] = int(open(os.path.join(base, k)).read())
        except OSError:
            out[k] = None
    try:
        info = json.loads(subprocess.run(["ip", "-details", "-json", "link", "show", "dev", channel],
                                         capture_output=True, text=True, timeout=0.5).stdout or "[]")
        link = info[0] if info else {}
        out["up"] = "UP" in link.get("flags", [])
        out["state"] = ((link.get("linkinfo") or {}).get("info_data") or {}).get("state")
    except (OSError, ValueError, subprocess.SubprocessError):
        out["up"], out["state"] = None, None
    return out


def _per_arm(text: str | None, n: int) -> dict | None:
    """"a,b,..." (both arms) or "left=a,b,...;right=a,b,..." -> {arm: [floats]}."""
    if not text:
        return None
    if "=" not in text:
        vals = [float(v) for v in text.split(",")]
        out = {a: vals for a in S.ARMS}
    else:
        out = {}
        for part in text.split(";"):
            arm, vals = part.split("=", 1)
            out[arm.strip()] = [float(v) for v in vals.split(",")]
    for arm, vals in out.items():
        if arm not in S.ARMS or len(vals) != n:
            raise ValueError(f"expected {n} numbers per arm ({list(S.ARMS)}), got {arm}={vals}")
    return out


class CellBusy(RuntimeError):
    pass


def _take_cell_lock(purpose: str):
    path = os.environ.get("ORA_CELL_LOCK", "/tmp/ora-cell.lock")
    try:                                         # no O_CREAT: in /tmp, fs.protected_regular refuses it on a lock file
        fd = os.open(path, os.O_RDWR)            # another user created, even a world-writable one
    except FileNotFoundError:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        os.fchmod(fd, 0o666)                     # so the other users' tools can take it too
    handle = os.fdopen(fd, "r+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "another program"
        handle.close()
        raise CellBusy(f"the cell is in use by {holder}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{purpose} (pid {os.getpid()})")
    handle.flush()
    return handle


def _snapshot(robot) -> dict:
    """i2rt's latest observations of one arm without its state lock, as get_observations() returns them.

    get_observations() takes i2rt's _state_lock, which i2rt's own server thread holds through every update(): the
    gravity model (three MuJoCo inverse-dynamics calls) and the bus exchange with all seven motors. The control tick
    waited on it twice (once per arm): 1.8 ms of a 2.6 ms tick at rest, 2.5 of 4.4 ms with both arms servoing, on the
    cell on 2026-11-04. i2rt (v1.4.0) only ever replaces _joint_state with a new object (motor_chain_robot.py: two
    assignments, no in-place writes), so one reference read is a consistent snapshot. Another i2rt without those
    fields gets the locked call."""
    js, gi = getattr(robot, "_joint_state", None), getattr(robot, "_gripper_index", None)
    if js is None or gi is None:
        return robot.get_observations()
    o = {"joint_pos": js.pos[:gi], "joint_vel": js.vel[:gi], "joint_eff": js.eff[:gi],
         "gripper_pos": js.pos[gi:gi + 1], "gripper_vel": js.vel[gi:gi + 1], "gripper_eff": js.eff[gi:gi + 1]}
    if getattr(robot, "temp_record_flag", False):
        o["temp_mos"], o["temp_rotor"] = js.temp_mos, js.temp_rotor
    return o


def _motors_off(robot) -> None:
    """Stop i2rt's CAN thread for one arm, then send each motor DM's DISABLE.

    i2rt v1.4.0's close() stops its server thread and closes the bus but never disables the motors, and its CAN thread
    can still be sending: the motors then time out enabled and latch DM fault 0xD ('lost communication'; the
    dashboard shows FAULT on every motor until the next start clears it), and the thread dies on the closed socket.
    Disabled here, they end as they were before robotd started. Nothing for i2rt's simulated or monitored robots."""
    chain = getattr(robot, "motor_chain", None)
    iface, motors = getattr(chain, "motor_interface", None), getattr(chain, "motor_list", None)
    if iface is None or not motors:
        return
    stop, server = getattr(robot, "_stop_event", None), getattr(robot, "_server_thread", None)
    if stop is not None and server is not None:          # i2rt's server thread first: on a stopped chain it raises
        stop.set()
        server.join(1.0)
    chain.running = False                                # the CAN loop ends at its next period (about 4 ms)
    time.sleep(0.02)
    lock = getattr(chain, "command_lock", None)          # held by the loop while it sends: wait for a send under way
    if lock is not None:
        lock.acquire()
    try:
        for motor_id, _ in motors:
            iface.motor_off(motor_id)
    finally:
        if lock is not None:
            lock.release()


def _release_cell_lock(handle) -> None:
    """Clear the holder text (ORA's tools read a non-empty file as busy) and let the lock go."""
    try:
        handle.seek(0)
        handle.truncate()
        handle.flush()
    finally:
        handle.close()


class YamI2rtBackend(Backend):
    name = "yam_i2rt"
    lockstep = False
    arm_force_trusted = False       # the arms' force estimate read "contact" with the hand 30 cm in the air
    gripper_empty_width_m = 0.0029  # closed on nothing: GRIPPER_HOLD's closed end, 2.8-3.1 mm on the cell (median 2.93)

    def __init__(self, channels: dict, sim: bool = False, park_q: dict | None = None,
                 camera_serials: dict | None = None, calibration: str | None = None, gripper_limits: dict | None = None,
                 passive: bool = False, gravity_factors: dict | None = None):
        """park_q / gripper_limits / gravity_factors: {arm: values}, overriding the calibration's rest pose / gripper
        limits / i2rt's gravity_comp_factor.
        passive: monitor mode, the motors stay off (every motion request is refused by the driver)."""
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.sim = sim
        self.passive = passive and not sim
        self.name = "yam_i2rt_sim" if sim else ("yam_i2rt_monitor" if self.passive else "yam_i2rt")
        self.channels = dict(channels)
        self.dynamics = _dynamics(os.environ.get("YAM_DYNAMICS"))
        # i2rt reports the gripper's effort as its motor's torque (Nm, + in the motor's direction); the API's finger
        # effort is jaw force (N, + opening), so it is scaled, and flipped where the motor's reading grows closing
        self.grip_force_per_nm = 1.0 if sim else _gripper_force_per_torque()
        self._obs = {}                             # arm -> last observation (the dashboard's temperatures)
        self._bus_cache = (0.0, {})
        self._bus_prev = {}                        # channel -> (time, rx, tx): packet rates
        self.calibration = load_calibration(calibration)
        self.offsets = joint_offsets(self.calibration)
        limits = dict(self.calibration.get("gripper_limits") or {})
        limits.update(gripper_limits or {})
        self.lock = None if sim else _take_cell_lock("robotd monitor, motors off" if self.passive
                                                     else "robotd driver")
        # Cameras start before the motors do, never after: pyrealsense2 holds the GIL for ~0.35 s while four pipelines
        # start, which starves i2rt's control threads past the motors' CAN timeout (they latch "loss communication"
        # and drop out). Grabbing frames afterwards stalls the other threads only ~4 ms.
        self.camera_serials = camera_serials or {}
        self._rs, self._rs_error = None, None
        if self.camera_serials and not sim:
            try:
                from robotd.backends.realsense import RealSenseRig
                self._rs = RealSenseRig(self.camera_serials)
            except Exception as e:                 # noqa: BLE001 -- the arms still come up; cameras() says why
                self._rs_error = f"the cameras did not start: {e}"
        self.robots = {}
        try:
            for arm in S.ARMS:
                if self.passive:
                    from robotd.backends.yam_passive import PassiveYam
                    self.robots[arm] = PassiveYam(channels[arm], limits.get(arm))
                    continue
                extra = {}
                if arm in limits and not sim:          # SimRobot would clip its normalized command to motor limits
                    extra["gripper_limits_override"] = np.asarray(limits[arm], float)
                if gravity_factors and arm in gravity_factors:
                    extra["gravity_comp_factor"] = np.asarray(gravity_factors[arm], float)
                    print(f"robotd: {arm} gravity factors {list(gravity_factors[arm])}", flush=True)
                self.robots[arm] = get_yam_robot(channel=channels[arm], gripper_type=GripperType.LINEAR_4310,
                                                 zero_gravity_mode=False, sim=sim, **extra)
                self.robots[arm].temp_record_flag = True    # observations carry the motor temperatures
                if not sim:
                    note = _unwrap_gripper(self.robots[arm], arm)
                    if note:
                        print(f"robotd: {note}", flush=True)
        except BaseException:                          # an arm did not come up: leave the cell as we found it
            for r in self.robots.values():
                if self.passive:
                    r.close()
            if self._rs is not None:
                self._rs.close()
            if self.lock is not None:
                _release_cell_lock(self.lock)
            raise
        self.joints = []
        for arm in S.ARMS:
            self.joints += S.ARM_JOINTS[arm] + [S.FINGER_JOINTS[arm][0]]
        self.gripper_kp = {a: float(np.asarray(getattr(r, "_kp", [0] * 7))[-1] or 1.0) for a, r in self.robots.items()}
        self.grip_sign = {}
        for a, r in self.robots.items():
            lim = getattr(r, "_gripper_limits", None) if not self.passive else r.limits     # motor rad [closed, open]
            self.grip_sign[a] = -1.0 if (not sim and lim is not None and lim[0] > lim[1]) else 1.0
        self.gripper_kd = {a: float(np.asarray(getattr(r, "_kd", [0] * 7))[-1]) for a, r in self.robots.items()}
        self.grip_force = {a: None for a in S.ARMS}
        self.t0 = time.perf_counter()
        self.next_tick = self.t0
        m = self.read()
        self.start_q = m.q.copy()
        rest = self.calibration.get("rest_reference") or {}
        self.park_q = {a: list(np.asarray(rest[a], float)[:6] + self.offsets[a]) for a in S.ARMS if a in rest}
        self.park_q.update(park_q or {})
        self.park_q = self.park_q or None
        self._kin_model = None

    @classmethod
    def from_env(cls, passive: bool = False):
        ch = dict(kv.split("=") for kv in os.environ.get("YAM_CHANNELS", "left=can_left,right=can_right").split(","))
        cams = os.environ.get("YAM_CAMERAS")
        return cls(ch, sim=os.environ.get("YAM_SIM") == "1", passive=passive,
                   gripper_limits=_per_arm(os.environ.get("YAM_GRIPPER_LIMITS"), 2),
                   park_q=_per_arm(os.environ.get("YAM_PARK_Q"), 6),
                   gravity_factors=_per_arm(os.environ.get("YAM_GRAVITY_FACTORS"), 6),
                   camera_serials=dict(kv.split("=") for kv in cams.split(",")) if cams else None,
                   calibration=os.environ.get("YAM_CALIBRATION"))

    # ---------------------------------------------------------------- control
    def read(self) -> Measured:
        q, dq, tau, tau_model, width = [], [], [], [], {}
        for arm in S.ARMS:
            r = self.robots[arm]
            o = _snapshot(r)
            self._obs[arm] = o
            g = float(np.clip(o["gripper_pos"][0], 0.0, 1.0))
            q += list(np.asarray(o["joint_pos"][:6]) + self.offsets[arm]) + [g * S.FINGER_OPEN_M]   # finger: per-finger travel
            dq += list(o["joint_vel"][:6]) + [float(o["gripper_vel"][0]) * S.FINGER_OPEN_M]
            tau += list(o["joint_eff"][:6]) + [float(o["gripper_eff"][0]) * self.grip_force_per_nm * self.grip_sign[arm]]
            ff = r.get_motor_torques()
            tau_model += list(np.asarray(ff)[:6]) if ff is not None else [np.nan] * 6
            tau_model.append(0.0)
            width[arm] = 2 * g * S.FINGER_OPEN_M
        return Measured(t=time.perf_counter() - self.t0, q=np.array(q), dq=np.array(dq), tau=np.array(tau),
                        gripper_width=width, tau_model=np.array(tau_model))

    def write(self, q_cmd, dq_cmd, kp, kd, effort_limit) -> None:
        if self.passive:                          # monitor mode: nothing is ever sent to the motors
            return
        for k, arm in enumerate(S.ARMS):
            sl = slice(7 * k, 7 * k + 7)
            pos = np.array(q_cmd[sl], float)
            pos[:6] -= self.offsets[arm]                                      # model angle -> encoder reading
            g = float(np.clip(pos[6] / S.FINGER_OPEN_M, *GRIPPER_HOLD))
            pos[6] = g
            vel = np.array(dq_cmd[sl], float)
            vel[6] = 0.0
            kps = np.array(kp[sl], float)
            kds = np.array(kd[sl], float)
            if kps[6] > 0:
                kps[6], kds[6] = self.gripper_kp[arm], self.gripper_kd[arm]   # i2rt's own gripper gains
            else:
                kps[6], kds[6] = 0.0, 0.0                                     # relaxed by the overload guard: limp
            self._set_grip_force(arm, float(effort_limit[sl][6]))
            self.robots[arm].command_joint_state({"pos": pos, "vel": vel, "kp": kps, "kd": kds})

    def _set_grip_force(self, arm: str, force: float) -> None:
        """gripper(force=N): i2rt's force limiter backs the jaw off when its estimated force exceeds the limit.

        i2rt v1.4.0 binds the limit into the limiter's force->torque map when the robot is built
        (partial(..., gripper_force=max_force), at get_yam_robot's 50 N), so setting max_force alone changed nothing:
        on the cell on 2026-11-04 a force=15 grip on a tennis ball held at ~71 N (the driver's estimate). The map is
        rebuilt with the new limit."""
        if force == self.grip_force[arm]:
            return
        r = self.robots[arm]
        limiter = getattr(r, "_gripper_force_limiter", None)
        if limiter is not None:
            limiter.max_force = force
            fmap = getattr(limiter, "gripper_force_torque_map", None)
            if isinstance(fmap, functools.partial):
                limiter.gripper_force_torque_map = functools.partial(fmap.func, *fmap.args,
                                                                     **{**fmap.keywords, "gripper_force": force})
            r._limit_gripper_force = force
        self.grip_force[arm] = force

    def advance(self, dt: float) -> None:
        self.next_tick += dt
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.1:                       # fell far behind (e.g. a slow request): resynchronise
            self.next_tick = time.perf_counter()

    # ---------------------------------------------------------------- sensing
    def health(self) -> dict:
        """Per arm: motor temperatures (MOSFET and rotor, joints 1-6 then the gripper, deg C) and whether i2rt's
        control loop still runs; per CAN bus: state and packet counters (refreshed at most once a second)."""
        arms = {}
        for arm in S.ARMS:
            o = self._obs.get(arm) or {}
            chain = getattr(self.robots[arm], "motor_chain", None)
            arms[arm] = {"channel": self.channels[arm],
                         "temp_mos": _floats(o.get("temp_mos")), "temp_rotor": _floats(o.get("temp_rotor")),
                         "running": None if chain is None else bool(getattr(chain, "running", True))}
            try:                                         # i2rt's own CAN loop: how often it sends every motor its
                hz = getattr(chain, "comm_freq", None)   # command and reads the replies (no fixed period)
                arms[arm]["comm_hz"] = None if hz is None else round(float(hz), 1)
            except Exception:                            # noqa: BLE001 -- not tracked (simulated or monitored arm)
                arms[arm]["comm_hz"] = None
            motors = getattr(self.robots[arm], "motors", None) or getattr(chain, "motor_list", None)
            if motors is not None:
                arms[arm]["motors"] = [[int(m[0]), str(m[1])] for m in motors]     # CAN id and DM model per joint
            if self.passive:
                arms[arm].update(motor_status=list(self.robots[arm].status), error=self.robots[arm].last_error)
            elif not self.sim and chain is not None and getattr(chain, "state", None):
                from robotd.backends.yam_passive import status_text      # each motor's last reply, as i2rt keeps it
                arms[arm]["motor_status"] = [status_text(int(str(s.error_code), 16)) for s in list(chain.state)]
        out = {"arms": arms, "sim": self.sim, "monitor": self.passive}
        if not self.sim:
            now = time.monotonic()
            if now - self._bus_cache[0] > 1.0:
                buses = {}
                for ch in self.channels.values():
                    b = buses[ch] = _bus_stats(ch)
                    prev = self._bus_prev.get(ch)
                    if prev and b["rx_packets"] is not None and prev[1] is not None:
                        b["rx_per_s"] = (b["rx_packets"] - prev[1]) / (now - prev[0])
                        b["tx_per_s"] = (b["tx_packets"] - prev[2]) / (now - prev[0])
                    self._bus_prev[ch] = (now, b["rx_packets"], b["tx_packets"])
                self._bus_cache = (now, buses)
            out["buses"] = self._bus_cache[1]
        return out

    def kinematics_model(self):
        """The station model with the cell's measured base corrections (kinematics_model_for)."""
        if self._kin_model is None:
            self._kin_model = kinematics_model_for(self.calibration)
        return self._kin_model

    def cameras(self, names=None, depth: bool = True) -> dict:
        if not self.camera_serials:
            raise NotImplementedError("no cameras configured: set YAM_CAMERAS=name=serial,...")
        if self._rs is None:
            if self._rs_error or not self.sim:     # a real rig only starts with the motors off (see __init__)
                raise NotImplementedError(self._rs_error or "the cameras are not running")
            from robotd.backends.realsense import RealSenseRig
            self._rs = RealSenseRig(self.camera_serials)
        frames = self._rs.grab(names or list(self.camera_serials), depth)
        for name, f in frames.items():
            try:
                f["T"] = self._camera_pose(name)
            except NotImplementedError as e:      # uncalibrated: the image, but no pose to put its points anywhere
                f["T"], f["pose_error"] = None, str(e)
            f["t"] = time.perf_counter() - self.t0
        return frames

    def _camera_pose(self, name: str) -> np.ndarray:
        return camera_pose(name, self.calibration, self.kinematics_model, lambda: dict(zip(self.joints, self.read().q)))

    def close(self) -> None:
        # The driver has parked the arms. i2rt's threads must stop now: they would keep commanding the park pose, the
        # process would never exit, and the lock below would be free while the motors are still driven. The motors
        # are switched off at the park pose (a resting pose); the lock goes last.
        for arm, r in self.robots.items():
            try:
                _motors_off(r)
            except Exception as e:                       # noqa: BLE001 -- they then time out as before
                print(f"robotd: could not switch the {arm} motors off ({e!r}); they time out instead", flush=True)
            try:
                r.close()
            except Exception:
                pass
        if self._rs is not None:
            self._rs.close()
        if self.lock is not None:
            _release_cell_lock(self.lock)
