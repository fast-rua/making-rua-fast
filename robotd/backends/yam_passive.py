"""One YAM arm read with its motors off: the backend's monitor mode (robotd --real yam --monitor).

Each poll sends every motor DM's DISABLE frame, which a DM motor answers with its state (position, velocity, torque,
both temperatures, its status) without enabling. Nothing here can command a torque. The motor-to-joint conversion is
i2rt's own (v1.4.0): unwrap the 16-bit position, offset it into (-pi, pi], apply the direction, the linkage
transmissions and the gripper's normalisation by its limits.

A DISABLE frame turns a running motor off, so this runs only when no other program drives the arm; the backend holds
the cell lock while it does.
"""
from __future__ import annotations

import threading
import time

import numpy as np

DISABLE = [0xFF] * 7 + [0xFD]
FAULTS = {0x8: "over voltage", 0x9: "under voltage", 0xA: "over current", 0xB: "MOSFET over temperature",
          0xC: "motor over temperature", 0xD: "lost communication", 0xE: "overload"}     # DM's documented faults


def status_text(code: int) -> str:
    """The status nibble of a DM reply. Disabled DM4310s on this cell answer 0x3, which DM does not document; it is
    not one of the faults."""
    if code in FAULTS:
        return "error: " + FAULTS[code]
    return {0x0: "disabled", 0x1: "enabled"}.get(code, f"status 0x{code:x}")


class PassiveYam:
    """The subset of i2rt's robot that YamI2rtBackend reads (get_observations, get_motor_torques, motor_chain.running),
    from DISABLE replies polled at rate_hz."""

    def __init__(self, channel: str, gripper_limits, rate_hz: float = 50.0, iface=None):
        from i2rt.robots.utils import ArmType, GripperType, JointMapper, _load_arm_config
        hw = _load_arm_config(ArmType.YAM)
        g = GripperType.LINEAR_4310
        limits = gripper_limits if gripper_limits is not None else g.get_gripper_limits(ArmType.YAM)
        if limits is None:
            raise ValueError("monitor mode needs the gripper limits (YAM_GRIPPER_LIMITS or the cell calibration): "
                             "without them the gripper reading means nothing")
        self.channel = channel
        self.motors = [(int(i), str(t)) for i, t in hw.motor_list] + [(int(hw.gripper_can_id), g.get_motor_type(ArmType.YAM))]
        self.direction = np.array(list(hw.directions) + [g.get_motor_direction(ArmType.YAM)], float)
        self.transmissions = dict(hw.transmissions)
        self.limits = tuple(float(v) for v in limits)
        self.mapper = JointMapper({len(self.motors) - 1: self.limits}, len(self.motors))
        if iface is None:
            from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, ReceiveMode
            iface = DMSingleMotorCanInterface(channel=channel, receive_mode=ReceiveMode.p16,
                                              name=f"robotd_monitor_{channel}")
        self.iface = iface
        self.temp_record_flag = True
        self.motor_chain = self                       # health() reads motor_chain.running
        self.running = True
        self.status = [None] * len(self.motors)
        self.last_error = None
        self._abs = None
        self._offset = np.zeros(len(self.motors))
        self._obs = None
        self._lock = threading.Lock()
        if hasattr(self.iface, "_drain_bus"):
            self.iface._drain_bus(timeout_s=0.05)
        self._poll()                                  # the first read raises if the arm does not answer
        self._period = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"monitor-{channel}")
        self._thread.start()

    # ------------------------------------------------------------------ polling
    def _read(self, motor_id: int, motor_type: str):
        """One DISABLE frame, one reply: (position, velocity, torque, temp_mos, temp_rotor, status) in motor units."""
        from i2rt.motor_drivers.utils import MotorType, uint_to_float
        msg = self.iface._send_message_get_response(self.iface._get_frame_id(motor_id), motor_id, DISABLE)
        d = msg.data
        c = MotorType.get_motor_constants(motor_type)
        p = uint_to_float((d[1] << 8) | d[2], c.POSITION_MIN, c.POSITION_MAX, 16)
        v = uint_to_float((d[3] << 4) | (d[4] >> 4), c.VELOCITY_MIN, c.VELOCITY_MAX, 12)
        t = uint_to_float(((d[4] & 0xF) << 8) | d[5], c.TORQUE_MIN, c.TORQUE_MAX, 12)
        return p, v, t, float(d[6]), float(d[7]), (d[0] & 0xF0) >> 4, c.POSITION_MAX - c.POSITION_MIN

    def _poll(self):
        rows = [self._read(mid, mtype) for mid, mtype in self.motors]
        raw = np.array([r[0] for r in rows])
        span = np.array([r[6] for r in rows])
        if self._abs is None:                         # as get_yam_robot: offsets that put each motor in (-pi, pi]
            self._abs = raw.copy()
            pos = self._abs * self.direction
            self._offset = np.round(pos / (2 * np.pi)) * 2 * np.pi * self.direction
            # ... except the gripper, whose stroke spans more than pi: the turn that lands it inside its limits
            g, lo, hi = len(self.motors) - 1, min(self.limits), max(self.limits)
            turns = [k * 2 * np.pi * self.direction[g] for k in (-1, 0, 1)]
            dist = [max(lo - p, 0.0, p - hi) for p in ((self._abs[g] - k) * self.direction[g] for k in turns)]
            self._offset[g] = turns[int(np.argmin(dist))]
        else:                                         # as DMChainCanInterface: follow wrap-arounds of the raw value
            delta = raw - np.mod(self._abs, span)
            delta = np.where(delta > span / 2, delta - span, np.where(delta < -span / 2, delta + span, delta))
            self._abs = self._abs + delta
        pos = (self._abs - self._offset) * self.direction
        vel = np.array([r[1] for r in rows]) * self.direction
        eff = np.array([r[2] for r in rows]) * self.direction
        for idx, linkage in self.transmissions.items():
            ratio = linkage.ratio(pos[idx])
            pos[idx] = linkage.joint_from_motor(pos[idx])
            vel[idx] *= ratio
            eff[idx] /= ratio
        pos = self.mapper.to_command_joint_pos_space(pos)
        vel = self.mapper.to_command_joint_vel_space(vel)
        n = len(self.motors) - 1
        obs = {"joint_pos": pos[:n], "joint_vel": vel[:n], "joint_eff": eff[:n],
               "gripper_pos": pos[n:], "gripper_vel": vel[n:], "gripper_eff": eff[n:],
               "temp_mos": np.array([r[3] for r in rows]), "temp_rotor": np.array([r[4] for r in rows])}
        with self._lock:
            self._obs = obs
            self.status = [status_text(r[5]) for r in rows]

    def _loop(self):
        fails = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._poll()
                fails, self.running, self.last_error = 0, True, None
            except Exception as e:                    # noqa: BLE001 -- a motor did not answer: report, keep trying
                fails += 1
                self.last_error = str(e)
                self.running = fails < 5
            self._stop.wait(max(0.0, self._period - (time.monotonic() - t0)))

    # ------------------------------------------------------------------ what the backend reads
    def get_observations(self) -> dict:
        with self._lock:
            return {k: v.copy() for k, v in self._obs.items()}

    def get_motor_torques(self):
        return None

    def command_joint_state(self, joint_state):
        raise RuntimeError(f"{self.channel}: monitor mode, the motors are off")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if hasattr(self.iface, "close"):
            self.iface.close()
