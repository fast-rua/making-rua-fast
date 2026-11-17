"""Monitor mode: the real YAM read with its motors off (backends/yam_passive.py), and the driver refusing motion."""
import types

import numpy as np
import pytest

from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks
from robotd.server import Driver, NotSupported

LIMITS = [6.3796, 1.1114]                             # the left arm's gripper: motor rad [closed, open]


class FakeBus:
    """Answers each frame like a disabled DM motor at the given motor position (rad)."""

    def __init__(self, pos):
        self.pos, self.sent = dict(pos), []

    def _get_frame_id(self, motor_id):
        return motor_id

    def _send_message_get_response(self, frame_id, motor_id, data):
        from i2rt.motor_drivers.utils import MotorType, float_to_uint
        self.sent.append(list(data))
        c = MotorType.get_motor_constants("DM4340" if motor_id <= 3 else "DM4310")
        p = float_to_uint(self.pos[motor_id], c.POSITION_MIN, c.POSITION_MAX, 16)
        v = float_to_uint(0.0, c.VELOCITY_MIN, c.VELOCITY_MAX, 12)
        t = float_to_uint(0.5, c.TORQUE_MIN, c.TORQUE_MAX, 12)
        return types.SimpleNamespace(data=bytes([motor_id, p >> 8, p & 0xFF, v >> 4, ((v & 0xF) << 4) | (t >> 8),
                                                 t & 0xFF, 41, 37]))


def test_passive_reader_converts_like_i2rt_and_only_sends_disable():
    pytest.importorskip("i2rt")
    from robotd.backends.yam_passive import DISABLE, PassiveYam
    q = [0.05, -0.03, 0.08, 0.0, 0.01, -0.02]
    bus = FakeBus({**{i + 1: v for i, v in enumerate(q)}, 7: LIMITS[1] + 0.5 * (LIMITS[0] - LIMITS[1])})
    r = PassiveYam("can_test", LIMITS, rate_hz=200, iface=bus)
    try:
        o = r.get_observations()
        assert np.allclose(o["joint_pos"], q, atol=1e-3)
        assert abs(o["gripper_pos"][0] - 0.5) < 0.01
        assert o["temp_mos"][0] == 41 and o["temp_rotor"][0] == 37 and r.status[0] == "disabled"
        bus.pos[2] = 0.4                                  # someone lifts the shoulder by hand
        import time
        time.sleep(0.05)
        assert abs(r.get_observations()["joint_pos"][1] - 0.4) < 1e-3
        assert bus.sent and all(f == DISABLE for f in bus.sent)
        with pytest.raises(RuntimeError, match="monitor"):
            r.command_joint_state({"pos": np.zeros(7)})
        assert r.get_motor_torques() is None
    finally:
        r.close()


def test_passive_reader_keeps_a_closed_gripper_inside_its_stroke():
    pytest.importorskip("i2rt")
    from robotd.backends.yam_passive import PassiveYam
    bus = FakeBus({**{i: 0.0 for i in range(1, 7)}, 7: LIMITS[0]})          # jaws closed at start-up (> pi)
    r = PassiveYam("can_test", LIMITS, rate_hz=200, iface=bus)
    try:
        g = r.get_observations()["gripper_pos"][0]
        assert abs(g - 0.0) < 0.01 or abs(g - 1.0) < 0.01                    # at the closed end, not a turn away
        assert 0.0 - 0.01 <= g <= 1.0 + 0.01
    finally:
        r.close()


def test_monitor_driver_refuses_motion_and_does_not_park():
    cfg = tasks.get("empty").scene()
    b = MujocoYamBackend(cfg)
    b.passive = True                                       # what YamI2rtBackend(passive=True) reports
    d = Driver(b, description())
    js = S.ARM_JOINTS["left"]
    with pytest.raises(NotSupported, match="monitor"):
        d.handle({"op": "command", "spec": {"op": "move_to", "joints": js, "target": [0.0] * 6}})
    with pytest.raises(NotSupported):
        d.handle({"op": "park"})
    assert d.handle({"op": "hello"})["passive"] is True
    assert d.handle({"op": "state"})["q"]                  # reading works
    ev = d.handle({"op": "events", "since": 0})["events"]
    assert [e["kind"] for e in ev].count("refused") >= 1
    called = []
    d._park = lambda: called.append(1)
    b.lockstep = False
    d.lockstep = False
    d.close()
    assert not called


def test_cell_lock_excludes_and_is_cleared_on_release(tmp_path, monkeypatch):
    from robotd.backends import yam_i2rt as Y
    path = tmp_path / "ora-cell.lock"
    monkeypatch.setenv("ORA_CELL_LOCK", str(path))
    h = Y._take_cell_lock("robotd monitor, motors off")
    assert "robotd monitor" in path.read_text()
    with pytest.raises(Y.CellBusy, match="robotd monitor"):
        Y._take_cell_lock("someone else")
    Y._release_cell_lock(h)
    assert path.read_text() == ""                          # ORA's tools read a non-empty file as busy
    Y._release_cell_lock(Y._take_cell_lock("again"))       # free again


def test_gripper_effort_is_jaw_force():
    pytest.importorskip("i2rt")
    from robotd.backends.yam_i2rt import _gripper_force_per_torque
    assert abs(_gripper_force_per_torque() - 6.57 / 0.096) < 1e-6        # ~68 N at the jaws per Nm at the motor
