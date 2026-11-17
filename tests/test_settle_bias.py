"""The settled-error correction (controller._settle_bias) on the MuJoCo YAM station with a deliberately wrong gravity
compensation (SceneConfig.joint_torque_bias / gravity_comp_scale), and what the agent is told about a grip."""
import numpy as np
import pytest

from robot import Compliance, pose
from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.core.controller import BIAS_GAP_S, FLOOR_REF_TOL, Controller
from robotd.core.recorder import Recorder, load_ticks
from robotd.robots.yam_station.description import description
import tasks

DOWN = dict(rpy=[np.pi, 0, np.pi])
# the real YAM's steady offset near the table (joint 2 ~0.8 deg, joint 3 ~1.2 deg: the hand ~1 cm high), as a
# constant torque on the left arm's joints 2 and 3 (kp 80 Nm/rad)
OFFSET = {"left_joint2": -1.12, "left_joint3": 1.68}
NEAR_TABLE = pose([0.40, 0.25, 0.010], **DOWN)
IN_AIR = pose([0.30, 0.25, 0.30], rpy=[np.pi / 2, 0, np.pi / 2])


# the real YAM's joints stick 0.3-0.5 deg off a target (breakaway, sliding friction in Nm)
STICK = {"left_joint1": (0.6, 0.35), "left_joint2": (0.7, 0.4), "left_joint3": (0.6, 0.35), "left_joint4": (0.08, 0.05)}


def make(torque=None, scale=1.0, settle=True, fence=None, scene="empty", stiction=None):
    cfg = tasks.get(scene).scene()
    cfg.joint_torque_bias, cfg.gravity_comp_scale = dict(torque or {}), scale
    cfg.joint_stiction = dict(stiction or {})
    b = MujocoYamBackend(cfg)
    d = description()
    if not settle:
        d["settle_bias"] = None
    if fence is not None:
        d["fence"] = fence
    return b, Controller(b, d)


def run(c, s, tmax=15.0):
    cmd = c.submit(s)
    t0 = c.t
    while not cmd.chain_finished() and c.t - t0 < tmax:
        c.tick()
    assert cmd.chain_finished(), f"did not finish in {tmax}s"
    return cmd.last().result


def ticks(c, seconds):
    for _ in range(int(round(seconds / 0.004))):
        c.tick()


def tcp_err(c, arm="left"):
    """The measured TCP's distance from the reference TCP (m)."""
    q, qr = c._arm(c.m.q, arm), c._arm(c.q_ref, arm)
    return float(np.linalg.norm(c.kin.fk(arm, q)[:3, 3] - c.kin.fk(arm, qr)[:3, 3]))


def deg(c, arm="left"):
    return np.degrees(np.abs(c.bias[c.arm_cols[arm]]).max())


def test_perfect_tracking_keeps_the_bias_near_zero():
    b, c = make()
    for T in (NEAR_TABLE, IN_AIR):
        run(c, {"op": "move_to", "ee": "left", "target": T.tolist()})
        ticks(c, 1.0)
        assert deg(c) < 0.2 and tcp_err(c) < 0.001
    assert deg(c, "right") < 0.2
    assert np.abs(c.bias[[c.jidx[n] for n in c.finger.values()]]).max() == 0.0     # never on the fingers


@pytest.mark.parametrize("torque,scale", [(OFFSET, 1.0), ({}, 0.85)])
def test_settled_error_drops_from_cm_to_mm(torque, scale):
    """Without the bias the hand settles 8-16 mm off its reference; with it, under 2 mm within 1.5 s, holding still
    (no overshoot or ringing) once settled."""
    for settle in (False, True):
        b, c = make(torque, scale, settle)
        errs = []
        for T in (NEAR_TABLE, IN_AIR, NEAR_TABLE):
            run(c, {"op": "move_to", "ee": "left", "target": T.tolist()})
            ticks(c, 1.5)
            errs.append(tcp_err(c))
            if settle:                                       # settled: steady, no oscillation
                e, dq = [], []
                for _ in range(250):
                    c.tick()
                    e.append(tcp_err(c))
                    dq.append(np.abs(c._arm(c.m.dq, "left")).max())
                assert max(e) < 0.002 and max(dq) < 0.01, (max(e), max(dq))
        if settle:
            assert max(errs) < 0.002, errs
            assert 0.5 < deg(c) <= 1.5
        else:
            assert min(errs) > 0.007, errs


def test_sticky_joints_settle_without_hunting():
    """Stick-slip on joints 1-4: each hold ends within 2.5 mm (the deadband), and once settled the joints stay put (a joint that sticks,
    is pushed free and jumps past its target is frozen there, not driven back and forth)."""
    b, c = make(OFFSET, stiction=STICK)
    idx = c.arm_cols["left"]
    for p in ((0.35, 0.25, 0.03), (0.45, 0.15, 0.03), (0.30, 0.25, 0.18), (0.40, 0.20, 0.02)):
        run(c, {"op": "move_to", "ee": "left", "target": pose(list(p), **DOWN).tolist(), "duration": 3.0})
        ticks(c, 2.0)
        assert tcp_err(c) < 0.0025, (p, tcp_err(c))
        bias0, qs = c.bias[idx].copy(), []
        for _ in range(500):
            c.tick()
            qs.append(c.m.q[idx].copy())
        qs = np.degrees(qs)                              # (MuJoCo's soft friction lets a held joint creep slowly)
        assert np.allclose(c.bias[idx], bias0) and np.abs(qs[12:] - qs[:-12]).max() < 0.05, p   # no jump in 50 ms


def test_bias_is_held_through_motion_and_a_grasp_close_lands():
    """Air -> 30 mm above the table (0.5 s) -> a line down to 8 mm: the TCP is there at the close, not 1 cm high."""
    out = {}
    for settle in (False, True):
        b, c = make(OFFSET, settle=settle)
        run(c, {"op": "move_to", "ee": "left", "target": IN_AIR.tolist()})
        ticks(c, 1.0)
        A = pose([0.40, 0.25, 0.030], **DOWN)
        run(c, {"op": "move_to", "ee": "left", "target": A.tolist()})
        ticks(c, 0.5)
        G = A.copy()
        G[2, 3] = 0.008
        held = c.bias.copy()
        cmd = c.submit({"op": "move_to", "ee": "left", "target": G.tolist(), "line": True, "duration": 1.5})
        while not cmd.finished:
            c.tick()
            if settle and np.abs(c._arm(c.dq_ref, "left")).max() > c.settle["speed"]:
                assert np.allclose(c.bias, held)              # fast: held, still applied
            held = c.bias.copy()
        ticks(c, 0.3)
        out[settle] = c.T["left"][2, 3]
    assert out[False] > 0.016 and abs(out[True] - 0.008) < 0.002, out


def test_soft_stop_and_fault_turn_it_off():
    b, c = make(OFFSET)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    assert deg(c) > 0.5
    c.operator_stop(soft=False)                              # freeze: held where they are, the bias bleeds
    ticks(c, 1.0)
    assert deg(c) < 0.01
    c.recover(operator=True)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    assert deg(c) > 0.5
    c.operator_stop(soft=True)
    c.tick()
    assert deg(c) == 0.0 and np.allclose(c.q_cmd, c.q_ref)   # no stiffness: nothing to correct


def test_compliance_bleeds_it_and_a_finished_press_keeps_it_off():
    b, c = make(OFFSET)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    assert deg(c) > 0.5
    comp = Compliance(frame="world", stiffness={"z": 300})
    cmd = c.submit({"op": "follow", "ee": "left", "points": [NEAR_TABLE.tolist()], "times": [2.0],
                    "compliance": dict(comp)})
    ticks(c, 1.0)
    assert deg(c) < 0.01                                     # never fights a compliance
    while not cmd.finished:
        c.tick()
    ticks(c, 1.0)
    assert deg(c) < 0.01 and "left" in c.bias_blocked        # the reference may sit inside what it pressed
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    assert deg(c) > 0.5                                      # the next command learns it again


def test_pressing_into_the_table_does_not_wind_up():
    """No fence: a target 4 mm below the table. The joints stop following the bias, which bleeds off."""
    b, c = make(OFFSET)
    ev = []
    c.listener = lambda kind, text, **info: ev.append(text) if kind == "warning" else None
    c.submit({"op": "move_to", "ee": "left", "target": pose([0.40, 0.25, -0.004], **DOWN).tolist()})
    peak = 0.0
    for _ in range(1500):
        c.tick()
        peak = max(peak, deg(c))
    assert "left" in c.bias_blocked and deg(c) < 0.01 and peak < 2.0, peak
    assert any("do not follow the settle bias" in e for e in ev)


def test_never_lowers_the_measured_gripper_below_the_fence_floor():
    """The fence's floor 12 mm above the table, a target on it: the hand, ~1 cm high without the bias, comes down to
    it, never below it by more than a reference may go."""
    floor = 0.012
    fence = {"left": {"z": [0.0, 0.65]}, "gripper_floor": floor}
    b, c = make(OFFSET, fence=fence)
    T = pose([0.40, 0.25, floor + 0.0022], **DOWN)
    run(c, {"op": "move_to", "ee": "left", "target": pose([0.40, 0.25, 0.05], **DOWN).tolist()})
    run(c, {"op": "move_to", "ee": "left", "target": T.tolist(), "line": True, "duration": 1.0})
    lows = []
    for _ in range(750):
        c.tick()
        lows.append(c.floor.lowest("left", c.T["left"]))
    assert min(lows) > floor - FLOOR_REF_TOL, min(lows)
    assert abs(lows[-1] - floor) < 0.002 and deg(c) > 0.5


def test_held_while_the_gripper_closes():
    b, c = make(OFFSET)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    run(c, {"op": "gripper", "name": "left", "width": 0.07})
    held = c.bias.copy()
    cmd = c.submit({"op": "gripper", "name": "left", "width": 0.0})
    while not cmd.finished:
        c.tick()
        assert np.allclose(c.bias, held)


def test_reset_on_a_measurement_gap_and_on_park():
    b, c = make(OFFSET)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.5)
    assert deg(c) > 0.5
    b.advance(BIAS_GAP_S + 0.02)                             # the backend went away and came back
    c.tick()
    assert deg(c) < 0.05                                     # (and this tick's step)
    ticks(c, 1.5)
    assert deg(c) > 0.5
    c.operator_park()
    ticks(c, 1.0)
    assert deg(c) < 0.01


def test_off_by_description_and_recorded(tmp_path):
    b, c = make(OFFSET, settle=False)
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 0.5)
    assert c.settle is None and not c.bias.any() and np.allclose(c.q_cmd, c.q_ref, atol=1e-3)
    b, c = make(OFFSET)
    c.recorder = Recorder(str(tmp_path / "run"))
    run(c, {"op": "move_to", "ee": "left", "target": NEAR_TABLE.tolist()})
    ticks(c, 1.0)
    c.recorder.close()
    t = load_ticks(str(tmp_path / "run"))
    assert t["bias"].shape == t["q"].shape and np.allclose(t["bias"][-1], c.bias)
    assert np.allclose(c.state()["bias"], c.bias)


def test_a_close_on_nothing_says_so():
    b, c = make()
    assert c.desc["gripper_empty_width_m"] == 0.0 and c.desc["arm_force_trusted"] is True     # the sim's
    run(c, {"op": "gripper", "name": "left", "width": 0.06})
    g = run(c, {"op": "gripper", "name": "left", "width": 0.0})
    assert g["empty"] and c.gripper_state["left"]["empty"] and not c.gripper_state["left"]["stalled"]
    g = run(c, {"op": "gripper", "name": "left", "width": 0.03})
    assert not g["empty"] and not c.state()["grippers"]["left"]["empty"]
    # the real linear_4310 ends an empty close at 2.8-3.1 mm, which can read as stalled
    b, c = make(scene="pick_cube")
    c.desc["gripper_empty_width_m"] = 0.0029
    run(c, {"op": "gripper", "name": "left", "width": 0.07})
    run(c, {"op": "move_to", "ee": "left", "target": pose([0.36, 0.28, 0.10], **DOWN).tolist()})
    run(c, {"op": "move_to", "ee": "left", "target": pose([0.36, 0.28, 0.025], **DOWN).tolist(), "line": True})
    g = run(c, {"op": "gripper", "name": "left", "width": 0.0, "force": 15})
    assert g["reason"] == "stalled" and not g["empty"] and c.gripper_state["left"]["stalled"]      # holds the cube
