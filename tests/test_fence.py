"""The workspace fence (controller.parse_fence): per-arm TCP boxes and a planar speed limit, in process on the MuJoCo
YAM station (lockstep, no RPC)."""
import numpy as np
import pytest

from robot import pose
from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.core.controller import Controller, Fault, InvalidCommand, parse_fence
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks

DOWN = dict(rpy=[np.pi, 0, np.pi])
FENCE = {"left": {"x": [0.20, 0.45], "y": [0.10, 0.45], "z": [0.05, 0.20]}, "max_speed_xy": 0.5}
INSIDE = pose([0.30, 0.30, 0.10], **DOWN)


def make(fence=FENCE):
    b = MujocoYamBackend(tasks.get("empty").scene())
    desc = description()
    desc["fence"] = parse_fence(fence)
    return b, Controller(b, desc)


def run(c, s, tmax=10.0):
    cmd = c.submit(s)
    t0 = c.t
    while not cmd.chain_finished() and c.t - t0 < tmax:
        c.tick()
    assert cmd.chain_finished(), f"did not finish in {tmax}s"
    return cmd.last().result


def armed(c):
    """Move the left TCP inside its box (the fence starts off: the arm rests outside it)."""
    assert not c.fence_armed["left"]
    r = run(c, {"op": "move_to", "ee": "left", "target": INSIDE.tolist()})
    assert r["status"] == "done", r
    assert c.fence_armed["left"]


def test_parse():
    f = parse_fence('{"right": {"x": [0.1, 0.5], "z": [null, 0.2]}}')
    assert f["right"]["x"] == [0.1, 0.5] and f["right"]["y"] == [None, None] and f["right"]["z"] == [None, 0.2]
    assert f["max_speed_xy"] == 1.0 and f["margin"] == 0.02
    with pytest.raises(ValueError):
        parse_fence({"left": {"x": [0.5, 0.1]}})
    with pytest.raises(ValueError):
        parse_fence({"left": {"w": [0, 1]}})
    assert parse_fence(None) is None


def test_targets_outside_are_refused():
    b, c = make()
    with pytest.raises(InvalidCommand, match=r"fence: the target: left TCP x 0\.500 m > x_max 0\.450 m"):
        c.submit({"op": "move_to", "ee": "left", "target": pose([0.50, 0.30, 0.10], **DOWN).tolist()})
    with pytest.raises(InvalidCommand, match="z_min"):
        c.submit({"op": "move_to", "ee": "left", "target": pose([0.30, 0.30, 0.03], **DOWN).tolist(), "line": True})
    with pytest.raises(InvalidCommand, match=r"point 1: left TCP y"):
        c.submit({"op": "follow", "ee": "left", "points": [INSIDE.tolist(), pose([0.3, 0.0, 0.1], **DOWN).tolist()]})
    # a joint target is checked through forward kinematics (the rest pose's TCP is at z 0.23); the right arm has no box
    with pytest.raises(InvalidCommand, match=r"fence: the joint target: left TCP z 0\.2"):
        c.submit({"op": "move_to", "joints": S.ARM_JOINTS["left"], "target": list(S.READY_Q["left"])})
    c.submit({"op": "move_to", "ee": "right", "target": pose([0.45, -0.30, 0.10], **DOWN).tolist()})


def test_too_fast_is_refused():
    b, c = make()
    armed(c)
    with pytest.raises(InvalidCommand, match=r"point 0 needs 1\.00 m/s .*limit 0\.50"):
        c.submit({"op": "follow", "ee": "left", "points": [pose([0.40, 0.30, 0.10], **DOWN).tolist()],
                  "times": [0.1]})
    with pytest.raises(InvalidCommand, match="speed="):
        c.submit({"op": "move_to", "ee": "left", "target": pose([0.40, 0.30, 0.10], **DOWN).tolist(), "speed": 0.8})
    with pytest.raises(InvalidCommand, match="velocities"):
        c.submit({"op": "follow", "ee": "left", "points": [pose([0.40, 0.30, 0.10], **DOWN).tolist()],
                  "times": [0.5], "velocities": [[0.6, 0.0, 0.0]]})
    with pytest.raises(InvalidCommand, match="twist"):
        c.submit({"op": "move_at", "ee": "left", "velocity": {"y": 0.7}})


def test_reference_speed_is_limited():
    """A quick move (0.12 m sideways as fast as the joints go, ~0.4 m/s at its peak) runs at a 0.25 m/s planar limit
    instead."""
    b, c = make({**FENCE, "max_speed_xy": 0.25})
    armed(c)
    cmd = c.submit({"op": "move_to", "ee": "left", "target": pose([0.30, 0.42, 0.10], **DOWN).tolist(),
                    "duration": 0.15})
    ref, meas = [], []
    while not cmd.chain_finished():
        p0 = c.fence_p["left"].copy()
        c.tick()
        ref.append(np.linalg.norm((c.fence_p["left"] - p0)[:2]) / 0.004)
        meas.append(np.linalg.norm(c.v["left"][:2]))
    assert max(ref) <= 0.25 + 1e-6
    assert 0.24 < max(ref)
    assert max(meas) < 0.25 * 1.3
    assert c.fault is None
    assert np.allclose(c.T["left"][:3, 3], [0.30, 0.42, 0.10], atol=0.006)


def test_reference_leaving_the_box_stops_at_its_edge():
    """A joint-space move whose reference would leave the box (here the box shrinks under a running move, as a path
    that swings out between two targets inside does) ends aborted at the edge (reason fence), holding inside."""
    b, c = make()
    armed(c)
    q = c.solve_ik("left", pose([0.42, 0.30, 0.10], **DOWN))
    assert q["ok"]
    cmd = c.submit({"op": "move_to", "joints": q["joints"], "target": q["q"], "duration": 1.0})
    for _ in range(25):
        c.tick()
    c.fence["left"][1][0] = 0.36
    t0 = c.t
    while not cmd.chain_finished() and c.t - t0 < 3.0:
        c.tick()
    r = cmd.last().result
    assert r["status"] == "aborted" and r["reason"].startswith("fence: left TCP x"), r
    for _ in range(100):
        c.tick()
    assert c.fault is None
    assert c.fence_violation("left", c.fence_p["left"], 0.002) is None
    assert c.T["left"][0, 3] < 0.36 + c.fence_margin


def test_move_at_stops_at_the_fence():
    b, c = make()
    armed(c)
    r = run(c, {"op": "move_at", "ee": "left", "velocity": {"x": 0.1}, "max_travel": 0.5})
    assert r["status"] == "done" and r["reason"] == "fence", r
    assert 0.43 < c.T["left"][0, 3] < 0.46
    r = run(c, {"op": "move_at", "ee": "left", "velocity": {"x": -0.1}, "max_travel": 0.05})
    assert r["reason"] == "max_travel"


def test_measured_outside_latches_a_fault_and_recovers():
    b, c = make()
    armed(c)
    c.fence["left"][1][0] = 0.25              # the box shrinks under the TCP (as a push would move the TCP out)
    c.tick()
    assert c.fault and c.fault["reason"].startswith("fence: left TCP x 0.30") and "margin" in c.fault["reason"]
    with pytest.raises(Fault):
        c.submit({"op": "move_to", "ee": "left", "target": INSIDE.tolist()})
    c.recover()
    assert not c.fence_armed["left"] and c.fault is None
    r = run(c, {"op": "move_to", "ee": "left", "target": pose([0.24, 0.30, 0.10], **DOWN).tolist()})
    assert r["status"] == "done" and c.fence_armed["left"] and c.fault is None


def test_park_ignores_the_fence():
    """The exit park (c.parking) and the operator's PARK (park=True) go out of the box; afterwards the arm rests
    outside it without a Fault, and its checks start again once it is back inside."""
    b, c = make()
    armed(c)
    c.parking = True
    run(c, {"op": "move_to", "joints": S.ARM_JOINTS["left"], "target": list(S.READY_Q["left"]), "speed": 1.0})
    assert c.fault is None
    c.parking = False
    armed(c)
    cmd = c.submit({"op": "move_to", "joints": S.ARM_JOINTS["left"], "target": list(S.READY_Q["left"]),
                    "speed": 1.0}, park=True)
    while not cmd.chain_finished():
        c.tick()
    for _ in range(100):
        c.tick()
    assert c.fault is None and not c.fence_armed["left"]
    armed(c)


def test_no_fence_no_change():
    b, c = make(None)
    assert c.fence == {}
    r = run(c, {"op": "move_to", "ee": "left", "target": pose([0.47, 0.30, 0.10], **DOWN).tolist()})
    assert r["status"] == "done"
