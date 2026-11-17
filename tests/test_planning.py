"""Plans that pass the check and fail at run time: IK remembered from reachable() for the move, checks from where the
queue will have put the arms (after=), separation against where the other arm's running move ends, the ghost's IK
from after=, and kernel.planned_end()."""
import numpy as np
import pytest

import robot
from robot import pose
from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.core.controller import Controller
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks
from test_server import start

DOWN = dict(rpy=[np.pi, 0, np.pi])
L = S.ARM_JOINTS["left"]
# Near the edge of reach (x 0.59 m, joint 4 0.03 rad from its limit): IK from the start pose misses by 0.9 mm, the
# polish too (its seeds: the start, halfway to the range's centre, 6 fixed random ones); from Q_B it reaches.
Q_B = [2.74, 0.3, 2.35, 1.54, 0.74, 0.76]


def _fk(q, arm="left"):
    """The TCP pose at joints q, on the robot's own model (as robotd computes it)."""
    c = Controller(MujocoYamBackend(tasks.get("empty").scene()), description())
    return c.kin_plan.fk(arm, np.asarray(q, float))


def test_a_target_reachable_said_yes_to_is_not_refused_from_another_start(tmp_path):
    """reachable() from B (after=) solves the edge target; the move, solved from A (where the arm is), would miss
    it, and now takes the remembered solution instead of refusing."""
    T = _fk(Q_B)
    r = robot.connect(start(tmp_path))
    q_a = r.state().q(L)
    assert not r.reachable([T], "left")[0]["ok"], "the target should be out of IK's reach from the start pose"
    with pytest.raises(robot.InvalidCommand, match="unreachable"):         # nothing remembered yet: refused
        r.move(T, ee="left")
    assert np.allclose(r.state().q(L), q_a, atol=1e-6)
    chk = r.reachable([T], "left", after={"left": Q_B})[0]
    assert chk["ok"] and chk["why"] is None
    res = r.move(T, ee="left", duration=4)                                # from A, as before: now it goes
    assert res.ok
    assert np.linalg.norm(r.T("left")[:3, 3] - T[:3, 3]) < 0.006
    r._call("shutdown")


def test_the_remembered_solution_after_moving_there_and_back(tmp_path):
    """The same with the arm really at B for the check, then back at A when the move is sent."""
    T = _fk(Q_B)
    r = robot.connect(start(tmp_path))
    q_a = r.state().q(L)
    r.move(Q_B, joints=L, duration=3)
    assert r.reachable([T], "left")[0]["ok"]
    r.move(q_a, joints=L, duration=3)
    assert r.move(T, ee="left", duration=4).ok
    assert np.linalg.norm(r.T("left")[:3, 3] - T[:3, 3]) < 0.006
    r._call("shutdown")


def test_after_gives_the_verdict_and_solution_a_move_from_there_gets(tmp_path):
    """reachable / ik with after={arm: joints} agree with what the move does once the arm is there: the same verdict,
    and the same joint goal."""
    r = robot.connect(start(tmp_path))
    q_s = r.ik(pose([0.30, 0.42, 0.12], **DOWN), "left")["q"]
    cands = [pose([0.42, 0.20, 0.06], rpy=[np.pi, 0.0, np.pi / 2]), pose([0.30, 0.45, 0.04], **DOWN),
             pose([1.20, 0.30, 0.08], **DOWN)]
    planned = r.reachable(cands, "left", free="yaw", after={"left": q_s})
    sols = [r.ik(T, "left", free="yaw", after={"left": q_s}) for T in cands]
    now = r.reachable(cands, "left", free="yaw")
    assert [o["ok"] for o in planned] == [True, True, False]
    assert any(np.abs(p["q"] - n["q"]).max() > 0.1 for p, n in zip(planned, now) if p["ok"]), \
        "the start should matter for at least one candidate"
    r.move(q_s, joints=L, duration=3)
    for T, o, s in zip(cands, planned, sols):
        if o["ok"]:
            plan = r.move(T, ee="left", free="yaw", dry_run=True)
            assert plan.ok and np.allclose(plan.q_goal["left"], o["q"], atol=1e-6)
            assert np.allclose(s["q"], o["q"], atol=1e-6)
        else:
            with pytest.raises(robot.InvalidCommand):
                r.move(T, ee="left", free="yaw")
    r._call("shutdown")


def test_after_takes_a_pose_and_the_move_it_was_given(tmp_path):
    """after={arm: 4x4} or {"target", "tcp", "free"}: the joints the remembered solution for that move has."""
    r = robot.connect(start(tmp_path))
    P = pose([0.30, 0.42, 0.12], **DOWN)
    sol = r.ik(P, "left", free="yaw")
    T = pose([0.35, 0.35, 0.06], **DOWN)
    by_q = r.ik(T, "left", after={"left": sol["q"]})
    by_move = r.ik(T, "left", after={"left": {"target": P, "free": "yaw"}})
    assert np.allclose(by_q["q"], by_move["q"], atol=1e-9)
    with pytest.raises(robot.InvalidCommand, match="after"):
        r.ik(T, "left", after={"left": pose([1.5, 0.3, 0.1], **DOWN)})
    r._call("shutdown")


def test_separation_against_where_the_other_arm_is_going(tmp_path):
    """The right arm is on its way into the middle: a left target clear of where it is now, but not of where it will
    be, was ok before; now reachable refuses it with a separation why, and so does the move. after= that puts the
    right arm elsewhere checks it there instead."""
    r = robot.connect(start(tmp_path, separation=0.02))
    G_right = pose([0.40, -0.06, 0.10], **DOWN)
    T = pose([0.40, 0.10, 0.10], **DOWN)
    q_right_now = r.state().q(S.ARM_JOINTS["right"])
    before = r.reachable([T], "left", free="yaw")[0]
    assert before["ok"], before["why"]                                    # the old check: the right arm where it is
    m = r.move(G_right, ee="right", free="yaw", duration=6, wait=False)   # lockstep: it has not moved yet
    out = r.reachable([T], "left", free="yaw")[0]
    assert not out["ok"] and out["why"].startswith("separation") and "once right gets where its running move_to" \
        in out["why"], out["why"]
    assert out["q"].shape == (6,)                                         # the IK solution, for the record
    with pytest.raises(robot.InvalidCommand, match="separation"):
        r.move(T, ee="left", free="yaw")
    planned = r.reachable([T], "left", free="yaw", after={"right": q_right_now})[0]
    assert planned["ok"], planned["why"]                                  # the right arm will be back home by then
    into = r.reachable([T], "left", free="yaw", after={"right": {"target": G_right, "free": "yaw"}})[0]
    assert not into["ok"] and "where after= puts it" in into["why"], into["why"]
    assert m.result().ok
    r._call("shutdown")


def test_the_ghost_solves_from_after_on_the_latest_frame(tmp_path):
    """A ghost with after= draws the joints r.ik finds from that start, on the frame and arm state of now."""
    from robot import viz
    r = robot.connect(start(tmp_path))
    q_s = r.ik(pose([0.30, 0.42, 0.12], **DOWN), "left")["q"]
    T = pose([0.42, 0.20, 0.06], rpy=[np.pi, 0.0, np.pi / 2])
    want = r.ik(T, "left", free="yaw", after={"left": q_s})["q"]
    plain = r.ik(T, "left", free="yaw")["q"]
    assert np.abs(want - plain).max() > 0.1, "the start should change the solution"
    r.step(25)                                                            # time passes: the frame must be the newest
    t_now = r.state().t
    o = viz.Overlay(r, "top")
    assert abs(o.frame.t - t_now) < 0.05
    g = o.ghost("left", T, free="yaw", after={"left": q_s})
    assert g["ik_ok"] and np.allclose([g["q"][n] for n in L], want, atol=1e-6)
    g0 = o.ghost("left", T, free="yaw")
    assert np.allclose([g0["q"][n] for n in L], plain, atol=1e-6)
    v = viz.View3D(r, cameras=["top"])
    g3 = v.ghost("left", T, free="yaw", after={"left": q_s})
    assert np.allclose([g3["q"][n] for n in L], want, atol=1e-6)
    r._call("shutdown")


def test_planned_end_collects_the_moves_sent_and_the_ends_declared():
    from robot import kernel
    aj = description()["arm_joints"]
    G = pose([0.3, 0.3, 0.1], **DOWN).tolist()
    ends = kernel._ends_of({"op": "move_to", "ee": "left", "target": G, "tcp": None, "free": "yaw",
                            "then": {"op": "move_to", "joints": aj["right"], "target": [0.1] * 6}}, aj)
    assert ends == {"left": {"target": G, "tcp": None, "free": "yaw"}, "right": [0.1] * 6}
    assert kernel._ends_of({"op": "move_at", "ee": "left", "velocity": {"z": 0.01}}, aj) == {"left": None}
    assert kernel._ends_of({"op": "move_to", "ee": "left", "target": G, "frame": "tool"}, aj) == {"left": None}
    assert kernel._ends_of({"op": "gripper", "name": "left", "width": 0.02}, aj) == {}
    run = kernel.Action(lambda: None, "run", 10)
    run.ends = {"left": {"target": G, "tcp": None, "free": None}, "right": None}
    nxt = kernel.Action(lambda: None, "nxt", 10)
    nxt.declared = {"right": [0.2] * 6}
    saved = list(kernel._CURRENT), list(kernel._QUEUE)
    try:
        kernel._CURRENT[:], kernel._QUEUE[:] = [run], [nxt]
        assert kernel.planned_end() == {"left": run.ends["left"], "right": [0.2] * 6}
        kernel._QUEUE[:] = []
        assert kernel.planned_end() == {"left": run.ends["left"]}           # what it cannot tell is left out
    finally:
        kernel._CURRENT[:], kernel._QUEUE[:] = saved


def test_a_solve_that_reaches_stands_over_a_remembered_one_that_is_not_quicker():
    """The remembered solutions change nothing when the solve from the start reaches, unless one is much quicker to go
    to: a costlier solution for the same target (found from another start) is ignored, and the memory keeps one
    solution per target, the newest."""
    c = Controller(MujocoYamBackend(tasks.get("empty").scene()), description())
    q0 = c._arm(c.q_ref, "left")
    T = pose([0.42, 0.20, 0.06], rpy=[np.pi, 0.0, np.pi / 2])
    with c.plan_lock:
        q_s = c._ik_free("left", pose([0.30, 0.42, 0.12], **DOWN), q0, None, None, remember=False)[0]
        far, ok, _ = c._ik_free("left", T, q_s, None, "yaw")
        assert ok and len(c.ik_memory["left"]) == 1
        q1, ok1, _ = c._ik_free("left", T, q0, None, "yaw")
        assert ok1 and np.abs(far - q1).max() > 0.1 and c._cost("left", q0, far) > c._cost("left", q0, q1)
        c.ik_memory["left"].clear()
        q_plain = c._ik_free("left", T, q0, None, "yaw")[0]
    assert np.allclose(q1, q_plain, atol=1e-9)                    # the same as with nothing remembered
    assert len(c.ik_memory["left"]) == 1 and np.allclose(c.ik_memory["left"][-1][3], q_plain)
