"""The driver server: concurrent clients, stop from a second client, and disconnect stopping a client's motions."""
import os
import threading
import time

import numpy as np
import pytest

import robot
from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks
from robotd.server import Driver, serve


def start(tmp_path, realtime=False, separation=None):
    cfg = tasks.get("empty").scene()
    b = MujocoYamBackend(cfg, realtime=realtime)    # realtime: on the wall clock, as robotd --realtime (and a real arm)
    desc = description()
    if separation is not None:                  # the arms kept this far apart (robotd --separation)
        from robotd.core.separation import YAM
        desc["separation"] = {"min_m": separation, "fault_m": 0.0, **YAM}
    addr = f"unix:{tmp_path}/robot.sock"
    ready = str(tmp_path / "ready")
    threading.Thread(target=serve, args=(Driver(b, desc), addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    return addr


def slow_move(r):
    q0 = r.state().q(S.ARM_JOINTS["left"])
    return dict(target=q0 + np.array([0.6, 0, 0, 0, 0, 0]), joints=S.ARM_JOINTS["left"], duration=8.0)


def test_stop_from_a_second_client_while_one_blocks(tmp_path):
    addr = start(tmp_path, realtime=True)
    a, b = robot.connect(addr), robot.connect(addr)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("res", a.move(**slow_move(a))))
    th.start()
    time.sleep(0.5)
    b.stop()
    th.join(timeout=5)
    assert not th.is_alive(), "the blocked call did not return after stop()"
    assert out["res"].status == "stopped" and out["res"].reason == "stop"
    a._call("shutdown")


def test_disconnect_stops_the_clients_motion(tmp_path):
    for realtime in (False, True):
        d = tmp_path / ("realtime" if realtime else "lockstep")
        d.mkdir()
        addr = start(d, realtime)
        a = robot.connect(addr)
        m = a.move(**slow_move(a), wait=False)
        a.step(5)
        a.close()
        time.sleep(0.3)
        b = robot.connect(addr)
        st = b._call("status", id=m.id)
        assert st["finished"] and st["result"]["reason"] == "client_disconnected", st
        q1 = b.state().q(S.ARM_JOINTS["left"])
        b.step(10)
        assert np.abs(b.state().q(S.ARM_JOINTS["left"]) - q1).max() < 0.02, "the arm kept moving"
        b._call("shutdown")


def test_step_until_and_a_camera_window(tmp_path):
    addr = start(tmp_path)
    r = robot.connect(addr)
    t0 = r.state().t
    s = r.step_until(t0 + 0.1)
    assert abs(s.t - (t0 + 0.1)) < 0.0041
    assert abs(r.step_until(t0).t - s.t) < 1e-9, "a time already passed returns at once"
    full = r.cameras(["top"])["top"]
    win = r.cameras(["top"], roi=[100, 50, 300, 250])["top"]
    assert win.rgb.shape == (200, 200, 3) and win.depth.shape == (200, 200) and win.roi == [100, 50, 300, 250]
    assert np.abs(win.rgb.astype(int) - full.rgb[50:250, 100:300]).mean() < 2.0       # two renders: a few pixels differ
    near = win.depth < 5.0
    assert np.allclose(win.depth[near], full.depth[50:250, 100:300][near], atol=1e-3)
    pw = win.points(near)
    pf = full.points(np.pad(near, ((50, 230), (100, 340))))
    assert len(pw) == len(pf) and np.abs(pw - pf).max() < 0.01
    r._call("shutdown")


def test_ik_query(tmp_path):
    """r.ik solves without moving: its answer, commanded as joints, puts the TCP at the target; a target out of reach
    comes back ok=False with the miss."""
    from robot import pose
    addr = start(tmp_path)
    r = robot.connect(addr)
    q_before = r.state().q(S.ARM_JOINTS["left"])
    target = pose([0.32, 0.25, 0.12], rpy=[np.pi, 0, np.pi])
    sol = r.ik(target, ee="left")
    assert sol["ok"] and sol["joints"] == S.ARM_JOINTS["left"] and sol["q"].shape == (6,)
    assert np.allclose(r.state().q(S.ARM_JOINTS["left"]), q_before), "the query moved the arm"
    r.move(sol["q"], joints=sol["joints"])
    assert np.linalg.norm(r.T("left")[:3, 3] - target[:3, 3]) < 0.01
    far = r.ik(pose([1.2, 0.3, 0.1], rpy=[np.pi, 0, np.pi]), ee="left")
    assert not far["ok"] and far["error_m"] > 0.1
    r._call("shutdown")


def test_an_aborted_move_raises(tmp_path):
    """A motion that cannot go on raises Aborted (here a then= move refused as it fires); the Result rides along."""
    import robot
    from robot import when
    r = robot.connect(start(tmp_path))
    far = r.T("left").copy()
    far[:3, 3] += [0.9, 0.0, 0.0]
    with pytest.raises(robot.Aborted) as e:
        r.move_at({"z": 0.01}, ee="left", max_travel=0.05, until={"soon": when.time() > 0.05},
                  then=robot.Move(far, ee="left"))
    assert e.value.result.status == "aborted" and "InvalidCommand" in e.value.result.reason


def test_reachable_free_and_dry_run_over_rpc(tmp_path):
    import robot
    r = robot.connect(start(tmp_path))
    T = robot.pose([0.40, 0.30, 0.05], rpy=[np.pi, 0, 0.0])
    full, free = r.reachable([T], "left"), r.reachable([T], "left", free="yaw")
    assert not full[0]["ok"] and free[0]["ok"] and free[0]["T"].shape == (4, 4)
    assert set(free[0]["margin"]) == {"limit_rad", "limit_joint", "singular_m"}
    plan = r.move(T, ee="left", free="yaw", dry_run=True)
    assert plan.ok and plan.goal["left"].shape == (4, 4)
    res = r.move(T, ee="left", free="yaw")
    assert res.goal is not None and np.linalg.norm(r.T("left")[:3, 3] - T[:3, 3]) < 0.006


def test_reachable_says_which_candidates_a_move_would_refuse_and_why(tmp_path):
    """One call sorts candidates as r.move would: accepted, out of IK's reach, into the other arm, a line that fails."""
    import robot
    r = robot.connect(start(tmp_path, separation=0.02))
    down = dict(rpy=[np.pi, 0, np.pi])
    r.move(robot.pose([0.40, -0.06, 0.10], **down), ee="right", free="yaw")
    cands = [robot.pose([0.40, 0.30, 0.08], **down), robot.pose([1.20, 0.30, 0.08], **down),
             robot.pose([0.40, 0.10, 0.10], **down)]
    out = r.reachable(cands, "left", free="yaw")
    assert out[0]["ok"] and out[0]["why"] is None and out[0]["q"].shape == (6,)
    assert not out[1]["ok"] and "unreachable" in out[1]["why"]
    assert not out[2]["ok"] and "separation" in out[2]["why"]
    for T, o in zip(cands, out):                       # what r.move then does agrees
        if o["ok"]:
            assert r.move(T, ee="left", free="yaw", dry_run=True).ok
        else:
            with pytest.raises(robot.InvalidCommand):
                r.move(T, ee="left", free="yaw")
    assert r.reachable([cands[0]], "left", free="yaw", line=True)[0]["ok"]
