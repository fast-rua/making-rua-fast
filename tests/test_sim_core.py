"""Integration tests of the control core on the MuJoCo YAM station (in process, no RPC)."""
import numpy as np
import pytest

from robot import Compliance, Gripper, pose, when
from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.core.controller import Controller, Fault, InvalidCommand
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks

DOWN = dict(rpy=[np.pi, 0, np.pi])          # jaws straight down (reachable up to ~0.10 m above the table)


def make(scene="empty"):
    cfg = tasks.get(scene).scene()
    b = MujocoYamBackend(cfg)
    return b, Controller(b, description())


def spec(**kw):
    """Plain dicts as the client sends them (conditions as JSON nodes)."""
    out = {}
    for k, v in kw.items():
        out[k] = v.node if hasattr(v, "node") else v
    if isinstance(out.get("until"), dict):
        out["until"] = {k: (v.node if hasattr(v, "node") else v) for k, v in out["until"].items()}
    return out


def run(c, s, tmax=15.0):
    cmd = c.submit(s)
    t0 = c.t
    while not cmd.chain_finished() and c.t - t0 < tmax:
        c.tick()
    assert cmd.chain_finished(), f"did not finish in {tmax}s"
    return cmd.last().result


def ticks(c, seconds):
    for _ in range(int(seconds / 0.004)):
        c.tick()


def test_line_time_scaling_stays_in_range():
    """Near its end a straight-line move's quintic time scaling rounded a hair above 1, which the rotation slerp
    rejected; the failing command then stayed active and wedged every later command (found replaying a towel run)."""
    from robotd.core.references import PoseLine
    from scipy.spatial.transform import Rotation
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_euler("z", 0.5).as_matrix()
    T1[:3, 3] = [0.02, 0.0, -0.03]
    line = PoseLine(np.eye(4), T1, 1.0)
    for s in np.linspace(0.999, 1.0, 20001):            # 196 of these overshoot 1 before clamping
        line.step(float(s))


def test_hold_under_gravity():
    b, c = make()
    q0 = c.m.q.copy()
    ticks(c, 2.0)
    assert np.abs(c.m.q - q0).max() < 1e-3


def test_joint_move_reaches_goal():
    b, c = make()
    goal = [0.2, 1.2, 1.0, -0.6, 0.1, 0.0]
    r = run(c, spec(op="move_to", joints=S.ARM_JOINTS["left"], target=goal))
    assert r["status"] == "done" and r["reason"] == "reached"
    assert np.abs(c.m.q[:6] - goal).max() < 0.01


def test_pose_move_and_line_straightness():
    b, c = make()
    T = pose([0.36, 0.30, 0.10], **DOWN)
    r = run(c, spec(op="move_to", ee="left", target=T.tolist()))
    assert r["status"] == "done"
    assert np.linalg.norm(c.T["left"][:3, 3] - T[:3, 3]) < 0.006
    start = c.T["left"][:3, 3].copy()
    T2 = T.copy()
    T2[:3, 3] += [0.08, -0.05, 0.0]
    cmd = c.submit(spec(op="move_to", ee="left", target=T2.tolist(), line=True))
    dev = 0.0
    while not cmd.finished:
        c.tick()
        p = c.T["left"][:3, 3]
        u = (T2[:3, 3] - start) / np.linalg.norm(T2[:3, 3] - start)
        dev = max(dev, np.linalg.norm((p - start) - ((p - start) @ u) * u))
    assert np.linalg.norm(c.T["left"][:3, 3] - T2[:3, 3]) < 0.006
    assert dev < 0.010, f"line deviated {dev * 1000:.1f} mm"


def test_unreachable_and_bad_names_are_invalid():
    b, c = make()
    with pytest.raises(InvalidCommand, match="unreachable"):
        c.submit(spec(op="move_to", ee="left", target=pose([0.9, 0.3, 0.5], **DOWN).tolist()))
    with pytest.raises(InvalidCommand, match="unknown joints"):
        c.submit(spec(op="move_to", joints=["elbow"], target=[0.0]))


def test_replacement_is_smooth():
    b, c = make()
    js = S.ARM_JOINTS["left"]
    first = c.submit(spec(op="move_to", joints=js, target=[0.5, 1.4, 1.2, -0.6, 0.0, 0.0], wait=False))
    ticks(c, 0.3)
    dq_before = c.dq_ref[:6].copy()
    second = c.submit(spec(op="move_to", joints=js, target=[-0.3, 1.0, 0.9, -0.4, 0.0, 0.0]))
    c.tick()
    assert first.status == "replaced"
    assert np.abs(c.dq_ref[:6] - dq_before).max() < 0.1, "velocity jumped at replacement"
    t0 = c.t
    while not second.finished and c.t - t0 < 10:
        c.tick()
    assert np.abs(c.m.q[:6] - [-0.3, 1.0, 0.9, -0.4, 0.0, 0.0]).max() < 0.01


def test_touch_stops_on_the_table():
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.36, 0.30, 0.08], **DOWN).tolist()))
    run(c, spec(op="gripper", name="left", width=0.0))
    c.zero_force("left")
    r = run(c, spec(op="move_at", ee="left", velocity={"z": -0.02}, max_travel=0.12,
                    until={"contact": when.force("left", "-z") > 3}))
    assert r["status"] == "stopped" and r["reason"] == "contact"
    assert 0.0 < c.T["left"][2, 3] < 0.03


def test_move_at_default_max_travel():
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.30, 0.30, 0.10], **DOWN).tolist()))
    r = run(c, spec(op="move_at", ee="left", velocity={"x": 0.05}, max_travel=0.05))
    assert r["reason"] == "max_travel"
    assert abs(c.T["left"][0, 3] - 0.35) < 0.008


def test_press_force_and_contact_loss_bound():
    b, c = make("press_box")
    run(c, spec(op="move_to", ee="right", target=pose([0.35, -0.30, 0.16], **DOWN).tolist()))
    run(c, spec(op="gripper", name="right", width=0.0))
    ticks(c, 0.2)
    c.zero_force("right")
    run(c, spec(op="move_at", ee="right", velocity={"z": -0.02}, max_travel=0.12,
                until={"contact": when.force("right", "-z") > 2}))
    z_contact = c.T["right"][2, 3]
    comp = Compliance(frame="world", stiffness={"z": 500}, force={"-z": 5.0}, max_penetration=0.03)
    cmd = c.submit(spec(op="follow", ee="right", points=[c.T["right"].tolist()], times=[4.0], compliance=dict(comp)))
    fz, zs = [], []
    while not cmd.finished:
        tr = c.t - cmd.t0
        if 2.0 < tr and "gone" not in cmd.__dict__:
            b.set_object_pose("box", [0.35, -0.75, 0.05])
            cmd.gone = True
        c.tick()
        if 1.0 < tr < 2.0:
            fz.append(-c.wrench("right")[2])
        if tr > 2.0:
            zs.append(c.T["right"][2, 3])
    assert abs(np.mean(fz) - 5.0) < 1.0, f"pressing {np.mean(fz):.2f} N"
    assert z_contact - min(zs) < 0.03 + 0.005, f"dropped {1000 * (z_contact - min(zs)):.0f} mm after removal"


def test_grasp_and_lift_cube():
    b, c = make("pick_cube")
    run(c, spec(op="gripper", name="left", width=0.07))
    run(c, spec(op="move_to", ee="left", target=pose([0.36, 0.28, 0.10], **DOWN).tolist()))
    c.zero_force("left")
    run(c, spec(op="move_at", ee="left", velocity={"z": -0.03}, max_travel=0.15,
                until={"contact": when.force("left", "-z") > 3}))
    run(c, spec(op="follow", ee="left", points=[pose([0, 0, 0.008]).tolist()], times=[0.3], relative=True))
    g = run(c, spec(op="gripper", name="left", width=0.0, force=15))
    assert g["reason"] == "stalled" and abs(g["width"] - 0.04) < 0.004
    run(c, spec(op="follow", ee="left", points=[pose([0, 0, 0.12]).tolist()], times=[1.5], relative=True))
    ticks(c, 0.3)
    assert b.object_pose("cube")[2, 3] > 0.12


def test_two_arm_timeline_is_synchronised():
    b, c = make()
    r = run(c, spec(op="follow", ee=["left", "right"], times=[1.0],
                    points={"left": [pose([0.0, 0.0, -0.05]).tolist()], "right": [pose([0.0, 0.0, -0.05]).tolist()]},
                    relative=True))
    assert r["status"] == "done"
    assert abs((c.T["left"][2, 3]) - (c.T["right"][2, 3])) < 0.004


def test_on_trigger_opens_gripper_mid_motion():
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.30, 0.30, 0.10], **DOWN).tolist()))
    run(c, spec(op="gripper", name="left", width=0.0))
    s = spec(op="follow", ee="left", points=[pose([0.06, 0.0, 0.0]).tolist()], times=[1.2], relative=True)
    s["on"] = {"release": [(when.progress() > 0.5).node, Gripper("left", 0.06)]}
    cmd = c.submit(s)
    widths = []
    while not cmd.finished:
        c.tick()
        widths.append((cmd.ref is None and c._progress(cmd), c.gripper_state["left"]["width"]))
    assert "release" in cmd.on_fired
    assert c.gripper_state["left"]["width"] > 0.03


def test_tcp_offset_rotates_about_the_point():
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.32, 0.30, 0.09], **DOWN).tolist()))
    tcp = pose([0.0, 0.0, 0.05])                         # a point 5 cm ahead of the TCP (a tool tip)
    tip0 = (c.T["left"] @ tcp)[:3, 3].copy()
    angles = np.radians(np.linspace(3, 20, 6))
    pts = [pose(rpy=[0.0, a, 0.0]).tolist() for a in angles]
    r = run(c, spec(op="follow", ee="left", points=pts, times=list(np.linspace(0.3, 1.8, 6)), relative=True, tcp=tcp.tolist()))
    tip1 = (c.T["left"] @ tcp)[:3, 3]
    assert r["status"] == "done"
    assert np.linalg.norm(tip1 - tip0) < 0.006, f"tool tip moved {1000 * np.linalg.norm(tip1 - tip0):.1f} mm"


def test_max_force_latches_a_fault():
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.36, 0.30, 0.06], **DOWN).tolist()))
    run(c, spec(op="gripper", name="left", width=0.0))
    c.zero_force("left")
    comp = Compliance(frame="world", stiffness={"z": 2000}, max_force=4.0)
    s = spec(op="follow", ee="left", points=[pose([0, 0, -0.10]).tolist()], times=[3.0], relative=True,
             compliance=dict(comp))
    cmd = c.submit(s)
    t0 = c.t
    while not cmd.finished and c.t - t0 < 5:
        c.tick()
    assert cmd.status == "aborted" and c.fault is not None
    with pytest.raises(Fault):
        c.submit(spec(op="move_to", ee="left", target=pose([0.36, 0.30, 0.08], **DOWN).tolist()))
    c.recover()
    run(c, spec(op="move_to", ee="left", target=pose([0.36, 0.30, 0.08], **DOWN).tolist()))


def test_ending_a_press_is_not_a_step():
    b, c = make("press_box")
    run(c, spec(op="move_to", ee="right", target=pose([0.35, -0.30, 0.16], **DOWN).tolist()))
    run(c, spec(op="gripper", name="right", width=0.0))
    c.zero_force("right")
    run(c, spec(op="move_at", ee="right", velocity={"z": -0.02}, max_travel=0.12,
                until={"contact": when.force("right", "-z") > 2}))
    comp = Compliance(frame="world", stiffness={"z": 500}, force={"-z": 5.0})
    run(c, spec(op="follow", ee="right", points=[c.T["right"].tolist()], times=[1.5], compliance=dict(comp)))
    before = c.q_cmd.copy()
    ticks(c, 0.1)
    assert np.abs(c.q_cmd - before).max() < 2e-3, "the command jumped when the press ended"
    assert -c.wrench("right")[2] > 3.0, "the finished press should keep pressing until the next command"
    z0 = c.T["right"][2, 3]
    r = run(c, spec(op="follow", ee="right", points=[pose([0, 0, 0.05]).tolist()], times=[1.0], relative=True))
    # the press left the reference ~1 cm inside the box, so the tool itself rises ~4 cm of the 5
    assert r["status"] == "done" and c.T["right"][2, 3] - z0 > 0.035 and abs(c.wrench("right")[2]) < 2.0


def test_too_fast_joint_path_is_refused():
    b, c = make()
    js = S.ARM_JOINTS["left"]
    q0 = c.m.q[:6]
    with pytest.raises(InvalidCommand, match="stretch the times"):
        c.submit(spec(op="follow", joints=js, points=[(q0 + [1.0, 0, 0, 0, 0, 0]).tolist()], times=[0.1]))
    with pytest.raises(InvalidCommand, match="increasing"):
        c.submit(spec(op="follow", joints=js, points=[q0.tolist(), q0.tolist()], times=[0.5, 0.5]))
    r = run(c, spec(op="follow", joints=js, points=[(q0 + [0.3, 0, 0, 0, 0, 0]).tolist()], times=[1.0]))
    assert r["status"] == "done"


def test_follow_meets_a_pose_at_a_time_with_a_velocity():
    """Cartesian waypoints with velocities: the reference passes the waypoint at its time with that velocity."""
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.32, 0.30, 0.12], **DOWN).tolist()))
    T0 = c.kin.fk("left", c._arm(c.q_ref, "left"))
    hit, up = T0.copy(), np.array([0.0, 0.0, 0.25])
    hit[:3, 3] += [0.0, 0.0, 0.03]
    end = hit.copy()
    end[:3, 3] += [0.0, 0.0, 0.02]
    cmd = c.submit(spec(op="follow", ee="left", points=[hit.tolist(), end.tolist()], times=[0.4, 0.6],
                        velocities=[up.tolist(), [0, 0, 0]]))
    prev = None
    while not cmd.finished:
        c.tick()
        T = c.kin.fk("left", c._arm(c.q_ref, "left"))
        if prev is not None and abs(c.t - cmd.t0 - 0.4) < 0.002:
            v = (T[:3, 3] - prev[:3, 3]) / 0.004
            assert np.linalg.norm(T[:3, 3] - hit[:3, 3]) < 0.002 and np.linalg.norm(v - up) < 0.03, (T[:3, 3], v)
        prev = T
    with pytest.raises(InvalidCommand, match="velocities"):
        c.submit(spec(op="follow", ee="left", points=[hit.tolist()], times=[0.4], velocities=[[0, 0, 0, 0]]))


def test_replacing_a_moving_path_keeps_its_velocity_and_start_at_waits():
    """A follow that replaces a moving one starts at the current reference velocity; with start_at the running one
    keeps going until then."""
    b, c = make()
    run(c, spec(op="move_to", ee="left", target=pose([0.32, 0.30, 0.12], **DOWN).tolist()))
    far = pose([0.0, 0.10, 0.0]).tolist()
    first = c.submit(spec(op="follow", ee="left", points=[far], times=[1.0], relative=True))
    ticks(c, 0.4)
    v_before = c.kin.jac("left", c._arm(c.q_ref, "left"))[:3] @ c._arm(c.dq_ref, "left")
    later = c.t + 0.1
    second = c.submit(spec(op="follow", ee="left", points=[pose([0.32, 0.36, 0.12], **DOWN).tolist()], times=[0.6],
                           start_at=later))
    ticks(c, 0.05)
    assert first.status == "running" and second.status == "pending", "start_at replaced the running path early"
    while c.t < later - 1e-9:
        c.tick()
    v_at = c.kin.jac("left", c._arm(c.q_ref, "left"))[:3] @ c._arm(c.dq_ref, "left")
    c.tick()
    v_after = c.kin.jac("left", c._arm(c.q_ref, "left"))[:3] @ c._arm(c.dq_ref, "left")
    assert first.status == "replaced" and second.status == "running"
    assert np.linalg.norm(v_before) > 0.05 and np.linalg.norm(v_after - v_at) < 0.02, (v_at, v_after)


def test_inertia_feedforward_tracks_a_quick_joint_path():
    """A 2 Hz joint sine: without the feed-forward the soft servos lag by ~0.09 rad RMS, with it ~0.02."""
    errs = {}
    for ff in (False, True):
        b, c = make()
        c.feedforward = ff
        js = S.ARM_JOINTS["left"]
        q0 = c.m.q[:6].copy()
        ts = np.arange(0.02, 1.6, 0.02)
        w, A = 2 * np.pi * 2.0, 0.15
        pts = [(q0 + [0, A * np.sin(w * t), -A * np.sin(w * t), 0, 0, 0]).tolist() for t in ts]
        vel = [[0, A * w * np.cos(w * t), -A * w * np.cos(w * t), 0, 0, 0] for t in ts]
        cmd = c.submit(spec(op="follow", joints=js, points=pts, times=ts.tolist(), velocities=vel))
        e = []
        while not cmd.finished:
            c.tick()
            if c.t - cmd.t0 > 0.8:
                e.append(c.m.q[1:3] - c.q_ref[1:3])
        errs[ff] = float(np.sqrt(np.mean(np.square(e))))
    assert errs[True] < 0.4 * errs[False], errs


def test_a_refused_on_command_aborts_its_command_not_the_tick():
    """An on= command refused as it fires (here a joint path far too fast for its joints) ends the command that fired
    it with the refusal as its reason; the arm holds and later commands run."""
    b, c = make()
    js = S.ARM_JOINTS["left"]
    q0 = c.m.q[:6]
    too_fast = {"op": "follow", "joints": js, "points": [(q0 + [1.5, 0, 0, 0, 0, 0]).tolist()], "times": [0.05]}
    s = spec(op="follow", ee="left", points=[pose([0.0, 0.0, 0.02]).tolist()], times=[1.0], relative=True)
    s["on"] = {"jump": [(when.time() > 0.2).node, too_fast]}
    cmd = c.submit(s)
    ticks(c, 0.5)
    assert cmd.status == "aborted" and cmd.result["reason"].startswith("jump: InvalidCommand"), cmd.result
    r = run(c, spec(op="move_to", ee="left", target=pose([0.32, 0.30, 0.12], **DOWN).tolist()))
    assert r["status"] == "done"


def test_gains_and_grip_force_belong_to_their_command():
    """A command's gains= (or the defaults) apply from when it takes over and stay while it holds, until the next
    command on those joints. They used to stick to every later command, and a command waiting for start_at changed
    the gains of the one still running; a grip's force= stuck the same way."""
    b, c = make()
    wrist = S.ARM_JOINTS["left"][4]
    w, default = c.jidx[wrist], c.desc["default_kp"][wrist]
    stiff = {"kp": [80, 80, 80, 30, 30, 30], "kd": [5, 5, 5, 1.5, 1.5, 1.5]}
    T = pose([0.32, 0.30, 0.12], **DOWN).tolist()
    run(c, spec(op="move_to", ee="left", target=T, gains=stiff))
    ticks(c, 0.2)
    assert c.kp[w] == 30
    run(c, spec(op="move_to", ee="left", target=T))
    assert c.kp[w] == default
    cmd = c.submit(spec(op="move_to", ee="left", target=T, gains=stiff, start_at=c.t + 0.2))
    ticks(c, 0.1)
    assert cmd.status == "pending" and c.kp[w] == default
    ticks(c, 0.2)
    assert c.kp[w] == 30
    f, cap = c.jidx[c.finger["left"]], c.desc["effort_limit"][c.finger["left"]]
    run(c, spec(op="gripper", name="left", width=0.0, force=15))
    assert c.effort_limit[f] == min(15.0, cap)
    run(c, spec(op="gripper", name="left", width=0.08))
    assert c.effort_limit[f] == cap


def test_move_is_move_to_for_one_target_and_a_timed_path_for_several():
    """r.move: one target with nothing timing it is a move_to; a list (or times=/velocities=/relative=) is a follow,
    timed from the speed limits when it has no times, both arms on one timeline."""
    from robot import Move, InvalidCommand as ClientInvalid
    b, c = make()
    js = S.ARM_JOINTS["left"]
    A = pose([0.32, 0.30, 0.12], **DOWN)
    q = c.m.q[[c.jidx[n] for n in js]]
    assert Move(A, ee="left")["op"] == "move_to" and Move(q, joints=js)["op"] == "move_to"
    assert Move([A], ee="left")["op"] == "follow" and Move([q, q], joints=js)["op"] == "follow"
    one = Move(A, ee="left", times=1.0)
    assert one["op"] == "follow" and one["points"] == [A.tolist()] and one["times"] == [1.0]
    for v in ([0.1, 0.0, 0.0], [[0.1, 0.0, 0.0]]):        # the servo idiom's velocities=[v], or v bare
        assert Move(A, ee="left", times=[0.08], velocities=v)["velocities"] == [[0.1, 0.0, 0.0]]
        assert Move({"left": A}, ee=["left"], times=[0.08], velocities={"left": v})["velocities"] == \
            {"left": [[0.1, 0.0, 0.0]]}
    tool =Move(pose([0.0, 0.0, 0.01]), ee="left", frame="tool", times=0.5)
    assert tool["relative"] and tool["frame"] == "tool"
    with pytest.raises(ClientInvalid, match="line"):
        Move([A, A], ee="left", line=True)
    run(c, Move(A, ee="left"))
    p0 = c.kin.fk("left", c._arm(c.q_ref, "left"))[:3, 3]
    legs = [[0.04, 0.0, 0.03], [0.08, -0.02, 0.0]]
    cmd = c.submit(Move([pose(d) for d in legs], ee="left", relative=True))
    prev, peak, near = p0, 0.0, 1.0
    while not cmd.finished:
        c.tick()
        p = c.kin.fk("left", c._arm(c.q_ref, "left"))[:3, 3]
        peak = max(peak, np.linalg.norm(p - prev) / 0.004)
        near = min(near, np.linalg.norm(p - (p0 + legs[0])))
        prev = p
    assert near < 0.003 and np.linalg.norm(p - (p0 + legs[1])) < 0.003, (near, p - p0)
    assert peak < 1.5 * c.desc["linear_speed"], peak
    cmd = c.submit(Move({a: [pose([0.0, 0.0, -0.03]), pose([0.03, 0.0, -0.03])] for a in ("left", "right")},
                        ee=["left", "right"], relative=True, speed=0.05))
    assert list(cmd.cart["left"].gen.tm) == list(cmd.cart["right"].gen.tm) and cmd.cart["left"].gen.tm[-1] > 1.0
    while not cmd.finished:
        c.tick()
    q1 = q + np.array([0.3, -0.2, 0.0, 0.0, 0.4, 0.0])
    cmd = c.submit(Move([q + np.array([0.15, -0.1, 0.0, 0.0, 0.2, 0.0]), q1], joints=js))
    idx = [c.jidx[n] for n in js]
    fastest = 0.0
    while not cmd.finished:
        c.tick()
        fastest = max(fastest, float(np.max(np.abs(c.dq_ref[idx]) / c.vmax[idx])))
    assert fastest <= 1.0 + 1e-6 and np.allclose(c.q_ref[idx], q1, atol=1e-3), fastest


def _replace_mid_motion(second, t_rep=0.35):
    """Start a quick joint move, replace it after t_rep s with second(c) (a spec); the reference's worst velocity step
    per tick and acceleration step around the replacement."""
    b, c = make()
    js = S.ARM_JOINTS["left"]
    idx = [c.jidx[n] for n in js]
    c.submit(spec(op="move_to", joints=js, target=(c.m.q[idx] + [0.6, -0.4, 0.5, 0.6, 0.5, 0.8]).tolist()))
    dq, k_rep = [], int(round(t_rep / 0.004))
    for k in range(int(0.8 / 0.004)):
        if k == k_rep:
            c.submit(second(c))
        c.tick()
        dq.append(c.dq_ref[idx].copy())
    acc = np.diff(np.array(dq), axis=0) / 0.004
    v_step = np.abs(np.diff(np.array(dq), axis=0))[k_rep - 1:k_rep + 3].max()
    a_step = np.abs(np.diff(acc, axis=0))[k_rep - 2:k_rep + 2].max()
    return v_step, a_step


def test_every_kind_of_command_replaces_a_moving_one_smoothly():
    """A command sent while the arm moves continues its velocity and acceleration: before, a line, a pose path or a
    move_at started from rest (velocity steps of 2 rad/s in one tick at 0.35 s into a quick move) and a joint goal
    from zero acceleration."""
    js = S.ARM_JOINTS["left"]

    def T1(c):
        T = c.T["left"].copy()
        T[:3, 3] += [0.03, -0.04, 0.02]
        return T.tolist()
    q = lambda c: c.q_ref[[c.jidx[n] for n in js]]
    seconds = {
        "joint goal": lambda c: spec(op="move_to", joints=js, target=(q(c) + 0.2).tolist()),
        "pose goal": lambda c: spec(op="move_to", ee="left", target=T1(c)),
        "line": lambda c: spec(op="move_to", ee="left", target=T1(c), line=True),
        "joint path": lambda c: spec(op="follow", joints=js, points=[(q(c) + 0.1).tolist(), (q(c) + 0.15).tolist()],
                                     times=[0.3, 0.6]),
        "pose path": lambda c: spec(op="follow", ee="left", points=[T1(c)], times=[0.5]),
        "move_at twist": lambda c: spec(op="move_at", ee="left", velocity=[0.05, 0, 0, 0, 0, 0], max_travel=0.05),
        "move_at joints": lambda c: spec(op="move_at", joints=js, velocity=[0.3] * 6, duration=0.4),
    }
    for name, second in seconds.items():
        v_step, a_step = _replace_mid_motion(second)
        assert v_step < 0.15 and a_step < 15.0, (name, v_step, a_step)


def test_a_servo_loop_of_short_paths_has_no_acceleration_kicks():
    """Re-sending a 80 ms pose path every 32 ms toward a circling target: each new path starts where the last one was
    asking (the damped per-tick IK trails it) and blends its acceleration in, so the reference has no kick at each
    re-send (worst acceleration step 15 rad/s^2 before, ~3 now in steady state)."""
    b, c = make()
    idx = [c.jidx[n] for n in S.ARM_JOINTS["left"]]
    Tc = c.T["left"].copy()
    dq = []
    for k in range(25):
        th = 2 * np.pi * k * 0.032 / 1.5
        T = Tc.copy()
        T[:3, 3] += 0.024 * np.array([np.cos(th) - 1, np.sin(th), 0])
        c.submit(spec(op="follow", ee="left", points=[T.tolist()], times=[0.08]))
        for _ in range(8):
            c.tick()
            dq.append(c.dq_ref[idx].copy())
    acc = np.diff(np.array(dq), axis=0) / 0.004
    assert np.abs(np.diff(acc, axis=0))[40:].max() < 5.0


def test_free_yaw_reaches_a_pinch_the_full_pose_cannot():
    """A top-down pinch at yaw 0 out at x 0.40 is out of reach; free='yaw' turns it about the tool axis to one that
    is reachable: same position and approach axis, the chosen pose in the result."""
    b, c = make()
    T = pose([0.40, 0.30, 0.05], rpy=[np.pi, 0, 0.0])
    with pytest.raises(InvalidCommand, match="unreachable"):
        c.prepare(spec(op="move_to", ee="left", target=T.tolist()))
    res = run(c, spec(op="move_to", ee="left", target=T.tolist(), free="yaw"))
    G = np.asarray(res["goal"]["left"])
    assert np.linalg.norm(G[:3, 3] - T[:3, 3]) < 1e-3 and G[:3, 2] @ T[:3, 2] > 0.999
    assert np.linalg.norm(c.T["left"][:3, 3] - T[:3, 3]) < 0.006


def test_free_yaw180_takes_the_nearer_of_the_two_grasps():
    """A parallel grasp and its half-turn are the same grasp: yaw180 picks the one the wrist reaches with less turn."""
    b, c = make()
    q_wrist = c.q_ref[c.jidx["left_joint6"]]
    T = pose([0.30, 0.30, 0.08], rpy=[np.pi, 0, 0.3])
    a = c.solve_ik("left", T.tolist())
    flip = c.solve_ik("left", (T @ np.diag([-1.0, -1.0, 1.0, 1.0])).tolist())
    pick = c.solve_ik("left", T.tolist(), free="yaw180")
    best = min((x for x in (a, flip) if x["ok"]), key=lambda x: np.max(np.abs(np.array(x["q"]) - c._arm(c.q_ref, "left"))))
    assert pick["ok"] and np.allclose(pick["q"], best["q"], atol=0.05), (pick["q"], best["q"], q_wrist)


def test_a_free_path_picks_turns_without_wrist_flips():
    """Five top-down pinches along y with free yaw: the turns chosen change smoothly from point to point."""
    b, c = make()
    pts = [pose([0.36, y, 0.06], rpy=[np.pi, 0, 0.0]).tolist() for y in np.linspace(0.38, 0.22, 5)]
    cmd = c.prepare(spec(op="follow", ee="left", points=pts, times=[0.6, 1.2, 1.8, 2.4, 3.0], free="yaw"))
    G = [np.asarray(T) for T in cmd.resolved["left"]]
    yaw = np.unwrap([np.arctan2(T[1, 0], T[0, 0]) for T in G])
    assert np.max(np.abs(np.diff(yaw))) < np.radians(61), np.degrees(yaw)
    for T, P in zip(G, pts):
        assert np.linalg.norm(T[:3, 3] - np.asarray(P)[:3, 3]) < 1e-3


def test_dry_run_walks_a_line_and_says_where_it_fails():
    b, c = make()
    T0 = c.T["left"]
    near = T0.copy()
    near[:3, 3] += [0.05, 0.0, -0.05]
    ok = c.dry_run(spec(op="move_to", ee="left", target=near.tolist(), line=True))
    assert ok["ok"] and ok["margin"]["left"]["limit_rad"] > 0
    q_before = c.q_ref.copy()
    far = T0.copy()
    far[:3, 3] += [0.9, 0.0, 0.0]
    path = c.dry_run(spec(op="follow", ee="left", points=[near.tolist(), far.tolist()], times=[1.0, 3.0]))
    assert not path["ok"] and path["reason"] == "unreachable" and path["point"] == 1 and 0 < path["fraction"] < 1
    assert np.array_equal(c.q_ref, q_before) and not c.commands          # nothing moved, nothing queued


def test_then_runs_after_a_move_that_arrives():
    """then= is what comes next: it starts when a move arrives, not only when an until= condition fires."""
    b, c = make()
    T = c.T["left"].copy()
    T[:3, 3] += [0.0, 0.0, 0.03]
    res = run(c, spec(op="move_to", ee="left", target=T.tolist(), then=Gripper("left", 0.03)))
    assert res["op"] == "gripper" and res["status"] == "done"


def test_a_planner_path_may_start_with_the_current_state():
    """Planners return paths whose first point is the start: at t=0 (dropped), or as the first of dt-spaced points
    (then point i is at i*dt). A t=0 point elsewhere, or a point outside the joint limits, is refused by name."""
    b, c = make()
    js = S.ARM_JOINTS["left"]
    q0 = c._arm(c.q_ref, "left")
    qs = [q0 + d * np.array([0.2, 0, 0, 0, 0, 0]) for d in np.linspace(0, 1, 11)]
    cmd = c.prepare(spec(op="follow", joints=js, points=[q.tolist() for q in qs], times=np.linspace(0, 1, 11).tolist()))
    assert len(cmd.spec["points"]) == 10 and cmd.spec["times"][0] == pytest.approx(0.1)
    cmd = c.prepare(spec(op="follow", joints=js, points=[q.tolist() for q in qs], dt=0.1))
    assert len(cmd.spec["points"]) == 10 and cmd.spec["times"][-1] == pytest.approx(1.0)
    with pytest.raises(InvalidCommand, match="t=0 must be where the joints are"):
        c.prepare(spec(op="follow", joints=js, points=[(q0 + 0.3).tolist(), (q0 + 0.4).tolist()], times=[0.0, 1.0]))
    lo = c.kin.limits["left_joint1"][0]
    with pytest.raises(InvalidCommand, match="left_joint1 = .* outside its limits"):
        c.prepare(spec(op="follow", joints=js, points=[[lo - 0.2] + list(q0[1:])], times=[2.0]))


def test_a_gripper_stopped_just_short_of_its_target_still_finishes():
    """Jaws that stop 1 mm short (between 'reached' and 'stalled') used to wait forever; still, they now end."""
    from robotd.core.references import GripperRef
    g = GripperRef(0.0, 0.02, 0.1, 0.004)
    for k in range(400):
        g.step(k * 0.004, min(k * 0.0004, 0.019), 0.0 if k * 0.0004 >= 0.019 else 0.1)
        if g.done:
            break
    assert g.done and g.reason == "reached"


def _urdf_fk(path, tip, q):
    """Forward kinematics of a URDF from its root to link `tip` (revolute, prismatic and fixed joints)."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    joints = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
    chain, link = [], tip
    while link in joints:
        chain.append(joints[link])
        link = joints[link].find("parent").get("link")
    T = np.eye(4)
    for j in reversed(chain):
        o = j.find("origin")
        A = np.eye(4)
        A[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
        A[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T = T @ A
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            M = np.eye(4)
            v = q.get(j.get("name"), 0.0)
            if j.get("type") == "prismatic":
                M[:3, 3] = axis * v
            else:
                M[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * v).as_matrix()
            T = T @ M
    return T


def test_generated_urdfs_agree_with_the_driver(tmp_path):
    """The URDFs libraries load reach where the driver does: the bimanual one in the world frame, each arm's from its
    base with T_world_base, at the tool-centre link; limits are the driver's (no placeholder velocity 1)."""
    import xml.etree.ElementTree as ET
    from robotd.robots.yam_station.model_files import write_model_files
    b, c = make()
    desc = description()
    gen = write_model_files(b.kinematics_model(), desc, str(tmp_path))
    rng = np.random.default_rng(1)
    for _ in range(5):
        for a in ("left", "right"):
            names = desc["arm_joints"][a]
            q = {n: rng.uniform(*c.kin.limits[n]) * 0.8 for n in names}
            T = c.kin.fk(a, np.array([q[n] for n in names]))
            assert np.allclose(_urdf_fk(gen["urdf"], f"{a}_tcp", q), T, atol=1e-6)
            arm = gen["arms"][a]
            assert np.allclose(np.asarray(arm["T_world_base"]) @ _urdf_fk(arm["urdf"], arm["tcp_link"], q), T, atol=1e-6)
    lim = {j.get("name"): j.find("limit") for j in ET.parse(gen["urdf"]).getroot().findall("joint")
           if j.find("limit") is not None}
    assert float(lim["left_joint1"].get("velocity")) == desc["velocity_limit"]["left_joint1"]
    assert float(lim["left_joint4"].get("effort")) == desc["effort_limit"]["left_joint4"]


def test_then_is_built_off_the_tick_in_real_time():
    """On a real-time backend a then= is built (IK, free= candidates) by a worker while the arm holds its reference:
    an unreachable one still ends its move aborted with the refusal, a free='yaw' one still runs, stop() drops one
    that is still being built, and no tick waits for any of them (the unreachable IK alone took 0.25 s in the tick)."""
    import sys
    import time
    b = MujocoYamBackend(tasks.get("empty").scene(), realtime=True)
    c = Controller(b, description())
    js = S.ARM_JOINTS["left"]
    idx = [c.jidx[n] for n in js]
    q0 = c.m.q[idx].copy()
    gaps, stopped = [], []

    def run_rt(cmd, on_tick=lambda: None, tmax=8.0):
        t0, t_prev = c.t, time.perf_counter()
        while not cmd.chain_finished() and c.t - t0 < tmax:
            c.tick()
            now = time.perf_counter()
            gaps.append(now - t_prev)
            t_prev = now
            on_tick()
        return cmd.last().result

    switch = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)                          # as robotd sets it for a real robot
    try:
        far = pose([2.0, 0.3, 0.2]).tolist()
        res = run_rt(c.submit(spec(op="move_to", joints=js, target=(q0 + [0.05, 0, 0, 0, 0, 0]).tolist(),
                                   then=spec(op="move_to", ee="left", target=far))))
        assert res["status"] == "aborted" and res["reason"].startswith("reached: InvalidCommand"), res
        T = c.T["left"] @ pose([0.0, 0.0, 0.02], rpy=[0, 0, 0.8])
        yaw = spec(op="move_to", ee="left", target=T.tolist(), free="yaw")
        res = run_rt(c.submit(spec(op="move_to", joints=js, target=q0.tolist(), then=yaw)))
        assert res["status"] == "done" and np.linalg.norm(c.T["left"][:3, 3] - T[:3, 3]) < 0.005, res
        cmd = c.submit(spec(op="move_to", joints=js, target=q0.tolist(), then=yaw))

        def stop_while_building():
            if not stopped and cmd.next is not None and cmd.next.status == "building":
                stopped.append(c.t)
                c.stop()
        res = run_rt(cmd, stop_while_building)
        ticks(c, 0.3)                                      # what was built is dropped: nothing moves the arm
        assert stopped and res["status"] == "stopped" and res["reason"] == "stop", res
        assert all(x.finished for x in c.commands) and np.allclose(c.m.q[idx], q0, atol=0.01)
    finally:
        sys.setswitchinterval(switch)
    assert max(gaps) < 0.1, f"a tick waited {max(gaps):.3f} s"


def test_a_disconnect_stops_what_on_started():
    """A client that goes away stops every command it started, the ones its on= conditions started included (they
    went on running); the arms hold."""
    from robot import Move
    from robotd.server import Client, Driver
    d = Driver(MujocoYamBackend(tasks.get("empty").scene()), description())
    cl = Client()
    js = S.ARM_JOINTS["left"]
    q0 = d.c.m.q[[d.c.jidx[n] for n in js]]
    s = Move(q0 + [0.6, 0, 0, 0, 0, 0], joints=js, speed=0.3, wait=False,
             on={"open": (when.time() > 0.1, Gripper("left", 0.08, speed=0.01))})
    d.handle({"op": "command", "spec": s}, cl)
    d.handle({"op": "step", "ticks": 5}, cl)                 # 0.2 s: the gripper command on= started is running
    grip = next(c for c in d.c.commands if c.kind == "gripper")
    assert grip.status == "running"
    d.release(cl)
    w = d.c.gripper_state["left"]["width"]
    ticks(d.c, 0.4)
    assert all(c.finished for c in d.c.commands) and grip.result["reason"] == "client_disconnected"
    assert abs(d.c.gripper_state["left"]["width"] - w) < 1e-3


def test_an_opening_that_stalls_does_not_close_the_jaws():
    """push_T replay n=1732: an opening that stopped short on the T block ended 'stalled', and the stall hold (meant
    for a grip: the reference 50 mm inside what it holds) then closed the jaws with 50 N. Now an opening that stalls
    keeps its reference at the width asked for, and the jaws stay where they stopped."""
    b, c = make("press_box")                                   # a 10 cm box, 2 kg, its +y face at y = -0.25
    run(c, spec(op="gripper", name="right", width=0.0))
    run(c, spec(op="move_to", ee="right", target=pose([0.35, -0.22, 0.16], **DOWN).tolist()))
    run(c, spec(op="move_to", ee="right", line=True, target=pose([0.35, -0.22, 0.06], **DOWN).tolist()))
    r = run(c, spec(op="gripper", name="right", width=0.08, force=5))     # one jaw opens into the box's face
    assert r["reason"] == "stalled" and r["width"] < 0.077, r
    ticks(c, 0.3)
    assert c.gripper_state["right"]["width"] > r["width"] - 0.002, f"the jaws closed: {c.gripper_state['right']}"
    run(c, spec(op="move_to", ee="right", line=True, target=pose([0.35, -0.22, 0.16], **DOWN).tolist()))
    full = run(c, spec(op="gripper", name="right", width=0.08))
    assert full["reason"] == "reached" and full["width"] > 0.078
