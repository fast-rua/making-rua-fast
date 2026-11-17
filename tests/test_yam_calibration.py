"""The real YAM backend's reading of ORA's cell calibration (and, with i2rt installed, on i2rt's SimRobot)."""
import json

import numpy as np
import pytest

from robotd.backends import yam_i2rt as Y

CAL = {
    "schema": "ora.cell-calibration.v1",
    "cameras": {"overhead": {"cam2world_cv": [[1, 0, 0, 0.3], [0, -1, 0, 0], [0, 0, -1, 0.8], [0, 0, 0, 1]]}},
    "gripper_limits": {"left": [0.1, -2.5], "right": [0.2, -2.4]},
    "rest_reference": {"left": [0, 0.1, 0.2, 0, 0, 0], "right": [0, 0.1, 0.2, 0, 0, 0]},
    "joint_zero_offsets": {"left": [0.0, 0.01, -0.02, 0, 0, 0], "right": {"rad": [0.0, 0.0, 0.03, 0, 0, 0]}},
    "arm_base_corrections": {"right": {"rotvec_rad": [0, 0, 0.01], "translation_m": [0.002, 0, 0]}},
    "wrist_camera_corrections": {"left": {"rotvec_rad": [0.01, 0, 0], "translation_m": [0, 0, 0.001]}},
}


def test_parsing(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(CAL))
    cal = Y.load_calibration(str(p))
    off = Y.joint_offsets(cal)
    assert np.allclose(off["left"][:3], [0, 0.01, -0.02]) and np.allclose(off["right"][2], 0.03)
    assert Y._per_arm("0.1,-2.5", 2) == {"left": [0.1, -2.5], "right": [0.1, -2.5]}
    assert Y._per_arm("left=1,2;right=3,4", 2)["right"] == [3.0, 4.0]
    with pytest.raises(ValueError):
        Y._per_arm("left=1,2,3", 2)
    (tmp_path / "bad.json").write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError):
        Y.load_calibration(str(tmp_path / "bad.json"))


def test_backend_applies_the_calibration(tmp_path):
    pytest.importorskip("i2rt")
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(CAL))
    b = Y.YamI2rtBackend({"left": "can_left", "right": "can_right"}, sim=True, calibration=str(p))
    try:
        raw = np.asarray(b.robots["left"].get_observations()["joint_pos"][:6])
        assert np.allclose(b.read().q[:6], raw + [0, 0.01, -0.02, 0, 0, 0])
        assert np.allclose(b.park_q["left"], np.array([0, 0.1, 0.2, 0, 0, 0]) + [0, 0.01, -0.02, 0, 0, 0])
        b.camera_serials = {"top": "x"}
        assert np.allclose(b._camera_pose("top")[:3, 3], [0.3, 0, 0.8])
        import mujoco
        from scipy.spatial.transform import Rotation
        from robotd.robots.yam_station import scene as S

        def base_world(m):
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
            return {a: d.body(f"{a}_base").xpos.copy() for a in ("left", "right")}
        cad, corrected = base_world(S.kinematic_model()), base_world(b.kinematics_model())
        assert np.allclose(corrected["left"], cad["left"])
        want = Rotation.from_rotvec([0, 0, 0.01]).as_matrix() @ cad["right"] + [0.002, 0, 0]
        assert np.allclose(corrected["right"], want, atol=1e-9)
    finally:
        b.close()


def _sim_backend(tmp_path, cal):
    pytest.importorskip("i2rt")
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"schema": "ora.cell-calibration.v1", **cal}))
    return Y.YamI2rtBackend({"left": "can_left", "right": "can_right"}, sim=True, calibration=str(p))


def test_a_joint_starting_outside_its_limit_is_not_stepped_in(tmp_path):
    """A zero offset can put a resting joint just outside its model limit (left joint 3 at -0.02 rad, the limit is 0):
    the command starts where the joint is (it stepped to the limit on the first tick) and moves in with the reference;
    inside, the limit applies again."""
    from robotd.core.controller import Controller
    from robotd.robots.yam_station import scene as S
    from robotd.robots.yam_station.description import description
    b = _sim_backend(tmp_path, {"joint_zero_offsets": {"left": [0, 0, -0.02, 0, 0, 0]}})
    try:
        c = Controller(b, description(0.5))
        j3 = c.jidx["left_joint3"]
        assert c.m.q[j3] < c.lo[j3] - 0.019
        for _ in range(5):
            c.tick()
        assert abs(c.q_cmd[j3] - c.lo[j3] + 0.02) < 1e-6 and abs(c.m.q[j3] - c.lo[j3] + 0.02) < 1e-6
        cmd = c.submit({"op": "move_to", "joints": S.ARM_JOINTS["left"], "target": [0.0, 0.3, 0.3, 0.0, 0.0, 0.0]})
        steps, prev = [], c.q_cmd[j3]
        while not cmd.finished:
            c.tick()
            steps.append(abs(c.q_cmd[j3] - prev))
            prev = c.q_cmd[j3]
        assert max(steps) < 0.01 and abs(c.m.q[j3] - 0.3) < 1e-3, max(steps)
        c.tick()
        assert c.lo_cmd[j3] == c.lo[j3]
    finally:
        b.close()


def _latched_driver(tmp_path):
    """A real-time driver on SimRobot whose effort guard keeps latching: SimRobot reads ~30 Nm on a joint held past
    its model range (left joint 3 at its limit with a +0.02 rad zero offset), so it latches again right after every
    recover. Returns the backend, the driver and the rest pose."""
    from robotd.robots.yam_station import scene as S
    from robotd.robots.yam_station.description import description
    from robotd.server import Driver
    rest = [0.0, 0.1, 0.12, 0.0, 0.0, 0.0]
    b = _sim_backend(tmp_path, {"joint_zero_offsets": {"left": [0, 0, 0.02, 0, 0, 0]},
                                "rest_reference": {"left": rest, "right": rest}})
    d = Driver(b, description(0.5))
    with d.lock:
        d.c.submit({"op": "move_to", "joints": S.ARM_JOINTS["left"], "target": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]})
    assert d.advance_until(lambda: d.c.fault is not None, 10.0) and "effort" in d.c.fault["reason"]
    with d.lock:
        d.c.recover()
    assert d.advance_until(lambda: d.c.fault is not None, 1.0)             # it latches again at once
    return b, d, rest


def _parked(b, rest):
    q = b.read().q
    for k, arm in enumerate(("left", "right")):
        assert np.allclose(q[7 * k:7 * k + 6], np.array(rest) + b.offsets[arm], atol=2e-3), (arm, q[7 * k:7 * k + 6])


def test_exit_park_goes_through_a_guard_that_keeps_latching(tmp_path):
    """Stopping the driver parks both arms although the effort guard keeps latching: guards stop programs, and the
    park is the operator's way out (without it the arms drop)."""
    b, d, rest = _latched_driver(tmp_path)
    d.close()
    _parked(b, rest)
    assert any(e["text"].endswith("parking on") for e in d.events)


def test_dashboard_park_goes_through_a_guard_that_keeps_latching(tmp_path):
    """The dashboard's PARK is the operator's way out too: it clears the guard's fault and parks both arms although
    the guard would latch again on the way (it warns); afterwards the guard is back. It parks after the operator's
    stop too (an arm stopped pressing on something must be able to get off it without a RELEASE first)."""
    from robotd.server import Client
    b, d, rest = _latched_driver(tmp_path)
    dash = Client()
    try:
        ids = d.handle({"op": "park"}, dash)["ids"]
        assert d.advance_until(lambda: all(d.cmds[i].chain_finished() for i in ids), 15.0)
        assert all(d.cmds[i].result["status"] == "done" for i in ids) and d.c.fault is None
        _parked(b, rest)
        assert any(e["text"].endswith("parking on") for e in d.events)
        assert not any(c.park and not c.finished for c in d.c.commands)
        d.handle({"op": "operator_stop", "soft": False}, dash)
        ids = d.handle({"op": "park"}, dash)["ids"]
        assert d.advance_until(lambda: all(d.cmds[i].chain_finished() for i in ids), 15.0)
        _parked(b, rest)
        d.handle({"op": "release"}, dash)
    finally:
        d.close()


def test_the_description_mjcf_has_the_calibrated_bases(tmp_path):
    """A cell whose calibration moves a base gets an MJCF with the base where the driver has it (the dashboard and
    the agents' libraries load it); the simulation keeps i2rt's file."""
    import mujoco
    from robotd.dashboard.model import DisplayModel
    from robotd.robots.yam_station import scene as S
    from robotd.robots.yam_station.description import description
    from robotd.robots.yam_station.model_files import with_model_files
    cal = {"arm_base_corrections": {"right": {"rotvec_rad": [0.0, 0.02, -0.03], "translation_m": [-0.01, 0.1, 0.015]}}}
    km = Y.kinematics_model_for(cal)
    desc = with_model_files(description(), km)
    assert desc["files"]["mjcf"] != S.STATION_XML
    assert with_model_files(description(), S.kinematic_model())["files"]["mjcf"] == S.STATION_XML
    joints = [km.joint(i).name for i in range(km.njnt)]
    dm = DisplayModel(desc["files"]["mjcf"], joints, list(desc["arm_joints"]), desc["tcp_sites"])
    kd = mujoco.MjData(km)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = [rng.uniform(*km.jnt_range[i]) if km.jnt_limited[i] else 0.0 for i in range(km.njnt)]
        kd.qpos[[km.jnt_qposadr[i] for i in range(km.njnt)]] = q
        mujoco.mj_kinematics(km, kd)
        dm._set(q)
        for a in desc["arm_joints"]:
            site = desc["tcp_sites"][a]
            assert np.linalg.norm(kd.site(site).xpos - dm.d.site(site).xpos) < 1e-6
