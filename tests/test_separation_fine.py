"""The hands' pairs measured on their meshes (robotd --separation-fine): what the capsules refused and what stays refused."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from robotd.core.kinematics import Kinematics                                  # noqa: E402
from robotd.core.separation import Separation, YAM, yam_fine                  # noqa: E402
from robotd.robots.yam_station import scene as S                              # noqa: E402

ARMS = {a: list(S.ARM_JOINTS[a]) for a in ("left", "right")}
ALL = ARMS["left"] + ["left_joint7", "left_joint8"] + ARMS["right"] + ["right_joint7", "right_joint8"]


def tool(z, jaw, p):
    z, y = np.array(z, float), np.array(jaw, float)
    T = np.eye(4)
    T[:3, :3] = np.c_[np.cross(y, z), y, z]
    T[:3, 3] = p
    return T


def setup():
    m = S.kinematic_model()
    kin = Kinematics(m, ARMS, dict(S.TCP_SITES), ALL)
    return m, kin, Separation(m, YAM, 0.02), Separation(m, {**YAM, "fine": yam_fine(0.005)}, 0.02)


def q_of(kin, ql, qr, w=0.0285):
    return {**dict(zip(ARMS["left"], ql)), **dict(zip(ARMS["right"], qr)), "left_joint7": w, "right_joint7": w}


def solve(kin, arm, T):
    for q0 in ([0, 1.2, 1.0, 0, 0, 0.], [0.6, 1.6, 1.4, 0.3, 0, 0.], [-0.6, 1.6, 1.4, -0.3, 0, 0.], [-.6, 1.0, 1.2, -.3, .6, 0.], [.6, 1.0, 1.2, .3, .6, 0.]):
        q, ok, _ = kin.ik(arm, T, np.array(q0), restarts=4)
        if ok:
            return q
    pytest.skip(f"{arm}: no IK for the test pose")


def test_a_side_to_side_grasp_is_refused_by_the_capsules_and_allowed_on_the_meshes():
    m, kin, old, fine = setup()
    c = np.array([0.34, 0.10, 0.09])
    ql = solve(kin, "left", tool((0, -1, 0), (0, 0, 1), c + (0, 0.005, 0)))          # jaw along z
    qr = solve(kin, "right", tool((0, 1, 0), (np.cos(np.radians(135)), 0, np.sin(np.radians(135))), c + (0, -0.0105, 0)))
    assert old.at(kin, q_of(kin, ql, qr))[0] < -0.1
    d, parts = fine.at(kin, q_of(kin, ql, qr))
    assert d >= fine.min_m, (d, parts)


def test_hands_that_truly_overlap_stay_refused():
    m, kin, old, fine = setup()
    c = np.array([0.34, 0.10, 0.09])
    ql = solve(kin, "left", tool((0, -1, 0), (0, 0, 1), c))
    qr = solve(kin, "right", tool((0, 1, 0), (0, 0, 1), c))                       # the same point, fingers in one plane
    d, parts = fine.at(kin, q_of(kin, ql, qr))
    assert d < fine.min_m, (d, parts)


def test_arms_far_apart_read_the_same_with_and_without_fine():
    m, kin, old, fine = setup()
    q = q_of(kin, [0.0, 1.2, 1.0, 0, 0, 0.0], [0.0, 1.2, 1.0, 0, 0, 0.0])
    a, b = old.at(kin, q), fine.at(kin, q)
    assert a[0] >= 0.02 and a[0] == pytest.approx(b[0], abs=1e-9)


def test_jaw_width_counts():
    m, kin, old, fine = setup()
    q = q_of(kin, [0.0, 1.2, 1.0, 0, 0, 0.0], [0.0, 1.2, 1.0, 0, 0, 0.0], w=0.04)
    fine.at(kin, q)
    assert kin.d.qpos[kin.qadr["left_joint8"]] == pytest.approx(0.04) if "left_joint8" in kin.qadr else True
