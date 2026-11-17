"""robot.viz: projection, the ghost arm (its placement, the calibration fit, hidden parts) and show()'s files.

A two-joint 'arm' built here stands in for the station's MJCF, so the test needs MuJoCo and OpenCV but no i2rt."""
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")

from robot import Frame, State, viz  # noqa: E402
from robotd.views import ViewBoard  # noqa: E402

MJCF = """<mujoco>
  <asset><mesh name="block" vertex="-0.02 -0.02 -0.05  0.02 -0.02 -0.05  0.02 0.02 -0.05  -0.02 0.02 -0.05
                                     -0.02 -0.02 0.05   0.02 -0.02 0.05   0.02 0.02 0.05   -0.02 0.02 0.05"/></asset>
  <worldbody>
    <body name="left_base" pos="0 0 0.3">
      <geom type="mesh" mesh="block" group="2"/>
      <body name="left_link1">
      <joint name="left_joint1" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <body name="left_gripper" pos="0.1 0 0">
        <joint name="left_joint2" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.02 0.02 0.02" group="2"/>
        <body name="left_finger" pos="0 0 -0.04">
          <joint name="left_joint7" type="slide" axis="0 1 0" range="0 0.047"/>
          <geom type="box" size="0.005 0.005 0.02" group="2"/>
        </body>
        <site name="tcp_left" pos="0 0 -0.06"/>
      </body>
      </body>
      <body name="right_base" pos="0 -0.6 0"><geom type="box" size="0.03 0.03 0.03" group="2"/></body>
    </body>
  </worldbody>
</mujoco>"""


TWO_FINGERS = """<mujoco>
  <worldbody>
    <body name="left_base" pos="0 0 0.3">
      <geom type="box" size="0.03 0.03 0.03" group="2"/>
      <body name="left_link1">
        <joint name="left_joint1" type="hinge" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        <body name="left_gripper" pos="0.1 0 0">
          <joint name="left_joint2" type="slide" axis="1 0 0"/>
          <geom type="box" size="0.02 0.03 0.01" group="2"/>
          <body name="left_tip_a" pos="0 -0.005 -0.04">
            <joint name="left_joint7" type="slide" axis="0 -1 0" range="0 0.047"/>
            <geom type="box" size="0.005 0.005 0.02" group="2"/>
          </body>
          <body name="left_tip_b" pos="0 0.005 -0.04">
            <joint name="left_joint8" type="slide" axis="0 1 0" range="0 0.047"/>
            <geom type="box" size="0.005 0.005 0.02" group="2"/>
          </body>
          <site name="tcp_left" pos="0 0 -0.06" zaxis="0 0 -1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality><joint joint1="left_joint8" joint2="left_joint7" polycoef="0 1 0 0 0"/></equality>
</mujoco>"""


def _setup(tmp_path, shift=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.01), mjcf=MJCF):
    path = tmp_path / ("arm.xml" if mjcf is MJCF else "two.xml")
    path.write_text(mjcf)
    desc = {"files": {"mjcf": str(path)}, "arm_joints": {"left": ["left_joint1", "left_joint2"]},
            "tcp_sites": {"left": "tcp_left"}, "finger_joint": {"left": "left_joint7"},
            "gripper_width_range_m": [0.0, 0.094]}
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    names = ["left_joint1", "left_joint2", "left_joint7"]
    for n, v in zip(names, q):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    T = np.eye(4)
    T[:3, :3] = d.site("tcp_left").xmat.reshape(3, 3)
    T[:3, 3] = d.site("tcp_left").xpos + np.asarray(shift)       # the 'driver' sees the base shifted by `shift`
    st = State({"t": 0.0, "joints": names, "q": list(q), "dq": [0] * 3, "effort": [0] * 3, "effort_ext": [0] * 3,
                "q_ref": list(q), "grippers": {},
                "ee": {"left": {"T": T.tolist(), "v": [0, 0, 0], "wrench": [0] * 6, "deflection": [0] * 6,
                                "contact": False}}})
    return desc, st


def _frame(depth=None, W=320, H=240, f=300.0):
    """A camera 1 m above the origin looking straight down (+x of the image along world +x)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0, 0, 1.0]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return Frame({"rgb": np.full((H, W, 3), 90, np.uint8), "depth": depth, "K": K, "T": T})


def test_uv_projects_world_points():
    o = viz.Overlay(frame=_frame())
    uv, z = o.uv([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.5]])
    assert np.allclose(uv[0], [160, 120]) and np.isclose(z[0], 1.0)
    assert np.allclose(uv[1], [190, 120])                         # +x right
    assert np.allclose(uv[2], [160, 60]) and np.isclose(z[2], 0.5)  # +y up the image; 0.5 m closer: twice the offset


def test_ghost_follows_the_driver_tcp_and_width(tmp_path):
    desc, st = _setup(tmp_path, shift=(0.03, 0.0, 0.0))           # a 3 cm base correction the model does not know
    o = viz.Overlay(frame=_frame(), state=st, description=desc)
    g = o.ghost("left", width=0.04)
    assert np.allclose(g["T"][:3, 3], st.ee["left"].T[:3, 3], atol=1e-9)   # drawn where the driver says it is
    assert g["pixels"] > 200 and g["behind_pixels"] == 0
    u, v = o.uv(g["T"][:3, 3])[0][0]
    assert abs(u - (160 + 300 * (0.13 / 0.76))) < 1.5             # tcp at x 0.13, z 0.24 -> 0.76 m from the camera
    g2 = o.ghost("left", q=[np.pi / 2, 0.05])                     # a quarter turn and 5 cm out: tcp at (0, 0.15)
    assert np.allclose(g2["T"][:3, 3], [0.03, 0.15, 0.24], atol=1e-6)
    assert "left ghost" in o.legend[0][0]


def test_ghost_behind_a_surface_is_counted(tmp_path):
    # the arm now: turned half a turn, its link out of view (only its base block, which does not move, is in view).
    # Its ghost at q = 0 puts the link and finger in view, all of it below z 0.35; a 'surface' seen at z 0.40 hides
    # them, except the base block's pixels: the arm is there now, and the arm's own pixels are not a surface
    desc, st = _setup(tmp_path, q=(np.pi, 0.4, 0.0))
    o = viz.Overlay(frame=_frame(np.full((240, 320), 0.60, np.float32)), state=st, description=desc)
    g = o.ghost("left", q=[0.0, 0.0], parts="arm")
    assert g["pixels"] > 200 and 0.3 < g["behind_fraction"] < 0.95
    assert "behind a seen surface" in o.legend[-1][0]
    o2 = viz.Overlay(frame=_frame(np.full((240, 320), 0.95, np.float32)), state=st, description=desc)
    assert o2.ghost("left", q=[0.0, 0.0], parts="arm")["behind_pixels"] == 0  # a surface beyond it hides nothing
    o3 = viz.Overlay(frame=_frame(np.full((240, 320), 0.60, np.float32)), state=st, description=desc)
    assert o3.ghost("left", parts="arm")["behind_pixels"] == 0                 # where it is now: its own pixels never hide it


def test_show_writes_the_view_and_its_history(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROBOT_RUN_DIR", str(tmp_path))
    o = viz.Overlay(frame=_frame())
    o.point([0, 0, 0], "origin")
    o.path([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]], "way")
    o.pose(np.eye(4), "world")
    o.mask(np.eye(240, 320, dtype=bool), label="diag")
    o.text("a note")
    p = o.show("check one")
    assert p == str(tmp_path / "views" / "check-one.png") and os.path.isfile(p)
    assert "[view] check-one:" in capsys.readouterr().out
    viz.show("depth", np.random.rand(20, 30))
    viz.show("mask", np.zeros((20, 30), bool), quiet=True)
    idx = [json.loads(x) for x in open(tmp_path / "views" / "index.jsonl")]
    assert [x["name"] for x in idx] == ["check-one", "depth", "mask"]
    assert all(os.path.isfile(x["history"]) for x in idx)
    img = cv2.imread(p)
    assert img.shape == (240, 320, 3)


def test_view_board_keeps_the_newest():
    b = ViewBoard()
    ok, jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    for i in range(14):
        b.show({"name": f"v{i}", "jpeg": np.frombuffer(jpg.tobytes(), np.uint8)})
    got = b.since(12)
    assert got["seq"] == 14 and [v["name"] for v in got["views"]] == ["v12", "v13"]
    assert len(got["names"]) == 12 and "jpeg" not in got["names"][0]
    with pytest.raises(ValueError):
        b.show({"name": "x", "jpeg": np.zeros(10, np.uint8)})


def test_ghost_parts(tmp_path):
    desc, st = _setup(tmp_path)
    px = {}
    for parts in ("arm", "gripper", "fingers"):
        o = viz.Overlay(frame=_frame(), state=st, description=desc)
        px[parts] = o.ghost("left", parts=parts)["pixels"]
    assert px["arm"] > px["gripper"] > px["fingers"] > 0          # base + link + finger, link + finger, finger
    o = viz.Overlay(frame=_frame(), camera="left_wrist", state=st, description=desc)
    assert o.ghost("left")["pixels"] == px["fingers"]              # a wrist camera draws the fingers by default
    with pytest.raises(ValueError):
        o.ghost("left", parts="hand")


def test_an_arm_is_not_everything_its_base_carries(tmp_path):
    # as in the station model, the left base carries the other arm's base: the left arm must not include it
    desc, st = _setup(tmp_path)
    m = viz._model(desc["files"]["mjcf"])
    names = {m.m.body(b).name for b in m.bodies("left", "arm", "left_joint1")}
    assert names == {"left_base", "left_link1", "left_gripper", "left_finger"}


def test_look_points_the_camera():
    T = viz.look([0, 0, 0.5], [1, 0, 0.5])                         # looking along +x, z up
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 2], [1, 0, 0]) and np.allclose(T[:3, 0], [0, -1, 0]) and np.allclose(T[:3, 1], [0, 0, -1])
    Td = viz.look([0, 0, 1], [0, 0, 0])                           # straight down still makes a camera
    assert np.allclose(Td[:3, 2], [0, 0, -1]) and abs(np.linalg.det(Td[:3, :3]) - 1) < 1e-9


def _plane_frame(eye, at, plane_z, W=320, H=240, f=280.0):
    """A camera at eye looking at `at`, seeing only a horizontal plane at plane_z (a checkerboard)."""
    T = viz.look(eye, at)
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u, float)], -1) @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z - T[2, 3]) / d[..., 2]
    depth = np.where(t > 0, t, 0).astype(np.float32)               # z-depth: the ray's camera z is 1
    P = T[:3, 3] + d * t[..., None]
    rgb = np.where((((P[..., 0] // 0.02) + (P[..., 1] // 0.02)) % 2 == 0)[..., None], 200, 120).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=2)
    return Frame({"rgb": rgb, "depth": depth, "K": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "T": T})


def test_view3d_marks_what_is_under_a_seen_surface(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_RUN_DIR", str(tmp_path))
    desc, st = _setup(tmp_path, q=(np.pi, 0.4, 0.0))               # the arm now: its gripper far out of view
    for plane_z, lo, hi in ((0.27, 0.3, 0.95), (0.10, -1, 0.001)):  # the finger spans z 0.24-0.28; a plane at 0.27
        frames = {"top": _plane_frame([0.1, 0, 1.0], [0.1, 0, 0], plane_z),          # hides its lower
                  "side": _plane_frame([0.6, 0.3, 0.6], [0.1, 0, plane_z], plane_z)}            # three quarters
        v = viz.View3D(frames=frames, state=st, description=desc, radius=0.08)
        g = v.ghost("left", q=[0.0, 0.0], parts="fingers")         # the finger: about its lower half hidden
        assert lo < g["behind_fraction"] < hi, (plane_z, g["behind_fraction"])
        assert np.allclose(v.center, g["T"][:3, 3])
        v.point(g["T"], "tcp")
        v.path([[0.1, 0, 0.35], g["T"][:3, 3]], "down")
    img = v.render()                                               # jaws, along, top, iso: 2 x 2
    assert img.shape[1] == 2 * 480 and img.shape[0] > 2 * 360
    every = list(viz.GRIPPER_VIEWS + viz.WORLD_VIEWS) + [(45, 30), viz.look([0.5, 0.5, 0.5], v.center)]
    img = v.render(views=every)
    assert img.shape[1] == 3 * 480 and img.shape[0] > 4 * 360     # 12 views, three to a row
    assert v.show("v3d", views="jaws").endswith("v3d.png")
    with pytest.raises(ValueError):
        v.render(views=["sideways"])
    w = viz.View3D(frames=frames, state=st, description=desc, center=[0.1, 0, 0.27], radius=0.08)
    with pytest.raises(ValueError):                                # a gripper view needs a ghost
        w.render(views=["jaws"])
    assert w.render().shape[1] == 2 * 480                          # front, left, top, iso


def test_splat_keeps_the_nearest_point():
    K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    P = np.array([[0, 0, 1.0], [0, 0, 0.5]])                         # same pixel, the second nearer
    C = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    img, Z = viz._splat([(P, C, 1)], K, np.eye(4), 100, 100)
    assert tuple(img[50, 50]) == (0, 255, 0) and np.isclose(Z[50, 50], 0.5)


def test_points_between_the_jaws(tmp_path):
    # jaws 4 cm apart, tips at z 0.24, the fingers 1 cm wide (along x): a 'cloth' 1.5 cm above the tips lies between
    # them; one below the tips does not
    desc, st = _setup(tmp_path, q=(np.pi, 0.4, 0.0), mjcf=TWO_FINGERS)
    o = viz.Overlay(frame=_plane_frame([0.1, 0, 1.0], [0.1, 0, 0], 0.255), state=st, description=desc)
    g = o.ghost("left", q=[0.0, 0.0], width=0.04)
    box = g["jaw_box"]
    assert np.allclose(box["s"], [-0.02, 0.02], atol=1e-6) and np.allclose(box["a"], [-0.03, 0.0], atol=1e-6)
    assert np.allclose(box["n"], [-0.005, 0.005], atol=1e-6)
    b = g["between"]
    assert b["unit"] == "pixels" and b["count"] > 20 and b["span_fraction"] > 0.6 and b["gap_mm"] == 40.0  # 2.7 mm px
    assert 13 < b["above_tips_mm"][0] <= b["above_tips_mm"][1] < 17
    assert "between the jaws:" in o.legend[-1][0]
    frames = {"top": _plane_frame([0.1, 0, 1.0], [0.1, 0, 0], 0.255),
              "side": _plane_frame([0.5, 0.4, 0.5], [0.1, 0, 0.255], 0.255)}
    v = viz.View3D(frames=frames, state=st, description=desc, radius=0.06)
    b3 = v.ghost("left", q=[0.0, 0.0], width=0.04)["between"]
    assert b3["unit"] == "points" and b3["count"] > 20 and b3["span_fraction"] > 0.9
    assert v.render().shape[1] == 2 * 480
    below = viz.View3D(frames={"top": _plane_frame([0.1, 0, 1.0], [0.1, 0, 0], 0.20)}, state=st, description=desc)
    nb = below.ghost("left", q=[0.0, 0.0], width=0.04)["between"]
    assert nb["count"] == 0 and nb["above_tips_mm"] is None
    shut = viz.View3D(frames=frames, state=st, description=desc).ghost("left", q=[0.0, 0.0], width=0.0)["between"]
    assert shut["gap_mm"] <= 0 and shut["count"] == 0
