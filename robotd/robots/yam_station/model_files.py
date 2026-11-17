"""Model files for libraries (cuRobo, PyRoKi, Pinocchio, MoveIt, viewers), generated from the driver's own model so
that they agree with it: what the driver reaches, they reach.

i2rt's station URDF has placeholder limits (velocity 1, effort 1 on every joint), no tool-centre link and the left
arm's base as its origin. From it and the driver's kinematic model (which carries a real cell's measured base
corrections) this writes:
- yam_station.urdf: both arms in the world frame (a fixed joint from `world` to each base), a `<arm>_tcp` link at
  the tool centre, the driver's joint limits, velocity limits and effort limits;
- yam_<arm>.urdf: one arm with its base link as the root (for a planner per arm), the same links and limits, and the
  base's pose in the world given in the description (`arms[arm]["T_world_base"]`).
Mesh paths are absolute. Files are written once per content into a cache directory ($ROBOTD_CACHE, default .cache/
in this repository) and reused.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robotd.robots.yam_station import scene as S

REPO_CACHE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".cache"))


def _origin(T: np.ndarray) -> dict:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return {"xyz": " ".join(f"{v:.9g}" for v in T[:3, 3]), "rpy": " ".join(f"{v:.9g}" for v in rpy)}


def _body_T(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> np.ndarray:
    b = m.body(name).id
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def _site_in_body(m: mujoco.MjModel, site: str) -> tuple[str, np.ndarray]:
    sid = m.site(site).id
    T = np.eye(4)
    w, x, y, z = m.site_quat[sid]
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = m.site_pos[sid]
    return m.body(int(m.site_bodyid[sid])).name, T


def _subtree(root: ET.Element, base: str) -> tuple[list, list]:
    """The links and joints below (and including) link `base`."""
    joints = root.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    links, keep_j, todo = [base], [], [base]
    while todo:
        p = todo.pop()
        for j in children.get(p, []):
            c = j.find("child").get("link")
            keep_j.append(j)
            links.append(c)
            todo.append(c)
    return links, keep_j


def _calibrated_mjcf(kin_model: mujoco.MjModel, d: mujoco.MjData) -> str | None:
    """i2rt's MJCF with the bases where the driver's model has them (a real cell's calibration moves them), still in
    the left base's frame; None when they are where i2rt's file puts them (the simulation)."""
    spec = mujoco.MjSpec.from_file(S.STATION_XML)
    ref = spec.compile()
    rd = mujoco.MjData(ref)
    mujoco.mj_kinematics(ref, rd)
    shift = np.eye(4)
    shift[1, 3] = S.STATION_TO_WORLD_Y                  # the file's frame (left base at 0) -> the public world
    moved = False
    for a in S.ARMS:
        name = f"{a}_base"
        b = kin_model.body(name).id
        parent = int(kin_model.body_parentid[b])
        if parent == 0:                                 # a child of the world: its pose in the file's frame
            T = np.linalg.inv(shift) @ _body_T(kin_model, d, name)
            pos, quat = T[:3, 3], np.zeros(4)
            mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
        else:                                           # relative to its parent, as the file has it
            pos, quat = kin_model.body_pos[b].copy(), kin_model.body_quat[b].copy()
        rb = ref.body(name).id
        if np.abs(pos - ref.body_pos[rb]).max() > 1e-7 or min(np.abs(quat - ref.body_quat[rb]).max(),
                                                             np.abs(quat + ref.body_quat[rb]).max()) > 1e-7:
            moved = True
        spec.body(name).pos, spec.body(name).quat = pos, quat
    if not moved:
        return None
    base = os.path.dirname(os.path.abspath(S.STATION_XML))
    spec.meshdir = os.path.normpath(os.path.join(base, spec.meshdir or ""))
    spec.texturedir = os.path.normpath(os.path.join(base, spec.texturedir or ""))
    return spec.to_xml()


def write_model_files(kin_model: mujoco.MjModel, desc: dict, out_dir: str | None = None) -> dict:
    """Write the URDFs for this model and description, and an MJCF when the driver's bases differ from i2rt's (a real
    cell's calibration); returns {"urdf": the bimanual one, "mjcf": that MJCF or None, "arms": {arm: {...}}}."""
    src = S.STATION_URDF
    tree = ET.parse(src)
    root = tree.getroot()
    src_dir = os.path.dirname(os.path.abspath(src))
    for mesh in root.iter("mesh"):
        f = mesh.get("filename")
        if f and not f.startswith(("/", "package://")):
            mesh.set("filename", os.path.normpath(os.path.join(src_dir, f)))
    lo_hi = {kin_model.joint(i).name: kin_model.jnt_range[i] for i in range(kin_model.njnt)}
    for j in root.findall("joint"):
        name, lim = j.get("name"), j.find("limit")
        if lim is None or name not in desc.get("effort_limit", {}):
            continue
        lo, hi = lo_hi[name]
        lim.set("lower", f"{lo:.9g}")
        lim.set("upper", f"{hi:.9g}")
        lim.set("effort", f"{desc['effort_limit'][name]:.9g}")
        lim.set("velocity", f"{desc['velocity_limit'][name]:.9g}")
    d = mujoco.MjData(kin_model)
    mujoco.mj_kinematics(kin_model, d)
    T_base = {a: _body_T(kin_model, d, f"{a}_base") for a in S.ARMS}
    for a in S.ARMS:                                    # the tool-centre link, as the driver's TCP site
        body, T = _site_in_body(kin_model, S.TCP_SITES[a])
        ET.SubElement(root, "link", name=f"{a}_tcp")
        j = ET.SubElement(root, "joint", name=f"{a}_tcp_joint", type="fixed")
        ET.SubElement(j, "origin", **_origin(T))
        ET.SubElement(j, "parent", link=body)
        ET.SubElement(j, "child", link=f"{a}_tcp")
    # both arms in the world: world -> each base (the right base was a child of the left one)
    for j in list(root.findall("joint")):
        if j.find("child").get("link") in {f"{a}_base" for a in S.ARMS}:
            root.remove(j)
    world = ET.Element("link", name="world")
    root.insert(0, world)
    for a in S.ARMS:
        j = ET.SubElement(root, "joint", name=f"world_to_{a}_base", type="fixed")
        ET.SubElement(j, "origin", **_origin(T_base[a]))
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=f"{a}_base")
    # the top camera hung off the left base in the source: keep it in the world
    for j in root.findall("joint"):
        if j.find("child").get("link") == "top_camera" and j.find("parent").get("link") == "left_base":
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            T[:3, 3] = xyz
            j.find("parent").set("link", "world")
            for k, v in _origin(T_base["left"] @ T).items():
                o.set(k, v)
    texts = {"yam_station.urdf": ET.tostring(root, encoding="unicode")}
    mjcf = _calibrated_mjcf(kin_model, d)
    if mjcf is not None:
        texts["yam_station.xml"] = mjcf
    for a in S.ARMS:
        links, joints = _subtree(root, f"{a}_base")
        one = ET.Element("robot", name=f"yam_{a}")
        for el in root.findall("link"):
            if el.get("name") in links:
                one.append(el)
        for el in joints:
            one.append(el)
        texts[f"yam_{a}.urdf"] = ET.tostring(one, encoding="unicode")
    h = hashlib.sha256("".join(texts[k] for k in sorted(texts)).encode()).hexdigest()[:16]
    out = out_dir or os.path.join(os.environ.get("ROBOTD_CACHE") or REPO_CACHE, "yam_station", h)
    os.makedirs(out, exist_ok=True)
    for name, text in texts.items():
        path = os.path.join(out, name)
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write('<?xml version="1.0"?>\n' + text)
            os.replace(tmp, path)
    return {"urdf": os.path.join(out, "yam_station.urdf"),
            "mjcf": os.path.join(out, "yam_station.xml") if mjcf is not None else None,
            "arms": {a: {"urdf": os.path.join(out, f"yam_{a}.urdf"), "base_link": f"{a}_base",
                         "tcp_link": f"{a}_tcp", "joints": list(desc["arm_joints"][a]),
                         "finger_joints": list(S.FINGER_JOINTS[a]), "T_world_base": T_base[a].tolist()}
                     for a in S.ARMS}}


def with_model_files(desc: dict, kin_model: mujoco.MjModel) -> dict:
    """The description with the generated files: files["urdf"] the bimanual URDF in the world frame; files["mjcf"]
    i2rt's MJCF in the left base's frame, or on a calibrated cell a copy with the bases where the driver has them;
    and per arm its URDF, base and tool links, joints and base pose."""
    gen = write_model_files(kin_model, desc)
    out = dict(desc)
    out["files"] = {**desc["files"], "urdf": gen["urdf"], **({"mjcf": gen["mjcf"]} if gen["mjcf"] else {})}
    out["arms"] = gen["arms"]
    return out
