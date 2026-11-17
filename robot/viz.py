"""robot.viz: pictures to check your work with, for you and for the operator.

Draw on a camera frame in world coordinates, render an arm where a move would put it (its real meshes, tested against
the camera's depth), and show the result. viz.show writes the image to a file you can open and look at, and sends a
copy to the operator's dashboard.

    from robot import viz
    o = viz.Overlay(r, "top")                     # a fresh frame from that camera to draw on
    g = o.ghost("left", T_grasp, width=0.06)      # the left arm where r.ik puts that TCP pose, jaws 6 cm apart
    o.path([T_above, T_grasp])                    # the TCP's way there
    o.point(corner, "corner")                     # world points (3 or N x 3), labelled
    path = o.show("grasp-check")                  # the file: open it and look. The dashboard shows it too.

    viz.show("towel-mask", mask)                  # imshow: any image (H x W x 3 RGB, H x W grey, bool, or depth in m)
    viz.show_cameras("now", r)                    # every camera in ONE labelled grid image (one image, not four)
    pts = viz.marks(frame, "towel-marks", mask=m) # numbered points on the image; pts[7] is marker 7's world xyz

A program that showed images prints, as it exits, `[views] ["/path/a.png", ...]`: the files it wrote, one JSON list,
so whatever ran it can attach them right away.

The ghost is translucent where the camera would see it, and RED where it would be behind a surface the camera sees:
inside or under it (a finger that would go into the table, or under the cloth). The arms' own pixels, where they are
now, do not count as surfaces. ghost() with no target draws the arm where it is: it should sit on the real arm, a check
of the camera's pose. Images are RGB, as r.cameras() gives them: colours are (r, g, b) or a name in viz.COLORS.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import time

import numpy as np

COLORS = {"red": (235, 50, 60), "green": (60, 210, 90), "blue": (60, 130, 255), "yellow": (255, 215, 40),
          "cyan": (40, 225, 235), "magenta": (235, 70, 225), "orange": (255, 150, 30), "white": (255, 255, 255),
          "black": (0, 0, 0)}
BEHIND = COLORS["red"]
BETWEEN = (255, 40, 200)                                      # points between the jaws: hot pink, rare in a scene
JAW_DEPTH = 0.03                                               # how far up from the tips "between the jaws" reaches
SELF_MARGIN = 0.006            # depth points within this of the arms' own surfaces (as the model poses them now) are the
                               # arms themselves, seen past the edge of their pixel mask: left out of clouds and counts.
SELF_MARGIN_FIXED = 0.012      # ... for a camera that does not ride on the arm: the arm's pose has mm of error there
                               # (on the real cell, ~30-40 points of the left finger stayed at 6 mm in the front camera)


def _self_margin(camera) -> float:
    return SELF_MARGIN if "wrist" in str(camera) else SELF_MARGIN_FIXED
_MODELS: dict = {}


def _cv2():
    try:
        import cv2
    except ImportError as e:                                   # pragma: no cover -- the cell's venv has it
        raise ImportError("robot.viz needs OpenCV (pip install opencv-python-headless)") from e
    return cv2


def _color(c, default="yellow"):
    c = default if c is None else c
    if isinstance(c, str):
        if c not in COLORS:
            raise ValueError(f"unknown colour {c!r}; names: {', '.join(COLORS)}, or an (r, g, b) tuple")
        return COLORS[c]
    return tuple(int(x) for x in c[:3])


def _text(img, s, org, color=(255, 255, 255), scale=None):
    """Text with a dark edge, readable on any background."""
    cv2 = _cv2()
    scale = scale or max(0.4, img.shape[1] / 1300)
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _color(color), 1, cv2.LINE_AA)


def _text_width(s, scale) -> int:
    cv2 = _cv2()
    return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]


def _wrap(s, width, scale) -> list:
    """A legend line cut into lines that fit `width` pixels: at " | " first, then between words."""
    out = []
    for part in s.split(" | "):
        line = ""
        for word in part.split(" "):
            cand = f"{line} {word}" if line else word
            if line and _text_width(cand, scale) > width:
                out.append(line)
                line = "  " + word
            else:
                line = cand
        out.append(line)
    return out


# ------------------------------------------------------------------------------------------------ show (imshow)
def views_dir() -> str:
    """Where shown images go: the program run's directory (views/) under `python -m robot run`, else ./.robot/views."""
    run = os.environ.get("ROBOT_RUN_DIR")
    d = os.path.join(run, "views") if run else os.path.join(os.getcwd(), ".robot", "views")
    os.makedirs(os.path.join(d, "history"), exist_ok=True)
    return d


def as_rgb(img) -> np.ndarray:
    """Any image as H x W x 3 uint8 RGB: RGB stays; bool and grey become grey; a float H x W (a depth map in m, a score)
    is coloured by value (turbo: low blue, high red; NaN and 0 black)."""
    cv2 = _cv2()
    if isinstance(img, Overlay):
        return img.render()
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        if a.dtype != np.uint8:
            a = np.clip(a * (255 if a.max() <= 1.0 else 1), 0, 255)
        return np.ascontiguousarray(a[..., :3], np.uint8)
    if a.ndim != 2:
        raise ValueError(f"not an image: shape {a.shape}")
    if a.dtype == bool:
        g = a.astype(np.uint8) * 255
    elif a.dtype == np.uint8:
        g = a
    else:
        f = a.astype(np.float64)
        ok = np.isfinite(f) & (f != 0)
        if not ok.any():
            return np.zeros(a.shape + (3,), np.uint8)
        lo, hi = np.percentile(f[ok], [1, 99])
        g = np.clip((f - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        out[~ok] = 0
        return out
    return np.repeat(g[..., None], 3, axis=2)


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-.")[:60]
    return s or "view"


_SHOWN: list = []                                             # this program's views, for the [views] line at exit
_SINK = threading.local()     # a thread's own list of shown paths instead of _SHOWN (robot.kernel: an action's report)


def _print_views():
    if _SHOWN and os.environ.get("ROBOT_EXIT_VIEWS", "1") != "0":
        print("[views] " + json.dumps(list(dict.fromkeys(_SHOWN))), flush=True)


def show(name: str, img, r=None, note: str | None = None, quiet: bool = False) -> str:
    """Save an image as views/NAME.png (each show of a name replaces it; views/history keeps every one) and send a
    copy to the operator's dashboard. Returns the file's path: open it to look. Prints `[view] NAME: PATH`, and
    the program prints `[views] [...]` (every path it showed, one JSON list) as it exits. quiet=True: neither."""
    cv2 = _cv2()
    rgb = as_rgb(img)
    name = _safe_name(name)
    d = views_dir()
    path = os.path.join(d, name + ".png")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tmp = path + ".tmp.png"
    cv2.imwrite(tmp, bgr)
    os.replace(tmp, path)                       # a reader never sees half a file
    try:
        with open(os.path.join(d, "index.jsonl")) as f:
            n = sum(1 for _ in f)
    except OSError:
        n = 0
    hist = os.path.join(d, "history", f"{n:04d}-{name}.jpg")
    cv2.imwrite(hist, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(os.path.join(d, "index.jsonl"), "a") as f:
        f.write(json.dumps({"n": n, "name": name, "wall": time.time(), "file": path, "history": hist,
                            "note": note, "shape": list(rgb.shape)}) + "\n")
    _push(name, bgr, r, note)
    if not quiet:
        print(f"[view] {name}: {path}", flush=True)
        sink = getattr(_SINK, "paths", None)
        if sink is not None:
            sink.append(path)
        else:
            if not _SHOWN:
                atexit.register(_print_views)
            _SHOWN.append(path)
    return path


def grid(images, labels=None, cols: int = 2, width: int = 960) -> np.ndarray:
    """Images (anything show takes) tiled in one RGB image `width` pixels wide, each scaled to fit its cell and
    labelled in its corner. One image of four cameras costs the model less than four images (fewer images and,
    at width 960, fewer pixels than four full frames)."""
    cv2 = _cv2()
    ims = [as_rgb(i) for i in images]
    if not ims:
        raise ValueError("grid: no images")
    cols = max(1, min(cols, len(ims)))
    rows = -(-len(ims) // cols)
    cw = width // cols
    h0, w0 = ims[0].shape[:2]
    ch = int(round(cw * h0 / w0))
    out = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    for i, im in enumerate(ims):
        h, w = im.shape[:2]
        k = min(cw / w, ch / h)
        small = cv2.resize(im, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * ch, (i % cols) * cw
        out[y:y + small.shape[0], x:x + small.shape[1]] = small
        if labels is not None and i < len(labels) and labels[i]:
            _text(out, str(labels[i]), (x + 6, y + 18), (255, 255, 255), 0.5)
    return out


def show_cameras(name: str = "cameras", r=None, frames=None, names=None, width: int = 960, note: str | None = None,
                 quiet: bool = False) -> str:
    """Every camera (or `names`) as ONE labelled grid image, views/NAME.png: frames (from r.cameras()) or a fresh
    grab from r (colour only). Returns the path. width=640 is a quick look; 1280 keeps every camera at full size."""
    if frames is None:
        if r is None:
            import robot
            r = robot._latest() or robot.connect()
        frames = r.cameras(names, depth=False)
    order = [n for n in (names or ("top", "front", "left_wrist", "right_wrist")) if n in frames]
    order += [n for n in frames if n not in order and (names is None or n in names)]
    img = grid([frames[n].rgb for n in order], labels=order, cols=2 if len(order) > 1 else 1, width=width)
    return show(name, img, r=r, note=note, quiet=quiet)


def _pixel_xyz(frame, u: int, v: int, win: int = 2):
    """The world point at pixel (u, v): the median of the valid depths around it. None where there is no depth."""
    if frame.depth is None or frame.T is None:
        return None
    H, W = frame.depth.shape
    patch = frame.depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1].astype(np.float64)
    ok = np.isfinite(patch) & (patch > 0.02) & (patch < 5.0)
    if not ok.any():
        return None
    z = float(np.median(patch[ok]))
    x = (u - frame.K[0, 2]) * z / frame.K[0, 0]
    y = (v - frame.K[1, 2]) * z / frame.K[1, 1]
    return (frame.T @ np.array([x, y, z, 1.0]))[:3]


TABLE_PAD = 0.10        # m added around the arms' fence boxes (x and y) for table_box


def table_box(r=None) -> tuple:
    """The volume over the table, ((x0, x1), (y0, y1), (z0, z1)) in world metres: the arms' workspace fence boxes
    (the driver's description) padded by TABLE_PAD in x and y, from 2 cm below z = 0 to 10 cm over the fence's top.
    Without a fence: x 0-0.9, y -0.6-0.6, z -0.02-0.7."""
    import robot
    if r is None or r is False:
        r = robot._latest()
    fence = None
    try:
        fence = (r.description or {}).get("fence") if r is not None else None
    except Exception:                                          # noqa: BLE001
        fence = None
    boxes = [fence[a] for a in ("left", "right") if isinstance(fence, dict) and isinstance(fence.get(a), dict)]
    boxes = [b for b in boxes if all(k in b for k in ("x", "y", "z"))]
    if not boxes:
        return (0.0, 0.9), (-0.6, 0.6), (-0.02, 0.7)
    lo = {k: min(b[k][0] for b in boxes) for k in "xyz"}
    hi = {k: max(b[k][1] for b in boxes) for k in "xyz"}
    return ((lo["x"] - TABLE_PAD, hi["x"] + TABLE_PAD), (lo["y"] - TABLE_PAD, hi["y"] + TABLE_PAD),
            (-0.02, hi["z"] + 0.10))


def inside(frame, box) -> np.ndarray:
    """H x W bool: the pixels whose depth puts them inside box, ((x0, x1), (y0, y1), (z0, z1)) in world metres.
    Pixels without depth are False, except small holes inside an area that is in (closed over 7 px)."""
    if frame.depth is None or frame.T is None:
        raise ValueError("inside()/on_table need the frame's depth and the camera's pose")
    cv2 = _cv2()
    z = np.asarray(frame.depth, dtype=np.float64)
    H, W = z.shape
    ok = np.isfinite(z) & (z > 0.02) & (z < 5.0)
    K, T = np.asarray(frame.K, dtype=np.float64), np.asarray(frame.T, dtype=np.float64)
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = [T[i, 0] * x + T[i, 1] * y + T[i, 2] * z + T[i, 3] for i in range(3)]
    keep = ok.copy()
    for c, (a, b) in zip(P, box):
        keep &= (c >= a) & (c <= b)
    closed = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    return keep | (closed & ~ok)                              # in the box, or a depth hole closed over


def marks(frame, name: str | None = None, mask=None, step: int = 40, extra=None, max_marks: int = 60,
          r=None, quiet: bool = False, on_table: bool = False, within=None, picture: bool = True) -> dict:
    """Numbered points on a camera frame (it needs depth, K and T), each with its world position, so that you can
    point at a place by its number instead of measuring pixels and looking up depth in another step. Points: a grid
    every `step` pixels (inside `mask` when given, e.g. the towel's), the outline's corners when a mask is given, and
    `extra` pixels [(u, v), ...] of your own. on_table=True keeps only what lies over the table (table_box: the floor,
    the stands and the backdrop beyond the table are left out, mask or no mask); within=((x0, x1), (y0, y1),
    (z0, z1)) keeps a box of your own. With `name`, prints one line, `[marks NAME] 1:(x,y,z) 2:...` in metres, and
    shows the image (views/NAME.png) unless picture=False. Returns {number: world xyz (3,)}; numbers run 1, 2, ... in
    reading order, outline corners first."""
    cv2 = _cv2()
    H, W = frame.depth.shape if frame.depth is not None else frame.rgb.shape[:2]
    cand = []
    n_corners = 0
    m = None if mask is None else np.asarray(mask).astype(bool)
    if on_table or within is not None:
        keep = inside(frame, within if within is not None else table_box(r))
        m = keep if m is None else (m & keep)
    if m is not None:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
            cand += [(int(u), int(v)) for u, v in poly]
            n_corners = len(poly)
    for v in range(step // 2, H, step):
        for u in range(step // 2, W, step):
            if m is None or m[v, u]:
                cand.append((u, v))
    cand += [(int(u), int(v)) for u, v in (extra or [])]
    pts = {}
    corners, rest = [], []
    for i, (u, v) in enumerate(cand):
        if not (0 <= u < W and 0 <= v < H):
            continue
        p = _pixel_xyz(frame, u, v)
        if p is not None:
            (corners if i < n_corners else rest).append((u, v, p))
    room = max(0, max_marks - len(corners))
    if len(rest) > room:                                        # thin the grid evenly; the outline's corners stay
        idx = np.linspace(0, len(rest) - 1, room).round().astype(int) if room else []
        rest = [rest[i] for i in sorted(set(idx))]
    kept = corners + rest
    img = as_rgb(frame.rgb).copy()
    scale = max(0.35, W / 1800)
    for k, (u, v, p) in enumerate(kept, 1):
        pts[k] = p
        cv2.circle(img, (u, v), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, COLORS["yellow"], -1, cv2.LINE_AA)
        _text(img, str(k), (u + 5, v - 4), COLORS["yellow"], scale)
    if name is not None:
        if picture:
            show(name, img, r=r, quiet=quiet)
        if not quiet:
            print(f"[marks {_safe_name(name)}] " + " ".join(
                f"{k}:({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for k, p in pts.items()), flush=True)
    return pts


_PUSH_WARNED = []


def _push(name, bgr, r, note):
    """A copy for the dashboard, through the image server (camerad, or robotd in simulation). Best effort."""
    if r is None:
        import robot
        r = robot._latest()
    if r is None:
        return
    cv2 = _cv2()
    h, w = bgr.shape[:2]
    if w > 800:
        bgr = cv2.resize(bgr, (800, int(h * 800 / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    try:
        r._views("show", name=name, jpeg=np.frombuffer(buf.tobytes(), np.uint8), note=note)
    except Exception as e:                                     # noqa: BLE001 -- the file is what matters
        if not _PUSH_WARNED:
            _PUSH_WARNED.append(1)
            print(f"[view] (not on the dashboard: {type(e).__name__}: {e})", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ the arm's meshes
class _ArmModel:
    """The description's MJCF: every arm's visual meshes once (lightly simplified), their poses at any joints."""

    def __init__(self, mjcf: str, cell: float = 0.003):
        import mujoco
        self.mujoco = mujoco
        self.m = m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(m)
        self.jadr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        self.couplings = []                                    # joint equalities: the second finger follows the first
        for i in range(m.neq):
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                dep, ind = int(m.jnt_qposadr[m.eq_obj1id[i]]), int(m.jnt_qposadr[m.eq_obj2id[i]])
                self.couplings.append((dep, ind, m.eq_data[i][:5].copy()))
        self.parts = {}                                        # geom id -> (verts in the geom frame, faces)
        for g in range(m.ngeom):
            if m.geom_group[g] >= 3 or m.geom_rgba[g][3] == 0:
                continue
            part = self._geometry(g, cell)
            if part is not None:
                self.parts[g] = part
        self.body_of = {g: int(m.geom_bodyid[g]) for g in self.parts}

    def _geometry(self, g, cell):
        m, mj = self.m, self.mujoco
        t = m.geom_type[g]
        if t == mj.mjtGeom.mjGEOM_MESH:
            k = int(m.geom_dataid[g])
            v0, nv, f0, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
            return _simplify(np.array(m.mesh_vert[v0:v0 + nv], float), np.array(m.mesh_face[f0:f0 + nf], int), cell)
        if t == mj.mjtGeom.mjGEOM_BOX:
            return _box(m.geom_size[g])
        if t in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            r, h = m.geom_size[g][0], m.geom_size[g][1] + (m.geom_size[g][0] if t == mj.mjtGeom.mjGEOM_CAPSULE else 0)
            return _prism(r, h)
        if t == mj.mjtGeom.mjGEOM_SPHERE:
            return _prism(m.geom_size[g][0], m.geom_size[g][0])
        return None

    def subtree(self, root_body: str) -> set:
        m = self.m
        try:
            root = m.body(root_body).id
        except KeyError:
            return set()
        out = set()
        for b in range(m.nbody):
            x = b
            while x > 0 and x != root:
                x = int(m.body_parentid[x])
            if x == root:
                out.add(b)
        return out

    def bodies(self, arm: str, parts: str = "arm", first_joint: str | None = None) -> set:
        """The bodies drawn for `parts` of an arm: "arm" (its base and everything its first joint moves), "gripper"
        (from its gripper body) or "fingers" (the gripper's bodies that move on joints of their own, and what they
        carry). Not the base's subtree: in the station model the left base carries the whole station."""
        if parts not in ("arm", "gripper", "fingers"):
            raise ValueError(f"parts= 'arm', 'gripper' or 'fingers', not {parts!r}")
        m = self.m
        arm_bodies = set()
        if first_joint is not None and first_joint in self.jadr:
            arm_bodies = self.subtree(m.body(int(m.jnt_bodyid[m.joint(first_joint).id])).name)
            try:
                arm_bodies.add(m.body(f"{arm}_base").id)
            except KeyError:
                pass
        arm_bodies = arm_bodies or {b for b in range(m.nbody) if m.body(b).name.startswith(arm)}
        if parts == "arm":
            return arm_bodies
        grip = self.subtree(f"{arm}_gripper") or arm_bodies
        if parts == "gripper":
            return grip
        moving = [b for b in grip if self.m.body_jntnum[b] > 0 and self.m.body(b).name != f"{arm}_gripper"]
        out = set()
        for b in moving:
            out |= self.subtree(self.m.body(b).name)
        return out or grip

    def set(self, q: dict) -> None:
        d = self.d
        given = set()
        for n, v in q.items():
            a = self.jadr.get(n)
            if a is not None:
                d.qpos[a] = float(v)
                given.add(a)
        for dep, ind, c in self.couplings:                     # whichever side was given drives the other
            if ind in given and dep not in given:
                x = d.qpos[ind]
                d.qpos[dep] = c[0] + c[1] * x + c[2] * x * x + c[3] * x ** 3 + c[4] * x ** 4
            elif dep in given and ind not in given and abs(c[1]) > 1e-9:
                d.qpos[ind] = (d.qpos[dep] - c[0]) / c[1]
        self.mujoco.mj_kinematics(self.m, d)

    def site_T(self, site: str) -> np.ndarray:
        i = self.m.site(site).id
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[i].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[i]
        return T

    def triangles(self, geoms, T_fix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The world vertices (n x 3) and faces (k x 3) of these geoms at the joints last set, moved by T_fix."""
        Vs, Fs, off = [], [], 0
        for g in geoms:
            V, F = self.parts[g]
            R, p = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            W = V @ R.T + p
            Vs.append(W @ T_fix[:3, :3].T + T_fix[:3, 3])
            Fs.append(F + off)
            off += len(V)
        if not Vs:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        return np.concatenate(Vs), np.concatenate(Fs)


def _simplify(V, F, cell):
    """Vertex clustering on a cell-sized grid: a mesh of tens of thousands of faces becomes a few thousand, with the
    shape kept to a few mm (a ghost to look at, not a collision model). Face winding is kept."""
    if len(F) < 400:
        return V, F
    key = np.floor(V / cell).astype(np.int64)
    _, idx, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    idx = idx.reshape(-1)
    W = np.zeros((len(counts), 3))
    np.add.at(W, idx, V)
    W /= counts[:, None]
    G = idx[F]
    G = G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])]
    _, first = np.unique(np.sort(G, axis=1), axis=0, return_index=True)
    return W, G[np.sort(first)]


def _box(size):
    x, y, z = size[:3]
    V = np.array([[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], int)
    return V, F


def _prism(r, h, n=12):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    V = np.concatenate([np.c_[ring, np.full(n, -h)], np.c_[ring, np.full(n, h)], [[0, 0, -h], [0, 0, h]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F, int)


def _model(mjcf: str) -> _ArmModel:
    if mjcf not in _MODELS:
        _MODELS[mjcf] = _ArmModel(mjcf)
    return _MODELS[mjcf]


# ------------------------------------------------------------------------------------------------ the overlay
class Overlay:
    """A camera frame to draw on in world coordinates. Everything drawn goes onto .img (H x W x 3 RGB, yours to draw on
    with cv2 too); show() adds a legend and saves it. Overlay(r, "top") takes a fresh frame; pass frame= to draw on
    one you have (camera= its name, for the legend)."""

    def __init__(self, r=None, camera: str = "top", frame=None, state=None, description=None):
        cv2 = _cv2()
        self.r, self.camera = r, camera
        if frame is None:
            if r is None:
                raise ValueError("give r (to take a frame) or frame=")
            frame = r.cameras([camera])[camera]
        self.frame = frame
        self.img = np.ascontiguousarray(as_rgb(frame.rgb)).copy()
        H, W = self.img.shape[:2]
        K = np.asarray(frame.K, float)
        depth = getattr(frame, "depth", None)
        if depth is not None and depth.shape[:2] != (H, W):       # K is the depth image's: scale it to the colour's
            sx, sy = W / depth.shape[1], H / depth.shape[0]
            K = np.diag([sx, sy, 1.0]) @ K
            depth = cv2.resize(np.asarray(depth, np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        self.K = K
        self.depth = None if depth is None else np.asarray(depth, np.float32)
        self.T = None if frame.T is None else np.asarray(frame.T, float)
        self.T_cam = None if self.T is None else np.linalg.inv(self.T)
        self._state = state
        self.description = description if description is not None else (r.description if r is not None else None)
        self.legend: list = []
        self.ghosts: list = []

    # ---- geometry
    def state(self):
        if self._state is None and self.r is not None:
            self._state = self.r.state()
        return self._state

    def uv(self, points):
        """World points (3 or N x 3, or 4x4 poses) -> (pixels N x 2, depth along the camera's axis N; <= 0 behind it)."""
        if self.T_cam is None:
            raise ValueError(f"camera {self.camera} has no calibrated pose: world points cannot be placed on it")
        P = np.asarray(points, float)
        if P.ndim == 2 and P.shape == (4, 4):
            P = P[None]
        if P.ndim == 3:
            P = P[:, :3, 3]
        P = np.atleast_2d(P)[:, :3]
        C = P @ self.T_cam[:3, :3].T + self.T_cam[:3, 3]
        z = C[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * C[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * C[:, 1] / z + self.K[1, 2]
        return np.stack([u, v], 1), z

    def _px(self, uv):
        return int(round(float(uv[0]))), int(round(float(uv[1])))

    def _inside(self, uv, z):
        H, W = self.img.shape[:2]
        return z > 0 and -W < uv[0] < 2 * W and -H < uv[1] < 2 * H

    # ---- drawing
    def text(self, line: str, color="white"):
        """A line for the legend (top left), e.g. what a check found."""
        self.legend.append((str(line), _color(color, "white")))

    def point(self, xyz, label=None, color="yellow", radius=None):
        """World point(s), 3 or N x 3 (or 4x4 poses: their positions); label: one string, or one per point."""
        cv2 = _cv2()
        col = _color(color)
        radius = radius or max(4, self.img.shape[1] // 120)
        uv, z = self.uv(xyz)
        labels = label if isinstance(label, (list, tuple)) else [label] * len(uv)
        for p, zz, lab in zip(uv, z, labels):
            if not self._inside(p, zz):
                continue
            c = self._px(p)
            cv2.circle(self.img, c, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(self.img, c, radius, col, -1, cv2.LINE_AA)
            if lab:
                _text(self.img, str(lab), (c[0] + radius + 4, c[1] - radius - 2), col)

    def points(self, P, color="yellow", size=2):
        """Many world points as small dots (a cloud, a fitted edge)."""
        uv, z = self.uv(P)
        H, W = self.img.shape[:2]
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        col = np.array(_color(color), np.uint8)
        for du in range(-(size // 2), size - size // 2):
            for dv in range(-(size // 2), size - size // 2):
                self.img[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = col

    def pose(self, T, label=None, size=0.05):
        """A pose's axes: x red, y green, z blue, size m long."""
        cv2 = _cv2()
        T = np.asarray(T, float)
        ends = [T[:3, 3]] + [T[:3, 3] + size * T[:3, i] for i in range(3)]
        uv, z = self.uv(np.array(ends))
        if not self._inside(uv[0], z[0]):
            return
        o = self._px(uv[0])
        for i, col in enumerate(("red", "green", "blue")):
            if z[i + 1] > 0:
                cv2.arrowedLine(self.img, o, self._px(uv[i + 1]), (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.2)
                cv2.arrowedLine(self.img, o, self._px(uv[i + 1]), COLORS[col], 2, cv2.LINE_AA, tipLength=0.2)
        if label:
            _text(self.img, str(label), (o[0] + 6, o[1] + 16))

    def path(self, points, label=None, color="cyan"):
        """A path through world positions (N x 3, or 4x4 poses): a line with a dot at each point, an arrow at the end."""
        cv2 = _cv2()
        col = _color(color)
        uv, z = self.uv(points)
        pts = [self._px(p) for p, zz in zip(uv, z) if self._inside(p, zz)]
        if len(pts) >= 2:
            arr = np.array(pts, np.int32)
            cv2.polylines(self.img, [arr], False, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(self.img, [arr], False, col, 2, cv2.LINE_AA)
            cv2.arrowedLine(self.img, pts[-2], pts[-1], col, 2, cv2.LINE_AA, tipLength=min(0.5, 12 / max(
                1, np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]))))
        for p in pts:
            cv2.circle(self.img, p, 3, col, -1, cv2.LINE_AA)
        if label and pts:
            _text(self.img, str(label), (pts[-1][0] + 6, pts[-1][1] + 14), col)

    def mask(self, m, color="orange", label=None, alpha=0.35):
        """Tint a pixel mask of this image (H x W bool) and outline it."""
        cv2 = _cv2()
        m = np.asarray(m).astype(bool)
        H, W = self.img.shape[:2]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        col = np.array(_color(color), float)
        self.img[m] = (self.img[m] * (1 - alpha) + col * alpha).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.img, cs, -1, tuple(int(c) for c in col), 1, cv2.LINE_AA)
        if label and cs:
            x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
            _text(self.img, str(label), (x, max(12, y - 4)), tuple(int(c) for c in col))

    def box(self, u0, v0, u1, v1, label=None, color="yellow"):
        """A pixel rectangle (e.g. a detection)."""
        cv2 = _cv2()
        col = _color(color)
        cv2.rectangle(self.img, (int(u0), int(v0)), (int(u1), int(v1)), col, 2, cv2.LINE_AA)
        if label:
            _text(self.img, str(label), (int(u0), max(12, int(v0) - 4)), col)

    # ---- the ghost
    def ghost(self, arm: str, target=None, q=None, width=None, free=None, color="cyan", label=None,
              alpha=0.45, parts=None, after=None) -> dict:
        """Draw `arm` where it would be: at the joints r.ik finds for the TCP pose `target` (4x4; free= as for move),
        or at joint values q (that arm's joints in order, or {name: value}), or where it is now (neither). width= the
        jaws' opening in m (default: as they are now). parts= "fingers", "gripper" (its body and fingers) or "arm"
        (every link); default "fingers" in a wrist camera, "gripper" in the others. Translucent where the camera would
        see it, red where a surface the camera sees is in front of it. Returns {"q", "T" (the TCP pose drawn),
        "jaw_axis", "ik_ok", "error_m", "pixels", "behind_pixels", "behind_fraction"}, also in self.ghosts.
        after= (as for r.reachable: where the queue will have put the arms) seeds r.ik from there, as the move will
        solve it; the picture is still this frame, the rest of the robot where it is now."""
        g = _posed_arm(self.r, self.description, self.state(), arm, target, q, width, free,
                       parts or ("fingers" if "wrist" in str(self.camera) else "gripper"), after)
        info = g["info"]
        rgb, Z = self._raster(g["V"], g["F"], _color(color, "cyan"), near=0.06)
        cover = np.isfinite(Z)
        behind = np.zeros_like(cover)
        if self.depth is not None:
            real = self.depth
            seen = np.isfinite(real) & (real > 0.02) & ~self._arms_now()
            behind = cover & seen & (real < Z - _tol(np.where(cover, Z, 0)))
        front = cover & ~behind
        img = self.img.astype(np.float32)
        img[front] = img[front] * (1 - alpha) + rgb[front] * alpha
        img[behind] = img[behind] * (1 - alpha) + np.array(BEHIND, np.float32) * alpha
        self.img = img.astype(np.uint8)
        _outline_and_mark(self, cover, info["T"], _color(color, "cyan"), label or f"{arm} ghost")
        n, nb = int(cover.sum()), int(behind.sum())
        info.update(pixels=n, behind_pixels=nb, behind_fraction=round(nb / n, 3) if n else 0.0, between=None)
        box = info.get("jaw_box")
        if box is not None and self.depth is not None and self.T is not None:
            H, W = self.depth.shape
            v, u = np.mgrid[0:H, 0:W].reshape(2, -1)
            z = self.depth.reshape(-1).astype(np.float64)
            ok = np.isfinite(z) & (z > 0.02) & ~self._arms_now().reshape(-1)
            u, v, z = u[ok], v[ok], z[ok]
            Pc = np.stack([(u - self.K[0, 2]) * z / self.K[0, 0], (v - self.K[1, 2]) * z / self.K[1, 1], z], 1)
            Pw = _to_world(Pc, self.T)                        # this camera's points, in the world
            c = info["T"][:3, 3]
            S = _self_surface(self.description, self.state(), c - 0.15, c + 0.15)
            mine = _not_self(Pw, S, _self_margin(self.camera)) if len(S) else np.ones(len(Pw), bool)
            u, v, Pw = u[mine], v[mine], Pw[mine]
            inside, L = _between(Pw, box)
            img = self.img.astype(np.float32)
            img[v[inside], u[inside]] = img[v[inside], u[inside]] * 0.3 + np.array(BETWEEN, np.float32) * 0.7
            self.img = img.astype(np.uint8)
            _box_edges(self, box, BETWEEN)
            info["between"] = _between_stats(L[inside], box, "pixels")
        self.text(_ghost_line(arm, label, target, q, width, info, bool(nb), bool(n)) + _between_line(info["between"]),
                  "white" if info["ik_ok"] else "red")
        self.ghosts.append(info)
        return info

    def _arms_now(self) -> np.ndarray:
        """Pixels the arms cover where they are now (from the model, slightly grown): not surfaces to hide behind."""
        if getattr(self, "_arms_mask", None) is None:
            self._arms_mask = _arms_now_mask(self.description, self.state(), self.K, self.T_cam,
                                             *self.img.shape[:2])
        return self._arms_mask

    def _raster(self, V, F, color, shade=True, near=0.03):
        return _raster(V, F, color, self.K, self.T_cam, *self.img.shape[:2], shade=shade, near=near)

    # ---- out
    def render(self) -> np.ndarray:
        """The image with the legend drawn on (a copy)."""
        img = self.img.copy()
        if self.legend:
            scale = max(0.4, img.shape[1] / 1300)
            lines = [(part, col) for s, col in self.legend for part in _wrap(s, img.shape[1] - 12, scale)]
            lh = int(22 * scale / 0.5)
            wmax = max(_text_width(s, scale) for s, _ in lines) + 12
            h = lh * len(lines) + 8
            band = img[:h, :min(wmax, img.shape[1])]
            band[:] = (band * 0.35).astype(np.uint8)
            for i, (s, col) in enumerate(lines):
                _text(img, s, (6, lh * (i + 1)), col, scale)
        return img

    def show(self, name: str, note: str | None = None, quiet: bool = False) -> str:
        """Save this overlay as views/NAME.png (with its legend) and put it on the dashboard; returns the path."""
        return show(name, self.render(), r=self.r, note=note, quiet=quiet)


# ------------------------------------------------------------------------------------------------ shared by 2D and 3D
def _tol(z):
    """How far behind a seen surface something must be to count as hidden: depth noise grows with distance."""
    return np.maximum(0.006, 0.01 * z)


def _posed_arm(r, desc, st, arm, target, q, width, free, parts, after=None) -> dict:
    """An arm's triangles in the world where it would be (r.ik for a target, from after= if given, or joint values q,
    or where it is now), moved by the base correction between the model and the driver (fitted at the joints it has
    now)."""
    if desc is None:
        raise ValueError("a ghost needs the robot's description: make the view with r, or pass description=")
    if st is None:
        raise ValueError("a ghost needs the arms' state: make the view with r, or pass state=")
    model = _model(desc["files"]["mjcf"])
    joints = list(desc["arm_joints"][arm])
    q_now = dict(zip(st.joints, map(float, st.q())))
    info = {"arm": arm, "ik_ok": True, "error_m": 0.0}
    if target is not None:
        if r is None:
            raise ValueError("ghost(target=...) asks r.ik: make the view with r, or pass q=")
        res = r.ik(np.asarray(target, float), ee=arm, free=free, **({"after": after} if after else {}))
        q_arm = dict(zip(res["joints"], map(float, res["q"])))
        info.update(ik_ok=bool(res["ok"]), error_m=float(res["error_m"]))
    elif q is not None:
        q_arm = dict(q) if isinstance(q, dict) else dict(zip(joints, map(float, q)))
    else:
        q_arm = {n: q_now[n] for n in joints if n in q_now}
    q_ghost = dict(q_now, **q_arm)
    finger = desc.get("finger_joint", {}).get(arm)
    if width is not None and finger:
        hi = desc.get("gripper_width_range_m", [0.0, 0.094])[1]
        q_ghost[finger] = float(np.clip(width, 0.0, hi)) / 2
    tcp_site = desc["tcp_sites"][arm]
    model.set(q_now)
    T_fix = st.ee[arm].T @ np.linalg.inv(model.site_T(tcp_site)) if arm in st.ee else np.eye(4)
    bodies = model.bodies(arm, parts, joints[0])
    geoms = [g for g in model.parts if model.body_of[g] in bodies]
    model.set(q_ghost)
    V, F = model.triangles(geoms, T_fix)
    T_tcp = T_fix @ model.site_T(tcp_site)
    jaw = None
    if finger and finger in model.jadr:                   # the line the jaws close along: the finger's slide axis
        jaw = T_fix[:3, :3] @ model.d.xaxis[model.m.joint(finger).id]
        jaw = jaw / np.linalg.norm(jaw)
    fingers = model.bodies(arm, "fingers", joints[0])
    is_finger = np.concatenate([np.full(len(model.parts[g][1]), model.body_of[g] in fingers) for g in geoms]) \
        if geoms else np.zeros(0, bool)
    info.update(q={n: q_ghost[n] for n in joints}, T=T_tcp, jaw_axis=jaw, jaw_box=_jaw_box(model, fingers, T_fix,
                                                                                         T_tcp, jaw))
    return {"V": V, "F": F, "info": info, "is_finger": is_finger}


def _jaw_box(model, fingers, T_fix, T_tcp, jaw):
    """The space between the jaws, as the fingers are posed now: a box in the gripper's frame (columns: the closing
    line s, the normal n, the approach a, at the TCP) from one finger's inner face to the other's, as wide as both
    fingers, and from the tips up JAW_DEPTH. None without two fingers and a closing line."""
    if jaw is None:
        return None
    app = T_tcp[:3, 2] / np.linalg.norm(T_tcp[:3, 2])
    j = jaw - (jaw @ app) * app
    j /= np.linalg.norm(j)
    G = np.eye(4)
    G[:3, 0], G[:3, 1], G[:3, 2], G[:3, 3] = j, np.cross(app, j), app, T_tcp[:3, 3]
    per = []
    for b in sorted(fingers):
        geoms = [g for g in model.parts if model.body_of[g] == b]
        if geoms:
            Wv, _ = model.triangles(geoms, T_fix)
            per.append(_in_frame(Wv, G))
    if len(per) != 2:
        return None
    tip = max(L[:, 2].max() for L in per)
    a0 = tip - JAW_DEPTH                                       # only the pads: higher up, a finger's mesh can
    per = [L[L[:, 2] >= a0] for L in per]                      # carry its slide, reaching across the middle
    if any(len(L) == 0 for L in per):
        return None
    A, B = sorted(per, key=lambda L: L[:, 0].mean())
    return {"T": G, "s": [float(A[:, 0].max()), float(B[:, 0].min())],
            "n": [float(max(A[:, 1].min(), B[:, 1].min())), float(min(A[:, 1].max(), B[:, 1].max()))],
            "a": [float(a0), float(tip)]}


def _to_world(P, T):
    """Points in a frame T (4x4, world <- frame) -> world: N x 3."""
    P = np.asarray(P, float)
    return np.stack([(P * T[i, :3]).sum(1) + T[i, 3] for i in range(3)], 1)


def _in_frame(P, G):
    """World points in the frame G (4x4): N x 3."""
    D = np.asarray(P, float) - G[:3, 3]
    return np.stack([(D * G[:3, i]).sum(1) for i in range(3)], 1)


def _between(P, box):
    """(which world points are between the jaws, their coordinates in the jaw box's frame)."""
    L = _in_frame(P, box["T"])
    inside = np.ones(len(L), bool)
    for i, k in enumerate("sna"):
        inside &= (L[:, i] > box[k][0]) & (L[:, i] < box[k][1])
    return inside, L


def _between_stats(L, box, unit="points") -> dict:
    """What is between the jaws: how many points, how much of the gap they span (2 mm bins along the closing line),
    and how far above the fingertips (mm)."""
    gap = box["s"][1] - box["s"][0]
    out = {"count": int(len(L)), "unit": unit, "gap_mm": round(1000 * gap, 1), "span_fraction": 0.0,
           "above_tips_mm": None}
    if len(L) and gap > 0:
        nb = max(1, int(np.ceil(gap / 0.002)))
        bins = np.clip(((L[:, 0] - box["s"][0]) / 0.002).astype(int), 0, nb - 1)
        out["span_fraction"] = round(len(np.unique(bins)) / nb, 2)
        up = 1000 * (box["a"][1] - L[:, 2])
        out["above_tips_mm"] = [round(float(up.min()), 1), round(float(up.max()), 1)]
    return out


def _between_line(st) -> str:
    if st is None:
        return ""
    if st["gap_mm"] <= 0:
        return " | jaws closed: no gap"
    if not st["count"]:
        return f" | between the jaws: nothing seen (gap {st['gap_mm']:.0f} mm)"
    lo, hi = st["above_tips_mm"]
    return (f" | between the jaws: {st['count']} {st['unit']} across {st['span_fraction']:.0%} of the "
            f"{st['gap_mm']:.0f} mm gap, {lo:.0f}-{hi:.0f} mm above the tips")


def _box_edges(o, box, col):
    """The jaw box's edges, thin, on an Overlay."""
    cv2 = _cv2()
    G = box["T"]
    corners = np.array([[s, n, a] for s in box["s"] for n in box["n"] for a in box["a"]])
    Wc = corners @ G[:3, :3].T + G[:3, 3]
    uv, z = o.uv(Wc)
    if (z <= 0).any():
        return
    for i in range(8):
        for k in (1, 2, 4):                                   # corners differing in one coordinate
            if i & k == 0:
                cv2.line(o.img, o._px(uv[i]), o._px(uv[i | k]), col, 1, cv2.LINE_AA)


def _ghost_line(arm, label, target, q, width, info, any_behind, any_shown) -> str:
    where = "where it is now" if target is None and q is None else "at the target" if target is not None else "at q"
    line = f"{label or arm + ' ghost'} ({where}"
    if width is not None:
        line += f", jaws {1000 * width:.0f} mm"
    if target is not None:
        line += ", IK ok" if info["ik_ok"] else f", IK MISSES by {1000 * info['error_m']:.0f} mm"
    line += ")" + (f": {info['behind_fraction']:.0%} of it behind a seen surface (red)" if any_behind else "")
    if not any_shown:
        line += ": out of this camera's view"
    return line


def _outline_and_mark(o, cover, T_tcp, col, label):
    cv2 = _cv2()
    cs, _ = cv2.findContours(cover.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o.img, cs, -1, col, 1, cv2.LINE_AA)
    uv, z = o.uv(T_tcp[:3, 3])
    if o._inside(uv[0], z[0]):
        c = o._px(uv[0])
        cv2.drawMarker(o.img, c, (0, 0, 0), cv2.MARKER_CROSS, 16, 4, cv2.LINE_AA)
        cv2.drawMarker(o.img, c, col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        _text(o.img, label, (c[0] + 9, c[1] - 9), col)


def _arms_now_mask(desc, st, K, T_cam, H, W) -> np.ndarray:
    """Pixels the arms cover where they are now, in a camera (from the model, slightly grown)."""
    cv2 = _cv2()
    mask = np.zeros((H, W), bool)
    if desc is None or st is None or T_cam is None:
        return mask
    model = _model(desc["files"]["mjcf"])
    model.set(dict(zip(st.joints, map(float, st.q()))))
    for arm in desc["arm_joints"]:
        if arm not in st.ee:
            continue
        T_fix = st.ee[arm].T @ np.linalg.inv(model.site_T(desc["tcp_sites"][arm]))
        bodies = model.bodies(arm, "arm", desc["arm_joints"][arm][0])
        geoms = [g for g in model.parts if model.body_of[g] in bodies]
        V, F = model.triangles(geoms, T_fix)
        _, Z = _raster(V, F, (255, 255, 255), K, T_cam, H, W, shade=False)
        mask |= np.isfinite(Z)
    return cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def _raster(V, F, color, K, T_cam, H, W, shade=True, near=0.03):
    """Triangles drawn far to near, each its own shade and its mean depth into Z: (colour H x W x 3 float, Z with inf
    where none). color: one (r, g, b), or one per face (len(F) x 3)."""
    cv2 = _cv2()
    out = np.zeros((H, W, 3), np.float32)
    Z = np.full((H, W), np.inf, np.float32)
    if len(F) == 0 or T_cam is None:
        return out, Z
    cols = np.asarray(color, np.float32)
    cols = np.broadcast_to(cols, (len(F), 3)) if cols.ndim == 1 else cols
    C = V @ T_cam[:3, :3].T + T_cam[:3, 3]
    tri = C[F]                                                 # k x 3 x 3, camera frame
    keep = tri[:, :, 2].min(1) > near                         # parts at the lens itself say nothing
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(1)
    keep &= (n * cen).sum(1) < 0                               # facing the camera
    tri, n, cen, cols = tri[keep], n[keep], cen[keep], cols[keep]
    uv = np.stack([K[0, 0] * tri[..., 0] / tri[..., 2] + K[0, 2], K[1, 1] * tri[..., 1] / tri[..., 2] + K[1, 2]], -1)
    lo, hi = uv.min(1), uv.max(1)
    keep = (hi[:, 0] >= 0) & (lo[:, 0] < W) & (hi[:, 1] >= 0) & (lo[:, 1] < H) & (np.abs(uv) < 1e5).all((1, 2))
    uv, n, cen, cols = uv[keep], n[keep], cen[keep], cols[keep]
    if not len(uv):
        return out, Z
    cos = np.abs((n * cen).sum(1)) / (np.linalg.norm(n, axis=1) * np.linalg.norm(cen, axis=1) + 1e-12)
    level = 0.35 + 0.65 * cos if shade else np.ones(len(cos))
    zmean = cen[:, 2]
    pts = np.round(uv * 16).astype(np.int32)
    for i in np.argsort(-zmean):
        cv2.fillConvexPoly(out, pts[i], tuple(float(x) for x in cols[i] * level[i]), cv2.LINE_8, 4)
        cv2.fillConvexPoly(Z, pts[i], float(zmean[i]), cv2.LINE_8, 4)
    return out, Z


# ------------------------------------------------------------------------------------------------ 3D views
BACKGROUND = (34, 36, 42)
GRIPPER_VIEWS = ("jaws", "along", "approach")
WORLD_VIEWS = ("top", "front", "back", "left", "right", "iso", "iso-right")
_WORLD_AZ_EL = {"front": (0, 0), "back": (180, 0), "left": (90, 0), "right": (-90, 0), "iso": (45, 35.26),
                "iso-right": (-45, 35.26)}           # the direction the camera looks from: azimuth from +x, elevation
FOV_DEG = 24.0                                                 # narrow: close to a true side view, heights compare


def look(eye, at, up=(0, 0, 1)) -> np.ndarray:
    """A camera pose (world <- optical: +z forward, +y down) at `eye` looking at `at`, `up` pointing up in the image."""
    eye, at, up = (np.asarray(x, float)[:3] for x in (eye, at, up))
    z = at - eye
    z /= np.linalg.norm(z)
    y = -(up - (up @ z) * z)
    if np.linalg.norm(y) < 1e-6:                               # looking along `up`: any turn of the image will do
        alt = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        y = -(alt - (alt @ z) * z)
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = np.cross(y, z), y, z, eye
    return T


def _self_surface(desc, st, lo, hi, spacing=0.003) -> np.ndarray:
    """Points on the arms' surfaces where they are now (the model at the state's joints, every arm), inside the box
    lo..hi (each triangle with a corner in it), about `spacing` apart: what the depth cameras see of the arms
    themselves."""
    if desc is None or st is None:
        return np.zeros((0, 3))
    model = _model(desc["files"]["mjcf"])
    model.set(dict(zip(st.joints, map(float, st.q()))))
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    rng = np.random.default_rng(0)
    for arm in desc["arm_joints"]:
        if arm not in st.ee:
            continue
        T_fix = st.ee[arm].T @ np.linalg.inv(model.site_T(desc["tcp_sites"][arm]))
        bodies = model.bodies(arm, "arm", desc["arm_joints"][arm][0])
        geoms = [g for g in model.parts if model.body_of[g] in bodies]
        V, F = model.triangles(geoms, T_fix)
        if not len(F):
            continue
        tri = V[F]
        near = np.any(np.all((tri >= lo) & (tri <= hi), axis=2), axis=1)
        tri = tri[near]
        if not len(tri):
            continue
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = np.ceil(2.0 * area / spacing ** 2).astype(int)
        idx = np.repeat(np.arange(len(tri)), n)
        a, b = rng.random(len(idx)), rng.random(len(idx))
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        t = tri[idx]
        out.append(t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0]))
        out.append(tri.reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def _not_self(P, S, margin=None) -> np.ndarray:
    """True for the points P that are farther than `margin` (default SELF_MARGIN) from every point of S."""
    margin = SELF_MARGIN if margin is None else margin
    if margin <= 0 or not len(S) or not len(P):
        return np.ones(len(P), bool)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(S).query(P, distance_upper_bound=margin)
    return ~np.isfinite(d)


def _cloud(frame, box, drop=None):
    """World points and their colours from one RGB-D frame, inside box [[x0, y0, z0], [x1, y1, z1]], with the
    spacing of neighbouring points (m) at their distance. Pixels in `drop` (the arms' own) and pixels at a jump in
    depth (where a sensor mixes the near and the far side into points in mid-air) are left out."""
    d = np.asarray(frame.depth, np.float32)
    H, W = d.shape
    dd = np.where(np.isfinite(d), d, 0)
    jump = np.zeros((H, W), np.float32)
    jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(dd[:, 1:] - dd[:, :-1]))
    jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(dd[:, 1:] - dd[:, :-1]))
    jump[1:] = np.maximum(jump[1:], np.abs(dd[1:] - dd[:-1]))
    jump[:-1] = np.maximum(jump[:-1], np.abs(dd[1:] - dd[:-1]))
    bad = jump > np.maximum(0.01, 0.02 * dd)
    if drop is not None and drop.shape == (H, W):
        bad |= drop
    v, u = np.mgrid[0:H, 0:W].reshape(2, -1)
    z = d.reshape(-1).astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < 3.0) & ~bad.reshape(-1)
    u, v, z = u[ok], v[ok], z[ok]
    K, T = np.asarray(frame.K, float), np.asarray(frame.T, float)
    x, y = (u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1]
    with np.errstate(all="ignore"):                            # (Accelerate's matmul warns spuriously on macOS)
        P = np.stack([x, y, z], 1) @ T[:3, :3].T + T[:3, 3]
    keep = np.all((P >= box[0]) & (P <= box[1]), axis=1)
    rgb = as_rgb(frame.rgb)
    if rgb.shape[:2] != (H, W):
        rgb = _cv2().resize(rgb, (W, H), interpolation=_cv2().INTER_NEAREST)
    return P[keep], rgb[v[keep], u[keep]], z[keep] / K[0, 0]


def _splat(groups, K, T_cam, H, W):
    """Points drawn as small squares, the nearest winning each pixel: (image H x W x 3 uint8, depth H x W)."""
    img = np.empty((H, W, 3), np.uint8)
    img[:] = BACKGROUND
    Z = np.full((H, W), np.inf, np.float32)
    us, vs, zs, cs = [], [], [], []
    for P, C, size in groups:
        if not len(P):
            continue
        with np.errstate(all="ignore"):
            Pc = P @ T_cam[:3, :3].T + T_cam[:3, 3]
        ok = Pc[:, 2] > 0.02
        Pc, C = Pc[ok], C[ok]
        u = np.round(K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]).astype(np.int64)
        v = np.round(K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]).astype(np.int64)
        for du in range(-(size // 2), size - size // 2):
            for dv in range(-(size // 2), size - size // 2):
                us.append(u + du), vs.append(v + dv), zs.append(Pc[:, 2]), cs.append(C)
    if not us:
        return img, Z
    u, v, z, c = (np.concatenate(a) for a in (us, vs, zs, cs))
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    li, z, c = v[ok] * W + u[ok], z[ok], c[ok]
    if not len(li):
        return img, Z
    order = np.lexsort((z, li))                                # by pixel, nearest first
    li, z, c = li[order], z[order], c[order]
    first = np.r_[True, li[1:] != li[:-1]]
    img.reshape(-1, 3)[li[first]] = c[first]
    Z.reshape(-1)[li[first]] = z[first]
    return img, Z


class View3D:
    """The cameras' depth fused into one coloured point cloud around `center`, seen from virtual cameras of your
    choosing, with Overlay's drawing calls (ghost, point, points, path, pose, text).

        v = viz.View3D(r, center=T)                   # fresh frames from every camera with a pose; T a 4x4 or a point
        g = v.ghost("left", T, width=0.05)            # red: inside or under a surface the real cameras see
        v.point(p, "edge")
        v.show("grasp-3d")                            # jaws, along, top, iso in a 2 x 2 grid
        v.show("grasp-3d", views=["jaws", (30, 15)])  # any of the presets, (azimuth, elevation) in degrees about the
                                                      # centre, or a camera pose (viz.look(eye, at))
    Presets about the first ghost's gripper (GRIPPER_VIEWS):
        "jaws"      through the gap between the fingers, along the gripper's normal: what would be between the jaws
                    (a slice jaw_section= thick each side of the fingers' plane, default 2 cm)
        "along"     along the line the jaws close on: the fingers' heights against the surface, in profile
        "approach"  down the gripper's approach axis from behind it: what the jaws come down on
    Presets about the world (WORLD_VIEWS): "top" (the far side up), "front" (from +x, facing the arms), "back",
    "left" (from +y), "right", "iso" (front-left, from above), "iso-right".
    radius= m around center that the views frame. A level view (a side preset, or within 20 degrees of level) is a
    cross-section: only the points within section= m (default 4 cm) of the centre along its line of sight, so a
    surface reads as its profile, with height lines every cm (section=None: all points). The cloud is what the cameras
    see: surfaces facing away from every camera, and anything they are blocked from, are missing.
    """

    def __init__(self, r=None, center=None, radius=0.12, cameras=None, frames=None, state=None, description=None,
                 size=(480, 360), section=0.04, jaw_section=0.02):
        if frames is None:
            if r is None:
                raise ValueError("give r (to take frames) or frames=")
            frames = r.cameras(cameras)
        self.r = r
        self.frames = {k: f for k, f in frames.items() if f.T is not None and getattr(f, "depth", None) is not None}
        self._state = state
        self.description = description if description is not None else (r.description if r is not None else None)
        self.center = None if center is None else self._point(center)
        self.radius, self.size, self.section, self.jaw_section = float(radius), tuple(size), section, jaw_section
        self.legend: list = []
        self.ghosts: list = []
        self._ghost_draw: list = []                            # (V, F, face colours, outline colour, label, T_tcp)
        self._marks: list = []
        self._views2d: dict = {}
        self._cloud_groups = None

    @staticmethod
    def _point(p):
        p = np.asarray(p, float)
        return p[:3, 3].copy() if p.shape == (4, 4) else p.reshape(-1)[:3].copy()

    def state(self):
        if self._state is None and self.r is not None:
            self._state = self.r.state()
        return self._state

    def _camera(self, name) -> Overlay:
        if name not in self._views2d:
            self._views2d[name] = Overlay(self.r, name, frame=self.frames[name], state=self.state(),
                                          description=self.description)
        return self._views2d[name]

    # ---- what to draw (drawn in every view)
    def text(self, line: str, color="white"):
        self.legend.append((str(line), _color(color, "white")))

    def point(self, *a, **kw):
        self._marks.append(("point", a, kw))

    def points(self, *a, **kw):
        self._marks.append(("points", a, kw))

    def path(self, *a, **kw):
        self._marks.append(("path", a, kw))

    def pose(self, *a, **kw):
        self._marks.append(("pose", a, kw))

    def ghost(self, arm: str, target=None, q=None, width=None, free=None, color="cyan", label=None,
              parts="gripper", after=None) -> dict:
        """As Overlay.ghost (parts default "gripper": its body drawn faintly, the fingers plainly). A face is red when,
        in at least half of the cameras with a depth reading where it would be (and at least two, when two have one),
        a surface they see is in front of it: it would be inside or under that surface.
        Returns {"q", "T", "jaw_axis", "ik_ok", "error_m", "behind_fraction"} (of its area). after= as Overlay.ghost."""
        g = _posed_arm(self.r, self.description, self.state(), arm, target, q, width, free, parts, after)
        V, F, info = g["V"], g["F"], g["info"]
        tri = V[F]
        cen = tri.mean(1)
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        seen_n = np.zeros(len(F), int)
        hid_n = np.zeros(len(F), int)
        for name in self.frames:
            o = self._camera(name)
            if o.depth is None:
                continue
            uv, z = o.uv(cen)
            H, W = o.depth.shape
            u, v = np.round(uv[:, 0]), np.round(uv[:, 1])
            inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui, vi = u[inside].astype(int), v[inside].astype(int)
            real = o.depth[vi, ui]
            seen = np.isfinite(real) & (real > 0.02) & ~o._arms_now()[vi, ui]
            hid = seen & (real < z[inside] - _tol(z[inside]))
            seen_n[inside] += seen
            hid_n[inside] += hid
        hidden = (2 * hid_n >= seen_n) & ((hid_n >= 2) | ((hid_n == 1) & (seen_n == 1)))   # one camera alone
                                                                                    # does not decide
        col = np.array(_color(color, "cyan"), np.float32)
        colors = np.where(hidden[:, None], np.array(BEHIND, np.float32), col)
        frac = float(area[hidden].sum() / area.sum()) if area.sum() > 0 else 0.0
        info.update(behind_fraction=round(frac, 3))
        self._ghost_draw.append((V, F, colors, tuple(int(c) for c in col), label or f"{arm} ghost", info["T"],
                                 g["is_finger"]))
        if self.center is None:
            self.center = info["T"][:3, 3].copy()
        info["between"] = None
        if info.get("jaw_box") is not None:                    # the cloud's points between the jaws
            L = [_between(P, info["jaw_box"]) for P, _, _ in self._groups()]
            L = np.concatenate([l[inside] for inside, l in L]) if L else np.zeros((0, 3))
            info["between"] = _between_stats(L, info["jaw_box"])
        self.legend.append((_ghost_line(arm, label, target, q, width, info, bool(hidden.any()), True)
                            + _between_line(info["between"]), (255, 255, 255) if info["ik_ok"] else _color("red")))
        self.ghosts.append(info)
        return info

    # ---- the views
    def _gripper_axes(self):
        """(TCP, approach, jaw line, normal) of the first ghost with a jaw line; the normal points away from +x so its
        views look from the front side."""
        g = next((g for g in self.ghosts if g.get("jaw_axis") is not None), None)
        if g is None:
            return None
        T = g["T"]
        app = T[:3, 2] / np.linalg.norm(T[:3, 2])
        jaw = g["jaw_axis"] - (g["jaw_axis"] @ app) * app
        jaw /= np.linalg.norm(jaw)
        nrm = np.cross(app, jaw)
        if nrm[0] > 0:
            nrm = -nrm
        if jaw[1] < 0:                                         # along the jaws: from the +y side
            jaw = -jaw
        return T[:3, 3].copy(), app, jaw, nrm

    def _view(self, spec) -> dict:
        """A view spec -> {"title", "T" (camera pose), "section" (half thickness, m, or None), "about" (the section's
        centre), "keep" (another point filter, or None)}."""
        c, d = self.center, self.radius * 1.25 / np.tan(np.radians(FOV_DEG / 2))
        out = {"section": None, "about": c, "keep": None}
        level = self.section
        if isinstance(spec, str):
            if spec in GRIPPER_VIEWS:
                axes = self._gripper_axes()
                if axes is None:
                    raise ValueError(f"view {spec!r} looks at a gripper: draw a ghost first (or use {WORLD_VIEWS})")
                tcp, app, jaw, nrm = axes
                if spec == "jaws":
                    return dict(out, title="jaws: through the gap, along the gripper's normal", about=tcp,
                                T=look(tcp - d * nrm, tcp, up=-app), section=self.jaw_section)
                if spec == "along":
                    return dict(out, title="along the jaws' closing line", about=tcp,
                                T=look(tcp - d * jaw, tcp, up=-app), section=level)
                return dict(out, title="approach: down the gripper's axis", T=look(tcp - d * app, tcp, up=nrm),
                            keep=lambda P: ((P - tcp) * app).sum(1) > -0.06)
            if spec == "top":
                return dict(out, title="top (far side up)", T=look(c + [0, 0, d], c, up=(1, 0, 0)))
            if spec not in WORLD_VIEWS:
                raise ValueError(f"views: {GRIPPER_VIEWS + WORLD_VIEWS}, (azimuth, elevation) or a 4x4 pose, "
                                 f"not {spec!r}")
            az, el = _WORLD_AZ_EL[spec]
            title = {"front": "front (from +x)", "back": "back (from -x)", "left": "left (from +y)",
                     "right": "right (from -y)", "iso": "iso (front-left)", "iso-right": "iso (front-right)"}[spec]
        else:
            arr = np.asarray(spec, float)
            if arr.shape == (4, 4):
                return dict(out, title="your camera", T=arr, section=level if abs(arr[2, 2]) < 0.35 else None)
            if arr.shape != (2,):
                raise ValueError(f"views: {GRIPPER_VIEWS + WORLD_VIEWS}, (azimuth, elevation) or a 4x4 pose, "
                                 f"not {spec!r}")
            az, el = (float(x) for x in arr)
            title = f"az {az:.0f} deg, el {el:.0f} deg"
        e = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                      np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
        return dict(out, title=title, T=look(c + d * e, c, up=(1, 0, 0) if el > 80 else (0, 0, 1)),
                    section=level if abs(el) <= 20 else None)

    def _groups(self):
        """The cloud around the centre, one group per camera: (points, colours, splat size)."""
        if self._cloud_groups is None:
            r = self.radius
            box = np.array([self.center - [1.6 * r, 1.6 * r, 1.6 * r], self.center + [1.6 * r, 1.6 * r, 1.6 * r]])
            box[0, 2] = min(box[0, 2], -0.02)                  # keep the table under it
            dist = r * 1.25 / np.tan(np.radians(FOV_DEG / 2))
            pixel_m = dist / ((self.size[0] / 2) / np.tan(np.radians(FOV_DEG / 2)))
            self._cloud_groups = []
            S = _self_surface(self.description, self.state(), box[0] - SELF_MARGIN_FIXED, box[1] + SELF_MARGIN_FIXED) \
                if self.description is not None and self.state() is not None else np.zeros((0, 3))
            for name, f in self.frames.items():
                drop = self._camera(name)._arms_now() if self.description is not None and self.state() is not None \
                    else None
                P, C, spacing = _cloud(f, box, drop)
                if len(P) and len(S):                          # the arms' own points, past their pixel mask
                    keep = _not_self(P, S, _self_margin(name))
                    P, C, spacing = P[keep], C[keep], spacing[keep]
                if len(P):                                     # each camera's points as big as their spacing here
                    self._cloud_groups.append((P, C, int(np.clip(np.ceil(np.median(spacing) / pixel_m), 1, 4))))
        return self._cloud_groups

    def _render_one(self, spec, groups):
        cv2 = _cv2()
        W, H = self.size
        view = self._view(spec)
        title, T = view["title"], view["T"]
        side = abs(T[2, 2]) < 0.35                             # the line of sight within 20 degrees of level
        f = (W / 2) / np.tan(np.radians(FOV_DEG / 2))
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        T_cam = np.linalg.inv(T)
        if view["section"]:                                    # a cross-section along the line of sight
            fwd, about, sec = T[:3, 2], view["about"], view["section"]
            groups = [(P[m], C[m], sz) for P, C, sz in groups for m in [np.abs(((P - about) * fwd).sum(1)) <= sec]]
            title += f", slice +-{100 * sec:.0f} cm"
        if view["keep"] is not None:
            groups = [(P[m], C[m], sz) for P, C, sz in groups for m in [view["keep"](P)]]
        img, Zp = _splat(groups, K, T_cam, H, W)
        from types import SimpleNamespace
        o = Overlay(frame=SimpleNamespace(rgb=img, depth=None, K=K, T=T), camera=title)
        c, r = self.center, self.radius
        dist = float(np.linalg.norm(T[:3, 3] - c))
        labels = []
        if side:                                               # height lines, every cm, labelled every 5 cm
            right = T[:3, 0] * 2.5 * r
            guide = o.img.copy()
            for zc in range(int(np.floor((c[2] - 1.3 * r) * 100)), int(np.ceil((c[2] + 1.3 * r) * 100)) + 1):
                z = zc / 100
                uv, zz = o.uv(np.array([[c[0], c[1], z]]) + np.stack([-right, right]))
                if (zz <= 0).any():
                    continue
                p0, p1 = o._px(uv[0]), o._px(uv[1])
                strong = zc % 5 == 0
                cv2.line(guide, p0, p1, (150, 150, 150) if strong else (95, 95, 95), 1, cv2.LINE_AA)
                if strong:
                    labels.append((f"z {zc} cm", (4, p0[1] - 3)))
            o.img = (0.6 * o.img + 0.4 * guide).astype(np.uint8)
            o.img[np.isfinite(Zp)] = img[np.isfinite(Zp)]     # the cloud over the lines
            for s, at in labels:
                _text(o.img, s, at, (190, 190, 190), 0.4)
        for V, F, colors, col, label, T_tcp, is_finger in self._ghost_draw:
            cover_all = np.zeros((H, W), bool)
            # the gripper's body faintly (what is under it stays visible), then the fingers plainly
            for faces, a_front, a_back in ((~is_finger, 0.25, 0.1), (is_finger, 0.6, 0.2)):
                if not faces.any():
                    continue
                rgb, Z = _raster(V, F[faces], colors[faces], K, T_cam, H, W, near=0.02)
                cover = np.isfinite(Z)
                front = cover & (Z <= Zp + 0.002)              # the ghost nearer than the cloud: plain to see
                back = cover & ~front                          # points in front of it: seen through them, faintly
                im = o.img.astype(np.float32)
                im[front] = im[front] * (1 - a_front) + rgb[front] * a_front
                im[back] = im[back] * (1 - a_back) + rgb[back] * a_back
                o.img = im.astype(np.uint8)
                cover_all |= cover
            _outline_and_mark(o, cover_all, T_tcp, col, label)
        for g in self.ghosts:
            if g.get("jaw_box") is not None:
                _box_edges(o, g["jaw_box"], BETWEEN)
        for name, a, kw in self._marks:
            getattr(o, name)(*a, **kw)
        px_per_m = f / dist                                    # a 2 cm scale bar at the centre's distance
        L = int(round(0.02 * px_per_m))
        cv2.line(o.img, (W - 14 - L, H - 14), (W - 14, H - 14), (230, 230, 230), 2)
        _text(o.img, "2 cm", (W - 14 - L, H - 20), (230, 230, 230), 0.4)
        R = T_cam[:3, :3]                                      # the world's axes, bottom left
        for i, colr in enumerate(("red", "green", "blue")):
            dvec = R[:, i][:2]
            if np.linalg.norm(dvec) < 1e-3:
                cv2.circle(o.img, (30, H - 30), 3, COLORS[colr], -1)
                continue
            end = (int(30 + 22 * dvec[0] / max(np.linalg.norm(dvec), 0.35)),
                   int(H - 30 + 22 * dvec[1] / max(np.linalg.norm(dvec), 0.35)))
            cv2.arrowedLine(o.img, (30, H - 30), end, COLORS[colr], 2, cv2.LINE_AA, tipLength=0.25)
            _text(o.img, "xyz"[i], (end[0] + 2, end[1] + 4), COLORS[colr], 0.4)
        _text(o.img, title, (6, 16), (255, 255, 255), 0.45)
        cv2.rectangle(o.img, (0, 0), (W - 1, H - 1), (70, 70, 70), 1)
        return o.img

    def render(self, views=None, legend: bool = True) -> np.ndarray:
        """The views (default: jaws, along, top and iso about a ghost; front, left, top and iso without one) in a grid
        under the legend (legend=False: the grid alone), as an RGB image."""
        if self.center is None:
            marks = [self._point(a[0]) for n, a, _ in self._marks if n in ("point", "pose") and a]
            if not marks:
                raise ValueError("View3D needs center= (or a ghost, or a point, to centre on)")
            self.center = np.mean(marks, axis=0)
        if views is None:
            views = ["jaws", "along", "top", "iso"] if self._gripper_axes() is not None else \
                ["front", "left", "top", "iso"]
        views = [views] if isinstance(views, str) else list(views)
        groups = []
        for P, C, size in self._groups():                      # points between any ghost's jaws stand out
            between = np.zeros(len(P), bool)
            for g in self.ghosts:
                if g.get("jaw_box") is not None:
                    between |= _between(P, g["jaw_box"])[0]
            if between.any():
                C = C.copy()
                C[between] = BETWEEN
            groups.append((P, C, size))
        tiles = [self._render_one(v, groups) for v in views]
        W, H = self.size
        cols = 2 if len(tiles) == 4 else min(3, len(tiles))
        rows = -(-len(tiles) // cols)
        grid = np.zeros((rows * H, cols * W, 3), np.uint8)
        for i, t in enumerate(tiles):
            grid[(i // cols) * H:(i // cols + 1) * H, (i % cols) * W:(i % cols + 1) * W] = t
        if not legend:
            return grid
        legend = self.legend + [(f"point cloud from {', '.join(self.frames) or 'no camera'}; red: inside or under a "
                                 "surface the cameras see; faint: behind points in this view; pink: points between "
                                 "the jaws (the box)", (170, 170, 170))]
        legend = [(part, col) for s, col in legend for part in _wrap(s, grid.shape[1] - 12, 0.45)]
        lh = 20
        band = np.zeros((lh * len(legend) + 8, grid.shape[1], 3), np.uint8)
        band[:] = (20, 20, 24)
        for i, (s, col) in enumerate(legend):
            _text(band, s, (6, lh * (i + 1)), col, 0.45)
        return np.concatenate([band, grid])

    def show(self, name: str, views=None, note: str | None = None, quiet: bool = False) -> str:
        """Save the views as views/NAME.png (and on the dashboard); returns the path."""
        return show(name, self.render(views), r=self.r, note=note, quiet=quiet)
