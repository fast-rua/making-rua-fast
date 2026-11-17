"""Tasks on the bimanual YAM station (world: origin on the table top midway between the bases, +x forward, +z up)."""
from __future__ import annotations

import numpy as np

from robotd.robots.yam_station.scene import Cloth, Obj, SceneConfig
from tasks import Task

RED, GREEN, BLUE = (0.85, 0.15, 0.15, 1), (0.15, 0.7, 0.2, 1), (0.2, 0.4, 0.8, 1)


def _pick_cube_check(backend):
    z = float(backend.object_pose("cube")[2, 3])
    return {"success": z > 0.08, "cube_z": z}


def _two_cubes_check(backend):
    zr, zg = float(backend.object_pose("red")[2, 3]), float(backend.object_pose("green")[2, 3])
    return {"success": zr > 0.08 and zg > 0.08, "red_z": zr, "green_z": zg}


# ------------------------------------------------------------------------------------------------ towel
TOWEL = dict(w=0.30, h=0.22)          # m; across (y) and along (x) when laid flat. The YAM holds cloth up to
                                       # ~0.45 m, so a towel hanging from any point (<= its 0.37 m diagonal) clears
                                       # the table
TOWEL_SPACING = 0.015
TOWEL_RGBA = (0.85, 0.1, 0.1, 1)       # red: no part of the robot or the table is red
TOWEL_DROP = (0.32, 0.0)               # where the crumpled towel lands (xy)
TOWEL_TABLE_X = (-0.20, 0.52)          # the towel tasks' table ends 0.52 m in front of the bases: a ledge in reach


def towel_mesh(w=0.30, h=0.22, spacing=0.015):
    """A flat rectangular towel on a square grid in its own frame: x along h (0..h), y across w (-w/2..w/2).
    Returns points (N x 2), triangles, and the grid shape (rows along x, columns along y)."""
    nx, ny = int(round(h / spacing)) + 1, int(round(w / spacing)) + 1
    xs, ys = np.linspace(0, h, nx), np.linspace(-w / 2, w / 2, ny)
    pts = np.array([(x, y) for x in xs for y in ys])
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1
            tris += [(a, b, d), (a, d, c)]
    return pts, np.array(tris), (nx, ny)


def _towel_cloth(initial=None):
    pts, tris, _ = towel_mesh(spacing=TOWEL_SPACING, **TOWEL)
    return Cloth("towel", pts, tris, pos=(TOWEL_DROP[0] - TOWEL["h"] / 2, TOWEL_DROP[1], 0.005), rgba=TOWEL_RGBA,
                 mass=0.12, initial=initial)


def crumpled_towel(seed: int = 0) -> np.ndarray:
    """A crumpled pile, made the way a person makes one: the towel is picked up by a point near its middle, lifted
    about 45 cm while turning and swinging, and let go above the drop point. Simulated once per seed and cached."""
    import hashlib
    import os
    import mujoco
    from robotd.robots.yam_station import scene as S
    key = hashlib.sha1(repr((seed, TOWEL, TOWEL_SPACING, TOWEL_DROP, TOWEL_TABLE_X, 8)).encode()).hexdigest()[:12]
    cache = os.path.join(os.path.expanduser("~/.cache/robot-cell"), f"towel-crumpled-{key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    rng = np.random.default_rng(seed)
    pts, _, (nx, ny) = towel_mesh(spacing=TOWEL_SPACING, **TOWEL)
    cloth = _towel_cloth()
    held = int(np.argmin(np.linalg.norm(pts - (pts.mean(0) + rng.uniform(-0.06, 0.06, 2)), axis=1)))
    spec = S.build_spec(SceneConfig(cloths=[cloth], table_x=TOWEL_TABLE_X))
    m = spec.compile()
    body = m.body(m.flex_vertbodyid[m.flex_vertadr[0] + held]).name
    start = cloth.world_points()[held]
    spec.worldbody.add_body(name="lifter", mocap=True, pos=list(start))
    spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1="lifter", name2=body, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      data=[0.0] * 12)
    m = spec.compile()
    d = mujoco.MjData(m)
    arm_q = [m.jnt_qposadr[m.joint(n).id] for a in S.ARMS for n in S.ARM_JOINTS[a]]
    d.qpos[arm_q] = np.concatenate([S.READY_Q[a] for a in S.ARMS])
    acts = [m.actuator(f"{n}_motor").id for a in S.ARMS for n in S.ARM_JOINTS[a]]
    d.ctrl[acts] = d.qpos[arm_q]
    robot_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.joint(j).name.startswith(("left_joint", "right_joint"))]
    mujoco.mj_forward(m, d)
    eq = m.equality("lifter_" + body).id if False else m.neq - 1
    lift_s, hold_s, fall_s = 1.2, 0.6, 1.6
    top = np.array([TOWEL_DROP[0], TOWEL_DROP[1], 0.45]) + np.r_[rng.uniform(-0.04, 0.04, 2), 0]
    swing = rng.uniform(0.02, 0.06) * np.array([np.cos(rng.uniform(0, 6.3)), np.sin(rng.uniform(0, 6.3)), 0])
    for k in range(int((lift_s + hold_s + fall_s) / m.opt.timestep)):
        t = k * m.opt.timestep
        if t < lift_s + hold_s:
            a = min(1.0, t / lift_s)
            s_ = a * a * (3 - 2 * a)
            d.mocap_pos[0] = start + s_ * (top - start) + np.sin(2 * np.pi * 1.5 * t) * swing * (t < lift_s + hold_s)
        else:
            d.eq_active[eq] = 0
        d.qfrc_applied[robot_dofs] = d.qfrc_bias[robot_dofs]
        mujoco.mj_step(m, d)
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "towel_flex")
    v = d.flexvert_xpos[m.flex_vertadr[f]:m.flex_vertadr[f] + m.flex_vertnum[f]].copy()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, v)
    return v


def _coverage(v, tris, cell=0.004):
    """Area (m^2) the towel covers seen from above: its triangles rasterised on a grid."""
    xy = v[:, :2]
    lo = xy.min(0) - cell
    shape = np.ceil((xy.max(0) + cell - lo) / cell).astype(int) + 1
    gx, gy = np.meshgrid(np.arange(shape[0]) * cell + lo[0], np.arange(shape[1]) * cell + lo[1], indexing="ij")
    cov = np.zeros(shape, bool)
    for t in tris:
        a, b_, c_ = xy[t]
        i0, j0 = np.floor((np.minimum(np.minimum(a, b_), c_) - lo) / cell).astype(int)
        i1, j1 = np.ceil((np.maximum(np.maximum(a, b_), c_) - lo) / cell).astype(int) + 1
        X, Y = gx[i0:i1, j0:j1], gy[i0:i1, j0:j1]
        d = (b_[1] - c_[1]) * (a[0] - c_[0]) + (c_[0] - b_[0]) * (a[1] - c_[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b_[1] - c_[1]) * (X - c_[0]) + (c_[0] - b_[0]) * (Y - c_[1])) / d
        l2 = ((c_[1] - a[1]) * (X - c_[0]) + (a[0] - c_[0]) * (Y - c_[1])) / d
        cov[i0:i1, j0:j1] |= (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
    return float(cov.sum() * cell * cell)


def _flatten_check(backend):
    v = backend.cloth_vertices("towel")
    _, tris, _ = towel_mesh(spacing=TOWEL_SPACING, **TOWEL)
    frac = _coverage(v, tris) / (TOWEL["w"] * TOWEL["h"])
    return {"success": frac > 0.9 and v[:, 2].max() < 0.03, "coverage": frac, "max_height_m": float(v[:, 2].max())}


def _fold_towel_check(backend):
    """Folded once in half, either way: every vertex lies on its mirror image across one of the towel's centre
    lines (mean distance under 3 cm), the towel covers about half its flat area, and it lies on the table."""
    v = backend.cloth_vertices("towel")
    _, tris, (nx, ny) = towel_mesh(spacing=TOWEL_SPACING, **TOWEL)
    grid = v.reshape(nx, ny, 3)
    pair = {"across_length": np.linalg.norm(grid - grid[::-1], axis=2).mean(),
            "across_width": np.linalg.norm(grid - grid[:, ::-1], axis=2).mean()}
    best = min(pair.values())
    frac = _coverage(v, tris) / (TOWEL["w"] * TOWEL["h"])
    return {"success": bool(best < 0.03 and 0.4 < frac < 0.62 and v[:, 2].max() < 0.04),
            "mirror_distance_m": {k: float(x) for k, x in pair.items()}, "coverage": frac,
            "max_height_m": float(v[:, 2].max())}


def _towel_scene(seed=0):
    return SceneConfig(cloths=[_towel_cloth(initial=crumpled_towel(seed))], table_x=TOWEL_TABLE_X)


TASKS = {t.name: t for t in [
    Task("empty", "yam_station", lambda: SceneConfig(), None),
    Task("pick_cube", "yam_station",
         lambda: SceneConfig(objects=[Obj("cube", size=(0.02, 0.02, 0.02), pos=(0.36, 0.28, 0.02), mass=0.05,
                                          rgba=RED)]),
         "Pick up the red cube on the table and lift it so that its bottom is at least 6 cm above the table. "
         "Keep holding it, then call r.finish() with a short summary.",
         _pick_cube_check),
    Task("press_box", "yam_station",
         lambda: SceneConfig(objects=[Obj("box", size=(0.05, 0.05, 0.05), pos=(0.35, -0.30, 0.05), mass=2.0,
                                          rgba=BLUE)]),
         None),
    Task("two_cubes", "yam_station",
         lambda: SceneConfig(objects=[Obj("red", size=(0.02, 0.02, 0.02), pos=(0.36, 0.25, 0.02), rgba=RED),
                                      Obj("green", size=(0.02, 0.02, 0.02), pos=(0.36, -0.25, 0.02), rgba=GREEN)]),
         "There is a red cube and a green cube on the table. Pick up both, one in each gripper, and hold them so "
         "that their bottoms are at least 6 cm above the table. Then call r.finish() with a short summary.",
         _two_cubes_check),
    Task("flatten_towel", "yam_station", _towel_scene,
         "A red towel, 30 x 22 cm, lies crumpled on the table. The table ends 0.52 m in front of the arm bases. "
         "Spread the towel out flat on the table, then call r.finish() with a short summary.", _flatten_check),
    Task("fold_towel", "yam_station", _towel_scene,
         "A red towel, 30 x 22 cm, lies crumpled on the table. The table ends 0.52 m in front of the arm bases. "
         "Spread it out flat, then fold it once in half, edge to edge. Leave it lying folded on the table, then call r.finish() with a short summary.",
         _fold_towel_check),
    Task("fold_flat_towel", "yam_station", lambda: SceneConfig(cloths=[_towel_cloth()], table_x=TOWEL_TABLE_X),
         "A red towel, 30 x 22 cm, lies flat on the table. The table ends 0.52 m in front of the arm bases. Fold it "
         "once in half, edge to edge. Leave it lying folded on the table, then call r.finish() with a short summary.",
         _fold_towel_check),
]}
