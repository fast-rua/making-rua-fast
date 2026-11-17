"""Shared helpers for the towel policies: seeing the towel, pinching cloth, folding over. Ordinary code over the
robot API; nothing here is part of the API."""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import robot
from robot import when

r = robot.connect()
BASE_Y = {"left": 0.305, "right": -0.305}
TABLE_PINCH_Z = 0.005             # TCP height for pinching cloth off the table: fingertips level with the cloth
_PEEK = __import__("os").environ.get("TOWEL_DEV_PEEK")       # developers only: needs a driver started with --dev


def dev_peek(name):
    """Save free-viewpoint renders (front, side) and print the true coverage. No-op unless TOWEL_DEV_PEEK is set."""
    if not _PEEK:
        return
    from robotd.core.recorder import write_png
    import tasks.yam_station as Y
    for view, az, el in (("front", 180, -25), ("side", 90, -15)):
        img = r._call("render_free", view=dict(lookat=[0.38, 0.0, 0.1], distance=0.95, azimuth=az, elevation=el))
        write_png(f"{_PEEK}/{name}-{view}.png", img["rgb"])
    v = np.array(r._call("dev_truth")["cloth"]["towel"])
    _, tris, _ = Y.towel_mesh(spacing=Y.TOWEL_SPACING, **Y.TOWEL)
    print(f"  [dev] {name}: true coverage {Y._coverage(v, tris) / (Y.TOWEL['w'] * Y.TOWEL['h']):.2f}, "
          f"x {v[:, 0].min():.2f}..{v[:, 0].max():.2f}, z {v[:, 2].min():.2f}..{v[:, 2].max():.2f}")


def towel_points(max_z=0.2):
    """The towel as world points (N x 3) from the top camera: the largest red region."""
    f = r.cameras(["top"])["top"]
    mask = red(f.rgb)
    if not mask.any():                             # hidden (e.g. behind an arm): no points
        return np.zeros((0, 3))
    # drop the outline pixels: at a silhouette edge, colour and depth can come from different surfaces (towel
    # colour, table depth), which puts stray points far behind the towel
    pts = f.points(ndimage.binary_erosion(largest(mask)))
    return pts[pts[:, 2] < max_z]


def red(rgb):
    """The towel's colour: red well above green and blue (the robot has no red parts; shadows keep the ratio)."""
    c = rgb.astype(int)
    return (c[..., 0] > 2.2 * c[..., 1]) & (c[..., 0] > 2.2 * c[..., 2]) & (c[..., 0] > 40)


def largest(mask):
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    return labels == 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))


_HAND_TO_CAM = {}


def look_at(arm, target, distances=(0.28, 0.34, 0.22), azimuths_deg=(0, 25, -25, 50, -50), height=0.05):
    """Point arm's wrist camera at a world point: the hand-to-camera offset comes from one camera reading; the
    camera is placed `distance` away on the arm's own side, a little above the target, looking at it. Returns
    the towel's points seen from there (N x 3, world)."""
    cam = f"{arm}_wrist"
    if arm not in _HAND_TO_CAM:
        f = r.cameras([cam])[cam]
        _HAND_TO_CAM[arm] = np.linalg.inv(r.T(arm)) @ f.T
    X = _HAND_TO_CAM[arm]
    target = np.asarray(target, float)
    out = np.array([0.0, 1.0 if arm == "left" else -1.0, 0.0])
    for dist in distances:
        for az in np.radians(azimuths_deg):
            d = np.array([np.cos(az) * out[0] - np.sin(az) * out[1], np.sin(az) * out[0] + np.cos(az) * out[1], 0])
            c = target + dist * d + np.array([0, 0, height])
            z = (target - c) / np.linalg.norm(target - c)
            y = np.array([0, 0, -1.0]) - z * (-z[2])
            y /= np.linalg.norm(y)
            T_cam = np.eye(4)
            T_cam[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
            T_cam[:3, 3] = c
            try:
                r.move(T_cam @ np.linalg.inv(X), ee=arm, speed=0.6)
            except robot.InvalidCommand:
                continue
            f = r.cameras([cam])[cam]
            return f.points(largest(red(f.rgb)))
    raise RuntimeError(f"{arm} cannot point its camera at {np.round(target, 3)}")


def hanging_bottom(arm, x, y):
    """Height of the lowest point of a towel hanging near (x, y), seen from the side by arm's wrist camera; None if
    not seen. The camera aims low enough that the bottom is in the upper part of the image: the arm's own fingers
    fill the bottom of a wrist image and would hide it."""
    cam = f"{arm}_wrist"
    for z, above in ((0.15, 0.10), (0.12, 0.16), (0.20, 0.05), (0.08, 0.22)):     # aim point, camera height above it
        try:
            look_at(arm, [x, y, z], height=above)
        except RuntimeError:
            continue
        f = r.cameras([cam])[cam]
        m = largest(red(f.rgb))
        rows = np.nonzero(m.any(axis=1))[0]
        if len(rows) and rows.max() < 0.7 * m.shape[0]:
            return float(f.points(m)[:, 2].min())
    return None


def down(p, z, yaw=0.0, tilt=0.0):
    """Jaws down at (x, y, z), closing along the horizontal direction yaw, the jaws tilted by `tilt` rad toward
    +x (straight down reaches only ~0.10-0.15 m above the table; a tilt reaches higher)."""
    return robot.pose([p[0], p[1], z], rpy=[np.pi, -tilt, np.pi + yaw])


def pitched_pinch(p, z, jaw_angle, pitch_deg):
    """Jaws reaching forward and down by pitch_deg, closing horizontally along jaw_angle (far out, near the table
    edge, jaws straight down do not reach)."""
    pr = np.radians(pitch_deg)
    ja = np.array([np.cos(jaw_angle), np.sin(jaw_angle), 0.0])
    hz = np.array([-ja[1], ja[0], 0.0])
    hz = hz if hz[0] >= 0 else -hz
    zax = np.cos(pr) * hz + np.sin(pr) * np.array([0, 0, -1.0])
    return robot.pose([p[0], p[1], z], R_mat=np.stack([ja, np.cross(zax, ja), zax], 1))


def down_any(p, z, yaw=0.0):
    """The candidate grasp poses at p, z for a jaw direction yaw, simplest first: straight down, then tilted, then
    reaching forward and down; each also with yaw + pi (the same grasp for a parallel gripper)."""
    out = [down(p, z, y, t) for t in (0.0, 0.35, -0.35) for y in (yaw, yaw + np.pi)]
    return out + [pitched_pinch(p, z, y, pd) for pd in (75, 60, 45) for y in (yaw, yaw + np.pi)]


def pitched(p, pitch_deg=45.0, yaw=0.0):
    """Jaws pointing forward and pitched down by pitch_deg (0: straight forward, 90: straight down). Near the table
    edge (x 0.22-0.58 m) and low, 15-60 deg reaches everywhere; straight down and straight forward do not."""
    return robot.pose(p, rpy=[0, np.radians(90 + pitch_deg), yaw])


def forward(p, yaw=0.0):
    """Jaws pointing forward (+x turned by yaw), fingers vertical: how the arms hold cloth up high."""
    return robot.pose(p, rpy=[0, np.pi / 2, yaw])


def reach(arm, poses_by_yaw, **kw):
    """Move to the first reachable pose (a parallel gripper is symmetric: yaw and yaw + pi are the same grasp)."""
    last = None
    for T in poses_by_yaw:
        try:
            return r.move(T, ee=arm, **kw)
        except robot.InvalidCommand as e:
            last = e
    raise last


def pinch(arm, p, z_top, n=None, insets=(0.004, 0.0, 0.008), force=3.0, opening=0.035, depth=0.0):
    """Pinch cloth at p (xy): from above, down until the fingers press on it (force), close until they stall.
    With n (outward direction of an edge or corner), the jaws go across it with the outer finger just past it;
    without n, they pinch a fold of the pile. Retries a little further out or in. Returns the yaw used."""
    if n is None:
        tries = [(0.0, np.asarray(p, float)), (np.pi / 2, np.asarray(p, float))]
    else:
        n = np.asarray(n, float) / np.linalg.norm(n)
        tries = [(np.arctan2(n[1], n[0]), np.asarray(p, float) - i * n) for i in insets]
    for yaw, c in tries:
        r.gripper(arm, opening)
        reach(arm, down_any(c, z_top + 0.04, yaw))
        r.zero_force(arm)
        t = r.move_at({"z": -0.02}, ee=arm, max_travel=z_top + 0.04, until={"contact": when.force(arm, "-z") > force})
        if depth:                              # into a pile: the jaws close round a handful, not a surface fold
            r.move_at({"z": -0.02}, ee=arm, max_travel=depth, until={"hard": when.force(arm, "-z") > 4 * force})
        elif n is not None and r.T(arm)[2, 3] > TABLE_PINCH_Z:   # an edge on the table: fingertips down to its height
            T = r.T(arm)
            T[2, 3] = TABLE_PINCH_Z
            r.move(T, ee=arm, line=True, speed=0.02)
        g = r.gripper(arm, 0.0, force=15)
        print(f"  {arm}: pinch at {np.round(c, 3)} yaw {np.degrees(yaw):.0f}: touch {t.reason} z {t.at.T[arm][2, 3]:.3f}"
              f", grip {g.reason} {g.width * 1000:.1f} mm")
        if g.reason == "stalled" and holding(arm):
            r.step(10)                         # a grip can stall, then slip as the cloth squeezes out
            if holding(arm):
                return yaw
            held = r.state().grippers[arm]["width"]
            print(f"  {arm}: the cloth slipped out (jaws now at {held * 1000:.1f} mm)")
        r.move([robot.pose([0, 0, 0.04])], times=[0.5], ee=arm, relative=True)
    raise RuntimeError(f"{arm}: could not pinch the cloth at {np.round(p, 3)}")


def corner_pinch(arm, tip, n, opening=0.08, beyond=0.03, inside=0.03):
    """Pinch a corner lying on the table, pointing along n: the open jaws close across n; they go down onto the bare
    table just beyond the tip, slide in along the table past the tip, and close. The fingertips stay at table height,
    so they close under the corner's edges (from above, a fingertip on the cloth holds both up and the jaws close
    over the corner). Wide open and sliding 3 cm past the tip, a corner seen a few cm off still ends up between the
    fingers (a right-angled corner is 6 cm wide 3 cm in). Returns True when holding; otherwise opens and backs up."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    yaw = np.arctan2(n[1], n[0]) + np.pi / 2
    start = np.asarray(tip, float) + beyond * n
    r.gripper(arm, opening)
    reach(arm, down_any(start, 0.05, yaw))
    r.zero_force(arm)
    t = r.move_at({"z": -0.02}, ee=arm, max_travel=0.06, until={"contact": when.force(arm, "-z") > 3.0})
    T = r.T(arm)
    # slide with the fingertips down at the cloth's own height, whatever they touched: on the cloth's surface they
    # would close over it (the jaws reach TCP z 0.005 without pressing the table)
    T[2, 3] = min(T[2, 3] + 0.002, TABLE_PINCH_Z)
    T[:2, 3] += (np.asarray(tip, float) - inside * n) - start
    r.move(T, ee=arm, line=True, speed=0.03)
    g = r.gripper(arm, 0.0, force=15)
    r.step(10)
    ok = holding(arm)
    print(f"  {arm}: corner pinch at {np.round(tip, 3)} from {np.round(n, 2)}: touch {t.reason} z "
          f"{t.at.T[arm][2, 3]:.3f}, grip {g.reason} {g.width * 1000:.1f} mm{'' if ok else ', not holding'}")
    if not ok:
        r.gripper(arm, opening)
        r.move([robot.pose([0, 0, 0.04])], times=[0.5], ee=arm, relative=True)
    return ok


def grab_corner(arm, tip, n, **kw):
    """A corner pinch, else pinches across the edge at the tip (raises RuntimeError when both fail)."""
    if not corner_pinch(arm, tip, n, **kw):
        pinch(arm, tip, 0.01, n=n)


def park(arms=("left", "right")):
    """Out of the top camera's view of the table."""
    for a in arms:
        r.move(down([0.2, BASE_Y[a] * 0.9], 0.15), ee=a)


def holding(arm):
    """Cloth in the jaws: closed on cloth they stop at its thickness (one layer: 4-6 mm); closed on nothing they meet
    below 2 mm; open, they hold nothing."""
    return 0.0025 < r.state().grippers[arm]["width"] < 0.05


def lift_off(arm, dz=0.06):
    """Straight up by dz in the current orientation (off the table before turning the hand)."""
    T = r.T(arm)
    T[2, 3] += dz
    r.move(T, ee=arm, line=True)


def shake(arms, strokes=3, up=0.05, down=0.03, period=0.44):
    """The whisk: quick up-and-down strokes of the hand(s) holding the towel, shaking out its folds."""
    offsets = [robot.pose([0, 0, dz]) for _ in range(strokes) for dz in (up, -down)] + [robot.pose([0, 0, 0])]
    r.move({a: offsets for a in arms}, times=list(np.arange(1, len(offsets) + 1) * period / 2), ee=list(arms),
             relative=True)


def rectangle(pts):
    """Smallest enclosing rectangle of the towel seen from above: centre, long axis, short axis, side lengths."""
    xy = pts[:, :2]
    hull = xy[ConvexHull(xy).vertices]
    best = None
    for e in np.diff(np.vstack([hull, hull[:1]]), axis=0):
        a = e / np.linalg.norm(e)
        b = np.array([-a[1], a[0]])
        la, lb = np.ptp(hull @ a), np.ptp(hull @ b)
        if best is None or la * lb < best[0]:
            best = (la * lb, a, b, la, lb)
    _, a, b, la, lb = best
    if la < lb:
        a, b, la, lb = b, -a, lb, la
    c = 0.5 * ((hull @ a).min() + (hull @ a).max()) * a + 0.5 * ((hull @ b).min() + (hull @ b).max()) * b
    return c, a, b, la, lb
