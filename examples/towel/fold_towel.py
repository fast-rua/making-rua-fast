"""Fold a crumpled towel in half (the fold_towel task: a red 30 x 22 cm towel; the table's front edge is 0.52 m in
front of the arm bases).

1. hang: pinch the top of the pile and lift it; the towel hangs from that point.
2. corner: a hanging rectangle's lowest point is a corner. Lay the towel down sideways so the lowest point lands
   first and the bottom of the towel lies out in a strip with that point at its tip; the free arm pinches the tip
   (corner A) and the first arm lets go.
3. opposite corner: hanging from A, the lowest point is the opposite corner C; the same lay-down, and the other arm
   pinches it. The arms now hold the ends of a diagonal. (If the towel hangs much shorter than its diagonal from A,
   A is not a corner, and its lowest point is: pick that instead, as in 2.)
4. adjacent corner: held by a diagonal, about its length apart, the towel hangs folded along it in two triangular
   flaps whose lowest points are the other two corners. Lay it down while drawing it back toward the robot: the
   corners touch first and stay, and the flaps lie out flat as a doubled triangle with the corners at its apexes.
   The apex nearer C is joined to A by a long edge (and the one nearer A to C), so the hand nearer the apex lets go
   and pinches it: the arms hold the ends of a long edge.
5. flatten: lift the towel by that edge, shake it gently, hang it over the front edge and draw it back toward the
   robot: the hanging part keeps it taut and the edge combs it flat. Again until it lies flat. The hands keep hold
   (but if it lies out reaching too far from them, or a fold keeps standing, they take the near corners again,
   which the camera now shows sharply).
6. fold: carry the held edge up and away over the towel onto the far edge, and let go; draw back a fold that landed
   well past the far edge (where a held corner lands depends on how far in from it the hand holds it), press lumps
   down.

Anything that goes wrong in a way starting over fixes (a pinch closing on nothing, the towel the wrong size) lets
go of the towel and starts again from 1.
"""
import traceback
import numpy as np
import robot
from robot import when
from towel_lib import (r, towel_points, park, forward, pitched, down_any, reach, hanging_bottom, rectangle, holding,
                       lift_off, shake, grab_corner, corner_pinch, pinch, dev_peek)

EDGE_X = 0.52                  # the table's front edge (told by the task)
W, L = 0.30, 0.22              # the towel (told by the task)
DIAG = float(np.hypot(W, L))
BOTH = ("left", "right")


class StartOver(Exception):
    pass


class NotACorner(Exception):
    """Hanging from the held point, the towel hangs too short for that point to be a corner."""


def other(arm):
    return "right" if arm == "left" else "left"


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def on_table(pts, below=0.03):
    return pts[(pts[:, 2] < below) & (pts[:, 2] > -0.01)]          # not what hangs down past the table's edge


def let_go(arms=BOTH):
    """Open, rise slowly out of the cloth, park."""
    for a in arms:
        r.gripper(a, 0.08)
    r.step(10)
    for a in arms:
        try:
            lift_off(a, 0.08)
        except robot.InvalidCommand:
            pass
    park(arms)


# ------------------------------------------------------------------------------------------------ 1. hang
def hang():
    """Pinch the top of the pile and lift it high (jaws forward: straight down reaches only ~0.15 m above the table).
    Returns the arm holding it."""
    for attempt in range(3):
        pts = towel_points()
        if not len(pts):
            raise StartOver("the towel is not in view")
        top = pts[pts[:, 2] >= np.percentile(pts[:, 2], 97)]
        p, z_top = top[:, :2].mean(0), float(np.percentile(pts[:, 2], 99))
        p = p + 0.02 * attempt * unit(pts[:, :2].mean(0) - p + 1e-9)     # retries a little toward the pile's middle
        arm = "left" if p[1] >= 0 else "right"
        print(f"pile top at {np.round(p, 3)}, z {z_top:.3f} -> {arm} arm")
        try:
            pinch(arm, p, z_top, depth=0.015)
        except RuntimeError as e:
            print(f"  {e}")
            park([arm])
            continue
        lift_off(arm, 0.08)
        r.move(forward([0.33, 0.08 if arm == "left" else -0.08, 0.45]), ee=arm, speed=0.6)
        r.step(25)                                     # 1 s for the towel to stop swinging
        if holding(arm):
            return arm
        print(f"  lost the towel during the lift; trying again")
        let_go([arm])
    raise StartOver("could not pick the towel up")


# ------------------------------------------------------------------------------------------------ 2, 3. corners
def pick_lowest(holder, min_hang=0.0):
    """The towel hangs from `holder`. Lay it down sideways: the lowest point touches first and stays, and the bottom
    of the towel lies out in a strip with that point at its tip, clear of the holding arm and in view of the top
    camera. The other arm pinches the tip. Returns that arm."""
    free = other(holder)
    side = 1 if free == "left" else -1                 # the strip goes toward the free arm
    y = 0.06 * side
    for attempt in range(3):                           # a missed pinch: lift the towel again and lay it down again
        park([free])
        r.move(forward([0.30, y, 0.45]), ee=holder, speed=0.5)
        shake([holder])
        r.step(25)
        # how far it hangs, seen from the side by the free arm's wrist camera (from above it hides its own bottom)
        bottom = hanging_bottom(free, 0.30, y)
        hang = float(np.clip(0.45 - bottom, 0.15, 0.40)) if bottom is not None else 0.33
        print(f"  the towel hangs {hang:.2f} m below the hand")
        if hang < min_hang:
            raise NotACorner(f"the towel hangs only {hang:.2f} m from the held point")
        park([free])
        # down until the lowest point touches, then down and sideways about 1.5x as far, so the towel unrolls flat
        # onto the table instead of folding under itself
        x = 0.36
        path = [forward([x, y, hang + 0.01])] + [forward([x, y - 0.055 * k * side, hang + 0.01 - 0.035 * k])
                                                  for k in range(1, 5)]
        r.move(path, times=[1.5, 2.5, 3.5, 4.5, 5.5], ee=holder)
        r.step(25)
        flat = on_table(towel_points(max_z=0.6))
        if len(flat) < 30:
            print(f"  only {len(flat)} towel points lie on the table after the lay-down")
            continue
        hand = r.T(holder)[:2, 3]
        d = np.linalg.norm(flat[:, :2] - hand, axis=1)
        tip = flat[d > d.max() - 0.01, :2].mean(0)      # the far end of the strip: its last centimetre
        n = unit(tip - flat[:, :2].mean(0))
        print(f"  {len(flat)} points on the table; the lowest point (strip tip) at {np.round(tip, 3)}")
        try:
            grab_corner(free, tip, n)
            return free
        except RuntimeError as e:
            print(f"  {e}")
            r.gripper(free, 0.08)
    raise StartOver("could not pick the towel's lowest point")


def corner(holder):
    """Hold the towel by a corner: returns the arm that holds it."""
    a_hand = pick_lowest(holder)
    r.gripper(holder, 0.08)                             # the first arm lets go and backs off up and away
    T = r.T(holder)
    T[0, 3] -= 0.08
    T[2, 3] += 0.05
    r.move(T, ee=holder, line=True)
    print(f"the {a_hand} arm holds a corner")
    return a_hand


def opposite(a_hand):
    """Holding corner A, also hold the opposite corner C with the other arm. Returns the C arm. Raises NotACorner
    when the towel hangs too short from A (from a corner it hangs about its 0.37 m diagonal)."""
    lift_off(a_hand)
    c_hand = pick_lowest(a_hand, min_hang=0.30)
    print(f"the {c_hand} arm holds the opposite corner")
    return c_hand


# ------------------------------------------------------------------------------------------------ the ledge
def hold_apart(spread, x=0.32, z=0.38):
    """Both hands up with the jaws forward, `spread` apart across the table (pulling the cloth harder than that
    stretches it: it is soft)."""
    for a in BOTH:
        lift_off(a, 0.05)
    m = r.move(forward([x, spread / 2, z]), ee="left", speed=0.6, wait=False)
    r.move(forward([x, -spread / 2, z]), ee="right", speed=0.6)
    m.result()
    r.step(25)


# held by two points, the towel is shaken gently: a hard stroke throws the cloth up over the grippers, where it can
# stay draped instead of falling back to hang
GENTLE = dict(up=0.025, down=0.025, period=0.8)
DRAPE = dict(draw_s=10.0, x_end=0.22, relax=True, out_high=True)       # measured on the flat towel (scratch/fold_test.py, DRAPE=1)


def drape(label):
    """Both arms hold the ends of an edge. Lift it, hands an edge's length apart, shake it, swing it out over the
    front edge so it hangs straight down off the edge, then draw it back low over the table toward the robot, slowly:
    the hanging part keeps it taut and the edge combs it flat as it slides up onto the table. Ends with the held
    points set down on the table, still held."""
    X, H = 0.32, 0.38
    hold_apart(W - 0.01, X, H)
    Y = {a: float(r.T(a)[1, 3]) for a in BOTH}
    shake(BOTH, **GENTLE)
    r.step(40)                                         # 1.6 s to hang still
    dev_peek(f"{label}-shaken")
    both = list(BOTH)
    if DRAPE["out_high"]:
        # out past the edge at full height, wait until the towel hangs straight down there, then lower it: going out
        # and down at once, the towel trails behind the hands and lands on the table before the edge
        r.move({a: [forward([EDGE_X + 0.05, Y[a], H])] for a in BOTH}, times=[1.5], ee=both)
        r.step(40)
        r.move({a: [forward([EDGE_X + 0.05, Y[a], 0.05])] for a in BOTH}, times=[2.0], ee=both)
    else:
        r.move({a: [forward([X + 0.10, Y[a], H + 0.03]), forward([EDGE_X + 0.04, Y[a], 0.22]),
                      forward([EDGE_X + 0.05, Y[a], 0.05])] for a in BOTH}, times=[0.5, 1.4, 2.4], ee=both)
    r.step(25)
    dev_peek(f"{label}-over-edge")
    x_end, n, T = DRAPE["x_end"], 7, DRAPE["draw_s"]
    r.move({a: [pitched([EDGE_X + 0.05 - k * (EDGE_X + 0.05 - x_end) / n, Y[a], 0.035], pitch_deg=10 + 35 * k / n)
                  for k in range(1, n + 1)] for a in BOTH}, times=list(np.linspace(T / n, T, n)), ee=both)
    r.move({a: [pitched([x_end, Y[a], 0.015], pitch_deg=45)] for a in BOTH}, times=[0.8], ee=both)
    if DRAPE["relax"]:                                 # lift the held edge and lay it back: the towel, held taut by
        r.move({a: [pitched([x_end + 0.03, Y[a], 0.06], pitch_deg=45),        # the table, lets go of the strain
                      pitched([x_end, Y[a], 0.015], pitch_deg=45)] for a in BOTH}, times=[0.8, 1.8], ee=both)
    r.step(15)
    dev_peek(f"{label}-drawn-back")


def lying(margin=0.04):
    """The towel on the table as seen from above, without the part in and near the hands."""
    pts = towel_points()
    for a in BOTH:
        pts = pts[np.linalg.norm(pts[:, :2] - r.T(a)[:2, 3], axis=1) > margin]
    return pts


def area(pts, cell=0.005):
    """Area the points cover seen from above (m^2)."""
    return len(np.unique(np.floor(pts[:, :2] / cell).astype(int), axis=0)) * cell * cell if len(pts) else 0.0


# ------------------------------------------------------------------------------------------------ 4. adjacent
def find_apex(PA, PC):
    """The doubled triangle lying in front of the diagonal PA-PC (the held corners): one apex, how far it lies out
    from the diagonal and how far along it (0 at A, 1 at C), and the apex's outward direction (its bisector)."""
    pts = on_table(lying())
    if len(pts) < 50:
        raise StartOver("the towel is not on the table after laying the diagonal down")
    u = unit(PC - PA)
    v = np.array([-u[1], u[0]])
    v = v if v[0] > 0 else -v                          # away from the robot: where the flaps lie
    uv = (pts[:, :2] - PA) @ np.stack([u, v], 1)
    # one apex: the farthest point out and those near it (the two apexes stand out about equally far)
    far = pts[np.argmax(uv[:, 1]), :2]
    tip = (uv[:, 1] > uv[:, 1].max() - 0.01) & (np.linalg.norm(pts[:, :2] - far, axis=1) < 0.03)
    apex = pts[tip, :2].mean(0)
    along, height = uv[tip, 0].mean() / np.linalg.norm(PC - PA), uv[:, 1].max()
    print(f"  apex at {np.round(apex, 3)}: {height:.2f} m out from the diagonal, {along:.2f} of the way to C")
    return apex, along, height, unit(-(unit(PA - apex) + unit(PC - apex)))


def adjacent(a_hand, c_hand):
    """Holding a diagonal (A, C), end up holding the ends of a long edge."""
    for attempt in range(2):
        hold_apart(DIAG - 0.02 - 0.03 * attempt, x=0.40, z=0.30)   # again a little slacker, so it falls otherwise
        shake(BOTH, **GENTLE)
        r.step(25)
        dev_peek("diagonal-hanging")
        Y = {a: float(r.T(a)[1, 3]) for a in BOTH}
        n = 6
        r.move({a: [forward([0.40, Y[a], 0.20])] + [pitched([0.40 - 0.17 * k / n, Y[a], 0.20 - 0.17 * k / n],
                                                              pitch_deg=10 + 35 * k / n) for k in range(1, n + 1)]
                  for a in BOTH}, times=list(np.linspace(1.0, 5.0, n + 1)), ee=list(BOTH))
        r.step(25)
        dev_peek("diagonal-laid")
        PA, PC = r.T(a_hand)[:2, 3], r.T(c_hand)[:2, 3]
        apex, along, height, n = find_apex(PA, PC)
        # a doubled triangle: the apexes ~0.18 m out (less with a corner tucked under), at 0.35 and 0.65 of the way;
        # an apex in the middle means no clean fold along the diagonal
        if 0.11 < height < 0.24 and abs(along - 0.5) > 0.1 and apex[0] < EDGE_X - 0.02:
            break
        print("  the flaps did not lie out as a doubled triangle; laying it down again")
    else:
        raise StartOver("the diagonal's flaps would not lie flat")
    # the hand nearer the apex lets go and pinches the apex (a right-angled corner: jaws across its bisector);
    # letting go can tug the cloth, so look again before each try
    swap = a_hand if along < 0.5 else c_hand
    r.gripper(swap, 0.08)
    r.step(10)
    lift_off(swap, 0.10)
    for attempt in range(2):
        apex, _, _, n = find_apex(PA, PC)
        try:
            grab_corner(swap, apex, n, inside=0.01)    # a clean corner seen sharply: no need to reach far past it
            print(f"  the {swap} arm now holds the apex: the arms hold the ends of a long edge")
            return
        except RuntimeError as e:
            print(f"  {e}")
    raise StartOver("could not pinch the apex")


# ------------------------------------------------------------------------------------------------ 5. flatten
def lay_flat(label):
    """Both arms hold the ends of an edge. Lift it, hands an edge's length apart, shake it so it hangs straight, then
    lay it down while drawing it back toward the robot: the bottom edge touches first and stays, and the towel lies
    down flat in front of the hands. Ends with the held corners set down on the table, still held."""
    hold_apart(W - 0.01, x=0.44, z=0.30)
    shake(BOTH, **GENTLE)
    r.step(25)
    Y = {a: float(r.T(a)[1, 3]) for a in BOTH}
    n = 6
    r.move({a: [forward([0.44, Y[a], 0.25])] + [pitched([0.44 - 0.22 * k / n, Y[a], 0.25 - 0.215 * k / n],
                                                          pitch_deg=10 + 35 * k / n) for k in range(1, n + 1)]
              for a in BOTH}, times=list(np.linspace(1.0, 5.5, n + 1)), ee=list(BOTH))
    r.move({a: [pitched([0.22, Y[a], 0.015], pitch_deg=45)] for a in BOTH}, times=[0.8], ee=list(BOTH))
    r.step(15)
    dev_peek(f"{label}-laid")


def regrip_near_corners(inside=0.01):
    """Let go of the towel lying on the table and pinch the two corners of its edge nearest the robot (on a towel
    lying out, the camera shows its corners sharply), then hold them as the flatten step does."""
    for a in BOTH:
        r.gripper(a, 0.08)
    r.step(10)
    for a in BOTH:
        lift_off(a, 0.06)
    park()
    pts = towel_points()
    pts = pts[pts[:, 2] > -0.01]
    c, ax, bx, la, lb = rectangle(pts)
    corners = [c + sa * ax * la / 2 + sb * bx * lb / 2 for sa in (1, -1) for sb in (1, -1)]
    near = sorted(corners, key=lambda p: p[0])[:2]     # nearest the robot
    xy = pts[:, :2]
    for arm, p in zip(("left", "right"), sorted(near, key=lambda p: -p[1])):
        d = unit(p - c)
        q = xy[(xy - c) @ d > ((xy - c) @ d).max() - 0.006].mean(0)     # the towel's own corner there
        print(f"  {arm}: taking the corner at {np.round(q, 3)} again")
        if not corner_pinch(arm, q, d, inside=inside):
            raise StartOver(f"{arm} missed the corner at {np.round(q, 3)}")
        lift_off(arm, 0.02)
    for a in BOTH:
        p = r.T(a)[:3, 3]
        r.move(pitched([p[0], p[1], 0.015], pitch_deg=45), ee=a)


def flatten():
    """Until the towel lies flat (about its full size, nothing standing up): drape it over the front edge and draw it
    back (the ledge), or, when that leaves it caught or wrinkled, lay it down from hanging straight."""
    regripped = False
    for k, how in enumerate(("ledge",) * 4):            # the lay-down made things worse on the (stiff) towel
        (drape if how == "ledge" else lay_flat)(f"flatten-{k + 1}-{how}")
        pts = lying()
        over = int((pts[:, 2] < -0.01).sum())          # hanging down past the table's edge
        pts = pts[pts[:, 2] > -0.01]
        cover, high = area(pts) / (W * L), float(np.percentile(pts[:, 2], 98)) if len(pts) else 1.0
        # how far it reaches from the held edge: ~L when it lies out flat (less: part folded under; more: it hangs
        # over the table's edge, or the hands hold a short edge)
        P = {a: r.T(a)[:2, 3] for a in BOTH}
        u = unit(P["left"] - P["right"])
        v = np.array([-u[1], u[0]])
        v = v if v[0] > 0 else -v
        depth = float(np.percentile((pts[:, :2] - 0.5 * (P["left"] + P["right"])) @ v, 99)) if len(pts) else 0.0
        print(f"  {how}: the towel covers {cover:.2f} of its area, reaches {depth:.2f} m out, 98% of it below "
              f"{high:.3f} m{f', {over} points hang over the edge' if over else ''}")
        if over > 50:
            continue
        # one layer lies ~6 mm high; a small lump is fine (pressed down after the fold); the hands hide a little of
        # it from the camera
        if cover > 0.75 and high < 0.04 and L - 0.03 < depth < L + 0.04:
            return
        if cover > 0.6 and depth > L + 0.04:           # lies out, but reaches too far: held some way in from its
            regrip_near_corners()                      # corners; take the corners themselves, then again
            regripped = True
        elif cover > 0.55 and not regripped and k >= 1:   # mostly out but a fold stays standing pass after pass:
            regrip_near_corners()                      # fresh grips on the corners themselves, then again
            regripped = True
        if cover < 0.5:                                # held by the ends of an edge, it never lies this small
            raise StartOver(f"the towel covers only {cover:.2f} of its area: the hands do not hold an edge")
    raise StartOver("the towel would not lie flat")


# ------------------------------------------------------------------------------------------------ 6. fold
# measured on the flat-towel fold test (scratch/fold_test.py, after a ledge pass): a half circle 2 cm higher than
# the towel's half depth, the corners onto the far corners; a corner landing well past the far edge (held corners land
# 2-7 cm past the jaws, depending on the grip) is moved back before letting go (after letting go, both layers move
# together); the open jaws rise straight up (sliding them out dragged a deep-held corner several cm further);
# lumps pressed down
FOLD = dict(radius_extra=0.02, land=0.0, pull=0.0, pitch_end=80.0, slide_out=0.0, pat=True, pat_force=5.0,
            pat_above=0.03, tidy=False, tidy_above=0.03, adjust=True, adjust_above=0.025, arc_s=5.0, mirror_grip=False)


def fold():
    """Carry the held near edge up and over on a half circle, each held corner onto the far corner on its side, and
    let go."""
    P = {a: r.T(a)[:3, 3].copy() for a in BOTH}
    u = unit(P["left"][:2] - P["right"][:2])           # toward the left hand
    v = np.array([-u[1], u[0]])
    v = v if v[0] > 0 else -v                          # away from the robot
    mid = 0.5 * (P["left"][:2] + P["right"][:2])
    pts = lying()
    uv = (pts[:, :2] - mid) @ np.stack([u, v], 1)
    depth = float(np.percentile(uv[:, 1], 99))         # held edge to far edge
    band = pts[uv[:, 1] > depth - 0.03, :2]            # along the far edge
    bu = (band - mid) @ u
    goal = {"left": band[bu > bu.max() - 0.01].mean(0), "right": band[bu < bu.min() + 0.01].mean(0)}
    far = dict(goal)                                   # the far corners themselves
    if FOLD["mirror_grip"]:
        # folding is a mirror: a point held a along the edge and b in from the near corner lands a along and b in from
        # the far corner. The hands rarely hold the corners exactly; the camera shows the near corners beside them
        allpts = towel_points()
        for a, side in (("left", 1.0), ("right", -1.0)):
            loc = allpts[np.linalg.norm(allpts[:, :2] - P[a][:2], axis=1) < 0.08]
            if len(loc) < 20:
                continue
            dn = unit(-v + side * u)                                       # toward the near corner on this side
            N = loc[(loc[:, :2] @ dn) > (loc[:, :2] @ dn).max() - 0.006, :2].mean(0)
            off = P[a][:2] - N
            along, inward = float(np.clip(off @ u, -0.06, 0.06)), float(np.clip(off @ v, -0.06, 0.06))
            print(f"  {a} hand holds the towel {along * 100:+.1f} cm along and {inward * 100:+.1f} cm in from its corner")
            goal[a] = goal[a] + along * u - inward * v
    goal = {a: g + FOLD["land"] * v for a, g in goal.items()}  # the corner sits a few mm out of the jaws
    print(f"folding the near edge over by {depth:.2f} m: corners onto " +
          ", ".join(f"{a} {np.round(goal[a], 3)}" for a in BOTH))
    th = np.linspace(0, np.pi, 9)[1:]
    h = min(0.12, depth / 2 + FOLD["radius_extra"])
    end = FOLD["pitch_end"]
    r.move({a: [pitched(np.r_[P[a][:2] + (goal[a] - P[a][:2]) * (1 - np.cos(t)) / 2, 0.035 + h * np.sin(t)],
                          pitch_deg=45 + (end - 45) * t / np.pi) for t in th] for a in BOTH},
             times=list(np.linspace(FOLD["arc_s"] / len(th), FOLD["arc_s"], len(th))), ee=list(BOTH))
    if FOLD["pull"]:                                   # draw the top layer straight before letting go
        r.move({a: [pitched(np.r_[goal[a] + FOLD["pull"] * v, 0.035], pitch_deg=end)] for a in BOTH}, times=[0.8],
                 ee=list(BOTH))
    if FOLD["adjust"]:
        # still holding: where each held corner landed past the far edge (it lands 2-7 cm beyond the jaws, depending
        # on how far in from it the hand holds it), seen past the jaws by the top camera; move that hand back by it
        pts = towel_points()
        back = {}
        for a in BOTH:
            near = pts[np.abs((pts[:, :2] - far[a]) @ u) < 0.06]
            over = float(((near[:, :2] - far[a]) @ v).max()) if len(near) else 0.0
            back[a] = over if over > FOLD["adjust_above"] else 0.0     # small readings are mostly the camera's error
            print(f"  {a} corner landed {over * 100:.1f} cm past the far edge" + (": moving it back" if back[a] else ""))
        if max(back.values()) > 0.005:
            r.move({a: [pitched(np.r_[r.T(a)[:2, 3] - back[a] * v, 0.035], pitch_deg=end)] for a in BOTH},
                     times=[1.2], ee=list(BOTH))
    for a in BOTH:
        r.gripper(a, 0.09)                             # wide open: a wad of cloth between the pads must not catch
    r.step(15)
    for a in BOTH:                                     # and out slowly at first: rising fast lifted the top layer
        T = r.T(a)
        T[2, 3] += 0.03
        r.move(T, ee=a, line=True, speed=0.015)
    if FOLD["slide_out"]:                              # the open jaws slide out past the corner, then rise
        r.move({a: [pitched(np.r_[r.T(a)[:2, 3] + FOLD["slide_out"] * v, 0.035], pitch_deg=end)] for a in BOTH},
                 times=[1.0], ee=list(BOTH))
    for a in BOTH:
        T = r.T(a)
        T[2, 3] += 0.08
        r.move(T, ee=a, line=True, speed=0.05)
    if FOLD["tidy"]:
        park()
        tidy_overshoot(goal, u, v)
    if FOLD["pat"]:                                    # press the folded towel down wherever it stands up
        park()
        for k in range(4):
            pts = towel_points()
            if not len(pts) or pts[:, 2].max() < FOLD["pat_above"]:
                break
            p = pts[np.argmax(pts[:, 2]), :2]
            arm = "left" if p[1] >= 0 else "right"
            print(f"  pressing down a lump {pts[:, 2].max():.3f} m high at {np.round(p, 3)}")
            r.gripper(arm, 0.0)
            try:                                       # a lump out of reach is left as it is: never undo the fold
                reach(arm, down_any(p, 0.08))
            except robot.InvalidCommand:
                print("  out of reach; leaving it")
                park([arm])
                break
            r.move_at({"z": -0.02}, ee=arm, max_travel=0.08, until={"press": when.force(arm, "-z") > FOLD["pat_force"]})
            r.move([robot.pose([0, 0, 0.05])], times=[0.6], ee=arm, relative=True)
            park([arm])
    park()


def tidy_overshoot(far, u, v):
    """Where the folded edge landed past the far edge (`far`: the far corners before folding), the part sticking out
    is the top layer alone: pinch its corner from outside and drag it back. How far past the jaws a held corner lands
    depends on how far in from the corner the hand held it, so the fold is checked, not trusted."""
    pts = towel_points()
    for arm, side in (("left", 1.0), ("right", -1.0)):
        near = pts[np.abs((pts[:, :2] - far[arm]) @ u) < 0.06]          # this end of the folded towel
        if not len(near):
            continue
        over = float(((near[:, :2] - far[arm]) @ v).max())
        if over < FOLD["tidy_above"]:
            print(f"  {arm} end of the fold: {over * 100:.1f} cm past the far edge")
            continue
        d = unit(v + side * u)                                          # the flap's outer corner, pointing out
        q = near[np.argmax(near[:, :2] @ d), :2]
        print(f"  {arm} end of the fold landed {over * 100:.1f} cm past the far edge: drawing the flap back")
        n = v if q[0] + 0.035 * v[0] < EDGE_X - 0.01 else side * u       # from outside; from the side at the
        try:                                                             # table's edge (no table beyond it)
            if not corner_pinch(arm, q, n, inside=0.008):
                park([arm])
                continue
        except robot.InvalidCommand:
            park([arm])
            continue
        T = r.T(arm)
        T[:2, 3] -= (over - 0.005) * v
        r.move(T, ee=arm, line=True, speed=0.03)
        r.gripper(arm, 0.08)
        r.step(10)
        lift_off(arm, 0.06)
        park([arm])


def folded():
    """From above: about half the towel's area, a rectangle of the towel's length by half its width (or the other
    way), lying low."""
    pts = towel_points()
    if not len(pts):
        return False, "towel not in view"
    c, a, b, la, lb = rectangle(pts)
    cover, high = area(pts) / (W * L), float(pts[:, 2].max())
    # half as wide as the towel (a fold landing short by d leaves it d/2 wider: 1 cm wider is 2 cm short); the
    # crease stands a few cm high
    ok = 0.35 < cover < 0.55 and high < 0.045 and ((la < W + 0.04 and lb < L / 2 + 0.012) or
                                                    (la < L + 0.04 and lb < W / 2 + 0.012))
    return ok, f"{la:.2f} x {lb:.2f} m, {cover:.2f} of the flat area, highest {high:.3f} m"


# ------------------------------------------------------------------------------------------------ the task
def main():
    for attempt in range(1, 5):
        print(f"==== attempt {attempt}", flush=True)
        try:
            holder = hang()
            a_hand = corner(holder)
            for k in range(3):
                try:
                    c_hand = opposite(a_hand)
                    break
                except NotACorner as e:            # its lowest point is a corner: pick that, let this one go
                    print(f"  {e}: not a corner; picking its lowest point instead")
                    a_hand = corner(a_hand)
            else:
                raise StartOver("could not get hold of a corner")
            adjacent(a_hand, c_hand)
            flatten()
            fold()
            ok, how = folded()
            print(f"folded: {how}")
            if ok:
                break
            print("the fold did not come out right; starting over")
        except Exception as e:                             # whatever went wrong, starting over from the pile fixes it
            print(f"starting over: {type(e).__name__}: {e}")
            if not isinstance(e, (StartOver, robot.RobotError)):
                traceback.print_exc()
            if r.state().fault:
                r.recover()
            let_go()
    else:
        raise SystemExit("could not fold the towel in 4 attempts")
    print(r.finish(f"folded the towel in half ({how}): found a corner, the opposite corner and an adjacent corner by "
                   f"hanging and laying it down, combed it flat over the table edge, folded the held edge onto the far "
                   f"edge"))


if __name__ == "__main__":
    main()
