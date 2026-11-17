"""Tasks on the real YAM cell (scripts/agent_run_real.py): what the agent is told, and a check from the cameras, since
there is no simulator to ask. The launcher measures the scene before the agent starts (measure) and checks it after
the episode (check), both from outside the agent's sandbox and with the arms parked, so they do not hide the towel.
The operator's verdict, from the recording and the witness frames, is the one that counts; the check is a first read.

The towel tasks follow tasks/yam_station.py's thresholds on the area the towel covers seen from above (folded in half:
0.40-0.62 of its flat area; spread flat: over 0.9), measured here from the top camera's depth instead of the
simulator's cloth vertices. The towel is found by its colour's hue (--towel WxH:COLOUR; red, blue or green), only on
the table in the arms' work area (the cell has a blue background beyond the table), as the largest patch there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage

CELL = 0.005                    # m: the grid the covered area is counted on
TOWEL = dict(w_cm=30, h_cm=22, colour="red")
HUES = {"red": (345, 15), "blue": (180, 250), "green": (80, 160)}    # degrees (red wraps around 0)
WORK = dict(x=(0.10, 0.62), y=(-0.47, 0.52), z=(-0.03, 0.15))      # m: the table in the arms' reach


def colour_mask(rgb, colour: str) -> np.ndarray:
    """Pixels of a saturated colour: its hue range (HUES), saturation over 0.3, not too dark."""
    x = rgb.astype(float) / 255
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-9
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == R, ((G - B) / d) % 6, np.where(mx == G, (B - R) / d + 2, (R - G) / d + 4)) * 60
    lo, hi = HUES[colour]
    hue = (h >= lo) | (h <= hi) if lo > hi else (h >= lo) & (h <= hi)
    return hue & (d / (mx + 1e-9) > 0.3) & (mx > 0.12)


def towel_view(r, colour: str = "red", camera: str = "top") -> dict:
    """The towel from above: the area its points cover on the table (m^2, on a CELL grid with small holes closed), its
    height above the table (95th percentile of those points, m), where it lies (xy of its centre, m) and how many of
    its colour's pixels the camera saw. Only points on the table in the arms' work area count (WORK), and of those the
    largest patch. The table height is the median of the other points within 5 cm of z = 0 around the towel."""
    f = r.cameras([camera])[camera]
    # drop the outline: at a silhouette edge colour and depth can come from different surfaces
    mask = ndimage.binary_erosion(colour_mask(f.rgb, colour), iterations=2)
    out = {"camera": camera, "colour": colour, "pixels": int(mask.sum()), "area_m2": 0.0, "height_m": None,
           "centre_xy": None}
    pts = f.points(mask)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] > WORK["x"][0]) & (pts[:, 0] < WORK["x"][1]) & (pts[:, 1] > WORK["y"][0])
              & (pts[:, 1] < WORK["y"][1]) & (pts[:, 2] > WORK["z"][0]) & (pts[:, 2] < WORK["z"][1])]
    if len(pts) < 50:
        return out
    ij = np.floor((pts[:, :2] - pts[:, :2].min(0)) / CELL).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, n = ndimage.label(ndimage.binary_closing(grid, iterations=2, border_value=0) | grid)
    keep = labels == 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))
    pts = pts[keep[ij[:, 0], ij[:, 1]]]
    near = f.points(~colour_mask(f.rgb, colour))
    near = near[np.isfinite(near).all(axis=1)]
    lo, hi = pts[:, :2].min(0) - 0.1, pts[:, :2].max(0) + 0.1
    near = near[(near[:, :2] >= lo).all(1) & (near[:, :2] <= hi).all(1) & (np.abs(near[:, 2]) < 0.05)]
    table_z = float(np.median(near[:, 2])) if len(near) > 100 else 0.0
    ij = np.floor((pts[:, :2] - pts[:, :2].min(0)) / CELL).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=2, border_value=0) | grid
    out.update(area_m2=round(float(grid.sum()) * CELL * CELL, 5),
               height_m=round(float(np.percentile(pts[:, 2], 95)) - table_z, 4),
               table_z=round(table_z, 4), centre_xy=np.round(pts[:, :2].mean(0), 3).tolist(), points=len(pts))
    return out


def _objects(f, z_min=0.03) -> np.ndarray:
    """A camera's points on the table's objects: in the work area (WORK, z up to 0.40 here) and z_min above the
    table."""
    pts = f.points(np.ones(f.rgb.shape[:2], bool))
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts[(pts[:, 0] > WORK["x"][0]) & (pts[:, 0] < WORK["x"][1]) & (pts[:, 1] > WORK["y"][0])
               & (pts[:, 1] < WORK["y"][1]) & (pts[:, 2] > z_min) & (pts[:, 2] < 0.40)]


def ball_cup_view(r, towel=None) -> dict:
    """The red ball, the bowl and the cup (dump_ball): where the cameras see the ball (each camera: its red pixels and
    the median of their points, on the table's work area), and the tallest object on the right half (y < 0.05) in
    the front camera's depth, the cup: its top's height and where it stands. The bowl is the left half's objects
    (y > 0.05) under 10 cm."""
    out = {"ball": {}, "cup": None, "bowl": None}
    frames = r.cameras(["top", "front"])
    for name, f in frames.items():
        mask = ndimage.binary_erosion(colour_mask(f.rgb, "red"), iterations=1)
        pts = f.points(mask)
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[(pts[:, 0] > WORK["x"][0]) & (pts[:, 0] < WORK["x"][1]) & (pts[:, 1] > WORK["y"][0])
                  & (pts[:, 1] < WORK["y"][1]) & (pts[:, 2] > -0.03) & (pts[:, 2] < 0.40)]
        out["ball"][name] = {"pixels": int(mask.sum()), "points": len(pts),
                             "xyz": np.round(np.median(pts, 0), 3).tolist() if len(pts) >= 20 else None}
    # the objects: clusters of the front camera's points above the table (2 cm cells), less the arms' (any cluster
    # within 8 cm of a TCP); the cup is the tallest over 10 cm, the bowl the largest under 10 cm
    obj = _objects(frames["front"], z_min=0.02)
    tcps = np.array([np.asarray(r.state().ee[a].T)[:3, 3] for a in ("left", "right")])
    clusters = []
    if len(obj):
        ij = np.floor((obj[:, :2] - obj[:, :2].min(0)) / 0.02).astype(int)
        grid = np.zeros(ij.max(0) + 1, bool)
        grid[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(grid)
        L = lab[ij[:, 0], ij[:, 1]]
        for k in range(1, n + 1):
            p = obj[L == k]
            if len(p) < 150 or np.min(np.linalg.norm(p[:, None, :] - tcps[None], axis=2)) < 0.08:
                continue
            top = float(np.percentile(p[:, 2], 98))
            clusters.append({"n": len(p), "top_z": round(top, 3),
                             "xy": np.round(np.median(p[p[:, 2] > top - 0.03][:, :2], 0), 3).tolist()})
    tall = [c for c in clusters if c["top_z"] > 0.10]
    low = [c for c in clusters if c["top_z"] <= 0.10]
    if tall:
        c = max(tall, key=lambda c: c["top_z"])
        out["cup"] = {"top_z": c["top_z"], "xy": c["xy"], "standing": True}
    if low:
        c = max(low, key=lambda c: c["n"])
        out["bowl"] = {"top_z": c["top_z"], "xy": c["xy"]}
    seen = [v["xyz"] for v in out["ball"].values() if v["xyz"] is not None]
    out["ball_xyz"] = seen[0] if seen else None
    return out


def _ball_check(before, after, towel):
    """The ball in the cup: neither camera sees it (the cup is opaque; inside it is out of their sight) and the cup
    still stands. A ball that left the table is not seen either: the operator confirms from the frames."""
    cup = after.get("cup") or {}
    hidden = after.get("ball_xyz") is None
    ok = hidden and bool(cup.get("standing"))
    return {"success": bool(ok), "ball_seen_at": after.get("ball_xyz"), "cup": cup or None,
            "rule": "the ball seen by neither camera, the cup's top over 10 cm (standing); the operator confirms the "
                    "ball is in the cup and not elsewhere"}


@dataclass
class RealTask:
    name: str
    text: Callable[[dict], str]                        # the instruction, from the towel's description (TOWEL)
    check: Callable[[dict, dict, dict], dict]          # check(before, after, towel) -> {"success", ...}
    notes: list = field(default_factory=list)
    view: Callable | None = None                       # measure(r, towel) -> dict; default: the towel from above

    def instruction(self, towel: dict | None = None) -> str:
        return self.text({**TOWEL, **(towel or {})})

    def measure(self, r, towel: dict) -> dict:
        """The scene, before the agent and after the episode (the launcher, with the arms parked)."""
        return self.view(r, towel) if self.view is not None else towel_view(r, towel["colour"])


def flat_area(towel: dict) -> float | None:
    return towel["w_cm"] * towel["h_cm"] * 1e-4 if towel.get("w_cm") and towel.get("h_cm") else None


def size(t: dict) -> str:
    """", about W x H cm," when the towel's size is known (the instruction says no size it does not know)."""
    return f", about {t['w_cm']:g} x {t['h_cm']:g} cm," if t.get("w_cm") and t.get("h_cm") else ""


def _unknown(after, why):
    return {"success": None, "area_m2": after["area_m2"], "height_m": after["height_m"],
            "note": f"no towel size given, so no {why} threshold: the operator judges from the witness frames"}


def _fold_check(before, after, towel):
    """Folded once in half: about half the flat area (as the simulator's check), lying on the table."""
    flat = flat_area(towel)
    if flat is None:
        return _unknown(after, "coverage")
    frac = after["area_m2"] / flat
    ok = 0.40 < frac < 0.62 and after["height_m"] is not None and after["height_m"] < 0.04
    return {"success": bool(ok), "coverage": round(frac, 3),
            "coverage_before": round(before["area_m2"] / flat, 3) if before else None,
            "height_m": after["height_m"], "flat_area_m2": flat,
            "rule": "covers 0.40-0.62 of its flat area, under 4 cm high (tasks/yam_station.py _fold_towel_check)"}


def _flat_check(before, after, towel):
    """Spread out flat: over 0.9 of the flat area, lying on the table."""
    flat = flat_area(towel)
    if flat is None:
        return _unknown(after, "coverage")
    frac = after["area_m2"] / flat
    ok = frac > 0.9 and after["height_m"] is not None and after["height_m"] < 0.03
    return {"success": bool(ok), "coverage": round(frac, 3),
            "coverage_before": round(before["area_m2"] / flat, 3) if before else None,
            "height_m": after["height_m"], "flat_area_m2": flat,
            "rule": "covers over 0.9 of its flat area, under 3 cm high (tasks/yam_station.py _flatten_check)"}


# The simulator's instructions (tasks/yam_station.py), less what is not true of the cell (its table does not end
# 0.52 m in front of the bases) and with the towel's colour and size as given (--towel).
TASKS = {
    "fold_towel": RealTask(
        "fold_towel",
        lambda t: f"A {t['colour']} towel{size(t)} lies crumpled on the table between the two arms. Spread it out "
                  "flat, then fold it once in half, edge to edge. Leave it lying folded on the table, then call "
                  "r.finish() with a short summary.",
        _fold_check),
    "fold_flat_towel": RealTask(
        "fold_flat_towel",
        lambda t: f"A {t['colour']} towel{size(t)} lies flat on the table between the two arms. Fold it once in half, "
                  "edge to edge. Leave it lying folded on the table, then call r.finish() with a short summary.",
        _fold_check),
    "unfold_towel": RealTask(
        "unfold_towel",
        lambda t: f"A {t['colour']} towel{size(t)} lies on the table between the two arms, folded once in half. "
                  "Unfold it: spread it out flat on the table, then call r.finish() with a short summary.",
        _flat_check),
    "find_pear": RealTask(                             # the cup demo (cups-02 in the blog); no camera check
        "find_pear",
        lambda t: "There are three red cups on the table in front of you. Two of them have something under them: one "
                  "has a pear and one has a ball. The third has nothing under it. Find the cup that has the pear, and "
                  "put the pear under the cup that has nothing in it. Then call r.finish() with a short summary.",
        lambda before, after, towel: {"success": None, "note": "no camera check: the operator judges from the "
                                                               "witness frames"},
        view=lambda r, towel: {}),
    "dump_ball": RealTask(
        "dump_ball",
        lambda t: "A red ball lies in a black bowl on the left side of the table, and a tall dark cup stands on the "
                  "right side. Dump the red ball into the cup on the right: it should end inside the cup, with the "
                  "cup still standing. Then call r.finish() with a short summary.",
        _ball_check, view=ball_cup_view),
}
