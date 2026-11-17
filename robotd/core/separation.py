"""Keep the two arms apart. Each arm is a few capsules around its links (segments between body origins, and from the
gripper body to the TCP) and spheres (a wrist camera); the arms' clearance is the smallest distance between one arm's
capsule surfaces and the other's. The controller (desc["separation"]) refuses a command whose goal brings the arms
closer than min_m, stops an arm whose reference would (as the fence stops one at its box), and latches a Fault when
the measured arms are closer than fault_m. Moving apart is always allowed.

The capsule around a gripper is 8 cm in radius (the jaws fully open), so two hands could never work on one small object.
With spec["fine"] (robotd --separation-fine) the capsules stay as a first pass: while they say the arms are clear,
nothing else runs. When they say otherwise, the hands' pairs are measured on the real meshes instead (convex hulls, with
the fingers where the jaw width puts them), against fine["min_m"] (5 mm, say); the arms' links keep their capsules.

The capsules are conservative (radii from the link meshes' 95th percentile to their maximum around each segment), as
collision spheres in MoveIt or cuRobo are: the real surfaces stay further apart than the clearance says. One
kinematics pass and 36 segment pairs: about 30 us a tick.
"""
from __future__ import annotations

import mujoco
import numpy as np

# The YAM station (robotd/robots/yam_station), measured on the cell's model 2026-11-04 (mesh distance to each segment:
# p95 / max): upper arm 5.4 / 7.5 cm, forearm 7.9 / 10.9 (the elbow housing), wrist 6.9 / 7.4 and 6.3 / 6.4, gripper
# to TCP 5.8 cm (7.7 with the jaws open); the wrist camera's mesh within 5.6 cm of its geom's centre.
YAM = {"capsules": {a: [[f"{a}_link2", f"{a}_link3", 0.065], [f"{a}_link3", f"{a}_link4", 0.085],
                        [f"{a}_link4", f"{a}_link5", 0.075], [f"{a}_link5", f"{a}_gripper", 0.065],
                        [f"{a}_gripper", f"site:tcp_{a}", 0.08]] for a in ("left", "right")},
       "spheres": {a: [[f"{a}_camera", 0.06]] for a in ("left", "right")}}
YAM_HAND = {a: [f"{a}_gripper", f"{a}_tip_left", f"{a}_tip_right", f"{a}_camera"] for a in ("left", "right")}
YAM_NEAR = {a: [f"{a}_link3", f"{a}_link4", f"{a}_link5"] for a in ("left", "right")}


def yam_fine(min_m: float) -> dict:
    """spec["fine"] for the YAM station. fingers: each jaw's joint and its mirror (the kinematics do not apply the
    coupling)."""
    return {"min_m": float(min_m), "hand": YAM_HAND, "near": YAM_NEAR,
            "fingers": {a: [f"{a}_joint7", f"{a}_joint8"] for a in ("left", "right")}}


def seg_seg(p1, q1, p2, q2) -> np.ndarray:
    """Distance between segments p1-q1 and p2-q2 (broadcast over leading axes; a segment may be a point). Ericson,
    Real-Time Collision Detection, 5.1.9."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    eps = 1e-12
    pa, pe = a <= eps, e <= eps
    safe = lambda x: np.where(np.abs(x) > eps, x, 1.0)
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / safe(denom), 0.0, 1.0), 0.0)
    t = (b * s + f) / safe(e)
    s = np.where(t < 0, np.clip(-c / safe(a), 0.0, 1.0), np.where(t > 1, np.clip((b - c) / safe(a), 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    # degenerate segments: a point against a segment, or two points
    s = np.where(pa, 0.0, np.where(pe, np.clip(-c / safe(a), 0.0, 1.0), s))
    t = np.where(pa & ~pe, np.clip(f / safe(e), 0.0, 1.0), np.where(pe, 0.0, t))
    c1, c2 = p1 + d1 * s[..., None], p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


class Separation:
    """The two arms' clearance on a MuJoCo model, from spec ({"capsules": {arm: [[A, B, r], ...]}, "spheres": {arm:
    [[BODY, r], ...]}}; A and B are body names or "site:NAME"; a sphere sits at the centre of the body's first geom)."""

    def __init__(self, model: mujoco.MjModel, spec: dict, min_m: float, fault_m: float = 0.0):
        self.m, self.min_m, self.fault_m = model, float(min_m), float(fault_m)
        self.arms = list(spec["capsules"])
        if len(self.arms) != 2:
            raise ValueError(f"separation needs two arms, got {self.arms}")
        ref = lambda name: (("site", model.site(name[5:]).id) if name.startswith("site:")
                            else ("body", model.body(name).id))
        self.caps = {a: [(ref(x), ref(y), float(rad)) for x, y, rad in spec["capsules"][a]] for a in self.arms}
        self.spheres = {a: [(int(np.flatnonzero(model.geom_bodyid == model.body(b).id)[0]), float(rad))
                            for b, rad in (spec.get("spheres") or {}).get(a, [])] for a in self.arms}
        self.names = {a: [f"{x}-{y}" for x, y, _ in spec["capsules"][a]] + [b for b, _ in
                                                                            (spec.get("spheres") or {}).get(a, [])]
                      for a in self.arms}
        self.fine = spec.get("fine")
        if self.fine:
            mesh = lambda b: [g for g in np.flatnonzero(model.geom_bodyid == model.body(b).id)
                              if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            hand = {a: [g for b in self.fine["hand"][a] for g in mesh(b)] for a in self.arms}
            near = {a: [g for b in self.fine["near"][a] for g in mesh(b)] for a in self.arms}
            a, b = self.arms
            self.fine_pairs = [(int(x), int(y)) for x in hand[a] for y in hand[b] + near[b]] + \
                              [(int(x), int(y)) for x in near[a] for y in hand[b]]
            self.fine_shift = self.min_m - float(self.fine["min_m"])
            adr = lambda n: int(model.jnt_qposadr[model.joint(n).id])
            self.mimic = [(adr(x), adr(y)) for x, y in self.fine["fingers"].values()]
            self.hand_shape = {a: np.array([y.startswith("site:") for _, y, _ in spec["capsules"][a]] +
                                           [True] * len((spec.get("spheres") or {}).get(a, []))) for a in self.arms}

    def _shapes(self, d, arm):
        pt = lambda r: d.site_xpos[r[1]] if r[0] == "site" else d.xpos[r[1]]
        P = [pt(x) for x, _, _ in self.caps[arm]] + [d.geom_xpos[g] for g, _ in self.spheres[arm]]
        Q = [pt(y) for _, y, _ in self.caps[arm]] + [d.geom_xpos[g] for g, _ in self.spheres[arm]]
        R = [r for _, _, r in self.caps[arm]] + [r for _, r in self.spheres[arm]]
        return np.array(P), np.array(Q), np.array(R)

    def clearance(self, d) -> tuple:
        """(clearance in m, (left shape, right shape)) for the positions in d (kinematics already run)."""
        (P1, Q1, R1), (P2, Q2, R2) = (self._shapes(d, a) for a in self.arms)
        dist = seg_seg(P1[:, None], Q1[:, None], P2[None], Q2[None]) - R1[:, None] - R2[None]
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        first = float(dist[i, j]), (self.names[self.arms[0]][i], self.names[self.arms[1]][j])
        if not self.fine or first[0] >= self.min_m:
            return first
        a, b = self.arms                                # the hands' pairs on their meshes, the links' on capsules
        rest = np.where(self.hand_shape[a][:, None] | self.hand_shape[b][None], np.inf, dist)
        i, j = np.unravel_index(int(np.argmin(rest)), rest.shape)
        best = float(rest[i, j]), (self.names[a][i], self.names[b][j])
        fromto = np.zeros(6)
        for x, y in self.fine_pairs:
            gap = mujoco.mj_geomDistance(self.m, d, x, y, 0.2, fromto) + self.fine_shift
            if gap < best[0]:
                best = gap, (self.m.body(self.m.geom_bodyid[x]).name, self.m.body(self.m.geom_bodyid[y]).name)
        return best

    def at(self, kin, q: dict) -> tuple:
        """The clearance with the arms at joint values q ({name: rad}, every arm joint), on kin's own data."""
        kin.set_q(q)
        if self.fine:
            for x, y in self.mimic:
                kin.d.qpos[y] = kin.d.qpos[x]
        mujoco.mj_kinematics(kin.m, kin.d)
        return self.clearance(kin.d)
