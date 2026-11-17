"""The gripper's lowest point, for the fence's gripper_floor (robotd/core/controller.py, parse_fence).

The fence's box bounds the TCP, a point between the fingertips. A tilted gripper's fingers, body and wrist camera reach
below it, and meet the table while the TCP is still above the box's floor (on the real YAM: a gripper tilted 20 deg,
sent to the floor at double stiffness, pressed into the table and chattered). gripper_floor bounds the gripper itself.

Its geometry: the convex hulls of every geom below each arm's last arm joint (the gripper, its fingers, the wrist
camera), in the TCP site's frame, with the fingers at both ends of their travel. A finger slides on a line, so its
lowest point over any opening is at one of the two, and the hull of both is exact for the lowest point at any opening.
Then the lowest point at a TCP pose T is one dot product per hull point (a few hundred).
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def _below(model, body: int, top: int) -> bool:
    """body is top or hangs from it."""
    while body not in (0, top):
        body = int(model.body_parentid[body])
    return body == top


def _geom_points(model, g: int) -> np.ndarray:
    """A geom's extreme points in its own frame: a mesh's convex hull, a primitive's bounding box corners."""
    t, s = int(model.geom_type[g]), model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        V = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]],
                       float)
        return V[ConvexHull(V).vertices] if len(V) > 4 else V
    half = {mujoco.mjtGeom.mjGEOM_BOX: s[:3],
            mujoco.mjtGeom.mjGEOM_SPHERE: [s[0]] * 3,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID: s[:3],
            mujoco.mjtGeom.mjGEOM_CYLINDER: [s[0], s[0], s[1]],
            mujoco.mjtGeom.mjGEOM_CAPSULE: [s[0], s[0], s[1] + s[0]]}.get(t)
    if half is None:                                    # a plane or a height field: not part of a gripper
        return np.zeros((0, 3))
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half, float)


class GripperFloor:
    def __init__(self, model, arm_joints: dict, tcp_sites: dict):
        d = mujoco.MjData(model)
        self.pts = {}
        for arm, js in arm_joints.items():
            top = int(model.jnt_bodyid[model.joint(js[-1]).id])
            geoms = [g for g in range(model.ngeom) if _below(model, int(model.geom_bodyid[g]), top)]
            slides = [j for j in range(model.njnt) if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_SLIDE
                      and _below(model, int(model.jnt_bodyid[j]), top)]
            sid = model.site(tcp_sites[arm]).id
            local = {g: _geom_points(model, g) for g in geoms}
            pts = []
            for end in (0, 1):
                d.qpos[:] = model.qpos0
                for j in slides:
                    d.qpos[model.jnt_qposadr[j]] = model.jnt_range[j][end]
                mujoco.mj_kinematics(model, d)
                R, p = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
                for g in geoms:
                    W = d.geom_xpos[g] + local[g] @ d.geom_xmat[g].reshape(3, 3).T
                    pts.append((W - p) @ R)                       # in the TCP site's frame
            P = np.concatenate(pts) if pts else np.zeros((1, 3))
            self.pts[arm] = P[ConvexHull(P).vertices] if len(P) > 4 else P

    def lowest(self, arm: str, T) -> float:
        """World z of the gripper's lowest point with its TCP site at pose T (4x4)."""
        T = np.asarray(T, float)
        return float(T[2, 3] + (self.pts[arm] @ T[2, :3]).min())
