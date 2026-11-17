"""The robot as the dashboard draws it: the station model's meshes once, then their poses for any joint vector.

The model is the CAD station (the description's MJCF) in the public world frame. On a real cell the driver's own
kinematics also carry the calibration's base corrections, so the drawn arms can sit a few millimetres off the point
cloud.
"""
from __future__ import annotations

import mujoco
import numpy as np

from robotd.robots.yam_station import scene as S

TABLE = {"size": [0.55, 0.7, 0.025], "pos": [0.35, 0.0, -0.025]}     # as in the simulation scene
DIAGRAM_BODIES = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]


class DisplayModel:
    def __init__(self, mjcf: str, joints: list[str], arms: list[str], tcp_sites: dict):
        spec = mujoco.MjSpec.from_file(mjcf)
        S._shift_world_children(spec, S.STATION_TO_WORLD_Y)
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        self.joints, self.arms = list(joints), list(arms)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.joints])
        self.limits = [[float(v) for v in m.jnt_range[m.joint(n).id]] for n in self.joints]
        driven = set(int(a) for a in self.qadr)
        self.couplings = []                      # joint equalities: the second finger follows the first
        for i in range(m.neq):
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                dep, ind = int(m.jnt_qposadr[m.eq_obj1id[i]]), int(m.jnt_qposadr[m.eq_obj2id[i]])
                c = m.eq_data[i][:5].copy()
                if ind in driven and dep not in driven:
                    self.couplings.append((dep, ind, c, False))
                elif dep in driven and ind not in driven and abs(c[1]) > 1e-9:
                    self.couplings.append((ind, dep, c, True))
        self.geoms = [g for g in range(m.ngeom) if m.geom_group[g] < 3]
        self.diagram_bodies = {a: [m.body(f"{a}_{b}").id for b in DIAGRAM_BODIES] for a in self.arms}
        self.tcp_sites = {a: m.site(tcp_sites[a]).id for a in self.arms}
        self.j1 = {a: self.joints.index(f"{a}_joint1") for a in self.arms}

    def payload(self) -> tuple[dict, bytes]:
        """The geometry, once: geoms (type, size, colour, mesh) and every mesh's vertices and faces in one blob."""
        m = self.m
        meshes, blob, index = [], bytearray(), {}
        geoms = []
        for g in self.geoms:
            mesh = int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1
            if mesh >= 0 and mesh not in index:
                v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
                vert = np.ascontiguousarray(m.mesh_vert[v0:v0 + nv], np.float32)
                face = np.ascontiguousarray(m.mesh_face[f0:f0 + nf], np.uint32)
                index[mesh] = len(meshes)
                meshes.append({"vert_offset": len(blob), "vert_count": int(nv)})
                blob += vert.tobytes()
                meshes[-1].update(face_offset=len(blob), face_count=int(nf))
                blob += face.tobytes()
            geoms.append({"type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                          "rgba": m.geom_rgba[g].tolist(), "mesh": index.get(mesh, -1),
                          "body": m.body(int(m.geom_bodyid[g])).name})
        return {"geoms": geoms, "meshes": meshes, "table": TABLE}, bytes(blob)

    def _set(self, q):
        d = self.d
        d.qpos[self.qadr] = np.asarray(q, float)
        for dst, src, c, invert in self.couplings:
            x = d.qpos[src]
            d.qpos[dst] = (x - c[0]) / c[1] if invert else c[0] + c[1] * x + c[2] * x * x + c[3] * x ** 3 + c[4] * x ** 4
        mujoco.mj_kinematics(self.m, d)

    def poses(self, q) -> list:
        """Every drawn geom's pose at joints q: [x, y, z, r00 .. r22] per geom, flattened."""
        self._set(q)
        d = self.d
        P = np.concatenate([d.geom_xpos[self.geoms], d.geom_xmat[self.geoms]], axis=1)
        return np.round(P, 5).ravel().tolist()

    def diagram(self, q) -> dict:
        """Each arm seen from the side, in the vertical plane joint 1 turns it into: [[h, z], ...] in metres from the
        base for the base, links 1-5, the gripper and the TCP (joint k sits at the k-th point)."""
        self._set(q)
        d = self.d
        out = {}
        for a in self.arms:
            ids = self.diagram_bodies[a]
            base = d.xpos[ids[0]].copy()
            pts = [d.xpos[i] - base for i in ids] + [d.site_xpos[self.tcp_sites[a]] - base]
            yaw = float(q[self.j1[a]])
            c, s = np.cos(yaw), np.sin(yaw)
            out[a] = [[round(float(p[0] * c + p[1] * s), 4), round(float(p[2]), 4)] for p in pts]
        return out
