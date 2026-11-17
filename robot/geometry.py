"""Pose helpers. Poses are 4x4 numpy matrices (world <- frame). Quaternion order is always in the function name."""
from __future__ import annotations

from robot._lazy import Lazy

np = Lazy("numpy", __name__, "np")
R = Lazy("scipy.spatial.transform:Rotation", __name__, "R")


def asarray(x) -> np.ndarray:
    """Any array-like (numpy, torch on GPU, JAX, lists) -> numpy float64."""
    if hasattr(x, "detach"):                      # torch
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def pose(xyz=(0, 0, 0), rpy=None, quat_wxyz=None, quat_xyzw=None, rotvec=None, R_mat=None) -> np.ndarray:
    """4x4 from a position and at most one rotation: rpy (rad, extrinsic xyz), quat_wxyz, quat_xyzw, rotvec, R_mat."""
    T = np.eye(4)
    T[:3, 3] = asarray(xyz)
    given = [v is not None for v in (rpy, quat_wxyz, quat_xyzw, rotvec, R_mat)]
    if sum(given) > 1:
        raise ValueError("give at most one rotation")
    if rpy is not None:
        T[:3, :3] = R.from_euler("xyz", asarray(rpy)).as_matrix()
    elif quat_wxyz is not None:
        w, x, y, z = asarray(quat_wxyz)
        T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    elif quat_xyzw is not None:
        T[:3, :3] = R.from_quat(asarray(quat_xyzw)).as_matrix()
    elif rotvec is not None:
        T[:3, :3] = R.from_rotvec(asarray(rotvec)).as_matrix()
    elif R_mat is not None:
        T[:3, :3] = asarray(R_mat)
    return T


def wxyz(T) -> np.ndarray:
    """Quaternion (w, x, y, z) of a 4x4 or 3x3: cuRobo, MuJoCo, PyRoKi/jaxlie, Isaac order."""
    x, y, z, w = R.from_matrix(asarray(T)[:3, :3]).as_quat()
    return np.array([w, x, y, z])


def xyzw(T) -> np.ndarray:
    """Quaternion (x, y, z, w) of a 4x4 or 3x3: ROS, scipy order."""
    return R.from_matrix(asarray(T)[:3, :3]).as_quat()


def T_from(pos, quat, order: str) -> np.ndarray:
    """4x4 from a position and a quaternion; order is 'wxyz' or 'xyzw' (required: the classic bug)."""
    if order == "wxyz":
        return pose(pos, quat_wxyz=quat)
    if order == "xyzw":
        return pose(pos, quat_xyzw=quat)
    raise ValueError("order must be 'wxyz' or 'xyzw'")


def inv(T) -> np.ndarray:
    T = asarray(T)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def twist(direction, speed: float) -> dict:
    """{"x","y","z"} twist of `speed` m/s along a signed axis ('-z') or a 3-vector."""
    if isinstance(direction, str):
        sign = -1.0 if direction.startswith("-") else 1.0
        return {direction.lstrip("+-"): sign * float(speed)}
    u = asarray(direction)
    u = u / np.linalg.norm(u) * float(speed)
    return {"x": u[0], "y": u[1], "z": u[2]}
