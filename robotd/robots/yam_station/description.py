"""Description of the bimanual YAM station: what the driver needs and what the agent receives as `r.description`."""
from __future__ import annotations

from robotd.robots.yam_station import scene as S

MAX_JOINT_VELOCITY = [1.5, 1.5, 1.5, 2.0, 2.0, 2.0]   # rad/s driver defaults (agent may pass speed=)
VELOCITY_LIMIT = [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]       # rad/s hard cap: faster paths are refused, references clamped


def description(speed_scale: float = 1.0) -> dict:
    """speed_scale multiplies the default joint and line speeds (e.g. 0.5 on a real robot's first runs)."""
    arm_joints = {a: list(S.ARM_JOINTS[a]) for a in S.ARMS}
    finger = {a: S.FINGER_JOINTS[a][0] for a in S.ARMS}
    kp, kd, soft_kd, lim, vmax, vcap = {}, {}, {}, {}, {}, {}
    for a in S.ARMS:
        for i, n in enumerate(arm_joints[a]):
            kp[n], kd[n], lim[n] = S.DEFAULT_KP[i], S.DEFAULT_KD[i], S.TORQUE_LIMIT_NM[i]
            soft_kd[n] = S.SOFT_KD[i]
            vmax[n] = MAX_JOINT_VELOCITY[i] * speed_scale
            vcap[n] = VELOCITY_LIMIT[i] * speed_scale
        kp[finger[a]], kd[finger[a]], lim[finger[a]], vmax[finger[a]] = S.GRIPPER_KP, S.GRIPPER_KD, S.GRIPPER_FORCE_LIMIT_N, 0.2
        soft_kd[finger[a]] = S.GRIPPER_KD
        vcap[finger[a]] = 0.4
    return {
        "robot": "yam_station_linear_4310_d405",
        "files": {"mjcf": S.STATION_XML, "urdf": S.STATION_URDF},
        "world_frame": "origin midway between the arm bases on the table top; +x forward, +z up; "
                       "left base at y=+0.305, right base at y=-0.305. files['urdf'] is in this frame (both arms, "
                       "a <arm>_tcp link at the tool centre, the driver's limits); arms[arm] has a one-arm URDF in "
                       "its base frame and T_world_base. The MJCF is in the left-base frame: add y=+0.305",
        "arm_joints": arm_joints,
        "groups": {**{f"{a}_arm": arm_joints[a] for a in S.ARMS}, **{f"{a}_gripper": [finger[a]] for a in S.ARMS}},
        "end_effectors": {a: {"tcp_site": S.TCP_SITES[a], "group": f"{a}_arm", "gripper": a} for a in S.ARMS},
        "tcp_sites": dict(S.TCP_SITES),
        "finger_joint": finger,
        "gripper_width_range_m": [0.0, 2 * S.FINGER_OPEN_M],
        "cameras": {"left_wrist": "on the left wrist (D405)", "right_wrist": "on the right wrist (D405)",
                    "top": "overhead (D405), looking down at the table"},
        "default_kp": kp, "default_kd": kd, "soft_kd": soft_kd, "effort_limit": lim, "max_velocity": vmax,
        "velocity_limit": vcap,
        "gripper_speed": 0.08,          # m/s of the opening
        "linear_speed": 0.10 * speed_scale,  # m/s default for line moves
        "angular_speed": 0.8 * speed_scale,  # rad/s
        "tick_s": 0.04, "fast_tick_s": 0.004,
        # the settled-error correction (controller._settle_bias): rate 1/s, cap per joint (rad), held above speed rad/s.
        # A trim: with the cell's measured gravity factors (YAM_GRAVITY_FACTORS) what is left is friction, +-0.4-0.8 deg
        "settle_bias": {"rate": 3.0, "max_rad": 0.0262, "speed": 0.1},
        "ready_q": {a: list(map(float, S.READY_Q[a])) for a in S.ARMS},
        "quaternions": "the API uses 4x4 matrices; robot.wxyz(T) / robot.xyzw(T) give quaternions in either order",
    }
