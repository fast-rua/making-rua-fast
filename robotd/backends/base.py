"""The backend protocol: the only robot-specific code in the driver.

A backend measures joints, applies joint PD targets with gains (plus gravity compensation, as the robot's own
firmware or SDK does), advances time, and returns camera frames. Joint order is `backend.joints`: each arm's joints,
then that arm's driven finger joint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measured:
    t: float                 # robot clock, s
    q: np.ndarray            # joint positions (rad, or m for finger slides)
    dq: np.ndarray           # joint velocities
    tau: np.ndarray          # total motor effort (Nm, or N for fingers), gravity compensation included
    gripper_width: dict = None   # arm -> measured opening between the pads (m)
    tau_model: np.ndarray = None  # the robot's own feed-forward (gravity comp) per joint, if it reports one


class Backend:
    joints: list[str]
    lockstep: bool           # True: time advances only through advance() (simulation)
    name: str
    arm_force_trusted = False      # the arms' force estimate (contact) is worth telling the agent
    gripper_empty_width_m = 0.0    # the width a close on nothing ends at (m)

    def read(self) -> Measured: ...

    def write(self, q_cmd: np.ndarray, dq_cmd: np.ndarray, kp: np.ndarray, kd: np.ndarray,
              effort_limit: np.ndarray) -> None:
        """Hold these PD targets until the next write. effort_limit caps each motor's effort (|Nm| or |N|)."""

    def advance(self, dt: float) -> None:
        """Simulation: integrate physics for dt. Real robot: wait until the next control period."""

    def cameras(self, names=None, depth: bool = True) -> dict:
        """name -> {"rgb": HxWx3 uint8, "depth": HxW float32 m, "K": 3x3, "T": 4x4 world<-optical, "t": s}."""
        raise NotImplementedError

    def kinematics_model(self):
        """A MuJoCo model of the robot (world frame) for kinematics and the gravity model."""
        raise NotImplementedError

    def health(self) -> dict:
        """Robot-specific health for the operator's dashboard (motor temperatures, buses, ...). Must not block."""
        return {}

    def close(self) -> None:
        pass
