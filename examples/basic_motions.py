"""Robot-agnostic motion check (no objects needed): joint move, pose move, straight line, gripper, two-arm path.
Safe for a first real-robot run: small, slow motions around the start pose."""
import numpy as np
import robot

r = robot.connect()
s = r.state()
print("start:", {a: np.round(e.T[:3, 3], 3).tolist() for a, e in s.ee.items()})
for arm in r.end_effectors:
    js = r.description["groups"][f"{arm}_arm"]
    q0 = s.q(js)
    print(arm, "joint wiggle:", r.move(q0 + np.array([0.1, 0, 0, 0, 0, 0]), joints=js, speed=0.3))
    print(arm, "back:", r.move(q0, joints=js, speed=0.3))
    print(arm, "up 3 cm (line):", r.move([robot.pose([0, 0, 0.03])], times=[1.5], ee=arm, relative=True))
    print(arm, "gripper half:", r.gripper(arm, 0.04))
    print(arm, "gripper open:", r.gripper(arm, 0.08))
both = {a: [robot.pose([0, 0, -0.03])] for a in r.end_effectors}
print("both down 3 cm together:", r.move(both, times=[1.5], ee=list(r.end_effectors), relative=True))
print("final:", {a: np.round(e.T[:3, 3], 3).tolist() for a, e in r.state().ee.items()})
