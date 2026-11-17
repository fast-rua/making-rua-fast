"""Pick the red cube and lift it ~12 cm: top-camera colour segmentation, guarded descent, force-limited grasp."""
import numpy as np
import robot
from helpers import touch

r = robot.connect()
f = r.cameras(["top"])["top"]
rgb = f.rgb.astype(int)
red = (rgb[..., 0] > 140) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
assert red.sum() > 20, "no red cube in the top camera"
pts = f.points(red)
centre, top = np.median(pts, axis=0), np.percentile(pts[:, 2], 95)
arm = "left" if centre[1] > 0 else "right"
print(f"cube at {np.round(centre, 3)}, top z {top:.3f} -> {arm} arm")

jaws_down = dict(rpy=[np.pi, 0, np.pi])
r.gripper(arm, 0.07)
r.move(robot.pose([centre[0], centre[1], top + 0.06], **jaws_down), ee=arm)
r.zero_force(arm)
res = touch(r, arm, "-z", speed=0.03, force=3.0, max_travel=0.15)
print("touch:", res, "tcp z", round(res.at.T[arm][2, 3], 4))
r.move([robot.pose([0, 0, 0.008])], times=[0.3], ee=arm, relative=True)       # off the table a little
g = r.gripper(arm, 0.0, force=15)
print("grasp:", g)
assert g.reason == "stalled", "closed on nothing"
lift = r.move([robot.pose([0, 0, 0.12])], times=[1.5], ee=arm, relative=True)
print("lift:", lift)

# verify before claiming success: the cube is gone from its spot on the table and the jaws still hold something
f = r.cameras(["top"])["top"]
rgb = f.rgb.astype(int)
red = (rgb[..., 0] > 140) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
on_table = f.points(red) if red.sum() else np.zeros((0, 3))
still_there = int(np.sum((np.linalg.norm(on_table[:, :2] - centre[:2], axis=1) < 0.03) & (on_table[:, 2] < 0.06)))
width = r.state().grippers[arm]["width"]
print(f"red points left at the old spot: {still_there}; jaw opening {width:.3f} m")
assert still_there < 20 and width > 0.02, "the cube did not come up with the gripper"
print("finish:", r.finish(f"lifted the red cube about 12 cm with the {arm} arm; its table spot is now empty"))
