# The real cell

This is the real YAM cell the robot API drives. The robot manual describes the station as simulated; where this
description, `r.description` or your own measurements differ from it, they win. The numbers here were measured on this
cell in November 2026; the camera figures come from 110 saved frames of three episodes.

## Layout

- **Arms.** Two I2RT YAM arms, `left` and `right`, on one table, in the manual's world frame: origin on the table top
  midway between the bases, +x forward (away from the bases), +z up.
- **Bases.** The left base is at (0, +0.305, 0). The right base is not where the simulated one is: the cell's
  calibration puts it at (-0.013, -0.208, +0.015), 9.7 cm closer to the left in y and 1.5 cm higher, turned 2.4
  degrees. `r.description["arms"][arm]["T_world_base"]`, the URDF and the MJCF in `r.description["files"]` all have
  the calibrated bases: they are the driver's own model, so forward kinematics from them agree with `r.state()`.
- **Joints.** The calibration also corrects each joint's encoder zero (by up to about 3 degrees). The joint angles
  `r.state()` reports, and those you command, are already corrected: they mean what the model files mean.
- **The table.** Light wood; the arms' work area spans about x 0.1-0.8 m. Its top reads between 3 and 13 mm above
  z = 0, depending on the camera (see Cameras); the driver's model puts it at about 5 mm. Overhead lights
  make a bright glare patch on it in the fixed cameras.
- **Around the table.** Off its edges the cameras see the floor, light stands and the cell's frame. Beyond the table
  there is a blue backdrop.
- **Time.** Everything runs on the wall clock. Anything left moving keeps moving while you read, think or write code.

## Cameras

Four Intel RealSense RGB-D cameras. Colour and depth are both 640 x 480, depth registered to colour (one `K` per
frame), streaming at 30 frames a second; `r.cameras()` returns the latest frames.

| Camera | Sensor | Where | Field of view | Sees on the table |
|---|---|---|---|---|
| `top` | D435i | fixed at (0.77, 0.42, 0.86), looking straight down | 56 x 43 deg | x 0.26-0.77, y 0.04-0.59: the left half only. The right arm's side is outside it. |
| `front` | D435i | fixed at (0.06, 0.09, 0.98) above the bases, looking forward, 70 deg below level | 56 x 43 deg | x 0.11-0.76, y -0.53-0.54: the whole table, at a slant |
| `left_wrist` | D405 | on the left gripper, moving with it | 79 x 63 deg | wherever the gripper points |
| `right_wrist` | D405 | on the right gripper, moving with it | 79 x 63 deg | wherever the gripper points |

**Depth and point-cloud precision**, measured on the bare table in 110 saved frames:

| Camera | Pixels with depth | Distances in view | Table height | Table tilt | Scatter about the table (sd / 95%) | One pixel covers |
|---|---|---|---|---|---|---|
| `top` | 95% | 0.6-1.7 m | 3.4 mm | 0.6 deg | 3.6 / 7.0 mm | 1.4 mm |
| `front` | 94% | 0.5-1.7 m | 13.3 mm | 0.4 deg | 5.6 / 11.4 mm | 1.7 mm |
| `left_wrist` | 82% | 0.07-0.39 m | 7.6 mm | 3.2 deg | 2.5 / 5.0 mm | 0.5 mm |
| `right_wrist` | 84% | 0.07-0.38 m | 13.4 mm | 2.1 deg | 1.7 / 3.4 mm | 0.7 mm |

The table is flat, so each column shows a camera's error:

- **Absolute height.** The same table comes out between 3 and 13 mm high depending on the camera.
- **Between cameras.** Laid on `top`'s cloud, `front`'s fits best shifted by (+2, +8, -9) mm, about 12 mm in all, and
  that shift is steady from frame to frame. The wrist cameras' clouds sit 5-10 mm from the fixed cameras', and their
  offset changes with the arm's pose (by about 1 cm from one frame to another), because a wrist camera's pose comes
  from the arm's joint readings. A point seen by one camera is therefore placed in the world to about 1-2 cm.
- **Tilt.** The wrist cameras see the level table tilted 2-3 degrees. That is an error in their orientation: across a
  10 cm patch it makes a false slope of 3-5 mm, and more further away. The fixed cameras are within about 0.5
  degrees, about 1 mm per 10 cm.
- **Within one camera.** Points on a flat surface scatter about it by the amounts in the table: a few mm for the
  wrist cameras, 4-6 mm for the fixed ones, with one point in twenty further off. Relative positions within one wrist
  camera's view are good to a few millimetres: it separates steps of about 3 mm close up. A thin layer of a few mm on
  the table is within the fixed cameras' scatter.
- **Edges.** One pixel covers the size in the last column at the camera's working distance, and an edge blurs over a
  couple of pixels. Where the depth jumps (an object's edge against what is behind it), single points can float
  between the two surfaces.
- **The wrist cameras.** The D405 reads depth from about 7 cm; nearer than that is blank. The fingertips are about
  9 cm in front of the camera, so what they touch is in range. The TCP falls near pixel (331, 385) in `left_wrist`
  and (313, 369) in `right_wrist`. The fingers fill part of the bottom of the image and hide what is between them
  from the camera's side. The camera moves with every motion of its arm.
- **Gaps.** Missing depth is most common on dark or shiny surfaces, at depth edges, and nearer than a camera's
  minimum range (7 cm for the wrist cameras).

**Recordings.** From the start of an episode, every camera's colour is recorded, a few frames a second (6 in recent
episodes), in `/var/record/cameras`: one MP4 per camera, with each frame's wall time in `frames.csv`. A few times a
second (3 by default) every camera's colour and depth from one grab are recorded too, with K and the camera's pose
(`rgbd/`, indexed in `rgbd.jsonl`). `robot.recording.Recording()` reads the recordings by time (`frame_at`,
`frames`, `clip`; `cameras_at` returns an RGB-D grab as frames like `r.cameras()`, which `robot.viz` takes), and
`ffmpeg` is on `PATH`. A pause's `pause-N/` also has depth.

## Arms

- **Stopping short.** An arm stops short of a commanded pose in the direction it moved: about 1 cm after a move down,
  and up to 3 cm after a move up (the manual's ~4 mm is the simulation). `r.state()` says where it really is.
- **Force.** The arms have no force sensor, and their derived force and wrench estimates (`s.ee[arm].force`,
  `when.force`, `contact`) are not reliable here. The raw signals are there to read and judge for yourself
  (`s = r.state()`, joints in `s.joints` order):
  - `s.q - s.q_cmd`: each joint's tracking error (rad). Joint friction alone holds a joint anywhere within about
    0.4-0.8 deg of its command on joints 1-3 (kp 80 Nm/rad, so ~1 Nm); a joint pushed or blocked falls further behind.
  - `s.effort()`: the motors' measured torque (Nm, gravity included); `s.effort_ext()`: that minus the model's
    gravity (about +-1 Nm of friction and model error at rest).
  - `s.ee[arm].deflection`: the TCP's measured minus commanded position (m).
  - The grippers' width and effort (`s.grippers`) are reliable.
  Compare against a baseline taken just before (the same pose, not moving) rather than against zero.
- **Gains.** `gains=` is capped at the defaults (`r.description["max_gain_scale"]` is 1). Stiffer than that, the arm
  chatters against what it presses on.
- **Speed.** The joint speed limits are in `r.description["max_velocity"]` and `["velocity_limit"]`.
- **Reach.** Measured by sampling the arm's model, in each arm's own base frame:

  | Gripper | TCP height | Reach at table height |
  |---|---|---|
  | jaws straight down | up to about 0.22 m | about 0.50 m forward |
  | tilted 35-60 deg from vertical | up to about 0.48 m | about 0.73 m |
  | sideways | up to about 0.7 m | |

  In world terms, with the jaws down at table height, the left arm reaches y from about -0.15 to +0.55 and the right
  from about -0.50 to +0.20, out to x of about 0.45. The fence (below) narrows this to each arm's half of the table.

## Grippers

- **Opening.** 0-94 mm between the pads; `force=` up to 50 N per pad (the default).
- **Squeeze.** The width target sets most of the squeeze. The jaws are stiff: each mm the target lies inside the
  object adds roughly 90 N. `force=` takes hold only once the jaws are blocked and moving slowly. A target of 0 on a
  wide object (a 6.7 cm ball at full speed, 2026-11-04) crushed it to 37 mm at up to 307 N before the overload guard
  aborted the grasp (`gripper_overload`).
- **Closing on a thin layer.** A gripper's width and its stall flag cannot tell a thin layer (one layer of cloth)
  from an empty close: both end near 2.9 mm, and `stalled` is unreliable there. Something thick enough to stop the
  jaws shows as `stalled` with that width.
- **Seeing the jaws.** The wrist cameras look at the jaws from 9 cm, and `robot.viz` (in the manual) can draw the
  fingers where a move would put them, mark what lies between them, and count it.

## Safety limits, always on

- **The fence.** Each arm's TCP is kept in a box over its half of the table, and the TCPs' speed in the table plane
  is capped. Both are in `r.description["fence"]`; in the towel episodes the box was x 0.12-0.70, y 0.05-0.50 for the
  left arm and -0.45-0.05 for the right, z up to 0.65, at 1 m/s.
  - A command that would leave the box, or go faster, is refused with `robot.InvalidCommand`.
  - The box's floor bounds only the TCP. A tilted gripper's fingers and body reach below it, so the fence's
    `gripper_floor` (4.5 mm) bounds the whole gripper: a target that would take any part of it lower is refused.
  - The manual's "The workspace fence" section has the details.
- **Separation.** The two arms are kept at least 2 cm apart, measured between capsules around their links and
  grippers (`r.description["separation"]`). A command whose goal is closer is refused (`robot.InvalidCommand`). An arm
  that moves closer stops, and its command ends `aborted` with the reason `separation: ...`.
- **The operator.** Someone may be watching remotely and can stop or park the arms at any time. Until they release
  them, every command is refused with `robot.Fault`.
