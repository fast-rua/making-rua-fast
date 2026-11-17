# Robot manual (agent-facing)

You control a robot by writing and running Python. `import robot; r = robot.connect()` gives you the robot. Everything
else is ordinary Python, with whatever libraries are installed. When this manual is unclear, read the code:
`robot/__init__.py` (the client) and `robotd/core/controller.py` (what the driver does every 4 ms).

This manual describes the station as simulated. A real cell comes with a description of its own (its layout, its
cameras as measured, how its arms and grippers behave), and where the two differ, the cell's description, `r.description`
and your measurements win.

## The robot: bimanual I2RT YAM station

| Part | Facts |
|---|---|
| Arms | `left`, `right`: 6 revolute joints each (`left_joint1..6`). Bases at world (0, +0.305, 0) and (0, -0.305, 0). Reach about 0.6 m from the base. Joint limits are in `r.description["files"]["urdf"]` (see Using libraries). |
| Gripper | Linear, two fingers. Opening 0–0.094 m between the pads. `r.gripper(arm, width, force=N)`: N is each pad's squeeze (at most 50 N, the default). A grasp that stops on an object ends with reason `stalled`, and `Result.width` is what it holds. Each fingertip's face is 12 mm wide at its end and 25 mm wide 15 mm up, with a window in its middle above that. |
| Tool point (TCP) | The centre between the fingertip pads (site `tcp_left`/`tcp_right`). With the jaws closed the pads meet there. Poses you command are TCP poses unless you pass `tcp=` (a 4x4 offset from the TCP, e.g. to a tool tip). |
| Cameras | `top` (overhead, looking down at the table), `left_wrist`, `right_wrist` (D405 RGB-D cameras that move with their grippers). Colour and depth, 640 x 480: see Cameras and point clouds below. |
| World frame | Origin on the table top midway between the bases; +x forward (away from the bases), +z up. The table top is z = 0. |
| Reachability | Jaws straight down (`rpy=[pi, 0, pi]`) reach the whole table area x 0.2–0.5 m in front of a base when the TCP is at most ~0.10 m above the table; higher than that, tilt the jaws about 20° toward +x. An unreachable goal raises `InvalidCommand` with the IK miss distance. When some axes do not matter, leave them to the driver with `free=` (below) instead of trying poses one by one, and when they do, check the candidates together with `r.reachable` rather than moving to one and reading the refusal. `r.ik`, `r.reachable` and `r.move(..., dry_run=True)` ask without moving. |
| Accuracy | The wrist is soft (gains kp 10): a moving TCP lags its reference by up to ~1 cm, and settles within ~4 mm (a still arm's steady offset, from the gravity model, is integrated away within ~1 s: `r.description["settle_bias"]`). Do not rely on mm-level position accuracy; use guarded moves (`until=` conditions) to find surfaces. A quick path (a stroke under ~0.2 s) lags more, a few cm and degrees at a tool 10 cm out, and a held tool's weight sags the wrist (a 0.14 kg paddle: ~1 cm, ~1.5 deg): `r.state()` says where the tool really is, so measure and compensate. Near a joint limit or a straight elbow, small tool motions need large joint motions and track worst. |
| Speed | Joint velocity limits are in `r.description["velocity_limit"]` (rad/s): 3 on joints 1-3, 4 on the wrist. A joint path faster than that is refused with `InvalidCommand` (the message says how much to stretch the times). Turning a held tool over (wrist roll, joint 6, half a turn) takes at least 0.8 s. |
| Force | There is no force sensor. Force is estimated from joint torques: about ±0.5 N in simulation and ±1–3 N on the real arm. Call `r.zero_force(arm)` just before contact and after grasping. On the real arm, deflection (`when.deflection`, encoder only) is the more reliable contact signal under a soft compliance. Rely on force as little as you can: as a stop when something is touched, not to align, insert or place by feel. |
| Timing | Simulation is paused whenever your code is not inside a blocking call or `r.step()`, so thinking costs no time. On the real robot time keeps running: a slow loop is a slow robot. `r.lockstep` says which (a simulation can also run in real time, like the robot). |
| Workspace fence | When the driver runs with one (`r.description["fence"]`, else `None`), each fenced arm's TCP must stay in its box, and the TCPs' speed in the table plane is capped. See below. |

## The calls

| You want to | Call |
|---|---|
| Look | `f = r.cameras(["top"])["top"]`, then `f.rgb`, `f.depth`, `f.points(mask)`; `roi=` for a window |
| Read the robot | `s = r.state()`: `s.q(names)`, `s.ee["left"].T` / `.v` (TCP linear velocity, 3 values, m/s, world) / `.force` / `.contact` / `.deflection`, `s.grippers["left"]` (`width`, `stalled`: closed on something, `empty`: closed on nothing). `.contact` comes from the arms' force estimate: `r.description["arm_force_trusted"]` is False on the real arm (it read contact with the hand 30 cm in the air), and the `[state]` line leaves it out there |
| Go to a pose | `r.move(T, ee="left")`: there and stop, smoothly (`line=True` for a straight line, `speed=` m/s, `duration=` s) |
| Leave axes free | `r.move(T, ee="left", free="yaw")`: any turn about the tool's z axis (a top-down pinch); `"yaw180"`: as given or half a turn more (the same grasp for a parallel gripper); `"rotation"`: position only. The driver picks the solution quickest to reach that keeps away from joint limits (along a path, the turns that make the whole path cheapest, without wrist flips); `res.goal["left"]` is the pose it chose |
| Go to joint angles | `r.move(q, joints=names)` |
| Follow a path | `r.move(poses, ee="left")` passes through them, timed from the speed limits (`speed=`; `duration=` for the whole path), or at `times=[...]` (s from the start, one per pose) or `dt=`. `velocities=`: the controlled point's m/s at each pose, to pass a pose at a time with a speed; `relative=True`: offsets from the start; `free=` one value or one per pose |
| Run a planner's path | `r.move(qs, joints=names, dt=0.02)` or `times=`, `velocities=` (rad/s per point) as the planner gives them (numpy, torch or JAX). A first point that is the current state (at t=0, or the first of dt-spaced points) is dropped; a point outside the joint limits or too fast is refused, naming the point, the joint and the limit |
| Servo on something moving | re-send short paths in a loop: `r.move(T, ee="left", times=[tau], velocities=[v], wait=False)`, tau 2-3x the loop's period (0.08 s at 30 Hz). Every move continues from the arm's current position, velocity and acceleration, so re-sending is smooth |
| Check before moving | `r.ik(T, ee="left", free=None)` → `{"joints", "q", "ok", "error_m", "T", "margin"}`; `r.reachable([T1, T2, ...], ee="left", free=..., line=False)`: many candidates in one call (approach angles, tilts, heights), each checked as `r.move` would check it (fence, gripper floor, IK, separation from the other arm where it is commanded and where its running move ends; `line=True`: the straight line too), with `why` the refusal it would get, None when `ok` (filter on `ok`, rank by `margin`); `r.move(..., dry_run=True)` → a `Plan`: ok, or where a line or path would fail (point, fraction of its leg, pose, miss) and the tightest margin on the way (`limit_rad` to the nearest joint limit, `singular_m` near 0 at a straight elbow) |
| Move until something happens | `r.move_at({"z": -0.02}, ee="left", until=when.force("left", "-z") > 3)`; `max_travel=` m (default 0.30) ends it with reason `max_travel` |
| Find a surface | the guarded move above with `until={"contact": when.force(arm, "-z") > 3}`: reason `contact` or `max_travel`; `res.at.T["left"]` is where |
| Grasp / release | `r.gripper("left", w - 0.003, speed=0.02, force=15)` (w: the object's width) / `r.gripper("left", 0.08)` |
| Press with a force | any motion with `compliance=Compliance(frame="world", stiffness={"z": 500}, force={"-z": 5})` |
| Let time pass | `r.step(ticks)` (40 ms each), `r.step_until(t)` (to a robot time, in 4 ms ticks: a fixed-rate loop) |
| See a detection or a plan before acting on it | `robot.viz` (below): draw it on a frame, render the gripper where a move would put it, and look |
| Stop everything | `r.stop()` |
| End the episode | `r.finish("what you did and what you believe the outcome is")`. No success signal comes back: check with your own eyes (cameras) before you finish. |

Every motion call takes the same extra arguments:
- `until=` a condition, or `{name: condition}`. When it fires, the motion ends with status `stopped` and reason = the name.
- `then=` what comes next: another command or a list of them (built with `robot.Move/MoveAt/Gripper`), started when the motion ends as asked, whether it arrived or a condition fired (in simulation in the same 4 ms tick; on a real robot as soon as it is built, the arm holding meanwhile: a few ms for a goal, ~75 ms with `free="yaw"`, seconds for a long `free=` path); a dict picks by the reason (`{"contact": ..., "max_travel": ...}`). `"hold"` (default) and `("back_off", metres)` are reactions to an `until=` condition.
- `on={name: (condition, command)}` fires a command once without stopping the motion, e.g. open the gripper at 60% of a
  throw, or start the next motion of the same arm at a time (`when.time() > 0.4`): it takes over smoothly. A command
  that is refused as it fires (an `on=` or `then=` one) ends the command that fired it, `aborted` with the refusal as
  its reason (it raises `Aborted`), and the arm holds.
- `compliance=Compliance(...)` and `gains={"scale": 0.5}` or `{"kp": [...], "kd": [...]}`: joint stiffness for this
  command (a command without `gains=` runs with the defaults; one that ends holds with its gains until your next
  command on those joints).
  A compliant command that ends keeps its push (a press keeps pressing) until your next command on that arm;
  `then="hold"` stops pushing.
- `tcp=4x4` moves the controlled point: rotate about a hinge, or press with a tool tip.
- `start_at=` starts on the robot clock: what runs on those joints keeps going until then, and a timed path that arrives
  late keeps its waypoints on that clock. `replace=True` (default) lets a new command take over joints a running one
  uses, smoothly: it continues from the current reference and its velocity (resending a path mid-motion is fine).
- `wait=False` returns a handle (`.result()`, `.done`, `.stop()`); then step or wait to let it run.
- `timeout=` in seconds.

Conditions run in the driver every 4 ms, not in your loop:
- `when.force(ee, "-z")`, `when.torque`, `when.effort(joint)`, `when.deflection(ee, "z")`, `when.travel(ee, "-z")`, `when.pose(ee, "z")`, `when.speed(ee)`, `when.time()`, `when.progress()`, `when.contact(ee)`, `when.gripper(name).width` (compare it) / `.stalled` (already a condition: `until=when.gripper("left").stalled`), `when.signal(name)`;
- combine with `>` `<` `&` `|` `~` `abs()` `+` `-`, `.held(seconds)` and `when.after(first, then)`.

## The workspace fence

The operator can start the driver with a fence (`robotd --fence`, or a scene's own): a box per arm for its TCP and a
speed cap in the table plane, for working near people. `r.description["fence"]` shows it, e.g.
`{"left": {"x": [0.13, 0.54], "y": [-0.02, 0.19], "z": [0.11, 0.24]}, "right": {...}, "max_speed_xy": 1.0,
"margin": 0.02}` (metres, world frame; `null` leaves that end open; an arm left out is not fenced).
- A target outside the box, or a path point outside it, is refused with `InvalidCommand` naming the limit
  (`fence: the target: left TCP x 0.600 m > x_max 0.540 m`). So is a command that asks for more planar speed than
  `max_speed_xy` (`speed=`, a timed path's legs or `velocities=`, a `move_at` twist): stretch the times.
- A motion whose reference would leave the box on the way (a joint move that swings out) stops at the edge: it ends
  `aborted` with a `fence: ...` reason and the arm holds inside. The reference never moves faster than the cap in
  the table plane. A `move_at` toward a wall ends `done` with reason `fence` at the wall.
- The measured TCP outside its box by more than `margin`, or moving faster than 1.5 x the cap for 20 ms (pushed,
  struck), latches a `Fault`. `r.recover()` clears it; an arm outside its box may then only move back in.
- An arm starts outside its box (at rest): its checks start once its TCP is inside. Parking (the operator's PARK,
  the driver's exit) ignores the fence, and the arm's checks start again when it is back inside.
- `"gripper_floor": z`, when the fence has one, bounds the whole gripper, not only its TCP: the fingers, the body and
  the wrist camera stay above that height (the table). A tilted gripper's fingers and body reach below its TCP, so
  it meets this floor before its TCP meets the box's. A target or path point that would put the gripper lower is
  refused (`fence: the target: left gripper's lowest point z ... < gripper_floor ...`), and a motion whose reference
  would take it lower stops there (`aborted`), as at the box's walls.
- `r.description["max_gain_scale"]`, when set (the real robot: 1), caps a command's `gains=`: a `scale` above it,
  or `kp`/`kd` above the defaults times it (`kd` times its square root), is refused with `InvalidCommand`.

## Results and errors

- A command returns `Result(status, reason, t, duration, at)`. `status` is one of `done`, `stopped` (a condition fired), `replaced` or `timeout`. `at` is the state when it ended; `goal` the pose(s) free axes settled on.
- A condition firing is normal. Read `reason`.
- Exceptions mean something you did not ask for:
  - `Aborted` (the motion could not go on: the arm could not track its path, `cannot_track`; a `then=`/`on=` command was refused as it fired; a gripper jammed, `gripper_overload`). `e.result` is its Result and the arm holds where it is: look at what happened, then decide;
  - `InvalidCommand` (bad arguments, unreachable goal; the message says what);
  - `Fault` (a guard latched, e.g. `max_force`; the arm holds; call `r.recover()`). When the operator stops the robot,
    your running command ends `aborted` with reason `operator_stop` and every command raises `Fault` until the
    operator releases it; `r.recover()` cannot. Stop and report;
  - `Busy` (`replace=False` and the joints are in use);
  - `NotSupported` (this robot or backend lacks the feature asked for).
  All are subclasses of `robot.RobotError`.

## Cameras and point clouds

`r.cameras(names=None, depth=True, roi=None)` takes a frame from each camera named (all of them by default) and
returns `{name: Frame}`:

| Field | |
|---|---|
| `rgb` | H x W x 3 uint8 in RGB order (OpenCV's own functions expect BGR: `cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)`). |
| `depth` | H x W float32, metres along the camera's axis (the z of the point in the camera's frame, not its distance along the ray). 0 or NaN where there is no reading. It is registered to `rgb`: pixel (u, v) of both looks along the same ray, and one `K` serves both. |
| `K` | The 3 x 3 intrinsics of this image: fx, fy (pixels), and the principal point cx, cy. |
| `T` | The camera's pose, world <- optical: +z looks forward, +x is right in the image and +y is down. `None` when the camera has no calibrated pose (`pose_error` says why), and then `points()` refuses. |
| `t` | The robot-clock time the image shows. On a real robot the frame reaches you 10-50 ms later. |
| `roi` | With `roi=[u0, v0, u1, v1]` (one box, or `{name: box}`), only that pixel box comes back and `K` is shifted to match, so everything below works on the crop as on the full image. It is quicker to send and to search when you track something small. |

- **Pixel to world.** With `z = depth[v, u]`, the point in the camera's frame is `((u - cx) z / fx, (v - cy) z / fy,
  z)`, and `T @ [x, y, z, 1]` puts it in the world. `f.points(mask)` does this for every pixel of a boolean mask
  (`None`: all of them) and returns N x 3 world points; pixels without a reading are left out, and `stride=n` takes
  every n-th row and column. Keep the mask's pixel indices if you need to go back.
- **World to pixel.** `p = inv(T) @ [x, y, z, 1]`, then `u = fx p_x / p_z + cx`, `v = fy p_y / p_z + cy`; a point
  with `p_z <= 0` is behind the camera. `viz.Overlay(r, cam).uv(points)` does it for many points.
- **Fusing cameras.** Every camera's points are in the same world frame, so clouds from several cameras can be
  stacked. Each camera's pose is calibrated on its own, though, so on a real cell the same surface can sit a few
  millimetres to a centimetre apart from one camera to another. Compare shapes and heights within one camera before
  comparing single points across cameras.
- **What real depth cameras miss.** Black, shiny and transparent surfaces give no reading or a wrong one. So does
  anything nearer than the camera's minimum range. At an object's edge a pixel can mix the near and the far surface
  into a point that floats between them ("flying pixels"). Noise grows with distance, roughly with its square, and a
  thin layer on a surface (cloth, paper) is within that noise unless the camera is close. A simulated camera renders
  exact depth and has none of these.
- **Being seen.** The arms appear in the fixed cameras' images, and a wrist camera always sees its own fingers. A
  camera sees only the surfaces facing it: the underside of anything, and what the arm or an object hides, are not in
  its cloud.
- **Cost.** Each call sends whole images through the camera server: ask for the cameras you need, `depth=False` when
  colour is enough, and `roi=` in a tracking loop. In lockstep simulation no time passes while you look.

## Pictures: `robot.viz`

Pictures of what your code found and what it plans: drawn on camera frames, or in 3D from the cameras' point clouds,
with the arm rendered where a move would put it. They are image files you can open and look at, and the operator's
dashboard shows them too.

```python
from robot import viz
o = viz.Overlay(r, "left_wrist")          # a fresh frame from that camera to draw on
g = o.ghost("left", T, width=0.05)        # the gripper where r.ik puts TCP pose T, jaws 5 cm apart
o.point(p, "target")                      # world points (3 or N x 3), labelled
o.path([T_above, T])                      # the TCP's way there
o.mask(m, label="object")                 # a pixel mask of this image
o.show("check")                           # writes views/check.png and prints its path: open it and look
viz.show("depth", f.depth)                # any image: RGB, grey, bool, or a float map (coloured by value)
```

- `o.ghost(arm, target=None, q=None, width=None, free=None, parts=None, after=None)` renders the arm's own meshes at
  the joints `r.ik` finds for `target` (from `after=`, as for `r.reachable`, if given; or at joint values `q`, or where
  the arm is now). It is translucent where the camera
  would see it and **red where a surface the camera sees is in front of it**: there it would be inside or under that
  surface. The arms' own pixels where they are now are not counted as surfaces. A surface within about 6 mm (or 1% of
  its distance) of the ghost does not hide it. `parts`: `"fingers"` (the default in a wrist camera), `"gripper"`
  (the default elsewhere) or `"arm"`. It returns `{"ik_ok", "error_m", "q", "T", "pixels", "behind_fraction",
  "jaw_box", "between"}`, so code can use the result as well as you.
- **The arms' own points are left out** of `between` and of the 3D views' point cloud: depth points within 6 mm of
  where the model puts the arms now (12 mm in a camera that does not ride on the arm, whose view of the arm carries
  the arm pose's error) are the arms themselves (`viz.SELF_MARGIN`, `viz.SELF_MARGIN_FIXED`). An empty gripper reads
  0-4 points on the real cell. Material that close to a finger, such as a sheet pinched flat between closed pads, is
  not counted either: read a hold from the gripper's width and whether it stalled.
- **Between the jaws.** The ghost also outlines the space between its fingers. That space runs from one finger pad's
  inner face to the other's, is as wide as the pads, and reaches 3 cm up from the tips. Whatever the camera sees
  inside it is coloured pink. `between` says what is there: `{"count", "unit", "gap_mm", "span_fraction",
  "above_tips_mm"}`. Here `span_fraction` is how much of the gap the points cover (in 2 mm bins along the closing
  line), and `above_tips_mm` is their [lowest, highest] height above the fingertips. Zero points means that camera
  sees nothing there that the jaws would close on.
- With no target, the ghost is drawn where the arm is, on top of the real arm when the camera's pose is right. A wrist
  camera moves with its arm, so a ghost far from the arm's present pose is seen from where the camera is now.
- `o.uv(points)` gives the pixels and depths of world points. `o.img` is the RGB image to draw on with cv2 yourself
  (colours are `(r, g, b)`). `o.text(line)` adds a line to the legend.
- `show` writes `views/NAME.png` in the program's run directory (`.robot/runs/<id>/views/`; `.robot/views/` outside a
  run) and replaces the last image of that name. `views/history/` keeps every one. It prints `[view] NAME: PATH` and
  puts a copy on the operator's dashboard. A ghost takes about 30-50 ms.

**3D views.** `viz.View3D` fuses every calibrated camera's depth into one coloured point cloud around a point, and
draws it from virtual cameras you choose, with the same calls (`ghost`, `point`, `points`, `path`, `pose`, `text`):

```python
v = viz.View3D(r, center=T)                    # T: a 4x4 or a point; radius= m framed (default 0.12)
g = v.ghost("left", T, width=0.05)             # body faint, fingers plain, red where it would be inside a surface
v.show("grasp-3d")                             # the default four views in one image
v.show("grasp-3d", views=["jaws", "front", (30, 15)])
```

- Views of the gripper (they need a ghost):
  - `"jaws"` looks through the gap between the fingers, along the gripper's normal: what would be between the jaws.
  - `"along"` looks along the line the jaws close on: the fingertips' heights against the surface.
  - `"approach"` looks down the gripper's axis from behind it.
- Views of the world: `"top"` (the far side up), `"front"` (from +x, facing the arms), `"back"`, `"left"` (from +y),
  `"right"`, `"iso"` (from the front-left, above) and `"iso-right"`.
- Any other view: `(azimuth, elevation)` in degrees, looking at the centre from that direction (azimuth from +x), or
  a camera pose from `viz.look(eye, at)`.
- The default is jaws, along, top and iso about a ghost, and front, left, top and iso without one.
- A level view is a slice: only the points within 4 cm of the centre along the line of sight (2 cm for `"jaws"`), so
  a surface shows as its profile, with height lines every cm.
- A face of the ghost is red when the cameras that see the spot (at least two, when two have a reading) see a
  surface in front of it: it would be inside or under that surface.
- Points between the jaws are pink in every view, and the jaw box is outlined. The ghost's `between` counts them
  over all the cameras together.
- The cloud only has what the cameras see. Surfaces facing away from every camera, and anything they are blocked from,
  are missing. Four views take about 0.3 s.

**Fewer, smaller images.** `viz.show_cameras(name, r, width=960)` shows every camera (or `names=`) as one labelled
grid image; `viz.grid(images, labels, cols=2, width=960)` tiles any images into one. `viz.marks(frame, name,
mask=None, step=40, extra=None, max_marks=60)` puts numbered points on a camera frame (a grid every `step` pixels,
inside `mask` when given, the mask outline's corners first, and `extra` pixels of yours) and returns
`{number: world xyz}`; with a name it shows the image and prints `[marks NAME] 1:(x,y,z) 2:...` (metres).
`on_table=True` keeps only what lies over the table (`viz.table_box(r)`: the arms' fence boxes padded by 10 cm, from
2 cm below z = 0 up), so a colour mask that also catches the floor or a backdrop needs no crop; `within=((x0, x1), (y0,
y1), (z0, z1))` keeps a box of yours; `picture=False` prints the points without the image. `viz.inside(frame, box)` is
the pixel mask behind both.

**Hooks: what you would fetch after a motion, with the motion.** In `python -m robot.kernel`, `kernel.after_motion(fn)`
runs `fn(r, frames)` at the end of every call that moved the arms: `r` is the call's robot, `frames` one grab of every
camera with depth (taken the first time the hook reads it). It runs after the call's code and its `[state]` line,
before its `[views]` line, so what it prints and the images it shows come back in that call's own output.
`kernel.after_call(fn)` runs after every call. Registering a function of the same name again replaces it;
`kernel.remove_hook(name)`, `kernel.hooks()`. A hook that raises prints `[hook NAME] error ...` and the call ends as it
would have; one that takes over a second says so, since every such call pays for it. A mask and its points take
about 0.1 s; a 3D render 0.5-1 s.

```python
def towel(r, frames):
    f = frames["front"]; h = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2HSV)
    viz.marks(f, "towel", mask=(h[..., 0] > 85) & (h[..., 0] < 120) & (h[..., 1] > 90), on_table=True, picture=False)
kernel.after_motion(towel)        # every later call that moves the arms ends with "[marks towel] 1:(x,y,z) ..."
```

**Actions: look at a plan while it runs.** In `python -m robot.kernel`, `kernel.start(fn, name=None, timeout=120)` runs
`fn(r)` (or `fn()`) in the kernel and returns its `Action` at once; `r` is the actions' connection, not closed
when the call ends. Every later call reports it (`[action NAME] running for 3.1 s`, what it printed, and as it ends
`[action NAME] done in 9.8 s`, or stopped, or its error); the call that sees it end prints the arms' `[state]`, shows
every camera as one grid (`views/after-NAME.png`; `views=False`: none) and runs the after-motion hooks. Actions share one
robot connection, kept open from one to the next. `a.stop()` halts every motion (the arms hold) and the action's code goes no further;
`a.wait(timeout=None)` waits for it (True if it ended on its own, without an error); its timeout stops it too. One
action at a time; a call that sends the arms a command stops the running action first. `kernel.action()` is the latest
and `kernel.actions()` lists them.

```python
def grasp(r):
    r.gripper("left", .07); r.move(A, ee="left", duration=3); r.move(G, ee="left", line=True, duration=3)
    r.gripper("left", w - 0.003, speed=0.02, force=30); r.move(L, ee="left", line=True, duration=3)   # w: its width
kernel.start(grasp)               # returns at once; next call: kernel.action().stop(), or .wait(): its output is the look
```

**Pushed action reports (in an episode).** When the harness sets `ROBOT_KERNEL_EVENTS` (a directory), the action makes
its own report as it ends: how it ended, what it printed, the arms' `[state]`, the `after-NAME` grid and the
after-motion hooks (one camera grab shared with them). The harness sends it to you as a message, `[action NAME] ...`
with the images attached, into your running turn or as a new turn if you have ended yours. Nothing waits or polls:
start, look at the plan, then do something else or end your turn. A call that does `a.wait()` gets the report in its
own output instead; other calls only say `[action NAME] done in 9.8 s (its report is sent to you as a message)`.
`EVENTS/actions.json` names the running action, and the episode stays open while one runs.

**The queue: the next action while this one runs.** `kernel.queue(fn, name=None, timeout=120, views=True)` starts `fn` as
the running action ends, or at once if none is running, and returns its `Action`; at most three wait. As an action ends the
next one starts first, and only then is its report made; the report says what started (`[queue] started: B`). An action that
ended with an error or a stop starts nothing: everything queued is dropped (`[queue] dropped: B`) and the arms hold. A check
at the top of `fn` that raises is the same thing. `kernel.queued()` lists what waits, `kernel.flush()` drops it, and
`a.stop()` on a queued action takes it out of the queue. A timeout counts from the action's start.

Every action has a number (`#12`). `kernel.queue(fn, replace_from=12)` takes #12 and everything behind it out of the queue and
puts `fn` (or a list) in their place, in one step; if #12 has started, nothing changes and it raises. `kernel.flush(from_item=12)`
drops #12 and what is behind it. In an episode with a queue, each message carries `[queue]` lines: what runs (`running #14 grasp:
3.1 s in, command 3 (move)`) and what waits, in order; and when your turn ends while an action runs, a `[look N]` message
brings the arms' state, every camera as one grid and those lines.

**Checking the next action from where the queue will leave the arms.** A check made while an action runs sees the arms
where they are now, but the next action's moves are solved when they run, from where the running action ended.
`r.reachable(cands, ee, after=...)`, `r.ik(T, ee, after=...)` and the ghosts (`o.ghost(arm, T, after=...)`,
`View3D.ghost`) take `after={arm: where it will be}`: that arm's joint values, a 4x4 TCP pose, or `{"target": T, "tcp":
..., "free": ...}` as its move was given. IK starts from there (as the move will), and an arm named in `after` is checked
for separation there instead of where it is now. A ghost is still drawn on the latest frame, with the rest of the robot
where it is now; only its IK's starting point changes. `kernel.planned_end()` gives that dict: for each arm, the end of
the last move the running action has sent it so far, unless an action declared its end with `kernel.queue(fn,
ends={arm: pose or joints})` (the latest that says wins); an arm none of them says anything about is left out. It cannot
see moves the running action has yet to send, nor where a queued action without `ends=` goes.

```python
nxt = kernel.planned_end()                                   # {"left": {"target": G, ...}, ...}
out = r.reachable([A2, G2, L2], ee="left", after=nxt)        # as the next action's moves will be solved
kernel.queue(grasp2, ends={"left": L2})                      # the action after it can plan from L2
```

Without `after`, `r.reachable` checks the other arm both where it is commanded now and where its running move ends, so a
goal it would collide with once it gets there is refused at check time (`why`: `separation: ... once right gets where its
running move_to (command 12) ends`); `r.move` refuses it as well. robotd also remembers the IK solutions it finds for
`r.reachable`, `r.ik`, dry runs and moves (a few dozen per arm, keyed by target, `tcp` and `free`). When a move's own solve,
from where the arm is as it starts, misses, the remembered solutions for a target within 1 cm and 0.1 rad are seeds too:
a target `r.reachable` passed near the edge of reach is not refused later because the arm started elsewhere. A solve that
reaches stands, unless a remembered solution is at least 0.5 s quicker to go to (the solve found another branch). The
fence, gripper floor, joint limits and separation apply to whichever solution is chosen.

**What a program prints as it exits.** A program that showed images prints `[views] ["/path/a.png", ...]`, every
path it showed, as one JSON list (`quiet=True` shows are left out; `ROBOT_EXIT_VIEWS=0` turns it off). A program that
sent the arms a command prints `[state] ...` (`State.summary()`): each end effector's position, its gripper's width
and whether it stalled on something or closed on nothing (a close asked to go below the empty-jaw width,
`r.description["gripper_empty_width_m"]`, that ended within 0.5 mm of it; `Result.empty` says the same), contact
where the arms' force estimate is trusted (simulation), a latched fault (`ROBOT_EXIT_STATE=0` turns it off).

**Watching a motion.** `robot.watch.watch(r, arms=None, view_every=None, every=0.25, check=None, signal="watch",
cameras=None, depth=False, lines="changes")` is a with block whose threads (and connections) are its own, so it never
holds up the code inside it. With `view_every=N` it renders, about every N seconds, one compact image of the watched
arms, `views/monitor-K.png`: the wrist and front cameras with each arm's ghost where it is now and its jaw box, and
the point cloud's `jaws` and `along` views around its TCP (about 0.5 s per arm). It prints `[watch view +1.2s] PATH |
left: between the jaws: ... (at the start of the block: 2, so +158)`. `check(state, frames)` or `check(state, frames, between)` runs every `every`
seconds; when it returns a reason, the watcher prints `[watch STOP] reason` and sets the signal to 1, which ends any
motion started with `until=when.signal("watch") > 0`. A state line is printed when a gripper's width, a close on
nothing or (simulation) contact changes (`lines="all"`: every tick; `"none"`: never). The signal is set to 0 as the block starts and ends.

```python
from robot import watch, when
with watch.watch(r, every=0.25, check=lambda st, fr: "closed on nothing" if st.grippers["left"]["width"] < 0.004
                 else None):
    r.move(T_lift, ee="left", duration=3.0, until=when.signal("watch") > 0)
```

## Using libraries

- `r.description["files"]["urdf"]` is both arms in the world frame, written from the driver's own model: a
  `left_tcp` / `right_tcp` link at the tool centre (the point `r.move(T, ee=...)` puts at T) and the driver's joint
  limits, velocity limits and efforts. Load it into cuRobo, PyRoKi or any other URDF library or viewer directly.
- `r.description["arms"]["left"]` is one arm for a per-arm planner: `urdf` (rooted at its base), `base_link`,
  `tcp_link`, `joints`, `finger_joints` (lock them), and `T_world_base` (4x4): a world target T is
  `inv(T_world_base) @ T` in that URDF's frame.
- `r.description["files"]["mjcf"]` is the station's MuJoCo model with the bases where the driver has them (on a real
  cell, where its calibration puts them), in the left base's frame (add y = +0.305 for the world).
- Feed library joint vectors in with their own name lists: `r.move(traj, times=t, joints=names)`. Planning (collision-free paths, constraints beyond `free=`) is the library's job; the robot executes its joint path smoothly and checks it against the joint limits.
- Get the current joints in any order with `r.state().q(names)`.
- Poses are 4x4. `robot.wxyz(T)` and `robot.xyzw(T)` give quaternions in either order. `robot.pose(xyz, rpy=...)` builds one.

## Writing and running a program

Any Python file that uses `robot` runs as usual (`python file.py`). The program runner, `python -m robot run`, runs a
file in a process of its own and adds pauses, a debugger and a record of each run. Such a file lists its steps,
functions `def name(r, v, s)`, in `STEPS`; importing it does no work. `r` is the robot, `v` the vision client (None
when no vision service is configured), and `s` a `SimpleNamespace` kept for the whole run, for what one step hands
the next. It can import modules from its own directory.

```python
def find_cube(r, v, s):
    s.cube = locate(r, "top")                   # a helper from a module next to it
    assert s.cube is not None, "no cube in the top camera"     # a failed assert pauses the run here

def grasp(r, v, s):
    s.g = grasp_at(r, "left", s.cube)

STEPS = [find_cube, grasp, lift]
```

`python -m robot run task.py` starts the program in a process of its own and returns at the first of: it pauses, it
exits, `--wait` seconds pass (default 60). It prints the program's output since the last report, then where and why it
paused, or its exit status and traceback. `python -m robot wait` waits again. `python -m robot pause` pauses the
running program at the next line it runs in your files (a robot request in progress finishes first), then waits as
`wait` does. `python -m robot stop` ends the program, running or paused (`--unattended` too); the arms hold.
Each run has a directory `.robot/runs/<UTC time>/` (`.robot/current` is the latest) with `log.txt` and `status.json`. While a run is alive,
`run` refuses to start another.

| Option | |
|---|---|
| `--break STEP` | pause at the start of that step |
| `--break FILE:LINE` | pause at that line (a pdb breakpoint) |
| `--pause-each-step` | pause after each step |
| `--from STEP`, `--only STEP` | start at that step; run only that step |
| `--unattended` | never pause: a failed `assert` raises `AssertionError`, and any exception ends the program with its traceback and a non-zero exit. For runs nobody watches. |

The program pauses at a failed `assert` in your own files (the program and the modules it imports from its
directory), in that frame and before anything is raised; when any other exception leaves a step (post-mortem); at
`--break`, `--pause-each-step` and `breakpoint()`; and at `python -m robot pause`. Each pause saves `pause-N/` in the
run directory: every camera's rgb (`NAME.png`) and depth (`NAME_depth.png`, 16-bit, mm), their K and T
(`cameras.json`), `r.state()` (`state.json`) and the message. The stops after `n`, `s`, `unt` and `r` save `r.state()`
and the message only.

`python -m robot debug 'CMD'` sends one pdb command to the paused program and prints its output: `p EXPR`, `!STMT`,
`where`, `up`, `down`, `l`, `n`, `s`, `b FILE:LINE`, and the rest of pdb. `r`, `v` and `s` are defined in every
command, whichever frame is selected (after that frame's own names). A command that resumes the program then waits
as `wait` does.

| Command | |
|---|---|
| `c` | go on with the next line. After a failed assert that is the line after it, run as if the check had passed: values computed before your fix stay as they were |
| `reload` | reread your files and `STEPS`; helpers called from then on run the new code, also after `c`. The paused function itself goes on as it was |
| `retry` | reload, then run the current step again from its start; `s` is kept |
| `skip` | go on with the next step |
| `goto STEP` | reload, then go on from that step, also one added since the run started |
| `q` | end the program |

After an exception the step cannot go on: `c`, `n` and `s` are refused; use `retry`, `skip`, `goto` or `q`.

While the program is paused its code stops. Commands it already sent run to their end in the driver, then the arms
hold. In lockstep simulation the world stops too, since nothing steps it; on the real robot and in a real-time
simulation it goes on: a ball in flight or a box on a conveyor has moved on by the time anyone looks, while a
grasp that missed or a key half inserted waits.

What a rerun keeps: `q`, a crash and a driver restart lose `s`, while the world keeps what was done and the files
written stay. When a program ends, crashes or is interrupted, the driver stops the motions it started and the robot
holds.
