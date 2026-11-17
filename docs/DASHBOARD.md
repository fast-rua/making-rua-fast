# The operator's dashboard

A web page that runs on the robot's computer, next to `robotd`. It shows what the robot is doing and holds the
operator's stop.

```bash
python -m robotd.server --real yam --addr unix:/tmp/robot.sock ... &      # or: --sim yam --scene pick_cube --realtime
python -m robotd.dashboard --addr unix:/tmp/robot.sock                   # then open http://localhost:8780 (--programs DIR)
```

Open the page in a browser on the robot computer's own screen. No port forwarding is needed. It listens on
localhost only.

**To watch the real robot without powering it,** start the driver with `--monitor`:
`python -m robotd.server --real yam --monitor`. The motors stay off. The driver polls each motor with DM's DISABLE
frame, which a motor answers with its state without enabling, so the page shows real joint angles, temperatures and
cameras while you move nothing, and every motion request is refused.
- **It still takes the cell lock** and still sends frames on the buses, so it is only for when no other program is
  driving the arms.
- **It needs the gripper limits:** `YAM_GRIPPER_LIMITS` or the calibration.

## Layout

- **Main view:** the robot model in 3D, a translucent blue "ghost" at the commanded joints (where the arms are heading),
  the TCP frames, and the point clouds of every camera in the world frame. Drag to orbit, scroll to zoom. Presets:
  Iso, Front, Side, Top. Each camera's cloud can be toggled, and the cloud can be coloured by height.
- **Camera strip:** each camera's colour image. Click a tile to enlarge it.
- **Bottom: the arms.** For each arm:
  - a side view in the plane joint 1 turns it into, with each joint coloured by motor temperature;
  - every joint's angle;
  - its position in its range (white) and the commanded position (blue);
  - its velocity and effort (share of the motor limit);
  - the MOSFET and rotor temperatures (amber from 45 °C, red from 50 °C: a DM4310 latched over-temperature at a 53 °C rotor on the cell).

  - the tracking error (commanded minus measured);
  - each motor's state, as reported in every DM reply: off, on, FAULT (DM's fault codes 0x8-0xE), or 0x3 (an
    undocumented status that disabled DM4310s report). It also shows as a ring on the side view, like the motor's LED.

  **Hover** a joint (in the table or on the side view) for a card with everything about it: model and CAN id,
  position in degrees and radians, commanded and sent-to-motor positions, tracking error against its guard, range and
  distance to the nearest limit, velocity and its cap, measured effort, model feed-forward and estimated external
  torque, gains, temperatures, the command that owns it. **Click** it to open it as a tab on the right, beside the log,
  with the same details live and 30-second charts of position against commanded, velocity, effort against the guard,
  and temperatures.

  Beside the arms: the driver, the clock, the control-loop rate, what is running, the TCPs, the force estimates, the
  grippers, i2rt's motor loops and the CAN buses (on the real robot).
- **Right: tabs.** The log, and any joints you opened: The driver's events (commands, results, faults, stops, refused commands, clients connecting)
  and the dashboard's own actions with the time the driver took to acknowledge them.
- **Bottom bar: the stop.**

## Programs

The right panel's **Programs** tab runs, edits and stops the programs that drive the robot.

- **Your programs** live in `--programs` (default `~/robot-programs`).
  - **New** makes one from a template.
  - **Save** (Ctrl/Cmd-S) writes it and reports a syntax error, if any.
  - **Delete** takes two clicks.
- **The repository's examples** are listed read-only. **Copy to mine** makes an editable copy.
- **▶ Run** starts the selected program with the arguments in the box, as its own process, with `ROBOT_ADDR` set to
  the driver.
  - Only one runs at a time.
  - Its output streams into the console, with stderr in red.
  - The top bar shows what is running and for how long.
- **■ Stop** sends Ctrl-C (the program's `finally` blocks run), then SIGTERM, then SIGKILL.
  - However a program ends, the driver stops the motions it started and the arms hold.
  - **FREEZE, SOFT and OFF also stop the running program**, so a program that loops on errors cannot keep trying
    while the robot is stopped.
- **⇔** widens the panel for editing.
- While you type in the editor or the arguments box, Space and S are text, not FREEZE and SOFT. The buttons always
  work.
- `--no-programs` turns the tab off, for a viewer only.

Programs are code that runs as you on the robot's computer. Keep the dashboard on localhost, or behind `--token`.

## The stop

| Button | Key | What it does |
|---|---|---|
| **FREEZE** | Space | Aborts every command and holds both arms stiff where they are. |
| **SOFT** | S | Aborts every command and drops the arm stiffness to zero. Damping stays on top of the gravity compensation, so the arms float and yield to a push. The grippers stop squeezing. |
| **PARK** | click twice | Moves both arms to the park pose at 0.3 rad/s: the calibration's rest pose, else where they started. Works after FREEZE or SOFT too (it clears the stop and parks); programs stay refused until RELEASE. |
| **RELEASE** | click twice | Clears the stop. The arms hold stiff where they are, and programs may command them again. |

- **FREEZE and SOFT latch.** Until the operator releases, every command is refused with `Fault`. A program's
  `r.recover()` cannot clear an operator stop, so a policy that loops on errors cannot override you. Refused attempts
  appear in the log.
- **Which one to press:**
  - FREEZE, when the arm is about to hit something.
  - SOFT, when it is already pressing on something or someone. Push it free, or PARK.
- **Neither is limp.** i2rt can also command nothing at all (`enter_zero_torque_idle`), but on this arm that only lets
  it fall.
- **FREEZE and SOFT act on the down-press** of the button, not on its release.
- **Soft damping** is `SOFT_KD` in `robotd/robots/yam_station/scene.py`: 2.0 on joints 1-3 and 0.5 on 4-6 (Nm·s/rad).
  That is more than i2rt's grav-comp idle uses, so a gravity-model error drifts slowly.

**What the page cannot do:**
- **The stop goes through `robotd`.** If `robotd` itself dies or hangs, nothing in software holds the arms. The YAM's
  motors disable themselves about 0.4 s after the last command, and the arms drop. The page shows NO DRIVER.
- **The power switch is the last resort.** It drops the arms too. Keep the work low over the table and the space
  under the arms clear.

## Running

```bash
python -m robotd.dashboard --addr unix:/tmp/robot.sock --port 8780 --camera-hz 3 --stride 4 --max-points 25000 \
       --crop=-0.5,1.2,-1.0,1.0,-0.15,1.2       # point cloud work volume: x0,x1,y0,y1,z0,z1 (m, world)
```

- **Process:** the dashboard is a separate process and an ordinary client of the driver, so it can never delay a
  control tick. It opens three connections: state at 20 Hz, cameras at `--camera-hz`, and the operator's requests.
  If the driver restarts, the page reconnects by itself.
- **Cameras** come through `robotd` (its `cameras` request), so the dashboard never opens a camera itself. On the real
  YAM, start `robotd` with `YAM_CAMERAS=top=<serial>,left_wrist=<serial>,right_wrist=<serial>`. Without it, the camera
  tiles say why.
- **Simulation:** start `robotd` with `--realtime`. The default lockstep simulation only moves while a program blocks
  in a call, so the page would look frozen.
- **The 3D model** is the CAD station from the description's MJCF. On a real cell the driver also applies the
  calibration's base corrections, so the drawn arms can sit a few millimetres off the point cloud.
- **three.js** is loaded from jsDelivr, so the 3D view needs internet. Everything else, the stop included, works
  without it.
- **From another machine:**
  - `ssh -N -L 8780:localhost:8780 <robot computer>`, then open `http://localhost:8780`.
  - Or listen on the network with a token: `--host 0.0.0.0 --token <secret>`, then open `/?token=<secret>` once.
    The robot computer may have a public address, and anyone who reaches the page can press the buttons.
- **Request checks:** requests that act on the robot need the `X-Dashboard` header. A page from another site cannot
  send it without a CORS preflight, which the server never grants. Requests with an unknown `Host` are refused, which
  stops DNS rebinding.

## Driver requests behind it

These are not part of the agent's API:
- `operator_stop` (`soft=` true/false), `release`, `park`;
- `health`: the backend's motor temperatures, motor loops and buses, the control-loop tick count, connected clients;
- `events` (`since=`): the driver's event log.
