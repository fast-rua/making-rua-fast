# Architecture

This is a robot harness in which a coding agent writes and runs Python against one small, joint-level robot API.
Every robot (here a simulated YAM station and the real bimanual YAM cell) sits behind the same API, through a
driver process. The agent uses standard robotics libraries (cuRobo, Ruckig, Pinocchio, PyRoKi, Open3D) directly,
and nothing in between wraps them.

## Processes

```
 agent sandbox (any number of agent processes)          driver: robotd (one per robot)
 ┌──────────────────────────────────────────┐   RPC    ┌───────────────────────────────────────────────┐
 │ policy.py                                 │ (JSON over│ rpc server ── command scheduler ── control core │
 │   import robot; r = robot.connect()       │  unix or  │                 (per joint group)  (fast tick)  │
 │   r.move / move_at / gripper / ik         │   TCP)    │   references: goal (Ruckig), path (spline),     │
 │   when.*, Compliance, builders            │◄─────────►│   MoveAt (velocity), Gripper                    │
 │   cuRobo / Ruckig / Pinocchio / Open3D    │           │   law: joint PD gains + admittance compliance   │
 │   used directly on numpy state            │           │   conditions: until / on / then, guards         │
 └──────────────────────────────────────────┘           │   estimator: wrench from joint efforts          │
                                                        │   recorder: requests, ticks, frames             │
                                                        │                     │ backend protocol           │
                                                        └─────────────────────┼─────────────────────────────┘
                                         ┌────────────────────────────────────┼───────────────────────────┐
                                         │ MujocoBackend (YAM station scene)  │ YamI2rtBackend (real YAM) │
                                         │ torque actuators; PD + gravity in  │ i2rt command_joint_state  │
                                         │ the backend; lockstep physics      │ (pos, vel, kp, kd); 250 Hz│
                                         └────────────────────────────────────┴───────────────────────────┘
```

- **Client (`robot`):** a small pure-Python package with no simulator imports. Everything it sends is plain JSON. The
  numpy, torch and JAX arrays it accepts are converted to lists, and it always returns numpy float64.
  - **The program runner** (`python -m robot run task.py`; `robot/__main__.py`, `robot/runner.py`) runs a program's
    steps in a process of its own and pauses it at failed asserts (rewritten at import, as pytest does), exceptions
    and breaks. At a pause it saves every camera's frames and `r.state()` and serves pdb over a unix socket, one
    command per `python -m robot debug` call. MANUAL.md, "Writing and running a program", describes it.
- **Driver (`robotd`):** owns the robot. It runs a fixed-rate **fast tick** (4 ms, i.e. 250 Hz, the YAM's CAN rate),
  in which references are interpolated, the control law applied, conditions evaluated, guards enforced, and
  everything recorded.
  - **Simulation is lockstep:** physics advances only while a blocking call runs or on `step()`. The world is paused
    whenever control is with the agent. Requests from several clients are served one at a time.
  - **Real robots run the fast tick continuously** in a thread. Each client has its own thread, so a `stop()` from a
    second terminal takes effect while another client is blocked in a motion; IK solves and camera grabs run outside
    the control lock.
  - **A real-time simulation** (`--realtime`) runs the same way: the control thread ticks the physics and sleeps to the
    next 4 ms period (late ticks are counted in `timing.json`). Its clock starts when the first program connects, so
    a scene's timed events stay where lockstep puts them. Camera scenes are posed under the physics lock and rendered
    outside it; a render still holds Python's GIL for parts of its ~11 ms, so ticks run up to ~10 ms late around
    frames and then catch up (the simulated world stays exact; its pacing against the wall clock jitters).
  - **A client that disconnects** (its program ended, crashed or was interrupted) has its commands stopped; the arms
    hold where they are.
- **Backend:** the only robot-specific code. It reads the measured state, writes PD targets with gains, and renders or
  grabs cameras. The simulated one can also make the world imperfect, per scene: moving air (a steady draft plus
  Ornstein-Uhlenbeck gusts, through MuJoCo's wind) and objects that leave each contact with their speed and direction
  slightly changed (`Obj.bounce_noise`). Each process draws from its own stream seeded by the scene, in step order,
  so the same seed and the same commands replay the same run; `--scene-arg noise=0` (or `noise_seed=`) changes it.

## The API the agent sees

```python
r = robot.connect()                         # address from $ROBOT_ADDR
s = r.state()                               # q, dq, effort (by name), ee poses, wrench, deflection, contact, t
r.T("left"); r.cameras(); r.description     # measured TCP pose of an end effector, camera frames, the description

r.move(target, joints=|ee=, line=False, speed=, duration=, free=, **common)  # one target, no times: a goal
r.move(points, times=|dt=, joints=|ee=, relative=, velocities=, free=, ...)  # several, or timed: a path
                                                                             #   (velocities: joint rates, or the
                                                                             #   point's m/s at each pose; no times/
                                                                             #   dt: timed from the limits)
r.move(..., dry_run=True)                                                    # a Plan: where it would fail; no motion
r.move_at(velocity, joints=|ee=, frame=, max_travel=, duration=, **common)   # velocity / twist
r.gripper(name, width, force=, **common)                                     # force-limited; stall = success
r.step(ticks); r.step_until(t); r.stop(); r.zero_force(ee); r.finish(summary)  # one step tick = 40 ms
r.cameras(names, depth=True, roi=None)                                       # roi: a pixel box, K shifted
r.ik(target, ee, tcp=None, seed=None, free=None); r.reachable(targets, ee, free=)   # IK, margins; no motion

# common: compliance=Compliance(frame, stiffness={"z":300}, force={"-z":6}, max_force, max_deviation,
#         max_penetration, ramp), gains={...}, tcp=4x4, until=cond|{name: cond}, then=cmd|[cmds]|{reason: cmd}
#         (after arriving or a condition) or "hold"|("back_off",m) (after a condition),
#         on={name: (cond, cmd)}, start_at=t, replace=True, wait=True, timeout=s
#         (gains and a grip's force= hold for that command only; without them the defaults apply)
# conditions: when.force/torque/effort/deflection/travel/pose/speed/time/progress/contact/gripper/signal,
#             combined with & | ~, comparisons, abs, +, -, .held(s), when.after(a, b)
```

A fired condition is a normal `Result(status, reason, t, at=Snapshot)`. Exceptions are reserved for things the agent
didn't ask for: `Aborted` (a motion that could not go on; the arm holds), `Fault`, `InvalidCommand`, `NotSupported`,
`Busy`, all subclasses of `RobotError`.

## Control law (position-PD robots such as the YAM)

- **Joint PD with gravity compensation:** `tau = kp*(q_ref - q) + kd*(dq_ref - dq) + g(q)`. In simulation, MuJoCo applies
  this as motor torque; on the YAM, i2rt applies it in the motors (MIT mode) with its own gravity model.
- **Inertia feed-forward:** the torque that accelerates the arms along their reference, `M(q_ref) ddq_ref` (link and
  rotor inertia, from the description's model), goes through the same servo as a command offset `tau / kp`. The
  reference acceleration is low-passed and bounded at 40 rad/s^2. On a 2 Hz joint sine it cuts the soft servos' lag
  from 0.09 to 0.02 rad RMS. A held tool's mass and inertia are not in the model.
- **Cartesian references** become joint references every tick: the joints' current velocity predicts the step and one
  damped least-squares step corrects it (the damping only shaves the correction, so a moving reference neither trails
  nor dips when handed over); a step that would exceed a joint's velocity limit is scaled down as a whole, so the
  path keeps its direction and lags instead.
- **Replacing a command** continues from the current reference: its position, velocity and acceleration. A goal
  (Ruckig) takes all three. A timed path or a line starts where the reference is (a Cartesian one where the replaced
  Cartesian one was asking) and adds a blend, a short correction that starts at the velocity and acceleration the new
  path lacks and dies out within 0.25 s or the path's duration (never moving its end; a large missing acceleration is
  blended over a few ticks so the path bends by at most 2-3 mm or mrad). A velocity command ramps from the current
  velocity. Measured on a quick joint move replaced at 0.35 s: velocity steps under 0.1 rad/s per tick (2 rad/s
  before), acceleration steps under 12 rad/s^2 (up to 500 before); a servo loop of 80 ms paths re-sent every 32 ms
  has acceleration steps under 3 rad/s^2 (15 before).
- **Free axes** (`free=`) are solved when a command is sent: the candidate IK solutions (the masked solve, every 30
  deg about the tool axis, or the half turn) and the cheapest to reach, with penalties near joint limits and
  singularities; along a path, dynamic programming over those candidates per point (transitions that the joints
  cannot make in the given time are cut). The result carries the chosen pose(s).
- A command with `start_at` leaves the running one alone until then (one pending command per joint group: a second
  one replaces it; to chain two motions of an arm on the clock, fire the second from the first with `on=`).
- **IK** for commands and `ik` queries runs on a second copy of the kinematics (its own MjData, its own lock): a
  real-time driver solves it outside the control lock. It is Python, so it still takes the GIL from the control
  thread while it runs.
- **Compliance is an admittance outer loop in the fast tick:** on the requested Cartesian axes, the reference yields to
  the estimated external force like a spring of stiffness K, `x_off = clip(F_ext / K)`.
  - **Force targets are implicit:** the reference is pushed into the surface by `F_target * (c_servo + 1/K)`, where
    `c_servo = u^T J Kp^-1 J^T u` is the PD servo's own compliance along the axis (the two springs are in series;
    `F/K` alone gives only about 60% of the target on the YAM). A slow trim on the measured force corrects the model
    within +-30% while in contact. The push is bounded by `max_penetration` (default 0.03 m) and ramped.
  - **On contact loss** the arm travels at most `max_penetration` past the old surface. It cannot slam into the table.
  - **When a compliant command ends,** the reference takes over the commanded (pushed or yielded) position, so the
    end is not a step: a finished press keeps pressing until the next command moves on, and `then="hold"` releases.
- **Wrench estimate:** a generalized-momentum observer per arm (`robotd/core/estimator.py`) gives each joint's external
  torque, so the estimate holds while the arm moves: it integrates motor torque, Coriolis, gravity (the robot's own
  feed-forward when it reports one, as i2rt does) and joint friction against the momentum `M(q) qd`, never
  differentiating velocities, with a 25 ms lag. The model is MuJoCo's (CAD inertias) plus each backend's rotor inertia
  and friction (`backend.dynamics`; on the real YAM `YAM_DYNAMICS`, identified on the arm). The joint torques become a
  force at the TCP by a weighted least-squares fit over all six joints (joints with less friction weigh more), then
  the moment from what the force leaves; filtered, minus a per-arm bias that `zero_force` sets. In simulation, moving
  fast with nothing touching it reads at most 0.3 N (the static estimate, gravity minus measured, read 9 N), and a
  5-6 N push at the TCP is read to within 0.35 N while moving.

## Guards

- The commanded joint stays within 0.15 rad of the measured one (a blocked arm pushes with at most kp x 0.15).
- No reference moves a joint faster than `description["velocity_limit"]`; a joint path (`r.move(qs, joints=...)`) or
  joint velocity that would is refused with `InvalidCommand`, saying how much to stretch the times.
- An arm joint above 80% of its motor limit for 40 ms, or `Compliance(max_force=)` exceeded, latches a `Fault`; the
  arms hold at the measured position until `recover()`.
- A gripper above 1.5x its force limit for 0.3 s is jammed or blocked: its command is aborted and it is relaxed (no
  stiffness) until its next command. (On the cell a gripper jammed at its open stop was driven at ~7.6 Nm until its
  motor latched over-temperature.)
- Joint limits from the model.

## The operator's stop

The dashboard (`robotd.dashboard`, a separate process and an ordinary client) can latch an operator stop:
- **freeze** holds the arms stiff at the measured joints;
- **soft** sets the arm stiffness to zero, leaves damping (`soft_kd`) on top of the backend's gravity compensation, and
  makes every reference track the measured joints, so the arms float and yield to a push and releasing is
  step-free.

Either one aborts every command and refuses new ones until the operator releases it; a program's `recover()` cannot.
On the YAM nothing in software is limp-safe: without torque the arm falls, and if the driver dies the motors disable
themselves about 0.4 s later. See `docs/DASHBOARD.md`.

## Robot description

`r.description` points at standard files and names, which libraries load directly:
- a URDF of both arms in the world frame, written from the driver's own model, with an `<arm>_tcp` link and the
  driver's limits; per-arm URDFs under `arms[<arm>]` (with `base_link`, `tcp_link`, `joints`, `T_world_base`); i2rt's
  MJCF (`yam_station_linear_4310_d405`, in the left-base frame);
- joint groups (`left_arm`, `right_arm`, `left_gripper`, `right_gripper`);
- limits;
- TCP sites;
- cameras;
- default gains, default and limit velocities.

On a real cell the driver also reads the cell's calibration (ORA's `ora.cell-calibration.v1`): encoder zero offsets
and base corrections enter its kinematics, the fixed cameras' poses come from it, and the gripper limits and rest
pose configure the arms. The URDFs the agent loads are written from that calibrated model.

## Repository layout

```
robot/      client package (connect, when, Compliance, builders, geometry helpers, types), the program runner
            (python -m robot run / wait / debug) and the pipelining kernel (python -m robot.kernel)
robotd/     driver: server.py, camerad.py (the cameras, in their own process), core/ (scheduler, references,
            conditions, compliance, estimator, kinematics, recorder), backends/ (mujoco_sim, yam_i2rt, realsense),
            robots/yam_station/ (scene builder, description), dashboard/ (the operator's page)
tasks/      yam_station.py (simulated scenes with hidden success checks), real_yam.py (the real cell's tasks)
scripts/    agent_run_real.py (an agent episode), agent_monitor.py, agent_debrief.py, agent/ (the sandbox's
            pieces and AGENTS.md template), run_sim.py, run_log.py, profile_run.py, export_run.py, render_run.py, replay_requests.py,
            install.sh
examples/   scripted policies (pick_cube, basic_motions, gentle_close, gripper_yield, towel/)
configs/    real_yam/: the environment template and the workspace fences
demos/      the towel demos' instructions, word for word
tests/      unit tests (conditions, references, IK), integration tests in MuJoCo, RPC end to end, the operator's
            requests, the dashboard and the agent monitor
docs/       SETUP.md, ARCHITECTURE.md, MANUAL.md and AGENT_PLAYBOOK.md (agent-facing), REAL_CELL.md (agent-facing:
            the cell), DASHBOARD.md, RUNS.md
```
