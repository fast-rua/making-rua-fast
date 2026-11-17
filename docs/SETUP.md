# Setup: the simulator, the real YAM cell, and agent episodes

This guide takes you from a fresh checkout to a Codex agent driving the arms: first on a simulated YAM station (no
hardware), then on a real bimanual YAM cell. The README has the short version.

## What you need

| | Simulator | Real cell |
|---|---|---|
| OS | Linux | Linux |
| Python | 3.11, installed by `scripts/install.sh` through [uv](https://docs.astral.sh/uv/) | same |
| Robot | none | two YAM arms on CAN buses (driven by [i2rt](https://github.com/i2rt-robotics/i2rt) v1.4.0) |
| Cameras | rendered | four Intel RealSense cameras: `top`, `front`, `left_wrist`, `right_wrist` |
| Agent | the Codex CLI (0.165.0, installed by the script) and an OpenAI account with access to the model | same |
| Sandbox | [bubblewrap](https://github.com/containers/bubblewrap): the system's `bwrap`, else the copy that comes with Codex (it needs unprivileged user namespaces) | `/usr/bin/bwrap` (where AppArmor restricts user namespaces, as recent Ubuntu does by default, only the system's copy may create them) |

The demo runs used GPT-6 Astra in Codex, at medium reasoning effort, on the "ultrafast" service tier. Which models
and tiers you can use depends on your account (`codex` lists them).

## 1. Install

Everything goes into one folder, `$ROBOT_HOME` (default `~/robot-cell`). Nothing is installed system-wide and nothing
needs sudo; removing it means deleting the folder.

```bash
git clone https://github.com/fast-rua/making-rua-fast.git && cd making-rua-fast
bash scripts/install.sh            # simulator + robot API + Codex
bash scripts/install.sh --cell     # on the cell's computer: + the RealSense driver
source ~/robot-cell/env.sh         # in every shell you use
```

The script makes:

| Path | What |
|---|---|
| `$ROBOT_HOME/.venv` | Python 3.11 with numpy, scipy, MuJoCo, Ruckig, OpenCV, Pillow, mink, i2rt with its CAN driver, and (with `--cell`) pyrealsense2. The agent's sandbox runs this same Python |
| `$ROBOT_HOME/i2rt` | i2rt v1.4.0: the YAM model the simulator and the kinematics use, and the motor driver |
| `$ROBOT_HOME/codex/current` | the Codex CLI the episodes run; the script prints its SHA-256 to compare with the release page's |
| `$ROBOT_HOME/env.sh` | the environment (from `configs/real_yam/env.sh.example`); the real cell's values are commented out until you fill them in |

Episodes write to `$ROBOT_HOME/agent-runs/`, and robotd's recordings go wherever `--record` points (we use
`$ROBOT_HOME/runs/`).

Check the install with the tests (a few minutes; no hardware):

```bash
python -m pytest -q
```

## 2. Try it in the simulator

The simulator is a MuJoCo model of the YAM station with a few tasks (`tasks/yam_station.py`): `pick_cube`,
`two_cubes`, `flatten_towel`, `fold_towel`, `fold_flat_towel`. The driver, the API and the dashboard are the same
ones the real cell uses.

**A scripted policy:**

```bash
python scripts/run_sim.py --scene pick_cube --policy examples/pick_cube.py --record runs/pick-1
```

**Watch it:** run the driver yourself, open the dashboard, and run the policy against it.

```bash
python -m robotd.server --sim yam --scene pick_cube --realtime --addr unix:/tmp/robot.sock &
python -m robotd.dashboard --addr unix:/tmp/robot.sock &        # http://localhost:8780
ROBOT_ADDR=unix:/tmp/robot.sock python examples/pick_cube.py
```

**An agent episode.** Log Codex in once, for the episodes only:

```bash
CODEX_HOME=$ROBOT_HOME/codex-home $ROBOT_HOME/codex/current/bin/codex login --device-auth
```

Then run an episode on a simulated station. The launcher starts its own robotd, builds the sandbox, runs Codex in it
and records everything:

```bash
python scripts/agent_run_real.py --sim pick_cube --tag sim-pick-01 --pipelined --effort medium
python scripts/agent_monitor.py &                                 # http://localhost:8790: watch it live
```

`--sim fold_towel --seed 0` gives the agent a crumpled towel instead. Cloth is slow to simulate (about 5x slower than
real time on one core). The simulated cameras render headless through EGL (`MUJOCO_GL=egl` in `env.sh`); on a
machine with several NVIDIA GPUs, `--gpu N` picks the one EGL renders on.

`--print agents-md` prints what the agent would be told and exits, and `--agent standin` runs a script in place of
the model that probes the sandbox and sends no motion: both are good first checks.

## 3. The real cell

### Rules of the cell

- **Coordinate first.** Only one driver can run at a time: robotd takes a lock file (`/tmp/ora-cell.lock`, or
  `$ORA_CELL_LOCK`) and refuses to start while anyone holds it. `cat` it to see who.
- **Watch powered motors.** Whenever robotd runs without `--monitor`, the motors are powered. Someone watches the
  arms with the dashboard open, a hand near the arm power switch, and knows the dashboard's buttons:
  - **FREEZE** (Space): stop and hold stiff.
  - **SOFT** (S): stop and go limp, for an arm already pressing on something.
  - **PARK** (click twice): move to the rest pose. It also works after FREEZE or SOFT.
  - **RELEASE** (click twice): let programs move the arms again.
- **Stop robotd with Ctrl-C** (or `kill`): it parks the arms, then exits. **Never `kill -9`**: the motors time out
  0.4 s later and the arms drop.
- **Don't pipe robotd's output** through `tee` or `grep`: Ctrl-C would kill the pipe before robotd.
- **The arms' force estimate is not reliable on the YAM.** Only the grippers' force is. Programs should use positions,
  cameras and the grippers' `stalled` result for contact.

### The pieces

| Process | Port (localhost) | What it does |
|---|---|---|
| `robotd` (`python -m robotd.server --real yam`) | 7410 | The driver: a 250 Hz control loop on the CAN buses. It owns the motors and enforces the safety limits (below) |
| `camerad` (`python -m robotd.camerad`) | 7412 | The four cameras, in their own process so camera traffic never stalls the control loop |
| Dashboard (`python -m robotd.dashboard`) | 8780 | The operator's page: FREEZE, SOFT, PARK, RELEASE, the cameras, a 3D view, the driver's log |
| Agent monitor (`scripts/agent_monitor.py`) | 8790 | An episode, live: the agent's messages, its files and diffs, its programs, the witness frames, a box to steer it |
| Episode launcher (`scripts/agent_run_real.py`) | — | One Codex episode in a sandbox against robotd and camerad; then it parks, checks and logs |

### Configure `env.sh`

Uncomment and fill in the real-cell block of `$ROBOT_HOME/env.sh`:

| Variable | What |
|---|---|
| `YAM_CHANNELS` | the CAN bus of each arm, `left=can_left,right=can_right`. Set the buses up with i2rt's tools; robotd never configures them |
| `YAM_CAMERAS` | the cameras by RealSense serial, `top=...,front=...,left_wrist=...,right_wrist=...` (`rs-enumerate-devices -s` lists them) |
| `YAM_CALIBRATION` | the calibration file (next section) |
| `YAM_GRIPPER_LIMITS` | optional: each gripper's motor reading `closed,open`, overriding the calibration's |
| `YAM_PARK_Q` | optional: the rest pose (6 joint angles per arm), overriding the calibration's |
| `YAM_GRAVITY_FACTORS` | optional: i2rt's gravity compensation scaled per joint. The demos used `1.0,1.2,1.1,0.5,1.0,1.0`, measured on our arms; start from `1.0` everywhere on yours |
| `YAM_CALIBRATION_IGNORE` | optional: calibration keys to leave out. The demos left out `wrist_camera_corrections` |

Per-arm values are `a,b,...` for both arms or `left=a,b,...;right=a,b,...`.

### Calibration

robotd and camerad read one JSON file (`YAM_CALIBRATION`; they never write it). Every key except `schema` is
optional; the format is the one [ORA](https://github.com/hao-ai-lab/ORA)'s cell tools write:

```json
{
  "schema": "ora.cell-calibration.v1",
  "joint_zero_offsets": {"left": [0, 0, 0, 0, 0, 0], "right": [0, 0, 0, 0, 0, 0]},
  "gripper_limits": {"left": [6.38, 1.11], "right": [6.40, 0.98]},
  "rest_reference": {"left": [0, 0, 0, 0, 0, 0], "right": [0, 0, 0, 0, 0, 0]},
  "arm_base_corrections": {"left": {"rotvec_rad": [0, 0, 0], "translation_m": [0, 0, 0]}},
  "cameras": {"top": {"cam2world_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
              "front": {"cam2world_cv": "..."}},
  "wrist_camera_corrections": {"left": {"rotvec_rad": [0, 0, 0], "translation_m": [0, 0, 0]}}
}
```

| Key | Meaning |
|---|---|
| `joint_zero_offsets` | per arm, each joint's encoder zero error in rad: model angle = reading + offset |
| `gripper_limits` | per arm, the gripper motor's reading `[closed, open]` in rad. Without them, i2rt calibrates each gripper at start-up by driving the jaws to both stops |
| `rest_reference` | per arm, the rest pose (encoder readings): where the arms park, and where robotd leaves them when it exits. It must be a pose the arms can rest in unpowered |
| `arm_base_corrections` | per arm, a correction to the base's pose in the world frame (corrected = correction @ model) |
| `cameras.<name>.cam2world_cv` | each fixed camera's pose: world from optical frame (OpenCV: +z forward, +y down), 4x4. `top` also matches a role named `overhead`. A fixed camera without one is refused rather than guessed |
| `wrist_camera_corrections` | per arm, a correction to the wrist camera's mount on top of the model's |

The world frame has its origin on the table top midway between the arm bases, +x forward, +z up.

### Describe your cell to the agent

The agent is told what the cell looks like from `docs/REAL_CELL.md` (copied into its instructions before the
manual): where the cameras are, the table, what the grippers can and can't hold. It describes our cell. Edit it to
describe yours; keep the facts measurable.

### Bring-up, with the motors off

```bash
python -m robotd.server --real yam --monitor --addr tcp:127.0.0.1:7410     # reads joints and temperatures; refuses motion
python -m robotd.dashboard --addr tcp:127.0.0.1:7410 --port 8780           # in another shell
```

The arms at rest should read near their rest pose and both grippers their real opening. Then check the install and
the calibration file against i2rt's simulated robot behind the same backend:

```bash
YAM_SIM=1 python -m robotd.server --real yam --addr tcp:127.0.0.1:7410 &
ROBOT_ADDR=tcp:127.0.0.1:7410 python examples/basic_motions.py
kill %1                                            # robotd parks, then exits
```

Every line should end with `done: reached`.

### First real motions

```bash
python -m robotd.server --real yam --addr tcp:127.0.0.1:7410 --speed-scale 0.5 \
    --fence configs/real_yam/fence-gentle.json --record $ROBOT_HOME/runs/first &
ROBOT_ADDR=tcp:127.0.0.1:7410 python examples/basic_motions.py
```

`basic_motions.py` makes small, slow moves around the start pose: a 0.1 rad joint wiggle, 3 cm up and down, the
gripper half open and then open. Watch the first one with a hand on the power switch.

### Each session

One shell per process (tmux works well), each with `env.sh` sourced, in the repository:

```bash
python -u -m robotd.camerad --addr tcp:127.0.0.1:7412 --robot tcp:127.0.0.1:7410
```

```bash
python -u -m robotd.server --real yam --speed-scale 0.5 --separation 0.02 --separation-fine 0.005 \
    --cameras tcp:127.0.0.1:7412 --fence configs/real_yam/fence-cloth.json \
    --record $ROBOT_HOME/runs/<name> --addr tcp:127.0.0.1:7410
```

```bash
python -u -m robotd.dashboard --addr tcp:127.0.0.1:7410 --port 8780 --no-programs
```

```bash
python -u scripts/agent_monitor.py --port 8790
```

The robotd line is the one the demos ran with. On startup robotd prints the limits it enforces; read them.

- `--record DIR`: every request and every 4 ms tick, across all the episodes this robotd serves. Use a new name for
  each start.
- `--speed-scale 0.5` halves the default joint speeds and limits (the default with `--real`).
- The fences in `configs/real_yam/`: `fence-cloth.json` (what the demos used: near-full reach, x ≤ 0.70 m,
  z ≤ 0.65 m, 1.0 m/s, a 0.8 mm gripper floor) and `fence-gentle.json` (x ≤ 0.45 m, z ≤ 0.35 m, 0.3 m/s, for first
  runs and people nearby). Measure your table and set the boxes from that.
- The dashboard can also run your own programs from a folder (`--programs DIR`); `--no-programs` turns that off,
  which is what you want while an agent runs (see the README's security notes).

**Watching from another computer:** forward the pages over SSH and open `http://localhost:8780` and
`http://localhost:8790`:

```bash
ssh -N -L 8780:127.0.0.1:8780 -L 8790:127.0.0.1:8790 <you>@<cell>
```

**Shutting down:** PARK on the dashboard, Ctrl-C robotd (it parks, then exits), then Ctrl-C the rest.

## 4. Agent episodes on the real cell

Log Codex in for the episodes once (as in section 2), then:

```bash
python scripts/agent_run_real.py --task fold_towel --towel 30x22:blue --pipelined --effort medium \
    --release-park --tag towel-01
```

**What happens:**
1. **Checks.** The launcher refuses to start if robotd has a fault, something is moving the arms, or the operator
   parked them. `--release-park` releases an operator park first.
2. **The agent's world.** Codex gets `AGENTS.md` (the task, the cell's description, the rules and the robot manual,
   `docs/MANUAL.md`) and runs in a bubblewrap sandbox that sees its workspace, a read-only copy of the robot client
   code (at `/opt/robot-api`), the camera recordings and two sockets to robotd and camerad. Its only network is the
   model API, through an allow-listed proxy.
3. **Acting.** The agent writes Python against `import robot` and runs it in one long-lived Python kernel
   (`python -m robot.kernel`). Each action's end report, with the cameras and the arms' state, arrives in its running
   turn. With `--pipelined` it queues actions (`kernel.queue(fn)`) that the robot runs in order while the model plans
   the next ones.
4. **Recording.** All four cameras, a witness frame from the top and front cameras every 10 s, a 4x timelapse, and
   every robot command (robotd's recording).
5. **The end.** After the agent's last turn and `--linger-s` (60 s) with no operator message, the launcher parks the
   arms (latched: the next episode needs `--release-park`), checks the scene from the cameras, and writes the log.

**Steering:** type in the agent monitor's box. The message goes into the agent's running turn, or starts a new one.

**Stopping:** Ctrl-C the launcher (it stops the agent, then parks), or FREEZE, SOFT or PARK on the dashboard.

**Tasks** (`tasks/real_yam.py`): `fold_towel`, `fold_flat_towel`, `unfold_towel` (the towel tasks measure the towel
from above before and after; give its size and colour with `--towel WxH:COLOUR`), `dump_ball`, and `find_pear` (the
cup demo). `--instruction "..."` replaces a task's text. To add a task, add an entry to `TASKS`: the instruction, a
function that measures the scene from the cameras, and a check that reads those measurements. The check is a first
read; the operator's verdict, from the frames and the video, is the one that counts.

**Launcher options worth knowing:**

| Option | Meaning |
|---|---|
| `--pipelined` | pipelining, as in the demos: the agent queues its next action while the running one moves. Without it, it starts one action at a time and writes the next after the report, so the arms wait while the model thinks |
| `--effort low/medium/high/xhigh` | reasoning effort. Low and medium are the practical ones on a robot that keeps moving |
| `--service-tier TIER` | the model's service tier, if your account has more than one; `--fast` for "fast" |
| `--llm-model MODEL` | another model than GPT-6 Astra |
| `--codex-home DIR` | whose Codex login (`DIR/auth.json`); default `$ROBOT_HOME/codex-home` |
| `--wall-h H` | the agent's time limit, in hours |
| `--release-park`, `--keep-park` | release an operator park before starting, or keep it (for no-motion tests) |
| `--agent standin` | a scripted probe of the sandbox: reads the robot and cameras, sends no motion |
| `--print agents-md` | print what the agent would be told, and exit |
| `--exec` | `codex exec` instead of the app-server (no steering) |

## 5. Reproducing the demos

The three runs in the blog post ran with this robotd (section 3, "Each session"), `env.sh` with
`YAM_GRAVITY_FACTORS=1.0,1.2,1.1,0.5,1.0,1.0` and `YAM_CALIBRATION_IGNORE=wrist_camera_corrections` (camerad too),
Codex 0.165.0 and GPT-6 Astra. The towel instructions are in `demos/`, word for word:

```bash
# three cups hide a pear, a ball and nothing; put the pear under the empty cup (cups-02)
python scripts/agent_run_real.py --task find_pear --pipelined --effort medium --service-tier ultrafast \
    --wall-h 0.25 --release-park --tag cups-01

# spread a partly folded towel and fold it in half (towel-fold-18)
python scripts/agent_run_real.py --task fold_towel --instruction "$(cat demos/towel-spread-fold.txt)" \
    --pipelined --effort medium --service-tier ultrafast --wall-h 0.25 --release-park --tag towel-01

# the same with a suggested plan: smooth it first, then scoop the corners (towel-fold-23)
python scripts/agent_run_real.py --task fold_towel --instruction "$(cat demos/towel-smooth-fold.txt)" \
    --pipelined --effort medium --service-tier ultrafast --wall-h 0.25 --release-park --tag towel-02
```

The cup run in the post was launched under the `dump_ball` task with this text as `--instruction`; `find_pear`
gives the same instruction without `dump_ball`'s camera check, which doesn't apply to cups.

## 6. After a run

Each episode writes `$ROBOT_HOME/agent-runs/<tag>/`. Start with `TIMELINE.md`: the agent's messages and every command
it ran, the programs and their results, and every robot command, in one time order. `docs/RUNS.md` describes every
file.

- `scripts/run_log.py RUN` rebuilds the timeline (the launcher runs it at the end).
- `scripts/profile_run.py RUN` says where the time went: model, tools, motion, and how much of the model's time was
  hidden behind motion.
- `scripts/export_run.py` makes a compact copy (the session record with the images the model saw, the timeline, the
  code, the commands and 25 Hz joint data).
- `scripts/agent_debrief.py TAG` talks with an ended episode's agent: a fork of its Codex conversation, in its sandbox
  without the robot, with the run record read-only at `/episode`. Type in the terminal or in the agent monitor's box.

## 7. The limits robotd enforces on the real arms

| Limit | What it does | Set by |
|---|---|---|
| Fence box | each arm's TCP stays in its box: a target outside is refused, a motion that would leave it stops at the wall, and a TCP measured more than 2 cm out faults | `--fence` |
| Planar speed | the TCP's speed in the table plane is capped; moving faster than 1.5x the cap faults | the fence's `max_speed_xy` |
| Gripper floor | the whole gripper (fingers, body, wrist camera) stays above this height, since a tilted gripper reaches below its TCP | the fence's `gripper_floor` |
| Separation | the arms stay apart, measured as capsules around their links (and the hands' meshes with `--separation-fine`) | `--separation` (0.02 m with `--real`) |
| Gains cap | `gains=` stiffer than the defaults is refused | `--max-gain-scale` (1 with `--real`) |
| Joint speed | default joint speeds 1.5 / 2.0 rad/s, hard caps 3.0 / 4.0 rad/s at speed scale 1 | `--speed-scale` |
| Gripper overload | a gripper pushing over 1.5x its force cap for 0.3 s is relaxed until its next command | always |
| Operator stop and park | they latch: programs are refused until RELEASE | the dashboard |

The fence is a software limit inside the driver, checked every 4 ms against the encoders' forward kinematics. It does
not see people and does not replace the power switch. Parking (PARK, robotd's exit) ignores the fence, so check that
the path to the rest pose is clear.

## 8. What robotd does on the real arm

- **The 250 Hz loop** sends `command_joint_state(pos, vel, kp, kd)` to each arm. i2rt's motors run the PD, and i2rt
  adds its gravity compensation (scaled by `YAM_GRAVITY_FACTORS`).
- **Joint angles are model angles:** encoder reading + the calibration's zero offset, both ways.
- **The gripper is never held at its stops:** commands stay within 3–97% of its range. `gripper(force=N)` sets
  i2rt's gripper force limiter (default 50 N).
- **Guards:** commanded joints stay within 0.15 rad of the measured ones, joint speeds are limited, and a joint above
  80% of its motor's torque limit for 40 ms latches a fault.
- **A joint that starts outside its model limits** (a zero offset at a resting stop) is not stepped in: commands stay
  between the limit and where it is until it is back inside.
- **Recording** (`--record DIR`): every request goes to `DIR/requests.jsonl` and every tick to `DIR/ticks/` (one file a
  minute, written by a thread of its own). `scripts/replay_requests.py` replays a run deterministically in the
  simulator, to debug the driver.
- **Cameras:** robotd started with `--cameras ADDR` opens no camera itself; its description carries camerad's address,
  and `r.cameras()` fetches from camerad directly. Through an SSH tunnel, forward port 7412 too, or set
  `ROBOT_CAMERA_ADDR`. camerad reads the joints from robotd at each grab, for the wrist cameras' poses.

## 9. Known issues

- **Thin cloth slips.** The grippers close only to about 3 mm between the pads (robotd keeps them off the closed
  stop so the motor doesn't overheat), so a single towel layer slips. Grab folds, the doubled hem, or a ridge pushed
  up first.
- **Low force caps on rigid objects** can trip the gripper overload guard. Grips at 18–35 N held cups and bowls fine.
- **Reach depends on the gripper's angle.** Pointing straight down a TCP can't rise above about z = 0.22 m; tilted
  35–60° it reaches about 0.48 m, sideways about 0.7 m.
- **Cameras run in camerad, not robotd.** Grabbing and aligning RGB-D frames holds Python's interpreter for 15–28 ms
  at a time, which stalls the 250 Hz loop inside robotd.
- **"NO DRIVER" on the dashboard** means robotd is gone: nothing holds the arms, and they drop about 0.4 s later.
- **Service tiers** depend on the account: `--service-tier` must name one your account offers.
