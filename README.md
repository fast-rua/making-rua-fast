# Robot-use agents in real time

This is the code behind [Getting Robot-Use Agents to Run in Real Time](https://haoailab.com/blogs/realtime-rua/): a
coding agent (Codex) that drives a bimanual YAM cell by writing Python against a small robot API, and keeps the robot
moving while the model thinks about its next step (pipelining).

It has:

- **A robot API and driver.** `robotd` owns the arms and runs a 250 Hz control loop with the safety limits in it
  (workspace fences, arm separation, speed caps). Programs talk to it from any process with `import robot`: state,
  cameras with depth, moves, grippers, IK, reachability, conditions.
- **Pipelining.** A long-running Python kernel for the agent (`python -m robot.kernel`). The agent queues its next
  action while the current one runs (`kernel.queue(fn)`), and each action's end report, with the cameras and the arms'
  state, arrives in the agent's running turn.
- **The episode harness.** It runs Codex in a bubblewrap sandbox that can reach only the robot, the cameras and the
  model API, and records everything: the agent's turns, its code, every robot command, all cameras.
- **Two web pages.** An operator's dashboard (stop, park, cameras, a 3D view) and an agent monitor (the agent's
  messages, code and programs live, and a box to steer it).
- **A simulated YAM station** (MuJoCo) with a few tasks, behind the same driver, to try everything without hardware.
- **The real YAM backend** (through [i2rt](https://github.com/i2rt-robotics/i2rt)) and a camera server for four
  RealSense cameras.

It is research code, run on one cell. It is meant to be read, run and changed, not installed as a library.

## Quick start: the simulator

You need Linux, `git` and [uv](https://docs.astral.sh/uv/). Everything installs into one folder,
`$ROBOT_HOME` (default `~/robot-cell`), with no sudo:

```bash
git clone https://github.com/shaoxiongduan/making-rua-fast.git && cd making-rua-fast
bash scripts/install.sh
source ~/robot-cell/env.sh
python -m pytest -q                       # a few minutes, no hardware
```

Run a scripted policy on the simulated station:

```bash
python scripts/run_sim.py --scene pick_cube --policy examples/pick_cube.py
```

Or run the driver yourself, watch it in the dashboard, and run the policy against it:

```bash
python -m robotd.server --sim yam --scene pick_cube --realtime --addr unix:/tmp/robot.sock &
python -m robotd.dashboard --addr unix:/tmp/robot.sock &        # http://localhost:8780
ROBOT_ADDR=unix:/tmp/robot.sock python examples/pick_cube.py
```

A program looks like this:

```python
import robot
r = robot.connect()                                  # $ROBOT_ADDR
s = r.state()                                        # joints, TCP poses, grippers
frames = r.cameras(["top"])                          # RGB, depth, intrinsics, pose
r.gripper("left", 0.07)
r.move(robot.pose([0.35, 0.15, 0.10], rpy=[3.14159, 0, 3.14159]), ee="left")
```

`docs/MANUAL.md` is the full API: it is also exactly what the agent reads.

### An agent episode in the simulator

Episodes run Codex in a [bubblewrap](https://github.com/containers/bubblewrap) sandbox: the system's `bwrap` if there
is one, else the copy that comes with Codex (it needs unprivileged user namespaces). The install script puts the Codex
CLI in `$ROBOT_HOME/codex/current`. You need an OpenAI account for it; log it in once, for the episodes only:

```bash
CODEX_HOME=$ROBOT_HOME/codex-home $ROBOT_HOME/codex/current/bin/codex login --device-auth
python scripts/agent_run_real.py --sim pick_cube --tag sim-pick-01 --pipelined --effort medium
python scripts/agent_monitor.py &                               # http://localhost:8790
```

`--print agents-md` prints what the agent is told, and `--agent standin` runs a scripted probe of the sandbox in
place of the model. Both are good first checks.

## The real cell

The demos ran on two YAM arms on CAN buses, four Intel RealSense cameras (`top`, `front` and one on each wrist) and a
table, with one Linux computer. `docs/SETUP.md` walks through it:

1. `bash scripts/install.sh --cell` adds the RealSense driver.
2. Fill in the cell's block of `env.sh`: the CAN buses, the camera serials, the calibration file.
3. Bring it up with the motors off (`--monitor`), then check against i2rt's simulated robot (`YAM_SIM=1`).
4. First real motions with `examples/basic_motions.py`, slowly and fenced.
5. Each session: camerad, robotd, the dashboard and the agent monitor, then episodes.

Describe your cell to the agent in `docs/REAL_CELL.md` (it is copied into the agent's instructions). It describes ours.

## Reproducing the demos

The three runs in the post used this robotd, Codex CLI 0.165.0, GPT-6 Astra at medium reasoning effort and the
"ultrafast" service tier, which depends on your account. `docs/SETUP.md`, "Reproducing the demos", has the details.

```bash
python -u -m robotd.camerad --addr tcp:127.0.0.1:7412 --robot tcp:127.0.0.1:7410
python -u -m robotd.server --real yam --speed-scale 0.5 --separation 0.02 --separation-fine 0.005 \
    --cameras tcp:127.0.0.1:7412 --fence configs/real_yam/fence-cloth.json \
    --record $ROBOT_HOME/runs/demo-01 --addr tcp:127.0.0.1:7410
```

```bash
# find the cup with the pear, put the pear under the empty cup
python scripts/agent_run_real.py --task find_pear --pipelined --effort medium --service-tier ultrafast \
    --wall-h 0.25 --release-park --tag cups-01
# spread a partly folded towel and fold it in half (demos/ holds the instructions word for word)
python scripts/agent_run_real.py --task fold_towel --instruction "$(cat demos/towel-spread-fold.txt)" \
    --pipelined --effort medium --service-tier ultrafast --wall-h 0.25 --release-park --tag towel-01
python scripts/agent_run_real.py --task fold_towel --instruction "$(cat demos/towel-smooth-fold.txt)" \
    --pipelined --effort medium --service-tier ultrafast --wall-h 0.25 --release-park --tag towel-02
```

Without `--pipelined` the agent starts one action at a time (`kernel.start`) and writes the next after the last one's
report arrives, so the arms wait while the model thinks. `scripts/profile_run.py RUN` shows where an episode's time
went.

## Safety

- robotd enforces the limits in software, every 4 ms: per-arm fence boxes, a planar speed cap, a floor for the whole
  gripper, arm-to-arm separation, joint speed caps and a gripper overload guard (`docs/SETUP.md`, "The limits robotd
  enforces"). None of them sees people. Someone watches every run with the dashboard open and a hand near the arms'
  power switch.
- The dashboard's FREEZE, SOFT and PARK stop the programs and latch until RELEASE.
- Stop robotd with Ctrl-C: it parks the arms, then exits. Never `kill -9` it: the motors time out and the arms drop.
- The YAM's arm force estimate is not reliable; only the grippers' force is. The examples that use the arm's force
  estimate (`examples/helpers.py`'s `touch`, `pick_cube.py`) are for the simulator.

## Security

This was run on a single-user computer in a lab. Read this before running it anywhere else.

- **The sandbox is one bubblewrap layer.** The agent's commands run in the same sandbox as Codex, so they can read
  the Codex login (`auth.json`) and the egress proxy's password. Through the proxy they reach only the model API's
  hosts. Use a login made for the robot (`$ROBOT_HOME/codex-home`), not your own.
- **robotd's RPC has no authentication.** Anyone who can reach its port can move the arms. Keep it on `127.0.0.1` or a
  unix socket, and on a shared computer remember that every local user can reach a localhost port.
- **The operator's requests share the agent's socket.** RELEASE and PARK are not part of the agent's API, and it is
  never told about them, but robotd serves them on the socket the agent uses. A latched stop does not hold against a
  program that sends RELEASE. The power switch does.
- **The dashboard can run programs** from a folder (`--programs`), as your user. It listens on `127.0.0.1` only; reach
  it from another computer through an SSH tunnel, or with `--host` and `--token`. Run it with `--no-programs` while an
  agent runs.
- The agent's sandbox sees a read-only copy of the client code at `/opt/robot-api`. That path appears in the agent's
  instructions, so it stays as it was in the demos.

## Layout

```
robot/      the client the agent imports (connect, state, cameras, move, gripper, ik, reachable, when), the program
            runner (python -m robot run) and the pipelining kernel (python -m robot.kernel)
robotd/     the driver (server.py), the camera server (camerad.py), the control core, the backends (MuJoCo, YAM
            through i2rt, RealSense), the YAM station model and the operator's dashboard
tasks/      yam_station.py: simulated scenes with hidden success checks; real_yam.py: the real cell's tasks
scripts/    agent_run_real.py (an episode), agent_monitor.py, agent_debrief.py, agent/ (the sandbox's pieces and
            the AGENTS.md template), install.sh, run_sim.py, and tools for runs: run_log, profile_run, export_run,
            render_run, replay_requests
examples/   scripted policies: pick_cube, basic_motions, gentle_close, gripper_yield, towel/
configs/    real_yam/: the environment template and the workspace fences
demos/      the towel demos' instructions
docs/       below
tests/      the driver, the API, the operator's requests, the dashboard and the agent monitor (no hardware)
```

## Docs

| | |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Install, the simulator, the real cell, episodes, reproducing the demos, limits, known issues |
| [docs/MANUAL.md](docs/MANUAL.md) | The robot API and runtime, as the agent reads it |
| [docs/AGENT_PLAYBOOK.md](docs/AGENT_PLAYBOOK.md) | A worked guide to the kernel and pipelining, added to the agent's instructions with `--playbook` (the demos ran without it) |
| [docs/REAL_CELL.md](docs/REAL_CELL.md) | Our cell, as the agent is told about it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The processes, the control core, the control law, the guards |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | The operator's dashboard |
| [docs/RUNS.md](docs/RUNS.md) | What an episode and a robotd recording leave on disk |

The agent-facing files (`docs/MANUAL.md`, `docs/REAL_CELL.md`, `docs/AGENT_PLAYBOOK.md`,
`scripts/agent/AGENTS.md.template`, and the code under `robot/` and `robotd/core/` that the agent can read) are as
they were in the demo runs.

## License

Apache-2.0 (`LICENSE`). i2rt, MuJoCo, the Codex CLI and the other dependencies are installed separately, under their
own licenses.
