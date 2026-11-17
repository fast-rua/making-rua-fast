# Run records

What a run leaves on disk. An agent episode (`scripts/agent_run_real.py`, on the real cell or with `--sim SCENE`)
writes the layout in the first section; robotd's own recording, which `scripts/run_sim.py` also writes, is the second.
The agent never sees a run directory, apart from its `workspace/`.

## An agent episode: `$ROBOT_HOME/agent-runs/<tag>/`

```
<tag>/
  TIMELINE.md       the episode in one time order: the prompts and operator messages, the agent's messages, every
  timeline.jsonl    command it ran (duration, exit code), every program and how it ended, and every robot command
                    with its result or refusal. scripts/run_log.py writes them at the end (again by hand, or
                    --all $ROBOT_HOME/agent-runs)
  meta.json         robotapi.agent-run.v1 (below); rewritten as the episode progresses
  before.json       the task's measurement from the cameras before the agent starts (tasks/real_yam.py view=)
  verdict.json      the same after the park, and the task's check on the two: a first read, not the verdict
  profile.json      where the time went: model, tools, motion, and the model time hidden behind motion
                    (scripts/profile_run.py)
  workspace/        the agent's working directory (/workspace in the sandbox): AGENTS.md (the agent's instructions,
                    scripts/agent/AGENTS.md.template filled in), its files, and .robot/runs/<run>/ with each program
                    run's log, code and pause snapshots (cameras and state)
  workspace.git     the workspace's history: its state at the start and at the end
  codex/            events.jsonl (what the agent monitor shows), app-server.jsonl (the raw app-server protocol),
                    egress.jsonl (one line per connection through the egress proxy: host, port, allowed, bytes),
                    stderr.log, home/ (the episode's CODEX_HOME: config.toml, and sessions/**/rollout-*.jsonl,
                    Codex's own complete record of the session with every image the model saw)
  steer/inbox.jsonl the operator's messages from the agent monitor
  robotd/           README: where robotd's recording is and the byte offset of this episode's first request; after
                    the episode, run_log.py copies the episode's slice of it here (next section)
  cameras/          all four cameras at --streams-fps (and depth at --streams-depth-fps)
  witness/          a top and a front frame every 10 s (--witness-s)
  video-4x.mp4      the timelapse (--video-speed)
  sandbox/          robot-api/ (what the agent sees at /opt/robot-api: robot/, robotd/core/, docs/MANUAL.md,
                    docs/REAL_CELL.md) and bwrap.json (the sandbox command)
  logs/             program-host.jsonl (the programs the agent ran), streams.log, video.log, memory.jsonl (the
                    sandbox's memory use)
```

With `--sim SCENE` the launcher starts its own robotd, whose recording goes to `robotd-sim/` and its output to
`robotd-sim.log`, and `verdict.json` holds the scene's own success check, read through robotd's control socket.

### meta.json (`robotapi.agent-run.v1`)

```json
{"schema": "robotapi.agent-run.v1", "tag": "...", "task": "fold_towel", "real": true, "staggered": true,
 "towel": {"w_cm": 30, "h_cm": 22, "colour": "blue"}, "sim": null, "instruction": "...", "agent": "codex",
 "repo": {"commit": "...", "branch": "...", "dirty": ["..."]}, "host": "...",
 "codex": {"version": "codex-cli 0.165.0", "model": "gpt-6-astra", "reasoning_effort": "medium",
           "service_tier": "ultrafast", "argv": [...], "config": "<config.toml>", "prompt": "...",
           "steering": {"inbox": "steer/inbox.jsonl", "linger_s": 60}, "reasoning_summary": "none", "prune": {...}},
 "sandbox": {"kind": "bwrap, one layer", "network": "egress-proxy", "egress_allow": ["chatgpt.com", ...],
             "argv": "sandbox/bwrap.json"},
 "robotd": {"addr": "tcp:127.0.0.1:7410", "record": "...", "requests_offset": 0, "camera_addr": "...",
            "health_start": {...}},
 "arms_start": {...}, "arms_end": {...}, "park_end": {...}, "before": {...},
 "limits": {"wall_s": 900, "memory_gb": 4, "process_memory_gb": 3},
 "state": "starting" | "running" | "done", "started_utc": "...", "ended_utc": "...",
 "end_reason": "...", "program_host": {...}, "witness": {...}, "video": {...}, "camera_streams": {...},
 "memory": {...}, "notes": ["..."]}
```

## robotd's recording: `robotd/` (what `python -m robotd.server --record DIR` writes)

```
run.json         robotd.run.v1: backend, robot, lockstep, started_utc, fast_tick_s (0.004), tick_s (0.04), joints
                 (the order of every tick column), arm_joints, finger_joint, tcp_sites, and in simulation
                 task: {name, seed, args} (the scene variant; args: --scene-arg keywords, e.g. noise) and scene:
                 {objects: [{name, shape: box|cylinder|sphere, size (MuJoCo half extents), rgba, fixed}],
                  cloths: [{name, triangles}], table_top_z, table_x, station_to_world_y, urdf, mjcf, cameras}
requests.jsonl   one line per request: {n, wall, dur_s, client, t0, t1, req, rep}
                   client: the connection (one per program run); t0/t1: the robot clock at start and reply
                   req.op: hello (req.program: argv, cwd, pid) | state | command (req.spec: op, ee/joints, ...) |
                           wait | status | cancel | step | stop | zero_force | recover | signal | cameras | finish
                   req.src: [[file, line, function, code_sha], ...], innermost first: the policy code that sent it
                   rep: {ok, result} or {ok: false, error: {type, message}}; a command's result has status, reason,
                        t, duration, at (the state when it ended); cameras: {name: {png, depth_png, K, T, t[, roi]}}
                        (a frame taken with roi= is the cropped image, K for the crop)
code/<sha>.txt   the source of each policy file at the time it ran (code_sha above)
frames/<n>-<camera>.png   the RGB image the agent got in request n
frames/<n>-<camera>_depth.png   its depth, a 16-bit grey PNG in mm (0: no reading); cv2.imread(path,
                 cv2.IMREAD_UNCHANGED) reads it as uint16
ticks/NNNNN.npz  every 4 ms tick: t, q, dq, q_ref, q_cmd, tau (N x joints), wrench_<arm> (N x 6, what the tool
                 applies), obj/<name> (N x 7: x y z qw qx qy qz, simulation only), cloth/<name> (every 10th tick:
                 M x vertices x 3) with its times cloth_t/<name>; robotd.core.recorder.load_ticks
verdict.json     {verdict, summary, t}, written at r.finish() when the simulated task has a success check
timing.json      real-time simulation only: ticks, late_ticks, max_late_s, resyncs, physics wall time per tick
policy.log       the policy's printed output (scripts/run_sim.py keeps it)
replay/video.mp4 the run re-rendered from ticks/ (joints, free objects, cloth) from two free cameras side by side; video time
replay/video.json  is simulation time (t0/t1 above). scripts/render_run.py RUN_DIR; developer view, never for agents
```

Frames: world = origin on the table top midway between the arm bases, +x forward, +z up. The URDF/MJCF in
`scene.urdf` are in the left-base frame: add `station_to_world_y` (0.305) to y. The finger joints `*_joint7` and
`*_joint8` each travel 0..0.047 m; the gripper opening is their sum.
