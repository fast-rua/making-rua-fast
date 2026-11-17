"""Run a coding agent (Codex) on the real YAM cell: one episode of a task in tasks/real_yam.py, against the robotd and
camerad that already run there. Run it on the cell PC with the robot-api venv's Python:

    python scripts/agent_run_real.py --task fold_flat_towel --tag towel-fold-01            # an episode (1 h)
    python scripts/agent_run_real.py --task fold_flat_towel --tag probe-01 --agent standin  # the sandbox, no model,
                                                                                            #   no robot commands
    python scripts/agent_run_real.py --task fold_flat_towel --print agents-md              # what the agent is told

It is scripts/agent_run.py's episode (that module's sandbox, egress proxy, program host, memory watchdog and workspace
history, imported, with its paths pointed at the cell), with these differences:

- robotd and camerad are not started here: they are the operator's, on TCP. The sandbox has no network of its own,
  so each is reached through a unix socket bound into it (/run/robot-api/robot.sock and cameras.sock), forwarded
  here to its TCP address. robotd's own recording (its --record directory) is the episode's robot record.
- One sandbox layer. On this machine AppArmor (kernel.apparmor_restrict_unprivileged_userns = 1) lets /usr/bin/bwrap
  create namespaces, but not Codex's bundled bwrap, and no bwrap inside another sandbox. So the outer sandbox is the
  system bwrap, and inside it Codex runs its commands without a sandbox of their own (sandbox_mode
  danger-full-access) and the program host runs programs directly. Against the cluster's two layers this gives up:
  commands share Codex's PID namespace, so they could read the login (/codex-home/auth.json) and Codex's environment
  (the egress proxy's password), and through the proxy reach the model API's hosts (EGRESS_ALLOW), nothing else.
  Still out of reach: the repository (examples/, tasks/, scripts/), the run directories, the calibration and the
  home directory (nothing outside the binds exists in the sandbox), the CAN buses and the internet (a network
  namespace of its own), robotd's and camerad's processes (a PID namespace of its own).
- The arms. Before the agent: no fault, no command running, and no operator park (--release-park releases one, for
  an unattended series). After it: the arms are parked, which latches (programs that outlive the agent are refused
  until the next release), and the task's camera check runs. verdict.json holds the check and operator_verdict:
  null until the operator fills it in from the witness frames and robotd's recording.
- The witness: the top and front cameras every --witness-s seconds, small PNGs in witness/, for the operator; and a
  timelapse video of them (scripts/record_episode.py: video-4x.mp4, a frame every 0.5 s played at 4x), from before the
  agent starts until after the end-of-episode park and check.
- r.finish(): found in robotd's requests.jsonl (the summary, and when).
- Steering (the default; --exec runs `codex exec` as the simulation does, without it): the sandbox runs
  `codex app-server --stdio` and scripts/agent/codex_driver.py speaks to it from here. The operator's messages
  (RUN/steer/inbox.jsonl, which the monitor's steer box appends to) go into the agent's running turn, or start a new
  one if it has finished; after its last turn the episode waits --linger-s for one before it ends.

--sim SCENE runs the same episode on a simulated station instead of the cell, everything else unchanged (the sandbox,
the Codex driver, pruning, the prompt's sections, the recordings, the profile):

    python scripts/agent_run_real.py --sim fold_towel --seed 0 --tag sim-towel-01 --effort low --service-tier ultrafast

- robotd is started here (`robotd.server --sim yam --scene SCENE`, on the wall clock unless --lockstep, cameras
  rendered on --gpu), on a free local TCP port forwarded into the sandbox as the cell's is, recording into
  RUN/robotd-sim/; it is shut down at the end.
- AGENTS.md: the simulation's world paragraph instead of the real one, and no cell description; the rest is word for
  word the cell's.
- No park and no camera check at the end: the task's own success check (tasks/, through robotd's control socket, as
  scripts/agent_run.py takes it) is verdict.json's "verdict".
- Paths come from one install folder, $ROBOT_HOME (default ~/robot-cell; docs/SETUP.md), as on the cell.

RUNS/<tag>/ is agent_run.py's (meta.json, verdict.json, workspace/, workspace.git/, codex/, sandbox/, logs/), plus
witness/, before.json and the video. At the end, scripts/run_log.py makes it the episode's complete log: robotd/ gets
the episode's window of robotd's recording (its requests, tick files and program sources), and timeline.jsonl and
TIMELINE.md put the agent's every item (Codex's session record, in codex/home/sessions/), the programs and the robot
commands in one time order.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import agent_run as ar                                   # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent"))
from codex_driver import CodexDriver                     # noqa: E402
import run_log                                           # noqa: E402

import numpy as np                                       # noqa: E402

import robot                                             # noqa: E402
from tasks import real_yam                               # noqa: E402

REAL = ar.HOME                                           # the install folder ($ROBOT_HOME; agent_run.py's paths)
MEM_CAP_GB, MEM_CAP_PROCESS_GB = 4, 3                    # the sandbox's memory caps; the cell PC has 14 GB and robotd
                                                         #   keeps its share
# The system bwrap (on the cell the one AppArmor lets create namespaces), else the one on PATH, else the one that comes
# with Codex (it needs unprivileged user namespaces)
BWRAP = "/usr/bin/bwrap" if os.path.exists("/usr/bin/bwrap") else (
    shutil.which("bwrap") or os.path.join(ar.CODEX_PKG, "codex-resources", "bwrap"))
# Paths the stand-in agent checks it cannot read: the repository, the runs, the Codex login, the calibration, SSH keys
HIDDEN = [ROOT, os.path.join(REAL, "runs"), os.path.join(REAL, "agent-runs"), os.path.join(REAL, "codex-home"),
          os.path.expanduser("~/.ssh")]
if os.environ.get("YAM_CALIBRATION"):
    HIDDEN.append(os.path.dirname(os.path.abspath(os.environ["YAM_CALIBRATION"])))
WITNESS_S = 10.0
WITNESS_CAMERAS = ("top", "front")
STREAMS_FPS = 6.0                                        # the agent's recordings: all four cameras, colour
STREAMS_DEPTH_FPS = 3.0                                  # ... and RGB-D grabs (colour, depth, K, T) a second
RECORDINGS = "/var/record/cameras"                       # ... where the sandbox sees them (read-only)
TOOLS = "/opt/harness/tools"                             # ffmpeg, first on the sandbox's PATH
PARK_TIMEOUT_S = 90.0

REAL_WORLD = (
    "**The world** is the real robot cell described in \"The real cell\" below (`docs/REAL_CELL.md`): two YAM arms "
    "on a\n  table, with four cameras (top, front, and one on each wrist). It runs on the wall clock: anything you "
    "leave moving\n  keeps moving while you read, think or write code, and someone may be watching remotely who can "
    "stop or park\n  the arms at any time.")
REAL_CELL = os.path.join(ROOT, "docs", "REAL_CELL.md")    # the cell's description, copied before the manual
PLAYBOOK = os.path.join(ROOT, "docs", "AGENT_PLAYBOOK.md")  # --playbook: the worked guide, before the cell's description
# Advice on how to work, in the prompt only with --advice (the 2026-11-05 episodes had it; the default is facts).
REAL_WORK = """# Working on the real robot

- **Plan before the first motion.** Write `PLAN.md`, a few lines per object: which arm, how it is grasped and why,
  where it ends up, and what the next step needs from it (reach, the cameras' view, room for the other arm). Put an
  object where the next step needs it, not just somewhere free.
- **Write big programs made of small policies.** One function per subtask (a grasp with its approach and checks, a
  carry, a pour), each finding its inputs by looking (no typed-in coordinates), acting, and checking. The program
  strings them together for the whole task, or as much of it as you can plan, and runs straight through: as long as
  its checks pass it goes on to the next subtask by itself, and nobody waits. Subtasks skip work already done, so a
  rerun continues from the world.
- **Perception is code you keep.** Write each detector (an object, its parts, what a gripper holds) as a function, and
  check what it finds against the images before acting on it. When it is wrong, fix that function and re-run it on
  the frames where it failed, instead of writing a new one-off.
- **Check at every stage.** Before a subtask: what it needs (the object there, visible, in reach). While it runs:
  that things hold up (still gripped, still in view, still upright, the arm still making progress), inside the loop
  that watches the motion, with a short time margin so noise does not trip it. After it: that it did what it should.
- **Handle what you can in the code; stop only when you cannot.** Re-aim when the object moved; retry a missed grasp
  a few times, changing what the failure points to; look from another camera. What the code cannot handle ends in a
  failed assert, which pauses the program with the cameras and the arms' state saved in `pause-N/` and hands it to
  you. Run without `--unattended`.
- **Pauses are yours to place.** A pause holds the arm until you answer; a check that passes costs nothing. Decide
  where the program should stop for you, and change that as you learn which checks you can trust.
{recordings}- **Work in parallel with the arm; keep motion in order.** Motions stay in the program, one after another, each
  finishing before the next. Everything else can run at the same time: `python -m robot run task.py --wait 5` returns
  while the program runs on, and when it pauses or ends you are told in your running turn (a `[program] ...`
  message), so there is nothing to poll.
- **Closed loop wherever the world can change during a motion.** Keep looking while the arm moves and re-send the
  motion (`wait=False`), or let the driver end it (`until=`, e.g. the gripper's width when a carried object could
  slip).
- **Motions particular to the task are yours to write.** For a dynamic one, first state the effect you want and its
  timing; derive the phases, speeds and stops from it within the joint speed limits (`r.description["max_velocity"]`);
  then write the path, preview it with `dry_run=True`, and measure the result from the cameras.
- **When something fails, fix the policy, not the coordinate.** Change the subtask's function or a helper so the same
  failure cannot come back. If the same check trips twice for the same reason, the code has to change. If a check
  tripped although nothing was wrong, fix it or remove it. Keep the fixes in the program, so by the end it is the
  task's policy.

"""
REAL_RECORDINGS = (
    "- **Look back at anything.** Every pause saves the cameras and the arms' state in `pause-N/`. All the cameras "
    "are\n"
    "  also recorded from the start of the episode, colour, {fps:g} frames a second, in `{dir}` (one MP4 per\n"
    "  camera, each frame's wall time in `frames.csv`), and colour with depth, K and pose a few times a second\n"
    "  (`rgbd/`); `robot.recording` reads them by time (`cameras_at` gives frames as `r.cameras()` does, for\n"
    "  `robot.viz`), and `ffmpeg` is on `PATH`. Try a check on moments that went well and moments that went badly\n"
    "  before the program relies on it.\n")
REAL_PYTHONS = (
    "- **Python** (nothing can be installed; `PYTHONPATH` is `{code}`, so it can `import robot`):\n"
    "  `{python}` (first on `PATH`, as `python`): numpy, scipy, OpenCV (`cv2`), mujoco, mink, ruckig, pillow. No\n"
    "  torch, JAX, Pinocchio or Open3D. `ffmpeg` is on `PATH`.\n")


# ------------------------------------------------------------------------------------------------ helpers
class Forward:
    """A unix socket that forwards each connection to a TCP address: how the sandbox, which has no network of its
    own, reaches robotd and camerad."""

    def __init__(self, path: str, host: str, port: int):
        self.path, self.addr = path, (host, port)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        os.chmod(path, 0o600)
        self.srv.listen(64)
        self.connections = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections += 1
            try:
                up = socket.create_connection(self.addr, timeout=10)
                up.settimeout(None)
            except OSError:
                conn.close()
                continue
            for src, dst in ((conn, up), (up, conn)):
                threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        for s in (dst, src):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self):
        try:
            self.srv.close()
        except OSError:
            pass


def tcp(addr: str) -> tuple:
    m = re.fullmatch(r"tcp:([^:]+):(\d+)", addr)
    if not m:
        raise SystemExit(f"{addr}: expected tcp:HOST:PORT")
    return m.group(1), int(m.group(2))


def robotd_record_dir(addr: str) -> str | None:
    """robotd's --record directory, from the command line of the robotd serving addr (None if not found)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [x.decode("utf-8", "replace") for x in argv]
        if "robotd.server" in argv and addr in argv and "--record" in argv:
            return os.path.realpath(os.path.expanduser(argv[argv.index("--record") + 1]))
    return None


def finishes(record: str | None, offset: int, t0: float) -> list:
    """The r.finish() calls in robotd's requests.jsonl after byte offset (this episode's): [{wall, summary}]."""
    out = []
    path = os.path.join(record or "", "requests.jsonl")
    if not record or not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        f.seek(offset)
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            req = e.get("req") or {}
            if req.get("op") == "finish" and e.get("wall", 0) >= t0:
                out.append({"wall": e.get("wall"), "summary": req.get("summary"), "ok": (e.get("rep") or {}).get("ok")})
    return out


class Witness(threading.Thread):
    """The cameras every period, as small PNGs (every 4th pixel): what the operator judges the episode by."""

    def __init__(self, addr: str, out: str, period: float, cameras=WITNESS_CAMERAS):
        super().__init__(daemon=True)
        self.addr, self.out, self.period, self.cameras = addr, out, period, list(cameras)
        self.stop = threading.Event()
        self.frames, self.errors = 0, []
        os.makedirs(out, exist_ok=True)

    def run(self):
        from robot.images import write_png
        r = None
        while not self.stop.is_set():
            t = time.time()
            try:
                r = r or robot.connect(self.addr, name="agent_run_real/witness")
                for name, f in r.cameras(self.cameras, depth=False).items():
                    stamp = time.strftime("%H%M%S", time.localtime(t))
                    write_png(os.path.join(self.out, f"{stamp}-{name}.png"), np.ascontiguousarray(f.rgb[::4, ::4]))
                    self.frames += 1
            except Exception as e:                       # noqa: BLE001 -- keep watching
                if len(self.errors) < 20:
                    self.errors.append(f"{ar.utc(t)} {e!r}")
                r = None
            self.stop.wait(max(0.5, self.period - (time.time() - t)))


def arms_state(r) -> dict:
    st = r._call("state")
    return {"fault": st.get("fault"), "parked": st.get("parked"),
            "busy": {g: v for g, v in (st.get("busy") or {}).items() if v}}


def park(r, notes: list) -> dict:
    """The operator's park (it latches): wait until both park moves end."""
    try:
        ids = r._call("park")["ids"]
    except Exception as e:                               # noqa: BLE001
        notes.append(f"park refused: {e!r}")
        return {"ok": False, "error": repr(e)}
    t0 = time.time()
    while time.time() - t0 < PARK_TIMEOUT_S:
        if not arms_state(r)["busy"]:
            return {"ok": True, "ids": ids, "s": round(time.time() - t0, 1)}
        time.sleep(0.5)
    notes.append(f"the park had not ended after {PARK_TIMEOUT_S:.0f} s")
    return {"ok": False, "ids": ids, "timeout": True}


PROGRAM_MESSAGES = (
    "- **Messages from the cell:** `python -m robot run` starts a program in a process of its own, which keeps running\n"
    "  after the command returns (see the manual); when it pauses or ends, a `[program] ...` message comes into your\n"
    "  turn. When the operator stops, parks or releases the arms, an `[operator] ...` message says so.\n")
ACTION_MESSAGES = (
    "  As an action you started with `kernel.start` ends, an `[action NAME] ...` message brings its report.\n")
BEFORE_YOU_ACT = """# Before you act

- **Think through the physical consequences.** For each action you consider, picture the whole motion, not only
  where it ends: what each finger touches first, what gets pushed, dragged, folded or knocked over on the way, where a
  held object swings, and what state it leaves for the next step. Name what could go wrong. Choose the option that is
  most likely to work and easiest to recover from when it does not, and change the plan when what you see says it
  will not work.
- **Grasp deeply.** A grasp holds by how much material is between the pads. Closed on the last millimetres of an edge
  or a corner tip, it slips or tears free. Aim so the pads close well inside the edge, with most of their height on
  the material, and approach from a side where neither finger pushes the material away before it is between the
  jaws. After closing, confirm the hold (a short lift, then look) before building on it, and do not let go of a hold
  you would have to redo until its result is confirmed.
- **Review before you commit.** Before a grasp or a motion that is hard to undo, look at it with `robot.viz`: the
  ghost in the camera images, and the 3D views of the point cloud (`jaws`, `along`, `approach`, and from the side),
  where the depth of the grasp, the fingers' clearance and contact are visible. For a motion with several stages,
  review its key poses together.

"""


# Facts about where an episode's time goes (measured on the cell's runs, 2026-11-07), in the prompt unless
# --no-time-facts. Facts, not advice: what each step and image costs, and what the API offers to spend fewer.
TIME_FACTS = r"""# Time, steps and images

- **Each of your steps is a model round trip of about 1 to 3 s,** longer the more images the conversation holds; the
  arms and the world keep going meanwhile.
- **A fresh `python` per command costs about 0.3 s** (start and imports), more the first time a ghost loads the arm
  model. `python -m robot.kernel` runs code in one long-running Python instead (a call starts in about 40 ms): stdin or
  a file, one namespace that persists between calls (robot, np, cv2, viz, watch, when and kernel imported; your
  variables, functions and loaded data stay), and each call ends as a program does: its robot connections close
  (their motions stop) and it prints its `[state]` and `[views]` lines. `python -m robot.kernel --help` says more.
- **Every image in the conversation slows every later step,** by the number of images and their size. One image of
  several cameras costs less than one per camera: `viz.show_cameras(name, r)` shows every camera as one labelled grid
  (`viz.grid` tiles any images). A smaller image costs less (`viz.show_cameras(name, r, width=640)`). Attach images
  at the default detail: this Codex drops an image sent with detail "low" (the model gets a note instead). The
  harness moves the conversation to a fresh thread after many images: older images become notes naming their
  files, which you can view again.
- **A command's images can come back in the step that ran it.** A program that showed images prints
  `[views] ["/path/a.png", ...]` (one JSON list) as it exits, so the same exec cell can attach them, e.g.
  `const o = (await tools.exec_command({cmd, yield_time_ms: 20000})).output; text(o);`
  `for (const p of JSON.parse((o.match(/^\[views\] (.*)$/m) || [, "[]"])[1])) image((await tools.view_image({path: p})).image_url);`
  Viewing them in a later step costs another round trip.
- **A program that moved the arms prints `[state] ...` as it exits:** where each arm ended, its gripper's width and
  whether it is holding something, contact, any fault.
- **A command that runs longer than its `yield_time_ms` returns early,** and seeing how it ended takes another step;
  with a `yield_time_ms` longer than its motions, it all comes back at once. `python -m robot run` programs report
  when they pause or end (`[program] ...`, with their `[state]` and `[views]`), with nothing to check on.
- **While a motion runs, your time runs too.** `kernel.start(fn)` (below) runs a whole action while your steps go
  on; a motion started with `wait=False` returns at once too (its `.result()` waits for it, within the same
  call). `robot.watch` renders, while a motion runs and
  without holding it up, one compact image per arm about once a second (`views/monitor-*.png`: the wrist and front
  cameras with the arm's ghost where it is, and the point cloud's `jaws` and `along` views) with a line of what is
  between the jaws, and can stop the motion through a signal (`until=when.signal("watch") > 0`) when your check
  says so.
- **`viz.marks(frame, name, mask=...)` numbers points on a camera image** and returns their world positions, so a
  place can be named by its number instead of measuring pixels and looking up depth in another step.
  `on_table=True` keeps only what lies over the table (whatever is around and beyond it is left out).

%%ACTIONS%%## Make the tools yours: every step should end with what the next one needs

The robot API is yours to extend while you work. Functions you define in your kernel stay for the rest of the
episode, and hooks you register run at the end of your calls with their output in that call's own result, so you can
shape what each command does and returns to fit the task. Use them to need fewer steps: each step is a model round
trip, and one that only fetches something costs as much as one that acts.

- **Hooks run at the end of your calls.** `kernel.after_motion(fn)` runs `fn(r, frames)` at the end of every kernel
  call that moved the arms (`r`: the call's robot; `frames`: one grab of every camera with depth, taken when the hook
  first reads it); what it prints and shows comes back with the motion's result. `kernel.after_call(fn)` runs after
  every call. If you find yourself fetching the same thing after motions (the object's numbered points, a wrist
  view, a check of what is between the jaws), make it a hook, e.g. once the towel's colour is known:
  ```python
  def towel(r, frames):
      f = frames['front']; h = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2HSV)
      viz.marks(f, 'towel', mask=(h[..., 0] > 85) & (h[..., 0] < 120) & (h[..., 1] > 90), on_table=True, picture=False)
  kernel.after_motion(towel)
  ```
  From then on every call that moves the arms ends with `[marks towel] 1:(x,y,z) ...`. Registering a function of the
  same name again replaces it (`kernel.remove_hook(name)`, `kernel.hooks()`).
- **Helpers: write once what you repeat.** A grasp sequence, the review picture you draw before each move, a
  check you keep making: define it as a function in the kernel and call it by name. Shorter calls are quicker to
  write, and the same function runs the same way every time.
- **Keep hooks quick and their output short.** Every call that moves pays for every hook, and what a hook prints
  stays in the conversation: points as text cost little (a mask and its points take about 0.1 s), a 3D render
  0.5-1 s, and each image slows every later step. A hook that raises prints `[hook NAME] error ...` and never fails
  the call.
- **Change them as the task changes:** replace a hook when you need something else watched, drop it when you no
  longer do.
- **What stays as it is:** the driver's limits (the fence, the arms' separation, speed and force limits) are outside
  your kernel, and apply to everything as before.

"""

ACTIONS_PUSH = r'''## Start an action, look at your plan while it runs, and let its end come to you

A step to review a plan and another to execute it leave the arms still for two round trips, and a step that only
checks on a motion costs as much as one that acts. So look while the arms move, and never check on them:

- **Write the action as a function, start it, and show your plan in the same call.** `kernel.start(fn)` runs
  `fn(r)` in your kernel (`r`: the actions' robot connection) and returns at once; the arms begin while the rest of
  the call draws your plan picture, which comes back with the call.
  ```python
  def grasp(r):
      r.gripper('left', .07); r.move(A, ee='left', duration=3); r.move(G, ee='left', line=True, duration=3)
      r.gripper('left', w - 0.003, speed=0.02, force=30); r.move(L, ee='left', line=True, duration=3)   # w: its width
  kernel.start(grasp)
  review('left', [A, G])          # your plan picture (a helper of yours, or viz), drawn while the arm starts
  ```
- **When an action ends, its report comes to you as a message,** with no step of yours: `[action NAME] done in
  9.8 s` (or stopped, or its error), what it printed, the arms' `[state]`, every camera as one grid
  (`views/after-NAME.png`, attached to the message: no need to open it) and your after-motion hooks' lines and
  images. It reaches your running turn at your next step, or starts a new turn if you have ended yours. That message
  is your look: decide from it and start the next action in your reply. Fetching the cameras again costs a step.
- **While the arms move, check your plan picture.** Wrong: `kernel.action().stop()` halts every motion at once (the
  arms hold where they are) and the action's code goes no further; its report still comes. Right: nothing to do.
  Prepare the next action if that needs work (its perception, its plan), or end your turn. Do not wait on the
  action (`kernel.action().wait()`), sleep or poll: each of those is a step that only waits.
- **End your turn only while an action runs:** its report starts your next one. With no action running, ending
  your turn ends the episode.
- **When the next move does not depend on how this one ends, make both one action** (one function), so that the
  arms go straight from one to the next.
- **Give every kernel call a long `yield_time_ms` (60000).** A call returns as soon as its code ends, and
  `kernel.start` returns at once by itself; a short one makes a call that is still drawing come back empty, and
  getting its output then costs a step of polling.
- **Put the moves that commit least first** (the approach above the target), so that a stop comes before the ones
  that commit most (closing on the cloth, lifting it).
- **What stays as it is:** the driver checks every command (reach, fence, the arms' separation) before the arms
  move; an action's error comes in its report (`[action NAME] error ...`).
- Other calls mention the running action (`[action grasp] running for 3.1 s`). One action at a time; a call that
  sends the arms a command stops the running action first (its report still comes); its timeout (120 s,
  `kernel.start(fn, timeout=...)`) stops it too.

'''

ACTIONS_STAGGER = r'''## Keep a queue of actions, and keep looking while the arms work

The arms do not wait for you if the next action is queued before the running one ends: an action's end and the next
one's start are one moment, with no step of yours in between. You work the whole time they move, on the newest view.

- **Write the action as a function and queue it.** `kernel.queue(fn)` runs `fn(r)` in your kernel (`r`: the actions'
  robot connection) and returns at once. With nothing running it starts now; with an action running it starts as the
  last one queued before it ends. `kernel.queue([f, g])` queues several in order. At most three wait.
  ```python
  def grasp(r):
      r.gripper('left', .07); r.move(A, ee='left', duration=3); r.move(G, ee='left', line=True, duration=3)
      r.gripper('left', w - 0.003, speed=0.02, force=30); r.move(L, ee='left', line=True, duration=3)   # w: its width
  kernel.queue(grasp)
  ```
- **Messages come to you without a step of yours.** Each is the state of things now; it reaches your running turn at
  your next step, or starts a turn.
  - As an action ends: its report, `[action A] done in 9.8 s`, what it printed, `[queue] started: B (#14; ...)`,
    the arms' `[state]`, every camera as one grid (`views/after-A.png`, attached) and your after-motion hooks. The grid
    is taken as B starts, so it shows what A left.
  - Each time your turn ends while an action runs: a look, `[look N]`, with `[state]`, the grid (`views/look-N.png`,
    attached) and the queue, from that moment. Fetching the cameras yourself costs a step.
- **Every message says where the queue stands:** `[queue] running #14 grasp: 3.1 s in, command 3 (move)`, then
  `[queue] waiting 1: #15 lift`, `[queue] waiting 2: #16 lay` in order. `[queue] empty behind it` means the arms stop
  after the running action unless you queue one.
- **With each message, the queue is yours to change.**
  - Add: `kernel.queue(fn)`.
  - Change an item: `kernel.queue(new, replace_from=15)` takes #15 and everything behind it out of the queue and puts
    `new` in their place, in one step, since what stood behind it was planned on it. To replace more than one, pass a
    list: `kernel.queue([f, g], replace_from=15)`. If #15 has started by then, nothing changes and it raises.
  - Drop: `kernel.flush(from_item=15)` drops #15 and what is behind it; `kernel.flush()` drops the whole queue.
  - If what is queued still makes sense, keep working: write the action that comes after the last one and add it, from
    what you have and the code ahead of it, even with the running action unfinished. Each look is the moment to write
    the next action, not a check on this one. Only when three actions wait and all of them still make sense is there
    nothing to write: then end your turn, which costs the arms nothing, and the next look or report comes.
- **What you see is older than what is happening.** The newest picture shows the moment it was taken, and the running
  action has gone on since. What you queue starts from where the actions ahead of it leave things: you know their code
  and where they end the arms.
- **Code you queue runs later, so what it reads is newer than what you saw.** `r.state()` and `r.cameras()` inside `fn`
  are read as it runs, so a next move that depends on how the cloth lands can compute its target there, or open with a
  move that is useful either way. A check at the top of `fn` that raises ends it before it moves.
- **If an action ends with an error or is stopped, everything queued is dropped** (`[queue] dropped: B`) and the arms
  hold, so what you planned on its success does not run.
- **Wrong plan while the arms move:** `kernel.action().stop()` halts every motion at once (the arms hold where they are),
  its code goes no further and the queue is dropped; its report still comes.
- **Ending the episode:** with nothing running or queued, ending your turn ends it. Do not wait on an action, sleep or
  poll: each of those is a step that only waits.
- **Give every kernel call a long `yield_time_ms` (60000).** A call returns as soon as its code ends, and `kernel.queue`
  returns at once by itself.
- **What stays as it is:** the driver checks every command (reach, fence, the arms' separation) as it is sent; an
  action's error comes in its report (`[action NAME] error ...`). A call that sends the arms a command stops the running
  action first; its timeout (120 s, `kernel.queue(fn, timeout=...)`, counted from its start) stops it too.

'''

ACTIONS_WAIT = r'''## Start an action, then look at your plan while it runs

A step to review a plan and another to execute it leave the arms still for two round trips. Look while they move
instead:

- **Write the action as a function, start it, and show your plan in the same call.** `kernel.start(fn)` runs
  `fn(r)` in your kernel (`r`: the actions' robot connection) and returns at once; the arms begin while the rest of
  the call draws your plan picture, which comes back with the call.
  ```python
  def grasp(r):
      r.gripper('left', .07); r.move(A, ee='left', duration=3); r.move(G, ee='left', line=True, duration=3)
      r.gripper('left', w - 0.003, speed=0.02, force=30); r.move(L, ee='left', line=True, duration=3)   # w: its width
  kernel.start(grasp)
  review('left', [A, G])          # your plan picture (a helper of yours, or viz), drawn while the arm starts
  ```
- **Your next step, taken while they move, checks the picture.** Wrong: `kernel.action().stop()` halts every motion
  at once (the arms hold where they are) and the action's code goes no further. Right: `kernel.action().wait()`
  returns when it has ended.
- **The call that sees an action end is your look.** Its output carries how the action ended, the arms' `[state]`,
  every camera as one grid (`views/after-NAME.png`) and your after-motion hooks. Decide from that and write the next
  action in the same step; fetching the cameras again costs a step.
- **When the next action does not depend on how this one ends, chain them in one call:**
  `kernel.action().wait(); kernel.start(next_action)` takes the arms straight from one to the next.
- **Give every kernel call a long `yield_time_ms` (60000).** A call returns as soon as its code ends, and
  `kernel.start` returns at once by itself; a short one makes a call that is still drawing or waiting come back
  empty, and getting its output then costs a step of polling.
- **Put the moves that commit least first** (the approach above the target), so that a stop comes before the ones
  that commit most (closing on the cloth, lifting it).
- **What stays as it is:** the driver checks every command (reach, fence, the arms' separation) before the arms
  move; an action's error comes back in your next call's output (`[action NAME] error ...`).
- Every call reports the running action (`[action grasp] running for 3.1 s`) and how it ended. One action at a time;
  a call that sends the arms a command stops the running action first; its timeout (120 s,
  `kernel.start(fn, timeout=...)`) stops it too.

'''



def agents_md(instruction: str, wall_h: float, started: str, memory: tuple, streams_fps: float = 0.0,
              advice: bool = False, program_messages: bool = True, guidance: bool = True,
              time_facts: bool = True, sim: str | None = None, push: bool = True, staggered: bool = False,
              playbook: bool = False) -> str:
    """agent_run.py's AGENTS.md, for the cell: the real world paragraph, the cell's one Python, no GPU, two sockets,
    then a short section on deciding (BEFORE_YOU_ACT; guidance=False leaves it out) and the cell's description
    (docs/REAL_CELL.md) before the manual. advice=True adds the older, longer section on how to work on the real robot
    (REAL_WORK), as the episodes of 2026-11-05 had it. sim="realtime" or "lockstep": the same AGENTS.md for a simulated
    station (--sim): the simulation's world paragraph for that clock, and no cell description; the rest is word for
    word the same. push: kernel actions' end reports come as messages (ACTIONS_PUSH), else the call that waits on an
    action gets it (ACTIONS_WAIT). staggered (needs push): actions are queued behind the running one (ACTIONS_STAGGER).
    playbook: docs/AGENT_PLAYBOOK.md after those sections, a worked guide to the kernel and the queue from the runs
    that went best (for models that need more than the sections give)."""
    clock = sim or "realtime"
    text = ar.render_agents_md(instruction, False, wall_h, started, None, clock, memory)
    world = ar.WORLD[clock]
    assert world in text
    if not sim:
        text = text.replace(world, REAL_WORLD)
    i = text.index("# Robot manual")
    cell = "" if sim else open(REAL_CELL).read().strip() + "\n\n"
    work = ""
    if advice:
        rec = REAL_RECORDINGS.format(fps=streams_fps, dir=RECORDINGS) if streams_fps > 0 else ""
        work = REAL_WORK.replace("{recordings}", rec)
    facts = TIME_FACTS.replace("%%ACTIONS%%", ACTIONS_STAGGER if staggered else ACTIONS_PUSH if push else ACTIONS_WAIT)
    if staggered:                                        # the sections before it say how an action starts
        facts = facts.replace("`kernel.start(fn)` (below)", "`kernel.queue(fn)` (below)")
    guide = open(PLAYBOOK).read().strip() + "\n\n" if playbook else ""
    text = text[:i] + work + (BEFORE_YOU_ACT if guidance else "") + (facts if time_facts else "") + guide + cell + \
        text[i:]
    if program_messages:
        j = text.index("- **Ending:**")
        text = text[:j] + PROGRAM_MESSAGES + (ACTION_MESSAGES.replace("started with `kernel.start`", "queued") if staggered
                                              else ACTION_MESSAGES if push else "") + text[j:]
    text, n = re.subn(r"- \*\*Pythons\*\*.*?  - None of them has Pinocchio or Open3D\.\n",
                      REAL_PYTHONS.replace("{code}", ar.CODE).replace("{python}", f"{ar.ROBOT_VENV}/bin/python"),
                      text, flags=re.S)
    assert n == 1, "the Pythons paragraph of AGENTS.md.template changed"
    for old, new in (
            ("- **No GPU:** your programs see none; torch and JAX run on the CPU, and cuRobo cannot run.",
             "- **No GPU.**"),
            ("(`MANUAL.md` is the robot manual, copied below)",
             ("(`MANUAL.md` is the robot manual, copied below).\n" if sim else
              "(`REAL_CELL.md` describes this cell and `MANUAL.md` the robot; both are copied below).\n") +
             "- **Before you start,** read all of the code in `/opt/robot-api`, every file in `robot/` and `robotd/core/`:\n"
             "  the descriptions below summarise it, and the code says exactly what each call does and what it gives back"),
            ("your programs can connect only to the robot socket.",
             "your programs can connect only to the robot's two sockets (the driver, and the cameras\n  behind "
             "`r.cameras()`)."),):
        assert old in text, old
        text = text.replace(old, new)
    return text


def codex_config_one_layer(llm: dict, tier: str | None = None, summary: str = "none") -> str:
    """agent_run.py's config.toml with Codex's command sandbox off (module docstring): no permission profile. tier:
    the service tier ("fast" is what the TUI's /fast turns on; the account's model list names the others). summary:
    the reasoning summaries. "none" by default: with "detailed" the server holds each response until its reasoning
    summary is written, about 1.7 s per step that reasons (A/B on the cell 2026-11-07: 3.5 s vs 1.8 s per call), and
    the summaries are only for the monitor and the timeline."""
    text = ar.codex_config(False, llm)
    assert 'default_permissions = "robot"' in text
    text = text.replace('default_permissions = "robot"', 'sandbox_mode = "danger-full-access"')
    text = text.replace("approval_policy = ", f'model_reasoning_summary = "{summary}"\napproval_policy = ', 1)
    if tier:
        text = text.replace("approval_policy = ", f'service_tier = "{tier}"\napproval_policy = ', 1)
    return text[:text.index("[permissions.robot]")].rstrip() + "\n"


def sandbox_argv(run, tmp, inner, stop_r, host_log, password, models, llm, standin: bool,
                 recordings: str | None = None, events: str | None = None, staggered: bool = False) -> list:
    """agent_run.py's bwrap command, for one layer: the system bwrap, programs run directly, the camera socket, the
    camera recordings (read-only at RECORDINGS), ffmpeg first on PATH and (events: the sandbox's path of the push's
    directory) ROBOT_KERNEL_EVENTS."""
    a = ar.bwrap_argv(run, tmp, inner, False, False, stop_r, host_log, password, models, None, llm, None, False, None)
    a[0] = BWRAP
    cs = ar.command_sandbox()
    k = next(i for i in range(len(a)) if a[i:i + len(cs)] == cs)
    a[k:k + len(cs)] = ["/usr/bin/env"]                  # `env -- env -i -C CWD ...`: a program runs as it is
    for j in range(len(a) - 2):                          # ffmpeg (imageio-ffmpeg's static build) on PATH
        if a[j] == "--setenv" and a[j + 1] == "PATH":
            a[j + 2] = f"{TOOLS}:{a[j + 2]}"
            break
    i = a.index("--clearenv")
    extra = ["--setenv", "ROBOT_CAMERA_ADDR", f"unix:{ar.SOCKS}/cameras.sock"]
    binds = (["--ro-bind", os.path.join(ar.HERE, "standin_real.sh"), f"{ar.HARNESS}/standin_real.sh"]
             if standin else [])
    ff = ffmpeg_exe()
    binds += ["--ro-bind", ff, f"{TOOLS}/ffmpeg"] if ff else []
    if recordings:
        binds += ["--ro-bind", recordings, RECORDINGS]
        extra += ["--setenv", "ROBOT_RECORDING_DIR", RECORDINGS]
    if events:
        extra += ["--setenv", "ROBOT_KERNEL_EVENTS", events]
        if staggered:
            extra += ["--setenv", "ROBOT_KERNEL_STAGGER", "1"]
    return a[:i] + binds + a[i:i + 1] + extra + a[i + 1:]


def sim_task(name: str):
    """--sim: the task (tasks/), from tasks.yam_station first (tasks.get loads every task module, and some need scene
    assets a machine may not have)."""
    from tasks import yam_station
    if name in yam_station.TASKS:
        return yam_station.TASKS[name]
    import tasks
    return tasks.get(name)


def start_sim_robotd(a, run: str):
    """--sim: robotd on a simulated station (the task's scene and seed; on the wall clock unless --lockstep), on a free
    local TCP port that the episode forwards into the sandbox as it does the cell's robotd, recording into
    RUN/robotd-sim/ (its log: RUN/robotd-sim.log), with the control socket the sim launcher uses for the success check.
    Returns (process, control dir, address)."""
    import atexit
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    addr = f"tcp:127.0.0.1:{port}"
    ctl = tempfile.mkdtemp(prefix="robotd-sim-", dir="/tmp")
    os.makedirs(os.path.join(ctl, "ctl"))
    rec = os.path.join(run, "robotd-sim")
    os.makedirs(rec)
    argv = [sys.executable, "-m", "robotd.server", "--sim", "yam", "--scene", a.sim,
            *(["--seed", str(a.seed)] if a.seed is not None else []), *([] if a.lockstep else ["--realtime"]),
            *[x for kv in a.scene_arg for x in ("--scene-arg", kv)],
            *(["--separation", "0.02", "--separation-fine", str(a.separation_fine)] if a.separation_fine is not None else []),
            "--addr", addr, "--control-addr", f"unix:{ctl}/ctl/control.sock", "--record", rec,
            "--ready-file", f"{ctl}/ctl/robotd.ready", "--gpu", str(a.gpu)]
    env = dict(os.environ, PYTHONPATH=ROOT, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), PYTHONUNBUFFERED="1",
               ROBOTD_CACHE=os.path.join(run, "robot-files"))
    proc = subprocess.Popen(argv, env=env, cwd=ROOT, stdout=open(os.path.join(run, "robotd-sim.log"), "w"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    proc.argv = argv
    atexit.register(stop_sim_robotd, proc, ctl, None)  # whatever ends the launcher, the simulation ends with it
    t0 = time.time()
    while not os.path.exists(f"{ctl}/ctl/robotd.ready"):
        if proc.poll() is not None:
            raise SystemExit(f"the simulated robotd exited ({proc.returncode}): see {run}/robotd-sim.log")
        if time.time() - t0 > 180:
            stop_sim_robotd(proc, ctl, None)
            raise SystemExit(f"the simulated robotd was not ready after 180 s: see {run}/robotd-sim.log")
        time.sleep(0.2)
    print(f"simulated robotd: scene {a.sim}, seed {a.seed}, {'lockstep' if a.lockstep else 'wall clock'}, {addr} "
          f"(ready in {time.time() - t0:.1f} s)", flush=True)
    return proc, ctl, addr


def sim_verdict(task, run: str, proc, ctl: str, record: str | None, offset: int, t0: float, notes: list):
    """--sim: the task's own success check, as the sim launcher takes it (agent_run.py): after SETTLE_S with the arms
    holding, or at r.finish() for a task that ends there. verdict.json."""
    if proc is None or proc.poll() is not None:
        notes.append("the simulated robotd had exited before the success check")
        return None
    try:
        time.sleep(0.5)                                  # robotd releases the agent's dropped connections
        ends = bool(getattr(task, "ends_at_finish", False))
        ev = ar.control(ctl, {"op": "evaluate", "settle_s": 0.0 if ends else ar.SETTLE_S})
        finish_path = os.path.join(record or "", "verdict.json")
        at_finish = json.load(open(finish_path)) if record and os.path.exists(finish_path) else None
        fin = finishes(record, offset, t0)
        verdict = {"verdict": ev["verdict"], "verdict_from": "end", "t": ev["t"], "settled_s": ev["settle_s"],
                   "t_before_settle": ev["t_before_settle"], "evaluated_utc": ar.utc(),
                   "finished": fin[-1] if fin else ev.get("finished"), "at_finish": at_finish}
        if ends and at_finish is not None:
            verdict.update(verdict=at_finish["verdict"], verdict_from="finish", at_end=ev["verdict"])
        ar.write_json(os.path.join(run, "verdict.json"), verdict)
        return verdict
    except Exception as e:                               # noqa: BLE001
        notes.append(f"evaluation failed: {e!r}")
        return None


def stop_sim_robotd(proc, ctl: str, notes: list | None):
    """--sim: shut the simulated robotd down through its control socket (SIGTERM, then SIGKILL, if it does not go)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        ar.control(ctl, {"op": "shutdown"}, timeout=30)
        proc.wait(60)
    except Exception as e:                               # noqa: BLE001
        if notes is not None:
            notes.append(f"the simulated robotd did not shut down cleanly: {e!r}")
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    shutil.rmtree(ctl, ignore_errors=True)


def ffmpeg_exe() -> str | None:
    """imageio-ffmpeg's static build, else the ffmpeg on PATH, else None (no ffmpeg in the sandbox)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


class OperatorEvents:
    """Tells the agent, in its running turn, when the operator stops, parks or releases the arms: robotd's state, read
    every second. A stop or a park is a pause the operator ends, not the end of the episode."""

    def __init__(self, r, deliver):
        self.r, self.deliver = r, deliver
        self.next = 0.0
        self.seen = self._now()                          # how things are as the episode starts: not news

    def _now(self) -> tuple:
        st = self.r._call("state")
        f = st.get("fault") or {}
        return (f.get("reason") if f.get("operator") else None, bool(st.get("parked")))

    def poll(self):
        if time.time() < self.next:
            return
        self.next = time.time() + 1.0
        try:
            now = self._now()
        except Exception:                                # noqa: BLE001 -- the launcher's ping notices a lost robotd
            return
        if now == self.seen:
            return
        held_before, (stop, parked) = any(self.seen), now
        self.seen = now
        if stop or parked:
            kind = stop.split("(")[-1].rstrip(")") if stop and "(" in stop else stop      # "operator stop (soft)"
            what = " and ".join(x for x in ((f"stopped the arms ({kind})" if stop else ""),
                                            ("parked the arms" if parked else "")) if x)
            text = (f"[operator] The operator {what}. Every command is refused (robot.Fault) until they release "
                    "them. This is a pause, not the end of the episode: you are told when they release.")
        elif held_before:
            text = "[operator] The operator released the arms: commands work again."
        else:
            return
        try:
            self.deliver(text)
        except Exception:                                # noqa: BLE001
            pass


class ProgramEvents:
    """Tells the agent, in its running turn, when its program pauses or ends: the runner's status.json
    (WORKSPACE/.robot/current), read every half second. No polling on the agent's side. An ended program's own last
    `[state]` and `[views]` lines (robot and robot.viz print them as it exits) come with the message."""

    def __init__(self, ws: str, deliver):
        self.ws = ws
        self.path = os.path.join(ws, ".robot", "current", "status.json")
        self.deliver = deliver
        self.seen = None

    def _exit_lines(self, run_dir) -> list:
        """The last [state] and [views] lines of the run's log (its path as the sandbox sees it: under /workspace)."""
        if not run_dir:
            return []
        rel = str(run_dir).split("/workspace/", 1)[-1] if "/workspace/" in str(run_dir) else str(run_dir)
        log = os.path.join(self.ws, rel.lstrip("/"), "log.txt")
        try:
            with open(log, "rb") as f:
                f.seek(max(0, os.path.getsize(log) - 20000))
                tail = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []
        out = []
        for tag in ("[state] ", "[views] "):
            last = next((l.strip() for l in reversed(tail) if l.strip().startswith(tag)), None)
            if last:
                out.append(last[:4000])
        return out

    def poll(self):
        try:
            with open(self.path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return
        key = (st.get("run_dir"), st.get("state"), st.get("pause"))
        if key == self.seen:
            return
        first, self.seen = self.seen is None, key
        if st.get("state") == "running":
            return
        run_id = os.path.basename(str(st.get("run_dir") or ""))
        if first and time.time() - float(st.get("updated") or 0) > 5:    # an old run's status, not news
            return
        if st.get("state") == "paused":
            where = f"{os.path.basename(str(st.get('file') or ''))}:{st.get('line')}" if st.get("file") else ""
            text = (f"[program] paused ({st.get('kind') or 'pause'}) {where} in step {st.get('step')}: "
                    f"{st.get('message')} (cameras and state in .robot/runs/{run_id}/pause-{st.get('pause')}/)")
        elif st.get("state") == "exited":
            code = st.get("exit_code")
            text = (f"[program] ended (exit code {code})" + (f": {st.get('message')}" if code else "") +
                    f" (log in .robot/runs/{run_id}/log.txt)")
            text += "".join("\n" + line for line in self._exit_lines(st.get("run_dir")))
        else:
            return
        try:
            self.deliver(text)
        except Exception:                                # noqa: BLE001 -- the agent finds out by itself anyway
            pass


LOOK_LAPSE_S = 3.0


class ActionEvents:
    """The push: a kernel action's end report (robot/kernel.py writes WORKSPACE/.robot/events/*-action-*.json as the
    action ends: how it ended, what it printed, [state], the after-NAME camera grid and the hooks' output) goes to the
    agent as a message, with its images, into its running turn or as a new turn. No step of the agent's waits or polls.
    Each image is copied first (events/sent/), so that a later view of the same name cannot change it before Codex
    reads it. busy(): an action is running (EVENTS/actions.json), so the episode does not end while the agent waits
    for its report with no turn of its own."""

    def __init__(self, ws: str, deliver):
        self.ws, self.deliver = ws, deliver
        self.dir = os.path.join(ws, ".robot", "events")
        self.sent = os.path.join(self.dir, "sent")
        self.tries = {}
        self.log = []                                   # (file, written, delivered, how)
        self.look_asked = self.last_look = 0.0          # staggered: when a look was last asked for

    def _host(self, p: str) -> str:
        return os.path.join(self.ws, p[len(ar.WS) + 1:]) if p.startswith(ar.WS + "/") else p

    def _sandbox(self, host: str) -> str:
        return ar.WS + "/" + os.path.relpath(host, self.ws)

    def poll(self):
        try:
            names = sorted(n for n in os.listdir(self.dir) if n.endswith(".json") and "-action-" in n)
        except OSError:
            return
        for n in names:
            path = os.path.join(self.dir, n)
            try:
                with open(path) as f:
                    ev = json.load(f)
            except (OSError, ValueError):
                continue
            text = (ev.get("text") or f"[action {ev.get('name')}] {ev.get('status')}\n").rstrip("\n")
            images = []
            os.makedirs(self.sent, exist_ok=True)
            for k, p in enumerate(ev.get("images") or []):
                dst = os.path.join(self.sent, f"{n[:-5]}-{k}{os.path.splitext(p)[1] or '.png'}")
                try:
                    shutil.copyfile(self._host(p), dst)
                    images.append(self._sandbox(dst))
                except OSError:
                    pass
            if ev.get("images"):
                text += "\n[views] " + json.dumps(ev["images"])
            try:
                how = self.deliver(text, images)
            except Exception as e:                      # noqa: BLE001 -- again at the next tick, three times
                self.tries[n] = self.tries.get(n, 0) + 1
                if self.tries[n] < 3:
                    continue
                how = f"failed: {e!r}"[:300]
            self.log.append({"file": n, "written": ev.get("written"), "delivered": round(time.time(), 3),
                             "how": how, "report_s": ev.get("report_s"), "t1": ev.get("t1")})
            os.makedirs(os.path.join(self.dir, "delivered"), exist_ok=True)
            os.replace(path, os.path.join(self.dir, "delivered", n))
            if ev.get("phase") == "look":
                self.look_asked = 0.0

    def running(self) -> bool:
        """An action is running now (EVENTS/actions.json)."""
        try:
            with open(os.path.join(self.dir, "actions.json")) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return False
        return bool(st.get("running")) and time.time() < float(st.get("deadline") or 0)

    def look_if_idle(self, idle: bool, every_s: float = 0.0) -> bool:
        """staggered: the agent's turn is over and an action runs, so nothing would start its next one until that action
        ends: ask the kernel for a look (robot/kernel.py), which comes as a message and starts a turn. One at a time, at
        most one every every_s; a request that gets no answer (the action ended first) lapses after LOOK_LAPSE_S."""
        now = time.time()
        if not idle or not self.running() or self.pending():
            return False
        if self.look_asked and now - self.look_asked < LOOK_LAPSE_S:
            return False
        if self.last_look and now - self.last_look < every_s:
            return False
        try:
            with open(os.path.join(self.dir, "look.request"), "w"):
                pass
        except OSError:
            return False
        self.look_asked = self.last_look = now
        return True

    def pending(self) -> bool:
        """A report not yet delivered."""
        try:
            return any(n.endswith(".json") and "-action-" in n for n in os.listdir(self.dir))
        except OSError:
            return False

    def busy(self) -> bool:
        """An action running, or a report not yet delivered."""
        return self.pending() or self.running()


# ------------------------------------------------------------------------------------------------ the episode
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(real_yam.TASKS), help="a real-cell task (tasks/real_yam.py)")
    ap.add_argument("--sim", metavar="SCENE", help="instead of the cell: a simulated station running this task's scene "
                    "(tasks/, e.g. fold_towel), started here; its own success check")
    ap.add_argument("--seed", type=int, default=None, help="--sim: the scene's seed")
    ap.add_argument("--lockstep", action="store_true", help="--sim: lockstep (the world waits for the agent); the "
                    "default is the wall clock, as on the cell")
    ap.add_argument("--gpu", type=int, default=0, help="--sim: the GPU that renders the cameras (EGL)")
    ap.add_argument("--separation-fine", type=float, default=None, metavar="M",
                    help="--sim: keep the arms apart as on the cell, with the hands' pairs measured on their meshes, at "
                    "least M m apart")
    ap.add_argument("--scene-arg", action="append", default=[], metavar="KEY=VALUE", help="--sim: a keyword for the scene")
    ap.add_argument("--tag", help="run directory name under " + ar.RUNS)
    ap.add_argument("--agent", choices=["codex", "standin", "prompt-input"], default="codex",
                    help="codex (the default); standin: a script that probes the sandbox and sends no robot command; "
                    "prompt-input: Codex renders its model-visible input (no model call)")
    ap.add_argument("--prompt", default=ar.PROMPT, help="the first user message (AGENTS.md carries the rest)")
    ap.add_argument("--robot", default="tcp:127.0.0.1:7410", help="robotd's address")
    ap.add_argument("--towel", default=None, metavar="[WxH][:COLOUR]",
                    help=f"the towel's size in cm and colour (default {real_yam.TOWEL['w_cm']}x{real_yam.TOWEL['h_cm']}"
                    f":{real_yam.TOWEL['colour']}); the camera check knows red, blue and green")
    ap.add_argument("--instruction", help="the task text instead of the task's own")
    ap.add_argument("--wall-h", type=float, default=1.0, help="hours of wall time for the agent")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default=None,
                    help=f"the model's reasoning effort (default {ar.EFFORT}, as the simulation runs)")
    ap.add_argument("--playbook", action="store_true",
                    help="add docs/AGENT_PLAYBOOK.md to AGENTS.md: a worked guide to the kernel, the queue and the "
                         "reports from the runs that went best (for models that need more guidance than astra)")
    ap.add_argument("--llm-model", default=None, metavar="MODEL", help=f"the model, e.g. gpt-6.1-sol (default {ar.MODEL})")
    ap.add_argument("--fast", action="store_true", help="Codex's fast mode (service_tier \"fast\")")
    ap.add_argument("--service-tier", default=None, metavar="TIER",
                    help="any service tier the account offers (e.g. the ultrafast one); overrides --fast")
    ap.add_argument("--codex-home", default=None, metavar="DIR",
                    help="the Codex login to use: DIR/auth.json (default %(default)s; another account: log in with "
                    "CODEX_HOME=DIR codex login)")
    ap.add_argument("--exec", action="store_true", help="run `codex exec` (no steering), as the simulation runs do")
    ap.add_argument("--no-guidance", action="store_true", help="leave out the short 'Before you act' section "
                    "(facts only)")
    ap.add_argument("--advice", action="store_true", help="add the section on how to work on the real robot "
                    "(REAL_WORK, as on 2026-11-05) before the cell's description; by default the prompt is facts only")
    ap.add_argument("--linger-s", type=float, default=60.0,
                    help="steering: after the agent's last turn, seconds to wait for an operator message")
    ap.add_argument("--release-park", action="store_true",
                    help="release an operator park before the episode (an unattended series: each episode ends parked)")
    ap.add_argument("--keep-park", action="store_true",
                    help="start with the arms parked and leave them so (latched: every robot command is refused): "
                    "a test of the harness that cannot move anything")
    ap.add_argument("--witness-s", type=float, default=WITNESS_S)
    ap.add_argument("--no-video", action="store_true", help="no timelapse video (scripts/record_episode.py)")
    ap.add_argument("--video-every", type=float, default=0.5, help="seconds between video frames")
    ap.add_argument("--video-speed", type=float, default=4.0, help="the video plays at this x real time")
    ap.add_argument("--streams-fps", type=float, default=STREAMS_FPS,
                    help="the agent's camera recordings (RUN/cameras/, read-only at /var/record/cameras): frames a "
                    "second per camera; 0: none")
    ap.add_argument("--streams-depth-fps", type=float, default=STREAMS_DEPTH_FPS,
                    help="RGB-D grabs a second in the recordings (every camera's colour, depth, K and pose, in "
                    "RUN/cameras/rgbd/); 0: colour only")
    ap.add_argument("--no-events", action="store_true",
                    help="do not tell the agent when its program pauses or ends (nor push its actions' ends)")
    ap.add_argument("--no-push", action="store_true",
                    help="kernel actions' end reports are not sent as messages: the call that waits on an action gets "
                    "its report (the prompt says so), as before 2026-11-08")
    ap.add_argument("--staggered", "--pipelined", action="store_true",
                    help="pipelining, as in the demos: the agent queues its next action while the running one moves "
                    "(kernel.queue), and an action's end report is sent as the next one starts")
    ap.add_argument("--look-every-s", type=float, default=0.0, metavar="S",
                    help="--staggered: at most one look every S seconds (a look: the state, the cameras and the queue, "
                    "sent when the agent's turn ends while an action runs); 0: as soon as its turn ends")
    ap.add_argument("--push-detail", choices=["auto", "high", "original"], default="auto",
                    help="the detail of a pushed report's images (Codex 0.165 drops an image sent at low detail)")
    ap.add_argument("--mem-cap-gb", type=float, default=MEM_CAP_GB, help="the sandbox's memory cap, GB")
    ap.add_argument("--mem-cap-process-gb", type=float, default=MEM_CAP_PROCESS_GB, help="one process's memory cap, GB")
    ap.add_argument("--no-time-facts", action="store_true",
                    help="leave out the short section on what steps and images cost and how to spend fewer")
    ap.add_argument("--reasoning-summary", choices=["none", "auto", "concise", "detailed"], default="none",
                    help="Codex's reasoning summaries (default none: 'detailed' adds ~1.7 s to every step that "
                         "reasons, for the monitor's thinking text)")
    ap.add_argument("--prune-images", type=int, default=20, metavar="N",
                    help="after N new images, at a quiet moment, carry on in a new thread without the older images "
                         "(scripts/agent/codex_driver.py; 0: never)")
    ap.add_argument("--prune-keep", type=int, default=4, metavar="N", help="the newest images a prune keeps")
    ap.add_argument("--print", choices=["agents-md", "all"])
    a = ap.parse_args()
    if bool(a.task) == bool(a.sim):
        ap.error("one of --task (the real cell) or --sim SCENE (a simulated station)")
    memory = (a.mem_cap_process_gb, a.mem_cap_gb)
    if a.codex_home:
        ar.CODEX_AUTH = os.path.join(os.path.realpath(os.path.expanduser(a.codex_home)), "auth.json")
    tier = a.service_tier or ("fast" if a.fast else None)
    clock = ("lockstep" if a.lockstep else "realtime") if a.sim else None
    towel = dict(real_yam.TOWEL)
    if a.towel:
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?))?:?(red|blue|green)?", a.towel)
        if not m or not (m.group(1) or m.group(3)):
            ap.error("--towel: WxH, WxH:COLOUR or COLOUR (sizes in cm), e.g. 30x22:red or blue")
        if m.group(1):
            towel.update(w_cm=float(m.group(1)), h_cm=float(m.group(2)))
        else:                                        # a colour alone: the size is not known, and not stated
            towel.update(w_cm=None, h_cm=None)
        towel.update(colour=m.group(3) or towel["colour"])
    if a.sim:
        task = sim_task(a.sim)
        instruction = a.instruction or task.instruction_for(a.seed)
        towel = None
    else:
        task = real_yam.TASKS[a.task]
        instruction = a.instruction or task.instruction(towel)
    llm = ar.llm_spec("gpt", None, a.tag or "<tag>")
    if a.effort:
        llm["effort"] = a.effort
    if a.llm_model:
        llm["model"] = a.llm_model
    # the push: kernel actions' end reports as messages (needs the app-server driver: not with --exec)
    push = a.agent == "codex" and not a.exec and not a.no_events and not a.no_push
    if a.staggered and not push:
        ap.error("--staggered needs the push: the Codex agent with steering, without --no-events or --no-push")
    if a.print:
        print(agents_md(instruction, a.wall_h, "<launch time, UTC>", memory, a.streams_fps, a.advice, not a.exec,
                        not a.no_guidance, not a.no_time_facts, sim=clock, push=push, staggered=a.staggered,
                        playbook=a.playbook))
        if a.print == "all":
            print("\n" + "=" * 100 + "\nconfig.toml:\n" + codex_config_one_layer(llm, tier, a.reasoning_summary))
        return
    if not a.tag:
        ap.error("--tag is required for a run")
    run = os.path.join(ar.RUNS, a.tag)
    if os.path.exists(run):
        ap.error(f"{run} exists: pick another --tag")
    if a.agent == "codex" and not os.path.exists(ar.CODEX_AUTH):
        ap.error(f"no Codex login at {ar.CODEX_AUTH}: CODEX_HOME={os.path.dirname(ar.CODEX_AUTH)} "
                 f"{ar.CODEX_PKG}/bin/codex login --device-auth")
    if not os.path.exists(BWRAP):
        ap.error(f"{BWRAP} is missing")

    # ---- a simulated station (--sim): its robotd, started here
    sim_proc = sim_ctl = None
    if a.sim:
        sim_proc, sim_ctl, a.robot = start_sim_robotd(a, run)
    # ---- the robot, before anything is set up
    r = robot.connect(a.robot, name="agent_run_real")
    desc = r.description
    state = arms_state(r)
    if state["fault"]:
        ap.error(f"robotd has a latched fault: {state['fault'].get('reason')}")
    if state["busy"]:
        ap.error(f"something is commanding the arms ({state['busy']}): stop it first")
    if state["parked"] and not (a.release_park or a.keep_park):
        ap.error("the operator parked the arms: release them on the dashboard, or pass --release-park")
    record = robotd_record_dir(a.robot)
    req_path = os.path.join(record, "requests.jsonl") if record else None
    req_offset = os.path.getsize(req_path) if req_path and os.path.exists(req_path) else 0
    cam_addr = desc.get("camera_addr")
    if not cam_addr:
        if (r._call("health").get("backend") or {}).get("sim"):
            cam_addr = a.robot                       # a simulated robotd serves its own cameras: harness tests
        else:
            ap.error("robotd serves no camera_addr: start robotd with --cameras (camerad)")

    # ---- run directory (agent_run.py's layout)
    for sub in ("workspace", "sandbox", "logs", "codex/home", "robotd"):
        os.makedirs(os.path.join(run, sub))
    tmp = tempfile.mkdtemp(prefix=f"robotapi-{a.tag[:40]}-", dir="/tmp")
    for sub in ("sock", "egress", "host", "tmp"):
        os.makedirs(os.path.join(tmp, sub), mode=0o700)
    code = os.path.join(run, "sandbox", "robot-api")
    for rel in ar.AGENT_CODE + ["docs/REAL_CELL.md"]:
        src, dst = os.path.join(ROOT, rel), os.path.join(code, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ar.AGENT_CODE_SKIP))
        else:
            shutil.copy2(src, dst)
    with open(os.path.join(run, "robotd", "README"), "w") as f:
        f.write(f"robotd's recording is {record}; this episode's requests start at byte {req_offset} of its "
                "requests.jsonl\n")
    t_start = time.time()
    ws, history = os.path.join(run, "workspace"), os.path.join(run, "workspace.git")
    password = secrets.token_urlsafe(24)
    config = codex_config_one_layer(llm, tier, a.reasoning_summary)
    with open(os.path.join(run, "codex", "home", "config.toml"), "w") as f:
        f.write(config)
    open(os.path.join(run, "codex", "home", "auth.json"), "w").close()     # the mount point for the login's bind
    if a.agent != "codex" and not os.path.exists(ar.CODEX_AUTH):          # a stand-in without a login: an empty one
        ar.CODEX_AUTH = os.path.join(tmp, "no-login.json")
        open(ar.CODEX_AUTH, "w").close()
    stop_r, stop_w = os.pipe()
    steer = a.agent == "codex" and not a.exec
    inner = (["/bin/bash", f"{ar.HARNESS}/standin_real.sh", *HIDDEN, record or "/nonexistent"]
             if a.agent == "standin" else [f"{ar.CODEX_DIR}/bin/codex", "app-server", "--stdio"] if steer
             else ar.agent_argv(a.agent, a.prompt, run, llm))
    inbox = os.path.join(run, "steer", "inbox.jsonl")
    if steer:
        os.makedirs(os.path.dirname(inbox))
        open(inbox, "w").close()
    meta = {"schema": "robotapi.agent-run.v1", "tag": a.tag, "task": a.task or a.sim, "real": not a.sim,
            "staggered": a.staggered, "playbook": a.playbook, "look_every_s": a.look_every_s if a.staggered else None,
            "towel": towel, "sim": ({"scene": a.sim, "seed": a.seed, "clock": clock, "scene_args": a.scene_arg,
                                     "gpu": a.gpu} if a.sim else None),
            "instruction": instruction, "agent": a.agent, "clock": clock or "real", "repo": ar.repo_state(),
            "host": socket.gethostname(),
            "codex": {"version": ar.codex_version(), "model": llm["model"], "reasoning_effort": llm["effort"],
                      "service_tier": tier or "default", "login": os.path.dirname(ar.CODEX_AUTH),
                      "argv": inner, "config": config, "prompt": a.prompt,
                      "steering": {"inbox": "steer/inbox.jsonl", "linger_s": a.linger_s} if steer else None,
                      "reasoning_summary": a.reasoning_summary,
                      "prune": {"after_images": a.prune_images, "keep": a.prune_keep} if steer else None},
            "sandbox": {"kind": "bwrap, one layer", "bwrap": BWRAP, "network": "egress-proxy",
                        "egress_allow": llm["egress"], "argv": "sandbox/bwrap.json",
                        "why_one_layer": "AppArmor here: no bwrap inside another sandbox (module docstring)"},
            "robotd": {"addr": a.robot, "record": record, "requests_offset": req_offset, "camera_addr": cam_addr,
                       "health_start": r._call("health").get("loop")},
            "arms_start": state, "limits": {"wall_s": a.wall_h * 3600, "memory_gb": a.mem_cap_gb,
                                            "process_memory_gb": a.mem_cap_process_gb},
            "state": "starting", "started_utc": ar.utc(t_start), "ended_utc": None, "notes": []}
    meta_path = os.path.join(run, "meta.json")
    ar.write_json(meta_path, meta)
    print(f"run dir: {run}", flush=True)

    notes = meta["notes"]
    agent = proxy = watchdog = witness = video = streams = t_agent = None
    forwards = []
    end_reason, rc = None, None
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        # ---- the sockets the sandbox reaches, the egress proxy, the scene before the agent
        forwards = [Forward(f"{tmp}/sock/robot.sock", *tcp(a.robot)), Forward(f"{tmp}/sock/cameras.sock", *tcp(cam_addr))]
        proxy = ar.EgressProxy(f"{tmp}/egress/proxy.sock", password, llm["egress"],
                               os.path.join(run, "codex", "egress.jsonl"))
        models = ar.model_files(desc)
        meta["sandbox"]["model_files"] = models
        if state["parked"] and not a.keep_park:
            r._call("release")
            notes.append("released the operator's park before the episode (--release-park)")
            meta["arms_start"] = arms_state(r)
        before = None if a.sim else task.measure(r, towel)
        ar.write_json(os.path.join(run, "before.json"), before)
        meta["before"] = before
        host_log = os.open(os.path.join(run, "logs", "program-host.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        cams_dir = os.path.join(run, "cameras") if a.streams_fps > 0 else None
        if cams_dir:                                     # the recorder fills it; the sandbox sees it read-only
            os.makedirs(cams_dir)
            streams = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "record_episode.py"),
                                        "--streams", cams_dir, "--run", run, "--robot", a.robot,
                                        "--fps", str(a.streams_fps), "--depth-fps", str(a.streams_depth_fps)],
                                       stdout=open(os.path.join(run, "logs", "streams.log"), "w"),
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            for _ in range(100):                         # its index.json before the agent starts
                if os.path.exists(os.path.join(cams_dir, "index.json")) or streams.poll() is not None:
                    break
                time.sleep(0.1)
            if not os.path.exists(os.path.join(cams_dir, "index.json")):
                notes.append("the camera recordings did not start (logs/streams.log)")
        if push:
            os.makedirs(os.path.join(ws, ".robot", "events"), exist_ok=True)
        argv = sandbox_argv(run, tmp, inner, stop_r, host_log, password, models, llm, a.agent == "standin",
                            cams_dir, events=f"{ar.WS}/.robot/events" if push else None, staggered=a.staggered)
        ar.write_json(os.path.join(run, "sandbox", "bwrap.json"), [x.replace(password, "<password>") for x in argv])
        have = list(desc.get("cameras") or WITNESS_CAMERAS)   # the sim's station has no front camera
        wcams = [c for c in WITNESS_CAMERAS if c in have] or have[:2]
        witness = Witness(a.robot, os.path.join(run, "witness"), a.witness_s, cameras=wcams)
        witness.start()
        if not a.no_video:                               # stops itself once meta.json says done (or on SIGTERM)
            video = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "record_episode.py"),
                                      "--run", run, "--robot", a.robot, "--every", str(a.video_every),
                                      "--cameras", *wcams,
                                      "--speed", str(a.video_speed)],
                                     stdout=open(os.path.join(run, "logs", "video.log"), "w"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)

        # ---- the agent: its clock starts with AGENTS.md
        if stopping.is_set():
            end_reason = "interrupted"
            raise RuntimeError("interrupted before the agent started")
        t_agent = time.time()
        with open(os.path.join(ws, "AGENTS.md"), "w") as f:
            f.write(agents_md(instruction, a.wall_h, ar.utc(t_agent) + " UTC", memory, a.streams_fps, a.advice,
                              not a.exec, not a.no_guidance, not a.no_time_facts, sim=clock, push=push, staggered=a.staggered,
                              playbook=a.playbook))
        for gd in (None, history):
            ar.git(ws, "init", "-q", "-b", "main", git_dir=gd)
            ar.git(ws, "add", "AGENTS.md", git_dir=gd)
            ar.git(ws, "commit", "-q", "-m", "episode start", git_dir=gd)
        meta.update(state="running", agent_started_utc=ar.utc(t_agent))
        ar.write_json(meta_path, meta)
        events = None if steer else open(os.path.join(
            run, "codex", "prompt-input.json" if a.agent == "prompt-input" else "events.jsonl"), "w")
        stderr = open(os.path.join(run, "codex", "stderr.log"), "w")
        agent = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr,
                                 stdin=subprocess.PIPE if steer else subprocess.DEVNULL,
                                 pass_fds=(stop_r, host_log), start_new_session=True)
        watchdog = ar.MemoryWatchdog(agent.pid, a.mem_cap_process_gb, a.mem_cap_gb,
                                     os.path.join(run, "logs", "memory.jsonl"), notes, t_agent)
        os.close(stop_r)
        os.close(host_log)
        stats = {"events": 0, "thread_id": None, "usage": {}, "turns": 0, "errors": []}

        def read_events():
            for line in agent.stdout:
                events.write(line.decode("utf-8", "replace"))
                events.flush()
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                stats["events"] += 1
                if ev.get("type") == "thread.started":
                    stats["thread_id"] = ev.get("thread_id")
                elif ev.get("type") == "turn.completed":
                    stats["turns"] += 1
                    for k, v in (ev.get("usage") or {}).items():
                        if isinstance(v, (int, float)):
                            stats["usage"][k] = stats["usage"].get(k, 0) + v
                elif ev.get("type") in ("turn.failed", "error"):
                    stats["errors"].append(ev.get("error") or ev.get("message"))
        driver = action_events = None
        if steer:                                        # the driver reads the app-server; its stats are these
            driver = CodexDriver(agent, os.path.join(run, "codex", "events.jsonl"),
                                 os.path.join(run, "codex", "app-server.jsonl"), inbox, a.prompt, ar.WS, a.linger_s,
                                 codex_home=os.path.join(run, "codex", "home"), prune_after=a.prune_images,
                                 prune_keep=a.prune_keep)
            stats = driver.stats
            driver.start()
            reader = None
        else:
            reader = threading.Thread(target=read_events, daemon=True)
            reader.start()
        program_events = None if (driver is None or a.no_events) else \
            ProgramEvents(ws, lambda text: driver.deliver(text, source="program"))
        operator_events = None if (driver is None or a.no_events) else \
            OperatorEvents(r, lambda text: driver.deliver(text, source="program"))
        action_events = ActionEvents(ws, lambda text, images: driver.deliver(text, source="action", images=images,
                                                                             detail=a.push_detail)) \
            if (driver is not None and push) else None
        deadline, next_ping = t_agent + a.wall_h * 3600, 0.0
        while True:
            if agent.poll() is not None:
                end_reason = "agent_exit"
                break
            if driver is not None:
                driver.poll_inbox()                      # the operator's messages, as they come
                if program_events is not None:
                    program_events.poll()                # the program paused or ended: told in the turn
                if operator_events is not None:
                    operator_events.poll()               # the operator stopped, parked or released the arms
                if action_events is not None:
                    action_events.poll()                 # an action ended: its report, as a message
                driver.maybe_prune()                     # older images out of the thread, at a quiet moment
                if action_events is not None and a.staggered:
                    action_events.look_if_idle(driver.turn_id is None and not driver.pruning, a.look_every_s)
                if driver.finished() and not (action_events is not None and action_events.busy()):
                    end_reason = "agent_done"            # (not while an action runs: its report starts a turn)
                    break
            if time.time() > deadline:
                end_reason = "wall_limit"
                break
            if stopping.is_set():
                end_reason = "interrupted"
                break
            if time.time() >= next_ping:                 # robotd still there?
                try:
                    r._call("ping")
                except Exception as e:                   # noqa: BLE001
                    end_reason = "robotd_gone"
                    notes.append(f"robotd stopped answering: {e!r}")
                    break
                next_ping = time.time() + 10
            time.sleep(0.1 if action_events is not None else 0.5 if driver is not None else 1.0)
        if driver is not None:                           # its stdin closed, the app-server exits, and the sandbox
            driver.close()                               # with it (sandbox_init.py stops the program host first)
            if end_reason == "agent_done":
                try:
                    agent.wait(30)
                except subprocess.TimeoutExpired:
                    pass
        if agent.poll() is None:                         # sandbox_init.py stops the agent, then the program host
            os.write(stop_w, b"x")
            try:
                agent.wait(60)
            except subprocess.TimeoutExpired:
                notes.append("the sandbox did not exit 60 s after the stop request: killed")
                os.killpg(agent.pid, signal.SIGKILL)
                agent.wait(10)
        rc = agent.returncode
        if reader is not None:
            reader.join(10)
        if events is not None:
            events.close()
        stderr.close()
        if a.agent == "prompt-input":
            ar.write_prompt_md(os.path.join(run, "codex", "prompt-input.json"))
        t_end = time.time()
        meta.update(agent_ended_utc=ar.utc(t_end), agent_wall_s=round(t_end - t_agent, 1), end_reason=end_reason,
                    agent_exit_code=rc, codex_thread_id=stats["thread_id"], codex_events=stats["events"],
                    codex_turns=stats["turns"], codex_usage=stats["usage"], codex_errors=stats["errors"][:20],
                    **({"operator_messages": stats["operator_messages"], "codex_compactions": stats["compactions"],
                        "program_events": stats.get("program_events", []), "codex_threads": stats.get("threads", []),
                        "codex_prunes": stats.get("prunes", []),
                        "action_pushes": action_events.log if action_events is not None else None}
                       if driver is not None else {}))
    except Exception as e:                               # noqa: BLE001 -- still park, check and record
        import traceback
        traceback.print_exc()
        end_reason = end_reason or "runner_error"
        notes.append(f"runner error: {e!r}")
        meta["end_reason"] = end_reason
        if agent is not None and agent.poll() is None:
            os.killpg(agent.pid, signal.SIGKILL)
    finally:
        if agent is not None:
            meta["program_host"] = ar.program_host_summary(os.path.join(run, "logs", "program-host.jsonl"), notes)
        # ---- the arms parked (latched), then the check from the cameras
        verdict = None
        if a.sim:
            verdict = sim_verdict(task, run, sim_proc, sim_ctl, record, req_offset, t_agent or t_start, notes)
        try:
            if a.sim:
                raise StopIteration                      # no park, no camera check: the simulation judged it
            rr = robot.connect(a.robot, name="agent_run_real/end")    # fresh: the episode's may be mid-reply
            # the stand-in and prompt-input send no robot command: nothing to park
            meta["park_end"] = (park(rr, notes) if t_agent is not None and a.agent == "codex" and not a.keep_park
                                else None)
            time.sleep(1.0)
            after = task.measure(rr, towel)
            fin = finishes(record, req_offset, t_agent or t_start)
            verdict = {"check": task.check(meta.get("before"), after, towel), "after": after,
                       "before": meta.get("before"), "finished": fin[-1] if fin else None,
                       "operator_verdict": None, "evaluated_utc": ar.utc()}
            ar.write_json(os.path.join(run, "verdict.json"), verdict)
            meta["robotd"]["health_end"] = rr._call("health").get("loop")
            meta["arms_end"] = arms_state(rr)
        except StopIteration:
            pass
        except Exception as e:                           # noqa: BLE001
            notes.append(f"park or check failed: {e!r}")
        if witness is not None:
            witness.stop.set()
            witness.join(15)
            meta["witness"] = {"frames": witness.frames, "period_s": a.witness_s, "errors": witness.errors}
        for fw in forwards:
            fw.close()
        if proxy is not None:
            proxy.close()
        if watchdog is not None:
            watchdog.close()
            meta["memory"] = watchdog.summary()
        try:
            os.close(stop_w)
        except OSError:
            pass
        stub = os.path.join(run, "codex", "home", "auth.json")
        if os.path.exists(stub) and os.path.getsize(stub) == 0:
            os.remove(stub)
        try:
            if os.path.isdir(history):
                ar.git(ws, "add", "-A", git_dir=history)
                ar.git(ws, "commit", "-q", "--allow-empty", "-m", f"episode end ({end_reason})", git_dir=history)
        except subprocess.CalledProcessError as e:
            notes.append(f"workspace commit failed: {e.stderr.strip()}")
        try:
            ar.keep_tmp(f"{tmp}/tmp", os.path.join(run, "sandbox", "tmp"), notes)
        except OSError as e:
            notes.append(f"keeping the sandbox's /tmp failed: {e!r}")
        shutil.rmtree(tmp, ignore_errors=True)
        finished = (verdict or {}).get("finished")
        if a.agent == "codex" and not finished:
            notes.append("the agent never called r.finish()")
        meta.update(state="done", ended_utc=ar.utc(), wall_s=round(time.time() - t_start, 1), finished=finished,
                    check=(verdict or {}).get("check"), **({"verdict": (verdict or {}).get("verdict")} if a.sim else {}))
        ar.write_json(meta_path, meta)
        if video is not None:                            # it saw state done; else stop it
            try:
                video.wait(15)
            except subprocess.TimeoutExpired:
                video.send_signal(signal.SIGTERM)
                try:
                    video.wait(30)
                except subprocess.TimeoutExpired:
                    video.kill()
            path = os.path.join(run, f"video-{a.video_speed:g}x.mp4")
            meta["video"] = {"path": os.path.basename(path), "bytes": os.path.getsize(path) if os.path.exists(path)
                             else None, "every_s": a.video_every, "speed": a.video_speed}
            ar.write_json(meta_path, meta)
        if streams is not None:                          # the agent's recordings: same ending
            try:
                streams.wait(15)
            except subprocess.TimeoutExpired:
                streams.send_signal(signal.SIGTERM)
                try:
                    streams.wait(30)
                except subprocess.TimeoutExpired:
                    streams.kill()
            cams = os.path.join(run, "cameras")
            meta["camera_streams"] = {"dir": "cameras", "fps": a.streams_fps,
                                      "bytes": sum(os.path.getsize(os.path.join(cams, f)) for f in os.listdir(cams))
                                      if os.path.isdir(cams) else None}
            ar.write_json(meta_path, meta)
        if sim_proc is not None:                         # the recorders have stopped: the simulation can go
            stop_sim_robotd(sim_proc, sim_ctl, notes)
            ar.write_json(meta_path, meta)
        # ---- the run's log: robotd's slice of the episode (waiting a little for its last tick file) and the timeline;
        # then any earlier run left incomplete
        try:
            info = run_log.bundle(run, wait_s=90.0, quiet=True)
            print(f"log: timeline.jsonl, TIMELINE.md ({info.get('timeline_events')} events); robotd/ "
                  f"({info.get('requests')} requests, {len(info.get('ticks') or [])} tick files, complete "
                  f"{info.get('complete')})")
            run_log.refresh(os.path.dirname(run), quiet=True)
        except Exception as e:                           # noqa: BLE001 -- the episode is over; the log can be redone
            print(f"log failed ({e!r}): python scripts/run_log.py {run}")
        try:                                             # where the time went (profile.json): model, tools, motion
            import profile_run
            print(profile_run.summary(profile_run.profile(run)))
            with open(os.path.join(run, "profile.json"), "w") as f:
                json.dump(profile_run.profile(run), f, indent=1)
        except Exception as e:                           # noqa: BLE001
            print(f"profile failed ({e!r}): python scripts/profile_run.py {run}")
        print(f"end: {end_reason}, agent exit code {rc}; " + (f"verdict: {json.dumps((verdict or {}).get('verdict'))}"
              if a.sim else f"check: {json.dumps((verdict or {}).get('check'))}"))
        print(f"run dir: {run}")


if __name__ == "__main__":
    main()
