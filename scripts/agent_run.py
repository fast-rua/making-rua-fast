"""In this repository: the episode machinery that scripts/agent_run_real.py imports (the sandbox, the egress proxy,
the program host, the memory watchdog, the workspace history, AGENTS.md). Run episodes with agent_run_real.py, on the
real cell or with --sim SCENE. This file's own command line below is the launcher we used on a GPU cluster, with a
vision service and the Pi agent that this release does not include; it is not supported here.

Run a coding agent (Codex, or Pi) on a task: one episode for one (task, seed, tag).

    python scripts/agent_run.py --task pick_cube --seed 0 --tag pick_cube-s0-01      # an episode (on a GPU node)
    python scripts/agent_run.py --task pick_cube --print agents-md                  # the agent's instructions only
    python scripts/agent_run.py --task pick_cube --print all                        # + the Codex command and config
    python scripts/agent_run.py --task pick_cube --tag t1 --agent standin           # the sandbox with a scripted stand-in
    python scripts/agent_run.py --task pick_cube --tag p1 --agent prompt-input      # Codex's model-visible input
    python scripts/agent_run.py --task pick_cube --seed 0 --tag t2 --model deepseek --endpoint https://HOST:PORT/v1
                                                     # Codex on DeepSeek (DEEPSEEK) at an ora-router LLM allocation
    python scripts/agent_run.py --task pick_cube --seed 0 --tag t3-pi --agent pi --endpoint https://HOST:PORT/v1
                                                     # Pi (PI) on DeepSeek at an ora-router LLM allocation (below)
    python scripts/agent_run.py --task pick_cube --tag t4 --agent pi-standin --endpoint https://HOST:PORT/v1
                                                     # Pi's sandbox with the stand-in, which also runs Pi on a test image
    python scripts/agent_run.py ... --appendix scripts/agent/appendix-working-habits.md --examples DIR
                                                     # AGENTS.md + an appendix; DIR read-only at /opt/examples

Run it with the robot-api venv's Python. An episode:
1. creates RUNS/<tag>/ (below), with the workspace: a git repository holding AGENTS.md (scripts/agent/
   AGENTS.md.template filled in with the task instruction, the environment and docs/MANUAL.md verbatim; with
   --appendix FILE, FILE after the manual; with --examples DIR, a line pointing at /opt/examples/README.md, DIR
   bound there read-only; meta.json records both with their sha256);
2. starts robotd on the task's scene and seed (lockstep, or on the wall clock with --realtime, the default for a task
   with realtime=True; recording, EGL on --gpu, a control socket only this process reaches) and, unless --no-vision,
   the vision service (--vision-gpu); AGENTS.md's World paragraph says which;
3. runs `codex exec` (or Pi) in the sandbox (below) until it exits or --wall-h hours pass. An exec that ends, finished
   or not, is not resumed: meta.json records whether r.finish() was called;
4. evaluates the task's success check through robotd's control socket into verdict.json, stops robotd and vision,
   commits the workspace's end state (outside the sandbox) and completes meta.json.

RUNS/<tag>/
  meta.json      task, seed, instruction, clock (lockstep or realtime), repo commit, Codex command and config, sandbox,
                 times, end reason, usage, the programs the program host started and those it stopped at the end,
                 and in real time the simulation's timing (late ticks, from robotd/timing.json)
  verdict.json   the success check at the end of the episode, after SETTLE_S with the arms holding (simulated, or
                 wall time in real time), and the one robotd wrote at r.finish() (if called); for a task whose check
                 ends at r.finish() (ends_at_finish) the verdict is that one, with no settling
  workspace/     the agent's working directory, a git repo with the initial commit
  workspace.git/ the workspace's history outside the sandbox: the initial commit and then the end state
  robotd/        robotd's recording (docs/RUNS.md); robot-files/ holds the URDFs r.description points at
  vision/        the vision client's recording (requests.jsonl, arrays/<sha>.npz; see vision/__init__.py)
  codex/         events.jsonl (the --json stream), stderr.log, egress.jsonl (one line per proxy connection),
                 home/ (the episode's CODEX_HOME: config.toml, the session rollout, last_message.txt)
  sandbox/       robot-api/ (the code the agent sees at /opt/robot-api), bwrap.json (the exact sandbox command),
                 tmp/ (the sandbox's private /tmp at the end: the scripts and images an agent kept there; regular
                 files only, less caches, Codex's daemon directory and files over TMP_KEEP_MB)
  logs/          robotd.log, vision.log, program-host.jsonl (the program host's starts, exits and end-of-episode stops),
                 memory.jsonl (the memory watchdog's kills)
  pi/            (--agent pi, in place of codex/) events.jsonl (Pi's --mode json stream less its message_update and
                 tool_execution_update deltas), stderr.log, session.jsonl (Pi's session file), requests.jsonl (the
                 model proxy's log: one line per model request, its usage and the images it carried),
                 first-request.json (the first request's body: what the model saw first), last_message.txt, home/
                 (/pi-home: agent/ with models.json and settings.json, sessions/); sandbox/codex-home/ is the command
                 sandbox's CODEX_HOME (its config.toml only)

The sandbox: Codex runs under bwrap with an allowlisted filesystem and a network namespace of its own (loopback only).
  read-only   /usr, /etc, /sys; /opt/robot-api (a copy of robot/, robotd/core/ (the driver's control core) and
              docs/MANUAL.md, with vision also vision/ and docs/VISION.md; not robotd/backends|robots|dashboard (the
              simulator), examples/ (our solutions), tasks/ or scripts/); the three Pythons (VENVS), their base
              interpreter and the sources their path hooks name (i2rt, pyroki, sam2, ...); every file r.description
              names (the URDFs, the MJCF and their meshes), at their own paths; the Codex release at /opt/codex;
              with --examples DIR, DIR at /opt/examples
  read-write  /workspace; a private /tmp; /codex-home (CODEX_HOME, with the shared auth.json bound in); the vision
              recording at /var/record/vision; /run/program-host
  model API   --model gpt (the default): the shared ChatGPT login's auth.json is bound in at /codex-home/auth.json;
              --model deepseek: no auth.json, the internal CA at /opt/harness/ca.crt (CODEX_CA_CERTIFICATE) and the
              key in the first process's environment (bwrap --args from a pipe), which only Codex keeps: its commands
              and the program host get neither the key nor the provider variables
  sockets     /run/robot-api/robot.sock and vision.sock; /run/egress/proxy.sock, the egress proxy;
              /run/program-host/host.sock, the program host (ROBOT_PROGRAM_HOST)
  network     only the egress proxy: sandbox_init.py forwards 127.0.0.1:3128 to it for processes in its own PID
              namespace (Codex, not its commands), and the proxy CONNECTs to the --egress-allow hosts (port 443, or
              the gateway's host:port with --model deepseek) for whoever gives the episode's password, which only
              Codex's environment holds
Codex runs each command in a sandbox of its own (its own PID namespace): permission profile "robot" (config.toml) =
:workspace (writes to /workspace, its .git included, and /tmp only), auth.json, /run/egress and Codex's daemon
directory unreadable, network on (Codex's network-off mode also refuses unix sockets; the outer namespace has no
network anyway), and the proxy variables removed from commands' environment. Codex's bundled skills, connectors,
plugins, browsers, image generation and web search are off.
Everything a command started dies with it, so `python -m robot run` hands its program to the program host
(scripts/agent/program_host.py), which sandbox_init.py starts in the outer sandbox before the agent: the host runs
the program in a command sandbox of its own (`codex sandbox -P robot -C /workspace`, the command's environment less
the proxy variables, its working directory and log), keeps it until it exits, and stops every program still running
when the agent has ended, before the evaluation.
Memory: every 2 s the launcher sums the resident memory of the sandbox's processes (bwrap and all its descendants:
the agent, its commands, the program host's programs; not robotd or the vision server) and kills the largest when one
holds more than --mem-cap-process-gb (48) or all together more than --mem-cap-gb (64), as AGENTS.md says
(MemoryWatchdog; logs/memory.jsonl, meta.json memory and notes).

--agent pi: Pi runs in Codex's place in the same outer sandbox (binds, sockets, program host, GPU), headless (--mode
json), with AGENTS.md as its context file, its own read, bash, edit and write tools and nothing else (no extensions,
skills, prompt templates, MCP, codemode or project settings), on DeepSeek (DEEPSEEK) as an OpenAI-compatible chat
model with the request shape ORA's Pi provider gives that profile (ora-pi/src/play/runtime.ts): reasoning_effort
high, max_tokens 16384, a system message, streamed. Pi has no sandbox of its own, so:
  - its bash commands run through scripts/agent/pi_shell.sh in Codex's command sandbox, profile "robot", as a Codex
    agent's commands do, with /pi-home unreadable as well (Codex is only that sandbox here: CODEX_HOME holds its
    config.toml alone, no Codex agent runs and no model is reached through it); the program host is unchanged;
  - its write and edit tools are confined to /workspace and /tmp (scripts/agent/pi_guard.js);
  - its bash tool blocks until the command ends, and has no default time limit: a call that passes no timeout gets
    PI_BASH_TIMEOUT_S (600 s), one that passes a timeout keeps it (scripts/agent/pi_timeout.js); at the limit Pi
    kills the command (its Codex sandbox and everything in it; programs on the program host run on), and AGENTS.md
    says so in one line (PI_TIMEOUT_LINE), the one difference from Codex's;
  - a reply that spends all of max_tokens on reasoning ends with stopReason "length" and no tool call, which ends
    Pi's run: then a one-line harness note says so and asks for one more model request, at most 10 times an episode
    (scripts/agent/pi_length.js);
  - a model request carries the images of only the latest 3 tool results with images, as in ORA's Pi runtime (older
    ones become a line of text naming the file; scripts/agent/pi_images.js): sent in full, a long session's requests
    outgrow the gateway's body limit;
  - the key never enters the sandbox: Pi's provider is http://127.0.0.1:3128/v1 with the episode's token as its API
    key; sandbox_init.py bridges that port, for its own PID namespace only, to /run/egress/model.sock, and the model
    proxy here (ModelProxy) sends each request over TLS to the endpoint with the key and X-Session-Id (the tag). There
    is no egress proxy; the proxy log is the token count, per request and in total, in meta.json.
  read-only   + the node binary (NODE) at /opt/node/bin/node, Pi's node_modules (PI_MODULES) at /opt/pi/node_modules,
              pi_shell.sh and the extensions (PI_EXTENSIONS); /pi-home/agent/models.json and settings.json;
              /codex-home/config.toml
  read-write  /pi-home in place of /codex-home's Codex state: Pi's agent directory and sessions; /codex-home is
              the command sandbox's (sandbox/codex-home); no auth.json, CA or key
"""
from __future__ import annotations

import argparse
import ast
import base64
import ctypes
import datetime
import glob
import hashlib
import hmac
import http.client
import inspect
import json
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "scripts", "agent")
sys.path.insert(0, ROOT)

# Paths: one install folder, $ROBOT_HOME (default ~/robot-cell; docs/SETUP.md): agent-runs/, the Codex package in
# codex/current, the episodes' Codex login in codex-home/, the Python venv in .venv/
HOME = os.path.expanduser(os.environ.get("ROBOT_HOME", "~/robot-cell"))
RUNS = os.path.join(HOME, "agent-runs")
CODEX_PKG = os.path.realpath(os.path.join(HOME, "codex", "current"))
CODEX_AUTH = os.path.join(HOME, "codex-home", "auth.json")            # bound into the sandbox; never read here
ROBOT_VENV = os.path.join(HOME, ".venv")
PERCEPTION_VENV = ROBOT_LIBS_VENV = ROBOT_VENV                          # one venv for everything the agent runs
VENVS = [ROBOT_VENV]
CACHE_ENV = os.path.join(HOME, "cache.env")
MODEL, EFFORT = "gpt-6-astra", "xhigh"
EGRESS_ALLOW = ["chatgpt.com", "api.openai.com", "auth.openai.com"]
# --model deepseek (this script's own launcher only): a DeepSeek model served by vLLM behind an OpenAI-compatible
# endpoint (--endpoint), spoken to as a custom Codex model provider; the key is read from key_file (or key_env), the
# endpoint's CA from ca.
DEEPSEEK = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "effort": "high", "context_window": 1048576,
            "max_output_tokens": 16384, "key_env": "DEEPSEEK_API_KEY",
            "key_file": os.environ.get("DEEPSEEK_KEY_FILE", ""), "ca": os.environ.get("DEEPSEEK_CA", "")}
VISION_MODELS = "point,segment,features,detect,track"
AGENT_CODE = ["robot", "robotd/__init__.py", "robotd/core", "docs/MANUAL.md"]
AGENT_CODE_SKIP = ("__pycache__", "*.pyc")
MEM_CAP_GB, MEM_CAP_PROCESS_GB = 64, 48           # the sandbox's memory caps (MemoryWatchdog), GB of 2^30 bytes
TMP_KEEP_MB = 20                                    # sandbox/tmp keeps the agent's /tmp files up to this size
SETTLE_S = 2.0                                      # seconds, arms holding, before the success check (none for a
                                                    #   task whose check ends at r.finish())
PROMPT = "Do the task described in AGENTS.md."
# Codex features that reach past the sandbox (connectors, plugins, browsers) or add context of their own
DISABLED_FEATURES = ["apps", "plugins", "remote_plugin", "browser_use", "browser_use_external", "in_app_browser",
                     "computer_use", "image_generation", "daemon_auto_start", "tool_suggest", "memories"]
PROXY_VARS = ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]
# a custom provider's variables in Codex's environment, kept from its commands and the program host like PROXY_VARS
PROVIDER_VARS = [DEEPSEEK["key_env"], "CODEX_CA_CERTIFICATE", "SANDBOX_PRIVATE_VARS"]
# --agent pi: Pi (@earendil-works/pi-coding-agent 1.3.0, the version ORA's agent core pins, installed there) on Node,
# both bound read-only and never modified. Its provider is DEEPSEEK as ORA's Pi provider maps a profile
# (ora-pi/src/play/runtime.ts oraProvider): Chat Completions, a system message rather than "developer", no `store`,
# max_tokens, reasoning_effort; the profile has no request_options. Retries cover about the profile's 900 s recovery
# window (2, 4, ... 60 s apart).
NODE = shutil.which("node") or "/usr/bin/node"
PI_MODULES = os.environ.get("PI_MODULES", "")         # --agent pi only: Pi's node_modules
PI_PACKAGE = "@earendil-works/pi-coding-agent"
PI_PROVIDER, PI_TOOLS, PI_RETRIES = "ora-router", ["read", "bash", "edit", "write"], 19
PI_EXTENSIONS = ["pi_guard.js", "pi_images.js", "pi_timeout.js", "pi_length.js"]  # scripts/agent/, --extension
# Pi's bash time limit (pi_timeout.js DEFAULT_S: a call that passes no timeout gets it) and AGENTS.md's line for it;
# Codex's AGENTS.md has none
PI_BASH_TIMEOUT_S = 600
PI_TIMEOUT_LINE = ("\n- **Commands:** a command is killed after 10 minutes unless you pass a longer timeout; run long "
                   "computations in\n  the background (and poll), or as robot programs through the runner.")
# --examples DIR: bound read-only here, and AGENTS.md's line for it
EXAMPLES = "/opt/examples"
EXAMPLES_LINE = (f"\n- **Examples:** before you start, read {EXAMPLES}/README.md: how a stronger agent solved other "
                 "tasks on this robot,\n  step by step. Work in a similar way. The tasks differ, so don't copy their "
                 "numbers.")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# paths inside the sandbox
WS, CODE, SOCKS, EGRESS = "/workspace", "/opt/robot-api", "/run/robot-api", "/run/egress"
PROGRAM_HOST = "/run/program-host"
RECORD, CODEX_HOME, CODEX_DIR, HARNESS = "/var/record/vision", "/codex-home", "/opt/codex", "/opt/harness"
PI_HOME, NODE_DIR, PI_DIR = "/pi-home", "/opt/node", "/opt/pi"
PROXY_PORT = 3128


def utc(t: float | None = None) -> str:
    return datetime.datetime.fromtimestamp(time.time() if t is None else t, datetime.timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def tree_digest(root: str) -> dict:
    """Every regular file under root: its relative path -> [bytes, sha256] (meta.json's record of --examples)."""
    out = {}
    for d, _, names in sorted(os.walk(root)):
        for name in sorted(names):
            path = os.path.join(d, name)
            if os.path.isfile(path):
                data = open(path, "rb").read()
                out[os.path.relpath(path, root)] = [len(data), hashlib.sha256(data).hexdigest()]
    return out


# ------------------------------------------------------------------------------------------------ what the agent gets
WORLD = {
    "lockstep": "**The world** is a simulation of the robot station described in the manual below, run in lockstep: "
                "it moves only\n  while one of your programs is inside a blocking robot call. Every program you run "
                "acts on the same world, and the\n  world keeps its state between programs for the whole episode.",
    "realtime": "**The world** is a simulation of the robot station described in the manual below, running on the "
                "wall clock like the\n  real robot. Its clock starts when your first program connects; from then on "
                "it keeps moving while you\n  read, think or write code, between programs and while a program is "
                "paused. A paused or ended program's arms\n  hold where they are, but a thrown or falling ball keeps "
                "falling and anything already moving keeps moving. Camera\n  frames take real time to grab and "
                "arrive late. Every program you run acts on the same world, and the world keeps\n  its state between "
                "programs for the whole episode.",
}


def render_agents_md(instruction: str, vision: bool, wall_h: float, started: str, gpu: str | None,
                     clock: str = "lockstep", memory: tuple = (MEM_CAP_PROCESS_GB, MEM_CAP_GB), pi: bool = False,
                     examples: bool = False, appendix: str | None = None) -> str:
    """AGENTS.md: the template with the task, the environment facts and the robot manual verbatim. [[...]] spans are
    kept with vision and dropped without. gpu: what the sandbox's GPU is (gpu_info), None for none. clock: lockstep
    or realtime, the World paragraph. memory: the caps (per process, all together) in GB, MemoryWatchdog's. pi: the
    bash time limit's line (PI_TIMEOUT_LINE). examples: the line pointing at EXAMPLES. appendix: text appended after
    the manual (--appendix). Without these three, Codex's AGENTS.md is as it was."""
    text = open(os.path.join(HERE, "AGENTS.md.template")).read()
    text = re.sub(r"\[\[(.*?)\]\]", (lambda m: m.group(1)) if vision else "", text, flags=re.S)
    values = {"instruction": instruction.strip(), "world": WORLD[clock], "workspace": WS, "code": CODE,
              "robot_python": f"{ROBOT_VENV}/bin/python", "perception_python": f"{PERCEPTION_VENV}/bin/python",
              "libs_python": f"{ROBOT_LIBS_VENV}/bin/python",
              "robot_addr": f"unix:{SOCKS}/robot.sock", "vision_addr": f"unix:{SOCKS}/vision.sock",
              "wall_h": f"{wall_h:g}", "started": started,
              "mem_process_gb": f"{memory[0]:g}", "mem_total_gb": f"{memory[1]:g}",
              "commands": PI_TIMEOUT_LINE if pi else "", "examples": EXAMPLES_LINE if examples else "",
              "gpu": (f"**GPU:** {gpu}, shared with other processes. torch, warp, cuRobo and the robot-libs "
                      "Python's JAX (PyRoKi) use it; the perception Python's JAX runs on the CPU. JAX allocates as it "
                      "goes rather than taking most of the memory at start." if gpu else
                      "**No GPU:** your programs see none; torch and JAX run on the CPU, and cuRobo cannot run.")}
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    manual = open(os.path.join(ROOT, "docs", "MANUAL.md")).read().strip()
    text = text.replace("{manual}", manual)              # last: the manual's own braces stay as they are
    return text if appendix is None else text.rstrip("\n") + "\n\n" + appendix.strip() + "\n"


def llm_spec(name: str, endpoint: str | None, session: str) -> dict:
    """The model Codex runs on. gpt: MODEL through the ChatGPT login. deepseek: DEEPSEEK at endpoint (an ora-router
    allocation's gateway URL), with session (the episode's tag) as the X-Session-Id that keeps it on one replica."""
    if name == "gpt":
        return {"name": name, "model": MODEL, "effort": EFFORT, "provider": None, "egress": EGRESS_ALLOW}
    host, _, port = endpoint.split("://", 1)[-1].split("/", 1)[0].partition(":")
    return {"name": name, "model": DEEPSEEK["model"], "effort": DEEPSEEK["effort"],
            "provider": {"id": "deepseek", "base_url": endpoint.rstrip("/"), "wire_api": "responses",
                         "env_key": DEEPSEEK["key_env"], "http_headers": {"X-Session-Id": session},
                         "context_window": DEEPSEEK["context_window"],
                         "max_output_tokens": DEEPSEEK["max_output_tokens"], "ca": DEEPSEEK["ca"]},
            "egress": [f"{host}:{port or 443}"]}


def permissions(vision: bool, unreadable: list) -> list:
    """config.toml's permission profile "robot", for Codex's command sandbox: :workspace (writes to /workspace and
    /tmp) and its .git, the unreadable paths, the egress socket and Codex's daemon directory unreadable, the vision
    recording writable, network on with only the episode's unix sockets."""
    socks = [f"{SOCKS}/robot.sock"] + ([f"{SOCKS}/vision.sock"] if vision else []) + [f"{PROGRAM_HOST}/host.sock"]
    return [
        "[permissions.robot]",
        'extends = ":workspace"',
        "",
        "[permissions.robot.filesystem]",
        f'"{WS}/.git" = "write"',
        *[f'"{path}" = "none"' for path in unreadable],
        f'"{EGRESS}" = "none"',
        f'"/tmp/codex-daemon-{os.getuid()}" = "none"',
        *([f'"{RECORD}" = "write"'] if vision else []),
        "",
        "[permissions.robot.network]",
        "enabled = true",
        "unix_sockets = { " + ", ".join(f'"{s}" = "allow"' for s in socks) + " }",
        ""]


def codex_config(vision: bool, llm: dict) -> str:
    """The episode's CODEX_HOME/config.toml."""
    p = llm["provider"]
    return "\n".join([
        f'model = "{llm["model"]}"',
        f'model_reasoning_effort = "{llm["effort"]}"',
        *([f'model_provider = "{p["id"]}"', f'model_context_window = {p["context_window"]}'] if p else []),
        'approval_policy = "never"',
        'default_permissions = "robot"',
        'web_search = "disabled"',
        "",
        "[features]",
        *[f"{name} = false" for name in DISABLED_FEATURES + (["multi_agent"] if p else [])],
        "",
        *([f'[model_providers.{p["id"]}]', 'name = "DeepSeek (ora-router LLM gateway)"',
           f'base_url = "{p["base_url"]}"', f'env_key = "{p["env_key"]}"', f'wire_api = "{p["wire_api"]}"',
           "http_headers = { " + ", ".join(f'"{k}" = "{v}"' for k, v in p["http_headers"].items()) + " }",
           ""] if p else []),
        "[skills]",                                      # no bundled skills (image generation, docs, ...)
        "include_instructions = false",
        "",
        "[skills.bundled]",
        "enabled = false",
        "",
        "[shell_environment_policy]",                    # commands get the same environment for every model
        'inherit = "all"',
        f"exclude = {json.dumps(PROXY_VARS + (PROVIDER_VARS if p else []))}",
        "",
        *permissions(vision, [] if p else [f"{CODEX_HOME}/auth.json"])])


def command_sandbox_config(vision: bool) -> str:
    """--agent pi: CODEX_HOME/config.toml for Codex's command sandbox alone (pi_shell.sh and the program host run
    `codex sandbox -P robot`): a Codex agent's command profile with Pi's home unreadable; no model, provider or agent
    settings, since no Codex agent runs."""
    return "\n".join(['default_permissions = "robot"', "", "[features]",
                      *[f"{name} = false" for name in DISABLED_FEATURES], "", *permissions(vision, [PI_HOME])])


def pi_settings() -> dict:
    """--agent pi: Pi's settings.json (agent directory, read-only in the sandbox)."""
    return {"defaultProvider": PI_PROVIDER, "defaultModel": DEEPSEEK["model"], "defaultThinkingLevel": DEEPSEEK["effort"],
            "shellPath": f"{HARNESS}/pi_shell.sh", "defaultProjectTrust": "never", "quietStartup": True,
            "cacheWarming": "off", "retry": {"enabled": True, "maxRetries": PI_RETRIES, "baseDelayMs": 2000,
                                             "maxAgentDelayMs": 60000}}


def pi_models(token: str) -> dict:
    """--agent pi: Pi's models.json (read-only in the sandbox). The provider is the model proxy through the loopback
    bridge; token, the episode's, is all the API key Pi has."""
    return {"providers": {PI_PROVIDER: {
        "baseUrl": f"http://127.0.0.1:{PROXY_PORT}/v1", "api": "openai-completions", "apiKey": token,
        "models": [{"id": DEEPSEEK["model"], "name": DEEPSEEK["model"].split("/")[-1], "reasoning": True,
                    "input": ["text", "image"], "contextWindow": DEEPSEEK["context_window"],
                    "maxTokens": DEEPSEEK["max_output_tokens"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsDeveloperRole": False, "supportsStore": False, "maxTokensField": "max_tokens",
                               "supportsReasoningEffort": True}}]}}}


def pi_argv(prompt: str | None) -> list:
    """--agent pi: Pi's command inside the sandbox (prompt None: without it, for the stand-in)."""
    return [f"{NODE_DIR}/bin/node", f"{PI_DIR}/node_modules/{PI_PACKAGE}/dist/bundle/cli.js", "--mode", "json",
            "--provider", PI_PROVIDER, "--model", DEEPSEEK["model"], "--thinking", DEEPSEEK["effort"],
            "--tools", ",".join(PI_TOOLS), "--session-dir", f"{PI_HOME}/sessions", "--no-extensions",
            *[x for name in PI_EXTENSIONS for x in ("--extension", f"{HARNESS}/{name}")], "--no-mcp",
            "--no-skills", "--no-prompt-templates", "--no-themes", "--no-approve",
            *(["--", prompt] if prompt is not None else [])]


def pi_info() -> dict:
    pkg = json.load(open(os.path.join(PI_MODULES, PI_PACKAGE, "package.json")))
    node = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip()
    return {"package": pkg["name"], "version": pkg["version"], "node": node, "node_path": NODE,
            "modules": PI_MODULES}


def agent_argv(kind: str, prompt: str, run: str, llm: dict) -> list:
    """The agent's command inside the sandbox."""
    if kind == "codex":
        return [f"{CODEX_DIR}/bin/codex", "exec", "--json", "-m", llm["model"], "--skip-git-repo-check", "-C", WS,
                "--ignore-rules", "-o", f"{CODEX_HOME}/last_message.txt", "--", prompt]
    if kind == "prompt-input":
        return [f"{CODEX_DIR}/bin/codex", "debug", "prompt-input", prompt]
    if kind == "standin":                                # the real paths it must fail to read, and the model API
        p = llm["provider"]
        return ["/bin/bash", f"{HARNESS}/standin.sh", run, ROOT, os.path.dirname(CODEX_AUTH),
                *([p["base_url"], p["env_key"]] if p else [])]
    if kind == "pi":
        return pi_argv(prompt)
    if kind == "pi-standin":                             # the same, with Pi's shell, the model proxy and Pi itself
        return ["/bin/bash", f"{HARNESS}/standin.sh", run, ROOT, os.path.dirname(CODEX_AUTH), "--pi",
                f"http://127.0.0.1:{PROXY_PORT}/v1", *pi_argv(None)]
    raise ValueError(kind)


def command_sandbox() -> list:
    """How Codex runs a command (the program host runs programs the same way): profile "robot", cwd /workspace."""
    return [f"{CODEX_DIR}/bin/codex", "sandbox", "-P", "robot", "-C", WS]


def library_paths() -> list:
    """What the Pythons need besides their own directories: the base interpreter (pyvenv.cfg home), the directories
    their .pth files add to sys.path and the package directories their editable-install finders map."""
    out = set()
    for venv in VENVS:
        for line in open(os.path.join(venv, "pyvenv.cfg")):
            key, _, value = line.partition("=")
            if key.strip() == "home":
                out.add(os.path.dirname(value.strip()))
        site = os.path.join(venv, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        for name in os.listdir(site):
            path = os.path.join(site, name)
            if name.endswith(".pth"):
                for line in open(path):
                    line = line.strip()
                    if line.startswith("/") and os.path.isdir(line):
                        out.add(line)
            elif name.startswith("__editable___") and name.endswith("_finder.py"):
                m = re.search(r"^MAPPING[^=]*=\s*(\{.*?\})", open(path).read(), re.M | re.S)
                if m:
                    out.update(p for p in ast.literal_eval(m.group(1)).values() if os.path.isdir(p))
    return sorted(out)


def model_files(desc) -> list:
    """Every file r.description names (the URDFs, the MJCF) and the files those load (meshes, textures, includes)."""
    found, todo = set(), []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and v.startswith("/") and os.path.isfile(v):
            todo.append(v)
    walk(desc)
    while todo:
        path = os.path.normpath(todo.pop())
        if path in found or not os.path.isfile(path):
            continue
        found.add(path)
        if not path.endswith((".urdf", ".xml")):
            continue
        base = os.path.dirname(path)
        root = ET.parse(path).getroot()
        if path.endswith(".urdf"):
            todo += [os.path.join(base, el.get("filename")) for el in root.iter("mesh") if el.get("filename")]
            continue
        comp = root.find("compiler")
        dirs = {k: os.path.join(base, comp.get(k, "") if comp is not None else "")
                for k in ("meshdir", "texturedir")}
        for el in root.iter():
            f = el.get("file")
            if not f:
                continue
            where = {"mesh": dirs["meshdir"], "skin": dirs["meshdir"], "texture": dirs["texturedir"],
                     "hfield": dirs["texturedir"]}.get(el.tag, base)
            todo.append(os.path.join(where, f))
    return sorted(found)


def gpu_info(index: int) -> tuple:
    """GPU `index` as nvidia-smi numbers it -> (its /dev node, "one NVIDIA H200 (140 GB)"). The node is found through
    the PCI bus: the driver's device minors are not nvidia-smi's order (on one 8-GPU node card 2 was /dev/nvidia6)."""
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=pci.bus_id,name,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    bus, name, mib = [x.strip() for x in out.strip().split(",")]
    info = open(f"/proc/driver/nvidia/gpus/{bus[-12:].lower()}/information").read()
    minor = int(re.search(r"Device Minor:\s*(\d+)", info).group(1))
    return f"/dev/nvidia{minor}", f"one {name} ({float(mib) / 1024:.0f} GB)"


def bwrap_argv(run: str, tmp: str, inner: list, vision: bool, standin: bool, stop_fd: int, host_log: int,
               password: str, models: list, gpu_dev: str | None, llm: dict, key_fd: int | None,
               pi: bool = False, examples: str | None = None) -> list:
    """key_fd: a pipe holding `--setenv KEY_ENV KEY` for bwrap's --args, so the key is in no command line (or
    bwrap.json); bwrap sets it in the sandbox's first process, which keeps it for Codex alone. pi: Pi's sandbox
    (module docstring), where the bridge leads to the model proxy and no process holds a key or proxy password.
    examples: a directory bound read-only at EXAMPLES (--examples)."""
    bw = os.path.join(CODEX_PKG, "codex-resources", "bwrap")
    p = llm["provider"]
    proxy = f"http://codex:{password}@127.0.0.1:{PROXY_PORT}"
    gpu_devs = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", gpu_dev] if gpu_dev else []
    env = {"PATH": (f"{HARNESS}/bin:" if gpu_dev else "")      # gpu_bwrap.sh as `bwrap`, for Codex's sandbox
                   + f"{ROBOT_VENV}/bin:{CODEX_DIR}/bin:{CODEX_DIR}/codex-path:/usr/local/bin:/usr/bin:/bin",
           "HOME": "/home/agent", "USER": os.environ.get("USER", "agent"), "LANG": "C.UTF-8", "TMPDIR": "/tmp",
           "CODEX_HOME": CODEX_HOME, "PYTHONPATH": CODE, "PYTHONDONTWRITEBYTECODE": "1",
           "XDG_CACHE_HOME": "/tmp/.cache",               # warp's and torch's kernel caches: only /tmp is writable
           **({"XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # JAX would take 75% of a shared card
               "ROBOT_SANDBOX_GPU_DEVICES": " ".join(gpu_devs)} if gpu_dev else
              {"JAX_PLATFORMS": "cpu"}),                 # no GPU: JAX would first fail to find one
           "ROBOT_ADDR": f"unix:{SOCKS}/robot.sock", "ROBOT_PROGRAM_HOST": f"{PROGRAM_HOST}/host.sock",
           **({"VISION_ADDR": f"unix:{SOCKS}/vision.sock", "VISION_RECORD_DIR": RECORD} if vision else {}),
           **({"CODEX_CA_CERTIFICATE": f"{HARNESS}/ca.crt",          # the gateway's internal CA, for Codex
               "SANDBOX_PRIVATE_VARS": " ".join(PROVIDER_VARS)} if p else {}),   # sandbox_init keeps them from the host
           **{v: proxy for v in PROXY_VARS}}
    if pi:                                               # no proxy password, key or CA; Pi's own settings
        env = {k: v for k, v in env.items() if k not in PROXY_VARS + PROVIDER_VARS}
        env.update(PATH=(f"{HARNESS}/bin:" if gpu_dev else "")       # rg as a Codex agent's commands have it
                   + f"{ROBOT_VENV}/bin:{CODEX_DIR}/codex-path:/usr/local/bin:/usr/bin:/bin",
                   PI_CODING_AGENT_DIR=f"{PI_HOME}/agent", PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0",
                   NODE_COMPILE_CACHE="/tmp/.cache/node-compile-cache")
    home = (["--bind", f"{run}/sandbox/codex-home", CODEX_HOME,
             "--ro-bind", f"{run}/sandbox/codex-home/config.toml", f"{CODEX_HOME}/config.toml",
             "--bind", f"{run}/pi/home", PI_HOME,
             *[x for name in ("models.json", "settings.json")
               for x in ("--ro-bind", f"{run}/pi/home/agent/{name}", f"{PI_HOME}/agent/{name}")],
             "--ro-bind", NODE, f"{NODE_DIR}/bin/node", "--ro-bind", PI_MODULES, f"{PI_DIR}/node_modules",
             "--ro-bind", os.path.join(HERE, "pi_shell.sh"), f"{HARNESS}/pi_shell.sh",
             *[x for name in PI_EXTENSIONS for x in ("--ro-bind", os.path.join(HERE, name), f"{HARNESS}/{name}")]]
            if pi else
            ["--bind", f"{run}/codex/home", CODEX_HOME,
             *(["--ro-bind", p["ca"], f"{HARNESS}/ca.crt"] if p else ["--bind", CODEX_AUTH, f"{CODEX_HOME}/auth.json"])])
    a = [bw, "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
         "--unshare-cgroup-try", "--hostname", "sandbox",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
         "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
         "--ro-bind", "/etc", "/etc", "--ro-bind", "/sys", "/sys", "--proc", "/proc", "--dev", "/dev",
         "--bind", f"{tmp}/tmp", "/tmp", "--dir", "/home/agent",
         "--bind", f"{run}/workspace", WS,
         "--ro-bind", f"{run}/sandbox/robot-api", CODE,
         "--ro-bind", f"{tmp}/sock", SOCKS,
         "--ro-bind", f"{tmp}/egress", EGRESS,
         "--bind", f"{tmp}/host", PROGRAM_HOST,
         *home,
         "--ro-bind", CODEX_PKG, CODEX_DIR,
         "--ro-bind", os.path.join(HERE, "sandbox_init.py"), f"{HARNESS}/sandbox_init.py",
         "--ro-bind", os.path.join(HERE, "program_host.py"), f"{HARNESS}/program_host.py"]
    if standin:
        a += ["--ro-bind", os.path.join(HERE, "standin.sh"), f"{HARNESS}/standin.sh"]
    if vision:
        a += ["--bind", f"{run}/vision", RECORD]
    if examples:
        a += ["--ro-bind", examples, EXAMPLES]
    if gpu_dev:                                          # that one card's node: CUDA sees no other
        for dev in gpu_devs:
            a += ["--dev-bind", dev, dev]
        a += ["--ro-bind", os.path.join(HERE, "gpu_bwrap.sh"), f"{HARNESS}/bin/bwrap"]
    for path in [*VENVS, *library_paths(), *models]:
        a += ["--ro-bind", path, path]
    a += ["--clearenv", *[x for k, v in env.items() for x in ("--setenv", k, v)],
          *(["--args", str(key_fd)] if key_fd is not None else []), "--chdir", WS, "--",
          "/usr/bin/python3", f"{HARNESS}/sandbox_init.py", str(PROXY_PORT),
          f"{EGRESS}/{'model' if pi else 'proxy'}.sock", str(stop_fd),
          str(host_log), "/usr/bin/python3", f"{HARNESS}/program_host.py", f"{PROGRAM_HOST}/host.sock",
          *command_sandbox(), "--", *inner]
    return a


# ------------------------------------------------------------------------------------------------ egress proxy
class EgressProxy:
    """An HTTP CONNECT proxy on a unix socket: the sandbox's only way out. Allowed hosts (HOST for port 443,
    HOST:PORT for another), for clients that give the episode's password; one JSON line per connection in the log."""

    def __init__(self, path: str, password: str, allow: list, log_path: str):
        self.path, self.allow = path, {h if ":" in h else f"{h}:443" for h in allow}
        self.auth = "Basic " + base64.b64encode(f"codex:{password}".encode()).decode()
        self.log = open(log_path, "a")
        self.lock = threading.Lock()
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _record(self, **entry):
        with self.lock:
            self.log.write(json.dumps({"wall": round(time.time(), 3), **entry}) + "\n")
            self.log.flush()

    def _handle(self, conn):
        t0, head = time.time(), b""
        try:
            while b"\r\n\r\n" not in head and len(head) < 16384:
                chunk = conn.recv(4096)
                if not chunk:
                    conn.close()
                    return
                head += chunk
            text, _, extra = head.partition(b"\r\n\r\n")
            lines = text.decode("latin-1").split("\r\n")
            method, target = (lines[0].split(" ") + ["", ""])[:2]
            headers = {k.strip().lower(): v.strip() for k, _, v in (h.partition(":") for h in lines[1:])}
            host, _, port = target.rpartition(":")
            if headers.get("proxy-authorization") != self.auth:
                return self._refuse(conn, t0, target, 407, "Proxy Authentication Required", "no or wrong password")
            if method != "CONNECT" or not port.isdigit():
                return self._refuse(conn, t0, target, 405, "Method Not Allowed", f"method {method}")
            if f"{host}:{port}" not in self.allow:
                return self._refuse(conn, t0, target, 403, "Forbidden", "host not allowed")
            up = socket.create_connection((host, int(port)), timeout=30)
            up.settimeout(None)
        except OSError as e:
            return self._refuse(conn, t0, head[:200].decode("latin-1", "replace"), 502, "Bad Gateway", repr(e))
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if extra:
            up.sendall(extra)
        counts = {"up": len(extra), "down": 0}
        done = threading.Event()

        def pump(src, dst, key):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
                    counts[key] += len(data)
            except OSError:
                pass
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            if key == "down":
                done.set()
        t = threading.Thread(target=pump, args=(conn, up, "up"), daemon=True)
        t.start()
        pump(up, conn, "down")
        t.join(30)
        for s in (conn, up):
            s.close()
        self._record(host=host, port=int(port), allowed=True, status=200, up=counts["up"], down=counts["down"],
                     dur_s=round(time.time() - t0, 3))

    def _refuse(self, conn, t0, target, code, reason, why):
        try:
            extra = 'Proxy-Authenticate: Basic realm="egress"\r\n' if code == 407 else ""
            conn.sendall(f"HTTP/1.1 {code} {reason}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n".encode())
        except OSError:
            pass
        conn.close()
        self._record(target=target, allowed=False, status=code, reason=why, dur_s=round(time.time() - t0, 3))

    def close(self):
        self.srv.close()
        self.log.close()


class ModelProxy:
    """--agent pi: the sandbox's only way out, to the model API, and the only holder of the key. Plain HTTP on a unix
    socket (sandbox_init.py bridges Pi's 127.0.0.1:PROXY_PORT to it), sent over TLS (the internal CA) to the endpoint's
    host and port with Authorization (the key) and X-Session-Id (the episode's tag) set here; Pi has the episode's
    token as its API key, and nothing else. Only POST /v1/chat/completions and GET /v1/models, one request per
    connection, the response streamed back as it arrives. One JSON line per request in the log: status, times, sizes,
    the messages and images the request carried, how many of its leading messages are the previous request's, and the
    usage its stream or reply reported (self.requests holds the same); a connection closed before any request is a line
    with "empty": true. The first chat request's body goes to first_path: what the model saw at the start."""

    TIMEOUT = 600.0                                      # s without a byte from the gateway (Pi's own limit is 300)
    ROUTES = {("POST", "/v1/chat/completions"): "/chat/completions", ("GET", "/v1/models"): "/models"}
    DROP = {"connection", "keep-alive", "proxy-authorization", "proxy-authenticate", "proxy-connection", "te",
            "trailer", "transfer-encoding", "upgrade", "host", "content-length", "authorization", "x-session-id",
            "accept-encoding"}                         # hop-by-hop, recomputed, or set here

    def __init__(self, path: str, endpoint: str, key: str, session: str, token: str, ca: str, log_path: str,
                 first_path: str):
        u = urllib.parse.urlsplit(endpoint)
        self.host, self.port, self.base = u.hostname, u.port or 443, u.path.rstrip("/")
        self.set = {"Authorization": f"Bearer {key}", "X-Session-Id": session, "Accept-Encoding": "identity"}
        self.token = f"Bearer {token}"
        self.ctx = ssl.create_default_context(cafile=ca)
        self.first_path, self.requests, self.previous = first_path, [], []
        self.log, self.lock = open(log_path, "a"), threading.Lock()
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(16)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _record(self, entry: dict):
        with self.lock:
            self.requests.append(entry)
            self.log.write(json.dumps(entry) + "\n")
            self.log.flush()

    def _reply(self, conn, entry: dict, status: int, why: str):
        body = json.dumps({"error": {"message": f"model proxy: {why}"}}).encode()
        try:
            conn.sendall(f"HTTP/1.1 {status} {http.client.responses.get(status, '')}\r\nContent-Type: application/json"
                         f"\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        except OSError:
            pass
        entry.update(status=status, refused=why)

    @staticmethod
    def _describe(body: bytes) -> dict:
        try:
            req = json.loads(body)
            msgs = req.get("messages") or []
        except (ValueError, AttributeError):
            return {"req_bytes": len(body), "body": "not a JSON object"}
        images = sum(1 for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), list)
                     for c in m["content"] if isinstance(c, dict) and c.get("type") == "image_url")
        return {"req_bytes": len(body), "model": req.get("model"), "max_tokens": req.get("max_tokens"),
                "reasoning_effort": req.get("reasoning_effort"), "stream": req.get("stream"), "messages": len(msgs),
                "images": images, "tools": len(req.get("tools") or []),
                "_hashes": [hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest() for m in msgs]}

    @staticmethod
    def _usage(entry: dict, data: bytes):
        """A stream event's, or a whole reply's, usage and finish reason into entry."""
        try:
            ev = json.loads(data)
        except ValueError:
            return
        if isinstance(ev, dict):
            entry["usage"] = ev.get("usage") or entry.get("usage")
            for c in ev.get("choices") or []:
                if isinstance(c, dict) and c.get("finish_reason"):
                    entry["finish_reason"] = c["finish_reason"]

    def _handle(self, conn):
        t0, entry, up = time.time(), {"wall": round(time.time(), 3)}, None
        try:
            conn.settimeout(60)
            f = conn.makefile("rb")
            method, target, _ = (f.readline(8192).decode("latin-1").split(" ", 2) + ["", ""])[:3]
            headers = {}
            while (line := f.readline(65536)) not in (b"\r\n", b"\n", b""):
                k, _, v = line.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            if not method.strip():                       # closed before sending a request (Pi's HTTP client does)
                entry["empty"] = True
                return
            entry.update(method=method, path=target)
            route = self.ROUTES.get((method, target))
            if not hmac.compare_digest(headers.get("authorization", "").encode(), self.token.encode()):
                return self._reply(conn, entry, 401, "no or wrong token")
            if route is None:
                return self._reply(conn, entry, 404, "not a model API request")
            if "transfer-encoding" in headers:
                return self._reply(conn, entry, 411, "a request body needs Content-Length")
            body = f.read(int(headers.get("content-length") or 0)) if method == "POST" else None
            if body is not None:
                entry.update(self._describe(body))
                hashes = entry.pop("_hashes", [])
                with self.lock:                          # does it extend the previous request, message for message?
                    k = next((i for i, (x, y) in enumerate(zip(self.previous, hashes)) if x != y),
                             min(len(self.previous), len(hashes)))
                    entry.update(same_messages=k, extends_previous=bool(self.previous) and k == len(self.previous))
                    self.previous = hashes or self.previous
                if route == "/chat/completions" and not os.path.exists(self.first_path):
                    with open(self.first_path, "wb") as out:
                        out.write(body)
            conn.settimeout(None)
            up = http.client.HTTPSConnection(self.host, self.port, context=self.ctx, timeout=self.TIMEOUT)
            up.request(method, self.base + route, body=body,
                       headers={**{k: v for k, v in headers.items() if k not in self.DROP}, **self.set})
            resp = up.getresponse()
            entry.update(status=resp.status, ttfb_s=round(time.time() - t0, 3))
            head = [f"HTTP/1.1 {resp.status} {resp.reason}",
                    *[f"{k}: {v}" for k, v in resp.getheaders() if k.lower() not in self.DROP], "Connection: close"]
            conn.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
            buf, size, error, whole = b"", 0, b"", b""
            plain = (resp.getheader("Content-Type") or "").startswith("application/json")    # not a stream
            while chunk := resp.read1(65536):
                size += len(chunk)
                conn.sendall(chunk)
                if resp.status != 200:
                    error = (error + chunk)[:4000]
                    continue
                if plain:                                # a non-streamed reply: its usage is in the whole body
                    whole = (whole + chunk)[:1 << 22]
                    continue
                buf += chunk                             # the stream's events: its usage and finish reason
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln.startswith(b"data:") and ln[5:].strip() not in (b"", b"[DONE]"):
                        self._usage(entry, ln[5:])
            if whole:
                self._usage(entry, whole)
            entry["resp_bytes"] = size
            if error:
                entry["error_body"] = error.decode("utf-8", "replace")
        except (OSError, ValueError, http.client.HTTPException) as e:
            entry["error"] = repr(e)
            if "status" not in entry:
                self._reply(conn, entry, 502, f"the gateway: {e!r}")
        finally:
            for s in (up, conn):
                try:
                    s and s.close()
                except OSError:
                    pass
            entry["dur_s"] = round(time.time() - t0, 3)
            self._record(entry)

    def summary(self, t_agent: float) -> dict:
        """The model requests and their tokens, per request and in total: input (prompt tokens, cached included),
        cached, output (completion tokens, reasoning included), reasoning, as the gateway's usage reported them; the
        gateway's chat API reports no cached tokens (vLLM gives them only with --enable-prompt-tokens-details), so
        cached_est is the reusable prefix instead: the previous request's input when this one extends it message for
        message (prefix caching on the session's replica can reuse that much), else 0."""
        rows, total, last = [], {"input": 0, "cached": 0, "cached_est": 0, "output": 0, "reasoning": 0}, None
        for r in self.requests:
            if r.get("path") != "/v1/chat/completions":
                continue
            u = r.get("usage") or {}
            row = {"t_s": round(r["wall"] - t_agent, 1), "dur_s": r.get("dur_s"), "status": r.get("status"),
                   "input": u.get("prompt_tokens"),
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "cached_est": (last or 0) if r.get("extends_previous") else 0,
                   "output": u.get("completion_tokens"),
                   "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "messages": r.get("messages"), "same_messages": r.get("same_messages"), "images": r.get("images"),
                   "finish": r.get("finish_reason")}
            last = row["input"] if row["input"] is not None else last
            for k in total:
                total[k] += row[k] or 0
            rows.append(row)
        statuses = {}
        for row in rows:
            statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
        return {"count": len(rows), "statuses": statuses, "with_usage": sum(row["input"] is not None for row in rows),
                "cached_reported": sum(row["cached"] is not None for row in rows), "tokens": total,
                "log": "pi/requests.jsonl", "per_request": rows}

    def close(self):
        self.srv.close()
        with self.lock:
            self.log.close()


def sandbox_pids(root: int) -> list:
    """root (the sandbox's bwrap) and its descendants."""
    kids = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    kids.setdefault(int(f.read().rsplit(")", 1)[1].split()[1]), []).append(int(name))
            except (OSError, IndexError, ValueError):
                continue
    out, i = [root], 0
    while i < len(out):
        out += kids.get(out[i], [])
        i += 1
    return out


def scan_processes(root: int, secret: bytes, audit: dict):
    """--agent pi: the key in the environment or command line of any process in the sandbox (audit accumulates)."""
    audit["scans"] += 1
    for pid in sandbox_pids(root):
        audit["pids"].add(pid)
        for part in ("environ", "cmdline"):
            try:
                with open(f"/proc/{pid}/{part}", "rb") as f:
                    data = f.read()
            except OSError:
                audit["unreadable"] += 1
                continue
            audit["read"] += 1
            if secret in data:
                audit["hits"].append(f"pid {pid} {part}")


def scan_files(root: str, secret: bytes) -> dict:
    """--agent pi: the key in any regular file under root (the run directory, at the end)."""
    hits, n, size = [], 0, 0
    for here, dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(here, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            n += 1
            tail = b""
            with open(path, "rb") as f:
                while chunk := f.read(1 << 24):
                    size += len(chunk)
                    if secret in tail + chunk:
                        hits.append(os.path.relpath(path, root))
                        break
                    tail = chunk[-len(secret):]
    return {"files": n, "bytes": size, "hits": hits}


_LIBC = ctypes.CDLL(None, use_errno=True)
SYS_PIDFD_SEND_SIGNAL, SYS_PIDFD_OPEN = 424, 434          # Linux >= 5.3, the same on every architecture


def pidfd_kill(pid: int, same) -> bool:
    """SIGKILL pid through a pidfd (by syscall: the robot venv's Python has no os.pidfd_open), if same() still holds
    once the pidfd pins the process, so a PID reused by another process is never signalled. True if it was sent."""
    fd = _LIBC.syscall(ctypes.c_long(SYS_PIDFD_OPEN), ctypes.c_long(pid), ctypes.c_long(0))
    if fd < 0:
        return False
    try:
        return bool(same()) and _LIBC.syscall(ctypes.c_long(SYS_PIDFD_SEND_SIGNAL), ctypes.c_long(fd),
                                              ctypes.c_long(signal.SIGKILL), None, ctypes.c_long(0)) == 0
    finally:
        os.close(fd)


class MemoryWatchdog:
    """The episode's memory cap. Every PERIOD s it reads the resident memory (RSS) of every process in the sandbox:
    its bwrap and all descendants, so the agent, its commands and the program host's programs, but not robotd or the
    vision server, which this process starts outside it. When one process holds more than per_process_gb, or all of
    them together more than total_gb, the largest is killed (SIGKILL through a pidfd, after checking that its PID still
    names the process sampled). A killed process still freeing its memory, or a zombie, counts no more. Each kill is a
    line in log_path and a note; summary() gives the caps, the peaks and the kills. GB here are 2^30 bytes."""

    PERIOD = 2.0
    GB = 1 << 30
    PAGE = os.sysconf("SC_PAGE_SIZE")

    def __init__(self, root: int, per_process_gb: float, total_gb: float, log_path: str, notes: list, t0: float):
        self.root, self.caps, self.notes, self.t0 = root, (per_process_gb, total_gb), notes, t0
        self.log, self.kills, self.killed, self.samples, self.errors = open(log_path, "a"), [], set(), 0, []
        self.peak_total, self.peak = 0, (0, None)        # bytes; (bytes, "comm: cmdline") of the largest process
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _proc(pid: int):
        """(rss bytes, state, start time) of pid, or None once it has gone."""
        try:
            with open(f"/proc/{pid}/statm") as f:
                rss = int(f.read().split()[1]) * MemoryWatchdog.PAGE
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return rss, fields[0], int(fields[19])
        except (OSError, IndexError, ValueError):
            return None

    @staticmethod
    def _what(pid: int) -> str:
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            return "?"
        return f"{comm}: {cmd[:300]}"

    def _run(self):
        while not self.done.wait(self.PERIOD):
            try:
                self._sample()
            except Exception as e:                       # noqa: BLE001 -- keep watching; say so once
                if not self.errors:
                    self.notes.append(f"the memory watchdog: {e!r}")
                self.errors.append(repr(e))

    def _sample(self):
        rows = []
        for pid in sandbox_pids(self.root):
            p = self._proc(pid)
            if p is not None and p[1] not in "ZX" and (pid, p[2]) not in self.killed:
                rows.append((p[0], pid, p[2]))
        if not rows:
            return
        self.samples += 1
        total = sum(r[0] for r in rows)
        rss, pid, start = max(rows)
        self.peak_total = max(self.peak_total, total)
        if rss > self.peak[0]:
            self.peak = (rss, self._what(pid))
        why = ("per_process" if rss > self.caps[0] * self.GB else
               "total" if total > self.caps[1] * self.GB else None)
        if why:
            self._kill(pid, start, rss, total, why)

    def _kill(self, pid: int, start: int, rss: int, total: int, why: str):
        what = self._what(pid)
        same = lambda: (self._proc(pid) or (0, "", None))[2] == start     # noqa: E731 -- not gone, not reused
        if not pidfd_kill(pid, same):
            if same():
                self.errors.append(f"could not kill pid {pid} ({what[:80]}): errno {ctypes.get_errno()}")
            return
        self.killed.add((pid, start))
        entry = {"wall": round(time.time(), 3), "t_s": round(time.time() - self.t0, 1), "pid": pid, "what": what,
                 "rss_gb": round(rss / self.GB, 2), "sandbox_gb": round(total / self.GB, 2), "cap": why,
                 "caps_gb": {"per_process": self.caps[0], "total": self.caps[1]}}
        self.kills.append(entry)
        self.log.write(json.dumps(entry) + "\n")
        self.log.flush()
        self.notes.append(f"the memory watchdog killed pid {pid} ({what[:120]}) at {entry['rss_gb']} GB, the sandbox "
                          f"at {entry['sandbox_gb']} GB: over the {'per-process' if why == 'per_process' else 'total'}"
                          f" cap of {self.caps[0] if why == 'per_process' else self.caps[1]:g} GB")

    def summary(self) -> dict:
        return {"cap_process_gb": self.caps[0], "cap_total_gb": self.caps[1], "period_s": self.PERIOD,
                "samples": self.samples, "peak_sandbox_gb": round(self.peak_total / self.GB, 2),
                "peak_process_gb": round(self.peak[0] / self.GB, 2), "peak_process": self.peak[1],
                "kills": self.kills, "errors": self.errors[:10], "log": "logs/memory.jsonl"}

    def close(self):
        self.done.set()
        self.thread.join(10)
        self.log.close()


# ------------------------------------------------------------------------------------------------ services
def _wait_file(path: str, proc: subprocess.Popen, timeout: float, what: str):
    t0 = time.time()
    while not os.path.exists(path):
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited ({proc.returncode}) before it was ready; see its log")
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{what} not ready after {timeout:.0f} s")
        time.sleep(0.2)


def start_robotd(a, run: str, tmp: str) -> subprocess.Popen:
    argv = [f"{ROBOT_VENV}/bin/python", "-m", "robotd.server", "--sim", "yam", "--scene", a.task,
            *(["--seed", str(a.seed)] if a.seed is not None else []), *(["--realtime"] if a.realtime else []),
            *[x for kv in a.scene_arg for x in ("--scene-arg", kv)],
            "--addr", f"unix:{tmp}/sock/robot.sock", "--control-addr", f"unix:{tmp}/ctl/control.sock",
            "--record", f"{run}/robotd", "--ready-file", f"{tmp}/ctl/robotd.ready", "--gpu", str(a.gpu)]
    env = dict(os.environ, PYTHONPATH=ROOT, MUJOCO_GL="egl", PYTHONUNBUFFERED="1", ROBOTD_CACHE=f"{run}/robot-files")
    log = open(f"{run}/logs/robotd.log", "w")
    proc = subprocess.Popen(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    proc.argv = argv
    return proc


def start_vision(a, run: str, tmp: str) -> subprocess.Popen:
    models = a.vision_models
    argv = [f"{PERCEPTION_VENV}/bin/python", "-m", "vision.server", "--addr", f"unix:{tmp}/sock/vision.sock",
            "--gpu", str(a.vision_gpu), "--models", models, "--preload", models,
            "--ready-file", f"{tmp}/ctl/vision.ready"]
    cmd = f"source {CACHE_ENV} && exec " + " ".join(argv)
    log = open(f"{run}/logs/vision.log", "w")
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(["bash", "-c", cmd], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    proc.argv = argv
    return proc


def control(tmp: str, req: dict, timeout: float = 120.0):
    from robot import rpc
    s = rpc.connect(f"unix:{tmp}/ctl/control.sock")
    s.settimeout(timeout)
    try:
        rpc.send(s, req)
        rep = rpc.recv(s)
    finally:
        s.close()
    if not rep or not rep.get("ok"):
        raise RuntimeError(f"robotd control {req['op']}: {rep}")
    return rep["result"]


def stop_process(proc: subprocess.Popen, grace: float, name: str, notes: list):
    """SIGTERM to this exact process, SIGKILL after grace seconds."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        notes.append(f"{name} did not exit {grace:.0f} s after SIGTERM: killed")
        proc.kill()
        proc.wait(10)


def git(ws: str, *args, git_dir: str | None = None):
    """git on the workspace, with no hooks, fsmonitor or global config. With git_dir the history lives there and the
    workspace is only its work tree: once the agent has run, its /workspace/.git (config, hooks, a gitfile) could make
    a git command here run anything, so the end state goes into a history outside the sandbox."""
    where = ["--git-dir", git_dir, "--work-tree", ws] if git_dir else ["-C", ws]
    return subprocess.run(["git", "-c", "user.name=robot-api-runner", "-c", "user.email=runner@robot-api.local",
                           "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *where, *args],
                          check=True, capture_output=True, text=True,
                          env=dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")).stdout


def repo_state() -> dict:
    def out(*args):
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()
    return {"commit": out("rev-parse", "HEAD"), "branch": out("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": out("status", "--porcelain").splitlines()}


def codex_version() -> str:
    env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
    return subprocess.run([f"{CODEX_PKG}/bin/codex", "--version"], capture_output=True, text=True,
                          env=env).stdout.strip()


def program_host_summary(path: str, notes: list) -> dict:
    """What the program host's log says: how many programs it started, and which it stopped at the episode's end
    (signal and exit code), before the evaluation."""
    events = []
    for line in open(path) if os.path.exists(path) else []:
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    end = next((e for e in events if e.get("event") == "end"), None)
    out = {"log": "logs/program-host.jsonl", "started": sum(e.get("event") == "start" for e in events),
           "stopped_at_end": None if end is None else end["stopped"]}
    if end is None:
        notes.append("the program host did not report its end: programs it started may have been killed with the "
                     "sandbox instead of stopped")
    elif end["stopped"]:
        notes.append(f"the program host stopped {len(end['stopped'])} program(s) still running at the end: "
                     + ", ".join(f"{os.path.relpath(x['run_dir'], WS)} ({x['signal']}, exit {x['returncode']})"
                                 for x in end["stopped"]))
    return out


def keep_tmp(src: str, dst: str, notes: list) -> None:
    """The sandbox's /tmp (src) into the run directory (dst) before it is deleted: regular files only (no links,
    sockets or devices), less caches (.cache), Codex's daemon directory and files over TMP_KEEP_MB."""
    skipped = 0
    for here, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != ".cache" and not d.startswith("codex-daemon-")
                   and not os.path.islink(os.path.join(here, d))]
        for name in files:
            path = os.path.join(here, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            if os.path.getsize(path) > TMP_KEEP_MB * 2**20:
                skipped += 1
                continue
            out = os.path.join(dst, os.path.relpath(path, src))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(path, out, follow_symlinks=False)
    if skipped:
        notes.append(f"sandbox/tmp: {skipped} file(s) over {TMP_KEEP_MB} MB not kept")


def write_prompt_md(path: str) -> None:
    """codex debug prompt-input's JSON as readable markdown next to it (.md): each message's role and text, and the
    tools by name."""
    items = json.load(open(path))
    out = []

    def tools(ts, depth=0):
        for t in ts:
            out.append("  " * depth + f"- {t.get('name') or t.get('type')}")
            tools(t.get("tools") or [], depth + 1)
    for it in items:
        if it.get("type") == "additional_tools":
            out.append(f"## tools ({it.get('role')})\n")
            tools(it.get("tools") or [])
            out.append("")
            continue
        for c in it.get("content") or []:
            out.append(f"## {it.get('role')} message\n\n{c.get('text', '')}\n")
    with open(path[:-5] + ".md", "w") as f:
        f.write("\n".join(out))


def read_pi_events(stream, out, stats: dict):
    """--agent pi: Pi's --mode json stream into out, less its message_update and tool_execution_update deltas
    (message_end and tool_execution_end carry the whole), counted into stats."""
    for raw in stream:
        line = raw.decode("utf-8", "replace")
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        kind = ev.get("type") if isinstance(ev, dict) else None
        if kind not in ("message_update", "tool_execution_update"):
            out.write(line)
            out.flush()
        if kind is None:
            continue
        stats["events"] += 1
        if kind == "session":
            stats["session_id"] = ev.get("id")
        elif kind == "message_end" and isinstance(ev.get("message"), dict) and ev["message"].get("role") == "assistant":
            m = ev["message"]
            stats["assistant_messages"] += 1
            for k, v in (m.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    stats["usage"][k] = stats["usage"].get(k, 0) + v
            if m.get("stopReason") in ("error", "aborted"):
                stats["errors"].append(m.get("errorMessage") or m.get("stopReason"))
            text = "".join(c.get("text", "") for c in m.get("content") or []
                           if isinstance(c, dict) and c.get("type") == "text")
            if text.strip():
                stats["last_message"] = text
        elif kind == "tool_execution_start":
            name, args = str(ev.get("toolName")), ev.get("args") if isinstance(ev.get("args"), dict) else {}
            stats["tool_calls"][name] = stats["tool_calls"].get(name, 0) + 1
            if name == "read" and str(args.get("path") or "").lower().endswith(IMAGE_EXT):
                stats["image_reads"] += 1
        elif kind == "tool_execution_end" and ev.get("isError"):
            stats["tool_errors"] += 1
        elif kind == "auto_retry_start":
            stats["retries"] += 1
        elif kind == "compaction_start":
            stats["compactions"] += 1


# ------------------------------------------------------------------------------------------------ the episode
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="a task in tasks/ (its scene, instruction and success check)")
    ap.add_argument("--seed", type=int, default=None, help="the scene variant")
    ap.add_argument("--scene-arg", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", help="run directory name under " + RUNS)
    ap.add_argument("--agent", choices=["codex", "pi", "standin", "pi-standin", "prompt-input"], default="codex",
                    help="codex (the default); pi: Pi on DeepSeek at --endpoint (module docstring); standin: a "
                    "scripted stand-in that probes the sandbox; pi-standin: the same in Pi's sandbox, which also runs "
                    "Pi on a test image; prompt-input: Codex renders its model-visible input (no model call), written "
                    "to codex/prompt-input.json")
    ap.add_argument("--prompt", default=PROMPT, help="the first user message (AGENTS.md carries the rest)")
    ap.add_argument("--model", choices=["gpt", "deepseek"], default=None,
                    help=f"the model Codex runs on: gpt, {MODEL} through the ChatGPT login (the default); deepseek, "
                    f"{DEEPSEEK['model']} at --endpoint (Pi's only model)")
    ap.add_argument("--endpoint", help="--model deepseek and --agent pi: an ora-router LLM allocation's gateway URL "
                    "(its llm.endpoint, https://HOST:PORT/v1)")
    ap.add_argument("--llm-allocation", help="--model deepseek and --agent pi: that allocation's id, recorded in "
                    "meta.json")
    clock = ap.add_mutually_exclusive_group()
    clock.add_argument("--realtime", action="store_true", default=None,
                       help="run the simulation on the wall clock (the default for a task with realtime=True)")
    clock.add_argument("--lockstep", dest="realtime", action="store_false",
                       help="run it in lockstep (the default for the other tasks)")
    ap.add_argument("--wall-h", type=float, default=6.0, help="hours of wall time for the agent")
    ap.add_argument("--gpu", type=int, default=2, help="GPU for robotd's EGL rendering")
    ap.add_argument("--no-vision", action="store_true", help="no vision service")
    ap.add_argument("--vision-gpu", type=int, default=3)
    ap.add_argument("--vision-models", default=VISION_MODELS, help="served and preloaded")
    ap.add_argument("--agent-gpu", type=int, default=None,
                    help="the GPU the agent's programs see (default: the vision GPU, else robotd's); -1: none")
    ap.add_argument("--egress-allow", action="append", default=None, metavar="HOST",
                    help=f"hosts the egress proxy lets Codex reach, HOST (port 443) or HOST:PORT (default "
                    f"{', '.join(EGRESS_ALLOW)}; with --model deepseek, the endpoint's host and port)")
    ap.add_argument("--mem-cap-gb", type=float, default=MEM_CAP_GB,
                    help="memory cap of the sandbox's processes together (RSS, GB of 2^30 bytes): past it the largest "
                    "is killed (MemoryWatchdog); robotd and the vision server are outside it")
    ap.add_argument("--mem-cap-process-gb", type=float, default=MEM_CAP_PROCESS_GB,
                    help="memory cap of any one process in the sandbox (RSS, GB): past it that process is killed")
    ap.add_argument("--appendix", metavar="FILE", help="append FILE to AGENTS.md, after the manual (e.g. "
                    "scripts/agent/appendix-working-habits.md)")
    ap.add_argument("--examples", metavar="DIR", help=f"bind DIR read-only at {EXAMPLES} and point AGENTS.md at its "
                    "README.md (worked episodes of other tasks)")
    ap.add_argument("--print", choices=["agents-md", "all"], help="print what the agent would get and exit")
    a = ap.parse_args()
    if not 0 < a.mem_cap_process_gb <= a.mem_cap_gb:
        ap.error("--mem-cap-process-gb must be positive and at most --mem-cap-gb")
    memory = (a.mem_cap_process_gb, a.mem_cap_gb)
    pi = a.agent in ("pi", "pi-standin")
    if pi and a.model == "gpt":
        ap.error("Pi runs on DeepSeek only (--model deepseek, its default)")
    a.model = a.model or ("deepseek" if pi else "gpt")
    if pi and a.egress_allow:
        ap.error("--agent pi has no egress proxy: its only way out is the model proxy to --endpoint")
    if a.model == "deepseek" and not a.endpoint:
        ap.error("--model deepseek needs --endpoint (an ora-router LLM allocation's llm.endpoint)")
    appendix = open(a.appendix).read() if a.appendix else None
    if a.examples:
        a.examples = os.path.realpath(a.examples)
        if not os.path.isfile(os.path.join(a.examples, "README.md")):
            ap.error(f"--examples {a.examples} has no README.md")
    extra = {"pi": pi, "examples": bool(a.examples), "appendix": appendix}     # render_agents_md's agent lines
    llm = llm_spec(a.model, a.endpoint, a.tag or "<tag>")
    egress = a.egress_allow or llm["egress"]
    vision = not a.no_vision
    if a.agent_gpu is None:
        a.agent_gpu = a.vision_gpu if vision else a.gpu
    import tasks
    task = tasks.get(a.task)
    params = inspect.signature(task.scene).parameters.values()
    if a.seed is not None and not any(p.name == "seed" or p.kind is p.VAR_KEYWORD for p in params):
        ap.error(f"task {a.task} takes no seed (its scene is fixed): leave out --seed")
    instruction = task.instruction_for(a.seed)
    if not instruction:
        ap.error(f"task {a.task} has no instruction")
    settle_s = 0.0 if task.ends_at_finish else SETTLE_S
    if a.realtime is None:
        a.realtime = task.realtime
    clock = "realtime" if a.realtime else "lockstep"

    if a.print:
        gpu = None if a.agent_gpu < 0 else (gpu_info(a.agent_gpu)[1] if shutil.which("nvidia-smi") else "one GPU")
        print(render_agents_md(instruction, vision, a.wall_h, "<launch time, UTC>", gpu, clock, memory, **extra))
        if a.print == "all" and pi:
            print("\n" + "=" * 100 + "\nfirst user message:\n" + a.prompt)
            print("\n" + "=" * 100 + "\ncommand in the sandbox:\n" + " ".join(pi_argv(a.prompt)))
            print("\n" + "=" * 100 + f"\n{PI_HOME}/agent/settings.json:\n" + json.dumps(pi_settings(), indent=1))
            print("\n" + "=" * 100 + f"\n{PI_HOME}/agent/models.json:\n" + json.dumps(pi_models("<token>"), indent=1))
            print("\n" + "=" * 100 + f"\nmodel proxy: http://127.0.0.1:{PROXY_PORT}/v1 -> {a.endpoint} (X-Session-Id: "
                  f"{a.tag or '<tag>'})")
            print("\n" + "=" * 100 + f"\n{CODEX_HOME}/config.toml (the command sandbox):\n"
                  + command_sandbox_config(vision))
        elif a.print == "all":
            print("\n" + "=" * 100 + "\nfirst user message:\n" + a.prompt)
            print("\n" + "=" * 100 + "\ncommand in the sandbox:\n"
                  + " ".join(agent_argv("codex", a.prompt, "RUN", llm)))
            print("\n" + "=" * 100 + f"\n{CODEX_HOME}/config.toml:\n" + codex_config(vision, llm))
        return
    if not a.tag:
        ap.error("--tag is required for a run")
    run = os.path.join(RUNS, a.tag)
    if os.path.exists(run):
        ap.error(f"{run} exists: pick another --tag")
    if a.model == "gpt" and not os.path.exists(CODEX_AUTH):
        ap.error("no Codex login (auth.json) to bind into the sandbox")
    if a.model == "deepseek" and not os.access(DEEPSEEK["key_file"], os.R_OK):
        ap.error(f"cannot read the DeepSeek key file {DEEPSEEK['key_file']}")

    # ---- run directory
    for sub in ("workspace", "robotd", "robot-files", "vision", "sandbox", "logs",
                *(("pi/home/agent", "pi/home/sessions", "sandbox/codex-home") if pi else ("codex/home",))):
        os.makedirs(os.path.join(run, sub))
    tmp = tempfile.mkdtemp(prefix=f"robotapi-{a.tag[:40]}-", dir="/tmp")
    for sub in ("sock", "ctl", "egress", "host", "tmp"):
        os.makedirs(os.path.join(tmp, sub), mode=0o700)
    code = os.path.join(run, "sandbox", "robot-api")
    for rel in AGENT_CODE + (["vision", "docs/VISION.md"] if vision else []):
        src, dst = os.path.join(ROOT, rel), os.path.join(code, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*AGENT_CODE_SKIP))
        else:
            shutil.copy2(src, dst)
    t_start = time.time()
    ws = os.path.join(run, "workspace")
    history = os.path.join(run, "workspace.git")
    password = secrets.token_urlsafe(24)                # the egress proxy's password, or Pi's token
    if pi:
        for path, obj in (("pi/home/agent/settings.json", pi_settings()), ("pi/home/agent/models.json",
                                                                            pi_models(password))):
            write_json(os.path.join(run, path), obj)
        with open(os.path.join(run, "sandbox", "codex-home", "config.toml"), "w") as f:
            f.write(command_sandbox_config(vision))
    else:
        with open(os.path.join(run, "codex", "home", "config.toml"), "w") as f:
            f.write(codex_config(vision, llm))
    if llm["provider"] is None:
        open(os.path.join(run, "codex", "home", "auth.json"), "w").close()     # an empty mount point for the bind
    gpu_dev, gpu = gpu_info(a.agent_gpu) if a.agent_gpu >= 0 else (None, None)
    stop_r, stop_w = os.pipe()
    inner = agent_argv(a.agent, a.prompt, run, llm)
    meta = {"schema": "robotapi.agent-run.v1", "tag": a.tag, "task": a.task, "seed": a.seed,
            "scene_args": tasks.scene_args(a.scene_arg), "instruction": instruction, "agent": a.agent,
            "clock": clock, "repo": repo_state(), "host": socket.gethostname(),
            **({"pi": {**pi_info(), "model": DEEPSEEK["model"], "reasoning_effort": DEEPSEEK["effort"],
                       "max_output_tokens": DEEPSEEK["max_output_tokens"],
                       "context_window": DEEPSEEK["context_window"], "api": "openai-completions (streamed)",
                       "endpoint": a.endpoint, "llm_allocation": a.llm_allocation,
                       "session_affinity": {"header": "X-Session-Id", "value": a.tag}, "argv": inner,
                       "prompt": a.prompt, "settings": pi_settings(), "models": pi_models("<token>"),
                       "extensions": PI_EXTENSIONS, "bash_default_timeout_s": PI_BASH_TIMEOUT_S}} if pi else
               {"codex": {"version": codex_version(), "model": llm["model"], "reasoning_effort": llm["effort"],
                          "model_choice": a.model, "argv": inner, "config": codex_config(vision, llm),
                          "prompt": a.prompt,
                          **({"provider": {**llm["provider"], "llm_allocation": a.llm_allocation,
                                           "max_output_tokens_note": "the server's limit; Codex 0.165 sends none"}}
                             if llm["provider"] else {})}}),
            "sandbox": ({"kind": "bwrap+pi", "network": "model-proxy", "model_proxy": {
                             "bridge": f"127.0.0.1:{PROXY_PORT}", "socket": f"{EGRESS}/model.sock",
                             "upstream": egress, "routes": [" ".join(r) for r in ModelProxy.ROUTES]},
                         "command_sandbox": {"codex_version": codex_version(), "config": command_sandbox_config(vision)},
                         "argv": "sandbox/bwrap.json"} if pi else
                        {"kind": "bwrap+codex", "network": "egress-proxy",
                         "egress_allow": egress, "argv": "sandbox/bwrap.json"}),
            "robotd": {"gpu": a.gpu}, "vision": {"on": vision, "gpu": a.vision_gpu if vision else None,
                                                 "models": a.vision_models if vision else None},
            "agent_gpu": {"index": a.agent_gpu, "device": gpu_dev, "what": gpu} if gpu_dev else None,
            "limits": {"wall_s": a.wall_h * 3600, "memory_gb": a.mem_cap_gb,
                       "process_memory_gb": a.mem_cap_process_gb},
            **({"appendix": {"path": os.path.realpath(a.appendix), "sha256": hashlib.sha256(appendix.encode())
                             .hexdigest(), "bytes": len(appendix.encode())}} if appendix is not None else {}),
            **({"examples": {"dir": a.examples, "at": EXAMPLES, "files": tree_digest(a.examples)}}
               if a.examples else {}),
            "settle_s": settle_s, "state": "starting",
            "started_utc": utc(t_start),
            "ended_utc": None, "notes": []}
    meta_path = os.path.join(run, "meta.json")
    write_json(meta_path, meta)
    print(f"run dir: {run}", flush=True)

    robotd = vis = agent = proxy = secret = t_agent = watchdog = None
    audit = {"scans": 0, "pids": set(), "read": 0, "unreadable": 0, "hits": []}
    notes = meta["notes"]
    end_reason, rc = None, None
    stopping = threading.Event()

    def on_signal(signum, frame):
        stopping.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        # ---- services
        if a.agent != "prompt-input":
            robotd = start_robotd(a, run, tmp)
            meta["robotd"]["argv"] = robotd.argv
            if vision:
                vis = start_vision(a, run, tmp)
                meta["vision"]["argv"] = vis.argv
                with open(os.path.join(run, "vision", "server.pid"), "w") as f:
                    f.write(str(vis.pid))
            _wait_file(f"{tmp}/ctl/robotd.ready", robotd, 180, "robotd")
            if vis is not None:
                _wait_file(f"{tmp}/ctl/vision.ready", vis, 900, "the vision server")
        if pi:                                           # the key stays in this process: the model proxy adds it
            secret = open(DEEPSEEK["key_file"], "rb").read().strip()
            proxy = ModelProxy(f"{tmp}/egress/model.sock", a.endpoint, secret.decode(), a.tag, password,
                               DEEPSEEK["ca"], os.path.join(run, "pi", "requests.jsonl"),
                               os.path.join(run, "pi", "first-request.json"))
        else:
            proxy = EgressProxy(f"{tmp}/egress/proxy.sock", password, egress,
                                os.path.join(run, "codex", "egress.jsonl"))
        models = model_files(control(tmp, {"op": "description"})) if robotd is not None else []
        meta["sandbox"]["model_files"] = models
        host_log = os.open(os.path.join(run, "logs", "program-host.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        key_r = None
        if llm["provider"] and not pi:                       # the key goes to bwrap through a pipe (--args)
            key = open(DEEPSEEK["key_file"], "rb").read().strip()
            key_r, key_w = os.pipe()
            os.write(key_w, b"\0".join([b"--setenv", llm["provider"]["env_key"].encode(), key, b""]))
            os.close(key_w)
            del key
        argv = bwrap_argv(run, tmp, inner, vision, a.agent in ("standin", "pi-standin"), stop_r, host_log, password,
                          models, gpu_dev, llm, key_r, pi, a.examples)
        write_json(os.path.join(run, "sandbox", "bwrap.json"), [x.replace(password, "<password>") for x in argv])

        # ---- the agent: its clock starts with AGENTS.md
        if stopping.is_set():
            end_reason = "interrupted"
            raise RuntimeError("interrupted before the agent started")
        t_agent = time.time()
        with open(os.path.join(ws, "AGENTS.md"), "w") as f:
            f.write(render_agents_md(instruction, vision, a.wall_h, utc(t_agent) + " UTC", gpu, clock, memory,
                                     **extra))
        for gd in (None, history):             # the agent's own repo, and the history it cannot reach
            git(ws, "init", "-q", "-b", "main", git_dir=gd)
            git(ws, "add", "AGENTS.md", git_dir=gd)
            git(ws, "commit", "-q", "-m", "episode start", git_dir=gd)
        meta.update(state="running", agent_started_utc=utc(t_agent))
        write_json(meta_path, meta)
        out_name = "prompt-input.json" if a.agent == "prompt-input" else "events.jsonl"
        events = open(os.path.join(run, "pi" if pi else "codex", out_name), "w")
        stderr = open(os.path.join(run, "pi" if pi else "codex", "stderr.log"), "w")
        agent = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr, stdin=subprocess.DEVNULL,
                                 pass_fds=(stop_r, host_log, *([key_r] if key_r is not None else [])),
                                 start_new_session=True,
                                 # bwrap's PID 1 in the sandbox keeps the environment bwrap was started with, and Pi's
                                 # read tool, unsandboxed, could read it there (/proc/1/environ): the launcher's holds
                                 # credentials. The sandbox's processes get theirs from --clearenv/--setenv.
                                 **({"env": {}} if pi else {}))
        watchdog = MemoryWatchdog(agent.pid, a.mem_cap_process_gb, a.mem_cap_gb,
                                  os.path.join(run, "logs", "memory.jsonl"), notes, t_agent)
        os.close(stop_r)
        if key_r is not None:
            os.close(key_r)
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
        if pi:
            stats = {"events": 0, "session_id": None, "assistant_messages": 0, "usage": {}, "tool_calls": {},
                     "image_reads": 0, "tool_errors": 0, "retries": 0, "compactions": 0, "errors": [],
                     "last_message": None}
            reader = threading.Thread(target=read_pi_events, args=(agent.stdout, events, stats), daemon=True)
        else:
            reader = threading.Thread(target=read_events, daemon=True)
        reader.start()
        deadline, next_scan = t_agent + a.wall_h * 3600, 0.0
        while True:
            if agent.poll() is not None:
                end_reason = "agent_exit"
                break
            if time.time() > deadline:
                end_reason = "wall_limit"
                break
            if stopping.is_set():
                end_reason = "interrupted"
                break
            if robotd is not None and robotd.poll() is not None:
                end_reason = "robotd_exit"
                notes.append(f"robotd exited ({robotd.returncode}) during the episode")
                break
            if secret is not None and time.time() >= next_scan:     # the key in no sandbox process, every 30 s
                scan_processes(agent.pid, secret, audit)
                next_scan = time.time() + 30
            time.sleep(1.0)
        if agent.poll() is None:                         # sandbox_init.py stops the agent, then the program host
            os.write(stop_w, b"x")
            try:
                agent.wait(60)
            except subprocess.TimeoutExpired:
                notes.append("the sandbox did not exit 60 s after the stop request: killed")
                os.killpg(agent.pid, signal.SIGKILL)
                agent.wait(10)
        rc = agent.returncode
        reader.join(10)
        events.close()
        stderr.close()
        if a.agent == "prompt-input":
            write_prompt_md(os.path.join(run, "codex", "prompt-input.json"))
        t_end = time.time()
        meta.update(agent_ended_utc=utc(t_end), agent_wall_s=round(t_end - t_agent, 1), end_reason=end_reason,
                    agent_exit_code=rc)
        if pi:
            meta.update(pi_session_id=stats["session_id"], pi_events=stats["events"],
                        pi_assistant_messages=stats["assistant_messages"], pi_tool_calls=stats["tool_calls"],
                        pi_image_reads=stats["image_reads"], pi_tool_errors=stats["tool_errors"],
                        pi_retries=stats["retries"], pi_compactions=stats["compactions"], pi_usage=stats["usage"],
                        pi_errors=stats["errors"][:20])
            if stats["last_message"] is not None:
                with open(os.path.join(run, "pi", "last_message.txt"), "w") as f:
                    f.write(stats["last_message"])
        else:
            meta.update(codex_thread_id=stats["thread_id"], codex_events=stats["events"], codex_turns=stats["turns"],
                        codex_usage=stats["usage"], codex_errors=stats["errors"][:20])
    except Exception as e:                               # noqa: BLE001 -- still evaluate and stop what started
        import traceback
        traceback.print_exc()
        end_reason = end_reason or "runner_error"
        notes.append(f"runner error: {e!r}")
        meta["end_reason"] = end_reason
        if agent is not None and agent.poll() is None:
            os.killpg(agent.pid, signal.SIGKILL)
    finally:
        if agent is not None:                            # the sandbox has exited: the host's log is complete
            meta["program_host"] = program_host_summary(os.path.join(run, "logs", "program-host.jsonl"), notes)
        # ---- verdict, then stop the services
        verdict = None
        if robotd is not None and robotd.poll() is None:
            try:
                time.sleep(0.5)                          # robotd releases the agent's dropped connections
                ev = control(tmp, {"op": "evaluate", "settle_s": settle_s})
                finish_path = os.path.join(run, "robotd", "verdict.json")
                at_finish = json.load(open(finish_path)) if os.path.exists(finish_path) else None
                verdict = {"verdict": ev["verdict"], "verdict_from": "end", "t": ev["t"], "settled_s": ev["settle_s"],
                           "t_before_settle": ev["t_before_settle"], "evaluated_utc": utc(),
                           "finished": ev["finished"], "at_finish": at_finish}
                if task.ends_at_finish and at_finish is not None:   # the check covers the episode up to r.finish()
                    verdict.update(verdict=at_finish["verdict"], verdict_from="finish", at_end=ev["verdict"])
                write_json(os.path.join(run, "verdict.json"), verdict)
            except Exception as e:                       # noqa: BLE001
                notes.append(f"evaluation failed: {e!r}")
            try:
                control(tmp, {"op": "shutdown"})
                robotd.wait(60)
            except Exception as e:                       # noqa: BLE001
                notes.append(f"robotd shutdown: {e!r}")
        stop_process(robotd, 30, "robotd", notes)
        stop_process(vis, 20, "the vision server", notes)
        timing_path = os.path.join(run, "robotd", "timing.json")
        if os.path.exists(timing_path):                  # real time: did the simulation keep its 4 ms period?
            tm = json.load(open(timing_path))
            meta["timing"] = {**tm, "late_fraction": round(tm["late_ticks"] / max(tm["ticks"], 1), 4)}
            if tm.get("resyncs"):
                notes.append(f"the real-time simulation fell more than 0.1 s behind {tm['resyncs']} time(s)")
        elif a.realtime and robotd is not None:
            notes.append("real time, but robotd wrote no timing.json")
        if proxy is not None:
            proxy.close()
        if watchdog is not None:
            watchdog.close()
            meta["memory"] = watchdog.summary()
        try:
            os.close(stop_w)
        except OSError:
            pass
        auth_stub = os.path.join(run, "codex", "home", "auth.json")
        if os.path.exists(auth_stub) and os.path.getsize(auth_stub) == 0:
            os.remove(auth_stub)
        try:
            if os.path.isdir(history):
                git(ws, "add", "-A", git_dir=history)
                git(ws, "commit", "-q", "--allow-empty", "-m", f"episode end ({end_reason})", git_dir=history)
        except subprocess.CalledProcessError as e:
            notes.append(f"workspace commit failed: {e.stderr.strip()}")
        try:
            keep_tmp(f"{tmp}/tmp", os.path.join(run, "sandbox", "tmp"), notes)
        except OSError as e:
            notes.append(f"keeping the sandbox's /tmp failed: {e!r}")
        shutil.rmtree(tmp, ignore_errors=True)
        if pi:
            if isinstance(proxy, ModelProxy):
                meta["model_requests"] = proxy.summary(t_agent or t_start)
            sessions = sorted(glob.glob(os.path.join(run, "pi", "home", "sessions", "**", "*.jsonl"), recursive=True))
            meta["pi_session_files"] = [os.path.relpath(x, run) for x in sessions]
            if sessions:
                shutil.copy2(sessions[-1], os.path.join(run, "pi", "session.jsonl"))
            if len(sessions) > 1:
                notes.append(f"Pi wrote {len(sessions)} session files; pi/session.jsonl is the last")
            if secret is not None:                       # the key in no file of the run, and in no sandbox process
                files = scan_files(run, secret)
                meta["key_audit"] = {"processes": {k: len(v) if k == "pids" else v for k, v in audit.items()},
                                     "files": files}
                if audit["hits"] or files["hits"]:
                    notes.append(f"THE KEY WAS FOUND: processes {audit['hits']}, files {files['hits']}")
        finished = (verdict or {}).get("finished")
        if a.agent != "prompt-input" and finished is None:
            notes.append("the agent never called r.finish(); the episode was not resumed")
        meta.update(state="done", ended_utc=utc(), wall_s=round(time.time() - t_start, 1), finished=finished,
                    verdict=(verdict or {}).get("verdict"), resumed=False)
        write_json(meta_path, meta)
        print(f"end: {end_reason}, agent exit code {rc}; verdict: {json.dumps((verdict or {}).get('verdict'))}")
        print(f"run dir: {run}")


if __name__ == "__main__":
    main()
