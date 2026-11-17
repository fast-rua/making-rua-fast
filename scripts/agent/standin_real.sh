#!/bin/bash
# A stand-in for `codex exec` on the real cell (scripts/agent_run_real.py --agent standin), to test the one-layer
# sandbox without a model and WITHOUT MOVING THE ARMS: it reads the robot's state and one camera frame, and sends no
# command. Each command runs as Codex runs one there: `bash -c` in the outer sandbox with the proxy variables removed
# (Codex's shell_environment_policy; no command sandbox of its own). It prints one JSON event per command on stdout.
# The probes print only whether a path is readable, never its contents.
#
#     standin_real.sh HIDDEN_PATH...       the real paths a command must fail to read (the repo, the runs, ...)
HIDDEN=("$@")
UNSET=(-u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy)

emit() {    # emit COMMAND OUTPUT EXIT_CODE WHERE
  python3 -c 'import json, sys
print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "where": sys.argv[4],
  "command": sys.argv[1], "aggregated_output": sys.argv[2], "exit_code": int(sys.argv[3])}}), flush=True)' \
    "$1" "$2" "$3" "$4"
}
run() {     # a command as Codex runs one here
  local out rc
  out=$(cd /workspace && env "${UNSET[@]}" bash -c "$1" 2>&1); rc=$?
  emit "$1" "$out" $rc "command (one layer: the outer sandbox, proxy variables removed)"
}
outer() {   # as the Codex process itself
  local out rc
  out=$(bash -c "$1" 2>&1); rc=$?
  emit "$1" "$out" $rc "outer sandbox (the Codex process)"
}

echo '{"type":"thread.started","thread_id":"standin-real"}'
echo '{"type":"turn.started"}'
PROBE='p() { if [ -d "$1" ]; then ls "$1" >/dev/null 2>&1 && echo "READABLE dir $1" || echo "denied $1: $(ls "$1" 2>&1 | head -1)"; elif head -c 1 "$1" >/dev/null 2>&1; then echo "READABLE $1"; else echo "denied $1: $(head -c 1 "$1" 2>&1 >/dev/null | head -1)"; fi; }'

outer "id; hostname; ls /; ls /run /run/robot-api /opt; /opt/codex/bin/codex --version; $PROBE; p /codex-home/auth.json"
outer "curl -sS -m 15 -o /dev/null -w 'chatgpt.com via the proxy, with the password: HTTP %{http_code}\n' https://chatgpt.com; curl -sS -m 10 -o /dev/null https://example.com 2>&1 | head -1"

run "pwd; id; ls -la; head -3 AGENTS.md; echo PATH=\$PATH; which python; python -V; ls /opt/robot-api /opt/robot-api/docs; env | grep -c -i proxy | sed 's/^/proxy variables: /'"
run "$PROBE; for x in ${HIDDEN[*]} /root; do p \$x; done; p /codex-home/auth.json"
# a directory that exists only as the parent of a bound file (the model files r.description names) is readable but
# holds only that file's path: what is in them, and the solutions and private files by name
R=${HIDDEN[0]}
run "ls -A /home $R 2>&1; find /home -maxdepth 6 -type f 2>/dev/null | head -20; $PROBE; for x in $R/examples/towel/fold_towel.py $R/tasks/real_yam.py $R/scripts/agent_run.py; do p \$x; done"
run "touch /opt/robot-api/x 2>&1 | head -1; touch /usr/x 2>&1 | head -1; touch /workspace/ok && echo 'workspace writable'; touch /tmp/ok && echo '/tmp writable'"
run "python - <<'EOF'
import socket
for host, port in (('127.0.0.1', 7410), ('127.0.0.1', 7412), ('1.1.1.1', 443)):
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect((host, port))
        print('TCP', host, port, 'REACHED')
    except OSError as e:
        print('TCP', host, port, 'unreachable:', e)
    s.close()
EOF
curl -sS -m 10 -o /dev/null https://chatgpt.com 2>&1 | head -1; ip -br link 2>/dev/null || cat /proc/net/dev | tail -n +3 | cut -d: -f1"

# the robot: state and one frame through the sockets bound in, and the files the description names. NO COMMANDS.
run "cat > probe.py <<'EOF'
import os, xml.etree.ElementTree as ET
import numpy as np
import robot
r = robot.connect()
s = r.state()
print('connected:', r.address if hasattr(r, 'address') else os.environ['ROBOT_ADDR'], '| arms', list(r.end_effectors))
for a in r.end_effectors:
    print(' ', a, 'TCP', np.round(s.ee[a].T[:3, 3], 3).tolist())
print('fence', r.description.get('fence'))
for cam in ('top', 'left_wrist'):
    f = r.cameras([cam])[cam]
    print('camera', cam, f.rgb.shape, None if f.depth is None else f.depth.shape, 'pose', f.T is not None)
import cv2
f = r.cameras(['top'])['top']
hsv = cv2.cvtColor(np.ascontiguousarray(f.rgb), cv2.COLOR_RGB2HSV)
n, _ = cv2.connectedComponents((hsv[..., 1] > 80).astype(np.uint8))
print('cv2', cv2.__version__, 'top HSV', hsv.shape, 'saturated components', n - 1)
paths, bad, meshes = [], [], 0
def walk(v):
    if isinstance(v, dict):
        [walk(x) for x in v.values()]
    elif isinstance(v, list):
        [walk(x) for x in v]
    elif isinstance(v, str) and v.startswith('/'):
        paths.append(v)
walk(r.description)
for p in paths:
    try:
        open(p).close()
        if p.endswith('.urdf'):
            for el in ET.parse(p).getroot().iter('mesh'):
                open(os.path.join(os.path.dirname(p), el.get('filename'))).close()
                meshes += 1
    except OSError as e:
        bad.append(str(e))
print('description: paths', len(paths), 'opened, URDF meshes', meshes, 'opened, unreadable', bad)
EOF
python probe.py"

# the camera recordings, read-only, and ffmpeg on PATH (scripts/record_episode.py --streams writes them)
run "ls -l /var/record/cameras; which ffmpeg; ffmpeg -hide_banner -version | head -1; touch /var/record/cameras/x 2>&1 | head -1; python - <<'EOF'
import time
from robot.recording import Recording
rec = Recording()
img, t = rec.frame_at('top', time.time())
print('recordings:', rec.cameras, rec.fps, 'fps,', len(rec.times), 'frames; latest top frame', img.shape,
      round(time.time() - t, 2), 's old')
EOF"

# the hand-off runner: a program the program host keeps running after the command that asked returns (no robot)
run "cat > handoff_probe.py <<'EOF'
import time
def slow(r, v, s):
    time.sleep(6)                                     # outlives a --wait 2 return
    s.n = 41
def check(r, v, s):
    assert s.n == 42, 'deliberate: s.n is %d' % s.n
    s.after = True
STEPS = [slow, check]
EOF
python -m robot run handoff_probe.py --wait 2; echo exit=\$?"
run "python -m robot wait --wait 30; echo exit=\$?"
run "python -m robot debug '!s.n = 42'; python -m robot debug c; echo exit=\$?"
run "python -m robot wait --wait 30; echo exit=\$?; tail -3 .robot/current/log.txt 2>/dev/null"

# a program still running when the agent ends: the program host stops it
run "cat > sleeper.py <<'EOF'
import time
def sleep(r, v, s):
    print('sleeping past the end of the episode', flush=True)
    time.sleep(3600)
STEPS = [sleep]
EOF
python -m robot run sleeper.py --wait 2; echo exit=\$?"

echo '{"type":"item.completed","item":{"type":"agent_message","text":"stand-in done (no robot commands sent)"}}'
echo '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,"output_tokens":0}}'
