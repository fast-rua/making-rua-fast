#!/usr/bin/env bash
# Install everything this repository needs into one folder, $ROBOT_HOME (default ~/robot-cell), as your own user:
# no sudo, nothing system-wide; removing it means deleting the folder (docs/SETUP.md).
#
#   bash scripts/install.sh            # the simulator, the robot API and Codex (Linux)
#   bash scripts/install.sh --cell     # + the real cell's camera driver (RealSense)
#
# It makes $ROBOT_HOME/.venv (Python 3.11), $ROBOT_HOME/i2rt (i2rt v1.4.0: the YAM model and motor driver),
# $ROBOT_HOME/codex/current (the Codex CLI the agent episodes run, 0.165.0), and $ROBOT_HOME/env.sh to source in
# every shell. Needs uv (https://docs.astral.sh/uv/) and git.
set -euo pipefail
CELL=0; [ "${1:-}" = "--cell" ] && CELL=1
REPO=$(cd "$(dirname "$0")/.." && pwd)
H=${ROBOT_HOME:-$HOME/robot-cell}
CODEX_VERSION=0.165.0
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
mkdir -p "$H" && cd "$H"

echo "== i2rt v1.4.0 in $H/i2rt"
[ -d i2rt ] || git -c advice.detachedHead=false clone -q --depth 1 --branch v1.4.0 https://github.com/i2rt-robotics/i2rt.git i2rt

echo "== Python 3.11 venv in $H/.venv"
[ -d .venv ] || uv venv -q --python 3.11 .venv
PY=$H/.venv/bin/python
uv pip install -q --python "$PY" numpy scipy "mujoco>=3.2" ruckig pillow pytest \
    "opencv-python-headless==4.11.0.86" "imageio-ffmpeg==0.6.0"
uv pip install -q --python "$PY" --no-deps -e i2rt
uv pip install -q --python "$PY" "mink>=0.0.13" "qpsolvers[quadprog]==4.7.1" "pyyaml>=6.0.0" "pydantic>=2.0"
# i2rt's motor driver and its simulated robot (YAM_SIM=1): the real backend's tests use them, on any machine
uv pip install -q --python "$PY" "python-can==4.5.0" "crcmod-plus>=2.3.1" "dm-env==1.6" "portal==3.7.3" \
    "tyro>=1.0.0" "click<8.2.0" "threadpoolctl==3.6.0" "mcap-ros2-support>=0.5.5"
if [ $CELL = 1 ]; then
  echo "== the cell's camera driver: RealSense"
  uv pip install -q --python "$PY" pyrealsense2
fi

echo "== Codex CLI $CODEX_VERSION in $H/codex/current"
pkg=codex-package-$(uname -m)-unknown-linux-musl.tar.gz
if [ ! -x "codex/$CODEX_VERSION/bin/codex" ]; then
  mkdir -p "codex/$CODEX_VERSION"
  curl -fsSL -o "codex/$pkg" "https://github.com/openai/codex/releases/download/rust-v$CODEX_VERSION/$pkg"
  echo "   sha256 $(sha256sum "codex/$pkg" | cut -d' ' -f1): compare it with the release page's"
  tar xzf "codex/$pkg" -C "codex/$CODEX_VERSION"
fi
ln -sfn "$CODEX_VERSION" codex/current
command -v bwrap >/dev/null || echo "   note: no system bwrap; episodes use the one that comes with Codex (it needs unprivileged user namespaces)"

echo "== $H/env.sh"
if [ ! -f env.sh ]; then
  sed -e "s|^export ROBOT_HOME=.*|export ROBOT_HOME=$H|" -e "s|^export PYTHONPATH=\$ROBOT_HOME/code|export PYTHONPATH=$REPO|" \
      "$REPO/configs/real_yam/env.sh.example" > env.sh
  echo "   written; for the real cell, fill in its <...> values"
else
  echo "   kept (already there)"
fi

echo "== check"
source env.sh 2>/dev/null || true
cd /tmp && "$PY" -c "import numpy, scipy, mujoco, ruckig, i2rt, robot, robotd.server; print('ok: mujoco', mujoco.__version__)"
echo "next: source $H/env.sh (docs/SETUP.md)"
