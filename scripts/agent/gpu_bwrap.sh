#!/bin/bash
# Codex's command sandbox, plus the episode's GPU. Codex runs the `bwrap` it finds on PATH (else its bundled one) as
# `bwrap ... --ro-bind / / --dev /dev ...`, and that fresh /dev has no GPU. When the agent gets a GPU, agent_run.py
# puts this script first on the outer sandbox's PATH as `bwrap`: it adds the card's device nodes
# ($ROBOT_SANDBOX_GPU_DEVICES) right after `--dev /dev` and runs Codex's bundled bubblewrap with everything else as is.
args=()
while [ $# -gt 0 ]; do
  args+=("$1")
  if [ "$1" = "--dev" ] && [ "${2-}" = "/dev" ]; then
    args+=("$2")
    shift
    for dev in $ROBOT_SANDBOX_GPU_DEVICES; do
      args+=(--dev-bind "$dev" "$dev")
    done
  fi
  shift
done
exec /opt/codex/codex-resources/bwrap "${args[@]}"
