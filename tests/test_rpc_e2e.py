"""End to end: a driver process, a policy process, the hidden verdict."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pick_cube_policy_end_to_end(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_sim.py"), "--scene", "pick_cube",
                          "--policy", os.path.join(ROOT, "examples", "pick_cube.py"), "--record", str(tmp_path)],
                         capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert "policy exit code 0" in out.stdout, out.stdout + out.stderr
    verdict = json.load(open(tmp_path / "verdict.json"))
    assert verdict["verdict"]["success"], verdict
    assert (tmp_path / "requests.jsonl").exists()
    frames = os.listdir(tmp_path / "frames")                     # each camera fetch: rgb, and depth as 16-bit mm
    assert any(f.endswith("-top.png") for f in frames) and any(f.endswith("-top_depth.png") for f in frames)
    from robotd.core.recorder import load_ticks
    ticks = load_ticks(str(tmp_path))
    assert len(ticks["t"]) > 1000 and ticks["q"].shape[1] == 14
