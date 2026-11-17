"""Start a simulated driver, run one policy against it as a separate process, shut down, print the verdict.

    python scripts/run_sim.py --scene pick_cube --policy examples/pick_cube.py [--record runs/x]
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="empty")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=None, help="scene variant (e.g. how the towel lies crumpled)")
    ap.add_argument("--record", default=None)
    ap.add_argument("--timeout", type=float, default=3600, help="s of wall time for the policy")
    ap.add_argument("--realtime", action="store_true", help="run the simulation against the wall clock")
    ap.add_argument("--scene-arg", action="append", default=[], metavar="KEY=VALUE", help="passed to the driver")
    a = ap.parse_args()
    run_dir = a.record or tempfile.mkdtemp(prefix="robotapi-run-", dir=os.path.join(ROOT, "scratch"))
    os.makedirs(run_dir, exist_ok=True)
    sock = os.path.join(tempfile.mkdtemp(prefix="rsock-"), "robot.sock")
    ready = os.path.join(run_dir, "ready")
    env = dict(os.environ, PYTHONPATH=ROOT, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), PYTHONUNBUFFERED="1")
    srv = subprocess.Popen([sys.executable, "-m", "robotd.server", "--sim", "yam", "--scene", a.scene,
                            *(["--seed", str(a.seed)] if a.seed is not None else []),
                            *(["--realtime"] if a.realtime else []),
                            *[x for kv in a.scene_arg for x in ("--scene-arg", kv)],
                            "--addr", f"unix:{sock}", "--record", run_dir, "--ready-file", ready], env=env, cwd=ROOT)
    try:
        t0 = time.time()
        while not os.path.exists(ready):
            if srv.poll() is not None or time.time() - t0 > 60:
                raise SystemExit("driver did not start")
            time.sleep(0.1)
        t0 = time.time()
        # the policy's output goes to the terminal and to policy.log in the run directory (the run viewer shows it)
        with open(os.path.join(run_dir, "policy.log"), "w") as log:
            pol = subprocess.Popen([sys.executable, a.policy], env=dict(env, ROBOT_ADDR=f"unix:{sock}"), cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            timer = threading.Timer(a.timeout, pol.kill)
            timer.start()
            for line in pol.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            rc = pol.wait()
            timed_out = not timer.is_alive()
            timer.cancel()
        print(f"policy exit code {rc}{' (timed out)' if timed_out else ''}, wall {time.time() - t0:.1f}s")
    finally:
        sys.path.insert(0, ROOT)
        try:
            from robot import rpc
            s = rpc.connect(f"unix:{sock}"); rpc.send(s, {"op": "shutdown"}); rpc.recv(s)
        except Exception:
            srv.terminate()
        srv.wait(timeout=30)
    vpath = os.path.join(run_dir, "verdict.json")
    print("verdict:", open(vpath).read() if os.path.exists(vpath) else "none (no finish call, or the scene has no evaluator)")
    print("run dir:", run_dir)

if __name__ == "__main__":
    main()
