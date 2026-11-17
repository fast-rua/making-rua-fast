"""Replay a recorded simulated run: feed its requests.jsonl, in order, to a fresh in-process driver with the same
scene, seed and scene arguments (lockstep simulation is deterministic, its noise included), compare each reply with
the recording (all of it: results, times, joint values), print the full traceback of any error the driver raises,
and at the end compare the task's verdict with the recorded one.

    python scripts/replay_requests.py RUN_DIR [--until N] [--gpu 2] [--scene-arg noise=0 ...]

Camera requests are skipped (they do not move the simulation); everything else is sent as recorded. --scene-arg
overrides a scene argument: the recorded commands then run open loop in a changed world (e.g. another noise draw).
Runs recorded before scenes had noise replay with noise off.
"""
import argparse, inspect, json, os, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--until", type=int, default=None, help="stop after request N")
    ap.add_argument("--gpu", type=int, default=os.environ.get("ROBOT_GPU"))
    ap.add_argument("--scene-arg", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robotd.backends.egl import use_gpu
    use_gpu(a.gpu)
    import tasks
    from robotd.server import Driver
    from robotd.backends.mujoco_sim import MujocoYamBackend
    from robotd.robots.yam_station.description import description
    from robotd.core.controller import Busy, Fault, InvalidCommand
    from robotd.core.recorder import _small
    from robotd.server import _error
    meta = json.load(open(os.path.join(a.run, "run.json")))
    t = meta.get("task") or {}
    task = tasks.get(t["name"])
    kw = dict(t.get("args") or {})
    if "args" not in t and not (meta.get("scene") or {}).get("noise") and \
            "noise" in inspect.signature(task.scene).parameters:
        kw["noise"] = False                                # recorded before the scene had noise
    kw.update(tasks.scene_args(a.scene_arg))
    scene = task.scene(**kw) if t.get("seed") is None else task.scene(seed=t["seed"], **kw)
    backend = MujocoYamBackend(scene)
    driver = Driver(backend, description(1.0), None, task.evaluate, task=t)
    diverged = 0
    first = None
    for line in open(os.path.join(a.run, "requests.jsonl")):
        o = json.loads(line)
        req = {k: v for k, v in o["req"].items() if k not in ("src", "src_text")}
        if req.get("op") in ("hello", "shutdown", "cameras"):
            continue
        try:
            rep = {"ok": True, "result": driver.handle(req, None)}
        except Exception as e:                             # noqa: BLE001
            rep = {"ok": False, "error": _error(e)}
            if not isinstance(e, (InvalidCommand, Fault, Busy)):     # not a deliberate refusal: show where
                print(f"request {o['n']} raised {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        if json.loads(json.dumps(_small(rep), default=str)) != o["rep"]:
            diverged += 1
            if first is None:
                first = o["n"]
                mine, theirs = json.dumps(_small(rep), default=str)[:300], json.dumps(o["rep"])[:300]
                print(f"request {o['n']} ({req.get('op')}) first differs from the recording: replay {mine} | "
                      f"recorded {theirs}", flush=True)
        if a.until is not None and o["n"] >= a.until:
            break
    print(f"replayed to request {o['n']} at t {driver.c.t:.2f}; {diverged} replies differed from the recording")
    if task.evaluate is not None:
        v = json.loads(json.dumps(task.evaluate(backend), default=str))
        keys = ("success", "best_streak", "bounces", "hits", "touched_other")
        print("verdict (replay):  ", {k: v[k] for k in keys if k in v})
        vpath = os.path.join(a.run, "verdict.json")
        if os.path.exists(vpath):
            rec = json.load(open(vpath))["verdict"]
            print("verdict (recorded):", {k: rec[k] for k in keys if k in rec})
            print("verdict identical:", v == rec)


if __name__ == "__main__":
    main()
