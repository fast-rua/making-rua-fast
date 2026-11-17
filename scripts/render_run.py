"""Render a recorded simulated run as a video for the run viewer: the scene rebuilt from the run's task, posed from
the recorded joint positions, free objects' poses and cloth states, seen from two free cameras side by side.
Developer tool (it reads the simulator's full state); its output never goes to an agent.

    python scripts/render_run.py RUN_DIR [--scene fold_towel --seed 0] [--fps 20] [--gpu 2] [--out DIR]

Writes RUN_DIR/replay/video.mp4 and RUN_DIR/replay/video.json (fps, duration, views), or DIR/ with --out (an agent
episode's recording stays untouched: its video goes next to it). Video time is simulation time, the clock of the
t0/t1 fields in requests.jsonl.
"""
import argparse, json, os, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
def _ffmpeg() -> str:
    """$FFMPEG, else imageio-ffmpeg's static build (scripts/install.sh installs it), else the ffmpeg on PATH."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


FFMPEG = _ffmpeg()
VIEWS = [  # (name, lookat, distance, azimuth, elevation)
    ("from above, behind the robot", (0.38, 0.0, 0.0), 1.15, 0.0, -66.0),
    ("from the front left", (0.36, 0.0, 0.08), 1.2, 210.0, -24.0),
]
W, H = 640, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--scene", default=None, help="task name, if run.json does not say (older runs)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--gpu", type=int, default=os.environ.get("ROBOT_GPU"))
    ap.add_argument("--out", default=None, help="output folder (default RUN_DIR/replay)")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "run.json")))
    task_meta = meta.get("task") or {}
    name = a.scene or task_meta.get("name")
    seed = a.seed if a.seed is not None else task_meta.get("seed")
    assert name, "run.json names no task: pass --scene"

    os.environ.setdefault("MUJOCO_GL", "egl")
    from robotd.backends.egl import use_gpu
    use_gpu(a.gpu)
    import mujoco
    import tasks
    from robotd.backends.mujoco_sim import MujocoYamBackend
    from robotd.core.recorder import load_ticks
    task = tasks.get(name)
    kw = task_meta.get("args") or {}
    backend = MujocoYamBackend(task.scene(**kw) if seed is None else task.scene(seed=seed, **kw))
    m, d = backend.model, backend.data
    ticks = load_ticks(a.run)
    t = ticks["t"]
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in meta["joints"]]
    # joint equalities (each finger's twin is coupled to joint7): q1 = q1_0 + c0 + c1 (q2 - q2_0). The station model
    # writes joint7 as q1 (it follows joint8), so the twin is solved from the recorded joint, not the other way round
    recorded = set(qadr)
    couple = [(m.jnt_qposadr[m.eq_obj1id[e]], m.jnt_qposadr[m.eq_obj2id[e]], m.eq_data[e][:2].copy())
              for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT]
    cloths = [k.split("/", 1)[1] for k in ticks if k.startswith("cloth/")]
    # free objects (a cube, a ball, a held paddle): posed from their recorded x y z qw qx qy qz
    free = [(m.jnt_qposadr[m.joint(f"{k.split('/', 1)[1]}_free").id], ticks[k]) for k in ticks
            if k.startswith("obj/") and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{k.split('/', 1)[1]}_free") >= 0]

    rend = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    cams = []
    for _, lookat, dist, az, el in VIEWS:
        c = mujoco.MjvCamera()
        c.lookat[:], c.distance, c.azimuth, c.elevation = lookat, dist, az, el
        cams.append(c)

    out_dir = a.out or os.path.join(a.run, "replay")
    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(out_dir, "video.mp4")
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W * len(VIEWS)}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            video + ".part.mp4"], stdin=subprocess.PIPE)
    frames = np.arange(0.0, float(t[-1]) + 1e-9, 1.0 / a.fps)
    last_cloth = {}
    t_start = time.time()
    for k, tf in enumerate(frames):
        i = min(int(np.searchsorted(t, tf)), len(t) - 1)
        d.qpos[qadr] = ticks["q"][i]
        for j1, j2, c in couple:          # the recorded joint drives its twin, whichever side of the equality it is
            if j1 in recorded and j2 not in recorded:
                d.qpos[j2] = m.qpos0[j2] + (d.qpos[j1] - m.qpos0[j1] - c[0]) / c[1]
            else:
                d.qpos[j1] = m.qpos0[j1] + c[0] + c[1] * (d.qpos[j2] - m.qpos0[j2])
        for adr, pose in free:
            d.qpos[adr:adr + 7] = pose[i]
        for cname in cloths:
            ct = ticks[f"cloth_t/{cname}"]
            ci = max(0, min(int(np.searchsorted(ct, tf, side="right")) - 1, len(ct) - 1))
            if last_cloth.get(cname) != ci:
                backend.set_cloth_vertices(cname, ticks[f"cloth/{cname}"][ci])   # also runs mj_forward
                last_cloth[cname] = ci
        mujoco.mj_forward(m, d)
        img = []
        for c in cams:
            rend.update_scene(d, camera=c, scene_option=opt)
            img.append(rend.render())
        enc.stdin.write(np.concatenate(img, axis=1).tobytes())
        if k % 500 == 0:
            print(f"frame {k}/{len(frames)} ({time.time() - t_start:.0f}s)", flush=True)
    enc.stdin.close()
    assert enc.wait() == 0, "ffmpeg failed"
    os.replace(video + ".part.mp4", video)
    json.dump({"fps": a.fps, "duration_s": float(frames[-1]) if len(frames) else 0.0, "width": W * len(VIEWS),
               "height": H, "views": [v[0] for v in VIEWS], "task": name, "seed": seed},
              open(os.path.join(out_dir, "video.json"), "w"))
    print(f"wrote {video}: {len(frames)} frames, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
