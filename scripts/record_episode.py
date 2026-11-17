"""Record the cell's cameras as a timelapse video of an episode (scripts/agent_run_real.py starts one per episode;
it can also be started by hand on a running one):

    python scripts/record_episode.py --run ~/robot-cell/agent-runs/TAG          # RUN/video-4x.mp4 until it ends
    python scripts/record_episode.py --out clip.mp4 --every 0.25 --speed 2           # anything, until Ctrl-C

Every --every seconds it grabs the cameras' colour images (no depth) through the robot client (camerad), puts them side
by side under a bar with the run, the wall time and the time since the start, and pipes the frame to ffmpeg (the static
build that imageio-ffmpeg ships: H.264, CRF 28, a keyframe every 16 frames), which plays it at --speed x real time
(fps = speed / every). The MP4 is fragmented and written as it goes, so it plays while it records and even if the
recorder is killed. With --run it stops once RUN/meta.json says the episode is
done (after the end-of-episode park and check, so those are in it), or on SIGTERM/SIGINT.

    python scripts/record_episode.py --streams DIR --run RUN [--fps 6]           # the agent's camera recordings

--streams: every camera's colour as its own MP4 in real time (DIR/<camera>.mp4, full resolution, --fps frames a
second, a keyframe each second so a seek is quick), DIR/frames.csv with each frame's wall time (frame k of every
file was read at that time), and DIR/index.json. Fragmented and written as it goes, so they can be read while they
record (robot/recording.py reads them).
--depth-fps (default 3; 0: none): also every camera's depth, with the colour, K and pose T of the same grab, from a
connection of its own: DIR/rgbd/<k>-<camera>_depth.png (16-bit, mm; 0: no reading) and <k>-<camera>.jpg, indexed in
DIR/rgbd.jsonl (wall time; per camera the robot time t, K, T or pose_error, the files). recording.Recording.cameras_at
reads them back as frames like r.cameras(). A grab of four RGB-D frames is about 10 MB over the camera socket, so
this runs at a few frames a second, beside the colour streams.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import robot                                             # noqa: E402

BAR = 22                                                 # px: the text bar over the images


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:                                  # no wheel: the one on PATH
        import shutil
        return shutil.which("ffmpeg") or "ffmpeg"


def done(meta: str | None) -> bool:
    if not meta:
        return False
    try:
        return json.load(open(meta)).get("state") == "done"
    except (OSError, ValueError):
        return False


def streams(a, stop: threading.Event):
    """Each camera's colour as its own real-time MP4, with one frames.csv of wall times for all of them."""
    os.makedirs(a.streams, exist_ok=True)
    meta = os.path.join(os.path.expanduser(a.run), "meta.json") if a.run else None
    r = robot.connect(a.robot, name="record_streams")
    first = r.cameras(None, depth=False)
    names = sorted(first)
    enc = {}
    for n in names:
        h, w = first[n].rgb.shape[:2]
        enc[n] = subprocess.Popen([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{a.fps:g}",
                                   "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                                   "-crf", "23", "-g", str(max(1, int(round(a.fps)))), "-pix_fmt", "yuv420p",
                                   "-flush_packets", "1", "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                                   os.path.join(a.streams, f"{n}.mp4")],
                                  stdin=subprocess.PIPE, start_new_session=True)
    t0 = time.time()
    with open(os.path.join(a.streams, "index.json"), "w") as f:
        json.dump({"start_wall": t0, "fps": a.fps, "cameras": names,
                   "size": {n: list(first[n].rgb.shape[1::-1]) for n in names},
                   "files": {n: f"{n}.mp4" for n in names}, "times": "frames.csv",
                   "note": "frame k of every file was read at frames.csv's wall time k; it plays at k / fps s",
                   **({"rgbd": {"fps": a.depth_fps, "index": "rgbd.jsonl", "dir": "rgbd",
                                "note": "colour and depth (uint16 mm, 0: no reading) of the same grab, with K and T "
                                        "(world <- optical) per camera"}} if a.depth_fps > 0 else {})}, f,
                  indent=1)
    csv = open(os.path.join(a.streams, "frames.csv"), "w", buffering=1)
    csv.write("frame,wall_s\n")
    print(f"record_streams: {a.streams}, {', '.join(names)} at {a.fps:g} fps"
          + (f", RGB-D at {a.depth_fps:g} fps" if a.depth_fps > 0 else ""), flush=True)
    rgbd = None
    if a.depth_fps > 0:
        rgbd = threading.Thread(target=rgbd_stream, args=(a, stop, meta), daemon=True)
        rgbd.start()
    n, frames, last_meta, errors, period = 0, first, 0.0, 0, 1.0 / a.fps
    while not stop.is_set():
        t = time.time()
        try:
            if frames is None:
                frames = r.cameras(None, depth=False)
            for name in names:                           # a camera missing this time repeats its last frame
                img = frames.get(name)
                if img is not None and img.rgb is not None:
                    enc[name].last = np.ascontiguousarray(img.rgb).tobytes()
                enc[name].stdin.write(getattr(enc[name], "last"))
            csv.write(f"{n},{t:.3f}\n")
            n += 1
            errors = 0
        except BrokenPipeError:
            print("record_streams: an encoder exited", flush=True)
            break
        except Exception as e:                           # noqa: BLE001 -- a camera hiccup: try again next frame
            errors += 1
            if errors in (1, 10, 100):
                print(f"record_streams: frame failed ({errors}): {e!r}", flush=True)
            try:
                r = robot.connect(a.robot, name="record_streams")
            except Exception:                            # noqa: BLE001
                pass
        frames = None
        if meta and t - last_meta > 5:
            last_meta = t
            if done(meta):
                break
        stop.wait(max(0.0, period - (time.time() - t)))   # behind: the next read at once, the times say when
    for p in enc.values():
        try:
            p.stdin.close()
        except OSError:
            pass
    for p in enc.values():
        p.wait(60)
    csv.close()
    stop.set()
    if rgbd is not None:
        rgbd.join(120)
    print(f"record_streams: {n} frames per camera over {time.time() - t0:.0f} s", flush=True)


def rgbd_stream(a, stop: threading.Event, meta: str | None):
    """Every camera's colour and depth from one grab, --depth-fps times a second, as files (written by a thread of
    their own, so encoding never holds up the next grab), indexed in rgbd.jsonl."""
    import queue
    from robot.images import depth_mm, write_png
    out = os.path.join(a.streams, "rgbd")
    os.makedirs(out, exist_ok=True)
    jobs: queue.Queue = queue.Queue(maxsize=64)

    def writer():
        while True:
            job = jobs.get()
            if job is None:
                return
            fn, args = job
            try:
                fn(*args)
            except Exception as e:                       # noqa: BLE001 -- one lost file, not the recording
                print(f"record_rgbd: write failed: {e!r}", flush=True)

    def jpeg(path, rgb):
        Image.fromarray(np.ascontiguousarray(rgb)).save(path, quality=95)

    w = threading.Thread(target=writer, daemon=True)
    w.start()
    index = open(os.path.join(a.streams, "rgbd.jsonl"), "a", buffering=1)
    r, k, errors, period, dropped, t0 = None, 0, 0, 1.0 / a.depth_fps, 0, time.time()
    while not stop.is_set():
        t = time.time()
        try:
            if r is None:
                r = robot.connect(a.robot, name="record_rgbd")
            frames = r.cameras(None, depth=True)
            entry = {"k": k, "wall": round(t, 4), "cams": {}}
            for name, f in sorted(frames.items()):
                e = {"t": f.t, "K": np.asarray(f.K).tolist(), "T": None if f.T is None else np.asarray(f.T).tolist()}
                if f.T is None:
                    e["pose_error"] = f.pose_error
                if f.rgb is not None:
                    e["rgb"] = f"rgbd/{k:06d}-{name}.jpg"
                    _put(jobs, (jpeg, (os.path.join(a.streams, e["rgb"]), f.rgb)))
                if f.depth is not None:
                    e["depth"] = f"rgbd/{k:06d}-{name}_depth.png"
                    _put(jobs, (lambda p, d: write_png(p, depth_mm(d)), (os.path.join(a.streams, e["depth"]), f.depth)))
                entry["cams"][name] = e
            index.write(json.dumps(entry) + "\n")
            k += 1
            errors = 0
        except Exception as e:                           # noqa: BLE001 -- a camera hiccup: try again next time
            errors += 1
            if errors in (1, 10, 100):
                print(f"record_rgbd: grab failed ({errors}): {e!r}", flush=True)
            r = None
        if meta and done(meta):
            break
        late = time.time() - t
        if late > period:
            dropped += 1
        stop.wait(max(0.0, period - late))
    jobs.put(None)
    w.join(300)
    index.close()
    el = max(time.time() - t0, 1e-9)
    print(f"record_rgbd: {k} RGB-D grabs over {el:.0f} s ({k / el:.2f} a second; {dropped} took longer than "
          f"{period:.2f} s)", flush=True)


def _put(q, job):
    """Queue a file to write; when the writer falls far behind, wait for it rather than lose the file."""
    q.put(job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="an episode's run directory: writes RUN/video-<speed>x.mp4, stops when it is done")
    ap.add_argument("--out", help="the MP4 to write (instead of --run's)")
    ap.add_argument("--robot", default="tcp:127.0.0.1:7410")
    ap.add_argument("--cameras", nargs="+", default=["top", "front"])
    ap.add_argument("--every", type=float, default=0.5, help="seconds between frames")
    ap.add_argument("--speed", type=float, default=4.0, help="playback speed against real time")
    ap.add_argument("--width", type=int, default=480, help="px per camera (height keeps the aspect)")
    ap.add_argument("--streams", metavar="DIR", help="record every camera as its own real-time MP4 into DIR")
    ap.add_argument("--fps", type=float, default=6.0, help="--streams: frames a second")
    ap.add_argument("--depth-fps", type=float, default=3.0,
                    help="--streams: RGB-D grabs a second (colour, depth, K, T of every camera); 0: none")
    a = ap.parse_args()
    if a.streams:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, lambda *_: stop.set())
        streams(a, stop)
        return
    if not (a.run or a.out):
        ap.error("give --run or --out")
    out = a.out or os.path.join(os.path.expanduser(a.run), f"video-{a.speed:g}x.mp4")
    label = os.path.basename(os.path.normpath(a.run)) if a.run else os.path.basename(out)
    meta = os.path.join(os.path.expanduser(a.run), "meta.json") if a.run else None
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())

    r = robot.connect(a.robot, name="record_episode")
    first = r.cameras(a.cameras, depth=False)
    h0, w0 = next(iter(first.values())).rgb.shape[:2]
    w = a.width - a.width % 2
    h = int(round(w * h0 / w0 / 2)) * 2
    W, H = w * len(a.cameras), h + BAR
    fps = a.speed / a.every
    ff = subprocess.Popen([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                           # a keyframe (so a fragment) every 16 frames and no lookahead: written as it records,
                           # playable meanwhile, and a recorder killed loses seconds, not minutes
                           "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-crf", "28", "-g", "16",
                           "-pix_fmt", "yuv420p", "-flush_packets", "1",
                           "-movflags", "+frag_keyframe+empty_moov+default_base_moof", out],
                          stdin=subprocess.PIPE, start_new_session=True)
    print(f"record_episode: {out}, {len(a.cameras)} cameras every {a.every:g} s at {a.speed:g}x ({fps:g} fps, "
          f"{W}x{H})", flush=True)
    t0, n, last_meta, errors = time.time(), 0, 0.0, 0
    frames = first
    while not stop.is_set():
        t = time.time()
        try:
            if frames is None:
                frames = r.cameras(a.cameras, depth=False)
            canvas = Image.new("RGB", (W, H), (16, 18, 22))
            for i, name in enumerate(a.cameras):
                img = Image.fromarray(np.ascontiguousarray(frames[name].rgb)).resize((w, h), Image.BILINEAR)
                canvas.paste(img, (i * w, BAR))
            el = int(t - t0)
            ImageDraw.Draw(canvas).text(
                (6, 5), f"{label}   {time.strftime('%H:%M:%S', time.localtime(t))}   +{el // 60:d}:{el % 60:02d}   "
                        f"{' | '.join(a.cameras)}   {a.speed:g}x", fill=(220, 225, 232))
            ff.stdin.write(canvas.tobytes())
            n += 1
            errors = 0
        except BrokenPipeError:
            print("record_episode: ffmpeg exited", flush=True)
            break
        except Exception as e:                           # noqa: BLE001 -- a camera hiccup: try again next frame
            errors += 1
            if errors in (1, 10, 100):
                print(f"record_episode: frame failed ({errors}): {e!r}", flush=True)
            try:
                r = robot.connect(a.robot, name="record_episode")
            except Exception:                            # noqa: BLE001
                pass
        frames = None
        if meta and t - last_meta > 5:                   # the episode is over once meta.json says done
            last_meta = t
            try:
                if json.load(open(meta)).get("state") == "done":
                    break
            except (OSError, ValueError):
                pass
        stop.wait(max(0.0, a.every - (time.time() - t)))
    try:
        ff.stdin.close()
    except OSError:
        pass
    ff.wait(60)
    print(f"record_episode: {n} frames, {n / fps:.0f} s of video for {time.time() - t0:.0f} s, "
          f"{os.path.getsize(out) / 1e6:.1f} MB: {out}", flush=True)


if __name__ == "__main__":
    main()
