"""The episode's camera recordings, for looking back (on the real cell; scripts/record_episode.py --streams writes
them): every camera's colour as its own MP4 from the start of the episode, a few frames a second, and each frame's
wall time; and, a few times a second, every camera's colour and depth from one grab with its K and pose (rgbd/).
They are read while they record.

    from robot import recording
    rec = recording.Recording()                       # $ROBOT_RECORDING_DIR (default /var/record/cameras)
    rec.cameras, rec.fps, rec.times[-1]               # what is there, and the latest frame's wall time
    img, t = rec.frame_at("top", time.time() - 30)    # the top camera 30 s ago: RGB (H x W x 3), when it was read
    for t, img in rec.frames("left_wrist", t0, t1):   # every frame in a span (every=1.0: one a second)
        ...
    rec.clip("front", t0, t1, "fling.mp4")            # a span as its own MP4, to look at
    fs, t = rec.cameras_at(time.time() - 30)          # the RGB-D grab then: {camera: Frame} as r.cameras() gives,
    viz.Overlay(frame=fs["front"], camera="front", state=..., description=r.description)   # so robot.viz works on it

Times are wall-clock seconds (time.time()), the clock programs and r.state() use. ffmpeg (on PATH, or the one
imageio-ffmpeg ships) does the decoding: `ffmpeg -ss SECONDS -i top.mp4 ...` works too, with frame k at k / fps s.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess

import numpy as np

DIR = os.environ.get("ROBOT_RECORDING_DIR", "/var/record/cameras")


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Recording:
    def __init__(self, path: str = DIR):
        self.path = path
        with open(os.path.join(path, "index.json")) as f:
            self.index = json.load(f)
        self.cameras = list(self.index["cameras"])
        self.fps = float(self.index["fps"])
        self.size = {n: tuple(s) for n, s in self.index["size"].items()}

    @property
    def times(self) -> np.ndarray:
        """Each recorded frame's wall time (frame k of every camera's file), as of now."""
        with open(os.path.join(self.path, self.index["times"])) as f:
            rows = list(csv.reader(f))[1:]
        return np.array([float(r[1]) for r in rows if len(r) == 2])

    def frame_index(self, t: float) -> int:
        """The last frame read at or before wall time t (the first frame, if t is earlier)."""
        ts = self.times
        if not len(ts):
            raise LookupError("nothing recorded yet")
        return int(max(0, np.searchsorted(ts, t, side="right") - 1))

    def _decode(self, camera: str, start: float, n: int, every: int = 1) -> np.ndarray:
        w, h = self.size[camera]
        vf = ["-vf", f"select=not(mod(n\\,{every}))", "-fps_mode", "passthrough"] if every > 1 else []
        out = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.4f}",
                              "-i", os.path.join(self.path, self.index["files"][camera]), *vf, "-frames:v", str(n),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        k = len(out) // (w * h * 3)
        return np.frombuffer(out[:k * w * h * 3], np.uint8).reshape(k, h, w, 3)

    def frame_at(self, camera: str, t: float) -> tuple:
        """(RGB image, its wall time): the camera's last frame read at or before wall time t."""
        k = self.frame_index(t)
        img = self._decode(camera, k / self.fps, 1)
        if not len(img):
            raise LookupError(f"frame {k} of {camera} is not readable yet")
        return img[0], float(self.times[k])

    def frames(self, camera: str, t0: float, t1: float, every: float | None = None) -> list:
        """[(wall time, RGB image)] from wall time t0 to t1; every: seconds between them (default: all)."""
        ts = self.times
        k0, k1 = self.frame_index(t0), self.frame_index(t1)
        step = max(1, int(round((every or 0) * self.fps)))
        imgs = self._decode(camera, k0 / self.fps, (k1 - k0) // step + 1, step)
        return [(float(ts[k0 + i * step]), img) for i, img in enumerate(imgs) if k0 + i * step < len(ts)]

    def clip(self, camera: str, t0: float, t1: float, out: str) -> str:
        """The span from wall time t0 to t1 as its own MP4 at `out`."""
        k0, k1 = self.frame_index(t0), self.frame_index(t1)
        subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{k0 / self.fps:.4f}",
                        "-i", os.path.join(self.path, self.index["files"][camera]),
                        "-t", f"{(k1 - k0 + 1) / self.fps:.4f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", out], check=True)
        return out

    # ---- the RGB-D grabs (rgbd/, indexed in rgbd.jsonl)
    def _rgbd(self) -> list:
        p = os.path.join(self.path, (self.index.get("rgbd") or {}).get("index", "rgbd.jsonl"))
        if not os.path.exists(p):
            return []
        out = []
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:                       # the line being written
                    break
        return out

    @property
    def rgbd_times(self) -> np.ndarray:
        """Each RGB-D grab's wall time, as of now (empty when depth is not recorded)."""
        return np.array([e["wall"] for e in self._rgbd()])

    def cameras_at(self, t: float, names=None) -> tuple:
        """({camera: Frame}, its wall time): the last RGB-D grab at or before wall time t, as r.cameras() returns
        frames (rgb, depth in m with 0 for no reading, K, T world <- optical, t the robot time), so Frame.points and
        robot.viz work on it. A file not written yet comes back as None."""
        from PIL import Image

        from robot import Frame
        rows = self._rgbd()
        if not rows:
            raise LookupError("no RGB-D recorded here (index.json has no rgbd)")
        ts = np.array([e["wall"] for e in rows])
        e = rows[int(max(0, np.searchsorted(ts, t, side="right") - 1))]
        out = {}
        for name, c in e["cams"].items():
            if names is not None and name not in names:
                continue
            rgb = depth = None
            if c.get("rgb") and os.path.exists(os.path.join(self.path, c["rgb"])):
                rgb = np.asarray(Image.open(os.path.join(self.path, c["rgb"])).convert("RGB"))
            if c.get("depth") and os.path.exists(os.path.join(self.path, c["depth"])):
                depth = _read_depth(os.path.join(self.path, c["depth"]))
            out[name] = Frame({"rgb": rgb, "depth": depth, "K": c["K"], "T": c.get("T"), "t": c.get("t"),
                               "pose_error": c.get("pose_error")})
        return out, float(e["wall"])


def _read_depth(path: str) -> np.ndarray:
    """A 16-bit depth PNG (mm, 0: no reading) as float32 metres."""
    from PIL import Image
    with Image.open(path) as im:
        mm = np.asarray(im).astype(np.uint16)
    return mm.astype(np.float32) / 1000.0
