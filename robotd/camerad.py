"""camerad: the cell's RealSense cameras in a process of their own.

robotd's 250 Hz control loop shares a Python interpreter with everything else robotd does. Grabbing, aligning and
sending four RGB-D frames (about 11 MB of JSON per grab) held that interpreter for 15-28 ms at a time, and the loop
stalled with it: on the YAM cell on 2026-11-04 the tick period's p99 was 4.0 ms without camera traffic and 15-18 ms
with the dashboard's (motors on, 33 ms). camerad does the camera work instead. robotd started with --cameras ADDR never
opens a camera and names camerad's address in its description; robot.Robot.cameras() fetches from there. The frames
are as before: rgb, depth (m), K, T (world <- optical) and t (the robot clock, read from robotd at the grab).

    python -m robotd.camerad --addr tcp:127.0.0.1:7412 --robot tcp:127.0.0.1:7410 [--record DIR]

Environment: YAM_CAMERAS "name=serial,..." (required); YAM_CALIBRATION (the cell calibration: fixed cameras' poses,
base and wrist-mount corrections); I2RT_ROOT (the arm model, for the wrist cameras' poses).
--record DIR: the frames programs take (not the dashboard's, nor the episode recorders', which save their own) as
DIR/frames/<n>-<camera>.png and <n>-<camera>_depth.png (16-bit, mm), indexed in DIR/frames.jsonl (client, wall time,
robot time, K and T per camera). At most --record-max-mb (256) wait in memory to be written; a grab past that is not
recorded, and camerad prints how many were dropped (each index line's dropped_before counts them too).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import threading
import time
import traceback

import numpy as np

from robot import rpc
from robot.images import depth_mm, write_png
from robotd.views import ViewBoard

VIEWS = ViewBoard()          # images programs show (robot.viz.show), for the dashboard


class Cameras:
    """The rig, the calibration and the arm state the wrist cameras' poses need."""

    def __init__(self, serials: dict, calibration_path: str | None, robot_addr: str | None, parallel: bool = False):
        from robotd.backends import yam_i2rt
        from robotd.backends.realsense import RealSenseRig
        self.y = yam_i2rt
        self.serials = dict(serials)
        self.calibration = yam_i2rt.load_calibration(calibration_path)
        self.robot_addr = robot_addr
        self._model = None
        self._robot = None
        self.rig = RealSenseRig(self.serials)
        self.parallel = parallel
        self.grab_lock = threading.Lock()      # one grab at a time: a pipeline is not read from two threads
        self.robot_lock = threading.Lock()

    def model(self):
        if self._model is None:
            self._model = self.y.kinematics_model_for(self.calibration)
        return self._model

    def robot_state(self):
        """(robot clock t, {joint: model angle}) from robotd, or (None, None, why) when it cannot be read."""
        import robot
        with self.robot_lock:
            try:
                if self._robot is None:
                    if not self.robot_addr:
                        return None, None, "camerad was started without --robot"
                    self._robot = robot.connect(self.robot_addr, name="camerad")
                s = self._robot.state()
                return s.t, dict(zip(s.joints, map(float, s.q()))), None
            except Exception as e:                       # noqa: BLE001 -- robotd restarting: try again next grab
                with contextlib.suppress(Exception):
                    self._robot.close()
                self._robot = None
                return None, None, f"no arm state from robotd at {self.robot_addr}: {e}"

    def grab(self, names=None, depth: bool = True) -> dict:
        names = list(names or self.serials)
        unknown = [n for n in names if n not in self.serials]
        if unknown:
            raise ValueError(f"unknown camera(s) {unknown}; cameras: {sorted(self.serials)}")
        with self.grab_lock:
            frames = self.rig.grab(names, depth, parallel=self.parallel)
        t, q, why = self.robot_state()
        for name, f in frames.items():
            try:
                if q is None and name.endswith("_wrist"):
                    raise NotImplementedError(why)
                f["T"] = self.y.camera_pose(name, self.calibration, self.model, q or {})
            except NotImplementedError as e:             # uncalibrated: the image, but no pose for its points
                f["T"], f["pose_error"] = None, str(e)
            f["t"] = t
        return frames

    def close(self):
        self.rig.close()
        if self._robot is not None:
            with contextlib.suppress(Exception):
                self._robot.close()


SKIP_CLIENTS = ("dashboard", "record_")   # the dashboard's live view, and the episode recorders
                                          # (scripts/record_episode.py: record_streams, record_rgbd, record_episode),
                                          # which save their own frames


def _held(a) -> int:
    """The bytes an array keeps in memory: a crop (a view) keeps the whole frame it was cut from."""
    a = np.asarray(a)
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a.nbytes


class FrameRecorder:
    """The frames programs take, written by a thread of its own (a PNG takes ~20 ms).

    The queue is bounded: a grab that would take it past max_bytes is dropped, counted and reported, never held.
    On 2026-11-07 an unbounded queue grew to 8.8 GB in 12 minutes while the episode recorders' streams (about 50 PNGs
    a second) were saved here a second time; the cell PC ran out of memory and swapped, robotd's CAN threads starved,
    the motors' communication watchdog tripped on both arms and they went limp. Clients whose names start with one of
    `skip` are not recorded."""

    def __init__(self, run_dir: str, max_bytes: int = 256 << 20, skip=SKIP_CLIENTS):
        self.dir = run_dir
        os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
        self.index = open(os.path.join(run_dir, "frames.jsonl"), "a")
        self.n = 0
        self.max_bytes = int(max_bytes)
        self.skip = tuple(skip)
        self.queued = 0                          # bytes the queue keeps in memory now
        self.dropped = 0                         # grabs not recorded because the writer was behind
        self.lock = threading.Lock()
        self.q = queue.Queue()
        threading.Thread(target=self._write, daemon=True).start()

    def wants(self, client: str | None) -> bool:
        return not (client or "").startswith(self.skip)

    def _write(self):
        while True:
            fn, args, nbytes = self.q.get()
            try:
                fn(*args)
            except Exception:                            # noqa: BLE001 -- a failed file must not stop the rest
                traceback.print_exc()
            finally:
                with self.lock:
                    self.queued -= nbytes

    def save(self, frames: dict, client: str | None) -> bool:
        """Queue one grab's files (all of them or none); False if it was dropped."""
        sizes = {name: (_held(f["rgb"]), 0 if f.get("depth") is None else _held(f["depth"]))
                 for name, f in frames.items()}
        total = sum(a + b for a, b in sizes.values())
        with self.lock:
            if self.queued + total > self.max_bytes:
                self.dropped += 1
                if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                    print(f"camerad: the recorder is behind ({self.queued >> 20} MB waiting to be written): "
                          f"{self.dropped} grab(s) not recorded so far", flush=True)
                return False
            self.queued += total
            self.n += 1
            entry = {"n": self.n, "wall": time.time(), "client": client, "frames": {},
                     "dropped_before": self.dropped}
            for name, f in frames.items():
                e = {"t": f.get("t"), "K": np.asarray(f["K"]).tolist(),
                     "T": None if f.get("T") is None else np.asarray(f["T"]).tolist(), "roi": f.get("roi")}
                fn = f"{self.n:06d}-{name}.png"
                self.q.put((write_png, (os.path.join(self.dir, "frames", fn), f["rgb"]), sizes[name][0]))
                e["png"] = f"frames/{fn}"
                if f.get("depth") is not None:
                    fn = f"{self.n:06d}-{name}_depth.png"
                    self.q.put((lambda p, d: write_png(p, depth_mm(d)), (os.path.join(self.dir, "frames", fn),
                                                                          f["depth"]), sizes[name][1]))
                    e["depth_png"] = f"frames/{fn}"
                entry["frames"][name] = e
            self.index.write(json.dumps(entry) + "\n")
            self.index.flush()
        return True


def _error(e: Exception) -> dict:
    from robotd.core.controller import InvalidCommand
    kind = "InvalidCommand" if isinstance(e, (InvalidCommand, ValueError)) else "RuntimeError"
    return {"type": kind, "message": f"{type(e).__name__}: {e}"}


def _serve_client(cams: Cameras, rec: FrameRecorder | None, conn):
    from robotd.server import _crop
    name = None
    try:
        while True:
            try:
                req = rpc.recv(conn)
            except (ConnectionError, OSError):
                req = None
            if req is None:
                break
            op = req.get("op")
            try:
                if op == "hello":
                    name = req.get("client") or name
                    result = {"server": "camerad", "cameras": sorted(cams.serials)}
                elif op == "cameras":
                    frames = _crop(cams.grab(req.get("names"), req.get("depth", True)), req.get("roi"))
                    if rec is not None and rec.wants(name):
                        rec.save(frames, name)
                    result = frames
                elif op == "show":                       # robot.viz.show: an image for the dashboard
                    result = VIEWS.show(req, name)
                elif op == "views":
                    result = VIEWS.since(int(req.get("since", 0)))
                else:
                    raise ValueError(f"camerad answers hello, cameras, show and views, not {op!r}")
                rep = {"ok": True, "result": result}
            except Exception as e:                       # noqa: BLE001 -- every error goes back to the caller
                rep = {"ok": False, "error": _error(e)}
            try:
                rpc.send(conn, rep)
            except (ConnectionError, OSError):
                break
    finally:
        with contextlib.suppress(OSError):
            conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--addr", default="tcp:127.0.0.1:7412", help="where to serve frames (unix:/path or tcp:host:port)")
    ap.add_argument("--robot", default=None, help="robotd's address, for the wrist cameras' poses and the robot clock")
    ap.add_argument("--record", default=None, help="run directory for the frames programs take")
    ap.add_argument("--record-max-mb", type=float, default=256,
                    help="frames waiting to be written, at most (MB); grabs past it are not recorded (default 256)")
    ap.add_argument("--parallel-grab", action="store_true",
                    help="read the cameras at the same time (a thread each) instead of one after another")
    args = ap.parse_args()
    cams_env = os.environ.get("YAM_CAMERAS")
    if not cams_env:
        ap.error("set YAM_CAMERAS=name=serial,...")
    serials = dict(kv.split("=") for kv in cams_env.split(","))
    from robotd.server import _listen
    cams = Cameras(serials, os.environ.get("YAM_CALIBRATION"), args.robot, parallel=args.parallel_grab)
    rec = FrameRecorder(args.record, max_bytes=int(args.record_max_mb * (1 << 20))) if args.record else None
    srv = _listen(args.addr)
    print(f"camerad: {', '.join(f'{n}={s}' for n, s in serials.items())} on {args.addr}"
          + (f", arm state from {args.robot}" if args.robot else "") + (f", recording to {args.record}" if rec else ""),
          flush=True)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except TimeoutError:                         # _listen's sockets time out every 0.2 s
                continue
            conn.settimeout(None)
            threading.Thread(target=_serve_client, args=(cams, rec, conn), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        cams.close()


if __name__ == "__main__":
    main()
