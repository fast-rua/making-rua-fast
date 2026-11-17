"""Minimal RealSense capture for real cells (pyrealsense2): depth aligned to colour, intrinsics from the device.
Untested in this repo's CI (no hardware); the cell's own capture code is the reference.

Each camera has its own rs.align. One align shared by several cameras keeps state from the frames it has processed and
maps the next camera's depth wrongly: on the YAM cell (two D435i, two D405) the front D435i's depth landed ~109 px to
the right of its colour from the second grab on, which put the ball's bump beside the ball and shifted the arms."""
from __future__ import annotations

import numpy as np


class RealSenseRig:
    def __init__(self, serials: dict, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self.rs, self.pipes, self.aligns = rs, {}, {}
        for name, serial in serials.items():
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.pipes[name] = (pipe, scale)
            self.aligns[name] = rs.align(rs.stream.color)        # one per camera: never shared (see above)

    def _grab_one(self, name, depth):
        pipe, scale = self.pipes[name]
        frames = pipe.wait_for_frames()
        if depth:                                 # aligning depth to colour is the slow part: only when asked for
            frames = self.aligns[name].process(frames)
        color = frames.get_color_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        f = {"rgb": np.asanyarray(color.get_data()).copy(),
             "K": np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])}
        if depth:
            f["depth"] = (np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale)
        return f

    def grab(self, names, depth=True, parallel=False) -> dict:
        """parallel=True reads the cameras at the same time, one thread each (each has its own pipeline and align),
        instead of one after another, each waiting for its next frame."""
        names = list(names)
        if parallel and len(names) > 1:
            from concurrent.futures import ThreadPoolExecutor
            if getattr(self, "_pool", None) is None:
                self._pool = ThreadPoolExecutor(max_workers=len(self.pipes))
            futs = {n: self._pool.submit(self._grab_one, n, depth) for n in names}
            return {n: futs[n].result() for n in names}
        return {name: self._grab_one(name, depth) for name in names}

    def close(self):
        for pipe, _ in self.pipes.values():
            try:
                pipe.stop()
            except Exception:
                pass
