"""robot.watch: compact 3D views of the grasp while a motion runs, and an optional check that stops it.

    from robot import watch, when
    with watch.watch(r, arms=["left"], view_every=1.0) as w:
        r.move(T_approach, ee="left", duration=4.0, until=when.signal("watch") > 0)

While the code inside the block runs, the watcher, in threads of its own (they never hold up your program):

- every `view_every` seconds renders, for each watched arm, ONE compact image, views/monitor-K.png: the wrist camera
  and the front camera with the arm's ghost where it is now and its jaw box, and the point cloud's `jaws` (through
  the gap) and `along` (fingertips against the surface) views around its TCP. It prints one line:
  `[watch view +1.2s] /path/monitor-3.png | left: between the jaws: 160 points across 80% of the 40 mm gap, 4-18 mm
  above the tips (at the start of the block: 2, so +158)`. The arms' own points are left out (robot.viz SELF_MARGIN:
  on the real cell an empty gripper reads 0-4 points); the caption still compares with the block's first view while
  the gap stays within 5 mm of it. A render takes about 0.5-1 s per arm on the cell; a new one starts when the last
  is done and view_every has passed, and the block never waits for one.
  Names start with "monitor": an agent harness drops those from its history before other images.
- every `every` seconds reads the arms' state and runs `check`, if given. check(state, frames) or
  check(state, frames, between) (between: {arm: the latest between-the-jaws numbers, or None}) returns None or a
  reason; with a reason the watcher prints `[watch STOP] reason` and sets the signal (default "watch") to 1, which ends
  any motion started with until=when.signal("watch") > 0 within a tick (Result status "stopped", reason "until").
  frames is {} unless cameras= names some (a fresh grab of those, with depth when depth=True).
- prints a state line (`[watch +1.0s] left xyz=[...] grip=64.0mm ...`) only when something changes (a gripper's
  width by 2 mm or more, a close on nothing, contact where the arms' force estimate is trusted: simulation), or every
  tick with lines="all", or never with lines="none".

The signal is set to 0 as the block starts and again as it ends. The watcher's connections are its own; your program's
connection stays the one robot.viz and the [state] line use. w.views lists the images it wrote, w.between the latest
numbers per arm, w.reason why the check stopped the motion (None if it did not).
"""
from __future__ import annotations

import inspect
import threading
import time

import numpy as np


class Watcher:
    def __init__(self, r, every=0.25, check=None, signal="watch", cameras=None, depth=False, view_every=None,
                 arms=None, view_width=640, name="monitor", lines="changes"):
        self.r, self.every, self.check, self.signal = r, float(every), check, signal
        self.cameras, self.depth, self.view_every, self.view_width = cameras, depth, view_every, int(view_width)
        self.arms = list(arms) if arms else None
        self.name = name if str(name).startswith("monitor") else "monitor-" + str(name)
        self.lines_mode = lines
        self.reason = None                       # why the check stopped the motion, if it did
        self.lines, self.views, self.between = [], [], {}
        self.baseline = {}                       # arm -> (count, gap) at the first render: the fingers themselves
        self._check_args = 2
        if check is not None:
            try:
                self._check_args = len(inspect.signature(check).parameters)
            except (TypeError, ValueError):
                pass
        self._stop = threading.Event()
        self._threads = []
        self._conns = []
        self._t0 = None

    # ---- the with block
    def __enter__(self):
        import robot
        latest = list(robot._LATEST)
        tick = robot.connect(self.r.addr, name="watch")
        self._conns.append(tick)
        tick.signal(self.signal, 0.0)
        render = None
        if self.view_every:                      # the render thread's own connection: it closes it itself
            render = robot.connect(self.r.addr, name="watch-views")
        robot._LATEST[:] = latest                # the program's own connection stays the one viz and [state] use
        if self.arms is None:
            self.arms = [a for a in ("left", "right") if a in self.r.end_effectors] or list(self.r.end_effectors)
        self._t0 = time.time()
        self._threads.append(threading.Thread(target=self._tick, args=(tick,), daemon=True))
        if render is not None:
            self._threads.append(threading.Thread(target=self._render_loop, args=(render,), daemon=True))
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._threads[0].join(2.0)               # the tick thread (quick); a render under way finishes on its own
        try:
            self._conns[0].signal(self.signal, 0.0)
        except Exception:                        # noqa: BLE001
            pass
        for c in self._conns:
            try:
                c.close()
            except Exception:                    # noqa: BLE001
                pass
        return False

    def _say(self, s):
        self.lines.append(s)
        print(s, flush=True)

    def _t(self):
        return time.time() - self._t0

    # ---- the state, the check, the lines
    def _tick(self, conn):
        last = None
        while not self._stop.is_set():
            try:
                st = conn.state()
                frames = conn.cameras(self.cameras, depth=self.depth) if self.cameras else {}
                key = tuple((n, round(1000 * float((st.grippers.get(n) or {}).get("width") or 0) / 2),
                             bool((st.grippers.get(n) or {}).get("empty")),
                             bool(e.contact and st.arm_force_trusted)) for n, e in st.ee.items())
                if self.lines_mode == "all" or (self.lines_mode == "changes" and key != last):
                    speeds = " ".join(f"{n}:{1000 * float(np.linalg.norm(e.v[:3])):.0f}mm/s" for n, e in st.ee.items())
                    self._say(f"[watch +{self._t():.1f}s] {st.summary()}; speed {speeds}")
                last = key
                if self.check is not None and self.reason is None:
                    args = (st, frames, dict(self.between)) if self._check_args >= 3 else (st, frames)
                    why = self.check(*args)
                    if why:
                        self.reason = str(why)
                        conn.signal(self.signal, 1.0)
                        self._say(f"[watch STOP +{self._t():.1f}s] {self.reason} (signal '{self.signal}' set)")
            except Exception as e:                # noqa: BLE001 -- a watcher never takes the program down
                self._say(f"[watch error] {type(e).__name__}: {e}")
            self._stop.wait(self.every)

    # ---- the compact 3D views
    def _render_loop(self, conn):
        try:
            self._render_views(conn)
        finally:
            try:
                conn.close()
            except Exception:                    # noqa: BLE001
                pass

    def _render_views(self, conn):
        k = 0
        while not self._stop.is_set():
            t_start = time.time()
            try:
                k += 1
                path, caption = self.render(conn, f"{self.name}-{k}")
                self.views.append(path)
                self._say(f"[watch view +{self._t():.1f}s] {path} | {caption}")
            except Exception as e:                # noqa: BLE001
                self._say(f"[watch view error] {type(e).__name__}: {e}")
            self._stop.wait(max(0.0, self.view_every - (time.time() - t_start)))

    def render(self, conn, name):
        """One compact image for the watched arms (module docstring); returns (path, caption)."""
        import cv2

        from robot import viz
        need = {"front"} | {f"{a}_wrist" for a in self.arms}
        st = conn.state()
        fs = conn.cameras(sorted(need), depth=True)
        blocks, captions = [], []
        for arm in self.arms:
            T = st.ee[arm].T
            v = viz.View3D(conn, center=T, frames=fs, state=st, radius=0.10, size=(320, 240))
            info = v.ghost(arm)
            b = info.get("between")
            if b is not None:
                base = self.baseline.setdefault(arm, (b["count"], b["gap_mm"]))
                b = dict(b, baseline=base[0] if abs(base[1] - b["gap_mm"]) < 5 else None)
            self.between[arm] = b
            cap = viz._between_line(b).lstrip(" |") if b is not None else "no depth between the jaws"
            if b is not None and b.get("baseline") is not None:
                cap += f" (at the start of the block: {b['baseline']}, so {b['count'] - b['baseline']:+d})"
            captions.append(f"{arm}: {cap}")
            tiles = []
            for cam in (f"{arm}_wrist", "front"):
                if cam in fs:
                    o = viz.Overlay(conn, cam, frame=fs[cam], state=st)
                    o.ghost(arm)
                    tiles.append(o.render())
            tiles.append(v.render(["jaws", "along"], legend=False))
            blocks.append(_stack(tiles, self.view_width))
        img = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)     # one arm per band, each full width
        if img.shape[0] > 2 * self.view_width:                     # keep it compact: at most 2:1 tall
            s = 2 * self.view_width / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), 2 * self.view_width), interpolation=cv2.INTER_AREA)
        path = viz.show(name, img, r=conn, quiet=True)
        return path, "; ".join(captions)


def _stack(tiles, width):
    """Tiles side by side in rows of two (each scaled to half the width), rows stacked: one image `width` wide."""
    import cv2
    rows, i = [], 0
    while i < len(tiles):
        row = tiles[i:i + 2]
        i += 2
        if len(row) == 1 and row[0].shape[1] >= row[0].shape[0] * 1.5:    # a wide tile (the 3D pair) gets a row
            t = row[0]
            rows.append(cv2.resize(t, (width, max(1, int(t.shape[0] * width / t.shape[1]))),
                                   interpolation=cv2.INTER_AREA))
            continue
        cw = width // 2
        h = max(int(t.shape[0] * cw / t.shape[1]) for t in row)
        r = np.zeros((h, width, 3), np.uint8)
        for j, t in enumerate(row):
            s = cv2.resize(t, (cw, max(1, int(t.shape[0] * cw / t.shape[1]))), interpolation=cv2.INTER_AREA)
            r[:s.shape[0], j * cw:j * cw + cw] = s
        rows.append(r)
    return np.concatenate(rows, axis=0)


def watch(r, every=0.25, check=None, signal="watch", cameras=None, depth=False, view_every=None, arms=None,
          view_width=640, name="monitor", lines="changes") -> Watcher:
    """A Watcher for a with block (module docstring)."""
    return Watcher(r, every, check, signal, cameras, depth, view_every, arms, view_width, name, lines)
