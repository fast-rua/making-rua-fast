"""Recording, for replay and analysis (the dashboard reads it; the agent never sees it):

    run.json         backend, robot, joints, start time and, in simulation, the scene (objects, table, model files)
    requests.jsonl   every request and its reply: client, robot clock at start and end (t0, t1), the policy call
                     site (src: [[file, line, function, code sha], ...] innermost first) and the result summary
    code/<sha>.txt   the source of every policy file that issued a request, by content hash
    frames/          the camera images the agent took (PNG), named <request number>-<camera>.png, and their depth as
                     <request number>-<camera>_depth.png (16-bit, mm, 0 where there is no reading)
    ticks/NNNNN.npz  every fast tick, one file per minute so a long session does not grow in memory: t, q, dq,
                     q_ref, q_cmd (bias included), bias (the settle bias), tau, wrench_<arm>, and in simulation obj/<name> (x y z qw qx qy qz) and, every
                     10th tick, cloth/<name> (N x 3 vertices) with cloth_t/<name>

load_ticks(run_dir) joins the tick files."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback

import numpy as np

from robot.images import depth_mm, write_png

CHUNK_TICKS = 15000              # 60 s at 4 ms
CLOTH_EVERY = 10                 # ticks between recorded cloth vertex sets


class _Columns:
    """One tick file's worth of ticks in preallocated arrays: each tick writes a row per column. Per-tick lists of
    arrays had to be converted when a file was saved: in the writer thread, one long call per column that held the
    interpreter lock and stalled the control loop 22-31 ms once a minute on the cell (2026-11-04). A column first
    seen after row 0 keeps its rows from there, as the lists did. Sparse columns (cloth, every CLOTH_EVERY ticks) stay
    lists."""

    def __init__(self, n: int):
        self.n, self.rows = n, 0
        self.arr, self.first, self.sparse = {}, {}, {}

    def put(self, name: str, value) -> None:
        a = self.arr.get(name)
        if a is None:
            v = np.asarray(value)
            a = self.arr[name] = np.empty((self.n,) + v.shape, v.dtype)
            self.first[name] = self.rows
        a[self.rows] = value

    def add(self, name: str, value) -> None:
        self.sparse.setdefault(name, []).append(value)

    def take(self) -> dict:
        """The rows written so far, column by column (views of the arrays; this object is not written again)."""
        out = {k: a[self.first[k]:self.rows] for k, a in self.arr.items()}
        out.update({k: v for k, v in self.sparse.items() if v})
        return out


def _small(obj, depth=0):
    """A JSON-safe, image-free summary of a request or result."""
    if isinstance(obj, np.ndarray):
        return {"array": list(obj.shape), "dtype": str(obj.dtype)} if obj.size > 64 else obj.tolist()
    if isinstance(obj, dict):
        return {k: _small(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_small(v, depth + 1) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_depth(path: str, depth: np.ndarray) -> None:
    write_png(path, depth_mm(depth))


class Recorder:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.req = open(os.path.join(run_dir, "requests.jsonl"), "a")
        self.tick_dir = os.path.join(run_dir, "ticks")
        os.makedirs(self.tick_dir, exist_ok=True)
        for sub in ("code", "frames"):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        self.part = len([f for f in os.listdir(self.tick_dir) if f.endswith(".npz")])
        self.cols = _Columns(CHUNK_TICKS)
        self.n = sum(1 for _ in open(os.path.join(run_dir, "requests.jsonl")))
        self.code = {}                   # client id -> {path: sha}
        self.lock = threading.Lock()     # requests only; ticks are collected under the driver's lock
        # camera images and tick files are written by a thread of their own: a PNG takes ~20 ms, which must not delay
        # the reply, and a minute of ticks up to 0.5 s to compress, which must not delay a control tick
        self._out = queue.Queue()
        self._writer = threading.Thread(target=self._write, daemon=True)
        self._writer.start()

    def _write(self):
        while True:
            item = self._out.get()
            if item is None:
                return
            write, args = item
            try:
                write(*args)
            except Exception:                    # noqa: BLE001 -- one failed file (a full disk) must not stop the rest
                traceback.print_exc()

    def meta(self, info: dict) -> None:
        with open(os.path.join(self.dir, "run.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)

    def request(self, req: dict, rep: dict, wall_s: float, client: int | None = None, t0=None, t1=None) -> None:
        if self.req.closed:                  # robotd is shutting down: a late request (the dashboard's) goes unlogged
            return
        req = dict(req)
        shas = self.code.setdefault(client, {})
        for path, text in (req.pop("src_text", None) or {}).items():
            sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
            out = os.path.join(self.dir, "code", f"{sha}.txt")
            if not os.path.exists(out):
                with open(out, "w", encoding="utf-8") as f:
                    f.write(text)
            shas[path] = sha
        if req.get("src"):
            req["src"] = [[p, line, fn, shas.get(p)] for p, line, fn in req["src"]]
        if req.get("op") == "cameras" and rep.get("ok"):
            frames = {}
            for name, fr in (rep.get("result") or {}).items():
                entry = {k: fr.get(k) for k in ("K", "T", "t")}
                rgb = fr.get("rgb")
                if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
                    fn = f"{self.n:06d}-{name}.png"
                    self._out.put((write_png, (os.path.join(self.dir, "frames", fn), rgb)))
                    entry["png"] = f"frames/{fn}"
                depth = fr.get("depth")
                if isinstance(depth, np.ndarray) and depth.ndim == 2:
                    fn = f"{self.n:06d}-{name}_depth.png"
                    self._out.put((_write_depth, (os.path.join(self.dir, "frames", fn), depth)))
                    entry["depth_png"] = f"frames/{fn}"
                if fr.get("roi") is not None:
                    entry["roi"] = fr["roi"]
                frames[name] = entry
            rep = {"ok": True, "result": frames}
        self.req.write(json.dumps({"n": self.n, "wall": time.time(), "dur_s": round(wall_s, 4), "client": client,
                                   "t0": t0, "t1": t1, "req": _small(req), "rep": _small(rep)}, default=str) + "\n")
        self.req.flush()
        self.n += 1

    def tick(self, c) -> None:
        cols = self.cols
        cols.put("t", c.t)
        cols.put("q", c.m.q)
        cols.put("dq", c.m.dq)
        cols.put("q_ref", c.q_ref)
        cols.put("q_cmd", c.q_cmd)
        cols.put("bias", c.bias)
        cols.put("tau", c.m.tau)
        for a in c.arms:
            cols.put(f"wrench_{a}", c.wrench(a))
        poses = getattr(c.b, "privileged_poses", None)
        if poses is not None:
            for name, p in poses().items():
                cols.put(f"obj/{name}", p)
        noise = getattr(c.b, "privileged_noise", None)
        if noise is not None:
            for name, v in noise().items():
                cols.put(f"noise/{name}", v)
        deform = getattr(c.b, "privileged_deformables", None)
        if deform is not None and c.ticks % CLOTH_EVERY == 0:          # cloth vertices at 25 Hz
            for name, v in deform().items():
                cols.add(f"cloth/{name}", v.astype(np.float32))
                cols.add(f"cloth_t/{name}", c.t)
        cols.rows += 1
        if cols.rows >= cols.n:
            self.save()

    def save(self) -> str | None:
        """Hand the ticks so far to the writer thread (the tick only starts new arrays)."""
        if not self.cols.rows:
            return None
        path = os.path.join(self.tick_dir, f"{self.part:05d}.npz")
        self._out.put((_save_ticks, (path, self.cols.take())))
        self.part += 1
        self.cols = _Columns(CHUNK_TICKS)
        return path

    def close(self) -> None:
        self.save()
        self.req.close()
        self._out.put(None)
        self._writer.join()


def _save_ticks(path: str, cols: dict) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in cols.items() if len(v)})


def load_ticks(run_dir: str) -> dict:
    """All recorded ticks of a run, joined in time order: {column: array}."""
    tick_dir = os.path.join(run_dir, "ticks")
    parts = [np.load(os.path.join(tick_dir, f)) for f in sorted(os.listdir(tick_dir)) if f.endswith(".npz")]
    return {k: np.concatenate([p[k] for p in parts]) for k in (parts[0].files if parts else [])}
