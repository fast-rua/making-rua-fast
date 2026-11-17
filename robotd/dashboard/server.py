"""The operator's dashboard: a web page served next to robotd, on the robot's computer.

    python -m robotd.dashboard --addr unix:/tmp/robot.sock          # then open http://localhost:8780

It is a separate process and an ordinary client of the driver (three connections: state, cameras, control), so it
can never hold up the control loop, and a dashboard crash leaves the robot as it was. The control connection carries
only the operator's requests: freeze and soft (latched operator stops), release, park, zero force, off.

It also runs programs (programs.py): your Python files (--programs) and the repository's examples, one at a time, each
as its own process against the driver. FREEZE, SOFT and OFF also stop the running program.

It listens on localhost only, unless given --host and a --token. Requests that act on the robot need the
X-Dashboard header, which a page from another site cannot send without a CORS preflight that this server never grants,
and the Host header must be one this server answers to (no DNS rebinding).
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import socket
import struct
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import robot
from robotd.dashboard.programs import ProgramError, ProgramManager

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples")
STOPS_PROGRAMS = {"freeze": "operator FREEZE", "soft": "operator SOFT", "off": "operator OFF"}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
ACTIONS = {"freeze": ("operator_stop", {"soft": False}), "soft": ("operator_stop", {"soft": True}),
           "release": ("release", {}), "park": ("park", {}), "stop": ("stop", {}),
           "zero": ("zero_force", {}),                     # zero: every arm's force estimate, here and now
           "off": ("shutdown", {})}                        # off: robotd parks the arms, lets every motor go, exits
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class Hub:
    """Everything the page shows, refreshed by background threads; the HTTP handlers only read it."""

    def __init__(self, addr: str, state_hz=20.0, camera_hz=3.0, stride=4, max_points=25000, crop=None,
                 jpeg_width=480, programs_dir: str | None = None, examples_dir: str | None = EXAMPLES):
        self.addr = addr
        self.programs = (ProgramManager(programs_dir, examples_dir, addr,
                                        on_event=lambda kind, text, **i: self.log(kind, text, **i))
                         if programs_dir else None)
        self.state_period = 1.0 / state_hz
        self.camera_period = 1.0 / camera_hz
        self.stride, self.max_points, self.jpeg_width = stride, max_points, jpeg_width
        self.crop = np.asarray(crop if crop is not None else [-0.5, 1.2, -1.0, 1.0, -0.15, 1.2], float)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.connected = False
        self.meta, self.meta_seq = None, 0
        self.model, self.model_json, self.mesh_blob = None, {"error": "not connected"}, b""
        self.live = json.dumps({"connected": False}).encode()
        self.health, self.health_at, self.ticks_prev = {}, 0.0, None
        self.tick_hz = None
        self.cloud, self.cloud_seq = b"", 0
        self.jpeg, self.cam_seq, self.cam_error, self.cam_at = {}, 0, None, 0.0
        self.cam_info = {}                                   # name -> {"pose": bool, "pose_error": str | None}
        self.view_list, self.view_jpeg = [], {}              # images programs showed (robot.viz.show)
        self.view_seq, self.views_since = 0, 0
        self.log_items = collections.deque(maxlen=1500)
        self.log_seq = 0
        self.robot_events_seq = 0
        self.ctl, self.ctl_lock = None, threading.Lock()

    # ------------------------------------------------------------------ log
    def log(self, kind: str, text: str, source: str = "dashboard", **info):
        with self.lock:
            self.log_seq += 1
            if info.get("count", 1) > 1:                   # the driver collapsed a burst: update that entry, at the end
                for last in list(self.log_items)[-4:]:
                    if last["kind"] == kind and last["text"] == text:
                        last.update(info, seq=self.log_seq, wall=time.time())
                        self.log_items.remove(last)
                        self.log_items.append(last)
                        return
            self.log_items.append({"seq": self.log_seq, "wall": time.time(), "kind": kind, "text": text,
                                   "source": source, **info})

    def log_since(self, since: int) -> dict:
        with self.lock:
            return {"seq": self.log_seq, "events": [e for e in self.log_items if e["seq"] > since][-400:]}

    # ------------------------------------------------------------------ threads
    def start(self):
        for target, name in ((self._state_loop, "dash-state"), (self._camera_loop, "dash-cameras")):
            threading.Thread(target=target, daemon=True, name=name).start()

    def _connect(self, name: str, timeout: float) -> robot.Robot:
        r = robot.connect(self.addr, name=name)
        r.sock.settimeout(timeout)
        return r

    def _state_loop(self):
        while not self.stop_event.is_set():
            r = None
            try:
                r = self._connect("dashboard/state", 3.0)
                self._on_connect(r)
                last_health = last_events = 0.0
                while not self.stop_event.is_set():
                    t0 = time.monotonic()
                    st = r._call("state")
                    poll_ms = (time.monotonic() - t0) * 1000
                    if t0 - last_health > 0.5:
                        self._on_health(r._call("health"))
                        last_health = t0
                    if t0 - last_events > 0.25:
                        ev = r._call("events", since=self.robot_events_seq)
                        for e in ev["events"]:
                            self.log(e.pop("kind"), e.pop("text"), source="robotd", **{k: v for k, v in e.items()
                                                                                         if k not in ("seq", "wall")})
                        self.robot_events_seq = ev["seq"]
                        last_events = t0
                    self._publish(st, poll_ms)
                    time.sleep(max(0.0, self.state_period - (time.monotonic() - t0)))
            except Exception as e:                           # noqa: BLE001 -- show it, retry
                if self.connected:
                    self.log("disconnected", f"lost the driver: {e}", level="error")
                self.connected = False
                with self.lock:
                    self.live = json.dumps({"connected": False, "error": str(e), "wall": time.time(),
                                            "meta_seq": self.meta_seq}).encode()
                if r is not None:
                    r.close()
                time.sleep(1.0)

    def _on_connect(self, r: robot.Robot):
        hello = r._call("hello", client="dashboard/state")
        desc = hello["description"]
        st = r._call("state")
        joints = st["joints"]
        model, model_json, blob, limits = None, {"error": None}, b"", None
        try:
            from robotd.dashboard.model import DisplayModel
            model = DisplayModel(desc["files"]["mjcf"], joints, list(desc["arm_joints"]), desc["tcp_sites"])
            model_json, blob = model.payload()
            limits = model.limits
        except Exception as e:                               # noqa: BLE001 -- the 3D view says why; the rest works
            model_json = {"error": f"no 3D model: {e}"}
        meta = {"backend": hello["backend"], "lockstep": hello["lockstep"], "passive": hello.get("passive", False),
                "joints": joints,
                "arms": list(desc["arm_joints"]), "arm_joints": desc["arm_joints"],
                "finger_joint": desc["finger_joint"], "limits": limits,
                "effort_limit": [desc["effort_limit"][n] for n in joints],
                "velocity_limit": [desc["velocity_limit"].get(n) for n in joints],
                "max_velocity": [desc["max_velocity"].get(n) for n in joints],
                "soft_kd": [desc.get("soft_kd", {}).get(n) for n in joints],
                "cameras": list(desc.get("cameras", {})), "camera_info": desc.get("cameras", {}),
                "gripper_width_range_m": desc.get("gripper_width_range_m"), "robot": desc.get("robot")}
        with self.lock:
            self.model, self.model_json, self.mesh_blob = model, model_json, blob
            self.meta = meta
            self.meta_seq += 1
        self.connected = True
        self.log("connected", f"connected to robotd ({hello['backend']}, "
                              f"{'lockstep' if hello['lockstep'] else 'realtime'}) at {self.addr}")
        if hello.get("passive"):
            self.log("warning", "monitor mode: the motors are off; the page shows the robot, nothing can move it",
                     level="warn")
        if hello["lockstep"]:
            self.log("warning", "the driver is lockstep: the view only moves while a program blocks in a call "
                                "(start the simulation with --realtime to watch it live)", level="warn")
        self._ensure_control()

    def _on_health(self, h: dict):
        now = time.monotonic()
        if self.ticks_prev is not None and now > self.ticks_prev[1]:
            self.tick_hz = (h["ticks"] - self.ticks_prev[0]) / (now - self.ticks_prev[1])
        self.ticks_prev = (h["ticks"], now)
        self.health, self.health_at = h, now

    def _publish(self, st: dict, poll_ms: float):
        model = self.model
        q, q_ref = st["q"], st["q_ref"]
        geoms = ghost = diagram = None
        if model is not None:
            geoms = model.poses(q)
            if np.max(np.abs(np.asarray(q_ref) - np.asarray(q))) > 2e-3:
                ghost = model.poses(q_ref)
            diagram = model.diagram(q)
        live = {"connected": True, "wall": time.time(), "poll_ms": round(poll_ms, 2), "meta_seq": self.meta_seq,
                "t": st["t"], "q": q, "q_ref": q_ref, "q_cmd": st.get("q_cmd"), "kp": st.get("kp"), "kd": st.get("kd"),
                "dq": st["dq"], "effort": st["effort"],
                "effort_ext": st["effort_ext"], "grippers": st["grippers"], "busy": st["busy"],
                "fault": st["fault"], "parked": st.get("parked"),
                "ee": {a: {"T": e["T"], "wrench": e["wrench"], "contact": e["contact"]} for a, e in st["ee"].items()},
                "geoms": geoms, "ghost": ghost, "diagram": diagram,
                "health": self.health, "tick_hz": self.tick_hz,
                "cloud_seq": self.cloud_seq, "cam_seq": self.cam_seq, "cam_error": self.cam_error,
                "cams": self.cam_info,
                "cam_age": time.monotonic() - self.cam_at if self.cam_at else None, "log_seq": self.log_seq,
                "views": self.view_list, "view_seq": self.view_seq, "wall_now": time.time(),
                "program": self.programs.status() if self.programs else None}
        data = json.dumps(_finite(live)).encode()
        with self.lock:
            self.live = data

    def _camera_loop(self):
        r = None
        while not self.stop_event.is_set():
            if not self.connected or self.meta is None or not self.meta["cameras"]:
                time.sleep(0.5)
                continue
            t0 = time.monotonic()
            try:
                if r is None:
                    r = self._connect("dashboard/cameras", 10.0)
                frames = r.cameras(depth=True)
                self._on_frames(frames)
                self._poll_views(r)
                self.cam_error = None
            except robot.RobotError as e:                    # e.g. no cameras configured: say so, try again later
                if str(e) != self.cam_error:
                    self.log("cameras", f"cameras: {e}", level="warn")
                self.cam_error = str(e)
                if "closed the connection" in str(e):
                    r = None
                time.sleep(5.0)
                continue
            except Exception as e:                           # noqa: BLE001 -- connection trouble: reconnect
                self.cam_error = str(e)
                if r is not None:
                    r.close()
                r = None
                time.sleep(1.0)
                continue
            time.sleep(max(0.0, self.camera_period - (time.monotonic() - t0)))

    def _on_frames(self, frames: dict):
        jpeg, pts, cols, counts, info = {}, [], [], [], {}
        for name, f in frames.items():
            info[name] = {"pose": f.T is not None, "pose_error": f.pose_error}
            if f.rgb is not None:
                jpeg[name] = _jpeg(f.rgb, self.jpeg_width)
            if f.depth is None or f.T is None:              # no pose: the image only
                continue
            P, C = self._points(f)
            pts.append(P)
            cols.append(C)
            counts.append([name, int(len(P))])
        head = json.dumps({"seq": self.cloud_seq + 1, "counts": counts}).encode()
        head += b" " * (-(len(head) + 4) % 4)                # keep the float block 4-byte aligned
        P = np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32)
        C = np.concatenate(cols).astype(np.uint8) if cols else np.zeros((0, 3), np.uint8)
        blob = struct.pack("<I", len(head)) + head + P.tobytes() + C.tobytes()
        with self.lock:
            self.jpeg = jpeg
            self.cam_info = info
            self.cloud = blob
            self.cloud_seq += 1
            self.cam_seq += 1
        self.cam_at = time.monotonic()

    def _poll_views(self, r: robot.Robot):
        """The images programs showed since the last poll (robot.viz.show), from the image server's board."""
        try:
            rep = r._views("views", since=self.views_since)
        except Exception:                                    # noqa: BLE001 -- an image server without the board
            return
        if rep["seq"] < self.views_since:                    # the server restarted: everything again next time
            self.views_since = 0
            return
        with self.lock:
            for v in rep["views"]:
                self.view_jpeg[v["name"]] = bytes(np.asarray(v["jpeg"], np.uint8))
            self.view_list = rep["names"]
            keep = {v["name"] for v in rep["names"]}
            for name in [n for n in self.view_jpeg if n not in keep]:
                del self.view_jpeg[name]
            if rep["seq"] != self.views_since:
                self.view_seq += 1
            self.views_since = rep["seq"]

    def _points(self, f) -> tuple[np.ndarray, np.ndarray]:
        """World points and colours from one depth frame: strided, cropped to the work volume, capped."""
        depth = f.depth
        H, W = depth.shape
        v, u = np.mgrid[0:H:self.stride, 0:W:self.stride].reshape(2, -1)
        z = depth[v, u].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.05) & (z < 3.0)
        u, v, z = u[ok], v[ok], z[ok]
        x = (u - f.K[0, 2]) * z / f.K[0, 0]
        y = (v - f.K[1, 2]) * z / f.K[1, 1]
        P = (f.T[:3, :3] @ np.stack([x, y, z])).T + f.T[:3, 3]
        c = self.crop
        keep = ((P[:, 0] > c[0]) & (P[:, 0] < c[1]) & (P[:, 1] > c[2]) & (P[:, 1] < c[3])
                & (P[:, 2] > c[4]) & (P[:, 2] < c[5]))
        P, u, v = P[keep], u[keep], v[keep]
        if len(P) > self.max_points:
            pick = np.linspace(0, len(P) - 1, self.max_points).astype(int)
            P, u, v = P[pick], u[pick], v[pick]
        if f.rgb is not None:
            Hc, Wc = f.rgb.shape[:2]
            C = f.rgb[(v * Hc) // H, (u * Wc) // W][:, :3]
        else:
            C = np.full((len(P), 3), 200, np.uint8)
        return P, C

    # ------------------------------------------------------------------ the operator's requests
    def _ensure_control(self):
        if not self.ctl_lock.acquire(timeout=0.1):
            return
        try:
            if self.ctl is None:
                self.ctl = self._connect("dashboard/control", 2.0)
        except Exception:                                    # noqa: BLE001 -- the next action retries
            self.ctl = None
        finally:
            self.ctl_lock.release()

    def action(self, name: str) -> dict:
        res = self._action(name)
        if name in STOPS_PROGRAMS and self.programs is not None:      # the operator's stop stops the program too
            self.programs.stop(STOPS_PROGRAMS[name])
        return res

    def _action(self, name: str) -> dict:
        op, kw = ACTIONS[name]
        with self.ctl_lock:
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    if self.ctl is None:
                        self.ctl = self._connect("dashboard/control", 2.0)
                    if op == "zero_force":
                        res = [self.ctl._call(op, ee=a) for a in (self.meta or {}).get("arms", [])]
                    else:
                        res = self.ctl._call(op, **kw)
                except (OSError, socket.timeout) as e:       # connection trouble: reconnect once, then give up
                    if self.ctl is not None:
                        self.ctl.close()
                    self.ctl = None
                    if attempt == 0:
                        continue
                    self.log("action", f"{name.upper()} failed: driver unreachable ({e})", level="error")
                    return {"ok": False, "error": f"driver unreachable: {e}"}
                except robot.RobotError as e:
                    if "closed the connection" in str(e) and attempt == 0:
                        self.ctl = None
                        continue
                    self.log("action", f"{name.upper()} refused: {e}", level="warn")
                    return {"ok": False, "error": str(e)}
                ms = (time.perf_counter() - t0) * 1000
                self.log("action", f"operator {name.upper()} (acknowledged in {ms:.1f} ms)", level="action")
                return {"ok": True, "result": res, "ms": ms}
        return {"ok": False, "error": "unreachable"}


def _finite(x):
    """NaN and infinity as null: the browser's JSON parser rejects them."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _jpeg(rgb: np.ndarray, width: int) -> bytes:
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3]))
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None
    token: str | None = None
    hosts: set = LOOPBACK
    server_version = "robotd-dashboard"

    def log_message(self, fmt, *args):                       # quiet: the page has its own log
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        name = host[1:host.index("]")] if host.startswith("[") else host.rsplit(":", 1)[0]
        return self.token is not None or name in self.hosts

    def _token_ok(self, query: dict) -> bool:
        if self.token is None:
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        given = (query.get("token", [None])[0] or self.headers.get("X-Token")
                 or (jar["robotd_dash"].value if "robotd_dash" in jar else None))
        return given == self.token

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._host_ok():
            return self._send(421, b"unknown host", "text/plain")
        if not self._token_ok(query):
            return self._send(401, b"token required: open the page as /?token=...", "text/plain")
        hub, path = self.hub, url.path
        if path in ("/", "/index.html"):
            body = open(os.path.join(STATIC, "index.html"), "rb").read()
            extra = {}
            if self.token is not None and "token" in query:
                extra["Set-Cookie"] = f"robotd_dash={self.token}; HttpOnly; SameSite=Strict; Path=/"
            return self._send(200, body, CONTENT_TYPES[".html"], extra)
        if path.startswith("/static/"):
            name = os.path.basename(path)
            full = os.path.join(STATIC, name)
            if not os.path.isfile(full):
                return self._send(404, b"not found", "text/plain")
            return self._send(200, open(full, "rb").read(),
                              CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream"))
        if path == "/api/live":
            with hub.lock:
                data = hub.live
            return self._send(200, data, "application/json")
        if path == "/api/meta":
            with hub.lock:
                return self._json({"meta": hub.meta, "meta_seq": hub.meta_seq, "addr": hub.addr})
        if path == "/api/model":
            with hub.lock:
                return self._json({"model": hub.model_json, "meta_seq": hub.meta_seq})
        if path == "/api/mesh.bin":
            with hub.lock:
                blob = hub.mesh_blob
            return self._send(200, blob, "application/octet-stream")
        if path == "/api/events":
            return self._json(hub.log_since(int(query.get("since", ["0"])[0])))
        if path == "/api/cloud.bin":
            with hub.lock:
                blob = hub.cloud
            return self._send(200, blob, "application/octet-stream")
        if path.startswith("/api/programs"):
            return self._programs_get(path, query)
        if path.startswith("/api/view/"):
            name = os.path.splitext(os.path.basename(path))[0]
            with hub.lock:
                img = hub.view_jpeg.get(name)
            if img is None:
                return self._send(404, b"no such view", "text/plain")
            return self._send(200, img, "image/jpeg")
        if path.startswith("/api/camera/"):
            name = os.path.splitext(os.path.basename(path))[0]
            with hub.lock:
                img = hub.jpeg.get(name)
            if img is None:
                return self._send(404, b"no frame", "text/plain")
            return self._send(200, img, "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if not self._host_ok():
            return self._send(421, b"unknown host", "text/plain")
        if not self._token_ok(parse_qs(url.query)):
            return self._send(401, b"token required", "text/plain")
        if self.headers.get("X-Dashboard") != "1":
            return self._send(403, b"missing X-Dashboard header", "text/plain")
        if url.path.startswith("/api/action/"):
            name = url.path.rsplit("/", 1)[1]
            if name not in ACTIONS:
                return self._json({"ok": False, "error": f"unknown action {name}"}, 404)
            return self._json(self.hub.action(name))
        if url.path.startswith("/api/programs/"):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 600_000:
                return self._send(413, b"too large", "text/plain")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "body is not JSON"}, 400)
            return self._programs_post(url.path.rsplit("/", 1)[1], body)
        return self._send(404, b"not found", "text/plain")

    def _programs_get(self, path, query):
        pm = self.hub.programs
        if pm is None:
            return self._json({"ok": False, "error": "programs are off (--no-programs)"}, 404)
        q = lambda k, d="": query.get(k, [d])[0]
        try:
            if path == "/api/programs":
                return self._json({"ok": True, "programs": pm.list(), "status": pm.status(), "dir": pm.dirs["mine"]})
            if path == "/api/programs/source":
                return self._json({"ok": True, "name": q("name"), "origin": q("origin", "mine"),
                                   "source": pm.source(q("name"), q("origin", "mine"))})
            if path == "/api/programs/output":
                return self._json({"ok": True, **pm.output(int(q("since", "0")))})
        except (ProgramError, OSError) as e:
            return self._json({"ok": False, "error": str(e)})
        return self._send(404, b"not found", "text/plain")

    def _programs_post(self, verb, body):
        pm = self.hub.programs
        if pm is None:
            return self._json({"ok": False, "error": "programs are off (--no-programs)"}, 404)
        try:
            if verb == "run":
                return self._json({"ok": True, "running": pm.run(body["name"], body.get("origin", "mine"),
                                                                 body.get("args", ""))})
            if verb == "stop":
                return self._json({"ok": True, **pm.stop("stopped from the dashboard")})
            if verb == "save":
                return self._json({"ok": True, **pm.save(body["name"], body.get("source"))})
            if verb == "delete":
                return self._json({"ok": True, **pm.delete(body["name"])})
            if verb == "copy":
                return self._json({"ok": True, **pm.copy(body["name"], body.get("origin", "example"),
                                                         body.get("new_name"))})
        except (ProgramError, OSError, KeyError) as e:
            return self._json({"ok": False, "error": str(e)})
        return self._json({"ok": False, "error": f"unknown request {verb}"}, 404)


def serve(hub: Hub, host: str = "127.0.0.1", port: int = 8780, token: str | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"hub": hub, "token": token, "hosts": LOOPBACK | {host}})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main():
    ap = argparse.ArgumentParser(description="robotd operator dashboard")
    ap.add_argument("--addr", default=os.environ.get("ROBOT_ADDR", "unix:/tmp/robot.sock"), help="robotd address")
    ap.add_argument("--host", default="127.0.0.1", help="listen address (anything but localhost needs --token)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--token", default=None, help="required as /?token=... when listening beyond localhost")
    ap.add_argument("--camera-hz", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=4, help="point cloud: every n-th pixel")
    ap.add_argument("--max-points", type=int, default=25000, help="point cloud: per camera")
    ap.add_argument("--crop", default="-0.5,1.2,-1.0,1.0,-0.15,1.2", help="point cloud box: x0,x1,y0,y1,z0,z1 (m)")
    ap.add_argument("--programs", default="~/robot-programs", help="your programs: run, edit, add, delete here")
    ap.add_argument("--examples", default=EXAMPLES, help="read-only example programs to run or copy")
    ap.add_argument("--no-programs", action="store_true", help="a viewer only: no programs tab")
    args = ap.parse_args()
    if args.host not in LOOPBACK and not args.token:
        ap.error(f"--host {args.host} would let anyone who can reach this machine press the stop buttons and park: "
                 "add --token <secret>")
    hub = Hub(args.addr, camera_hz=args.camera_hz, stride=args.stride, max_points=args.max_points,
              crop=[float(v) for v in args.crop.split(",")],
              programs_dir=None if args.no_programs else args.programs, examples_dir=args.examples)
    hub.start()
    httpd = serve(hub, args.host, args.port, args.token)
    shown = "localhost" if args.host in LOOPBACK else args.host
    print(f"robotd dashboard: http://{shown}:{args.port}/" + (f"?token={args.token}" if args.token else "")
          + f"  (driver {args.addr})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop_event.set()
        if hub.programs is not None:
            hub.programs.stop("the dashboard is shutting down")
        httpd.server_close()


if __name__ == "__main__":
    main()
