"""A live monitor for agent episodes (scripts/agent_run_real.py, scripts/agent_run.py): what the agent is thinking and
doing, the files in its workspace and how they changed since the episode started, the programs it runs, and the
cameras. It never takes git's locks (the agent's own git keeps working), so it can watch a live episode. The one
thing it writes: an operator's steering message (the Agent tab's box), appended to RUN/steer/inbox.jsonl, which the
episode's launcher hands to the agent (scripts/agent/codex_driver.py). Only requests with the X-Monitor header are
taken, which a page from another site cannot send.

    python scripts/agent_monitor.py [--runs $ROBOT_HOME/agent-runs] [--port 8790]
    # from your computer: ssh -N -L 8790:127.0.0.1:8790 <you>@<cell>
    #                     then http://localhost:8790

What it reads in RUNS/<tag>/ (agent_run.py's layout):
  meta.json, verdict.json, before.json   the episode, its end and its check
  codex/events.jsonl                     Codex's --json stream: reasoning, messages, commands, file changes, usage
  workspace/                             the agent's files; its first commit is the episode start (the diff base)
  workspace/.robot/runs/<id>/            each program run (python -m robot run): status.json, log.txt, the code
                                         that ran (src/), pause snapshots
  logs/program-host.jsonl                which programs started, exited or were stopped
  witness/                               camera frames every few seconds (the real cell)
Standard library only.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import mimetypes
import os
import subprocess
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "agent", "monitor.html")
MAX_TEXT = 600_000                 # bytes of a file shown as text
MAX_EVENTS = 400                   # events per poll
SKIP_DIRS = {".git", "__pycache__", ".cache"}
GIT_ENV = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1")
# the workspace is the agent's, .git/config included: never let its settings run a command here
GIT_SAFE = ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.pager=cat"]


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def git(ws, *args, binary=False):
    try:
        out = subprocess.run(["git", *GIT_SAFE, "-C", ws, *args], capture_output=True, env=GIT_ENV, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Run:
    """One run directory, and the caches that make polling it cheap."""

    def __init__(self, path):
        self.path, self.ws = path, os.path.join(path, "workspace")
        self._base = None
        self.lock = threading.Lock()

    # ---- paths
    def inside(self, rel: str) -> str | None:
        """rel (relative to the run directory) as a real path that stays inside it, else None."""
        p = os.path.realpath(os.path.join(self.path, rel.lstrip("/")))
        root = os.path.realpath(self.path)
        return p if p == root or p.startswith(root + os.sep) else None

    def host_path(self, sandbox_path: str) -> str:
        """A path the agent saw (/workspace/...) as one in the run directory."""
        return os.path.join(self.ws, sandbox_path[len("/workspace/"):]) if sandbox_path.startswith("/workspace/") \
            else sandbox_path

    # ---- the episode
    def summary(self) -> dict:
        meta = read_json(os.path.join(self.path, "meta.json"), {}) or {}
        verdict = read_json(os.path.join(self.path, "verdict.json"))
        keep = ("tag", "task", "instruction", "agent", "state", "real", "started_utc", "agent_started_utc",
                "agent_ended_utc", "ended_utc", "end_reason", "agent_wall_s", "codex_turns", "codex_usage",
                "codex_errors", "notes", "limits", "towel", "check", "finished")
        out = {k: meta.get(k) for k in keep if k in meta}
        out["codex"] = {k: (meta.get("codex") or {}).get(k) for k in ("version", "model", "reasoning_effort")}
        out["verdict"] = verdict
        out["witness"] = self.witness(2)
        out["videos"] = [{"path": n, "bytes": os.path.getsize(os.path.join(self.path, n))}
                         for n in sorted(os.listdir(self.path)) if n.startswith("video") and n.endswith(".mp4")]
        out["programs"] = self.programs()
        out["running"] = [p for p in out["programs"] if p["state"] in ("running", "paused", "starting")]
        out["debrief"] = self.debrief()
        out["steerable"] = (meta.get("state") == "running" and os.path.exists(os.path.join(self.path, "steer",
                                                                                           "inbox.jsonl"))) \
            or out["debrief"]["live"]
        return out

    def debrief(self) -> dict:
        """scripts/agent_debrief.py's sessions for this run, and whether one is open now (its heartbeat)."""
        root = os.path.join(self.path, "debrief")
        try:
            sessions = sorted(int(x) for x in os.listdir(root) if x.isdigit())
        except OSError:
            sessions = []
        live = read_json(os.path.join(root, "live.json"), None)
        is_live = bool(live) and time.time() - live.get("wall", 0) < 20
        return {"live": is_live, "session": live.get("session") if is_live else None, "sessions": sessions}

    def _streams(self) -> list:
        """The episode's event stream, then each debrief's, in order."""
        out = [os.path.join(self.path, "codex", "events.jsonl")]
        for n in self.debrief()["sessions"]:
            out.append(os.path.join(self.path, "debrief", str(n), "events.jsonl"))
        return out

    def witness(self, n=2) -> list:
        d = os.path.join(self.path, "witness")
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return []
        latest, seen = [], set()
        for name in reversed(names):                       # the newest frame of each camera
            cam = name.split("-", 1)[-1].rsplit(".", 1)[0]
            if cam not in seen:
                seen.add(cam)
                latest.append({"camera": cam, "path": f"witness/{name}", "t": name.split("-", 1)[0]})
            if len(seen) >= n:
                break
        return sorted(latest, key=lambda x: x["camera"])

    def programs(self) -> list:
        """Every program the agent ran: from the program host's log, with each run's status.json."""
        host = {}
        try:
            with open(os.path.join(self.path, "logs", "program-host.jsonl")) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    rd = e.get("run_dir")
                    if not rd:
                        continue
                    h = host.setdefault(rd, {"run_dir": rd})
                    if e.get("event") == "start":
                        h.update(start=e.get("wall"), file=(e.get("argv") or [""])[-1], pid=e.get("pid"))
                    elif e.get("event") in ("exit", "stop"):
                        h.update(end=e.get("wall"), how=e.get("event"), returncode=e.get("returncode"),
                                 signal=e.get("signal"))
        except OSError:
            pass
        runs_dir = os.path.join(self.ws, ".robot", "runs")
        try:
            ids = sorted(os.listdir(runs_dir))
        except OSError:
            ids = []
        for rid in ids:                                    # runs the host does not know of (old ones) still count
            host.setdefault(f"/workspace/.robot/runs/{rid}", {"run_dir": f"/workspace/.robot/runs/{rid}"})
        out = []
        for rd, h in host.items():
            st = read_json(os.path.join(self.host_path(rd), "status.json"), {}) or {}
            ended = "end" in h
            state = st.get("state") or ("ended" if ended else "starting")
            if ended and state in ("running", "paused"):
                state = "stopped" if h.get("how") == "stop" else "ended"
            file = h.get("file") or st.get("task") or ""
            out.append({"id": os.path.basename(rd), "file": file, "rel": os.path.relpath(self.host_path(file), self.ws)
                        if file.startswith("/workspace/") else file, "state": state, "step": st.get("step"),
                        "line": st.get("line"), "function": st.get("function"), "message": st.get("message"),
                        "kind": st.get("kind"), "start": h.get("start") or st.get("started"),
                        "end": h.get("end"), "exit_code": st.get("exit_code", h.get("returncode")),
                        "signal": h.get("signal"), "pause": st.get("pause"), "unattended": st.get("unattended")})
        return sorted(out, key=lambda p: p.get("start") or 0)

    def views(self, limit: int = 120) -> dict:
        """Every image the agent's programs showed (robot.viz.show), newest first: the runs' views/history and the
        workspace's own .robot/views/history (shows from outside the program runner)."""
        found = []
        dirs = [(os.path.join(self.ws, ".robot", "views", "history"), "workspace/.robot/views/history", None)]
        try:
            for rid in os.listdir(os.path.join(self.ws, ".robot", "runs")):
                dirs.append((os.path.join(self.ws, ".robot", "runs", rid, "views", "history"),
                             f"workspace/.robot/runs/{rid}/views/history", rid))
        except OSError:
            pass
        for d, rel, rid in dirs:
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for x in names:
                if x.endswith((".jpg", ".png")):
                    try:
                        mt = os.path.getmtime(os.path.join(d, x))
                    except OSError:
                        continue
                    name = x.split("-", 1)[1].rsplit(".", 1)[0] if "-" in x else x.rsplit(".", 1)[0]
                    found.append({"path": f"{rel}/{x}", "name": name, "run": rid, "t": mt})
        found.sort(key=lambda v: -v["t"])
        return {"views": found[:limit], "total": len(found)}

    def program(self, rid: str) -> dict:
        d = os.path.join(self.ws, ".robot", "runs", os.path.basename(rid))
        out = {"id": rid, "status": read_json(os.path.join(d, "status.json")), "log": "", "versions": [],
               "pauses": []}
        try:
            with open(os.path.join(d, "log.txt"), "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 200_000))
                out["log"] = f.read().decode("utf-8", "replace")
                out["log_truncated"] = size > 200_000
        except OSError:
            pass
        try:
            with open(os.path.join(d, "src", "index.jsonl")) as f:
                for line in f:
                    try:
                        out["versions"].append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        try:                                    # images the program showed (robot.viz.show), oldest first
            vd = os.path.join(d, "views", "history")
            out["views"] = [f"workspace/.robot/runs/{rid}/views/history/{x}" for x in sorted(os.listdir(vd))
                            if x.endswith((".jpg", ".png"))][-60:]
        except OSError:
            out["views"] = []
        try:
            for p in sorted(x for x in os.listdir(d) if x.startswith("pause-")):
                imgs = sorted(x for x in os.listdir(os.path.join(d, p)) if x.endswith(".png") and "_depth" not in x)
                out["pauses"].append({"dir": p, "images": [f"workspace/.robot/runs/{rid}/{p}/{x}" for x in imgs]})
        except OSError:
            pass
        return out

    def version(self, rid: str, sha: str) -> str | None:
        p = os.path.join(self.ws, ".robot", "runs", os.path.basename(rid), "src", os.path.basename(sha) + ".py")
        try:
            with open(p, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return None

    # ---- the agent's stream
    def events(self, since: int) -> dict:
        """Events from offset `since` of the episode's stream and its debriefs' after it, read as one: a later
        file's offsets start where the earlier files end (an earlier one only grows while it is the last)."""
        out, offset, base = [], since, 0
        streams = self._streams()
        for k, path in enumerate(streams):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            last = k == len(streams) - 1
            if not last and offset >= base + size:
                base += size
                continue
            try:
                with open(path, "rb") as f:
                    f.seek(offset - base)
                    for line in f:
                        if not line.endswith(b"\n"):       # a line still being written: next poll
                            break
                        offset += len(line)
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            out.append({"type": "unparsed", "text": line.decode("utf-8", "replace")[:2000]})
                        if len(out) >= MAX_EVENTS:
                            return {"events": out, "offset": offset, "more": True}
            except OSError:
                pass
            if not last:
                if offset - base < size:                     # a partial line in a finished file: stop here
                    break
                base += size
        return {"events": out, "offset": offset, "more": False}

    # ---- the workspace against the episode start
    def base(self) -> str | None:
        if self._base is None:
            roots = git(self.ws, "rev-list", "--max-parents=0", "HEAD")
            if roots:
                self._base = roots.split()[-1]
        return self._base

    def base_tree(self) -> dict:
        b = self.base()
        if not b:
            return {}
        out = {}
        for line in (git(self.ws, "ls-tree", "-r", b) or "").splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                out[path] = parts[2]
        return out

    def files(self) -> dict:
        base = self.base_tree()
        cur = {}
        for root, dirs, names in os.walk(self.ws):
            rel_root = os.path.relpath(root, self.ws)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not (rel_root == ".robot" and d == "runs"))   # program runs: the Programs tab
            for n in names:
                p = os.path.join(root, n)
                rel = os.path.normpath(os.path.join(rel_root, n))
                try:
                    stt = os.lstat(p)
                    if not os.path.isfile(p) or os.path.islink(p):
                        continue
                    with open(p, "rb") as f:
                        data = f.read(MAX_TEXT + 1)
                except OSError:
                    continue
                sha = blob_sha(data) if stt.st_size <= MAX_TEXT else None
                status = "added" if rel not in base else ("unchanged" if sha == base[rel] else "modified")
                cur[rel] = {"path": rel, "size": stt.st_size, "mtime": stt.st_mtime, "status": status}
        for rel in base:
            if rel not in cur:
                cur[rel] = {"path": rel, "size": 0, "mtime": None, "status": "deleted"}
        commits = []
        for line in (git(self.ws, "log", "--format=%h%x09%ct%x09%s", "-n", "50") or "").splitlines():
            h, _, rest = line.partition("\t")
            ct, _, subject = rest.partition("\t")
            commits.append({"sha": h, "t": int(ct) if ct.isdigit() else None, "subject": subject})
        return {"files": sorted(cur.values(), key=lambda x: x["path"]), "base": self.base(), "commits": commits}

    def steer(self, text: str) -> dict:
        meta = read_json(os.path.join(self.path, "meta.json"), {}) or {}
        inbox = os.path.join(self.path, "steer", "inbox.jsonl")
        if meta.get("state") != "running" and not self.debrief()["live"]:
            return {"ok": False, "error": "the episode is not running (to talk with its agent: "
                                         "python scripts/agent_debrief.py TAG)"}
        if not os.path.exists(inbox):
            return {"ok": False, "error": "this episode cannot be steered (started with --exec)"}
        with open(inbox, "a") as f:
            f.write(json.dumps({"wall": round(time.time(), 3), "text": text}) + "\n")
        return {"ok": True}

    def file(self, rel: str) -> dict:
        p = self.inside(os.path.join("workspace", rel))
        out = {"path": rel}
        old = None
        if self.base():
            raw = git(self.ws, "show", f"{self.base()}:{rel}", binary=True)
            old = raw.decode("utf-8", "replace") if raw is not None else None
        new = None
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                data = f.read(MAX_TEXT + 1)
            out["size"] = os.path.getsize(p)
            out["mtime"] = os.path.getmtime(p)
            if b"\0" in data[:4096]:
                out["binary"] = True
            else:
                new = data[:MAX_TEXT].decode("utf-8", "replace")
                out["truncated"] = len(data) > MAX_TEXT
        out["text"] = new
        out["status"] = "added" if old is None and new is not None else ("deleted" if new is None and old is not None
                                                                          else ("unchanged" if old == new else "modified"))
        if old != new and not out.get("binary"):
            out["diff"] = "".join(difflib.unified_diff((old or "").splitlines(True), (new or "").splitlines(True),
                                                       f"a/{rel} (episode start)", f"b/{rel} (now)", n=3))
        return out


class Monitor:
    def __init__(self, runs_dir: str):
        self.dir = os.path.realpath(os.path.expanduser(runs_dir))
        self.runs = {}

    def run(self, tag: str) -> Run | None:
        tag = os.path.basename(tag or "")
        path = os.path.join(self.dir, tag)
        if not tag or not os.path.isdir(path):
            return None
        if tag not in self.runs:
            self.runs[tag] = Run(path)
        return self.runs[tag]

    def list(self) -> list:
        out = []
        try:
            tags = os.listdir(self.dir)
        except OSError:
            return out
        for tag in tags:
            meta = read_json(os.path.join(self.dir, tag, "meta.json"))
            if not isinstance(meta, dict):
                continue
            out.append({"tag": tag, "task": meta.get("task"), "state": meta.get("state"),
                        "agent": meta.get("agent"), "started_utc": meta.get("started_utc"),
                        "end_reason": meta.get("end_reason"), "real": meta.get("real"),
                        "success": (meta.get("check") or {}).get("success") if isinstance(meta.get("check"), dict)
                        else (meta.get("verdict") or {}).get("success") if isinstance(meta.get("verdict"), dict)
                        else None,
                        "mtime": os.path.getmtime(os.path.join(self.dir, tag, "meta.json"))})
        return sorted(out, key=lambda r: r.get("started_utc") or "", reverse=True)


def make_handler(mon: Monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):                 # quiet: polls every second
            pass

        def _send(self, body: bytes, ctype="application/json", code=HTTPStatus.OK):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path):
            """A file, with byte ranges (a browser seeks in a video with them)."""
            size = os.path.getsize(path)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                if a:
                    start, end = int(a), min(int(b), size - 1) if b else size - 1
                elif b:
                    start = max(0, size - int(b))
            self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def _json(self, obj, code=HTTPStatus.OK):
            self._send(json.dumps(obj, default=str).encode(), code=code)

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            try:
                if url.path != "/api/steer":
                    return self._json({"error": "unknown"}, HTTPStatus.NOT_FOUND)
                if self.headers.get("X-Monitor") != "1":         # a page from another site cannot set it
                    return self._json({"error": "missing X-Monitor header"}, HTTPStatus.FORBIDDEN)
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                run, text = mon.run(body.get("tag", "")), (body.get("text") or "").strip()
                if run is None or not text:
                    return self._json({"error": "give tag and text"}, HTTPStatus.BAD_REQUEST)
                out = run.steer(text[:8000])
                return self._json(out, HTTPStatus.OK if out["ok"] else HTTPStatus.CONFLICT)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:                         # noqa: BLE001
                try:
                    self._json({"error": repr(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                except OSError:
                    pass

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    with open(PAGE, "rb") as f:
                        return self._send(f.read(), "text/html; charset=utf-8")
                if url.path == "/api/runs":
                    return self._json({"runs": mon.list(), "dir": mon.dir, "now": time.time()})
                run = mon.run(q.get("tag", ""))
                if run is None:
                    return self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
                if url.path == "/api/summary":
                    return self._json(run.summary())
                if url.path == "/api/events":
                    return self._json(run.events(int(q.get("since", 0) or 0)))
                if url.path == "/api/files":
                    return self._json(run.files())
                if url.path == "/api/file":
                    return self._json(run.file(q.get("path", "")))
                if url.path == "/api/views":
                    return self._json(run.views())
                if url.path == "/api/program":
                    return self._json(run.program(q.get("id", "")))
                if url.path == "/api/version":
                    return self._json({"text": run.version(q.get("id", ""), q.get("sha", ""))})
                if url.path == "/api/raw":                 # images (witness, pauses) and the timelapse video
                    p = run.inside(q.get("path", ""))
                    if not p or not os.path.isfile(p) or not p.endswith((".png", ".jpg", ".jpeg", ".mp4")):
                        return self._json({"error": "not an image or video in the run"}, HTTPStatus.NOT_FOUND)
                    return self._file(p)
                return self._json({"error": "unknown"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:                         # noqa: BLE001 -- one bad request must not stop the monitor
                try:
                    self._json({"error": repr(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                except OSError:
                    pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("ROBOT_HOME", "~/robot-cell"), "agent-runs"),
                    help="the directory of run directories (default: $ROBOT_HOME/agent-runs)")
    ap.add_argument("--host", default="127.0.0.1", help="listen address (localhost: reach it through ssh -L)")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    mon = Monitor(a.runs)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(mon))
    print(f"agent monitor: http://{a.host}:{a.port}/  (runs in {mon.dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
