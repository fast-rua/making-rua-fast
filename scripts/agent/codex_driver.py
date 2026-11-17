"""Drive Codex through its app-server (JSON-RPC over stdio, one message per line) instead of `codex exec`, so an
operator can steer it while it works. Used by scripts/agent_run_real.py: the sandbox runs `codex app-server --stdio`,
and this side, outside it, speaks the protocol over the sandbox's stdin and stdout.

- start(): initialize, start a thread in the workspace and a first turn with the prompt.
- The app-server's notifications become `codex exec --json` events (what scripts/agent_monitor.py reads) in
  events.jsonl; every raw message goes to app-server.jsonl.
- deliver(text): the operator's message. Into the running turn (turn/steer: the agent reads it at its next step and
  keeps its context), or as a new turn when it has finished its last one. Messages arrive through an inbox file,
  RUN/steer/inbox.jsonl, one JSON object ({"text": ...}) per line, which the monitor's steer box appends to.
- finished(): a turn has completed and no message came within linger_s, the end of an episode (as `codex exec`
  ends when its turn does).
- continue_thread(thread_id): instead of start(), go on from an earlier thread (scripts/agent_debrief.py): a fork of
  it (the earlier thread's record stays as it was) or the thread itself, with no turn until a message comes.

Codex compacts the thread's context by itself when it nears the model's limit (auto_compact_token_limit); a
compaction shows as a context_compaction item. Standard library only.

Image pruning (maybe_prune(), the launcher calls it every tick): every image the agent has viewed stays in the thread
and slows every later request (measured 2026-11-07: a one-word reply took 3.4 s on towel-fold-11's 173k-token thread,
1.4 s with all but the last four images removed). So, after prune_images new images, at a quiet moment (no command of
the agent's running), the driver interrupts the turn, rebuilds the thread's model-visible history from its rollout with
all but the newest prune_keep images replaced by a short note naming the file, starts a new thread, loads that history
into it (thread/inject_items, an app-server call: no Codex file is changed and the sandbox keeps running), and starts
a turn there telling the agent to go on. The same happens at once when the app-server falls back from WebSockets to
HTTPS, which re-sends the whole history, images included, with every request (towel-fold-11's last 32 steps took a
median 12.5 s that way).

    python codex_driver.py --test CODEX_HOME WORKDIR      # a two-turn check without any sandbox or robot
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

WS_FALLBACK = "Falling back from WebSockets to HTTPS"
PRUNE_NOTE = ("[harness] This conversation was moved to a fresh thread so that requests stay fast: older images were "
              "replaced by notes naming their files (view a file again if you need it); everything else is as it was. "
              "Go on with the task from where you were.")
VIEW_PATH = re.compile(r"""view_image\(\s*\{[^}]*?path\s*:\s*["']([^"']+)["']""")


def _opening_context(item: dict) -> bool:
    """An item the new thread writes again by itself: tool declarations, developer instructions, AGENTS.md, the
    environment context."""
    if item.get("type") == "additional_tools":
        return True
    if item.get("type") != "message":
        return False
    if item.get("role") == "developer":
        return True
    content = item.get("content")
    text = " ".join(c.get("text", "") for c in content if isinstance(c, dict)) if isinstance(content, list) \
        else str(content or "")
    return item.get("role") == "user" and text.lstrip().startswith(
        ("# AGENTS.md instructions", "<environment_context>", "<user_instructions>"))


def model_history(rollout_path: str) -> list:
    """The thread's model-visible history from its rollout, without the opening context: after the last compaction,
    its replacement history and the items since; else every item after the opening context."""
    lines = [json.loads(l) for l in open(rollout_path) if l.strip()]
    comp = [i for i, e in enumerate(lines) if e.get("type") == "compacted"]
    if comp:
        head = (lines[comp[-1]].get("payload") or {}).get("replacement_history") or []
        tail = [e["payload"] for e in lines[comp[-1] + 1:] if e.get("type") == "response_item"]
        items = [x for x in head if not _opening_context(x)] + tail
    else:
        items = [e["payload"] for e in lines if e.get("type") == "response_item"]
        k = 0
        while k < len(items) and _opening_context(items[k]):
            k += 1
        items = items[k:]
    return items


def prune_images(items: list, keep: int, keep_monitor: int = 1) -> tuple:
    """Replace all but the newest `keep` images (and the newest `keep_monitor` monitor images, files named
    monitor*) in tool outputs and in messages (a pushed action report's images) by a text note naming the file when
    the call that showed it (or the message's `[view] NAME: PATH` lines) names one. Returns (items, removed)."""
    paths = {}                                           # call id -> the image paths its input names, in order
    for x in items:
        if x.get("type") in ("custom_tool_call", "function_call"):
            src = x.get("input") or x.get("arguments") or ""
            paths[x.get("call_id")] = VIEW_PATH.findall(src) or re.findall(r'"path"\s*:\s*"([^"]+)"', src)
    budget = {"other": keep, "monitor": keep_monitor}
    removed = 0
    out = [None] * len(items)
    for i in range(len(items) - 1, -1, -1):              # newest first, so the newest images keep their places
        x = items[i]
        if x.get("type") == "message" and x.get("role") == "user" and isinstance(x.get("content"), list):
            key = "content"                              # a message with images (a pushed action report)
            text = " ".join(c.get("text", "") for c in x["content"] if isinstance(c, dict))
            names = re.findall(r"\[view\] [^:\n]+: (\S+)", text)
        elif x.get("type") in ("custom_tool_call_output", "function_call_output") and isinstance(x.get("output"), list):
            key = "output"
            names = paths.get(x.get("call_id")) or []
        else:
            out[i] = x
            continue
        imgs = [j for j, c in enumerate(x[key]) if isinstance(c, dict) and c.get("type") == "input_image"]
        if not imgs:
            out[i] = x
            continue
        new = list(x[key])
        for n, j in reversed(list(enumerate(imgs))):
            path = names[n] if n < len(names) else None
            kind = "monitor" if path and os.path.basename(path).startswith("monitor") else "other"
            if budget[kind] > 0:
                budget[kind] -= 1
                continue
            new[j] = {"type": "input_text", "text": f"[image removed from the history: {path}]" if path
                      else "[image removed from the history]"}
            removed += 1
        out[i] = dict(x, **{key: new})
    return out, removed


def find_rollout(codex_home: str, thread_id: str) -> str | None:
    found = glob.glob(os.path.join(codex_home, "sessions", "**", f"rollout-*{thread_id}.jsonl"), recursive=True)
    return max(found, key=os.path.getmtime) if found else None


def convert(item: dict) -> dict:
    """An app-server ThreadItem as `codex exec --json` writes its item (the monitor's format)."""
    t, out = item.get("type"), {"id": item.get("id")}
    if t == "agentMessage":
        out.update(type="agent_message", text=item.get("text", ""))
    elif t == "reasoning":
        text = "\n".join(item.get("summary") or []) or "\n".join(item.get("content") or [])
        out.update(type="reasoning", text=text)
    elif t == "commandExecution":
        out.update(type="command_execution", command=item.get("command"), aggregated_output=item.get("aggregatedOutput"),
                   exit_code=item.get("exitCode"), status=item.get("status"), duration_ms=item.get("durationMs"))
    elif t == "fileChange":
        out.update(type="file_change", status=item.get("status"),
                   changes=[{"path": c.get("path"), "kind": (c.get("kind") or {}).get("type") if isinstance(c.get("kind"), dict)
                             else c.get("kind")} for c in item.get("changes") or []])
    elif t == "userMessage":
        parts = [c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
        out.update(type="user_message", text="\n".join(parts))
    elif t == "plan":
        out.update(type="agent_message", text="Plan:\n" + (item.get("text") or ""))
    elif t == "contextCompaction":
        out.update(type="context_compaction")
    else:
        out.update(type=t or "unknown", raw=item)
    return out


class CodexDriver:
    def __init__(self, proc: subprocess.Popen, events_path: str, raw_path: str, inbox_path: str, prompt: str,
                 cwd: str, linger_s: float = 30.0, codex_home: str | None = None, prune_after: int = 0,
                 prune_keep: int = 4):
        self.p, self.prompt, self.cwd, self.linger_s = proc, prompt, cwd, linger_s
        self.events = open(events_path, "a")
        self.raw = open(raw_path, "a")
        self.inbox_path, self.inbox_pos = inbox_path, 0
        self.lock, self.next_id, self.pending = threading.Lock(), 1, {}
        self.thread_id = self.turn_id = None
        self.idle_since = None                  # when the last turn completed (None while one runs)
        self.closed = threading.Event()         # the app-server's stdout ended
        self.prev_total = None
        self.stats = {"thread_id": None, "turns": 0, "usage": {}, "errors": [], "operator_messages": [],
                      "compactions": 0, "events": 0, "threads": [], "prunes": []}
        self.on_event = None                    # a callable(event dict), e.g. to echo the agent's words
        # image pruning (module docstring): where the rollouts are, after how many new images, how many to keep
        self.codex_home, self.prune_after, self.prune_keep = codex_home, prune_after, prune_keep
        self.running = set()                    # the agent's commands that have started and not ended
        self.images_since = 0                   # images viewed since the thread started or was last pruned
        self.ws_fallback = False                # the app-server fell back to HTTPS: prune at the next quiet moment
        self.pruning = False
        self.turn_ended = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    # ---- the wire
    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        with self.lock:
            self.p.stdin.write(line.encode())
            self.p.stdin.flush()
            self.raw.write(json.dumps({"wall": round(time.time(), 3), "out": msg}) + "\n")
            self.raw.flush()

    def request(self, method: str, params: dict, timeout: float = 120.0):
        with self.lock:
            i = self.next_id
            self.next_id += 1
            slot = self.pending[i] = {"event": threading.Event()}
        self._send({"id": i, "method": method, "params": params})
        if not slot["event"].wait(timeout):
            raise RuntimeError(f"{method}: no reply in {timeout:.0f} s")
        msg = self.pending.pop(i)["msg"]
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result") or {}

    def _read(self):
        for line in self.p.stdout:
            text = line.decode("utf-8", "replace")
            self.raw.write(json.dumps({"wall": round(time.time(), 3), "in": text.rstrip("\n")[:200000]}) + "\n")
            self.raw.flush()
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if "method" not in msg and "id" in msg:                     # a reply
                slot = self.pending.get(msg["id"])
                if slot is not None:
                    slot["msg"] = msg
                    slot["event"].set()
            elif "method" in msg and "id" in msg:                       # a request of the server's: none is expected
                self._send({"id": msg["id"], "error": {"code": -32601, "message": "not available in this harness "
                                                       "(approvals are off; nobody answers questions)"}})
            elif "method" in msg:
                try:
                    self._notification(msg["method"], msg.get("params") or {})
                except Exception as e:                                  # noqa: BLE001 -- keep reading
                    self.stats["errors"].append(f"notification {msg['method']}: {e!r}")
        self.closed.set()

    def _emit(self, obj: dict):
        ev = {"wall": round(time.time(), 3), **obj}             # when this side saw it
        self.events.write(json.dumps(ev) + "\n")
        self.events.flush()
        self.stats["events"] += 1
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:                                   # noqa: BLE001 -- an echo must not stop the driver
                pass

    # ---- what the app-server says
    def _notification(self, m: str, p: dict):
        tid = p.get("threadId") or (p.get("thread") or {}).get("id")
        if m == "warning" and WS_FALLBACK in str(p.get("message") or ""):
            self.ws_fallback = True                             # prune (and so re-send far less) at once
            self._emit({"type": "warning", "message": p.get("message"), "prune": "next quiet moment"})
            return
        if tid and self.thread_id and tid != self.thread_id and m != "thread/started":
            if m == "turn/completed":                           # the thread a prune left: its interrupted turn
                self.turn_ended.set()
            return                                              # (the raw log keeps all of it)
        if m == "thread/started":
            self.stats["thread_id"] = (p.get("thread") or {}).get("id") or p.get("threadId")
            self._emit({"type": "thread.started", "thread_id": self.stats["thread_id"]})
        elif m == "turn/started":
            self.turn_id, self.idle_since = (p.get("turn") or {}).get("id"), None
            self.turn_ended.clear()
            self._emit({"type": "turn.started", "turn_id": self.turn_id})
        elif m in ("item/started", "item/completed"):
            raw = p.get("item") or {}
            if raw.get("type") == "commandExecution":
                (self.running.add if m == "item/started" else self.running.discard)(raw.get("id"))
            if raw.get("type") == "imageView" and m == "item/completed":
                self.images_since += 1
            item = convert(raw)
            if item.get("type") == "context_compaction" and m == "item/completed":
                self.stats["compactions"] += 1
            self._emit({"type": "item." + m.split("/")[1], "item": item})
        elif m == "thread/tokenUsage/updated":
            self.usage = (p.get("tokenUsage") or {}).get("total") or {}
        elif m == "turn/completed":
            turn = p.get("turn") or {}
            total = getattr(self, "usage", {}) or {}
            prev = self.prev_total or {}
            usage = {"input_tokens": total.get("inputTokens", 0) - prev.get("inputTokens", 0),
                     "cached_input_tokens": total.get("cachedInputTokens", 0) - prev.get("cachedInputTokens", 0),
                     "output_tokens": total.get("outputTokens", 0) - prev.get("outputTokens", 0)}
            self.prev_total = dict(total)
            self.stats["turns"] += 1
            for k, v in usage.items():
                self.stats["usage"][k] = self.stats["usage"].get(k, 0) + v
            if turn.get("status") == "failed":
                err = turn.get("error") or {}
                self.stats["errors"].append(err.get("message"))
                self._emit({"type": "turn.failed", "error": err})
            else:
                self._emit({"type": "turn.completed", "usage": usage, "status": turn.get("status")})
            self.turn_id, self.idle_since = None, time.time()
            self.running.clear()
            self.turn_ended.set()
        elif m == "error":
            err = p.get("error") or {}
            if not p.get("willRetry"):
                self.stats["errors"].append(err.get("message"))
            self._emit({"type": "error", "message": err.get("message"), "will_retry": p.get("willRetry")})
        elif m == "thread/compacted":
            self._emit({"type": "item.completed", "item": {"id": f"compacted-{time.time():.0f}",
                                                           "type": "context_compaction"}})

    # ---- what this side does
    def start(self):
        self.request("initialize", {"clientInfo": {"name": "agent_run_real", "title": "robot-api episodes",
                                                   "version": "1"}})
        self._send({"method": "initialized"})
        r = self.request("thread/start", {"cwd": self.cwd})
        self.thread_id = (r.get("thread") or {}).get("id") or r.get("threadId")
        if not self.thread_id:
            raise RuntimeError(f"thread/start gave no thread id: {r}")
        self.stats["thread_id"] = self.thread_id
        self.stats["threads"].append(self.thread_id)
        self.request("turn/start", {"threadId": self.thread_id, "input": [{"type": "text", "text": self.prompt}]})

    def continue_thread(self, thread_id: str, fork: bool = True, developer_instructions: str | None = None,
                        effort: str | None = None) -> str:
        """Go on from an earlier thread: thread/fork (a new thread with its history; the earlier one's rollout is not
        touched) or thread/resume (the thread itself). No turn starts until deliver(). Returns the thread id."""
        self.request("initialize", {"clientInfo": {"name": "agent_debrief", "title": "robot-api debriefs",
                                                   "version": "1"}})
        self._send({"method": "initialized"})
        params = {"threadId": thread_id, "cwd": self.cwd}
        if developer_instructions:
            params["developerInstructions"] = developer_instructions
        if effort:
            params["config"] = {"model_reasoning_effort": effort}
        r = self.request("thread/fork" if fork else "thread/resume", params, timeout=300)
        self.thread_id = (r.get("thread") or {}).get("id") or r.get("threadId") or (None if fork else thread_id)
        if not self.thread_id:
            raise RuntimeError(f"thread/{'fork' if fork else 'resume'} gave no thread id: {r}")
        self.stats["thread_id"] = self.thread_id
        self.stats["threads"].append(self.thread_id)
        return self.thread_id

    def deliver(self, text: str, source: str = "operator", images: list | None = None, detail: str = "auto") -> str:
        """A message for the agent (the operator's, or source="program": the launcher's word that the agent's program
        paused or ended; "action": a kernel action's end report): into the running turn, else a new turn. images:
        files (paths the app-server can read) sent with it, at that detail (auto, high or original: Codex 0.165 drops
        an image sent at low, leaving a note). Returns how it went."""
        inp = [{"type": "text", "text": text}] + [{"type": "localImage", "path": p, "detail": detail}
                                                  for p in images or ()]
        how = None
        if self.turn_id:
            try:
                self.request("turn/steer", {"threadId": self.thread_id, "expectedTurnId": self.turn_id, "input": inp})
                how = "steer"
            except RuntimeError as e:                                   # the turn ended meanwhile
                how = f"steer failed ({e}); "
        if how != "steer":
            r = self.request("turn/start", {"threadId": self.thread_id, "input": inp})
            self.idle_since = None                                      # a turn is starting: not idle (finished()
            self.turn_id = self.turn_id or (r.get("turn") or {}).get("id")   # must not end the episode before
            how = (how or "") + "new turn"                              # its turn/started arrives)
        entry = {"wall": round(time.time(), 3), "text": text, "delivered_as": how}
        if images:
            entry["images"] = list(images)
            self.images_since += len(images)
        if source == "operator":
            self.stats["operator_messages"].append(entry)
        else:
            self.stats.setdefault("program_events", []).append(entry)
        self._emit({"type": f"{source}.message", **entry})
        return how

    def poll_inbox(self):
        """Deliver the inbox's new lines."""
        try:
            with open(self.inbox_path, "rb") as f:
                f.seek(self.inbox_pos)
                data = f.read()
        except OSError:
            return
        while b"\n" in data:
            line, data = data.split(b"\n", 1)
            self.inbox_pos += len(line) + 1
            try:
                text = (json.loads(line) or {}).get("text", "").strip()
            except ValueError:
                text = line.decode("utf-8", "replace").strip()
            if text:
                try:
                    self.deliver(text)
                except Exception as e:                                  # noqa: BLE001
                    self.stats["errors"].append(f"delivering an operator message: {e!r}")
                    self._emit({"type": "error", "message": f"could not deliver the operator's message: {e!r}"})

    def finished(self) -> bool:
        return (self.idle_since is not None and time.time() - self.idle_since > self.linger_s
                and not self.pruning)

    # ---- image pruning (module docstring)
    def maybe_prune(self) -> bool:
        """Prune when it is due and the moment is quiet: a turn running, none of the agent's commands running (an
        interrupt would end them), no prune under way. Returns whether one happened."""
        due = self.ws_fallback or (self.prune_after > 0 and self.images_since >= self.prune_after)
        if not due or self.pruning or not self.turn_id or self.running or not self.codex_home:
            return False
        return self.prune()

    def prune(self) -> bool:
        """Interrupt the turn, rebuild the history with older images removed, carry on in a new thread."""
        self.pruning = True
        old, t0 = self.thread_id, time.time()
        rec = {"wall": round(t0, 3), "from_thread": old, "images_since": self.images_since,
               "reason": "websocket fallback" if self.ws_fallback else "images"}
        try:
            self.turn_ended.clear()
            self.request("turn/interrupt", {"threadId": old, "turnId": self.turn_id}, timeout=30)
            self.turn_ended.wait(30)
            path = find_rollout(self.codex_home, old)
            if not path:
                raise RuntimeError(f"no rollout for thread {old} under {self.codex_home}")
            time.sleep(0.5)                             # the interrupted turn's last lines reach the file
            items = model_history(path)
            answered = {x.get("call_id") for x in items if x.get("type") in ("custom_tool_call_output",
                                                                               "function_call_output")}
            items = [x for x in items if x.get("type") not in ("custom_tool_call", "function_call")
                     or x.get("call_id") in answered]   # a call the interrupt cut off has no output: leave it out
            items, removed = prune_images(items, self.prune_keep)
            r = self.request("thread/start", {"cwd": self.cwd})
            new = (r.get("thread") or {}).get("id") or r.get("threadId")
            if not new:
                raise RuntimeError(f"thread/start gave no thread id: {r}")
            self.request("thread/inject_items", {"threadId": new, "items": items}, timeout=120)
            self.thread_id = self.stats["thread_id"] = new
            self.stats["threads"].append(new)
            self.prev_total = None                      # the new thread's token totals start from zero
            try:
                self.request("thread/unsubscribe", {"threadId": old}, timeout=10)
            except RuntimeError:
                pass
            self.images_since, self.ws_fallback = 0, False
            self.request("turn/start", {"threadId": new, "input": [{"type": "text", "text": PRUNE_NOTE}]})
            rec.update(to_thread=new, items=len(items), images_removed=removed, bytes=len(json.dumps(items)),
                       took_s=round(time.time() - t0, 2))
            self._emit({"type": "harness.prune", **rec})
            return True
        except Exception as e:                          # noqa: BLE001 -- go on in the old thread
            rec.update(error=repr(e)[:500], took_s=round(time.time() - t0, 2))
            self.stats["errors"].append(f"prune: {e!r}")
            self._emit({"type": "harness.prune", **rec})
            self.prune_after = 0 if "inject" in repr(e) else self.prune_after     # do not try it again and again
            self.ws_fallback = False
            if self.thread_id == old and not self.turn_id:
                try:
                    self.request("turn/start", {"threadId": old, "input": [{"type": "text", "text":
                                 "[harness] Your turn was interrupted by the harness by mistake. Go on with the task "
                                 "from where you were."}]})
                except Exception:                       # noqa: BLE001
                    pass
            return False
        finally:
            self.stats["prunes"].append(rec)
            self.pruning = False

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        self.events.close()
        self.raw.close()


def _test(codex_home: str, workdir: str):
    """Two short turns and a steer, straight against `codex app-server --stdio` (no sandbox, no robot)."""
    home = os.path.expanduser(os.environ.get("ROBOT_HOME", "~/robot-cell"))
    exe = os.path.realpath(os.path.join(home, "codex", "current", "bin", "codex"))
    os.makedirs(workdir, exist_ok=True)
    p = subprocess.Popen([exe, "app-server", "--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(workdir, "stderr.log"), "w"), cwd=workdir,
                         env=dict(os.environ, CODEX_HOME=codex_home))
    inbox = os.path.join(workdir, "inbox.jsonl")
    open(inbox, "w").close()
    d = CodexDriver(p, os.path.join(workdir, "events.jsonl"), os.path.join(workdir, "app-server.jsonl"), inbox,
                    "Run the shell command `sleep 20; echo first-step-done` and report its output in one line.",
                    workdir, linger_s=5)
    d.start()
    t0, sent = time.time(), False
    while not d.finished() and time.time() - t0 < 240 and not d.closed.is_set():
        if not sent and d.turn_id and time.time() - t0 > 8:          # steer while the first command runs
            with open(inbox, "a") as f:
                f.write(json.dumps({"text": "Also run `echo steered-ok` and include its output."}) + "\n")
            sent = True
        d.poll_inbox()
        time.sleep(0.5)
    d.close()
    p.wait(20)
    print(json.dumps(d.stats, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--test":
        _test(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
