"""The operator's requests (freeze, soft, release, park, health, events) and the dashboard serving them."""
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.core.controller import Fault
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks
from robotd.server import Driver, serve


def driver():
    cfg = tasks.get("empty").scene()
    return Driver(MujocoYamBackend(cfg), description())


def ticks(d, n):
    for _ in range(n):
        d.c.tick()


def slow_move(d):
    js = S.ARM_JOINTS["left"]
    q0 = np.array([d.c.m.q[d.c.jidx[n]] for n in js])
    return {"op": "command", "spec": {"op": "move_to", "joints": js, "target": (q0 + [0.6, 0, 0, 0, 0, 0]).tolist(),
                                      "duration": 8.0, "wait": False}}


def tcp(d, arm="left"):
    return d.c.T[arm][:3, 3].copy()


def push(d, arm, force):
    b = d.backend
    b.data.xfrc_applied[b.model.body(f"{arm}_gripper").id, :3] = force


def test_soft_stop_floats_and_yields_and_only_the_operator_releases_it():
    d = driver()
    cid = d.handle(slow_move(d))["id"]
    ticks(d, 50)
    writes = []
    real_write = d.backend.write
    d.backend.write = lambda q, dq, kp, kd, lim: (writes.append((np.array(kp), np.array(kd))), real_write(q, dq, kp, kd, lim))
    d.handle({"op": "operator_stop", "soft": True})
    res = d.cmds[cid].result
    assert res["status"] == "aborted" and res["reason"] == "operator_stop"
    start = tcp(d)
    ticks(d, 250)                                         # 1 s untouched: gravity compensation holds it up
    assert np.linalg.norm(tcp(d) - start) < 0.01
    kp, kd = writes[-1]
    assert np.all(kp[d.c.arm_idx] == 0) and np.allclose(kd[d.c.arm_idx], d.c.soft_kd[d.c.arm_idx])
    fingers = [d.c.jidx[S.FINGER_JOINTS[a][0]] for a in S.ARMS]
    assert np.all(kp[fingers] == 0) and np.all(kd[fingers] == 0)          # the grippers go limp too
    push(d, "left", [0, 0, 8.0])                          # a hand lifting the wrist: the arm gives way
    ticks(d, 125)
    push(d, "left", [0, 0, 0])
    ticks(d, 125)
    assert tcp(d)[2] - start[2] > 0.02                    # and stays where it was pushed
    with pytest.raises(Fault, match="operator"):
        d.handle(slow_move(d))
    with pytest.raises(Fault, match="only the operator"):
        d.handle({"op": "recover"})                       # a program cannot clear the operator's stop
    d.handle({"op": "release"})
    assert d.c.fault is None
    ticks(d, 5)
    kp, _ = writes[-1]
    assert np.allclose(kp[d.c.arm_idx], d.c.kp[d.c.arm_idx]) and kp[d.c.arm_idx].min() > 1.0


def test_freeze_holds_stiff_and_springs_back_after_a_push():
    d = driver()
    d.handle(slow_move(d))
    ticks(d, 50)
    d.handle({"op": "operator_stop", "soft": False})
    start = tcp(d)
    push(d, "left", [0, 0, 8.0])                          # the servo gives a little (soft wrist gains) ...
    ticks(d, 125)
    push(d, "left", [0, 0, 0])
    ticks(d, 250)
    assert np.linalg.norm(tcp(d) - start) < 0.005         # ... and springs back to where it froze
    assert d.c.fault["operator"] and not d.c.fault["soft"]
    with pytest.raises(Fault):
        d.handle(slow_move(d))
    d.handle({"op": "release"})
    d.handle(slow_move(d))                                # accepted again


def test_park_returns_to_the_start_pose():
    d = driver()
    home = d.home_q.copy()
    js = S.ARM_JOINTS["left"]
    q0 = np.array([home[d.c.jidx[n]] for n in js])
    d.handle({"op": "command", "spec": {"op": "move_to", "joints": js, "target": (q0 + [0.3, -0.2, 0.2, 0, 0, 0]).tolist()}})
    assert abs(d.c.m.q[d.c.jidx[js[0]]] - home[d.c.jidx[js[0]]]) > 0.2
    for cid in d.handle({"op": "park"})["ids"]:
        d.handle({"op": "wait", "id": cid})
    arm = [d.c.jidx[n] for a in d.c.arms for n in S.ARM_JOINTS[a]]
    assert np.max(np.abs(d.c.m.q[arm] - home[arm])) < 0.02


def test_park_latches_programs_out_until_the_operator_releases():
    """A program servoing an arm cannot take it back from the operator's park: its running command is stopped, every
    new one is refused (a servo loop's next step would otherwise replace the park at once), its own recover() does
    not clear the latch, and RELEASE does."""
    d = driver()
    ticks(d, 50)
    prog = slow_move(d)
    cid = d.handle(prog)["id"]
    ticks(d, 100)
    ids = d.handle({"op": "park"})["ids"]
    assert d.c.parked is not None
    assert d.cmds[cid].status == "stopped" and d.cmds[cid].result["reason"] == "operator park"
    with pytest.raises(Fault, match="parked"):
        d.handle(slow_move(d))
    with pytest.raises(Fault, match="parked"):
        d.handle({"op": "command", "spec": {"op": "gripper", "name": "left", "width": 0.05, "wait": False}})
    d.handle({"op": "recover"})                          # a program's recover() leaves the latch
    assert d.c.parked is not None
    ticks(d, 2500)
    assert all(d.cmds[i].result["status"] == "done" for i in ids)   # the park itself ran to the end
    assert d.handle({"op": "state"})["parked"]["reason"] == "the operator parked the arms"
    d.handle({"op": "release"})
    assert d.c.parked is None
    assert d.handle(slow_move(d))["id"] is not None      # programs may move the arms again


def test_a_servo_loop_collapses_into_two_log_entries():
    d = driver()
    js = S.ARM_JOINTS["left"]
    q0 = [float(d.c.m.q[d.c.jidx[n]]) for n in js]
    for k in range(20):                                   # command, finish, command, finish, ...
        d.handle({"op": "command", "spec": {"op": "move_to", "joints": js, "target": q0, "duration": 0.04}})
    ev = [e for e in d.handle({"op": "events", "since": 0})["events"] if e["kind"] in ("command", "finish")]
    assert len(ev) == 2 and {e["kind"] for e in ev} == {"command", "finish"} and all(e["count"] == 20 for e in ev)


def test_events_and_health():
    d = driver()
    d.handle(slow_move(d))
    d.handle({"op": "operator_stop", "soft": True})
    for _ in range(3):                                    # a program retrying while stopped: one collapsed entry
        with pytest.raises(Fault):
            d.handle(slow_move(d))
    d.handle({"op": "release"})
    ev = d.handle({"op": "events", "since": 0})
    kinds = [e["kind"] for e in ev["events"]]
    assert "command" in kinds and "fault" in kinds and "recover" in kinds and ev["seq"] == ev["events"][-1]["seq"]
    refused = [e for e in ev["events"] if e["kind"] == "refused"]
    assert len(refused) == 1 and refused[0]["count"] == 3 and "operator stop (soft)" in refused[0]["text"]
    assert d.handle({"op": "events", "since": ev["seq"]})["events"] == []
    h = d.handle({"op": "health"})
    assert h["backend"]["sim"] is True and h["ticks"] >= 0


# ------------------------------------------------------------------------------------------------ the dashboard
def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10) as r:
        return r.status, r.read()


def _post(url, headers=None):
    req = urllib.request.Request(url, data=b"", method="POST", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _until(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_dashboard_end_to_end(tmp_path):
    pytest.importorskip("PIL")
    from robotd.dashboard.server import Hub, serve as serve_http
    cfg = tasks.get("pick_cube").scene()
    backend = MujocoYamBackend(cfg, realtime=True)                # robotd --realtime
    addr = f"unix:{tmp_path}/robot.sock"
    ready = str(tmp_path / "ready")
    threading.Thread(target=serve, args=(Driver(backend, description()), addr, ready), daemon=True).start()
    _until(lambda: os.path.exists(ready))
    hub = Hub(addr, camera_hz=5.0)
    hub.start()
    httpd = serve_http(hub, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        live = _until(lambda: (lambda j: j if j.get("connected") and j.get("geoms") else None)(
            json.loads(_get(base + "/api/live")[1])))
        model = json.loads(_get(base + "/api/model")[1])["model"]
        assert len(live["geoms"]) == 12 * len(model["geoms"]) and model["meshes"]
        assert len(_get(base + "/api/mesh.bin")[1]) > 100000
        assert set(live["diagram"]) == {"left", "right"} and len(live["diagram"]["left"]) == 8
        assert b"FREEZE" in _get(base + "/")[1]
        _until(lambda: json.loads(_get(base + "/api/live")[1])["cloud_seq"] > 0)
        blob = _get(base + "/api/cloud.bin")[1]
        n = struct.unpack("<I", blob[:4])[0]
        head = json.loads(blob[4:4 + n])
        assert sum(c[1] for c in head["counts"]) > 1000
        assert _get(base + "/api/camera/top.jpg")[1][:2] == b"\xff\xd8"
        assert _post(base + "/api/action/freeze")[0] == 403          # no X-Dashboard header: refused
        assert _post(base + "/api/action/freeze", {"X-Dashboard": "1", "Host": "evil.example"})[0] == 421
        code, res = _post(base + "/api/action/soft", {"X-Dashboard": "1"})
        assert code == 200 and res["ok"] and res["result"]["soft"]
        _until(lambda: (json.loads(_get(base + "/api/live")[1])["fault"] or {}).get("soft"))
        code, res = _post(base + "/api/action/release", {"X-Dashboard": "1"})
        assert res["ok"]
        code, res = _post(base + "/api/action/zero", {"X-Dashboard": "1"})       # every arm's force estimate
        assert code == 200 and res["ok"] and len(res["result"]) == 2
        _until(lambda: json.loads(_get(base + "/api/live")[1])["fault"] is None)
        log = _until(lambda: (lambda ev: ev if any(e["kind"] == "fault" for e in ev) else None)(
            json.loads(_get(base + "/api/events?since=0")[1])["events"]))     # the log trails the state by <0.25 s
        assert any(e["kind"] == "action" for e in log)
        code, res = _post(base + "/api/action/off", {"X-Dashboard": "1"})        # park, motors off, the driver exits
        assert code == 200 and res["ok"]
        _until(lambda: not json.loads(_get(base + "/api/live")[1])["connected"])
    finally:
        hub.stop_event.set()
        httpd.shutdown()
