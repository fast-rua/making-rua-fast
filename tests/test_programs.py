"""The dashboard's programs: files, one run at a time as its own process against the driver, stop, operator stops."""
import json
import os
import threading
import time
import urllib.request

import pytest

from robotd.dashboard.programs import ProgramError, ProgramManager


def until(pred, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


def pm(tmp_path, addr="unix:/nonexistent"):
    ex = tmp_path / "examples"
    ex.mkdir()
    (ex / "wave.py").write_text('"""Wave hello."""\nprint("hello from the example")\n')
    return ProgramManager(str(tmp_path / "mine"), str(ex), addr)


def test_save_run_output_exit_code(tmp_path):
    p = pm(tmp_path)
    p.save("hello.py", 'import sys\nprint("out line")\nprint("err line", file=sys.stderr)\nsys.exit(3)\n')
    names = {(x["name"], x["origin"]) for x in p.list()}
    assert ("hello.py", "mine") in names and ("wave.py", "example") in names
    p.run("hello.py", args="--a 1")
    last = until(lambda: p.status()["last"])
    assert last["code"] == 3 and not last["stopped_by"]
    lines = p.output()["lines"]
    assert {"out line", "err line"} <= {l["text"] for l in lines}
    assert any(l["stream"] == "err" and l["text"] == "err line" for l in lines)


def test_stop_one_at_a_time_and_names(tmp_path):
    p = pm(tmp_path)
    p.save("loop.py", 'import time\nwhile True:\n    print("tick", flush=True)\n    time.sleep(0.05)\n')
    p.run("loop.py")
    until(lambda: any(l["text"] == "tick" for l in p.output()["lines"]))
    with pytest.raises(ProgramError, match="one program at a time"):
        p.run("loop.py")
    with pytest.raises(ProgramError, match="running"):
        p.delete("loop.py")
    p.stop("test")
    last = until(lambda: p.status()["last"], 10)
    assert last["stopped_by"] == "test" and p.status()["running"] is None
    for bad in ("../x.py", "a/b.py", "x.txt", ".py", "x y.py"):
        with pytest.raises(ProgramError):
            p.save(bad, "print(1)")
    with pytest.raises(ProgramError):
        p.source("wave.py", "nowhere")
    assert p.copy("wave.py", "example")["name"] == "wave.py"            # an example becomes yours
    assert "hello from the example" in p.source("wave.py", "mine")
    assert p.save("broken.py", "def f(:\n")["syntax"]                   # saved, with the syntax problem reported
    p.save("fresh.py")                                                  # a new one from the template
    assert "robot.connect" in p.source("fresh.py")


def test_programs_drive_the_robot_and_the_operator_stop_stops_them(tmp_path):
    pytest.importorskip("PIL")
    from robotd.backends.mujoco_sim import MujocoYamBackend
    from robotd.dashboard.server import Hub, serve as serve_http
    from robotd.robots.yam_station.description import description
    import tasks
    from robotd.server import Driver, serve
    cfg = tasks.get("empty").scene()
    b = MujocoYamBackend(cfg, realtime=True)
    addr, ready = f"unix:{tmp_path}/robot.sock", str(tmp_path / "ready")
    threading.Thread(target=serve, args=(Driver(b, description()), addr, ready), daemon=True).start()
    until(lambda: os.path.exists(ready))
    hub = Hub(addr, camera_hz=1.0, programs_dir=str(tmp_path / "mine"), examples_dir=None)
    hub.start()
    httpd = serve_http(hub, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(verb, body):
        req = urllib.request.Request(f"{base}/api/programs/{verb}", data=json.dumps(body).encode(), method="POST",
                                     headers={"X-Dashboard": "1", "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=10))

    def output():
        return [l["text"] for l in json.load(urllib.request.urlopen(f"{base}/api/programs/output?since=0"))["lines"]]
    try:
        until(lambda: hub.connected)
        assert post("save", {"name": "where.py", "source": 'import robot\nr = robot.connect(name="where")\n'
                                                          'print("t =", round(r.state().t, 3))\n'})["ok"]
        assert post("run", {"name": "where.py"})["ok"]
        until(lambda: any(t.startswith("t =") for t in output()))
        assert post("save", {"name": "sway.py", "source":
                    'import robot, numpy as np\nr = robot.connect(name="sway")\njs = r.description["groups"]["left_arm"]\n'
                    'q0 = r.state().q(js)\nwhile True:\n'
                    '    for d in (0.05, 0.0):\n        r.move(q0 + [d, 0, 0, 0, 0, 0], joints=js, speed=0.3)\n'})["ok"]
        until(lambda: hub.programs.status()["running"] is None)
        assert post("run", {"name": "sway.py"})["ok"]
        until(lambda: any(e["text"].startswith("move_to") for e in hub.log_since(0)["events"]))
        hub.action("freeze")                                     # the operator's stop stops the program too
        last = until(lambda: hub.programs.status()["last"] if hub.programs.status()["last"]["name"] == "sway.py" else None)
        assert last["stopped_by"] == "operator FREEZE"
        until(lambda: json.load(urllib.request.urlopen(f"{base}/api/live"))["program"]["running"] is None,
              timeout=2.0)                                       # the page's view lags by up to one publish (50 ms)
    finally:
        hub.stop_event.set()
        if hub.programs:
            hub.programs.stop("test over")
        httpd.shutdown()
