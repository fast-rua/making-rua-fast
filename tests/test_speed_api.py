"""The pieces that save the agent round trips and context (2026-11-07): the [views] and [state] lines a program prints
as it exits, viz.grid / show_cameras (one image instead of four), viz.marks (numbered points with their world
positions), and the harness's image pruning (scripts/agent/codex_driver.py: the history rebuilt from a rollout, older
images replaced by notes)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent"))
pytest.importorskip("cv2")


class _Frame:
    """A camera 0.8 m above a flat table at z=0, looking straight down (as robot.Frame gives it)."""

    def __init__(self, H=240, W=320, f=300.0, height=0.8):
        self.rgb = np.full((H, W, 3), 90, np.uint8)
        self.depth = np.full((H, W), height, np.float32)
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])            # optical +z (forward) points down, -z in the world
        T[2, 3] = height
        self.T = T


def test_grid_tiles_and_labels():
    from robot import viz
    ims = [np.full((480, 640, 3), c, np.uint8) for c in (10, 60, 120, 200)]
    g = viz.grid(ims, labels=["top", "front", "left_wrist", "right_wrist"], width=640)
    assert g.shape == (480, 640, 3)                       # 2 x 2 cells of 320 x 240
    assert g[200, 100].tolist() == [10, 10, 10] and g[440, 600].tolist() == [200, 200, 200]
    g1 = viz.grid(ims[:1], width=500)
    assert g1.shape[1] == 500


def test_marks_give_world_points_on_the_table(tmp_path, monkeypatch, capsys):
    from robot import viz
    monkeypatch.chdir(tmp_path)
    fr = _Frame()
    mask = np.zeros(fr.depth.shape, bool)
    mask[60:180, 80:240] = True                           # a 160 x 120 px patch: 0.43 x 0.32 m at 0.8 m
    pts = viz.marks(fr, "patch", mask=mask, step=40, r=False)
    assert len(pts) >= 4
    P = np.array(list(pts.values()))
    assert np.allclose(P[:, 2], 0.0, atol=1e-6)           # on the table
    first4 = np.array([pts[k] for k in (1, 2, 3, 4)])     # the outline's corners come first
    assert np.isclose(np.abs(first4[:, 0]).max(), 80 * 0.8 / 300, atol=0.01)
    out = capsys.readouterr().out
    assert "[view] patch:" in out and "[marks patch] 1:(" in out
    assert os.path.exists(tmp_path / ".robot" / "views" / "patch.png")
    few = viz.marks(fr, mask=mask, step=8, max_marks=10)  # thinned to max_marks, corners kept
    assert len(few) == 10


def test_a_program_prints_its_views_as_it_exits(tmp_path):
    code = ("import numpy as np\nfrom robot import viz\n"
            "viz.show('a', np.zeros((20, 30, 3), np.uint8), r=False)\n"
            "viz.show('b', np.zeros((20, 30), bool), r=False)\n"
            "viz.show('quiet', np.zeros((20, 30), bool), r=False, quiet=True)\n"
            "viz.show('a', np.ones((20, 30, 3), np.uint8), r=False)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))))
    assert out.returncode == 0, out.stderr
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("[views] ")
    paths = json.loads(last[len("[views] "):])
    assert [os.path.basename(p) for p in paths] == ["a.png", "b.png"]   # once each, in order; quiet ones left out


def test_state_summary_is_one_line():
    import robot
    T = np.eye(4)
    T[:3, 3] = [0.4, 0.2, 0.05]
    ee = {"T": T.tolist(), "v": [0] * 6, "wrench": [0] * 6, "deflection": [0] * 6, "contact": True}
    st = robot.State({"t": 1.0, "joints": ["a"], "q": [0], "dq": [0], "effort": [0], "effort_ext": [0],
                      "q_ref": [0], "ee": {"left": ee, "right": dict(ee, contact=False)},
                      "grippers": {"left": {"width": 0.0043, "stalled": 1.0}, "right": {"width": 0.065}},
                      "fault": None})
    s = st.summary()
    assert "\n" not in s
    assert "left xyz=[0.400, 0.200, 0.050] grip=4.3mm (stalled: holding something);" in s     # no untrusted contact
    assert "right xyz=[0.400, 0.200, 0.050] grip=65.0mm" in s and "FAULT" not in s and "contact" not in s
    sim = robot.State(dict(st.raw, arm_force_trusted=True))                   # simulation: its force estimate holds
    assert "grip=4.3mm (stalled: holding something) contact" in sim.summary()
    empty = robot.State(dict(st.raw, grippers={"left": {"width": 0.0029, "stalled": 0.0, "empty": True}}))
    assert "left xyz=[0.400, 0.200, 0.050] grip=2.9mm (closed on nothing)" in empty.summary()


# ------------------------------------------------------------------------------------------------ pruning
def _rollout(tmp_path, lines):
    p = tmp_path / "rollout-x-thread1.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return str(p)


def _img(n):
    return {"type": "input_image", "image_url": f"data:image/png;base64,{'A' * 10}{n}"}


def _call(cid, paths):
    src = " ".join(f'image((await tools.view_image({{path:"{p}"}})).image_url);' for p in paths)
    return {"type": "custom_tool_call", "call_id": cid, "name": "exec", "input": src}


def _out(cid, n_images, start):
    return {"type": "custom_tool_call_output", "call_id": cid,
            "output": [{"type": "input_text", "text": "Script completed"}] + [_img(start + i) for i in range(n_images)]}


def test_model_history_skips_the_opening_context_and_follows_compaction(tmp_path):
    from codex_driver import model_history
    opening = [{"type": "additional_tools", "role": "developer"},
               {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex"}]},
               {"type": "message", "role": "user", "content": [{"type": "input_text",
                                                                  "text": "# AGENTS.md instructions for /workspace"}]},
               {"type": "message", "role": "user", "content": [{"type": "input_text",
                                                                  "text": "<environment_context>..."}]}]
    task = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Do the task."}]}
    work = [_call("c1", ["/w/a.png"]), _out("c1", 1, 0)]
    lines = [{"type": "session_meta", "payload": {}}] + [{"type": "response_item", "payload": x}
                                                         for x in opening + [task] + work]
    items = model_history(_rollout(tmp_path, lines))
    assert items[0] == task and items[1:] == work
    comp = {"type": "compacted", "payload": {"replacement_history": opening + [task, {"type": "compaction",
                                                                                   "encrypted_content": "x"}]}}
    after = [_call("c2", ["/w/b.png"]), _out("c2", 1, 1)]
    lines2 = lines + [comp] + [{"type": "response_item", "payload": x} for x in after]
    items2 = model_history(_rollout(tmp_path, lines2))
    assert [x["type"] for x in items2] == ["message", "compaction", "custom_tool_call", "custom_tool_call_output"]


def test_prune_keeps_the_newest_images_and_names_the_files_of_the_rest():
    from codex_driver import prune_images
    items = [_call("c1", ["/w/a.png", "/w/b.png"]), _out("c1", 2, 0),
             _call("c2", ["/w/monitor-1.png"]), _out("c2", 1, 2),
             _call("c3", ["/w/c.png"]), _out("c3", 1, 3),
             _call("c4", ["/w/monitor-2.png"]), _out("c4", 1, 4)]
    out, removed = prune_images(items, keep=2, keep_monitor=1)
    imgs = [(x["call_id"], c.get("type"), c.get("text")) for x in out if x["type"].endswith("_output")
            for c in x["output"][1:]]
    assert removed == 2
    assert imgs == [("c1", "input_text", "[image removed from the history: /w/a.png]"),   # oldest goes
                    ("c1", "input_image", None),                                         # b.png: newest 2 others
                    ("c2", "input_text", "[image removed from the history: /w/monitor-1.png]"),
                    ("c3", "input_image", None),
                    ("c4", "input_image", None)]                                         # the newest monitor image
    assert items[1]["output"][1]["type"] == "input_image"                                # the input is not changed


# ------------------------------------------------------------------------------------------------ robot.kernel
def test_kernel_keeps_one_namespace_and_ends_calls_like_programs(tmp_path):
    import tempfile
    sock = os.path.join(tempfile.mkdtemp(dir="/tmp"), "k.sock")      # a unix socket path must be short (macOS: 104)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])), ROBOT_KERNEL_SOCK=sock)

    def call(code):
        return subprocess.run([sys.executable, "-m", "robot.kernel"], input=code, cwd=tmp_path, capture_output=True,
                              text=True, timeout=120, env=env)
    try:
        a = call("x = 41\nprint('first', x, np.__name__, viz.__name__)\n")
        assert a.returncode == 0 and "first 41 numpy robot.viz" in a.stdout, a.stdout + a.stderr
        b = call("x += 1\nprint('second', x)\nviz.show('k', np.zeros((8, 8), bool), r=False)\n")
        assert "second 42" in b.stdout
        assert b.stdout.strip().splitlines()[-1].startswith("[views] ")          # as a program prints at exit
        c = call("raise ValueError('boom')\n")
        assert c.returncode == 1 and "ValueError: boom" in c.stderr
        d = call("exit(3)\n")
        assert d.returncode == 3
        e = call("print('still', x)\n")                                           # the kernel survived both
        assert "still 42" in e.stdout
    finally:
        subprocess.run([sys.executable, "-m", "robot.kernel", "--stop"], env=env, capture_output=True, timeout=30)


# ------------------------------------------------------------------------------------------------ hooks, on_table
def test_marks_on_table_leaves_out_what_is_off_the_table(tmp_path, monkeypatch):
    from robot import viz
    monkeypatch.chdir(tmp_path)
    fr = _Frame()
    fr.T[0, 3] = 0.45                                     # the camera over x = 0.45: the table spans x 0.02-0.88
    fr.depth[:, 160:] = 1.6                               # the right half is the floor, 0.8 m below the table
    everything = np.ones(fr.depth.shape, bool)
    pts = viz.marks(fr, mask=everything, step=20, max_marks=500)
    assert min(p[2] for p in pts.values()) < -0.7         # without the filter: floor points
    on = viz.marks(fr, mask=everything, step=20, max_marks=500, on_table=True, r=False)
    assert on and all(abs(p[2]) < 1e-6 for p in on.values())
    box = viz.marks(fr, step=20, max_marks=500, within=((0.4, 0.9), (-1, 1), (-0.1, 0.1)))
    assert box and all(0.4 - 1e-6 <= p[0] <= 0.9 for p in box.values())
    quiet = viz.marks(fr, "nopic", mask=everything, on_table=True, r=False, picture=False)
    assert quiet and not os.path.exists(tmp_path / ".robot" / "views" / "nopic.png")


def _kernel(tmp, env_extra=None):
    import tempfile
    sock = os.path.join(tempfile.mkdtemp(dir="/tmp"), "k.sock")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])), ROBOT_KERNEL_SOCK=sock, ROBOT_ADDR="unix:/tmp/no-robot-here.sock")
    env.update(env_extra or {})

    def call(code):
        return subprocess.run([sys.executable, "-m", "robot.kernel"], input=code, cwd=tmp, capture_output=True,
                              text=True, timeout=120, env=env)

    def stop():
        subprocess.run([sys.executable, "-m", "robot.kernel", "--stop"], env=env, capture_output=True, timeout=30)
    return call, stop


def test_kernel_hooks_run_at_the_end_of_calls(tmp_path):
    call, stop = _kernel(tmp_path)
    try:
        a = call("def h(r, frames):\n    print('hook ran', r is None)\nkernel.after_call(h)\nprint('a')\n")
        assert a.returncode == 0, a.stderr
        assert a.stdout.splitlines()[:2] == ["a", "hook ran True"]           # the call that registered it, too
        b = call("print('b')\n")
        assert b.stdout.splitlines() == ["b", "hook ran True"]
        c = call("def bad(r):\n    raise ValueError('nope')\nkernel.after_call(bad)\n")
        assert c.returncode == 0 and "[hook bad] error ValueError: nope (line 2" in c.stdout
        d = call("kernel.after_motion(lambda r, f: print('motion hook'), name='m')\nprint(kernel.hooks())\n")
        assert "motion hook" not in d.stdout                                  # nothing moved
        assert "{'h': 'call', 'bad': 'call', 'm': 'motion'}" in d.stdout
        e = call("def h(r):\n    print('new h')\nkernel.after_call(h)\nkernel.remove_hook('bad')\n")
        assert "new h" in e.stdout and "hook ran" not in e.stdout and "[hook bad]" not in e.stdout
        f = call("def pic(r, frames):\n    viz.show('from-hook', np.zeros((8, 8), bool), r=False)\n"
                 "kernel.after_call(pic)\n")
        last = f.stdout.strip().splitlines()[-1]                              # the hook's image is in [views]
        assert last.startswith("[views] ") and "from-hook.png" in last
    finally:
        stop()


def test_an_after_motion_hook_runs_when_the_call_moved_the_arms(tmp_path):
    import tempfile
    import threading
    import time
    try:
        import mujoco                                                         # noqa: F401
        from robotd.backends.mujoco_sim import MujocoYamBackend
        from robotd.robots.yam_station.description import description
        from robotd.server import Driver, serve
        from tasks import yam_station                                         # not tasks.get: it loads every task
        b = MujocoYamBackend(yam_station.TASKS["empty"].scene())
    except Exception as e:                                                    # noqa: BLE001 -- no GL here (macOS)
        pytest.skip(f"no simulator: {e}")
    d = tempfile.mkdtemp(dir="/tmp")
    b.lockstep = False
    addr, ready = f"unix:{d}/robot.sock", os.path.join(d, "ready")
    threading.Thread(target=serve, args=(Driver(b, description()), addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    call, stop = _kernel(tmp_path, {"ROBOT_ADDR": addr})
    try:
        a = call("def seen(r):\n    print('hook after motion:', r.state().summary())\nkernel.after_motion(seen)\n"
                 "r = robot.connect(); print(r.state().summary())\n")
        assert a.returncode == 0, a.stderr
        assert "hook after motion" not in a.stdout                            # it read the state, nothing moved
        m = call("r = robot.connect(); js = r.description['arm_joints']['left']\n"
                 "r.move(r.state().q(js) + np.array([0.05, 0, 0, 0, 0, 0]), joints=js, duration=0.5)\n")
        assert m.returncode == 0, m.stderr
        lines = m.stdout.strip().splitlines()
        assert lines[0].startswith("[state] ") and lines[1].startswith("hook after motion: left xyz=")
    finally:
        stop()
        try:
            import robot
            robot.connect(addr)._call("shutdown")
        except Exception:                                                     # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------ actions
def test_kernel_actions_run_while_calls_go_on(tmp_path):
    import time
    call, stop = _kernel(tmp_path)
    try:
        t0 = time.time()
        a = call("import time\ndef act():\n    for i in range(20):\n        time.sleep(0.1)\n        if i == 3: print('half way')\n"
                 "kernel.start(act)\n")
        assert a.returncode == 0, a.stderr
        assert "[action act] started" in a.stdout and time.time() - t0 < 8     # returned before the action ended
        b = call("print(kernel.action().running)\n")
        assert "True" in b.stdout and "[action act] running for" in b.stdout
        c = call("print('ok', kernel.action().wait())\n")
        assert "ok True" in c.stdout and "[action act] done in" in c.stdout
        assert "[action act] half way" in b.stdout + c.stdout                 # what it printed, reported once
        d = call("def loop():\n    while True: time.sleep(0.05)\nkernel.start(loop)\n")
        assert "[action loop] started" in d.stdout
        e = call("kernel.start(loop)\n")                                       # one at a time
        assert e.returncode == 1 and "still running" in e.stderr
        f = call("a = kernel.action(); print(a.stop()); print(a.wait(3))\n")
        assert "True\nFalse" in f.stdout and "[action loop] stopped after" in f.stdout
        g = call("def bad():\n    raise ValueError('boom')\nkernel.start(bad).wait()\n")
        assert "[action bad] ValueError: boom" in g.stdout and "[action bad] error ValueError: boom" in g.stdout
        h = call("kernel.start(loop, timeout=0.5).wait()\n")
        assert "(its timeout of 0.5 s passed)" in h.stdout
    finally:
        stop()


def test_an_action_moves_the_arm_while_calls_go_on_and_stops_when_asked(tmp_path):
    import tempfile
    import threading
    import time
    try:
        import mujoco                                                         # noqa: F401
        from robotd.backends.mujoco_sim import MujocoYamBackend
        from robotd.robots.yam_station.description import description
        from robotd.server import Driver, serve
        from tasks import yam_station
        b = MujocoYamBackend(yam_station.TASKS["empty"].scene(), realtime=True)   # on the wall clock, as the cell
    except Exception as e:                                                    # noqa: BLE001 -- no GL here (macOS)
        pytest.skip(f"no simulator: {e}")
    d = tempfile.mkdtemp(dir="/tmp")
    addr, ready = f"unix:{d}/robot.sock", os.path.join(d, "ready")
    threading.Thread(target=serve, args=(Driver(b, description()), addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    call, stop = _kernel(tmp_path, {"ROBOT_ADDR": addr})
    slow = ("js = robot.connect().description['arm_joints']['left']\n"
            "def slow(r):\n    q = r.state().q(js)\n    r.move(q + np.array([0.6, 0, 0, 0, 0, 0]), joints=js, duration=4.0)\n"
            "    print('reached the end')\n")
    try:
        call("kernel.after_motion(lambda r, f: print('hook ran'), name='h')\n")
        t0 = time.time()
        a = call(slow + "q0 = robot.connect().state().q(js)\nkernel.start(slow)\n")
        assert a.returncode == 0, a.stderr
        assert time.time() - t0 < 3.5 and "[action slow] started" in a.stdout
        time.sleep(1.0)
        s = call("a = kernel.action(); a.stop(); a.wait(5)\nq1 = robot.connect().state().q(js); print('moved', round(q1[0] - q0[0], 2))\n")
        assert "[action slow] stopped after" in s.stdout and "reached the end" not in s.stdout
        moved = float(s.stdout.split("moved ")[1].split()[0])
        assert 0.02 < moved < 0.5                                             # it moved, and stopped part way
        assert "[state] " in s.stdout and "hook ran" in s.stdout              # as when a call's own motion ends
        last = s.stdout.strip().splitlines()[-1]
        assert last.startswith("[views] ") and "after-slow.png" in last           # every camera, in the same call
        c = call(slow + "kernel.start(slow)\n")
        m = call("r = robot.connect(); r.move(r.state().q(js), joints=js, duration=0.3)\n")
        assert "[action slow] stopped after" in m.stdout and "this call sent the arms a command" in m.stdout
    finally:
        stop()
        try:
            import robot
            robot.connect(addr)._call("shutdown")
        except Exception:                                                     # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------ the push
def _events(d):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*-action-*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def _until(pred, s=10.0):
    import time
    t_end = time.time() + s
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_an_ended_action_pushes_its_report_and_a_waiting_call_gets_it_instead(tmp_path):
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_KERNEL_EVENTS": ev})
    try:
        a = call("import time\ndef act():\n    time.sleep(0.3); print('half way'); time.sleep(0.3)\nkernel.start(act)\n")
        assert a.returncode == 0, a.stderr
        assert "[action act] started (as it ends, its report comes to you as a message)" in a.stdout
        with open(os.path.join(ev, "actions.json")) as f:
            assert json.load(f)["running"] == "act"
        assert _until(lambda: len(_events(ev)) == 1)                         # no call needed: a file for the harness
        r = _events(ev)[0]
        assert r["kind"] == "action" and r["name"] == "act" and r["ok"] is True
        assert "[action act] half way" in r["text"] and "[action act] done in" in r["text"]
        assert "no [state]" in r["text"]                                     # no robot here
        assert _until(lambda: json.load(open(os.path.join(ev, "actions.json")))["running"] is None)
        b = call("print('next')\n")
        assert "[action act] done in" in b.stdout and "(its report is sent to you as a message)" in b.stdout
        assert "half way" not in b.stdout                                    # the report went as the message
        c = call("print('quiet')\n")
        assert "[action act]" not in c.stdout                                # said once
        # a call that waits gets the report in its own output, and nothing is pushed
        d = call("def act2():\n    time.sleep(0.4); print('two')\nprint('ok', kernel.start(act2).wait())\n")
        assert "ok True" in d.stdout and "[action act2] two" in d.stdout and "[action act2] done in" in d.stdout
        assert "no [state]" in d.stdout
        assert len(_events(ev)) == 1
        # stopped by a call: pushed, stopped
        call("def loop():\n    while True: time.sleep(0.05)\nkernel.start(loop)\n")
        e = call("kernel.action().stop()\n")
        assert e.returncode == 0, e.stderr
        assert _until(lambda: len(_events(ev)) == 2)
        r2 = _events(ev)[1]
        assert r2["name"] == "loop" and r2["ok"] is False and "[action loop] stopped after" in r2["text"]
    finally:
        stop()


def test_a_pushed_report_carries_the_hooks_and_their_images(tmp_path):
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_KERNEL_EVENTS": ev})
    try:
        a = call("def pic(r, frames):\n    viz.show('hook-pic', np.zeros((8, 8), bool), r=False)\n"
                 "kernel.after_motion(pic)\n"
                 "import time\ndef act():\n    time.sleep(0.2)\nkernel.start(act, views=False)\n"
                 "time.sleep(1.5)\nviz.show('the-call', np.ones((8, 8), bool), r=False)\n")
        assert a.returncode == 0, a.stderr
        assert _until(lambda: len(_events(ev)) == 1)
        r = _events(ev)[0]
        assert [os.path.basename(p) for p in r["images"]] == ["hook-pic.png"]
        assert "[view] hook-pic:" in r["text"]
        last = a.stdout.strip().splitlines()[-1]                             # the call's own [views]: not the report's
        assert last.startswith("[views] ") and "the-call.png" in last and "hook-pic" not in last
    finally:
        stop()


def test_action_events_deliver_reports_with_copied_images(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import agent_run_real as arr
    except Exception as e:                                                   # noqa: BLE001
        pytest.skip(f"the launcher does not import here: {e}")
    ws = tmp_path / "workspace"
    (ws / ".robot" / "views").mkdir(parents=True)
    (ws / ".robot" / "events").mkdir()
    (ws / ".robot" / "views" / "after-grasp.png").write_bytes(b"png1")
    rep = {"kind": "action", "name": "grasp", "status": "done in 9.8 s", "written": 1.0,
           "text": "[action grasp] done in 9.8 s\n[state] left ...\n[view] after-grasp: "
                   f"{arr.ar.WS}/.robot/views/after-grasp.png\n",
           "images": [f"{arr.ar.WS}/.robot/views/after-grasp.png"]}
    (ws / ".robot" / "events" / "1000-0001-action-grasp.json").write_text(json.dumps(rep))
    (ws / ".robot" / "events" / "actions.json").write_text(json.dumps({"running": "next", "deadline": 4e9}))
    got = []
    e = arr.ActionEvents(str(ws), lambda text, images: got.append((text, images)) or "steer")
    assert e.busy()
    e.poll()
    assert len(got) == 1
    text, images = got[0]
    assert text.startswith("[action grasp] done in 9.8 s") and text.endswith(
        f'[views] ["{arr.ar.WS}/.robot/views/after-grasp.png"]')
    assert images == [f"{arr.ar.WS}/.robot/events/sent/1000-0001-action-grasp-0.png"]
    assert (ws / ".robot" / "events" / "sent" / "1000-0001-action-grasp-0.png").read_bytes() == b"png1"
    assert (ws / ".robot" / "events" / "delivered" / "1000-0001-action-grasp.json").exists()
    e.poll()
    assert len(got) == 1 and e.log[0]["how"] == "steer"
    (ws / ".robot" / "events" / "actions.json").write_text(json.dumps({"running": None}))
    assert not e.busy()


def test_prune_takes_images_out_of_pushed_messages_too():
    from codex_driver import prune_images
    msg = {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "[action a] done\n[view] after-a: /w/after-a.png\n"},
        {"type": "input_image", "image_url": "data:x"}]}
    items = [_call("c1", ["/w/a.png"]), _out("c1", 1, 0), msg, _call("c2", ["/w/b.png"]), _out("c2", 1, 2)]
    out, removed = prune_images(items, keep=1)
    assert removed == 2
    assert out[2]["content"][1] == {"type": "input_text", "text": "[image removed from the history: /w/after-a.png]"}
    assert out[4]["output"][1]["type"] == "input_image"


def test_in_the_sim_an_ended_action_pushes_state_cameras_and_hooks(tmp_path):
    import tempfile
    import threading
    import time
    try:
        import mujoco                                                         # noqa: F401
        from robotd.backends.mujoco_sim import MujocoYamBackend
        from robotd.robots.yam_station.description import description
        from robotd.server import Driver, serve
        from tasks import yam_station
        b = MujocoYamBackend(yam_station.TASKS["empty"].scene(), realtime=True)
    except Exception as e:                                                    # noqa: BLE001 -- no GL here (macOS)
        pytest.skip(f"no simulator: {e}")
    d = tempfile.mkdtemp(dir="/tmp")
    addr, ready = f"unix:{d}/robot.sock", os.path.join(d, "ready")
    threading.Thread(target=serve, args=(Driver(b, description()), addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_ADDR": addr, "ROBOT_KERNEL_EVENTS": ev})
    try:
        call("kernel.after_motion(lambda r, f: print('hook ran', sorted(f)[:1]), name='h')\n")
        a = call("js = robot.connect().description['arm_joints']['left']\n"
                 "def nudge(r):\n    q = r.state().q(js)\n    r.move(q + np.array([0.2, 0, 0, 0, 0, 0]), joints=js, duration=1.0)\n"
                 "kernel.start(nudge)\n")
        assert a.returncode == 0, a.stderr
        t_start = time.time()
        assert _until(lambda: len(_events(ev)) == 1, 30)
        r = _events(ev)[0]
        assert r["ok"] and "[action nudge] done in" in r["text"] and "[state] " in r["text"]
        assert "hook ran" in r["text"]
        assert [os.path.basename(p) for p in r["images"]] == ["after-nudge.png"]
        assert os.path.exists(r["images"][0])
        assert r["written"] - r["t1"] < 5.0, r                               # the report follows the motion's end
        print("report took", r["report_s"], "s; pushed", round(r["written"] - r["t1"], 2), "s after the end;",
              round(time.time() - t_start, 2), "s after the start call returned")
    finally:
        stop()
        try:
            import robot
            robot.connect(addr)._call("shutdown")
        except Exception:                                                     # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------ the queue
_ACT = "import time\ndef {n}():\n    print('in {n}', round(time.time(), 3)); time.sleep({s})\n"
_LOOP = "import time\ndef slow():\n    while True: time.sleep(0.05)\n"           # a stop reaches Python code between sleeps


def test_queued_actions_run_in_order_one_at_a_time(tmp_path):
    call, stop = _kernel(tmp_path)
    try:
        a = call(_ACT.format(n="one", s=0.6) + _ACT.format(n="two", s=0.3) + _ACT.format(n="three", s=0.1) +
                 "A = kernel.queue(one); B = kernel.queue(two); C = kernel.queue(three)\n"
                 "print('waiting', [x.name for x in kernel.queued()], A.running, B.running, B.status()[:6])\n")
        assert a.returncode == 0, a.stderr
        assert "[action one] started" in a.stdout and "[action two] queued behind one" in a.stdout
        assert "waiting ['two', 'three'] True False queued" in a.stdout
        b = call("print(C.wait(10), [round(y.t0 - x.t1, 3) for x, y in zip((A, B), (B, C))])\n")
        assert "True [" in b.stdout, b.stdout + b.stderr
        gaps = eval(b.stdout[b.stdout.index("["):b.stdout.index("]") + 1])
        assert all(0 <= g < 0.1 for g in gaps), gaps                           # each starts as the last one ends
        c = call("print('order', [x.name for x in kernel.actions()], [x.running for x in kernel.actions()])\n")
        assert "order ['one', 'two', 'three'] [False, False, False]" in c.stdout
    finally:
        stop()


def test_the_queue_is_dropped_when_an_action_does_not_finish(tmp_path):
    call, stop = _kernel(tmp_path)
    try:
        call(_ACT.format(n="after", s=0.1) + "def bad():\n    time.sleep(0.3); raise ValueError('boom')\n"
             "A = kernel.queue(bad); B = kernel.queue(after)\n")
        a = call("print(B.wait(5), A.error, B.status()[:7], B.stopped)\n")
        assert "False boom stopped dropped: bad did not finish" in a.stdout, a.stdout + a.stderr
        b = call("print('ran', [x.name for x in kernel.actions()][-1:])\n")
        assert "ran ['bad']" in b.stdout                                       # after never started
        c = call(_LOOP + "S = kernel.queue(slow, timeout=60); B = kernel.queue(after)\n"
                 "print(S.stop()); print(S.wait(5), B.wait(5), B.stopped)\n")
        assert "dropped: slow did not finish" in c.stdout, c.stdout + c.stderr  # a stop drops it too
    finally:
        stop()


def test_the_queue_limits_and_flushes(tmp_path):
    call, stop = _kernel(tmp_path)
    try:
        a = call(_LOOP + _ACT.format(n="x", s=0) + "S = kernel.queue(slow, timeout=60)\n"
                 "B = kernel.queue(x, name='b'); C = kernel.queue(x, name='c'); D = kernel.queue(x, name='d')\n"
                 "kernel.queue(x, name='e')\n")
        assert a.returncode == 1 and "the queue holds 3" in a.stderr, a.stderr
        b = call("print(B.stop(), [x.name for x in kernel.queued()], kernel.flush(), kernel.queued())\n")
        assert "True ['c', 'd'] ['c', 'd'] []" in b.stdout, b.stdout + b.stderr
        c = call("print(S.running)\nS.stop(); print(S.wait(5))\nkernel.queue(x, name='e').wait(5)\n"
                 "print(kernel.action().name)\n")
        assert "True\nFalse\n" in c.stdout and "\ne\n" in c.stdout, c.stdout + c.stderr
    finally:
        stop()


def test_a_queued_action_that_arrives_late_starts_at_once_and_says_so(tmp_path):
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_KERNEL_EVENTS": ev, "ROBOT_KERNEL_STAGGER": "1", "ROBOT_ADDR": "unix:/nonexistent"})
    try:
        a = call(_ACT.format(n="only", s=0.2) + "kernel.queue(only)\n")
        assert a.returncode == 0, a.stderr
        assert _until(lambda: len(_events(ev)) == 2, 20)
        st, end = sorted(_events(ev), key=lambda e: e["written"])
        assert st["phase"] == "start" and "started (the arms were holding)" in st["text"]
        assert "[queue] empty" in end["text"] and "done in" in end["text"]     # the arms hold: the report says so
        assert _until(lambda: json.load(open(os.path.join(ev, "actions.json")))["running"] is None, 5)
    finally:
        stop()


def test_in_the_sim_queued_actions_move_the_arm_without_a_stop_for_the_report(tmp_path):
    import tempfile
    import threading
    import time
    try:
        import mujoco                                                         # noqa: F401
        from robotd.backends.mujoco_sim import MujocoYamBackend
        from robotd.robots.yam_station.description import description
        from robotd.server import Driver, serve
        from tasks import yam_station
        b = MujocoYamBackend(yam_station.TASKS["empty"].scene(), realtime=True)
    except Exception as e:                                                    # noqa: BLE001 -- no GL here (macOS)
        pytest.skip(f"no simulator: {e}")
    d = tempfile.mkdtemp(dir="/tmp")
    addr, ready = f"unix:{d}/robot.sock", os.path.join(d, "ready")
    threading.Thread(target=serve, args=(Driver(b, description()), addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_ADDR": addr, "ROBOT_KERNEL_EVENTS": ev, "ROBOT_KERNEL_STAGGER": "1"})
    try:
        a = call("js = robot.connect().description['arm_joints']['left']\n"
                 "q0 = robot.connect().state().q(js)\n"
                 "def go(dq):\n    def f(r):\n        r.move(q0 + np.array([dq, 0, 0, 0, 0, 0]), joints=js, duration=0.8)\n    return f\n"
                 "A = kernel.queue(go(0.15), name='a'); B = kernel.queue(go(0.3), name='b'); C = kernel.queue(go(0.1), name='c')\n"
                 "import time; t = time.time()\n")
        assert a.returncode == 0, a.stderr
        assert _until(lambda: sum(e.get("phase") != "start" for e in _events(ev)) == 3, 40), _events(ev)
        ends = sorted((e for e in _events(ev) if e.get("phase") != "start"), key=lambda e: e["t0"])
        assert [e["name"] for e in ends] == ["a", "b", "c"]
        assert all(e["ok"] and "[state] " in e["text"] for e in ends), ends
        assert "[queue] started: b" in ends[0]["text"] and "[queue] started: c" in ends[1]["text"]
        assert "[queue] empty" in ends[2]["text"]
        gaps = [round(y["t0"] - x["t1"], 3) for x, y in zip(ends, ends[1:])]
        assert all(g < 0.1 for g in gaps), gaps                               # the report does not hold up the arms
        b = call("print('end', np.round(robot.connect().state().q(js) - q0, 2)[0])\n")
        assert "end 0.1" in b.stdout, b.stdout + b.stderr                     # the shared connection stayed in step
        print("gaps between actions", gaps, "report seconds", [e["report_s"] for e in ends])
    finally:
        stop()
        try:
            import robot
            robot.connect(addr)._call("shutdown")
        except Exception:                                                     # noqa: BLE001
            pass


def test_in_the_sim_a_park_during_an_action_starts_nothing_that_was_queued(tmp_path):
    import tempfile
    import threading
    import time
    try:
        import mujoco                                                         # noqa: F401
        from robotd.backends.mujoco_sim import MujocoYamBackend
        from robotd.robots.yam_station.description import description
        from robotd.server import Driver, serve
        from tasks import yam_station
        b = MujocoYamBackend(yam_station.TASKS["empty"].scene(), realtime=True)
    except Exception as e:                                                    # noqa: BLE001 -- no GL here (macOS)
        pytest.skip(f"no simulator: {e}")
    d = tempfile.mkdtemp(dir="/tmp")
    addr, ready = f"unix:{d}/robot.sock", os.path.join(d, "ready")
    drv = Driver(b, description())
    threading.Thread(target=serve, args=(drv, addr, ready), daemon=True).start()
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_ADDR": addr, "ROBOT_KERNEL_EVENTS": ev, "ROBOT_KERNEL_STAGGER": "1"})
    try:
        a = call("js = robot.connect().description['arm_joints']['left']\n"
                 "q0 = robot.connect().state().q(js)\n"
                 "def go(dq, name):\n    def f(r):\n        print('running', name)\n"
                 "        r.move(q0 + np.array([dq, 0, 0, 0, 0, 0]), joints=js, duration=2.0)\n    return f\n"
                 "A = kernel.queue(go(0.2, 'a'), name='a'); B = kernel.queue(go(0.5, 'b'), name='b')\n")
        assert a.returncode == 0, a.stderr
        time.sleep(0.7)
        import robot
        ctl = robot.connect(addr, name="dashboard/control")
        ctl._call("park", reason="the operator parked the arms", operator=True)
        assert _until(lambda: sum(e.get("phase") != "start" for e in _events(ev)) >= 2 or
                      any("dropped" in e["text"] for e in _events(ev)), 30), _events(ev)
        time.sleep(1.0)
        q1 = ctl.state().q(call and robot.connect(addr).description["arm_joints"]["left"])
        time.sleep(1.5)
        q2 = ctl.state().q(robot.connect(addr).description["arm_joints"]["left"])
        assert np.abs(q2 - q1).max() < 0.02, (q1, q2)                         # nothing queued moved the arm
        text = "\n".join(e["text"] for e in _events(ev))
        print(text)
        ran_b = "[action b] running b" in text
        assert not ran_b or "error" in text.split("[action b]", 1)[1], text   # b never moved
    finally:
        stop()
        try:
            import robot
            robot.connect(addr)._call("shutdown")
        except Exception:                                                     # noqa: BLE001
            pass


def test_replacing_an_item_drops_everything_behind_it_in_one_step(tmp_path):
    call, stop = _kernel(tmp_path, {"ROBOT_KERNEL_STAGGER": "1"})
    try:
        a = call(_LOOP + _ACT.format(n="x", s=0) + "S = kernel.queue(slow, timeout=60)\n"
                 "A = kernel.queue(x, name='a'); B = kernel.queue(x, name='b'); C = kernel.queue(x, name='c')\n"
                 "print('ids', A.id - S.id, B.id - S.id, C.id - S.id)\n")
        assert "ids 1 2 3" in a.stdout, a.stdout + a.stderr
        assert f"[queue] waiting 2:" in a.stdout and "[queue] running #" in a.stdout
        b = call("N = kernel.queue([x, x], replace_from=B.id)\n"
                 "print('now', [q.name for q in kernel.queued()], B.stopped, C.stopped)\n")
        assert "now ['a', 'x', 'x'] dropped: replaced by x dropped: replaced by x" in b.stdout, b.stdout + b.stderr
        assert "[queue] replaced:" in b.stdout
        c = call("kernel.queue(x, replace_from=A.id, name='n'); print('next', [q.name for q in kernel.queued()])\n")
        assert "next ['n']" in c.stdout, c.stdout + c.stderr                     # by id: A and both behind it went
        d = call("kernel.queue(x, replace_from=S.id)\n")
        assert d.returncode == 1 and "not waiting" in d.stderr, d.stderr         # it has started: nothing changes
        e = call("print('still', [q.name for q in kernel.queued()]); print(kernel.flush(from_item='n'), kernel.queued())\n"
                 "S.stop(); print(S.wait(5))\n")
        assert "still ['n']" in e.stdout and "['n'] []" in e.stdout, e.stdout + e.stderr
    finally:
        stop()


def test_a_replacement_too_long_for_the_queue_changes_nothing(tmp_path):
    call, stop = _kernel(tmp_path)
    try:
        call(_LOOP + _ACT.format(n="x", s=0) + "S = kernel.queue(slow, timeout=60); A = kernel.queue(x, name='a')\n"
             "B = kernel.queue(x, name='b')\n")
        a = call("kernel.queue([x, x, x], replace_from=B.id)\n")
        assert a.returncode == 1 and "the queue holds 3" in a.stderr, a.stderr
        b = call("print([q.name for q in kernel.queued()], B.stopped)\nkernel.flush(); S.stop()\n")
        assert "['a', 'b'] None" in b.stdout, b.stdout + b.stderr
    finally:
        stop()


def test_the_harness_asks_for_a_look_only_while_an_action_runs_and_the_agent_is_idle(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_run_real", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                                                  "agent_run_real.py"))
    try:
        m = importlib.util.module_from_spec(spec)
        sys.modules["agent_run_real"] = m
        spec.loader.exec_module(m)
    except Exception as e:                                                    # noqa: BLE001
        pytest.skip(f"launcher not importable: {e}")
    import time
    ws = tmp_path / "ws"
    d = ws / ".robot" / "events"
    d.mkdir(parents=True)
    ae = m.ActionEvents(str(ws), lambda text, images: "new turn")
    def running(name):
        (d / "actions.json").write_text(json.dumps({"running": name, "deadline": time.time() + 60}))
    req = d / "look.request"
    assert not ae.look_if_idle(True) and not req.exists()                     # no action
    running("a")
    assert not ae.look_if_idle(False) and not req.exists()                    # the agent is working
    assert ae.look_if_idle(True) and req.exists()
    req.unlink()
    assert not ae.look_if_idle(True) and not req.exists()                     # one at a time
    (d / "1-0001-action-a.json").write_text(json.dumps({"name": "a", "phase": "look", "text": "[look 1]", "images": []}))
    assert not ae.look_if_idle(True)                                          # a message is on its way
    ae.poll()
    assert not (d / "1-0001-action-a.json").exists()
    assert ae.look_if_idle(True) and req.exists()                             # answered: the next one
    req.unlink()
    ae.look_asked = 0.0
    assert not ae.look_if_idle(True, every_s=60)                              # the throttle


def test_a_look_brings_state_and_the_queue_while_an_action_runs(tmp_path):
    ev = str(tmp_path / "events")
    call, stop = _kernel(tmp_path, {"ROBOT_KERNEL_EVENTS": ev, "ROBOT_KERNEL_STAGGER": "1", "ROBOT_ADDR": "unix:/nonexistent"})
    try:
        a = call(_LOOP + _ACT.format(n="x", s=0) + "S = kernel.queue(slow, timeout=60); B = kernel.queue(x, name='b')\n")
        assert a.returncode == 0, a.stderr
        open(os.path.join(ev, "look.request"), "w").close()
        assert _until(lambda: any(e.get("phase") == "look" for e in _events(ev)), 15), _events(ev)
        lk = next(e for e in _events(ev) if e.get("phase") == "look")
        assert "[look 1]" in lk["text"] and "[queue] running #" in lk["text"] and "[queue] waiting 1:" in lk["text"], lk
        assert not os.path.exists(os.path.join(ev, "look.request"))
        call("kernel.flush(); S.stop()\n")
    finally:
        stop()
