"""The program runner (python -m robot run / wait / debug) against a fake driver: assert rewriting, pauses, the
debugger's names and commands, the run options."""
import glob
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import textwrap
import threading
import time
import zlib

import numpy as np
import pytest

from robot import rpc, runner
from robot.runner import ASSERT_HOOK, rewrite_asserts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------------------------ assert rewriting
def test_assert_rewrite_keeps_lines_and_c_runs_the_next_line():
    src = textwrap.dedent('''\
        def f(x, log):
            log.append("before")
            assert x > 1, f"x is {x}"
            log.append("after")
            assert True, 1 / 0
            assert (x >
                    5)
            if x < 0:
                raise ValueError("negative")
            return x
        ''')
    calls = []

    def hook(msg, test, next_line):          # a pause that returns at once: what c does
        calls.append((msg, test, next_line, sys._getframe(1).f_lineno))
    g = {ASSERT_HOOK: hook}
    exec(compile(rewrite_asserts(src, "t.py"), "t.py", "exec"), g)
    log = []
    assert g["f"](0, log) == 0
    assert log == ["before", "after"]                     # the line after the failed assert ran
    assert calls == [("x is 0", "x > 1", 4, 3), (None, "x > 5", 8, 6)]
    with pytest.raises(ValueError) as e:                  # line numbers after the rewritten asserts are unchanged
        g["f"](-1, [])
    assert e.traceback[-1].lineno + 1 == 9


def test_the_cli_does_not_load_numpy():
    code = ("import sys, robot, robot.__main__; print('numpy' in sys.modules, 'scipy' in sys.modules); "
            "robot.pose([0, 0, 1], rpy=[0, 0, 1]); print('numpy' in sys.modules, 'scipy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.stdout.split() == ["False", "False", "True", "True"], out.stdout + out.stderr


def test_failed_assert_without_a_runner_raises():
    g = {ASSERT_HOOK: runner._assert_failed}
    exec(compile(rewrite_asserts("def f(x):\n    assert x, 'no x'\n", "t.py"), "t.py", "exec"), g)
    with pytest.raises(AssertionError, match="no x"):
        g["f"](0)


# ------------------------------------------------------------------------------------------------ a fake driver
class FakeDriver:
    """Answers hello, state and cameras as robotd does; every other request with an empty result."""

    def __init__(self, path):
        self.src_text = []                                   # every request's src_text, in order
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(8)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            while (req := rpc.recv(conn)) is not None:
                if req.get("src_text"):
                    self.src_text.append(req["src_text"])
                if req["op"] == "sleep":                     # a slow request
                    time.sleep(req["s"])
                rpc.send(conn, {"ok": True, "result": self._reply(req["op"])})
        except OSError:
            pass

    @staticmethod
    def _reply(op):
        if op == "hello":
            return {"description": {"end_effectors": {"left": {}}}, "lockstep": True}
        if op == "state":
            return {"t": 1.5, "joints": ["j1"], "q": [0.1], "dq": [0.0], "effort": [0.0], "effort_ext": [0.0],
                    "q_ref": [0.1], "ee": {}, "grippers": {"left": {"width": 0.05}}}
        if op == "cameras":
            rgb = np.zeros((12, 16, 3), np.uint8)
            rgb[..., 0] = 200
            depth = np.full((12, 16), 0.75, np.float32)
            depth[0, 0] = 0.0
            return {"top": {"rgb": rgb, "depth": depth, "K": np.eye(3).tolist(), "T": np.eye(4).tolist(), "t": 1.5}}
        return {}

    def close(self):
        self.srv.close()


def read_png16(path):
    """A 16-bit grey PNG as written by robot.images (filter None or Up)."""
    data = open(path, "rb").read()
    w, h, bits, colour = struct.unpack(">IIBB", data[16:26])
    assert (bits, colour) == (16, 0)
    pos, idat = 8, b""
    while pos < len(data):
        n, kind = struct.unpack(">I", data[pos:pos + 4])[0], data[pos + 4:pos + 8]
        idat += data[pos + 8:pos + 8 + n] if kind == b"IDAT" else b""
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 2 * w)
    rows = raw[:, 1:].copy()
    for y in range(1, h):
        if raw[y, 0] == 2:
            rows[y] += rows[y - 1]
    return rows.view(">u2").reshape(h, w)


class Cli:
    def __init__(self, path):
        self.dir = path
        self.env = dict(os.environ, ROBOT_ADDR=f"unix:{path}/robot.sock", PYTHONPATH=ROOT)
        self.env.pop("VISION_ADDR", None)

    def __call__(self, *args, timeout=60):
        p = subprocess.run([sys.executable, "-m", "robot", *args], cwd=self.dir, env=self.env, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr

    def write(self, name, src):
        (self.dir / name).write_text(textwrap.dedent(src))

    def status(self):
        return json.load(open(self.dir / ".robot" / "current" / "status.json"))


@pytest.fixture
def cli(tmp_path):
    drv = FakeDriver(str(tmp_path / "robot.sock"))
    c = Cli(tmp_path)
    c.drv = drv
    c.write("lib.py", '''\
        def check(s):
            return False

        def second():
            return "old"

        def small(n):
            v = "mine"
            assert n > 5, "n too small"
            return n
        ''')
    c.write("task.py", '''\
        from lib import check, second

        def first(r, v, s):
            s.n = 1
            s.tries = 0
            print("first ran")

        def middle(r, v, s):
            s.tries += 1
            ok = check(s)
            assert ok, f"check failed (try {s.tries})"
            print("after assert", ok, second())

        def last(r, v, s):
            print("last ran, tries", s.tries, "n", s.n)

        STEPS = [first, middle, last]
        ''')
    yield c
    for path in glob.glob(str(tmp_path / ".robot" / "runs" / "*" / "status.json")):     # the programs this test began
        st = json.load(open(path))
        if st["state"] != "exited":
            try:
                os.kill(st["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    drv.close()


def test_assert_pauses_saves_frames_and_c_goes_on_as_if_it_passed(cli):
    code, out = cli("run", "task.py")
    assert code == 0, out
    assert "first ran" in out
    assert "--- pause 1: assert failed at task.py:11 in middle (step middle 2/3)" in out
    assert "check failed (try 1)" in out and "c goes on with line 12 as if the check had passed" in out
    st = cli.status()
    assert st["state"] == "paused" and st["kind"] == "assert" and st["line"] == 11 and st["step"] == "middle"
    pdir = st["pause_dir"]
    assert sorted(os.listdir(pdir)) == ["cameras.json", "message.txt", "state.json", "top.png", "top_depth.png"]
    depth = read_png16(os.path.join(pdir, "top_depth.png"))
    assert depth.shape == (12, 16) and depth[0, 0] == 0 and (depth[1:] == 750).all()
    assert json.load(open(os.path.join(pdir, "state.json")))["t"] == 1.5
    assert json.load(open(os.path.join(pdir, "cameras.json")))["top"]["K"] == np.eye(3).tolist()
    assert cli("debug", "p s.tries") == (0, "1\n")
    assert cli("debug", "p ok") == (0, "False\n")
    assert cli("debug", "!ok = True")[0] == 0                 # written back into the paused frame
    code, out = cli("debug", "c")
    assert code == 0, out
    assert "after assert True old" in out and "last ran, tries 1 n 1" in out and "--- exit code 0" in out
    assert cli.status()["state"] == "exited"


def test_unattended_raises_and_exits_nonzero(cli):
    code, out = cli("run", "task.py", "--unattended")
    assert code == 1, out
    assert "AssertionError: check failed (try 1)" in out
    assert 'task.py", line 11, in middle' in out and "runner.py" not in out
    assert cli.status()["kind"] == "exception"


def test_names_r_v_s_resolve_in_every_frame(cli):
    cli.write("task.py", '''\
        from lib import small

        def first(r, v, s):
            s.n = 1
            small(s.n)

        STEPS = [first]
        ''')
    code, out = cli("run", "task.py")
    assert "--- pause 1: assert failed at lib.py:9 in small (step first 1/1)" in out, out
    assert cli("debug", "p s.n") == (0, "1\n")                 # not in small's frame: the runner's s
    assert cli("debug", "p v") == (0, "'mine'\n")             # the frame's own name comes first
    assert cli("debug", "p r.state().t") == (0, "1.5\n")
    assert cli("debug", "p [s.n + i for i in range(2)]") == (0, "[1, 2]\n")     # a nested scope
    assert cli("debug", "!s.n = 7") == (0, "")
    code, out = cli("debug", "up")
    assert "first()" in out and "small(s.n)" in out
    assert cli("debug", "p v") == (0, "None\n")
    assert cli("debug", "pp s.n") == (0, "7\n")
    code, out = cli("debug", "up")
    assert "Oldest frame" in out                              # the runner's frames are not shown
    cli("debug", "down")
    code, out = cli("debug", "where")
    assert "runner.py" not in out and "lib.py" in out
    code, out = cli("debug", "q")
    assert code == 1 and "--- ended by q at pause 1" in out
    assert cli.status()["kind"] == "quit"


def test_retry_reruns_the_step_with_new_code_and_keeps_s(cli):
    cli("run", "task.py")
    cli.write("lib.py", open(cli.dir / "lib.py").read().replace("return False", "return True"))
    code, out = cli("debug", "retry")
    assert code == 0, out
    assert "reloaded lib.py, task.py; STEPS = [first, middle, last]" in out
    assert out.count("--- step middle (2/3)") == 1 and "after assert True old" in out
    assert "last ran, tries 2 n 1" in out


def test_reload_then_c_runs_the_new_helper(cli):
    cli("run", "task.py")
    cli.write("lib.py", open(cli.dir / "lib.py").read().replace('"old"', '"new"'))
    code, out = cli("debug", "reload")
    assert code == 0 and "reloaded lib.py, task.py" in out, out
    code, out = cli("debug", "c")
    assert "after assert False new" in out and "--- exit code 0" in out, out


def test_goto_a_step_added_since_the_start_and_a_typo(cli):
    cli("run", "task.py")
    code, out = cli("debug", "goto fixup")
    assert code == 0 and "*** no step 'fixup'; STEPS = [first, middle, last]" in out, out
    assert cli.status()["state"] == "paused"
    src = open(cli.dir / "task.py").read().replace(
        "STEPS = [first, middle, last]", 'def fixup(r, v, s):\n    print("fixup ran", s.n)\n\n\n'
                                         "STEPS = [first, middle, fixup, last]")
    cli.write("task.py", src)
    code, out = cli("debug", "goto fxup")
    assert "*** no step 'fxup'; STEPS = [first, middle, fixup, last]" in out, out
    code, out = cli("debug", "goto fixup")
    assert "fixup ran 1" in out and "last ran, tries 1" in out and "--- exit code 0" in out, out


def test_reload_with_a_syntax_error_changes_nothing(cli):
    cli("run", "task.py")
    cli.write("lib.py", "def check(s):\n    return (\n")
    code, out = cli("debug", "retry")
    assert "reload failed, nothing changed" in out and "SyntaxError" in out, out
    assert cli.status()["state"] == "paused"
    cli("debug", "q")


def test_an_exception_pauses_post_mortem_and_refuses_c(cli):
    cli.write("task.py", '''\
        def first(r, v, s):
            x = 41
            raise ValueError(f"boom {x}")

        def second(r, v, s):
            print("second ran")

        STEPS = [first, second]
        ''')
    code, out = cli("run", "task.py")
    assert "--- pause 1: exception at task.py:3 in first (step first 1/2)" in out, out
    assert "ValueError: boom 41" in out and 'task.py", line 3, in first' in out
    assert "c, n and s are refused" in out
    for cmd in ("c", "n", "s"):
        code, out = cli("debug", cmd)
        assert "the step raised and cannot go on; use retry, skip, goto STEP or q" in out, out
    assert cli("debug", "p x + 1") == (0, "42\n")
    code, out = cli("debug", "skip")
    assert "second ran" in out and "--- exit code 0" in out, out


def test_wait_returns_while_the_program_runs(cli):
    cli.write("task.py", '''\
        import time

        def nap(r, v, s):
            print("napping", flush=True)
            time.sleep(60)

        STEPS = [nap]
        ''')
    t0 = time.time()
    code, out = cli("run", "task.py", "--wait", "1")
    assert time.time() - t0 < 10 and code == 0
    assert "napping" in out and "--- still running after 1 s (step nap)" in out, out
    code, out = cli("debug", "p 1")
    assert code == 1 and "running, not paused" in out
    code, out = cli("wait", "--wait", "0.5")
    assert "still running" in out and "napping" not in out          # only output since the last report
    os.kill(cli.status()["pid"], signal.SIGTERM)
    code, out = cli("wait", "--wait", "10")
    assert code == 143 and "--- exit code 143" in out, out


def test_break_before_a_step_then_next(cli):
    code, out = cli("run", "task.py", "--break", "middle")
    assert "--- pause 1: break at task.py:9 in middle (step middle 2/3)" in out, out
    assert "before step middle" in out
    code, out = cli("debug", "n")
    assert "--- pause 2: stopped at task.py:10 in middle" in out and "after next" in out, out
    assert sorted(os.listdir(cli.status()["pause_dir"])) == ["message.txt", "state.json"]     # no camera frames
    assert cli("debug", "p s.tries") == (0, "1\n")
    code, out = cli("debug", "c")
    assert "--- pause 3: assert failed at task.py:11" in out, out
    cli("debug", "q")


def test_step_into_a_helper_and_next_past_the_end_of_a_step(cli):
    cli("run", "task.py", "--break", "middle")
    cli("debug", "n")
    code, out = cli("debug", "s")
    assert "--- pause 3: stopped at lib.py:1 in check (step middle 2/3)" in out and "calling check()" in out, out
    code, out = cli("debug", "r")
    assert "--- pause 4: stopped at lib.py:2 in check" in out and "returning from check()" in out, out
    cli("debug", "q")
    cli.write("lib.py", open(cli.dir / "lib.py").read().replace("return False", "return True"))
    code, out = cli("run", "task.py", "--break", "last")
    assert "--- pause 1: break at task.py:15 in last (step last 3/3)" in out, out
    code, out = cli("debug", "n")
    assert "last ran" in out and "--- pause 2: stopped at task.py:15 in last" in out and "returning from last()" in out
    code, out = cli("debug", "n")                           # out of the step: on to the end, no stop in the runner
    assert "--- pause" not in out and "--- exit code 0" in out, out


def test_a_stop_inside_the_robot_client_sends_it_no_requests(cli):
    cli.write("task.py", '''\
        def first(r, v, s):
            t = r.state().t
            print("t is", t)

        STEPS = [first]
        ''')
    cli("run", "task.py", "--break", "first")
    code, out = cli("debug", "s")
    assert "--- pause 2: stopped at" in out and " in state " in out, out
    assert "saved .robot/runs/" in out and "nothing (paused inside the robot client)" in out
    code, out = cli("debug", "c")
    assert "t is 1.5" in out and "--- exit code 0" in out, out


def test_pause_a_running_program_between_robot_requests(cli):
    cli.write("task.py", '''\
        def loop(r, v, s):
            s.n = 0
            while True:
                r._call("sleep", s=0.3)
                s.n += 1

        STEPS = [loop]
        ''')
    code, out = cli("run", "task.py", "--wait", "1")
    assert "still running" in out, out
    code, out = cli("pause")
    assert "--- pause 1: paused at task.py:" in out and "in loop (step loop 1/1)" in out, out
    assert "paused on request (python -m robot pause)" in out
    st = cli.status()
    assert st["kind"] == "pause" and st["line"] in (3, 4, 5)
    assert "top.png" in os.listdir(st["pause_dir"])
    n = int(cli("debug", "p s.n")[1])
    assert cli("debug", "p r.state().t") == (0, "1.5\n")     # the request in progress got its own reply
    code, out = cli("pause")
    assert "paused already" in out and "--- pause 1: paused" in out
    code, out = cli("debug", "c", "--wait", "1")
    assert "still running" in out, out
    code, out = cli("pause")
    assert "--- pause 2: paused at task.py:" in out, out
    assert int(cli("debug", "p s.n")[1]) > n
    code, out = cli("debug", "q")
    assert code == 1 and "--- ended by q at pause 2" in out
    cli("run", "task.py", "--unattended", "--wait", "0.5")
    code, out = cli("pause")
    assert code == 1 and "an --unattended run does not pause" in out
    os.kill(cli.status()["pid"], signal.SIGTERM)


def test_break_at_a_line_and_breakpoint(cli):
    code, out = cli("run", "task.py", "--break", "lib.py:2")
    assert "--- pause 1: breakpoint at lib.py:2 in check (step middle 2/3)" in out, out
    code, out = cli("debug", "c")
    assert "--- pause 2: assert failed at task.py:11" in out, out
    code, out = cli("debug", "retry")                        # the breakpoint holds for the rerun
    assert "--- pause 3: breakpoint at lib.py:2 in check" in out, out
    cli("debug", "q")
    code, out = cli("run", "task.py", "--break", "lib.py:3")
    assert code == 2 and "--break lib.py:3: no breakpoint set there" in out, out
    cli.write("task.py", '''\
        def first(r, v, s):
            x = 5
            breakpoint()
            print("x is", x)

        STEPS = [first]
        ''')
    code, out = cli("run", "task.py")
    assert "--- pause 1: breakpoint at task.py:3 in first" in out and "breakpoint()" in out, out
    cli("debug", "!x = 6")
    code, out = cli("debug", "c")
    assert "x is 6" in out and "--- exit code 0" in out, out


def test_pause_each_step(cli):
    cli.write("lib.py", open(cli.dir / "lib.py").read().replace("return False", "return True"))
    code, out = cli("run", "task.py", "--pause-each-step")
    assert "--- pause 1: after the step at task.py:6 in first (step first 1/3)" in out, out
    assert "c goes on with step middle" in out
    assert cli("debug", "p s.n") == (0, "1\n")
    code, out = cli("debug", "n")                           # between steps: pause at the next step's start
    assert "--- pause 2: break at task.py:9 in middle" in out, out
    code, out = cli("debug", "c")
    assert "--- pause 3: after the step at task.py:12 in middle" in out, out
    assert cli("debug", "p ok") == (0, "True\n")            # the finished step's locals
    code, out = cli("debug", "c")
    assert "--- pause 4: after the step" in out and "it was the last step" in out, out
    code, out = cli("debug", "c")
    assert "--- exit code 0" in out, out


def test_from_only_and_unknown_steps(cli):
    cli.write("abc.py", '''\
        def a(r, v, s):
            print("a ran")

        def b(r, v, s):
            print("b ran")

        def c(r, v, s):
            print("c ran")

        STEPS = [a, b, c]
        ''')
    code, out = cli("run", "abc.py", "--only", "b")
    assert code == 0 and "b ran" in out and "a ran" not in out and "c ran" not in out, out
    code, out = cli("run", "abc.py", "--from", "b")
    assert code == 0 and "b ran" in out and "c ran" in out and "a ran" not in out, out
    code, out = cli("run", "abc.py", "--from", "x")
    assert code == 2 and "no step 'x'; STEPS = [a, b, c]" in out, out
    code, out = cli("run", "abc.py", "--break", "bb")
    assert code == 2 and "no step 'bb'" in out, out


def test_one_run_at_a_time(cli):
    cli("run", "task.py")
    code, out = cli("run", "task.py")
    assert code == 1 and "is still paused" in out and "python -m robot debug q" in out, out
    cli("debug", "q")
    code, out = cli("run", "task.py", "--unattended")
    assert "AssertionError" in out


def test_a_program_named_like_a_standard_module(cli):
    cli.write("inspect.py", """\
        def look(r, v, s):
            print("look ran")
        STEPS = [look]
        """)
    code, out = cli("run", "inspect.py", "--unattended")
    assert code == 0, out
    assert "look ran" in out


def test_a_program_that_dies_says_why(cli):
    cli.write("bad.py", """\
        import os
        def die(r, v, s):
            print("about to die")
            os._exit(3)
        STEPS = [die]
        """)
    code, out = cli("run", "bad.py", "--unattended")
    assert code != 0
    assert "about to die" in cli.status()["message"], cli.status()


def test_an_edited_file_is_sent_to_the_driver_again(cli):
    cli.write("edit.py", """\
        def one(r, v, s):
            r.state()

        def two(r, v, s):
            r.state()
        STEPS = [one, two]
        """)
    code, out = cli("run", "edit.py", "--pause-each-step")
    assert "pause" in out, out
    src = (cli.dir / "edit.py").read_text().replace("def two(r, v, s):", "def two(r, v, s):\n    print('edited')")
    (cli.dir / "edit.py").write_text(src)
    assert cli("debug", "reload")[0] == 0
    code, out = cli("debug", "c")
    assert "edited" in out, out
    sent = [t[p] for t in cli.drv.src_text for p in t if p.endswith("edit.py")]
    assert len(sent) == 2 and "edited" not in sent[0] and "edited" in sent[1], sent


def test_a_post_mortem_pause_selects_the_programs_own_frame(cli):
    cli.write("pm.py", """\
        import json

        def parse(r, v, s):
            local_text = "not json"
            json.loads(local_text)
        STEPS = [parse]
        """)
    code, out = cli("run", "pm.py")
    assert "pm.py:5 in parse" in out, out
    code, out = cli("debug", "p local_text")
    assert "'not json'" in out, out
    assert cli("debug", "q")[0] in (0, 1)


def test_the_program_files_are_kept_at_the_start_and_at_each_reload(cli):
    code, out = cli("run", "task.py")
    assert "pause" in out, out
    cli.write("lib.py", (cli.dir / "lib.py").read_text().replace('return "old"', 'return "new"'))
    assert cli("debug", "reload")[0] == 0
    run = cli.dir / ".robot" / "current"
    idx = [json.loads(x) for x in (run / "src" / "index.jsonl").read_text().splitlines()]
    assert [e["event"] for e in idx] == ["start", "reload"], idx
    before, after = idx[0]["files"]["lib.py"], idx[1]["files"]["lib.py"]
    assert before != after and idx[0]["files"]["task.py"] == idx[1]["files"]["task.py"]
    assert 'return "old"' in (run / "src" / f"{before}.py").read_text()
    assert 'return "new"' in (run / "src" / f"{after}.py").read_text()
    assert idx[1]["step"] == "middle" and idx[1]["pause"] == 1
