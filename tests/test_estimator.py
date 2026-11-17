"""The external-force estimate while the arm moves (momentum observer), and the gripper overload guard."""
import numpy as np

from robotd.backends.mujoco_sim import MujocoYamBackend
from robotd.robots.yam_station import scene as S
from robotd.robots.yam_station.description import description
import tasks
from robotd.server import Driver

LEFT = S.ARM_JOINTS["left"]
LOWEST = [1.0]                                          # lowest TCP height seen in run(): the table is at z = 0


def driver():
    cfg = tasks.get("empty").scene()
    return Driver(MujocoYamBackend(cfg), description())


def static_force(d, arm="left"):
    """What the old estimate read: gravity minus measured torque, through the Jacobian (environment on the robot)."""
    c = d.c
    qd = {n: float(c.m.q[i]) for i, n in enumerate(c.joints)}
    g = c.kin.bias(qd, None)
    tau = np.array([g[n] - float(c.m.tau[c.jidx[n]]) for n in c.groups[arm]])
    J = c.kin.jac(arm, c._arm(c.m.q, arm))
    return np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), J @ tau)[:3]


def run(d, spec, ticks, push=None):
    """Tick a motion; return the estimated force on the gripper (environment on the robot) and the static one."""
    d.handle({"op": "command", "spec": {**spec, "wait": False}})
    est, old = [], []
    b = d.backend
    body = b.model.body("left_gripper").id
    tcp = b.model.site(S.TCP_SITES["left"]).id
    for _ in range(ticks):
        if push is not None:                            # a force at the TCP (a hand nudging the gripper)
            p = np.asarray(push, float)
            b.data.xfrc_applied[body, :3] = p
            b.data.xfrc_applied[body, 3:] = np.cross(b.data.site_xpos[tcp] - b.data.xipos[body], p)
        d.c.tick()
        est.append(-d.c.wrench("left")[:3])
        old.append(static_force(d))
        LOWEST[0] = min(LOWEST[0], float(b.data.site_xpos[tcp][2]))
    b.data.xfrc_applied[body, :] = 0
    return np.array(est), np.array(old)


def fast_move(d, dq=(0.8, -0.4, 0.5, 0.6, 0.0, 0.0)):
    q0 = np.array([d.c.m.q[d.c.jidx[n]] for n in LEFT])
    return {"op": "move_to", "joints": LEFT, "target": (q0 + np.array(dq)).tolist(), "speed": 1.5}


def test_the_estimate_holds_while_the_arm_moves_fast():
    d = driver()
    for _ in range(50):
        d.c.tick()
    d.c.zero_force("left")
    LOWEST[0] = 1.0
    est, old = run(d, fast_move(d), 500)                    # 2 s: accelerate, cruise, brake
    assert LOWEST[0] > 0.08
    worst_new, worst_old = np.max(np.linalg.norm(est, axis=1)), np.max(np.linalg.norm(old - old[0], axis=1))
    print(f"no contact, fast move: observer max {worst_new:.2f} N, static estimate max {worst_old:.2f} N")
    assert worst_new < 1.0 and worst_old > 3 * worst_new


def test_it_reads_a_known_force_still_and_moving():
    d = driver()
    for _ in range(50):
        d.c.tick()
    d.c.zero_force("left")
    F = np.array([3.0, -2.0, -5.0])                         # pushed down, forward and to the side
    est, _ = run(d, fast_move(d, (0, 0, 0, 0, 0, 0)), 250, push=F)    # holding still
    still = est[-50:].mean(axis=0)
    for _ in range(50):                                     # let go, let the estimate settle, zero in free space
        d.c.tick()
    d.c.zero_force("left")
    LOWEST[0] = 1.0
    est, _ = run(d, fast_move(d, (-0.6, -0.3, 0.3, 0.3, 0, 0)), 500, push=F)    # pushed while it moves fast
    assert LOWEST[0] > 0.08, f"the gripper came down to z = {LOWEST[0]:.3f} m: that is the table, not free space"
    moving = est[60:450]
    err_still = np.linalg.norm(still - F)
    err_moving = np.max(np.linalg.norm(moving - F, axis=1))
    print(f"applied {F} N: still {np.round(still, 2)} (error {err_still:.2f} N), moving worst error {err_moving:.2f} N")
    assert err_still < 0.6 and err_moving < 1.5


def test_a_jammed_gripper_is_relaxed_and_rearmed_by_its_next_command():
    d = driver()
    b = d.backend
    real_read = b.read
    fi = d.c.jidx[S.FINGER_JOINTS["left"][0]]

    def jammed():                                           # the gripper motor pushes 120 N (limit 50 N)
        m = real_read()
        m.tau[fi] = -120.0
        return m
    b.read = jammed
    writes = []
    real_write = b.write
    b.write = lambda q, dq, kp, kd, lim: (writes.append(np.array(kp)), real_write(q, dq, kp, kd, lim))
    for _ in range(100):                                    # 0.4 s
        d.c.tick()
    assert d.c.grip_relaxed["left"] and writes[-1][fi] == 0.0
    assert any(e["kind"] == "fault" and "relaxed" in e["text"] for e in d.handle({"op": "events", "since": 0})["events"])
    assert d.handle({"op": "state"})["grippers"]["left"]["relaxed"] is True
    b.read = real_read
    d.handle({"op": "command", "spec": {"op": "gripper", "name": "left", "width": 0.04}})
    assert not d.c.grip_relaxed["left"] and writes[-1][fi] > 0.0
