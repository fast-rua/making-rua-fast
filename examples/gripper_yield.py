"""Gripper only (the arm does not move): the jaws resist up to --yield N (default 5), push or pull. Squeeze them or pull
them apart harder than that and they move with you, faster the harder you push, so what you feel stays about --yield N:
a force cap that acts as a damper. Let go and they stay where you left them.

    ROBOT_ADDR=unix:/tmp/robot.sock python examples/gripper_yield.py --arm left             # Ctrl-C to stop

The jaw-force reading has no dependable zero, so it goes through the usual processing before it means "a push":
- tare: zero = the median while the jaws are known to be untouched and still (at the start, and after every push);
  while they are still and untouched the zero also follows the reading slowly (--track s), because a loaded screw
  relaxes and the reading creeps by a few N;
- low-pass: about 30 ms, well above the reading's noise (~0.2 N);
- hysteresis and debounce: a push starts above --yield N held for 40 ms and ends below --release N, so a 2-3 N
  wander never triggers anything and one noisy sample cannot flip the state; changing direction goes through a hold;
- friction compensation: at the start the jaws close and open 5 mm once, to learn what the reading shows from friction
  alone in each direction; that is subtracted while the policy moves them, or yielding would read as more push;
- the jaws must give way too: a push presses them --press mm past their target in its direction (5 N is ~0.08 mm on
  this gripper), and while the policy moves them a push in the other direction counts only if they move that way. On
  the cell the screw sticks: starting to move after a squeeze reads up to ~50 N with the jaws standing still.

Effort convention (the API's): positive = the motor pushing the jaws open. Squeezing them by hand reads positive (the
motor pushes back), pulling them apart reads negative.
"""
import argparse
import time

import numpy as np
import robot

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left", choices=["left", "right"])
ap.add_argument("--home", type=float, default=60.0, help="opening to start at, mm")
ap.add_argument("--min", type=float, default=15.0, help="never close below this, mm")
ap.add_argument("--max", type=float, default=88.0, help="never open beyond this, mm")
ap.add_argument("--yield", dest="yield_n", type=float, default=5.0, help="give way above this push or pull, N")
ap.add_argument("--release", type=float, default=2.5, help="the push has ended below this, N")
ap.add_argument("--gain", type=float, default=3.0, help="speed per N above --yield, mm/s per N")
ap.add_argument("--max-speed", type=float, default=30.0, help="fastest yield, mm/s")
ap.add_argument("--track", type=float, default=1.5, help="the zero follows a still, untouched reading this slowly, s")
ap.add_argument("--press", type=float, default=0.04, help="a push moves the jaws this far past their target, mm")
args = ap.parse_args()

r = robot.connect(name="gripper_yield")
arm = args.arm
DT = 0.01                                     # 100 Hz loop
home, w_min, w_max = args.home / 1000, args.min / 1000, args.max / 1000


def read():
    g = r.state().grippers[arm]
    return g["width"], g["effort"]


def tare(seconds):
    """Zero = median reading while untouched and still; also returns the noise."""
    vals = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        vals.append(read()[1])
        time.sleep(DT)
    return float(np.median(vals)), float(np.std(vals))


def move(width, speed=0.1, wait=True):
    return r.gripper(arm, float(np.clip(width, w_min, w_max)), speed=speed, wait=wait)


# --- set up: go to the start opening, learn the zero and the friction while moving
print(f"opening to {args.home:.0f} mm")
move(home)
time.sleep(0.5)
zero, noise = tare(1.0)
print(f"zero {zero:+.1f} N (noise {noise:.2f} N)")
friction = {}
for direction, target in (("close", home - 0.005), ("open", home)):
    m = move(target, speed=args.max_speed / 1000 / 2, wait=False)
    vals = []
    while not m.done:
        vals.append(read()[1] - zero)
        time.sleep(DT)
    friction[direction] = float(np.median(vals[len(vals) // 4:])) if len(vals) > 4 else 0.0
time.sleep(0.4)
zero, noise = tare(0.6)
print(f"friction while moving: closing {friction['close']:+.1f} N, opening {friction['open']:+.1f} N; zero {zero:+.1f} N")
print(f"push or pull the jaws: they give way above {args.yield_n:.1f} N")

# --- run. state: hold | close | open (yielding that way)
state, target, f_filt, above_since, last_print = "hold", home, 0.0, None, 0.0
alpha = DT / 0.03                              # ~30 ms low-pass
w_prev, v_filt = read()[0], 0.0                # jaw speed, + opening
try:
    while True:
        t0 = time.time()
        width, effort = read()
        v_filt += DT / 0.05 * ((width - w_prev) / DT - v_filt)
        w_prev = width
        moving = state if state in ("close", "open") else None
        f = effort - zero - (friction[moving] if moving else 0.0)           # + squeezed closed, - pulled open
        f_filt += alpha * (f - f_filt)

        # a push in a direction: a reading beyond --yield AND the jaws giving way that way. While the policy moves them,
        # a push the other way counts only if they really move that way (a high reading alone is the screw sticking)
        if state == "open":
            want_close = f_filt > args.yield_n and v_filt < -0.003
        else:
            want_close = f_filt > args.yield_n and target - width > args.press / 1000
        if state == "close":
            want_open = f_filt < -args.yield_n and v_filt > 0.003
        else:
            want_open = f_filt < -args.yield_n and width - target > args.press / 1000
        direction = "close" if want_close else "open" if want_open else None
        above_since = (above_since or t0) if direction else None
        still_pushing = (state == "close" and f_filt > args.release) or (state == "open" and f_filt < -args.release)

        if still_pushing or (direction and t0 - above_since >= 0.04 and state == "hold"):
            state = state if still_pushing else direction
            speed = min(args.gain * max(abs(f_filt) - args.yield_n, 0.0) + 2.0, args.max_speed) / 1000
            target = float(np.clip(target + (speed if state == "open" else -speed) * DT, w_min, w_max))
            move(target, wait=False)
        elif state in ("close", "open"):       # the push ended: hold where the jaws are, re-tare there
            state, target = "hold", width
            move(target, wait=False)
            time.sleep(0.3)
            zero, noise = tare(0.3)            # friction locks in a few N at the new spot: that is the zero now
            f_filt = 0.0
        if state == "hold" and abs(v_filt) < 0.001 and abs(f_filt) < args.release:
            zero += DT / args.track * (effort - zero)                       # follow slow creep, never a push
        if t0 - last_print >= 0.2:
            print(f"  {state:6s} width {width * 1000:5.1f} mm  target {target * 1000:5.1f} mm  "
                  f"push {f_filt:+5.1f} N  (reading {effort:+5.1f}, zero {zero:+5.1f})")
            last_print = t0
        time.sleep(max(0.0, DT - (time.time() - t0)))
except KeyboardInterrupt:
    print("stopping: the jaws hold where they are")
