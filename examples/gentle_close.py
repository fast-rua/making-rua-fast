"""Gripper only (the arm does not move): close slowly on an object, stop the moment the jaws touch it, hold it with as
little force as the jaws allow, and optionally squeeze up to a target force in small steps while you watch.

    ROBOT_ADDR=unix:/tmp/robot.sock python examples/gentle_close.py --arm left
    ROBOT_ADDR=unix:/tmp/robot.sock python examples/gentle_close.py --arm left --squeeze 15 --hold 10

Hold an object between the open jaws by its far end (fingers outside the jaws), or stand it between them. It must be
at least 15 mm narrower than the opening.

How it works:
- **Learn free motion first.** The jaws close freely for 10 mm while the policy samples the jaw-force reading. On a
  real gripper that reading is never zero while moving: the screw drive's friction shows up in it.
- **Watch for contact.** The jaws close on, and the driver checks every 4 ms whether the reading has stayed above that
  free-motion level (plus a margin) for 20 ms. The motion stops at that tick.
- **Hold at the contact width.** The driver's target is then set back to where the jaws actually stopped. The real
  jaws are very stiff (i2rt's gains: every mm the target runs past the object adds roughly 90 N of squeeze), so
  holding on the contact width is what keeps the grip gentle.
- **Squeeze (optional).** --squeeze N steps the target in --step mm at a time until the reading reaches N, or the
  object has given 5 mm. On the real jaws 0.02 mm is about 2 N.
"""
import argparse
import time

import numpy as np
import robot
from robot import when

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left", choices=["left", "right"])
ap.add_argument("--open", type=float, default=0.08, help="opening to start from, m")
ap.add_argument("--speed", type=float, default=0.005, help="closing speed of the opening, m/s")
ap.add_argument("--margin", type=float, default=4.0, help="contact = free-motion reading + this, N")
ap.add_argument("--squeeze", type=float, default=0.0, help="then squeeze up to this jaw force, N (0: just hold)")
ap.add_argument("--step", type=float, default=0.02, help="squeeze step, mm of opening")
ap.add_argument("--hold", type=float, default=5.0, help="hold this long at the end, s")
args = ap.parse_args()

r = robot.connect(name="gentle_close")
arm = args.arm


def grip():
    g = r.state().grippers[arm]
    return g["width"], abs(g["effort"])


def watch(motion, label, every=0.1):
    """Print width and force while a motion runs; return the samples."""
    rows, last = [], 0.0
    while not motion.done:
        w, f = grip()
        rows.append((w, f))
        if time.time() - last >= every:
            print(f"  {label}: width {w * 1000:5.1f} mm   jaw force {f:5.1f} N")
            last = time.time()
        time.sleep(0.02)
    return rows, motion.result()


print(f"opening to {args.open * 1000:.0f} mm")
r.gripper(arm, args.open)
time.sleep(0.5)

# 1. free motion: what the reading looks like with nothing between the jaws
rows, res = watch(r.gripper(arm, args.open - 0.010, speed=args.speed, wait=False), "free")
moving = [f for w, f in rows[len(rows) // 4:]]                 # skip the start-up transient
base = float(np.percentile(moving, 95)) if moving else 0.0
thr = base + args.margin
print(f"free motion: reading up to {base:.1f} N -> contact above {thr:.1f} N held 20 ms")

# 2. close on, stop at contact (the driver evaluates the condition every 4 ms)
contact = (abs(when.gripper(arm).effort) > thr).held(0.02)
rows, res = watch(r.gripper(arm, 0.0, speed=args.speed, until={"contact": contact}, wait=False), "closing")
print(f"closing ended: {res.status}, {res.reason}")
if res.reason != "contact":
    r.gripper(arm, args.open)
    raise SystemExit("no contact felt: the jaws closed fully (nothing between them, or the margin is too high)")

# 3. hold at the width the jaws actually stopped at
time.sleep(0.1)
w_c, f_c = grip()
r.gripper(arm, w_c, speed=0.002)
time.sleep(0.5)
w_h, f_h = grip()
print(f"contact at {w_c * 1000:.1f} mm ({f_c:.1f} N at the stop); holding at {w_h * 1000:.1f} mm with {f_h:.1f} N")

# 4. optional squeeze: --step mm at a time up to the target force
if args.squeeze > 0:
    target = w_c
    while True:
        w, f = grip()
        if f >= args.squeeze or w_c - w > 0.005 or target < 0.002:
            break
        target -= args.step / 1000
        r.gripper(arm, target, speed=0.002)
        time.sleep(0.1)
        w, f = grip()
        print(f"  squeeze: target {target * 1000:5.1f} mm   width {w * 1000:5.1f} mm   jaw force {f:5.1f} N")
    w, f = grip()
    print(f"squeezed to {f:.1f} N; the object gave {(w_c - w) * 1000:.1f} mm "
          f"({(w_c - w) * 1000 / max(f - f_h, 1e-6):.2f} mm per N above the hold)")

# 5. hold and report, then let go
t_end = time.time() + args.hold
while time.time() < t_end:
    w, f = grip()
    print(f"  holding: width {w * 1000:5.1f} mm   jaw force {f:5.1f} N")
    time.sleep(0.5)
print(f"opening to {args.open * 1000:.0f} mm")
r.gripper(arm, args.open)
