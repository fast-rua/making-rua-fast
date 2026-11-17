# Playbook: how the runs that worked used this API

This section is a worked guide to the kernel, the action queue and the pushed reports, taken from the real-cell
episodes that went best (cups-01: three cups, the pear moved under the empty cup and checked, 122 s; towel-fold-13,
-15 and -16: the towel spread and folded once). Everything here is also in the sections below; this part shows how the
pieces fit together, in the order an episode uses them, with the code those runs sent.

## 1. The shape of an episode

- **One step is one tool call of yours, plus the model round trip that writes it.** The arms run on the wall clock
  between your steps. Your output is slow to write (each line of code you send costs time while nothing you have not
  queued moves), so the episodes that went well kept the arms busy with actions queued ahead, and wrote short code by
  defining helpers once and calling them by name.
- **Everything runs in one Python kernel.** Every command you send is `python -m robot.kernel` with your code on
  stdin. Variables, functions and imports stay between calls: define `P`, `review` and your actions once, then reuse
  them.
- **An action is a function `fn(r)` that you queue.** `kernel.queue(fn)` returns at once; the action runs in the
  kernel while you keep working. When it ends, a report comes into your turn by itself (`[action fn] done in 9.3 s`,
  the queue, the arms' state, a camera grid).
- **The cycle that worked:** look, then measure, then check reach, then queue an action, then, while it runs, measure
  and check the next one and queue it too. Every report is the moment to adjust what waits. When the task is done:
  look once, release, move clear, `r.finish(summary)`, and stop.

## 2. Running code: the exact form

Send every command like this, with a long timeout on your shell tool (60 s; `yield_time_ms: 60000` in Codex):

```bash
python -m robot.kernel <<'PY'
r = robot.connect()
print(r.state().summary())
PY
```

- `robot`, `np`, `cv2`, `viz`, `watch`, `when` and `kernel` are already imported in the kernel.
- A call returns when its code ends. `kernel.queue(...)` returns at once, so a call that only queues comes back in
  well under a second.
- At its end, each call prints `[state] ...` (where each arm is, each gripper's width and whether it holds
  something or closed on nothing) and `[views] [...]` (the images it showed). Read them before your next step instead of asking again.
- **Inside a queued action, use the `r` it is given.** In your own calls, `r = robot.connect()` makes a connection
  for checks and pictures. Queries never disturb a running action: `r.state()`, `r.cameras()`, `r.ik`,
  `r.reachable`, `r.move(..., dry_run=True)`, `viz.*`.
- **A command that moves the arms, sent from your own call rather than from an action, stops the running action
  first** and drops the queue. Put every motion inside an action.
- Each call closes its robot connections as it ends, and their motions stop. A `wait=False` move started in your own
  call does not outlive that call. Queue an action instead.

## 3. The first steps (about 15 s in cups-01)

1. **Read the code once, in one command.** One `cat` of the files you need (`robot/__init__.py`, `robot/kernel.py`,
   `robot/viz.py`) costs one step. Reading them a few hundred lines at a time costs one step each: bottle-01 spent
   about 40 s before its first motion doing that.
2. **Connect and look.**
   ```python
   r = robot.connect()
   print(r.state().summary())
   viz.show_cameras('initial', r, width=800)       # every camera as one labelled grid: one image
   ```
3. **Find the objects as numbers, not by eye.** A colour mask, its connected components, and each component's 3D
   points from depth give positions in metres that you can use directly:
   ```python
   f = r.cameras()['front']
   h = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2HSV)
   mask = ((h[..., 0] < 12) | (h[..., 0] > 170)) & (h[..., 1] > 130) & (h[..., 2] > 60)     # red
   n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
   for i in range(1, n):
       if stats[i, 4] > 200:                                   # pixels
           p = f.points(lab == i)                              # N x 3 world points (m)
           print(i, np.quantile(p, [.1, .5, .9], axis=0).round(3).tolist())
   viz.marks(f, 'cups', mask=mask, on_table=True, step=25, max_marks=25)   # numbered points with their xyz
   ```
   The 10 %, 50 % and 90 % quantiles give the object's extent and centre. The 90 % z is its top, so a cup's grasp
   height is below that.
4. **Define your helpers now, once:**
   ```python
   def P(x, y, z, pitch=0.0):                  # TCP pose, jaws pointing down, tilted by pitch (rad) about y
       return robot.pose([x, y, z], rpy=[np.pi, pitch, np.pi])

   def review(arm, poses, name, width=.09):    # the arm's ghost at each pose on the front camera, and 3D views
       fs = r.cameras()
       ov = viz.Overlay(r, camera='front', frame=fs['front'])
       for T in poses:
           ov.ghost(arm, T, width=width)
       sc = viz.View3D(r, center=poses[0][:3, 3], radius=.13, frames=fs, size=(320, 240))
       sc.ghost(arm, poses[0], width=width)
       viz.show(name, viz.grid([ov.render(), sc.render(views=['jaws', 'along', 'approach', 'left'])], width=1000))
   ```

## 4. Before queuing: check reach in one call

Check a whole set of candidate poses at once. Moving to one pose and reading the refusal costs a step each time.

```python
cands = [P(.405, .33, .085, pitch) for pitch in (0, .25, .5, -.25, -.5)]
out = r.reachable(cands, ee='left')                  # each checked as r.move would check it
for T, o in zip(cands, out):
    print(o['ok'], o['why'], round(o['margin']['limit_rad'], 2))
G = next(T for T, o in zip(cands, out) if o['ok'])
```

- `why` is the refusal that move would get: the fence, the gripper floor, IK (with the miss), or the arms'
  separation with the other arm where it is commanded now. `line=True` also checks the straight line from where the
  arm is. Rank the ones that pass by `margin` (`limit_rad`: distance to the nearest joint limit).
- Check the approach (A), the grasp (G) and the lift (L) together: `r.reachable([A, G, L], ee='left')`.
- For a hard-to-undo motion, also look at it: `review('left', [A, G, L], 'cup1')`. The 3D views show how deep the
  jaws go and what they touch.
- Jaws straight down reach the table x 0.2–0.5 m in front of a base at TCP heights up to about 0.10 m. Higher than
  that, tilt the jaws about 0.3–0.5 rad (`pitch`) toward +x.

## 5. Writing an action

An action is a short sequence of moves for one purpose: approach, grasp, lift. Name it for what it does; the name
shows in every report.

```python
def lift1(r):
    r.gripper('left', .093)                          # open
    r.move(A, ee='left', duration=2)                 # above the cup
    r.move(G, ee='left', line=True, duration=2)      # straight down onto it
    r.gripper('left', w - .003, speed=.02, force=30) # w: the object's measured width
    r.move(L, ee='left', line=True, duration=2)      # straight up
kernel.queue(lift1)
```

- **Every move names its arm:** `ee='left'` (or `joints=[...]`). A move without one is refused
  (`give exactly one of joints= ... or ee= ...`). towel-fold-16's first action failed on exactly this.
- **Gripper width sets the squeeze.** For a rigid object, close to its measured width minus 3 mm; the jaws then
  press with about 90 N per mm they go into the object. Closing to 0 on a cup or a pear ended with
  `gripper_overload` twice in cups-01. For cloth or thin material, closing to 0 is right (the towel holds at
  about 3 mm). A report's `grip=51mm (stalled: holding something)` confirms a hold. `(closed on nothing)` means the
  jaws ended at the empty-jaw width (`r.description["gripper_empty_width_m"]`, 2.9 mm on the real gripper, + 0.5 mm):
  a miss, or a pinch of cloth too thin to tell apart, so look before you lift. The real arms report no `contact`
  (their force estimate is not trusted).
- **`line=True` for the last few centimetres** in and out, so the jaws come straight down and go straight up instead
  of sweeping sideways through the object.
- **`duration=2` to `3` s per move** is what those runs used. Faster moves knock things over; slower ones waste
  time.
- **A check at the top of an action stops it before it moves** when the situation is not what it was planned for:
  ```python
  def set2(r):
      if r.state().grippers['right']['width'] < .035:
          raise RuntimeError('cup grasp uncertain')
      r.move(H2, ee='right', duration=2)
      ...
  ```
- **Code inside an action runs later, on fresh readings.** `r.state()` and `r.cameras()` inside `fn` read the world
  as the action starts. A move that depends on where something landed can compute its target there.

## 6. Pipelining: keep the arms busy

In cups-01 the arms were moving through most of the episode, because the next action was usually queued before the
running one ended. The pattern, from that run:

```
24.5 s  queue lift1                      (lift the left cup)
27.6 s  review where cup 1 goes down     (while lift1 runs)
30.4 s  review the place, and cup 2's grasp
31.7 s  report: lift1 error gripper_overload
34.9 s  queue [rescue1, lift2]           (fix the hold, then the next cup: two actions, one step)
41.4 s  review the middle cup; queue restore_ball
46.6 s  queue [set2, lift3]              (three actions waiting: nothing more to write)
47.7 s  report: lift2 done; restore_ball started; set2 and lift3 wait
```

- **On every report or look, read the `[queue]` lines first.** `running #14 grasp: 3.1 s in` tells you what moves
  now. `waiting 1: #15 lift` tells you what follows. `empty behind it` means the arms stop after this action unless
  you queue something.
- **Then decide, in one step:**
  - The plan still holds: write the action after the last one queued and `kernel.queue` it. Do this even while
    the running action is unfinished, because you know where its code leaves the arms.
  - Something waiting no longer fits: `kernel.queue(new, replace_from=15)` replaces #15 and everything behind it.
  - Nothing waiting fits: `kernel.flush()`, then queue what does.
  - It is going wrong right now: `kernel.action().stop()` halts the arms where they are and drops the queue.
- **At most three actions wait.** With three waiting and all still right, end your turn. That costs the arms
  nothing, and the next report or look starts your next turn.
- **Several actions in one step:** `kernel.queue([f, g])`. cups-01 did this after every recovery.
- **What you see is older than what is happening.** A picture shows the moment it was taken. Plan from where the
  queued actions will leave the arms, which you know from their code.
- **Check the next action from there too.** `r.reachable(cands, ee, after=kernel.planned_end())` (and `r.ik`, a
  ghost's `after=`) solves IK from where the running and queued actions leave the arms, and checks the other arm there
  for separation. `planned_end()` knows the last move the running action has sent each arm so far and what queued
  actions declared with `kernel.queue(fn, ends={"left": L})`; give `after={"left": L}` yourself when you know better.
  Without `after`, a check sees the arms where they are commanded now (and the other arm where its running move ends).
- **An error or a stop drops everything queued** (`[queue] dropped: ...`) and the arms hold. Read the error, queue
  the fix and the rest of the plan again, in one step.
- **Do not wait, sleep or poll.** Each of those is a step that only waits; the reports come by themselves. With
  nothing running or queued, ending your turn ends the episode.

## 7. Hooks: let every report carry what you need

If you keep fetching the same measurement after motions, make it a hook. It then runs at the end of every call that
moved the arms, and what it prints comes with the report:

```python
def pear_position(r, frames):
    f = frames['front']; h = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2HSV)
    m = (h[..., 1] > 35) & ((h[..., 0] > 140) | (h[..., 0] < 15))
    p = f.points(m); p = p[(p[:, 2] > .025) & (p[:, 2] < .10)]
    if len(p) > 15:
        print('pear', np.quantile(p, [.1, .5, .9], axis=0).round(3).tolist())
kernel.after_motion(pear_position)
# ... later, when it is no longer needed:
kernel.remove_hook('pear_position')
```

Keep hooks quick and their output short. Printed numbers cost little; every image slows every later step.

## 8. When something fails

| Report says | What it means | What worked |
|---|---|---|
| `InvalidCommand: give exactly one of joints= ... or ee=` | a move without `ee=` | add `ee='left'` to every move |
| `target unreachable (closest IK solution misses by N mm)` | out of reach at that orientation | `r.reachable` over pitches and heights, then queue the best |
| `separation: the goal puts the arms' shapes N cm into each other` | the two arms would come too close | move the other arm clear first (its own action, queued before), or approach from the other side; `r.reachable` reports this. `... once right gets where its running move_to (command N) ends`: the clash is with where the other arm is going; give `after=` if it moves on before this move runs |
| `fence: ...` | outside the arm's box or below the gripper floor | stay over the table, keep the TCP higher or tilt less |
| `gripper aborted: gripper_overload` | squeezed too hard (closed far past a rigid object) | reopen to the object's width (`r.gripper(arm, w)`), then close to `w - .003` |
| the object moved or fell after a move | the jaws or the arm pushed it | approach from straight above with `line=True`, open wider, and keep the other hand away from it |

A tall, light object (a bottle, a cup) tips over when a finger or the arm brushes it. In bottle-01 it fell twice,
both times while a hand moved around it. Come down from directly above it, open wider than it, and do not slide a
hand around it once it is between the jaws.

## 9. Finishing

1. Check the result with one picture (`viz.show_cameras('check', r, width=800)`) or the numbers from your hook.
2. Release and move clear in one queued action:
   ```python
   def release(r):
       r.gripper('left', .093)
       r.move(CH, ee='left', line=True, duration=2.5)
       r.move(E, ee='left', duration=2)
   kernel.queue(release)
   ```
3. When its report comes, call `r.finish('what you did and what you see now')` in a call of its own, then stop.
   Do not call `r.finish` while an action is still running or queued.

## 10. A whole episode, condensed (cups-01)

```python
# step 1: look
r = robot.connect(); print(r.state().summary()); viz.show_cameras('initial', r, width=800)
# step 2: find the cups (mask, components, quantiles, marks), as in section 3
# step 3: helpers P and review, as in section 3; check the first grasp
G = P(.405, .33, .085, .5); A = P(.405, .33, .18, .5); L = P(.405, .33, .25, .5)
print([(o['ok'], o['why']) for o in r.reachable([A, G, L], ee='left')]); review('left', [A, G, L], 'cup1')
# step 4: queue it
def lift1(r):
    r.gripper('left', .093); r.move(A, ee='left', duration=2); r.move(G, ee='left', line=True, duration=2)
    r.gripper('left', w - .003, speed=.02, force=30); r.move(L, ee='left', line=True, duration=2)   # w: cup width
kernel.queue(lift1)
# steps 5-6, while lift1 runs: measure and check the next cup, queue lift2 behind it
# on each report: read [queue], keep or replace what waits, queue the next action
# end: one look, release and clear in one action, then r.finish(...) in its own call
```
