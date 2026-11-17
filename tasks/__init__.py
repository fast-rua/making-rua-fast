"""Tasks: a scene, the instruction the agent receives, and a privileged success check.

The check runs when the agent calls r.finish() and is written to the run record (verdict.json); it is never returned
to the agent. This package stays outside the agent's sandbox, so its source (scenes, object placements, success
predicates) is not readable by the policy either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    name: str
    robot: str                                  # e.g. "yam_station"
    scene: Callable                             # () -> the backend's scene config
    instruction: str | None                     # what the agent is told (None: no task, free play or tests)
    evaluate: Callable | None = None            # (backend) -> {"success": bool, ...}
    text: Callable | None = None                # (seed) -> the instruction, for a task whose text names what the seed
                                                #   drew (draw_shape's shape)
    ends_at_finish: bool = False                # the check covers the episode up to r.finish(): an episode's verdict
                                                #   is the one taken at r.finish() (in real time the world runs on
                                                #   after it), and the check at the end gets no settling
    realtime: bool = False                      # episodes run the simulation on the wall clock by default (a dynamic
                                                #   task: in lockstep a program could stop a ball in mid-air)

    def instruction_for(self, seed: int | None = None) -> str | None:
        """What the agent is told for this seed (a scene left without a seed builds seed 0)."""
        return self.text(0 if seed is None else seed) if self.text else self.instruction


def all_tasks() -> dict:
    from tasks import yam_station
    return dict(yam_station.TASKS)


def get(name: str) -> Task:
    tasks = all_tasks()
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; tasks: {sorted(tasks)}")
    return tasks[name]


def scene_args(pairs) -> dict:
    """KEY=VALUE strings (e.g. from --scene-arg noise=0) as keyword arguments for a task's scene(): numbers become
    numbers, true/false booleans."""
    out = {}
    for pair in pairs or []:
        key, _, text = pair.partition("=")
        low = text.strip().lower()
        if low in ("true", "false"):
            out[key.strip()] = low == "true"
            continue
        try:
            out[key.strip()] = int(text)
        except ValueError:
            try:
                out[key.strip()] = float(text)
            except ValueError:
                out[key.strip()] = text
    return out
