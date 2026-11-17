"""Compile the JSON condition trees built by `robot.when` into per-command evaluators run every fast tick.

A compiled condition keeps its own state (`held`, `after`), so one is compiled per command. Evaluation reads a
TickContext, a snapshot of measured and reference quantities filled by the controller each tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2, "rx": 0, "ry": 1, "rz": 2}


class ConditionError(ValueError):
    pass


@dataclass
class TickContext:
    t: float = 0.0
    t0: float = 0.0
    progress: float = 0.0
    T: dict = field(default_factory=dict)          # ee -> measured TCP pose (4x4)
    T_ref: dict = field(default_factory=dict)      # ee -> reference TCP pose
    T0: dict = field(default_factory=dict)         # ee -> TCP pose when the command started
    v: dict = field(default_factory=dict)          # ee -> TCP linear velocity (3)
    wrench: dict = field(default_factory=dict)     # ee -> force/torque the tool applies (6, world)
    q: dict = field(default_factory=dict)
    q_ref: dict = field(default_factory=dict)
    dq: dict = field(default_factory=dict)
    tau_ext: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)
    grippers: dict = field(default_factory=dict)   # name -> {"width", "effort", "stalled"}
    signals: dict = field(default_factory=dict)


def _vec(v3: np.ndarray, axis, frame: str, R_tool: np.ndarray | None) -> float:
    v3 = np.asarray(v3, float)
    if frame == "tool":
        if R_tool is None:
            raise ConditionError("frame='tool' needs an end effector")
        v3 = R_tool.T @ v3
    elif frame not in ("world", "base"):
        raise ConditionError(f"unknown frame {frame!r}: use 'world' or 'tool'")
    if axis is None:
        return float(np.linalg.norm(v3))
    if isinstance(axis, str):
        sign = -1.0 if axis.startswith("-") else 1.0
        name = axis.lstrip("+-")
        if name not in AXES:
            raise ConditionError(f"unknown axis {axis!r}: use x y z rx ry rz, optionally signed")
        return sign * float(v3[AXES[name]])
    u = np.asarray(axis, float)
    return float(v3 @ (u / np.linalg.norm(u)))


def _need(d: dict, key, what: str):
    if key not in d:
        raise ConditionError(f"unknown {what} {key!r}; known: {sorted(d)}")
    return d[key]


def compile_measure(node: dict):
    kind = node.get("m")
    if kind == "const":
        v = float(node["v"])
        return lambda c: v
    if kind in ("add", "sub", "mul"):
        a, b = compile_measure(node["a"]), compile_measure(node["b"])
        op = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y}[kind]
        return lambda c: op(a(c), b(c))
    if kind == "abs":
        a = compile_measure(node["a"])
        return lambda c: abs(a(c))
    if kind in ("force", "torque"):
        ee, axis, frame = node["ee"], node.get("axis"), node.get("frame", "world")
        sl = slice(0, 3) if kind == "force" else slice(3, 6)
        return lambda c: _vec(_need(c.wrench, ee, "end effector")[sl], axis, frame, c.T[ee][:3, :3])
    if kind == "effort":
        j = node["joint"]
        return lambda c: float(_need(c.tau_ext, j, "joint"))
    if kind == "deflection":
        tgt, axis, frame = node["target"], node.get("axis"), node.get("frame", "world")

        def f(c):
            if tgt in c.T:
                return _vec(c.T_ref[tgt][:3, 3] - c.T[tgt][:3, 3], axis, frame, c.T[tgt][:3, :3])
            return float(_need(c.q_ref, tgt, "end effector or joint") - c.q[tgt])
        return f
    if kind == "travel":
        ee, axis, frame = node["ee"], node.get("axis"), node.get("frame", "world")
        return lambda c: _vec(_need(c.T, ee, "end effector")[:3, 3] - c.T0[ee][:3, 3], axis, frame, c.T0[ee][:3, :3])
    if kind == "pose":
        ee, axis, frame = node["ee"], node["axis"], node.get("frame", "world")
        return lambda c: _vec(_need(c.T, ee, "end effector")[:3, 3], axis, frame, None if frame == "world" else c.T0[ee][:3, :3])
    if kind == "speed":
        tgt = node["target"]
        return lambda c: float(np.linalg.norm(c.v[tgt])) if tgt in c.v else abs(float(_need(c.dq, tgt, "end effector or joint")))
    if kind == "time":
        return lambda c: c.t - c.t0
    if kind == "progress":
        return lambda c: c.progress
    if kind == "signal":
        name = node["name"]
        return lambda c: float(c.signals.get(name, 0.0))
    if kind == "contact":
        ee = node["ee"]
        return lambda c: 1.0 if _need(c.contact, ee, "end effector") else 0.0
    if kind in ("gripper_width", "gripper_effort", "gripper_stalled"):
        name, key = node["name"], kind.split("_", 1)[1]
        return lambda c: float(_need(c.grippers, name, "gripper")[key])
    raise ConditionError(f"unknown measure {kind!r}")


def compile_cond(node: dict, dt: float):
    """-> f(ctx) -> bool, with private state for held/after."""
    op = node.get("c")
    if op == "const":
        v = bool(node["v"])
        return lambda c: v
    if op in ("gt", "ge", "lt", "le"):
        a, b = compile_measure(node["a"]), compile_measure(node["b"])
        cmp = {"gt": lambda x, y: x > y, "ge": lambda x, y: x >= y, "lt": lambda x, y: x < y, "le": lambda x, y: x <= y}[op]
        return lambda c: cmp(a(c), b(c))
    if op == "truthy":
        a = compile_measure(node["a"])
        return lambda c: a(c) > 0.5
    if op in ("and", "or"):
        a, b = compile_cond(node["a"], dt), compile_cond(node["b"], dt)
        if op == "and":
            return lambda c: a(c) and b(c)
        return lambda c: a(c) or b(c)
    if op == "not":
        a = compile_cond(node["a"], dt)
        return lambda c: not a(c)
    if op == "held":
        a, need = compile_cond(node["a"], dt), max(1, int(round(float(node["s"]) / dt)))
        state = {"n": 0}

        def held(c):
            state["n"] = state["n"] + 1 if a(c) else 0
            return state["n"] >= need
        return held
    if op == "after":
        a, b = compile_cond(node["a"], dt), compile_cond(node["b"], dt)
        state = {"armed": False}

        def after(c):
            if state["armed"]:
                return b(c)
            state["armed"] = a(c)
            return False
        return after
    raise ConditionError(f"unknown condition {op!r}")


def compile_named(spec, dt: float) -> dict:
    """until=/watch= value -> {name: evaluator}. A bare condition is named 'until'."""
    if spec is None:
        return {}
    if isinstance(spec, dict) and "c" not in spec:
        return {str(k): compile_cond(v, dt) for k, v in spec.items()}
    return {"until": compile_cond(spec, dt)}
