"""Conditions: measures you compare and combine; the driver evaluates them every fast tick (4 ms).

    until = when.force("left", "-z") > 4                  # pushing down harder than 4 N
    until = (when.deflection("left", "z") > 0.003) | (when.travel("left", "-z") > 0.05)
    until = {"contact": when.contact("left"), "far": when.travel("left") > 0.10}   # the name comes back as reason
    until = (when.force("left") > 2).held(0.03)            # true for 30 ms in a row
    until = when.after(when.force("left", "y") > 3, when.force("left", "y") < 1)   # rose above 3, then fell below 1

Axes are "x" "y" "z" "rx" "ry" "rz", optionally signed ("-z"); a signed axis reads the component along it, so
when.force("left", "-z") > 4 means the tool pushes 4 N along -z. axis=None reads the magnitude. frame is "world"
(default) or "tool" (the TCP frame, moving with the arm). Forces and torques are what the tool applies to the world,
estimated from joint efforts, with the bias removed by r.zero_force(ee).
"""
from __future__ import annotations

import numbers


def _node(x):
    if isinstance(x, Measure):
        return x.node
    if isinstance(x, numbers.Number):
        return {"m": "const", "v": float(x)}
    raise TypeError(f"expected a measure or a number, got {type(x).__name__}")


class Cond:
    def __init__(self, node: dict):
        self.node = node

    def __and__(self, other):
        return Cond({"c": "and", "a": self.node, "b": _cond(other).node})

    def __or__(self, other):
        return Cond({"c": "or", "a": self.node, "b": _cond(other).node})

    def __invert__(self):
        return Cond({"c": "not", "a": self.node})

    def held(self, seconds: float) -> "Cond":
        """True once this condition has been true for `seconds` without a break."""
        return Cond({"c": "held", "a": self.node, "s": float(seconds)})

    def __bool__(self):
        raise TypeError("a condition is evaluated by the driver, not in Python: pass it as until= or on=")

    def __repr__(self):
        return f"Cond({self.node})"


def _cond(x):
    if isinstance(x, Cond):
        return x
    raise TypeError("combine conditions with & | ~, e.g. (when.force('left') > 3) | when.contact('left')")


class Measure:
    def __init__(self, node: dict):
        self.node = node

    def _cmp(self, op, other):
        return Cond({"c": op, "a": self.node, "b": _node(other)})

    def __gt__(self, o): return self._cmp("gt", o)
    def __ge__(self, o): return self._cmp("ge", o)
    def __lt__(self, o): return self._cmp("lt", o)
    def __le__(self, o): return self._cmp("le", o)
    def __add__(self, o): return Measure({"m": "add", "a": self.node, "b": _node(o)})
    __radd__ = __add__
    def __sub__(self, o): return Measure({"m": "sub", "a": self.node, "b": _node(o)})
    def __rsub__(self, o): return Measure({"m": "sub", "a": _node(o), "b": self.node})
    def __mul__(self, o): return Measure({"m": "mul", "a": self.node, "b": _node(o)})
    __rmul__ = __mul__
    def __neg__(self): return Measure({"m": "mul", "a": self.node, "b": {"m": "const", "v": -1.0}})
    def __abs__(self): return Measure({"m": "abs", "a": self.node})

    def __bool__(self):
        raise TypeError("compare a measure with a number, e.g. when.force('left', '-z') > 4")

    def __repr__(self):
        return f"Measure({self.node})"


def _axis(axis):
    if axis is None or isinstance(axis, str):
        return axis
    return [float(v) for v in axis]            # a 3-vector: the component along it


def force(ee: str, axis=None, frame="world") -> Measure:
    """Force the tool applies to the world, N."""
    return Measure({"m": "force", "ee": ee, "axis": _axis(axis), "frame": frame})


def torque(ee: str, axis=None, frame="world") -> Measure:
    """Torque the tool applies to the world, Nm."""
    return Measure({"m": "torque", "ee": ee, "axis": _axis(axis), "frame": frame})


def effort(joint: str) -> Measure:
    """External effort on a joint (measured minus the model's gravity and motion terms), Nm or N."""
    return Measure({"m": "effort", "joint": joint})


def deflection(target: str, axis=None, frame="world") -> Measure:
    """Reference minus measured: an end effector in m (TCP position), or a joint in rad. Encoders only."""
    return Measure({"m": "deflection", "target": target, "axis": _axis(axis), "frame": frame})


def travel(ee: str, axis=None, frame="world") -> Measure:
    """TCP displacement since the command started, m (along a signed axis, or the distance)."""
    return Measure({"m": "travel", "ee": ee, "axis": _axis(axis), "frame": frame})


def pose(ee: str, axis: str, frame="world") -> Measure:
    """A TCP coordinate: x, y or z in m."""
    return Measure({"m": "pose", "ee": ee, "axis": axis, "frame": frame})


def speed(target: str) -> Measure:
    """TCP speed (m/s) of an end effector, or |velocity| of a joint (rad/s)."""
    return Measure({"m": "speed", "target": target})


def time() -> Measure:
    """Seconds since the command started."""
    return Measure({"m": "time"})


def progress() -> Measure:
    """0..1 along a move (its path or the way to its goal)."""
    return Measure({"m": "progress"})


def signal(name: str) -> Measure:
    """A number set by r.signal(name, value) from any process (e.g. a scale reading), default 0."""
    return Measure({"m": "signal", "name": name})


def contact(ee: str) -> Cond:
    """Debounced contact flag of an end effector (estimated force above the contact threshold)."""
    return Cond({"c": "truthy", "a": {"m": "contact", "ee": ee}})


class _Gripper:
    def __init__(self, name):
        self.width = Measure({"m": "gripper_width", "name": name})
        self.effort = Measure({"m": "gripper_effort", "name": name})
        self.stalled = Cond({"c": "truthy", "a": {"m": "gripper_stalled", "name": name}})


def gripper(name: str) -> _Gripper:
    """gripper(name).width (m between the pads), .effort (N), .stalled (closed on something)."""
    return _Gripper(name)


def after(first: Cond, then: Cond) -> Cond:
    """True once `then` holds at some tick after `first` has held."""
    return Cond({"c": "after", "a": _cond(first).node, "b": _cond(then).node})


def always() -> Cond:
    return Cond({"c": "const", "v": True})
