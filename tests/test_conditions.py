import json

import numpy as np
import pytest

from robot import when
from robotd.core.conditions import ConditionError, TickContext, compile_cond, compile_named

DT = 0.004


def ctx(**kw):
    c = TickContext(T={"left": np.eye(4)}, T0={"left": np.eye(4)}, T_ref={"left": np.eye(4)},
                    wrench={"left": np.zeros(6)}, v={"left": np.zeros(3)}, contact={"left": False},
                    grippers={"left": {"width": 0.05, "effort": 0.0, "stalled": 0.0}})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_round_trip_is_plain_json():
    cond = ((when.force("left", "-z") > 4) | when.contact("left")) & ~(when.travel("left") > 0.1)
    assert json.loads(json.dumps(cond.node)) == cond.node


def test_signed_axis_reads_component_along_it():
    f = compile_cond((when.force("left", "-z") > 4).node, DT)
    w = np.zeros(6)
    w[2] = -5.0                                     # the tool pushes 5 N downward
    assert f(ctx(wrench={"left": w}))
    w[2] = -3.0
    assert not f(ctx(wrench={"left": w}))


def test_magnitude_abs_and_arithmetic():
    w = np.array([3.0, 4.0, 0, 0, 0, 0])
    assert compile_cond((when.force("left") > 4.9).node, DT)(ctx(wrench={"left": w}))
    assert compile_cond((abs(when.force("left", "x") - 5) < 2.5).node, DT)(ctx(wrench={"left": w}))
    both = when.force("left", "x") + when.force("left", "y")
    assert compile_cond((both > 6.5).node, DT)(ctx(wrench={"left": w}))


def test_held_needs_consecutive_ticks():
    f = compile_cond((when.time() > 0).held(0.012).node, DT)    # 3 ticks
    c = ctx(t=1.0, t0=0.0)
    assert [f(c) for _ in range(4)] == [False, False, True, True]


def test_after_arms_then_fires():
    rising = when.signal("s") > 3
    falling = when.signal("s") < 1
    f = compile_cond(when.after(rising, falling).node, DT)
    seq = [0.0, 4.0, 2.0, 0.5]
    out = [f(ctx(signals={"s": v})) for v in seq]
    assert out == [False, False, False, True]


def test_travel_and_tool_frame():
    T = np.eye(4)
    T[:3, 3] = [0, 0, -0.05]
    f = compile_cond((when.travel("left", "-z") > 0.04).node, DT)
    assert f(ctx(T={"left": T}))
    Rt = np.eye(4)
    Rt[:3, :3] = np.diag([1, -1, -1])                # tool z points down: world -z is tool +z
    w = np.zeros(6)
    w[2] = -5.0
    g = compile_cond((when.force("left", "z", frame="tool") > 4).node, DT)
    assert g(ctx(T={"left": Rt}, wrench={"left": w}))


def test_named_until_and_errors():
    named = compile_named({"hit": (when.force("left") > 1).node, "far": (when.travel("left") > 1).node}, DT)
    assert set(named) == {"hit", "far"}
    with pytest.raises(ConditionError):
        compile_cond((when.force("nope") > 1).node, DT)(ctx())
    with pytest.raises(TypeError):
        bool(when.force("left") > 1)
