"""Small helpers the examples share. Ordinary code over the robot API; nothing here is part of the API."""
from robot import twist, when


def touch(r, ee, direction, speed=0.01, force=3.0, max_travel=0.05, frame="world", back_off=0.0, **common):
    """Guarded move: go along a signed axis ('-z') or a 3-vector until the tool feels `force` N or travels max_travel.
    Result.reason is 'contact' or 'max_travel'; Result.at.T[ee] is the TCP pose then. back_off= m: retreat after
    contact."""
    axis = direction if isinstance(direction, str) else [float(v) for v in direction]
    common.setdefault("until", {"contact": when.force(ee, axis, frame=frame) > force})
    common.setdefault("then", ["back_off", back_off] if back_off else "hold")
    return r.move_at(twist(direction, speed), ee=ee, frame=frame, max_travel=max_travel, **common)
