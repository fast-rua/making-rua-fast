"""The images programs show (robot.viz.show), kept for the operator's dashboard: the latest JPEG under each name.

The image server holds the board: camerad on the real cell, robotd itself in simulation. A program sends
{"op": "show", "name", "jpeg" (uint8 bytes), "note"}; the dashboard asks {"op": "views", "since": seq} for the ones
shown since. Only the newest MAX names are kept, and nothing is written to disk here (the program saved its own file).
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np

MAX = 12
MAX_BYTES = 3_000_000


class ViewBoard:
    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.views: collections.OrderedDict = collections.OrderedDict()

    def show(self, req: dict, client: str | None = None) -> dict:
        name = str(req.get("name") or "view")[:64]
        jpeg = np.asarray(req.get("jpeg", []), np.uint8)
        if jpeg.size == 0 or jpeg.size > MAX_BYTES or bytes(jpeg[:2]) != b"\xff\xd8":
            raise ValueError("show takes a JPEG (bytes as uint8) of at most 3 MB")
        with self.lock:
            self.seq += 1
            self.views.pop(name, None)
            self.views[name] = {"name": name, "seq": self.seq, "wall": time.time(), "client": req.get("client") or client,
                                "note": req.get("note"), "jpeg": jpeg}
            while len(self.views) > MAX:
                self.views.popitem(last=False)
            return {"seq": self.seq}

    def since(self, since: int = 0) -> dict:
        """The board's seq, every name it holds (newest last) and the images of those shown after `since`."""
        with self.lock:
            names = [{k: v for k, v in x.items() if k != "jpeg"} for x in self.views.values()]
            new = [dict(x) for x in self.views.values() if x["seq"] > since]
            return {"seq": self.seq, "names": names, "views": new}
