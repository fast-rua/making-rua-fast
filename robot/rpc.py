"""Length-prefixed JSON messages over a stream socket (unix or TCP). Arrays travel as base64 blobs with dtype/shape."""
from __future__ import annotations

import base64
import json
import socket
import struct

from robot._lazy import Lazy

np = Lazy("numpy", __name__, "np")


def _default(o):
    if isinstance(o, np.ndarray):
        if o.dtype in (np.uint8, np.float32) and o.size > 64:      # images and depth: a compact blob
            return {"__nd__": base64.b64encode(np.ascontiguousarray(o).tobytes()).decode(),
                    "dtype": str(o.dtype), "shape": list(o.shape)}
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "detach"):                                         # torch
        return o.detach().cpu().numpy().tolist()
    if hasattr(o, "__array__"):                                      # JAX and others
        return np.asarray(o).tolist()
    if hasattr(o, "node"):                                           # robot.when conditions and measures
        return o.node
    if hasattr(o, "to_json"):
        return o.to_json()
    raise TypeError(f"cannot send {type(o).__name__}")


def _hook(d):
    if "__nd__" in d:
        return np.frombuffer(base64.b64decode(d["__nd__"]), dtype=d["dtype"]).reshape(d["shape"])
    return d


def dumps(obj) -> bytes:
    return json.dumps(obj, default=_default).encode()


def loads(b: bytes):
    return json.loads(b, object_hook=_hook)


def send(sock: socket.socket, obj) -> None:
    data = dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv(sock: socket.socket):
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    (n,) = struct.unpack(">I", head)
    return loads(_recv_exact(sock, n))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


def connect(addr: str) -> socket.socket:
    """addr: 'unix:/path/to/sock' or 'tcp:host:port'."""
    if addr.startswith("unix:"):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(addr[5:])
        return s
    if addr.startswith("tcp:"):
        host, port = addr[4:].rsplit(":", 1)
        s = socket.create_connection((host, int(port)))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s
    raise ValueError("address must be unix:/path or tcp:host:port")
