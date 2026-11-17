"""PNG files from camera frames, with the standard library and numpy only (the runner's pauses and the driver's
recorder both write them).

    write_png(path, rgb)                  # H x W x 3 uint8
    write_png(path, depth_mm(f.depth))    # H x W uint16: depth in mm, 0 where there is no reading

cv2.imread(path, cv2.IMREAD_UNCHANGED) and PIL read the depth back as uint16 mm."""
from __future__ import annotations

import struct
import zlib

import numpy as np


def depth_mm(depth) -> np.ndarray:
    """Depth in m as uint16 mm (0: no reading, beyond 65.535 m clipped)."""
    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0)
    return np.where(ok, np.clip(np.round(d * 1000.0), 0, 65535), 0).astype(np.uint16)


def write_png(path: str, img) -> None:
    """An H x W x 3 uint8 image (RGB), or an H x W uint8 or uint16 image (grey, e.g. depth_mm), as PNG."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = np.ascontiguousarray(a[..., :3], dtype=np.uint8)
        bits, colour, rows = 8, 2, a.reshape(a.shape[0], -1)
    elif a.ndim == 2 and a.dtype == np.uint16:
        bits, colour, rows = 16, 0, np.ascontiguousarray(a, dtype=">u2").view(np.uint8)
    elif a.ndim == 2:
        bits, colour, rows = 8, 0, np.ascontiguousarray(a, dtype=np.uint8)
    else:
        raise ValueError(f"not an image: shape {a.shape}")
    h, w = a.shape[:2]
    if bits == 16:          # the Up filter (each byte minus the one above it): the sim's top depth 19 KB, not 26
        rows = np.concatenate([rows[:1], rows[1:] - rows[:-1]])
    raw = np.hstack([np.full((h, 1), 2 if bits == 16 else 0, np.uint8), rows]).tobytes()

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, colour, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
