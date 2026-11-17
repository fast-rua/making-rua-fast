"""camerad's --record (robotd/camerad.py FrameRecorder): the queue of frames waiting to be written stays under its
limit when the writer falls behind (it grew to 8.8 GB on the cell on 2026-11-07 and the arms went limp), and the
episode recorders' and the dashboard's grabs are not recorded."""
import json
import os
import threading
import time

import numpy as np


def _grab(H=480, W=640, cams=("front", "top", "left_wrist", "right_wrist")):
    K = [[600.0, 0, W / 2], [0, 600.0, H / 2], [0, 0, 1]]
    return {c: {"rgb": np.zeros((H, W, 3), np.uint8), "depth": np.full((H, W), 0.5, np.float32), "K": K,
                "T": np.eye(4), "t": 1.0} for c in cams}


def test_a_slow_writer_drops_grabs_instead_of_holding_them(tmp_path, monkeypatch):
    from robotd import camerad
    written, peak = [], [0]

    def slow_png(path, img):
        time.sleep(0.02)
        written.append(os.path.basename(path))

    monkeypatch.setattr(camerad, "write_png", slow_png)
    cap = 40 << 20                                         # ~4 grabs of four 640 x 480 RGB-D frames (8.6 MB each)
    rec = camerad.FrameRecorder(str(tmp_path), max_bytes=cap)
    stop = threading.Event()

    def watch_peak():
        while not stop.is_set():
            peak[0] = max(peak[0], rec.queued)
            time.sleep(0.001)

    threading.Thread(target=watch_peak, daemon=True).start()
    saved = sum(rec.save(_grab(), "agent-program") for _ in range(60))
    stop.set()
    assert peak[0] <= cap and rec.queued <= cap
    assert rec.dropped > 0 and saved + rec.dropped == 60
    t0 = time.time()
    while rec.queued > 0 and time.time() - t0 < 30:
        time.sleep(0.01)
    assert rec.queued == 0
    assert len(written) == saved * 8                       # each saved grab: all four cameras, colour and depth
    index = [json.loads(line) for line in open(tmp_path / "frames.jsonl")]
    assert [e["n"] for e in index] == list(range(1, saved + 1))
    assert all(a["dropped_before"] <= b["dropped_before"] <= rec.dropped for a, b in zip(index, index[1:]))


def test_the_recorders_and_the_dashboard_are_not_recorded(tmp_path):
    from robotd import camerad
    rec = camerad.FrameRecorder(str(tmp_path))
    for name in ("record_streams", "record_rgbd", "record_episode", "dashboard", "dashboard-live"):
        assert not rec.wants(name), name
    for name in ("agent", "watch", "watch-views", "towel", None):
        assert rec.wants(name), name


def test_a_crop_counts_the_whole_frame_it_keeps():
    from robotd.camerad import _held
    full = np.zeros((480, 640, 3), np.uint8)
    assert _held(full[100:200, 100:200]) == full.nbytes
