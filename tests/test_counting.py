"""Unit tests for the geometry-only finger counter (no camera needed).

Run with:  .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counting import analyze_hand, count_fingers  # noqa: E402
from src.tracking import CountStabilizer  # noqa: E402

# A canonical left-hand skeleton in "palm space": x points from pinky to
# index, y points from wrist to fingertips, units are arbitrary.
WRIST = (0.0, 0.0)
INDEX_MCP = (-0.42, 1.00)
MIDDLE_MCP = (0.00, 1.15)
RING_MCP = (0.34, 1.08)
PINKY_MCP = (0.62, 0.88)

FINGER_BASES = {
    "index": INDEX_MCP,
    "middle": MIDDLE_MCP,
    "ring": RING_MCP,
    "pinky": PINKY_MCP,
}

THUMB_CHAIN = [(-0.55, 0.25), (-0.95, 0.50), (-1.20, 0.80), (-1.35, 1.10)]
THUMB_CURLED = [(-0.55, 0.25), (-0.70, 0.55), (-0.45, 0.80), (-0.25, 0.95)]


def make_hand(extended, *, thumb_extended=None, curl_dir=-1.0):
    """Build a (21, 2) hand.

    extended: iterable of 4 booleans for index..pinky
    thumb_extended: bool, defaults to True
    curl_dir: -1 curls fingers towards the palm (towards -x), +1 the other way
    """
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[0] = WRIST
    pts[5] = INDEX_MCP
    pts[9] = MIDDLE_MCP
    pts[13] = RING_MCP
    pts[17] = PINKY_MCP

    thumb_extended = True if thumb_extended is None else thumb_extended
    pts[1:5] = np.array(THUMB_CHAIN if thumb_extended else THUMB_CURLED)

    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        mcp = np.array(FINGER_BASES[name], dtype=np.float64)
        base = 5 + 4 * i
        pts[base] = mcp

        # A little fan so fingers do not perfectly overlap.
        spread = (i - 1.5) * 0.06
        up = np.array([spread, 1.0])
        up /= np.linalg.norm(up)

        if extended[i]:
            pts[base + 1] = mcp + 0.45 * up
            pts[base + 2] = mcp + 0.75 * up
            pts[base + 3] = mcp + 1.00 * up
        else:
            # 90 degree bend at PIP, middle phalanx sideways, tip folds back.
            side = np.array([curl_dir, 0.0])
            pts[base + 1] = mcp + 0.45 * up
            pts[base + 2] = pts[base + 1] + 0.30 * side
            pts[base + 3] = pts[base + 2] - 0.25 * up
    return pts


def rotate(points, deg, offset=(0.0, 0.0), scale=1.0):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    rot = np.array([[c, -s], [s, c]])
    return (points @ rot.T) * scale + np.array(offset)


class TestCounting(unittest.TestCase):
    def test_open_palm_is_five(self):
        hand = make_hand([True, True, True, True], thumb_extended=True)
        analysis = analyze_hand(hand)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.count, 5)
        self.assertEqual(analysis.extended, (True,) * 5)

    def test_fist_is_zero(self):
        hand = make_hand([False, False, False, False], thumb_extended=False)
        self.assertEqual(count_fingers(hand), 0)

    def test_two_fingers(self):
        hand = make_hand([True, True, False, False], thumb_extended=False)
        analysis = analyze_hand(hand)
        self.assertEqual(analysis.count, 2)
        self.assertEqual(analysis.extended, (False, True, True, False, False))

    def test_three_fingers(self):
        hand = make_hand([True, True, True, False], thumb_extended=False)
        self.assertEqual(count_fingers(hand), 3)

    def test_thumbs_up(self):
        # one thumb only: other fingers curled
        hand = make_hand([False, False, False, False], thumb_extended=True)
        self.assertEqual(count_fingers(hand), 1)

    def test_rotation_invariance(self):
        hand = make_hand([True, False, True, False], thumb_extended=True)
        base = count_fingers(hand)
        for deg in (-75, -20, 30, 90, 160):
            moved = rotate(hand, deg, offset=(120, 80), scale=2.3)
            self.assertEqual(count_fingers(moved), base, msg=f"rotation {deg}")

    def test_mirror_invariance(self):
        hand = make_hand([True, True, False, False], thumb_extended=False)
        mirrored = hand * np.array([-1.0, 1.0])
        self.assertEqual(count_fingers(hand), count_fingers(mirrored))

    def test_scale_invariance(self):
        hand = make_hand([True, True, True, False], thumb_extended=True)
        self.assertEqual(count_fingers(hand * 0.01), count_fingers(hand * 100))

    def test_degenerate_returns_none(self):
        self.assertIsNone(analyze_hand(np.zeros((21, 2))))
        self.assertIsNone(analyze_hand(np.full((21, 2), np.nan)))
        self.assertIsNone(analyze_hand(np.zeros((5, 2))))


class TestStabilizer(unittest.TestCase):
    def test_single_frame_spike_ignored(self):
        s = CountStabilizer(window=9, min_votes=4)
        for v in [3] * 5:
            s.update(v)
        self.assertEqual(s.update(5), 3)  # spike ignored
        self.assertEqual(s.update(3), 3)

    def test_real_change_accepted(self):
        s = CountStabilizer(window=9, min_votes=4, hold_seconds=0.0)
        for _ in range(4):
            s.update(3)
        for _ in range(4):
            s.update(5)
        self.assertEqual(s.value, 5)

    def test_change_must_be_held(self):
        now = [0.0]
        s = CountStabilizer(window=9, min_votes=2, hold_seconds=0.5, clock=lambda: now[0])
        for _ in range(3):
            s.update(3)
        self.assertEqual(s.value, 3)

        now[0] = 0.1
        self.assertEqual(s.update(5), 3)   # vote still says 3
        now[0] = 0.4
        self.assertEqual(s.update(5), 3)
        now[0] = 0.7
        self.assertEqual(s.update(5), 3)   # candidate starts here
        now[0] = 0.9
        self.assertEqual(s.update(5), 3)   # only 0.2 s held
        now[0] = 1.3
        self.assertEqual(s.update(5), 5)   # 0.6 s held -> commit

    def test_hold_zero_switches_immediately(self):
        s = CountStabilizer(window=9, min_votes=2, hold_seconds=0.0)
        for _ in range(3):
            s.update(3)
        for _ in range(3):
            s.update(5)
        self.assertEqual(s.value, 5)


if __name__ == "__main__":
    unittest.main()
