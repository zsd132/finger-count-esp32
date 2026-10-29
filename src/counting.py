"""Count extended fingers from 21 hand landmarks.

The landmark order follows the MediaPipe / "Hand Keypoint Dataset" convention,
which is also what the YOLO11n-pose hand model was trained on:

    0  wrist
    1  thumb CMC     2  thumb MCP     3  thumb IP      4  thumb TIP
    5  index MCP     6  index PIP     7  index DIP     8  index TIP
    9  middle MCP   10  middle PIP   11  middle DIP   12  middle TIP
   13  ring MCP     14  ring PIP     15  ring DIP     16  ring TIP
   17  pinky MCP    18  pinky PIP    19  pinky DIP    20  pinky TIP

The decision is geometric and needs no training:

* For the four long fingers a finger counts as extended when its two distal
  joints are nearly straight (a bent finger folds back on itself) *and* the
  fingertip reaches further away from the wrist than the PIP joint.
* The thumb is judged by straightness at the MCP/IP joints plus how far the
  tip sits from the index knuckle / how far it sticks out sideways.  This is
  independent of whether the palm or the back of the hand faces the camera,
  and independent of image rotation.

All maths happens in pixel coordinates, so it is scale invariant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGER_CHAINS = (THUMB, INDEX, MIDDLE, RING, PINKY)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Skeleton edges used by the renderer.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class HandAnalysis:
    """Result of :func:`analyze_hand` for one hand."""

    extended: tuple[bool, ...]      # thumb, index, middle, ring, pinky
    count: int
    confidence: float               # 0..1, from landmark scores
    palm_length: float              # wrist -> middle MCP, pixels
    palm_width: float               # index MCP -> pinky MCP, pixels
    bend: tuple[float, ...]         # straightness (deg) per finger, debug info
    probs: tuple[float, ...] = ()   # classifier probabilities, when used


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (180 = perfectly straight)."""
    u = a - b
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def analyze_hand(
    points: Sequence[Sequence[float]],
    scores: Sequence[float] | None = None,
    *,
    classifier=None,
    straight_threshold: float = 135.0,
    reach_threshold: float = 0.0,
    thumb_straight_threshold: float = 125.0,
    thumb_index_ratio: float = 1.0,
    thumb_lateral_ratio: float = 0.9,
) -> HandAnalysis | None:
    """Analyse one hand.

    Parameters
    ----------
    points:
        (21, 2) array of pixel coordinates.
    scores:
        Optional (21,) per-landmark confidence values.
    classifier:
        Optional learned :class:`src.finger_model.FingerClassifier`.  When it
        is supplied the per-finger decision comes from the model instead of
        the geometric thresholds below.

    Returns
    -------
    HandAnalysis or None when the landmarks are degenerate (all points equal,
    too few points, NaNs, ...).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 21 or pts.shape[1] < 2:
        return None
    pts = pts[:21, :2]
    if not np.isfinite(pts).all():
        return None

    wrist = pts[WRIST]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    palm_length = float(np.linalg.norm(middle_mcp - wrist))
    palm_width = float(np.linalg.norm(index_mcp - pinky_mcp))
    if palm_length < 1e-6 or palm_width < 1e-6:
        # Degenerate / collapsed landmarks are noise, not hands.
        return None

    up = (middle_mcp - wrist) / palm_length          # wrist -> middle knuckle
    side = (index_mcp - pinky_mcp) / palm_width      # pinky -> index knuckle

    bend: list[float] = [
        min(_angle(pts[1], pts[2], pts[3]), _angle(pts[2], pts[3], pts[4])),
        *[
            min(_angle(pts[a], pts[b], pts[c]), _angle(pts[b], pts[c], pts[d]))
            for a, b, c, d in (INDEX, MIDDLE, RING, PINKY)
        ],
    ]

    probs: tuple[float, ...] = ()
    if classifier is not None:
        extended_tuple, proba = classifier.predict(pts)
        extended = list(extended_tuple)
        probs = tuple(float(p) for p in proba)
        if scores is None:
            confidence = 1.0
        else:
            sc = np.asarray(scores, dtype=np.float64)[:21]
            confidence = float(np.clip(np.nanmean(sc), 0.0, 1.0)) if sc.size else 1.0
        return HandAnalysis(
            extended=tuple(extended),
            count=int(sum(extended)),
            confidence=confidence,
            palm_length=palm_length,
            palm_width=palm_width,
            bend=tuple(bend),
            probs=probs,
        )

    # --- rules (fallback / no trained model) ------------------------------
    extended: list[bool] = []

    # --- thumb ------------------------------------------------------------
    thumb_straight = min(
        _angle(pts[1], pts[2], pts[3]),
        _angle(pts[2], pts[3], pts[4]),
    )
    tip_to_index = float(np.linalg.norm(pts[4] - index_mcp)) / palm_width
    lateral = float(np.dot(pts[4] - wrist, side)) / palm_width
    thumb_ext = (
        thumb_straight >= thumb_straight_threshold
        and (tip_to_index >= thumb_index_ratio or lateral >= thumb_lateral_ratio)
    )
    extended.append(bool(thumb_ext))

    # --- index / middle / ring / pinky -------------------------------------
    for mcp, pip, dip, tip in (INDEX, MIDDLE, RING, PINKY):
        straight = min(
            _angle(pts[mcp], pts[pip], pts[dip]),
            _angle(pts[pip], pts[dip], pts[tip]),
        )
        reach = float(np.dot(pts[tip] - pts[pip], up)) / palm_length
        extended.append(bool(straight >= straight_threshold and reach >= reach_threshold))

    if scores is None:
        confidence = 1.0
    else:
        sc = np.asarray(scores, dtype=np.float64)[:21]
        confidence = float(np.clip(np.nanmean(sc), 0.0, 1.0)) if sc.size else 1.0

    return HandAnalysis(
        extended=tuple(extended),
        count=int(sum(extended)),
        confidence=confidence,
        palm_length=palm_length,
        palm_width=palm_width,
        bend=tuple(bend),
    )


def count_fingers(points: Sequence[Sequence[float]], scores: Sequence[float] | None = None) -> int:
    """Convenience wrapper returning just the count (0 when unusable)."""
    analysis = analyze_hand(points, scores)
    return analysis.count if analysis else 0
