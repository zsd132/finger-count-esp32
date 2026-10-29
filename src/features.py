"""Scale/rotation invariant features for each of the five fingers.

The same function is used by the trainer (tools/train_finger_model.py) and by
the runtime, so the feature order must never change without bumping
FEATURE_VERSION and retraining.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 1
N_FEATURES = 10

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def finger_features(points: np.ndarray) -> np.ndarray:
    """Return a (5, N_FEATURES) float32 feature matrix.

    Row 0 is the thumb, rows 1-4 are index/middle/ring/pinky.  Everything is
    normalised by palm length/width, so the features do not depend on image
    scale, position or in-plane rotation.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (21, 2) or not np.isfinite(pts).all():
        return np.zeros((5, N_FEATURES), dtype=np.float32)

    wrist = pts[WRIST]
    index_mcp = pts[INDEX_MCP]
    middle_mcp = pts[MIDDLE_MCP]
    pinky_mcp = pts[PINKY_MCP]

    palm_len = float(np.linalg.norm(middle_mcp - wrist))
    palm_width = float(np.linalg.norm(index_mcp - pinky_mcp))
    if palm_len < 1e-9 or palm_width < 1e-9:
        return np.zeros((5, N_FEATURES), dtype=np.float32)

    up = (middle_mcp - wrist) / palm_len
    side = (index_mcp - pinky_mcp) / palm_width

    feats = np.zeros((5, N_FEATURES), dtype=np.float32)
    thumb_tip = pts[4]
    index_tip = pts[8]
    index_vec = index_tip - index_mcp

    for fi, (mcp, pip, dip, tip) in enumerate(FINGERS):
        p_mcp, p_pip, p_dip, p_tip = pts[mcp], pts[pip], pts[dip], pts[tip]
        a1 = _angle(p_mcp, p_pip, p_dip) / 180.0
        a2 = _angle(p_pip, p_dip, p_tip) / 180.0
        reach = float(np.dot(p_tip - p_pip, up)) / palm_len
        tip_rel = (float(np.linalg.norm(p_tip - wrist)) - float(np.linalg.norm(p_pip - wrist))) / palm_len
        tip_mcp = float(np.linalg.norm(p_tip - p_mcp)) / palm_len
        lateral = float(np.dot(p_tip - wrist, side)) / palm_width
        up_proj = float(np.dot(p_tip - wrist, up)) / palm_len
        straight = min(a1, a2)
        feats[fi] = (a1, a2, reach, tip_rel, tip_mcp, lateral, up_proj, straight, 0.0, 0.0)

    # Thumb-specific cues: distance/angle to the index finger.
    d_index = float(np.linalg.norm(thumb_tip - index_mcp)) / palm_width
    v_th = thumb_tip - pts[2]
    denom = float(np.linalg.norm(v_th)) * float(np.linalg.norm(index_vec))
    if denom > 1e-9:
        ang_ti = float(np.degrees(np.arccos(np.clip(np.dot(v_th, index_vec) / denom, -1.0, 1.0))))
    else:
        ang_ti = 0.0
    feats[0, 8] = d_index
    feats[0, 9] = ang_ti / 180.0

    return np.clip(feats, -4.0, 4.0)
