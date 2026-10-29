"""Learned per-finger classifier on top of the 21 landmarks.

`models/finger_lr.npz` holds five logistic-regression heads trained on HaGRID
MediaPipe landmarks (see tools/train_finger_model.py).  It is a drop-in
replacement for the hand-written geometric rules and is markedly better on
the thumb, which is the hardest finger to judge from 2D landmarks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .features import FEATURE_VERSION, finger_features

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "finger_lr.npz"


class FingerClassifier:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        data = np.load(str(path))
        version = int(np.asarray(data["version"]).reshape(-1)[0])
        if version != FEATURE_VERSION:
            raise ValueError(
                f"finger model feature version {version} != runtime {FEATURE_VERSION}; "
                "retrain with tools/train_finger_model.py"
            )
        self.W = np.asarray(data["W"], dtype=np.float32)        # (5, 10)
        self.b = np.asarray(data["b"], dtype=np.float32)        # (5,)
        self.mean = np.asarray(data["mean"], dtype=np.float32)  # (5, 10)
        self.std = np.asarray(data["std"], dtype=np.float32) + 1e-6

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        feats = finger_features(points)
        z = (((feats - self.mean) / self.std) * self.W).sum(axis=1) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def predict(self, points: np.ndarray, threshold: float = 0.5) -> tuple[tuple[bool, ...], np.ndarray]:
        proba = self.predict_proba(points)
        return tuple(bool(p >= threshold) for p in proba), proba


_DEFAULT: Optional[FingerClassifier] = None
_TRIED = False


def load_default_classifier() -> Optional[FingerClassifier]:
    """Load models/finger_lr.npz once; return None when it is unavailable."""
    global _DEFAULT, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            if DEFAULT_PATH.exists():
                _DEFAULT = FingerClassifier(DEFAULT_PATH)
        except Exception as exc:  # pragma: no cover - corrupt model file
            print(f"[warn] finger classifier unavailable ({exc}); using geometric rules")
            _DEFAULT = None
    return _DEFAULT
