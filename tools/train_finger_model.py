#!/usr/bin/env python3
"""Train the per-finger logistic-regression heads used by src/finger_model.py.

Data: HaGRID hand landmarks exported by
`GestureDetectionConnoisseurs/preprocessed_hand_skeletons` (MediaPipe, 21 pts).

    python tools/train_finger_model.py --csv data/basic_export.csv

The script prints per-finger validation accuracy and the overall 0-5 counting
accuracy, then writes models/finger_lr.npz.  Pure numpy, no sklearn needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_VERSION, finger_features  # noqa: E402

# Gesture name -> (thumb, index, middle, ring, pinky)
LABELS = {
    "fist": (0, 0, 0, 0, 0),
    "one": (0, 1, 0, 0, 0),
    "two": (0, 1, 1, 0, 0),
    "peace": (0, 1, 1, 0, 0),
    "peace_inverted": (0, 1, 1, 0, 0),
    "three": (0, 1, 1, 1, 0),
    "four": (0, 1, 1, 1, 1),
    "palm": (1, 1, 1, 1, 1),
    "stop": (1, 1, 1, 1, 1),
    "stop_inverted": (1, 1, 1, 1, 1),
}


def load_dataset(csv_path: Path):
    X, Y, cats = [], [], []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            cat = row["category"]
            if cat not in LABELS:
                continue
            try:
                skeleton = json.loads(row["skeleton_json"])
                landmarks = skeleton["hand_landmarks"][0]
                pts = np.array([[p["x"], p["y"]] for p in landmarks], dtype=np.float64)
            except Exception:
                continue
            if pts.shape != (21, 2) or not np.isfinite(pts).all():
                continue
            feats = finger_features(pts)
            if not np.any(feats):
                continue
            X.append(feats)
            Y.append(LABELS[cat])
            cats.append(cat)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), cats


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(X, y, epochs=600, lr=0.05, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    mw, vw = np.zeros(d), np.zeros(d)
    mb = vb = 0.0
    for t in range(1, epochs + 1):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        # Adam
        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w**2
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b**2
        mw_hat = mw / (1 - 0.9**t)
        vw_hat = vw / (1 - 0.999**t)
        mb_hat = mb / (1 - 0.9**t)
        vb_hat = vb / (1 - 0.999**t)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return w, b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "data" / "basic_export.csv"))
    parser.add_argument("--out", default=str(ROOT / "models" / "finger_lr.npz"))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    X, Y, cats = load_dataset(Path(args.csv))
    print(f"loaded {len(X)} samples")
    if len(X) < 200:
        print("not enough data", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, va = idx[:split], idx[split:]

    # Per-finger standardisation (each finger has its own feature scale).
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0) + 1e-6

    W = np.zeros((5, X.shape[-1]), dtype=np.float64)
    B = np.zeros(5, dtype=np.float64)
    names = ("thumb", "index", "middle", "ring", "pinky")

    for fi in range(5):
        Xtr = ((X[tr, fi, :] - mean[fi]) / std[fi]).astype(np.float64)
        Xva = ((X[va, fi, :] - mean[fi]) / std[fi]).astype(np.float64)
        w, b = train_logreg(Xtr, Y[tr, fi], epochs=args.epochs)
        W[fi], B[fi] = w, b
        pred = (sigmoid(Xva @ w + b) >= 0.5).astype(np.float32)
        acc = float((pred == Y[va, fi]).mean())
        print(f"{names[fi]:7s} val accuracy {acc * 100:5.1f}%")

    # Overall count accuracy
    probs = np.zeros((len(va), 5), dtype=np.float64)
    for fi in range(5):
        Xva = ((X[va, fi, :] - mean[fi]) / std[fi]).astype(np.float64)
        probs[:, fi] = sigmoid(Xva @ W[fi] + B[fi])
    pred = (probs >= 0.5).astype(np.float32)
    count_acc = float((pred.sum(axis=1) == Y[va].sum(axis=1)).mean())
    exact = float((pred == Y[va]).all(axis=1).mean())
    print(f"count accuracy (sum): {count_acc * 100:5.1f}%   exact 5-finger: {exact * 100:5.1f}%")

    np.savez(
        args.out,
        W=W.astype(np.float32),
        b=B.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        version=np.array([FEATURE_VERSION], dtype=np.int32),
    )
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
