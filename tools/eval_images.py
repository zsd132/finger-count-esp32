#!/usr/bin/env python3
"""Batch-evaluate the counter on a folder of images.

    python tools/eval_images.py samples/*.jpg --backend yolo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backends import create_backend  # noqa: E402
from src.counting import analyze_hand  # noqa: E402
from src.finger_model import load_default_classifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--backend", default="yolo", choices=["auto", "yolo", "mediapipe"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out-dir", default=str(ROOT / "out" / "eval"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(args.backend, conf=args.conf)
    classifier = load_default_classifier() if backend.name == "mediapipe" else None
    try:
        for path_str in args.images:
            path = Path(path_str)
            image = cv2.imread(str(path))
            if image is None:
                print(f"{path.name:34s} [cannot read]")
                continue
            hands = backend.detect(image)
            counts = [
                analyze_hand(h.points, h.scores, classifier=classifier) for h in hands
            ]
            values = [c.count if c else 0 for c in counts]
            print(f"{path.name:34s} hands={len(hands)} counts={values}")
            # annotated preview
            preview = image.copy()
            for hand, analysis in zip(hands, counts):
                p = hand.points.astype(int)
                for i, (x, y) in enumerate(p):
                    cv2.circle(preview, (x, y), 4, (0, 0, 255), -1)
                if hand.bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in hand.bbox]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{path.stem}_eval.jpg"), preview)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
