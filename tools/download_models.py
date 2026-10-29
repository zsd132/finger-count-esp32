#!/usr/bin/env python3
"""Download the YOLO hand-keypoint weights.

The weights live in a GitHub repository blob, and on some networks
(github.com / raw.githubusercontent.com blocked) the normal download URL does
not work.  The GitHub REST API is usually reachable, so this script uses the
blob endpoint and verifies the SHA-256.

Usage
-----
    python tools/download_models.py
    python tools/download_models.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models" / "yolo11n-pose-hands.pt"

# chrismuntean/YOLO11n-pose-hands -> runs/pose/train/weights/best.pt
BLOB_SHA = "dcfc8987767721fdfa4bbc99ab999572ade1551e"
SHA256 = "e60bb8d9a25d61c4b8238c84afe07453fc7daa8f0adfe98d14a58ff2f08c4af3"
URL = f"https://api.github.com/repos/chrismuntean/YOLO11n-pose-hands/git/blobs/{BLOB_SHA}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dest: Path) -> None:
    import requests

    print(f"downloading {URL}")
    resp = requests.get(
        URL,
        headers={"Accept": "application/vnd.github.raw", "User-Agent": "finger-count-demo"},
        timeout=120,
    )
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    got = sha256_of(tmp)
    if got != SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch: expected {SHA256}, got {got}\n"
            "The download may have been modified. Try again or download manually."
        )
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.force:
        current = sha256_of(MODEL_PATH)
        if current == SHA256:
            print(f"model OK: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
            _warmup_mediapipe()
            return 0
        print(f"existing model has wrong checksum ({current}); re-downloading")

    try:
        download(MODEL_PATH)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(
            "Manual option: open\n"
            f"  https://github.com/chrismuntean/YOLO11n-pose-hands/blob/main/runs/pose/train/weights/best.pt\n"
            f"and save it as {MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    _warmup_mediapipe()
    return 0


def _warmup_mediapipe() -> None:
    """Trigger the MediaPipe model download so the fallback works offline."""
    try:
        import numpy as np
        from src.backends import MediaPipeHandBackend
    except Exception as exc:  # pragma: no cover
        print(f"[warn] mediapipe warmup skipped: {exc}")
        return
    try:
        backend = MediaPipeHandBackend(max_hands=1)
        backend.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        backend.close()
        print("mediapipe hand model OK")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] mediapipe warmup failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
