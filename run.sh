#!/usr/bin/env bash
# One-command launcher: sets up the venv on first run, downloads the model,
# then starts the webcam demo. Extra arguments are forwarded to main.py.
set -euo pipefail
cd "$(dirname "$0")"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
  3.10|3.11|3.12) ;;
  *) echo "[warn] Python $PYVER detected; this project is tested with 3.10-3.12." ;;
esac

if [ ! -d .venv ]; then
  echo "[setup] creating virtualenv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import ultralytics, mediapipe" >/dev/null 2>&1; then
  echo "[setup] installing dependencies (first run, a few minutes)"
  python -m pip install -U pip setuptools wheel
  python -m pip install -r requirements.txt
fi

if [ ! -f models/yolo11n-pose-hands.pt ]; then
  echo "[setup] downloading hand-keypoint model"
  python tools/download_models.py
fi

exec python main.py "$@"
