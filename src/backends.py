"""Hand detection back-ends.

Two interchangeable implementations produce the same 21-landmark layout:

* ``yolo``      - YOLO11n-pose model trained on hand keypoints (the default).
* ``mediapipe`` - MediaPipe Hands, used as an automatic fallback and as a
                  second opinion when the YOLO model is unsure.

Both return pixel coordinates for a BGR frame.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# MediaPipe / TensorFlow are chatty on import.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models" / "yolo11n-pose-hands.pt"


@dataclass
class HandObservation:
    """One detected hand in pixel coordinates."""

    points: np.ndarray                      # (21, 2) float32
    scores: np.ndarray                      # (21,) float32, 0..1
    bbox: Optional[np.ndarray] = None       # (4,) x1, y1, x2, y2
    handedness: Optional[str] = None        # "Left" / "Right" when known
    handedness_score: float = 0.0


class HandBackend:
    name = "base"

    def detect(self, frame_bgr: np.ndarray) -> List[HandObservation]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class YoloHandBackend(HandBackend):
    """YOLO11n-pose hand-keypoint model."""

    name = "yolo11n-pose"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_MODEL,
        *,
        imgsz: int = 640,
        conf: float = 0.35,
        max_hands: int = 2,
        device: str = "cpu",
        refine: bool = True,
        refine_pad: float = 0.25,
        refine_imgsz: int | None = None,
        warmup: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_path}\n"
                "Run `python tools/download_models.py` first."
            )
        self.model = YOLO(str(self.model_path))
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_hands = int(max_hands)
        self.device = device
        self.refine = bool(refine)
        self.refine_pad = float(refine_pad)
        self.refine_imgsz = int(refine_imgsz or imgsz)

        if warmup:
            self.model.predict(
                np.zeros((480, 640, 3), dtype=np.uint8),
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                device=self.device,
            )

    # ------------------------------------------------------------------ utils
    def _predict(self, image: np.ndarray, imgsz: int, max_det: int):
        return self.model.predict(
            image,
            imgsz=imgsz,
            conf=self.conf,
            max_det=max_det,
            verbose=False,
            device=self.device,
        )[0]

    @staticmethod
    def _extract(result):
        """Return (points, scores, boxes) or None when nothing was found."""
        kp = result.keypoints
        if kp is None or kp.xy is None or result.boxes is None or len(result.boxes) == 0:
            return None
        xy = kp.xy.cpu().numpy().astype(np.float32)
        # Ultralytics returns a (1, 0, 2) placeholder when nothing was found.
        if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] < 21:
            return None
        if kp.conf is not None:
            scores = kp.conf.cpu().numpy().astype(np.float32)
        else:
            scores = np.ones(xy.shape[:2], dtype=np.float32)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        return xy, scores, boxes

    # ----------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        """Detect hands, then refine each hand on an enlarged crop.

        The second pass matters: the 21-keypoint head was trained on hands
        that fill the frame, and on a 1280x720 webcam frame a hand is often
        only ~150 px tall.  Cropping and upscaling the detected hand before
        the keypoint pass makes the predictions far more precise.
        """
        result = self._predict(frame_bgr, self.imgsz, self.max_hands)
        extracted = self._extract(result)
        if extracted is None:
            return []
        xy, scores, boxes = extracted

        h, w = frame_bgr.shape[:2]
        hands: List[HandObservation] = []
        for i in range(xy.shape[0]):
            points, point_scores = xy[i], scores[i]
            bbox = boxes[i] if i < len(boxes) else _bbox_from_points(points)

            if self.refine and self._should_refine(bbox, w, h):
                refined = self._refine(frame_bgr, bbox, w, h)
                if refined is not None:
                    points, point_scores = refined

            hands.append(
                HandObservation(
                    points=points,
                    scores=point_scores,
                    bbox=bbox,
                    handedness=None,
                )
            )
        return hands

    @staticmethod
    def _should_refine(bbox: np.ndarray, width: int, height: int) -> bool:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if bw < 48 or bh < 48:
            return False
        # Already filling the frame -> a crop would not add resolution.
        return (bw * bh) / float(width * height) < 0.55

    def _refine(self, frame: np.ndarray, bbox: np.ndarray, width: int, height: int):
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = (x2 - x1) * self.refine_pad, (y2 - y1) * self.refine_pad
        X1 = int(max(0, x1 - pad_x))
        Y1 = int(max(0, y1 - pad_y))
        X2 = int(min(width, x2 + pad_x))
        Y2 = int(min(height, y2 + pad_y))
        if X2 - X1 < 32 or Y2 - Y1 < 32:
            return None

        crop = frame[Y1:Y2, X1:X2]
        if crop.size == 0:
            return None
        result = self._predict(crop, self.refine_imgsz, 1)
        extracted = self._extract(result)
        if extracted is None:
            return None

        points, scores, _ = extracted
        points = points[0].copy()
        points[:, 0] += X1
        points[:, 1] += Y1
        return points, scores[0]


class FallbackHandBackend(HandBackend):
    """Use a primary backend, and let MediaPipe cover frames it missed.

    The YOLO model is trained on close-up hands; when the hand is small or
    partly out of frame it can return nothing at all.  MediaPipe is more
    tolerant, so ``auto`` mode quietly falls back on exactly those frames.
    """

    name = "auto:yolo+mediapipe"

    def __init__(self, primary: HandBackend, fallback_factory) -> None:
        self.primary = primary
        self._fallback_factory = fallback_factory
        self.fallback: Optional[HandBackend] = None
        self.fallback_frames = 0

    def detect(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        hands = self.primary.detect(frame_bgr)
        if hands:
            return hands
        if self.fallback is None:
            self.fallback = self._fallback_factory()
        self.fallback_frames += 1
        return self.fallback.detect(frame_bgr)

    def close(self) -> None:
        self.primary.close()
        if self.fallback is not None:
            self.fallback.close()


class MediaPipeHandBackend(HandBackend):
    """MediaPipe Hands (solutions API)."""

    name = "mediapipe"

    def __init__(
        self,
        *,
        max_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def detect(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hands: List[HandObservation] = []
        handedness_info = result.multi_handedness or []

        for idx, landmarks in enumerate(result.multi_hand_landmarks):
            pts = np.array(
                [[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.float32
            )
            scores = np.array(
                [getattr(lm, "visibility", 0.0) or 1.0 for lm in landmarks.landmark],
                dtype=np.float32,
            )
            label, score = None, 0.0
            if idx < len(handedness_info):
                cls = handedness_info[idx].classification[0]
                label, score = cls.label, float(cls.score)
            hands.append(
                HandObservation(
                    points=pts,
                    scores=scores,
                    bbox=_bbox_from_points(pts),
                    handedness=label,
                    handedness_score=score,
                )
            )
        return hands

    def close(self) -> None:
        self.hands.close()


def _bbox_from_points(points: np.ndarray, pad: float = 0.12) -> np.ndarray:
    x1, y1 = points[:, 0].min(), points[:, 1].min()
    x2, y2 = points[:, 0].max(), points[:, 1].max()
    w, h = x2 - x1, y2 - y1
    return np.array(
        [x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h], dtype=np.float32
    )


def backend_available(name: str) -> bool:
    if name == "yolo":
        if not DEFAULT_YOLO_MODEL.exists():
            return False
        try:
            import ultralytics  # noqa: F401
        except Exception:
            return False
        return True
    if name == "mediapipe":
        try:
            import mediapipe  # noqa: F401
        except Exception:
            return False
        return True
    return False


def create_backend(
    name: str = "auto",
    *,
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    imgsz: int = 640,
    conf: float = 0.35,
    max_hands: int = 2,
    refine: bool = True,
) -> HandBackend:
    """Create a backend.

    ``auto`` uses YOLO and falls back to MediaPipe on frames where YOLO sees
    no hand at all.  ``yolo`` / ``mediapipe`` force one engine.
    """
    name = (name or "auto").lower()
    model_ok = Path(model_path).exists()
    yolo_ok = model_ok and backend_available("yolo")
    mp_ok = backend_available("mediapipe")

    if name == "auto":
        if yolo_ok:
            try:
                primary = YoloHandBackend(
                    model_path,
                    imgsz=imgsz,
                    conf=conf,
                    max_hands=max_hands,
                    refine=refine,
                )
                if mp_ok:
                    return FallbackHandBackend(
                        primary,
                        lambda: MediaPipeHandBackend(max_hands=max_hands),
                    )
                return primary
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[warn] YOLO backend failed ({exc}); using MediaPipe")
        if mp_ok:
            return MediaPipeHandBackend(max_hands=max_hands)
        raise RuntimeError(
            "No hand backend available. Install dependencies with "
            "`pip install -r requirements.txt` and download the model with "
            "`python tools/download_models.py`."
        )

    if name == "yolo":
        if not yolo_ok:
            raise RuntimeError(
                f"YOLO backend is not available: missing weights ({model_path}) "
                "or ultralytics is not installed."
            )
        return YoloHandBackend(
            model_path,
            imgsz=imgsz,
            conf=conf,
            max_hands=max_hands,
            refine=refine,
        )

    if not mp_ok:
        raise RuntimeError("MediaPipe backend is not available: `pip install mediapipe`.")
    return MediaPipeHandBackend(max_hands=max_hands)
