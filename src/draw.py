"""On-screen rendering: skeleton, hand chips and the big counter panel."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .counting import FINGER_CHAINS, HAND_CONNECTIONS, HandAnalysis

FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)

MINT = (150, 255, 170)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK = (26, 28, 34)


def _find_font() -> Optional[str]:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def alpha_blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an RGBA uint8 image onto a BGR frame at (x, y)."""
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    sub = src[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    rgb = sub[..., :3].astype(np.float32)
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (region * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


class TextSprite:
    """A cached RGBA text render."""

    __slots__ = ("image", "width", "height")

    def __init__(self, text: str, size: int, color, font_path: Optional[str], stroke: int = 0):
        if font_path:
            font = ImageFont.truetype(font_path, size)
        else:  # pragma: no cover - only when no CJK font exists
            font = ImageFont.load_default()
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        bbox = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
        pad = 3
        w = max(1, bbox[2] - bbox[0] + pad * 2)
        h = max(1, bbox[3] - bbox[1] + pad * 2)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (pad - bbox[0], pad - bbox[1]),
            text,
            font=font,
            fill=tuple(color) + (255,),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 160),
        )
        self.image = np.array(img)
        self.width, self.height = w, h


class HUD:
    def __init__(self, width: int, height: int, *, panel_width: int = 300):
        self.width = width
        self.height = height
        self.font_path = _find_font()
        self.cjk = bool(self.font_path) and ("PingFang" in self.font_path or "Hiragino" in self.font_path)
        self.label = "根手指" if self.cjk else "FINGERS"
        self.hint = "把手举到摄像头前，比出 0~5" if self.cjk else "Show 0-5 fingers"
        self._cache: Dict[Tuple, TextSprite] = {}

        # Keep the HUD proportional on small sources (e.g. a 320x240 test clip).
        self.scale = max(0.40, min(1.0, min(width, height) / 720.0))
        self.panel_w = int(panel_width * self.scale)
        self.panel_h = int(208 * self.scale)
        self.panel_x = int(28 * self.scale)
        self.panel_y = int(26 * self.scale)
        self.panel = self._make_panel(self.panel_w, self.panel_h)

    # ------------------------------------------------------------------ text
    def sprite(self, text: str, size: int, color, stroke: int = 0) -> TextSprite:
        key = (text, size, tuple(color), stroke)
        if key not in self._cache:
            self._cache[key] = TextSprite(text, size, color, self.font_path, stroke)
        return self._cache[key]

    def text(self, frame, text: str, x: int, y: int, size: int, color, anchor="lt", stroke=0):
        sprite = self.sprite(text, size, color, stroke)
        dx = {"l": 0, "c": -sprite.width // 2, "r": -sprite.width}[anchor[0]]
        dy = {"t": 0, "c": -sprite.height // 2, "b": -sprite.height}[anchor[1]]
        alpha_blit(frame, sprite.image, x + dx, y + dy)

    # ----------------------------------------------------------------- panel
    def _make_panel(self, w: int, h: int) -> np.ndarray:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill=DARK + (168,))
        draw.rounded_rectangle(
            (1, 1, w - 2, h - 2), radius=21, outline=(255, 255, 255, 26), width=1
        )
        return np.array(img)

    # ---------------------------------------------------------------- render
    def render(
        self,
        frame: np.ndarray,
        tracks,
        *,
        fps: float,
        backend: str,
        show_hint: bool = True,
        serial_on: bool = False,
        total: int | None = None,
        hands: int | None = None,
    ) -> np.ndarray:
        for track in tracks:
            self._draw_hand(frame, track)

        if total is None:
            total = int(sum(t.count for t in tracks))
        if hands is None:
            hands = len(tracks)
        total = int(total)
        has_hand = hands > 0

        alpha_blit(frame, self.panel, self.panel_x, self.panel_y)

        accent = MINT if has_hand else GREY
        number = str(total) if has_hand else "—"
        size = int((132 if len(number) == 1 else 112) * self.scale)
        cx = self.panel_x + self.panel_w // 2
        cy = self.panel_y + int(self.panel_h * 0.40)
        self.text(frame, number, cx, cy, size, accent, anchor="cc", stroke=max(1, int(2 * self.scale)))
        self.text(
            frame,
            self.label,
            cx,
            self.panel_y + int(self.panel_h * 0.80),
            max(12, int(26 * self.scale)),
            (235, 235, 235),
            anchor="ct",
        )

        # status pill, bottom-left (stays readable on bright backgrounds)
        status = f"{backend}  |  {fps:4.1f} fps  |  {hands} hand(s)"
        if serial_on:
            status += "  |  SER" 
        st = self.sprite(status, max(11, int(17 * self.scale)), (225, 225, 225))
        px = self.panel_x
        py = self.height - st.height - max(10, int(18 * self.scale))
        self._rounded_rect(
            frame, px, py, px + st.width + 16, py + st.height + 8,
            max(5, int(9 * self.scale)), (22, 24, 30), 0.62,
        )
        alpha_blit(frame, st.image, px + 8, py + 4)

        if show_hint and not has_hand:
            self.text(
                frame,
                self.hint,
                self.width // 2,
                self.height - int(66 * self.scale),
                max(13, int(26 * self.scale)),
                (230, 230, 230),
                anchor="ct",
                stroke=1,
            )
        return frame

    # ------------------------------------------------------------------ hand
    def _draw_hand(self, frame: np.ndarray, track) -> None:
        pts = track.points.astype(int)
        analysis: Optional[HandAnalysis] = track.analysis
        color = track.color

        extended = analysis.extended if analysis else (False,) * 5
        edge_finger = {}
        for fi, chain in enumerate(FINGER_CHAINS):
            for a, b in zip(chain, chain[1:]):
                edge_finger[(a, b)] = fi
                edge_finger[(b, a)] = fi

        for a, b in HAND_CONNECTIONS:
            fi = edge_finger.get((a, b))
            on = fi is not None and extended[fi]
            col = color if on else (105, 110, 120)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(pts):
            if i in (4, 8, 12, 16, 20):
                fi = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}[i]
                col = color if extended[fi] else (130, 130, 130)
                cv2.circle(frame, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 3, (235, 235, 235), -1, cv2.LINE_AA)

        # bounding box with corner ticks
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width - 1, x2), min(self.height - 1, y2)
        self._corner_box(frame, x1, y1, x2, y2, color)

        # count chip
        label = f"{track.count}"
        suffix = ""
        if track.handedness == "Left":
            suffix = " 左手" if self.cjk else " L"
        elif track.handedness == "Right":
            suffix = " 右手" if self.cjk else " R"
        elif track.handedness:
            suffix = " 手"
        chip_text = label + suffix
        ts = self.sprite(chip_text, max(12, int(22 * self.scale)), color, stroke=0)
        pad_x, pad_y = max(6, int(9 * self.scale)), max(3, int(4 * self.scale))
        chip_w, chip_h = ts.width + pad_x * 2, ts.height + pad_y * 2
        cx = x1
        cy = max(4, y1 - chip_h - int(6 * self.scale))
        self._rounded_rect(frame, cx, cy, cx + chip_w, cy + chip_h, max(4, int(8 * self.scale)), (24, 26, 32), 0.72)
        alpha_blit(frame, ts.image, cx + pad_x, cy + pad_y)

    def _corner_box(self, frame, x1, y1, x2, y2, color, length=18):
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2, cv2.LINE_AA)

    @staticmethod
    def _rounded_rect(frame, x1, y1, x2, y2, r, color, alpha=0.7):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
