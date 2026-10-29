"""Temporal smoothing and tiny multi-hand tracker.

Raw per-frame counts flicker when a finger hovers around a decision
threshold.  Two cheap tricks remove almost all of it:

* majority vote over a short history (with a minimum number of votes, so a
  single frame cannot change the displayed number);
* nearest-centre hand tracking, so each hand keeps its own history when two
  hands are on screen.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from .counting import HandAnalysis
from .backends import HandObservation

COLORS: Tuple[Tuple[int, int, int], ...] = (
    (120, 255, 170),
    (120, 200, 255),
    (255, 170, 120),
    (200, 140, 255),
)


class CountStabilizer:
    """Majority vote plus a "held for N seconds" gate.

    The vote removes single-frame spikes; the hold gate makes the displayed
    number change only after the new value has been the vote winner for
    ``hold_seconds`` (0.5 s by default).  A brief misread while the hand is
    moving therefore never reaches the screen or the serial port.
    """

    def __init__(
        self,
        window: int = 9,
        min_votes: int = 4,
        hold_seconds: float = 0.5,
        clock=time.perf_counter,
    ) -> None:
        self.history: Deque[int] = deque(maxlen=max(3, window))
        self.min_votes = max(2, min_votes)
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.value: Optional[int] = None
        self._clock = clock
        self._candidate: Optional[int] = None
        self._candidate_since = 0.0

    def update(self, value: int, now: Optional[float] = None) -> int:
        if now is None:
            now = self._clock()
        self.history.append(int(value))
        if self.value is None:
            self.value = int(value)
            return self.value

        history = list(self.history)
        counts = Counter(history)
        # Majority vote; on a tie prefer the value seen most recently so the
        # display still follows a real change quickly.
        best, votes = max(
            counts.items(),
            key=lambda kv: (kv[1], len(history) - 1 - history[::-1].index(kv[0])),
        )
        needed = min(self.min_votes, len(history))

        if best == self.value:
            self._candidate = None
            return self.value

        if votes < needed:
            self._candidate = None
            return self.value

        if best != self._candidate:
            self._candidate = best
            self._candidate_since = now

        if now - self._candidate_since >= self.hold_seconds:
            self.value = best
            self._candidate = None
        return self.value


@dataclass
class Track:
    tid: int
    points: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray
    handedness: Optional[str]
    analysis: Optional[HandAnalysis]
    count: int
    color: Tuple[int, int, int]
    stabilizer: CountStabilizer = field(default_factory=CountStabilizer)
    missed: int = 0

    @property
    def bbox_center(self) -> np.ndarray:
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0]
        )


class HandTracker:
    def __init__(
        self,
        frame_width: int,
        *,
        match_ratio: float = 0.18,
        max_missed: int = 6,
        hold_seconds: float = 0.5,
    ) -> None:
        self.match_dist = max(40.0, frame_width * match_ratio)
        self.max_missed = max_missed
        self.hold_seconds = hold_seconds
        self.tracks: List[Track] = []
        self._next_id = 0

    @staticmethod
    def _center(obs: HandObservation) -> np.ndarray:
        if obs.bbox is not None:
            return np.array(
                [(obs.bbox[0] + obs.bbox[2]) / 2.0, (obs.bbox[1] + obs.bbox[3]) / 2.0]
            )
        return obs.points.mean(axis=0)

    def update(
        self, hands: Sequence[HandObservation], analyses: Sequence[Optional[HandAnalysis]]
    ) -> List[Track]:
        centers = [self._center(h) for h in hands]
        used = [False] * len(hands)
        result: List[Track] = []

        for track in self.tracks:
            track.missed += 1

        for track in list(self.tracks):
            if not centers:
                break
            distances = [
                float(np.linalg.norm(c - track.bbox_center)) if not used[i] else 1e9
                for i, c in enumerate(centers)
            ]
            i = int(np.argmin(distances))
            if distances[i] > self.match_dist:
                continue
            used[i] = True
            track.missed = 0
            self._refresh(track, hands[i], analyses[i])
            result.append(track)

        for i, hand in enumerate(hands):
            if used[i]:
                continue
            track = Track(
                tid=self._next_id,
                points=hand.points,
                scores=hand.scores,
                bbox=hand.bbox if hand.bbox is not None else _bbox_from_points(hand.points),
                handedness=hand.handedness,
                analysis=analyses[i],  # type: ignore[arg-type]
                count=0,
                color=COLORS[self._next_id % len(COLORS)],
                stabilizer=CountStabilizer(hold_seconds=self.hold_seconds),
            )
            self._next_id += 1
            if analyses[i] is not None:
                track.count = track.stabilizer.update(analyses[i].count)
            self.tracks.append(track)
            result.append(track)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return result

    def _refresh(
        self, track: Track, hand: HandObservation, analysis: Optional[HandAnalysis]
    ) -> None:
        track.missed = 0
        track.points = hand.points
        track.scores = hand.scores
        if hand.bbox is not None:
            track.bbox = hand.bbox
        if hand.handedness:
            track.handedness = hand.handedness
        if analysis is not None:
            track.analysis = analysis
            track.count = track.stabilizer.update(analysis.count)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0


def _bbox_from_points(points: np.ndarray, pad: float = 0.12) -> np.ndarray:
    x1, y1 = points[:, 0].min(), points[:, 1].min()
    x2, y2 = points[:, 0].max(), points[:, 1].max()
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h], dtype=np.float32)
