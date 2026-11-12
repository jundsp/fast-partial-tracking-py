"""Per-hop segments: which partials fade in, fade out or continue between two frames.

Shared by every synthesizer; they only differ in how a segment's phase is rendered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fast_partial_tracking.partials import Frame
from fast_partial_tracking.tracking.labeling import SetMatch, match_sets


@dataclass(frozen=True)
class Segments:
    """Endpoint parameters of sinusoid segments spanning one hop.

    Attributes:
        omega: Start/end angular frequencies (radians/sample), shape (P, 2).
        amplitude: Start/end linear amplitudes of the complex model, shape (P, 2).
        phase: Start/end phases, shape (P, 2).
        track_id: Partial label of each segment, shape (P,).
    """

    omega: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    track_id: np.ndarray

    def __len__(self) -> int:
        return int(self.omega.shape[0])

    @classmethod
    def concat(cls, parts: list[Segments]) -> Segments:
        fields = ((p.omega, p.amplitude, p.phase, p.track_id) for p in parts)
        return cls(*(np.concatenate(columns) for columns in zip(*fields)))


def _fade_in(frame: Frame, rows: np.ndarray, hop: int) -> Segments:
    """Partials born at ``frame``: fade in over the preceding hop at constant frequency."""
    w, ph = frame.omega[rows], frame.phase[rows]
    return Segments(
        np.stack([w, w], axis=1),
        np.stack([np.zeros_like(w), np.exp(frame.log_amplitude[rows])], axis=1),
        np.stack([ph - w * hop, ph], axis=1),
        frame.track_id[rows],
    )


def _fade_out(frame: Frame, rows: np.ndarray, hop: int) -> Segments:
    """Partials ending at ``frame``: fade out over the following hop at constant frequency."""
    w, ph = frame.omega[rows], frame.phase[rows]
    return Segments(
        np.stack([w, w], axis=1),
        np.stack([np.exp(frame.log_amplitude[rows]), np.zeros_like(w)], axis=1),
        np.stack([ph, ph + w * hop], axis=1),
        frame.track_id[rows],
    )


def _continue(prev: Frame, prev_rows: np.ndarray, curr: Frame, curr_rows: np.ndarray) -> Segments:
    """Partials present in both frames: interpolate between their parameters."""
    return Segments(
        np.stack([prev.omega[prev_rows], curr.omega[curr_rows]], axis=1),
        np.stack([np.exp(prev.log_amplitude[prev_rows]), np.exp(curr.log_amplitude[curr_rows])], axis=1),
        np.stack([prev.phase[prev_rows], curr.phase[curr_rows]], axis=1),
        prev.track_id[prev_rows],
    )


def hop_segments(prev: Frame | None, curr: Frame, hop: int) -> Segments | None:
    """All segments of the hop ending at ``curr``, or None if nothing sounds.

    Order follows jun_synthesize_partials.m: births (fade in), deaths (fade out), then
    continuations.
    """
    none = np.zeros(0, np.intp)
    if prev is None or len(prev) == 0:
        match = SetMatch(none, none, none, np.arange(len(curr)))
    else:
        match = match_sets(prev.track_id, curr.track_id)
    parts = [_fade_in(curr, match.unmatched_b, hop)]
    if prev is not None:
        parts += [_fade_out(prev, match.unmatched_a, hop), _continue(prev, match.a, curr, match.b)]
    segments = Segments.concat(parts)
    return segments if len(segments) else None
