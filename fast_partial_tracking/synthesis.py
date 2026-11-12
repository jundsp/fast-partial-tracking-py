"""Additive resynthesis of tracked partials (``jun_synthesize_partials.m``).

Between consecutive frames each partial is rendered with linearly interpolated
amplitude and cubic phase interpolation (R. McAulay and T. Quatieri, "Speech
analysis/synthesis based on a sinusoidal representation", IEEE Trans. ASSP, 1986,
eqs. 34-37). Partials that start (or end) fade in (or out) over one hop at constant
frequency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from fast_partial_tracking.config import matlab_round
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.labeling import SetMatch, match_sets
from fast_partial_tracking.partials import Frame, Partials


def cubic_phase_segments(
    omega: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    hop: int,
) -> np.ndarray:
    """Renders sinusoid segments between two parameter points, hop samples apart.

    The phase is the cubic polynomial
    theta(t) = phi_1 + omega_1 t + alpha t**2 + beta t**3 (MQ eq. 37), whose coefficients
    (eq. 34) match both endpoint phases and frequencies, with the 2*pi*M unwrapping term
    chosen for maximal smoothness (eq. 36).

    Args:
        omega: Start/end angular frequencies in radians/sample, shape (P, 2).
        amplitude: Start/end linear amplitudes of the complex model, shape (P, 2).
        phase: Start/end phases, shape (P, 2).
        hop: Segment length H in samples.

    Returns:
        Real samples t = 0..H-1 of each segment, shape (P, H). The factor 2 converts the
        analytic model of eq. (1) into a real sinusoid.
    """
    t = np.arange(hop + 1, dtype=float)
    w1, w2 = omega[:, :1], omega[:, 1:]
    p1, p2 = phase[:, :1], phase[:, 1:]
    m_star = matlab_round(((p1 + w1 * hop - p2) + (w2 - w1) * hop / 2.0) / (2.0 * np.pi))  # eq. (36)
    unwrapped = p2 - p1 - w1 * hop + 2.0 * np.pi * m_star
    slope = w2 - w1
    alpha = 3.0 / hop**2 * unwrapped - 1.0 / hop * slope  # eq. (34)
    beta = -2.0 / hop**3 * unwrapped + 1.0 / hop**2 * slope
    theta = p1 + w1 * t + alpha * t**2 + beta * t**3  # eq. (37)
    a1, a2 = amplitude[:, :1], amplitude[:, 1:]
    envelope = a1 + (a2 - a1) * (t / hop)
    return (2.0 * envelope * np.cos(theta))[:, :hop]


@dataclass(frozen=True)
class Segments:
    """Endpoint parameters of sinusoid segments spanning one hop.

    Attributes:
        omega: Start/end angular frequencies (radians/sample), shape (P, 2).
        amplitude: Start/end linear amplitudes of the complex model, shape (P, 2).
        phase: Start/end phases, shape (P, 2).
    """

    omega: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray

    def __len__(self) -> int:
        return int(self.omega.shape[0])

    @classmethod
    def concat(cls, parts: list[Segments]) -> Segments:
        return cls(*(np.concatenate(columns) for columns in zip(*((p.omega, p.amplitude, p.phase) for p in parts))))

    def render(self, hop: int) -> np.ndarray:
        """Sum of all segments, ``hop`` samples."""
        return cubic_phase_segments(self.omega, self.amplitude, self.phase, hop).sum(axis=0)


def _fade_in(frame: Frame, rows: np.ndarray, hop: int) -> Segments:
    """Partials born at ``frame``: fade in over the preceding hop at constant frequency."""
    w, ph = frame.omega[rows], frame.phase[rows]
    return Segments(
        np.stack([w, w], axis=1),
        np.stack([np.zeros_like(w), np.exp(frame.log_amp[rows])], axis=1),
        np.stack([ph - w * hop, ph], axis=1),
    )


def _fade_out(frame: Frame, rows: np.ndarray, hop: int) -> Segments:
    """Partials ending at ``frame``: fade out over the following hop at constant frequency."""
    w, ph = frame.omega[rows], frame.phase[rows]
    return Segments(
        np.stack([w, w], axis=1),
        np.stack([np.exp(frame.log_amp[rows]), np.zeros_like(w)], axis=1),
        np.stack([ph, ph + w * hop], axis=1),
    )


def _continue(prev: Frame, prev_rows: np.ndarray, curr: Frame, curr_rows: np.ndarray) -> Segments:
    """Partials present in both frames: interpolate between their parameters."""
    return Segments(
        np.stack([prev.omega[prev_rows], curr.omega[curr_rows]], axis=1),
        np.stack([np.exp(prev.log_amp[prev_rows]), np.exp(curr.log_amp[curr_rows])], axis=1),
        np.stack([prev.phase[prev_rows], curr.phase[curr_rows]], axis=1),
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


class Synthesizer:
    """Streaming McAulay-Quatieri synthesizer.

    Feed frames in order with ``push``; each call returns the output samples between the
    previous frame's origin and this one's, so output can be played or written while the
    analysis is still running.

    Args:
        grid: Frame geometry of the analysis (the padded time axis).
    """

    def __init__(self, grid: FrameGrid) -> None:
        self.grid = grid
        self._prev: Frame | None = None
        self._position = 0

    def push(self, frame: Frame) -> np.ndarray:
        """Synthesizes the hop that ends at ``frame``.

        Args:
            frame: The next frame (indices must be consecutive, starting at 0).

        Returns:
            Samples of the padded output from the previous frame origin (or 0 for the first
            frame) up to, but excluding, this frame's origin.
        """
        expected = 0 if self._prev is None else self._prev.index + 1
        if frame.index != expected:
            raise ValueError(f"expected frame {expected}, got {frame.index}")
        start = self._position
        hop = frame.center - start
        prev = self._prev
        self._prev = frame
        self._position = frame.center
        if hop <= 0:
            return np.zeros(0)

        segments = hop_segments(prev, frame, hop)
        return np.zeros(hop) if segments is None else segments.render(hop)

    def finish(self, padded_length: int) -> np.ndarray:
        """Returns the silent tail after the last frame origin.

        Args:
            padded_length: Total length of the padded output.

        Returns:
            Zeros up to ``padded_length``.
        """
        tail = max(padded_length - self._position, 0)
        self._position += tail
        return np.zeros(tail)


def synthesize_frames(frames: Iterable[Frame], grid: FrameGrid, trim: bool = True) -> np.ndarray:
    """Synthesizes a sequence of frames.

    Args:
        frames: All frames 0..M-1 in order.
        grid: Completed frame geometry.
        trim: If True, removes the analysis zero padding so the output aligns with the
            input signal (length L). Otherwise returns the padded signal.

    Returns:
        The resynthesized signal.
    """
    synth = Synthesizer(grid)
    blocks = [synth.push(frame) for frame in frames]
    blocks.append(synth.finish(grid.padded_length))
    out = np.concatenate(blocks)
    if trim:
        out = out[grid.pad_front : grid.pad_front + grid.signal_length]
    return out


def synthesize(partials: Partials, trim: bool = True) -> np.ndarray:
    """Resynthesizes a tracking result.

    Args:
        partials: Tracked partials.
        trim: If True, removes the analysis zero padding (output length L).

    Returns:
        The resynthesized signal.
    """
    return synthesize_frames(partials.frames(), partials.grid, trim=trim)
