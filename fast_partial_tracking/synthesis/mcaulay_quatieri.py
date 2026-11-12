"""Additive resynthesis of tracked partials (``jun_synthesize_partials.m``).

Between consecutive frames each partial is rendered with linearly interpolated
amplitude and cubic phase interpolation (R. McAulay and T. Quatieri, "Speech
analysis/synthesis based on a sinusoidal representation", IEEE Trans. ASSP, 1986,
eqs. 34-37). Partials that start (or end) fade in (or out) over one hop at constant
frequency.
"""

from __future__ import annotations


import numpy as np

from fast_partial_tracking.compat.matlab import matlab_round
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Frame
from fast_partial_tracking.synthesis.segments import Segments, hop_segments


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


class Synthesizer:
    """Streaming McAulay-Quatieri synthesizer (measured phases, cubic interpolation).

    Feed frames in order with ``push``; each call returns the output samples between the
    previous frame's origin and this one's, so output can be played or written while the
    analysis is still running. Subclasses change how a hop is rendered (``_render``).

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
        return np.zeros(hop) if segments is None else self._render(segments, hop, frame)

    def _render(self, segments: Segments, hop: int, frame: Frame) -> np.ndarray:
        """Sum of all segments of one hop (MQ eqs. 34-37)."""
        return cubic_phase_segments(segments.omega, segments.amplitude, segments.phase, hop).sum(axis=0)

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
