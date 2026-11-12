"""Oscillator-bank synthesis with phase integrated from frequency.

The McAulay-Quatieri synthesizer interpolates between *measured* phases, which is exact
for an unmodified analysis but breaks once frequencies or frame times are changed
(pitch shifting, time stretching): the measured phases no longer fit the new trajectory.
This synthesizer instead accumulates each partial's phase from its linearly interpolated
frequency, so any transformed set of partials renders smoothly. A partial starts from
its measured phase at birth.
"""

from __future__ import annotations

import numpy as np

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Frame
from fast_partial_tracking.synthesis.mcaulay_quatieri import Synthesizer
from fast_partial_tracking.synthesis.segments import Segments


class PhaseIntegratingSynthesizer(Synthesizer):
    """Streaming oscillator bank whose phases follow the integrated frequency.

    Args:
        grid: Frame geometry of the analysis (the padded time axis).
    """

    def __init__(self, grid: FrameGrid) -> None:
        super().__init__(grid)
        self._phase: dict[int, float] = {}

    def _render(self, segments: Segments, hop: int, frame: Frame) -> np.ndarray:
        """Renders one hop and carries every surviving partial's phase to the next."""
        start = self._start_phase(segments)
        t = np.arange(hop, dtype=float)
        w1, w2 = segments.omega[:, :1], segments.omega[:, 1:]
        # Phase of a linear frequency ramp: integral of w1 + (w2 - w1) t / H.
        theta = start[:, None] + w1 * t + (w2 - w1) * t**2 / (2.0 * hop)
        a1, a2 = segments.amplitude[:, :1], segments.amplitude[:, 1:]
        envelope = a1 + (a2 - a1) * (t / hop)
        end = np.mod(start + (segments.omega[:, 0] + segments.omega[:, 1]) * hop / 2.0, 2.0 * np.pi)
        alive = set(frame.track_id.tolist())
        self._phase = {int(i): float(p) for i, p in zip(segments.track_id, end) if int(i) in alive}
        return (2.0 * envelope * np.cos(theta)).sum(axis=0)

    def _start_phase(self, segments: Segments) -> np.ndarray:
        """Carried phase for continuing partials, measured start phase for new ones."""
        carried = [self._phase.get(int(i)) for i in segments.track_id]
        return np.array([p if p is not None else segments.phase[k, 0] for k, p in enumerate(carried)], dtype=float)
