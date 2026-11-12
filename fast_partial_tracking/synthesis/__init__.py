"""Additive resynthesis of tracked partials.

* ``phase="measured"``: McAulay-Quatieri cubic phase interpolation between the measured
  phases (``Synthesizer``). Exact for unmodified analyses; matches MATLAB.
* ``phase="integrated"``: oscillator bank whose phase follows the frequency
  (``PhaseIntegratingSynthesizer``). Use after pitch shifting or time stretching.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Frame, Partials
from fast_partial_tracking.synthesis.mcaulay_quatieri import Synthesizer, cubic_phase_segments
from fast_partial_tracking.synthesis.oscillator import PhaseIntegratingSynthesizer
from fast_partial_tracking.synthesis.segments import Segments, hop_segments

PhaseMode = Literal["measured", "integrated"]

__all__ = [
    "PhaseIntegratingSynthesizer",
    "PhaseMode",
    "Segments",
    "Synthesizer",
    "cubic_phase_segments",
    "hop_segments",
    "make_synthesizer",
    "synthesize",
    "synthesize_frames",
]


def make_synthesizer(grid: FrameGrid, phase: PhaseMode = "measured") -> Synthesizer:
    """Creates a streaming synthesizer.

    Args:
        grid: Frame geometry of the analysis.
        phase: ``"measured"`` (McAulay-Quatieri) or ``"integrated"`` (oscillator bank).
    """
    synthesizers = {"measured": Synthesizer, "integrated": PhaseIntegratingSynthesizer}
    if phase not in synthesizers:
        raise ValueError(f"phase must be one of {sorted(synthesizers)}, got {phase!r}")
    return synthesizers[phase](grid)


def synthesize_frames(
    frames: Iterable[Frame], grid: FrameGrid, trim: bool = True, phase: PhaseMode = "measured"
) -> np.ndarray:
    """Synthesizes a sequence of frames.

    Args:
        frames: All frames 0..M-1 in order.
        grid: Completed frame geometry.
        trim: If True, removes the analysis zero padding so the output aligns with the
            input signal (length L). Otherwise returns the padded signal.
        phase: Phase rendering, see the module docstring.

    Returns:
        The resynthesized signal.
    """
    synth = make_synthesizer(grid, phase)
    blocks = [synth.push(frame) for frame in frames]
    blocks.append(synth.finish(grid.padded_length))
    out = np.concatenate(blocks)
    if trim:
        out = out[grid.pad_front : grid.pad_front + grid.signal_length]
    return out


def synthesize(partials: Partials, trim: bool = True, phase: PhaseMode | None = None) -> np.ndarray:
    """Resynthesizes a tracking result.

    Args:
        partials: Tracked partials.
        trim: If True, removes the analysis zero padding (output length L).
        phase: ``"measured"`` or ``"integrated"``. Default: ``"integrated"`` for results
            produced by a transform (recorded in their metadata), else ``"measured"``.

    Returns:
        The resynthesized signal.
    """
    phase = phase or partials.metadata.get("synthesis_phase", "measured")
    return synthesize_frames(partials.frames(), partials.grid, trim=trim, phase=phase)
