"""Fast partial tracking of audio through linear programming.

Python implementation of J. Neri and P. Depalle, "Fast Partial Tracking of Audio with
Real-Time Capability through Linear Programming", Proc. DAFx-18, Aveiro, Portugal,
2018, pp. 326-333.

Example:
    >>> import fast_partial_tracking as fpt
    >>> partials = fpt.track(x, fs, fpt.AnalysisConfig(window_length=2047, oversample=2))
    >>> y = fpt.synthesize(partials)
    >>> fpt.rsnr(x, y)
    >>> partials.save("partials.parquet")
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from fast_partial_tracking import data, io, plot, selection, transforms
from fast_partial_tracking.config import AnalysisConfig, TrackingConfig
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.metrics import rsnr
from fast_partial_tracking.partials import Frame, Partials, Track
from fast_partial_tracking.synthesis import PhaseIntegratingSynthesizer, Synthesizer, synthesize, synthesize_frames
from fast_partial_tracking.tracking.tracker import PartialTracker, track

try:
    __version__ = _version("fast-partial-tracking")
except PackageNotFoundError:  # pragma: no cover - running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "AnalysisConfig",
    "Frame",
    "FrameGrid",
    "PartialTracker",
    "Partials",
    "PhaseIntegratingSynthesizer",
    "Synthesizer",
    "Track",
    "TrackingConfig",
    "__version__",
    "data",
    "io",
    "plot",
    "rsnr",
    "selection",
    "synthesize",
    "synthesize_frames",
    "track",
    "transforms",
]
