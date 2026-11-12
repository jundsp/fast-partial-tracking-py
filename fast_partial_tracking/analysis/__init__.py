"""Short-term analysis (Section 3.2 of the paper).

Windows, spectral peak picking and the Distribution Derivative Method (DDM), which
estimates the complex coefficients of the generalized sinusoidal model (eqs. 4-7) for
every peak of a frame.
"""

from __future__ import annotations

from fast_partial_tracking.analysis.ddm import DdmEstimator, FrameEstimate
from fast_partial_tracking.analysis.model import (
    angular_frequency,
    frame_parameters,
    log_amplitude,
    phase,
    polynomial_basis,
    polynomial_basis_derivative,
)
from fast_partial_tracking.analysis.peaks import Peaks, pick_peaks
from fast_partial_tracking.analysis.windows import blackman_harris, get_window, hann

__all__ = [
    "DdmEstimator",
    "FrameEstimate",
    "Peaks",
    "angular_frequency",
    "blackman_harris",
    "frame_parameters",
    "get_window",
    "hann",
    "log_amplitude",
    "phase",
    "pick_peaks",
    "polynomial_basis",
    "polynomial_basis_derivative",
]
