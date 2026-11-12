"""Numerical helpers that mimic MATLAB built-ins, used by the MATLAB-compatible preset."""

from __future__ import annotations

import numpy as np


def matlab_round(x: float | np.ndarray) -> float | np.ndarray:
    """Rounds half away from zero, like MATLAB's ``round`` (NumPy rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def nextpow2(x: float) -> int:
    """Returns the smallest p with 2**p >= |x| (MATLAB ``nextpow2``)."""
    return int(np.ceil(np.log2(abs(x)))) if x != 0 else 0
