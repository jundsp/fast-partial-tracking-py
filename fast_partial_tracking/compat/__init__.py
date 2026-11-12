"""Compatibility with the MATLAB implementation.

``AnalysisConfig.matlab`` / ``TrackingConfig.matlab`` reproduce its settings. This package
holds the MATLAB-style numeric helpers (``compat.matlab``) and its ``.bin`` partials
format (``compat.matlab_bin``).
"""

from __future__ import annotations

from fast_partial_tracking.compat.matlab import matlab_round, nextpow2

__all__ = ["matlab_round", "nextpow2"]
