"""User-facing configuration for analysis and tracking.

The defaults and the input sanitizing reproduce the optional-argument handling of
``jun_track_partials.m`` so that the same settings yield the same analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def matlab_round(x: float | np.ndarray) -> float | np.ndarray:
    """Rounds half away from zero, like MATLAB's ``round``.

    Python's built-in ``round`` and ``np.round`` round half to even, which would make
    window/hop/DFT sizes differ from the MATLAB implementation for exact .5 cases.

    Args:
        x: Scalar or array to round.

    Returns:
        The rounded value(s), as float.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def nextpow2(x: float) -> int:
    """Returns the smallest p with 2**p >= |x| (MATLAB ``nextpow2``)."""
    return int(np.ceil(np.log2(abs(x)))) if x != 0 else 0


@dataclass(frozen=True)
class AnalysisConfig:
    """Short-term analysis settings (Section 3.2 of the paper).

    Attributes:
        window_length: N, analysis window length in samples (the paper's frame spans
            N + 1 samples, n = -N/2..N/2).
        hop_factor: Hop size H = N / hop_factor. Rounded up to a power of two and raised
            to at least 4 (Hann) or 8 (Blackman-Harris), as in MATLAB.
        oversample: DFT oversampling factor, rounded up to a power of two. Values > 1
            select the Blackman-Harris window, otherwise a Hann window is used.
        peak_threshold_db: Spectral peaks more than this many dB below the reference
            level are ignored. The sign is ignored (-40 and 40 are equivalent).
        order: Q, the polynomial order of the short-term model (eq. 4). Q = 1 estimates
            frequency and log-amplitude slope (damping); Q = 2 adds their derivatives.
    """

    window_length: int = 2**11 - 1
    hop_factor: int = 4
    oversample: int = 1
    peak_threshold_db: float = -40.0
    order: int = 2

    def __post_init__(self) -> None:
        if self.window_length < 1:
            raise ValueError(f"window_length must be positive, got {self.window_length}")
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")

    @property
    def effective_window_length(self) -> int:
        """N after MATLAB's sanitizing: ``max(4, round(N))``."""
        return int(max(4, matlab_round(self.window_length)))

    @property
    def effective_oversample(self) -> int:
        """Oversampling factor rounded up to a power of two."""
        return int(2 ** nextpow2(max(1, self.oversample)))

    @property
    def effective_order(self) -> int:
        """Q after sanitizing: ``max(1, round(Q))``."""
        return int(max(1, matlab_round(self.order)))

    @property
    def effective_hop_factor(self) -> int:
        """Hop factor rounded up to a power of two with the window-dependent minimum."""
        hop_factor = int(2 ** nextpow2(max(1, self.hop_factor)))
        return max(8 if self.effective_oversample > 1 else 4, hop_factor)

    @property
    def window_name(self) -> str:
        """``"blackman_harris"`` when oversampling, else ``"hann"``."""
        return "blackman_harris" if self.effective_oversample > 1 else "hann"

    def to_dict(self) -> dict[str, float | int]:
        """Returns the settings as a plain dictionary (for file metadata)."""
        return asdict(self)


@dataclass(frozen=True)
class TrackingConfig:
    """Peak-to-peak assignment settings (Section 3.3 of the paper).

    Attributes:
        delta: δ in (0, 1). Changes the relative preference between useful and spurious
            assignments (eq. 9): larger values favor useful assignments.
        zeta_f: ζ_f in Hz, the midpoint frequency difference |Δf| at which an
            assignment switches from useful to spurious (eq. 12).
        zeta_a: ζ_a in dB, the corresponding log-amplitude difference |Δa| (eq. 13).
    """

    delta: float = 0.2
    zeta_f: float = 50.0
    zeta_a: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")

    @property
    def zeta_f_hz(self) -> float:
        """ζ_f clamped to at least 0.1 Hz (MATLAB sanitizing)."""
        return max(self.zeta_f, 0.1)

    @property
    def zeta_a_log(self) -> float:
        """ζ_a clamped to at least 0.1 dB and converted to natural-log units."""
        return max(self.zeta_a, 0.1) / 20.0 * np.log(10.0)

    def to_dict(self) -> dict[str, float]:
        """Returns the settings as a plain dictionary (for file metadata)."""
        return asdict(self)
