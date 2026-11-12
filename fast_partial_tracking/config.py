"""Analysis and tracking settings.

Settings are validated, never silently rewritten. ``AnalysisConfig.matlab`` builds the
exact configuration that ``jun_track_partials.m`` derives from its arguments (including
its power-of-two rounding and frame geometry) for reproducing the MATLAB results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from fast_partial_tracking.compat.matlab import matlab_round, nextpow2

WindowName = Literal["hann", "blackman_harris"]
Geometry = Literal["centered", "matlab"]
ThresholdReference = Literal["signal", "full_scale"]
AmplitudeGuard = Literal["center", "legacy"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Short-term analysis settings (Section 3.2 of the paper).

    Attributes:
        window_length: N, analysis window length in samples (odd lengths center the
            window exactly on a sample).
        hop_size: H in samples. Default: N // 8 for Blackman-Harris, N // 4 for Hann.
        n_fft: DFT size (>= N). Default: the next power of two >= 2N.
        window: Analysis window, ``"blackman_harris"`` or ``"hann"``.
        poly_order: Q, the polynomial order of the short-term model (eq. 4). Q = 1
            estimates frequency and amplitude slope; Q = 2 adds their derivatives.
        threshold_db: Peaks more than this many dB below the reference level are ignored
            (must be <= 0).
        threshold_reference: What ``threshold_db`` is relative to: ``"signal"`` (the
            loudest frame of the input; needs the whole signal), ``"full_scale"`` (a
            sinusoid of amplitude 1), or an absolute level in dB (DFT magnitude).
        ddm_bins: Bins on each side of a peak (peak included) entering the DDM system.
            Default: Q times the DFT oversampling factor.
        geometry: ``"centered"`` puts the model origin at the window centre and frame k
            at input sample k*H; ``"matlab"`` reproduces the MATLAB frame placement.
        amplitude_guard: Rejects exploding estimates. ``"center"`` drops a peak whose
            estimated amplitude at the frame centre exceeds the largest sample (also used
            by the MATLAB code since fast-partial-tracking 02c2fd6); ``"legacy"`` is the
            earlier jun_ddm.m check (log-amplitude compared with a linear bound, which almost
            never triggers and lets Q >= 3 estimates explode).
        max_peaks: Keep at most this many peaks per frame, the loudest by estimated
            amplitude (None for no limit). Limits the number of simultaneous partials.
    """

    window_length: int = 2047
    hop_size: int | None = None
    n_fft: int | None = None
    window: WindowName = "blackman_harris"
    poly_order: int = 2
    threshold_db: float = -60.0
    threshold_reference: ThresholdReference | float = "signal"
    ddm_bins: int | None = None
    geometry: Geometry = "centered"
    amplitude_guard: AmplitudeGuard = "center"
    max_peaks: int | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.window_length, (int, np.integer)) and self.window_length >= 4,
                 f"window_length must be an integer >= 4, got {self.window_length!r}")
        _require(self.window in ("hann", "blackman_harris"), f"unknown window {self.window!r}")
        _require(self.hop_size is None or 1 <= self.hop_size <= self.window_length,
                 f"hop_size must be in [1, window_length], got {self.hop_size}")
        _require(self.n_fft is None or self.n_fft >= self.window_length,
                 f"n_fft must be >= window_length, got {self.n_fft}")
        _require(isinstance(self.poly_order, (int, np.integer)) and self.poly_order >= 1,
                 f"poly_order must be an integer >= 1, got {self.poly_order!r}")
        _require(self.threshold_db <= 0, f"threshold_db is relative to the reference and must be <= 0, got {self.threshold_db}")
        _require(self.ddm_bins is None or self.ddm_bins >= 1, f"ddm_bins must be >= 1, got {self.ddm_bins}")
        _require(self.geometry in ("centered", "matlab"), f"unknown geometry {self.geometry!r}")
        _require(self.amplitude_guard in ("center", "legacy"), f"unknown amplitude_guard {self.amplitude_guard!r}")
        _require(self.max_peaks is None or (isinstance(self.max_peaks, (int, np.integer)) and self.max_peaks >= 1),
                 f"max_peaks must be None or an integer >= 1, got {self.max_peaks!r}")
        _require(isinstance(self.threshold_reference, (int, float)) or self.threshold_reference in ("signal", "full_scale"),
                 f"threshold_reference must be 'signal', 'full_scale' or a dB value, got {self.threshold_reference!r}")

    @classmethod
    def from_duration(cls, fs: float, window_ms: float = 46.0, hop_ms: float | None = None, **kwargs: Any) -> AnalysisConfig:
        """Builds a configuration from durations instead of sample counts.

        Args:
            fs: Sampling rate in Hz.
            window_ms: Window duration; rounded to the nearest odd number of samples.
            hop_ms: Hop duration (default: the window's default hop).
            **kwargs: Other ``AnalysisConfig`` fields.

        Returns:
            The configuration.
        """
        n = int(round(window_ms * 1e-3 * fs)) // 2 * 2 + 1
        if hop_ms is not None:
            kwargs["hop_size"] = max(1, int(round(hop_ms * 1e-3 * fs)))
        return cls(window_length=n, **kwargs)

    @classmethod
    def matlab(
        cls,
        window_length: int = 2**11 - 1,
        hop_factor: int = 4,
        oversample: int = 1,
        peak_threshold_db: float = -40.0,
        order: int = 2,
    ) -> AnalysisConfig:
        """The configuration ``jun_track_partials.m`` derives from its arguments.

        Reproduces its sanitizing exactly: N = max(4, round(N)); hop factor and
        oversampling rounded up to powers of two; Blackman-Harris and a hop factor of at
        least 8 when oversampling, else Hann and at least 4; its DFT size formula; the sign
        of the threshold ignored; and MATLAB's frame geometry.

        Args:
            window_length: N.
            hop_factor: H = N / hop_factor.
            oversample: DFT oversampling factor.
            peak_threshold_db: Peak threshold below the loudest frame.
            order: Q.

        Returns:
            A configuration that reproduces the MATLAB analysis.
        """
        n = int(max(4, matlab_round(window_length)))
        oversample = int(2 ** nextpow2(max(1, oversample)))
        hop_factor = int(2 ** nextpow2(max(1, hop_factor)))
        hop_factor = max(8 if oversample > 1 else 4, hop_factor)
        n_fft = int(2 ** matlab_round(np.log2(oversample * 2 ** matlab_round(np.log2(n)))))
        if n_fft < n:
            n_fft = int(2 ** matlab_round(np.log2(n) + 1))
        q = int(max(1, matlab_round(order)))
        return cls(
            window_length=n,
            hop_size=int(matlab_round((n - n % 2) / hop_factor)),
            n_fft=n_fft,
            window="blackman_harris" if oversample > 1 else "hann",
            poly_order=q,
            threshold_db=-abs(float(peak_threshold_db)),
            threshold_reference="signal",
            ddm_bins=q * oversample,
            geometry="matlab",
        )

    @property
    def resolved_hop_size(self) -> int:
        """H, with the window-dependent default applied."""
        if self.hop_size is not None:
            return int(self.hop_size)
        return max(1, self.window_length // (8 if self.window == "blackman_harris" else 4))

    @property
    def resolved_n_fft(self) -> int:
        """DFT size, with the default (next power of two >= 2N) applied."""
        return int(self.n_fft) if self.n_fft is not None else int(2 ** nextpow2(2 * self.window_length))

    @property
    def resolved_ddm_bins(self) -> int:
        """Bins per side of a peak used by the DDM, with the default applied."""
        if self.ddm_bins is not None:
            return int(self.ddm_bins)
        oversampling = self.resolved_n_fft // int(2 ** matlab_round(np.log2(self.window_length)))
        return self.poly_order * max(1, oversampling)

    def to_dict(self) -> dict[str, Any]:
        """Returns the settings as a plain dictionary (for file metadata)."""
        return asdict(self)


@dataclass(frozen=True)
class TrackingConfig:
    """Peak-to-peak assignment settings (Sections 3.3-3.4 of the paper).

    Attributes:
        delta: δ in (0, 1). Relative preference between useful and spurious assignments
            (eq. 9): larger values favor useful assignments.
        zeta_f_hz: ζ_f, the frequency difference |Δf| (Hz) at the frames' midpoint where an
            assignment switches from useful to spurious (eq. 12).
        zeta_a_db: ζ_a, the corresponding log-amplitude difference |Δa| in dB (eq. 13).
    """

    delta: float = 0.2
    zeta_f_hz: float = 50.0
    zeta_a_db: float = 15.0

    def __post_init__(self) -> None:
        _require(0.0 < self.delta < 1.0, f"delta must be in (0, 1), got {self.delta}")
        _require(self.zeta_f_hz > 0, f"zeta_f_hz must be > 0, got {self.zeta_f_hz}")
        _require(self.zeta_a_db > 0, f"zeta_a_db must be > 0, got {self.zeta_a_db}")

    @classmethod
    def matlab(cls, delta: float = 0.2, zeta_f: float = 50.0, zeta_a: float = 15.0) -> TrackingConfig:
        """Settings as ``jun_track_partials.m`` sanitizes them (ζ clamped to >= 0.1)."""
        return cls(delta=delta, zeta_f_hz=max(zeta_f, 0.1), zeta_a_db=max(zeta_a, 0.1))

    @property
    def zeta_a_log(self) -> float:
        """ζ_a in natural-log amplitude units (the model's log-amplitude scale)."""
        return self.zeta_a_db / 20.0 * np.log(10.0)

    def to_dict(self) -> dict[str, float]:
        """Returns the settings as a plain dictionary (for file metadata)."""
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
