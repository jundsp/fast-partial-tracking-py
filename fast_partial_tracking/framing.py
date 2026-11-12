"""Frame geometry: hop size, DFT size, zero padding and frame positions.

Reproduces the SETUP section of ``jun_track_partials.m``. All sample indices here are
0-based positions in the zero-padded signal ``[zeros(pad_front), x, zeros(pad_back)]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from fast_partial_tracking.config import AnalysisConfig, matlab_round


@dataclass(frozen=True)
class FrameGrid:
    """Positions of the short-term analysis frames.

    Frame k (0-based) covers padded samples ``k*hop .. k*hop + window_length - 1``. Its
    model time origin n = 0 (Section 3.2, eqs. 4-7) is the sample ``k*hop + origin``.

    Attributes:
        fs: Sampling rate in Hz.
        window_length: N, samples per analysis frame.
        hop: H, hop size in samples.
        n_dft: DFT size (zero-padded FFT length).
        origin: Index within a frame of the polynomial time origin n = 0. This is
            MATLAB's 1-based ``n_center`` used as a 0-based offset, so for odd N the
            origin sits one sample after the window's peak.
        pad_front: Zeros prepended to the signal (``Ndft - H``).
        signal_length: L, number of input samples; None while streaming.
        num_frames: M, number of frames; None while streaming.
        pad_back: Zeros appended to the signal; None while streaming.
    """

    fs: float
    window_length: int
    hop: int
    n_dft: int
    origin: int
    pad_front: int
    signal_length: int | None = None
    num_frames: int | None = None
    pad_back: int | None = None

    @classmethod
    def from_config(cls, fs: float, analysis: AnalysisConfig) -> FrameGrid:
        """Derives the frame geometry from the analysis settings.

        Args:
            fs: Sampling rate in Hz.
            analysis: Analysis settings.

        Returns:
            A grid without a signal length (see ``with_signal_length``).
        """
        n = analysis.effective_window_length
        hop = int(matlab_round((n - n % 2) / analysis.effective_hop_factor))
        n_dft = int(2 ** matlab_round(np.log2(analysis.effective_oversample * 2 ** matlab_round(np.log2(n)))))
        if n_dft < n:
            n_dft = int(2 ** matlab_round(np.log2(n) + 1))
        origin = (n - n % 2) // 2 + 1
        return cls(fs=float(fs), window_length=n, hop=hop, n_dft=n_dft, origin=origin, pad_front=n_dft - hop)

    def with_signal_length(self, signal_length: int) -> FrameGrid:
        """Returns a copy with the frame count and tail padding for L input samples.

        Args:
            signal_length: Number of input samples L.

        Returns:
            A completed grid where ``num_frames`` and ``pad_back`` are set.
        """
        num_frames = math.ceil((self.n_dft + signal_length - 2 * self.hop - 1) / self.hop + 1)
        pad_back = (num_frames - 1) * self.hop + self.n_dft - (self.pad_front + signal_length)
        return replace(self, signal_length=signal_length, num_frames=num_frames, pad_back=pad_back)

    @property
    def padded_length(self) -> int:
        """Length of the zero-padded signal (requires a signal length)."""
        return self.pad_front + self._require(self.signal_length) + self._require(self.pad_back)

    @property
    def offsets(self) -> np.ndarray:
        """Model time n of every sample in a frame, relative to the origin (shape (N,))."""
        return np.arange(self.window_length, dtype=float) - self.origin

    @property
    def midpoint_offsets(self) -> tuple[int, int]:
        """Model times (n_curr, n_prev) at which consecutive frames are compared.

        Eqs. (10)-(11) compare frame k-1 at +H/2 with frame k at -H/2, i.e. both at the
        midpoint between the two frame origins (n = -floor(H/2) and n = ceil(H/2)).
        """
        return -(self.hop // 2), -(-self.hop // 2)

    def center(self, frame: int | np.ndarray) -> int | np.ndarray:
        """Padded sample index of a frame's time origin n = 0."""
        return np.asarray(frame) * self.hop + self.origin

    def time_seconds(self, frame: int | np.ndarray) -> float | np.ndarray:
        """Time of a frame's origin in seconds, relative to the first input sample."""
        return (self.center(frame) - self.pad_front) / self.fs

    def to_dict(self) -> dict[str, float | int | None]:
        """Returns the grid as a plain dictionary (for file metadata)."""
        return asdict(self)

    @staticmethod
    def _require(value: int | None) -> int:
        if value is None:
            raise ValueError("signal length is unknown; call with_signal_length() first")
        return value
