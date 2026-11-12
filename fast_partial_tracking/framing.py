"""Frame geometry: hop size, DFT size, zero padding and frame positions.

All sample indices here are 0-based positions in the zero-padded signal
``[zeros(pad_front), x, zeros(pad_back)]``. Two geometries are supported:

* ``"centered"``: the model origin n = 0 is the window centre and frame k is centred on
  input sample k*H, so frame times are simply k*H/fs.
* ``"matlab"``: the SETUP section of ``jun_track_partials.m`` (front padding n_fft - H,
  origin one sample after the window centre for odd N).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from fast_partial_tracking.config import AnalysisConfig, Geometry


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
        origin: Index within a frame of the polynomial time origin n = 0.
        pad_front: Zeros prepended to the signal.
        signal_length: L, number of input samples; None while streaming.
        num_frames: M, number of frames; None while streaming.
        pad_back: Zeros appended to the signal; None while streaming.
        geometry: ``"centered"`` or ``"matlab"`` (see module docstring).
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
    geometry: Geometry = "centered"

    @classmethod
    def from_config(cls, fs: float, analysis: AnalysisConfig) -> FrameGrid:
        """Derives the frame geometry from the analysis settings.

        Args:
            fs: Sampling rate in Hz.
            analysis: Analysis settings.

        Returns:
            A grid without a signal length (see ``with_signal_length``).
        """
        n = analysis.window_length
        hop = analysis.resolved_hop_size
        n_fft = analysis.resolved_n_fft
        if analysis.geometry == "matlab":
            origin, pad_front = (n - n % 2) // 2 + 1, n_fft - hop
        else:
            origin = pad_front = n // 2
        return cls(
            fs=float(fs), window_length=n, hop=hop, n_dft=n_fft, origin=origin, pad_front=pad_front,
            geometry=analysis.geometry,
        )

    def with_signal_length(self, signal_length: int) -> FrameGrid:
        """Returns a copy with the frame count and tail padding for L input samples.

        The centered geometry places frames until one is centred at or after sample L;
        the MATLAB geometry uses ``ceil((Ndft + L - 2H - 1)/H + 1)`` frames.

        Args:
            signal_length: Number of input samples L.

        Returns:
            A completed grid where ``num_frames`` and ``pad_back`` are set.
        """
        if self.geometry == "matlab":
            num_frames = math.ceil((self.n_dft + signal_length - 2 * self.hop - 1) / self.hop + 1)
            last_frame_end = (num_frames - 1) * self.hop + self.n_dft
        else:
            num_frames = math.ceil(signal_length / self.hop) + 1
            last_frame_end = (num_frames - 1) * self.hop + self.window_length
        pad_back = last_frame_end - (self.pad_front + signal_length)
        return replace(self, signal_length=signal_length, num_frames=num_frames, pad_back=pad_back)

    def frame_times(self) -> np.ndarray:
        """Time in seconds of every frame origin (requires a signal length)."""
        return np.asarray(self.time_seconds(np.arange(self._require(self.num_frames))), dtype=float)

    def frame_index(self, time_s: float | np.ndarray) -> int | np.ndarray:
        """Index of the frame whose origin is nearest to ``time_s``."""
        index = np.rint((np.asarray(time_s) * self.fs + self.pad_front - self.origin) / self.hop).astype(int)
        return int(index) if index.ndim == 0 else index

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
