"""Containers for tracked partials.

``Frame`` holds the partials active in one analysis frame (what the streaming tracker
emits). ``Partials`` holds a whole analysis as flat columns, one row per (frame, partial),
plus an index of where each frame starts, so per-frame views are slices with no copy.
This "long" layout is also the on-disk schema (see ``fast_partial_tracking.io``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fast_partial_tracking.framing import FrameGrid


@dataclass(frozen=True)
class Frame:
    """Partials active in one analysis frame, evaluated at the frame origin n = 0.

    Parameters are stored in the model's native units (eqs. 5-7) so no precision is lost;
    use ``frequency_hz``, ``amplitude`` and ``amplitude_db`` for physical units.

    Attributes:
        index: 0-based frame number k.
        center: Index of the frame origin in the zero-padded signal.
        time_s: Time of the frame origin in seconds, relative to the first input sample.
        fs: Sampling rate in Hz.
        omega: Angular frequency in radians/sample, shape (R,).
        log_amplitude: Natural log of the complex (analytic) amplitude, shape (R,).
        phase: Phase in radians (not wrapped), shape (R,).
        track_id: 0-based partial label (Section 3.6), shape (R,).
        alpha: Full short-term model coefficients (eq. 4), shape (R, Q + 1), or None.
    """

    index: int
    center: int
    time_s: float
    fs: float
    omega: np.ndarray
    log_amplitude: np.ndarray
    phase: np.ndarray
    track_id: np.ndarray
    alpha: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.omega.size)

    @property
    def frequency_hz(self) -> np.ndarray:
        """Frequencies in Hz."""
        return self.omega * self.fs / (2.0 * np.pi)

    @property
    def amplitude(self) -> np.ndarray:
        """Linear amplitude of each real sinusoid, 2 * exp(log_amplitude)."""
        return 2.0 * np.exp(self.log_amplitude)

    @property
    def amplitude_db(self) -> np.ndarray:
        """Amplitude in dB (20 log10 of ``amplitude``)."""
        return 20.0 * np.log10(self.amplitude)


@dataclass
class Partials:
    """A complete partial-tracking result in long (one row per partial per frame) form.

    Attributes:
        grid: Frame geometry, including signal length and frame count.
        frame: Frame index of each row, non-decreasing.
        track_id: 0-based partial label of each row.
        omega: Angular frequency (radians/sample) of each row.
        log_amplitude: Natural log of the complex amplitude of each row.
        phase: Phase (radians) of each row.
        alpha: Model coefficients of each row, shape (rows, Q + 1), or None.
        num_tracks: Number of distinct partials (labels are 0..num_tracks-1).
        metadata: Settings and provenance stored alongside the data in files.
    """

    grid: FrameGrid
    frame: np.ndarray
    track_id: np.ndarray
    omega: np.ndarray
    log_amplitude: np.ndarray
    phase: np.ndarray
    alpha: np.ndarray | None = None
    num_tracks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grid.num_frames is None:
            raise ValueError("Partials requires a complete FrameGrid (with signal length)")
        self.frame = np.asarray(self.frame, dtype=np.int64)
        if self.frame.size and np.any(np.diff(self.frame) < 0):
            raise ValueError("rows must be ordered by frame")
        self._offsets = np.searchsorted(self.frame, np.arange(self.grid.num_frames + 1))

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[Frame],
        grid: FrameGrid,
        num_tracks: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Partials:
        """Concatenates per-frame results into one table.

        Args:
            frames: Frames in increasing order of index.
            grid: Completed frame geometry.
            num_tracks: Number of partials; inferred from the labels if None.
            metadata: Optional settings/provenance.

        Returns:
            The combined partials.
        """
        frames = list(frames)
        counts = [len(f) for f in frames]
        has_alpha = bool(frames) and all(f.alpha is not None for f in frames)
        track_id = np.concatenate([f.track_id for f in frames]) if frames else np.zeros(0, np.int64)
        if num_tracks is None:
            num_tracks = int(track_id.max()) + 1 if track_id.size else 0
        return cls(
            grid=grid,
            frame=np.repeat([f.index for f in frames], counts).astype(np.int64),
            track_id=track_id.astype(np.int64),
            omega=_concat([f.omega for f in frames]),
            log_amplitude=_concat([f.log_amplitude for f in frames]),
            phase=_concat([f.phase for f in frames]),
            alpha=np.concatenate([f.alpha for f in frames]) if has_alpha else None,
            num_tracks=num_tracks,
            metadata=dict(metadata or {}),
        )

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def num_frames(self) -> int:
        """Number of analysis frames M."""
        return int(self.grid.num_frames)  # type: ignore[arg-type]

    @property
    def frequency_hz(self) -> np.ndarray:
        """Frequency of each row in Hz."""
        return self.omega * self.grid.fs / (2.0 * np.pi)

    @property
    def amplitude(self) -> np.ndarray:
        """Linear amplitude of each row's real sinusoid, 2 * exp(log_amplitude)."""
        return 2.0 * np.exp(self.log_amplitude)

    @property
    def amplitude_db(self) -> np.ndarray:
        """Amplitude of each row in dB."""
        return 20.0 * np.log10(self.amplitude)

    @property
    def time_s(self) -> np.ndarray:
        """Time of each row's frame in seconds."""
        return np.asarray(self.grid.time_seconds(self.frame), dtype=float)

    def frame_slice(self, index: int) -> slice:
        """Row range of frame ``index``."""
        return slice(int(self._offsets[index]), int(self._offsets[index + 1]))

    def get_frame(self, index: int) -> Frame:
        """Returns frame ``index`` as a ``Frame`` view (no copy of the columns)."""
        rows = self.frame_slice(index)
        return Frame(
            index=index,
            center=int(self.grid.center(index)),
            time_s=float(self.grid.time_seconds(index)),
            fs=self.grid.fs,
            omega=self.omega[rows],
            log_amplitude=self.log_amplitude[rows],
            phase=self.phase[rows],
            track_id=self.track_id[rows],
            alpha=None if self.alpha is None else self.alpha[rows],
        )

    def frames(self) -> Iterator[Frame]:
        """Iterates over all M frames, including empty ones."""
        for index in range(self.num_frames):
            yield self.get_frame(index)

    def track(self, track_id: int) -> np.ndarray:
        """Row indices of one partial, in time order."""
        return np.flatnonzero(self.track_id == track_id)

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Writes the partials to a file; the format follows the extension.

        Args:
            path: Destination (.parquet, .arrows, .sdif or .bin).
            **kwargs: Passed to the format's writer.
        """
        from fast_partial_tracking.io import write

        write(self, path, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Partials:
        """Reads partials from a file; the format follows the extension."""
        from fast_partial_tracking.io import read

        return read(path)


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0)
