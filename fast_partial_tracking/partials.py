"""Containers for tracked partials.

``Frame`` holds the partials active in one analysis frame (what the streaming tracker
emits). ``Partials`` holds a whole analysis as flat columns, one row per (frame, partial),
plus an index of where each frame starts, so per-frame views are slices with no copy.
This "long" layout is also the on-disk schema (see ``fast_partial_tracking.io``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class Track:
    """One partial: its parameters in every frame it spans, in time order.

    Attributes:
        id: Partial label.
        frame: Frame index of each point.
        time_s: Time of each point in seconds.
        fs: Sampling rate in Hz.
        omega: Angular frequency (radians/sample) of each point.
        log_amplitude: Natural log of the complex amplitude of each point.
        phase: Phase (radians) of each point.
    """

    id: int
    frame: np.ndarray
    time_s: np.ndarray
    fs: float
    omega: np.ndarray
    log_amplitude: np.ndarray
    phase: np.ndarray

    def __len__(self) -> int:
        return int(self.frame.size)

    def __repr__(self) -> str:
        return (
            f"Track(id={self.id}, {len(self)} frames, {self.start_s:.3f}-{self.end_s:.3f} s, "
            f"{self.mean_frequency_hz:.1f} Hz, peak {self.amplitude_db.max():.1f} dB)"
        )

    @property
    def frequency_hz(self) -> np.ndarray:
        """Frequency of each point in Hz."""
        return self.omega * self.fs / (2.0 * np.pi)

    @property
    def amplitude(self) -> np.ndarray:
        """Linear amplitude of each point."""
        return 2.0 * np.exp(self.log_amplitude)

    @property
    def amplitude_db(self) -> np.ndarray:
        """Amplitude of each point in dB."""
        return 20.0 * np.log10(self.amplitude)

    @property
    def start_s(self) -> float:
        """Time of the first point."""
        return float(self.time_s[0])

    @property
    def end_s(self) -> float:
        """Time of the last point."""
        return float(self.time_s[-1])

    @property
    def duration_s(self) -> float:
        """Time between the first and last point."""
        return self.end_s - self.start_s

    @property
    def mean_frequency_hz(self) -> float:
        """Amplitude-weighted mean frequency."""
        return float(np.average(self.frequency_hz, weights=self.amplitude))


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

    def __repr__(self) -> str:
        duration = (self.grid.signal_length or 0) / self.grid.fs
        return (
            f"Partials({self.num_tracks} partials, {self.num_frames} frames, {len(self)} rows, "
            f"{duration:.2f} s at {self.grid.fs:g} Hz)"
        )

    def tracks(self) -> Iterator[Track]:
        """Iterates over all partials in label order."""
        order = np.argsort(self.track_id, kind="stable")
        ids = self.track_id[order]
        bounds = np.flatnonzero(np.r_[True, np.diff(ids) != 0, True])
        for start, end in zip(bounds[:-1], bounds[1:]):
            yield self._track(order[start:end])

    def track(self, track_id: int) -> Track:
        """One partial by label."""
        rows = np.flatnonzero(self.track_id == track_id)
        if rows.size == 0:
            raise KeyError(f"no partial with id {track_id}")
        return self._track(rows)

    def _track(self, rows: np.ndarray) -> Track:
        return Track(
            id=int(self.track_id[rows[0]]),
            frame=self.frame[rows],
            time_s=self.time_s[rows],
            fs=self.grid.fs,
            omega=self.omega[rows],
            log_amplitude=self.log_amplitude[rows],
            phase=self.phase[rows],
        )

    def track_lengths(self) -> np.ndarray:
        """Number of frames spanned by each label 0..num_tracks-1."""
        return np.bincount(self.track_id, minlength=self.num_tracks)

    def prune(self, min_frames: int = 3) -> Partials:
        """Removes short partials, which are mostly spurious.

        Args:
            min_frames: Minimum number of frames a partial must span to be kept.

        Returns:
            A new result with the remaining partials relabelled 0..n-1.
        """
        keep = self.track_lengths() >= min_frames
        return self.subset(keep[self.track_id])

    def select(
        self,
        time_s: tuple[float, float] | None = None,
        frequency_hz: tuple[float, float] | None = None,
        min_amplitude_db: float | None = None,
        track_ids: Iterable[int] | None = None,
    ) -> Partials:
        """Keeps the points inside a time/frequency region (point-wise, not per partial).

        Args:
            time_s: (start, end) time range in seconds, inclusive.
            frequency_hz: (low, high) frequency range in Hz, inclusive.
            min_amplitude_db: Drop points quieter than this.
            track_ids: Keep only these partial labels.

        Returns:
            A new result with the remaining partials relabelled 0..n-1.
        """
        mask = np.ones(len(self), dtype=bool)
        if time_s is not None:
            t = self.time_s
            mask &= (t >= time_s[0]) & (t <= time_s[1])
        if frequency_hz is not None:
            f = self.frequency_hz
            mask &= (f >= frequency_hz[0]) & (f <= frequency_hz[1])
        if min_amplitude_db is not None:
            mask &= self.amplitude_db >= min_amplitude_db
        if track_ids is not None:
            mask &= np.isin(self.track_id, np.fromiter(track_ids, dtype=np.int64))
        return self.subset(mask)

    def filter_tracks(self, predicate: Callable[[Track], bool]) -> Partials:
        """Keeps whole partials for which ``predicate(track)`` is true.

        Example:
            >>> low = partials.filter_tracks(lambda t: t.mean_frequency_hz < 1000)
        """
        keep = np.zeros(max(self.num_tracks, 1), dtype=bool)
        for track in self.tracks():
            keep[track.id] = bool(predicate(track))
        return self.subset(keep[self.track_id])

    def subset(self, rows: np.ndarray, relabel: bool = True) -> Partials:
        """A new result holding only the selected rows.

        Args:
            rows: Boolean mask (or index array, in frame order) of rows to keep.
            relabel: Renumber the remaining partials 0..n-1 (keeping their order).

        Returns:
            The subset, sharing grid and metadata.
        """
        track_id = self.track_id[rows]
        num_tracks = self.num_tracks
        if relabel:
            unique, track_id = np.unique(track_id, return_inverse=True)
            num_tracks = int(unique.size)
        return replace(
            self,
            frame=self.frame[rows],
            track_id=track_id.astype(np.int64),
            omega=self.omega[rows],
            log_amplitude=self.log_amplitude[rows],
            phase=self.phase[rows],
            alpha=None if self.alpha is None else self.alpha[rows],
            num_tracks=num_tracks,
            metadata=dict(self.metadata),
        )

    def stats(self) -> dict[str, float | int]:
        """Summary numbers: counts, partial lengths and frequency range."""
        lengths = self.track_lengths()
        lengths = lengths[lengths > 0]
        has_rows = len(self) > 0
        return {
            "partials": self.num_tracks,
            "frames": self.num_frames,
            "rows": len(self),
            "duration_s": (self.grid.signal_length or 0) / self.grid.fs,
            "median_partial_frames": float(np.median(lengths)) if lengths.size else 0.0,
            "max_partial_frames": int(lengths.max()) if lengths.size else 0,
            "min_frequency_hz": float(self.frequency_hz.min()) if has_rows else 0.0,
            "max_frequency_hz": float(self.frequency_hz.max()) if has_rows else 0.0,
            "max_amplitude_db": float(self.amplitude_db.max()) if has_rows else -np.inf,
        }

    def to_dict(self) -> dict[str, np.ndarray]:
        """Columns in physical units, one row per partial per frame."""
        return {
            "frame": self.frame,
            "time_s": self.time_s,
            "track_id": self.track_id,
            "frequency_hz": self.frequency_hz,
            "amplitude": self.amplitude,
            "amplitude_db": self.amplitude_db,
            "phase": self.phase,
        }

    def to_polars(self) -> Any:
        """The ``to_dict`` columns as a polars DataFrame (needs ``polars``)."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError("to_polars() needs polars: pip install 'fast-partial-tracking[dataframes]'") from exc
        return pl.DataFrame(self.to_dict())

    def to_pandas(self) -> Any:
        """The ``to_dict`` columns as a pandas DataFrame (needs ``pandas``)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_pandas() needs pandas: pip install 'fast-partial-tracking[dataframes]'") from exc
        return pd.DataFrame(self.to_dict())

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
