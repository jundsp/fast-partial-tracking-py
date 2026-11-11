"""Common interface for writing partials frame by frame.

Every format is written incrementally: frames are appended as soon as the tracker
finalizes them, so analyses of arbitrarily long or live input never need to be held in
memory. Settings that are only known at the end of a stream (signal length, frame
count, number of partials) are supplied with ``set_summary`` before closing.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Frame, Partials

FORMAT_NAME = "fast-partial-tracking"
FORMAT_VERSION = 1


class PartialsWriter(ABC):
    """Base class of all partial writers; use as a context manager.

    Args:
        path: Destination file.
        grid: Frame geometry (the signal length may still be unknown).
        metadata: Settings/provenance to store, if the format supports it.
    """

    def __init__(self, path: str | Path, grid: FrameGrid, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.grid = grid
        self.metadata = dict(metadata or {})
        self.num_tracks: int | None = None
        self._closed = False

    @abstractmethod
    def write_frame(self, frame: Frame) -> None:
        """Appends one frame (frames must arrive in increasing order)."""

    @abstractmethod
    def _finalize(self) -> None:
        """Flushes buffered data, writes trailers and releases the file."""

    def set_summary(self, grid: FrameGrid, num_tracks: int, metadata: dict[str, Any] | None = None) -> None:
        """Records end-of-stream information before closing.

        Args:
            grid: The completed frame geometry (with signal length and frame count).
            num_tracks: Number of partials found.
            metadata: Additional settings/provenance to merge in.
        """
        self.grid = grid
        self.num_tracks = num_tracks
        self.metadata.update(metadata or {})

    def write_partials(self, partials: Partials) -> None:
        """Writes a complete result (all frames plus summary)."""
        for frame in partials.frames():
            self.write_frame(frame)
        self.set_summary(partials.grid, partials.num_tracks, partials.metadata)

    def close(self) -> None:
        """Finishes the file. Safe to call more than once."""
        if not self._closed:
            self._finalize()
            self._closed = True

    def __enter__(self) -> PartialsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def header(self) -> dict[str, Any]:
        """Self-describing metadata block stored by the modern formats."""
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "grid": self.grid.to_dict(),
            "num_tracks": self.num_tracks,
            **self.metadata,
        }


def dumps(value: dict[str, Any]) -> bytes:
    """Serializes metadata as compact UTF-8 JSON."""
    return json.dumps(value, separators=(",", ":"), default=_json_default).encode()


def loads(raw: bytes | str | None) -> dict[str, Any]:
    """Parses metadata written by ``dumps``."""
    return json.loads(raw) if raw else {}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
