"""The binary format of the MATLAB implementation (``demo_partials.bin``).

Kept for interoperability and for the equivalence tests. Little-endian layout written
by ``jun_track_partials.m`` / ``jun_write_partials.m``:

* Header: num_tracks (uint32), L (uint32), fs (float64), M (uint32),
  padding front (uint32), padding back (uint32).
* Per frame: frame number (uint32, 1-based), time (uint32, 1-based padded sample
  index of the frame origin), rows (uint32), then per row omega (float64,
  radians/sample), log-amplitude (float64), phase (float64), track (uint32, 1-based).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io.base import PartialsWriter
from fast_partial_tracking.partials import Frame, Partials

_HEADER = np.dtype([("num_tracks", "<u4"), ("L", "<u4"), ("fs", "<f8"), ("M", "<u4"), ("pad1", "<u4"), ("pad2", "<u4")])
_FRAME = np.dtype([("frame", "<u4"), ("time", "<u4"), ("rows", "<u4")])
_ROW = np.dtype([("omega", "<f8"), ("log_amplitude", "<f8"), ("phase", "<f8"), ("track", "<u4")])


class MatlabBinWriter(PartialsWriter):
    """Writes the MATLAB ``.bin`` format.

    The header holds the signal length and partial count, which are only known at the
    end; like MATLAB, the header is rewritten when the file is closed.

    Args:
        path: Destination ``.bin`` file.
        grid: Frame geometry.
        metadata: Unused; accepted for a uniform interface.
    """

    def __init__(self, path: str | Path, grid: FrameGrid, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(path, grid, metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: BinaryIO = self.path.open("wb")
        self._file.write(self._header_bytes())

    def _header_bytes(self) -> bytes:
        header = np.zeros(1, _HEADER)
        header["num_tracks"] = self.num_tracks or 0
        header["L"] = self.grid.signal_length or 0
        header["fs"] = self.grid.fs
        header["M"] = self.grid.num_frames or 0
        header["pad1"] = self.grid.pad_front
        header["pad2"] = self.grid.pad_back or 0
        return header.tobytes()

    def write_frame(self, frame: Frame) -> None:
        head = np.zeros(1, _FRAME)
        head["frame"] = frame.index + 1
        head["time"] = frame.center + 1
        head["rows"] = len(frame)
        rows = np.zeros(len(frame), _ROW)
        rows["omega"] = frame.omega
        rows["log_amplitude"] = frame.log_amplitude
        rows["phase"] = frame.phase
        rows["track"] = frame.track_id + 1
        self._file.write(head.tobytes() + rows.tobytes())

    def _finalize(self) -> None:
        self._file.seek(0)
        self._file.write(self._header_bytes())
        self._file.close()


def read_matlab_bin(path: str | Path) -> Partials:
    """Reads a ``.bin`` file written by MATLAB ``jun_track_partials`` or ``MatlabBinWriter``.

    Args:
        path: Source ``.bin`` file.

    Returns:
        The partials, with 0-based frame and track indices. The window length and DFT
        size are not stored in this format and are inferred where possible.
    """
    raw = Path(path).read_bytes()
    header = np.frombuffer(raw, _HEADER, count=1)[0]
    offset = _HEADER.itemsize
    frames, centers, blocks = [], [], []
    while offset < len(raw):
        head = np.frombuffer(raw, _FRAME, count=1, offset=offset)[0]
        offset += _FRAME.itemsize
        rows = np.frombuffer(raw, _ROW, count=int(head["rows"]), offset=offset)
        offset += rows.nbytes
        frames.append(int(head["frame"]) - 1)
        centers.append(int(head["time"]) - 1)
        blocks.append(rows)
    hop = int(np.median(np.diff(centers))) if len(centers) > 1 else 1
    origin = centers[0] if centers else 0
    pad_front = int(header["pad1"])
    grid = FrameGrid(
        fs=float(header["fs"]),
        window_length=0,
        hop=hop,
        n_dft=pad_front + hop,
        origin=origin,
        pad_front=pad_front,
        signal_length=int(header["L"]),
        num_frames=int(header["M"]),
        pad_back=int(header["pad2"]),
    )
    data = np.concatenate(blocks) if blocks else np.zeros(0, _ROW)
    counts = [b.size for b in blocks]
    return Partials(
        grid=grid,
        frame=np.repeat(frames, counts),
        track_id=data["track"].astype(np.int64) - 1,
        omega=data["omega"].astype(float),
        log_amplitude=data["log_amplitude"].astype(float),
        phase=data["phase"].astype(float),
        num_tracks=int(header["num_tracks"]),
    )
