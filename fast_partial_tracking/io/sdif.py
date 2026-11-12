"""Binary SDIF (Sound Description Interchange Format) export and import.

Writes the standard ``1TRC`` (sinusoidal tracks) frame type directly, so files open in
SDIF tools such as IRCAM AudioSculpt or OpenMusic without the external ``tosdif``
converter that the MATLAB version needed. SDIF is block based (one frame per analysis
frame), so it is written incrementally like the other formats.

Layout (SDIF 3, all values big-endian):

* File header: ``"SDIF"``, size 8, specification version 3, types version 1.
* Frame header (24 bytes): signature ``"1TRC"``, size of the rest of the frame,
  time (float64, seconds), stream ID (int32), matrix count (int32).
* Matrix header (16 bytes): signature ``"1TRC"``, data type (0x0004 float32 or
  0x0008 float64), rows, columns; then row-major data padded to a multiple of 8 bytes.
* ``1TRC`` columns: Index (partial label), Frequency (Hz), Amplitude (linear), Phase.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io.base import PartialsWriter
from fast_partial_tracking.io.points import from_points
from fast_partial_tracking.partials import Frame, Partials

_FILE_HEADER = struct.Struct(">4sIII")
_FRAME_HEADER = struct.Struct(">4sIdiI")
_MATRIX_HEADER = struct.Struct(">4sIII")
_DTYPES = {0x0004: np.dtype(">f4"), 0x0008: np.dtype(">f8")}
TRACK_SIGNATURE = b"1TRC"


class SdifWriter(PartialsWriter):
    """Writes partials as a binary SDIF file of ``1TRC`` frames.

    Frames whose time is negative (they lie in the analysis zero padding before the
    signal starts) are skipped because SDIF times must be non-negative.

    Args:
        path: Destination ``.sdif`` file.
        grid: Frame geometry.
        metadata: Unused (SDIF stores no settings); accepted for a uniform interface.
        precision: ``"float64"`` (default) or ``"float32"`` matrix data.
        stream_id: SDIF stream ID of the frames.
    """

    def __init__(
        self,
        path: str | Path,
        grid: FrameGrid,
        metadata: dict[str, Any] | None = None,
        precision: str = "float64",
        stream_id: int = 0,
    ) -> None:
        super().__init__(path, grid, metadata)
        codes = {"float32": 0x0004, "float64": 0x0008}
        if precision not in codes:
            raise ValueError(f"precision must be one of {sorted(codes)}, got {precision!r}")
        self.data_type = codes[precision]
        self.stream_id = stream_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: BinaryIO = self.path.open("wb")
        self._file.write(_FILE_HEADER.pack(b"SDIF", 8, 3, 1))

    def write_frame(self, frame: Frame) -> None:
        if frame.time_s < 0:
            return
        matrix = np.column_stack(
            [frame.track_id.astype(float), frame.frequency_hz, frame.amplitude, frame.phase]
        ).astype(_DTYPES[self.data_type])
        data = matrix.tobytes()
        padding = (-len(data)) % 8
        size = _FRAME_HEADER.size - 8 + _MATRIX_HEADER.size + len(data) + padding
        self._file.write(_FRAME_HEADER.pack(TRACK_SIGNATURE, size, frame.time_s, self.stream_id, 1))
        self._file.write(_MATRIX_HEADER.pack(TRACK_SIGNATURE, self.data_type, *matrix.shape))
        self._file.write(data + b"\0" * padding)

    def _finalize(self) -> None:
        self._file.close()


@dataclass(frozen=True)
class SdifMatrix:
    """One SDIF matrix.

    Attributes:
        signature: Matrix type, e.g. ``"1TRC"``.
        data: Matrix values (rows x columns).
    """

    signature: str
    data: np.ndarray


@dataclass(frozen=True)
class SdifFrame:
    """One SDIF frame.

    Attributes:
        signature: Frame type, e.g. ``"1TRC"``.
        time: Time in seconds.
        stream_id: Stream ID.
        matrices: Matrices of the frame.
    """

    signature: str
    time: float
    stream_id: int
    matrices: list[SdifMatrix]


def iter_sdif(path: str | Path) -> Iterator[SdifFrame]:
    """Reads an SDIF file frame by frame (numeric matrices; others are skipped).

    Args:
        path: Source ``.sdif`` file.

    Yields:
        The frames in file order.
    """
    with Path(path).open("rb") as file:
        signature, size, *_ = _FILE_HEADER.unpack(file.read(_FILE_HEADER.size))
        if signature != b"SDIF":
            raise ValueError(f"{path} is not an SDIF file")
        file.seek(8 + size)
        while header := file.read(_FRAME_HEADER.size):
            if len(header) < _FRAME_HEADER.size:
                raise ValueError(f"{path}: truncated frame header")
            frame_sig, frame_size, time, stream_id, count = _FRAME_HEADER.unpack(header)
            end = file.tell() + frame_size - (_FRAME_HEADER.size - 8)
            matrices = []
            for _ in range(count):
                matrix_sig, data_type, rows, cols = _MATRIX_HEADER.unpack(file.read(_MATRIX_HEADER.size))
                width = data_type & 0xFF
                nbytes = rows * cols * width
                raw = file.read(nbytes + (-nbytes) % 8)[:nbytes]
                if data_type in _DTYPES:
                    data = np.frombuffer(raw, dtype=_DTYPES[data_type]).reshape(rows, cols).astype(float)
                    matrices.append(SdifMatrix(matrix_sig.decode(), data))
            file.seek(end)
            yield SdifFrame(frame_sig.decode(), time, stream_id, matrices)


def read_sdif_tracks(path: str | Path) -> list[tuple[float, np.ndarray]]:
    """Reads the ``1TRC`` matrices of an SDIF file.

    Args:
        path: Source ``.sdif`` file.

    Returns:
        (time in seconds, matrix with columns index/frequency/amplitude/phase) per frame.
    """
    return [
        (frame.time, matrix.data)
        for frame in iter_sdif(path)
        for matrix in frame.matrices
        if matrix.signature == TRACK_SIGNATURE.decode()
    ]


def read_sdif(path: str | Path, fs: float | None = None) -> Partials:
    """Reads the sinusoidal tracks of an SDIF file (``1TRC`` or ``1HRM`` matrices) as partials.

    Works for files from this package and from other SDIF tools. SDIF stores no sampling
    rate or analysis settings, so the frame grid is inferred from the frame times (see
    ``fast_partial_tracking.io.points.from_points``).

    Args:
        path: Source ``.sdif`` file.
        fs: Sampling rate in Hz (default 44.1 kHz).

    Returns:
        The partials. Stored phases are kept; if they are all zero the result is marked
        for phase-integrating synthesis.
    """
    signatures = {TRACK_SIGNATURE.decode(), "1HRM"}
    blocks = [(frame.time, matrix.data) for frame in iter_sdif(path) for matrix in frame.matrices
              if matrix.signature in signatures and matrix.data.shape[1] >= 3]
    if not blocks:
        raise ValueError(f"{path} contains no 1TRC or 1HRM tracks")
    time = np.concatenate([np.full(data.shape[0], t) for t, data in blocks])
    data = np.concatenate([np.pad(d[:, :4], ((0, 0), (0, max(0, 4 - d.shape[1])))) for _, d in blocks])
    return from_points(time, data[:, 0], data[:, 1], data[:, 2], phase=data[:, 3], fs=fs,
                       metadata={"source_format": "sdif"})
