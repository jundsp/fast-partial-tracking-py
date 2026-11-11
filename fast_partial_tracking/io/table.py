"""Arrow representation of partials, shared by the Parquet and Arrow-stream formats.

One row per (frame, partial), in physical units:

=================  ==============================  =========================================
column             type                            meaning
=================  ==============================  =========================================
frame              int32                           0-based analysis frame k
time_s             float64                         frame origin time, seconds from input start
track_id           int32                           0-based partial label
frequency_hz       float64                         instantaneous frequency
amplitude          float64                         real sinusoid amplitude, 2*exp(Re alpha_0)
phase              float64                         instantaneous phase in radians (unwrapped)
alpha_re/alpha_im  fixed_size_list<float64>[Q+1]   optional model coefficients (eq. 4)
=================  ==============================  =========================================

Settings and the frame grid are stored as JSON in the schema metadata so files are
self-describing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Frame, Partials

HEADER_KEY = b"fast_partial_tracking"
SUMMARY_KEY = b"fast_partial_tracking.summary"


def schema(order: int | None, metadata: dict[bytes, bytes] | None = None) -> pa.Schema:
    """Arrow schema of the partials table.

    Args:
        order: Q to include model coefficient columns, or None to omit them.
        metadata: Schema-level key/value metadata.
    """
    fields = [
        pa.field("frame", pa.int32(), nullable=False),
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("track_id", pa.int32(), nullable=False),
        pa.field("frequency_hz", pa.float64(), nullable=False),
        pa.field("amplitude", pa.float64(), nullable=False),
        pa.field("phase", pa.float64(), nullable=False),
    ]
    if order is not None:
        coeffs = pa.list_(pa.float64(), order + 1)
        fields += [pa.field("alpha_re", coeffs, nullable=False), pa.field("alpha_im", coeffs, nullable=False)]
    return pa.schema(fields, metadata=metadata)


def frames_to_batch(frames: list[Frame], fs: float, table_schema: pa.Schema) -> pa.RecordBatch:
    """Converts frames into one record batch.

    Args:
        frames: Frames to convert.
        fs: Sampling rate in Hz.
        table_schema: Target schema (decides whether coefficients are included).
    """
    counts = [len(f) for f in frames]
    columns: dict[str, Any] = {
        "frame": np.repeat([f.index for f in frames], counts).astype(np.int32),
        "time_s": np.repeat([f.time_s for f in frames], counts).astype(float),
        "track_id": _cat([f.track_id for f in frames]).astype(np.int32),
        "frequency_hz": _cat([f.omega for f in frames]) * fs / (2.0 * np.pi),
        "amplitude": 2.0 * np.exp(_cat([f.log_amp for f in frames])),
        "phase": _cat([f.phase for f in frames]),
    }
    if "alpha_re" in table_schema.names:
        width = table_schema.field("alpha_re").type.list_size
        alpha = np.concatenate([_alpha(f, width) for f in frames]) if frames else np.zeros((0, width), complex)
        columns["alpha_re"] = _fixed_list(alpha.real, width)
        columns["alpha_im"] = _fixed_list(alpha.imag, width)
    arrays = [pa.array(columns[name], type=table_schema.field(name).type) for name in table_schema.names]
    return pa.RecordBatch.from_arrays(arrays, schema=table_schema)


def table_to_partials(table: pa.Table, header: dict[str, Any]) -> Partials:
    """Rebuilds ``Partials`` from a table and its metadata header.

    Args:
        table: Table with the schema above.
        header: Parsed metadata (grid, num_tracks, settings).
    """
    grid_info = dict(header.get("grid") or {})
    grid = FrameGrid(**grid_info)
    frame = table.column("frame").to_numpy().astype(np.int64)
    if grid.num_frames is None:
        signal_length = int(frame.max() + 1) * grid.hop if frame.size else 0
        grid = grid.with_signal_length(signal_length)
    fs = grid.fs
    alpha = None
    if "alpha_re" in table.column_names:
        re = _list_to_matrix(table.column("alpha_re"))
        im = _list_to_matrix(table.column("alpha_im"))
        alpha = re + 1j * im
    num_tracks = header.get("num_tracks")
    track_id = table.column("track_id").to_numpy().astype(np.int64)
    if num_tracks is None:
        num_tracks = int(track_id.max()) + 1 if track_id.size else 0
    metadata = {k: v for k, v in header.items() if k not in {"grid", "num_tracks", "format", "format_version"}}
    return Partials(
        grid=grid,
        frame=frame,
        track_id=track_id,
        omega=table.column("frequency_hz").to_numpy() * 2.0 * np.pi / fs,
        log_amp=np.log(table.column("amplitude").to_numpy() / 2.0),
        phase=table.column("phase").to_numpy(),
        alpha=alpha,
        num_tracks=int(num_tracks),
        metadata=metadata,
    )


def batch_to_frames(batch: pa.RecordBatch, grid: FrameGrid) -> list[Frame]:
    """Splits a record batch back into ``Frame`` objects (non-empty frames only)."""
    table = pa.Table.from_batches([batch])
    if table.num_rows == 0:
        return []
    frame = table.column("frame").to_numpy().astype(np.int64)
    omega = table.column("frequency_hz").to_numpy() * 2.0 * np.pi / grid.fs
    log_amp = np.log(table.column("amplitude").to_numpy() / 2.0)
    phase = table.column("phase").to_numpy()
    track_id = table.column("track_id").to_numpy().astype(np.int64)
    alpha = None
    if "alpha_re" in table.column_names:
        alpha = _list_to_matrix(table.column("alpha_re")) + 1j * _list_to_matrix(table.column("alpha_im"))
    starts = np.flatnonzero(np.r_[True, np.diff(frame) != 0])
    ends = np.r_[starts[1:], frame.size]
    return [
        Frame(
            index=int(frame[s]),
            center=int(grid.center(int(frame[s]))),
            time_s=float(grid.time_seconds(int(frame[s]))),
            omega=omega[s:e],
            log_amp=log_amp[s:e],
            phase=phase[s:e],
            track_id=track_id[s:e],
            alpha=None if alpha is None else alpha[s:e],
        )
        for s, e in zip(starts, ends)
    ]


def _alpha(frame: Frame, width: int) -> np.ndarray:
    if frame.alpha is None:
        raise ValueError("frame has no model coefficients; open the writer with include_alpha=False")
    if frame.alpha.shape[1] != width:
        raise ValueError(f"expected {width} coefficients per partial, got {frame.alpha.shape[1]}")
    return frame.alpha


def _fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.ascontiguousarray(values, dtype=float).ravel(), type=pa.float64())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _list_to_matrix(column: pa.ChunkedArray) -> np.ndarray:
    width = column.type.list_size
    flat = pa.concat_arrays([chunk.flatten() for chunk in column.chunks]) if column.num_chunks else pa.array([], pa.float64())
    return flat.to_numpy(zero_copy_only=False).reshape(-1, width)


def _cat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0)
