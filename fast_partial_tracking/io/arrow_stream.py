"""Arrow IPC streaming format (``.arrows``) for live analysis.

Unlike Parquet, an Arrow stream can be read while it is still being written (for
example through a pipe or a growing file): every frame block is a self-contained record
batch. The end-of-stream summary travels as the custom metadata of a final empty batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import numpy as np
import pyarrow as pa

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io.base import PartialsWriter, dumps, loads
from fast_partial_tracking.io.table import (
    HEADER_KEY,
    SUMMARY_KEY,
    batch_to_frames,
    frames_to_batch,
    schema,
    table_to_partials,
)
from fast_partial_tracking.partials import Frame, Partials


class ArrowStreamWriter(PartialsWriter):
    """Writes partials as an Arrow IPC stream.

    Args:
        path: Destination file path, or a writable binary file object (e.g. a pipe).
        grid: Frame geometry.
        metadata: Settings/provenance stored in the stream schema.
        order: Q of the model; when given, the coefficient columns are written.
        frames_per_batch: Frames per record batch (1 gives the lowest latency).
    """

    def __init__(
        self,
        path: str | Path | IO[bytes],
        grid: FrameGrid,
        metadata: dict[str, Any] | None = None,
        order: int | None = None,
        frames_per_batch: int = 1,
    ) -> None:
        is_path = isinstance(path, (str, Path))
        super().__init__(path if is_path else "<stream>", grid, metadata)
        if is_path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sink: Any = pa.OSFile(str(self.path), "wb")
        else:
            self._sink = path
        self.schema = schema(order, {HEADER_KEY: dumps(self.header())})
        self.frames_per_batch = frames_per_batch
        self._writer = pa.ipc.new_stream(self._sink, self.schema)
        self._owns_sink = is_path
        self._pending: list[Frame] = []

    def write_frame(self, frame: Frame) -> None:
        self._pending.append(frame)
        if len(self._pending) >= self.frames_per_batch:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            self._writer.write_batch(frames_to_batch(self._pending, self.grid.fs, self.schema))
            self._pending = []

    def _finalize(self) -> None:
        self._flush()
        trailer = frames_to_batch([], self.grid.fs, self.schema)
        self._writer.write_batch(trailer, custom_metadata={SUMMARY_KEY: dumps(self.header())})
        self._writer.close()
        if self._owns_sink:
            self._sink.close()


def iter_arrow_stream(source: str | Path | IO[bytes]) -> Iterator[Frame]:
    """Yields frames from an Arrow stream as they are read, filling in empty frames.

    Frames come out with consecutive indices starting at 0, so the output can be fed
    straight into a ``Synthesizer``.

    Args:
        source: Path or readable binary file object.

    Yields:
        Frames in order.
    """
    reader = pa.ipc.open_stream(str(source) if isinstance(source, (str, Path)) else source)
    grid = FrameGrid(**loads((reader.schema.metadata or {}).get(HEADER_KEY))["grid"])
    next_index = 0
    while True:
        try:
            batch, custom = reader.read_next_batch_with_custom_metadata()
        except StopIteration:
            return
        for frame in batch_to_frames(batch, grid):
            while next_index < frame.index:
                yield _empty(next_index, grid)
                next_index += 1
            yield frame
            next_index += 1
        if custom is not None and SUMMARY_KEY in custom:
            grid = FrameGrid(**loads(custom[SUMMARY_KEY])["grid"])
            while grid.num_frames is not None and next_index < grid.num_frames:
                yield _empty(next_index, grid)
                next_index += 1


def read_arrow_stream(source: str | Path | IO[bytes]) -> Partials:
    """Reads a complete Arrow stream written by ``ArrowStreamWriter``.

    Args:
        source: Path or readable binary file object.

    Returns:
        The stored partials.
    """
    reader = pa.ipc.open_stream(str(source) if isinstance(source, (str, Path)) else source)
    header = loads((reader.schema.metadata or {}).get(HEADER_KEY))
    batches = []
    while True:
        try:
            batch, custom = reader.read_next_batch_with_custom_metadata()
        except StopIteration:
            break
        batches.append(batch)
        if custom is not None and SUMMARY_KEY in custom:
            header = loads(custom[SUMMARY_KEY])
    table = pa.Table.from_batches(batches, schema=reader.schema)
    return table_to_partials(table, header)


def _empty(index: int, grid: FrameGrid) -> Frame:
    return Frame(
        index=index,
        center=int(grid.center(index)),
        time_s=float(grid.time_seconds(index)),
        omega=np.zeros(0),
        log_amp=np.zeros(0),
        phase=np.zeros(0),
        track_id=np.zeros(0, np.int64),
        alpha=None,
    )
