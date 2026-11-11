"""Parquet storage (the recommended at-rest format).

Frames are buffered and written as one Parquet row group per block of frames, so memory
stays bounded while the file is being produced. The result is a standard columnar,
zstd-compressed table readable by pandas, polars, DuckDB and MATLAB's ``parquetread``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io.base import PartialsWriter, dumps, loads
from fast_partial_tracking.io.table import HEADER_KEY, SUMMARY_KEY, frames_to_batch, schema, table_to_partials
from fast_partial_tracking.partials import Frame, Partials


class ParquetWriter(PartialsWriter):
    """Writes partials to a Parquet file incrementally.

    Args:
        path: Destination ``.parquet`` file.
        grid: Frame geometry.
        metadata: Settings/provenance stored in the file metadata.
        order: Q of the model; when given, the coefficient columns are written.
        frames_per_group: Frames buffered per row group.
        compression: Parquet compression codec.
    """

    def __init__(
        self,
        path: str | Path,
        grid: FrameGrid,
        metadata: dict[str, Any] | None = None,
        order: int | None = None,
        frames_per_group: int = 256,
        compression: str = "zstd",
    ) -> None:
        super().__init__(path, grid, metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema(order, {HEADER_KEY: dumps(self.header())})
        self.frames_per_group = frames_per_group
        self._writer = pq.ParquetWriter(self.path, self.schema, compression=compression)
        self._pending: list[Frame] = []

    def write_frame(self, frame: Frame) -> None:
        self._pending.append(frame)
        if len(self._pending) >= self.frames_per_group:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            batch = frames_to_batch(self._pending, self.grid.fs, self.schema)
            self._writer.write_table(pa.Table.from_batches([batch]))
            self._pending = []

    def _finalize(self) -> None:
        self._flush()
        self._writer.add_key_value_metadata({SUMMARY_KEY: dumps(self.header())})
        self._writer.close()


def read_parquet(path: str | Path) -> Partials:
    """Reads a Parquet file written by ``ParquetWriter``.

    Args:
        path: Source ``.parquet`` file.

    Returns:
        The stored partials.
    """
    table = pq.read_table(path)
    meta = pq.read_metadata(path).metadata or {}
    header = loads(meta.get(SUMMARY_KEY) or meta.get(HEADER_KEY))
    return table_to_partials(table, header)
