"""Reading and writing partials.

========== ===================== ==============================================
extension  format                use
========== ===================== ==============================================
.parquet   Apache Parquet        recommended storage; pandas/polars/DuckDB/MATLAB
.arrows    Arrow IPC stream      live pipes; readable while being written
.sdif      SDIF 1TRC (binary)    compatibility with SDIF tools
.bin       MATLAB demo format    compatibility with the MATLAB implementation
========== ===================== ==============================================

All writers are incremental (frame by frame) and share the ``PartialsWriter`` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io.arrow_stream import ArrowStreamWriter, iter_arrow_stream, read_arrow_stream
from fast_partial_tracking.io.base import PartialsWriter
from fast_partial_tracking.compat.matlab_bin import MatlabBinWriter, read_matlab_bin
from fast_partial_tracking.io.parquet import ParquetWriter, read_parquet
from fast_partial_tracking.io.points import from_points
from fast_partial_tracking.io.sdif import SdifWriter, iter_sdif, read_sdif, read_sdif_tracks
from fast_partial_tracking.partials import Partials

__all__ = [
    "ArrowStreamWriter",
    "MatlabBinWriter",
    "ParquetWriter",
    "PartialsWriter",
    "SdifWriter",
    "iter_arrow_stream",
    "from_points",
    "iter_sdif",
    "open_writer",
    "read",
    "read_arrow_stream",
    "read_matlab_bin",
    "read_parquet",
    "read_sdif",
    "read_sdif_tracks",
    "write",
]

_WRITERS: dict[str, type[PartialsWriter]] = {
    ".parquet": ParquetWriter,
    ".arrows": ArrowStreamWriter,
    ".arrow": ArrowStreamWriter,
    ".sdif": SdifWriter,
    ".bin": MatlabBinWriter,
}
_ORDER_AWARE = (ParquetWriter, ArrowStreamWriter)


def open_writer(
    path: str | Path,
    grid: FrameGrid,
    metadata: dict[str, Any] | None = None,
    order: int | None = None,
    **kwargs: Any,
) -> PartialsWriter:
    """Opens an incremental writer chosen by file extension.

    Args:
        path: Destination file.
        grid: Frame geometry (e.g. ``PartialTracker.grid``).
        metadata: Settings/provenance to store where supported.
        order: Q; when given, Parquet/Arrow files also store the model coefficients.
        **kwargs: Format-specific options (see each writer class).

    Returns:
        A writer to use as a context manager.
    """
    writer_cls = _writer_class(path)
    if issubclass(writer_cls, _ORDER_AWARE):
        kwargs["order"] = order
    return writer_cls(path, grid, metadata, **kwargs)


def write(partials: Partials, path: str | Path, include_alpha: bool = True, **kwargs: Any) -> None:
    """Writes a complete result; the format follows the file extension.

    Args:
        partials: The result to write.
        path: Destination file.
        include_alpha: Store the model coefficients (Parquet/Arrow) when available.
        **kwargs: Format-specific options.
    """
    order = partials.alpha.shape[1] - 1 if include_alpha and partials.alpha is not None else None
    with open_writer(path, partials.grid, partials.metadata, order=order, **kwargs) as writer:
        writer.write_partials(partials)


def read(path: str | Path, fs: float | None = None) -> Partials:
    """Reads partials from a .parquet, .arrows/.arrow, .bin or .sdif file.

    Args:
        path: Source file.
        fs: Sampling rate for SDIF files, which do not store one (default 44.1 kHz).
            Ignored for the other formats.

    Returns:
        The stored partials.
    """
    suffix = Path(path).suffix.lower()
    readers = {
        ".parquet": read_parquet,
        ".arrows": read_arrow_stream,
        ".arrow": read_arrow_stream,
        ".bin": read_matlab_bin,
        ".sdif": lambda p: read_sdif(p, fs=fs),
    }
    if suffix not in readers:
        raise ValueError(f"cannot read {suffix!r} files into Partials; supported: {sorted(readers)}")
    return readers[suffix](path)


def _writer_class(path: str | Path) -> type[PartialsWriter]:
    suffix = Path(path).suffix.lower()
    if suffix not in _WRITERS:
        raise ValueError(f"unsupported extension {suffix!r}; supported: {sorted(_WRITERS)}")
    return _WRITERS[suffix]
