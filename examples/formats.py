"""File formats: Parquet, Arrow stream and SDIF round trips, and querying with polars.

Run:
    python examples/formats.py [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fast_partial_tracking as fpt
from fast_partial_tracking.io import iter_arrow_stream, read_sdif_tracks

log = logging.getLogger("fast_partial_tracking.examples.formats")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "formats"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    x, fs = fpt.data.demo_sound()
    partials = fpt.track(x, fs)
    data_dir = args.out / "data"
    for suffix in [".parquet", ".arrows", ".sdif", ".bin"]:
        path = data_dir / f"demo{suffix}"
        partials.save(path)
        log.info(f"{suffix:9s} {path.stat().st_size / 1024:7.1f} KiB  {path}")

    loaded = fpt.Partials.load(data_dir / "demo.parquet")
    log.info(f"Parquet round trip: {loaded}")
    first_frames = [f for f, _ in zip(iter_arrow_stream(data_dir / "demo.arrows"), range(3))]
    log.info(f"Arrow stream read frame by frame: frames {[f.index for f in first_frames]}")
    tracks = read_sdif_tracks(data_dir / "demo.sdif")
    log.info(f"SDIF: {len(tracks)} 1TRC frames (index, frequency, amplitude, phase)")

    try:
        import polars as pl
    except ImportError:
        log.info("install polars to run the query example: pip install polars")
        return
    table = (
        pl.read_parquet(data_dir / "demo.parquet")
        .group_by("track_id")
        .agg(pl.len().alias("frames"), pl.col("frequency_hz").mean().alias("mean_hz"), pl.col("amplitude").max())
        .sort("amplitude", descending=True)
        .head(5)
    )
    log.info(f"five loudest partials (polars on the Parquet file):\n{table}")


if __name__ == "__main__":
    main()
