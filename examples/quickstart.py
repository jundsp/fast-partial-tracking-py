"""Quickstart: track, resynthesize, measure and plot in a few lines.

Run:
    python examples/quickstart.py [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.quickstart")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "quickstart"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    x, fs = fpt.data.demo_sound()
    partials = fpt.track(x, fs)  # default settings: 46 ms Blackman-Harris window, Q = 2
    y = fpt.synthesize(partials)

    log.info(partials)
    log.info(f"R-SNR = {fpt.rsnr(x, y):.2f} dB")
    log.info(f"loudest partial: {max(partials.tracks(), key=lambda t: t.amplitude_db.max())}")

    partials.save(args.out / "data" / "partials.parquet")
    fpt.plot.overview(partials, x, y, max_frequency_hz=5000, out_path=args.out / "figures" / "overview.png")
    log.info(f"outputs -> {args.out}")


if __name__ == "__main__":
    main()
