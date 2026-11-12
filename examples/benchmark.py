"""Times each processing stage on a long signal and writes a table.

Run:
    python examples/benchmark.py [--seconds 10] [--workers N] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import cProfile
import logging
import os
import pstats
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.benchmark")

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "demo_sound.wav"
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "benchmark"


def time_call(func, repeat: int = 3) -> float:
    """Best-of-``repeat`` wall time of ``func()`` in seconds."""
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile", action="store_true", help="also log the top functions by time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source, fs = sf.read(DEFAULT_AUDIO)
    x = np.tile(source, int(np.ceil(args.seconds * fs / source.size)))[: int(args.seconds * fs)]
    analysis = fpt.AnalysisConfig(window_length=2047, threshold_db=-50)

    rows = []
    for workers in sorted({1, args.workers}):
        partials = fpt.track(x, fs, analysis, workers=workers)
        t_track = time_call(lambda: fpt.track(x, fs, analysis, workers=workers))
        t_synth = time_call(lambda: fpt.synthesize(partials))
        rows.append(
            {
                "workers": workers,
                "audio_s": round(args.seconds, 2),
                "track_s": round(t_track, 3),
                "synth_s": round(t_synth, 3),
                "realtime_factor": round(args.seconds / (t_track + t_synth), 1),
                "frames": partials.num_frames,
                "rows": len(partials),
            }
        )
        log.info(
            f"workers={workers:2d}: track {t_track:.3f} s, synth {t_synth:.3f} s "
            f"({rows[-1]['realtime_factor']}x real time, {partials.num_frames} frames)"
        )

    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(fpt.track, x, fs, analysis, workers=1)
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:10]
        for (filename, line, name), values in top:
            log.info(f"  {values[2]:7.3f} s  {Path(filename).name}:{line} {name}")

    table = args.out / "tables" / "benchmark.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"table -> {table}")


if __name__ == "__main__":
    main()
