"""Streams audio through the tracker and synthesizer in fixed-size blocks.

Simulates a live input by looping the demo sound for a long time, writes the partials
incrementally to an Arrow stream, resynthesizes on the fly, and checks that memory use
does not grow with the input length.

Run:
    python examples/streaming_demo.py [--seconds 120] [--block 512] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.streaming_demo")

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "demo_sound.wav"
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "streaming_demo"


def plot_memory(seconds: np.ndarray, megabytes: np.ndarray, out_path: Path) -> None:
    """Plots traced memory against the amount of audio processed.

    Args:
        seconds: Audio processed so far, in seconds.
        megabytes: Traced Python memory at that point.
        out_path: Figure destination.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(seconds, megabytes, color="tab:blue")
    ax.set_xlabel("audio processed (s)")
    ax.set_ylabel("traced memory (MB)")
    ax.set_ylim(0, max(1.0, megabytes.max() * 1.2))
    ax.set_title("Streaming tracker + Arrow writer + synthesizer")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=120.0, help="length of the simulated live input")
    parser.add_argument("--block", type=int, default=512, help="audio block size in samples")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source, fs = sf.read(DEFAULT_AUDIO)
    total = int(args.seconds * fs)
    analysis = fpt.AnalysisConfig(window_length=2047, oversample=2, peak_threshold_db=-50)
    tracker = fpt.PartialTracker(fs, analysis, threshold_db=None)
    synth = fpt.Synthesizer(tracker.grid)
    stream_path = args.out / "data" / "live.arrows"

    tracemalloc.start()
    processed, memory = [], []
    started = time.perf_counter()
    emitted_samples = 0
    with fpt.io.open_writer(stream_path, tracker.grid, order=tracker.estimator.order) as writer:
        for start in range(0, total, args.block):
            index = (start + np.arange(min(args.block, total - start))) % source.size
            for frame in tracker.push(source[index]):
                writer.write_frame(frame)
                emitted_samples += synth.push(frame).size
            if start // args.block % 200 == 0:
                processed.append(start / fs)
                memory.append(tracemalloc.get_traced_memory()[0] / 1e6)
        for frame in tracker.flush():
            writer.write_frame(frame)
            emitted_samples += synth.push(frame).size
        writer.set_summary(tracker.grid, tracker.num_tracks)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info(
        f"{args.seconds:.0f} s of audio in {elapsed:.1f} s ({args.seconds / elapsed:.1f}x real time), "
        f"{tracker.num_tracks} partials, peak traced memory {peak / 1e6:.1f} MB"
    )
    log.info(f"resynthesized {emitted_samples / fs:.1f} s on the fly; partial stream -> {stream_path}")
    plot_memory(np.array(processed), np.array(memory), args.out / "figures" / "memory.png")


if __name__ == "__main__":
    main()
