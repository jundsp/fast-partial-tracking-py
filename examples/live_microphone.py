"""Real-time partial tracking from the microphone, with a live plot.

Audio blocks from the default input device are pushed into a ``PartialTracker`` as they
arrive; finished frames are drawn on a scrolling time-frequency plot. Needs
``pip install 'fast-partial-tracking[live,plot]'`` and microphone permission.

``--simulate`` replays the demo recording in real-time-sized blocks instead (no device
needed), which exercises the same code path.

Run:
    python examples/live_microphone.py [--seconds 10] [--simulate] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import queue
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.live_microphone")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "live_microphone"
BLOCK = 1024


def microphone_blocks(fs: int, seconds: float):
    """Yields mono blocks from the default input device for ``seconds``."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "The microphone needs sounddevice: run `uv sync --extra dev` in the repo "
            "(or `pip install 'fast-partial-tracking[live]'`). Use --simulate to run without a microphone."
        ) from exc

    blocks: queue.Queue[np.ndarray] = queue.Queue()
    with sd.InputStream(samplerate=fs, channels=1, blocksize=BLOCK, callback=lambda data, *_: blocks.put(data[:, 0].copy())):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            yield blocks.get()


def simulated_blocks(seconds: float):
    """Yields the demo recording, looped, in real-time-sized blocks."""
    x, fs = fpt.data.demo_sound()
    total = int(seconds * fs)
    for start in range(0, total, BLOCK):
        yield x[(start + np.arange(min(BLOCK, total - start))) % x.size]
        time.sleep(BLOCK / fs * 0.2)


def plot_live(points: deque, ax: plt.Axes, window_s: float) -> None:
    """Redraws the last ``window_s`` seconds of partial points."""
    ax.clear()
    if points:
        t, f, level = np.array(points).T
        ax.scatter(t, f, c=level, s=2, cmap="plasma", vmin=level.max() - 70, vmax=level.max())
        ax.set_xlim(t.max() - window_s, t.max())
    ax.set_ylim(0, 5000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--simulate", action="store_true", help="replay the demo sound instead of the microphone")
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fs = 44100 if args.simulate else args.fs
    # A live input cannot know its loudest frame in advance: threshold relative to full scale.
    analysis = fpt.AnalysisConfig.from_duration(fs, window_ms=46, threshold_db=-75, threshold_reference="full_scale")
    tracker = fpt.PartialTracker(fs, analysis)
    blocks = simulated_blocks(args.seconds) if args.simulate else microphone_blocks(fs, args.seconds)

    interactive = "agg" not in plt.get_backend().lower()
    if interactive:
        plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))
    points: deque = deque(maxlen=20000)
    frames = 0
    for frame in tracker.process(blocks):
        frames += 1
        points.extend(zip(np.full(len(frame), frame.time_s), frame.frequency_hz, frame.amplitude_db))
        if interactive and frames % 8 == 0:
            plot_live(points, ax, window_s=5.0)
            plt.pause(0.001)
    plot_live(points, ax, window_s=5.0)
    out_path = args.out / "figures" / "last_seconds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info(f"{frames} frames, {tracker.num_tracks} partials; figure -> {out_path}")


if __name__ == "__main__":
    main()
