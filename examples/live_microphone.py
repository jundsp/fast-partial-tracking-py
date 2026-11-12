"""Real-time partial tracking from the microphone, with a live scrolling display.

Audio blocks arrive on a queue from the input device (or, with ``--simulate``, from the
demo recording played at real-time pace). The main loop tracks *all* queued audio, then
redraws at most ``--fps`` times per second. The display is a scrolling image (one column per
frame), so drawing costs the same however many partials there are.

The title shows the **lag**: audio waiting in the queue. If processing fell behind it would
grow without bound; a warning is logged above 0.5 s and the maximum is reported at the end.

Needs ``uv sync --extra dev`` in a clone (or ``pip install 'fast-partial-tracking[live,plot]'``)
and microphone permission for your terminal.

Run:
    python examples/live_microphone.py [--seconds 30] [--simulate] [--fps 8] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.live_microphone")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "live_microphone"
BLOCK = 1024
LAG_WARNING_S = 0.5


class AudioSource:
    """Audio blocks on a queue, filled in the background at real-time pace.

    Args:
        fs: Sampling rate in Hz.
        seconds: How long to record.
        simulate: Replay the demo recording instead of opening the microphone.
    """

    def __init__(self, fs: int, seconds: float, simulate: bool) -> None:
        self.fs = fs
        self.seconds = seconds
        self.blocks: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread = threading.Thread(target=self._simulate if simulate else self._record, daemon=True)
        self._thread.start()

    @property
    def lag_s(self) -> float:
        """Audio waiting to be processed, in seconds."""
        return self.blocks.qsize() * BLOCK / self.fs

    def drain(self, timeout: float) -> Iterator[np.ndarray] | None:
        """All blocks queued right now (waits up to ``timeout`` for the first); None at the end."""
        try:
            first = self.blocks.get(timeout=timeout)
        except queue.Empty:
            return iter(())
        if first is None:
            return None
        pending = [first]
        while True:
            try:
                block = self.blocks.get_nowait()
            except queue.Empty:
                break
            if block is None:
                self.blocks.put(None)
                break
            pending.append(block)
        return iter(pending)

    def _record(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            log.error("The microphone needs sounddevice: run `uv sync --extra dev` in the repo "
                      "(or `pip install 'fast-partial-tracking[live]'`). Use --simulate to run without one.")
            self.blocks.put(None)
            return
        callback = lambda data, *_: self.blocks.put(data[:, 0].copy())  # noqa: E731
        with sd.InputStream(samplerate=self.fs, channels=1, blocksize=BLOCK, callback=callback):
            time.sleep(self.seconds)
        self.blocks.put(None)

    def _simulate(self) -> None:
        x, _ = fpt.data.demo_sound()
        start = time.monotonic()
        total = int(self.seconds * self.fs)
        for offset in range(0, total, BLOCK):
            # Release each block when it would have been recorded.
            time.sleep(max(0.0, start + (offset + BLOCK) / self.fs - time.monotonic()))
            self.blocks.put(x[(offset + np.arange(min(BLOCK, total - offset))) % x.size])
        self.blocks.put(None)


class ScrollingDisplay:
    """A time-frequency image of the latest partials, updated in place.

    Args:
        hop_s: Time between frames in seconds.
        window_s: Visible duration.
        max_frequency_hz: Top of the frequency axis.
        bins: Frequency resolution of the image.
        dynamic_range_db: Levels shown below the loudest partial seen so far.
    """

    def __init__(self, hop_s: float, window_s: float = 6.0, max_frequency_hz: float = 5000.0,
                 bins: int = 400, dynamic_range_db: float = 70.0) -> None:
        self.columns = int(round(window_s / hop_s))
        self.max_frequency_hz = max_frequency_hz
        self.bins = bins
        self.dynamic_range_db = dynamic_range_db
        self.image = np.full((bins, self.columns), np.nan)
        self.loudest = -np.inf
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.artist = self.ax.imshow(self.image, origin="lower", aspect="auto", cmap="magma", interpolation="nearest",
                                     extent=(-window_s, 0.0, 0.0, max_frequency_hz))
        self.ax.set_xlabel("time relative to now (s)")
        self.ax.set_ylabel("frequency (Hz)")
        self.ax.set_facecolor("black")

    def add(self, frames: list[fpt.Frame]) -> None:
        """Scrolls the image by one column per frame and draws the frames' partials."""
        if not frames:
            return
        shift = min(len(frames), self.columns)
        self.image = np.roll(self.image, -shift, axis=1)
        self.image[:, -shift:] = np.nan
        for column, frame in zip(range(self.columns - shift, self.columns), frames[-shift:]):
            keep = frame.frequency_hz < self.max_frequency_hz
            if not np.any(keep):
                continue
            rows = (frame.frequency_hz[keep] / self.max_frequency_hz * self.bins).astype(int)
            np.fmax.at(self.image[:, column], rows, frame.amplitude_db[keep])
            self.loudest = max(self.loudest, float(frame.amplitude_db[keep].max()))

    def draw(self, title: str) -> None:
        """Pushes the current image to the screen."""
        self.artist.set_data(self.image)
        self.artist.set_clim(self.loudest - self.dynamic_range_db, self.loudest)
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--simulate", action="store_true", help="replay the demo sound instead of the microphone")
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--fps", type=float, default=8.0, help="display updates per second")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fpt.plot.use_style()

    fs = 44100 if args.simulate else args.fs
    # A live input cannot know its loudest frame in advance: threshold relative to full scale.
    analysis = fpt.AnalysisConfig.from_duration(fs, window_ms=46, threshold_db=-75, threshold_reference="full_scale")
    tracker = fpt.PartialTracker(fs, analysis)
    interactive = "agg" not in plt.get_backend().lower()
    if interactive:
        plt.ion()
    display = ScrollingDisplay(hop_s=tracker.grid.hop / fs)
    if interactive:
        plt.show(block=False)

    source = AudioSource(fs, args.seconds, args.simulate)
    max_lag, warned, frames_total = 0.0, False, 0
    next_draw = time.monotonic()
    while True:
        lag = source.lag_s
        max_lag = max(max_lag, lag)
        if lag > LAG_WARNING_S and not warned:
            log.warning(f"falling behind: {lag:.2f} s of audio queued")
            warned = True
        blocks = source.drain(timeout=0.1)
        if blocks is None:
            break
        frames = [frame for block in blocks for frame in tracker.push(block)]
        frames_total += len(frames)
        display.add(frames)
        if time.monotonic() >= next_draw:
            display.draw(f"{tracker.num_tracks} partials   lag {lag * 1000:.0f} ms")
            next_draw = time.monotonic() + 1.0 / args.fps
    display.add(tracker.flush())

    display.draw(f"{tracker.num_tracks} partials   max lag {max_lag * 1000:.0f} ms")
    out_path = args.out / "figures" / "last_seconds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display.fig.savefig(out_path, dpi=150)
    log.info(f"{frames_total} frames, {tracker.num_tracks} partials; max lag {max_lag * 1000:.0f} ms "
             f"({'kept up' if max_lag < LAG_WARNING_S else 'fell behind'}); figure -> {out_path}")


if __name__ == "__main__":
    main()
