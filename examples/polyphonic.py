"""Polyphony: overlapping notes and crossing partials.

Tracks a synthetic chord whose notes enter one after another and two chirps whose
frequencies cross, then colors each partial individually. Away from the crossing the
assignment keeps the two chirps apart; at the crossing itself both are one spectral peak
for a few frames, so one of them continues and the other restarts as a new partial.

Run:
    python examples/polyphonic.py [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.polyphonic")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "polyphonic"


def plot_cases(results: dict[str, fpt.Partials], out_path: Path) -> None:
    """One panel per signal, each partial in its own color.

    Args:
        results: Pruned results by signal name.
        out_path: Figure destination.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    for ax, (name, partials) in zip(axes, results.items()):
        fpt.plot.partials(partials, ax=ax, color_by="track", cmap="tab20", max_frequency_hz=3000, linewidth=1.5)
        ax.set_title(f"{name}: {partials.num_tracks} partials")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = {}
    for name in ["chord", "crossing"]:
        x, fs = fpt.data.synthetic(name, duration=2.0)
        partials = fpt.track(x, fs, fpt.AnalysisConfig(window_length=1023, threshold_db=-50))
        pruned = partials.prune(min_frames=10)
        results[name] = pruned
        log.info(f"{name:9s} {partials.num_tracks:4d} partials, {pruned.num_tracks} after pruning; "
                 f"R-SNR {fpt.rsnr(x, fpt.synthesize(partials)):.1f} dB")
    plot_cases(results, args.out / "figures" / "polyphonic.png")


if __name__ == "__main__":
    main()
