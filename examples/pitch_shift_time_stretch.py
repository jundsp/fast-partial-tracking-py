"""Pitch shifting and time stretching by editing partials.

Analysis once, then any number of transformations; the phase-integrating synthesizer
renders them without the phasing artefacts of naive resampling or overlap-add.

Run:
    python examples/pitch_shift_time_stretch.py [audio.wav] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.pitch_shift_time_stretch")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "pitch_shift_time_stretch"


def plot_variants(variants: dict[str, fpt.Partials], out_path: Path, max_frequency_hz: float = 5000) -> None:
    """Partials of every variant side by side, on a shared frequency axis.

    Args:
        variants: Results by name.
        out_path: Figure destination.
        max_frequency_hz: Upper frequency limit.
    """
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.5), sharey=True)
    for ax, (name, partials) in zip(axes, variants.items()):
        fpt.plot.partials(partials, ax=ax, max_frequency_hz=max_frequency_hz, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, nargs="?", default=fpt.data.demo_sound_path())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    audio, fs = sf.read(args.audio, always_2d=True)
    partials = fpt.track(audio[:, 0], fs).prune(3)
    variants = {
        "original": partials,
        "+5 semitones": fpt.transforms.pitch_shift(partials, semitones=5),
        "-7 semitones": fpt.transforms.pitch_shift(partials, semitones=-7),
        "2x slower": fpt.transforms.time_stretch(partials, 2.0),
    }
    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, result in variants.items():
        path = data_dir / f"{name.replace(' ', '_').replace('+', 'up').replace('-', 'down')}.wav"
        sf.write(path, fpt.synthesize(result), fs, subtype="FLOAT")
        log.info(f"{name:14s} -> {path.name}")
    plot_variants(variants, args.out / "figures" / "variants.png")


if __name__ == "__main__":
    main()
