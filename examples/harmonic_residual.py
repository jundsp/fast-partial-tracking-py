"""Splits a sound into its sinusoidal part and a noise residual.

The resynthesized partials form the deterministic (sinusoidal) part; subtracting them from
the input leaves the stochastic residual (breath, bow and attack noise). Writes both as WAV
files and plots their spectrograms.

Run:
    python examples/harmonic_residual.py [audio.wav] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.harmonic_residual")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "harmonic_residual"


def plot_decomposition(parts: dict[str, np.ndarray], fs: float, out_path: Path, max_frequency_hz: float = 8000) -> None:
    """Spectrograms of the input, sinusoidal part and residual, stacked.

    Args:
        parts: Signals by name.
        fs: Sampling rate in Hz.
        out_path: Figure destination.
        max_frequency_hz: Upper frequency limit.
    """
    fig, axes = plt.subplots(len(parts), 1, figsize=(8, 2.6 * len(parts)), sharex=True)
    for ax, (name, signal) in zip(axes, parts.items()):
        fpt.plot.spectrogram(signal, fs, ax=ax, n_fft=1024)
        ax.set_ylim(0, max_frequency_hz)
        energy = 10 * np.log10(np.sum(signal**2) / np.sum(parts["input"] ** 2))
        ax.set_title(f"{name} ({energy:+.1f} dB re input)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, nargs="?", default=fpt.data.demo_sound_path())
    parser.add_argument("--min-frames", type=int, default=4, help="drop partials shorter than this (likely noise)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fpt.plot.use_style()

    audio, fs = sf.read(args.audio, always_2d=True)
    x = audio[:, 0]
    partials = fpt.track(x, fs, fpt.AnalysisConfig.from_duration(fs, window_ms=46, threshold_db=-70))
    sinusoidal = fpt.synthesize(partials.prune(args.min_frames))
    parts = {"input": x, "sinusoidal": sinusoidal, "residual": x - sinusoidal}

    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, signal in parts.items():
        sf.write(data_dir / f"{name}.wav", signal, fs, subtype="FLOAT")
    log.info(f"sinusoidal part keeps {fpt.rsnr(x, sinusoidal):.1f} dB R-SNR; WAVs -> {data_dir}")
    plot_decomposition(parts, fs, args.out / "figures" / "decomposition.png")


if __name__ == "__main__":
    main()
