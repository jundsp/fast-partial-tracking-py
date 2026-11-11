"""Port of the MATLAB ``jun_demo.m``: track, resynthesize, report R-SNR and plot.

Run:
    python examples/track_demo.py [audio.wav] [--out DIR]

Writes the partials (.parquet and .sdif) and a figure of the partials drawn over the
spectrogram to ``~/Documents/output/fast-partial-tracking/examples/track_demo/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.track_demo")

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "demo_sound.wav"
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "track_demo"


def plot_partials(
    partials: fpt.Partials,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    max_freq: float = 5000.0,
) -> None:
    """Plots the input/resynthesis waveforms and the partials over the spectrogram.

    Args:
        partials: Tracking result.
        x: Input signal.
        y: Resynthesized signal.
        out_path: Figure destination.
        max_freq: Upper frequency limit of the plot in Hz.
    """
    fs = partials.grid.fs
    t = np.arange(x.size) / fs
    fig, (ax_wave, ax_tf) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[1, 3], sharex=True)
    ax_wave.plot(t, x, color="k", lw=0.6, label="input")
    ax_wave.plot(t, y, color="tab:red", lw=0.6, label="resynthesis")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_title(f"R-SNR = {fpt.rsnr(x, y):.2f} dB, {partials.num_tracks} partials")

    nfft = partials.grid.n_dft
    ax_tf.specgram(x, NFFT=min(nfft, 2048), Fs=fs, noverlap=min(nfft, 2048) * 3 // 4, cmap="gray_r", vmin=-120)
    level = 20 * np.log10(partials.amplitude)
    order = np.argsort(partials.track_id, kind="stable")
    tracks, starts = np.unique(partials.track_id[order], return_index=True)
    ends = np.r_[starts[1:], order.size]
    norm = plt.Normalize(level.max() - 80, level.max())
    for s, e in zip(starts, ends):
        rows = order[s:e]
        ax_tf.plot(partials.time_s[rows], partials.frequency_hz[rows], lw=1, color=plt.cm.plasma(norm(level[rows].mean())))
    ax_tf.set_ylim(0, max_freq)
    ax_tf.set_xlim(0, t[-1])
    ax_tf.set_xlabel("time (s)")
    ax_tf.set_ylabel("frequency (Hz)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, nargs="?", default=DEFAULT_AUDIO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    x, fs = sf.read(args.audio, always_2d=True)
    x = x[:, 0]
    # Same settings as jun_demo.m.
    analysis = fpt.AnalysisConfig(
        window_length=int(2 ** np.ceil(np.log2(0.025 * fs)) - 1),
        hop_factor=4,
        oversample=2,
        peak_threshold_db=-50,
        order=2,
    )
    tracking = fpt.TrackingConfig(delta=0.2, zeta_f=50, zeta_a=15)

    partials = fpt.track(x, fs, analysis, tracking)
    y = fpt.synthesize(partials)
    log.info(f"R-SNR = {fpt.rsnr(x, y):.2f} dB ({partials.num_tracks} partials, {partials.num_frames} frames)")

    data_dir = args.out / "data"
    partials.save(data_dir / "partials.parquet")
    partials.save(data_dir / "partials.sdif")
    sf.write(data_dir / "resynthesis.wav", y, int(fs), subtype="FLOAT")
    log.info(f"partials and resynthesis -> {data_dir}")
    plot_partials(partials, x, y, args.out / "figures" / "partials.png")


if __name__ == "__main__":
    main()
