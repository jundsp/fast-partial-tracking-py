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

import numpy as np
import soundfile as sf

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.track_demo")

DEFAULT_AUDIO = fpt.data.demo_sound_path()
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "track_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, nargs="?", default=DEFAULT_AUDIO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fpt.plot.use_style()

    x, fs = sf.read(args.audio, always_2d=True)
    x = x[:, 0]
    # Same settings as jun_demo.m, reproduced exactly by the MATLAB preset.
    analysis = fpt.AnalysisConfig.matlab(
        window_length=int(2 ** np.ceil(np.log2(0.025 * fs)) - 1),
        hop_factor=4,
        oversample=2,
        peak_threshold_db=-50,
        order=2,
    )
    tracking = fpt.TrackingConfig.matlab(delta=0.2, zeta_f=50, zeta_a=15)

    partials = fpt.track(x, fs, analysis, tracking)
    y = fpt.synthesize(partials)
    log.info(f"R-SNR = {fpt.rsnr(x, y):.2f} dB ({partials.num_tracks} partials, {partials.num_frames} frames)")

    data_dir = args.out / "data"
    partials.save(data_dir / "partials.parquet")
    partials.save(data_dir / "partials.sdif")
    sf.write(data_dir / "resynthesis.wav", y, int(fs), subtype="FLOAT")
    log.info(f"partials and resynthesis -> {data_dir}")
    fpt.plot.figure(partials, x, out_path=args.out / "figures" / "partials.png")
    fpt.plot.overview(partials, x, y, out_path=args.out / "figures" / "overview.png")


if __name__ == "__main__":
    main()
