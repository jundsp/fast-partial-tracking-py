"""Exports tracked partials to binary SDIF and reads the file back.

The MATLAB version wrote a text file that had to be converted with IRCAM's ``tosdif``
tool; this writes the binary ``1TRC`` SDIF file directly.

Run:
    python examples/sdif_export.py [audio.wav] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import soundfile as sf

import fast_partial_tracking as fpt
from fast_partial_tracking.io import read_sdif_tracks

log = logging.getLogger("fast_partial_tracking.examples.sdif_export")

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "demo_sound.wav"
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "sdif_export"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, nargs="?", default=DEFAULT_AUDIO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    x, fs = sf.read(args.audio, always_2d=True)
    partials = fpt.track(x[:, 0], fs, fpt.AnalysisConfig(window_length=2047, threshold_db=-50))
    path = args.out / "data" / f"{args.audio.stem}.sdif"
    partials.save(path)

    tracks = read_sdif_tracks(path)
    rows = sum(matrix.shape[0] for _, matrix in tracks)
    log.info(f"{path}: {len(tracks)} 1TRC frames, {rows} rows, {path.stat().st_size / 1024:.1f} KiB")
    time, matrix = tracks[len(tracks) // 2]
    log.info(f"frame at {time:.3f} s, first rows (index, frequency Hz, amplitude, phase):")
    for row in matrix[:5]:
        log.info(f"  {int(row[0]):5d} {row[1]:10.2f} {row[2]:10.5f} {row[3]:8.3f}")


if __name__ == "__main__":
    main()
