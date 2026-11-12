"""Command-line interface: ``fpt analyze | synth | convert | info``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from fast_partial_tracking.config import AnalysisConfig, TrackingConfig
from fast_partial_tracking.io import open_writer, read, write
from fast_partial_tracking.metrics import rsnr
from fast_partial_tracking.synthesis import synthesize
from fast_partial_tracking.tracker import PartialTracker, track

log = logging.getLogger("fast_partial_tracking.cli")


def _analysis_args(parser: argparse.ArgumentParser) -> None:
    defaults = AnalysisConfig()
    tracking = TrackingConfig()
    parser.add_argument("--window-length", type=int, default=None, help="N in samples (default: ~25 ms, 2^k - 1)")
    parser.add_argument("--hop-factor", type=int, default=defaults.hop_factor)
    parser.add_argument("--oversample", type=int, default=2, help="DFT oversampling (>1 uses Blackman-Harris)")
    parser.add_argument("--threshold-db", type=float, default=-50.0, help="peak threshold below the loudest frame")
    parser.add_argument("--order", type=int, default=defaults.order, help="polynomial order Q")
    parser.add_argument("--delta", type=float, default=tracking.delta)
    parser.add_argument("--zeta-f", type=float, default=tracking.zeta_f, help="Hz")
    parser.add_argument("--zeta-a", type=float, default=tracking.zeta_a, help="dB")
    parser.add_argument("--no-alpha", action="store_true", help="do not store model coefficients")


def _configs(args: argparse.Namespace, fs: float) -> tuple[AnalysisConfig, TrackingConfig]:
    window_length = args.window_length or int(2 ** np.ceil(np.log2(0.025 * fs)) - 1)
    analysis = AnalysisConfig(
        window_length=window_length,
        hop_factor=args.hop_factor,
        oversample=args.oversample,
        peak_threshold_db=args.threshold_db,
        order=args.order,
    )
    return analysis, TrackingConfig(delta=args.delta, zeta_f=args.zeta_f, zeta_a=args.zeta_a)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Tracks an audio file and writes the partials."""
    fs = sf.info(str(args.input)).samplerate
    analysis, tracking = _configs(args, fs)
    if args.stream:
        _analyze_stream(args, fs, analysis, tracking)
    else:
        _analyze_batch(args, analysis, tracking)


def _analyze_stream(args: argparse.Namespace, fs: float, analysis: AnalysisConfig, tracking: TrackingConfig) -> None:
    """Reads, tracks and writes block by block, with bounded memory."""
    order = None if args.no_alpha else analysis.effective_order
    tracker = PartialTracker(fs, analysis, tracking, keep_alpha=not args.no_alpha)
    with open_writer(args.output, tracker.grid, order=order) as writer:
        for block in sf.blocks(str(args.input), blocksize=args.block_size, always_2d=True):
            for frame in tracker.push(block[:, 0]):
                writer.write_frame(frame)
        for frame in tracker.flush():
            writer.write_frame(frame)
        writer.set_summary(tracker.grid, tracker.num_tracks)
    log.info(f"{tracker.num_tracks} partials, {tracker.grid.num_frames} frames -> {args.output}")


def _analyze_batch(args: argparse.Namespace, analysis: AnalysisConfig, tracking: TrackingConfig) -> None:
    """Tracks the whole file at once (threshold relative to the loudest frame, like MATLAB)."""
    x, fs = sf.read(str(args.input), always_2d=True)
    partials = track(x[:, 0], fs, analysis, tracking, keep_alpha=not args.no_alpha)
    write(partials, args.output, include_alpha=not args.no_alpha)
    log.info(f"{partials.num_tracks} partials, {partials.num_frames} frames -> {args.output}")
    if args.report:
        log.info(f"R-SNR = {rsnr(x[:, 0], synthesize(partials)):.2f} dB")


def cmd_synth(args: argparse.Namespace) -> None:
    """Resynthesizes a partials file to audio."""
    partials = read(args.input)
    y = synthesize(partials)
    sf.write(str(args.output), y, int(round(partials.grid.fs)), subtype="FLOAT")
    log.info(f"wrote {y.size} samples at {partials.grid.fs:g} Hz -> {args.output}")


def cmd_convert(args: argparse.Namespace) -> None:
    """Converts between partial file formats (e.g. Parquet to SDIF)."""
    write(read(args.input), args.output)
    log.info(f"{args.input} -> {args.output}")


def cmd_info(args: argparse.Namespace) -> None:
    """Summarizes a partials file."""
    partials = read(args.input)
    grid = partials.grid
    duration = (grid.signal_length or 0) / grid.fs
    log.info(f"{args.input}")
    log.info(f"  fs={grid.fs:g} Hz, duration={duration:.3f} s, frames={partials.num_frames}, hop={grid.hop}")
    log.info(f"  partials={partials.num_tracks}, rows={len(partials)}, alpha={'yes' if partials.alpha is not None else 'no'}")
    if len(partials):
        log.info(f"  frequency range {partials.frequency_hz.min():.1f}-{partials.frequency_hz.max():.1f} Hz")


def build_parser() -> argparse.ArgumentParser:
    """Creates the argument parser."""
    parser = argparse.ArgumentParser(prog="fpt", description="Fast partial tracking (Neri & Depalle, DAFx-18)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="track partials of an audio file")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("-o", "--output", type=Path, required=True, help=".parquet, .arrows, .sdif or .bin")
    analyze.add_argument("--stream", action="store_true", help="read and write block by block (bounded memory)")
    analyze.add_argument("--block-size", type=int, default=4096)
    analyze.add_argument("--report", action="store_true", help="log the resynthesis R-SNR")
    _analysis_args(analyze)
    analyze.set_defaults(func=cmd_analyze)

    synth = sub.add_parser("synth", help="resynthesize a partials file")
    synth.add_argument("input", type=Path)
    synth.add_argument("-o", "--output", type=Path, required=True, help="audio file, e.g. .wav")
    synth.set_defaults(func=cmd_synth)

    convert = sub.add_parser("convert", help="convert between partial file formats")
    convert.add_argument("input", type=Path)
    convert.add_argument("-o", "--output", type=Path, required=True)
    convert.set_defaults(func=cmd_convert)

    info = sub.add_parser("info", help="summarize a partials file")
    info.add_argument("input", type=Path)
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``fpt`` command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
