"""Command-line interface: ``fpt analyze | synth | convert | info | plot | ui``."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from fast_partial_tracking.analysis.windows import get_window
from fast_partial_tracking.config import AnalysisConfig, TrackingConfig
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.io import open_writer, read, write
from fast_partial_tracking.metrics import rsnr
from fast_partial_tracking.partials import Partials
from fast_partial_tracking.synthesis import synthesize
from fast_partial_tracking.tracking.tracker import PartialTracker, reference_level_at, track

log = logging.getLogger("fast_partial_tracking.cli")


def _analysis_args(parser: argparse.ArgumentParser) -> None:
    tracking = TrackingConfig()
    group = parser.add_argument_group("analysis")
    group.add_argument("--config", type=Path, help="JSON file with 'analysis'/'tracking' settings (overrides flags)")
    group.add_argument("--preset", choices=["default", "matlab"], default="default",
                       help="'matlab' reproduces jun_track_partials.m (uses --hop-factor/--oversample)")
    group.add_argument("--window-ms", type=float, default=None, help="window duration (default 46 ms)")
    group.add_argument("--window-length", type=int, default=None, help="window length N in samples")
    group.add_argument("--hop", type=int, default=None, help="hop size H in samples")
    group.add_argument("--n-fft", type=int, default=None, help="DFT size")
    group.add_argument("--window", choices=["blackman_harris", "hann"], default="blackman_harris")
    group.add_argument("--order", type=int, default=2, help="polynomial order Q")
    group.add_argument("--threshold-db", type=float, default=-60.0, help="peak threshold below the loudest frame")
    group.add_argument("--hop-factor", type=int, default=4, help="matlab preset: H = N / hop-factor")
    group.add_argument("--oversample", type=int, default=2, help="matlab preset: DFT oversampling")
    group = parser.add_argument_group("tracking")
    group.add_argument("--delta", type=float, default=tracking.delta)
    group.add_argument("--zeta-f", type=float, default=tracking.zeta_f_hz, help="Hz")
    group.add_argument("--zeta-a", type=float, default=tracking.zeta_a_db, help="dB")
    group.add_argument("--prune", type=int, default=0, metavar="N", help="drop partials shorter than N frames")
    parser.add_argument("--no-alpha", action="store_true", help="do not store model coefficients")


def _configs(args: argparse.Namespace, fs: float) -> tuple[AnalysisConfig, TrackingConfig]:
    if args.config:
        settings = json.loads(args.config.read_text())
        return AnalysisConfig(**settings.get("analysis", {})), TrackingConfig(**settings.get("tracking", {}))
    tracking = TrackingConfig(delta=args.delta, zeta_f_hz=args.zeta_f, zeta_a_db=args.zeta_a)
    if args.preset == "matlab":
        n = args.window_length or int(2 ** np.ceil(np.log2(0.025 * fs)) - 1)
        return AnalysisConfig.matlab(n, args.hop_factor, args.oversample, args.threshold_db, args.order), tracking
    common = dict(hop_size=args.hop, n_fft=args.n_fft, window=args.window, poly_order=args.order,
                  threshold_db=args.threshold_db)
    if args.window_length:
        return AnalysisConfig(window_length=args.window_length, **common), tracking
    return AnalysisConfig.from_duration(fs, window_ms=args.window_ms or 46.0, **common), tracking


def cmd_analyze(args: argparse.Namespace) -> None:
    """Tracks an audio file and writes the partials."""
    fs = sf.info(str(args.input)).samplerate
    analysis, tracking = _configs(args, fs)
    if args.stream:
        if args.prune:
            raise SystemExit("--prune needs the whole result; drop --stream or prune later")
        _analyze_stream(args, fs, analysis, tracking)
    else:
        _analyze_batch(args, analysis, tracking)


def _analyze_stream(args: argparse.Namespace, fs: float, analysis: AnalysisConfig, tracking: TrackingConfig) -> None:
    """Reads, tracks and writes block by block, with bounded memory."""
    order = None if args.no_alpha else analysis.poly_order
    reference = _file_reference_db(args.input, fs, analysis) if analysis.threshold_reference == "signal" else None
    tracker = PartialTracker(fs, analysis, tracking, reference_db=reference, keep_alpha=not args.no_alpha)
    blocks = (block[:, 0] for block in sf.blocks(str(args.input), blocksize=args.block_size, always_2d=True))
    with open_writer(args.output, tracker.grid, order=order) as writer:
        for frame in tracker.process(blocks):
            writer.write_frame(frame)
        writer.set_summary(tracker.grid, tracker.num_tracks)
    log.info(f"{tracker.num_tracks} partials, {tracker.grid.num_frames} frames -> {args.output}")


def _file_reference_db(path: Path, fs: float, analysis: AnalysisConfig) -> float:
    """Signal-relative threshold reference of a file, found with one bounded-memory pass."""
    grid = FrameGrid.from_config(fs, analysis)
    window = get_window(analysis.window, grid.window_length)[0]
    loudest, peak, offset = 0, -1.0, 0
    for block in sf.blocks(str(path), blocksize=65536, always_2d=True):
        magnitude = np.abs(block[:, 0])
        if magnitude.size and magnitude.max() > peak:
            peak, loudest = float(magnitude.max()), offset + int(np.argmax(magnitude))
        offset += magnitude.size
    if peak <= 0:
        return -np.inf
    start = max(loudest - grid.window_length, 0)
    excerpt, _ = sf.read(str(path), start=start, frames=2 * grid.window_length, always_2d=True)
    return reference_level_at(excerpt[:, 0], loudest - start, grid, window)


def _analyze_batch(args: argparse.Namespace, analysis: AnalysisConfig, tracking: TrackingConfig) -> None:
    """Tracks the whole file at once (threshold relative to the loudest frame)."""
    x, fs = sf.read(str(args.input), always_2d=True)
    partials = track(x[:, 0], fs, analysis, tracking, keep_alpha=not args.no_alpha)
    if args.prune:
        partials = partials.prune(args.prune)
    write(partials, args.output, include_alpha=not args.no_alpha)
    log.info(f"{partials.num_tracks} partials, {partials.num_frames} frames -> {args.output}")
    if args.report:
        log.info(f"R-SNR = {rsnr(x[:, 0], synthesize(partials)):.2f} dB")


def cmd_synth(args: argparse.Namespace) -> None:
    """Resynthesizes a partials file to audio."""
    partials = read(args.input, fs=args.fs)
    y = synthesize(partials, phase=args.phase)
    sf.write(str(args.output), y, int(round(partials.grid.fs)), subtype="FLOAT")
    log.info(f"wrote {y.size} samples at {partials.grid.fs:g} Hz -> {args.output}")


def cmd_convert(args: argparse.Namespace) -> None:
    """Converts between partial file formats (e.g. Parquet to SDIF)."""
    write(read(args.input, fs=args.fs), args.output)
    log.info(f"{args.input} -> {args.output}")


def cmd_info(args: argparse.Namespace) -> None:
    """Summarizes a partials file: settings, counts and a partial-length histogram."""
    partials = read(args.input, fs=args.fs)
    for line in describe(partials, str(args.input)):
        log.info(line)


def describe(partials: Partials, name: str = "") -> list[str]:
    """Human-readable summary lines of a result."""
    grid, stats = partials.grid, partials.stats()
    lines = [
        name,
        f"  audio     {stats['duration_s']:.3f} s at {grid.fs:g} Hz",
        f"  frames    {stats['frames']} (N={grid.window_length}, hop={grid.hop}, n_fft={grid.n_dft}, {grid.geometry})",
        f"  partials  {stats['partials']} ({stats['rows']} points, median length {stats['median_partial_frames']:.0f} frames)",
    ]
    if stats["rows"]:
        lines.append(f"  range     {stats['min_frequency_hz']:.1f}-{stats['max_frequency_hz']:.1f} Hz, "
                     f"loudest {stats['max_amplitude_db']:.1f} dB")
        lengths = partials.track_lengths()
        edges = [1, 2, 4, 8, 16, 32, 64, np.inf]
        counts = np.histogram(lengths[lengths > 0], bins=edges)[0]
        width = max(counts.max(), 1)
        lines.append("  length (frames)")
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            label = f"{lo}+" if np.isinf(hi) else f"{lo}-{int(hi) - 1}"
            lines.append(f"    {label:>6}  {'#' * int(round(30 * count / width)):<30} {count}")
    analysis = partials.metadata.get("analysis")
    if analysis:
        lines.append("  settings  " + ", ".join(f"{k}={v}" for k, v in analysis.items() if v is not None))
    return [line for line in lines if line]


def cmd_plot(args: argparse.Namespace) -> None:
    """Plots a partials file (optionally over the audio's spectrogram) to an image."""
    from fast_partial_tracking.plot import figure

    partials = read(args.input, fs=args.fs)
    audio = None
    if args.audio:
        audio, _ = sf.read(str(args.audio), always_2d=True)
        audio = audio[:, 0]
    fig = figure(partials, audio=audio, max_frequency_hz=args.max_freq, min_frames=args.min_frames,
                 color_by=args.color_by)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    log.info(f"figure -> {args.output}")


def cmd_ui(args: argparse.Namespace) -> None:
    """Opens the interactive web app (needs the [ui] extra)."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit("the UI needs streamlit and plotly: pip install streamlit plotly") from exc
    app = Path(__file__).parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", str(args.headless).lower()]
    if args.port:
        command += ["--server.port", str(args.port)]
    if args.audio:
        command += ["--", "--audio", str(args.audio.resolve())]
    subprocess.run(command, check=False)


def _package_version() -> str:
    from fast_partial_tracking import __version__

    return __version__


FS_HELP = "sampling rate of an .sdif input, which SDIF does not store (default 44100)"


def build_parser() -> argparse.ArgumentParser:
    """Creates the argument parser."""
    parser = argparse.ArgumentParser(prog="fpt", description="Fast partial tracking (Neri & Depalle, DAFx-18)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
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
    synth.add_argument("--fs", type=float, default=None, help=FS_HELP)
    synth.add_argument("-o", "--output", type=Path, required=True, help="audio file, e.g. .wav")
    synth.add_argument("--phase", choices=["measured", "integrated"], default=None)
    synth.set_defaults(func=cmd_synth)

    convert = sub.add_parser("convert", help="convert between partial file formats")
    convert.add_argument("input", type=Path)
    convert.add_argument("--fs", type=float, default=None, help=FS_HELP)
    convert.add_argument("-o", "--output", type=Path, required=True)
    convert.set_defaults(func=cmd_convert)

    info = sub.add_parser("info", help="summarize a partials file")
    info.add_argument("input", type=Path)
    info.add_argument("--fs", type=float, default=None, help=FS_HELP)
    info.set_defaults(func=cmd_info)

    plot = sub.add_parser("plot", help="plot a partials file to an image")
    plot.add_argument("input", type=Path)
    plot.add_argument("--fs", type=float, default=None, help=FS_HELP)
    plot.add_argument("-o", "--output", type=Path, required=True, help="image file, e.g. .png or .pdf")
    plot.add_argument("--audio", type=Path, help="draw this audio's spectrogram underneath")
    plot.add_argument("--color-by", choices=["track", "amplitude", "none"], default="track")
    plot.add_argument("--min-frames", type=int, default=3, help="hide partials shorter than this")
    plot.add_argument("--max-freq", type=float, default=None, help="upper frequency limit in Hz")
    plot.add_argument("--dpi", type=int, default=150)
    plot.set_defaults(func=cmd_plot)

    ui = sub.add_parser("ui", help="open the interactive web app")
    ui.add_argument("audio", type=Path, nargs="?", help="audio file to open")
    ui.add_argument("--port", type=int, default=None)
    ui.add_argument("--headless", action="store_true", help="do not open a browser")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``fpt`` command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
