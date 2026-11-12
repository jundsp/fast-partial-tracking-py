"""Interactive partial-tracking app (Streamlit). Start it with ``fpt ui [audio.wav]``."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import streamlit as st

import fast_partial_tracking as fpt
from fast_partial_tracking.cli import describe
from fast_partial_tracking.compat.matlab import nextpow2
from fast_partial_tracking.ui import figures

DEMO = "Demo voice (/kara/)"
UPLOAD = "Upload a file..."


def cli_audio() -> Path | None:
    """Audio path passed as ``fpt ui path`` (forwarded after ``--`` by streamlit)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audio", type=Path, default=None)
    return parser.parse_known_args(sys.argv[1:])[0].audio


def load_source() -> tuple[np.ndarray, int, str] | None:
    """Sidebar: choose the input signal (demo, synthetic, upload or a CLI path)."""
    st.sidebar.header("Audio")
    options = [DEMO] + [f"Synthetic: {name}" for name in fpt.data.SYNTHETIC] + [UPLOAD]
    path = cli_audio()
    if path is not None:
        options.insert(0, path.name)
    choice = st.sidebar.selectbox("Source", options)
    if path is not None and choice == path.name:
        audio, fs = sf.read(path, always_2d=True)
    elif choice == DEMO:
        audio, fs = sf.read(fpt.data.demo_sound_path(), always_2d=True)
    elif choice.startswith("Synthetic: "):
        x, fs = fpt.data.synthetic(choice.removeprefix("Synthetic: "), duration=2.0)
        audio = x[:, None]
    else:
        upload = st.sidebar.file_uploader("Audio file", type=["wav", "flac", "ogg", "aiff", "aif"])
        if upload is None:
            return None
        audio, fs = sf.read(io.BytesIO(upload.getvalue()), always_2d=True)
    channel = st.sidebar.selectbox("Channel", range(audio.shape[1])) if audio.shape[1] > 1 else 0
    x = audio[:, channel]
    duration = x.size / fs
    if duration > 1.0:
        start, end = st.sidebar.slider("Time range (s)", 0.0, float(duration), (0.0, float(duration)), step=0.05)
        x = x[int(start * fs) : max(int(end * fs), int(start * fs) + 1)]
    return x, int(fs), choice


def settings(fs: int) -> tuple[fpt.AnalysisConfig, fpt.TrackingConfig, int]:
    """Sidebar: analysis and tracking settings."""
    st.sidebar.header("Analysis")
    preset = st.sidebar.radio("Preset", ["Default", "MATLAB (jun_demo)"], horizontal=True)
    threshold = st.sidebar.slider("Peak threshold (dB below loudest frame)", -120, -20, -60)
    order = st.sidebar.select_slider("Polynomial order Q", [1, 2, 3], value=2)
    if preset == "Default":
        window_ms = st.sidebar.slider("Window (ms)", 10, 120, 46)
        window = st.sidebar.selectbox("Window type", ["blackman_harris", "hann"])
        hop_div = st.sidebar.select_slider("Hop (fraction of window)", [2, 4, 8, 16], value=8 if window == "blackman_harris" else 4,
                                           format_func=lambda d: f"1/{d}")
        analysis = fpt.AnalysisConfig.from_duration(fs, window_ms=window_ms, window=window, poly_order=order,
                                                    threshold_db=float(threshold))
        analysis = fpt.AnalysisConfig(**{**asdict(analysis), "hop_size": max(1, analysis.window_length // hop_div)})
    else:
        n = int(2 ** nextpow2(0.025 * fs) - 1)
        analysis = fpt.AnalysisConfig.matlab(n, hop_factor=4, oversample=2, peak_threshold_db=threshold, order=order)
    st.sidebar.header("Tracking")
    delta = st.sidebar.slider("δ (preference for continuing)", 0.05, 0.95, 0.2, 0.05)
    zeta_f = st.sidebar.slider("ζ_f (Hz)", 5, 300, 50, 5)
    zeta_a = st.sidebar.slider("ζ_a (dB)", 1, 40, 15)
    prune = st.sidebar.slider("Drop partials shorter than (frames)", 0, 20, 0)
    return analysis, fpt.TrackingConfig(delta=delta, zeta_f_hz=float(zeta_f), zeta_a_db=float(zeta_a)), prune


@st.cache_data(max_entries=16, show_spinner="Tracking partials...")
def analyze(x: np.ndarray, fs: int, analysis: dict[str, Any], tracking: dict[str, float], prune: int) -> tuple[fpt.Partials, np.ndarray, float]:
    """Tracks and resynthesizes (cached per audio and settings)."""
    started = time.perf_counter()
    partials = fpt.track(x, fs, fpt.AnalysisConfig(**analysis), fpt.TrackingConfig(**tracking))
    if prune:
        partials = partials.prune(prune)
    elapsed = time.perf_counter() - started
    return partials, fpt.synthesize(partials), elapsed


def wav_bytes(y: np.ndarray, fs: int) -> bytes:
    """Encodes audio as WAV, scaled down only if it would clip."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    buffer = io.BytesIO()
    sf.write(buffer, y / max(peak, 1.0), fs, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def file_bytes(partials: fpt.Partials, suffix: str) -> bytes:
    """Serializes partials to a file format by writing a temporary file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"partials{suffix}"
        partials.save(path)
        return path.read_bytes()


def overview_tab(partials: fpt.Partials, x: np.ndarray, y: np.ndarray, fs: int) -> None:
    left, right = st.columns([3, 1])
    max_freq = left.slider("Max frequency (Hz)", 500, fs // 2, min(5000, fs // 2), 100)
    color_by = right.radio("Color", ["track", "amplitude"], horizontal=True,
                           format_func=lambda c: "by partial" if c == "track" else "by level")
    figure = figures.overview_figure(partials, x, y, max_frequency_hz=max_freq, color_by=color_by)
    st.plotly_chart(figure, use_container_width=True)


def listen_tab(partials: fpt.Partials, x: np.ndarray, y: np.ndarray, fs: int) -> None:
    columns = st.columns(3)
    for column, (label, signal) in zip(columns, [("Original", x), ("Resynthesis", y), ("Residual (noise)", x - y)]):
        column.caption(label)
        column.audio(wav_bytes(signal, fs), format="audio/wav")
    st.subheader("Solo")
    band = st.slider("Frequency band (Hz)", 0, fs // 2, (0, fs // 2), 50)
    loudest = sorted(partials.tracks(), key=lambda t: -t.amplitude_db.max())[:40]
    labels = {f"#{t.id}: {t.mean_frequency_hz:.0f} Hz, {t.amplitude_db.max():.0f} dB, {len(t)} frames": t.id for t in loudest}
    picked = st.multiselect("Partials (40 loudest)", list(labels))
    solo = partials.select(frequency_hz=band, track_ids=[labels[p] for p in picked] if picked else None)
    st.audio(wav_bytes(fpt.synthesize(solo), fs), format="audio/wav")
    st.caption(f"{solo.num_tracks} partials in the solo selection")


def transform_tab(partials: fpt.Partials, fs: int) -> None:
    left, middle, right = st.columns(3)
    semitones = left.slider("Pitch shift (semitones)", -12.0, 12.0, 0.0, 0.5)
    stretch = middle.slider("Time stretch (x)", 0.25, 4.0, 1.0, 0.05)
    gain_db = right.slider("Gain (dB)", -24.0, 12.0, 0.0, 1.0)
    changed = fpt.transforms.gain(partials, gain_db)
    if semitones:
        changed = fpt.transforms.pitch_shift(changed, semitones)
    if stretch != 1.0:
        changed = fpt.transforms.time_stretch(changed, stretch)
    st.audio(wav_bytes(fpt.synthesize(changed, phase="integrated"), fs), format="audio/wav")
    st.plotly_chart(figures.partials_figure(changed, max_frequency_hz=min(8000, fs // 2), height=380),
                    use_container_width=True)


def details_tab(partials: fpt.Partials) -> None:
    st.code("\n".join(describe(partials)), language=None)
    st.plotly_chart(figures.length_histogram(partials), use_container_width=True)
    st.dataframe(partials.to_pandas().head(500), use_container_width=True, hide_index=True)


def export_tab(partials: fpt.Partials, y: np.ndarray, fs: int, name: str, analysis: fpt.AnalysisConfig,
               tracking: fpt.TrackingConfig) -> None:
    stem = Path(name).stem.replace(" ", "_").replace(":", "")
    columns = st.columns(4)
    columns[0].download_button("Partials (.parquet)", file_bytes(partials, ".parquet"), f"{stem}.parquet")
    columns[1].download_button("Partials (.sdif)", file_bytes(partials, ".sdif"), f"{stem}.sdif")
    columns[2].download_button("Resynthesis (.wav)", wav_bytes(y, fs), f"{stem}_resynthesis.wav")
    settings_json = json.dumps({"analysis": analysis.to_dict(), "tracking": tracking.to_dict()}, indent=2)
    columns[3].download_button("Settings (.json)", settings_json, f"{stem}_settings.json")
    st.caption("Settings files work with `fpt analyze --config settings.json`.")


def main() -> None:
    st.set_page_config(page_title="Partial tracking", layout="wide")
    st.title("Fast partial tracking")
    source = load_source()
    if source is None:
        st.info("Choose an audio file in the sidebar.")
        return
    x, fs, name = source
    analysis, tracking, prune = settings(fs)
    partials, y, elapsed = analyze(x, fs, analysis.to_dict(), tracking.to_dict(), prune)

    metrics = st.columns(4)
    metrics[0].metric("R-SNR", f"{fpt.rsnr(x, y):.2f} dB")
    metrics[1].metric("Partials", partials.num_tracks)
    metrics[2].metric("Frames", partials.num_frames)
    metrics[3].metric("Analysis time", f"{elapsed * 1000:.0f} ms")

    tabs = st.tabs(["Partials", "Listen", "Transform", "Details", "Export"])
    with tabs[0]:
        overview_tab(partials, x, y, fs)
    with tabs[1]:
        listen_tab(partials, x, y, fs)
    with tabs[2]:
        transform_tab(partials, fs)
    with tabs[3]:
        details_tab(partials)
    with tabs[4]:
        export_tab(partials, y, fs, name, analysis, tracking)


main()
