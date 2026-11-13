# Changelog

## Unreleased

### Added
- **Example recordings**: `fpt.data.example(name)` and `example_path(name)` load bundled
  recordings: `"speech"` (15 s of spoken English), `"hulusi"` (a flute melody, 15 s) and
  `"orchestra"` (12 s of Mozart), next to the `"kara"` demo voice. All three are CC0; their
  sources are listed in `fast_partial_tracking/data/README.md`. The web app offers them as sources.

## 0.2.0

A redesign for Python: explicit settings, a partials toolkit, transforms, plotting, a web
app and about 5× faster tracking. Exact MATLAB results remain available through
`AnalysisConfig.matlab(...)` / `TrackingConfig.matlab(...)`.

### Added
- **Settings**: explicit, validated settings: `window`, `hop_size`, `n_fft`, `threshold_reference`, `ddm_bins`, `geometry` and `amplitude_guard`. Also `AnalysisConfig.from_duration(fs, window_ms=...)`, plus the `matlab()` presets.
- **Partials toolkit**: `tracks()` and `track(id)` returning `Track` objects; `prune`, `select`, `filter_tracks`, `subset`, `stats`; `to_polars` and `to_pandas`; and a readable `repr`.
- **Transforms**: `fpt.transforms.pitch_shift`, `time_stretch` and `gain`.
- **Synthesis**: phase-integrating synthesis via `PhaseIntegratingSynthesizer` and `synthesize(phase="integrated")`. Transformed results use it automatically.
- **Streaming**: `PartialTracker.process(blocks)`.
- **Plotting**: `fpt.plot` in the style of the original MATLAB figures: each partial in its own saturated color over a light, smooth spectrogram, with serif (Computer Modern) type. `figure` (single publication panel), `overview`, `partials`, `spectrogram`, the `style()` / `use_style()` helpers, and `Partials.plot()`. The web app and `fpt plot` use the same look.
- **Example audio**: `fpt.data` provides the bundled demo voice and synthetic test signals.
- **Web app**: `fpt ui`, an interactive Streamlit app.
- **Command line**: `fpt plot`, `fpt ui`, a richer `fpt info`, and `analyze --preset/--config/--prune`. Streaming file analysis now uses the signal-relative threshold.
- **Examples gallery**: 11 runnable examples, all exercised by the test suite.
- **Dependencies**: numexpr (DDM speed-up) and matplotlib (plotting) install by default. Extras `ui`, `dataframes`, `live`, and `all` for everything. `dev` includes every optional feature, so `uv sync --extra dev` is the one setup command for a clone.
- **Version**: `fast_partial_tracking.__version__` and `fpt --version`.

### Changed
- **Package layout**: organized by processing stage into `analysis/`, `tracking/`, `synthesis/` and `compat/`.
- **Default frame placement**: centered. Frame k sits on input sample k·H and the model origin is the window centre (MATLAB placement: `geometry="matlab"`).
- **Amplitude check**: `amplitude_guard="center"`. It rejects estimates whose amplitude at the frame centre exceeds the largest sample, which keeps Q = 3 models stable. The MATLAB code was fixed the same way, so the MATLAB preset uses it too; `amplitude_guard="legacy"` reproduces the old check, which compared a log value with a linear bound and almost never triggered.
- **File format version 2**: the grid records its geometry. Version 1 files are still read.
- **Speed**: about 5× faster tracking, from a fused exponential in the α₀ estimate, one batched FFT per frame and frame-parallel analysis in `track(workers=...)`.

### Migration from 0.1

| 0.1 | 0.2 |
|---|---|
| `AnalysisConfig(window_length, hop_factor, oversample, peak_threshold_db, order)` | `AnalysisConfig.matlab(window_length, hop_factor, oversample, peak_threshold_db, order)` for identical results, or the explicit fields below |
| `order` | `poly_order` |
| `peak_threshold_db` (sign ignored) | `threshold_db` (must be ≤ 0) |
| `oversample` | `n_fft` (and `window`, no longer implied) |
| `hop_factor` | `hop_size` |
| `TrackingConfig(zeta_f=..., zeta_a=...)` | `TrackingConfig(zeta_f_hz=..., zeta_a_db=...)` |
| `PartialTracker(..., threshold_db=abs_level)` | `PartialTracker(..., reference_db=level)` plus `threshold_reference` |
| `Frame.log_amp`, `Partials.log_amp` | `.log_amplitude` |
| `Frame.frequency_hz(fs)` | `Frame.frequency_hz` (property) |
| `Partials.track(id)` returned row indices | `Partials.track(id)` returns a `Track` |
| `fast_partial_tracking.tracker`, `.ddm`, `.cost`, … | `fast_partial_tracking.tracking.tracker`, `.analysis.ddm`, `.tracking.cost`, … |
| `fast_partial_tracking.io.matlab_bin` | `fast_partial_tracking.compat.matlab_bin` |
| `examples/audio/demo_sound.wav` | `fpt.data.demo_sound()` / `fpt.data.demo_sound_path()` |

## 0.1.0

First release: a Python port of the MATLAB implementation, equivalent to it on eight reference signals. Includes streaming tracking and synthesis; Parquet, Arrow, SDIF and MATLAB `.bin` I/O; the `fpt` command; and examples.
