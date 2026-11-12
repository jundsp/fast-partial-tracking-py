# fast-partial-tracking

Python implementation of **fast partial tracking**: J. Neri and P. Depalle, [*Fast Partial Tracking of Audio with Real-Time Capability through Linear Programming*](https://www.dafx.de/paper-archive/2018/papers/DAFx2018_paper_26.pdf), DAFx-18.

Sinusoidal parameters are estimated per frame with the Distribution Derivative Method,
peaks are linked across frames by solving a linear assignment problem, and the result
can be resynthesized or exported. It produces the same output as the original
[MATLAB code](https://github.com/jundsp/Fast-Partial-Tracking), checked by tests against
reference data exported from MATLAB.

## Install

```sh
pip install fast-partial-tracking            # library + `fpt` command
pip install "fast-partial-tracking[plot]"    # + matplotlib for the examples
```

## Quick start

```python
import soundfile as sf
import fast_partial_tracking as fpt

x, fs = sf.read("examples/audio/demo_sound.wav")
analysis = fpt.AnalysisConfig(window_length=2047, oversample=2, peak_threshold_db=-50, order=2)
tracking = fpt.TrackingConfig(delta=0.2, zeta_f=50, zeta_a=15)

partials = fpt.track(x, fs, analysis, tracking)
y = fpt.synthesize(partials)
print(f"R-SNR = {fpt.rsnr(x, y):.2f} dB")        # 22.34 dB, same as MATLAB jun_demo

partials.save("demo.parquet")                     # or .arrows / .sdif / .bin
partials.frequency_hz, partials.amplitude, partials.track_id   # flat columns
```

### Streaming / real time

Linking only needs frames k-1 and k (Section 3.7 of the paper), so audio can be processed
in blocks of any size with bounded memory. Each frame is emitted one frame after it is
analyzed:

```python
tracker = fpt.PartialTracker(fs, analysis, tracking)
synth = fpt.Synthesizer(tracker.grid)
with fpt.io.open_writer("live.arrows", tracker.grid, order=2) as writer:
    for block in audio_blocks:
        for frame in tracker.push(block):
            writer.write_frame(frame)
            play(synth.push(frame))
    for frame in tracker.flush():
        writer.write_frame(frame)
    writer.set_summary(tracker.grid, tracker.num_tracks)
```

`fpt.track` uses the loudest frame of the whole signal as the peak-threshold reference,
like MATLAB. A live `PartialTracker` can't look ahead, so by default its threshold is
relative to a full-scale sinusoid (or pass `threshold_db`).

## File formats

All formats use the same table, one row per partial per frame, and are written
incrementally (frame by frame), so long or live analyses never need to fit in memory.

| extension  | format              | notes |
|------------|---------------------|-------|
| `.parquet` | Apache Parquet      | Recommended. Columnar and zstd-compressed, one row group per block of frames. Reads directly into pandas, polars, DuckDB and MATLAB (`parquetread`). |
| `.arrows`  | Arrow IPC stream    | For live pipelines: readable while it is being written (`fpt.io.iter_arrow_stream`). |
| `.sdif`    | SDIF `1TRC`, binary | For SDIF tools (AudioSculpt, OpenMusic). Written directly, so `tosdif` isn't needed. |
| `.bin`     | MATLAB demo format  | Byte-compatible with `jun_write_partials.m` / `jun_read_partials.m`. |

Columns: `frame`, `time_s`, `track_id`, `frequency_hz`, `amplitude` (linear amplitude of
the real sinusoid), `phase`, and optionally `alpha_re`/`alpha_im`. The last two hold the
Q + 1 complex coefficients of the short-term model (eq. 4), which give exact frequency
and amplitude slopes. Analysis settings and the frame grid are stored as JSON metadata.

```python
import polars as pl
df = pl.read_parquet("demo.parquet")
df.group_by("track_id").agg(pl.col("frequency_hz").mean(), pl.len())
```

## Command line

```sh
fpt analyze input.wav -o partials.parquet --report      # R-SNR of the resynthesis
fpt analyze long.wav -o partials.arrows --stream        # block by block, bounded memory
fpt convert partials.parquet -o partials.sdif
fpt synth partials.parquet -o resynthesis.wav
fpt info partials.parquet
```

## Where the paper lives in the code

| paper | module |
|---|---|
| Eq. 1–2, sinusoidal model | `partials.py` (`amplitude = 2·exp(a)`), `synthesis.py` |
| Eq. 3, assignment problem; §2.2 Hungarian algorithm | `assignment.py` (`scipy.optimize.linear_sum_assignment`) |
| Eq. 4–7, short-term model; DDM estimation [Betser 2009] | `model.py`, `ddm.py`, `windows.py`, `peaks.py` |
| Eq. 8–11, useful/spurious costs at the frame midpoint | `cost.py` |
| Eq. 12–13, σ from ζ and δ | `cost.py::cost_variances` |
| Eq. 14, cost matrix; §3.5 useful assignments | `cost.py`, `assignment.py` |
| §3.6, partial labeling | `labeling.py` |
| §3.7, frame-by-frame (real-time) operation | `tracker.py::PartialTracker` |
| Eq. 18, R-SNR; McAulay–Quatieri synthesis [12] | `metrics.py`, `synthesis.py` |

## MATLAB equivalence

`tests/fixtures/matlab/` holds reference outputs exported by
[`matlab/export_reference.m`](matlab/export_reference.m) from the MATLAB repository:

- **Signals:** the demo sound, a sine, harmonic chirps, vibrato, crossing chirps, tone bursts and noise.
- **Settings:** Hann and Blackman-Harris windows, Q = 1–3, and odd and even window lengths.

The tests check that the frame grids, windows, spectra, peaks, DDM coefficients, cost
matrices, assignments, track labels, partial parameters (to about 1e-12) and
resynthesized signals (to about 1e-14) all agree.

```sh
uv sync --extra dev
uv run pytest                    # uses the committed references
uv run pytest --matlab           # regenerates them first (needs MATLAB; set FPT_MATLAB_DIR)
uv run python examples/matlab_comparison_report.py   # plots + CSV per case
```

The SDIF output was also verified with IRCAM's SDIF library through `pysdif3` 1.0.0.
`pysdif3` 1.0.1 cannot open files, so that test skips unless 1.0.0 is installed with
numpy < 2.

## Examples

`examples/track_demo.py` (the `jun_demo.m` port), `streaming_demo.py`, `sdif_export.py`
and `matlab_comparison_report.py`. Outputs go to
`~/Documents/output/fast-partial-tracking/examples/<name>/` (override with `--out`).

## Citation

```bibtex
@inproceedings{neri2018fast,
  author    = {Neri, Julian and Depalle, Philippe},
  title     = {Fast Partial Tracking of Audio with Real-Time Capability through Linear Programming},
  booktitle = {Proceedings of the 21st International Conference on Digital Audio Effects (DAFx-18)},
  address   = {Aveiro, Portugal},
  pages     = {326--333},
  year      = {2018}
}
```

## License

MIT. See [LICENSE](LICENSE).
