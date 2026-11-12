"""How the tracking parameters affect quality: R-SNR and partial count over δ × ζ_f and Q.

Also compares the default (centered) frame geometry with the MATLAB preset on every
synthetic signal. Writes heatmaps and a CSV table.

Run:
    python examples/parameter_sweep.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.parameter_sweep")

DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "parameter_sweep"
DELTAS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
ZETAS_F = [10, 25, 50, 100, 200]


def plot_heatmaps(rsnr: np.ndarray, counts: np.ndarray, out_path: Path) -> None:
    """R-SNR and partial-count heatmaps over δ (rows) and ζ_f (columns).

    Args:
        rsnr: R-SNR in dB, shape (len(DELTAS), len(ZETAS_F)).
        counts: Partial counts, same shape.
        out_path: Figure destination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title, fmt in [(axes[0], rsnr, "R-SNR (dB)", "{:.1f}"), (axes[1], counts, "partials", "{:.0f}")]:
        image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ZETAS_F)), [str(z) for z in ZETAS_F])
        ax.set_yticks(range(len(DELTAS)), [str(d) for d in DELTAS])
        ax.set_xlabel("ζ_f (Hz)")
        ax.set_ylabel("δ")
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, fmt.format(value), ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"figure -> {out_path}")


def sweep_delta_zeta(x: np.ndarray, fs: float, analysis: fpt.AnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    """R-SNR and partial count for every (δ, ζ_f) pair."""
    rsnr = np.zeros((len(DELTAS), len(ZETAS_F)))
    counts = np.zeros_like(rsnr)
    for (i, delta), (j, zeta) in ((a, b) for a in enumerate(DELTAS) for b in enumerate(ZETAS_F)):
        partials = fpt.track(x, fs, analysis, fpt.TrackingConfig(delta=delta, zeta_f_hz=zeta))
        rsnr[i, j] = fpt.rsnr(x, fpt.synthesize(partials))
        counts[i, j] = partials.num_tracks
    return rsnr, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fpt.plot.use_style()

    x, fs = fpt.data.synthetic("vibrato", duration=1.0, noise_db=-40)
    analysis = fpt.AnalysisConfig(window_length=1023)
    rsnr, counts = sweep_delta_zeta(x, fs, analysis)
    plot_heatmaps(rsnr, counts, args.out / "figures" / "delta_zeta.png")

    rows = []
    for name in fpt.data.SYNTHETIC:
        x, fs = fpt.data.synthetic(name, duration=1.0, noise_db=-50)
        for order in (1, 2, 3):
            matlab = fpt.AnalysisConfig.matlab(1023, hop_factor=4, oversample=2, peak_threshold_db=-60, order=order)
            row = {"signal": name, "Q": order}
            for label, config in [("matlab", matlab), ("centered", replace(matlab, geometry="centered"))]:
                partials = fpt.track(x, fs, config)
                row[f"rsnr_{label}"] = round(fpt.rsnr(x, fpt.synthesize(partials)), 2)
                row[f"partials_{label}"] = partials.num_tracks
            rows.append(row)
            log.info(f"{name:9s} Q={order}: matlab {row['rsnr_matlab']:6.2f} dB, centered {row['rsnr_centered']:6.2f} dB")
    table = args.out / "tables" / "geometry_and_order.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"table -> {table}")


if __name__ == "__main__":
    main()
