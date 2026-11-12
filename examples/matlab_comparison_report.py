"""Compares the Python port with the MATLAB reference outputs, case by case.

Reads the references in ``tests/fixtures/matlab`` (regenerate with
``matlab -batch export_reference`` in ``matlab/``), runs the Python tracker on the same
inputs and settings, and reports structural agreement, parameter errors and R-SNR. It
plots the partials of both implementations and the resynthesis difference per case.

Run:
    python examples/matlab_comparison_report.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import fast_partial_tracking as fpt

log = logging.getLogger("fast_partial_tracking.examples.matlab_comparison_report")

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "matlab"
DEFAULT_OUT = Path.home() / "Documents" / "output" / "fast-partial-tracking" / "examples" / "matlab_comparison_report"


def run_case(ref: dict[str, Any]) -> tuple[fpt.Partials, np.ndarray, dict[str, Any]]:
    """Tracks one reference input with the reference settings and compares the results."""
    n, hop_factor, oversample, threshold, order, delta, zeta_f, zeta_a = ref["params"]
    analysis = fpt.AnalysisConfig.matlab(int(n), int(hop_factor), int(oversample), float(threshold), int(order))
    tracking = fpt.TrackingConfig.matlab(float(delta), float(zeta_f), float(zeta_a))
    x = np.atleast_1d(ref["x"]).astype(float)
    partials = fpt.track(x, float(ref["fs"]), analysis, tracking)
    y = fpt.synthesize(partials)
    rows = np.atleast_2d(ref["partials"])
    same = (
        len(partials) == rows.shape[0]
        and np.array_equal(partials.frame + 1, rows[:, 0])
        and np.array_equal(partials.track_id + 1, rows[:, 5])
    )
    summary = {
        "case": str(ref["name"]),
        "partials_matlab": int(ref["num_tracks"]),
        "partials_python": partials.num_tracks,
        "identical_structure": same,
        "max_omega_error": float(np.max(np.abs(partials.omega - rows[:, 2]))) if same else np.nan,
        "max_log_amp_error": float(np.max(np.abs(partials.log_amp - rows[:, 3]))) if same else np.nan,
        "max_output_error": float(np.max(np.abs(y - ref["y"]))),
        "rsnr_matlab": float(ref["rsnr"]),
        "rsnr_python": fpt.rsnr(x, y),
    }
    return partials, y, summary


def plot_case(ref: dict[str, Any], partials: fpt.Partials, y: np.ndarray, out_path: Path) -> None:
    """Overlays MATLAB and Python partials and plots the resynthesis difference.

    Args:
        ref: MATLAB reference case.
        partials: Python result.
        y: Python resynthesis.
        out_path: Figure destination.
    """
    rows = np.atleast_2d(ref["partials"])
    fs = float(ref["fs"])
    pad_front = int(ref["padding"][0])
    fig, (ax_tf, ax_err) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    t_ref = (rows[:, 1] - 1 - pad_front) / fs
    ax_tf.scatter(t_ref, rows[:, 2] * fs / (2 * np.pi), s=14, facecolors="none", edgecolors="k", lw=0.5, label="MATLAB")
    ax_tf.scatter(partials.time_s, partials.frequency_hz, s=3, c="tab:red", label="Python")
    ax_tf.set_ylabel("frequency (Hz)")
    ax_tf.legend(loc="upper right", fontsize=8)
    ax_tf.set_title(str(ref["name"]))
    t = np.arange(y.size) / fs
    ax_err.plot(t, y - ref["y"], color="tab:blue", lw=0.6)
    ax_err.set_ylabel("y_py - y_matlab")
    ax_err.set_xlabel("time (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summaries = []
    for path in sorted(FIXTURES.glob("*.mat")):
        ref = sio.loadmat(path, squeeze_me=True)
        if "partials" not in ref:
            continue
        partials, y, summary = run_case(ref)
        summaries.append(summary)
        plot_case(ref, partials, y, args.out / "figures" / f"{summary['case']}.png")
        log.info(
            f"{summary['case']:14s} identical={summary['identical_structure']!s:5s} "
            f"omega err={summary['max_omega_error']:.1e} y err={summary['max_output_error']:.1e} "
            f"R-SNR {summary['rsnr_python']:.4f} / {summary['rsnr_matlab']:.4f} dB"
        )

    table = args.out / "tables" / "comparison.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    log.info(f"table -> {table}; figures -> {args.out / 'figures'}")


if __name__ == "__main__":
    main()
