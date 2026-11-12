from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from fast_partial_tracking import AnalysisConfig, TrackingConfig

FIXTURES = Path(__file__).parent / "fixtures" / "matlab"
REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_BIN = Path(os.environ.get("MATLAB_BIN", "/Applications/MATLAB_R2026b.app/bin/matlab"))
MATLAB_REPO = Path(os.environ.get("FPT_MATLAB_DIR", REPO_ROOT.parent / "fast-partial-tracking"))
CASES = ["demo", "sine", "chirp", "chirp_params", "vibrato", "crossing", "bursts", "noise"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--matlab",
        action="store_true",
        default=False,
        help="regenerate tests/fixtures/matlab with a local MATLAB before running",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--matlab"):
        regenerate_fixtures()


def regenerate_fixtures() -> None:
    """Runs matlab/export_reference.m to rebuild the reference data."""
    matlab = MATLAB_BIN if MATLAB_BIN.exists() else shutil.which("matlab")
    if matlab is None:
        raise pytest.UsageError("--matlab given but no MATLAB found (set MATLAB_BIN)")
    subprocess.run(
        [str(matlab), "-nodisplay", "-batch", f"export_reference('{MATLAB_REPO}', '{FIXTURES}')"],
        cwd=REPO_ROOT / "matlab",
        check=True,
    )


@dataclass
class MatlabCase:
    """A reference case exported by matlab/export_reference.m."""

    name: str
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def x(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.data["x"], dtype=float))

    @property
    def fs(self) -> float:
        return float(self.data["fs"])

    @property
    def analysis(self) -> AnalysisConfig:
        n, hop_factor, oversample, threshold, order, *_ = self.data["params"]
        return AnalysisConfig.matlab(int(n), int(hop_factor), int(oversample), float(threshold), int(order))

    @property
    def tracking(self) -> TrackingConfig:
        *_, delta, zeta_f, zeta_a = self.data["params"]
        return TrackingConfig.matlab(float(delta), float(zeta_f), float(zeta_a))

    @property
    def partials(self) -> np.ndarray:
        """Rows: frame, time, omega, log-amp, phase, track (1-based MATLAB indices), row."""
        return np.atleast_2d(self.data["partials"])

    def ddm_frames(self) -> list[dict[str, Any]]:
        return [mat_struct(s) for s in np.atleast_1d(self.data["ddm"])]

    def cost_frames(self) -> list[dict[str, Any]]:
        return [mat_struct(s) for s in np.atleast_1d(self.data["cost"])]


def mat_struct(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return {name: _unwrap(value[name]) for name in value.dtype.names}


def _unwrap(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def load_mat(name: str) -> dict[str, Any]:
    return sio.loadmat(FIXTURES / f"{name}.mat", squeeze_me=True)


@pytest.fixture(scope="session", params=CASES)
def matlab_case(request: pytest.FixtureRequest) -> MatlabCase:
    return MatlabCase(request.param, load_mat(request.param))


@pytest.fixture(scope="session")
def demo_case() -> MatlabCase:
    return MatlabCase("demo", load_mat("demo"))


def harmonic_signal(fs: float = 16000.0, duration: float = 0.5, f0: float = 220.0, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    return sum((0.4 / h) * np.sin(2 * np.pi * h * f0 * t + h) for h in range(1, harmonics + 1))
