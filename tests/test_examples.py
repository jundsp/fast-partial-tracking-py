"""Runs every example script end to end (short settings) so the gallery stays working."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = {
    "benchmark.py": ["--seconds", "1", "--workers", "2"],
    "streaming_demo.py": ["--seconds", "2"],
    "live_microphone.py": ["--simulate", "--seconds", "1"],
}


@pytest.mark.examples
@pytest.mark.parametrize("script", sorted(p.name for p in EXAMPLES.glob("*.py")))
def test_example_runs(script: str, tmp_path: Path) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    command = [sys.executable, str(EXAMPLES / script), "--out", str(tmp_path), *ARGS.get(script, [])]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert any(tmp_path.rglob("*")), "example wrote no output"
