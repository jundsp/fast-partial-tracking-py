from __future__ import annotations

import numpy as np
import pytest

import fast_partial_tracking as fpt


@pytest.mark.parametrize("name", list(fpt.data.EXAMPLES))
def test_examples_load_as_mono_44k(name: str) -> None:
    x, fs = fpt.data.example(name)
    assert fs == 44100 and x.ndim == 1 and x.size > fs // 4
    assert 0.0 < np.abs(x).max() <= 1.0


def test_example_credits_list_every_file() -> None:
    credits = (fpt.data.example_path("kara").parent / "README.md").read_text()
    for filename, _ in fpt.data.EXAMPLES.values():
        assert f"`{filename}`" in credits


def test_kara_is_the_demo_sound() -> None:
    assert fpt.data.example_path("kara") == fpt.data.demo_sound_path()


def test_unknown_example_raises() -> None:
    with pytest.raises(ValueError, match="unknown example"):
        fpt.data.example("nope")


def test_recordings_track() -> None:
    x, fs = fpt.data.example("hulusi")
    partials = fpt.track(x[: fs * 2], fs)
    assert partials.frequency_hz.size > 0
