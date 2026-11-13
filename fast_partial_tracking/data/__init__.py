"""Example audio: bundled recordings and synthetic test signals.

The recordings and their licenses are listed in ``data/README.md``.

Example:
    >>> from fast_partial_tracking import data
    >>> x, fs = data.demo_sound()
    >>> x, fs = data.example("hulusi")
    >>> x, fs = data.synthetic("vibrato")
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf

SYNTHETIC = ("harmonic", "chirp", "vibrato", "crossing", "chord")

#: Bundled recordings: name -> (file, short description). All are mono, 44.1 kHz.
EXAMPLES: dict[str, tuple[str, str]] = {
    "kara": ("demo_sound.wav", "male voice saying /kara/ (0.5 s)"),
    "speech": ("speech.wav", "female voice, spoken English (5.7 s)"),
    "hulusi": ("hulusi.wav", "hulusi (gourd flute) playing a melody (15 s)"),
    "orchestra": ("orchestra.wav", "orchestra, opening of Mozart's Figaro overture (12 s)"),
}


def example_path(name: str) -> Path:
    """Path of a bundled recording; ``name`` is one of ``EXAMPLES``."""
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; choose from {tuple(EXAMPLES)}")
    return Path(str(files(__package__).joinpath(EXAMPLES[name][0])))


def example(name: str) -> tuple[np.ndarray, int]:
    """A bundled recording; ``name`` is one of ``EXAMPLES``.

    Returns:
        (mono samples, sampling rate).
    """
    x, fs = sf.read(example_path(name))
    return x, int(fs)


def demo_sound_path() -> Path:
    """Path of the bundled demo recording (a male voice saying /kara/, 44.1 kHz)."""
    return example_path("kara")


def demo_sound() -> tuple[np.ndarray, int]:
    """The bundled demo recording used by the MATLAB ``jun_demo.m``.

    Returns:
        (mono samples, sampling rate).
    """
    x, fs = sf.read(demo_sound_path())
    return x, int(fs)


def synthetic(name: str, fs: int = 16000, duration: float = 1.0, noise_db: float = -60.0, seed: int = 0) -> tuple[np.ndarray, int]:
    """Synthetic test signals with known partials.

    * ``"harmonic"``: 220 Hz tone with 8 harmonics, 1/h amplitudes.
    * ``"chirp"``: harmonic exponential chirp from 200 to 2000 Hz.
    * ``"vibrato"``: 330 Hz harmonic tone with ±1 semitone vibrato at 6 Hz.
    * ``"crossing"``: two linear chirps whose frequencies cross.
    * ``"chord"``: three harmonic notes (C4, E4, G4) entering one after another.

    Args:
        name: One of ``SYNTHETIC``.
        fs: Sampling rate in Hz.
        duration: Length in seconds.
        noise_db: Level of added white noise relative to full scale (``-inf`` for none).
        seed: Noise seed.

    Returns:
        (samples, sampling rate).
    """
    t = np.arange(int(round(duration * fs))) / fs
    builders = {
        "harmonic": lambda: _harmonics(np.full_like(t, 220.0), fs, 8),
        "chirp": lambda: _harmonics(200.0 * 10.0 ** (t / duration), fs, 5),
        "vibrato": lambda: _harmonics(330.0 * 2.0 ** (np.sin(2 * np.pi * 6.0 * t) / 12.0), fs, 8),
        "crossing": lambda: 0.4 * np.sin(2 * np.pi * np.cumsum(500 + 2000 * t / duration) / fs)
        + 0.3 * np.sin(2 * np.pi * np.cumsum(2500 - 2000 * t / duration) / fs + 1.0),
        "chord": lambda: sum(
            (t >= onset) * _harmonics(np.full_like(t, f0), fs, 6) / 3.0
            for f0, onset in [(261.63, 0.0), (329.63, duration / 4), (392.0, duration / 2)]
        ),
    }
    if name not in builders:
        raise ValueError(f"unknown signal {name!r}; choose from {SYNTHETIC}")
    x = builders[name]()
    if np.isfinite(noise_db):
        x = x + 10.0 ** (noise_db / 20.0) * np.random.default_rng(seed).standard_normal(t.size)
    return x, fs


def _harmonics(f0: np.ndarray, fs: int, count: int) -> np.ndarray:
    """Sum of harmonics h*f0(t), h = 1..count, with amplitudes 0.4/h, below Nyquist."""
    phase = 2.0 * np.pi * np.cumsum(f0) / fs
    return sum((0.4 / h) * np.sin(h * phase + h) * (h * f0 < fs / 2) for h in range(1, count + 1))
