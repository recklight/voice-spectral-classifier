"""Synthetic dataset: reproducibility, on-disk layout and the designed voice qualities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert

from voice_spectral_classifier import (
    SYNTHETIC_CLASSES,
    SyntheticConfig,
    make_sustained_vowel,
    write_synthetic_dataset,
)

RATE = 16_000
SMALL = SyntheticConfig(n_speakers_per_class=(2, 2, 3), n_recordings_per_speaker=2, duration_s=0.6)


def _all_bytes(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_the_same_seed_writes_identical_files(tmp_path):
    write_synthetic_dataset(tmp_path / "one", SMALL, rng=3)
    write_synthetic_dataset(tmp_path / "two", SMALL, rng=3)
    assert _all_bytes(tmp_path / "one") == _all_bytes(tmp_path / "two")


def test_another_seed_writes_other_recordings(tmp_path):
    write_synthetic_dataset(tmp_path / "one", SMALL, rng=3)
    write_synthetic_dataset(tmp_path / "two", SMALL, rng=4)
    first, second = _all_bytes(tmp_path / "one"), _all_bytes(tmp_path / "two")
    assert first.keys() == second.keys()
    assert first["modal/modal_000.wav"] != second["modal/modal_000.wav"]


def test_layout_naming_and_groups(tmp_path):
    index = write_synthetic_dataset(tmp_path, SMALL, rng=0)
    table = pd.read_csv(index)
    assert index == tmp_path / "dataset.csv"
    assert list(table.columns) == ["path", "label", "group"]
    assert table["label"].value_counts().to_dict() == {"rough": 6, "breathy": 4, "modal": 4}
    assert table["path"].iloc[:4].tolist() == [
        "breathy/breathy_000.wav",
        "breathy/breathy_001.wav",
        "breathy/breathy_002.wav",
        "breathy/breathy_003.wav",
    ]
    assert table["group"].nunique() == 7
    assert table.groupby("group").size().eq(2).all()
    assert table.groupby("group")["label"].nunique().eq(1).all()


def test_recordings_alternate_between_the_configured_rates(tmp_path):
    index = write_synthetic_dataset(tmp_path, SMALL, rng=0)
    rates = [wavfile.read(tmp_path / path)[0] for path in pd.read_csv(index)["path"]]
    assert set(rates) == set(SMALL.sample_rates)
    assert rates[:2] == list(SMALL.sample_rates)
    expected = round(SMALL.duration_s * rates[0]) + 2 * round(SMALL.silence_s * rates[0])
    assert wavfile.read(tmp_path / "breathy" / "breathy_000.wav")[1].shape == (expected,)


def _band_share(signal, low, high):
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, 1 / RATE)
    return spectrum[(frequencies >= low) & (frequencies < high)].sum() / spectrum.sum()


def test_breathy_voices_carry_more_high_frequency_energy():
    breathy = make_sustained_vowel("breathy", RATE, 1.0, rng=1)
    modal = make_sustained_vowel("modal", RATE, 1.0, rng=1)
    assert _band_share(breathy, 3000, 8000) > 5 * _band_share(modal, 3000, 8000)


def test_rough_voices_tremble_at_the_tremor_rate():
    def modulation_at(signal, rate_hz):
        envelope = np.abs(hilbert(signal))
        spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
        frequencies = np.fft.rfftfreq(envelope.size, 1 / RATE)
        return spectrum[np.argmin(np.abs(frequencies - rate_hz))] / envelope.mean()

    rough = make_sustained_vowel("rough", RATE, 2.0, tremor_rate_hz=5.0, rng=2)
    modal = make_sustained_vowel("modal", RATE, 2.0, tremor_rate_hz=5.0, rng=2)
    assert modulation_at(rough, 5.0) > 10 * modulation_at(modal, 5.0)


def test_rough_voices_have_subharmonic_energy():
    def energy_near(signal, frequency):
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
        frequencies = np.fft.rfftfreq(signal.size, 1 / RATE)
        return spectrum[np.abs(frequencies - frequency) < 10].sum()

    rough = make_sustained_vowel("rough", RATE, 2.0, f0_hz=200.0, rng=5)
    modal = make_sustained_vowel("modal", RATE, 2.0, f0_hz=200.0, rng=5)
    ratio_rough = energy_near(rough, 100.0) / energy_near(rough, 200.0)
    ratio_modal = energy_near(modal, 100.0) / energy_near(modal, 200.0)
    assert ratio_rough > 100 * ratio_modal


def test_silence_padding_is_quiet_and_the_peak_is_fixed():
    signal = make_sustained_vowel("modal", RATE, 0.5, silence_s=0.25, rng=0)
    assert signal.shape == (16_000,)
    assert np.abs(signal).max() == pytest.approx(0.8)
    assert np.abs(signal[:3000]).max() < 0.01


def test_every_listed_quality_can_be_synthesized():
    for quality in SYNTHETIC_CLASSES:
        assert np.isfinite(make_sustained_vowel(quality, RATE, 0.2, rng=0)).all()
    with pytest.raises(ValueError, match="unknown voice quality"):
        make_sustained_vowel("creaky", RATE, 0.2)


def test_severity_moves_a_quality_toward_the_modal_voice():
    def subharmonic_ratio(signal):
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
        frequencies = np.fft.rfftfreq(signal.size, 1 / RATE)
        near = [spectrum[np.abs(frequencies - f) < 10].sum() for f in (100.0, 200.0)]
        return near[0] / near[1]

    ratios = [
        subharmonic_ratio(make_sustained_vowel("rough", RATE, 2.0, f0_hz=200.0, severity=s, rng=5))
        for s in (0.0, 0.5, 1.0)
    ]
    assert ratios[0] < 1e-3 < ratios[1] < ratios[2]
    with pytest.raises(ValueError, match="severity"):
        make_sustained_vowel("rough", RATE, 0.2, severity=1.5)


def test_modal_speakers_have_no_severity(tmp_path):
    synthetic_config = SyntheticConfig(
        class_names=("modal", "rough"),
        n_speakers_per_class=(2, 2),
        severity_range=(1.0, 1.0),
        duration_s=0.5,
    )
    index = pd.read_csv(write_synthetic_dataset(tmp_path, synthetic_config, rng=0))
    assert index["label"].tolist() == ["modal"] * 4 + ["rough"] * 4
