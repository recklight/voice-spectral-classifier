"""Resampling, WAV loading, silence trimming and fixed-length segmentation."""

from __future__ import annotations

import numpy as np
import pytest
import speechdsp
from scipy.io import wavfile

from voice_spectral_classifier import (
    SegmentationConfig,
    load_recording,
    resample,
    segment_recording,
    split_into_segments,
)


def _tone(frequency, rate, seconds):
    return np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)


@pytest.mark.parametrize("source_rate", [44_100, 48_000, 22_050])
def test_resampling_keeps_duration_and_frequency(source_rate):
    resampled = resample(_tone(440.0, source_rate, 1.0), source_rate, 16_000)
    assert resampled.shape == (16_000,)
    spectrum = np.abs(np.fft.rfft(resampled * np.hanning(resampled.size)))
    assert np.argmax(spectrum) == pytest.approx(440, abs=1)


def test_equal_rates_return_a_copy():
    signal = np.arange(5.0)
    copied = resample(signal, 16_000, 16_000)
    assert np.array_equal(copied, signal) and copied is not signal


@pytest.mark.parametrize("rate", [0, -8000, 1.5])
def test_resampling_rejects_bad_rates(rate):
    with pytest.raises(ValueError, match="must be a positive integer"):
        resample(np.zeros(10), rate, 16_000)


def test_complete_segments_are_exact_slices(rng):
    signal = rng.standard_normal(3 * 1000)
    segments, starts = split_into_segments(signal, 1000, segment_s=1.0, hop_s=1.0)
    assert segments.shape == (3, 1000)
    assert starts.tolist() == [0, 1000, 2000]
    assert np.array_equal(segments.ravel(), signal)


def test_overlapping_segments_start_every_hop(rng):
    signal = rng.standard_normal(2000)
    segments, starts = split_into_segments(
        signal, 1000, segment_s=1.0, hop_s=0.25, min_fraction=1.0
    )
    assert starts.tolist() == [0, 250, 500, 750, 1000]
    for segment, start in zip(segments, starts, strict=True):
        assert np.array_equal(segment, signal[start : start + 1000])


@pytest.mark.parametrize(
    ("n_samples", "min_fraction", "expected"),
    [(2500, 0.5, 3), (2400, 0.5, 2), (2400, 0.4, 3), (2999, 1.0, 2), (3000, 1.0, 3)],
)
def test_trailing_pieces_follow_min_fraction(n_samples, min_fraction, expected):
    segments, _ = split_into_segments(np.ones(n_samples), 1000, min_fraction=min_fraction)
    assert segments.shape[0] == expected
    if expected * 1000 > n_samples:
        assert np.all(segments[-1, n_samples - (expected - 1) * 1000 :] == 0.0)


def test_a_signal_shorter_than_the_threshold_gives_no_segment():
    segments, starts = split_into_segments(np.ones(300), 1000, min_fraction=0.5)
    assert segments.shape == (0, 1000) and starts.size == 0
    padded, _ = split_into_segments(np.ones(600), 1000, min_fraction=0.5)
    assert padded.shape == (1, 1000)


@pytest.mark.parametrize(
    ("keyword", "value"), [("min_fraction", 0.0), ("min_fraction", 1.2), ("segment_s", 0.0001)]
)
def test_invalid_segmentation_settings_are_rejected(keyword, value):
    with pytest.raises(ValueError):
        split_into_segments(np.ones(100), 1000, **{keyword: value})


def test_silence_trimming_removes_the_quiet_ends(rng):
    rate = 16_000
    quiet = 1e-4 * rng.standard_normal(rate // 2)
    tone = 0.5 * _tone(200.0, rate, 1.7)
    signal = np.concatenate([quiet, tone, quiet])
    trimmed, indices = segment_recording(signal, SegmentationConfig(sample_rate=rate))
    untrimmed, _ = segment_recording(
        signal, SegmentationConfig(sample_rate=rate, trim_silence=False)
    )
    assert trimmed.shape[0] == 2 and indices.tolist() == [0, 1]
    assert untrimmed.shape[0] == 3
    assert np.abs(trimmed[0, :1000]).max() > 0.1


def test_max_segments_spreads_the_selection():
    segmentation_config = SegmentationConfig(sample_rate=1000, trim_silence=False, max_segments=3)
    segments, indices = segment_recording(np.arange(9000.0), segmentation_config)
    assert indices.tolist() == [0, 4, 8]
    assert segments[1, 0] == 4000.0


def test_loading_reads_mono_float_and_resamples(tmp_path):
    path = tmp_path / "tone.wav"
    speechdsp.write_wav(path, 0.5 * _tone(300.0, 48_000, 0.5), 48_000)
    signal = load_recording(path, 16_000)
    assert signal.dtype == np.float64 and signal.shape == (8000,)
    assert np.abs(signal).max() == pytest.approx(0.5, abs=0.01)


def test_loading_reports_missing_and_foreign_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path / "absent.wav", 16_000)
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"this is not audio")
    with pytest.raises(ValueError, match=r"cannot read .*fake\.wav as a WAV file"):
        load_recording(fake, 16_000)


def test_loading_rejects_a_float_file_with_a_nan_sample(tmp_path):
    samples = (0.5 * _tone(300.0, 16_000, 1.0)).astype(np.float32)
    samples[8000] = np.nan
    path = tmp_path / "broken.wav"
    wavfile.write(path, 16_000, samples)
    with pytest.raises(ValueError, match=r"broken\.wav holds NaN or infinite samples"):
        load_recording(path, 16_000)
