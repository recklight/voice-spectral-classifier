"""Power spectra, band pooling and the 8-bit spectrogram images."""

from __future__ import annotations

import numpy as np
import pytest
import speechdsp

from voice_spectral_classifier import (
    SpectrogramConfig,
    band_weights,
    power_spectrogram,
    spectrogram_image,
    spectrogram_images,
)
from voice_spectral_classifier.spectrogram import (
    analysis_sizes,
    band_frequencies,
    fit_frames,
    quantize,
    to_decibels,
)

RATE = 16_000


def _tone(frequency, seconds=1.0, rate=RATE):
    return np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)


def test_one_second_gives_98_frames_of_25_ms():
    assert analysis_sizes(RATE, 25.0, 10.0) == (400, 160, 512)
    power, frequencies = power_spectrogram(_tone(1000.0), RATE)
    assert power.shape == (98, 257)
    assert frequencies[-1] == 8000.0


def test_power_spectral_density_integrates_to_the_variance(rng):
    noise = 0.3 * rng.standard_normal(4 * RATE)
    power, frequencies = power_spectrogram(noise, RATE, window="hann")
    variance = power.mean(axis=0).sum() * (frequencies[1] - frequencies[0])
    assert variance == pytest.approx(0.09, rel=0.05)


@pytest.mark.parametrize("frequency", [300.0, 1100.0, 2500.0, 4100.0, 7300.0])
def test_a_tone_lands_in_its_linear_band(frequency):
    image = spectrogram_image(_tone(frequency), RATE)
    assert image.shape == (40, 98) and image.dtype == np.uint8
    assert int(image.max()) == 255
    assert int(image.mean(axis=1).argmax()) == int(frequency // 200)


def test_row_zero_is_the_lowest_band_for_both_scales():
    for scale in ("linear", "mel"):
        spectrogram_config = SpectrogramConfig(scale=scale)
        low = spectrogram_image(_tone(150.0), RATE, spectrogram_config).mean(axis=1).argmax()
        high = spectrogram_image(_tone(7100.0), RATE, spectrogram_config).mean(axis=1).argmax()
        assert low < 5 and high > 34, scale


def test_mel_bands_widen_with_frequency():
    centers = band_frequencies("mel", 40, 8000.0)
    widths = np.diff(centers)
    assert np.all(widths > 0) and widths[-1] > 5 * widths[0]
    linear = band_frequencies("linear", 40, 8000.0)
    assert np.allclose(np.diff(linear), 200.0)


def test_mel_images_agree_with_the_speechdsp_reference(rng):
    # A 32 ms Hann frame equals speechdsp's frame (= n_fft), and 16 032 samples
    # give exactly 98 frames in both implementations.
    signal = _tone(440.0, 16_032 / RATE) + 0.3 * rng.standard_normal(16_032)
    spectrogram_config = SpectrogramConfig(scale="mel", frame_ms=32.0, n_fft=512, window="hann")
    ours = spectrogram_image(signal, RATE, spectrogram_config).astype(int)
    reference = speechdsp.spectrogram_image(signal, RATE, shape=(40, 98), n_fft=512, hop=160)
    assert np.abs(ours - reference.astype(int)).max() <= 1


def test_linear_band_rows_average_the_bins_below_fmax():
    weights = band_weights("linear", 40, RATE, 512, 6000.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    frequencies = np.fft.rfftfreq(512, 1 / RATE)
    assert np.all(weights[:, frequencies > 6000.0 + 16.0] == 0.0)
    fine = band_weights("linear", 400, RATE, 512, 8000.0)
    assert np.all(fine.sum(axis=1) > 0.99)


def test_fmax_below_nyquist_restricts_the_image():
    spectrogram_config = SpectrogramConfig(fmax_hz=4000.0)
    image = spectrogram_image(_tone(3950.0), RATE, spectrogram_config)
    assert int(image.mean(axis=1).argmax()) == 39


def test_other_rates_and_frame_counts_work():
    spectrogram_config = SpectrogramConfig(n_bands=20, n_frames=50, fmax_hz=4000.0)
    images = spectrogram_images(np.stack([_tone(500.0, 1.0, 8000)] * 2), 8000, spectrogram_config)
    assert images.shape == (2, 20, 50)
    assert int(images[0].mean(axis=1).argmax()) == 2


def test_each_segment_is_scaled_to_its_own_peak():
    loud, soft = _tone(1100.0), 1e-4 * _tone(1100.0)
    images = spectrogram_images(np.stack([loud, soft]), RATE)
    assert np.array_equal(images[0], images[1])


def test_silence_gives_a_black_image():
    image = spectrogram_image(np.zeros(RATE), RATE)
    assert int(image.max()) == 0


def test_decibels_quantization_and_interpolation():
    assert to_decibels([1.0, 0.01, 1e-20], top_db=30.0).tolist() == [0.0, -20.0, -30.0]
    assert quantize([-100.0, -80.0, 0.0, 5.0]).tolist() == [0, 0, 255, 255]
    values = np.arange(12.0).reshape(2, 6)
    assert np.array_equal(fit_frames(values, 6), values)
    stretched = fit_frames(values, 11)
    assert stretched[:, 0].tolist() == [0.0, 6.0] and stretched[:, -1].tolist() == [5.0, 11.0]
    assert fit_frames(np.zeros((3, 0)), 4, fill=-80.0).tolist() == [[-80.0] * 4] * 3


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_energy_is_an_error_not_silence(bad):
    with pytest.raises(ValueError, match="finite"):
        to_decibels([1.0, bad, 0.1])


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="window"):
        power_spectrogram(np.zeros(1000), RATE, window="kaiser")
    with pytest.raises(ValueError, match="shorter than"):
        analysis_sizes(RATE, 25.0, 10.0, n_fft=256)
    with pytest.raises(ValueError, match="fmax_hz"):
        band_weights("linear", 40, RATE, 512, 9000.0)
    with pytest.raises(ValueError, match="scale"):
        band_weights("bark", 40, RATE, 512, 8000.0)
    with pytest.raises(ValueError, match="segments of shape"):
        spectrogram_images(np.zeros(100), RATE)
