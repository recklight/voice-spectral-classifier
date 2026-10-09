"""Configuration defaults, YAML round trips and validation messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_spectral_classifier import FEATURE_SECTIONS, Config, ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))


def test_default_yaml_equals_the_dataclass_defaults():
    assert Config.from_yaml(PROJECT_ROOT / "configs" / "default.yaml") == Config()


def test_default_yaml_lists_every_key_in_dataclass_order():
    text = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# Default configuration.")
    assert lines[1].startswith("# Override single values on the command line with")
    keys = [line.split(":")[0].strip() for line in lines[2:] if not line.lstrip().startswith("#")]
    expected = []
    for name, value in Config().to_dict().items():
        expected.append(name)
        if isinstance(value, dict):
            expected.extend(value)
    assert keys == expected


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_every_configuration_file_loads(path):
    assert isinstance(Config.from_yaml(path), Config)


def test_quick_demo_points_at_the_synthetic_dataset():
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    assert config.data.dataset == "data/synthetic/dataset.csv"
    assert config.model.name == "mlp"
    first = (PROJECT_ROOT / "configs" / "quick_demo.yaml").read_text(encoding="utf-8")
    assert first.startswith(
        "# Small, fast settings for synthetic data (examples, smoke tests, CI)."
    )


def test_yaml_round_trip_keeps_every_value(tmp_path):
    config = Config().override(
        {
            "seed": 7,
            "spectrogram.scale": "mel",
            "segmentation.max_segments": 3,
            "model.hidden_units": [64, 32],
        }
    )
    assert Config.from_yaml(config.to_yaml(tmp_path / "nested" / "config.yaml")) == config
    assert config.model.hidden_units == (64, 32)


def test_feature_sections_are_the_ones_prepare_depends_on():
    assert FEATURE_SECTIONS == ("segmentation", "spectrogram")


def test_derived_sizes():
    config = Config()
    assert config.segmentation.segment_length == 16_000
    assert config.spectrogram.image_shape == (40, 98)


def test_unknown_key_is_rejected_and_the_valid_keys_listed():
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['n_band'\].*valid keys are"):
        Config.from_dict({"spectrogram": {"n_band": 20}})


def test_unknown_top_level_setting_is_rejected():
    with pytest.raises(ConfigError, match="valid top-level keys"):
        Config.from_dict({"spectrograms": {}})


def test_override_rejects_unknown_sections_and_keys():
    with pytest.raises(ConfigError, match="unknown configuration section"):
        Config().override({"features.n_bands": 3})
    with pytest.raises(ConfigError, match=r"unknown configuration key 'model\.layers'"):
        Config().override({"model.layers": 3})


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("seed", -1, "seed must be a non-negative integer"),
        ("spectrogram.scale", "bark", "spectrogram.scale='bark'"),
        ("spectrogram.window", "kaiser", "spectrogram.window"),
        ("spectrogram.n_bands", 0, "spectrogram.n_bands"),
        ("spectrogram.fmax_hz", 9000.0, "spectrogram.fmax_hz=9000.0 is above the Nyquist"),
        ("spectrogram.frame_ms", 2000.0, "spectrogram.frame_ms=2000.0 is longer than"),
        ("spectrogram.n_fft", 256, "spectrogram.n_fft=256 is shorter than"),
        ("spectrogram.top_db", 0, "spectrogram.top_db"),
        ("segmentation.min_fraction", 1.5, "segmentation.min_fraction"),
        ("segmentation.segment_s", "long", "segmentation.segment_s must be a finite number"),
        ("segmentation.segment_s", float("inf"), "segmentation.segment_s must be a finite number"),
        ("segmentation.hop_s", float("inf"), "segmentation.hop_s must be a finite number"),
        ("synthetic.silence_s", float("nan"), "synthetic.silence_s must be a finite number"),
        ("model.weight_decay", float("nan"), "model.weight_decay must be a finite number"),
        ("segmentation.trim_silence", "yes", "segmentation.trim_silence"),
        ("segmentation.max_segments", 0, "segmentation.max_segments"),
        ("model.name", "svm", "model.name='svm'"),
        ("model.n_epochs", "many", "model.n_epochs"),
        ("model.dropout", 1.0, "model.dropout"),
        ("model.class_weight", "inverse", "model.class_weight"),
        ("model.hidden_units", [], "model.hidden_units"),
        ("model.device", "tpu", "model.device"),
        ("evaluation.n_splits", 1, "evaluation.n_splits must be at least 2"),
        ("output.figure_format", "gif", "output.figure_format"),
        ("synthetic.class_names", ["modal"], "synthetic.class_names"),
        ("synthetic.class_names", ["modal", "creaky"], "synthetic.class_names='creaky'"),
        ("synthetic.n_speakers_per_class", [3, 3], "synthetic.n_speakers_per_class"),
        ("synthetic.sample_rates", [8000], "synthetic.sample_rates"),
        ("synthetic.severity_range", [0.8, 0.2], "synthetic.severity_range must satisfy"),
        ("synthetic.severity_range", [0.5], "synthetic.severity_range must be"),
    ],
)
def test_invalid_values_name_the_dotted_key(key, value, fragment):
    with pytest.raises(ConfigError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        Config().override({key: value})


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        Config.from_yaml(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        Config.from_yaml(broken)
    listing = tmp_path / "listing.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a mapping"):
        Config.from_yaml(listing)


def test_a_section_must_be_a_mapping():
    with pytest.raises(ConfigError, match="section 'model' must be a mapping"):
        Config.from_dict({"model": 3})


def test_configurations_are_hashable_and_comparable():
    assert hash(Config()) == hash(Config())
    assert Config().override({"seed": 1}) != Config()
