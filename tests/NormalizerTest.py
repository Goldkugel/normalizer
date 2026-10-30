"""
Pytest suite for the Normalizer and its configuration schema.

Tests the installed `normalizer` package (`pip install -e ".[test]"`), which
also provides the en_core_sci_sm model. Run with:

    pytest tests/NormalizerTest.py -v

The tests write their own temporary config and log files, so they can be
started from any directory.
"""

import math
import pathlib
import sys
import pytest
import yaml
from pydantic import ValidationError

from logger import Logger, LoggerConfig
from normalizer import Normalizer, NormalizerConfig

# --- Fixtures -------------------------------------------------------------
def _reset_logger():
    """Forget the Logger singleton and its configuration, so the next
    Logger() call starts from scratch."""
    Logger._instance  = None
    Logger.config     = None
    Logger.start_time = 0


@pytest.fixture(scope="module", autouse=True)
def standard_config_path(tmp_path_factory):
    """Writes a temporary config file with a logger and a normalizer section
    and returns its path.

    The Logger is reset and configured from this file, so test logs end up
    in the temporary folder instead of the repository, whichever test runs
    first.

    Unlike the shipped config/config.yaml, the test config enables stemming,
    so the shared normalizer exercises every pipeline step at once. Tests that
    need other settings write their own file (see TestConfigFlagToggles).
    """
    tmp_dir = tmp_path_factory.mktemp("config")
    config_data = {
        "logger": {
            "folder": str(tmp_dir / "logs") + "/",
            "file_name": "test.log",
        },
        "normalizer": {
            "model_name": "en_core_sci_sm",
            "lower_case": True,
            "lemmatizing": True,
            "stemming": True,
            "stop_word_removal": True,
        },
    }
    config_file = tmp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    _reset_logger()
    Logger(str(config_file))
    yield str(config_file)
    _reset_logger()


@pytest.fixture(scope="module")
def standard_normalizer(standard_config_path):
    """Provides a shared Normalizer loaded with the test config (all steps
    enabled), to avoid re-loading the spaCy model in every test.
    """
    return Normalizer(config=standard_config_path)


# --- Test Classes ---------------------------------------------------------
class TestNormalizerConfig:

    def test_config_initializes_with_expected_defaults(self):
        config = NormalizerConfig()
        assert config.model_name == "en_core_sci_sm"
        assert config.lower_case is True
        assert config.lemmatizing is True
        assert config.stemming is False  # Pydantic default
        assert config.stop_word_removal is True

    def test_config_rejects_extra_undefined_fields(self):
        with pytest.raises(ValidationError):
            NormalizerConfig(invalid_param=123)  # type: ignore


# All tests in this class use the shared normalizer, i.e. with stemming.
class TestDefaultPipelineBehavior:

    def test_normalizes_standard_medical_text(self, standard_normalizer):
        input_text = "Severe abnormalities of the kidneys"
        # "severe" -> "sever", "abnormality" -> "abnorm", "kidney" -> "kidney"
        assert standard_normalizer.normalize(input_text) == {
            "sever",
            "abnorm",
            "kidney",
        }

    def test_preserves_single_character_medical_terms(self, standard_normalizer):
        assert standard_normalizer.normalize("Aspirin C") == {"aspirin", "c"}
        assert standard_normalizer.normalize("Hepatitis C infection") == {
            "hepat",
            "c",
            "infect",
        }
        assert standard_normalizer.normalize("Vitamin D deficiency") == {
            "vitamin",
            "d",
            "defici",
        }

    def test_removes_punctuation_and_collapses_whitespace(self, standard_normalizer):
        input_text = "Kidney disease, chronic!"
        assert standard_normalizer.normalize(input_text) == {
            "kidney",
            "diseas",
            "chronic",
        }

        whitespace_text = "   Patients  were  suffering   "
        assert standard_normalizer.normalize(whitespace_text) == {
            "patient",
            "suffer",
        }

    def test_handles_empty_or_whitespace_only_input(self, standard_normalizer):
        assert standard_normalizer.normalize("") == set()
        assert standard_normalizer.normalize("   ") == set()

    def test_preserves_single_digit_tokens(self, standard_normalizer):
        type_1 = standard_normalizer.normalize("Type 1 diabetes mellitus")
        type_2 = standard_normalizer.normalize("Type 2 diabetes mellitus")
        assert "1" in type_1
        assert "2" in type_2
        assert type_1 != type_2

    def test_treats_missing_values_as_empty_input(self, standard_normalizer):
        assert standard_normalizer.normalize(None) == set()
        assert standard_normalizer.normalize(math.nan) == set()


class TestBatchNormalization:

    def test_batch_matches_single_normalization(self, standard_normalizer):
        texts = [
            "Severe abnormalities of the kidneys",
            "Hepatitis C infection",
            "Type 1 diabetes mellitus",
            "",
        ]
        assert standard_normalizer.normalize_batch(texts) == [
            standard_normalizer.normalize(t) for t in texts
        ]

    def test_batch_treats_missing_values_as_empty_input(self, standard_normalizer):
        # Empty pandas cells arrive as NaN floats; they must not become "nan"
        assert standard_normalizer.normalize_batch([None, math.nan, "kidney"]) == [
            set(),
            set(),
            {"kidney"},
        ]


class TestConfigFlagToggles:

    def _create_custom_config(self, tmp_path, overrides: dict) -> str:
        """Writes a normalizer config with all steps enabled, changed by
        `overrides`, to a temporary file and returns its path.

        The file has no logger section; the Logger keeps the configuration
        it got from the module fixture's file.
        """
        base_config = {
            "model_name": "en_core_sci_sm",
            "lower_case": True,
            "lemmatizing": True,
            "stemming": True,
            "stop_word_removal": True,
        }
        base_config.update(overrides)
        config_path = tmp_path / "custom_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"normalizer": base_config}, f)
        return str(config_path)

    def test_preserves_casing_when_lowercasing_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"lower_case": False, "stemming": False}
        )
        normalizer = Normalizer(config=config_file)

        # "The" is removed (stop word lookup ignores the casing); "Kidneys"
        # keeps its capital letter and, tagged as a proper noun, also its
        # plural form, although lemmatization is enabled
        result = normalizer.normalize("The Kidneys")
        assert result == {"Kidneys"}

    def test_retains_raw_tokens_when_lemmatization_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"lemmatizing": False, "stemming": False}
        )
        normalizer = Normalizer(config=config_file)

        result = normalizer.normalize("abnormalities of kidneys")
        assert result == {"abnormalities", "kidneys"}

    def test_retains_stopwords_when_stopword_removal_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"stop_word_removal": False, "stemming": False}
        )
        normalizer = Normalizer(config=config_file)

        result = normalizer.normalize("Abnormality of the kidney")
        assert result == {"abnormality", "of", "the", "kidney"}

    def test_disables_stemming_when_stemming_flag_is_false(self, tmp_path):
        config_file = self._create_custom_config(tmp_path, {"stemming": False})
        normalizer = Normalizer(config=config_file)

        # Lemmatized without Porter stemming ("abnormalities" -> "abnormality"
        # instead of "abnorm")
        result = normalizer.normalize("abnormalities")
        assert result == {"abnormality"}

    def test_stemming_preserves_casing_when_lowercasing_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"lower_case": False, "lemmatizing": False}
        )
        normalizer = Normalizer(config=config_file)

        # PorterStemmer lowercases by default; the configured casing must win
        result = normalizer.normalize("Kidneys")
        assert result == {"Kidney"}

    def test_empty_section_falls_back_to_defaults(self, tmp_path):
        config_path = tmp_path / "empty_section.yaml"
        config_path.write_text("normalizer:\n")
        normalizer = Normalizer(config=str(config_path))
        assert normalizer.config == NormalizerConfig()

    def test_missing_section_raises_key_error(self, tmp_path):
        config_path = tmp_path / "no_section.yaml"
        config_path.write_text("other: {}\n")
        with pytest.raises(KeyError, match="normalizer"):
            Normalizer(config=str(config_path))


class TestConfigSources:
    """Normalizer(config=...) accepts a YAML path, a dict (the whole
    configuration or just the normalizer section) or a NormalizerConfig."""

    def test_accepts_a_path_object(self, standard_config_path):
        normalizer = Normalizer(config=pathlib.Path(standard_config_path))
        assert normalizer.config.stemming is True

    def test_accepts_a_dict_of_normalizer_settings(self):
        settings   = {"stemming": False, "lemmatizing": False, "lower_case": False}
        normalizer = Normalizer(config=settings)
        assert normalizer.config == NormalizerConfig(**settings)
        assert normalizer.normalize("Kidneys") == {"Kidneys"}

    def test_accepts_a_full_configuration_dict(self):
        # The Logger is already configured by the module fixture, so the
        # dict's logger section is ignored and no folder is created there
        normalizer = Normalizer(config={
            "logger"    : {"folder": "/should/not/be/used/"},
            "normalizer": {"stemming": False},
        })
        assert normalizer.config == NormalizerConfig(stemming=False)

    def test_empty_section_in_a_dict_falls_back_to_defaults(self):
        assert Normalizer(config={"normalizer": None}).config == NormalizerConfig()

    def test_rejects_unknown_keys_in_a_dict(self):
        with pytest.raises(ValidationError):
            Normalizer(config={"unknown_key": True})

    def test_accepts_a_normalizer_config(self):
        config     = NormalizerConfig(stemming=False)
        normalizer = Normalizer(config=config)
        assert normalizer.config == config

        # Later changes to the passed object must not affect the normalizer.
        config.lower_case = False
        assert normalizer.config.lower_case is True

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Normalizer(config=str(tmp_path / "missing.yaml"))


class TestLoggerConfiguration:

    def test_unconfigured_logger_is_configured_from_the_normalizer_config(
        self, standard_config_path
    ):
        # Start without a Logger configuration, as in a fresh process
        _reset_logger()
        Normalizer(config=standard_config_path)

        with open(standard_config_path) as f:
            expected = yaml.safe_load(f)["logger"]["folder"]
        assert Logger.config.folder == expected

    def test_unconfigured_logger_is_configured_from_a_full_configuration_dict(
        self, tmp_path
    ):
        _reset_logger()
        Normalizer(config={
            "logger"    : {"folder": str(tmp_path) + "/", "file_name": "dict.log"},
            "normalizer": {"stemming": False},
        })
        assert Logger.config.folder    == str(tmp_path) + "/"
        assert Logger.config.file_name == "dict.log"

    def test_unconfigured_logger_uses_defaults_without_a_default_config_file(
        self, tmp_path, monkeypatch, standard_config_path
    ):
        # A fresh process outside a project directory: no logger settings in
        # the source and no ./config/config.yaml to fall back to
        monkeypatch.chdir(tmp_path)
        _reset_logger()
        Normalizer(config=NormalizerConfig())
        assert Logger.config == LoggerConfig()

        # The default log folder is relative to the working directory; point
        # the Logger back to the temporary test folder for later tests
        Logger(standard_config_path)

    def test_unconfigured_logger_falls_back_to_the_default_config_file(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.yaml").write_text(
            f"logger:\n  folder: \"{tmp_path}/default_logs/\"\n"
        )
        monkeypatch.chdir(tmp_path)
        _reset_logger()
        Normalizer(config={"stemming": False})
        assert Logger.config.folder == f"{tmp_path}/default_logs/"

    def test_existing_logger_configuration_is_kept(self, tmp_path, standard_normalizer):
        other_config = tmp_path / "other.yaml"
        other_config.write_text(
            "logger:\n  folder: \"/should/not/be/used/\"\nnormalizer: {}\n"
        )
        folder_before = Logger.config.folder
        Normalizer(config=str(other_config))
        assert Logger.config.folder == folder_before


class TestOntologyMatchingAlignment:

    def test_hpo_and_snomed_terms_normalize_to_equivalent_token_sets(
        self, standard_normalizer
    ):
        hpo_label = "Abnormality of the kidney"
        snomed_description = "Kidney abnormalities"

        tokens_hpo = standard_normalizer.normalize(hpo_label)
        tokens_snomed = standard_normalizer.normalize(snomed_description)

        # Both terms reduce to the same token set, so they match despite
        # different word order, inflection and the stop words "of the"
        assert tokens_hpo == tokens_snomed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
