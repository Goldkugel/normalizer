import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

import warnings
import pytest
import yaml
from pydantic import ValidationError

from .Normalizer import Normalizer
from .NormalizerConfig import NormalizerConfig

# Ignore third-party spaCy regex union warnings in Python 3.10+
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="spacy.language"
)

# --- Fixtures -------------------------------------------------------------
@pytest.fixture(scope="module")
def standard_config_path(tmp_path_factory):
    """Writes a temporary YAML config file containing your project's standard

    configuration settings and returns its file path.
    """
    config_data = {
        "normalizer": {
            "model_name": "en_core_sci_sm",
            "lower_case": True,
            "lemmatizing": True,
            "stemming": True,
            "stop_word_removal": True,
        }
    }
    tmp_dir = tmp_path_factory.mktemp("config")
    config_file = tmp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return str(config_file)


@pytest.fixture(scope="module")
def standard_normalizer(standard_config_path):
    """Provides a shared instance of Normalizer loaded with the standard config

    to avoid re-loading the spaCy model across multiple test methods.
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


class TestDefaultPipelineBehavior:

    def test_normalizes_standard_medical_text(self, standard_normalizer):
        input_text = "Severe abnormalities of the kidneys"
        # "severe" -> "sever", "abnormality" -> "abnorm", "kidney" -> "kidney"
        assert standard_normalizer.normalize(input_text) == [
            "sever",
            "abnorm",
            "kidney",
        ]

    def test_preserves_single_character_medical_terms(self, standard_normalizer):
        assert standard_normalizer.normalize("Aspirin C") == ["aspirin", "c"]
        assert standard_normalizer.normalize("Hepatitis C infection") == [
            "hepat",
            "c",
            "infect",
        ]
        assert standard_normalizer.normalize("Vitamin D deficiency") == [
            "vitamin",
            "d",
            "defici",
        ]

    def test_removes_punctuation_and_collapses_whitespace(self, standard_normalizer):
        input_text = "Kidney disease, chronic!"
        assert standard_normalizer.normalize(input_text) == [
            "kidney",
            "diseas",
            "chronic",
        ]

        whitespace_text = "   Patients  were  suffering   "
        assert standard_normalizer.normalize(whitespace_text) == [
            "patient",
            "suffer",
        ]

    def test_handles_empty_or_whitespace_only_input(self, standard_normalizer):
        assert standard_normalizer.normalize("") == ""
        assert standard_normalizer.normalize("   ") == []


class TestConfigFlagToggles:

    def _create_custom_config(self, tmp_path, overrides: dict) -> str:
        """Helper to dump custom normalizer config flags into a temporary file."""
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

        # "The" (stopword) is removed; "Kidneys" keeps initial capital letter
        result = normalizer.normalize("The Kidneys")
        assert result == ["Kidneys"]

    def test_retains_raw_tokens_when_lemmatization_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"lemmatizing": False, "stemming": False}
        )
        normalizer = Normalizer(config=config_file)

        result = normalizer.normalize("abnormalities of kidneys")
        assert result == ["abnormalities", "kidneys"]

    def test_retains_stopwords_when_stopword_removal_is_disabled(self, tmp_path):
        config_file = self._create_custom_config(
            tmp_path, {"stop_word_removal": False, "stemming": False}
        )
        normalizer = Normalizer(config=config_file)

        result = normalizer.normalize("Abnormality of the kidney")
        assert result == ["abnormality", "of", "the", "kidney"]

    def test_disables_stemming_when_stemming_flag_is_false(self, tmp_path):
        config_file = self._create_custom_config(tmp_path, {"stemming": False})
        normalizer = Normalizer(config=config_file)

        # Lemmatized without Porter Stemming ("abnormalities" -> "abnormality" instead of "abnorm")
        result = normalizer.normalize("abnormalities")
        assert result == ["abnormality"]


class TestOntologyMatchingAlignment:

    def test_hpo_and_snomed_terms_normalize_to_equivalent_token_sets(
        self, standard_normalizer
    ):
        hpo_label = "Abnormality of the kidney"
        snomed_description = "Kidney abnormalities"

        tokens_hpo = standard_normalizer.normalize(hpo_label)
        tokens_snomed = standard_normalizer.normalize(snomed_description)

        # Set equality verifies semantic token alignment regardless of original word order
        assert set(tokens_hpo) == set(tokens_snomed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))