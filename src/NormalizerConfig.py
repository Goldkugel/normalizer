"""
Pydantic model for the settings of the Normalizer.

Validates the "normalizer" section of the YAML config file, or the
equivalent dict, before the Normalizer loads its spaCy model.
"""

from pydantic import BaseModel, ConfigDict

class NormalizerConfig(BaseModel):
    """Pydantic model defining and validating the "normalizer" config section.

    Instances control which pipeline components (lowercasing, lemmatization,
    stemming, stopword removal) are active and specify the underlying NLP
    model to use. Every field has a default, so all keys are optional.
    Instances are read-only; use model_copy(update={...}) for a variant.
    """

    # Reject any keys in the input data that aren't explicitly defined below.
    # This catches typos or outdated config keys early, instead of silently
    # ignoring them. Frozen, because the Normalizer loads the model and the
    # stemmer once from these settings: changing them afterwards would have
    # no effect (model_name) or break normalize() (stemming), so assigning a
    # field raises a ValidationError instead.
    model_config            = ConfigDict(extra="forbid", frozen=True)

    # The spaCy / scispacy model name to load for tokenization and lemmatization.
    # The model must be installed as a Python package; only the default,
    # en_core_sci_sm, is a dependency of this package (see pyproject.toml).
    model_name: str         = "en_core_sci_sm"

    # Convert all input text to lowercase before further processing. Also
    # keeps the Porter stemmer from lowercasing when set to False.
    lower_case: bool        = True

    # Apply POS-guided lemmatization to reduce words to their base/dictionary form.
    lemmatizing: bool       = True

    # Apply NLTK Porter stemming to reduce words to their root form, on top of
    # the lemma (or the raw token). Disabled by default: stems such as "abnorm"
    # or "sever" merge more word forms, but also unrelated words.
    stemming: bool          = False

    # Drop tokens that are on spaCy's general English stop word list. Note that
    # this list is not domain-specific and includes words that can matter in
    # clinical terms, such as "no", "not", "without" and "back".
    stop_word_removal: bool = True
