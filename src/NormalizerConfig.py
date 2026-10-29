import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from pydantic import BaseModel, ConfigDict

class NormalizerConfig(BaseModel):
    """Pydantic model defining and validating the "normalizer" config section.

    Instances control which pipeline components (lowercasing, lemmatization,
    stemming, stopword removal) are active and specify the underlying NLP
    model to use. Every field has a default, so all keys are optional.
    """

    # Reject any keys in the input data that aren't explicitly defined below.
    # This catches typos or outdated config keys early, instead of silently
    # ignoring them.
    model_config            = ConfigDict(extra="forbid")

    # The spaCy / scispacy model name to load for tokenization and lemmatization.
    # The model must be installed as a Python package (see pyproject.toml).
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