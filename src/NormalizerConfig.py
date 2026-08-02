import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from pydantic import BaseModel, ConfigDict

class NormalizerConfig(BaseModel):
    """Pydantic model defining and validating the configuration schema for the

    lexical normalizer.

    Instances control which pipeline components (lowercasing, lemmatization,
    stemming, stopword removal) are active and specify the underlying NLP
    model to use.
    """

    # Reject any keys in the input data that aren't explicitly defined below.
    # This catches typos or outdated config keys early, instead of silently
    # ignoring them.
    model_config            = ConfigDict(extra="forbid")

    # The spaCy / scispacy model name to load for tokenization and lemmatization.
    model_name: str         = "en_core_sci_sm"

    # Convert all input text to lowercase before further processing.
    lower_case: bool        = True

    # Apply POS-guided lemmatization to reduce words to their base/dictionary form.
    lemmatizing: bool       = True

    # Apply stemming to reduce words to their root form (disabled by default for biomedical text).
    stemming: bool          = False

    # Filter out language- and domain-specific stop words from token streams.
    stop_word_removal: bool = True