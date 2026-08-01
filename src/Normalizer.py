import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from nltk.stem          import PorterStemmer
from NormalizerConfig   import NormalizerConfig
import nltk
import scispacy
import spacy
import yaml

# Key under which model settings are expected to live in the YAML config file.
configuration_section: str  = "normalizer"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str     = "../config/config.yaml"

class Normalizer:
    """Configurable lexical normalization pipeline for biomedical text.

    Uses spaCy / scispacy for biomedical tokenization, stopword removal,
    and POS-guided lemmatization, with optional NLTK stemming support.
    """
    config: NormalizerConfig = None

    def __init__(self, config: str = standard_directory):
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        self.config = NormalizerConfig.model_validate(data[configuration_section])

        # Load the specified spaCy / scispacy model
        self.nlp = spacy.load(self.config.model_name)

        # Initialize stemmer only if stemming is explicitly enabled
        if self.config.stemming:
            self.stemmer = PorterStemmer()

    def normalize(self, text: str) -> list:
        """Normalizes an input string according to the configured options."""
        if not text:
            return ""

        # 1. Lowercasing
        if self.config.lower_case:
            text = text.lower()

        # Process through spaCy pipeline
        doc = self.nlp(text)

        tokens: list[str] = []

        for token in doc:
            # Skip punctuation, whitespace, and single-character tokens
            if token.is_punct or token.is_space:
                continue

            # Keep single-character alphanumeric tokens (like 'c' or 'b'), 
            # but filter out non-alphanumeric noise if any remain
            if not token.is_alpha and len(token) <= 1:
                continue

            # 2. Stopword Removal
            if self.config.stop_word_removal and token.is_stop:
                continue

            # 3. Lemmatization vs Raw Text
            token_str = token.lemma_ if self.config.lemmatizing else token.text

            # 4. Optional Stemming
            if self.config.stemming:
                token_str = self.stemmer.stem(token_str)

            tokens.append(token_str)

        return tokens