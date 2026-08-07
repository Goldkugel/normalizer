import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from nltk.stem          import PorterStemmer
from .NormalizerConfig  import NormalizerConfig
from logger             import Logger
import nltk
import scispacy
import spacy
import yaml

# Key under which normalizer settings are expected to live in the YAML config file.
configuration_section: str  = "normalizer"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str     = "./config/config.yaml"

class Normalizer:
    """Configurable lexical normalization pipeline for biomedical text.

    Uses spaCy / scispacy for biomedical tokenization, stopword removal,
    and POS-guided lemmatization, with optional NLTK stemming support.
    """
    config: NormalizerConfig = None

    def __init__(self, config: str = standard_directory):
        l = Logger()
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        self.config = NormalizerConfig.model_validate(data[configuration_section])

        l.log(f"Loading spacy model '{self.config.model_name}'...")
        # Load the specified spaCy / scispacy model
        self.nlp = spacy.load(self.config.model_name, exclude=["ner", "parser"])
        l.log(f"Loading spacy model '{self.config.model_name}' completed.")

        # Initialize stemmer only if stemming is explicitly enabled
        if self.config.stemming:
            l.log("Loading stemmer...")
            self.stemmer = PorterStemmer()
            l.log("Loading stemmer completed.")

    def normalize_batch(self, texts: list[str], batch_size: int = 256, n_process: int = 1) -> list[list[str]]:
        """Normalizes many strings at once using spaCy's batched pipeline."""
        if self.config.lower_case:
            texts = [t.lower() if t else t for t in texts]

        results: list[list[str]] = []
        docs = self.nlp.pipe(
            (t if t else "" for t in texts),
            batch_size=batch_size,
            n_process=n_process,
        )
        for doc in docs:
            tokens: list[str] = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                if not token.is_alpha and len(token) <= 1:
                    continue
                if self.config.stop_word_removal and token.is_stop:
                    continue
                token_str = token.lemma_ if self.config.lemmatizing else token.text
                if self.config.stemming:
                    token_str = self.stemmer.stem(token_str)
                tokens.append(token_str)
            results.append(tokens)
        return results

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