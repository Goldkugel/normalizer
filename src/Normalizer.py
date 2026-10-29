"""
Lexical normalizer for biomedical terms.

Provides the Normalizer class, which turns a term such as "Abnormality of
the kidney" into an unordered set of normalized tokens ({"abnormality",
"kidney"}). Two terms with equal (or overlapping) token sets can then be
matched regardless of word order, inflection or filler words, e.g. when
aligning HPO labels with SNOMED CT descriptions.
"""

from __future__ import annotations

from collections.abc    import Iterable
from nltk.stem          import PorterStemmer
from spacy.language     import Language
from spacy.tokens       import Doc
from .NormalizerConfig  import NormalizerConfig
from logger             import Logger
import math
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

    Every input goes through the same steps, each of which can be switched
    on or off in the config:

    1. Lowercasing                       (_prepare)
    2. Stopword removal                  (_tokens)
    3. Lemmatization                     (_tokens)
    4. Stemming                          (_tokens)

    Punctuation, whitespace and pure symbol tokens are always dropped.
    """

    # The validated "normalizer" section of the config file.
    config: NormalizerConfig

    # The loaded spaCy pipeline.
    nlp: Language

    # Only set when stemming is enabled.
    stemmer: PorterStemmer

    def __init__(self, config: str = standard_directory) -> None:
        """
        Load the configuration, the spaCy model and, if enabled, the stemmer.

        Parameters
        ----------
        config : str, optional
            Path to the YAML configuration file. Defaults to
            ./config/config.yaml relative to the working directory.

            If no other module has configured the Logger yet and the file
            has a "logger" section, the Logger is configured from the same
            file.

        Raises
        ------
        KeyError
            If the file has no "normalizer" section.
        pydantic.ValidationError
            If the section contains an unknown key or an invalid value.
        """
        with open(config, "r", encoding="utf-8") as f:
            # An empty file is parsed as None; treat it like a file without
            # sections so the KeyError below explains what is missing.
            data = yaml.safe_load(f) or {}

        # The Logger is a process-wide singleton. If it has not been
        # configured yet, configure it from this file, so a custom config path
        # also covers logging. If it already has a configuration (e.g. set by
        # the calling pipeline), keep it; a bare Logger() call does not reload.
        # Files without a "logger" section leave the Logger to its own default
        # path (./config/config.yaml).
        if Logger.config is None and "logger" in data:
            log = Logger(config)
        else:
            log = Logger()

        if configuration_section not in data:
            raise KeyError(
                f"Config file '{config}' has no '{configuration_section}' section."
            )
        # An empty section ("normalizer:" with no keys) means "use all defaults".
        self.config = NormalizerConfig.model_validate(data[configuration_section] or {})

        log.log(f"Loading spacy model '{self.config.model_name}'...")
        # The named-entity recognizer and the dependency parser are not needed
        # for normalization, so they are not loaded, which saves loading and
        # processing time. The tagger stays, because the lemmatizer uses its
        # part-of-speech tags.
        self.nlp = spacy.load(self.config.model_name, exclude=["ner", "parser"])
        log.log(f"Loading spacy model '{self.config.model_name}' completed.")

        # Initialize stemmer only if stemming is explicitly enabled
        if self.config.stemming:
            log.log("Loading stemmer...")
            self.stemmer = PorterStemmer()
            log.log("Loading stemmer completed.")

    def normalize_batch(
        self,
        texts: Iterable[object],
        batch_size: int = 256,
        n_process: int = 1,
    ) -> list[set[str]]:
        """
        Normalize many terms at once using spaCy's batched pipeline.

        Gives the same result as calling normalize() on every element, but is
        much faster for large term lists.

        Parameters
        ----------
        texts : Iterable[object]
            The terms to normalize. Missing values (None, NaN) are allowed
            and yield an empty set, like an empty string.
        batch_size : int, optional
            Number of terms spaCy processes per batch (default 256).
        n_process : int, optional
            Number of worker processes spaCy starts (default 1). Each process
            loads its own copy of the model, so more processes only pay off
            for long lists.

        Returns
        -------
        list[set[str]]
            One token set per input term, in input order.
        """
        docs = self.nlp.pipe(
            (self._prepare(t) for t in texts),
            batch_size=batch_size,
            n_process=n_process,
        )
        return [self._tokens(doc) for doc in docs]

    def normalize(self, text: object) -> set[str]:
        """
        Normalize a single term according to the configured options.

        Parameters
        ----------
        text : object
            The term to normalize, normally a str. None and NaN are allowed
            and yield an empty set; other values are converted with str().

        Returns
        -------
        set[str]
            The normalized tokens. A set is returned because matching is
            meant to ignore word order and repeated words.
        """
        prepared = self._prepare(text)

        # Nothing to tokenize; skip the spaCy call.
        if not prepared:
            return set()
        return self._tokens(self.nlp(prepared))

    def _prepare(self, text: object) -> str:
        """Turns an input value into the string fed to spaCy.

        None and NaN (e.g. empty cells from a pandas column) become "" instead
        of the literal token "nan"; other non-strings are converted with str().
        """
        # numpy.float64 is a subclass of float, so NumPy/pandas NaN is caught too.
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return ""
        result = str(text)

        # 1. Lowercasing (done before spaCy, so stop word lookup and
        #    lemmatization see the same form regardless of the input casing)
        if self.config.lower_case:
            result = result.lower()
        return result

    def _tokens(self, doc: Doc) -> set[str]:
        """Applies the configured token filters and transformations to a doc."""
        tokens: set[str] = set()

        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue

            # Keep letters and digits, including single characters ('c' in
            # "Hepatitis C", '1' in "Type 1 diabetes"), but drop leftover
            # symbol tokens such as '+' or '%'
            if not any(c.isalnum() for c in token.text):
                continue

            # 2. Stopword Removal (spaCy's general English stop word list,
            #    which also contains words such as "no", "not" and "back")
            if self.config.stop_word_removal and token.is_stop:
                continue

            # 3. Lemmatization vs Raw Text
            token_str = token.lemma_ if self.config.lemmatizing else token.text

            # 4. Optional Stemming (PorterStemmer lowercases unless told not to)
            if self.config.stemming:
                token_str = self.stemmer.stem(
                    token_str, to_lowercase=self.config.lower_case
                )

            tokens.add(token_str)

        return tokens
