"""
Lexical normalizer for biomedical terms.

The Normalizer turns a term such as "Abnormality of the kidney" into a set
of normalized tokens ({"abnormality", "kidney"}), so that terms can be
matched regardless of word order, inflection or filler words, e.g. HPO
labels with SNOMED CT descriptions.
"""

from __future__ import annotations

from collections.abc    import Iterable
from nltk.stem          import PorterStemmer
from spacy.language     import Language
from spacy.tokens       import Doc, Token
from .NormalizerConfig  import NormalizerConfig
from logger             import Logger, LoggerConfig
import os
import spacy
import sys
import yaml

# Key of the normalizer settings in the YAML config file.
configuration_section: str  = "normalizer"

# Key of the logger settings in the same file (see the logger package).
logger_section: str         = "logger"

# Default config file; also the Logger's own default path.
standard_directory: str     = "./config/config.yaml"

class Normalizer:
    """Configurable lexical normalization pipeline for biomedical text.

    Uses spaCy / scispacy for biomedical tokenization, stopword removal,
    and POS-guided lemmatization, with optional NLTK stemming support.

    Every input goes through these steps, each switchable in the config:

    1. Lowercasing                       (_prepare)
    2. Stopword removal                  (_tokens)
    3. Lemmatization                     (_tokens)
    4. Stemming                          (_tokens)

    Punctuation, whitespace and pure symbol tokens are always dropped, and
    the remaining tokens are collected in a set (no order, no duplicates).
    """

    config: NormalizerConfig

    nlp: Language

    # Only set when stemming is enabled.
    stemmer: PorterStemmer

    def __init__(
        self, config: str | os.PathLike | dict | NormalizerConfig = standard_directory
    ) -> None:
        """
        Load the configuration, the spaCy model and, if enabled, the stemmer.

        Parameters
        ----------
        config : str, os.PathLike, dict or NormalizerConfig, optional
            Where the settings come from:
            - a path to a YAML configuration file with a "normalizer"
              section (default: ./config/config.yaml relative to the
              working directory);
            - a dict, either the content of such a file (with "normalizer"
              and optionally "logger" keys) or just the normalizer settings
              themselves;
            - a NormalizerConfig.

            The settings are read-only afterwards (see NormalizerConfig).

            If no other module has configured the Logger yet, it is
            configured from the same source: the "logger" section of the
            file, or the "logger" key of a full configuration dict (an empty
            section means the logger defaults). Sources without logger
            settings leave the Logger to ./config/config.yaml if that file
            exists, and to its built-in defaults otherwise.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        KeyError
            If the file is not a mapping or has no "normalizer" section, or
            if the Logger falls back to an existing ./config/config.yaml
            without a "logger" section.
        pydantic.ValidationError
            If the normalizer or logger settings are not a mapping, or
            contain an unknown key or an invalid value.
        OSError
            If the spaCy model named in the settings is not installed.
        """
        self.config, logger_settings = Normalizer._resolveConfig(config)

        # A Logger configured by the caller is kept (see the docstring).
        # LoggerConfig.model_validate() rather than Logger(settings): it
        # rejects a section that is not a mapping (e.g. "logger: false")
        # with a ValidationError, where Logger() would take a string for a
        # path. The defaults let the normalizer work outside a project
        # directory.
        if Logger.config is None:
            if logger_settings is not None:
                Logger.config = LoggerConfig.model_validate(logger_settings)
            elif not os.path.isfile(standard_directory):
                Logger.config = LoggerConfig()
        log = Logger()

        # NER and parser are not needed; the tagger stays, because the
        # lemmatizer uses its part-of-speech tags.
        log.log(f"Loading spaCy model '{self.config.model_name}'...")
        self.nlp = spacy.load(self.config.model_name, exclude=["ner", "parser"])
        log.log(f"Loading spaCy model '{self.config.model_name}' completed.")

        # Without a lemmatizer every lemma is "", so the tokens are kept as
        # they are (see _tokens()).
        if self.config.lemmatizing and "lemmatizer" not in self.nlp.pipe_names:
            log.log(
                f"spaCy model '{self.config.model_name}' has no lemmatizer; "
                f"tokens are not lemmatized.",
                level="warning",
            )

        if self.config.stemming:
            log.log("Loading Porter stemmer...")
            self.stemmer = PorterStemmer()
            log.log("Loading Porter stemmer completed.")

    @staticmethod
    def _resolveConfig(
        config: str | os.PathLike | dict | NormalizerConfig,
    ) -> tuple[NormalizerConfig, object | None]:
        """
        Turn the `config` argument of __init__ into a NormalizerConfig.

        Also returns the "logger" section of a file or full configuration
        dict, for __init__ to configure the Logger: None if there is none,
        {} if it is empty, anything else as is for LoggerConfig to validate.
        """
        # NormalizerConfig is frozen, so it can be used without a copy.
        if isinstance(config, NormalizerConfig):
            return config, None

        if isinstance(config, dict):
            # "normalizer" is not a normalizer setting itself, so a full
            # configuration and bare settings cannot be confused.
            if configuration_section in config:
                return Normalizer._fromSections(config)
            return NormalizerConfig.model_validate(config), None

        path = os.fspath(config)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # An empty file is parsed as None, plain text as a str.
        if not isinstance(data, dict) or configuration_section not in data:
            raise KeyError(
                f"Config file '{path}' has no '{configuration_section}' section."
            )
        return Normalizer._fromSections(data)

    @staticmethod
    def _fromSections(data: dict) -> tuple[NormalizerConfig, object | None]:
        """
        Read the "normalizer" and "logger" sections of a full configuration.

        An empty section (parsed as None) means "use all defaults". Other
        non-mappings, such as false or a list, are passed on, so validation
        rejects them instead of silently using the defaults.
        """
        section = data[configuration_section]
        settings = NormalizerConfig.model_validate({} if section is None else section)

        if logger_section not in data:
            return settings, None
        logger_settings = data[logger_section]
        return settings, {} if logger_settings is None else logger_settings

    def normalize_batch(
        self,
        texts: Iterable[object],
        batch_size: int = 256,
        n_process: int = 1,
    ) -> list[set[str]]:
        """
        Normalize many terms at once using spaCy's batched pipeline.

        Same result as normalize() on every element, but much faster for
        large term lists.

        Parameters
        ----------
        texts : Iterable[object]
            The terms to normalize. Missing values (None, NaN, pandas.NA,
            NaT) are allowed and yield an empty set, like an empty string.
        batch_size : int, optional
            Number of terms spaCy processes per batch (default 256).
        n_process : int, optional
            Number of worker processes spaCy starts (default 1), or -1 for
            one per CPU. Each process loads its own copy of the model, so
            more processes only pay off for long lists.

        Returns
        -------
        list[set[str]]
            One token set per input term, in input order.

        Raises
        ------
        TypeError
            If `texts` is a single string, which would otherwise be
            normalized character by character.
        ValueError
            If `batch_size` is smaller than 1 (spaCy would silently return
            no results for 0), or `n_process` is neither -1 nor at least 1.
        """
        # spaCy gives no or only cryptic errors for these.
        if isinstance(texts, str):
            raise TypeError(
                "normalize_batch() expects an iterable of terms, not a single "
                "string; use normalize() for one term."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}.")
        if n_process != -1 and n_process < 1:
            raise ValueError(f"n_process must be -1 or at least 1, not {n_process}.")

        # Unlike normalize(), empty strings are not skipped: one doc per
        # input keeps the results in input order.
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
            The term to normalize, normally a str. Missing values (None,
            NaN, pandas.NA, NaT) are allowed and yield an empty set; other
            values are converted with str().

        Returns
        -------
        set[str]
            The normalized tokens. A set is returned because matching is
            meant to ignore word order and repeated words.
        """
        prepared = self._prepare(text)

        if not prepared:
            return set()
        return self._tokens(self.nlp(prepared))

    def _prepare(self, text: object) -> str:
        """Turn an input value into the string fed to spaCy.

        Missing values (e.g. empty cells from a pandas column) become ""
        instead of literal tokens such as "nan", "na" or "nat"; other
        non-strings are converted with str().
        """
        # Strings, by far the most common input, skip the missing-value check.
        if isinstance(text, str):
            result = text
        elif Normalizer._isMissing(text):
            return ""
        else:
            result = str(text)

        # 1. Lowercasing, before spaCy, because the tagger often tags
        #    capitalized words as proper nouns, which the lemmatizer keeps
        #    unchanged ("Kidneys" stays "Kidneys", "kidneys" -> "kidney").
        if self.config.lower_case:
            result = result.lower()
        return result

    @staticmethod
    def _isMissing(value: object) -> bool:
        """Tell whether a non-string input value stands for a missing value.

        Missing are None, NaN of any float type (float, numpy.float32/64),
        NaT (numpy.datetime64 and pandas) and pandas.NA.
        """
        if value is None:
            return True

        # pandas values only exist if pandas is imported, so it is never
        # imported here. pandas.isna also knows pandas.NA, which cannot be
        # compared (bool(NA != NA) raises); containers are checked
        # element-wise, so only scalars are passed.
        pandas = sys.modules.get("pandas")
        if pandas is not None:
            return bool(pandas.api.types.is_scalar(value) and pandas.isna(value))

        # NaN and NaT are the only values not equal to themselves. Arrays
        # compare element-wise and raise; they are not missing values.
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            return False

    def _isStopWord(self, token: Token) -> bool:
        """Tell whether a token is removed as a stop word.

        keep_stop_words and extra_stop_words match the lowercase text or
        lemma, so casing and word form do not matter ("abnormalities"
        matches "abnormality", "n't" matches "not").
        """
        forms = {token.lower_, token.lemma_.lower()}
        if forms & self.config.keep_stop_words:
            return False
        return token.is_stop or bool(forms & self.config.extra_stop_words)

    def _tokens(self, doc: Doc) -> set[str]:
        """Apply the configured token filters and transformations to a doc."""
        tokens: set[str] = set()

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            # Keeps single characters ('c' in "Hepatitis C", '1' in "Type 1
            # diabetes"), drops symbol tokens such as '+' or '%'.
            if not any(c.isalnum() for c in token.text):
                continue

            # 2. Stopword removal (spaCy's English list, adjusted by
            #    keep_stop_words and extra_stop_words).
            if self.config.stop_word_removal and self._isStopWord(token):
                continue

            # 3. Lemmatization. A token without a lemma (a model without a
            #    lemmatizer) keeps its text; "" would make all terms match.
            token_str = (token.lemma_ or token.text) if self.config.lemmatizing else token.text

            # 4. Stemming. PorterStemmer lowercases by default; to_lowercase
            #    keeps the casing when lower_case is False.
            if self.config.stemming:
                token_str = self.stemmer.stem(
                    token_str, to_lowercase=self.config.lower_case
                )

            tokens.add(token_str)

        return tokens
