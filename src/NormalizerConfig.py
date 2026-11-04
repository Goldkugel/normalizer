"""
Pydantic model for the settings of the Normalizer.

Validates the "normalizer" section of the YAML config file, or the
equivalent dict, before the Normalizer loads its spaCy model.
"""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

class NormalizerConfig(BaseModel):
    """Pydantic model defining and validating the "normalizer" config section.

    Selects the spaCy model and the active steps (lowercasing,
    lemmatization, stemming, stopword removal). All keys are optional.
    Instances are read-only; use model_copy(update={...}) for a variant.
    """

    # extra="forbid" catches typos and outdated keys. Frozen, because the
    # model and stemmer are loaded once: a later change would have no effect
    # (model_name) or break normalize() (stemming).
    model_config            = ConfigDict(extra="forbid", frozen=True)

    # spaCy / scispacy model; must be installed. Only the default is a
    # dependency of this package (see pyproject.toml).
    model_name: str         = "en_core_sci_sm"

    # Also keeps the Porter stemmer from lowercasing when False.
    lower_case: bool        = True

    # POS-guided lemmatization.
    lemmatizing: bool       = True

    # NLTK Porter stemming on top of the lemma (or raw token). Off by
    # default: stems such as "abnorm" or "sever" also merge unrelated words.
    stemming: bool          = False

    # spaCy's general English list, which includes words that matter in
    # clinical terms, such as "no", "not", "without" and "back".
    stop_word_removal: bool = True

    # Adjust the stop word list (only with stop_word_removal): keep_stop_words
    # are never removed, extra_stop_words in addition. Entries match a
    # token's text or lemma, ignoring case. Frozensets keep the settings
    # read-only; YAML lists are converted.
    keep_stop_words: frozenset[str]  = frozenset()
    extra_stop_words: frozenset[str] = frozenset()

    @field_validator("keep_stop_words", "extra_stop_words")
    @classmethod
    def check_stop_words(cls, words: frozenset[str]) -> frozenset[str]:
        """
        Lowercase the entries and reject those that can never match a token.

        Tokens are single words, so an empty entry or one with whitespace
        (e.g. "of the") would silently have no effect.
        """
        for word in words:
            if not word or any(c.isspace() for c in word):
                raise ValueError(
                    f"Stop word entries must be single words, not {word!r}."
                )
        return frozenset(word.lower() for word in words)

    @model_validator(mode="after")
    def check_stop_word_overlap(self) -> "NormalizerConfig":
        """Reject words that are both kept and removed, which is ambiguous."""
        both = self.keep_stop_words & self.extra_stop_words
        if both:
            raise ValueError(
                "Words in both keep_stop_words and extra_stop_words: "
                f"{sorted(both)}."
            )
        return self
