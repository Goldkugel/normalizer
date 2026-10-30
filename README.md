---

# Biomedical Lexical Normalizer

A configurable lexical normalization pipeline tailored for biomedical text, ontologies, and clinical terminology matching (such as **HPO** and **SNOMED CT**).

Built on top of [scispacy](https://allenai.github.io/scispacy/) and [spaCy](https://spacy.io/), this tool provides biomedical tokenization, lowercasing, stopword removal, POS-guided lemmatization, and optional stemming while preserving single-character clinical tokens (e.g., *Hepatitis C*, *Vitamin D*, *Type 1 diabetes*). Each term is turned into an unordered set of tokens, so terms can be compared regardless of word order.

---

## Features

* **Biomedical Tokenization:** Powered by `scispacy` (`en_core_sci_sm`) trained on scientific literature.
* **Single-Character Term Preservation:** Retains single-letter and single-digit tokens (*Hepatitis C*, *Vitamin D*, *Type 1 diabetes*) that general-purpose pipelines often discard as noise, so e.g. *Type 1* and *Type 2 diabetes* stay distinguishable.
* **Missing-Value Safe:** Missing values — `None`, `NaN` of any float type, `NaT` and `pandas.NA` (e.g. empty pandas cells, also in columns of the `"string"` dtype) — normalize to an empty set instead of spurious tokens such as `"nan"` or `"na"`. pandas is not a dependency; its values are recognized if it is installed.
* **Configurable Pipeline:** Easily toggle pipeline stages (lowercasing, stopword removal, lemmatization, stemming) via a YAML configuration file, a dict or a `NormalizerConfig`, all validated by **Pydantic**.
* **Ontology Alignment Ready:** Normalizes variant labels into comparable token representations for exact and set-based entity matching across ontologies like HPO and SNOMED CT.

---

## Project Structure

```text
.
├── config/
│   ├── config.yaml          # Configuration file (normalizer and logger sections)
│   └── README.md            # Documentation of the normalizer section
├── data/
│   └── logs/                # Log files written by the logger
├── src/
│   ├── __init__.py          # Package entry point (exposes Normalizer, NormalizerConfig)
│   ├── NormalizerConfig.py  # Pydantic schema for config validation
│   ├── Normalizer.py        # Core Normalizer pipeline
│   └── README.md            # Module overview
├── tests/
│   └── NormalizerTest.py    # Pytest test suite (not part of the installed package)
├── .github/workflows/
│   └── tests.yml            # CI: tests with coverage on Python 3.9, 3.11 and 3.12, ruff, mypy
├── LICENSE
├── pyproject.toml           # Package metadata, dependencies, ruff and mypy settings
└── README.md

```

---

## Installation

### 1. Prerequisites

Ensure you have **Python 3.9+** installed.

### 2. Install the Package

All dependencies, including the `en_core_sci_sm` scispacy model, are declared in `pyproject.toml`:

```bash
pip install "normalizer @ git+https://github.com/Goldkugel/normalizer.git@v1.0.6"
```

For development, install from a local checkout together with the test dependencies:

```bash
pip install -e ".[test]"
```

---

## Configuration

The pipeline behavior is controlled via a YAML file (`config/config.yaml`) or the equivalent settings passed in code (see [Configuration sources](#configuration-sources)). The schema is strictly validated using Pydantic (`NormalizerConfig`): unknown keys raise a `ValidationError`, and omitted keys use the defaults below.

### Standard Configuration (`config/config.yaml`)

```yaml
normalizer:
  model_name: "en_core_sci_sm"
  lower_case: True
  lemmatizing: True
  stemming: False
  stop_word_removal: True
  keep_stop_words: []
  extra_stop_words: []
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"en_core_sci_sm"` | spaCy/scispacy model to load for tokenization & lemmatization. |
| `lower_case` | `bool` | `True` | Convert all input text to lower case before tokenization (capitalized words are otherwise often tagged as proper nouns and not lemmatized). |
| `lemmatizing` | `bool` | `True` | Apply POS-guided lemmatization to base form. |
| `stemming` | `bool` | `False` | Apply NLTK Porter Stemmer to tokens (respects `lower_case`). |
| `stop_word_removal` | `bool` | `True` | Remove words on spaCy's general English stop word list (see [Known Limitations](#known-limitations)). |
| `keep_stop_words` | `list[str]` | `[]` | Words that are never removed, e.g. `["no", "not", "without", "back"]`. |
| `extra_stop_words` | `list[str]` | `[]` | Words that are removed in addition to spaCy's list, e.g. filler words of ontology labels. |

`keep_stop_words` and `extra_stop_words` only take effect with `stop_word_removal: True`. A token matches an entry if its text or its lemma does, ignoring the casing (`Back` matches `back`, `abnormalities` matches `abnormality`, `n't` matches `not`). Entries must be single words, and a word must not be in both lists; otherwise a `pydantic.ValidationError` is raised.

---

## Quickstart & Usage

```python
from normalizer import Normalizer

# Initialize normalizer using the standard YAML config (stemming disabled)
normalizer = Normalizer(config="config/config.yaml")

# Example 1: HPO Term
tokens_hpo = normalizer.normalize("Abnormality of the kidney")
print("HPO Tokens:", tokens_hpo)
# Output: {'abnormality', 'kidney'}

# Example 2: Single-character term preservation
print(normalizer.normalize("Hepatitis C infection"))
# Output: {'hepatitis', 'c', 'infection'}

# Example 3: Set-based ontology alignment, regardless of word order
tokens_snomed = normalizer.normalize("Kidney abnormalities")
print(f"HPO and SNOMED Match: {tokens_hpo == tokens_snomed}")
# Output: True

# Example 4: Batch normalization (missing values give an empty set)
print(normalizer.normalize_batch(["Kidney abnormalities", None]))
# Output: [{'kidney', 'abnormality'}, set()]
```

`normalize_batch(texts, batch_size=256, n_process=1)` expects an iterable of terms (a list, a pandas column, a generator); a single string raises a `TypeError` instead of being normalized character by character. `batch_size` must be at least 1 and `n_process` either -1 (one process per CPU) or at least 1, otherwise a `ValueError` is raised.

### Configuration sources

Besides a YAML path, `Normalizer` accepts a dict or a `NormalizerConfig`:

```python
from normalizer import Normalizer, NormalizerConfig

# From a YAML file (default: "./config/config.yaml" relative to the
# working directory). A pathlib.Path works as well.
normalizer = Normalizer("config/config.yaml")

# From a dict: either the whole configuration (with "normalizer" and
# optionally "logger" keys) or just the normalizer settings.
normalizer = Normalizer({"normalizer": {"stemming": True}, "logger": {"folder": "./logs/"}})
normalizer = Normalizer({"stemming": True})

# From a NormalizerConfig.
normalizer = Normalizer(NormalizerConfig(stemming=True))
```

The settings are read-only (`normalizer.config.stemming = True` raises a `pydantic.ValidationError`), since the model and the stemmer are loaded only once; create a new `Normalizer` for other settings. A variant of a `NormalizerConfig` can be made with `config.model_copy(update={"stemming": True})`.

`Normalizer` logs through the shared `logger` package. If no other module has configured the logger yet, `Normalizer` configures it from the same source: the `logger` section of the config file, or the `logger` key of a full configuration dict (an empty section means the logger's defaults); otherwise the existing logger configuration is kept. With the shipped config, log messages are appended to `./data/logs/output.log` (the folder is created if missing).

---

## Known Limitations

* **Stop words are not domain-specific.** `stop_word_removal` uses spaCy's general English stop word list, which also contains words that carry meaning in clinical terms, e.g. negation (`no`, `not`, `without`), location (`back`, `front`, `side`, `above`, `below`) and numbers (`first`, `two`). For example, *Back pain* normalizes to `{'pain'}` and *No fever* to `{'fever'}` with the defaults. List such words in `keep_stop_words` to keep them (*Back pain* → `{'back', 'pain'}`).
* **Logger configuration.** The logger is a process-wide singleton, so only the first module that creates it decides its configuration. If `Normalizer` is created first and its configuration has no `logger` section (or is given as normalizer settings only or as a `NormalizerConfig`), the logger falls back to `./config/config.yaml` relative to the working directory, or to its built-in defaults (logging to `./data/logs/output.log`) if that file does not exist.

---

## Running Tests

The test suite in `tests/` is only meant for testing this repository and is not part of the installed package. It tests the installed `normalizer` package, so install it first, then run pytest from the project root:

```bash
pip install -e ".[test]"
pytest -v
```

`pyproject.toml` points pytest at every `tests/*Test.py` file, so `pytest` without arguments runs the whole suite, and new test files are picked up by CI as well. A single file can also be run directly with `python3 tests/NormalizerTest.py`.

To also see which lines and branches of the package the tests cover:

```bash
pytest --cov=normalizer --cov-branch --cov-report=term-missing
```

The tests write their own temporary config and log files and do not touch `data/logs/`; `config/config.yaml` is only read, to check that it is valid and matches the documented defaults. The `test` extra installs `pytest`, `pytest-cov` and `pandas` (used to build pandas missing values); `pandas` is limited to version 2, since pandas 3 needs numpy 2, which spaCy 3.7 does not support.

### Linting

The code is checked with [ruff](https://docs.astral.sh/ruff/) and [mypy](https://mypy-lang.org/); both are configured in `pyproject.toml`:

```bash
pip install -e ".[lint]"
ruff check .
mypy
```

GitHub Actions runs the tests (with a coverage report on the run's summary page) and both linters on every push and pull request (`.github/workflows/tests.yml`).