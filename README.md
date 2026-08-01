---

# Biomedical Lexical Normalizer

A high-performance, configurable lexical normalization pipeline tailored for biomedical text, ontologies, and clinical terminology matching (such as **HPO** and **SNOMED CT**).

Built on top of [scispacy](https://allenai.github.io/scispacy/) and [spaCy](https://spacy.io/), this tool provides POS-guided biomedical tokenization, lowercasing, stopword removal, lemmatization, and optional stemming while preserving crucial single-character clinical tokens (e.g., *Hepatitis C*, *Vitamin D*, *Aspirin C*).

---

## Features

* **Biomedical Tokenization:** Powered by `scispacy` (`en_core_sci_sm`) trained on scientific literature.
* **Single-Character Medical Term Preservation:** Retains critical single-letter clinical entities (*Hepatitis C*, *Blood Group A*, *Vitamin D*) that general-purpose NLP tokenizers often discard as noise.
* **Configurable Pipeline:** Easily toggle pipeline stages (lowercasing, stopword removal, lemmatization, stemming) via a YAML configuration file validated by **Pydantic**.
* **Ontology Alignment Ready:** Normalizes variant labels into comparable token representations for exact and set-based entity matching across ontologies like HPO and SNOMED CT.

---

## Project Structure

```text
.
├── config/
│   └── config.yaml          # Pipeline configuration file
├── src/
│   ├── NormalizerConfig.py  # Pydantic schema for config validation
│   ├── Normalizer.py        # Core BiomedicalLexicalNormalizer pipeline
│   └── NormalizerTest.py    # Pytest test suite
├── requirements.txt         # Project dependencies
└── README.md

```

---

## Installation

### 1. Prerequisites

Ensure you have **Python 3.10+** installed.

### 2. Install Dependencies

Install the required packages and download the `scispacy` biomedical model:

```bash
pip install spacy scispacy pydantic pyyaml nltk pytest
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

```

---

## Configuration

The pipeline behavior is controlled via a YAML file (`config/config.yaml`). The schema is strictly validated using Pydantic (`NormalizerConfig`).

### Standard Configuration (`config/config.yaml`)

```yaml
normalizer:
  model_name: "en_core_sci_sm"
  lower_case: True
  lemmatizing: True
  stemming: True
  stop_word_removal: True

```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"en_core_sci_sm"` | spaCy/scispacy model to load for tokenization & lemmatization. |
| `lower_case` | `bool` | `True` | Convert all input text to lower case. |
| `lemmatizing` | `bool` | `True` | Apply POS-guided lemmatization to base form. |
| `stemming` | `bool` | `False` | Apply NLTK Porter Stemmer to tokens. |
| `stop_word_removal` | `bool` | `True` | Remove domain- and language-specific stop words. |

---

## Quickstart & Usage

```python
from Normalizer import Normalizer

# Initialize normalizer using standard YAML config
normalizer = Normalizer(config="config/config.yaml")

# Example 1: HPO Term
hpo_term = "Abnormality of the kidney"
tokens_hpo = normalizer.normalize(hpo_term)
print("HPO Tokens:", tokens_hpo)
# Output: ['abnorm', 'kidney']

# Example 2: Medical single-character term preservation
drug_term = "Aspirin C"
tokens_drug = normalizer.normalize(drug_term)
print("Drug Tokens:", tokens_drug)
# Output: ['aspirin', 'c']

# Example 3: Set-based ontology alignment
snomed_term = "Kidney abnormalities"
tokens_snomed = normalizer.normalize(snomed_term)

# Check set-based match regardless of word order
is_match = set(tokens_hpo) == set(tokens_snomed)
print(f"HPO and SNOMED Match: {is_match}")
# Output: True

```

---

## Running Tests

The test suite is built with `pytest` and can be run directly as an executable Python script or via Pytest.

### Run with Python 3:

```bash
python3 src/NormalizerTest.py

```

### Run with Pytest:

```bash
pytest src/NormalizerTest.py -v

```