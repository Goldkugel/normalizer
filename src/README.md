# src

Source code for the biomedical lexical normalization pipeline. See the [top-level README](../README.md) for installation, configuration, and usage instructions.

## Modules

| File | Purpose |
| --- | --- |
| [`__init__.py`](./__init__.py) | Package entry point; exposes `Normalizer` as `normalizer.Normalizer`. |
| [`Normalizer.py`](./Normalizer.py) | Core pipeline. Loads a spaCy/scispacy model and an optional NLTK `PorterStemmer`, then exposes `normalize(text)` and `normalize_batch(texts)` to turn raw text into a token `set` (lowercasing, tokenization, stopword removal, lemmatization, stemming — each stage toggled via `NormalizerConfig`). |
| [`NormalizerConfig.py`](./NormalizerConfig.py) | Pydantic schema (`NormalizerConfig`) validating the `normalizer` section of `config/config.yaml`. Rejects unknown keys (`extra="forbid"`) to catch typos early. |
| [`NormalizerTest.py`](./NormalizerTest.py) | Pytest suite covering config validation, default pipeline behavior, per-flag toggles (lowercasing, lemmatization, stemming, stopword removal), and set-based HPO/SNOMED token alignment. |

## API at a glance

```python
from normalizer import Normalizer

normalizer = Normalizer(config="config/config.yaml")

normalizer.normalize("Abnormality of the kidney")
# -> {"abnorm", "kidney"}

normalizer.normalize_batch(["Abnormality of the kidney", "Kidney abnormalities"])
# -> [{"abnorm", "kidney"}, {"kidney", "abnorm"}]
```

Both methods return `set[str]` (per input string for `normalize_batch`), since the pipeline is designed for order-independent, deduplicated ontology term matching rather than sequence-preserving tokenization.
