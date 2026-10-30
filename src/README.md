# src

Source code for the biomedical lexical normalization pipeline. See the [top-level README](../README.md) for installation, configuration, and usage instructions.

## Modules

| File | Purpose |
| --- | --- |
| [`__init__.py`](./__init__.py) | Package entry point; exposes `normalizer.Normalizer` and `normalizer.NormalizerConfig`. |
| [`Normalizer.py`](./Normalizer.py) | Core pipeline. Loads a spaCy/scispacy model and an optional NLTK `PorterStemmer`, then exposes `normalize(text)` and `normalize_batch(texts)` to turn raw text into a token `set` (lowercasing, tokenization, stopword removal, lemmatization, stemming — each stage toggled via `NormalizerConfig`). Both go through the same two private helpers — `_prepare` (missing-value handling and lowercasing) and `_tokens` (token filtering, stop word removal, lemmatization, stemming) — so they always return the same result; missing values (`None`, `NaN`, `NaT`, `pandas.NA`, detected by `_isMissing`) give an empty set. The constructor accepts a YAML path, a dict or a `NormalizerConfig` (resolved by `_resolveConfig`) and configures the shared logger if no other module has done so. |
| [`NormalizerConfig.py`](./NormalizerConfig.py) | Pydantic schema (`NormalizerConfig`) validating the `normalizer` section of `config/config.yaml` or the equivalent dict. Rejects unknown keys (`extra="forbid"`) to catch typos early, and is read-only (`frozen=True`), since the model and the stemmer are loaded once from it. |

The test suite lives in [`tests/NormalizerTest.py`](../tests/NormalizerTest.py), outside the package. It covers config validation, default pipeline behavior, per-flag toggles (lowercasing, lemmatization, stemming, stopword removal), digit, symbol and hyphen handling, missing-value handling, batch/single consistency (also across batches and worker processes), model loading, the shipped `config/config.yaml`, the accepted configuration sources and their errors, logger configuration and fallbacks, the package exports, and set-based HPO/SNOMED token alignment.

## API at a glance

```python
from normalizer import Normalizer

normalizer = Normalizer(config="config/config.yaml")

normalizer.normalize("Abnormality of the kidney")
# -> {"abnormality", "kidney"}

normalizer.normalize_batch(["Abnormality of the kidney", "Kidney abnormalities"])
# -> [{"abnormality", "kidney"}, {"kidney", "abnormality"}]

# With stemming: True in the config, both would be {"abnorm", "kidney"}.
```

Both methods return `set[str]` (per input string for `normalize_batch`), since the pipeline is designed for order-independent, deduplicated ontology term matching rather than sequence-preserving tokenization.
