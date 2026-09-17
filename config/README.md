# config

[`config.yaml`](./config.yaml) is a shared configuration file with sections for several modules (`adapter`, `logger`, `llm`, ...). This README documents only the `normalizer` section, consumed by [`Normalizer`](../src/Normalizer.py) via [`NormalizerConfig`](../src/NormalizerConfig.py).

## `normalizer` section

```yaml
normalizer:
  model_name: "en_core_sci_sm"
  lower_case: True
  lemmatizing: True
  stemming: False
  stop_word_removal: True
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"en_core_sci_sm"` | spaCy/scispacy model to load for tokenization and lemmatization. |
| `lower_case` | `bool` | `True` | Convert input text to lowercase before tokenization. |
| `lemmatizing` | `bool` | `True` | Reduce tokens to their POS-guided lemma instead of keeping raw text. |
| `stemming` | `bool` | `False` | Apply an NLTK `PorterStemmer` on top of the (lemmatized or raw) token. Only loaded/applied when set to `True`. |
| `stop_word_removal` | `bool` | `True` | Drop tokens spaCy marks as stop words. |

All keys are optional — omitted keys fall back to the defaults above. The schema is validated with `extra="forbid"`, so any key not listed in this table (e.g. a typo) raises a `pydantic.ValidationError` at `Normalizer` construction time rather than being silently ignored.

## Loading

`Normalizer(config=path)` reads the YAML file at `path` (default: `./config/config.yaml`) and validates the `normalizer` key against `NormalizerConfig`. Other top-level sections in the same file (e.g. `adapter`, `logger`) are ignored by `Normalizer`.
