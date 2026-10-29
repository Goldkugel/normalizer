# config

[`config.yaml`](./config.yaml) contains two sections:

- `normalizer` — every option of [`Normalizer`](../src/Normalizer.py), validated by [`NormalizerConfig`](../src/NormalizerConfig.py). Documented below.
- `logger` — settings of the [`logger`](https://github.com/Goldkugel/logger) package, which `Normalizer` uses for its progress messages. The logger requires this section; see the logger's own documentation for its keys.

When the normalizer is used inside a larger project (e.g. `hpo_sct_mapper`), the `normalizer` section is copied into that project's `config/config.yaml` alongside the other modules' sections.

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
| `stemming` | `bool` | `False` | Apply an NLTK `PorterStemmer` on top of the (lemmatized or raw) token. Only loaded/applied when set to `True`. Keeps the casing when `lower_case` is `False`. |
| `stop_word_removal` | `bool` | `True` | Drop tokens spaCy marks as stop words. This is spaCy's general English list, so it also removes words such as `no`, `not`, `without` and `back`. |

All keys are optional — omitted keys fall back to the defaults above, and an empty `normalizer:` section uses all defaults. The schema is validated with `extra="forbid"`, so any key not listed in this table (e.g. a typo) raises a `pydantic.ValidationError` at `Normalizer` construction time rather than being silently ignored.

## Loading

`Normalizer(config=path)` reads the YAML file at `path` (default: `./config/config.yaml`) and validates the `normalizer` key against `NormalizerConfig`; a file without a `normalizer` section raises a `KeyError`. Other top-level sections in the same file are ignored by `Normalizer`.

Note: the `logger` section is read by the `logger` package, which always loads `./config/config.yaml` relative to the working directory, regardless of the path passed to `Normalizer`.
