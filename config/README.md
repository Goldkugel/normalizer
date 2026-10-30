# config

[`config.yaml`](./config.yaml) contains two sections:

- `normalizer` — every option of [`Normalizer`](../src/Normalizer.py), validated by [`NormalizerConfig`](../src/NormalizerConfig.py). Documented below.
- `logger` — settings of the [`logger`](https://github.com/Goldkugel/logger) package, which `Normalizer` uses for its progress messages. See the logger's own documentation for its keys.

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
| `lower_case` | `bool` | `True` | Convert input text to lowercase before tokenization. Without it, capitalized words are often tagged as proper nouns and keep their form (`Kidneys` is not lemmatized to `Kidney`). |
| `lemmatizing` | `bool` | `True` | Reduce tokens to their POS-guided lemma instead of keeping raw text. |
| `stemming` | `bool` | `False` | Apply an NLTK `PorterStemmer` on top of the (lemmatized or raw) token. Only loaded/applied when set to `True`. Keeps the casing when `lower_case` is `False`. |
| `stop_word_removal` | `bool` | `True` | Drop tokens spaCy marks as stop words. This is spaCy's general English list, so it also removes words such as `no`, `not`, `without` and `back`. |

All keys are optional — omitted keys fall back to the defaults above, and an empty `normalizer:` section uses all defaults. The schema is validated with `extra="forbid"`, so any key not listed in this table (e.g. a typo) raises a `pydantic.ValidationError` at `Normalizer` construction time rather than being silently ignored.

## Loading

`Normalizer(config=path)` reads the YAML file at `path` (default: `./config/config.yaml`) and validates the `normalizer` key against `NormalizerConfig`; a file without a `normalizer` section raises a `KeyError`. Other top-level sections in the same file are ignored by `Normalizer`. Instead of a path, the same settings can be passed as a dict or a `NormalizerConfig` (see [Configuration sources](../README.md#configuration-sources)).

If the logger has not been configured yet (by another module of the calling pipeline), `Normalizer` configures it from the `logger` section of the same file. An existing logger configuration is kept. If the file has no `logger` section and the logger is not configured yet, the logger falls back to its default `./config/config.yaml` relative to the working directory, or to its built-in defaults if that file does not exist.
