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
  keep_stop_words: []
  extra_stop_words: []
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"en_core_sci_sm"` | spaCy/scispacy model to load for tokenization and lemmatization. |
| `lower_case` | `bool` | `True` | Convert input text to lowercase before tokenization. Without it, capitalized words are often tagged as proper nouns and keep their form (`Kidneys` is not lemmatized to `Kidney`). |
| `lemmatizing` | `bool` | `True` | Reduce tokens to their POS-guided lemma instead of keeping raw text. |
| `stemming` | `bool` | `False` | Apply an NLTK `PorterStemmer` on top of the (lemmatized or raw) token. Only loaded/applied when set to `True`. Keeps the casing when `lower_case` is `False`. |
| `stop_word_removal` | `bool` | `True` | Drop tokens spaCy marks as stop words. This is spaCy's general English list, so it also removes words such as `no`, `not`, `without` and `back`. |
| `keep_stop_words` | `list[str]` | `[]` | Words that are never removed, e.g. `["no", "not", "without", "back"]`. Only used with `stop_word_removal: True`. |
| `extra_stop_words` | `list[str]` | `[]` | Words that are removed in addition to spaCy's list. Only used with `stop_word_removal: True`. |

A token matches an entry of `keep_stop_words` or `extra_stop_words` if its text or its lemma does, ignoring the casing, so `Back` matches `back`, `abnormalities` matches `abnormality` and `n't` matches `not`. Entries must be single words (an entry such as `of the` could never match a token), and a word must not be in both lists.

All keys are optional — omitted keys fall back to the defaults above, and an empty `normalizer:` section uses all defaults. A section that is not a mapping (e.g. `normalizer: false`) raises a `pydantic.ValidationError`. The schema is validated with `extra="forbid"`, so any key not listed in this table (e.g. a typo) raises a `pydantic.ValidationError` at `Normalizer` construction time rather than being silently ignored.

## Loading

`Normalizer(config=path)` reads the YAML file at `path` (default: `./config/config.yaml`) and validates the `normalizer` key against `NormalizerConfig`; a file that is not a mapping (e.g. empty, plain text or a list) or has no `normalizer` section raises a `KeyError`. Other top-level sections in the same file are ignored by `Normalizer`. Instead of a path, the same settings can be passed as a dict or a `NormalizerConfig` (see [Configuration sources](../README.md#configuration-sources)).

If the logger has not been configured yet (by another module of the calling pipeline), `Normalizer` configures it from the `logger` section of the same file; an empty `logger:` section means the logger's defaults. An existing logger configuration is kept. If the file has no `logger` section and the logger is not configured yet, the logger falls back to its default `./config/config.yaml` relative to the working directory, or to its built-in defaults if that file does not exist.
