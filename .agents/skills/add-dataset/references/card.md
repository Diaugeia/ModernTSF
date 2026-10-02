# Writing the dataset card

Card layout and front matter keys are defined in `.agents/STANDARDS.md` (Dataset
cards). `tsf repo cards` creates `catalog/datasets/<name>/README.md` with `TODO`
placeholders and the generated runtime block; replace every placeholder with
verified facts.

- Measure length, channels, span, frequency, and missing values from the local file
  (read-only) and set `stats_basis: "measured"`.
- For facts you cannot measure, fetch the primary source (data page, paper,
  repository license) and label them `source-reported`; use `mixed` for both.
- Write `license: "unknown"` and `redistribution: "unknown"` when no explicit terms
  exist; never guess a license or a number.
- Document split conventions, `drop_last`, scaling, zeros or sentinels, and leakage
  risks under "Standard protocol and known pitfalls"; list sibling presets in
  `related`. A literature protocol that differs goes in `literature_protocol`.
- Verify with `uv run tsf catalog show <name>` and `uv run tsf data audit`.
