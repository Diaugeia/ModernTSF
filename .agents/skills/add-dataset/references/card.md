# Writing the dataset card

The card format is in `.agents/STANDARDS.md` (Cards); copy the layout of a sibling
card (`uv run tsf catalog show <sibling> --depth 3` prints its paths). A card is
`catalog/datasets/<name>/card.toml` (facts), `README.md` (front matter `name` and
`description`, then Overview and Protocol and pitfalls, <= 60 lines), and an
optional `reference.md` for long provenance or statistics. Loader, files, task
modes, and parameters are derived from code; never copy them into the card.

- `card.toml`: `domain`, `tags`, `related`, `[source]` (name, url, citation,
  citation_url, license, redistribution), `[shape]` (frequency, time_span, length,
  channels, channel_kind, target, missing_values, stats_basis), `[protocol]`
  (protocol, literature, split, seq_lens, pred_lens).
- Measure length, channels, span, frequency, and missing values from the local file
  (read-only) and set `stats_basis = "measured"`. For facts you cannot measure,
  fetch the primary source and label them `source-reported`; use `mixed` for both.
- Record `characteristics` with `uv run tsf data analyze <preset> --write-card`
  (data terms from the train-only profile, with `characteristics_basis`); add task
  terms (`spatial-graph`, `calendar-effects`, `exogenous-covariates`, ...) by hand
  only when the loader or source supports them.
- Write `license = "unknown"` and `redistribution = "unknown"` when no explicit
  terms exist; never guess a license or a number.
- Document split conventions, `drop_last`, scaling, zeros or sentinels, shift, and
  leakage risks under Protocol and pitfalls; a differing literature protocol goes
  in `[protocol].literature`.
- Verify with `uv run tsf catalog show <name>`, `uv run tsf data audit`, and
  `uv run tsf catalog match <name>`.
