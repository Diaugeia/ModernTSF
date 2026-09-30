# Rolling real-time tracks

← [Documentation index](README.md)

Static benchmarks are fixed snapshots, so they cannot show how a model
generalizes to data that arrive after it was designed. Real-time tracks evaluate
every method in weekly **rounds** whose target data do not exist yet when the
forecasts are made.

## Tracks

Tracks are declared in `configs/realtime/<id>.toml`:

| Track | Source | Bootstrap history | Frequency · horizon |
| --- | --- | --- | --- |
| `stock_hs300` | AKShare, forward-adjusted closes → daily log returns | AKShare, from 2019-01-02 | trading days · 5 |
| `stock_nasdaq100` | Sina (via AKShare) forward-adjusted closes, Nasdaq API fallback → daily log returns | source, from 2019-01-02 | trading days · 5 |
| `traffic_pems_{ba,la,sac,sb}` | Caltrans PeMS clearinghouse `station_5min` (Districts 4, 7, 3, 8), summed to hourly flow | UltraTraffic store (2019–2023) | hourly · 24 |
| `air_openaq_cn` | OpenAQ v3 hourly averages, PM2.5 | API backfill (365 days) | hourly · 24 |

Credentials are read from the environment and never written to disk:
`PEMS_USER` / `PEMS_PASSWORD` (free PeMS account) and `OPENAQ_API_KEY` (free
OpenAQ key). The stock tracks need no key. The NASDAQ-100 fallback closes are
split- but not dividend-adjusted, so a day served by it can differ from the
primary source by the dividend yield. Stock fetches cache each symbol under
`dataset/realtime/_cache/`, so an interrupted bootstrap resumes.

## Data releases

Each track has an append-only panel store (`dataset/realtime/<track>/`,
parquet by year plus a manifest). An update only adds new timestamps or fills
cells that were missing, never rewrites observed history, and records a release
with a content hash. Releases are mirrored to the Hugging Face dataset
`Diaugeia/ModernTSF-RealTime` (owner overridable with `MODERNTSF_HUB_OWNER`), one commit per release, so every round can be
reproduced from a pinned revision.

## Rounds

`tsf realtime open` creates a round on the latest release:

- `cutoff` — the last observed timestamp;
- `target_timestamps` — `horizon` steps that start after a submission window
  (`submission_hours`), so forecasts always precede their truth;
- `deadline` — one second before the first target;
- `norm_mean` / `norm_std` — per-channel statistics frozen at the cutoff, used to
  z-score errors so that channels with different scales are comparable.

Rounds, forecasts, and scores live in
`apps/web/submissions/realtime/<track>/rounds/<round_id>/` as reviewable
evidence. Every stored instant carries its timezone offset.

## Forecasts

A forecast is a `ForecastSubmission` (`predictions` of shape
`(horizon, channels)` in raw units). Three sources fill a round:

- **Baselines** (`Naive`, `SeasonalNaive`, `WindowMean`) are submitted by the
  weekly workflow on CPU, so every round has anchors.
- **Catalog models**: `tsf realtime forecast --track T --model M` exports the
  round's history in the index-windowed `cauair` layout, trains and early-stops
  through the standard runner, and forecasts the window at the cutoff.
  Time-series models see stations as channels; spatiotemporal models also get
  calendar covariates. Needs the full install; a GPU is recommended.
- **Anyone** can open a pull request adding `forecasts/<Model>.json`. CI rejects
  it if the pull request was last updated after the deadline.

## Scoring and stability

Once a later release covers at least `min_coverage` of the target cells,
`tsf realtime score` computes MSE and MAE over the observed cells and ranks the
forecasts. `apps/web/data/realtime/<track>.json` summarizes mean rank and mean
MSE per method over scored rounds, plus rank stability: Kendall's tau between
the rankings of consecutive rounds.

## Weekly automation

The `realtime-weekly` workflow runs `python -m moderntsf.realtime weekly` on a
CPU runner every Monday (update → score → open → baselines) and proposes the
results as a pull request. The real-time package imports no torch outside
`forecast`, so the job installs only `pydantic`, `pandas`, `pyarrow`, `requests`,
`huggingface_hub`, and `akshare`.

## Replaying history

`tsf realtime replay --track T --end DATE --weeks N [--models ...]` opens rounds
at past weekly cutoffs on a store that already contains the following weeks,
forecasts them (catalog models train only on data up to each cutoff), and scores
them immediately. Replayed rounds are written under `work_dirs/_realtime_replay/`
and never mix with live rounds.
