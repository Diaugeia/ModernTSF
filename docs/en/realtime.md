# Rolling real-time tracks

← [Documentation index](README.md)

Static benchmarks are fixed snapshots, so they cannot show how a model
generalizes to data that arrive after it was designed. Real-time tracks evaluate
every method in weekly **rounds** whose target data do not exist yet when the
forecasts are made.

## Tracks

Tracks are declared in `configs/realtime/<id>.toml`. Hourly tracks use a
168-step input and a 24-step horizon (a week of context, one forecast day, the
same as the PeMS and OpenAQ tracks); daily stock tracks use 20 and 5 trading
days. All hourly stamps are naive in the track's `tz` (UTC unless noted).

| Track | Source | Auth | License / terms | Bootstrap history | Frequency · horizon |
| --- | --- | --- | --- | --- | --- |
| `stock_hs300` | AKShare, forward-adjusted closes → daily log returns | none | vendor terms, research use | AKShare, from 2019-01-02 | trading days · 5 |
| `stock_nasdaq100` | Nasdaq API, then Yahoo chart API, Sina last resort → daily log returns | none | vendor terms, research use | Nasdaq API, from 2019-01-02 | trading days · 5 |
| `stock_sp500` | constituents from `datasets/s-and-p-500-companies` (Wikipedia fallback); closes as above | none | list ODC-PDDL; prices vendor terms | Nasdaq API, from 2019-01-02 | trading days · 5 |
| `traffic_pems_{ba,la,sac,sb}` | Caltrans PeMS clearinghouse `station_5min` (Districts 4, 7, 3, 8), summed to hourly flow | free PeMS account | Caltrans PeMS terms | UltraTraffic store (2019–2023) | hourly · 24 |
| `air_openaq_cn` | OpenAQ v3 hourly averages, PM2.5, China | free OpenAQ key | OpenAQ CC BY 4.0 (per-provider licenses) | API backfill (365 days) | hourly · 24 |
| `air_openaq_us` | OpenAQ v3 PM2.5, US reference monitors only | free OpenAQ key | as above | API backfill (365 days) | hourly · 24 |
| `air_openaq_eu` | OpenAQ v3 PM2.5, reference monitors in DE, FR, ES, IT, PL, GB, NL, AT, CZ, SE | free OpenAQ key | as above | API backfill (365 days) | hourly · 24 |
| `air_airnow_us` | EPA AirNow `HourlyData` file products, US PM2.5 (preliminary) | none | US government, public (preliminary data, not for regulatory use) | AirNow archive files, 56 days | hourly · 24 |
| `weather_openmeteo_temp` | Open-Meteo Historical Forecast API, 2 m temperature at 82 US and EU cities | none | CC BY 4.0, non-commercial free tier, attribution | same API, 365 days | hourly · 24 |
| `solar_openmeteo_ghi` | Open-Meteo, global horizontal irradiance (`shortwave_radiation`) at 55 PV-relevant sites | none | as above | same API, 365 days | hourly · 24 |
| `grid_ercot` | ERCOT hourly load, 8 weather zones: `Native_Load` archives + MIS report NP6-345-CD | none | ERCOT public data | `Native_Load` archives, from 2019 | hourly · 24 (America/Chicago) |
| `grid_eia_us` | EIA-930 hourly demand per US balancing authority | free EIA key | US government, public domain | API backfill (365 days) | hourly · 24 |
| `solar_eia_us` | EIA-930 hourly utility-scale solar generation per balancing authority | free EIA key | US government, public domain | API backfill (365 days) | hourly · 24 |

Credentials are read from the environment and never written to disk:

| Variable | Used by | Where to get it |
| --- | --- | --- |
| `PEMS_USER`, `PEMS_PASSWORD` | `traffic_pems_*` | free account at https://pems.dot.ca.gov |
| `OPENAQ_API_KEY` | `air_openaq_*` | free key at https://explore.openaq.org (account settings) |
| `EIA_API_KEY` | `grid_eia_us`, `solar_eia_us` | free key, instant, at https://www.eia.gov/opendata/ |

The stock, AirNow, Open-Meteo, and ERCOT tracks need no key. Notes per source:

- **Stocks.** Nasdaq and Yahoo both return split-adjusted, not dividend-adjusted,
  closes, so either serves a day identically; the Sina fallback is
  dividend-adjusted and can differ by the dividend yield on ex-dividend days.
  The Nasdaq historical endpoint answers only windows that end near the present,
  which is how updates and bootstraps call it. Each symbol is cached under
  `dataset/realtime/_cache/`, so an interrupted bootstrap resumes. Constituents
  are frozen at bootstrap.
- **Open-Meteo.** The Historical Forecast API stitches the analysis hours of the
  operational models, which is the same product the live forecast endpoint shows
  for recent hours, so history and weekly updates share one definition (the ERA5
  archive endpoint lags by days and differs slightly). The newest three hours
  are forecasts and are not stored. Temperature is its own track so all channels
  share one scale and unit; irradiance is a second track because it is the
  weather driver of PV output and needs no key. Free-tier limits (600 calls per
  minute, 5,000 per hour, 10,000 per day, counted per location and 14 days)
  allow the 365-day bootstrap in batches of 10 sites.
- **AirNow.** The files are preliminary and unvalidated (AQS holds the validated
  values months later); negative PM2.5 readings are raw instrument noise and
  are kept. Each file is about 0.75 MB, so the 8-week bootstrap is about 1 GB
  fetched once and then pulled from the Hub.
- **ERCOT.** Hour-ending values in local clock time are stored at the start of the
  hour they cover; the repeated hour of the autumn clock change is averaged. The
  archive is refreshed monthly and the daily report files stay listed for about a
  month, so a weekly update never leaves a gap. The ERCOT Public API
  (`api.ercot.com`, registration at https://apiexplorer.ercot.com plus a
  subscription key) offers deeper history and the solar and wind reports, whose
  public files keep only about a week; it is not used by default.
- **EIA-930.** Operator-reported values (a few outliers and gaps are normal);
  regional aggregates such as `US48` or `TEX` are excluded so channels do not
  overlap. Demand lags about an hour, solar generation about half a day.
- **OpenAQ.** `monitor_only` keeps government reference monitors and drops
  low-cost sensor networks; sensors are spread evenly over the listed countries.

A new track is bootstrapped once by a maintainer (`tsf realtime bootstrap --track T
--push`); the weekly workflow then pulls the store from the Hub and appends.

## Data releases

Each track has an append-only panel store (`dataset/realtime/<track>/`,
parquet by year plus a manifest). An update only adds new timestamps or fills
cells that were missing, never rewrites observed history, and records a release
with a content hash. Releases are mirrored to the Hugging Face dataset
`Diaugeia/TSFLab-RealTime` (owner overridable with `TSFLAB_HUB_OWNER`), one commit per release, so every round can be
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

The `realtime-weekly` workflow runs `python -m tsflab.realtime weekly` on a
CPU runner every Monday (update → score → open → baselines) and proposes the
results as a pull request. The real-time package imports no torch outside
`forecast`, so the job installs only `pydantic`, `pandas`, `pyarrow`, `requests`,
`huggingface_hub`, and `akshare` (only the Sina fallback and CSI-300 need it).

## Replaying history

`tsf realtime replay --track T --end DATE --weeks N [--models ...]` opens rounds
at past weekly cutoffs on a store that already contains the following weeks,
forecasts them (catalog models train only on data up to each cutoff), and scores
them immediately. Replayed rounds are written under `work_dirs/_realtime_replay/`
and never mix with live rounds.
