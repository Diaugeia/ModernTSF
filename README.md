<div align="center">

# 🚀 TSFLab

**A fully automated, continuously updated platform for time series forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models: 199](https://img.shields.io/badge/models-199-orange.svg)](docs/en/models.md)
[![Real-time tracks: 16](https://img.shields.io/badge/real--time%20tracks-16-purple.svg)](docs/en/realtime.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Every forecasting method, one interface, one protocol, evaluated on the data
we already have **and** on data that did not exist when the method was written.

</div>

> 🧪 **Latest features land on the [`dev`](https://github.com/Diaugeia/TSFLab/tree/dev) branch first.** `main` is the stable, versioned release line.

---

## 🧭 Why TSFLab

Forecasting papers multiply every year, yet each one can compare against only a
few baselines, re-run under its own code and data conventions. Keeping hundreds
of methods in one benchmark by hand, verifying their code, and re-evaluating them
on new data no longer scales, and fixed benchmark snapshots cannot show how a
model holds up as the world moves on.

TSFLab automates that loop. Coding agents read new papers, implement them
behind one verified interface, and evaluate them under one protocol on static
datasets and on **rolling real-time tracks** that refresh every week. TSFLab
ships no agent of its own: it is the infrastructure (catalog, contracts, data,
protocols, evidence) that any coding agent operates through declarative skills.

---

## ✨ What is inside

| Module | What it does |
| --- | --- |
| 📚 **Paper reading** | Scans arXiv and Hugging Face Papers, deduplicates against the catalog, and records each paper's structure, equations, and pinned official code |
| 🧩 **Code & interface** | 199 methods as peers in one flat catalog, composed from 48 shared components, one forecasting signature, a 13-check verification battery with pinned-reference comparison |
| 🗃️ **Data** | 84 static presets across `time_series`, `spatiotemporal`, and `covariate` settings (incl. PeMS traffic for four Caltrans districts, 2003–2023), plus rolling real-time tracks (stocks, traffic, air quality) |
| ⚙️ **Experiments** | Declarative TOML sweeps, pre-run validation, seeds, budgets, GPU leases, queues, and recovery |
| 🏆 **Release & compare** | Run records → submissions → a leaderboard recomputed from evidence; weights as pinned `hf://` bundles |

---

## 🏁 Quick start

**Work in the repository with an agent:**

```bash
git clone https://github.com/Diaugeia/TSFLab.git
cd TSFLab
codex          # or any other coding agent
```

```text
> Set up the environment for my GPU.
> Benchmark DLinear, PatchTST and iTransformer on ETTh1 and give me a leaderboard.
> Implement the paper at <arXiv URL> as a catalog model and verify it.
> Forecast this week's traffic round with PatchTST and submit it.
```

**Or install the framework and scaffold your own project:**

```bash
uv tool install "git+https://github.com/Diaugeia/TSFLab"   # provides `tsf`
tsf init my-forecasting-project && cd my-forecasting-project
tsf dataset download etth1          # pinned, checksum-verified from the Hub
tsf inspect --config configs/runs/example.toml
tsf run configs/runs/example.toml
```

Scaffolded run configs inherit the installed catalog through `tsflab://`
paths, so upgrading TSFLab upgrades their defaults.

**Direct CLI discovery:**

```bash
uv run tsf model list --details
uv run tsf dataset list
uv run tsf catalog search "reversible normalization"   # one line per match
uv run tsf component show revin --depth 1             # 0 line, 1 interface, 2 card, 3 paths
uv run tsf realtime list
uv run tsf agent task list
```

---

## 📈 Rolling real-time evaluation

Each week the `realtime-weekly` workflow releases new observations (versioned on
the Hugging Face Hub), scores the rounds whose target window is now observed, and
opens a new round. Forecasts must be submitted **before** their targets exist, so
no model, including ours, can have seen its evaluation data.

| Track | Data | Setting | Horizon |
| --- | --- | --- | --- |
| `stock_hs300` | CSI-300 constituents, daily log returns (AKShare) | time series | 5 trading days |
| `stock_nasdaq100` | NASDAQ-100 constituents, daily log returns (Nasdaq API, Yahoo fallback) | time series | 5 trading days |
| `stock_sp500` | S&P 500 constituents, daily log returns (Nasdaq API, Yahoo fallback) | time series | 5 trading days |
| `traffic_pems_{ba,la,sac,sb}` | Caltrans PeMS Districts 4, 7, 3, 8: hourly flow at 2,472 / 1,926 / 801 / 1,105 stations | spatiotemporal | 24 h |
| `air_openaq_cn` | OpenAQ hourly PM2.5, government monitors in China | spatiotemporal | 24 h |
| `air_openaq_{us,eu}` | OpenAQ hourly PM2.5, US / European reference monitors | spatiotemporal | 24 h |
| `air_airnow_us` | EPA AirNow hourly PM2.5, US monitors (no key) | spatiotemporal | 24 h |
| `weather_openmeteo_temp`, `solar_openmeteo_ghi` | Open-Meteo hourly temperature at 82 US/EU cities, irradiance at 55 PV sites | spatiotemporal | 24 h |
| `grid_ercot` | ERCOT hourly load in 8 weather zones | spatiotemporal | 24 h |
| `grid_eia_us`, `solar_eia_us` | EIA-930 hourly demand / solar generation per US balancing authority | spatiotemporal | 24 h |

```bash
uv run tsf realtime forecast --track traffic_pems_sb --model DLinear   # produce a forecast
uv run tsf realtime replay --track traffic_pems_sb --end 2023-12-25 --weeks 12   # backtest the protocol
```

See [docs/en/realtime.md](docs/en/realtime.md).

---

## 🤝 Contributing

There are three ways to take part, all reviewed through pull requests:

1. **Propose a method** — open a *Submit a new model* issue. Once a maintainer
   approves it, the `paper-intake` workflow has a coding agent implement and
   verify it and opens a pull request.
2. **Submit results** — add a `submission.json` under `apps/web/submissions/`
   (see [SUBMITTING.md](apps/web/SUBMITTING.md)); CI validates it against the
   contract before it can reach the leaderboard.
3. **Forecast a real-time round** — add `forecasts/<YourModel>.json` to an open
   round before its deadline.

The literature is also scanned weekly by the `paper-intake` workflow. See
[CONTRIBUTING.md](CONTRIBUTING.md) for code contributions.

---

## 📖 Documentation

- [Workflow documentation](docs/en/README.md): models, data, verification, experiments
- [Projects and the Hub](docs/en/hub.md): `tsf init`, `hf://` assets, weights bundles
- [Real-time tracks](docs/en/realtime.md): rounds, forecasts, scoring, weekly automation

Exact command options stay in `tsf <command> --help`.

---

## 🗂️ Repository layout

| Path | Contents |
| --- | --- |
| `src/tsflab/` | The framework: `benchmark` (CLI, runner, registries), `models`, `data`, `realtime`, `hub`, `tsf_core` (contracts) |
| `configs/`, `catalog/`, `verification/` | Experiment and real-time track configs, dataset cards, verification evidence |
| `apps/web/` | TSFLab Leaderboard: static site, submission pipeline, `submissions/`, real-time rounds |
| `experiments/` | Local research workspace; only `*/scripts/` is tracked |

---

## 📜 License

TSFLab is released under the [MIT License](LICENSE). Copyright © 2026 **Diaugeia.AI**.

Ordinary paper architectures are maintained locally under the project license.
Released pretrained foundation models use optional official packages and unchanged
checkpoints through the offline runtime boundary; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Real-time data remain subject to
their providers' terms (Caltrans PeMS, OpenAQ, exchange data via AKShare).

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Diaugeia/TSFLab&type=Date)](https://star-history.com/#Diaugeia/TSFLab&Date)
