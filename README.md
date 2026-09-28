<div align="center">

# 🚀 ModernTSF

**Modern Time Series Forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Time Series Forecasting](https://img.shields.io/badge/task-time%20series%20forecasting-blue.svg)](docs/en/models.md)
[![Models: 178](https://img.shields.io/badge/models-178-orange.svg)](docs/en/models.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Agent Infrastructure for time-series forecasting** — not just another toolkit.
A unified, reproducible substrate where humans and agents spend their time on the
*idea*, not the plumbing around it.

🗣️ **Clone the repo, open it in [Codex](https://developers.openai.com/codex), Claude Code, Pi, or DeepSeek Harness, and speak your idea. ModernTSF works out of the box in each.**

</div>

> 🧪 **Latest features land on the [`dev`](https://github.com/Diaugeia/ModernTSF/tree/dev) branch first.** `main` is the stable, versioned release line — if you want the newest (pre-release) capabilities, track or install from `dev`.

---

## 🧭 What is ModernTSF

You don't build a car to drive one, mill flour to bake a loaf, or grow your
own beans for a cup of coffee — you reach for something ready-made. AI
research needs the same layer: today's agents can write code and run
experiments, yet most of the effort — human and agent alike — still goes into
reproducing prior work, validating baselines, debugging environments, and
writing glue code. ModernTSF is that missing infrastructure layer for
time-series forecasting. You bring the idea; the substrate handles everything
around it.

---

## ✨ Highlights

- 🧠 **178 model/method entries, 80 dataset presets** — a flat catalog spanning baselines, neural forecasters, graph models, custom CSVs, traffic graphs, and GIFT-EVAL
- 🤖 **Agent-ready** — open the repository in Codex, Claude Code, Pi, or DeepSeek Harness and request a complete workflow in plain language
- 🎛️ **Three data settings** — `time_series`, `spatiotemporal`, and `covariate`, switchable per run
- 🔁 **Reproducible & auditable** — TOML configs, fixed seeds, profiled outputs, and optional research rounds keep results comparable without burdening one-off runs
- 🛠️ **One entry point** — `tsf` scaffolds, smoke-tests, sweeps, aggregates, ranks, plots, and reports

---

## 🏁 How to use

```bash
git clone https://github.com/Diaugeia/ModernTSF.git
cd ModernTSF
codex
```

Then just say what you want, in plain language:

```text
> Set up the environment for my GPU.
> Benchmark DLinear, PatchTST and iTransformer on ETTh1 and give me a leaderboard.
> Here is my CSV of hourly sales — add it as a dataset and find the best model for it.
> I have an idea: <describe it>. Scaffold a model, implement it, and compare it against strong baselines.
```

The Agent owns research planning, code/config editing, diagnosis, interpretation,
and reporting through its native capabilities. ModernTSF supplies callable
contracts, evidence, budgets, resource leases, and recoverable execution. Python
APIs and the optional CLI expose the same services; no extra Agent loop is required.

The wheel includes read-only model cards, configs, verification
evidence, components, skills, and task harnesses, so catalog and Agent discovery
also work after installation. Use a git checkout for commands that add or rewrite
models, datasets, documentation, or verification evidence.

For direct discovery, the public catalogs are equally lightweight:

```bash
uv run tsf model list --details
uv run tsf component list
uv run tsf agent task list
uv run tsf agent task render autoresearch --set 'question=<your question>'
uv run tsf agent task start autoresearch --set 'question=<your question>' --json
```

`task start` prepares the bounded research round and prompt; it does not dispatch
another Agent process.

Optional [execution controls](docs/en/execution.md) add environment audits,
TensorBoard/W&B, resource budgets, GPU scheduling, and epoch-boundary recovery
without changing the basic run command.

The [workflow guide](docs/en/workflows.md) explains the model interface, shared
components, offline official Foundation runtimes, data layers, verification, and
experiments.

Dataset resources have three deliberately separate layers: ignored local files
live under `dataset/`, executable loaders and schemas live under `src/data/`,
and readable catalog cards live under `catalog/datasets/`. Code and cards never
embed local dataset payloads.

---

## 📖 Documentation

The compact workflow reference covers models, data, verification, and experiments;
exact command options stay in CLI help:

[Workflow documentation](docs/en/README.md)

> But chances are you'll never need any of this — let the agent do the reading.

---

## 🗂️ Repository layout

| Path | Contents |
| --- | --- |
| `src/` | The installable framework (`pip install modern-tsf`, CLI `tsf`) |
| `configs/`, `catalog/`, `verification/` | Experiment configs, dataset catalog, model verification evidence |
| `tests/` | Test suite |
| `apps/web/` | ModernTSF Leaderboard — static site, submission pipeline, and `submissions/` |
| `experiments/` | Local research workspace; only `*/scripts/` is tracked |

---

## 📜 License

ModernTSF is released under the [MIT License](LICENSE) — open by default, free to
use, modify, and build upon.

Copyright © 2026 **Diaugeia.AI**.

Ordinary paper architectures are maintained locally under the project license.
Released pretrained Foundation Models use optional official packages and unchanged
checkpoints through the offline runtime boundary; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency attribution.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Diaugeia/ModernTSF&type=Date)](https://star-history.com/#Diaugeia/ModernTSF&Date)
