---
license: mit
pretty_name: TSEval Submissions
tags: [time-series, forecasting, leaderboard, benchmark]
---

# TSEval — Submissions

Append-only evidence index of **leaderboard submissions** for the [Diaugeia.AI](https://diaugeia.ai) TSEval benchmark,
living inside the canonical [`github.com/Diaugeia/TSEval`](https://github.com/Diaugeia/TSEval) repo — one source of truth.
This holds only the lightweight **evidence** of each run (result + trajectory + report) so it stays cheap to clone and to
rebuild the leaderboard from. Weights are **not** part of a submission; they may *optionally* be archived in the public
[`Diaugeia/TSEval-Weights`](https://huggingface.co/datasets/Diaugeia/TSEval-Weights) dataset for bit-level
reproducibility, but are never required to get on the board.

## Layout

```
<track>/<dataset>/<model>/<submission_id>/
    submission.json     # tsf_core.SubmissionReport: manifest + DatasetSpec + RunRecord(s)
    trajectory.jsonl    # the agent's experiment process (synthetic for bulk imports)
    report.md           # human-readable per-submission summary (Markdown)
```

A submission carries its **result + trajectory + report only** — there is no weight reference in the bundle.
Trained weights MAY *optionally* be archived in the public [`Diaugeia/TSEval-Weights`](https://huggingface.co/datasets/Diaugeia/TSEval-Weights)
dataset for bit-level reproducibility, but are never required.

## Tracks

- `time_series/` — 8 static benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, electricity, solar, traffic, weather), horizon 192, **108 models** (see `Diaugeia/TSEval-Static`).
- `realtime/` — live, periodically-refreshed datasets: `stock_hs300` → **Stock-HS300 (CSI-300)**, seq_len 20 → pred_len 5, **135 models** (regression + quant backtest).
- `air_quality/` — **Air-CHNCities**, 6 pollutants (PM2.5, PM10, O₃, NO₂, SO₂, CO), **134 models** (curated).

## How the leaderboard is built

```bash
python pipeline/build_leaderboard.py            # validate → aggregate → write data/leaderboard.json
python pipeline/build_leaderboard.py --no-write # dry-run summary
```

Stdlib-only (no torch). Every `submission.json` is schema-validated (`pipeline/validate.py`),
then rows are aggregated per `(track, dataset, horizon, model)` — metrics **averaged across
runs/seeds** with `n_runs` + `<metric>_std` — and ranked by MSE. Blocks with no raw submissions
yet (air-quality, the stock *quant* view) are preserved as a curated overlay.

**See [`SUBMITTING.md`](../SUBMITTING.md) for the full submission format + multi-seed averaging rules.**

## Current contents

- `realtime/stock_hs300/` — **135 models** × CSI-300 (seq 20 → pred 5), 135 submissions.
