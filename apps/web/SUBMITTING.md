# Submitting data to the TSEval leaderboard

The leaderboard at **[tseval.diaugeia.ai](https://tseval.diaugeia.ai)** is an open board
you can check: every row is rebuilt from the submission evidence under
[`submissions/`](submissions/) in this repo — the single source of truth. To add or
update results you commit submission files and push — CI validates, aggregates, and
deploys automatically, so the board is always a function of the committed evidence.

## TL;DR

1. Write one `submission.json` per run under
   `submissions/<track>/<dataset>/<model>/<run_id>/submission.json`.
2. `python pipeline/build_leaderboard.py --no-write` to preview locally.
3. Commit + push to `main`. CI runs `pipeline/validate.py` → `pipeline/build_leaderboard.py`
   → deploys to Cloudflare (and the HF Space mirror).

## Directory layout

```
submissions/
  time_series/<dataset>/<model>/<run_id>/submission.json   # ETTh1, ETTh2, electricity, solar, traffic, weather, …
  realtime/stock_hs300/<model>/<run_id>/submission.json    # → shown as the "Stock" track
```

`run_id` is any unique folder name; the convention is
`<model>_<dataset>_sl<seqlen>_pl<predlen>_seed<seed>_<timestamp>`.

## `submission.json` format

Two shapes are accepted (same record fields). **Prefer the flat shape** for new
submissions:

```jsonc
{
  "schema_version": "1.0.0",
  "model": "PatchTST",          // must match a ModernTSF model name (see data/model-meta.json)
  "dataset_id": "ETTh1",        // ETTh1 … weather, or stock_hs300
  "track": "time_series",       // "time_series" or "realtime"
  "seed": 2021,                 // one seed per file (see "Multiple runs" below)
  "results": [
    {
      "horizon": 192,           // pred_len: 192 for time_series, 5 for stock
      "metrics": {              // nulls allowed; mse drives ranking
        "mse": 0.4521, "mae": 0.4310, "rmse": 0.6724,
        "corr": 0.62, "wape": 0.51, "rse": 0.60
      },
      "timing": { "fit_time_sec": 812.4, "inference_time_sec": 7.1 }
    }
  ]
}
```

The bundle shape `{ "manifest": {...}, "datasets": [...], "records": [ <flat record>, … ] }`
is also accepted — the aggregator reads `records[]`. Both are validated against
`pipeline/contract.schema.json`.

`dataset_id → display name`: `stock_hs300 → Stock-HS300`; every other id is shown
verbatim. `track=realtime` + `dataset_id=stock_hs300` lands in the **Stock** track.

## Multiple runs / averaging across seeds

To report a mean over several seeds, **submit one file per run** (same `model` +
`dataset_id` + `horizon`, different `seed` and `run_id`). The aggregator groups them
and the leaderboard row shows:

- each metric = **mean across runs**,
- `<metric>_std` (e.g. `mse_std`) = sample standard deviation (omitted for a single run),
- `n_runs` = number of runs (the **Runs** column),
- `submission_ids` = every run folded in.

No config needed — drop 5 seed files in and the row reports `n_runs: 5` with averaged
metrics. (Example today: `weather` MoFo/Kronos are already `n_runs: 2`.)

## What is and isn't submission-driven

| Block | Source |
|---|---|
| `time_series/*` (all 8 datasets) | aggregated from `submissions/time_series/` |
| Stock **regression** (mse/mae/corr) | aggregated from `submissions/realtime/stock_hs300/` |
| Stock **quant** (returns/Sharpe/…) | **curated** — no raw quant submissions yet |
| **Air quality** (`Air-CHNCities`) | **curated** — raw inputs not uploaded |

Curated blocks are preserved on every rebuild (see `overlay_curated` in
`pipeline/build_leaderboard.py`). To make them submission-driven, add the
corresponding `submission.json` files and they'll replace the curated rows.

## Build & validate locally

```bash
python pipeline/build_leaderboard.py --no-validate --no-write   # preview summary
python pipeline/build_leaderboard.py                            # validate + aggregate + write data/leaderboard.json
bun run build                                                   # confirm the site builds
```

Ranking is per `(track, dataset, horizon)` by **MSE** (lower is better). Weights are
**not** part of a submission and are never required to get on the board — a row earns
its place with its result, trajectory, and report. If you *want* bit-level
reproducibility, you may optionally archive your trained weights in the public
[`Diaugeia/TSEval-Weights`](https://huggingface.co/datasets/Diaugeia/TSEval-Weights)
dataset, but that is an invitation, not a gate.
