---
name: forecast-realtime-round
description: Produce, validate, and submit a forecast for an open ModernTSF real-time round (stocks, PeMS traffic, air quality, weather, solar, grid load), then read its score once the truth arrives. Use for live or replayed rolling evaluation; not for static benchmark runs or leaderboard result bundles.
---

# Forecast a real-time round

Real-time tracks open a round each week; forecasts must exist before their target
window is observed and are scored when a later data release covers it.

## Inputs

- A track id (`uv run tsf realtime list`) and an open round under
  `apps/web/submissions/realtime/<track>/rounds/<round_id>/round.json`: its cutoff,
  target timestamps, channels, and deadline.
- A method: a catalog model (trained on data up to the cutoff) or an external
  forecast in raw units with shape `(len(target_timestamps), len(channels))`.
- A local store for the track (`dataset/realtime/<track>/`); bootstrapping or
  updating it may need data-source credentials and is a maintainer action.

## Steps

1. Read `round.json`; confirm the deadline has not passed and the channels match.
2. Produce the forecast:

   ```bash
   uv run tsf realtime forecast --track <track> --model <Name> [--round <id>] \
     [--set training.epochs=10 --set experiment.runtime.device=cuda]
   ```

   The command exports history to the cutoff, trains through the standard runner,
   and writes `forecasts/<Name>.json`. It trains a model, so run it only where
   training is allowed (typically a GPU machine). An external method writes the
   same `ForecastSubmission` JSON itself.
3. Validate exactly as CI does:

   ```bash
   uv run tsf realtime validate apps/web/submissions/realtime/<track>/rounds/<id>/forecasts/<Name>.json
   ```

4. Submit by pull request adding only the forecast file; CI rejects it if the pull
   request was last updated after the deadline, and only the weekly workflow may
   write `round.json` or `scores.json`.
5. After the truth arrives, read `scores.json` and the track summary in
   `apps/web/data/realtime/<track>.json` (mean rank, mean z-scored MSE, Kendall τ).

To study the protocol on history instead, use
`uv run tsf realtime replay --track <track> --end <date> --weeks <n> [--models ...]`;
replayed rounds live under `work_dirs/_realtime_replay/` and never mix with live ones.

## Success

- A forecast file that passes `tsf realtime validate` before the deadline, and,
  after scoring, its rank and error reported against the round's other methods.

## Stop and hand off

- Stop if the deadline has passed or the channel set differs; never edit a round
  or its scores. Opening a pull request needs explicit authorization.
- Publishing the trained weights behind a forecast belongs to `publish-weights`.
