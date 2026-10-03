# rt/solar_eia_us — reference

## Provenance and license

- Source: the EIA-930 Hourly Electric Grid Monitor (https://www.eia.gov/electricity/gridmonitor/) through the EIA Open Data API (free key). `docs/en/realtime.md` records US government public-domain terms.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`solar_eia_us/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/solar_eia_us.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/solar_eia_us` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/grid_eia_us`](../grid_eia_us/README.md): same source, demand
- [`rt/solar_openmeteo_ghi`](../solar_openmeteo_ghi/README.md): irradiance driver
- [`solar`](../../solar/README.md): static PV power set
