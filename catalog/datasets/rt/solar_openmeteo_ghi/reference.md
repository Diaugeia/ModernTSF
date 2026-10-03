# rt/solar_openmeteo_ghi — reference

## Provenance and license

- Source: the Open-Meteo Historical Forecast API (https://open-meteo.com), which stitches the analysis hours of operational weather models. `docs/en/realtime.md` records CC BY 4.0, a non-commercial free tier, and attribution.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`solar_openmeteo_ghi/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of the same API | source-reported (config) |
| Track config | `configs/realtime/solar_openmeteo_ghi.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/solar_openmeteo_ghi` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/solar_eia_us`](../solar_eia_us/README.md): US PV generation as a real-time preset
- [`solar`](../../solar/README.md): static PV power set
