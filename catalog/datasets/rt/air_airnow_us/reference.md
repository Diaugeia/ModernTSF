# rt/air_airnow_us — reference

## Provenance and license

- Source: EPA AirNow `HourlyData` files (https://www.airnow.gov). `docs/en/realtime.md` records them as US government public data that are preliminary and not for regulatory use; AQS holds the validated values months later.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_airnow_us/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 56 days of AirNow archive files | source-reported (config) |
| Track config | `configs/realtime/air_airnow_us.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_airnow_us` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/air_openaq_us`](../air_openaq_us/README.md): US, OpenAQ source
- [`rt/air_openaq_cn`](../air_openaq_cn/README.md): China, OpenAQ source
- [`rt/air_openaq_eu`](../air_openaq_eu/README.md): Europe, OpenAQ source
