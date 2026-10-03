# rt/air_openaq_eu — reference

## Provenance and license

- Source: OpenAQ v3 (https://openaq.org), reference monitors only; an API key is needed to fetch updates. Terms are CC BY 4.0 with per-provider licenses (`docs/en/realtime.md`), hence `conditional`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_openaq_eu/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/air_openaq_eu.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_openaq_eu` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/air_openaq_us`](../air_openaq_us/README.md): US, same source
- [`rt/air_openaq_cn`](../air_openaq_cn/README.md): China, same source
