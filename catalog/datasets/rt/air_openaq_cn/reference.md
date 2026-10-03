# rt/air_openaq_cn — reference

## Provenance and license

- Source: OpenAQ v3 (https://openaq.org); an API key is needed to fetch updates. `docs/en/realtime.md` records the terms as CC BY 4.0 with per-provider licenses, which can be stricter; `conditional` reflects that.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_openaq_cn/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/air_openaq_cn.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_openaq_cn` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/air_openaq_us`](../air_openaq_us/README.md): US reference monitors
- [`rt/air_openaq_eu`](../air_openaq_eu/README.md): European reference monitors
- [`beijing_air`](../../beijing_air/README.md): static Chinese air-quality set
