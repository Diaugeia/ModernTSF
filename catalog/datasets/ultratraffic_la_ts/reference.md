# ultratraffic_la_ts — reference

## Provenance and license

- Source system: Caltrans PeMS (http://pems.dot.ca.gov), District 7.
- Packaging: the store manifest records `UltraTraffic_CL.zip` as its source archive. No publication or repository named UltraTraffic was found in primary sources during this card's research; the closest published relatives (XXLTraffic, CC BY-NC 4.0; TrafficStream) were not confirmed to be the same data, so provenance beyond PeMS is unverified. If the archive derives from a CC BY-NC dataset, commercial use and republication are restricted.
- Checked for terms: the PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say information is "in the public domain" unless otherwise indicated, which covers the raw PeMS data only. The `UltraTraffic_CL.zip` archive is not documented anywhere that was found (no paper, repository, or dataset card), so the packager's terms and possible upstream derivation (for example from XXLTraffic, CC BY-NC 4.0) cannot be verified. `license` and `redistribution` stay `unknown`; do not run `tsf data publish` for these presets without confirming the archive's terms.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Stations (static 2023) | 1,926 (this preset: 1,926) | measured |
| Rows (hours in 2023) | 8,760 | measured |
| Value range, 2023 (all stations) | 0 to 20,314, mean 3,527.3 | measured |
| Exact zeros, 2023 (all stations) | 0.95% of all values | measured |
| NaN, 2023 | 0.0% | measured |
| 2023 added / common stations (continual-learning split) | 5 added, 1,921 common | measured (manifest) |
| Stations per static year (2003-2023) | 1,295 to 1,927 | measured (manifest) |
| Real-time store `dataset/realtime/traffic_pems_la` | 43,824 hourly rows, 2019-01-01 to 2023-12-31, 1,926 channels, 1.3% NaN (bootstrap release only) | measured |

Measured from the local parquet store and manifest under `dataset/` (read-only).

## Related datasets

- [`ultratraffic_la_st`](../ultratraffic_la_st/README.md): same stations, other layout
- [`ultratraffic_ba_st`](../ultratraffic_ba_st/README.md): another district, same pipeline
- [`pems07`](../pems07/README.md): 5-minute PeMS flow graph of the same district (STSGCN)
