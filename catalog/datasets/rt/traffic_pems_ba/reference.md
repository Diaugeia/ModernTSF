# rt/traffic_pems_ba — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 4. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_ba_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 2,472 | measured |
| NaN cells in the store | 2.15% (per year 2019-2023: 4.3, 3.0, 2.5, 0.9, 0.0 %) | measured |
| Stations with more than 50% NaN | 62 | measured |
| Stations whose first reading is after the training split ends | 23 | measured |
| Value range | 0 to 15,101; mean 2710.0, standard deviation 1914.7 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 0.57% of observed values | measured |
| Mean flow by year | 2,871 (2019), 2,438 (2020), 2,714 (2021), 2,747 (2022), 2,780 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_ba.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_ba` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`ultratraffic_ba_st`](../../ultratraffic_ba_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_ba_ts`](../../ultratraffic_ba_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset
