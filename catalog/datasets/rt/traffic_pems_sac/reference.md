# rt/traffic_pems_sac — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 3. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_sac_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 801 | measured |
| NaN cells in the store | 23.83% (per year 2019-2023: 39.7, 36.3, 29.3, 13.7, 0.0 %) | measured |
| Stations with more than 50% NaN | 227 | measured |
| Stations whose first reading is after the training split ends | 99 | measured |
| Value range | 0 to 20,671; mean 1914.1, standard deviation 1663.1 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 2.29% of observed values | measured |
| Mean flow by year | 2,275 (2019), 1,911 (2020), 2,041 (2021), 1,850 (2022), 1,664 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_sac.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_sac` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`ultratraffic_sac_st`](../../ultratraffic_sac_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_sac_ts`](../../ultratraffic_sac_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset
