# rt/traffic_pems_sb — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 8. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_sb_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-1933` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 1,105 | measured |
| NaN cells in the store | 4.54% (per year 2019-2023: 6.8, 6.7, 4.6, 4.6, 0.0 %) | measured |
| Stations with more than 50% NaN | 49 | measured |
| Stations whose first reading is after the training split ends | 47 | measured |
| Value range | 0 to 16,278; mean 2535.7, standard deviation 1674.4 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 2.03% of observed values | measured |
| Mean flow by year | 2,671 (2019), 2,427 (2020), 2,582 (2021), 2,562 (2022), 2,442 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_sb.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_sb` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`ultratraffic_sb_st`](../../ultratraffic_sb_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_sb_ts`](../../ultratraffic_sb_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
