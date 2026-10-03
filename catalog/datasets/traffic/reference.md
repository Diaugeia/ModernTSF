# traffic — reference

## Provenance and license

- Source system: Caltrans Performance Measurement System (PeMS), http://pems.dot.ca.gov. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". That covers the raw PeMS data; the packaged copy below carries no license of its own, so redistribute it only after confirming the packager's terms.
- Packaging: Lai et al. 2018, https://github.com/laiguokun/multivariate-time-series-data, which has no license file (GitHub API: none) and no data terms in its README; the preprocessed copy therefore carries no license of its own. Raw source and packaging are separate: the PeMS public-domain policy applies to the raw data, but redistribute the LSTNet copy only after weighing both, hence `conditional`.
- Cite LSTNet (SIGIR 2018) and credit Caltrans PeMS.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 17,544 | measured |
| Channels | 862 sensors (columns named 1 to 862; no `OT` column) | measured |
| Time span | 2015-01-01 00:00:01 to 2016-12-31 23:00:01, step 1 hour, no gaps | measured |
| Missing values | 0 | measured |
| Value range | 0.0000 to 0.7240, mean 0.0567, std 0.0540 | measured |
| Exact zeros | 0.90% of all values | measured |
| Quantity | occupancy rate in [0, 1]; the 2015-2016 span is 731 days | source-reported (LSTNet) |

Measured on `dataset/traffic/traffic.csv` (read-only).

## Related datasets

- [`electricity`](../electricity/README.md): other hourly LTSF set with hundreds of channels
- [`pems04`](../pems04/README.md): 5-minute PeMS flow graph with adjacency
- [`ultratraffic_ba_ts`](../ultratraffic_ba_ts/README.md): hourly PeMS flow per station, 2023, stations as channels
