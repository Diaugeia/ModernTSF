---
name: prepare-dataset
description: Convert or normalize existing time-series files into ModernTSF-ready data. Use for windowing CSV data, producing NPZ splits, converting traffic bundles, building the UltraTraffic PeMS store, or downloading GIFT-Eval data; not for registering a new loader.
---

# Prepare a dataset

Turn source files into the layout a registered loader reads, without altering the
source.

## Inputs

- The source path and layout, the destination directory, and the target loader.
- Window sizes and split policy when the loader needs pre-windowed data.

## Steps

1. Inspect the source layout and the destination before writing.
2. Choose the converter:

   ```bash
   # window a CSV (use --input-dir for pre-split CSVs)
   uv run tsf dataset prepare --input-csv <file.csv> --output-dir <dir> \
     --seq-len 96 --label-len 48 --pred-len 96
   # graph traffic bundles and GIFT-Eval: read the options first
   uv run tsf dataset convert-traffic --help
   uv run tsf dataset gift-download --help
   # UltraTraffic archive -> dataset/ultratraffic parquet store
   uv run tsf dataset convert-ultratraffic --archive <TrafficCL.zip>
   ```

   The UltraTraffic converter stores each region and year once (the CL full-year
   file equals the static one and a duplicated region is skipped) and records row,
   station, and missing-value counts plus source hashes in `manifest.json`.
3. Verify every split, shape, window, and the train-only scaling policy.

## Success

- Destination data that the target loader reads, source files preserved, and the
  manifest or split facts reported.

## Stop and hand off

- Stop before replacing an existing output directory without authorization.
- Register the result with `add-dataset`; profile it with `inspect-dataset`.
