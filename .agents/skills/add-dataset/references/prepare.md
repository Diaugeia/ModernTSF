# Prepare data for a loader

Turn source files into the layout a registered loader reads, without altering the
source. Stop before replacing an existing output directory without authorization.

1. If the preset is published, download it instead of converting anything:

   ```bash
   uv run tsf dataset download --list
   uv run tsf dataset download <preset>      # verified against configs/hub/datasets.json
   ```

2. Otherwise inspect the source layout and the destination before writing, then
   choose the converter:

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
4. Publishing local files (`uv run tsf dataset publish <preset>`) is a maintainer
   action that needs explicit authorization and redistributable source terms.
   Read the card's `license` and `redistribution`; `unknown` or `restricted`
   means do not publish without that authorization.
