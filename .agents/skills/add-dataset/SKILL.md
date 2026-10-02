---
name: add-dataset
description: Register a new dataset in TSFLab from a standard CSV, a custom loader, a traffic bundle, or the UltraTraffic PeMS store. Use for dataset integration and preset configuration; not merely for inspecting or preprocessing an existing dataset.
---

# Add a dataset

Make one dataset runnable through a registered loader, a strict parameter schema,
a preset, and a generated card.

## Inputs

- The source files and their layout, target and feature columns, frequency,
  intended task modes, split policy, and any adjacency.
- Which pattern fits, smallest first: `custom` (standard CSV), `single` (a distinct
  loader), the traffic converter (values plus adjacency), or an existing store
  loader such as `ultratraffic_st` / `ultratraffic_ts`.

## Steps

1. Keep bytes in `dataset/`, loader and schema code in `src/tsflab/data/`, the
   preset in `configs/datasets/`, and the card in `catalog/datasets/`.
2. Scaffold a CSV-backed preset:

   ```bash
   uv run tsf dataset add --name my_data --pattern custom \
     --path ./dataset/my_data/my_data.csv --target OT
   ```

   For traffic bundles read `uv run tsf dataset convert-traffic --help` and pass
   explicit inputs, splits, and windows.
3. For a PeMS district from UltraTraffic, reuse the store loaders instead of a new
   loader: set `name = "ultratraffic_st"` (or `_ts`), `path = "./dataset/ultratraffic"`,
   and `[dataset.params]` with `region`, `years`, `variant` (`static`, `cl_common`,
   `cl_added`), and `stations` (`last` or `intersection`).
4. Keep `[dataset]` to `name`, optional display/track fields, `path`, optional `id`
   (catalog-style datasets such as GIFT-Eval), and `[dataset.params]`. Loader
   options live in a strict registered schema that rejects misspellings; a file
   loader receives a file, a directory loader a directory, a selector loader both
   `path` and `id`.
5. Write the card (see `.agents/STANDARDS.md`, Dataset cards). `tsf repo cards`
   creates `catalog/datasets/<name>/README.md` with `TODO` placeholders and the
   generated runtime block; replace every placeholder with verified facts. Measure
   length, channels, span, frequency, and missing values from the local file
   (read-only) and set `stats_basis: "measured"`. For facts you cannot measure,
   fetch the primary source (data page, paper, repository license) and label them
   `source-reported`. Write `license: "unknown"` and `redistribution: "unknown"`
   when no explicit terms exist; never guess a license or a number. Document split
   conventions, `drop_last`, scaling, zeros or sentinels, and leakage risks under
   "Standard protocol and known pitfalls", and list sibling presets in `related`.
6. Exercise the data and regenerate cards:

   ```bash
   uv run tsf dataset inspect --config configs/datasets/my_data.toml
   uv run tsf inspect --config configs/runs/<run>.toml
   uv run tsf repo cards
   uv run tsf dataset show my_data
   uv run tsf dataset audit
   ```

## Success

- Train-only scaling, correct feature/target selection, stable split boundaries,
  and adjacency injection where declared.
- `uv run tsf dataset audit` passes: required card facts are curated, no `TODO`
  remains, and the generated block is current.
- Tests that pin the dataset count are updated to the new total.

## Stop and hand off

- Stop before overwriting an existing dataset unless replacement was requested.
- Conversion of raw files belongs to `prepare-dataset`; profiling to
  `inspect-dataset`.
