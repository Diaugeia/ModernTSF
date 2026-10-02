---
name: add-dataset
description: "Register a new dataset in TSFLab, from a standard CSV, a custom loader, a traffic bundle, or the UltraTraffic PeMS store, including fetching or converting its files into loader-ready data. Use for dataset integration, data preparation, and preset configuration; not for profiling an existing dataset."
---

# Add a dataset

Data module, step one: make one dataset runnable through a registered loader, a
strict parameter schema, a preset, and a card. The card and the profile from
`inspect-dataset` are the context AutoResearch reads about data.

## Inputs

- The source files and their layout, target and feature columns, frequency,
  intended task modes, split policy, and any adjacency.
- Which pattern fits, smallest first: `custom` (standard CSV), `single` (a distinct
  loader), the traffic converter (values plus adjacency), or an existing store
  loader such as `ultratraffic_st` / `ultratraffic_ts`.

## Steps

1. Keep bytes in `dataset/`, loader and schema code in `src/tsflab/data/`, the
   preset in `configs/datasets/`, and the card in `catalog/datasets/`. Loader, preset,
   and card changes need a TSFLab checkout; in a standalone project, only plan them.
2. Get the data loader-ready (download a published preset, window a CSV, convert a
   traffic bundle or the UltraTraffic archive) per
   [references/prepare.md](references/prepare.md); never alter the source files.
3. Scaffold a CSV-backed preset; for traffic bundles read
   `uv run tsf data prepare --from traffic --help` and pass explicit inputs and splits:

   ```bash
   uv run tsf data add --name my_data --pattern custom \
     --path ./dataset/my_data/my_data.csv --target OT
   ```

4. For a PeMS district from UltraTraffic, reuse the store loaders: set
   `name = "ultratraffic_st"` (or `_ts`), `path = "./dataset/ultratraffic"`, and
   `[dataset.params]` with `region`, `years`, `variant` (`static`, `cl_common`,
   `cl_added`), and `stations` (`last` or `intersection`).
5. Keep `[dataset]` to `name`, optional display/track fields, `path`, optional `id`,
   and `[dataset.params]`. Loader options live in a strict registered schema that
   rejects misspellings.
6. Create the card skeleton, write it from measured or cited facts per
   [references/card.md](references/card.md), then exercise the data:

   ```bash
   uv run tsf repo cards              # creates catalog/datasets/my_data/README.md with TODOs
   uv run tsf data inspect --config configs/datasets/my_data.toml
   uv run tsf data audit
   ```

## Chain

- Module: Data.
- Reads: preset `configs/datasets/<name>.toml` and the card at L1/L2 for conventions of sibling presets.
- Produces: preset, card `catalog/datasets/<name>/README.md` (protocol, pitfalls, license), loader-ready bytes in `dataset/`.
- Hands off to: `inspect-dataset` (profile), `run-experiment` (preset), `run-autoresearch` (dataset card).

## Success

- Train-only scaling, correct feature/target selection, stable split boundaries,
  and adjacency injection where declared.
- `uv run tsf data audit` passes: required facts curated, no `TODO`, generated
  block current. Tests that pin the dataset count are updated.

## Stop and hand off

- Stop before overwriting an existing dataset or output directory unless asked.
  Publishing files needs explicit authorization and redistributable terms.
- Profile the result with `inspect-dataset`.
