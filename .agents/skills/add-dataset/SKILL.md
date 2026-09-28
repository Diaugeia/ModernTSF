---
name: add-dataset
description: Register a new dataset in ModernTSF from a standard CSV, custom loader, or traffic bundle. Use for dataset integration and configuration; not merely for inspecting or preprocessing an existing dataset.
---

# Add a dataset

Choose the smallest supported pattern: `custom` for a standard CSV, `single`
only for a distinct loader, or the traffic converter for values plus adjacency.

```bash
uv run tsf dataset add --name my_data --pattern custom \
  --path ./dataset/my_data/my_data.csv --target OT
```

Keep bytes in `dataset/`, loader/schema code in `src/moderntsf/data/`, the runnable preset
in `configs/datasets/`, and the generated card in `catalog/datasets/`. A preset
uses one `dataset.path`; only catalog-style datasets such as GIFT-Eval add a
separate `dataset.id`.

Keep `[dataset]` limited to `name`, optional display/track fields, `path`, optional
`id`, and `[dataset.params]`. Put loader-specific options in a strict registered
parameter schema; reject misspellings and catch-all keyword arguments. A file
loader receives a full file path, a directory loader receives a directory, and a
selector loader requires both `path` and `id`.

For traffic bundles, inspect `uv run tsf dataset convert-traffic --help` and
provide explicit inputs, splits, and windows. Then inspect and exercise the data:

```bash
uv run tsf dataset inspect --config configs/datasets/my_data.toml
uv run tsf inspect --config configs/runs/<run>.toml
uv run tsf run configs/runs/<run>.toml
```

Confirm train-only scaling, feature/target selection, split boundaries, and adjacency injection. Stop before overwriting an existing dataset unless replacement was requested.

`tsf dataset add` also regenerates the preset's canonical README card. Complete
any loader-specific facts in executable config/code, regenerate with
`uv run python scripts/generate_resource_cards.py`, then require both
`uv run tsf dataset show my_data` and `uv run tsf dataset audit` to pass.
