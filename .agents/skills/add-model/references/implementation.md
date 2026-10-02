# Implement a model locally

Write one model as local, verified code built from cataloged components where the
semantics match. A released pretrained foundation model is not rewritten; use
`integrate-foundation-model`.

1. Resolve paper omissions (tensor order, padding, initialization, defaults,
   train/eval behavior) from the paper and pinned official code. Never copy, rename,
   mechanically rewrite, import, or depend on external model source. A missing
   official license does not block an independent rewrite; record it as missing.
2. Reuse `src/tsflab/models/_components/` only after proving mathematical and runtime
   equivalence. Implement each `extract-new` operator as a component with its own
   card, tests, and `ComponentSpec`; keep paper-specific glue local.
3. Implement inside the flat `src/tsflab/models/<slug>/` package with the exact
   four-input `forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)`.
   Model-specific operations belong in named methods, not extra public inputs. Keep
   useful paper formulas as comments, never source-derived code or comments.
4. Keep `spec.py` to construction, a strict parameter schema (no `**kwargs` or hidden
   aliases), config, capabilities, declared components, runtime contract.
5. For pretrained weights or tokenizers, declare checksum-pinned `ModelArtifact`
   entries (`hf://`, `https://`, or `file://`), never download implicitly, and add an
   `artifact_factory(cfg, params, paths)` that uses verified local paths only.
6. Complete the card: paper and official-code facts, local mapping, reused and
   model-local blocks, deviations, limits, and verification.
7. Add focused equation/structure checks and, when official code exists, a bounded
   `reference_comparison`; otherwise record it as `not-applicable`.

```bash
uv run tsf catalog show <Name>     # L1; --depth 2 for the full card, --depth 3 for paths
uv run tsf model verify <Name>   # existing model; new entries verify during model add
uv run tsf model audit <Name>
uv run tsf repo check --contracts strict --models <Name>
uv run tsf model audit --components
```

Success: unified evidence covers every check, the card is truthful, no peer-model
import, no unresolved defining operation. Stop rather than add a placeholder when the
paper, inputs, or defining behavior are too ambiguous to implement truthfully.
