# Audit one model

Check that one catalog entry's claims, code, and evidence agree.

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
```

1. Treat README front matter as canonical metadata. Check paper, codebase, license
   (or its recorded absence), and revision.
2. Compare equations, shapes, defaults, objective, preprocessing, initialization, and
   output semantics. Confirm the code is local, the official revision was inspected
   when available, nothing was copied or imported, and every defining operation has
   a justified component decision.
3. For runtime artifacts: every required asset is checksum-pinned and fetched
   explicitly, the artifact factory receives only verified local paths, offline
   absence fails before construction, and the card claims no checkpoint behavior
   beyond what verification covers.
4. For an inference-only foundation runtime: it uses the official loader through
   `src/tsflab/models/_foundation/`, never downloads implicitly, skips training, and
   records training/gradient checks as `not-applicable`.
5. Run the executable checks and inspect the evidence (finite outputs, active
   gradients, state-dict round trip, CPU, batch and sequence bounds, declared marks
   or adjacency, and `reference_comparison` executed for official code or
   `not-applicable` without it):

   ```bash
   uv run tsf verify model <Name>
   uv run tsf repo doctor --strict --models <Name>
   uv run tsf smoke --model <Name>   # only when model show reports a smoke_config
   ```

Report findings per layer (metadata, source facts, implementation, evidence), each
traced to a file and line. Never persist status or blockers in metadata. For several
models, audit each independently and let the task own partitioning, shared-file
writes, and the final repository gate.
