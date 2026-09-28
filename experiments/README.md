# experiments/

Local research workspace. Only this README and each experiment's `scripts/`
are version-controlled; results stay on the machine that produced them
(see `.gitignore`).

| Path | Contents | Tracked |
| --- | --- | --- |
| `repo-bench/scripts/` | `baseline.sh` / `final.sh` — code-quality measurements of ModernTSF vs. Time-Series-Library, TFB, PyOmniTS | yes |
| `repo-bench/measurements/` | ruff / radon / vulture / jscpd reports, before and after | no |
| `repo-bench/repos/` | cloned comparison repositories | no |
| `data/` | raw server result bundles | no |
