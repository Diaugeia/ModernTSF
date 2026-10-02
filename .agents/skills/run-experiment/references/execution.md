# Optional execution controls

Load this reference only for budgets, queueing, tracking, cancellation, or recovery.
Discover APIs through `tsflab.benchmark.infra.api.describe_modules()` or the optional
`tsf interface` adapter. The Agent owns decisions; services own atomic state and
execution constraints. CLI examples below do not mandate a subprocess hop.

Keep scientific TOML independent of execution policy. Use `tsf run <config>
--policy <policy.toml>` for resource/budget/tracking settings. First call the same
command with `--dry-run --json`; no compute starts until the whole matrix passes.
Missing environment/data/weights are readiness failures, not model evidence.

Use the printed run/sweep directory with `tsf run status`, `cancel`, or `resume`.
Resume preserves logical run IDs, verifies config/data/code fingerprints, skips
completed cells, and resumes the most recent committed checkpoint. Optional batch
recovery requires zero data-loader workers and replayable input. LatentTSF
pretraining and fixed-window evaluation have stage checkpoints; external state
requires paired runtime hooks. Rolling evaluation restarts. Do not reinterpret
weight-only initialization as full recovery.

GPU allocation defaults to exclusive; optional sharing reserves declared memory
among cooperating local processes and is not physical isolation. Never infer that
free-memory admission guarantees fitting. Do not silently alter workload after
OOM. Automatic retry is opt-in and must remain within the declared budget.

Tracking is disabled by default. TensorBoard/W&B are optional mirrors of local
scalar evidence; select W&B online only when sending the configured metadata is
within the user's requested scope. Offline attempts remain grouped by logical run.
No tracking account, round, daemon, or policy file is required for an ordinary run.

Round run counts, parallel slots, wall time, and GPU-hours are executable limits.
The Agent defines research iterations and claims each stable operation ID once
when an iteration budget applies; matrix preparation and resume do not count. Reserve external tokens/USD with `tsf usage`
before dispatch and settle the same operation ID afterward. Unreported external
reasoning and billing cannot be metered by this repository. Host permissions stay with the
host. Do not describe task permissions as an OS sandbox or launch another agent.

For persistent queue or cluster submission, prepare the matrix once with
`tsf run --prepare-only`. Queue controllers preserve identity on restart. Slurm
submission is external dispatch and requires authorization. Storage cleanup is a
preview unless `--apply` is supplied; preserve user data and never automate
cleanup outside the authorized scope. Use `tsf --format json <command>` when a
Harness needs one envelope across old and new command surfaces.
