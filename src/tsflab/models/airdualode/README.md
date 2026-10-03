---
name: "AirDualODE"
description: "Dual neural ODE: boundary-aware diffusion-advection physics plus masked-attention latent ODE, fused on the graph. Use for multi-station air-quality forecasting with a geographic graph and meteorology covariates; not for data without a meaningful spatial graph."
---

# AirDualODE

## Idea

- `BoundaryAwareDynamics` implements the open-system diffusion-advection equation with source/sink correction; a GRU estimates its coefficients from history.
- `DataDrivenDynamics` is a latent ODE field using attention masked by the geographic graph to model dependencies the physics omits.
- Both states roll out one step per forecast horizon with Euler or RK4 (`ode_method`).
- Physics and latent states are concatenated and fused over the geographic graph in `graph_fusion` before the linear decoder.

## When to use

- Pollutant concentration forecasting across stations, where transport (diffusion by distance, advection by wind) and open-system boundary effects shape the dynamics.
- Needs a real distance graph and a directed flow graph; without them a ring placeholder graph is used and the physics branch loses its meaning.
- Uses meteorology covariates for every step; point output only.

## Configure

- `enc_in`: number of stations.
- `cov_dim`: covariate features per station; inputs must carry exactly this many.
- `adj_mx`: `[enc_in, enc_in]` distance adjacency (diffusion).
- `flow_mx`: `[enc_in, enc_in]` directed wind/flow adjacency (advection).

Other hyperparameters: preset defaults in `configs/models/AirDualODE.toml`; tune generically. `unk_latent_dim` must be divisible by `n_heads`.

## Differences

Local implementation of Eq. (6)-(10) and the graph-fusion description after inspecting the official `models/Air_DualODE.py`, `models/layers/Explicit_odefunc.py`, and `models/layers/Unk_Dynamics.py` at the pinned revision; nothing copied (no license file). Eq. (6) maps to `BoundaryAwareDynamics`, Eq. (7)-(8) to the explicit rollout and projection, Eq. (9)-(10) to the GRU and masked-attention latent ODE, and the GNN fusion to `graph_fusion`. The paper's Decay-TCL objective is not part of the forward pass; training uses the configured loss.
