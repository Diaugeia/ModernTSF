---
name: "STDMAE"
summary: "STD-MAE decouples masked pre-training of spatiotemporal traffic data into a temporal masked autoencoder (attention along time within each node) and a spatial one (attention across nodes within each time patch), both trained on a long history. In the forecasting stage the two encoders encode that long history, their last-patch node representations are injected into the skip path of a Graph WaveNet that forecasts from the recent short history. This entry is the forecasting stage; the masked pre-training is a separate stage that the runner does not express."
paper: "https://arxiv.org/abs/2312.00516"
paper_title: "Spatio-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting"
venue: "IJCAI 2024"
year: 2024
code: "https://github.com/Jimmy-7664/STD-MAE"
revision: "29fba76216f7ff9f2b51d097786cb5c91a8dae8b"
license: "NOASSERTION"
tagline: "Temporal and spatial masked-autoencoder encoders over a long history feed a Graph WaveNet's skip path."
tags: ["gnn", "transformer", "spatiotemporal", "graph-learning", "pretraining", "dilated-convolution", "covariates", "marks"]
composition: ["normalization=none", "decomposition=none", "temporal=component:tst_transformer", "channel=component:diffusion_conv", "head=local:context-injected-skip-head", "loss=loss:masked_mae"]
---
# STDMAE

## Key ideas

- `DecoupledMaskedEncoder(spatial=False)` is the T-MAE encoder: non-overlapping patch embedding, 2-D sinusoidal position over (node, patch), and a `tst_transformer` encoder attending along the patch axis within each node.
- `DecoupledMaskedEncoder(spatial=True)` is the S-MAE encoder: the same tokens, attending across nodes within each patch.
- `Model.encode_long_history` takes the last-patch representation of each node from both encoders; two MLPs map them to the skip width and add them to the Graph WaveNet skip path before the end convolutions.
- The backend is a valid (shrinking) dilated gated convolution Graph WaveNet over forward, reverse and adaptive supports (`diffusion_conv`, `graph_utils`, `adaptive_node_embedding_adjacency`) reading the last `history_len` steps with value and time-of-day.
- Masked pre-training (mask tokens, decoders, reconstruction loss) is a separate stage and is not in this module; `freeze_encoders=True` freezes the encoders for use with externally pre-trained weights.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2312.00516); title: Spatio-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting; venue/year: IJCAI 2024 / 2024
- [codebase](https://github.com/Jimmy-7664/STD-MAE); revision: `29fba76216f7ff9f2b51d097786cb5c91a8dae8b`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STDMAE.toml`](../../../../configs/models/STDMAE.toml).

## Differences

Independent implementation; no official code was copied (the official repository carries no licence, so it is reference-only; pinned revision above; checked `stdmae/stdmae_arch/stdmae.py`, `mask/mask.py`, `mask/patch.py`, `mask/positional_encoding.py`, `mask/transformer_layers.py`, `graphwavenet/model.py`, `stdmae_runner/stdmae_runner.py`, `stdmae_data/forecasting_dataset.py`, and `stdmae/STDMAE_PEMS04.py`). Inputs are `[B, seq_len, nodes]` values plus calendar marks; `seq_len` is the long-history length (a multiple of `patch_size`, 864 for the official PEMS04 setting) and the short history is its last `history_len` steps (12), which is how the official dataset builds both windows. The output is `[B, pred_len, nodes]`.

- **Pre-training stage is not implemented.** The official forecasting model loads two frozen encoders from separately trained checkpoints; the masked-autoencoder training (random token masking at ratio 0.25 for the released configs, mask tokens, a one-layer decoder, the reconstruction output layer and loss) is a different task that the runner cannot express, so it is omitted, and the decoder, mask-token and reconstruction parameters are absent. The released checkpoints are not loaded or shipped. By default (`freeze_encoders=False`) the two encoders are therefore trained end to end from random initialisation together with the backend, which is not the paper's frozen-encoder protocol; `freeze_encoders=True` freezes them (and keeps them in evaluation mode) for weights loaded externally into `temporal_encoder` and `spatial_encoder`. Results of this entry are not comparable with the paper's pre-trained numbers.
- **Encoders.** Reproduced from the code: patch embedding by a `(patch_size, 1)` strided convolution, sinusoidal 2-D position over (node, patch) added to the tokens, input scaled by `sqrt(embed_dim)`, post-norm ReLU Transformer layers (heads 4, feed-forward 4x, dropout 0.1) and a final LayerNorm, with no masking in the forecasting stage. The official position table comes from the `positional_encodings` package; here `sincos_2d` is written from its documented layout (interleaved sin/cos, row code then column code) and has not been compared numerically with the package. Official code leaves the encoders in training mode when the runner calls `train()` (dropout stays active under no-grad); frozen encoders here stay in evaluation mode.
- **Backend.** Graph WaveNet with kernel 2, 4 blocks of 2 layers, 32/32/256/512 channels, input value and time-of-day, left padding of one step then padding to the receptive field (13), valid dilated convolutions, per-layer batch norm, summed skips cropped to the shortest length, and a fixed final length of one step. The official always-created but unused residual 1x1 convolutions are omitted. The context MLPs map `embed_dim -> 512 -> skip_channels` (the official code hard-codes 96 and 256). The adaptive adjacency is initialised from standard normals as officially, and the two predefined supports are the double-transition matrices of the dataset adjacency (identity when none is given).
- **Windows.** The official dataset gives early samples an all-zero long history when fewer than `seq_len` earlier steps exist; the runner's windows start where a full window exists. Official time features are BasicTS scaled values; here calendar features come from the repository's marks (time-of-day, day-of-week) and only the first `in_dim = 2` are used.
- **Training recipe.** The official run uses masked MAE with null value 0, curriculum learning over the horizon (6 epochs), gradient clipping at 3.0, Adam with weight decay and a multi-step schedule; these are runner configuration here (`masked_mae` is the smoke loss, without a validity mask), not model facts.
- **Evidence.** Structure and reference-formula tests are in `tests/test_stdmae.py`; unified verification evidence has not been produced yet and is left stale for CI.

## Shared components

- [`adaptive_node_embedding_adjacency`](../_components/adaptive_node_embedding_adjacency/README.md)
- [`diffusion_conv`](../_components/diffusion_conv/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)
- [`tst_transformer`](../_components/tst_transformer/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `history_len=12`, `patch_size=12`, `embed_dim=96`, `num_heads=4`, `mlp_ratio=4`, `encoder_depth=4`, `encoder_dropout=0.1`, `in_dim=2`, `dropout=0.3`, `residual_channels=32`, `dilation_channels=32`, `skip_channels=256`, `end_channels=512`, `kernel_size=2`, `blocks=4`, `layers=2`, `freeze_encoders=False`
<!-- model-card:canonical:end -->

## Verification

Independent implementation; no official code was copied (the official repository carries no licence, so it is reference-only; pinned revision above; checked `stdmae/stdmae_arch/stdmae.py`, `mask/mask.py`, `mask/patch.py`, `mask/positional_encoding.py`, `mask/transformer_layers.py`, `graphwavenet/model.py`, `stdmae_runner/stdmae_runner.py`, `stdmae_data/forecasting_dataset.py`, and `stdmae/STDMAE_PEMS04.py`). Inputs are `[B, seq_len, nodes]` values plus calendar marks; `seq_len` is the long-history length (a multiple of `patch_size`, 864 for the official PEMS04 setting) and the short history is its last `history_len` steps (12), which is how the official dataset builds both windows. The output is `[B, pred_len, nodes]`.

- **Pre-training stage is not implemented.** The official forecasting model loads two frozen encoders from separately trained checkpoints; the masked-autoencoder training (random token masking at ratio 0.25 for the released configs, mask tokens, a one-layer decoder, the reconstruction output layer and loss) is a different task that the runner cannot express, so it is omitted, and the decoder, mask-token and reconstruction parameters are absent. The released checkpoints are not loaded or shipped. By default (`freeze_encoders=False`) the two encoders are therefore trained end to end from random initialisation together with the backend, which is not the paper's frozen-encoder protocol; `freeze_encoders=True` freezes them (and keeps them in evaluation mode) for weights loaded externally into `temporal_encoder` and `spatial_encoder`. Results of this entry are not comparable with the paper's pre-trained numbers.
- **Encoders.** Reproduced from the code: patch embedding by a `(patch_size, 1)` strided convolution, sinusoidal 2-D position over (node, patch) added to the tokens, input scaled by `sqrt(embed_dim)`, post-norm ReLU Transformer layers (heads 4, feed-forward 4x, dropout 0.1) and a final LayerNorm, with no masking in the forecasting stage. The official position table comes from the `positional_encodings` package; here `sincos_2d` is written from its documented layout (interleaved sin/cos, row code then column code) and has not been compared numerically with the package. Official code leaves the encoders in training mode when the runner calls `train()` (dropout stays active under no-grad); frozen encoders here stay in evaluation mode.
- **Backend.** Graph WaveNet with kernel 2, 4 blocks of 2 layers, 32/32/256/512 channels, input value and time-of-day, left padding of one step then padding to the receptive field (13), valid dilated convolutions, per-layer batch norm, summed skips cropped to the shortest length, and a fixed final length of one step. The official always-created but unused residual 1x1 convolutions are omitted. The context MLPs map `embed_dim -> 512 -> skip_channels` (the official code hard-codes 96 and 256). The adaptive adjacency is initialised from standard normals as officially, and the two predefined supports are the double-transition matrices of the dataset adjacency (identity when none is given).
- **Windows.** The official dataset gives early samples an all-zero long history when fewer than `seq_len` earlier steps exist; the runner's windows start where a full window exists. Official time features are BasicTS scaled values; here calendar features come from the repository's marks (time-of-day, day-of-week) and only the first `in_dim = 2` are used.
- **Training recipe.** The official run uses masked MAE with null value 0, curriculum learning over the horizon (6 epochs), gradient clipping at 3.0, Adam with weight decay and a multi-step schedule; these are runner configuration here (`masked_mae` is the smoke loss, without a validity mask), not model facts.
- **Evidence.** Structure and reference-formula tests are in `tests/test_stdmae.py`; unified verification evidence has not been produced yet and is left stale for CI.
