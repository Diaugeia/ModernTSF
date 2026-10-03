# MoFo — reference

## Paper

- **Title**: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling
- **Venue**: NeurIPS 2025
- **Abstract (shortened)**: Stable periodic patterns underpin long-term forecasting, but continuous, chaotic input partitioning and weak inductive biases hide them. MoFo treats periodicity as the correlation of period-aligned steps and the trend of period-offset steps. Period-structured patches (2D tensors from discrete sampling) put period-aligned steps in rows and offset steps in columns. A period-aware modulator adds an adaptive inductive bias through an end-to-end trainable regulated relaxation function. MoFo is competitive on standard benchmarks with high memory efficiency and fast training.

## Implementation mapping

- Discrete sampling: the normalized input is left-padded (replicate) to `ceil(seq_len / periodic)` whole cycles and reshaped to `[batch, channels, phase, cycle]`.
- Each future step `h` has a learned query; it attends to the embedded history values at phase `(seq_len + h) % periodic` across all cycles.
- Relaxation (Eq. 9): `sigmoid(-alpha (d - beta)) + exp(-d) sigmoid(-alpha beta)` with softplus-positive `alpha`, `beta`; `d` is the cycle distance between the future step and each history cycle. Its log is added to the attention scores.
- Post-norm layers (attention and GELU feed-forward), stacked `d_layers` times; `projection` maps each query to one value.
- RevIN without affine parameters normalizes and de-normalizes each channel.

## Citation

```bibtex
@inproceedings{ma2025mofo,
  author    = {Jiaming Ma and Binwu Wang and Qihe Huang and Guanjun Wang and Pengkun Wang and Zhengyang Zhou and Yang Wang},
  title     = {{MoFo}: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MoFo}
}
```
