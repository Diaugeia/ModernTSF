---
name: "Fredformer"
description: "Frequency-debiased Transformer: rFFT bands rescaled to equal energy, each processed by a shared channel-attention Transformer, then a linear head. Use for multivariate forecasting where low-amplitude frequency components matter and channels interact; not for very many channels."
---

# Fredformer

## Idea

- Splits the rFFT spectrum into contiguous bands of `band_width` bins (zero-padded to a whole band) and rescales each band to unit RMS energy (`FrequencyEqualization`), so high-energy low frequencies do not dominate learning.
- `FrequencyBandAttention` applies one shared `nn.TransformerEncoder` whose tokens are the channels, independently within every band.
- A sigmoid `band_gate` blends the Transformer output into the equalized band before restoring band energy and inverting the FFT.
- The reconstructed history goes through a linear `seq_len -> pred_len` head; `revin` wraps the model.

## When to use

- Designed against frequency bias: Transformers tend to learn high-energy low-frequency features and overlook lower-amplitude ones that matter for accuracy.
- Channel attention within each band suits correlated channels; cost grows quadratically with channel count.

## Configure

- `enc_in`: number of channels.

Other hyperparameters: preset defaults in `configs/models/Fredformer.toml`; tune generically.

## Differences

Clean-room implementation from the paper's frequency-equalization and band-local attention design; `chenzRG/Fredformer` at `fa64775e` (no license file, recorded `NOASSERTION`) was not copied or reused.

- The paper's lightweight variant with attention-matrix approximation is not implemented.

Citation: Piao, X., Chen, Z., Murayama, T., Matsubara, Y., Sakurai, Y. "Fredformer: Frequency Debiased Transformer for Time Series Forecasting." KDD 2024, pp. 2400-2410. doi:10.1145/3637528.3671928.
