---
name: "S_Mamba"
summary: "S_Mamba (Simple-Mamba) is a time series forecasting model that applies selective state space modeling in an iTransformer-style inverted embedding scheme. It tokenizes each variate's time points via a linear layer, uses a bidirectional Mamba layer to extract inter-variate correlations across the channel dimension, and applies a feed-forward network to learn temporal dependencies, finally mapping to forecasts through a linear layer without requiring custom CUDA operators for selective scanning."
paper: "https://arxiv.org/abs/2403.11144"
paper_title: "Is Mamba Effective for Time Series Forecasting?"
venue: "arXiv preprint"
year: 2024
code: "https://github.com/wzhwzhwzh0921/S-D-Mamba"
revision: "e7e8bf04066135afa43d85b0a87afa97cda16e3f"
license: "NOASSERTION"
---
# S_Mamba

S_Mamba (Simple-Mamba) is a time series forecasting model that applies selective state space modeling in an iTransformer-style inverted embedding scheme. It tokenizes each variate's time points via a linear layer, uses a bidirectional Mamba layer to extract inter-variate correlations across the channel dimension, and applies a feed-forward network to learn temporal dependencies, finally mapping to forecasts through a linear layer without requiring custom CUDA operators for selective scanning.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2403.11144); title: Is Mamba Effective for Time Series Forecasting?; venue/year: arXiv preprint / 2024
- [codebase](https://github.com/wzhwzhwzh0921/S-D-Mamba); revision: `e7e8bf04066135afa43d85b0a87afa97cda16e3f`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/S_Mamba.toml`](../../../../configs/models/S_Mamba.toml).

## Differences

**Clean-room implementation: confirmed.** The paper algorithm has executable token-axis, bidirectional-scan, FFN, and projection tests. Inputs are `[B, seq_len, variates]`, outputs are `[B, pred_len, variates]`, and marks are ignored. The author repository remains reference-only and no code was copied.

## Shared components

- [`mamba`](../_components/mamba/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_state=16`, `d_ff=128`, `e_layers=2`, `d_conv=2`, `expand=1`, `dropout=0.1`, `activation='gelu'`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Is Mamba Effective for Time Series Forecasting?
- **Venue**: arXiv preprint
- **Published**: 2024
- **arXiv**: https://arxiv.org/abs/2403.11144

## Abstract
In the realm of time series forecasting (TSF), it is imperative for models to adeptly discern and distill hidden patterns within historical time series data to forecast future states. Transformer-based models exhibit formidable efficacy in TSF, primarily attributed to their advantage in apprehending these patterns. However, the quadratic complexity of the Transformer leads to low computational efficiency and high costs, which somewhat hinders the deployment of the TSF model in real-world scenarios. Recently, Mamba, a selective state space model, has gained traction due to its ability to process dependencies in sequences while maintaining near-linear complexity. For TSF tasks, these characteristics enable Mamba to comprehend hidden patterns as the Transformer and reduce computational overhead compared to the Transformer. Therefore, we propose a Mamba-based model named Simple-Mamba (S-Mamba) for TSF. Specifically, we tokenize the time points of each variate autonomously via a linear layer. A bidirectional Mamba layer is utilized to extract inter-variate correlations and a Feed-Forward Network is set to learn temporal dependencies. Finally, the generation of forecast outcomes through a linear mapping layer. Experiments on thirteen public datasets prove that S-Mamba maintains low computational overhead and achieves leading performance. Furthermore, we conduct extensive experiments to explore Mamba's potential in TSF tasks.

## In TSFLab
Default config: `configs/models/S_Mamba.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The paper algorithm has executable token-axis, bidirectional-scan, FFN, and projection tests. Inputs are `[B, seq_len, variates]`, outputs are `[B, pred_len, variates]`, and marks are ignored. The author repository remains reference-only and no code was copied.

## Citation

```bibtex
@article{DBLP:journals/ijon/WangKFWYZWZ25,
  author       = {Zihan Wang and
                  Fanheng Kong and
                  Shi Feng and
                  Ming Wang and
                  Xiaocui Yang and
                  Han Zhao and
                  Daling Wang and
                  Yifei Zhang},
  title        = {Is Mamba effective for time series forecasting?},
  journal      = {Neurocomputing},
  volume       = {619},
  pages        = {129178},
  year         = {2025},
  url          = {https://doi.org/10.1016/j.neucom.2024.129178},
  doi          = {10.1016/J.NEUCOM.2024.129178},
  timestamp    = {Sat, 15 Nov 2025 13:50:19 +0100},
  biburl       = {https://dblp.org/rec/journals/ijon/WangKFWYZWZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
