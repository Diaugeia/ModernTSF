# MICN — reference

## Paper

Wang et al., ICLR 2023.

Transformer forecasters have high-complexity global attention and no targeted local-feature modelling. MICN combines
local features and global correlations with a multi-scale branch structure: each pattern is extracted with
down-sampled convolution (local) and isometric convolution (global), with linear complexity in sequence length for
suitable kernels. On six benchmarks it improves on the state of the art by 17.2% (multivariate) and 21.6% (univariate).

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Wang0HWCX23,
  author    = {Huiqiang Wang and Jian Peng and Feihu Huang and Jince Wang and Junhui Chen and Yifei Xiao},
  title     = {{MICN:} Multi-scale Local and Global Context Modeling for Long-term Series Forecasting},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://openreview.net/forum?id=zt53IDUR1U}
}
```
