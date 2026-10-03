# gift_eval/bizitobs_l2c_H — reference

## Provenance and license

- Original source: BizITObs, processed per AutoMixer (Palaskar et al., 2024); https://github.com/BizITObs/BizITObservabilityData.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CDLA-Sharing-1.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift`.
- CDLA-Sharing-1.0 asks that shared data stay under the same terms; check it before republishing derived files.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 7 | source-reported |
| Mean length per series | 2,664 | source-reported |
| Total observations | 2,664 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 6 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/bizitobs_l2c_5T`](../bizitobs_l2c_5T/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
