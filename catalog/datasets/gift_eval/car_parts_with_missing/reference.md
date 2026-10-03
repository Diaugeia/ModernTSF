# gift_eval/car_parts_with_missing — reference

## Provenance and license

- Original source: Car Parts (Hyndman et al.), via the Monash Time Series Forecasting Repository; https://zenodo.org/records/4656022.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 2,674 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 51 | source-reported |
| Total observations | 136,374 | source-reported |
| Frequency | monthly (1mo) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval`](../README.md): the GIFT-Eval family card.
