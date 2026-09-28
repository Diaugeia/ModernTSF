---
name: "revin"
kind: "component"
module: "moderntsf.models._components.revin"
summary: "Reversible instance normalization."
---

# revin

## Purpose

Reversible instance normalization.

Reversible per-instance normalization for batch-time-channel tensors.

Implementation: [`__init__.py`](__init__.py)

## Public API

- `RevIN(num_features: int, eps: float=1e-05, affine: bool=True, subtract_last: bool=False, enabled: bool=True)`
  Normalize one sequence instance and later restore its original scale.

```python
from moderntsf.models._components.revin import RevIN
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `revin` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: `denormalization`, `instance`, `normalization`, `reversible`.

## Current model consumers

- [`amplifier`](../../amplifier/README.md)
- [`amrc`](../../amrc/README.md)
- [`aurora`](../../aurora/README.md)
- [`awemixer`](../../awemixer/README.md)
- [`catboost_ts`](../../catboost_ts/README.md)
- [`cmos`](../../cmos/README.md)
- [`cora`](../../cora/README.md)
- [`crosslinear`](../../crosslinear/README.md)
- [`cyclenet`](../../cyclenet/README.md)
- [`decision_tree_ts`](../../decision_tree_ts/README.md)
- [`distdf`](../../distdf/README.md)
- [`dpwmixer`](../../dpwmixer/README.md)
- [`dsformer`](../../dsformer/README.md)
- [`dualformer`](../../dualformer/README.md)
- [`duet`](../../duet/README.md)
- [`dynamic_tmoe`](../../dynamic_tmoe/README.md)
- [`extra_trees_ts`](../../extra_trees_ts/README.md)
- [`fets`](../../fets/README.md)
- [`film`](../../film/README.md)
- [`fredformer`](../../fredformer/README.md)
- [`ftp`](../../ftp/README.md)
- [`glocalib`](../../glocalib/README.md)
- [`gradient_boosting_ts`](../../gradient_boosting_ts/README.md)
- [`gru_forecaster_ts`](../../gru_forecaster_ts/README.md)
- [`gtr`](../../gtr/README.md)
- [`hdmixer`](../../hdmixer/README.md)
- [`hn_mvts`](../../hn_mvts/README.md)
- [`implicitforecaster`](../../implicitforecaster/README.md)
- [`interpdn`](../../interpdn/README.md)
- [`lightgbm_ts`](../../lightgbm_ts/README.md)
- [`lsinet`](../../lsinet/README.md)
- [`lstm_forecaster_ts`](../../lstm_forecaster_ts/README.md)
- [`mgsfformer`](../../mgsfformer/README.md)
- [`mlp_forecaster_ts`](../../mlp_forecaster_ts/README.md)
- [`moderntcn`](../../moderntcn/README.md)
- [`mofo`](../../mofo/README.md)
- [`mtsmixer`](../../mtsmixer/README.md)
- [`nhits`](../../nhits/README.md)
- [`occamvts`](../../occamvts/README.md)
- [`olinear`](../../olinear/README.md)
- [`paifilter`](../../paifilter/README.md)
- [`patchtst`](../../patchtst/README.md)
- [`pathformer`](../../pathformer/README.md)
- [`phaseformer`](../../phaseformer/README.md)
- [`phat`](../../phat/README.md)
- [`pmdformer`](../../pmdformer/README.md)
- [`pws`](../../pws/README.md)
- [`random_forest_ts`](../../random_forest_ts/README.md)
- [`refocus`](../../refocus/README.md)
- [`rlinear`](../../rlinear/README.md)
- [`rnn_forecaster_ts`](../../rnn_forecaster_ts/README.md)
- [`sdmixer`](../../sdmixer/README.md)
- [`semixer`](../../semixer/README.md)
- [`sempo`](../../sempo/README.md)
- [`srsnet`](../../srsnet/README.md)
- [`svtime`](../../svtime/README.md)
- [`swift`](../../swift/README.md)
- [`symtime`](../../symtime/README.md)
- [`tcn_forecaster_ts`](../../tcn_forecaster_ts/README.md)
- [`texfilter`](../../texfilter/README.md)
- [`timealign`](../../timealign/README.md)
- [`timebridge`](../../timebridge/README.md)
- [`timecap`](../../timecap/README.md)
- [`timeemb`](../../timeemb/README.md)
- [`timefilter`](../../timefilter/README.md)
- [`timekan`](../../timekan/README.md)
- [`timemixer`](../../timemixer/README.md)
- [`timemosaic`](../../timemosaic/README.md)
- [`timeperceiver`](../../timeperceiver/README.md)
- [`timexer`](../../timexer/README.md)
- [`tqnet`](../../tqnet/README.md)
- [`tsrag`](../../tsrag/README.md)
- [`umixer`](../../umixer/README.md)
- [`wavenet`](../../wavenet/README.md)
- [`wpmixer`](../../wpmixer/README.md)
- [`xgboost_ts`](../../xgboost_ts/README.md)
- [`xpatch`](../../xpatch/README.md)

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
