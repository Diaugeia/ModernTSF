# Models and methods

TSFLab exposes 200 model and method entries through one flat public catalog. Architecture families are retrieval tags on each card, not directories or categories. Presets configure runs and do not create additional entries.

Every entry is maintained as a local implementation; verification status is derived from executable evidence.

| Name | Preset | Capabilities | Model card |
|---|---|---|---|
| `AGCRN` | [`configs/models/AGCRN.toml`](../../configs/models/AGCRN.toml) | spatiotemporal | [README](../../src/tsflab/models/agcrn/README.md) |
| `AirCade` | [`configs/models/AirCade.toml`](../../configs/models/AirCade.toml) | covariate | [README](../../src/tsflab/models/aircade/README.md) |
| `AirDualODE` | [`configs/models/AirDualODE.toml`](../../configs/models/AirDualODE.toml) | covariate | [README](../../src/tsflab/models/airdualode/README.md) |
| `AirFormer` | [`configs/models/AirFormer.toml`](../../configs/models/AirFormer.toml) | covariate | [README](../../src/tsflab/models/airformer/README.md) |
| `AirPhyNet` | [`configs/models/AirPhyNet.toml`](../../configs/models/AirPhyNet.toml) | covariate | [README](../../src/tsflab/models/airphynet/README.md) |
| `Amplifier` | [`configs/models/Amplifier.toml`](../../configs/models/Amplifier.toml) | time-series | [README](../../src/tsflab/models/amplifier/README.md) |
| `AMRC` | [`configs/models/AMRC.toml`](../../configs/models/AMRC.toml) | time-series | [README](../../src/tsflab/models/amrc/README.md) |
| `APN` | [`configs/models/APN.toml`](../../configs/models/APN.toml) | time-series | [README](../../src/tsflab/models/apn/README.md) |
| `ARIMATS` | [`configs/models/ARIMATS.toml`](../../configs/models/ARIMATS.toml) | time-series | [README](../../src/tsflab/models/arima_ts/README.md) |
| `ARMD` | [`configs/models/ARMD.toml`](../../configs/models/ARMD.toml) | time-series | [README](../../src/tsflab/models/armd/README.md) |
| `ASTGCN` | [`configs/models/ASTGCN.toml`](../../configs/models/ASTGCN.toml) | covariate, spatiotemporal | [README](../../src/tsflab/models/astgcn/README.md) |
| `Aurora` | [`configs/models/Aurora.toml`](../../configs/models/Aurora.toml) | dense-modality-context, time-series | [README](../../src/tsflab/models/aurora/README.md) |
| `Autoformer` | [`configs/models/Autoformer.toml`](../../configs/models/Autoformer.toml) | time-series | [README](../../src/tsflab/models/autoformer/README.md) |
| `AutoRegressiveTS` | [`configs/models/AutoRegressiveTS.toml`](../../configs/models/AutoRegressiveTS.toml) | time-series | [README](../../src/tsflab/models/autoregressive_ts/README.md) |
| `AWEMixer` | [`configs/models/AWEMixer.toml`](../../configs/models/AWEMixer.toml) | time-series | [README](../../src/tsflab/models/awemixer/README.md) |
| `BayesianRidgeTS` | [`configs/models/BayesianRidgeTS.toml`](../../configs/models/BayesianRidgeTS.toml) | time-series | [README](../../src/tsflab/models/bayesian_ridge_ts/README.md) |
| `BigST` | [`configs/models/BigST.toml`](../../configs/models/BigST.toml) | spatiotemporal | [README](../../src/tsflab/models/bigst/README.md) |
| `BiMamba` | [`configs/models/BiMamba.toml`](../../configs/models/BiMamba.toml) | time-series | [README](../../src/tsflab/models/bimamba/README.md) |
| `BiST` | [`configs/models/BiST.toml`](../../configs/models/BiST.toml) | spatiotemporal | [README](../../src/tsflab/models/bist/README.md) |
| `CANet` | [`configs/models/CANet.toml`](../../configs/models/CANet.toml) | time-series | [README](../../src/tsflab/models/canet/README.md) |
| `CARD` | [`configs/models/CARD.toml`](../../configs/models/CARD.toml) | time-series | [README](../../src/tsflab/models/card/README.md) |
| `CatBoostTS` | [`configs/models/CatBoostTS.toml`](../../configs/models/CatBoostTS.toml) | time-series | [README](../../src/tsflab/models/catboost_ts/README.md) |
| `CATS` | [`configs/models/CATS.toml`](../../configs/models/CATS.toml) | time-series | [README](../../src/tsflab/models/cats/README.md) |
| `CauAir` | [`configs/models/CauAir.toml`](../../configs/models/CauAir.toml) | covariate | [README](../../src/tsflab/models/cauair/README.md) |
| `CMoS` | [`configs/models/CMoS.toml`](../../configs/models/CMoS.toml) | time-series | [README](../../src/tsflab/models/cmos/README.md) |
| `Composed` | [`configs/models/Composed.toml`](../../configs/models/Composed.toml) | time-series | [README](../../src/tsflab/models/composed/README.md) |
| `CoRA` | [`configs/models/CoRA.toml`](../../configs/models/CoRA.toml) | time-series | [README](../../src/tsflab/models/cora/README.md) |
| `CoRe` | [`configs/models/CoRe.toml`](../../configs/models/CoRe.toml) | test-time-adaptation, time-series | [README](../../src/tsflab/models/core/README.md) |
| `COSA` | [`configs/models/COSA.toml`](../../configs/models/COSA.toml) | test-time-adaptation, time-series | [README](../../src/tsflab/models/cosa/README.md) |
| `CRIB` | [`configs/models/CRIB.toml`](../../configs/models/CRIB.toml) | missing-values, time-series | [README](../../src/tsflab/models/crib/README.md) |
| `Crossformer` | [`configs/models/Crossformer.toml`](../../configs/models/Crossformer.toml) | time-series | [README](../../src/tsflab/models/crossformer/README.md) |
| `CrossGNN` | [`configs/models/CrossGNN.toml`](../../configs/models/CrossGNN.toml) | time-series | [README](../../src/tsflab/models/crossgnn/README.md) |
| `CrossLinear` | [`configs/models/CrossLinear.toml`](../../configs/models/CrossLinear.toml) | time-series | [README](../../src/tsflab/models/crosslinear/README.md) |
| `CycleNet` | [`configs/models/CycleNet.toml`](../../configs/models/CycleNet.toml) | time-series | [README](../../src/tsflab/models/cyclenet/README.md) |
| `D2STGNN` | [`configs/models/D2STGNN.toml`](../../configs/models/D2STGNN.toml) | spatiotemporal | [README](../../src/tsflab/models/d2stgnn/README.md) |
| `DCRNN` | [`configs/models/DCRNN.toml`](../../configs/models/DCRNN.toml) | spatiotemporal | [README](../../src/tsflab/models/dcrnn/README.md) |
| `DecisionTreeTS` | [`configs/models/DecisionTreeTS.toml`](../../configs/models/DecisionTreeTS.toml) | time-series | [README](../../src/tsflab/models/decision_tree_ts/README.md) |
| `DeepAir` | [`configs/models/DeepAir.toml`](../../configs/models/DeepAir.toml) | covariate | [README](../../src/tsflab/models/deepair/README.md) |
| `DeepAR` | [`configs/models/DeepAR.toml`](../../configs/models/DeepAR.toml) | distribution-output, time-series | [README](../../src/tsflab/models/deepar/README.md) |
| `DFDGCN` | [`configs/models/DFDGCN.toml`](../../configs/models/DFDGCN.toml) | spatiotemporal | [README](../../src/tsflab/models/dfdgcn/README.md) |
| `DGCRN` | [`configs/models/DGCRN.toml`](../../configs/models/DGCRN.toml) | spatiotemporal | [README](../../src/tsflab/models/dgcrn/README.md) |
| `DistDF` | [`configs/models/DistDF.toml`](../../configs/models/DistDF.toml) | time-series | [README](../../src/tsflab/models/distdf/README.md) |
| `DLinear` | [`configs/models/DLinear.toml`](../../configs/models/DLinear.toml) | time-series | [README](../../src/tsflab/models/dlinear/README.md) |
| `DPWMixer` | [`configs/models/DPWMixer.toml`](../../configs/models/DPWMixer.toml) | time-series | [README](../../src/tsflab/models/dpwmixer/README.md) |
| `DSFormer` | [`configs/models/DSFormer.toml`](../../configs/models/DSFormer.toml) | time-series | [README](../../src/tsflab/models/dsformer/README.md) |
| `DSTAGNN` | [`configs/models/DSTAGNN.toml`](../../configs/models/DSTAGNN.toml) | spatiotemporal | [README](../../src/tsflab/models/dstagnn/README.md) |
| `DTAF` | [`configs/models/DTAF.toml`](../../configs/models/DTAF.toml) | time-series | [README](../../src/tsflab/models/dtaf/README.md) |
| `Dualformer` | [`configs/models/Dualformer.toml`](../../configs/models/Dualformer.toml) | time-series | [README](../../src/tsflab/models/dualformer/README.md) |
| `DUET` | [`configs/models/DUET.toml`](../../configs/models/DUET.toml) | time-series | [README](../../src/tsflab/models/duet/README.md) |
| `DynamicTMoE` | [`configs/models/DynamicTMoE.toml`](../../configs/models/DynamicTMoE.toml) | time-series | [README](../../src/tsflab/models/dynamic_tmoe/README.md) |
| `ElasticNetTS` | [`configs/models/ElasticNetTS.toml`](../../configs/models/ElasticNetTS.toml) | time-series | [README](../../src/tsflab/models/elastic_net_ts/README.md) |
| `ETSformer` | [`configs/models/ETSformer.toml`](../../configs/models/ETSformer.toml) | time-series | [README](../../src/tsflab/models/etsformer/README.md) |
| `ExpSmoothingTS` | [`configs/models/ExpSmoothingTS.toml`](../../configs/models/ExpSmoothingTS.toml) | time-series | [README](../../src/tsflab/models/exp_smoothing_ts/README.md) |
| `Extralonger` | [`configs/models/Extralonger.toml`](../../configs/models/Extralonger.toml) | spatiotemporal | [README](../../src/tsflab/models/extralonger/README.md) |
| `ExtraTreesTS` | [`configs/models/ExtraTreesTS.toml`](../../configs/models/ExtraTreesTS.toml) | time-series | [README](../../src/tsflab/models/extra_trees_ts/README.md) |
| `FEDformer` | [`configs/models/FEDformer.toml`](../../configs/models/FEDformer.toml) | time-series | [README](../../src/tsflab/models/fedformer/README.md) |
| `FeTS` | [`configs/models/FeTS.toml`](../../configs/models/FeTS.toml) | time-series | [README](../../src/tsflab/models/fets/README.md) |
| `FiLM` | [`configs/models/FiLM.toml`](../../configs/models/FiLM.toml) | time-series | [README](../../src/tsflab/models/film/README.md) |
| `FITS` | [`configs/models/FITS.toml`](../../configs/models/FITS.toml) | time-series | [README](../../src/tsflab/models/fits/README.md) |
| `Fredformer` | [`configs/models/Fredformer.toml`](../../configs/models/Fredformer.toml) | time-series | [README](../../src/tsflab/models/fredformer/README.md) |
| `FreqMoE` | [`configs/models/FreqMoE.toml`](../../configs/models/FreqMoE.toml) | time-series | [README](../../src/tsflab/models/freqmoe/README.md) |
| `FreTS` | [`configs/models/FreTS.toml`](../../configs/models/FreTS.toml) | time-series | [README](../../src/tsflab/models/frets/README.md) |
| `FTP` | [`configs/models/FTP.toml`](../../configs/models/FTP.toml) | time-series | [README](../../src/tsflab/models/ftp/README.md) |
| `GAGNN` | [`configs/models/GAGNN.toml`](../../configs/models/GAGNN.toml) | covariate | [README](../../src/tsflab/models/gagnn/README.md) |
| `Gateformer` | [`configs/models/Gateformer.toml`](../../configs/models/Gateformer.toml) | time-series | [README](../../src/tsflab/models/gateformer/README.md) |
| `GaussianMLP` | [`configs/models/GaussianMLP.toml`](../../configs/models/GaussianMLP.toml) | distribution-output, time-series | [README](../../src/tsflab/models/gaussian_mlp/README.md) |
| `GaussianProcessTS` | [`configs/models/GaussianProcessTS.toml`](../../configs/models/GaussianProcessTS.toml) | time-series | [README](../../src/tsflab/models/gaussian_process_ts/README.md) |
| `GCLSTM` | [`configs/models/GCLSTM.toml`](../../configs/models/GCLSTM.toml) | covariate, spatiotemporal | [README](../../src/tsflab/models/gclstm/README.md) |
| `GlocalIB` | [`configs/models/GlocalIB.toml`](../../configs/models/GlocalIB.toml) | time-series | [README](../../src/tsflab/models/glocalib/README.md) |
| `GOTSF` | [`configs/models/GOTSF.toml`](../../configs/models/GOTSF.toml) | time-series | [README](../../src/tsflab/models/gotsf/README.md) |
| `GradientBoostingTS` | [`configs/models/GradientBoostingTS.toml`](../../configs/models/GradientBoostingTS.toml) | time-series | [README](../../src/tsflab/models/gradient_boosting_ts/README.md) |
| `GRUForecasterTS` | [`configs/models/GRUForecasterTS.toml`](../../configs/models/GRUForecasterTS.toml) | time-series | [README](../../src/tsflab/models/gru_forecaster_ts/README.md) |
| `GTR` | [`configs/models/GTR.toml`](../../configs/models/GTR.toml) | time-series | [README](../../src/tsflab/models/gtr/README.md) |
| `GTS` | [`configs/models/GTS.toml`](../../configs/models/GTS.toml) | spatiotemporal | [README](../../src/tsflab/models/gts/README.md) |
| `GWNet` | [`configs/models/GWNet.toml`](../../configs/models/GWNet.toml) | spatiotemporal | [README](../../src/tsflab/models/gwnet/README.md) |
| `HDMixer` | [`configs/models/HDMixer.toml`](../../configs/models/HDMixer.toml) | time-series | [README](../../src/tsflab/models/hdmixer/README.md) |
| `HimNet` | [`configs/models/HimNet.toml`](../../configs/models/HimNet.toml) | spatiotemporal | [README](../../src/tsflab/models/himnet/README.md) |
| `HL` | [`configs/models/HL.toml`](../../configs/models/HL.toml) | spatiotemporal | [README](../../src/tsflab/models/hl/README.md) |
| `HMformer` | [`configs/models/HMformer.toml`](../../configs/models/HMformer.toml) | time-series | [README](../../src/tsflab/models/hmformer/README.md) |
| `HN_MVTS` | [`configs/models/HN_MVTS.toml`](../../configs/models/HN_MVTS.toml) | time-series | [README](../../src/tsflab/models/hn_mvts/README.md) |
| `ImplicitForecaster` | [`configs/models/ImplicitForecaster.toml`](../../configs/models/ImplicitForecaster.toml) | time-series | [README](../../src/tsflab/models/implicitforecaster/README.md) |
| `Informer` | [`configs/models/Informer.toml`](../../configs/models/Informer.toml) | time-series | [README](../../src/tsflab/models/informer/README.md) |
| `InterPDN` | [`configs/models/InterPDN.toml`](../../configs/models/InterPDN.toml) | time-series | [README](../../src/tsflab/models/interpdn/README.md) |
| `iTransformer` | [`configs/models/iTransformer.toml`](../../configs/models/iTransformer.toml) | time-series | [README](../../src/tsflab/models/itransformer/README.md) |
| `KalmanFilterTS` | [`configs/models/KalmanFilterTS.toml`](../../configs/models/KalmanFilterTS.toml) | time-series | [README](../../src/tsflab/models/kalman_filter_ts/README.md) |
| `KNNForecasterTS` | [`configs/models/KNNForecasterTS.toml`](../../configs/models/KNNForecasterTS.toml) | time-series | [README](../../src/tsflab/models/knn_forecaster_ts/README.md) |
| `Koopa` | [`configs/models/Koopa.toml`](../../configs/models/Koopa.toml) | time-series | [README](../../src/tsflab/models/koopa/README.md) |
| `Kronos` | [`configs/models/Kronos.toml`](../../configs/models/Kronos.toml) | time-series | [README](../../src/tsflab/models/kronos/README.md) |
| `LassoRegressionTS` | [`configs/models/LassoRegressionTS.toml`](../../configs/models/LassoRegressionTS.toml) | time-series | [README](../../src/tsflab/models/lasso_regression_ts/README.md) |
| `LatentTSF` | [`configs/models/LatentTSF.toml`](../../configs/models/LatentTSF.toml) | pretraining-stage, time-series | [README](../../src/tsflab/models/latenttsf/README.md) |
| `LightGBMTS` | [`configs/models/LightGBMTS.toml`](../../configs/models/LightGBMTS.toml) | time-series | [README](../../src/tsflab/models/lightgbm_ts/README.md) |
| `LightTS` | [`configs/models/LightTS.toml`](../../configs/models/LightTS.toml) | time-series | [README](../../src/tsflab/models/lightts/README.md) |
| `Linear` | [`configs/models/Linear.toml`](../../configs/models/Linear.toml) | time-series | [README](../../src/tsflab/models/linear/README.md) |
| `LSINet` | [`configs/models/LSINet.toml`](../../configs/models/LSINet.toml) | time-series | [README](../../src/tsflab/models/lsinet/README.md) |
| `LSTM` | [`configs/models/LSTM.toml`](../../configs/models/LSTM.toml) | spatiotemporal | [README](../../src/tsflab/models/lstm/README.md) |
| `LSTMForecasterTS` | [`configs/models/LSTMForecasterTS.toml`](../../configs/models/LSTMForecasterTS.toml) | time-series | [README](../../src/tsflab/models/lstm_forecaster_ts/README.md) |
| `MAFS` | [`configs/models/MAFS.toml`](../../configs/models/MAFS.toml) | time-series | [README](../../src/tsflab/models/mafs/README.md) |
| `MAGE` | [`configs/models/MAGE.toml`](../../configs/models/MAGE.toml) | spatiotemporal | [README](../../src/tsflab/models/mage/README.md) |
| `MambaSimple` | [`configs/models/MambaSimple.toml`](../../configs/models/MambaSimple.toml) | time-series | [README](../../src/tsflab/models/mambasimple/README.md) |
| `MambaTS` | [`configs/models/MambaTS.toml`](../../configs/models/MambaTS.toml) | time-series | [README](../../src/tsflab/models/mambats/README.md) |
| `MegaCRN` | [`configs/models/MegaCRN.toml`](../../configs/models/MegaCRN.toml) | spatiotemporal | [README](../../src/tsflab/models/megacrn/README.md) |
| `MGSFformer` | [`configs/models/MGSFformer.toml`](../../configs/models/MGSFformer.toml) | spatiotemporal | [README](../../src/tsflab/models/mgsfformer/README.md) |
| `MICN` | [`configs/models/MICN.toml`](../../configs/models/MICN.toml) | time-series | [README](../../src/tsflab/models/micn/README.md) |
| `MixLinear` | [`configs/models/MixLinear.toml`](../../configs/models/MixLinear.toml) | time-series | [README](../../src/tsflab/models/mixlinear/README.md) |
| `MLPForecasterTS` | [`configs/models/MLPForecasterTS.toml`](../../configs/models/MLPForecasterTS.toml) | time-series | [README](../../src/tsflab/models/mlp_forecaster_ts/README.md) |
| `MMPD` | [`configs/models/MMPD.toml`](../../configs/models/MMPD.toml) | time-series | [README](../../src/tsflab/models/mmpd/README.md) |
| `ModernTCN` | [`configs/models/ModernTCN.toml`](../../configs/models/ModernTCN.toml) | time-series | [README](../../src/tsflab/models/moderntcn/README.md) |
| `MoFo` | [`configs/models/MoFo.toml`](../../configs/models/MoFo.toml) | time-series | [README](../../src/tsflab/models/mofo/README.md) |
| `MQRNN` | [`configs/models/MQRNN.toml`](../../configs/models/MQRNN.toml) | covariate, quantile-output, time-series | [README](../../src/tsflab/models/mqrnn/README.md) |
| `MSGNet` | [`configs/models/MSGNet.toml`](../../configs/models/MSGNet.toml) | time-series | [README](../../src/tsflab/models/msgnet/README.md) |
| `MTGNN` | [`configs/models/MTGNN.toml`](../../configs/models/MTGNN.toml) | spatiotemporal | [README](../../src/tsflab/models/mtgnn/README.md) |
| `MTSMixer` | [`configs/models/MTSMixer.toml`](../../configs/models/MTSMixer.toml) | time-series | [README](../../src/tsflab/models/mtsmixer/README.md) |
| `MultiPatchFormer` | [`configs/models/MultiPatchFormer.toml`](../../configs/models/MultiPatchFormer.toml) | time-series | [README](../../src/tsflab/models/multipatchformer/README.md) |
| `NBeats` | [`configs/models/NBeats.toml`](../../configs/models/NBeats.toml) | time-series | [README](../../src/tsflab/models/nbeats/README.md) |
| `NHiTS` | [`configs/models/NHiTS.toml`](../../configs/models/NHiTS.toml) | time-series | [README](../../src/tsflab/models/nhits/README.md) |
| `NLinear` | [`configs/models/NLinear.toml`](../../configs/models/NLinear.toml) | time-series | [README](../../src/tsflab/models/nlinear/README.md) |
| `NSTransformer` | [`configs/models/NSTransformer.toml`](../../configs/models/NSTransformer.toml) | time-series | [README](../../src/tsflab/models/nstransformer/README.md) |
| `OccamVTS` | [`configs/models/OccamVTS.toml`](../../configs/models/OccamVTS.toml) | time-series | [README](../../src/tsflab/models/occamvts/README.md) |
| `OLinear` | [`configs/models/OLinear.toml`](../../configs/models/OLinear.toml) | time-series | [README](../../src/tsflab/models/olinear/README.md) |
| `PaiFilter` | [`configs/models/PaiFilter.toml`](../../configs/models/PaiFilter.toml) | time-series | [README](../../src/tsflab/models/paifilter/README.md) |
| `PatchMLP` | [`configs/models/PatchMLP.toml`](../../configs/models/PatchMLP.toml) | time-series | [README](../../src/tsflab/models/patchmlp/README.md) |
| `PatchTST` | [`configs/models/PatchTST.toml`](../../configs/models/PatchTST.toml) | time-series | [README](../../src/tsflab/models/patchtst/README.md) |
| `Pathformer` | [`configs/models/Pathformer.toml`](../../configs/models/Pathformer.toml) | time-series | [README](../../src/tsflab/models/pathformer/README.md) |
| `PAttn` | [`configs/models/PAttn.toml`](../../configs/models/PAttn.toml) | time-series | [README](../../src/tsflab/models/pattn/README.md) |
| `PCDCNet` | [`configs/models/PCDCNet.toml`](../../configs/models/PCDCNet.toml) | covariate | [README](../../src/tsflab/models/pcdcnet/README.md) |
| `PhaseFormer` | [`configs/models/PhaseFormer.toml`](../../configs/models/PhaseFormer.toml) | time-series | [README](../../src/tsflab/models/phaseformer/README.md) |
| `PHAT` | [`configs/models/PHAT.toml`](../../configs/models/PHAT.toml) | time-series | [README](../../src/tsflab/models/phat/README.md) |
| `PM25_GNN` | [`configs/models/PM25_GNN.toml`](../../configs/models/PM25_GNN.toml) | covariate | [README](../../src/tsflab/models/pm25gnn/README.md) |
| `PMDformer` | [`configs/models/PMDformer.toml`](../../configs/models/PMDformer.toml) | time-series | [README](../../src/tsflab/models/pmdformer/README.md) |
| `PolynomialRegressionTS` | [`configs/models/PolynomialRegressionTS.toml`](../../configs/models/PolynomialRegressionTS.toml) | time-series | [README](../../src/tsflab/models/polynomial_regression_ts/README.md) |
| `PULSE` | [`configs/models/PULSE.toml`](../../configs/models/PULSE.toml) | time-series | [README](../../src/tsflab/models/pulse/README.md) |
| `PWS` | [`configs/models/PWS.toml`](../../configs/models/PWS.toml) | time-series | [README](../../src/tsflab/models/pws/README.md) |
| `Pyraformer` | [`configs/models/Pyraformer.toml`](../../configs/models/Pyraformer.toml) | time-series | [README](../../src/tsflab/models/pyraformer/README.md) |
| `QuantileDLinear` | [`configs/models/QuantileDLinear.toml`](../../configs/models/QuantileDLinear.toml) | quantile-output, time-series | [README](../../src/tsflab/models/quantile_dlinear/README.md) |
| `QuantilePatchTST` | [`configs/models/QuantilePatchTST.toml`](../../configs/models/QuantilePatchTST.toml) | quantile-output, time-series | [README](../../src/tsflab/models/quantile_patchtst/README.md) |
| `RAGC` | [`configs/models/RAGC.toml`](../../configs/models/RAGC.toml) | spatiotemporal | [README](../../src/tsflab/models/ragc/README.md) |
| `RandomForestTS` | [`configs/models/RandomForestTS.toml`](../../configs/models/RandomForestTS.toml) | time-series | [README](../../src/tsflab/models/random_forest_ts/README.md) |
| `ReFocus` | [`configs/models/ReFocus.toml`](../../configs/models/ReFocus.toml) | time-series | [README](../../src/tsflab/models/refocus/README.md) |
| `Reformer` | [`configs/models/Reformer.toml`](../../configs/models/Reformer.toml) | time-series | [README](../../src/tsflab/models/reformer/README.md) |
| `RidgeRegressionTS` | [`configs/models/RidgeRegressionTS.toml`](../../configs/models/RidgeRegressionTS.toml) | time-series | [README](../../src/tsflab/models/ridge_regression_ts/README.md) |
| `RLinear` | [`configs/models/RLinear.toml`](../../configs/models/RLinear.toml) | time-series | [README](../../src/tsflab/models/rlinear/README.md) |
| `RNNForecasterTS` | [`configs/models/RNNForecasterTS.toml`](../../configs/models/RNNForecasterTS.toml) | time-series | [README](../../src/tsflab/models/rnn_forecaster_ts/README.md) |
| `RPMixer` | [`configs/models/RPMixer.toml`](../../configs/models/RPMixer.toml) | spatiotemporal | [README](../../src/tsflab/models/rpmixer/README.md) |
| `S4` | [`configs/models/S4.toml`](../../configs/models/S4.toml) | time-series | [README](../../src/tsflab/models/s4/README.md) |
| `S_Mamba` | [`configs/models/S_Mamba.toml`](../../configs/models/S_Mamba.toml) | time-series | [README](../../src/tsflab/models/s_mamba/README.md) |
| `SCINet` | [`configs/models/SCINet.toml`](../../configs/models/SCINet.toml) | time-series | [README](../../src/tsflab/models/scinet/README.md) |
| `SDMixer` | [`configs/models/SDMixer.toml`](../../configs/models/SDMixer.toml) | time-series | [README](../../src/tsflab/models/sdmixer/README.md) |
| `SegRNN` | [`configs/models/SegRNN.toml`](../../configs/models/SegRNN.toml) | time-series | [README](../../src/tsflab/models/segrnn/README.md) |
| `SEMixer` | [`configs/models/SEMixer.toml`](../../configs/models/SEMixer.toml) | time-series | [README](../../src/tsflab/models/semixer/README.md) |
| `SEMPO` | [`configs/models/SEMPO.toml`](../../configs/models/SEMPO.toml) | time-series | [README](../../src/tsflab/models/sempo/README.md) |
| `Sensorformer` | [`configs/models/Sensorformer.toml`](../../configs/models/Sensorformer.toml) | time-series | [README](../../src/tsflab/models/sensorformer/README.md) |
| `SOFTS` | [`configs/models/SOFTS.toml`](../../configs/models/SOFTS.toml) | time-series | [README](../../src/tsflab/models/softs/README.md) |
| `Sonnet` | [`configs/models/Sonnet.toml`](../../configs/models/Sonnet.toml) | time-series | [README](../../src/tsflab/models/sonnet/README.md) |
| `SparseTSF` | [`configs/models/SparseTSF.toml`](../../configs/models/SparseTSF.toml) | time-series | [README](../../src/tsflab/models/sparsetsf/README.md) |
| `SRSNet` | [`configs/models/SRSNet.toml`](../../configs/models/SRSNet.toml) | time-series | [README](../../src/tsflab/models/srsnet/README.md) |
| `ST-SSDL` | [`configs/models/ST-SSDL.toml`](../../configs/models/ST-SSDL.toml) | spatiotemporal | [README](../../src/tsflab/models/st_ssdl/README.md) |
| `STAEformer` | [`configs/models/STAEformer.toml`](../../configs/models/STAEformer.toml) | spatiotemporal | [README](../../src/tsflab/models/staeformer/README.md) |
| `STDMAE` | [`configs/models/STDMAE.toml`](../../configs/models/STDMAE.toml) | spatiotemporal | [README](../../src/tsflab/models/stdmae/README.md) |
| `STDN` | [`configs/models/STDN.toml`](../../configs/models/STDN.toml) | spatiotemporal | [README](../../src/tsflab/models/stdn/README.md) |
| `StemGNN` | [`configs/models/StemGNN.toml`](../../configs/models/StemGNN.toml) | spatiotemporal | [README](../../src/tsflab/models/stemgnn/README.md) |
| `STGCN` | [`configs/models/STGCN.toml`](../../configs/models/STGCN.toml) | spatiotemporal | [README](../../src/tsflab/models/stgcn/README.md) |
| `STGODE` | [`configs/models/STGODE.toml`](../../configs/models/STGODE.toml) | spatiotemporal | [README](../../src/tsflab/models/stgode/README.md) |
| `STID` | [`configs/models/STID.toml`](../../configs/models/STID.toml) | spatiotemporal | [README](../../src/tsflab/models/stid/README.md) |
| `STNorm` | [`configs/models/STNorm.toml`](../../configs/models/STNorm.toml) | spatiotemporal | [README](../../src/tsflab/models/stnorm/README.md) |
| `STOP` | [`configs/models/STOP.toml`](../../configs/models/STOP.toml) | spatiotemporal | [README](../../src/tsflab/models/stop/README.md) |
| `STPGNN` | [`configs/models/STPGNN.toml`](../../configs/models/STPGNN.toml) | spatiotemporal | [README](../../src/tsflab/models/stpgnn/README.md) |
| `STTN` | [`configs/models/STTN.toml`](../../configs/models/STTN.toml) | spatiotemporal | [README](../../src/tsflab/models/sttn/README.md) |
| `STWave` | [`configs/models/STWave.toml`](../../configs/models/STWave.toml) | spatiotemporal | [README](../../src/tsflab/models/stwave/README.md) |
| `Sumba` | [`configs/models/Sumba.toml`](../../configs/models/Sumba.toml) | time-series | [README](../../src/tsflab/models/sumba/README.md) |
| `SVRForecasterTS` | [`configs/models/SVRForecasterTS.toml`](../../configs/models/SVRForecasterTS.toml) | time-series | [README](../../src/tsflab/models/svr_forecaster_ts/README.md) |
| `SVTime` | [`configs/models/SVTime.toml`](../../configs/models/SVTime.toml) | time-series | [README](../../src/tsflab/models/svtime/README.md) |
| `SWIFT` | [`configs/models/SWIFT.toml`](../../configs/models/SWIFT.toml) | time-series | [README](../../src/tsflab/models/swift/README.md) |
| `SymTime` | [`configs/models/SymTime.toml`](../../configs/models/SymTime.toml) | time-series | [README](../../src/tsflab/models/symtime/README.md) |
| `TCNForecasterTS` | [`configs/models/TCNForecasterTS.toml`](../../configs/models/TCNForecasterTS.toml) | time-series | [README](../../src/tsflab/models/tcn_forecaster_ts/README.md) |
| `TexFilter` | [`configs/models/TexFilter.toml`](../../configs/models/TexFilter.toml) | time-series | [README](../../src/tsflab/models/texfilter/README.md) |
| `TiDE` | [`configs/models/TiDE.toml`](../../configs/models/TiDE.toml) | time-series | [README](../../src/tsflab/models/tide/README.md) |
| `TimeAlign` | [`configs/models/TimeAlign.toml`](../../configs/models/TimeAlign.toml) | time-series | [README](../../src/tsflab/models/timealign/README.md) |
| `TimeBase` | [`configs/models/TimeBase.toml`](../../configs/models/TimeBase.toml) | time-series | [README](../../src/tsflab/models/timebase/README.md) |
| `TimeBridge` | [`configs/models/TimeBridge.toml`](../../configs/models/TimeBridge.toml) | time-series | [README](../../src/tsflab/models/timebridge/README.md) |
| `TimeCAP` | [`configs/models/TimeCAP.toml`](../../configs/models/TimeCAP.toml) | time-series | [README](../../src/tsflab/models/timecap/README.md) |
| `TimeEmb` | [`configs/models/TimeEmb.toml`](../../configs/models/TimeEmb.toml) | time-series | [README](../../src/tsflab/models/timeemb/README.md) |
| `TimeExpert` | [`configs/models/TimeExpert.toml`](../../configs/models/TimeExpert.toml) | time-series | [README](../../src/tsflab/models/timeexpert/README.md) |
| `TimeFilter` | [`configs/models/TimeFilter.toml`](../../configs/models/TimeFilter.toml) | time-series | [README](../../src/tsflab/models/timefilter/README.md) |
| `TimeKAN` | [`configs/models/TimeKAN.toml`](../../configs/models/TimeKAN.toml) | time-series | [README](../../src/tsflab/models/timekan/README.md) |
| `TimeMixer` | [`configs/models/TimeMixer.toml`](../../configs/models/TimeMixer.toml) | time-series | [README](../../src/tsflab/models/timemixer/README.md) |
| `TimeMosaic` | [`configs/models/TimeMosaic.toml`](../../configs/models/TimeMosaic.toml) | time-series | [README](../../src/tsflab/models/timemosaic/README.md) |
| `TimeO1` | [`configs/models/TimeO1.toml`](../../configs/models/TimeO1.toml) | time-series | [README](../../src/tsflab/models/timeo1/README.md) |
| `TimePerceiver` | [`configs/models/TimePerceiver.toml`](../../configs/models/TimePerceiver.toml) | time-series | [README](../../src/tsflab/models/timeperceiver/README.md) |
| `TimePro` | [`configs/models/TimePro.toml`](../../configs/models/TimePro.toml) | time-series | [README](../../src/tsflab/models/timepro/README.md) |
| `TimesNet` | [`configs/models/TimesNet.toml`](../../configs/models/TimesNet.toml) | time-series | [README](../../src/tsflab/models/timesnet/README.md) |
| `TimeXer` | [`configs/models/TimeXer.toml`](../../configs/models/TimeXer.toml) | time-series | [README](../../src/tsflab/models/timexer/README.md) |
| `TiRex` | [`configs/models/TiRex.toml`](../../configs/models/TiRex.toml) | quantile-output, time-series | [README](../../src/tsflab/models/tirex/README.md) |
| `TQNet` | [`configs/models/TQNet.toml`](../../configs/models/TQNet.toml) | time-series | [README](../../src/tsflab/models/tqnet/README.md) |
| `Transformer` | [`configs/models/Transformer.toml`](../../configs/models/Transformer.toml) | time-series | [README](../../src/tsflab/models/transformer/README.md) |
| `TSMixer` | [`configs/models/TSMixer.toml`](../../configs/models/TSMixer.toml) | time-series | [README](../../src/tsflab/models/tsmixer/README.md) |
| `TSRAG` | [`configs/models/TSRAG.toml`](../../configs/models/TSRAG.toml) | time-series | [README](../../src/tsflab/models/tsrag/README.md) |
| `UMixer` | [`configs/models/UMixer.toml`](../../configs/models/UMixer.toml) | time-series | [README](../../src/tsflab/models/umixer/README.md) |
| `VisiFold` | [`configs/models/VisiFold.toml`](../../configs/models/VisiFold.toml) | spatiotemporal | [README](../../src/tsflab/models/visifold/README.md) |
| `WaveNet` | [`configs/models/WaveNet.toml`](../../configs/models/WaveNet.toml) | time-series | [README](../../src/tsflab/models/wavenet/README.md) |
| `WDformer` | [`configs/models/WDformer.toml`](../../configs/models/WDformer.toml) | time-series | [README](../../src/tsflab/models/wdformer/README.md) |
| `WPMixer` | [`configs/models/WPMixer.toml`](../../configs/models/WPMixer.toml) | time-series | [README](../../src/tsflab/models/wpmixer/README.md) |
| `XGBoostTS` | [`configs/models/XGBoostTS.toml`](../../configs/models/XGBoostTS.toml) | time-series | [README](../../src/tsflab/models/xgboost_ts/README.md) |
| `xPatch` | [`configs/models/xPatch.toml`](../../configs/models/xPatch.toml) | time-series | [README](../../src/tsflab/models/xpatch/README.md) |
