# CV-LB

用于记录本地 CV 与 Kaggle 线上 LB 的对应关系，方便后续判断哪些改动是真正有效的。

## 当前记录

| 日期 | 实验目录 | 模型 | Local CV | Public LB | LB - CV | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-26 | `outputs/birdclef2026_gm/20260426_224316_tf_efficientnetv2_s.in21k_ft_in1k` | `tf_efficientnetv2_s.in21k_ft_in1k` | `0.664928` | `0.782` | `+0.117072` | Local CV 取自 `train.log` 中的 `Final OOF local CV` |
| 2026-04-27 | `outputs/birdclef2026_gm/20260427_001810_convnext_atto.d2_in1k` | `convnext_atto.d2_in1k` | `0.696086` | `0.841` | `+0.144914` | Local CV 取自 `train.log` 中的 `Final OOF local CV` |
| 2026-04-27 | `outputs/birdclef2026_gm/20260427_022128_convnext_atto.d2_in1k` | `convnext_atto.d2_in1k` | `0.711801` | `0.835` | `+0.123199` | Local CV 更高，但 Public LB 低于 `20260427_001810`，提示线下与线上并非单调对应 |
| 2026-04-27 | `outputs/birdclef2026_gm/20260427_160037_convnext_atto.d2_in1k` | `convnext_atto.d2_in1k` | `0.720155` | `0.848` | `+0.127845` | 当时刷新过稳定版 Stage 2 提交成绩，但后续已被 `20260428_164427_convnextv2_atto.fcmae_ft_in1k` 的 `0.851` 超过 |
| 2026-04-27 | `outputs/birdclef2026_gm/20260427_160037_convnext_atto.d2_in1k` | `convnext_atto.d2_in1k` | `0.720155` | `0.867` | `+0.146845` | 使用增强版 `birdclef2026_gm_kaggle_infer_ensemble.py` 的一条强基线；但后续已被 `20260428_164427_convnextv2_atto.fcmae_ft_in1k` 的 `0.871` 超过 |
| 2026-04-28 | `outputs/birdclef2026_gm/20260427_232617_eca_nfnet_l0.ra2_in1k` | `eca_nfnet_l0.ra2_in1k` | `0.758286` | `0.838` | `+0.079714` | 使用稳定版 `birdclef2026_gm_kaggle_infer.py` 推理；本地 CV 很高，但当前公榜仍低于最佳 `convnext_atto` 提交 |
| 2026-04-28 | `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.729685` | `0.851` | `+0.121315` | 当前最佳稳定版 Stage 2 单模型提交，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py` |
| 2026-04-29 | `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.729685` | `0.871` | `+0.141315` | 强力 Stage 2 + 增强版推理基线；但后续已被 `20260429_033312...` 的 `0.873` 超过 |
| 2026-05-01 | `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.734594` | `0.856` | `+0.121406` | `stage1 only cutmix` 版本，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py`；较旧 Stage 2 稳定版 `0.851` 有提升 |
| 2026-05-01 | `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.734594` | `0.861` | `+0.126406` | 同一 run 使用增强版推理脚本 `birdclef2026_gm_kaggle_infer_ensemble.py`，但关闭额外 TTA；快于三路 TTA，但低于同 run 的 `0.869` |
| 2026-05-01 | `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.734594` | `0.869` | `+0.134406` | 同一 run 使用增强版推理脚本 `birdclef2026_gm_kaggle_infer_ensemble.py`；优于标准推理，但仍略低于旧 Stage 2 增强版基线 `0.871` |
| 2026-05-01 | `outputs/birdclef2026_gm/{20260428_164427_convnextv2_atto.fcmae_ft_in1k,20260501_165002_convnextv2_atto.fcmae_ft_in1k}` | `convnextv2_atto.fcmae_ft_in1k x2` | `n/a` | `0.868` | `n/a` | 双 Stage 2 run ensemble，使用增强版推理脚本且关闭额外 TTA；比单个新 run 的 no-TTA `0.861` 更强，但仍略低于单 run + TTA 的 `0.869` |
| 2026-05-02 | `outputs/birdclef2026_gm/20260502_211407_eca_nfnet_l0.ra2_in1k` | `eca_nfnet_l0.ra2_in1k` | `0.773772` | `0.852` | `+0.078228` | NFNet + stage1 cutmix/label smoothing 版本，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py`；推理约 70 分钟，不适合作为最终线上模型，但可作为 teacher 候选 |
| 2026-05-05 | `outputs/birdclef2026_gm/20260505_145600_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.749210` | `0.842` | `+0.092790` | 固定 deterministic seed 后的一条 ConvNeXt run，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py`；Local CV 提升但 Public LB 未同步提升 |
| 2026-05-05 | `outputs/birdclef2026_gm/20260505_182506_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.761989` | `0.840` | `+0.078011` | `csiro_conv_v1` head 对照，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py`；Local CV 高于旧主线但 Public LB 偏低 |
| 2026-05-05 | `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k` | `convnextv2_atto.fcmae_ft_in1k` | `0.776060` | `0.843` | `+0.066940` | 当前 CNN 侧最高 Local CV 主线，`csiro_conv_v1` + stage1 waveform mixup only，使用标准推理脚本 `birdclef2026_gm_kaggle_infer.py`；线上仍明显弱于 Perch |
| 2026-04-28 | `outputs/birdclef2026_gm_stage3_pseudo/20260428_220148_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` | `convnextv2_atto.fcmae_ft_in1k` | `0.719661` | `0.854` | `+0.134339` | 早期 Stage 3 pseudo 提交，使用 `birdclef2026_gm_kaggle_infer_stage3.py`；后续已被 `20260429_033312...` 的 `0.855` 超过 |
| 2026-04-29 | `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` | `convnextv2_atto.fcmae_ft_in1k` | `0.723835` | `0.855` | `+0.131165` | 同一 Stage 3 run 使用 `birdclef2026_gm_kaggle_infer_stage3.py` 的成绩；较上一版 Stage 3 有提升 |
| 2026-04-29 | `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` | `convnextv2_atto.fcmae_ft_in1k` | `0.723835` | `0.873` | `+0.149165` | 当时表内最高 Public LB；同一 Stage 3 run 使用增强版 `birdclef2026_gm_kaggle_infer_ensemble.py` 推理，说明 Stage 3 + 增强版推理可以叠加收益 |
| 2026-05-07 | `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` | `convnextv2_atto.fcmae_ft_in1k` | `0.811590` | `0.839` | `+0.027410` | Perch+CNN white-list pseudo teacher 训练出的 Stage3-only 提交；本地 Stage3 OOF 明显高于旧 Stage3，但 Public LB 低于主线 CNN `0.843` 和旧 Stage3 增强推理 `0.873`，暂不作为 standalone 主线 |
| 2026-05-06 | `outputs/perch_context_deploy_labeled_all_v1` | `Perch v2 frozen + context LogReg` | `0.867057` | `0.894` | `+0.026943` | 当前 Perch-only 最高 Public LB；Perch ONNX backend 推理，后接 5-fold context LogisticRegression artifact；本地 CV 使用 online-like OOF，即 fitted classes 用 LogReg 概率、unfitted classes 用 `sigmoid(raw Perch logits)`；旧训练 summary 的 `0.790411` 是 mixed-scale OOF，不再作为 LB 对比口径 |
| 2026-05-07 | `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027` | `Perch v2 frozen + context MLP` | `0.878995` | `0.891` | `+0.012005` | MLP 本地 CV 高于 LogReg，但 Public LB 低于 LogReg 的 `0.894`；说明这版 MLP 可能对 739-row local OOF 更贴合，但线上泛化不如 LogReg，暂不替代 Perch-only 主线 |
| 2026-05-10 | `outputs/perch_spatial_mamba_labeled_all_nopca_noraw_v1` | `Perch v2 frozen spatial_embedding + Mamba head` | `0.875406` | `0.899` | `+0.023594` | Perch spatial 单模首个线上提交；使用 `spatial_embedding [B,16,4,1536] -> mean over freq -> [B,16,1536]`，不做 PCA，不拼接 raw Perch score，2 层 Mamba-style block + MLP head；Public LB 高于 Perch context LogReg `0.894`，但仍低于 Perch+Stage3 融合 `0.916` |
| 2026-05-11 | `outputs/perch_spatial_mamba_labeled_all_mean_perchmambav1_cnn195634folds_nopca_noraw_v1` | `Perch v2 frozen spatial_embedding + strict PerchMambaHead` | `0.889574` | `0.897` | `+0.007426` | 使用 CNN `20260505_195634` 对齐 3fold；`spatial_embedding [B,16,4,1536] -> mean(freq) -> 2x LocalMambaBlock -> avg pool -> MLP`，无 PCA、无 raw Perch score；线上低于旧 spatial Mamba 首提 `0.899`，但 CV 口径更利于后续与 CNN/stage3/pseudo 对齐 |
| 2026-05-11 | `outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1` | `Perch v2 frozen spatial_embedding + strict PerchMambaHead + conservative pseudo` | `0.890960` | `0.898` | `+0.007040` | 使用 LB `0.916` 的 Perch+Stage3 teacher conservative pseudo：`row_min_max_prob=0.93, margin=0.25, top_k=1, pseudo_loss_weight=0.25`；相对 labeled-only 本地 `+0.001386`、线上 `+0.001`，说明 conservative pseudo 是一致小正收益 |
| 2026-05-06 | `outputs/blend_cnn195634_perch_context_deploy_v1` | `Perch context LogReg + ConvNeXt CNN logit blend` | `0.883238` | `0.903` | `+0.019762` | 此前最高 Public LB；线上提交使用 `Perch 0.82 / CNN 0.18` logit 融合；Perch 来自 `outputs/perch_context_deploy_labeled_all_v1`，CNN 来自 `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k`；fine grid 最优为 `Perch 0.824 / CNN 0.176`，Local CV `0.883276` |
| 2026-05-07 | `outputs/whitelist_blend_stage3perchcnn_perch_logreg_v1` | `Perch context LogReg + Stage3 CNN logit blend` | `0.905907` | `0.916` | `+0.010093` | 此前最高 Public LB；Perch 来自 `outputs/perch_context_deploy_labeled_all_v1`，Stage3 CNN 来自 `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；白名单 OOF 选参为 `perch_weight=0.74, file_scale=max_power(0.4), smooth=plain(0.15)` |
| 2026-05-09 | `outputs/whitelist_blend_threeway_perch_stage3_cnn195634_v1` | `Perch context LogReg + Stage3 CNN + base CNN` | `0.906158` | `pending` | `pending` | 三路白名单 OOF 选参，只用本地 OOF/labels：`logit, Perch=0.741275, Stage3=0.253725, base CNN=0.005, file_scale=topk_mean(2), smooth=plain(0.10)`；相对二路 `0.905907` 仅提升 `+0.000251`，收益很小，若提交应视为低风险小幅试探 |
| 2026-05-11 | `20260511-perch-mam-attn-cnn-cv9283` upload package | `Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention` | `0.9283` | `0.920` | `-0.0083` | 当前最高 Public LB；线上 unified 脚本共享一次 Perch ONNX 推理给三个 Perch head，融合权重为 `PerchLR=0.25, Mamba=0.30, Stage3=0.15, Attention=0.30`，并使用 `file_scale_topk=2`；说明 Perch 多 head + CNN/Stage3 多样性有效，但本地 CV 对该强 ensemble 略偏乐观 |
| 2026-05-14 | `outputs/whitelist_blend_unified_strict_raw_w100_20260514` | `Unified PerchLR + PerchMamba + old Stage3 CNN + PerchAttention + strict-teacher RawWave` | `0.934884` | `pending` | `pending` | 新本地最高 OOF；只替换 RawWave 为 `outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_164133_raw_wave_conv_tokenizer_base_strictteacher_w100`，保留旧 Stage3；最优权重为 `PerchLR=0.2275, Mamba=0.273, Stage3=0.1365, Attention=0.273, RawWave=0.09, file_scale_topk=2`；只替换 strict Stage3 或 strict Stage3+strict RawWave 都不如该组合 |
| 2026-06-01 | `20260531-ensemble-cv9422 safe family3 no RawWave no TTA` | `Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention + PerchSSM` | `0.937399` | `0.922` | `-0.015399` | `family3 + file_scale_topk2`，关闭 RawWave、Mamba TTA、Stage3 TTA 以保证线上时间；相对旧四路 `0.920` 有小幅线上提升，但 local-to-LB gap 明显扩大，说明 SSM/family3 的本地增益只有一部分转移到线上 |
| 2026-06-02 | `20260531-ensemble-cv9422 retuned safe family3 no RawWave no TTA` | `Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention + PerchSSM` | `0.938967` | `0.922` | `-0.016967` | 只重调全局权重为 `PerchLR=0.20, Mamba=0.112, Stage3=0.1375, Attention=0.208, SSM=0.3425`，仍关闭 RawWave 和所有 TTA；线上与旧安全版持平，说明该 OOF 权重提升未转成 Public LB |

## 记录说明

- `Local CV` 优先记录 `stage2` 最终 OOF 对应的 `Final OOF local CV`。
- Perch deploy artifact 的 `Local CV` 记录 online-like OOF 口径；不要使用旧训练 summary 里的 mixed-scale `probe_oof_auc` 做 LB 对比。
- `Public LB` 记录 Kaggle 公榜分数。
- `LB - CV` 用于观察线下与线上的 gap。
- 如果同一个模型有多次提交，建议按提交版本分别追加，不要覆盖历史记录。
- 当前建议分三种口径看“最佳”：
- 标准 Stage 2 单模型提交：`0.851`
- Stage 3 pseudo 提交：`0.873`
- 增强版 CNN inference-only 口径最高分：`0.873`
- Perch context LogReg 口径最高分：`0.894`
- Perch context MLP 对照分：`0.891`
- Perch + Stage3 CNN logit 融合分：`0.916`
- Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention 总体最高分：`0.920`
- Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention + PerchSSM 安全版最高 Public LB：`0.922`
- Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention + strict-teacher RawWave 本地最高 CV：`0.934884`，Public LB 待验证

## 后续追加模板

```md
| 日期 | 实验目录 | 模型 | Local CV | Public LB | LB - CV | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| YYYY-MM-DD | `outputs/...` | `model_name` | `0.xxxxxx` | `0.xxxxxx` | `+0.xxxxxx` | 备注 |
```
