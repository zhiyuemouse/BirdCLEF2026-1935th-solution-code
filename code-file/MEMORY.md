# MEMORY

用于记录这个 BirdCLEF 2026 项目的关键约束、当前方案、环境问题、实验状态与后续实验日志。

## 1. 项目约束

- 当前项目是 BirdCLEF 2026 Kaggle 比赛项目。
- 只能操作当前文件夹下的文件，不允许操作外部文件。
- 不修改原始的 `train.py` 和 `infer.py`，新方案通过新建脚本实现。
- 当前目标非常明确：提分，冲击 Kaggle 金牌。
- 当前主要使用 `conda` 的 `transformers` 环境。
- 为本地 `Perch v2` 探索，当前额外创建了独立的 `conda` 环境 `perch`，并在其中安装了 `TensorFlow 2.20`。

## 2. 当前核心文件

- [AGENT.md](AGENT.md)：项目约束与目标。
- [MEMORY.md](MEMORY.md)：项目记忆与实验记录。
- [CV-LB.md](CV-LB.md)：记录本地 CV 与 Kaggle 线上 LB 的对应关系。
- [代码思路.md](代码思路.md)：训练、分折、防泄露、CV、特征与推理思路说明。
- [birdclef2026_gm_train.py](birdclef2026_gm_train.py)：新的训练脚本。
- [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py)：当前稳定版 Kaggle 线上推理脚本。
- [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py)：增强版推理脚本，支持多实验目录 ensemble、TTA、平滑和 soundscape 级后处理。
- [birdclef2026_gm_make_pseudo_labels.py](birdclef2026_gm_make_pseudo_labels.py)：生成 `train_soundscapes` 伪标签的脚本，支持单模或多模型 teacher。
- [birdclef2026_gm_train_stage3_pseudo.py](birdclef2026_gm_train_stage3_pseudo.py)：基于 Stage 2 fold checkpoint 做 Stage 3 pseudo finetune 的脚本。
- [run_birdclef2026_gm_pseudo_stage3.sh](run_birdclef2026_gm_pseudo_stage3.sh)：`pseudo / stage3 / both` 三种模式的运行模板脚本。
- [birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py)：本地运行 `Perch v2` 并生成等价 `perch-meta` 缓存的脚本。
- [birdclef2026_perch_context_train.py](birdclef2026_perch_context_train.py)：训练 `Perch embedding/raw score/context feature -> context LogReg` deploy artifact 的脚本。
- [birdclef2026_perch_context_lgbm_train.py](birdclef2026_perch_context_lgbm_train.py)：训练 `Perch context LGBM` 对照 head 的脚本；当前已验证不优于 LogReg。
- [birdclef2026_perch_context_mlp_train.py](birdclef2026_perch_context_mlp_train.py)：训练 `Perch context MLP` head 的脚本；当前 Perch-only 本地 CV 最强候选来自该脚本。
- [birdclef2026_perch_kaggle_infer_context_logreg.py](birdclef2026_perch_kaggle_infer_context_logreg.py)：Kaggle 线上 Perch context artifact 推理脚本，当前同时支持 `perch_context_logreg` 与 `perch_context_mlp` artifact。
- [birdclef2026_perch_protossm_cv.py](birdclef2026_perch_protossm_cv.py)：基于本地 `Perch` cache 做 `12 x 5s` 时序建模 honest local CV 的脚本，当前实现为轻量 `ProtoSSM-lite / BiGRU residual` 口径。

## 3. 当前训练方案摘要

- 训练分成两个阶段。
- Stage 1 使用 `input/train.csv + input/train_audio` 做预训练。
- Stage 2 使用 `input/train_soundscapes_labels.csv + input/train_soundscapes` 做主训练与可信 local CV。
- 最终可信 local CV 只看 Stage 2 的 OOF，不看 Stage 1 的验证分数。

## 4. 防数据泄露与 CV 口径

- Stage 2 的 fold 不是按 5 秒片段随机分，而是按完整 `filename` 分组。
- 同一个 1 分钟 soundscape 中的所有 5 秒窗口，必须落在同一个 fold 内。
- 不会出现同一个 soundscape 的前半段进训练、后半段进验证的情况。
- local CV 只在人工标注的 soundscape 5 秒窗口上计算。
- 当前本地 Stage 2 local CV 实际覆盖 `66` 条 1 分钟 soundscape 文件，共 `739` 个有标签 5 秒窗口；理论满窗应为 `66 x 12 = 792`，但其中 `7` 个文件只提供了部分标注窗口，因此缺少 `53` 个窗口。
- 最终 CV 是所有 fold 的 soundscape OOF 汇总后的 macro ROC-AUC。

## 5. 特征、模型与损失

- 不依赖 `torchaudio`，改为 `numpy + torch.stft` 手工构建 mel 频谱特征。
- 输入是 3 通道 mel 频谱图，最终 resize 到 `(3, 256, 320)`。
- 主干网络通过 `timm.create_model(...)` 构建。
- 分类头当前支持两种口径：
- `linear`：`Dropout + Linear` 多标签输出头；
- `csiro_conv_v1`：在 backbone feature sequence 上叠加两层局部时序卷积块后再池化分类；
- 当前本地最强主线使用 `csiro_conv_v1`。
- 损失函数使用 `BCEWithLogitsLoss`。
- CV 指标使用纯 `numpy` 实现的 macro ROC-AUC，跳过无正样本或退化类别。

## 6. 环境问题与规避方案

- `sklearn` 曾因依赖链中的 `scipy` 问题报错，因此训练脚本改成不依赖 `sklearn` 的 `numpy` 版 AUC 计算。
- `torchaudio` 曾因缺失 `libtorch_cuda_cpp.so` 等动态库无法导入，因此训练和推理脚本都移除了 `torchaudio` 依赖。
- 当前策略是不在环境修复上浪费时间，优先采用代码规避方案，先把实验跑通并完成提分验证。
- `perch` 环境当前没有预装 `pyarrow` / `fastparquet`，因此若要直接保存 `full_perch_meta.parquet`，需要额外安装其中之一；[birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py) 已支持在缺少 parquet 依赖时改用 `.csv` 做烟雾测试。
- `perch` 环境当前也没有 `tqdm`，但 [birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py) 已做兼容处理：无 `tqdm` 时仍可正常运行，只是不显示进度条。

## 7. 已完成的工程修复

- 为训练脚本加入了每次运行目录下的 `train.log` 日志保存。
- 对 `tqdm` 的日志进行了收敛处理，只保留完整进度条，不保存每一步刷新。
- 验证阶段显存爆掉的问题已经处理，主要方法是减小默认 `eval_batch_size`、使用 `torch.inference_mode()`，并在验证循环中及时释放张量。
- 训练脚本的 `parse_args()` 已开放四个学习率参数：`stage1_backbone_lr`、`stage1_head_lr`、`stage2_backbone_lr`、`stage2_head_lr`，方便通过命令行或 shell 脚本切换训练配置。
- 默认早停耐心值已从 `4` 调整为 `5`，即连续 `5` 轮验证集 `valid_auc` 未提升才会触发 early stopping。
- 为 `ViT / DeiT / BEiT / EVA / Swin / XCiT / MViT` 这类 Transformer backbone 补充了更稳的 AMP 策略：
- `--amp-mode auto` 下，若 GPU 支持 `bf16`，则优先使用 `bf16`；
- 若 GPU 不支持 `bf16`，则自动回退到 `fp32`，避免 `fp16` 混合精度在 ViT 上更容易出现 `nan`。
- 训练脚本新增了 `--disable-amp`、`--amp-mode {auto,fp16,bf16,off}`、`--grad-clip-norm` 三个参数，方便针对不同 backbone 单独控制混合精度和梯度裁剪策略。
- 为 ViT 类 backbone 增加了默认梯度裁剪策略：当未手动指定 `--grad-clip-norm` 时，自动使用 `1.0`，以降低训练中梯度爆炸导致的数值不稳定风险。
- 训练循环已经加入非有限数值保护：若某个 batch 的 `loss` 或梯度出现 `inf/nan`，则跳过该 batch，并在使用 `GradScaler` 时回退 scale；只有真正执行了 `optimizer.step()` 才会继续 `scheduler.step()`。
- 修复了 `bf16` 验证阶段的兼容性问题：在将预测转成 `numpy` 前，先显式转换为 `float32`，避免 `torch 1.11` 下 `BFloat16` 不能直接 `.cpu().numpy()` 的报错。
- 已为 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 接入 batch 级 `mixup / cutmix` 增强，参数为：
- `--mixup-alpha`
- `--mixup-prob`
- `--cutmix-alpha`
- `--cutmix-prob`
- 已为 [birdclef2026_gm_train_stage3_pseudo.py](birdclef2026_gm_train_stage3_pseudo.py) 接入同口径的 batch 级 `mixup / cutmix` 增强。
- 当前实现默认关闭，仅在训练阶段生效，不污染验证；Stage 3 中 `pseudo loss weight` 也会按混合系数同步插值。
- 已同步更新 [run_birdclef2026_gm_train.sh](run_birdclef2026_gm_train.sh) 与 [run_birdclef2026_gm_stage3.sh](run_birdclef2026_gm_stage3.sh)，支持在脚本顶部直接修改 mix augmentation 开关。
- 已为 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 进一步加入分阶段增强控制：
- 兼容保留原有全局参数 `--mixup-alpha / --mixup-prob / --cutmix-alpha / --cutmix-prob`；
- 新增 `--stage1-*` 与 `--stage2-*` 两套 batch augmentation 开关，便于单独测试 `stage1 only cutmix`、`stage2 no cutmix` 这类更细粒度实验；
- 当前 [run_birdclef2026_gm_train.sh](run_birdclef2026_gm_train.sh) 已切到首个分阶段模板：`stage1_cutmix_alpha=0.5, stage1_cutmix_prob=0.10`，`stage2_cutmix_* = 0`。
- 已为 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 接入 fold-safe 的 Perch -> CNN layer-to-layer distillation：
- 使用 `perch_spatial_cache_labeled_all/perch_spatial_meta.parquet` + `perch_spatial_arrays.npz` 作为 teacher cache；
- 默认对齐 `spatial_tokens`，stage2 训练时通过 `perch_distill_weight` 叠加 token-level MSE；
- 已用 smoke test 验证整条链路可跑通，stage2 中能稳定看到 `train_distill_loss`，当前首个可用配置是 `csiro_conv_v1 + weight=0.05`。
- 随后的正式训练结果也已确认：`convnextv2_atto.fcmae_ft_in1k + csiro_conv_v1 + stage1 waveform mixup only + Perch distill weight=0.05` 的最终 OOF 为 `0.729443`，低于同主线不加 distill 的 `0.776060`，因此当前 distill 仅算“通路验证成功”，不能作为主线提分方案。
- 已为当前主线训练补齐完整的确定性控制，避免“同代码重复跑但结果漂移”：
- [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 现在会同时固定 `random / numpy / torch / cuda` 随机种子；
- 显式关闭 `cudnn.benchmark`，开启 `cudnn.deterministic` 与 `torch.use_deterministic_algorithms(True, warn_only=True)`；
- 为 `DataLoader worker`、`WeightedRandomSampler`、各个 train/valid loader 单独注入确定性 seed/generator；
- [run_birdclef2026_gm_train.sh](run_birdclef2026_gm_train.sh) 已补充 `SEED=2026`、`PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG` 并显式传入 `--seed`；
- 新日志中会打印 `Seed=... | deterministic_algorithms=... | cudnn_deterministic=... | cudnn_benchmark=...`，方便后续核对复现口径。
- 三份推理脚本 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py)、[birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py)、[birdclef2026_gm_kaggle_infer_stage3.py](birdclef2026_gm_kaggle_infer_stage3.py) 也已补齐同级别的确定性 seed 控制，并会在日志中打印 deterministic 状态。
- 已修复稳定版推理脚本 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 的 soundscape 切窗语义，使其与训练口径对齐：
- 每个 `row_id` 仍按 `end_sec` 命名；
- 真实取窗起点改为 `end_sec - 5`；
- `clip_seconds` 表示从该起点向右看的上下文长度。
- 已为训练与推理统一接入 `head_type` 概念，`config.json` 中会记录 `linear / csiro_conv_v1`，三份推理脚本均可自动按该字段恢复模型头。
- 已为 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 加入 `mixup_domain={image,waveform}`，当前支持 waveform-level mixup，并在训练日志中额外统计 `waveform_mixup_batches / waveform_mixup_samples`。

## 8. 推理脚本关键约束

- 推理脚本会读取官方 `sample_submission.csv` 的列顺序作为最终类别顺序。
- 训练输出维度与最终 `submission.csv` 的 columns 是一一对应的。
- 推理脚本支持 `--soundscapes-dir` 手工指定输入目录，方便线上切换 `test_soundscapes` 和 `train_soundscapes` 做 debug。
- Kaggle 线上代码文件默认位于 `/kaggle/working/`，推理脚本已按该目录结构兼容。
- 当前保留两套推理脚本。
- 稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 不做额外实验性增强，优先保证可稳定提交。
- 增强版 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 支持多 `model_root` ensemble、`tta_offsets`、`smoothing_kernel` 和 `soundscape_top_k` 后处理。
- 三份推理脚本当前都支持从 `config.json` 自动读取 `head_type`，因此已兼容 `csiro_conv_v1` 这类非线性头实验。
- 稳定版推理脚本当前已对齐训练的 5 秒标注窗口语义，避免线上推理与本地 Stage 2 评估使用不同切窗规则。
- 在确认 `site prior` 线上明显失效后，已将主线增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 中的 `site prior` 相关参数、加载逻辑和预测后调整逻辑全部移除，恢复到不含 `site` 分支的主线推理版本，以减少线上推理开销并避免误开该后处理。
- [birdclef2026_gm_local_infer_eval.py](birdclef2026_gm_local_infer_eval.py) 已同步移除旧的 `site prior` 依赖，保证本地 inference-debug 口径与当前主线增强推理脚本一致。
- 已新增 [birdclef2026_gm_local_infer_grid.py](birdclef2026_gm_local_infer_grid.py)，用于在本地低成本扫单模型或多模型的 `TTA / smoothing / soundscape_top_k` 组合，输出 `grid_results.csv` 与最佳配置摘要，方便先做 inference-side trick 排序，再决定是否值得上线上提交。
- 但最新一轮线上验证进一步说明：这个 `local infer grid` 只能当作 debug / ranking 信号，不能直接当线上最优配置代理。
- 对 `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k`，本地 grid 更偏向 `no TTA + light smoothing`；
- 但线上实测显示，同一 run 关闭额外 `TTA` 只有 `0.861`，而开启此前增强版推理口径时可到 `0.869`；
- 这再次证明：`train_soundscapes` 上的本地 inference-debug 分数会高估某些“省算力配置”的价值，不能拿来直接替代真正的线上判断。

## 9. 当前实验状态

- `tf_efficientnetv2_s.in21k_ft_in1k` 的训练、推理、提交流程已经打通。
- `tf_efficientnetv2_s.in21k_ft_in1k` 当前已知结果为：Local CV `0.664928`，Public LB `0.782`。
- `convnext_atto.d2_in1k` 当前已知结果为：Local CV `0.696086`，Public LB `0.841`。
- `convnext_atto.d2_in1k` 另一条新实验 `20260427_022128_convnext_atto.d2_in1k` 当前已知结果为：Local CV `0.711801`，Public LB `0.835`。
- `20260427_160037_convnext_atto.d2_in1k` 当前已知结果为：Local CV `0.720155`，Public LB `0.848`。
- 使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 且仅使用单个 `model_root=20260427_160037_convnext_atto.d2_in1k` 时，线上得分进一步提升到 `0.867`。
- `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k` 当前已知结果为：Stage 2 Local CV `0.729685`，使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 推理时 Public LB `0.851`。
- 同一个 `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`，使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 在线上推理时，Public LB 进一步提升到 `0.871`。
- `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k` 当前已知结果为：Stage 2 Local CV `0.734594`，使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 推理时 Public LB `0.856`。
- 同一个 `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k`，使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 且关闭额外 `TTA` 时，Public LB 为 `0.861`。
- 同一个 `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k`，使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 在线上推理时，Public LB 进一步提升到 `0.869`。
- `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k` 与 `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k` 的双 Stage 2 run ensemble，在增强版推理脚本下关闭额外 `TTA` 时，Public LB 为 `0.868`。
- `outputs/birdclef2026_gm_stage3_pseudo/20260428_220148_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` 当前已知结果为：Stage 3 Local CV `0.719661`，使用 [birdclef2026_gm_kaggle_infer_stage3.py](birdclef2026_gm_kaggle_infer_stage3.py) 推理时 Public LB `0.854`。
- `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` 当前已知结果为：Stage 3 Local CV `0.723835`，使用 [birdclef2026_gm_kaggle_infer_stage3.py](birdclef2026_gm_kaggle_infer_stage3.py) 推理时 Public LB `0.855`。
- 同一个 `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`，使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 在线上推理时，Public LB 进一步提升到 `0.873`。
- 当前建议分三种口径看“最佳成绩”：
- 标准 Stage 2 单模型提交最佳：`0.856`
- Stage 3 pseudo 提交最佳：`0.873`
- 增强版 inference-only 口径最高分：`0.873`
- 这说明当前 local CV 与 Public LB 具备相关性，但并不是严格单调；更高的本地 CV 不一定带来更高的公榜分数。
- 同时这也说明当前增强版推理脚本本身就是一个有效提分点，即使不做多模型 ensemble，仅做 inference-side 增强也能显著提升 LB。
- `convnext_atto.d2_in1k` 仍然是当前很强的 teacher / inference-side 候选；而当前最高的增强版推理分数口径已经更新为 `20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo + birdclef2026_gm_kaggle_infer_ensemble.py = 0.873`。
- 当前工程目录下已确认存在以下 backbone 权重：`ckpt/tf_efficientnetv2_s.in21k_ft_in1k.pth`、`ckpt/convnext_atto.d2_in1k.pth`、`ckpt/tf_efficientnet_b0.ns_jft_in1k.pth`。
- 当前 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 的默认模型已经切换为 `convnext_atto.d2_in1k`。
- 结合 BirdCLEF 2025 top 方案的经验，当前下一步重点之一是尝试 `tf_efficientnet_b3.ns_jft_in1k` 这类在往年高排名方案中高频出现的 backbone。
- 当前下一步重点还包括：在不影响稳定版提交流程的前提下，验证增强版推理脚本的多模型 ensemble、overlap/TTA、时间平滑与 soundscape 级后处理是否能继续提升 LB。
- 在恢复分支 `recover-07346-mainline` 上，当前已确认的稳定单模型 baseline 为：
- `outputs/birdclef2026_gm/20260505_145600_convnextv2_atto.fcmae_ft_in1k`
- `Final OOF local CV = 0.749210`
- 使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 线上推理时，Public LB `0.842`。
- `outputs/birdclef2026_gm/20260505_182506_convnextv2_atto.fcmae_ft_in1k` 为首个 `csiro_conv_v1 + stage1 only cutmix` 结果：
- `Final OOF local CV = 0.761989`
- 使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 线上推理时，Public LB `0.840`。
- 这说明 `csiro_head` 在当前小规模本地 CV 上给出了明显正收益，但单模公榜暂时没有同步受益，因此它更适合作为“多样性候选 / teacher 候选”，而不是直接替代 `145600` 成为唯一提交流程。
- 当前这条恢复主线上的最高已验证本地 CV 已进一步刷新为：
- `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k`
- `head_type=csiro_conv_v1`
- `stage1 waveform mixup only: alpha=0.2, prob=0.1`
- `Final OOF local CV = 0.776060`
- 使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 线上推理时，Public LB `0.843`。
- 这是当前这条分支上本地最强的已验证配置，也是当前恢复主线里已知的单模型稳定版推理最高公榜分数；公榜只给了弱确认，但没有反向打脸本地 CV。
- 当前已完成的 waveform mixup sweep 结论也比较明确：
- `alpha=0.1, prob=0.1` -> `0.725502`
- `alpha=0.3, prob=0.1` -> `0.742956`
- `alpha=0.2, prob=0.2` -> `0.728587`
- 因此当前 `stage1 waveform mixup` 的局部最优点仍然是 `alpha=0.2, prob=0.1`。
- `waveform mixup + cutmix` 的联合增强已经验证过，结果为：
- `outputs/birdclef2026_gm/20260505_220045_convnextv2_atto.fcmae_ft_in1k`
- `Final OOF local CV = 0.759318`
- 明显低于 `waveform mixup only` 的 `0.776060`，因此当前不建议继续把两者叠加作为主线。
- 轻量 `stage2 waveform mixup = 0.1 / 0.05` 已完成验证：
- `outputs/birdclef2026_gm/20260506_022923_convnextv2_atto.fcmae_ft_in1k`
- `Final OOF local CV = 0.740252`
- 该结果明显低于 `stage1 waveform mixup only` 的 `0.776060`，说明 stage2 mixup 当前是稳定负收益，不建议继续作为主线。
- `tf_efficientnetv2_b0.in1k + csiro_conv_v1 + stage1 waveform mixup only` 也已完成一次对照：
- `outputs/birdclef2026_gm/20260506_014724_tf_efficientnetv2_b0.in1k`
- `Final OOF local CV = 0.764883`
- 该结果低于当前最强的 `convnextv2_atto + csiro_conv_v1 + stage1 waveform mixup only`，但仍可作为模型互补候选观察。
- 当前已经开始搭建完整的 pseudo label 路线。
- 在最新一轮保守版 pseudo 实验模板中，teacher / student 默认已经切换到 `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`，因为它具备较强的 Stage 2 单模型成绩 `0.851`，并且其延伸出的 Stage 3 路线目前已经拿到全表最高的 `0.873`。
- 当前 pseudo label 方案默认采用 `fold-specific` 方式生成，即每个 fold 只使用对应 fold 的 Stage 2 checkpoint 给未标注 `train_soundscapes` 打伪标签，尽量保持后续 Stage 3 local CV 的可信度。
- 当前 Stage 3 方案会从已有 Stage 2 `fold_x/stage2_foldx_best.pth` 初始化学生模型，并将真实标注 soundscape 片段与伪标签片段混合训练。
- 当前已经有一份可直接运行的 shell 模板 [run_birdclef2026_gm_pseudo_stage3.sh](run_birdclef2026_gm_pseudo_stage3.sh)，可通过修改顶部参数快速切换 teacher、student、阈值、学习率和 epoch。
- 此外当前还维护两份更适合日常直接运行的独立脚本：
- [run_birdclef2026_gm_pseudo.sh](run_birdclef2026_gm_pseudo.sh) 负责生成 pseudo label，默认使用 `convnextv2_atto.fcmae_ft_in1k` teacher，并采用更保守的过滤参数：`prob_threshold=0.35`、`row_min_max_prob=0.85`、`top_k_labels=2`。
- [run_birdclef2026_gm_stage3.sh](run_birdclef2026_gm_stage3.sh) 负责 Stage 3 训练，默认使用同一个 `convnextv2_atto.fcmae_ft_in1k` 作为 student 基座，并采用更像轻量 finetune 的参数：`stage3_epochs=3`、`freeze_backbone_epochs=1`、`max_pseudo_rows=5000`、`pseudo_loss_weight=0.15`、`pseudo_sampler_weight=0.05`。
- 当前 [run_birdclef2026_gm_stage3.sh](run_birdclef2026_gm_stage3.sh) 还支持按 `PSEUDO_SUFFIX=conservative_fold_specific_convnextv2_atto_fcmae` 自动寻找最新一份 pseudo 目录，通常不需要再手动填写 `PSEUDO_ROOT`。
- `outputs/birdclef2026_gm/20260427_232617_eca_nfnet_l0.ra2_in1k` 当前已知结果为：Local CV `0.758286`，使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 推理时 Public LB `0.838`。
- 这是一个很重要的对照样本：`eca_nfnet_l0.ra2_in1k` 的本地 CV 明显更高，但公榜仍然没有超过当前标准 Stage 2 最佳的 `convnextv2_atto.fcmae_ft_in1k`，说明当前线下 CV 和线上泛化之间仍存在明显 gap，teacher 选择不能只看本地 CV。
- 当前正在尝试更大的 DINOv3 / ViT / ConvNeXt 系 backbone。
- 其中 ViT 训练链路上的两个主要稳定性问题已经定位并修复：
- `fp16` 混合精度下容易出现 `nan`；
- `bf16` 验证阶段在 `torch 1.11` 下不能直接转 `numpy`。
- 已完成 smoke test。
- `outputs/pseudo_labels/20260427_225558_smoke_pseudo_test` 已成功生成一份小规模伪标签。
- `outputs/birdclef2026_gm_stage3_pseudo/20260427_225809_convnext_atto.d2_in1k_stage3_pseudo` 已成功完成 `fold 0` 的 Stage 3 烟雾测试。
- 上述 smoke test 主要用于验证脚本链路、参数衔接和产物格式是否正确，不用于与正式实验的 Local CV 做横向比较。
- 这是当前 pseudo 路线里一个非常重要的现象：Stage 3 的本地 OOF CV 低于对应 Stage 2，但线上 Public LB 反而略有提升。
- 这说明当前 Stage 2 soundscape OOF 指标虽然仍然重要，但对 pseudo finetune 之后的线上泛化并不完全等价；后续评估 pseudo 策略时，不能只看 OOF 是否上涨，还要结合真实提交结果一起判断。
- 最新这版 `20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` 也支持这一判断：它的 Stage 3 OOF CV 从上一版 `0.719661` 提升到 `0.723835`，线上 Public LB 也从 `0.854` 提升到 `0.855`，说明在更保守的 pseudo 配置下，这条路线仍然有继续优化的空间。
- 更进一步地，同一个 `20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` 在增强版推理脚本下还能把 Public LB 从 `0.855` 再推到 `0.873`，说明“更保守的 pseudo 训练”与“增强版 inference-side 技巧”是可以叠加的，而不是二选一。
- 当前已经开始系统查看 `现阶段线上code区开源高分代码` 中的公开 notebook。
- 结论是：这批高分 notebook 大致分成两类。
- 一类是相对朴素的音频模型路线，如 `EfficientNetV2-S + LSE pooling + EMA`。
- 另一类是明显偏 `Perch v2 + ProtoSSM + metadata prior + 多层后处理` 的高公榜堆叠路线。
- 当前判断真正值得借鉴的点主要是：更强 teacher、更好利用 1 分钟时序信息、以及轻量 TTA / smoothing。
- 当前判断不应盲抄的点主要是：`site/hour/month prior` 的强融合、`per-class threshold`、`rank-aware scaling`、`per-taxon temperature scaling` 这类容易公榜拟合的后处理。
- 当前对 `Perch v2` 公开方案的理解已经明确：
- 这些 notebook 主流做法是把 `Perch v2` 作为冻结的预训练特征提取器使用，而不是先 finetune `Perch` 再拿微调权重使用。
- `Perch v2` 原始 `label` 输出维度是 `14795`，但后续真正参与 `OOF / ProtoSSM / probe / submission` 的分数，会先映射到比赛的 `234` 类空间。
- 它们主流程里主要对 `train_soundscapes` 做编码，而不是对 `train_audio` 做同类缓存；这是因为后续建模都是围绕 `1min -> 12 x 5s` soundscape 序列展开的。
- 当前本地 `Perch` 运行环境已经打通。
- 已确认当前目录下的 [Perch](Perch) 模型目录结构完整，包含 `saved_model.pb`、`variables/`、`assets/labels.csv` 等必要文件。
- 已在 `conda env perch` 中成功加载本地 `Perch` 模型，确认输入签名为 `(None, 160000)` 的 `float32` 波形，对应 `32kHz` 单声道 `5s` 音频。
- 已确认本地 `Perch` 输出包括：
- `label: (None, 14795)`
- `embedding: (None, 1536)`
- `spatial_embedding: (None, 16, 4, 1536)`
- `spectrogram: (None, 500, 128)`
- 已新增 [birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py)，用于在本地基于当前目录结构生成等价 `perch-meta` 缓存。
- 该脚本默认读取 `input/train_soundscapes`、`input/train_soundscapes_labels.csv`、`input/taxonomy.csv`、`input/sample_submission.csv` 与本地 [Perch](Perch) 模型目录。
- 脚本默认只处理 `train_soundscapes_labels.csv` 中拥有完整 `12` 个 `5s` 窗口标注的 `full_files`，与公开 `Perch/ProtoSSM` notebook 的缓存口径保持一致。
- 当前本地数据统计为：带标签的 soundscape 文件共 `66` 个，其中 `full_files` 共 `59` 个。
- 已完成本地 `Perch` cache 脚本的 smoke test：
- `--limit-files 1` 时成功生成了 `meta (12, 4)`、`scores_full_raw (12, 234)`、`emb_full (12, 1536)`。
- 这说明当前本地已经具备完整的 `Perch v2 -> BirdCLEF 234 类映射 -> perch-meta 缓存生成` 能力。
- Perch 线上推理当前优先使用 `ONNX` 后端，不再优先依赖 `SavedModel`。
- 原因是 Kaggle 默认 `TensorFlow 2.19` 对当前 `Perch SavedModel` 中的 `vhlo.cosine_v2` 不兼容；手动升级到 `TensorFlow 2.20` 后可以跑通但速度偏慢。
- `perch_v2.onnx` 来自冻结的官方 `Perch v2` 转换权重，不包含我们自己的训练权重；只要 Perch 本体不 finetune，可以持续复用同一个 ONNX 文件。
- 即使用 ONNX 后端，线上仍需要提供 `PERCH_DIR/assets/labels.csv`，用于将 Perch 原始 `14795` 类输出映射到比赛 `234` 类。
- 当前 Perch deploy 的本地 CV 与 LB 对比口径已经统一：
- `outputs/perch_context_deploy_labeled_all_v1` 的 online-like Local CV 为 `0.867057`，Public LB 为 `0.894`；
- 旧训练 summary 中的 `0.790411` 是 mixed-scale OOF，不再用于 Perch 与 LB 的主对比；
- 后续 Perch 融合、选权重、CV-LB 对比均使用 online-like OOF 口径。
- 当前 CNN 主线仍是 `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k`，Local CV `0.776060`，Public LB `0.843`。
- 此前总体最高提交是 `Perch context LogReg + CNN` 的 logit 融合：
- 本地 coarse grid 最优为 `Perch 0.82 / CNN 0.18`，Local CV `0.883238`；
- fine grid 最优为 `Perch 0.824 / CNN 0.176`，Local CV `0.883276`；
- 线上已提交 `Perch 0.82 / CNN 0.18`，Public LB `0.903`；后续已被 `Perch + Stage3 CNN` 的 `0.916` 超过。
- `Perch-only` 后接模型对照当前结论：
- `LogReg` deploy v1：Local CV `0.867057`，Public LB `0.894`，是当前已验证线上 Perch-only 最强；
- `LGBM`：最佳 Local CV `0.864695`，低于 LogReg，不建议替换；
- `MLP`：最佳目录 `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027`，Local CV `0.878995`，Public LB `0.891`，低于 LogReg 的 `0.894`，暂不替代 Perch-only 主线。
- `Perch spatial_embedding + Mamba-style head` 已成为新的 Perch-only 主线：
- `outputs/perch_spatial_mamba_labeled_all_nopca_noraw_v1`：使用 `spatial_embedding [B,16,4,1536] -> mean over freq -> [B,16,1536]`，不做 PCA、不拼接 raw Perch score，Local CV `0.875406`，Public LB `0.899`，已超过 Perch context LogReg 的 `0.894`；
- `outputs/perch_spatial_mamba_labeled_all_meanmax_nopca_noraw_v1`：mean+max frequency pooling，Local CV `0.869538`，低于 mean-only，暂不作为主线；
- `outputs/perch_spatial_mamba_labeled_all_flat64_nopca_noraw_v1`：使用完整 `16 x 4 = 64` tokens，`kernel_size=9`，不做 PCA、不拼接 raw score，Local CV `0.876872`，是当前 spatial Mamba 本地最强版本，值得线上提交验证；
- 已新增 [birdclef2026_cache_perch_spatial_onnx.py](birdclef2026_cache_perch_spatial_onnx.py)、[birdclef2026_perch_spatial_mamba_train.py](birdclef2026_perch_spatial_mamba_train.py)、[birdclef2026_perch_kaggle_infer_spatial_mamba.py](birdclef2026_perch_kaggle_infer_spatial_mamba.py) 与对应 run 脚本，支持 `mean / meanmax / flat64` 三种 spatial token 口径；
- Kaggle spatial Mamba 推理脚本已通过 debug features smoke，`flat64` artifact 会正确使用 `spatial_tokens_64` / ONNX 原始 `spatial_embedding.reshape(B,64,1536)`，不会错误退回 mean tokens。
- 线上提交 `Perch + MLP` 时，最小上传/挂载内容为：
- 推理代码 [birdclef2026_perch_kaggle_infer_context_logreg.py](birdclef2026_perch_kaggle_infer_context_logreg.py)、[birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py)、[run_birdclef2026_perch_context_kaggle_infer.sh](run_birdclef2026_perch_context_kaggle_infer.sh)；
- `perch_v2.onnx`；
- 含 `assets/labels.csv` 的 `PERCH_DIR`；
- `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027/perch_context_mlp_artifacts.joblib`。
- `ProtoSSM / BiGRU residual` 当前已按防泄露口径修过：
- 严格按 `filename` 分 outer fold；
- 禁止 train/valid filename overlap；
- early stopping 改为只在 outer train 内部再切 inner validation，不再偷看 outer valid；
- 但当前严格结果主要基于 `full_files` 的 `708` 行子集，不是主 Perch deploy 的 `739` 行 labeled-all 口径，因此暂不直接作为线上提交依据。
- 在 `708` 行 full-file subset 上，严格 BiGRU residual 的最佳 tiny blend 可到 `0.875607`，说明时序残差有潜力；下一步若继续做，应先扩展到 `739` 行 labeled-all 的 partial-file padding/mask 口径，再与 MLP `0.878995` 对齐比较。

## 10. 实验日志

### 2026-04-26

- 新建了训练脚本 [birdclef2026_gm_train.py](birdclef2026_gm_train.py)，采用两阶段训练方案。
- 新建了推理脚本 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py)，适配 Kaggle 线上目录结构与官方提交格式。
- 新建了文档 [代码思路.md](代码思路.md)，记录训练思路、分折策略、防泄露逻辑、特征构造和 CV 口径。
- 解决了 `sklearn` 与 `torchaudio` 依赖问题，改成纯 `numpy` AUC 和无 `torchaudio` 特征管线。
- 为训练脚本加入日志落盘与验证显存优化。
- 将训练脚本默认 backbone 从 `tf_efficientnetv2_s.in21k_ft_in1k` 切换为 `convnext_atto.d2_in1k`，准备开始轻量模型对比实验。

### 2026-04-27

- 整理了 [MEMORY.md](MEMORY.md) 的结构，使其更适合作为长期实验记录与项目记忆文件。
- 新建了 [CV-LB.md](CV-LB.md)，用于集中记录 Local CV 与 Public LB 的对应关系。
- 补记了 `tf_efficientnetv2_s.in21k_ft_in1k` 的线上结果：Local CV `0.664928`，Public LB `0.782`。
- 补记了 `convnext_atto.d2_in1k` 的线上结果：Local CV `0.696086`，Public LB `0.841`。
- 补记了 `20260427_022128_convnext_atto.d2_in1k` 的线上结果：Local CV `0.711801`，Public LB `0.835`。
- 补记了 `20260427_160037_convnext_atto.d2_in1k` 的线上结果：Local CV `0.720155`，Public LB `0.848`。
- 补记了一个非常重要的 inference-only 提升：同一个 `20260427_160037_convnext_atto.d2_in1k`，使用增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 且只用单个 `model_root` 时，Public LB 提升到了 `0.867`。
- 观察到一个重要现象：这次 `convnext_atto.d2_in1k` 的本地 CV 高于上一版，但 Public LB 低于上一版，说明当前线下指标对线上排序有帮助，但仍存在 gap 与抖动。
- 最新一版 `20260427_160037_convnext_atto.d2_in1k` 同时刷新了当前已知的 Local CV 与 Public LB，说明在当前阶段线下提升仍然具有较强参考价值，但单次波动仍需警惕。
- 当时的最高提交口径为 `20260427_160037_convnext_atto.d2_in1k + birdclef2026_gm_kaggle_infer_ensemble.py`，公榜 `0.867`；这一口径后来先被 `20260428_164427_convnextv2_atto.fcmae_ft_in1k + birdclef2026_gm_kaggle_infer_ensemble.py = 0.871` 超过，随后又被 `20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo + birdclef2026_gm_kaggle_infer_ensemble.py = 0.873` 继续刷新。
- 为训练脚本补充了四个学习率命令行参数，方便通过 `.sh` 文件快速切换 Stage 1 和 Stage 2 的 backbone/head learning rate。
- 将默认早停策略调整为连续 `5` 轮验证集指标不提升再停止训练。
- 新建了增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py)，并保留 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 作为稳定版。
- 增强版推理脚本已支持多实验目录 ensemble、row-level TTA、时间平滑与 soundscape 级后处理，并完成了本地 debug 跑通验证。
- 基于 BirdCLEF 2025 top 方案经验，决定将 `tf_efficientnet_b3.ns_jft_in1k` 作为下一批重点尝试 backbone 之一。
- 新建了 [birdclef2026_gm_make_pseudo_labels.py](birdclef2026_gm_make_pseudo_labels.py)，用于给未标注 `train_soundscapes` 生成 5 秒级 soft pseudo labels。
- 新建了 [birdclef2026_gm_train_stage3_pseudo.py](birdclef2026_gm_train_stage3_pseudo.py)，用于加载 Stage 2 fold checkpoint 并在真实标签 + 伪标签的混合数据上继续训练。
- 新建了 [run_birdclef2026_gm_pseudo_stage3.sh](run_birdclef2026_gm_pseudo_stage3.sh)，方便后续通过修改顶部配置快速运行 `pseudo / stage3 / both`。
- 第一版 pseudo 方案默认采用 `fold-specific` 生成逻辑，以降低 Stage 3 本地 CV 被 teacher 泄露污染的风险。
- 使用 `20260427_160037_convnext_atto.d2_in1k` 做 teacher，已经完成了一次小规模 pseudo 生成 smoke test，产物目录为 `outputs/pseudo_labels/20260427_225558_smoke_pseudo_test`。
- 使用上述 smoke pseudo，已经完成了一次 `fold 0` 的 Stage 3 烟雾测试，产物目录为 `outputs/birdclef2026_gm_stage3_pseudo/20260427_225809_convnext_atto.d2_in1k_stage3_pseudo`。
- 该 Stage 3 smoke test 主要用于验证脚本链路打通，不代表正式可比较实验结果。

### 2026-04-28

- 补记了 `outputs/birdclef2026_gm/20260427_232617_eca_nfnet_l0.ra2_in1k` 的线上结果：Local CV `0.758286`，使用稳定版 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py) 推理时 Public LB `0.838`。
- 这个结果进一步提醒我们：当前本地 CV 虽然有参考价值，但不能单独作为 teacher 选择标准；像 `eca_nfnet_l0.ra2_in1k` 这样的高 CV 模型，仍然可能在线上不如 `convnext_atto.d2_in1k`。
- 开始尝试 ViT / DINOv3 系 backbone 时，发现原始 `fp16` 混合精度在训练阶段会出现 `nan`。
- 已对 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 做了数值稳定性增强：
- 新增 `--amp-mode`、`--disable-amp`、`--grad-clip-norm` 参数；
- `auto` 模式下对 ViT 类模型优先使用 `bf16`，若 GPU 不支持则直接回退 `fp32`；
- 在训练循环里对非有限 `loss/grad` 自动跳过并回退 AMP scale；
- 仅在真正执行 `optimizer.step()` 后再执行 `scheduler.step()`。
- 随后又发现 ViT 在 `bf16` 验证阶段会报错：`TypeError: Got unsupported ScalarType BFloat16`。
- 该问题已经修复：在验证和 Stage 3 评估阶段，将 `logits` 先转为 `float32` 再做 `sigmoid` 和 `numpy` 转换。
- 补记了 `convnextv2_atto.fcmae_ft_in1k` 的一组 Stage 2 / Stage 3 对照结果：
- Stage 2 实验目录 `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`，Local CV `0.729685`，Public LB `0.851`。
- 同一个 Stage 2 实验目录在增强版推理脚本 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 下，Public LB 可进一步提升到 `0.871`。
- Stage 3 实验目录 `outputs/birdclef2026_gm_stage3_pseudo/20260428_220148_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`，Local CV `0.719661`，Public LB `0.854`。
- 这次结果再次说明：pseudo label 可能会让当前这套本地 OOF 口径下降，但仍然有机会改善线上分布下的泛化表现，因此 pseudo 路线不能仅凭本地 CV 一票否决。
- 同时也进一步明确了当前三种成绩口径：
- 标准 Stage 2 单模型最佳为 `0.851`
- Stage 3 pseudo 最佳为 `0.873`
- 增强版 inference-only 口径最高为 `0.873`

### 2026-04-29

- 将 [run_birdclef2026_gm_pseudo.sh](run_birdclef2026_gm_pseudo.sh) 默认 teacher 从旧的 `convnext_atto.d2_in1k` 切换为 `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`。
- 同时把 pseudo 输出命名规范调整为 `conservative_fold_specific_convnextv2_atto_fcmae`，方便与旧的 `convnext_atto` pseudo 结果区分。
- 将 [run_birdclef2026_gm_stage3.sh](run_birdclef2026_gm_stage3.sh) 默认 student 保持为同一个 `convnextv2_atto.fcmae_ft_in1k` run，并加入按 `PSEUDO_SUFFIX` 自动定位最新 pseudo 目录的逻辑，减少每次手动改 `PSEUDO_ROOT` 的麻烦。
- 这一轮 shell 模板的目标很明确：让 pseudo / stage3 更像“少量高质量伪标签的保守微调”，而不是被海量噪声伪标签主导。
- 基于上述保守版 pseudo 模板产出的 `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`，本地 Stage 3 OOF CV 提升到 `0.723835`，并在使用 [birdclef2026_gm_kaggle_infer_stage3.py](birdclef2026_gm_kaggle_infer_stage3.py) 线上推理时拿到 Public LB `0.855`。
- 这说明新的 `convnextv2_atto.fcmae` teacher + 保守版 pseudo 策略，相比上一版 Stage 3 已经出现了稳定的小幅正收益。
- 随后又验证了同一个 Stage 3 run 在 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 下可以进一步把 Public LB 提升到 `0.873`，并刷新当前全表最高分。
- 这也是一个很强的正反馈：说明我们前面为增强版推理脚本补上的 Stage 3 兼容支持，在线上确实转化成了真实收益。
- 随后还尝试了把 `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo` 与 `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k` 一起用增强版推理脚本做线上 ensemble。
- 结果是 Kaggle 提交会超时。
- 这个结果很重要：说明当前线上时间预算下，“双模型 ensemble” 至少对这组强模型来说已经不够安全，因此下一步更适合优先走“把复杂度留在线下、线上仍单模型”的方向，例如多 teacher pseudo 或迭代 pseudo。
- 进一步系统查看了 [现阶段线上code区开源高分代码](现阶段线上code区开源高分代码) 中的多份公开 notebook，并将结论整理进了 [TricksForOurs.md](TricksForOurs.md)。
- 当前最重要的判断是：这批高分公开代码真正值得借鉴的不是复杂的 metadata / threshold / calibration 公榜后处理，而是更强的 teacher、更好地利用 `1min` 时序信息，以及更温和的 TTA / smoothing。
- 同时也已经把 `Perch v2` 与 `ProtoSSM` 的主流用法梳理清楚：
- `Perch v2` 通常被当作冻结的预训练特征提取器，而不是先 finetune 再用；
- `Perch` 原始输出 `14795` 维标签分数，但下游会映射到比赛 `234` 类；
- `ProtoSSM` 的核心作用是对 `1min` 内 `12 x 5s` 的序列做时序建模，而不是把每个 `5s` 完全独立看待。
- 为了本地复现这条路线，额外创建了 `conda env perch`，并在其中安装了 `TensorFlow 2.20`。
- 已确认当前目录下的 [Perch](Perch) 模型目录完整，且可以在本地成功 `tf.saved_model.load(...)`。
- 已确认本地 `Perch` 模型的输入为 `(None, 160000)` 的 `32kHz / 5s / mono` 波形，输出包括 `label(14795)`、`embedding(1536)`、`spatial_embedding(16,4,1536)` 与 `spectrogram(500,128)`。
- 新建了 [birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py)，用于在当前本地目录下直接运行 `Perch v2`，并生成等价的 `full_perch_meta` 与 `full_perch_arrays.npz`。
- 该脚本默认基于 `train_soundscapes_labels.csv` 自动筛选出 `full_files`，并将 `Perch` 原始输出映射到比赛 `234` 类，同时保留公开 notebook 中用于未直接映射类别的 genus proxy 逻辑。
- 已完成脚本 smoke test：
- 在 `--limit-files 1`、`--meta-path /tmp/perch_meta_smoke.csv`、`--arrays-path /tmp/perch_arrays_smoke.npz` 下成功跑通；
- 输出 shape 分别为 `meta (12, 4)`、`scores_full_raw (12, 234)`、`emb_full (12, 1536)`。
- 当前本地统计结果也已确认：
- `train_soundscapes_labels.csv` 去重后共有 `66` 个带标签 soundscape 文件；
- 其中满足完整 `12` 个 `5s` 标注窗口的 `full_files` 有 `59` 个。
- 此外还确认了一点：
- 公开 `Perch/ProtoSSM` notebook 的主流程里主要对 `train_soundscapes` 做 `Perch` 编码与缓存，并没有把 `train_audio` 也做成同类 `perch-meta` 主缓存。

### 2026-05-01

- 明确决定放弃 `site prior / site stacker` 这条线，原因是：
- 直接 `site prior` 线上 Public LB 掉到 `0.727`；
- `birdclef2026_gm_site_stacker.py` 的 stacked OOF 也明显劣于 base OOF；
- 因此当前重新回到 `stage3 pseudo + enhanced inference` 主线，并将 `Perch + ProtoSSM` 作为并行探索支线。
- 新建了 [birdclef2026_perch_protossm_cv.py](birdclef2026_perch_protossm_cv.py)。
- 该脚本会把 `perch_cache/full_perch_meta + full_perch_arrays.npz` 重组成按 `filename` 的 `12 x 5s` 序列；
- 继续使用 `row_id` 对齐标签，按 `filename` 分 fold；
- 以 `raw Perch scores` 为 baseline logit，再训练一个轻量的时序残差模型去做修正；
- 当前支持两种最小时序模型口径：
- `protossm_lite`：双向 selective state mixer + prototype residual head；
- `bigru`：双向 GRU + prototype residual head。
- 已完成脚本 smoke test：
- 在 `conda run -n transformers` 环境下使用 `--limit-files 9 --n-folds 3 --epochs 1 --d-model 64 --num-layers 1 --device cpu` 成功跑通；
- 输出目录为 `/tmp/perch_protossm_smoke`；
- 说明当前本地已经具备 `Perch cache -> sequence rebuild -> temporal model CV -> OOF 保存` 的完整能力。
- 随后对 `Perch + 时序模型` 做了两轮正式对比：
- `ProtoSSM-lite v2`：
  `outputs/perch_protossm_cv_v2/summary.json`
  `protossm_oof_auc = 0.699447`，明显低于 `raw_perch_auc = 0.739018`；
- `BiGRU v2`：
  `outputs/perch_bigru_cv_v2/summary.json`
  `protossm_oof_auc = 0.750058`，略高于 raw Perch，但仍低于先前更强的 `Perch logreg/context` 基线；
- 基于这个结果，把脚本进一步改成了 `v3`：
  默认先构造 `context_logreg` first-pass base scores，再让时序模型只学习 residual 修正；
- `v3` 的第一版曾出现 `OOF` 回填 bug：
  因为子集序列重组后误用了局部行号，导致 `temporal_oof_auc` 被错误写低；
- 该 bug 已修复，当前 `group_rows_into_sequences(...)` 已支持显式传入 `source_row_indices` 来保持全局行号一致。
- 修复后重新跑 `BiGRU residual v3`：
  `outputs/perch_bigru_residual_v3_fix/summary.json`
  结果为：
  `raw_perch_auc = 0.739018`
  `base_oof_auc = 0.778155`
  `temporal_oof_auc = 0.748514`
- 结论也因此更明确：
  当前最强的 `Perch` 支线依然是便宜且稳定的 `context_logreg` first-pass；
  即便修复 bug 后，`BiGRU residual` 也仍然没有超过 `base_oof_auc = 0.778155`；
  因此 `Perch + temporal NN` 暂时不值得进入主线，优先级低于我们现有的 `stage3 pseudo + enhanced inference` 主路线。
- 在确认暂停 `ProtoSSM / GRU residual` 深挖后，当前下一步重新回到主线训练增强，优先测试更传统且更可能稳定泛化的 `mixup / cutmix`。
- 当前本地仓库已经完成第一次 git 基线快照提交：
  分支 `main`
  commit `9e2b1d3`
  message: `chore: snapshot baseline before mixup experiments`
- 这意味着后续所有 `mixup / cutmix` 相关实验都可以明确地相对于该基线回退或对比。
- 已完成三组主线增强对比，全部基于 `convnextv2_atto.fcmae_ft_in1k` 的 stage2 local CV：
- baseline：
  `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`
  `Final OOF local CV = 0.729685`
- only mixup：
  `outputs/birdclef2026_gm/20260501_013831_convnextv2_atto.fcmae_ft_in1k`
  `mixup_alpha=0.4, mixup_prob=0.5`
  `Final OOF local CV = 0.698432`
- only cutmix（较强）：
  `outputs/birdclef2026_gm/20260501_021931_convnextv2_atto.fcmae_ft_in1k`
  `cutmix_alpha=1.0, cutmix_prob=0.25`
  `Final OOF local CV = 0.723514`
- mixup + cutmix 轻量 hybrid：
  `outputs/birdclef2026_gm/20260501_132623_convnextv2_atto.fcmae_ft_in1k`
  `mixup_alpha=0.2, mixup_prob=0.2, cutmix_alpha=0.5, cutmix_prob=0.15`
  `Final OOF local CV = 0.718183`
- only cutmix（更轻）：
  `outputs/birdclef2026_gm/20260501_155038_convnextv2_atto.fcmae_ft_in1k`
  `cutmix_alpha=0.5, cutmix_prob=0.10`
  `Final OOF local CV = 0.726670`
- 当前结论：
- `mixup` 明显拖后腿，应暂时停掉；
- `cutmix` 往更轻方向调是有效的，但“全局同时打 stage1 + stage2” 仍未超过 baseline；
- 下一步更合理的验证方向是拆分阶段控制，优先测试 `stage1 only cutmix`，而非继续盲目调整统一 `cutmix_prob`。
- 随后完成了第一组 `stage1 / stage2` 分离增强实验：
  `outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k`
  `stage1_cutmix_alpha=0.5, stage1_cutmix_prob=0.10`
  `stage2_cutmix_alpha=0.0, stage2_cutmix_prob=0.0`
  `Final OOF local CV = 0.734594`
- 这次实验给出了一个很有价值的信号：
- 相比旧 Stage 2 baseline `0.729685`，本地 CV 提升了 `+0.004909`；
- 使用稳定版推理脚本时，Public LB 从旧的 `0.851` 提升到 `0.856`；
- 但使用增强版推理脚本时，Public LB 为 `0.869`，仍略低于旧 Stage 2 增强版基线 `0.871`。
- 因此当前更稳妥的判断是：
- `stage1 only cutmix` 已经是新的更强 Stage 2 “稳定版单模型”候选；
- 但它还不能直接替代旧的 `20260428_164427...` 作为最佳增强版推理底座；
- 下一步最值得尝试的是把新旧两个 Stage 2 run 一起喂给 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 做双 run ensemble，而不是立刻单方面淘汰旧主线。
- 随后线上又补做了两组关键对照：
- 新 run 单模型 + 增强版推理 + no TTA：
  `0.861`
- 旧 run + 新 run 双 Stage 2 ensemble + 增强版推理 + no TTA：
  `0.868`
- 这两个结果合在一起说明：
- 线上 `TTA` 仍然是一个真实有效的增益来源，至少对当前单模型提交仍然如此；
- 双 run ensemble 的多样性也确实有效，因为它把 no-TTA 成绩从 `0.861` 推到了 `0.868`；
- 但在 no-TTA 约束下，双 run ensemble 仍未超过新 run 单模型 + TTA 的 `0.869`，也未超过旧 run 单模型 + 增强版推理的 `0.871`。
- 在通过 git 回退到 `c2d8b29` 并新建恢复分支 `recover-07346-mainline` 后，当前主线再次回到“`stage1 only cutmix` + 旧版 plain log-mel”口径。
- 在补齐所有确定性 seed 控制后，新的已验证强基线为：
  `outputs/birdclef2026_gm/20260505_145600_convnextv2_atto.fcmae_ft_in1k`
  `Final OOF local CV = 0.749210`
- 这是当前这条恢复主线上的最高已验证本地 CV，也明显高于此前的 `0.734594` 基线。
- 当前这一版代码已经在本地 git 完成快照：
  分支 `recover-07346-mainline`
  commit `fba7d4f`
  message: `fix: snapshot deterministic training baseline`

### 2026-05-05

- 在发现“相同代码重复跑两次，最终 local CV 仍然明显不同”之后，系统检查了训练链路中的所有随机源。
- 确认旧版脚本虽然有 `seed_everything(cfg.seed)`，但仍遗漏了几个关键入口：
- `cudnn.benchmark=True` 会导致卷积算法选择随输入形状和环境波动；
- `DataLoader worker` 没有单独 seed；
- `WeightedRandomSampler` 没有显式 generator；
- 各 train/valid loader 也没有显式的 torch generator。
- 上述入口现已全部补齐并固化到主线代码中。
- 随后在恢复分支 `recover-07346-mainline` 上重新运行 [run_birdclef2026_gm_train.sh](run_birdclef2026_gm_train.sh)，得到：
- `outputs/birdclef2026_gm/20260505_145600_convnextv2_atto.fcmae_ft_in1k`
- `fold_scores = [0.881265, 0.906407, 0.864333]`
- `Final OOF local CV = 0.749210`
- 这一结果说明两件事：
- 先前的“复现不了 0.734594”并不一定只是模型结构问题，随机性失控本身就是一个真实干扰项；
- 在当前恢复主线上，把确定性控制补齐后，`stage1 only cutmix` 这条线的上限明显比我们此前以为的更高。
- 随后又把三份推理脚本的确定性控制补齐，并修复了稳定版推理脚本的切窗语义，使线上推理与 Stage 2 训练 / 评估口径重新对齐。
- 在这个确定性基线上，把分类头扩展为可切换的 `linear / csiro_conv_v1` 两种口径，并同步让三份推理脚本支持从 `config.json` 自动恢复对应头部结构。
- `linear` 头的复现实验 `outputs/birdclef2026_gm/20260505_182450_convnextv2_atto.fcmae_ft_in1k` 与 `145600` 的 local CV 完全一致，这证明当前恢复分支上的复现口径已经是稳定的。
- 首个 `csiro_conv_v1 + stage1 only cutmix` 实验为：
- `outputs/birdclef2026_gm/20260505_182506_convnextv2_atto.fcmae_ft_in1k`
- `Final OOF local CV = 0.761989`
- 使用稳定版推理脚本线上推理时，Public LB `0.840`。
- 这个结果的启发是：`csiro_head` 在小规模本地 CV 上明显更强，但单模公榜暂时没有同步受益，因此它更像“保留做多样性 / teacher 候选”的结构，而非立刻替换全部单模提交流程。
- 随后为训练脚本加入了 `mixup_domain={image,waveform}`，开始系统测试 waveform-level mixup。
- 第一组对照结果如下：
- 无增强：
  `outputs/birdclef2026_gm/20260505_195611_convnextv2_atto.fcmae_ft_in1k`
  `Final OOF local CV = 0.709124`
- `stage1 waveform mixup only`：
  `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k`
  `alpha=0.2, prob=0.1`
  `Final OOF local CV = 0.776060`
- `stage1 cutmix only`：
  `outputs/birdclef2026_gm/20260505_205630_convnextv2_atto.fcmae_ft_in1k`
  `Final OOF local CV = 0.761989`
- 其中 `205630` 与 `182506` 的 local CV 完全一致，说明在 `mixup_prob=0` 的前提下，`MIXUP_DOMAIN=waveform` 不会污染 pure cutmix-only 结果。
- 之后又补做了 `waveform mixup + cutmix` 的联合增强：
  `outputs/birdclef2026_gm/20260505_220045_convnextv2_atto.fcmae_ft_in1k`
  `Final OOF local CV = 0.759318`
- 该结果低于 `waveform mixup only` 的 `0.776060`，也略低于 pure cutmix-only 的 `0.761989`，因此当前不建议把两者叠加作为主线。
- 同一天还进一步确认了当前 local CV 的样本口径：
- Stage 2 本地评估使用 `66` 条 1 分钟 soundscape 文件，共 `739` 个有标签 5 秒窗口；
- 不是理论上的 `66 x 12 = 792` 全窗；
- 缺失的 `53` 个窗口来自 `7` 个只提供部分标注窗口的 soundscape 文件；
- 这也是当前本地 CV 与线上 Public LB 无法完全同步升降的一个重要原因。

### 2026-05-06

- 在确认 `stage1 waveform mixup only (alpha=0.2, prob=0.1)` 是当前最强点之后，对其参数做了第一轮小范围 sweep。
- 结果如下：
- `outputs/birdclef2026_gm/20260505_234016_convnextv2_atto.fcmae_ft_in1k`
  `alpha=0.1, prob=0.1`
  `Final OOF local CV = 0.725502`
- `outputs/birdclef2026_gm/20260505_234055_convnextv2_atto.fcmae_ft_in1k`
  `alpha=0.3, prob=0.1`
  `Final OOF local CV = 0.742956`
- `outputs/birdclef2026_gm/20260506_003759_convnextv2_atto.fcmae_ft_in1k`
  `alpha=0.2, prob=0.2`
  `Final OOF local CV = 0.728587`
- 这三组 sweep 全部输给 `195634` 的 `0.776060`，因此当前最优点仍明确停留在：
- `stage1 waveform mixup`
- `alpha=0.2`
- `prob=0.1`
- 轻量 `stage2 waveform mixup = 0.1 / 0.05` 随后完成验证：
- `outputs/birdclef2026_gm/20260506_022923_convnextv2_atto.fcmae_ft_in1k`
- `Final OOF local CV = 0.740252`
- 这条结果可复现地低于 `stage1 waveform mixup only` 的 `0.776060`，因此当前结论是：stage2 mixup 对这条主线是干扰，不再继续压。
- 同时补做了 `tf_efficientnetv2_b0.in1k + csiro_conv_v1 + stage1 waveform mixup only`：
- `outputs/birdclef2026_gm/20260506_014724_tf_efficientnetv2_b0.in1k`
- `Final OOF local CV = 0.764883`
- 该结果没有超过 convnextv2 当前主线，但分数不差，后续若做 ensemble 多样性可以保留为候选。
- 完成了 Perch context LogReg deploy v1 的线上验证：
- artifact 目录为 `outputs/perch_context_deploy_labeled_all_v1`；
- online-like Local CV 为 `0.867057`；
- Public LB 为 `0.894`；
- 这是当前已验证的 Perch-only 最高线上分数。
- 明确统一了 Perch 本地 CV 口径：
- 旧的 `0.790411` 来自 mixed-scale OOF，不再用于 LB 对比；
- 后续记录、融合选权和 CV-LB 分析都使用 online-like OOF `0.867057`。
- 完成了 `Perch context LogReg + CNN` 本地融合实验：
- 融合实验目录为 `outputs/blend_cnn195634_perch_context_deploy_v1`；
- CNN 来自 `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k`，Local CV `0.776060`；
- Perch 来自 `outputs/perch_context_deploy_labeled_all_v1`，Local CV `0.867057`；
- coarse grid 最优为 `logit_mean, Perch 0.82 / CNN 0.18`，Local CV `0.883238`；
- fine grid 最优为 `logit_mean, Perch 0.824 / CNN 0.176`，Local CV `0.883276`。
- 已线上提交 `Perch 0.82 / CNN 0.18` 的 logit 融合，Public LB 为 `0.903`，刷新当时总体最高分；后续已被 `Perch + Stage3 CNN` 的 `0.916` 超过。
- 完成了 Perch 后接 head 的第一轮替换实验：
- `LGBM` 最佳 Local CV 为 `0.864695`，低于 LogReg `0.867057`，暂不作为线上候选；
- `MLP` 最佳目录为 `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027`，Local CV 为 `0.878995`，高于 LogReg；后续线上验证为 Public LB `0.891`，低于 LogReg `0.894`。
- `Perch + MLP` 线上 artifact 最小只需要 `perch_context_mlp_artifacts.joblib`，不需要上传该目录下的 `oof_predictions.npz`、`fold_metrics.csv` 或 `summary.json`。
- 对 `ProtoSSM / BiGRU residual` 做了防泄露修正和复查：
- 当前严格版本已经避免 train/valid filename overlap；
- early stopping 不再使用 outer valid；
- 但结果仍主要基于 `708` 行 full-file subset，不是 `739` 行 labeled-all 主口径；
- 因此目前只把 `BiGRU residual full708 best blend = 0.875607` 当作潜力信号，不直接作为线上主线。

### 2026-05-07

- 更新 [CV-LB.md](CV-LB.md)，正式记录 `Perch 0.82 / CNN 0.18` logit 融合的线上 Public LB `0.903`。
- 补记 `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027` 的线上 Public LB `0.891`。
- 这条结果说明：MLP 的 Local CV `0.878995` 虽然高于 LogReg 的 `0.867057`，但线上低于 LogReg 的 `0.894`；因此当前 Perch-only 主线仍应保留 LogReg，MLP 不作为替代提交基线。
- 当前 `CV-LB` 结论同步调整为：
- 标准 Stage 2 单模型提交最佳：`0.851`
- Stage 3 pseudo 提交最佳：`0.873`
- 增强版 CNN inference-only 口径最高分：`0.873`
- Perch context LogReg 口径最高分：`0.894`
- Perch context MLP 对照分：`0.891`
- Perch + CNN logit 融合此前最高分：`0.903`
- Perch + Stage3 CNN logit 融合总体最高分：`0.916`
- 已完成 `Perch + MLP` 线上提交；若后续复现该提交，需要以下内容：
- 代码文件需要 [birdclef2026_perch_kaggle_infer_context_logreg.py](birdclef2026_perch_kaggle_infer_context_logreg.py)、[birdclef2026_run_perch_local.py](birdclef2026_run_perch_local.py)、[run_birdclef2026_perch_context_kaggle_infer.sh](run_birdclef2026_perch_context_kaggle_infer.sh)；
- Perch 后端优先使用 ONNX，需要 `perch_v2.onnx`；
- 仍需要 `PERCH_DIR/assets/labels.csv` 做 label mapping；
- MLP artifact 使用 `outputs/perch_context_mlp_labeled_all_v1_base_p64_seed2027/perch_context_mlp_artifacts.joblib`。
- 完成第一版 CNN long-context 实验：
- 目录为 `outputs/birdclef2026_gm/20260507_012705_convnextv2_atto.fcmae_ft_in1k`；
- 结构为 `15s input -> 3 x 5s slot logits + global logits`，head 为 `csiro_multicontext_v1`；
- Stage 2 fold soundscape AUC 分别为 `0.772407 / 0.852411 / 0.827600`，但最终全局 OOF local CV 只有 `0.622288`；
- 已从保存的 `soundscape_oof_predictions.csv` 和各 fold `valid_predictions.csv` 复算确认，`0.622288` 不是 CSV 保存或列对齐 bug；
- 根因是当前 739-row CV 中有大量极稀有类只在单个 fold 有正样本：75 个可评分类里，30 个类只覆盖 1 个 fold，这组类全局 OOF 平均 AUC 只有 `0.310389`；
- 这说明逐 fold AUC 在这种稀有类分布下会偏乐观，后续判断 CNN long-context 必须以最终全局 OOF CV 为准；
- 与主线 CNN `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k` 做 OOF 融合检查后，long-context 任意正权重都会拉低主线 CV，例如概率均值 `weight_long=0.01` 已从 `0.776060` 降到 `0.732376`；
- 因此第一版 long-context 先判负，不提交，不纳入当前融合候选。若后续重启这条线，应优先解决稀有类折间覆盖 / 采样 / head 校准问题，而不是直接加长上下文。
- 完成“白名单”Perch + CNN 后处理网格，只使用训练 OOF 和训练标签，不使用 hidden test 或 LB 反馈：
- 脚本为 `birdclef2026_whitelist_blend_grid.py`；
- 输出目录为 `outputs/whitelist_blend_cnn195634_perch_logreg_v1`；
- 泄露控制：Perch OOF 使用 validation-fold LogReg 输出，未 fitted classes 用 `sigmoid(raw Perch logits)` fallback；CNN 使用 `20260505_195634` 的 OOF；参数只由 739-row OOF 全局 AUC 选择；
- baseline 为 `Perch 0.824 / CNN 0.176` logit blend，Local CV `0.883276`；
- 最佳白名单配置为 `logit blend, Perch 0.83 / CNN 0.17 -> file top-2 mean scaling -> adaptive temporal smoothing alpha=0.10`；
- 该配置 Local CV 为 `0.889634`，相对 baseline `+0.006358`；
- class 维度上 `46` 类提升、`20` 类下降，median class delta 为 `+0.001459`，不是单一类别造成的假提升；
- 线上复现脚本为 `birdclef2026_blend_submissions_postprocess.py`，默认读取 `/kaggle/working/submission.csv` 作为 Perch、`/kaggle/working/submission_cnn.csv` 作为 CNN，并覆盖输出 `/kaggle/working/submission.csv`；
- 这条配置后续已被 `Perch + Stage3 CNN` 融合的 `0.916` 超过；如果继续扩大这类后处理，需要坚持只用本地 OOF 选参，避免借公榜噪声调参。
- 启动了 `Perch + CNN white-list blend teacher -> fold-specific pseudo -> Stage3` 链路，重点防止伪标签泄露：
- 发现不能直接用 `outputs/perch_context_deploy_labeled_all_v1` 的 5 折 Perch artifact 给 3 折 CNN 主线生成 fold pseudo，因为两套 fold 不对齐会让 CNN valid fold 可能通过 Perch teacher 间接看到标签；
- 因此新增 `--fold-assignment-path`，训练了与 CNN 主线 `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv` 完全对齐的 3 折 Perch LogReg artifact：
- `outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1`
- 该 aligned artifact 的 Local OOF 为 `0.787011`，低于线上 deploy artifact，但它是 Stage3 fold-specific pseudo 的安全 teacher；
- 新增/修复的脚本：
- `birdclef2026_make_pseudo_perch_cnn_blend.py`：支持 `perch-cache` 与 `pseudo-from-cache` 两阶段，解决本地 `perch` 环境有 TF/sklearn 但无 torch、`transformers` 环境有 torch 但无 TF 的依赖拆分问题；
- `run_birdclef2026_make_pseudo_perch_cnn_blend.sh`：生成 fold-specific Perch+CNN 伪标签；
- `run_birdclef2026_perch_context_aligned_train.sh`：训练 CNN-fold-aligned Perch LogReg artifact；
- `run_birdclef2026_perchcnn_pseudo_stage3_after_cache.sh`：等待 Perch cache 完成后自动生成伪标签并启动 Stage3；
- 同时修复 `birdclef2026_gm_kaggle_infer_ensemble.py`：补上 `head_type` 传递和 `CSIROHead` / `LocalSequenceBlock`，否则主线 `csiro_conv_v1` checkpoint 无法在 ensemble 推理库中加载；
- 已通过 smoke：
- Perch cache 2 files -> `24 x 234` scores + `24 x 1536` embeddings + fold0 pred；
- CNN+blend pseudo 2 files fold0 -> 保留 `3 / 24` 行高置信伪标签；
- `Perch + CNN white-list blend teacher -> fold-specific pseudo -> Stage3` 首轮全量已经跑完：
- Perch cache 已切到本地 `PerchV2Onnx/perch_v2.onnx` + `onnxruntime`，输出目录为 `outputs/pseudo_labels/perch_cnn_blend_white_v1_perch_cache`，全量 `10592` 个 soundscape 文件耗时约 `7073.6s`，明显快于 SavedModel CPU/XLA；
- fold-specific pseudo 输出目录为 `outputs/pseudo_labels/20260507_165105_perch_cnn_blend_white_v1`；
- 伪标签过滤参数为 `perch_weight=0.83`、`file_scale_mode=topk_mean`、`file_scale_value=2.0`、`smooth_mode=adaptive`、`smooth_alpha=0.10`、`prob_threshold=0.35`、`row_min_max_prob=0.85`、`top_k_labels=2`；
- 三个 fold 的伪标签保留率分别为 `121536 / 127104 = 95.62%`、`114986 / 127104 = 90.47%`、`122951 / 127104 = 96.73%`，整体明显偏激进；
- Stage3 输出目录为 `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；
- Stage3 配置为 `stage3_epochs=6`、`pseudo_loss_weight=0.35`、`pseudo_sampler_weight=0.5`、`min_pseudo_max_prob=0.85`、无 mixup/cutmix；
- 最终 Stage3 OOF local CV 为 `0.811590`，fold AUC 分别为 `0.860860 / 0.875203 / 0.842884`；
- 已提交 Stage3-only，Public LB 为 `0.839`；
- 这说明该模型虽然刷新了 Stage3-only 本地 OOF，但 standalone 线上泛化弱于当前主线 CNN `20260505_195634` 的 `0.843`，也弱于旧 Stage3 增强推理 `0.873`，不能作为单独提交主线；
- 三个 fold 都是 epoch 1 最好，后续继续训练基本下降，说明这轮伪标签数量和权重都过强，模型快速贴合 teacher 后开始损伤验证泛化；
- 单看 Stage3 训练动态，这轮不适合替代主线 CNN；但它作为 `Perch` 的互补分支很强，值得提交一次真实 LB 验证；
- 用 `outputs/perch_context_deploy_labeled_all_v1` 的 strict online-like Perch OOF，与该 Stage3 的 `soundscape_oof_predictions.csv` 重新跑白名单融合，输出目录为 `outputs/whitelist_blend_stage3perchcnn_perch_logreg_v1`；
- `Perch + Stage3 CNN` baseline `logit, perch_weight=0.824` 的 Local CV 为 `0.896853`，已明显高于旧 `Perch + Stage2 CNN` 白名单最佳 `0.889634`；
- 当前白名单最优配置为 `logit blend, perch_weight=0.74, file_scale_mode=max_power, file_scale_value=0.4, smooth_mode=plain, smooth_alpha=0.15`，Local CV 为 `0.905907`；
- 这组参数只由本地 OOF 选择，未使用 hidden test 或 LB 反馈；线上 Public LB 为 `0.916`，刷新当前总体最高分；
- 这说明 `Stage3-only` 虽然只有 Public LB `0.839`，但作为 `Perch` 的互补分支非常有效。后续 pseudo teacher / 蒸馏 / 融合实验应围绕这条 `0.916` 口径展开，而不是围绕 Stage3 standalone。
- 下一轮若继续 pseudo，应收紧到更保守配置，例如 `row_min_max_prob >= 0.93`、`top_k_labels=1`、`pseudo_loss_weight=0.15~0.25`、更低 `pseudo_sampler_weight`，并优先减少 epoch 或只做更轻量 finetune。
- 已修复 [birdclef2026_make_pseudo_perch_cnn_blend.py](birdclef2026_make_pseudo_perch_cnn_blend.py) 的 `RunLogger`：以后 `make_pseudo.log` 会压缩 tqdm 的 carriage-return 刷新，只保存最终进度行和关键信息，避免长任务日志每个 step 占一行。
- 已进一步修复 Stage3 artifact 的线上自描述问题：
- [birdclef2026_gm_train_stage3_pseudo.py](birdclef2026_gm_train_stage3_pseudo.py) 后续会把 `model_name / sample_rate / clip_seconds / image_height / image_width / dropout / drop_path / head_type` 写入 Stage3 `config.json`；
- [birdclef2026_gm_kaggle_infer_stage3.py](birdclef2026_gm_kaggle_infer_stage3.py) 与 [birdclef2026_gm_kaggle_infer_ensemble.py](birdclef2026_gm_kaggle_infer_ensemble.py) 的 fallback 也会优先读取这些字段；
- 当前 Stage3 输出目录的 `config.json` 已本地补齐 `head_type=csiro_conv_v1` 等字段，便于单独上传模型目录时正确恢复结构。

### 2026-05-08

- 明确了当前 `Perch` 路线与 `SED` 的关系：
- 思想上接近 `SED`，因为它也是对 soundscape 做时间局部化、多标签、逐窗口检测；
- 但当前实现不是 frame-level onset/offset SED，而是 `60s -> 12 x 5s` 的 segment-level SED-like classifier；
- 更准确地说，当前 Perch 支线是 `pretrained audio embedding + 5s segment-level SED-like classifier + temporal smoothing / file-level scaling`。
- 将长任务运行规范进一步固化：
- 训练、长时间推理、长等待监控都应放到 `tmux` 中；
- 后续 tmux 启动默认使用 `2>&1 | tee outputs/xxx.log`，这样 `tmux attach` 能看到关键输出，同时日志也会落盘；
- 非交互日志中继续通过 `TQDM_DISABLE=1` 关闭 tqdm 刷屏，避免日志每个 step 一行。
- 围绕 `0.916` 的 `Perch + Stage3 CNN` 融合提交，系统尝试将它作为 teacher 生成新的 fold-safe pseudo，再训练 Stage3 student。
- 为这条线新增/固化了保守伪标签流水线：
- [birdclef2026_make_pseudo_perch_cnn_blend.py](birdclef2026_make_pseudo_perch_cnn_blend.py) 新增 `min_top1_top2_margin / max_topk_entropy / entropy_top_k` 等筛选项；
- 新增 [run_birdclef2026_perchstage3_pseudo_stage3_conservative.sh](run_birdclef2026_perchstage3_pseudo_stage3_conservative.sh)，默认使用 `0.916-style teacher`：aligned Perch fold_k + Stage3 CNN fold_k + 本地 OOF 选出的 `perch_weight=0.74, file_scale=max_power(0.4), smooth=plain(0.15)`；
- 关键防泄露边界为：`pseudo_scope=fold-specific`、`teacher_folds=[0,1,2]`、`include_labeled=false`、目标 soundscape 数为 `10592 = 10658 - 66 labeled`；
- Stage3 训练脚本仍默认禁止 global pseudo，除非显式传入 `--allow-global-pseudo`，本轮所有实验都没有传该开关。
- 第一轮 `0.916 teacher conservative v1` 判负：
- pseudo 目录为 `outputs/pseudo_labels/20260507_203341_perch_stage3_teacher_conservative_v1`；
- Stage3 输出目录为 `outputs/birdclef2026_gm_stage3_perchstage3_teacher_conservative_v1/20260507_211924_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；
- 主要参数：`row_min_max_prob=0.93`、`top_k_labels=1`、`min_top1_top2_margin=0.25`、`pseudo_loss_weight=0.20`、`pseudo_sampler_weight=0.20`；
- pseudo 保留行数为 `341 / 343 / 228`，只覆盖极少量高置信窗口；
- Stage3 final OOF local CV 为 `0.766586`，低于旧 Stage3 主线 `0.811590`；
- 泄露审计通过：`labeled_overlap=0`、`valid_overlap=0`、`teacher_fold` 严格匹配对应 fold、`OVERALL_SAFE_CHECK=True`。
- 第二轮 `0.916 teacher relaxed v1` 继续判负：
- pseudo 目录为 `outputs/pseudo_labels/20260507_213817_perch_stage3_teacher_relaxed_v1`；
- Stage3 输出目录为 `outputs/birdclef2026_gm_stage3_perchstage3_teacher_relaxed_v1/20260507_222351_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；
- 主要参数：`row_min_max_prob=0.90`、`top_k_labels=1`、`min_top1_top2_margin=0.20`、`pseudo_loss_weight=0.20`、`pseudo_sampler_weight=0.20`；
- pseudo 保留行数为 `1478 / 1538 / 914`，仍不足以提供稳定监督；
- Stage3 final OOF local CV 为 `0.764577`；
- 泄露审计同样通过，说明该负结果可信，不是泄露或 fold 对齐问题。
- 第三轮 `0.916 teacher soft v1` 回升但仍未超过旧主线：
- pseudo 目录为 `outputs/pseudo_labels/20260507_224130_perch_stage3_teacher_soft_v1`；
- Stage3 输出目录为 `outputs/birdclef2026_gm_stage3_perchstage3_teacher_soft_v1/20260507_232722_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；
- 主要参数：`row_min_max_prob=0.75`、`top_k_labels=2`、`min_top1_top2_margin=0.0`、`min_pseudo_max_prob=0.75`、`pseudo_loss_weight=0.08`、`pseudo_sampler_weight=0.08`；
- pseudo 保留行数为 `126223 / 126179 / 126532`，接近全量 soft pseudo；
- Stage3 final OOF local CV 为 `0.803577`，明显强于前两轮高置信 hard pseudo，但仍低于旧 `Perch + CNN white-list blend teacher -> Stage3` 的 `0.811590`；
- 泄露审计通过：几乎覆盖全部 unlabeled 文件，但没有混入 labeled/valid，`teacher_fold` 仍严格匹配。
- 第四轮复用 `soft_v1` pseudo，仅把 Stage3 权重提高到 `0.15 / 0.15`，结果没有实质提升：
- Stage3 输出目录为 `outputs/birdclef2026_gm_stage3_perchstage3_teacher_soft_w015_v1/20260507_234551_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`；
- 主要参数：复用 `outputs/pseudo_labels/20260507_224130_perch_stage3_teacher_soft_v1`，`pseudo_loss_weight=0.15`、`pseudo_sampler_weight=0.15`、`min_pseudo_max_prob=0.75`；
- Stage3 final OOF local CV 为 `0.803828`，仅比 `w008` 的 `0.803577` 高 `+0.00025`，基本可视为噪声；
- 三折均为 epoch 1 最好，后续 epoch 下降，说明 teacher 权重继续变强会更快拉偏 student。
- 综上，`0.916 teacher -> CNN Stage3 student` 蒸馏路线暂时判负：
- `0.916` 这条组合非常适合作为线上提交融合，但并不比旧的 `Perch + CNN white-list blend teacher` 更适合训练 CNN student；
- 后续不建议继续沿着 `0.916 teacher` 的 `row_min_max_prob / pseudo_loss_weight / pseudo_sampler_weight` 小范围 sweep 消耗时间；
- 当前最佳可提交路线仍是 `Perch context LogReg + Stage3 CNN` 的本地白名单后处理融合，Local CV `0.905907`，Public LB `0.916`；
- 当前最佳 Stage3 student 仍是 `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`，Stage3 local CV `0.811590`，但 standalone Public LB 只有 `0.839`，它的主要价值是作为 Perch 融合互补分支。

### 2026-05-09

- 完成 `Perch + Stage3 CNN + base CNN` 三路 OOF 白名单融合实验：
- 新增 [birdclef2026_whitelist_blend_grid_threeway.py](birdclef2026_whitelist_blend_grid_threeway.py)，只读取本地 OOF prediction 和 train labels，不读取 hidden test 或 LB 反馈；
- 输入为：
- `Perch`: `outputs/perch_context_deploy_labeled_all_v1` 的 strict online-like OOF；
- `Stage3 CNN`: `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo/soundscape_oof_predictions.csv`；
- `base CNN`: `outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_oof_predictions.csv`；
- 输出目录为 `outputs/whitelist_blend_threeway_perch_stage3_cnn195634_v1`；
- 二路基线仍是 `Perch + Stage3 CNN` 的 `logit, Perch=0.74, Stage3=0.26, file_scale=max_power(0.4), smooth=plain(0.15)`，Local CV `0.905906851`；
- 三路最优为 `logit, Perch=0.741275, Stage3=0.253725, base CNN=0.005, file_scale=topk_mean(2), smooth=plain(0.10)`；
- 三路最优 Local CV 为 `0.906157965`，相对二路提升 `+0.000251`；
- class 维度上 `41` 类提升、`25` 类下降，median class delta 为 `+0.000261`，整体是很小的正信号；
- 新增 [birdclef2026_blend_three_submissions_postprocess.py](birdclef2026_blend_three_submissions_postprocess.py)，用于线上把三份 submission 合成为最终提交，默认参数就是上述 OOF 最优三路配置；
- 结论：三路融合可以作为低风险小幅试探提交，但本地提升太小，不应把它视为新的强主线；如果线上 LB 没提升或波动，仍以 `Perch + Stage3 CNN` 二路 `0.916` 口径为当前最佳主线。

### 2026-05-10

- 在多轮 CNN long-context / SED-like 实验判负后，明确把后续突破重点切到 `Perch spatial_embedding`。
- 新增 Perch spatial 特征链路：
- [birdclef2026_cache_perch_spatial_onnx.py](birdclef2026_cache_perch_spatial_onnx.py)：使用 `PerchV2Onnx/perch_v2.onnx` 导出 frozen `spatial_embedding [B,16,4,1536]`，支持缓存 `mean tokens [B,16,1536]`、`meanmax` 的 max tokens、以及 `flat64 [B,64,1536]`；
- [birdclef2026_perch_spatial_mamba_train.py](birdclef2026_perch_spatial_mamba_train.py)：fold-safe 训练 `Perch spatial tokens -> Mamba-style head`，按 `filename` 做 GroupKFold，scaler/PCA/early stopping/class mask 均只在 outer train fold 内拟合；
- [birdclef2026_perch_kaggle_infer_spatial_mamba.py](birdclef2026_perch_kaggle_infer_spatial_mamba.py)：Kaggle 线上 spatial Mamba 推理脚本，直接从 ONNX 取 `spatial_embedding`，支持 `mean / meanmax / flat64` artifact；
- 对应 run 脚本为 [run_birdclef2026_cache_perch_spatial_onnx.sh](run_birdclef2026_cache_perch_spatial_onnx.sh)、[run_birdclef2026_perch_spatial_mamba_train.sh](run_birdclef2026_perch_spatial_mamba_train.sh)、[run_birdclef2026_perch_spatial_mamba_kaggle_infer.sh](run_birdclef2026_perch_spatial_mamba_kaggle_infer.sh)。
- 完成三组 Perch spatial 单模实验：
- `outputs/perch_spatial_mamba_labeled_all_nopca_noraw_v1`：mean-only，`[B,16,4,1536] -> mean -> [B,16,1536]`，不做 PCA、不拼接 raw Perch score，Local CV `0.875406`，Public LB `0.899`，超过 Perch context LogReg 的 `0.894`；
- `outputs/perch_spatial_mamba_labeled_all_meanmax_nopca_noraw_v1`：mean+max frequency pooling，Local CV `0.869538`，判负；
- `outputs/perch_spatial_mamba_labeled_all_flat64_nopca_noraw_v1`：`[B,16,4,1536] -> reshape -> [B,64,1536]`，`kernel_size=9`，不做 PCA、不拼接 raw score，Local CV `0.876872`，当前 spatial Mamba 本地最强，值得线上提交验证。
- 已更新 [CV-LB.md](CV-LB.md)：记录 mean-only spatial Mamba 的 Local CV `0.875406` 与 Public LB `0.899`。
- 已修复 spatial Mamba artifact 序列化问题：不再把自定义 `TokenProjector` 对象直接 joblib dump 为 `__main__` 类，而是保存纯 dict + sklearn PCA，线上/离线均可反序列化。
- 已修复 Kaggle spatial Mamba 推理脚本的依赖边界：脚本不再 import TensorFlow 相关 Perch local 模块，避免仅 spatial ONNX 推理时引入不必要的 TF/SavedModel 依赖。
- 已完成 flat64 推理 debug smoke：`flat64` artifact 会正确使用 `spatial_tokens_64` 或 ONNX 原始 `spatial_embedding.reshape(B,64,1536)`，不会错误退回 mean tokens。
- 当前判断：`Perch spatial_embedding + Mamba-style head` 是新的 Perch-only 主线；`flat64` 是当前最值得下一发线上验证的 Perch-only 单模。

### 2026-05-11

- 补充线上验证两条 CNN 3fold 对齐的 strict `PerchMambaHead`：
- `outputs/perch_spatial_mamba_labeled_all_mean_perchmambav1_cnn195634folds_nopca_noraw_v1`：Local CV `0.889574`，Public LB `0.897`；
- `outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1`：使用 LB `0.916` Perch+Stage3 teacher 的 conservative pseudo，`row_min_max_prob=0.93, margin=0.25, top_k=1, pseudo_loss_weight=0.25`，Local CV `0.890960`，Public LB `0.898`；
- 结论：conservative pseudo 对 strict `PerchMambaHead` 是本地和线上一致的小正收益，但这两条 Public LB 都低于 2026-05-10 mean-only spatial Mamba 首提 `0.899`，因此它们更适合作为“fold 对齐 / pseudo 验证口径”，不直接替代当前 Perch spatial 单模最高 LB 记录。
- `Unified PerchLR + PerchMamba + Stage3 CNN + PerchAttention` 四路 ensemble 已完成线上验证：
- Kaggle 上传包为 `20260511-perch-mam-attn-cnn-cv9283`；
- 本地 CV 记录为 `0.9283`，Public LB 为 `0.920`，刷新当前总体最高线上分；
- 线上使用 [birdclef2026_kaggle_infer_unified_perch_stage3.py](birdclef2026_kaggle_infer_unified_perch_stage3.py) 与 [run_birdclef2026_unified_perch_stage3_kaggle_infer.sh](run_birdclef2026_unified_perch_stage3_kaggle_infer.sh)，共享一次 Perch ONNX 推理给 `PerchLR / PerchMamba / PerchAttention` 三个 head，再叠加 Stage3 CNN；
- 默认融合权重为 `PerchLR=0.25, Mamba=0.30, Stage3=0.15, Attention=0.30`，并使用 `file_scale_topk=2`；
- 这条结果说明：单个 Perch head 之间线上差距很小，但多个 Perch head 与 CNN/Stage3 分支确实存在可用多样性；同时 `CV 0.9283 -> LB 0.920` 表明当前强 ensemble 的 local CV 仍略偏乐观，后续选参不能只追 OOF 小数点。

### 2026-05-14

- 完成以 `0.921` 五路融合为 teacher 的严格防泄露蒸馏实验：
- 新增 [birdclef2026_make_strict_fold_teacher.py](birdclef2026_make_strict_fold_teacher.py)，输出 `outputs/strict_fold_teacher_20260514/strict_fold_teacher_predictions.npz`；
- strict teacher package 使用 `pred_by_fold[k]`，每个 student fold 只读取对应 teacher fold 的预测，避免 teacher 训练时见过 student validation fold；
- strict teacher 自检 OOF-like CV 为 `0.934508`，融合权重为 `PerchLR=0.221875, Mamba=0.26625, Stage3=0.133125, Attention=0.26625, RawWave=0.1125, file_scale_topk=2`；
- [birdclef2026_teacher_oof.py](birdclef2026_teacher_oof.py) 新增 `load_teacher_predictions_for_fold`，训练脚本可按 fold 读取 `pred_by_fold`，验证/CV 仍只用 hard label。
- RawWave strict-teacher student 权重扫描完成：
- 基线 RawWave best 为 `outputs/birdclef2026_raw_waveform_transformer/20260512_013731_raw_wave_conv_tokenizer_base_long_n32_d768`，Local CV `0.710473`；
- `teacher_loss_weight=0.10`：`outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_033211_raw_wave_conv_tokenizer_base_strictteacher_w010`，Local CV `0.773312`；
- `teacher_loss_weight=0.25`：`outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_032109_raw_wave_conv_tokenizer_base_strictteacher_w025`，Local CV `0.778978`；
- `teacher_loss_weight=0.50`：`outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_034059_raw_wave_conv_tokenizer_base_strictteacher_w050`，Local CV `0.788847`；
- `teacher_loss_weight=1.00`：`outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_164133_raw_wave_conv_tokenizer_base_strictteacher_w100`，Local CV `0.812600`，当前 RawWave strict-teacher 最优；
- `teacher_loss_weight=2.00`：`outputs/birdclef2026_raw_waveform_transformer_strict_teacher/20260514_165248_raw_wave_conv_tokenizer_base_strictteacher_w200`，Local CV `0.793196`，出现过蒸馏，判负；
- 结论：强 teacher 对弱 RawWave student 非常有效，但 `w=2.0` 已经压过 hard label 泛化，后续 RawWave distill 默认应从 `w=1.0` 开始。
- Stage3 CNN strict-teacher student 权重扫描完成：
- `teacher_loss_weight=0.10`：Local CV `0.823610`；
- `teacher_loss_weight=0.25`：Local CV `0.835787`；
- `teacher_loss_weight=0.50`：Local CV `0.836034`，当前 strict-teacher Stage3 最高但只比 `w=0.25` 高 `+0.000247`；
- `teacher_loss_weight=1.00`：Local CV `0.832568`，过蒸馏，判负；
- 结论：Stage3 CNN student 可以被 teacher 小幅校准，但不如旧 `Perch+CNN white-list pseudo` 训练出的 Stage3 分支适合最终 ensemble，不建议替换旧 Stage3。
- 完成 strict RawWave 接入五路融合的泄露安全 OOF 验证：
- 为 [birdclef2026_whitelist_blend_unified_raw_waveform.py](birdclef2026_whitelist_blend_unified_raw_waveform.py) 增加 `numpy._core` 兼容 shim，用于 numpy1 环境读取 numpy2 保存的 OOF object arrays，不改变融合逻辑；
- 只替换 RawWave 为 strict-teacher `w=1.0`，保留旧 Stage3，输出目录 `outputs/whitelist_blend_unified_strict_raw_w100_20260514`，最优 Local CV `0.934884`；
- 最优权重为 `PerchLR=0.2275, Mamba=0.273, Stage3=0.1365, Attention=0.273, RawWave=0.09, file_scale_topk=2`；
- 只替换 Stage3 为 strict-teacher `w=0.5` 的结果为 `0.933942`，不如旧五路 `0.934044`；
- 同时替换 strict Stage3 + strict RawWave 的结果为 `0.934238`，不如只替换 strict RawWave；
- 当前可提交本地最高组合是：旧 `PerchLR + PerchMamba + old Stage3 CNN + PerchAttention`，加 strict-teacher RawWave `w=1.0`。
- 已同步更新线上 unified 推理默认权重：
- [run_birdclef2026_unified_perch_stage3_kaggle_infer.sh](run_birdclef2026_unified_perch_stage3_kaggle_infer.sh)：默认 `PERCH_LR_WEIGHT=0.2275, MAMBA_WEIGHT=0.273, STAGE3_WEIGHT=0.1365, ATTENTION_WEIGHT=0.273, RAW_WAVE_WEIGHT=0.09`；
- [birdclef2026_kaggle_infer_unified_perch_stage3.py](birdclef2026_kaggle_infer_unified_perch_stage3.py)：`DEFAULT_WEIGHTS` 同步为上述本地最优配置；
- 线上提交时应上传 strict RawWave `w=1.0` 的 `config.json` 与三个 `fold_*/stage2_fold*_best.pth`，Stage3 仍用旧 `outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716...`，不要换成 strict-teacher Stage3。

### 2026-05-19

- 复现 [discussion/hgnet训练配置.png](discussion/hgnet训练配置.png) 中的 HGNet V8 配置，新增 [run_birdclef2026_gm_train_hgnet_v8.sh](run_birdclef2026_gm_train_hgnet_v8.sh)，并扩展 [birdclef2026_gm_train.py](birdclef2026_gm_train.py) 支持 `logmel_v8`、`input_channels=1`、`LSEHead`、`OneCycleLR`、以及 MixUp/CutMix 按 epoch 延迟开启；
- 关键配置：`hgnetv2_b0.ssld_stage2_ft_in1k`、单通道 `logmel_v8 [1,256,256]`、`n_fft=2048 / win=626 / hop=313 / n_mels=256 / f_min=20 / slaney norm`、`LSEHead(T=1.0, dropout=0.5)`、`4-fold seed=1086`、`MixUp alpha=1.0 prob=0.8 from epoch 5`、`OneCycleLR warmup 5/20`；
- 运行目录为 `outputs/birdclef2026_gm_hgnet_v8/20260519_022330_hgnetv2_b0.ssld_stage2_ft_in1k`；
- Stage1 train_audio best valid AUC 约 `0.89541`，说明特征和模型本身可以正常学习；
- Soundscape fold AUC 分别为 `0.869500 / 0.890486 / 0.860667 / 0.897172`，但 final OOF local CV 只有 `0.668820`；
- 结论：HGNet V8 这条线存在严重 fold 间校准/排序不一致，fold 内分数漂亮但合并 OOF 崩掉，不适合作为当前主线继续投入；
- 后续如继续研究 HGNet，应先做校准/OOF rank 诊断或只作为异质弱分支观察，不建议大规模调参。

### 2026-05-20

- 为了提交 HGNet V8 做线上验证，更新 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py)，使通用 GM 推理脚本支持 `logmel_v8`、`input_channels=1`、`image_normalize=zero_one` 与 `lse_head_v1`；
- HGNet V8 线上最小模型包只需要上传 `outputs/birdclef2026_gm_hgnet_v8/20260519_022330_hgnetv2_b0.ssld_stage2_ft_in1k` 下的 `config.json` 与四个 `fold_*/stage2_fold*_best.pth`；
- 不需要上传 `stage1_audio_best.pth`、`train.log`、`metrics.json`、`soundscape_oof_predictions.csv` 等训练辅助文件；
- Kaggle Notebook 中需要使用更新后的 [birdclef2026_gm_kaggle_infer.py](birdclef2026_gm_kaggle_infer.py)，否则旧脚本无法加载 `logmel_v8 + 1ch + LSEHead` 配置；
- 推荐线上命令：
- `python birdclef2026_gm_kaggle_infer.py --model-root /kaggle/input/<hgnet-dataset>/20260519_022330_hgnetv2_b0.ssld_stage2_ft_in1k --output-path /kaggle/working/submission.csv --segment-batch-size 12`；
- 这次提交属于低成本线上探测：虽然本地 final OOF `0.668820` 很低，但参考朋友的 HGNet 经验，本地 CV 可能严重低估线上 LB，因此值得交一次验证。
- 根据 [AudioProtoPNet.md](前沿论文中的核心创新点/AudioProtoPNet.md) 的核心思想，完成 Perch spatial prototype head v1：
- 在 [birdclef2026_perch_spatial_mamba_train.py](birdclef2026_perch_spatial_mamba_train.py) 中新增 `head_variant=prototype_pooling`；
- 输入为 Perch `spatial_embedding [B,16,4,1536] -> flat64 [B,64,1536]`；
- 每个类别学习 `5` 个 prototype，输入 token 与 class-specific prototype 做 cosine similarity，对 `64` 个局部 token 取 max；
- 最后分类层只连接本类 prototype，prototype 权重用 `softplus` 保证非负，并加入同类 prototype orthogonality loss，尽量贴近 AudioProtoPNet 的“局部原型证据”思想；
- [birdclef2026_perch_kaggle_infer_spatial_mamba.py](birdclef2026_perch_kaggle_infer_spatial_mamba.py) 已同步支持 `prototype_pooling` artifact；
- 新增 [run_birdclef2026_perch_spatial_protopnet_train.sh](run_birdclef2026_perch_spatial_protopnet_train.sh)，默认使用 CNN `20260505_195634` 的 3fold assignment，保持与主融合 OOF 对齐；
- smoke 中修复了 prototype similarity 取 max 维度错误：应对 token 维取 max，而不是 prototype 维；
- 正式输出目录为 `outputs/perch_spatial_protopnet_labeled_all_cnn195634folds_nopca_noraw_v1`；
- 参数：`flat64`、`token_pca_dim=0`、`include_raw_scores=false`、`prototype_per_class=5`、`prototype_temperature=12.0`、`prototype_orth_weight=0.01`、`epochs=260`；
- 3fold OOF 结果：fold AUC `0.851929 / 0.880182 / 0.871555`，`spatial_oof_auc = 0.872171`，`mean_fold_spatial_auc = 0.867889`；
- 后续做了 prototype 容量扫描 `prototype_per_class=2/3/8/12`，输出目录分别为 `outputs/perch_spatial_protopnet_labeled_all_cnn195634folds_nopca_noraw_ppc{2,3,8,12}_v1`；
- 扫描结果分别为 `0.862400 / 0.868417 / 0.870644 / 0.868345`，其中 `ppc=8` 最好，但仍低于原始 `ppc=5` 的 `0.872171` 和 long600 的 `0.873867`；
- 结论：prototype 数量并不是当前瓶颈，继续盲调 `prototype_per_class` 的收益已经很小；如果后续还要推进 prototype 方向，更值得试真正的 audio prototype 初始化或更改 prototype 语义来源。
- 尝试了 `train_audio` token 初始化 prototype：`outputs/perch_spatial_protopnet_audioinit_max100_ppc5_cnn195634folds_nopca_noraw_v1`；
- 该实验仍只在 labeled soundscape 上训练 head，`train_audio` 仅用于 prototype 初值，防泄露边界安全；
- 结果明显判负：fold AUC `0.775716 / 0.811728 / 0.772726`，`spatial_oof_auc = 0.816932`；
- 观察：audio token 原型与 soundscape 域差很大，强行用 audio class centroid 初始化反而把 prototype 拉到错误局部模式。
- 又尝试了 `soundscape_token` 初始化 prototype：`outputs/perch_spatial_protopnet_soundscapeinit_ppc5_cnn195634folds_nopca_noraw_v1`；
- 这版只用 outer-train soundscape token 初始化，不看 validation，仍然是 fold-safe；
- 结果更差：fold AUC `0.810956 / 0.859087 / 0.844449`，`spatial_oof_auc = 0.806403`，且 `fold_gap = +0.031761`，明显过拟合；
- 结论：无论 audio 还是 outer-train soundscape 初始化，都没有把 prototype head 推成强线，说明这条 AudioProtoPNet-style 路线在当前 Perch spatial 表示上不值得继续加码。
- 结论：AudioProtoPNet-style prototype head 能稳定学习 Perch spatial 局部模式，但当前 v1 明显弱于现有 Perch spatial Mamba / attention / SSM 强线，不建议直接加入主融合；后续若继续，只适合作为可解释性诊断或轻量 ablation，不作为短期提分主线。

### 2026-05-21

- 根据 [Domain-Invariant.md](前沿论文中的核心创新点/Domain-Invariant.md) 中 ProtoCLR 思想，给 [birdclef2026_perch_embedding_mlp_train.py](birdclef2026_perch_embedding_mlp_train.py) 增加 batch-level multilabel ProtoCLR 辅助损失；
- 实现方式：`Perch embedding -> MLP encoder -> hidden embedding`，主损失仍是 fold-safe BCE，辅助损失在 batch 内按有正样本的类别构造 prototype，正样本拉近本类 prototype、推远其他 batch prototype；
- 默认参数保持关闭，新增 `--protoclr-weight-stage1 / --protoclr-weight-stage2 / --protoclr-temperature` 等开关，[run_birdclef2026_perch_embedding_mlp_train.sh](run_birdclef2026_perch_embedding_mlp_train.sh) 已同步；
- 对照基线：`outputs/perch_embedding_mlp_labeled_cnn195634folds_h768_384_v1`，labeled-only embedding MLP Local CV `0.881232`；
- ProtoCLR stage2-only 权重扫描：
- `w=0.02`：`outputs/perch_embedding_mlp_protoclr_w002_labeled_cnn195634folds_h768_384_v1`，Local CV `0.881565`；
- `w=0.05`：`outputs/perch_embedding_mlp_protoclr_w005_labeled_cnn195634folds_h768_384_v1`，Local CV `0.882017`，当前该线最好；
- `w=0.10`：`outputs/perch_embedding_mlp_protoclr_w010_labeled_cnn195634folds_h768_384_v1`，Local CV `0.876221`，过强判负；
- 结论：ProtoCLR 在 Perch global embedding MLP 上有小正收益，但幅度只有 `+0.000785`，不是大分 trick；若继续，应谨慎尝试 audio stage1 轻权重 ProtoCLR 或把该思想迁移到更强的 spatial/SSM head。
- 进一步尝试了 `train_audio` stage1 + ProtoCLR：`outputs/perch_embedding_mlp_protoclr_audio_max100_s1w002_s2w005_cnn195634folds_h768_384_v1`；
- 配置为 `stage1 weight=0.02, stage2 weight=0.05`，仍只在 fold-safe 训练折内做 soundscape 评估；
- 结果回落到 `embedding_oof_auc = 0.873098`，明显低于 labeled-only ProtoCLR `0.882017`，也低于原始 labeled-only `0.881232`；
- 结论：ProtoCLR 思想本身对 Perch embedding MLP 有轻微帮助，但一旦叠加 train_audio stage1，收益被域差抵消，说明这条线不适合继续往 audio 方向加深。

### 2026-05-22

- 将 [Domain-Invariant.md](前沿论文中的核心创新点/Domain-Invariant.md) 的 ProtoCLR 思想从 Perch embedding MLP 迁移到更强的 Perch 60s sequence SSM；
- 修改 [birdclef2026_perch_sequence_ssm_train.py](birdclef2026_perch_sequence_ssm_train.py)：`ProtoSSMHead` 新增 `encode()` / `return_features`，训练时在已标注窗口 hidden feature 上额外计算 batch-level multilabel ProtoCLR；
- 修改 [run_birdclef2026_perch_sequence_ssm_train.sh](run_birdclef2026_perch_sequence_ssm_train.sh)：新增 `PROTOCLR_WEIGHT / PROTOCLR_TEMPERATURE / PROTOCLR_MIN_CLASSES / PROTOCLR_MIN_POS_PER_CLASS`，默认关闭，旧实验行为不变；
- 泄露边界：standardizer 仍只 fit outer-train files，ProtoCLR prototype 只由当前训练 batch 的 outer-train labeled windows 构造，validation fold 不参与 prototype、scaler 或 loss；
- 对照 baseline：`outputs/perch_sequence_ssm_d192_l2_crossattn_cnn195634folds_v1`，`ssm_oof_auc = 0.896786`；
- ProtoCLR 权重扫描：
- `w=0.02`：`outputs/perch_sequence_ssm_protoclr_w002_d192_l2_crossattn_cnn195634folds_v1`，`ssm_oof_auc = 0.896967`，基本持平；
- `w=0.05`：`outputs/perch_sequence_ssm_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1`，`ssm_oof_auc = 0.899790`，当前 SSM+ProtoCLR 最好；
- `w=0.10`：`outputs/perch_sequence_ssm_protoclr_w010_d192_l2_crossattn_cnn195634folds_v1`，`ssm_oof_auc = 0.896803`，过强后收益消失；
- 结论：ProtoCLR 对 SSM 有明确但小幅正收益，最佳约 `+0.003004`；它更像 representation regularizer，不是大分 trick，后续可作为 SSM 分支默认候选，但不值得继续大规模扫权重。
- 根据 [Animal2vec.md](前沿论文中的核心创新点/Animal2vec.md) 尝试 raw waveform 的 SincNet-style 可学习滤波前端；
- 修改 [waveform_model.py](waveform_model.py)：新增 `SincConv1D`、`SincRawAudioTokenizer`，并让 `RawWaveTransformerModel`/`MixerModel` 支持 `tokenizer_type=sinc_stack`；
- 修改 [birdclef2026_raw_waveform_transformer_train.py](birdclef2026_raw_waveform_transformer_train.py) 和 [run_birdclef2026_raw_waveform_transformer_train.sh](run_birdclef2026_raw_waveform_transformer_train.sh)：raw wave 启动从 shell env 改为命令行直传，避免本环境里 `VAR=value command`、`tee`、`CUBLAS_WORKSPACE_CONFIG`、`PYTHONHASHSEED` 触发 CUDA 不可见；
- smoke 通过后正式跑 `outputs/birdclef2026_raw_waveform_transformer/20260522_154155_raw_wave_sinc_tokenizer_base_long_n32_d768`，但结果明显判负：`final_oof_cv = 0.517147`；
- 进一步修正 Sinc 公式后重新 smoke/正式验证，仍然判负，最终 `final_oof_cv = 0.483708`；
- 观察：stage1 AUC 长期停在接近随机，说明这条 Sinc 前端在当前监督训练范式下没有学成有效滤波器，至少当前实现不适合作为主线 raw branch；
- 结论：Animal2vec 的“可学习滤波前端”思路值得保留，但直接替换我们现有 raw Conv tokenizer 并没有收益，后续若继续，只能走真正的 self-supervised pretrain 或更强的 waveform tokenizer 设计。
- 对当前 unified Perch+Stage3+RawWave 4/5 路 OOF 做了后处理小网格，只扫 `file_scale_topk=1/2/3`，不动模型、不碰 hidden test；
- 结果：`file_scale_topk=2` 仍然最好，`topk=1` 的 best_auc `0.933797`，`topk=3` 的 best_auc `0.933549`，都低于此前 `topk=2` 的 `0.934044`；
- 结论：文件级 top-k 缩放已经基本摸到头，继续扫 `1/2/3` 不会带来更大收益，后面若还想抠分，更值得看模型多样性而不是继续加重后处理。

### 2026-06-01

- 线上验证 `20260531-ensemble-cv9422 safe family3 no RawWave no TTA`：
- 模型为 `PerchLR + PerchMamba + PerchAttention + PerchSSM + Stage3 CNN`，线上关闭 RawWave、Mamba TTA、Stage3 TTA，使用 `BLEND_MODE=family3` 与 `file_scale_topk=2`；
- 本地 CV 口径为 `family3 + file_scale_topk2 = 0.937399`，Public LB 为 `0.922`，`LB-CV = -0.015399`；
- 该分数略高于旧四路 unified `0.920`，说明 SSM/family3 快版线上仍有正收益；
- 但相对本地 CV 折损明显，说明去掉 RawWave/TTA 后本地 0.9374 不能按比例外推，family3/SSM 的 OOF 增益只有一部分能转到线上；
- 后续提交选择上，若时间风险优先，仍可把 `0.922` 快版作为安全底线；若还有尝试额度，更值得测试“只加低成本 Stage3 OpenVINO TTA”或“少量恢复 RawWave/不加 Mamba TTA”的中间版，而不是直接回到最重 `0.942178` 配置。
- 线上继续验证 `OPENVINO_FORCE_TRACE=1 + STAGE3_BACKEND=openvino + RAW_WAVE_BACKEND=openvino` 的重线仍然超时；原因是 RawWave 在 Kaggle 上可能回退 PyTorch，且 Mamba TTA 需要额外两次 Perch ONNX，二者叠加不可控。后续正式提交不再推荐 RawWave 或 Mamba TTA。

### 2026-06-02

- 线上验证 retuned safe family3 no RawWave no TTA：
- 本地 OOF 小网格把权重调为 `PerchLR=0.20, Mamba=0.112, Stage3=0.1375, Attention=0.208, SSM=0.3425`，CV 从安全版约 `0.9374` 涨到 `0.938967`；
- 线上 Public LB 仍为 `0.922`，与旧安全版持平；
- 结论：最后阶段 OOF 全局权重小涨不再可信，尤其高 SSM 权重可能只是在 739-row OOF 上贴合；后续若无新结构信息，不建议继续靠重调权重期待线上提升。

### 2026-06-03

- 为 SSM head 增加 fold-safe teacher distillation：
- 修改 [birdclef2026_perch_sequence_ssm_train.py](birdclef2026_perch_sequence_ssm_train.py)，新增 `--teacher-oof-path / --teacher-key / --teacher-loss-weight`；
- teacher OOF 会按 `row_id` 对齐成 `[file, 12, class]`，只在 labeled windows 上参与训练 loss，validation/OOF 仍只用 hard labels 评估；
- 修改 [run_birdclef2026_perch_sequence_ssm_train.sh](run_birdclef2026_perch_sequence_ssm_train.sh)，新增 `TEACHER_OOF_PATH / TEACHER_KEY / TEACHER_LOSS_WEIGHT`，默认关闭；
- smoke 通过：`outputs/perch_sequence_ssm_teacher_distill_smoke`；
- 正式尝试 1：`outputs/perch_sequence_ssm_teacher_best_w005_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1`，`teacher_key=best, teacher_loss_weight=0.05, ProtoCLR=0.05`，SSM OOF `0.898452`；
- 正式尝试 2：`outputs/perch_sequence_ssm_teacher_baseline_w002_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1`，`teacher_key=baseline, teacher_loss_weight=0.02, ProtoCLR=0.05`，SSM OOF `0.896067`；
- 对照原始 SSM+ProtoCLR `outputs/perch_sequence_ssm_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1` 为 `0.899790`；
- 结论：SSM teacher distillation 判负，不替换最终 PerchSSM artifact。teacher 似乎把 SSM 拉向当前 OOF ensemble 的本地偏差，没有增强线上可用的独立信息。
- 继续做 safe no-RawWave/no-TTA 分支替换探针：
- 新增 [birdclef2026_safe_oof_branch_probe.py](birdclef2026_safe_oof_branch_probe.py)，只读取 OOF，复现线上安全口径 `PerchLR + Mamba + Attention + SSM + Stage3`、`BLEND_MODE=family3`、`file_scale_topk=2`；
- `seed2027` SSM 稳定性尝试判负：`outputs/perch_sequence_ssm_protoclr_w005_d192_l2_crossattn_seed2027_cnn195634folds_v1`，`ssm_oof_auc = 0.892954`，低于原 SSM `0.899790`；
- 分支替换最强结果：
- `Stage3` 替换为 `outputs/birdclef2026_gm_stage3_oof_teacher/20260514_003349_convnextv2_atto.fcmae_ft_in1k_stage3_oof_teacher_w0.25`，retuned safe local CV 从 `0.938961` 到 `0.943824`；
- `Mamba` 替换为 `outputs/perch_spatial_mamba_labeled_all_flat64_pos_nopca_noraw_v1/perch_spatial_mamba_artifacts.joblib`，retuned safe local CV 到 `0.943169`；
- 二者同时替换，retuned weights `PerchLR=0.20, Mamba=0.112, Stage3=0.1375, Attention=0.208, SSM=0.3425`，local CV 到 `0.945506`，fold AUC 为 `0.939471 / 0.949704 / 0.945736`；
- 默认权重下二者同时替换也有 `0.943903`，说明增益主要来自分支替换而不是单纯权重重调；
- 本地 smoke 通过：`Stage3 OOF-teacher + Mamba flat64_pos + SSM + family3 + no RawWave/no TTA`，4 个 train soundscape CPU 推理 `16.8s`，约 `4.19s/file`，输出 235 列正常；
- 结论：这是当前最值得线上再试的一刀。它不增加 RawWave/TTA，不额外跑 Perch，只换 `MAMBA_MODEL_PATH` 和 `STAGE3_MODEL_ROOT` 两个 artifact；风险在于 Public LB 已显示 OOF-LB gap 较大，但这条是“更强/更异质支线”而非继续调全局权重。
- 泄露复查：`0.945506` 使用的 `Stage3 OOF-teacher w0.25` teacher 文件是普通 OOF `pred`，不含 `pred_by_fold` strict teacher slice；训练时验证 fold 不直接参与 loss，valid loss/CV 只用 hard label，但 teacher 对 train rows 可能来自曾见过当前 student-valid fold 标签的 teacher fold，因此存在 indirect leakage / optimistic CV 风险；
- 更严格替代口径：`Mamba flat64_pos + strict Stage3 teacher w0.1` 为 `0.943794`，fold AUC `0.938676 / 0.942151 / 0.944045`；`strict w0.25` 为 `0.943327`。若要完全避开普通 OOF-teacher 的间接泄露争议，优先提交 strict teacher 版。

## 11. 后续实验记录模板

后续每次实验建议按下面格式补充，方便回顾与横向比较：

```md
### YYYY-MM-DD 实验名

- 模型：
- 训练命令：
- 主要参数：
- local CV：
- 是否提交 Kaggle：
- Public LB：
- 关键观察：
- 下一步：
```
