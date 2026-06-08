# Tricks For Ours

用于记录这个 BirdCLEF 2026 项目里：

- 哪些 trick 我们已经试过
- 哪些 trick 还没试
- 对于还没试的部分，下一步应该先做什么

本文档只站在“我们当前这套 pipeline”的角度整理，不追求覆盖所有学术或比赛技巧，而是优先服务提分。

## 1. 当前主线

我们当前已经验证有效的主线是：

- `CNN / ConvNeXt / EfficientNet` 类 backbone
- `stage1 train_audio -> stage2 train_soundscapes -> stage3 pseudo`
- 强化版 inference script 后处理

当前最重要的结论是：

- `stage3 pseudo` 是有效的
- `增强版推理` 也是有效的
- 二者可以叠加，而不是二选一

当前已知最高线上分数：

- `0.873`
- 对应目录：
  `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`
- 对应推理脚本：
  `birdclef2026_gm_kaggle_infer_ensemble.py`

## 2. 已试过的 Tricks

### 2.1 训练框架类

- `stage1 + stage2` 两阶段训练
  - `stage1` 用 `train_audio`
  - `stage2` 用有标签 `train_soundscapes`
  - 这是当前主框架，已验证有效

- `stage3 pseudo finetune`
  - 已完成完整链路
  - 已验证相比早期 stage3 版本有提升
  - 已验证和增强版推理可以叠加收益

- `secondary_labels`
  - 当前 `stage1 train_audio` 已使用 `secondary_labels`
  - 属于已经启用的标签增强

- `freeze backbone`
  - `stage2 / stage3` 都用过前若干轮冻结 backbone 的策略

### 2.2 数据与 CV 类

- `按 filename 分 fold`
  - `stage2` 不是按 5 秒片段随机分，而是按完整 soundscape `filename` 分组
  - 这是当前本地 CV 可信度的核心

- `fold-specific pseudo label`
  - 生成 pseudo 时按 fold teacher 分开生成
  - 目的是避免本地 CV 泄露

- `保守版 pseudo 筛选`
  - 已试过更严格的 `prob_threshold / row_min_max_prob / top_k_labels`
  - 已试过限制 `max_pseudo_rows`
  - 已试过降低 `pseudo_loss_weight / pseudo_sampler_weight`

### 2.3 特征与增强类

- `3 通道多尺度 mel`
  - 当前 mel 不是单尺度，而是 3 组不同 `n_fft / hop / f_max` 的 mel 图拼成 3 通道

- `SpecAug`
  - 已启用 `time mask`
  - 已启用 `freq mask`

### 2.4 推理与后处理类

- `row-level TTA`
  - 增强版推理脚本已支持 `tta_offsets`

- `temporal smoothing`
  - 增强版推理脚本已支持 `smoothing_kernel`

- `soundscape-level top-k postprocess`
  - 增强版推理脚本已支持 `soundscape_top_k`

- `同一模型换更强推理脚本`
  - 已明确验证有效
  - 这是当前最稳定、最直接的提分点之一

- `stage3 + 增强版推理`
  - 已验证有效
  - 当前全表最高分 `0.873` 就来自这个组合

### 2.5 backbone 尝试类

- 已尝试过多种 `timm` backbone
  - `tf_efficientnetv2_s.in21k_ft_in1k`
  - `convnext_atto.d2_in1k`
  - `tf_efficientnet_b0.ns_jft_in1k`
  - `eca_nfnet_l0.ra2_in1k`
  - `convnextv2_atto.fcmae_ft_in1k`
  - 若干 `ViT / DINOv3 / ConvNeXt` 变体

- 已验证一个重要事实
  - 本地 CV 高，不一定线上更高
  - teacher 选择不能只看本地 CV

## 3. 还没试过的 Tricks

下面按“我认为我们现在最应该优先尝试的顺序”排列。

### Priority 1. 多 teacher pseudo label

我们现在主线 pseudo 更接近：

- 单 teacher 或单 run 主导

还没系统试过：

- 多个强模型一起生成 pseudo
- teacher 先 ensemble 再生成 pseudo
- 用增强版推理脚本的输出作为 teacher pseudo 来源

优先级最高的原因：

- 和我们当前 pseudo 框架最接近
- 可以把复杂度放在线下 teacher 侧，线上仍保持单模型推理
- 不会像双模型线上 ensemble 那样直接吃掉 Kaggle 的推理时限
- 往年高分方案常见

### Priority 2. 迭代 pseudo label

我们现在严格来说还是第一轮 pseudo。

还没试：

- `teacher -> pseudo -> stage3`
- 再拿更强 `stage3 / ensemble` 反过来生成第二轮 pseudo
- 做第 2 轮或第 3 轮 pseudo 训练

推荐原因：

- 和当前 pipeline 最连续
- 往年解法里经常有效
- 很可能比完全换范式更稳

### Priority 3. 更完整的 overlap inference

我们现在有：

- `tta_offsets`

但还没系统做成 BirdCLEF 往年常见那种：

- `2.5s overlap`
- `overlap average`
- `overlap max`
- `overlap average + local max`

推荐原因：

- 属于推理侧 trick
- 不需要重训模型
- 有机会继续榨出线上收益

### Priority 4. 真正的多模型 ensemble

当前虽然有增强版 inference script，但我们的最高分本质上还是：

- 单个 `stage3 run`
- 加增强版推理

我们已经做过一次非常关键的线上验证：

- `outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo`
- 加上
  `outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k`
- 在线上用增强版推理脚本做双模型 ensemble
- 会超时

这说明：

- “多模型 ensemble” 不是无效
- 而是当前 Kaggle 线上时限下，至少这个双模型组合已经不够安全

如果后面还要继续尝试这个方向，建议只考虑：

- 更轻的 backbone 组合
- 更小的 TTA
- 更少的 overlap / smoothing
- 或者只在线下把 ensemble 用作 teacher，不直接线上双跑

所以它依然有价值，但优先级应低于“线上仍保持单模型”的 trick。

### Priority 5. 多时长模型

当前主训练基本围绕 `5 秒 clip`。

还没系统试：

- `8 秒`
- `10 秒`
- `15 秒`
- `5s + 8s`
- `5s + 10s`

推荐原因：

- 往年方案里经常有效
- 和多模型 ensemble 非常搭
- 不同时长模型通常具有互补性

### Priority 6. mixup / cutmix / specmix / sumix

目前我们有 `SpecAug`，但没有真正上：

- `audio mixup`
- `cutmix`
- `specmix`
- `sumix`

推荐原因：

- 属于经典增强
- 容易和当前代码融合

但为什么不是更高优先级：

- 它们对 BirdCLEF 这类任务不一定稳定涨
- 有时更适合增加 ensemble 多样性，而不是单模型冲榜

### Priority 7. 更换 loss

当前主损失还是：

- `BCEWithLogitsLoss`

还没试：

- `ASL / Asymmetric Loss`
- `Focal Loss`
- `class-balanced BCE`
- `pos_weight`

推荐原因：

- 对长尾类别、多标签不均衡任务可能有效

但为什么不是前 3 优先级：

- 这类改动有时会明显改掉训练动力学
- 本地与线上不一定稳定同向

### Priority 8. train_audio 的自蒸馏 secondary label

我们现在只是用了官方给的：

- `secondary_labels`

还没试：

- 先训 teacher
- 给 `train_audio` 自动补更多可能遗漏的 secondary labels
- 再重新训练 stage1

推荐原因：

- 往年方案里这类思路是有价值的
- 对 `train_audio` 的监督质量可能有帮助

但工程量已经比前面几项更大。

### Priority 9. Head / pooling 改造

当前 head 比较朴素：

- `global pooling + dropout + linear`

还没试：

- `GeM`
- attention pooling
- 更复杂的 pooling head

推荐原因：

- 有机会带来单模型提升

但优先级不高：

- 单独改 head 的收益未必比 ensemble / pseudo 明显

### Priority 10. 更强音频增强

还没试：

- background noise
- pink noise
- gain
- Gaussian noise

推荐原因：

- 可以增加鲁棒性

但优先级更靠后：

- 当前 pipeline 已经有足够多更高收益的候选方向

### Priority 11. EMA / SWA

当前还没试：

- `EMA`
- `SWA`

优点：

- 理论上能让模型更稳

缺点：

- 需要改训练保存与验证逻辑
- 当前不如 ensemble / pseudo / overlap 直接

### Priority 12. SED 路线

还没试：

- framewise / segmentwise 输出
- SED 风格 head
- SED 风格 pseudo

为什么现在不优先：

- 改动很大
- 不只是加一个 trick，而是换一套训练范式
- 当前 `CNN/ConvNeXt + pseudo + inference trick` 这条线还在继续涨分

结论：

- `SED` 是未来可考虑的大方向
- 但不是我们眼下最应该先做的事

## 4. 当前建议的下一步顺序

如果只按“最值得马上做”的顺序，我建议：

1. `多 teacher pseudo`
2. `迭代 pseudo`
3. `更完整 overlap inference`
4. `多模型 ensemble（仅限不超时的轻量版本，或仅作为离线 teacher）`
5. `多时长模型`

## 5. 从现阶段 code 区开源高分方案里学到什么

我额外看了 `现阶段线上code区开源高分代码` 里的多份 notebook，结论是：

- 它们大致分成两类
- 一类是相对正常的音频模型方案
- 另一类是明显偏 `Perch + metadata prior + 多层后处理` 的高公榜方案

### 5.1 两大流派

- 朴素音频模型流
  - 代表思路类似：
    `EfficientNetV2-S + LSE pooling + EMA + 小 fold ensemble + ONNX 加速`
  - 这类更像强 baseline 或正常工程优化

- 高公榜堆叠流
  - 代表思路类似：
    `Perch v2 -> ProtoSSM -> MLP / probe -> ResidualSSM -> TTA -> smoothing / scaling / threshold`
  - 这类 notebook 往往公榜很高，但里面混有大量公榜特化成分

### 5.2 他们反复在用的 trick

- 强预训练音频前端
  - `Perch v2`

- 时序建模
  - 不把每个 `5s` 窗口完全独立处理
  - 而是把完整 `1min` 的 `12 x 5s` 序列一起建模

- embedding 上的二级分类器
  - `MLP`
  - `Logistic Regression`
  - `probe / stacker`

- 双阶段修正
  - first-pass 先预测
  - second-pass 再学残差修正

- 训练 trick
  - `mixup`
  - `cutmix`
  - `focal loss`
  - `label smoothing`
  - `SWA`
  - `knowledge distillation`
  - `taxonomic auxiliary head`

- 推理 trick
  - `TTA`
  - `temporal smoothing`
  - `file-level scaling`
  - `rank-aware scaling`
  - `temperature scaling`
  - `per-class threshold`

- 元数据 trick
  - `site / hour / month prior`
  - 有些方案把这些 prior 融合得非常重

### 5.3 我认为真正有泛化价值的部分

- `强 teacher / 强音频前端`
  - 这些高分方案里，最像“硬实力提升”的部分，不是后处理，而是底座本身更强
  - 对我们最可迁移的含义是：
    继续找更强 teacher，用于 `pseudo` 或 `distill`

- `显式利用 1 分钟时序信息`
  - 这是我认为第二有价值的点
  - 它们不是只做单个 `5s` 分类，而是在利用 12 个窗口之间的上下文关系
  - 对我们来说，对应的可迁移方向包括：
    `更完整 overlap inference`
    或后续考虑更轻量的时序聚合

- `轻量且通用的推理增强`
  - `TTA`
  - 温和 `smoothing`
  - 这类通常比“精细校准每个类阈值”更健康

- `常规训练稳定器`
  - `mixup / focal / label smoothing / SWA`
  - 这类不是必涨分，但属于可以认真排队测试的标准组件

- `knowledge distillation`
  - 他们很多高分方案其实隐含着“强 teacher -> 小模型/次级模型”的思想
  - 这和我们当前的 `pseudo` 主线非常一致

### 5.4 我认为不要盲抄的部分

下面这些不是说一定没用，而是非常容易变成“针对公榜分布调参”：

- `site / hour / month prior` 的强融合
- `per-class threshold` 优化
- `per-taxon temperature scaling`
- `rank-aware scaling`
- `file-level top-k confidence scaling`
- `ResidualSSM + prior + threshold + temperature + smoothing` 一整串叠满
- `proxy species / genus proxy / unmapped class` 一类的比赛特化处理

原因很简单：

- 这些东西很多都依赖当前测试集分布
- 本地 CV 即便做得再认真，也不一定能忠实反映它们的真实泛化
- 很容易把我们带到“公榜看起来涨了，但私榜风险变大”的方向

### 5.5 对我们最有价值的可执行优先级

基于这批 notebook，我会把“值得借到我们当前 pipeline 里”的优先级排成这样：

1. `更强 teacher -> 多 teacher pseudo / distill`
2. `更完整地利用 1min 时序`
3. `轻量 TTA + 温和 smoothing`
4. `mixup / focal / label smoothing / SWA`
5. `如果后面确实需要，再考虑更轻量的时序模型或 SED 化`

不建议优先投入的方向：

- `metadata prior`
- `per-class threshold`
- `一大串公榜型校准后处理`

### 5.6 一句话总结这批开源代码

这批高分 code 真正给我们的启发不是：

- 去模仿它们那套复杂的公榜后处理

而是：

- `更强 teacher`
- `更好利用 1 分钟时序`
- `把复杂度尽量放在线下训练和 pseudo 侧`
- `线上尽量保持单模型、稳定、不卡时限`

## 6. 一句话版结论

我们已经验证有效的核心组合是：

- `stage2 / stage3 pseudo`
- `增强版推理`

下一步最值得做的不是立刻切 SED，而是继续深挖：

- `ensemble`
- `更强 teacher pseudo`
- `迭代 pseudo`
- `overlap inference`

这些方向和我们当前工程最兼容，也最可能继续稳定提分。

## 7. 2026-05-13 当前新主线：Perch 五路融合之后还能做什么

截至 `2026-05-13`，我们已经把主线从早期纯 CNN / mel 体系推进到：

- `Perch LR`
- `Perch Mamba`
- `Perch Attention`
- `Stage3 CNN`
- `RawWave Transformer`

当前最强本地与线上结果：

- local CV: `0.934044`
- online LB: `0.921`
- 线上推理没有超时

这说明：

- `Perch` 特征已经是当前最强底座
- `CNN` 与 `RawWave` 的价值主要是提供互补信息源
- 继续微调已有五路融合权重的收益会越来越小
- 下一步应该优先寻找新的互补结构，而不是只做小数点级别的权重搜索

### Priority 1. Perch 60s temporal head

当前多数 Perch head 仍然近似把每个 `5s` 窗口独立预测。

更值得试的方向是：

- 一次读取完整 `60s soundscape`
- 得到 12 个窗口的 Perch spatial tokens
- 输入形状类似：`[12, 16, 4, 1536]`
- 经过 temporal Transformer / Mamba / attention
- 输出 12 个窗口各自的 234 类预测

为什么优先级最高：

- Perch 仍然只需要跑一次，不显著增加前端推理成本
- 可以利用前后窗口的连续鸟叫信息
- 正好补上当前 `5s independent head` 的短板
- 比继续调融合权重更可能带来真正的新信息

注意事项：

- 训练和验证必须保持 fold-safe
- 如果用 pseudo teacher，训练集内必须使用 OOF teacher
- 不要让同一个 soundscape 的验证窗口信息泄露进训练

### Priority 2. 继续强化 RawWave 互补分支

RawWave 单模 CV 不高，但加入 ensemble 后能明显提分，说明它学到的是 Perch / CNN 没完全覆盖的东西。

下一步不建议只做更小模型，因为小模型虽然略快，但损失了大量融合增益。

更值得试：

- 不同 seed 的 RawWave
- 更强 waveform augment
- time shift / gain / polarity flip / noise
- 轻微结构变化，但保持线上可承受
- 只保留一个最有互补性的 RawWave 分支上线

判断标准：

- 不只看 RawWave 单模 CV
- 更要看加入五路融合后的 OOF 增益
- 如果推理仍不超时，则优先选择 ensemble 增益更大的版本

### Priority 3. 用当前 0.921 五路融合做 teacher

当前 `0.921 LB` 的五路融合可以作为很强 teacher。

可尝试：

- 训练 Perch temporal student
- 训练 RawWave student
- 训练更轻的线上 student
- 使用 soft pseudo target，而不是只用 hard label

最关键风险：

- 必须 fold-safe
- 训练集内只能用 OOF teacher
- 不能用同折模型预测同折训练样本再回灌训练

推荐策略：

- 对 train soundscapes 使用 OOF teacher
- 对无标签或额外 soundscapes 可以使用 full teacher
- pseudo loss weight 不要一开始太大
- 先让 teacher 起校准作用，而不是压过真实标签

### Priority 4. Submission-level 风险控制

当前 `0.934044 local CV -> 0.921 LB` 应该作为新的主线 best。

后续实验的提交原则：

- 如果本地只涨 `0.001`，但复杂度明显增加，优先不替换主线
- 如果新增分支推理接近超时，必须先 debug 计时再提交
- 如果提升主要来自少数类别暴涨，要谨慎判断是否 OOF 过拟合
- 不做 per-class 权重、per-class threshold 这类高风险公榜型调参

当前最推荐的下一步：

1. `Perch 60s temporal head`
2. `RawWave 互补分支增强`
3. `0.921 五路 teacher -> fold-safe student / pseudo`
4. `只在可解释、可复现、不过拟合的范围内做 submission-level 调整`
