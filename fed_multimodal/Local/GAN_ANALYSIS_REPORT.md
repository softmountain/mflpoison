# GAN 训练、结构、评估与作用分析

本文分析当前仓库中 UCF101 多模态特征 GAN 的实现、训练数据来源、网络结构、评估手段、已有训练效果与用途。重点以当前 `fed_multimodal/Local` 下的本地 GAN 主线为准，同时补充早期联邦版和 demo 版实现。

## 1. 总体结论

当前 GAN 不是直接生成原始音频或视频，而是生成已经提取好的多模态特征：

- 音频特征：MFCC，形状通常为 `[T_audio, 80]`，默认最长约 500 帧。
- 视频特征：MobileNetV2 特征，形状通常为 `[T_video, 1280]`，默认约 9 帧。
- 标签：UCF101 子集中的类别标签，当前本地数据管理器实际从 feature 文件中构造 51 类索引，但评估抽样中出现了 42 个有足够样本的类别。

训练目标是让生成器在给定噪声 `z` 和类别标签 `label` 的情况下，生成一对音频/视频特征，使其：

1. 单模态上看起来像真实音频/视频特征；
2. 保留类别语义，能被 teacher 分类器识别为指定类别；
3. 音频和视频这两个模态之间的联合关系尽量接近真实样本；
4. 具备一定样本多样性，可用于合成数据训练、标签翻转攻击和鲁棒性分析。

从已有结果看，当前 GAN 的边缘分布对齐较好，teacher 语义一致性很强，但生成样本多样性明显不足，联合模态关系仍不稳定。换言之，它已经能生成“分类器很容易认出来的特征”，但还没有完全生成“和真实数据难以区分、且多样性充分的真实特征分布”。

## 2. 训练入口与流程

### 2.1 当前本地主线入口

当前最重要的本地训练入口是：

- `fed_multimodal/Local/train_local_gan.py`
- 自动化脚本：`fed_multimodal/Local/run_gan_experiment.sh`
- 评估脚本：`fed_multimodal/Local/eval_local_gan_quality.py`

典型流程是：

1. 先训练或准备一个本地多模态分类器 teacher：默认路径为 `results/local_training/best_model.pt`。
2. 调用 `train_local_gan.py` 加载 teacher。
3. 构建 UCF101 本地 DataLoader，使用完整训练集 `full_train` 训练 GAN。
4. 每个 batch 取真实音频特征、真实视频特征、长度和标签。
5. 生成器输入随机噪声和标签，输出 fake audio / fake video。
6. 训练生成器、单模态判别器和 joint critic。
7. 保存 checkpoint 到 `fed_multimodal/Local/results/local_gan/ckpt_<epoch>_<exp>.pt`。
8. 用 `eval_local_gan_quality.py` 对 checkpoint 做质量分析，输出到 `fed_multimodal/Local/results/gan_analysis/ckpt_<epoch>_<exp>/`。

`run_gan_experiment.sh` 默认参数包括：

- `GAN_EPOCHS=200`
- `BATCH_SIZE=32`
- `GAN_LR_G=2e-4`
- `GAN_LR_D=1e-4`
- `GAN_RF_WEIGHT=2.0`
- `GAN_AUX_WEIGHT=1.0`
- `GAN_CLS_WEIGHT=0.1`
- `GAN_JOINT_WEIGHT=0.05`
- `GAN_FM_WEIGHT=0.05`
- `GAN_MOM_WEIGHT=0.05`
- `GAN_JOINT_D_STEPS=3`
- `GAN_JOINT_LR_MULT=2.0`

训练中还有 warmup/ramp 策略：

- 前 `warmup_ratio` 比例的 epoch 暂时关闭 teacher loss 和 joint loss，让单模态 GAN 先稳定；
- 后续 `ramp_ratio` 阶段逐步把 `cls_weight` 和 `joint_weight` 拉升到目标值。

这样做是为了避免训练初期 teacher 语义约束或 joint critic 过强，导致生成器还没学会基本分布时就被复杂目标拖垮。

### 2.2 联邦版入口

早期或联邦版入口是：

- `fed_multimodal/experiment/ucf101/train_gan_generator.py`
- 脚本：`fed_multimodal/experiment/ucf101/run_gan_generator.sh`

这个版本会先训练或加载联邦学习得到的全局模型，再以它作为 teacher 训练 GAN。输出路径主要是：

- `fed_multimodal/result/gan_generator/ucf101/`
- `fed_multimodal/result/gan_attack_improved/ucf101/`

该分支更贴近“联邦模型 + GAN 生成攻击/增强”的原始实验设置，但当前 `Local` 目录下的版本更加精简、可控，也有更多评估和消融结果。

### 2.3 Demo 版入口

还有 demo 版：

- `fed_multimodal/demo/ucf101_demo_gan/train_gan.py`
- `fed_multimodal/demo/ucf101_demo_gan/eval_gan.py`

这部分用于演示，不是当前主要分析对象。

## 3. 训练数据

### 3.1 数据来源

本地 GAN 使用的不是原始 UCF101 视频文件，而是预先提取好的 feature pickle：

- 音频：`fed_multimodal/results/feature/audio/mfcc/ucf101/feature.pkl`
- 视频：`fed_multimodal/results/feature/video/mobilenet_v2/ucf101/feature.pkl`
- 官方划分文件：`fed_multimodal/datasets/ucf101/ucfTrainTestlist/trainlist01.txt` 和 `testlist01.txt`

加载逻辑在：

- `fed_multimodal/Local/dataloader.py`

`UCF101LocalDataManager` 会：

1. 读取 audio/video 两个 feature 字典；
2. 从视频特征 key 中解析类别名，构建实际可用类别集合；
3. 读取 UCF101 split 文件；
4. 过滤掉 feature 文件中不存在的类别或样本；
5. 按官方 train/test split 形成训练集和测试集；
6. 从训练集中额外切出一部分 validation；
7. 同时构造 `full_train`，供 GAN 使用完整训练集训练。

### 3.2 样本格式

每个样本最终形如：

```text
(audio_feat, video_feat, len_audio, len_video, label)
```

其中：

- `audio_feat`：二维 tensor，形状约为 `[audio_len, 80]`；
- `video_feat`：二维 tensor，形状约为 `[video_len, 1280]`；
- `len_audio` / `len_video`：padding 前的真实长度；
- `label`：类别索引。

batch collate 时会按当前 batch 内最大长度 padding，并返回长度向量。GAN 训练和评估中都会使用长度 mask，避免 padding 部分污染 pooled feature 统计。

### 3.3 特征预处理对 GAN 的影响

音频特征有一个关键特点：真实音频在预处理时做了 per-sample / per-feature 的 z-score。结果是，在时间维度做 masked mean pooling 后，真实 audio pooled mean 接近 0。

因此当前生成器对 fake audio 也做了 `_apply_per_sample_znorm`，使 fake audio 与真实 audio 位于相同表征空间。这是当前版本的关键修复之一：如果不这样做，fake audio 在 pooled 空间会偏离真实分布很远。

## 4. 网络结构

核心结构定义在：

- `fed_multimodal/generator/gan_generator.py`

当前 GAN 是一个 conditional multimodal feature GAN，包含：

1. Audio Generator
2. Video Generator
3. Audio Discriminator
4. Video Discriminator
5. Joint Critic
6. 冻结的 teacher classifier

### 4.1 配置类

`FeatureGANConfig` 控制主要结构与训练超参数：

- `z_dim=128`
- `num_classes=51`
- `audio_seq_len=500`
- `audio_feat_dim=80`
- `video_seq_len=9`
- `video_feat_dim=1280`
- `hidden_dim=256`
- `lr_g=2e-4`
- `lr_d=1e-4`
- `rf_weight`：真假对抗损失权重
- `aux_weight`：辅助分类损失权重
- `cls_weight`：teacher 语义损失权重
- `joint_weight`：joint critic 损失权重
- `fm_weight`：feature matching 权重
- `mom_weight`：moment matching 权重
- `audio_out_max` / `video_out_max` 等输出范围约束

### 4.2 Audio Generator

`AudioFeatureGenerator` 输入：

- 随机噪声 `z`，维度 128；
- 类别标签 embedding。

结构：

1. label embedding，维度为类别数；
2. 拼接 `z` 和 label embedding；
3. 全连接投影到 `init_len * hidden_dim`；
4. LayerNorm + LeakyReLU；
5. 多层 ConvTranspose1d 上采样；
6. GroupNorm + LeakyReLU；
7. 最后一层线性输出到 80 维音频特征；
8. AdaptiveAvgPool1d 调整到目标长度；
9. 可学习 scale/bias 加输出 clamp；
10. 输出形状 `[B, T_audio, 80]`。

几个重要设计：

- 移除了 BatchNorm，改用 LayerNorm / GroupNorm，减少小 batch 不稳定；
- 最后一层移除了 Tanh，避免过强限制；
- fake audio 后续会做 per-sample z-normalization；
- 输出通过 `audio_scale_max`、`audio_bias_max`、`audio_out_max` 控制范围。

### 4.3 Video Generator

`VideoFeatureGenerator` 输入同样是 `z + label embedding`。

结构：

1. 全连接 512；
2. LayerNorm + LeakyReLU；
3. 全连接 1024；
4. LayerNorm + LeakyReLU；
5. 全连接 2048；
6. LayerNorm + LeakyReLU；
7. 全连接到 `video_seq_len * video_feat_dim`；
8. ReLU 保证视频特征非负；
9. 可学习 scale/bias；
10. clamp 到 `[0, video_out_max]`；
11. 输出形状 `[B, T_video, 1280]`。

视频特征来自 MobileNetV2，真实分布大多非负，因此这里使用 ReLU 和非负 clamp 是合理的。

### 4.4 单模态判别器

Audio / Video Discriminator 都是 ACGAN 风格，即同时输出：

- adversarial logit：判断 real / fake；
- auxiliary class logits：判断类别。

Audio Discriminator：

- Conv1d 输入通道为 80；
- 多层 spectral norm Conv1d；
- AdaptiveAvgPool1d；
- real/fake head；
- class head。

Video Discriminator：

- Conv1d 输入通道为 1280；
- spectral norm Conv1d；
- AdaptiveAvgPool1d；
- real/fake head；
- class head。

关键点：

- 判别器输出 logits，不直接用 Sigmoid；
- 损失使用 `BCEWithLogitsLoss`；
- spectral normalization 默认开启；
- 去掉 BatchNorm 和 Dropout，提高 adversarial 训练稳定性。

### 4.5 Joint Critic

`JointCritic` 用来判断一对 pooled audio/video 是否像真实同一样本的跨模态组合。

输入：

- masked mean pooling 后的 audio feature，维度 80；
- masked mean pooling 后的 video feature，维度 1280。

结构：

1. audio tower：LayerNorm -> Linear -> LeakyReLU -> Linear，投影到 `proj_dim`；
2. video tower：LayerNorm -> Linear -> LeakyReLU -> Linear，投影到 `proj_dim`；
3. 拼接 audio/video 投影；
4. spectral norm Linear -> LeakyReLU -> spectral norm Linear；
5. 输出 real/fake logit。

这样设计是为了解决音频 80 维、视频 1280 维之间的维度不平衡，避免 joint 判别完全被视频特征主导。

### 4.6 Teacher Classifier

GAN 训练时会复制并冻结一个已经训练好的 `MMActionClassifier`：

- 本地 teacher 默认来自 `results/local_training/best_model.pt`；
- teacher 不更新参数；
- 生成器训练时会把 fake audio/video 输入 teacher；
- teacher 预测类别需要和条件 label 一致。

teacher loss 的作用是让生成特征具有“分类语义”。如果没有 teacher loss，生成器可能只学到边缘统计，却不一定生成对应类别的可识别样本。已有消融中 `CLS0` 的 fake teacher accuracy 明显下降，说明 teacher 语义约束非常重要。

## 5. 损失函数与训练机制

每个 batch 的训练大致包括一个 G step 和一个 D step。

### 5.1 生成器损失

生成器总损失包括：

```text
G_loss =
  rf_weight    * adversarial_loss
+ aux_weight   * discriminator_aux_class_loss
+ cls_weight   * teacher_classification_loss
+ joint_weight * joint_critic_loss
+ fm_weight    * feature_matching_loss
+ mom_weight   * moment_matching_loss
+ audio_std_weight * audio_std_regularization
```

各项含义：

- `adversarial_loss`：骗过 audio/video 单模态判别器；
- `discriminator_aux_class_loss`：让单模态判别器的类别头也认为 fake 属于目标 label；
- `teacher_classification_loss`：让冻结 teacher 认为 fake 属于目标 label；
- `joint_critic_loss`：骗过 joint critic，使 fake audio/video 组合看起来像真实配对；
- `feature_matching_loss`：对齐判别器中间特征均值；
- `moment_matching_loss`：对齐判别器中间特征的均值和标准差；
- `audio_std_regularization`：可选，用于约束 fake audio pooled std。

### 5.2 判别器损失

判别器损失包括：

- real/fake adversarial loss；
- auxiliary class loss；
- joint critic real/fake loss。

真实标签使用 label smoothing：

- real label 默认 0.9；
- fake label 默认 0.1。

这有助于稳定 GAN 训练，避免判别器过早过强。

### 5.3 Joint Critic 强化训练

当前默认对 joint critic 做多步训练：

- `joint_d_steps=3`
- `joint_lr_mult=2.0`

也就是 joint critic 每个生成器 step 后训练多次，并使用更大学习率。这样做的动机是：跨模态关系比单模态边缘分布更难学，如果 joint critic 太弱，生成器可能只学会分别生成看起来合理的 audio/video，但二者组合不真实。

不过已有消融显示 joint gap 仍然不稳定，不同配置下 gap 可能为正也可能为负，说明 joint critic 的度量仍需谨慎解读。

## 6. 评估手段

当前评估主要在：

- `fed_multimodal/Local/eval_local_gan_quality.py`
- `fed_multimodal/generator/eval_gan_quality.py`

评估输出主要保存为：

- `analysis_results.json`
- `dist_audio.png`
- `dist_video.png`
- `tsne_audio.png`
- `tsne_video.png`
- `joint_gap_curve.png`

### 6.1 Teacher accuracy

`teacher_acc` 是最直接的语义指标。已有 `analysis_results.json` 中包含：

- `fake_fake`：teacher 在 fake audio + fake video 上的准确率；
- `real_real`：teacher 在 real audio + real video 上的准确率；
- `real_fake` / `fake_real`：混合真实与生成模态时的准确率。

这个指标回答：“生成样本是否能被下游分类器识别为指定类别？”

注意：teacher accuracy 高不代表生成分布真实，它只说明生成特征满足 teacher 的判别边界，可能存在“过度迎合 teacher”的情况。

### 6.2 边缘分布统计

音频和视频分别统计：

- mean
- std
- min
- max
- 每维均值/方差分布

音频还区分：

- masked temporal mean 的统计；
- masked temporal std 的统计。

这个指标回答：“fake 特征的数值范围和一阶/二阶统计是否接近 real？”

当前结果中，视频 mean/std 和真实值已经非常接近；音频经过 z-normalization 后，fake audio 的时序 std 也接近真实值。

### 6.3 Joint logit gap

Joint Critic 输出：

```text
joint_gap = real_logit_mean - fake_logit_mean
```

理想情况下：

- 如果 joint critic 能区分 real 和 fake，且 real logit 更高，则 gap 应为正；
- 如果 gap 绝对值接近 0，可能说明 fake 与 real 接近，也可能说明 critic 较弱；
- 如果 gap 为负，说明当前 joint critic 对 fake 给了更高 realness 分数，可能是训练不稳定或 critic 标定不可靠。

因此 joint gap 只能作为辅助诊断，不能单独代表生成质量。

### 6.4 多样性比

`diversity_ratio` 使用同类样本内部的 pairwise L2 距离来衡量：

```text
ratio = fake 类内平均距离 / real 类内平均距离
```

理想值接近 1。当前已有结果中：

- audio mean ratio 大多约 0.31～0.33；
- video mean ratio 大多约 0.22～0.24。

这说明 fake 样本类内多样性显著低于真实样本，存在明显 mode collapse 或生成分布过窄问题。

### 6.5 t-SNE 与分布图

评估脚本会生成：

- audio t-SNE；
- video t-SNE；
- audio distribution plots；
- video distribution plots。

这些图用于人工观察 real/fake 是否重叠、是否形成独立簇、是否类内分布过窄。它们不是严格指标，但对诊断 mode collapse 和分布偏移很有用。

### 6.6 FID / MMD / Domain Classifier

`eval_local_gan_quality.py` 还实现了：

- Fréchet distance；
- RBF MMD；
- real/fake domain classifier accuracy / AUC。

这些指标用于回答：“如果训练一个额外分类器区分 real 和 fake，是否容易区分？”

不过当前已保存的主要 `analysis_results.json` 里并不总是包含这些额外指标，可能是部分实验关闭了 extra metrics，或不同版本输出格式有所不同。

## 7. 已有训练效果

### 7.1 早期本地 GAN 结果

`fed_multimodal/Local/results/local_gan/local_gan_results.json` 中记录了一个 100 epoch 早期本地结果：

- teacher 模型测试准确率：约 75.62%；
- G loss：从 3.19 降到 2.27；
- D loss：从 2.90 降到 0.87；
- D accuracy：从 0.816 升到 1.0；
- generator quality：最高约 0.875；
- ML efficacy：约 0.084。

这个结果说明：

- 判别器最终几乎能完美区分真假，说明 fake 与 real 仍有明显差距；
- teacher 语义质量不差，但用于训练独立分类器的效果很弱；
- 早期版本的生成数据还不足以作为高质量替代训练集。

### 7.2 联邦版 GAN 结果

`fed_multimodal/result/gan_generator/ucf101/gan_generator_results.json` 中：

- final FL accuracy：约 77.52%；
- G loss：约 11.51 降到 10.63；
- D loss：约 17.78 降到 10.96；
- gen quality：0.95～1.0；
- D catches fake：最高接近 0.975；
- ML efficacy：约 0.319。

这个版本 teacher 语义一致性很强，ML efficacy 比早期本地结果好，但 D 仍能较明显地区分 fake。

`fed_multimodal/result/gan_attack_improved/ucf101/improved_gan_results.json` 中：

- warmup accuracy：约 64.56%；
- final accuracy：约 74.85%；
- gen quality：0.975～1.0；
- D catches fake：长期为 1.0；
- ML efficacy：约 0.169。

这说明该攻击/改进版本生成的样本非常符合 teacher 语义，但真实性不足，判别器非常容易抓到 fake。

### 7.3 当前 200 epoch 本地分析结果

当前 `fed_multimodal/Local/results/gan_analysis/` 下有多组 200 epoch 消融实验，例如：

- `ckpt_200_BASE`
- `ckpt_200_0309BASE`
- `ckpt_200_BASE_NO_FM_MOM`
- `ckpt_200_CLS0`
- `ckpt_200_CLS_DELAY`
- `ckpt_200_CLS_LOW`
- `ckpt_200_FM_MOM_0.1`
- `ckpt_200_JOINT0`
- `ckpt_200_JOINT_HIGH`
- `ckpt_200_OUTFIX`

以 `ckpt_200_BASE` 为例：

- 评估样本数：1600；
- fake_fake teacher accuracy：0.999375；
- real_real teacher accuracy：0.729375；
- real_fake teacher accuracy：0.999375；
- fake_real teacher accuracy：0.73625；
- video real mean/std：0.4617 / 0.4261；
- video fake mean/std：0.4528 / 0.4188；
- audio real temporal std mean：0.9907；
- audio fake temporal std mean：0.9998；
- audio diversity ratio：约 0.322；
- video diversity ratio：约 0.218；
- joint gap：约 -82.04。

这些数值说明：

1. **语义一致性非常强**：fake+fake 几乎 100% 被 teacher 识别为条件类别。甚至 real audio + fake video 也接近 100%，说明 fake video 可能非常贴合 teacher 的类别判别特征。
2. **边缘统计对齐较好**：video mean/std 已接近真实；audio temporal std 也接近真实。
3. **多样性不足明显**：audio/video diversity ratio 远低于 1，说明生成样本类内变化范围只有真实数据的大约 20%～33%。
4. **joint critic 解释不稳定**：BASE 的 joint gap 为负，`0309BASE` 为正，`FM_MOM_0.1` 接近 0 但仍为负。这说明 joint critic 既是训练约束，也是诊断工具，但当前不能作为单一可靠质量指标。

### 7.4 消融观察

#### Teacher loss 的作用

`ckpt_200_CLS0` 关闭 teacher loss 后：

- fake_fake teacher accuracy 降到约 0.814；
- 其他配置大多接近 1.0。

这说明 teacher loss 对生成类别语义非常关键。没有 teacher loss，生成器虽然仍有 aux 分类头约束，但语义可识别性明显下降。

#### Joint critic 的作用

`ckpt_200_JOINT0` 关闭 joint 后：

- fake_fake teacher accuracy 仍可达到 1.0；
- video diversity ratio 约 0.239，略高于 BASE；
- joint gap 很负，说明独立训练后的 joint 指标不理想。

这表明单靠单模态判别器和 teacher 也能生成可分类特征，但跨模态真实配对关系未必学得好。Joint critic 的必要性主要体现在跨模态一致性，而不是 teacher accuracy。

#### FM/MoM 的作用

`ckpt_200_FM_MOM_0.1`：

- fake_fake teacher accuracy 为 1.0；
- joint gap 约 -1.85，是所有列出的配置里较接近 0 的一个；
- audio/video diversity ratio 仍只有约 0.314 / 0.226。

这说明 feature matching 和 moment matching 对稳定统计分布、缓和 joint gap 有帮助，但尚未解决多样性不足。

#### 输出范围修复

`ckpt_200_OUTFIX`：

- video fake mean/std：0.4674 / 0.4279，和真实 0.4617 / 0.4261 非常接近；
- audio fake temporal std mean：0.9999；
- fake_fake teacher accuracy 为 1.0；
- diversity ratio 仍偏低。

输出范围约束对边缘分布对齐很有帮助，但不会自动提升类内多样性。

## 8. 当前 GAN 的作用

当前 GAN 在项目中主要有四类作用。

### 8.1 生成合成多模态特征

它可以按类别生成 audio/video feature pair，用作 synthetic dataset。由于生成的是 feature 而不是原始视频，成本低、速度快，也绕开了原始多媒体生成的复杂性。

### 8.2 支持 fake training

`run_gan_experiment.sh` 支持 `fake_train` 模式，即用 GAN 生成的特征训练分类器。相关结果输出在：

- `fed_multimodal/Local/results/fake_training/`

这可以验证生成数据是否具有训练价值。如果只在 teacher 上 accuracy 高，但用来训练新分类器效果差，就说明 GAN 更像是在拟合 teacher 决策边界，而不是真正覆盖真实数据分布。

### 8.3 支持 fake attack / 标签翻转攻击

`run_gan_experiment.sh` 支持 `fake_attack` 模式，可以生成指定类别或被翻转标签的 fake 特征，用于研究数据投毒或标签翻转攻击对模型的影响。

仓库中也有攻击相关代码和结果，例如：

- `fed_multimodal/generator/`
- `fed_multimodal/demo/attack/`
- `fed_multimodal/results/demo/ucf101/attack/`

这部分用途偏安全/鲁棒性研究：模拟恶意客户端或污染样本对多模态联邦学习的影响。

### 8.4 诊断多模态模型依赖

通过混合 real/fake 模态，例如：

- real audio + fake video；
- fake audio + real video；
- fake audio + fake video；

可以观察 teacher 更依赖哪个模态，或者生成的哪个模态更强。从已有 `ckpt_200_BASE` 看，real audio + fake video 的准确率接近 1.0，而 fake audio + real video 约 0.736，暗示 fake video 对 teacher 的类别判别贡献很强，甚至可能比 fake audio 更直接贴合分类边界。

## 9. 当前问题与风险

### 9.1 多样性不足

最明显问题是 diversity ratio 偏低：

- audio 多数配置约 0.31；
- video 多数配置约 0.22。

这表示同一类别内 fake 样本变化范围远小于 real。实际风险是：

- 用 fake 数据训练模型时泛化能力差；
- 攻击评估可能过于集中在少数模式；
- teacher accuracy 虽高，但样本缺乏真实分布覆盖。

### 9.2 Teacher 过拟合风险

fake_fake teacher accuracy 接近 100%，高于 real_real 约 72.9%。这不是单纯的好事，可能说明生成器学到了 teacher 喜欢的判别特征，而不是完整真实分布。

这类似“生成对 teacher 来说非常典型的类别原型”，但这些样本不一定像真实数据。

### 9.3 Joint 指标不稳定

不同 checkpoint 中 joint gap 差异很大：

- `0309BASE`：约 +27.18；
- `BASE`：约 -82.04；
- `FM_MOM_0.1`：约 -1.85；
- `JOINT0`：约 -102.47。

这说明 joint critic 的训练和标定仍不稳定。它可以作为相对消融参考，但不能简单解释为“gap 越大/越小就一定越好”。如果要把 joint gap 作为核心指标，建议使用独立 oracle joint critic 重新训练评估，避免训练中 critic 自身状态带来的偏差。

### 9.4 判别器过强问题

早期结果中 D accuracy 最终接近 1.0，说明判别器能够轻易区分 fake。当前版本虽然边缘统计改善很多，但如果 domain classifier/FID/MMD 指标仍显示可分，则说明 fake 与 real 仍有可检测差异。

## 10. 建议的后续改进方向

### 10.1 优先解决多样性

当前最需要优化的是生成多样性。可以考虑：

- 增强 latent z 对输出的影响，避免 label 主导后变成类别原型生成；
- 加 minibatch discrimination 或 batch diversity regularization；
- 增加 mode-seeking loss，例如鼓励不同 z 产生不同输出；
- 对同类样本引入 intra-class variance matching；
- 降低 teacher loss 或改成 margin-based teacher loss，避免生成器只追求 teacher 高置信度。

### 10.2 重新设计语义指标

teacher accuracy 已接近饱和，区分不了更好的模型。建议补充：

- teacher confidence 分布，而不仅是 accuracy；
- real/fake 在 teacher embedding 空间的 FID/MMD；
- 新训练 student 在 fake 数据上的训练效果；
- fake-to-real retrieval 或 nearest-neighbor distance，检查是否生成过窄。

### 10.3 强化独立评估

建议每个 checkpoint 都统一输出：

- FID audio/video；
- MMD audio/video；
- domain classifier acc/AUC；
- oracle joint critic gap；
- diversity ratio；
- fake training downstream accuracy。

当前不同结果文件字段不完全一致，不利于横向比较。

### 10.4 谨慎使用 GAN 结果作为结论

如果论文或报告中要使用当前 GAN 结果，建议表述为：

- “GAN 能生成具有强 teacher 语义一致性的多模态特征”；
- “边缘统计已经能较好匹配真实 MFCC/MobileNetV2 特征”；
- “但类内多样性和跨模态联合真实性仍是主要瓶颈”；
- “因此当前 GAN 更适合用于攻击/鲁棒性诊断和 controlled synthetic feature experiments，而不是作为完全替代真实数据的数据增强方法”。

## 11. 文件索引

核心文件：

- `fed_multimodal/generator/gan_generator.py`：GAN 网络结构和训练 step。
- `fed_multimodal/Local/train_local_gan.py`：本地 GAN 训练入口。
- `fed_multimodal/Local/eval_local_gan_quality.py`：本地 GAN 评估入口。
- `fed_multimodal/Local/dataloader.py`：UCF101 本地 feature 数据加载。
- `fed_multimodal/Local/run_gan_experiment.sh`：训练、评估、fake_train、fake_attack 自动化脚本。
- `fed_multimodal/Local/run_minimal_matrix.sh`：Joint Critic 消融实验脚本。
- `fed_multimodal/experiment/ucf101/train_gan_generator.py`：联邦版 GAN 训练入口。
- `fed_multimodal/generator/eval_gan_quality.py`：通用 GAN 质量评估工具。

主要结果：

- `fed_multimodal/Local/results/local_gan/`：本地 GAN checkpoint 和早期结果。
- `fed_multimodal/Local/results/gan_analysis/`：本地 GAN 评估结果、t-SNE、分布图、joint gap 曲线。
- `fed_multimodal/Local/results/fake_training/`：使用 GAN fake feature 训练分类器的结果。
- `fed_multimodal/result/gan_generator/ucf101/`：联邦版 GAN 结果。
- `fed_multimodal/result/gan_attack_improved/ucf101/`：改进攻击版 GAN 结果。

## 12. 最终判断

当前 GAN 的训练链路已经比较完整：有明确的数据加载、条件生成器、单模态判别器、joint critic、teacher 语义约束、自动训练脚本和多维度评估脚本。它的主要成果是能稳定生成 teacher 高度认可的 UCF101 多模态特征，并且在音频/视频边缘统计上接近真实特征。

但从已有结果看，生成质量还不能简单视为“逼真数据生成”。最大短板是类内多样性不足，且 joint critic 指标波动较大。当前 GAN 更适合作为多模态联邦学习中的合成特征生成器、攻击样本生成器和鲁棒性分析工具；如果目标是高质量数据增强或替代真实训练数据，还需要进一步提升多样性、统一独立评估指标，并验证 fake 数据训练出的 student 是否能在真实测试集上取得接近真实训练的效果。
