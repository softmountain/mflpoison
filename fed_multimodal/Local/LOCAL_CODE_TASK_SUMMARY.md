# Local 目录代码任务总结

## 1. 先说结论

`fed_multimodal/Local` 不是一个单独的模型文件夹，而是一套围绕 **UCF101 音频-视频特征** 做的“本地集中式实验流水线”。

这套代码的核心任务可以概括为四件事：

1. 用完整的本地数据训练一个多模态动作分类器，作为非联邦场景下的上界基线和 Teacher。
2. 在这个 Teacher 的约束下训练一个 **特征级** 多模态 GAN，生成假的音频特征和视频特征。
3. 评估这些假特征是否像真的特征，是否保留了正确类别语义，是否具备跨模态一致性。
4. 把 GAN 生成的假特征再拿去训练或攻击分类器，验证这些假样本是否真的“有用”。

换句话说，这个目录做的不是原始视频生成，也不是图像 GAN，而是：

**在预提取好的音频/视频特征空间里，构建一个本地实验平台，用来研究多模态分类、特征生成、生成质量评估，以及基于假特征的训练/攻击。**

---

## 2. Local 目录在整个项目里的角色

从代码依赖看，`Local` 目录本身主要负责“实验流程编排”，而不是定义所有底层模型：

- 分类器结构来自 `fed_multimodal/model/mm_models.py` 中的 `MMActionClassifier`
- GAN 结构来自 `fed_multimodal/generator/gan_generator.py` 中的 `MultimodalFeatureGAN`
- `Local` 目录负责把数据、分类器、GAN、评估和实验脚本串成一个完整的本地实验闭环

所以它更像是一个 **centralized baseline + GAN sandbox + fake-feature attack sandbox**。

---

## 3. 它处理的数据是什么

这个目录假设输入已经不是原始媒体，而是预先抽好的特征：

- 音频特征：默认 `mfcc`
- 视频特征：默认 `mobilenet_v2`
- 数据集：`UCF101`
- 数据来源路径：
  - `results/feature/audio/mfcc/ucf101/feature.pkl`
  - `results/feature/video/mobilenet_v2/ucf101/feature.pkl`
- 划分方式：使用官方 `ucfTrainTestlist` 中的 train/test split

在 `Local/dataloader.py` 里可以看出：

- 音频默认长度是 `500 x 80`
- 视频默认长度是 `9 x 1280`
- 实验只使用特征文件里实际存在的类别，因此这里不是完整的 101 类，而是一个可用子集，代码里通常是 `51` 类

这意味着 `Local` 做的事情本质上是：

**在 UCF101 的音频特征和视频特征上做多模态动作识别与特征生成研究。**

---

## 4. 整体执行链路

这个目录最核心的执行链路如下：

### 第一步：加载本地多模态特征

`dataloader.py` 负责：

- 从 `feature.pkl` 加载音频和视频特征字典
- 根据 UCF101 官方 split 生成 train/test 列表
- 过滤掉特征文件中不存在的样本或类别
- 对变长序列做 padding
- 返回：
  - `train`
  - `val`
  - `test`
  - `full_train`

其中 `full_train` 很关键，因为后续 GAN 训练通常不使用 train/val 拆分后的子集，而是直接用完整训练集来学特征分布。

### 第二步：训练本地分类器

`train_local.py` 负责：

- 构建 `MMActionClassifier`
- 用真实音频特征和真实视频特征做集中式监督训练
- 在 `val/test` 上持续评估
- 保存：
  - 最优模型 `best_model.pt`
  - 最终模型 `final_model.pt`
  - 训练历史 `training_history.json`
  - 汇总信息 `summary.json`

这一步的意义有两个：

- 给项目提供“非联邦条件下能做到多好”的上界
- 给后面的 GAN 提供一个 Teacher 分类器

### 第三步：训练本地多模态特征 GAN

`train_local_gan.py` 负责：

- 读取第二步训练好的分类器 checkpoint
- 用它恢复 Teacher 模型
- 基于 `MultimodalFeatureGAN` 构建一个特征级生成器
- 在 `full_train` 上学习生成假的音频特征和视频特征

这个 GAN 不是普通的单判别器结构，而是：

- 一个音频生成器
- 一个视频生成器
- 一个音频判别器
- 一个视频判别器
- 一个联合判别器 `Joint Critic`

训练目标也不是单一 adversarial loss，而是多种约束同时存在：

- real/fake 对抗损失
- auxiliary 分类损失
- Teacher 分类语义约束
- joint 对齐约束
- 可选的 feature matching / moment matching

并且代码里还有两个稳定训练的重要机制：

- `warmup`: 前期先不让 Teacher / Joint 约束过早介入
- `ramp`: 之后逐步增加 `cls_weight` 和 `joint_weight`

最终输出是 `results/local_gan/ckpt_*.pt` 这样的 GAN checkpoint。

### 第四步：评估 GAN 生成质量

`eval_local_gan_quality.py` 负责：

- 读取 GAN checkpoint
- 读取 Teacher checkpoint
- 在 test 集或 train 集上生成 fake 特征
- 对 fake 特征做多角度质量评估

它评估的不是“看起来像不像图片”，而是特征层面的几类指标：

- Teacher 在 fake+fake 上的分类正确率
- Teacher 在 real+real、real+fake、fake+real 上的表现
- real/fake 的边缘统计是否接近
- Joint Critic 对 real/fake 的 logit gap
- 类内多样性比，检查 mode collapse
- 可选：
  - FID
  - MMD
  - domain classifier 指标
  - t-SNE 可视化

输出是：

- `analysis_results.json`
- 分布图
- joint gap 曲线图
- 可选的 t-SNE 图

### 第五步：用 fake 特征继续训练或攻击分类器

`train_with_fake.py` 负责：

- 读取已训练好的 GAN
- 直接按标签采样 fake audio / fake video
- 用这些假特征训练一个新的分类器，或者微调已有分类器

这一步的目标不是生成更好的 GAN，而是回答两个问题：

1. GAN 生成的特征能不能支撑分类训练？
2. 如果对 fake 样本做标签翻转，能不能构成有效攻击？

这里支持一种很明确的攻击方式：

- 特征仍然按原始标签生成
- 但训练监督标签会按指定概率从 `src_label` 翻到 `dst_label`

这实际上是在模拟一种 **label-flip based fake-feature attack**。

---

## 5. 各代码文件分别在做什么

下面按“主流程文件 / 诊断调试文件 / 自动化脚本”来说明。

### 5.1 `dataloader.py`

这是整个 `Local` 目录的基础输入层，主要任务是把磁盘上的特征组织成可训练的 PyTorch DataLoader。

它做了几件关键的事：

- `pad_tensor`
  - 把每条样本补到当前 batch 内的最大长度
- `collate_mm_fn_padd`
  - 把一个 batch 组织成：
    - `x_audio`
    - `x_video`
    - `len_a`
    - `len_v`
    - `label`
- `UCF101LocalDataset`
  - 返回单样本的音频特征、视频特征、长度和标签
  - 如果特征缺失，会用全零张量占位
- `UCF101LocalDataManager`
  - 读取特征文件
  - 从视频 key 中提取类别名
  - 基于实际可用特征建立 `class_to_idx`
  - 按官方 split 过滤并组织 train/test
  - 再从 train 中切出 val
  - 额外提供 `full_train`

所以它的实际任务不是简单“读文件”，而是：

**把原始离散的特征字典和官方 split 文件，转换成后续分类训练、GAN 训练和评估都能统一复用的数据接口。**

### 5.2 `train_local.py`

这是本地监督训练脚本。

它的主要执行过程是：

- 解析训练参数
- 初始化随机种子和设备
- 调用 `UCF101LocalDataManager`
- 构建 `MMActionClassifier`
- 用交叉熵损失做多分类训练
- 每个 epoch 在 `train / val / test` 三个集合上记录：
  - loss
  - accuracy
  - UAR
  - macro F1
- 保存 best checkpoint 和 final checkpoint

这里训练的模型是一个典型的音频-视频融合分类器：

- 音频分支先过 `Conv1dEncoder`，再过 GRU
- 视频分支过 GRU
- 可选 attention
- 两个模态的表示拼接后做分类

因此 `train_local.py` 承担的是：

**建立本地集中式多模态动作识别基线，并输出一个可以当 Teacher 使用的高质量分类器。**

### 5.3 `train_local_gan.py`

这是本地特征 GAN 的训练入口。

它做的事情可以概括成：

- 加载本地 Teacher 分类器
- 从 dataloader 中取一个 batch 推断真实特征维度
- 构建 `FeatureGANConfig`
- 初始化 `MultimodalFeatureGAN`
- 在 `full_train` 上循环训练
- 根据 warmup/ramp 动态调节：
  - `cls_weight`
  - `joint_weight`
- 按配置保存中间或最终 checkpoint

从依赖的 `gan_generator.py` 看，实际训练内容包括：

- 生成音频 fake 特征
- 生成视频 fake 特征
- 用独立判别器区分每个模态的 real/fake
- 用 `Joint Critic` 判断音频池化特征和视频池化特征的联合一致性
- 用 Teacher 检查生成样本是否还保留目标语义

所以这个脚本的真正任务是：

**在真实多模态特征空间里学一个“按类别条件生成音频特征和视频特征”的生成模型，并让生成样本既像真实分布，又能保留跨模态语义。**

### 5.4 `eval_local_gan_quality.py`

这是 GAN 的核心评估脚本。

它不是只给一个简单准确率，而是系统地回答“fake 特征到底好不好”。

它的评估逻辑主要包括：

- 生成 fake audio / fake video
- 让 Teacher 分别看四种组合：
  - `real_real`
  - `fake_fake`
  - `real_fake`
  - `fake_real`
- 比较 real/fake 的：
  - 均值
  - 标准差
  - 极值范围
- 看 joint discriminator 对 real 和 fake 的平均 logit 差值
- 计算类内多样性比例
- 视情况计算 FID、MMD、domain classifier
- 输出图像可视化结果

因此它承担的是：

**从语义一致性、边缘分布相似性、跨模态耦合关系和多样性几个角度，对本地 GAN 的特征生成质量做完整体检。**

### 5.5 `train_with_fake.py`

这是“假特征能不能用”的验证脚本。

它和 `train_local.py` 的区别在于：

- `train_local.py` 用的是真实特征
- `train_with_fake.py` 用的是 GAN 即时生成的 fake 特征

它的主要流程是：

- 读取 GAN checkpoint 和可选的真实分类器 checkpoint
- 构建 Teacher 和 GAN
- 每个 batch 不取真实特征内容，只取真实标签和长度
- 根据标签采样 fake audio / fake video
- 用 fake 特征训练分类器
- 再在真实的 val/test 上评估

如果开启攻击模式，它还会：

- 在训练时对某个类别的监督标签做概率性翻转
- 但生成特征本身仍按原标签条件生成

这等于测试：

- fake 样本能不能支持正常训练
- fake 样本是否能承载投毒式监督攻击

所以这个脚本的角色是：

**把 GAN 从“可视化上看起来还行”推进到“对下游分类任务是否真的有影响”的验证层。**

### 5.6 `diagnose_audio_features.py`

这是一个针对 audio 分支的诊断工具，不是常规训练主流程的一部分。

它的目的非常明确：

**快速定位 audio GAN 为什么会崩，尤其是为什么 fake audio 的统计特性和 real audio 对不上。**

它做的诊断包括：

- 分析真实音频特征每一维的均值、方差、极值
- 找接近常量、接近二值、疑似 bimodal 的维度
- 模拟 per-sample Z-norm 后的统计变化
- 分析 pooled feature 的分布
- 读取 `AudioFeatureGenerator` 的输出层配置，检查：
  - scale
  - bias
  - clamp 范围
- 判断是否需要把音频特征拆成离散段和连续段分别建模

它本质上是在做：

**数据分布诊断 + 生成器输出约束诊断 + 架构改造建议。**

### 5.7 `debug/debug_joint_critic.py`

这是针对 Joint Critic 的排障脚本。

它内置了两种实验：

- `R0-1`
  - 冻结生成器
  - 只训练 Joint Critic
  - 观察它能不能把 real/fake 分开
- `R0-2`
  - 强化 Joint Critic 训练强度
  - 例如增加每轮 D step，或提高 joint 学习率

它要验证的是：

- joint 机制本身是不是坏的
- 是不是 loss 符号、label、学习率或训练强度出了问题
- 为什么 joint gap 会变负或不稳定

所以这个脚本是一个：

**面向联合判别器机制的专项排雷工具。**

---

## 6. 自动化 shell 脚本的任务

### 6.1 `run_gan_experiment.sh`

这是总控脚本，负责统一启动三类模式：

- `gan`
  - 训练 GAN
  - 然后自动跑 `eval_local_gan_quality.py`
- `fake_train`
  - 用 fake 特征训练分类器
- `fake_attack`
  - 用 fake 特征执行标签翻转攻击训练

它的作用是把多步 Python 流程包成一个统一入口，并把日志、checkpoint、分析目录都整理好。

### 6.2 `run_gan_experiments_batch.sh`

这是批量组件开关实验脚本。

它会围绕一个 base template，一次只改一个因素，例如：

- 关掉 FM/MoM
- 降低或关闭 Teacher 约束
- 调大或关闭 Joint 约束
- 修改输出范围约束

它的任务是：

**做可控消融，确认每个模块到底有没有贡献。**

### 6.3 `run_joint_experiments.sh`

这是围绕 Joint Critic 的专题实验脚本。

它聚焦的问题不是“整体 GAN 好不好”，而是：

- joint 约束会不会破坏 audio 边缘分布
- 新 joint 结构在不同权重下是否更安全

它本质上是：

**一个针对 joint 机制的实验设计脚本。**

### 6.4 `run_minimal_matrix.sh`

这是更小规模、更聚焦的 joint 消融矩阵。

它通常固定 seed 和 epoch，只比较少数几个关键配置，例如：

- joint=0 的基线
- 强化 D_joint 的方案
- 更小 joint 权重的稳健方案

目的是快速确认 Joint Critic 调整方向是否有效。

### 6.5 `test_automation.sh`

这是最轻量的 smoke test。

它不会追求结果质量，只是用较小 epoch 跑一遍 `run_gan_experiment.sh`，检查自动化流程是否能从头到尾打通。

---

## 7. 从“任务目标”角度重新理解这个目录

如果不按文件看，而按研究任务看，`Local` 目录其实在做三层事情。

### 第一层：建立本地上界

对应文件：

- `dataloader.py`
- `train_local.py`

目标：

- 在非联邦条件下训练出一个可靠的多模态分类器
- 提供性能上界
- 为 GAN 提供 Teacher

### 第二层：学习和评估特征级生成

对应文件：

- `train_local_gan.py`
- `eval_local_gan_quality.py`
- `diagnose_audio_features.py`
- `debug/debug_joint_critic.py`

目标：

- 让模型能按类别生成音频和视频特征
- 检查生成样本是否语义正确、分布合理、跨模态对齐
- 在生成失败时定位是 audio、joint 还是训练策略出了问题

### 第三层：把 fake 特征用于下游任务验证

对应文件：

- `train_with_fake.py`

目标：

- 验证 fake 特征能否支撑分类训练
- 验证 fake 特征能否承载数据投毒/标签翻转类攻击

---

## 8. 一句话总结每个主文件

- `dataloader.py`: 把 UCF101 的音频/视频特征和官方 split 组织成可复用的本地 DataLoader。
- `train_local.py`: 用真实特征训练本地多模态分类器，得到上界基线和 Teacher。
- `train_local_gan.py`: 用 Teacher 约束训练特征级多模态 GAN，生成 fake audio/video 特征。
- `eval_local_gan_quality.py`: 从语义、分布、joint 对齐和多样性几个角度评估 GAN 质量。
- `train_with_fake.py`: 用 GAN 生成的假特征训练或攻击分类器，验证 fake 样本的下游影响。
- `diagnose_audio_features.py`: 专门诊断 audio 特征分布与生成器约束是否匹配。
- `debug/debug_joint_critic.py`: 专门验证和排查 Joint Critic 是否真正起作用。

---

## 9. 最后给出一句总判断

`Local` 目录执行的不是“单个模型训练任务”，而是一个完整的本地实验系统：

**它以 UCF101 的多模态特征为对象，先训练本地分类器，再训练特征级 GAN，再评估 GAN 质量，最后检验 fake 特征在分类训练和攻击中的实际效果。**

如果把整个目录只压缩成一句更短的话，那就是：

**这是项目里用来研究“真实多模态特征 -> 生成假特征 -> 用假特征影响下游分类器”这一整条链路的本地实验工作台。**
