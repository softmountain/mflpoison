下面给你一个**最可行、最容易直接落地的 synthetic-feature GAN 方案**。核心设定是：**把全局模型输出改成 `K+1` 类，其中前 `K` 类是 UCF101 原类别，第 `K` 类是 synthetic/fake 类**。这样可以用全局模型副本作为生成器质量判别器，推动 G 生成接近真实多模态特征分布的合成样本。

这个方案的核心目标是：

```text
G 生成目标类别 y_target 的 audio/video feature；
D 是当前通信轮的全局模型副本；
D 输出 K+1 类；
D 学习 real UCF101 类别 vs synthetic/fake 类；
G 的目标不是破坏主任务，而是生成真假难辨、可带真实标签参与训练的合成样本；
如果 synthetic-only 数据也能训练出真实测试集表现不错的模型，说明生成器已经具备偷渡式数据替代能力。
```

你之前的报告显示，旧 GAN 已经能让 teacher 几乎 100% 识别 fake，但最大问题是**类内多样性很差**，audio/video diversity ratio 只有约 0.31 / 0.22，而且 joint critic 指标不稳定。 所以这个新方案不能只优化“骗过全局模型”，还必须加 **diversity loss、feature matching、class-wise variance matching**。PoisonGAN 类方法本身也常被理解为用全局模型初始化判别器来生成质量较高的样本。([arXiv][1])

# 方案名称

```text
Fed-PoisonGAN-K+1
```

适用于：

```text
UCF101 联邦学习
多模态 feature-level 输入
audio: MFCC [T_audio, 80]
video: MobileNetV2 feature [T_video, 1280]
全局模型: MMActionClassifier
输出: K+1 类，其中 K 是 fake/poison 类
```

---

# 一、全局模型改造

## 1. 原模型输出

原来：

```text
GlobalModel(x_audio, x_video) -> logits [B, K]
```

改成：

```text
GlobalModel_Kplus1(x_audio, x_video) -> logits [B, K+1]
```

其中：

```text
0 ~ K-1: UCF101 原始类别
K: fake / poison 类
```

训练和评估时：

```text
正常分类准确率只看 logits[:, :K]
fake 判别看 logits[:, K]
```

## 2. 初始化方式

如果你已经有原来的 `K` 类模型 checkpoint：

```python
old_fc.weight: [K, d]
old_fc.bias:   [K]

new_fc.weight: [K+1, d]
new_fc.bias:   [K+1]
```

初始化：

```python
new_fc.weight[:K] = old_fc.weight
new_fc.bias[:K] = old_fc.bias

new_fc.weight[K] = torch.randn_like(old_fc.weight[0]) * 0.01
new_fc.bias[K] = 0.0
```

更稳一点的 fake 类初始化：

```python
new_fc.weight[K] = old_fc.weight.mean(dim=0) + 0.01 * torch.randn_like(old_fc.weight[0])
new_fc.bias[K] = old_fc.bias.mean()
```

这样 fake 类不是完全随机离群，训练初期会更稳定。

---

# 二、判别器 D 设计

在第 `t` 轮通信中：

```text
D_t = 当前全局模型 GlobalModel_t 的本地副本
```

也就是说：

```text
D_t(x) -> logits_{K+1}
```

判别器没有额外结构，直接使用全局模型：

```python
class PoisonDiscriminator(nn.Module):
    def __init__(self, global_model_kplus1):
        super().__init__()
        self.model = global_model_kplus1

    def forward(self, audio, video, len_audio=None, len_video=None, return_embed=False):
        logits, emb = self.model(
            audio,
            video,
            len_audio=len_audio,
            len_video=len_video,
            return_embed=True
        )
        if return_embed:
            return logits, emb
        return logits
```

如果当前 `MMActionClassifier` 还不能返回 embedding，建议改成：

```python
return logits, fused_feature
```

其中 `fused_feature` 是 classifier 前一层的多模态融合表示。

---

# 三、生成器 G 设计

生成器仍然是 feature generator，不生成原始视频/音频。

## 输入

```text
z: 随机噪声 [B, z_dim]
y_target: 目标类别 [B]
```

可选：

```text
source_label: 源类别
attack_code: 攻击强度 / 触发条件
```

第一版不建议加太复杂，先用：

```text
G(z, y_target) -> fake_audio, fake_video
```

## 输出

```text
fake_audio: [B, T_audio, 80]
fake_video: [B, T_video, 1280]
```

其中：

```text
T_audio = 500
T_video = 9
```

---

# 四、生成器结构建议

你可以在旧 GAN 的基础上改，不需要重写所有东西。

## 1. Shared latent trunk

```text
z + label_embedding -> shared_hidden
```

建议：

```text
z_dim = 256
label_emb_dim = 128
hidden_dim = 512
```

结构：

```python
h = concat(z, label_emb)
h = Linear -> LayerNorm -> LeakyReLU
h = Linear -> LayerNorm -> LeakyReLU
h_shared = h
```

## 2. 关键改进：FiLM 条件注入

旧 GAN 很可能 label embedding 压过 z，导致每类只生成少数模板。新方案里，label 不只在输入拼接，而是在 decoder 每层调制：

```python
gamma, beta = label_mlp(y)
h = gamma * h + beta
```

这样：

```text
label 控制类别语义；
z 控制类内变化。
```

## 3. Audio branch

```text
h_shared -> temporal decoder -> [B, 500, 80]
```

建议：

```python
Linear(h_shared -> 64 * 256)
reshape [B, 256, 64]
ConvTranspose1d blocks
AdaptiveAvgPool1d(500)
Linear/Conv1d -> 80
per-sample z-normalization
```

最后必须保留旧报告里提到的 audio z-normalization，因为真实音频 MFCC 特征经过 per-sample/per-feature z-score，fake audio 如果不做会偏离真实空间。

```python
fake_audio = apply_per_sample_znorm(fake_audio)
```

## 4. Video branch

```text
h_shared -> MLP decoder -> [B, 9, 1280]
```

建议：

```python
Linear -> LayerNorm -> LeakyReLU
Linear -> LayerNorm -> LeakyReLU
Linear -> 9 * 1280
reshape [B, 9, 1280]
ReLU
clamp [0, video_out_max]
```

因为 MobileNetV2 video feature 多数非负，ReLU + clamp 是合理的。

---

# 五、判别器损失

因为你采用 `K+1` 输出，判别器训练非常简单。

## 1. Real 样本

真实样本标签是原始类别：

```text
y_real in [0, K-1]
```

判别器希望：

```text
D(real) -> y_real
```

损失：

```python
loss_D_real = CE(logits_real, y_real)
```

注意，`CE` 是对 `K+1` 个 logit 做的。这样真实样本不仅会被推到正确类别，也会自动把 fake 类 logit 压低。

## 2. Fake 样本

生成样本标签设成：

```text
y_fake = K
```

判别器希望：

```text
D(fake.detach()) -> fake class K
```

损失：

```python
fake_label = torch.full_like(y_target, fill_value=K)
loss_D_fake = CE(logits_fake_detached, fake_label)
```

## 3. 判别器总损失

```python
loss_D = loss_D_real + lambda_fake * loss_D_fake
```

推荐：

```text
lambda_fake = 0.5 ~ 1.0
```

第一版用：

```text
lambda_fake = 1.0
```

---

# 六、生成器损失

生成器希望 fake 不被判成 fake，而是被判成目标类别：

```text
D(G(z, y_target)) -> y_target
```

但只做这个会导致旧问题：fake 很容易变成 target 类原型，diversity 很低。你之前的报告里已经出现过 teacher accuracy 近乎 1.0 但生成分布过窄的问题。

所以生成器损失设计为：

```text
L_G =
  λ_adv  * L_adv_target
+ λ_avoid * L_avoid_fake
+ λ_fm   * L_feature_matching
+ λ_var  * L_class_variance
+ λ_div  * L_diversity
+ λ_mod  * L_cross_modal
+ λ_stat * L_stat
```

## 1. 目标类别攻击损失

```python
loss_G_target = CE(logits_fake, y_target)
```

让 fake 被 D 判成目标类别。

推荐：

```text
λ_adv = 1.0
```

## 2. 避免 fake 类损失

虽然 `CE(logits_fake, y_target)` 已经会压低 fake 类，但可以显式加一个 fake-logit penalty：

```python
prob_fake_class = softmax(logits_fake, dim=1)[:, K]
loss_avoid_fake = -torch.log(1.0 - prob_fake_class + 1e-6).mean()
```

或更简单：

```python
loss_avoid_fake = prob_fake_class.mean()
```

推荐：

```text
λ_avoid = 0.2
```

这个损失的作用是：即使目标类别不够强，也先避免被判成 fake。

## 3. Feature matching loss

用全局模型 embedding 对齐 real/fake。

对每个 target 类，取本地真实样本中对应类别的 embedding 均值：

```python
real_emb_y = D(real_y, return_embed=True)
fake_emb_y = D(fake_y, return_embed=True)

loss_fm = || mean(fake_emb_y) - mean(real_emb_y) ||_2
```

如果 batch 内某个类别没有真实样本，可以用 memory bank：

```text
class_embedding_bank[y] = 最近若干轮真实样本 embedding 均值
```

推荐：

```text
λ_fm = 0.2
```

这个比原始 feature 空间的 mean matching 更有效，因为它对齐的是全局模型真正用于分类的表示。

## 4. Class-wise variance matching

旧 GAN 最大问题是类内变化小，所以只对齐均值不够，要对齐类内方差。

```python
loss_var = || var(fake_emb_y) - var(real_emb_y) ||_1
```

推荐：

```text
λ_var = 0.1
```

如果本地每类样本少，可以先按 batch 全局算，后续再改成 memory bank。

## 5. Mode-seeking diversity loss

这是这个方案里最关键的改进之一。

对同一个 `y_target`，采样两个不同噪声：

```python
fake1 = G(z1, y_target)
fake2 = G(z2, y_target)
```

希望不同 z 生成不同样本：

```python
dist_audio = ||fake_audio1 - fake_audio2||
dist_video = ||fake_video1 - fake_video2||
dist_z = ||z1 - z2||

loss_div = - (dist_audio + alpha * dist_video) / (dist_z + eps)
```

推荐：

```text
λ_div = 0.05 ~ 0.1
```

第一版用：

```text
λ_div = 0.05
```

不要太大，否则 fake 会变成离群样本。

## 6. Cross-modal consistency loss

让 fake audio 和 fake video 是一对，而不是两个独立骗分类器的模态。

轻量实现：

```python
a_pool = masked_mean(fake_audio)
v_pool = masked_mean(fake_video)

a_proj = audio_proj(a_pool)
v_proj = video_proj(v_pool)

loss_mod = 1 - cosine_similarity(a_proj, v_proj).mean()
```

推荐：

```text
λ_mod = 0.05
```

如果你暂时不想加 projection head，第一版可以先不加，等基础 GAN 跑通后再加入。

## 7. 原始统计约束

保留：

```text
audio temporal std matching
video mean/std matching
```

推荐：

```text
λ_stat = 0.05
```

---

# 七、最终损失汇总

## 判别器

```python
logits_real, emb_real = D(real_audio, real_video, return_embed=True)
logits_fake, emb_fake = D(fake_audio.detach(), fake_video.detach(), return_embed=True)

fake_label = torch.full_like(y_target, K)

loss_D_real = CE(logits_real, y_real)
loss_D_fake = CE(logits_fake, fake_label)

loss_D = loss_D_real + loss_D_fake
```

## 生成器

```python
fake_audio, fake_video = G(z, y_target)

logits_fake, emb_fake = D(fake_audio, fake_video, return_embed=True)

loss_G_target = CE(logits_fake, y_target)

prob_fake = softmax(logits_fake, dim=1)[:, K]
loss_G_avoid = prob_fake.mean()

loss_G_fm = feature_matching(emb_fake, emb_real_bank, y_target)
loss_G_var = variance_matching(emb_fake, emb_real_bank, y_target)
loss_G_div = mode_seeking_loss(G, z1, z2, y_target)
loss_G_mod = cross_modal_loss(fake_audio, fake_video)
loss_G_stat = stat_matching(fake_audio, fake_video, real_audio, real_video)

loss_G = (
    1.0  * loss_G_target
  + 0.2  * loss_G_avoid
  + 0.2  * loss_G_fm
  + 0.1  * loss_G_var
  + 0.05 * loss_G_div
  + 0.05 * loss_G_mod
  + 0.05 * loss_G_stat
)
```

---

# 八、生成器训练与下游验证流程

每一轮 synthetic-data 客户端执行：

```text
输入:
  GlobalModel_t
  本地真实数据 D_local
  当前生成器 G_t
```

## Step 1：加载全局模型作为判别器

```python
D_t.load_state_dict(GlobalModel_t.state_dict())
```

如果 fake 类 head 是全局模型的一部分，也一起加载。

## Step 2：冻结策略

推荐第一版：

```text
冻结 D 的低层 backbone；
训练 D 的 fusion 层 + classifier K+1 head。
```

学习率：

```text
D backbone: 0 或 1e-5
D head:     1e-4
G:          2e-4
```

如果 D 太弱：

```text
解冻更多高层
```

如果 D 太强：

```text
减少 D_steps 或冻结更多层
```

## Step 3：训练 D

每个 batch：

```python
real_audio, real_video, y_real = batch
z = sample_noise()
y_target = sample_target_label()

fake_audio, fake_video = G(z, y_target)

loss_D = CE(D(real), y_real) + CE(D(fake.detach()), fake_class_K)
update(D)
```

## Step 4：训练 G

```python
z = sample_noise()
y_target = sample_target_label()

fake_audio, fake_video = G(z, y_target)

loss_G = target CE + avoid fake + fm + var + div + mod + stat
update(G)
```

推荐训练步数：

```text
D_steps = 1
G_steps = 2
```

原因：D 是全局模型初始化的，本来就很强；如果 D 每轮训练太多，G 很容易崩。

## Step 5：生成合成训练样本

```python
synthetic_audio, synthetic_video = G(z, y_target)
train_label = y_target
```

第一目标是让合成样本具备真实数据替代能力：

```text
synthetic_train_data = {(G(z, y), y)}
```

如果完全由合成样本组成的数据集训练出的同结构分类器，能在真实 test 集上取得较高表现，说明 G 生成的 feature 已经足够接近真实分布，可用于联邦流程中的偷渡式数据注入。

在生成质量足够高之后，可以保留标签翻转中毒扩展：

```text
生成 source/target 类外观特征
上传时使用攻击指定标签
```

但标签翻转不是第一阶段的质量评价标准，第一阶段先验证 clean-label synthetic-only 训练能力。

## Step 6：本地训练并上传

可以做三组对照：

```text
real_only:            真实训练集
synthetic_only:       完全由 G 生成的数据集
real_plus_synthetic:  真实数据 + 合成数据混合
```

第一阶段重点看：

```text
synthetic_only 模型在真实 val/test 上的 accuracy / loss
```

如果 synthetic_only 表现接近 real_only，再进入联邦学习客户端替换/混入合成数据实验。

---

# 九、target label 采样策略

不要每次只生成一个固定目标类，否则 G 更容易坍缩。

推荐：

```text
70%: 攻击目标类
30%: 随机其他类
```

例如：

```python
if random() < 0.7:
    y_target = attack_target
else:
    y_target = random_class()
```

如果你的攻击是多目标的：

```text
均匀采样 target class set
```

这样 G 不会只学一个极窄目标类原型。

---

# 十、训练 warmup 策略

强烈建议加 warmup。

## 阶段 1：语义 warmup

前 5~10 轮，只训练 G 的基本生成能力：

```text
λ_target = 1.0
λ_avoid  = 0.1
λ_fm     = 0.1
λ_var    = 0.0
λ_div    = 0.0
λ_mod    = 0.0
λ_stat   = 0.05
```

目标：

```text
先让 G 生成能被 D 判成目标类的样本。
```

## 阶段 2：多样性 ramp-up

第 10 轮后逐步打开：

```text
λ_var -> 0.1
λ_div -> 0.05
λ_mod -> 0.05
```

目标：

```text
防止 G 只生成目标类原型。
```

## 阶段 3：下游验证阶段

当生成质量稳定后，先用 synthetic-only 数据训练同结构分类器，并在真实 val/test 上验证。

只有 synthetic-only 表现足够好后，再进入 real+synthetic 混合训练或标签翻转中毒实验。

---

# 十一、推荐超参数表

| 模块            |  参数 |            推荐值 |
| ------------- | --: | -------------: |
| 类别数           |   K |   UCF101 子集类别数 |
| 输出维度          | K+1 | 原 K 类 + fake 类 |
| z_dim         |     |            256 |
| label_emb_dim |     |            128 |
| hidden_dim    |     |            512 |
| batch_size    |     |             32 |
| G_lr          |     |           2e-4 |
| D_head_lr     |     |           1e-4 |
| D_backbone_lr |     |       0 或 1e-5 |
| D_steps       |     |              1 |
| G_steps       |     |              2 |
| synthetic_ratio |     |            0.2 |
| λ_target      |     |            1.0 |
| λ_avoid       |     |            0.2 |
| λ_fm          |     |            0.2 |
| λ_var         |     |            0.1 |
| λ_div         |     |           0.05 |
| λ_mod         |     |           0.05 |
| λ_stat        |     |           0.05 |

---

# 十二、必须记录的评估指标

不要只看：

```text
D(fake) 是否等于 target
```

因为旧 GAN 已经证明这个指标会饱和。

必须记录：

## 生成质量

```text
target_success_rate:
  D(fake) 在前 K 类中是否预测为 y_target

fake_escape_rate:
  D(fake) 是否没有预测为 fake 类 K

fake_class_prob:
  softmax(logits_fake)[K]

audio diversity ratio
video diversity ratio

embedding FID / MMD
domain classifier AUC
```

## 下游效果

```text
synthetic-only real-test accuracy
synthetic-only real-test loss
real-only vs synthetic-only gap
real+synthetic 是否保持或提升真实测试表现
```

## 标签翻转扩展

```text
ASR: attack success rate
main task accuracy
target class accuracy
source -> target confusion rate
update norm
cosine similarity with benign updates
```

## 隐蔽性

```text
local clean accuracy
global clean accuracy
malicious update L2 norm
malicious update cosine similarity
fake class activation on benign validation data
```

---

# 十三、最小实现版本

如果你想最快跑起来，按这个最小版本做：

```text
1. 把全局模型最后分类层改成 K+1。
2. 真实数据训练时 label 仍是 0~K-1。
3. G 生成 synthetic feature。
4. D 训练:
   real -> 原类别
   synthetic -> 第 K 类
5. G 训练:
   synthetic -> target 类
   同时降低 fake 类概率并匹配真实 embedding/stat/diversity
6. 质量验证:
   用 synthetic-only 数据训练同结构分类器
   在真实 val/test 上评估
7. 联邦扩展:
   先尝试 clean-label synthetic 数据偷渡
   生成质量足够高后再执行标签翻转中毒
```

这个最小版本已经足够形成完整的 synthetic-feature GAN 验证闭环。

---

# 十四、最终推荐架构图

```text
               Federated Server
                     |
              GlobalModel_t
            output: K+1 logits
                     |
             synthetic-data client
                     |
        load GlobalModel_t as D_t
                     |
        -------------------------
        |                       |
   real local data          G(z, y_target)
        |                       |
        v                       v
   D(real) -> y_real       D(synthetic) -> fake K      # update D
                            |
                            v
                   D(synthetic) -> y_target            # update G
                            |
                            v
                  generated synthetic features
                            |
                            v
          synthetic-only / real+synthetic training
                            |
                            v
              evaluate on real val/test
```

---

# 十五、我的最终建议

你应该采用这个版本：

```text
K+1 Global-Discriminator Conditional Multimodal Synthetic Feature GAN
```

核心设计是：

```text
D = 每轮同步的全局模型，输出 K+1 类；
第 K 类代表 synthetic/fake；
G 生成 audio/video feature；
D 训练目标: real -> 原类别，synthetic -> fake 类；
G 训练目标: synthetic -> target 类，且 fake 类概率低；
额外加入 feature matching、class-wise variance、mode-seeking diversity；
第一阶段用 synthetic-only 训练同结构分类器，并在真实测试集上评估生成质量；
生成质量足够好之后，再进入 clean-label 偷渡或标签翻转中毒实验。
```

最重要的三个实现点是：

```text
1. K+1 softmax 可以用，但 G 必须优化 target 类，而不是只优化“非 fake”。
2. 必须加 mode-seeking diversity loss，否则会重复旧 GAN 的 mode collapse。
3. 必须用全局模型 embedding 做 feature matching，而不是只看 teacher accuracy。
```

这套方案更适合你的任务，因为它把“生成器”和“当前联邦全局模型”直接耦合起来，并把第一评价目标放在 synthetic-only 下游训练效果上：合成数据训练出的同结构模型在真实测试集上越好，说明生成特征越接近真实分布。标签翻转中毒保留为生成质量达标后的第二阶段实验。

[1]: https://arxiv.org/html/2405.11440v3?utm_source=chatgpt.com "A Model Consistency-Based Countermeasure to GAN- ..."
