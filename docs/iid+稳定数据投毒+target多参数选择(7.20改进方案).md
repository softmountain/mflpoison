数据读完了。Baseline 的加入让整个画面发生了根本性变化，先说分析结论，然后逐条评价你的新方案。

---

## 一、Baseline 校正后的关键发现

### 信噪比是灾难性的

**无攻击 baseline 的 per-class recall 漂移标准差是 28.6。** 也就是说，在完全不攻击的情况下，任意一个类的 recall 变动 ±29 个百分点是"正常噪声"。而你 55 组实验中只有 **24%（13/55）** 的 net 效果超过了这个噪声阈值。

最极端的例子：class 0（ApplyEyeMakeup）在无攻击下从 77.3% 崩到 4.5%（−72.7），比你的任何攻击组都猛。这不是攻击，纯粹是 `sample_rate=0.2 + Dirichlet α=1.0` 下的随机采样导致的灾难性遗忘。

### 几个关键 target 的真相被揭示了

| Target | 旧结论 | Baseline 校正后 | 真相 |
|---|---|---|---|
| 40 (Shotput) | "破坏 −61.6" | net −48.6 (仍然最强) | 攻击真实有效，但被高估了 13 点 |
| 9 (BodyWeight) | "几乎无效 +1.1" | net **−25.6** | 攻击被自然上升 +26.7 完全掩盖 |
| 28 (IceDancing) | "激活 +30.4" | net **0.0** | "激活"100% 来自自然漂移 |
| 2 (Archery) | "破坏 −21.9" | net **+9.8** | "破坏"实际是自然下降 |
| 31 (MoppingFloor) | "微弱 −2.9" | net **−38.2** | 被自然上升 +35.3 掩盖的强攻击 |

Target 9 和 31 的反转特别重要——它们在旧实验中看起来没被攻击，但 baseline 校正后是第二和第三强的攻击效果。

### 但这个 baseline 本身也只是一次采样

`sample_rate=0.2` 意味着每轮只 2 个客户端参与，不同的随机种子会产生完全不同的 per-class drift。你的 baseline 是**单次运行**——class 0 恰好被遗忘了 72.7 点，换个种子可能只遗忘 20 点。要做可靠的校正，至少需要 3-5 次 baseline 取均值。但更根本的问题是：这个实验框架的噪声太大，校正的意义有限。

---

## 二、评价你的新方案

你提的每一条调整方向都是对的，下面逐条细化。

### ✅ IID 分布（Dirichlet α=5.0）

这是最关键的改进。α=1.0 下 per-class drift 的 std 是 28.6；α=5.0 接近 IID，每个客户端的类分布基本均匀，**灾难性遗忘不再发生**，baseline drift 的 std 应该能压到 5 以内。这一条单独就足以让实验从"不可解读"变成"可解读"。

### ✅ 收敛后再攻击（~75% acc）

63.5% 的 M\* 有太大的上升空间，正常训练的增益盖过攻击。75% 接近天花板后：正常训练几乎没有增益、攻击的破坏不会被"治愈"、baseline 的 per-class drift 更加稳定。这比简单提高 sample\_rate 更有效。

### ✅ 稳定 inject 数量和攻击者参与比例

上一轮实验中 inject 从 13 到 51 不等，与 target 身份混淆。固定 inject 数量后，跨 target 的比较才有意义。同时保证恶意客户端每轮都被采样（或用 `sample_rate=1.0`），消除"攻击轮次 vs 纯训练轮次"的随机性。

### ⚠️ 关于 source 的选取——你的方向对，但要注意一个陷阱

用 TSTR best5 类作为 source 是合理的：生成质量高意味着合成特征足够"真实"，能向模型注入强信号。但上一轮的教训是：**best5 source 的真实样本 recall 太高（90-100%），模型对它们的表征太过坚固，s2t 永远是 0。** 如果你的新目标只是 target destruction 而不追求 s2t 误判，这不是问题。但如果仍然希望看到 s2t > 0，需要选 TSTR 中等偏上（70-90%）但真实 recall 不那么极端的 source。

### ⚠️⚠️ 关于 target 的三条标准——逻辑链需要拆开讲

你的直觉是对的，但三条标准之间有隐含的张力，需要理清。

**标准 1（特征距离大）+ 标准 2（TSTR 差）** 这两条在上一轮实验中已经得到间接验证：within-target 分析显示 cosine 越低（距离越大）破坏越强（r=+0.42），TSTR 差的类被生成的特征更"异质"。但这两条共同指向的其实是同一个机制——**注入特征与 target 真实特征的差异程度**。特征距离大是从 source 端看的差异，TSTR 差是从 target 端看的差异（生成器对 target 类的表征能力弱 → 注入的特征与 target 真实特征差异大）。

**标准 3（baseline recall 高）** 上一轮实验清楚地证明了：baseline recall 高的 target 被破坏最多（Pearson r = −0.561）。因为模型对这类的决策边界是"精修"过的，引入噪声特征直接破坏精修结果。

**"没学明白"——怎么操作化？** 你说"分类能力看起来强但是没学明白"，这个概念可以用以下指标量化：你说的"没学明白"可以用 **softmax 熵** 来量化。具体做法：对全局模型 M\* 做推理，看每个类 $c$ 的**正确预测**的 softmax 输出分布。如果模型"真懂"这个类，正确预测时 softmax 会是一个尖锐的 one-hot（低熵，比如 p=0.95 给正确标签）。但如果模型"没学明白"只是靠排除法或浅层特征碰巧分对了，softmax 会是一个扁平的分布（高熵，比如 p=0.4 给正确标签、p=0.15 分散在几个其他类上）。

高 recall + 高 softmax 熵 = "纸老虎"——看起来分得对，但决策边界很薄，最容易被攻击打穿。

**TSTR gap（real\_recall − TSTR\_recall）** 补充了另一个维度：如果一个类的真实 recall 是 85% 但 TSTR 只有 15%，说明这个类的视觉/特征模式复杂到 GAN 无法复现。攻击者用 GAN 生成的 source 特征替换这种 target 时，注入的东西和 target 的真实表征差异巨大——破坏力最强。

---

## 三、完整方案建议

结合你的思路和上一轮的教训，我建议的实验设计如下：

### 固定设置

| 项 | 值 | 理由 |
|---|---|---|
| 客户端数 | 10 | 不变 |
| 恶意客户端 | 4 (client 0,1,2,3) | 40% 恶意率 |
| 数据分布 | Dirichlet α=5.0 | 近 IID，消除灾难性遗忘噪声 |
| M\* | 全局 acc ≥ 75% 的收敛模型 | 消除"正常训练增益盖过攻击"的问题 |
| sample\_rate | **1.0** | 全员参与，消除采样随机性 |
| 攻击轮数 | 15 轮（保持不变） | 与旧实验可比 |
| inject 数 | **固定为 N**（所有 target 统一） | 消除 inject 数混淆 |
| Baseline | 同参数无攻击跑 **3 次**取均值 | 单次不可靠 |

### Source 选择

TSTR recall 排名前 5 的类，但增加一个过滤条件：**真实 recall 不超过 ~90%**。如果 best5 全都是 95%+，考虑往下取到 TSTR top 8-10 里面挑真实 recall 相对低的。这样既保证合成质量好（攻击载荷强），又留一点"可迁移"的余地给 s2t 指标。

### Target 选择

按照三阶段 pipeline 筛选：

**阶段 1**：M\* recall ≥ 70% 的类（约 20 个候选）

**阶段 2**：计算 fragility score

$$\text{fragility}(c) = \underbrace{\bar{H}(p(y|x_c))}_{\text{正确预测的平均 softmax 熵}} + \lambda \cdot \underbrace{(\text{recall}(c) - \text{TSTR}(c))}_{\text{TSTR gap}}$$

$\lambda$ 做归一化让两项量纲一致即可。取 fragility 排名前 10 的候选。

**阶段 3**：对每个候选 target，与每个 source 计算 penultimate cosine。保留 cosine 最低的 source-target 配对。最终取 fragility 最高的 5 个 target，每个配 1 个最远 source。

### 实验矩阵

| 组 | Source | Target | 变量 |
|---|---|---|---|
| A1-A5 | 最远 source | Top-5 fragility target | 主实验 |
| B1-B5 | **最近 source** | 同上 5 target | 对照：cosine 效应 |
| C1-C5 | 同 A 的 source | **bottom-5 fragility target**（recall高但不fragile） | 对照：fragility 效应 |
| Baseline | — | — | 无攻击 ×3 次 |

总共 15 组攻击 + 3 组 baseline = 18 次实验。

A vs B 控制 source-target 距离（stage 3），A vs C 控制 target fragility（stage 2），Baseline 做校正。这三组对比构成一个干净的消融实验。

### 评估指标的调整

鉴于上一轮 s2t 几乎全为 0，建议把指标体系调整为：

- **主指标**：NET Δt\_recall = 攻击 Δt\_recall − baseline 均值 drift(t)。这是 target destruction 的净效应。
- **辅助指标 1**：s2t 误判率。仍然监测，如果新方案下仍为 0，则正式确认"特征碰撞攻击在此场景下不产生定向误判"。
- **辅助指标 2**：全局 acc 的 net 变化。
- **新指标**：target 类在攻击前后的 **softmax 熵变化**。如果攻击有效，被攻击 target 的预测熵应该从低（自信正确）变高（开始困惑），甚至在 recall 还没掉之前就能看到这个早期信号。

### 一个可能成立的新故事线

如果新实验验证了"high recall + high entropy + low TSTR = 最脆弱"这个假设，你就有了一篇论文可以讲的完整故事：

> 在联邦学习中，具有 GAN 生成器的恶意客户端可以通过特征碰撞攻击定向破坏全局模型对特定类别的分类能力。攻击的强度不取决于 source-target 的特征空间距离（旧假设被证伪），而取决于 target 类本身的"表面韧性"——一个组合指标，由 baseline recall、预测置信度（softmax 熵）、和合成数据复现能力（TSTR gap）共同决定。我们提出了 fragility score 来刻画这种脆弱性，并验证了高 fragility 类在攻击下的 recall 降幅显著大于低 fragility 类。

这个故事既有攻击方的新发现（fragility-driven target selection），也有防御方的启示（监控 fragility score 高的类可以早期预警），比纯粹的"特征碰撞假设被证伪"更有建设性。

---

总结来说，你的方案方向完全可行。α=5.0 + 收敛后攻击 + 固定 inject + 三条件 target 选择，这些改进把上一轮实验的几个核心缺陷全部修复了。唯一要补充的是用 softmax 熵来量化"没学明白"，以及把 B 组（近距离 source）和 C 组（低 fragility target）加上做对照，这样实验在论文中的消融分析就是自包含的。