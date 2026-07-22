# Fragility 中 z(H) 与 z(gap) 独立贡献实验实施方案

## 0. 文档状态

- 研究问题：拆解 `fragility(c) = z(H(c)) + z(gap(c))` 中两项对生成特征投毒破坏强度的独立贡献，并检验训练轮数与攻击放大因子的调节作用。
- 数据集与任务：UCF101 51 类多模态联邦学习，定向类别可用性攻击（targeted availability poisoning）。
- 基准实验：2026-07-21 `fragility_exp_7.21`。
- 本方案性质：预注册式实施协议。未完成本方案规定的 Gate 0 和代码测试前，不得启动正式矩阵。

## 1. 证据边界与执行前提

### 1.1 已从 7.21 结果包核实的事实

1. 全局模型 `M*`：FedAvg、Dirichlet `alpha=5.0`、fold1、10 clients、`fuse_base`、`hid128`，训练 150 epochs，best@27，test accuracy `74.95%`。
2. 攻击阶段：从 `M*` 出发继续训练 15 rounds，`sample_rate=1.0`、batch size 16、local epoch 1、learning rate 0.05，恶意客户端为 0、1、2、3。
3. 输入特征：audio MFCC `[500,80]`，video MobileNetV2 `[10,1280]`。本方案以 7.21 结果包中的形状为准，不使用其他仓库版本中可能存在的 `[9,1280]`。
4. 攻击语义：把“由 source 类条件生成的特征”作为 target 标签样本暴露给恶意客户端训练，即：

   ```text
   condition_labels = source_class = s
   source_labels    = source_class = s
   train_labels     = target_class = t, s != t
   ```

5. 7.21 的 `attack_n_inject=30` 是每客户端上限，不是固定注入数；实际总注入量为 23--49，因此主实验必须改成严格固定暴露量。
6. 7.21 只有 3 个 benign seeds（8、9、10），攻击组随机种子没有写入结果；新实验必须记录所有随机流。

### 1.2 运行环境边界

实际服务器环境是唯一执行依据，本方案不约束仓库目录、脚本名称、函数名称、配置格式或命令行接口。无论使用现有 runner、配置系统还是重新封装实验入口，都必须满足以下行为契约：

- 能明确区分生成条件类、样本语义 source 类和训练标签类。
- 能确定性地控制恶意客户端、毒样本位置、毒样本ID、每轮暴露量和攻击调度。
- 能从相同初始模型和相同随机状态生成一一配对的 benign/attack 轨迹。
- 能按本方案规定的粒度保存逐轮、逐类、逐客户端指标和实验来源信息。
- 能实现数据级放大；更新级放大必须作为独立实验能力启用。

实施者可按实际环境映射这些能力，但不得因接口差异改变实验语义、剂量定义、随机性控制或统计终点。

## 2. 研究目标与预注册假设

### 2.1 主要目标

估计 `z(H)` 和 `z(gap)` 对 target 净破坏强度的标准化独立效应，并判断 `z(H)` 是否比 `z(gap)` 具有更高的解释力和跨条件预测力。

### 2.2 次要目标

1. 检验高 `H` target 是否随攻击轮数增加更快、更持续地崩溃。
2. 检验攻击数据剂量增加是否优先放大高 `H` target 的脆弱性。
3. 区分纯数据级放大与恶意客户端更新级放大。
4. 判断攻击停止后 target 是否恢复，区分暂态扰动和持久污染。
5. 检查 `gap` 是否只是 target 难度代理量，而不是直接生成器机制。

### 2.3 假设

- `H1`：控制 baseline recall、TSTR、source-target 特征差异、baseline 漂移和攻击剂量后，`z(H)` 的系数仍显著大于 0（以正向破坏量为因变量）。
- `H2`：`z(H)` 的增量半偏 `R^2` 和留一 target 预测增益均大于 `z(gap)`。
- `H3`：存在正向 `z(H) x log(1+round)` 交互，高 `H` target 的破坏随轮数增长更快。
- `H4`：存在正向 `z(H) x log2(A_data)` 交互，高 `H` target 对数据级放大更敏感。
- `H5`：如果只有更新级放大 `gamma_update > 1` 才产生强破坏，则结论应归因于混合模型中毒，而不能归因于生成数据中毒本身。

## 3. 变量定义

### 3.1 Fragility 分量

固定使用攻击前的同一个 `M*` 和同一个真实测试集计算：

```text
H(c)       = 类 c 中被 M* 正确预测样本的平均 softmax 熵
R_real(c)  = M* 在真实测试集上的类 c recall
R_tstr(c)  = 仅用固定生成器合成训练集训练后，在真实测试集上的类 c recall
gap(c)     = R_real(c) - R_tstr(c)
```

标准化规则固定为 51 类总体均值和总体标准差（`ddof=0`）：

```text
z_H(c)   = (H(c)   - mean_51(H))   / std_51(H)
z_gap(c) = (gap(c) - mean_51(gap)) / std_51(gap)
```

一旦 Gate 0 通过，51类总体的标准化均值和标准差必须作为实验来源信息冻结。不得按实验子集重新标准化，也不得根据攻击结果调整 target。

需要明确：当前攻击注入的是 **source 生成特征**，并不会把 target 生成特征注入训练。因此 `gap(t)` 不是直接施加到模型上的攻击剂量，而是 target 类复杂度/生成器难复现程度的代理变量。本实验首先检验其预测贡献；直接机制应由 source synthetic 到 target real 的 cosine、MMD/Fréchet distance和梯度对齐解释。除非额外设计能在固定 target 下操纵 gap 的干预，否则不得把 gap 的回归关系表述为因果效应。

### 3.2 主要因变量

对 target `t`、round `r`、seed `k`：

```text
delta_attack(t,r,k) = Recall_attack(t,r,k) - Recall_M*(t)
delta_benign(t,r,k) = Recall_benign(t,r,k) - Recall_M*(t)
D(t,r,k) = -(delta_attack(t,r,k) - delta_benign(t,r,k))
```

`D > 0` 表示攻击造成净破坏。攻击与 benign 必须使用相同的 `M*`、partition、客户端顺序和训练 seed。

主要终点：

- `D15`：round 15 的净破坏。
- `D30`：round 30 的净破坏。
- `D60`：round 60 的净破坏。
- `AUC_D`：round 0--60 的正向净破坏曲线面积。
- `T5`、`T10`：首次连续两个评估点达到 `D>=5`、`D>=10` 的 round；未达到记为删失。

### 3.3 协变量

必须同时保存：

- `R_real`、`R_tstr`，不能只保存 `gap`。
- target baseline 漂移均值与标准差。
- source synthetic center 与 target real center 的 cosine similarity。
- source-target MMD 或 Fréchet feature distance，至少实现其中一种分布级距离。
- 每客户端实际 poison count、poison exposure ratio、原始数据量和 FedAvg 权重。
- target 测试样本数、正确预测样本数、平均 top1-top2 margin。
- 客户端更新总范数以及分类头更新范数。

## 4. Gate 0：冻结代码、数据和模型来源

以下检查全部通过后才允许运行实验：

1. 冻结实际运行环境中生成本轮结果所需的代码、配置和分析逻辑；不要求特定目录结构或版本控制工具。
2. 保存可唯一识别环境的代码版本或快照摘要、未提交修改、Python版本、CUDA/cuDNN版本、PyTorch版本和依赖清单。
3. 对以下文件计算 SHA256：
   - UCF101 train/test manifest 或 feature pickle；
   - 客户端 partition；
   - `M*` checkpoint；
   - DTM-GAN checkpoint；
   - `dtm_final_train5100` 合成特征；
   - 真实和合成 class centers。
4. 从实际数据加载流程导出并冻结 class ID 到类别名称的映射。不得从其他环境的 demo 或辅助文件推导 class ID。
5. 断言类别数为 51，audio shape 为 `[500,80]`，video shape 为 `[10,1280]`。
6. 断言 `M*` test accuracy 与 7.21 的 `74.95%` 差异不超过 0.05 个百分点，并逐类对比 recall，最大绝对差异不超过 0.1 个百分点。
7. 重新计算 `H`、TSTR、gap、fragility，逐类与 7.21 已保存的逐类结果比较，误差不超过 `1e-6`（浮点序列化差异除外）。
8. 复现 7.21 的 A1、A3、A4 三组和一个 benign run，确认结果格式、攻击语义和趋势一致。由于旧攻击 seed 缺失，只要求趋势复现，不要求逐位相同。

Gate 0 任何一项失败时必须停止并定位来源，不得进入正式矩阵。

## 5. Target 设计

### 5.1 广泛筛查集合

沿用 `R_real >= 70%` 的可下降空间要求。7.21 中共有 34 类满足条件。

主筛查固定使用 source class 42；为避免 `s=t`，排除 target 42，得到 33 个 target。固定 source 的目的，是消除 source 身份和 source 生成质量差异。

### 5.2 H 匹配对：gap 和初始 recall 近似，H 不同

| pair | target 1 | target 2 | recall 1/2 | H 1/2 | gap 1/2 | 主要对比 |
|---|---:|---:|---:|---:|---:|---|
| H-P1 | 49 | 2 | 80.0/75.6 | 0.645/0.296 | 54.3/53.7 | 高 H vs 中 H |
| H-P2 | 14 | 25 | 74.4/71.4 | 0.523/0.232 | 38.5/35.7 | 高 H vs 低 H |
| H-P3 | 20 | 21 | 78.4/83.8 | 0.359/0.052 | 8.1/8.1 | 高 H vs 低 H |

### 5.3 Gap 匹配对：H 和初始 recall 近似，gap 不同

| pair | target 1 | target 2 | recall 1/2 | H 1/2 | gap 1/2 | 主要对比 |
|---|---:|---:|---:|---:|---:|---|
| G-P1 | 23 | 43 | 88.9/90.6 | 0.156/0.157 | 51.1/12.5 | 高 gap vs 低 gap |
| G-P2 | 50 | 10 | 97.8/95.3 | 0.108/0.102 | 53.3/14.0 | 高 gap vs 低 gap |
| G-P3 | 35 | 34 | 85.7/80.5 | 0.205/0.211 | 69.4/24.4 | 高 gap vs 中 gap |

这些配对是预注册候选。同步完整 cosine 矩阵后还必须检查同一 source 下的 pair 内 cosine 差异；若差异大于 0.10，则保留类别但在模型中显式控制 cosine，不得静默更换类别。探索性替代配对必须单独标记，不能进入确认性检验。

## 6. Source 与合成样本控制

### 6.1 主实验 source

- 固定 `source_class=42`。
- `tstr_pool`：冻结的 `dtm_final_train5100`，仅用于计算 TSTR、gap 和 target 选择，不作为主实验唯一攻击载荷。
- `attack_pool`：使用同一个 DTM-GAN checkpoint、独立 generator seed 新生成的 source 42 特征池，至少 512 条；不得参与 TSTR/gap 计算。
- 对同一 seed，不同 target 使用完全相同的 source synthetic sample IDs，仅改变 `train_labels=t`。
- 四个恶意客户端在同一 round 内使用互不重叠的 synthetic IDs；`A_data=4` 每轮共需 128 条。
- 为模拟固定本地恶意数据集，同一 client 分配到的静态 poison 子集可以跨联邦 rounds 重复训练；但同一 client 的单个 local epoch 内不得重复同一条特征。
- attack pool 不足时必须重新生成并记录 generator seed，不能在同一 round 内循环凑数。

### 6.2 Source 鲁棒性扩展

主结论形成后，选用 source 38 和 source 21 复验代表性 target 对：H-P1、G-P3、H-P3。该阶段检验 `H/gap` 结论是否依赖 source 42，不参与主假设首次判定。

### 6.3 标签语义硬校验

无论实际环境采用什么参数名或数据结构，都必须在运行前解析为以下显式语义：

```text
condition_class = s
source_class    = s
train_label     = t
s != t
```

每个保存的合成批次或等价来源记录至少能恢复：生成条件类、source语义类、训练标签类，以及“source生成特征使用target训练标签”的攻击语义标识。

进入客户端训练前断言 `condition_class == source_class != train_label`。任何无法从配置和产物中恢复这三个值的实现均不允许进入正式矩阵。

## 7. 固定注入量与数据级放大因子

### 7.1 为什么不再使用旧 `n_inject=30` 上限

旧实现只替换恶意客户端已有的 target 类条目，注入量由 target 在客户端上的自然数量决定，无法比较不同 target。主实验改为确定性 fixed-exposure sampler。

### 7.2 Fixed-exposure 规则

- 恶意客户端固定为 0、1、2、3。
- 每个恶意客户端每个 local epoch 的基础 poison exposure 为 `N0=8`。
- 数据级放大因子 `A_data in {1,2,4}`：

  ```text
  A_data=1 -> 8 poison samples/client/round，合计 32
  A_data=2 -> 16 poison samples/client/round，合计 64
  A_data=4 -> 32 poison samples/client/round，合计 128
  ```

- 训练集长度、batch size、local epochs 和 FedAvg 样本权重保持不变；poison 样本替换等量 clean exposure，不增加客户端聚合权重。
- 对同一 client/seed，`A=1` 的 poison index 集必须是 `A=2` 的子集，`A=2` 必须是 `A=4` 的子集，形成嵌套剂量设计。
- 四个恶意客户端必须各自达到规定数量；若客户端有效训练长度小于 32，Gate 失败并重新选择 `N0`，不能截断后继续。
- poison sample IDs、被替换的 clean indices 和每轮暴露次数必须写入 manifest。

这一定义把“放大因子”解释为纯数据层面的毒样本暴露倍数，仍属于数据中毒。

## 8. 更新级放大扩展

更新级放大不得混入主矩阵。单独定义：

```text
Delta_i = local_state_i - global_state
Delta_i_scaled = gamma_update * Delta_i
gamma_update in {1,2,4}
```

仅对恶意客户端更新应用，benign 更新保持不变。`gamma_update > 1` 属于数据中毒与模型更新操纵的混合攻击。

该扩展固定 `A_data=1`，在 target `{49,2,35,34,20,21}` 上运行。必须保存放大前后总更新范数、audio分支范数、video分支范数和classifier范数。若出现 NaN/Inf 或全局准确率单轮下降超过 10 个百分点，立即停止该 run。

## 9. 训练轮数与攻击调度

每条正式轨迹统一训练到 60 rounds，一次运行同时提供 15、30、60 轮终点，禁止为三个终点分别启动不同随机轨迹。

### 9.1 连续攻击

```text
round 1--60: 四个恶意客户端持续使用固定剂量 poison exposure
```

用于检验累积攻击和 `H/gap x round` 交互。

### 9.2 脉冲攻击

```text
round 1--15: 攻击
round 16--60: 恶意客户端恢复完全 clean 训练
```

用于区分攻击持续性、模型自愈和普通短期漂移。攻击停止后不得继续保留 poison sampler、poison loss 权重或更新放大。

### 9.3 评估点

必须在 `r={0,5,10,15,20,30,45,60}` 评估；若计算预算允许，round 1--15 每轮评估 target recall、entropy 和 margin，以核验早期信号。

## 10. 固定训练参数与随机性

| 参数 | 固定值 |
|---|---|
| aggregation | FedAvg，无防御 |
| clients | 10 |
| malicious clients | 0,1,2,3 |
| Dirichlet alpha | 5.0 |
| sample rate | 1.0 |
| fold | 1（主实验） |
| initial model | 冻结的 74.95% `M*` |
| batch size | 16 |
| local epochs | 1 |
| local learning rate | 0.05 |
| attack horizon | 60 rounds |
| evaluation | 0,5,10,15,20,30,45,60 |
| main seeds | 8,9,10,11,12 |

随机流必须拆分并写入配置：

```text
partition_seed
client_sampling_seed
local_data_order_seed
model_training_seed
poison_index_seed
synthetic_sample_seed
```

同一 seed 的 benign/attack 条件共享前四项；只允许 poison 和 synthetic 两项因攻击条件而存在。`sample_rate=1.0` 下仍应保存 clients-per-round 列表作为完整性证据。

## 11. 正式运行矩阵

### Phase R0：7.21 锚点复现（强制）

- A1、A3、A4 三个攻击条件，15 rounds，旧语义、旧注入模式。
- 1 个 matched benign。
- 目的：验证同步代码确实对应 7.21 路径，不纳入新假设统计。

### Phase S1：33 target 广泛筛查（主效应识别）

- target：`R_real>=70%` 且 `t!=42` 的 33 类。
- source：42。
- `A_data=1`，连续攻击，60 rounds。
- seeds：8、9、10。
- 数量：`33 x 3 = 99` attack trajectories，加 3 条共享 matched benign。
- 用途：获得足够 target 数估计 `z(H)` 与 `z(gap)` 的跨类独立贡献。重复 seed 不替代 target 数量。

### Phase M1：12 target 匹配剂量实验（核心确认）

- target：第 5 节 12 个匹配 target。
- source：42。
- `A_data={1,2,4}`，连续攻击，60 rounds。
- seeds：8--12。
- 总矩阵：`12 x 3 x 5 = 180` attack trajectories。
- S1 已覆盖其中 `12 x 1 x 3 = 36` 条，可直接复用；需新增 144 条 attack 和 seeds 11、12 的 2 条 benign。

### Phase M2：攻击停止后的持久性（强制）

- target：`{49,2,20,21,35,34}`，覆盖三个代表性匹配对。
- source：42。
- `A_data=4`，round 1--15 攻击，16--60 clean。
- seeds：8--12。
- 数量：`6 x 5 = 30` trajectories。

### Phase R1：source 鲁棒性（扩展）

- target：`{49,2,20,21,35,34}`。
- source：38、21。
- `A_data={1,4}`，连续攻击，60 rounds。
- seeds：8、9、10。
- 数量：`6 x 2 x 2 x 3 = 72` trajectories。

### Phase U1：更新级放大（扩展，混合攻击）

- target：`{49,2,20,21,35,34}`。
- source：42，`A_data=1`。
- `gamma_update={1,2,4}`，连续攻击，60 rounds。
- seeds：8--12。
- `gamma=1` 复用 M1；新增 `6 x 2 x 5 = 60` trajectories。

### Phase C1：生成器与标签语义对照（强制）

代表 target：`{49,2,35,23}`；source 42；seeds 8、9、10；`A_data={1,4}`。比较：

1. `G-poison`：source 生成特征 + target 标签。
2. `G-clean`：完全相同的生成特征 + 正确 source 标签。
3. `Random-target`：按 source 特征逐维置换或匹配均值方差的随机特征 + target 标签。
4. `Real-poison`：真实 source 特征 + target 标签（上界对照）。

四种条件使用相同的客户端、替换 indices 和有效暴露数。若真实 source 样本不足以构造 128 条互异样本，则所有四种条件统一采用相同的有放回规则，并把唯一样本数和重复率写入结果；不得只对 `Real-poison` 改变剂量定义。

数量：`4 targets x 4 conditions x 2 doses x 3 seeds = 96`。M1 中的 `G-poison` 24 条可复用，新增 72 条。

### 运行规模

- 主结论所需强制阶段：R0、S1、M1、M2、C1。
- 扣除复用后约 350 条新 attack/controls trajectories，外加 5 条60轮 matched benign 和 R0 benign。
- R1、U1 共增加 132 条扩展轨迹。
- 所有 60 轮终点来自同一轨迹，因此不得再乘以 15/30/60 三个终点。

## 12. 评估指标

### 12.1 主要指标

- target baseline-corrected destruction：`D15`、`D30`、`D60`、`AUC_D`。
- target recall 原始变化与 benign drift。
- target attack success：连续两个评估点满足 `D >= max(5, 2*sigma_benign(t,r))`。

### 12.2 隐蔽性与副作用

- global accuracy、Macro-F1、balanced accuracy。
- non-target Macro Recall。
- source recall。
- source-to-target ASR、target false-positive rate。
- 每类 recall 和混淆矩阵。

### 12.3 机制指标

- target entropy：对全部 target 样本和正确预测 target 样本分别计算。
- target top1-top2 margin、target NLL。
- source synthetic 到 target real 的 center cosine、MMD/Fréchet distance。
- clean target 梯度与 poison 梯度的 cosine similarity。
- 恶意/正常客户端总更新范数及 classifier/audio/video 分块范数。

熵只能作为候选机制指标。只有当其变化在多个 target、多个 seed 中先于 `D` 上升，才能称为早期信号，不能由 c49 单类决定。

## 13. 统计分析

### 13.1 先分析分量，不直接分析总 fragility

以 target 级 seed 均值作横截面主分析，同时保留完整重复测量模型。

基础模型：

```text
M0: D ~ R_real + baseline_std + cosine + feature_distance
MH: M0 + z_H
MG: M0 + z_gap
MHG: M0 + z_H + z_gap + z_H:z_gap
```

由于 `gap = R_real - R_tstr`，必须增加分量模型避免代数混淆：

```text
MCOMP: D ~ z_H + z_R_real + z_R_tstr + cosine + feature_distance + baseline_std
```

### 13.2 轮数和剂量模型

完整重复测量模型使用：

```text
D ~ z_H + z_gap
  + log(1 + round) + log2(A_data)
  + z_H:log(1 + round)
  + z_gap:log(1 + round)
  + z_H:log2(A_data)
  + z_gap:log2(A_data)
  + R_real + cosine + feature_distance + baseline_std
  + (1 | target) + (1 | seed)
```

若混合模型收敛失败，使用 target-cluster bootstrap 和 HC3 robust OLS，不得删除失败结果后重新选择模型。

### 13.3 独立贡献报告

必须报告：

- 标准化 `beta_H`、`beta_gap` 和 95% CI。
- 加入每个变量后的增量半偏 `R^2`。
- AIC/BIC 仅作辅助，不作为唯一判据。
- leave-one-target-out cross-validation 的 RMSE 和预测相关性。
- target-cluster bootstrap 95% CI，至少 5000 次重采样。
- 匹配对内差值和配对 bootstrap CI。
- 多 target 单独检验使用 Benjamini-Hochberg FDR 校正。

### 13.4 “H 更核心”的预注册判据

只有同时满足以下条件才能接受：

1. S1 中 `beta_H > 0` 且 95% CI 不跨 0。
2. `MH` 相对 `M0` 的增量半偏 `R^2` 大于 `MG` 相对 `M0` 的增量。
3. 加入 `z_H` 后 leave-one-target-out RMSE 的改善大于加入 `z_gap`。
4. 三个 H 匹配对中至少两个在预期方向，且合并配对效应 CI 不跨 0。
5. 结论在 round 15 和 round 60 同向，且不由 c49 单类驱动；移除 c49 后仍成立。
6. `MCOMP` 中 `z_H` 的效应仍存在，排除 `gap` 内含 `R_real` 导致的伪贡献。

若只满足部分条件，表述为“H 是较强候选预测子”，不得写成核心机制已验证。

## 14. 攻击成功、隐蔽性和停止规则

### 14.1 单 target 攻击成功

同时满足：

1. 连续两个评估点 `D >= max(5, 2*sigma_benign)`。
2. global accuracy 的 baseline-corrected drop 不超过 1 个百分点。
3. non-target Macro Recall 的 baseline-corrected drop 不超过 2 个百分点。
4. 至少 3/5 seeds 达到相同方向。

不满足隐蔽性条件时只能称为全局破坏，不能称为隐蔽定向攻击。

### 14.2 Run 停止规则

- loss、参数或更新出现 NaN/Inf。
- global accuracy 相对 matched benign 单轮额外下降超过 10 个百分点。
- 实际 poison exposure 与配置不一致。
- benign 和 attack 的客户端顺序、partition 或初始 checkpoint hash 不一致。
- synthetic artifact 语义断言失败。

停止的 run 必须保留日志和失败原因，不得从统计分母中静默删除。

## 15. 实现能力要求

本方案不规定文件或函数接口。实际环境需要提供以下逻辑能力，既可在一个入口中实现，也可拆分为多个组件：

1. **实验清单生成**：根据 phase、target、source、seed、剂量和调度生成唯一 run ID，并在启动前检查重复与缺失组合。
2. **语义明确的合成批次**：能同时记录生成条件类、source 语义类、训练标签类和合成样本ID。
3. **确定性 fixed-exposure 注入**：按客户端精确选择替换位置，支持 8/16/32 的嵌套剂量，并保持训练长度和聚合权重不变。
4. **连续与脉冲调度**：能在指定轮次启停攻击，并证明停止后 poison exposure 为 0。
5. **更新级放大**：只对恶意客户端更新应用缩放，同时保存放大前后的分块范数。
6. **配对 benign/attack 执行**：从相同模型、partition和训练随机流出发，生成可逐轮归因的配对结果。
7. **逐轮评估与断点恢复**：在指定 checkpoints 计算全部指标，异常中断后可以从冻结状态恢复。
8. **归因与统计分析**：完成 baseline correction、匹配对分析、分量回归、交互模型、bootstrap和敏感性分析。

每条 run 无论以何种文件格式保存，都必须能恢复以下信息组：

- 解析后的完整实验条件与状态（planned/running/completed/failed）。
- 代码、环境、数据、模型、生成器和partition来源摘要。
- 每客户端 poison indices、synthetic IDs、实际暴露数和聚合权重。
- 逐轮全局指标、逐类指标和逐客户端更新指标。
- round 0、15、30、60 的模型状态或等价可恢复快照。
- 最终摘要、标准输出/错误日志和失败原因。

## 16. 必须实现的自动化测试

1. **标签语义测试**：生成条件为 `s`，训练标签为 `t`，并断言 `s!=t`。
2. **固定剂量测试**：四个恶意客户端在每轮分别得到 8/16/32 个 poison exposure。
3. **嵌套剂量测试**：A1 indices 是 A2 子集，A2 是 A4 子集。
4. **数据长度测试**：所有剂量下客户端训练长度和 FedAvg 权重不变。
5. **benign 配对测试**：同 seed 的 benign/attack 初始 checkpoint、partition、客户端顺序和 clean index 顺序一致。
6. **调度测试**：pulse 模式在 round 16 后 poison count 严格为 0。
7. **更新放大测试**：只缩放恶意客户端 delta；`gamma=1` 与未启用放大逐位一致。
8. **cosine 完整性测试**：保存每个 target 对候选 source 的完整矩阵；farthest 为最小 similarity，nearest 为最大 similarity。
9. **断点续跑测试**：从 checkpoint 恢复后的指标与不中断运行一致。
10. **结果模式测试**：所有必需字段存在，失败 run 也有状态和原因。

正式矩阵启动前必须运行全部单元测试，并完成一个 target、一个 seed、5 rounds 的 smoke test。

## 17. 执行顺序

1. 完成 Gate 0，冻结实际环境来源摘要、所有 hash、class map 和 scaler。
2. 实现 fixed-exposure、显式标签语义、连续/脉冲调度和完整日志。
3. 运行自动化测试与 5-round smoke test。
4. 运行 R0，核对 7.21 锚点。
5. 运行 5 个 benign seeds 到 60 rounds；先完成 baseline 再启动大矩阵。
6. 运行 S1；冻结分析脚本并完成主效应初检。
7. 无论 S1 是否支持 H，按预注册方案完整运行 M1，不得根据中间结果更换 target 对。
8. 运行 M2 和 C1，完成强制证据链。
9. 主结论冻结后，再运行 R1 和 U1 扩展。
10. 输出完整报告、机器可读表格、失败清单和所有 run manifests。

## 18. 最终结论允许的表述

- 若满足第 13.4 节全部判据：`z(H)` 是比 `z(gap)` 更稳定的 target 脆弱性预测子，并且其效应随训练轮数/数据剂量增强。
- 若 `z(gap)` 匹配对稳定分化：gap 具有独立预测贡献，原 `z(H)+z(gap)` 需要重新估计权重而不是删除 gap。
- 若 c35 类现象由 source-target MMD、梯度对齐或 cosine 解释：H 不是唯一机制，应报告交互模型。
- 若纯数据放大无效、更新放大有效：生成数据中毒本身较弱，强效果来自混合模型中毒。
- 若攻击停止后快速恢复：攻击属于持续暴露依赖的暂态可用性攻击。
- 若 global accuracy 或非 target 类同时大幅下降：攻击不是隐蔽定向攻击，而是普通全局退化。

## 19. 交付验收清单

- [ ] 实际运行环境、代码状态和中间产物来源已冻结。
- [ ] 所有代码快照、数据、模型、生成器和 partition 已记录唯一摘要或 hash。
- [ ] class map 和 feature shape 已从实际数据加载器导出并验证。
- [ ] `condition/source/train` 三类标签语义已自动断言。
- [ ] 5 个 matched benign 60-round runs 已完成。
- [ ] 固定注入量与嵌套剂量测试已通过。
- [ ] S1、M1、M2、C1 强制阶段无静默缺失。
- [ ] 所有失败 run 均有日志和失败原因。
- [ ] 统计脚本按预注册模型运行，包含去除 c49 的敏感性分析。
- [ ] 报告明确区分纯数据中毒和更新级混合攻击。
- [ ] 结论用语符合第 18 节证据边界。
