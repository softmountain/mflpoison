# UCF101 合成数据特征碰撞攻击实验报告

> 日期：2026-07-19
> 分支：`temporal-adaptive-gan-evaluation`
> 状态：方式1（10 组）+ 方式2（45 组）= 55 组全部完成

## 1. 目标与假设

验证核心假设：**若某 source 类的 GAN 合成数据在收敛全局模型 M\* 的倒数第二层特征空间与某 target 真实类高度相似，则把"source 合成内容 + target 标签"注入恶意客户端后，攻击效果（target 类被破坏 / source 被误判为 target）更强。**

即"合成数据与真实数据的特征相似度 → 可解释攻击效果"。

## 2. 攻击语义（特征碰撞，非纯标签翻转）

对 4 个恶意客户端（10 客户端的 40%），把其 **target 类真实样本的 feature 替换为 source 类 GAN 合成特征，label 保持 target**。模型因此学到"source 合成特征 → target"的映射。

- 合成数据源：dtm_poison_gan final50（TSTR 57%，5100 样本 51 类 × 100）
- 注入：恶意客户端 0,1,2,3 的 target 类条目，feature 替换为 source 合成（label 不变）
- 中毒比例：恶意客户端 target 类样本 100% 替换

## 3. 实验设置

| 项 | 值 |
|---|---|
| 数据 | UCF101 51 类子集，从 feature.pkl 重新 Dirichlet 划分 10 客户端（alpha=1.0） |
| 收敛全局模型 M\* | fed_avg, fuse_base hid128, sample_rate=0.2, **test acc 63.5%**（sample_rate 0.2 + non-IID 的 FL 天花板，未达 70%，已确认接受） |
| penultimate 特征 | `x_mm`，fuse_base 下 dim=768 |
| 攻击 | 从 M\* load 出发，注入后继续 15 轮（sample_rate 0.2, eval_every 5） |
| source 类 | TSTR per-class recall **最好5**（SkyDiving/Rafting/PlayingFlute/PlayingSitar/TableTennisShot）+ **最差5**（BrushingTeeth/HandstandWalking/Shotput/PlayingDhol/BodyWeightSquats）= 10 个 |
| target 选择 | 方式1：source 合成特征中心最近的真实类（10 组）；方式2：target ∈ TSTR 最差5（45 组） |

## 4. source/target 选择（P0 TSTR per-class）

final50 合成数据 TSTR per-class recall 排序后取极端 5 类：

- **source_best5（合成质量最好）**: 41 SkyDiving(100%), 38 Rafting(100%), 36 PlayingFlute(100%), 37 PlayingSitar(97.7%), 46 TableTennisShot(97.4%)
- **source_worst5（合成质量最差）**: 13 BrushingTeeth(2.8%), 26 HandstandWalking(2.9%), 40 Shotput(15.2%), 35 PlayingDhol(16.3%), 9 BodyWeightSquats(16.7%)

## 5. 方式1 结果（特征驱动 target，10 组）

每组 target = source 合成特征中心余弦最近的真实类（相似度 0.76–0.88）。

| source | target | cosine | s2t base→after | Δs2t | Δt_recall | Δacc |
|---|---|---|---|---|---|---|
| 13 | 39 | 0.798 | 22.2→44.4 | **+22.2** | +11.6 | +6.9 |
| 40 | 23 | 0.876 | 4.3→6.5 | +2.2 | -2.2 | +4.0 |
| 9 | 26 | 0.869 | 0.0→0.0 | 0.0 | +11.8 | +3.1 |
| 26 | 23 | 0.779 | 0.0→0.0 | 0.0 | +6.7 | +2.9 |
| 35 | 31 | 0.799 | 0.0→0.0 | 0.0 | -2.9 | +5.8 |
| 37 | 2 | 0.759 | 0.0→0.0 | 0.0 | **-21.9** | +3.9 |
| 36 | 26 | 0.762 | 0.0→0.0 | 0.0 | +0.0 | +3.2 |
| 38 | 26 | 0.800 | 0.0→0.0 | 0.0 | +5.9 | +3.5 |
| 41 | 28 | 0.822 | 0.0→0.0 | 0.0 | **+30.4** | +3.1 |
| 46 | 30 | 0.844 | 0.0→0.0 | 0.0 | **+30.8** | +4.9 |

**相关性（cosine 相似度 vs 攻击效果，n=10）：**

| 攻击效果度量 | Pearson | Spearman |
|---|---|---|
| source→target 误判（最直接） | **-0.06** | 0.19 |
| target 类被破坏（-Δt_recall） | -0.45 | -0.45 |
| source 类被误判（-Δs_recall） | 0.07 | 0.09 |
| 全局 acc 下降 | 0.02 | -0.05 |

### 5.1 关键观察

1. **source→target 误判（s2t）大多为 0**：10 组中 8 组 Δs2t=0。攻击**没有**让 source 真实样本被误判为 target。仅 source13（+22.2）和 source40（+2.2）有效。
2. **但攻击确实改变了 target 类**：Δt_recall 大幅波动（source41/46 **+30**，source37 **-22**）。攻击更像"激活或压制 target 类预测"，而非"把 source 误判成 target"。
3. **全局 acc 全部上升**（+2.9~+6.9）：15 轮继续训练让模型整体进步，掩盖了定向攻击的破坏性。
4. **相似度与攻击效果几乎无关**（s2t pearson -0.06）。假设在方式1 中**不成立**。

### 5.2 为什么 s2t 多为 0？

- **best5 source**（36/37/38/41/46）合成质量好，但其真实样本分类极准（recall 90-100%），即便注入合成内容+target 标签，模型对真实 source 仍分类正确，不误判为 target。
- **worst5 source**（9/13/26/35/40）合成数据偏离真实分布（final50 audio std 3.9 远超真实 0.77），注入的合成特征与真实 source 特征差异大，模型学到的"合成特征→target"映射对真实 source 样本不生效。
- s2t 升幅实际取决于 **source 真实样本与 target 的天然混淆度**（baseline s2t 是否非 0），而非合成相似度。source13 baseline s2t=22.2（本就与 target 39 混淆），攻击加剧到 44.4。

## 6. 方式2 结果（TSTR 驱动 target，45 组）

target 固定为 TSTR 最差 5 类（9,13,26,35,40），10 source × 5 target = 45 组。

**source→target 误判（s2t）几乎不发生**：45 组中仅 1 组 Δs2t>0（source41→26，+3.2）。特征碰撞在方式2 同样无法建立 source→target 误判。

**Δtarget_recall 分布**：破坏(<-1) **26 组**，激活(>+1) 14 组，微变 5 组。方式2 以**破坏 target** 为主。

**按 target 分组的平均 Δtarget_recall**（揭示真正驱动因素）：

| target | 类名 | TSTR recall | 平均 Δt_recall | 效果 |
|---|---|---|---|---|
| 40 | Shotput | 15.2% | **-61.6** | 强烈破坏 |
| 13 | BrushingTeeth | 2.8% | -24.4 | 破坏 |
| 35 | PlayingDhol | 16.3% | +2.5 | 微 |
| 9 | BodyWeightSquats | 16.7% | +1.1 | 微 |
| 26 | HandstandWalking | 2.9% | +11.1 | 激活 |

**关键洞察**：攻击效果（破坏 vs 激活）**主要由 target 类本身的固有难度决定**，与 source 或相似度无关。target 40（Shotput）被强烈破坏（-61.6），无论哪个 source 注入。

## 7. 全量相关性（55 组）与最终结论

**相关性（cosine 相似度 vs 攻击效果，n=55）：**

| 攻击效果度量 | Pearson | Spearman | 解读 |
|---|---|---|---|
| source→target 误判 | +0.11 | -0.07 | 无相关 |
| target 类被破坏 | **-0.33** | **-0.42** | **负相关（反假设）** |
| source 类被误判 | -0.18 | -0.14 | 弱负相关 |
| 全局 acc 下降 | +0.13 | +0.14 | 弱正相关 |

### 7.1 核心假设被证伪

**"penultimate 特征相似度 → 攻击效果"假设不成立。** 55 组中，相似度与最直接的攻击度量（source→target 误判）几乎无关（pearson 0.11），与 target 破坏负相关（-0.33，反方向）。

### 7.2 真正的驱动因素：target 固有难度

攻击效果由 **target 类的基线分类难度**驱动，而非 source-target 相似度：
- 模型本就不擅长的 target（TSTR recall 最低的 Shotput/BrushingTeeth）被破坏最严重（target 40 recall 降 61.6）；
- 注入 source 内容 + target 标签，让模型对"本就脆弱"的 target 类更混乱；
- 相似度是**混淆变量**：相似度高的组合其 target 往往是模型较擅长的（baseline recall 较高、难破坏），故相似度 vs 破坏呈负相关。

### 7.3 为什么 source→target 误判几乎为 0（55 组仅 3 组）

- **best5 source**（合成好）真实样本分类极准（recall 90-100%），即便注入合成内容+target 标签，模型仍正确分类真实 source，不误判为 target；
- **worst5 source** 合成数据偏离真实（final50 audio std 3.9 vs 真实 0.77），注入的合成特征与真实 source 差异大，学到的映射对真实 source 不生效；
- s2t 升幅实际取决于 baseline 天然混淆度（source 真实样本本就被误判为 target 的比例），而非合成相似度。

### 7.4 攻击是否有效？

**有效，但机制与假设不同。** 攻击对脆弱 target 类造成显著破坏（target 40 recall -61.6），这是有意义的联邦中毒效果，但其强弱由 target 难度而非合成相似度决定。

### 7.5 启示

1. 评估 GAN 合成数据中毒攻击，不应假设"特征相似度→攻击效果"；
2. 特征碰撞攻击对"模型薄弱类"破坏最强——防御应重点关注低 recall 类的梯度；
3. 欲验证相似度假设，需**控制 target 难度**（同一 target 下扫 source 相似度），而非跨 target 比较。

### 7.6 baseline 校正：扣除自然漂移后的真实攻击效果

§7.1–7.4 的 Δt_recall / Δacc 把"攻击破坏"和"继续训练 15 轮的自然漂移"混在一起。为分离两者，跑了一次**无攻击 baseline**：完全相同的 M\*、15 轮、sample_rate 0.2、客户端划分，但 **0 恶意客户端**（`train_label_flip.py --malicious_clients ""`，不加载合成数据、不注入）。每个攻击组的净效果 = 攻击 Δ − baseline 该类自然漂移（per-class recall 用对应类的 drift，acc 用全局 drift）。

**baseline 自然漂移（令人意外）：**
- 全局 acc **自然上升 +4.94**（63.48 → 68.42）——15 轮继续训练，模型还在变好；
- 多个类 recall 自然**崩塌**：class 0 ApplyEyeMakeup **-72.7**（77→4.5）、class 11 BoxingPunchingBag -51.0、class 15 CricketBowling -47.2、class 4 BalanceBeam -41.9。这是 sample_rate 0.2 + Dirichlet non-IID 下，少数客户端被反复采样导致的类间遗忘，与攻击无关。

**关键 target 的校正（旧 Δt_recall → NET Δt_recall）：**

| target | 类 | 自然漂移 | 旧 Δt | **NET Δt** | 解读 |
|---|---|---|---|---|---|
| 40 | Shotput | -13.0 | -61.6 | **-48.6** | 攻击被高估 13，但仍是最大破坏 |
| 31 | MoppingFloor | +35.3 | -2.9 | **-38.2** | ⚠ 被自然上升掩盖的强破坏 |
| 9 | BodyWeightSquats | +26.7 | +1.1 | **-25.6** | ⚠ 被自然上升掩盖，实为有效攻击 |
| 28 | IceDancing | +30.4 | +30.4 | **+0.0** | ⚠ "激活"完全是自然漂移，与攻击无关 |
| 2 | Archery | -31.7 | -21.9 | **+9.8** | 旧"破坏"大半是自然下降 |
| 13 | BrushingTeeth | -11.1 | -24.4 | **-13.3** | 攻击破坏被高估约一半 |

**相关性（55 组，cosine vs target 破坏）：** OLD **-0.326** → NET **-0.234**。校正后负相关减弱（自然漂移贡献了部分反向信号），但方向不变——**相似度假设依然被证伪**，且现在排除了一个混淆源。注：cosine vs acc 下降不变（0.131→0.131），因为 acc 漂移是全局常数，对所有组扣除相同值，Pearson 对常数平移不变。

**对 §7 结论的修正：**
1. §7.2 称"target 40 破坏 -61.6 由 target 难度驱动"——部分正确，但其中 -13 是纯自然漂移，真正攻击贡献 **-48.6**；所谓"target 难度"里混入了 sample_rate 0.2 的随机遗忘效应；
2. §7.4 称"攻击有效"——校正后**更强**：target 31/9 的强破坏此前被自然上升完全掩盖，看起来像无效/微弱攻击；
3. 一个普遍教训：**联邦中毒实验必须配无攻击 baseline**，否则 non-IID + 低采样率下的类间遗忘会被误记为攻击效果，或反过来掩盖真实攻击。

## 8. 交付物

- 选择：`Local/results/poison_attack/source_target_selection.json`、`combos_method{1,2}.json`
- 类中心：`Local/results/poison_attack/centers/{real,synth}_centers.pt`
- M\*：`Local/results/poison_attack/M_star.pt`
- 方式1 分析：`Local/results/poison_attack/analysis_m1/{poison_summary.json,poison_summary.md,scatter_*.png}`
- 攻击结果：`fed_multimodal/result/fed_avg_poison/ucf101/.../source*_target*_m{1,2}/result.json`
- **无攻击 baseline**：`fed_multimodal/result/fed_avg_poison/ucf101/.../baseline_noattack/result.json`（含每轮 per_class_recall 的 eval_history）
- **baseline 校正分析**：`Local/results/poison_attack/analysis_all/poison_summary_attributed.{json,md}`
- 归因脚本：`experiment/ucf101/attribute_against_baseline.py`；baseline 跑法：`train_label_flip.py --malicious_clients ""`（其余参数同攻击组）

## 附录 A: 55 组完整数据表

色盘 Okabe-Ito（CVD 安全，dataviz validator PASS）。`base` = M\*（攻击前），`4/9/14` = 攻击轮次（eval_every 5）。折线图列点开见大图。

| method | source | target | cosine | s2t base→after | Δs2t | Δt_recall | Δs_recall | Δacc | inject | 折线图 |
|---|---|---|---|---|---|---|---|---|---|---|
| m1 | 9 BodyWeightSquats | 26 HandstandWalking | 0.869 | 0.0→0.0 | +0.0 | +11.8 | +10.0 | +3.1 | 29 | [📈](line_plots/line_s9_t26_m1.png) |
| m1 | 13 BrushingTeeth | 39 ShavingBeard | 0.798 | 22.2→44.4 | +22.2 | +11.6 | -16.7 | +6.9 | 29 | [📈](line_plots/line_s13_t39_m1.png) |
| m1 | 26 HandstandWalking | 23 HammerThrow | 0.779 | 0.0→0.0 | +0.0 | +6.7 | +8.8 | +2.9 | 35 | [📈](line_plots/line_s26_t23_m1.png) |
| m1 | 35 PlayingDhol | 31 MoppingFloor | 0.799 | 0.0→0.0 | +0.0 | -2.9 | +30.6 | +5.8 | 23 | [📈](line_plots/line_s35_t31_m1.png) |
| m1 | 36 PlayingFlute | 26 HandstandWalking | 0.762 | 0.0→0.0 | +0.0 | +0.0 | +14.6 | +3.2 | 29 | [📈](line_plots/line_s36_t26_m1.png) |
| m1 | 37 PlayingSitar | 2 Archery | 0.759 | 0.0→0.0 | +0.0 | -21.9 | +0.0 | +3.9 | 43 | [📈](line_plots/line_s37_t2_m1.png) |
| m1 | 38 Rafting | 26 HandstandWalking | 0.800 | 0.0→0.0 | +0.0 | +5.9 | -3.6 | +3.5 | 29 | [📈](line_plots/line_s38_t26_m1.png) |
| m1 | 40 Shotput | 23 HammerThrow | 0.876 | 4.3→6.5 | +2.2 | -2.2 | +4.3 | +4.0 | 35 | [📈](line_plots/line_s40_t23_m1.png) |
| m1 | 41 SkyDiving | 28 IceDancing | 0.822 | 0.0→0.0 | +0.0 | +30.4 | -3.2 | +3.1 | 51 | [📈](line_plots/line_s41_t28_m1.png) |
| m1 | 46 TableTennisShot | 30 LongJump | 0.844 | 0.0→0.0 | +0.0 | +30.8 | -2.6 | +4.9 | 38 | [📈](line_plots/line_s46_t30_m1.png) |
| m2 | 9 BodyWeightSquats | 13 BrushingTeeth | 0.540 | 0.0→0.0 | +0.0 | -25.0 | -3.3 | +2.5 | 35 | [📈](line_plots/line_s9_t13_m2.png) |
| m2 | 9 BodyWeightSquats | 26 HandstandWalking | 0.869 | 0.0→0.0 | +0.0 | +11.8 | +10.0 | +3.1 | 29 | [📈](line_plots/line_s9_t26_m2.png) |
| m2 | 9 BodyWeightSquats | 35 PlayingDhol | 0.598 | 3.3→0.0 | -3.3 | +24.5 | -3.3 | +5.2 | 24 | [📈](line_plots/line_s9_t35_m2.png) |
| m2 | 9 BodyWeightSquats | 40 Shotput | 0.733 | 13.3→0.0 | -13.3 | -80.4 | -10.0 | +1.8 | 36 | [📈](line_plots/line_s9_t40_m2.png) |
| m2 | 13 BrushingTeeth | 9 BodyWeightSquats | 0.724 | 0.0→0.0 | +0.0 | -3.3 | -30.6 | +3.5 | 13 | [📈](line_plots/line_s13_t9_m2.png) |
| m2 | 13 BrushingTeeth | 26 HandstandWalking | 0.750 | 0.0→0.0 | +0.0 | +11.8 | -25.0 | +5.9 | 29 | [📈](line_plots/line_s13_t26_m2.png) |
| m2 | 13 BrushingTeeth | 35 PlayingDhol | 0.527 | 0.0→0.0 | +0.0 | -4.1 | -36.1 | +4.5 | 24 | [📈](line_plots/line_s13_t35_m2.png) |
| m2 | 13 BrushingTeeth | 40 Shotput | 0.647 | 0.0→0.0 | +0.0 | -60.9 | -27.8 | +6.5 | 36 | [📈](line_plots/line_s13_t40_m2.png) |
| m2 | 26 HandstandWalking | 9 BodyWeightSquats | 0.667 | 8.8→5.9 | -2.9 | -6.7 | +11.8 | +6.4 | 13 | [📈](line_plots/line_s26_t9_m2.png) |
| m2 | 26 HandstandWalking | 13 BrushingTeeth | 0.489 | 0.0→0.0 | +0.0 | -16.7 | +5.9 | +5.6 | 35 | [📈](line_plots/line_s26_t13_m2.png) |
| m2 | 26 HandstandWalking | 35 PlayingDhol | 0.543 | 0.0→0.0 | +0.0 | +12.2 | +11.8 | +5.6 | 24 | [📈](line_plots/line_s26_t35_m2.png) |
| m2 | 26 HandstandWalking | 40 Shotput | 0.673 | 23.5→0.0 | -23.5 | -63.0 | +11.8 | +5.0 | 36 | [📈](line_plots/line_s26_t40_m2.png) |
| m2 | 35 PlayingDhol | 9 BodyWeightSquats | 0.669 | 0.0→0.0 | +0.0 | -3.3 | +2.0 | +3.3 | 13 | [📈](line_plots/line_s35_t9_m2.png) |
| m2 | 35 PlayingDhol | 13 BrushingTeeth | 0.483 | 0.0→0.0 | +0.0 | -19.4 | +28.6 | +4.6 | 35 | [📈](line_plots/line_s35_t13_m2.png) |
| m2 | 35 PlayingDhol | 26 HandstandWalking | 0.695 | 0.0→0.0 | +0.0 | +14.7 | +26.5 | +6.6 | 29 | [📈](line_plots/line_s35_t26_m2.png) |
| m2 | 35 PlayingDhol | 40 Shotput | 0.662 | 0.0→0.0 | +0.0 | -52.2 | +18.4 | +4.9 | 36 | [📈](line_plots/line_s35_t40_m2.png) |
| m2 | 36 PlayingFlute | 9 BodyWeightSquats | 0.684 | 0.0→0.0 | +0.0 | +0.0 | +6.2 | +6.8 | 13 | [📈](line_plots/line_s36_t9_m2.png) |
| m2 | 36 PlayingFlute | 13 BrushingTeeth | 0.686 | 2.1→0.0 | -2.1 | -16.7 | +4.2 | +4.6 | 35 | [📈](line_plots/line_s36_t13_m2.png) |
| m2 | 36 PlayingFlute | 26 HandstandWalking | 0.762 | 0.0→0.0 | +0.0 | +0.0 | +14.6 | +3.2 | 29 | [📈](line_plots/line_s36_t26_m2.png) |
| m2 | 36 PlayingFlute | 35 PlayingDhol | 0.658 | 0.0→0.0 | +0.0 | -8.2 | +12.5 | +5.0 | 24 | [📈](line_plots/line_s36_t35_m2.png) |
| m2 | 36 PlayingFlute | 40 Shotput | 0.677 | 0.0→0.0 | +0.0 | -69.6 | +16.7 | +3.0 | 36 | [📈](line_plots/line_s36_t40_m2.png) |
| m2 | 37 PlayingSitar | 9 BodyWeightSquats | 0.663 | 0.0→0.0 | +0.0 | +0.0 | +0.0 | +4.3 | 13 | [📈](line_plots/line_s37_t9_m2.png) |
| m2 | 37 PlayingSitar | 13 BrushingTeeth | 0.466 | 0.0→0.0 | +0.0 | -30.6 | +0.0 | +6.1 | 35 | [📈](line_plots/line_s37_t13_m2.png) |
| m2 | 37 PlayingSitar | 26 HandstandWalking | 0.639 | 0.0→0.0 | +0.0 | +14.7 | +0.0 | +4.7 | 29 | [📈](line_plots/line_s37_t26_m2.png) |
| m2 | 37 PlayingSitar | 35 PlayingDhol | 0.748 | 2.3→0.0 | -2.3 | +4.1 | +0.0 | +5.3 | 24 | [📈](line_plots/line_s37_t35_m2.png) |
| m2 | 37 PlayingSitar | 40 Shotput | 0.606 | 0.0→0.0 | +0.0 | -67.4 | +0.0 | +3.9 | 36 | [📈](line_plots/line_s37_t40_m2.png) |
| m2 | 38 Rafting | 9 BodyWeightSquats | 0.629 | 0.0→0.0 | +0.0 | +16.7 | -7.2 | +3.8 | 13 | [📈](line_plots/line_s38_t9_m2.png) |
| m2 | 38 Rafting | 13 BrushingTeeth | 0.511 | 0.0→0.0 | +0.0 | -22.2 | -3.6 | +4.9 | 35 | [📈](line_plots/line_s38_t13_m2.png) |
| m2 | 38 Rafting | 26 HandstandWalking | 0.800 | 0.0→0.0 | +0.0 | +5.9 | -3.6 | +3.5 | 29 | [📈](line_plots/line_s38_t26_m2.png) |
| m2 | 38 Rafting | 35 PlayingDhol | 0.684 | 0.0→0.0 | +0.0 | +0.0 | -3.6 | +3.4 | 24 | [📈](line_plots/line_s38_t35_m2.png) |
| m2 | 38 Rafting | 40 Shotput | 0.754 | 0.0→0.0 | +0.0 | -56.5 | -7.2 | +2.6 | 36 | [📈](line_plots/line_s38_t40_m2.png) |
| m2 | 40 Shotput | 9 BodyWeightSquats | 0.722 | 0.0→0.0 | +0.0 | +0.0 | +6.5 | +5.4 | 13 | [📈](line_plots/line_s40_t9_m2.png) |
| m2 | 40 Shotput | 13 BrushingTeeth | 0.483 | 0.0→0.0 | +0.0 | -27.8 | -50.0 | +4.3 | 35 | [📈](line_plots/line_s40_t13_m2.png) |
| m2 | 40 Shotput | 26 HandstandWalking | 0.847 | 0.0→0.0 | +0.0 | +11.8 | +4.3 | +7.1 | 29 | [📈](line_plots/line_s40_t26_m2.png) |
| m2 | 40 Shotput | 35 PlayingDhol | 0.588 | 0.0→0.0 | +0.0 | -14.3 | -15.2 | +3.7 | 24 | [📈](line_plots/line_s40_t35_m2.png) |
| m2 | 41 SkyDiving | 9 BodyWeightSquats | 0.633 | 0.0→0.0 | +0.0 | -10.0 | -3.2 | +4.7 | 13 | [📈](line_plots/line_s41_t9_m2.png) |
| m2 | 41 SkyDiving | 13 BrushingTeeth | 0.454 | 0.0→0.0 | +0.0 | -33.3 | -9.7 | +7.4 | 35 | [📈](line_plots/line_s41_t13_m2.png) |
| m2 | 41 SkyDiving | 26 HandstandWalking | 0.803 | 0.0→3.2 | +3.2 | +26.5 | +0.0 | +6.4 | 29 | [📈](line_plots/line_s41_t26_m2.png) |
| m2 | 41 SkyDiving | 35 PlayingDhol | 0.575 | 0.0→0.0 | +0.0 | +12.2 | -3.2 | +2.3 | 24 | [📈](line_plots/line_s41_t35_m2.png) |
| m2 | 41 SkyDiving | 40 Shotput | 0.696 | 0.0→0.0 | +0.0 | -76.1 | +3.2 | +2.8 | 36 | [📈](line_plots/line_s41_t40_m2.png) |
| m2 | 46 TableTennisShot | 9 BodyWeightSquats | 0.751 | 0.0→0.0 | +0.0 | +16.7 | -5.1 | +2.6 | 13 | [📈](line_plots/line_s46_t9_m2.png) |
| m2 | 46 TableTennisShot | 13 BrushingTeeth | 0.524 | 0.0→0.0 | +0.0 | -27.8 | +2.6 | +4.4 | 35 | [📈](line_plots/line_s46_t13_m2.png) |
| m2 | 46 TableTennisShot | 26 HandstandWalking | 0.778 | 0.0→0.0 | +0.0 | +2.9 | -7.7 | +5.5 | 29 | [📈](line_plots/line_s46_t26_m2.png) |
| m2 | 46 TableTennisShot | 35 PlayingDhol | 0.696 | 0.0→0.0 | +0.0 | -4.1 | +0.0 | +3.7 | 24 | [📈](line_plots/line_s46_t35_m2.png) |
| m2 | 46 TableTennisShot | 40 Shotput | 0.784 | 0.0→0.0 | +0.0 | -28.3 | -2.6 | +6.6 | 36 | [📈](line_plots/line_s46_t40_m2.png) |

## 附录 B: 55 组折线图（source/target recall + global acc 随攻击轮次）


### B.1 方式1（特征驱动 target，10 组）

![s9_t26_m1](images/line_s9_t26_m1.png)
*source 9 (BodyWeightSquats) → target 26 (HandstandWalking), cosine=0.869*

![s13_t39_m1](images/line_s13_t39_m1.png)
*source 13 (BrushingTeeth) → target 39 (ShavingBeard), cosine=0.798*

![s26_t23_m1](images/line_s26_t23_m1.png)
*source 26 (HandstandWalking) → target 23 (HammerThrow), cosine=0.779*

![s35_t31_m1](images/line_s35_t31_m1.png)
*source 35 (PlayingDhol) → target 31 (MoppingFloor), cosine=0.799*

![s36_t26_m1](images/line_s36_t26_m1.png)
*source 36 (PlayingFlute) → target 26 (HandstandWalking), cosine=0.762*

![s37_t2_m1](images/line_s37_t2_m1.png)
*source 37 (PlayingSitar) → target 2 (Archery), cosine=0.759*

![s38_t26_m1](images/line_s38_t26_m1.png)
*source 38 (Rafting) → target 26 (HandstandWalking), cosine=0.800*

![s40_t23_m1](images/line_s40_t23_m1.png)
*source 40 (Shotput) → target 23 (HammerThrow), cosine=0.876*

![s41_t28_m1](images/line_s41_t28_m1.png)
*source 41 (SkyDiving) → target 28 (IceDancing), cosine=0.822*

![s46_t30_m1](images/line_s46_t30_m1.png)
*source 46 (TableTennisShot) → target 30 (LongJump), cosine=0.844*


### B.2 方式2（TSTR 驱动 target，45 组，按 target 分组）


#### target 9 (BodyWeightSquats) — 9 组

![s13_t9_m2](images/line_s13_t9_m2.png)
*source 13 (BrushingTeeth), cosine=0.724*

![s26_t9_m2](images/line_s26_t9_m2.png)
*source 26 (HandstandWalking), cosine=0.667*

![s35_t9_m2](images/line_s35_t9_m2.png)
*source 35 (PlayingDhol), cosine=0.669*

![s36_t9_m2](images/line_s36_t9_m2.png)
*source 36 (PlayingFlute), cosine=0.684*

![s37_t9_m2](images/line_s37_t9_m2.png)
*source 37 (PlayingSitar), cosine=0.663*

![s38_t9_m2](images/line_s38_t9_m2.png)
*source 38 (Rafting), cosine=0.629*

![s40_t9_m2](images/line_s40_t9_m2.png)
*source 40 (Shotput), cosine=0.722*

![s41_t9_m2](images/line_s41_t9_m2.png)
*source 41 (SkyDiving), cosine=0.633*

![s46_t9_m2](images/line_s46_t9_m2.png)
*source 46 (TableTennisShot), cosine=0.751*


#### target 13 (BrushingTeeth) — 9 组

![s9_t13_m2](images/line_s9_t13_m2.png)
*source 9 (BodyWeightSquats), cosine=0.540*

![s26_t13_m2](images/line_s26_t13_m2.png)
*source 26 (HandstandWalking), cosine=0.489*

![s35_t13_m2](images/line_s35_t13_m2.png)
*source 35 (PlayingDhol), cosine=0.483*

![s36_t13_m2](images/line_s36_t13_m2.png)
*source 36 (PlayingFlute), cosine=0.686*

![s37_t13_m2](images/line_s37_t13_m2.png)
*source 37 (PlayingSitar), cosine=0.466*

![s38_t13_m2](images/line_s38_t13_m2.png)
*source 38 (Rafting), cosine=0.511*

![s40_t13_m2](images/line_s40_t13_m2.png)
*source 40 (Shotput), cosine=0.483*

![s41_t13_m2](images/line_s41_t13_m2.png)
*source 41 (SkyDiving), cosine=0.454*

![s46_t13_m2](images/line_s46_t13_m2.png)
*source 46 (TableTennisShot), cosine=0.524*


#### target 26 (HandstandWalking) — 9 组

![s9_t26_m2](images/line_s9_t26_m2.png)
*source 9 (BodyWeightSquats), cosine=0.869*

![s13_t26_m2](images/line_s13_t26_m2.png)
*source 13 (BrushingTeeth), cosine=0.750*

![s35_t26_m2](images/line_s35_t26_m2.png)
*source 35 (PlayingDhol), cosine=0.695*

![s36_t26_m2](images/line_s36_t26_m2.png)
*source 36 (PlayingFlute), cosine=0.762*

![s37_t26_m2](images/line_s37_t26_m2.png)
*source 37 (PlayingSitar), cosine=0.639*

![s38_t26_m2](images/line_s38_t26_m2.png)
*source 38 (Rafting), cosine=0.800*

![s40_t26_m2](images/line_s40_t26_m2.png)
*source 40 (Shotput), cosine=0.847*

![s41_t26_m2](images/line_s41_t26_m2.png)
*source 41 (SkyDiving), cosine=0.803*

![s46_t26_m2](images/line_s46_t26_m2.png)
*source 46 (TableTennisShot), cosine=0.778*


#### target 35 (PlayingDhol) — 9 组

![s9_t35_m2](images/line_s9_t35_m2.png)
*source 9 (BodyWeightSquats), cosine=0.598*

![s13_t35_m2](images/line_s13_t35_m2.png)
*source 13 (BrushingTeeth), cosine=0.527*

![s26_t35_m2](images/line_s26_t35_m2.png)
*source 26 (HandstandWalking), cosine=0.543*

![s36_t35_m2](images/line_s36_t35_m2.png)
*source 36 (PlayingFlute), cosine=0.658*

![s37_t35_m2](images/line_s37_t35_m2.png)
*source 37 (PlayingSitar), cosine=0.748*

![s38_t35_m2](images/line_s38_t35_m2.png)
*source 38 (Rafting), cosine=0.684*

![s40_t35_m2](images/line_s40_t35_m2.png)
*source 40 (Shotput), cosine=0.588*

![s41_t35_m2](images/line_s41_t35_m2.png)
*source 41 (SkyDiving), cosine=0.575*

![s46_t35_m2](images/line_s46_t35_m2.png)
*source 46 (TableTennisShot), cosine=0.696*


#### target 40 (Shotput) — 9 组

![s9_t40_m2](images/line_s9_t40_m2.png)
*source 9 (BodyWeightSquats), cosine=0.733*

![s13_t40_m2](images/line_s13_t40_m2.png)
*source 13 (BrushingTeeth), cosine=0.647*

![s26_t40_m2](images/line_s26_t40_m2.png)
*source 26 (HandstandWalking), cosine=0.673*

![s35_t40_m2](images/line_s35_t40_m2.png)
*source 35 (PlayingDhol), cosine=0.662*

![s36_t40_m2](images/line_s36_t40_m2.png)
*source 36 (PlayingFlute), cosine=0.677*

![s37_t40_m2](images/line_s37_t40_m2.png)
*source 37 (PlayingSitar), cosine=0.606*

![s38_t40_m2](images/line_s38_t40_m2.png)
*source 38 (Rafting), cosine=0.754*

![s41_t40_m2](images/line_s41_t40_m2.png)
*source 41 (SkyDiving), cosine=0.696*

![s46_t40_m2](images/line_s46_t40_m2.png)
*source 46 (TableTennisShot), cosine=0.784*
