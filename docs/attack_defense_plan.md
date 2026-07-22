# UCF101 + DTM-GAN 攻击与防御 Baseline 计划

> 起草日期：2026-07-19
> 当前分支：`temporal-adaptive-gan-evaluation`
> 状态：计划草案（待确认）

## 1. 背景与目标

在已有的 **UCF101 + dtm_poison_gan** 工作基础上，补齐**攻击**与**防御**两条线，跑通一个可对照、可消融的完整 baseline。

**研究定位：** 多模态联邦学习（UCF101，audio + video）下的数据中毒攻击与鲁棒聚合防御。

### 1.1 攻击构想（两条路线）

| 编号 | 攻击类型 | 核心思路 |
| --- | --- | --- |
| **A1** | 标签翻转数据中毒 | 用 GAN 合成数据 + **错误/目标标签**，直接灌入恶意客户端，构成标签翻转攻击 |
| **A2** | 后门攻击（Backdoor） | 利用合成数据"贴近全局分布"的优势，在其中**悄然植入后门 trigger**，训练后执行后门攻击 |

### 1.2 防御构想（一条 pipeline，四段）

```
客户端上传 update
      │
      ▼
[D1] 梯度检查  ──►  [D2] 客户端打分(信任分)
                          │
                          ▼
                   [D3] 信任分加权聚合（替代按样本数加权）
                          │
                          ▼
              [D4] 分数过低 → 遗忘学习规避其梯度
```

---

## 2. 现状评估（复用 vs 新建）

### 2.1 可直接复用（节省时间）

- **dtm_poison_gan 生成器已跑通**：50 epoch 完整训练，`target_success_rate ≈ 1.0`；TSTR 下游评估 final50 = **57.15%**、ckpt10 = 50.36%、全局基线 75.62%。
  - 生成脚本：`generate_dtm_poison_features.py`（可生成 clean_label + balanced 合成特征，每类 100 共 5100 样本）
  - 质量报告：`fed_multimodal/Local/results/dtm_poison_gan/GENERATOR_QUALITY_REPORT.md`
- **联邦训练框架成熟**：`fed_multimodal/trainers/server_trainer.py`
  - 已有 `average_weights`（按样本数加权）、`sample_clients`、`update_weights` / `update_gradients`
  - 支持 `fed_avg / fed_prox / fed_opt / scaffold / fed_rs`
- **UCF101 实验脚手架完整**：`fed_multimodal/experiment/ucf101/`（51 类，train 4893 / test 1944，audio `[B,500,80]`、video `[B,10,1280]`）。

### 2.2 几乎从零新建（吃时间）

- **`label_flip_attack.py` 是 UCI-HAR 专用**：写死 acc/gyro 双模态、`src_label=1 → dst_label=2`，**不能用于 UCF101 的 audio/video**，逻辑可借鉴但需重写。
- **后门攻击**：全仓库无 `backdoor / trigger` 相关代码 —— 完全新建。
- **整套防御**：全仓库无 `unlearn / trust_score / gradient_check / robust_agg` —— 完全新建：
  - 现有 `average_weights` 仅按样本数加权，没有信任分机制。
  - 没有任何梯度异常检测 / 客户端打分 / 遗忘学习模块。

> **结论：** 攻击侧重写 A1 + 全新 A2；防御侧 D1–D4 全新建。难点不在写代码，而在**实验矩阵**（详见 §5）。

---

## 3. 时间估计与前提假设

**约 7–8 周（全职、单卡、实验串行）。**

前提假设：
- 全职投入；若半职，时间大致 ×2。
- 本机按 `CLAUDE.md` 约定**串行**跑联邦实验（`taskset -c 1-30`，单 GPU）。
- UCF101 联邦单轮训练本身耗时可观，是计划跨 8 周而非 3–4 周的主因。

---

## 4. 周计划表

| 周 | 主题 | 关键工作 | 交付物 |
| --- | --- | --- | --- |
| **W1** | 基线复现 + 投毒管线 | ① 跑通**干净 UCF101 联邦基线**（fed_avg），记录 clean acc 作为对照锚点；② 写通用**恶意客户端注入管线**：把 GAN 合成特征 + 指定标签灌进指定 client 的 train split，按 `client_id` 后缀选恶意客户端 | 干净基线 acc；可复用的 poison injection 工具（与攻击类型无关） |
| **W2** | A1 标签翻转 | 重写 UCF101 版 label-flip（合成数据打**错误/目标标签**注入恶意客户端）；扫描 poison ratio / 恶意客户端比例；评估：全局 clean acc 下降幅度 + 对目标类的破坏 | A1 完整结果 + 曲线 |
| **W3** | A2 后门（设计） | 设计 **trigger**（feature 空间：建议在 video 某几帧 / audio 某段叠加固定扰动 pattern，或借 GAN 生成"带 trigger 的合成样本"）；构造后门数据（trigger + 目标标签） | trigger 注入器 + 后门数据集 |
| **W4** | A2 后门（跑通 + 评估） | 后门客户端训练；评估 **ASR**（attack success rate）与 clean acc；扫描 trigger 强度 / poison ratio | A2 完整结果；两种攻击对照齐 |
| **W5** | 防御 D1+D2+D3：鲁棒聚合 | server 端对每轮上传 update 做**梯度检查**（建议先上 cosine 异常 / FoolsGold 式聚类，简单有效）→ 输出**客户端信任分** → 用信任分做**加权聚合**（替换 `average_weights` 的样本数加权）。在 A1、A2 上验证防御效果 | robust aggregation 模块 + 防御前后对比 |
| **W6** | 防御 D4：遗忘学习 | 集成"低分客户端遗忘"（**方案需先定，见 §6.1**）；与 W5 降权做消融对比 | 遗忘模块 + 消融 |
| **W7** | 完整实验矩阵 + 消融 | 攻击{A1, A2} × 防御{无, 鲁棒聚合, +遗忘} × 关键超参 × 多 seed；统计显著性 | 完整结果表 |
| **W8** | 分析 / 可视化 / 报告 + buffer | 结果图、失败案例分析、报告或论文初稿；吸收实验意外 | 可交付报告 |

---

## 5. 实验矩阵预览（W7 目标）

最小关键格（先跑通）：

|  | 无防御 | 鲁棒聚合 (D1–D3) | + 遗忘 (D1–D4) |
| --- | --- | --- | --- |
| **干净（无攻击）** | ✓ | ✓ | ✓ |
| **A1 标签翻转** | ✓ | ✓ | ✓ |
| **A2 后门** | ✓ | ✓ | ✓ |

> 每格至少 2–3 个 seed；再补 poison ratio / trigger 强度 / 恶意客户端比例的扫描曲线。串行单机是这个矩阵吃时间的主因。

---

## 6. 关键风险与待澄清问题

### 6.1 ⚠️ "遗忘学习规避梯度"必须先钉死含义

真正的 **machine unlearning**（influence function / 重训抹除）成本高、研究深度大；FL 防御里更常用的是**直接剔除 / 强降权恶意客户端 update**。

- 若采用后者：W6 可压缩到半周。
- 若坚持前者（有论文卖点）：W6 可能扩到 1.5–2 周，并入 W7 buffer。
- **建议：** 先按"剔除 + 降权"跑通，再决定是否升级到真 unlearning。

### 6.2 ⚠️ 后门 trigger 设计是 A2 成败关键

"在合成数据里悄悄加后门"听起来优雅，但 feature 空间的 trigger 需同时满足：

- **ASR 高**（攻击有效）
- **不破坏 clean acc**（隐蔽）
- **不易被梯度检查抓到**（对抗 D1）

W3 给 trigger 留了整周，**不要压缩**。

### 6.3 评估指标清单（提前对齐）

- 攻击侧：clean acc 下降幅度、ASR、目标类破坏率。
- 防御侧：抵御后 clean acc 恢复程度、ASR 压制程度、误伤率（把良性客户端判为恶意的比例）。

---

## 7. 加速建议

- A1、A2 **共享 W1 的注入管线与评估框架**，避免重复造轮子。
- 防御先做 D1+D2+D3（W5，相对成熟、出效果快），D4 作为进阶。
- 实验矩阵先跑"关键格"（2 攻击 × 3 防御档 × 1 超参 × 2 seed），再补全，避免 W7 卡死。

---

## 8. 关键代码位置索引

| 用途 | 路径 |
| --- | --- |
| GAN 生成器主体 | `fed_multimodal/generator/gan_generator.py` |
| GAN 质量评估 | `fed_multimodal/generator/eval_gan_quality.py` |
| 现有标签翻转（UCI-HAR，需重写） | `fed_multimodal/generator/label_flip_attack.py` |
| 合成数据生成 | `generate_dtm_poison_features.py` |
| 合成数据下游评估 (TSTR) | `fed_multimodal/Local/train_synthetic.py` |
| 服务器端聚合（防御改造入口） | `fed_multimodal/trainers/server_trainer.py` |
| 客户端训练 | `fed_multimodal/trainers/fed_avg_trainer.py` 等 |
| 指标计算 | `fed_multimodal/trainers/evaluation.py` |
| UCF101 实验入口 | `fed_multimodal/experiment/ucf101/train.py` |
| 数据加载与路径映射 | `fed_multimodal/dataloader/dataload_manager.py` |

### 运行环境备忘

- conda env：`poigan`（`/home/xp/anaconda3/envs/poigan/bin/python`，torch 1.13+cu117, py3.9）
- 必须 `export PYTHONPATH=/home/xp/fedpoi`（第二版代码只在 fedpoi）
- UCF101 split 软链嵌套坑：训练/评估加 `--dataset_dir /home/xp/fedpoigan/fed_multimodal/datasets/ucf101`
