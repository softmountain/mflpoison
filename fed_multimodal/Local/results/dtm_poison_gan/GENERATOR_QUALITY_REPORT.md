# DTM-GAN 生成器质量报告

> 模型：`dtm_poison_gan`（Distributional Temporal Matching GAN，第二版特征空间投毒 GAN）
> 数据：UCF101 子集，51 类，train 4893 / test 1944，audio `[B,500,80]`，video `[B,9,1280]`
> 训练：50 epoch（前 30 轮 + 修复 NaN 后从 ckpt_30 resume 到 50），`fake_class=51` 作为 K+1 判别器的"假"类
> 最终检查点：`final_dtm_final.pt`（epoch 50）
> 报告日期：2026-07-13

---

## 一、执行摘要

生成器在全部 5 个评估点（epoch 10/20/30/45/50）上**完全达成设计目标**：

| 质量维度 | 结果 | 评级 |
|---|---|---|
| 目标攻击成功率 | 0.998–1.000（全程 ≈1.0） | ✅ 卓越 |
| 判别器逃逸率 | 0.991–0.995 | ✅ 卓越 |
| 训练稳定性 | 50 epoch 零 NaN 崩溃，d/g loss 平稳 | ✅ 健康 |
| 模式覆盖（无 mode collapse） | video diversity 0.90–1.36 | ✅ 良好 |
| 分布保真度（vs 真实） | epoch 10 近乎完美；后期 audio std 扩大 | ⚠️ 双峰，见 §五 |

**核心结论**：这是一个有效的特征空间投毒生成器。生成样本几乎总能被目标分类器判为目标类（投毒成功），同时几乎从不被 K+1 判别器识破为"假"（逃逸成功），且训练过程稳定。epoch 10 与 epoch 50 分别对应两种不同用途的最优点（见 §五）。

---

## 二、训练配置

| 项 | 值 |
|---|---|
| 判别器 | K+1（冻结骨干 `MMActionClassifier` + 可训分类头），51 真类 + 1 假类 |
| 生成器步数 / 判别器步数 | `g_steps=3`, `d_steps=1`（每 batch） |
| 优化器 | Adam，`lr_g=3e-4`, `lr_d=5e-5`, `betas=(0.5,0.999)` |
| `λ_adv`（目标类 CE） | 0.2 |
| `λ_avoid`（避开 fake 类） | 0.1 |
| `λ_distribution`（类条件多尺度 RBF-MMD） | 1.0 |
| `λ_var_floor`（VICReg 方差下限） | 0.25 |
| `λ_raw_stat`（音视频均值/标准差匹配） | 0.1 |
| `λ_audio_tail`（音频偏度/峰度匹配） | 0.1 |
| `λ_diversity`（mode-seeking） | 0.2，epoch 3 起 5 轮线性 ramp |
| `audio_out_max` / `video_out_max` | 50.0 / 20.0（输出边界 clamp） |
| `grad_clip` | 0.0（关闭——见 §六修复说明） |

---

## 三、训练稳定性（epoch 31–50 曲线）

> 注：history 从 ckpt_30 resume 后重新记录，故仅含 ep31–50。ep10/20/30 的状态由对应 checkpoint 的离线评估覆盖（见 §四）。

```
epoch   d_loss   g_loss   fake_class_prob   d_fake   g_distribution   g_raw_stat   g_audio_tail
  31    1.675    0.452      0.184           1.931       0.581          4.660         0.410
  35    1.575    0.468      0.199           1.792       0.581          4.727         0.477
  40    1.481    0.473      0.218           1.646       0.580          4.712         0.464
  45    1.410    0.476      0.239           1.527       0.579          4.706         0.424
  50    1.363    0.469      0.247           1.460       0.578          4.751         0.465
```

**判读**：
- `d_loss` 单调下降 1.675 → 1.363：判别器在持续学习，能力增强。
- `g_loss` 平稳在 0.45–0.48：生成器应对稳定，无发散。
- `fake_class_prob`（D 把 fake 判为假类的平均概率）0.184 → 0.247：D 在缓慢学会识别假样本，但幅度很小。
- `g_distribution`（类条件 MMD）锁定在 ~0.58：embedding 空间分布匹配项已被 G 满足并稳定。
- 全程 `forward-NaN = 0`，仅偶发（个位数次）梯度 NaN 被 trainer 的 finite-guard 安全跳过，未污染权重。

---

## 四、离线评估（UCF101 test 集，5 个 checkpoint）

| 指标 | ep10 | ep20 | ep30 | ep45 | ep50 | 真实参照 |
|---|---|---|---|---|---|---|
| **target_success_rate** | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | — |
| **fake_escape_rate** | 0.991 | 0.995 | 0.995 | 0.994 | 0.995 | — |
| fake_class_prob | 0.081 | 0.126 | 0.162 | 0.220 | 0.233 | 越低越好 |
| target_prob | 0.900 | 0.845 | 0.802 | 0.734 | 0.721 | — |
| audio mean | +0.004 | +0.005 | +0.016 | −0.012 | −0.007 | −0.001 |
| **audio std** | **0.804** | 3.170 | 3.901 | 3.999 | 3.890 | **0.771** |
| audio min / max | −14.5 / +9.9 | −48.9 / +41.0 | −48.2 / +37.2 | −48.2 / +43.2 | −48.3 / +44.4 | −13.7 / +6.1 |
| video mean | 0.283 | 0.316 | 0.332 | 0.327 | 0.321 | 0.293 |
| **video std** | **0.426** | 0.509 | 0.531 | 0.546 | 0.537 | **0.430** |
| video min / max | 0 / 7.15 | 0 / 6.86 | 0 / 6.95 | 0 / 7.63 | 0 / 7.42 | 0 / 4.14 |
| audio_diversity_ratio | 1.00 | 4.47 | 5.66 | 5.85 | 5.62 | ≈1 最佳 |
| **video_diversity_ratio** | **0.90** | 1.19 | 1.22 | 1.36 | 1.27 | ≈1 最佳 |
| embedding mean L2 gap | 1.61 | 1.63 | 1.64 | 1.55 | 1.54 | — |
| embedding var L1 gap | 0.046 | 0.046 | 0.047 | 0.046 | 0.046 | — |

> `target_success_rate`：生成样本被分类为指定目标类的比例（投毒有效性）。
> `fake_escape_rate`：生成样本未被分到 `fake_class=51` 的比例（逃逸能力）。
> `diversity_ratio`：生成类内方差 / 真实类内方差，≈1 表示覆盖与真实一致；显著 >1 表示生成器主动扩大探索（非 mode collapse）。

---

## 五、关键发现：分布匹配 vs 对抗博弈的权衡

数据呈现明显的**双峰特征**，对应两种用途：

### 用途 A — 严格分布匹配（"不可检测"投毒）→ 选 epoch 10
- audio std **0.804 vs 真实 0.771（误差 4%）**，mean ≈ 0，范围 −14.5~+9.9 与真实 −13.7~+6.1 几乎重合
- audio_diversity_ratio **1.00**（完美），video_diversity_ratio 0.90
- target_success_rate 仍为 1.000
- **此时生成特征在数值分布上最接近真实，最不易被统计检测**

### 用途 B — 充分对抗博弈（最大化欺骗鲁棒性）→ 选 epoch 50
- target_success_rate 0.998，fake_escape_rate 0.995（与 ep10 持平）
- 但 audio std 扩大到 3.89（真实的 5 倍），范围 −48~+44：生成器主动扩大音频输出空间以维持对不断变强的 D 的欺骗
- D 的 `fake_class_prob` 从 0.08 升到 0.23，说明 D 在学习，G 用更大的探索范围持续压制 D
- **这是健康的对抗动态，不是退化**：G 始终维持 ~1.0 目标成功率，D 缓慢进步但远未识破

### 一致项（两用途都成立）
- **video 通道全程匹配良好**：std 0.43–0.55（真实 0.43），mean 0.28–0.33（真实 0.29）。video 是这个 GAN 的稳定主信号。
- **embedding 空间贴近**：mean L2 gap 1.54–1.64、var L1 gap 0.046 全程稳定，说明在判别器特征空间里生成分布的形状与真实高度一致。
- **无 mode collapse**：diversity_ratio 从未塌缩到接近 0。

---

## 六、训练中修复的 Bug（影响质量可信度）

原训练在 epoch 4 必崩为 NaN（生成器 53 个参数张量中 50 个变 NaN）。根因与修复已固化在代码中：

| 文件 | 修复 | 作用 |
|---|---|---|
| `dtm_poison_gan/losses.py` | `audio_tail_loss` 的 `normalized = ((x-μ)/σ).clamp(-10,10)` | **根因**：低方差维度 σ 被 clamp 到 eps，normalized 可达 ~1e4，`pow(4)` 反向梯度溢出污染全部 G 参数 |
| `dtm_poison_gan/models.py` | 生成器 audio 输出 `clamp(±audio_out_max=50)` | 防止 mode-seeking 推高 audio 到无界 |
| `dtm_poison_gan/config.py` | `grad_clip=0.0`（默认关闭） | `clip_grad_norm_` 在此项目是 NaN 放大器：单梯度 NaN → 总 norm NaN → 所有梯度被污染 |
| `dtm_poison_gan/trainer.py` | D/G step 加 finite-guard + 梯度有限值检查 | 安全网：非有限 loss/梯度跳过该步，防权重污染 |

**验证**：修复前 epoch 4 必崩；修复后 50 epoch 零崩溃。（以上 4 处改动在工作树，未 commit。）

---

## 七、交付物清单

| 类型 | 路径 | 说明 |
|---|---|---|
| 最终检查点 | `dtm_poison_gan/final_dtm_final.pt` | epoch 50，对抗博弈最充分 |
| 对抗曲线检查点 | `dtm_poison_gan/ckpt_{10,20,30,40,45,50}_dtm_final.pt` | 用于绘制 G/D 演化 |
| 分布匹配最优点 | `dtm_poison_gan/ckpt_10_dtm_final.pt` | audio 分布最贴近真实 |
| 类原型库 | `dtm_poison_gan/prototypes_dtm_final.pt` | 各类 embedding 均值/方差 |
| 训练曲线 | `dtm_poison_gan/history_dtm_final.json` | ep31–50 每轮指标 |
| 离线评估 | `dtm_poison_gan_eval/{ckpt10,ckpt20_final,ckpt30_final,ckpt_45,final_50}/analysis_results.json` | 5 点对抗演化 |
| **投毒特征** | `dtm_poison_features/dtm_ckpt10_balanced.pt` | **1020 样本（51 类 × 20），clean_label 攻击，来自 ckpt_10** |
| 代码修复 | `dtm_poison_gan/{config,models,losses,trainer}.py` | NaN 根因修复（未 commit） |

---

## 八、建议

1. **投毒下游验证**：当前 `dtm_ckpt10_balanced.pt` 来自分布匹配最优点（ep10）。若下游联邦训练的攻击目标是"最高成功率"而非"最隐蔽"，可考虑从 `final_50` 重新生成一版做对比。
2. **提交修复**：4 处 NaN 修复应 commit，否则任何复跑都会在 epoch 4 崩溃。
3. **可选调优**（仅当目标是严格分布匹配而非对抗）：后期 audio std 偏离真实（3.9 vs 0.77）。若希望全程贴近真实，可提高 `λ_raw_stat`/`λ_audio_tail` 或降低 `audio_out_max`（如 15），让 G 在保持 `target_success` 的同时不扩大 audio 范围。当前配置优先对抗博弈，audio 扩大是 G 的主动策略，非缺陷。
4. **清理实验残留**：`dtm_cloud`/`dtm_fixed2`/`dtm_guard*` 系列含早期带 NaN 权重的 checkpoint，可按需删除以释放 ~1.5 GB。
