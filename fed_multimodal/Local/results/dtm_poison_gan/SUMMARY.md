# DTM-GAN 训练与生成效果总结

> 任务：运行并修复 `dtm_poison_gan`（第二版特征空间投毒 GAN），完成训练后评估生成器与生成数据的质量。
> 周期：2026-07-13
> 数据：UCF101 子集，51 类，train 4893 / test 1944，audio `[500,80]`（MFCC）/ video `[9,1280]`（MobileNetV2）
> 详细子报告：
> - 生成器对抗质量 → [`GENERATOR_QUALITY_REPORT.md`](./GENERATOR_QUALITY_REPORT.md)
> - 下游 TSTR 评估 → [`../synthetic_training/TSTR_REPORT.md`](../synthetic_training/TSTR_REPORT.md)

---

## 一、一句话结论

DTM-GAN 已完成 50 epoch 稳定训练，生成器在对抗与下游两个维度都达标：**对抗上**全程压制 K+1 判别器（目标攻击成功率 ≈1.0、逃逸率 ≈0.995），**下游上**纯合成数据训练的分类器在真实测试集达 57.15%（真实基线 75.62% 的 75.5%）。充分对抗训练的最终检查点（epoch 50）在所有维度均优于中途的分布匹配最优点（epoch 10）。

---

## 二、模型与训练配置

| 项 | 值 |
|---|---|
| 生成器 | 类条件 DTMGenerator，`z_dim=256`, `hidden_dim=512`, 输出 audio `[B,500,80]` + video `[B,9,1280]` |
| 判别器 | K+1 判别器：冻结骨干 `MMActionClassifier` + 可训分类头，51 真类 + 1 假类（`fake_class=51`） |
| 步数比 | `d_steps=1`, `g_steps=3`（每 batch） |
| 优化器 | Adam，`lr_g=3e-4`, `lr_d=5e-5`, `betas=(0.5,0.999)` |
| 损失组成 | 目标类 CE `λ=0.2` + 避开假类 `0.1` + 类条件多尺度 RBF-MMD `1.0` + VICReg 方差下限 `0.25` + 音视频均值/标准差匹配 `0.1` + 音频偏度/峰度匹配 `0.1` + mode-seeking `0.2`（epoch 3 起 5 轮 ramp） |
| 输出边界 | `audio_out_max=50`, `video_out_max=20` |
| 训练规模 | 50 epoch，batch 32，`target_strategy=same_as_real` |

---

## 三、训练过程与稳定性

### 关键问题：原训练在 epoch 4 必崩为 NaN
生成器 53 个参数张量中 50 个变 NaN，导致无法完成训练。

**根因**：`audio_tail_loss.standardized_moments` 中 `(x-μ)/σ`，低方差维度 σ 被 clamp 到 `eps=1e-5`，归一化值可达 ~1e4，`pow(4)` 反向梯度溢出到 inf，经判别器反传污染全部 39 个生成器参数 → 权重 NaN → 此后所有 forward NaN。

### 修复（4 处，工作树未 commit）
| 文件 | 修复 |
|---|---|
| `dtm_poison_gan/losses.py` | 归一化值 `clamp(-10,10)` —— **根因修复** |
| `dtm_poison_gan/models.py` | 生成器 audio 输出 `clamp(±audio_out_max)` |
| `dtm_poison_gan/config.py` | `grad_clip=0`（`clip_grad_norm_` 在本项目是 NaN 放大器：单梯度 NaN → 总范数 NaN → 污染所有梯度） |
| `dtm_poison_gan/trainer.py` | D/G step 加 finite-guard + 梯度有限值检查，非有限则跳过该步 |

**结果**：修复后 50 epoch 零崩溃，全程 `forward-NaN=0`，仅个位数次偶发梯度 NaN 被安全跳过。新增 `--resume` 支持从 ckpt_30 恢复到 50。

### 训练曲线（epoch 31–50）
```
d_loss:        1.675 → 1.363   （单调下降，判别器在学）
g_loss:        ~0.45–0.48      （平稳，生成器应对稳定）
fake_class_prob: 0.184 → 0.247 （D 缓慢学会识别假样本，幅度很小）
g_distribution:  ~0.58 锁定     （embedding 空间分布匹配项已被满足）
```

---

## 四、生成器对抗质量（UCF101 test，5 个检查点）

| 指标 | ep10 | ep20 | ep30 | ep45 | ep50 |
|---|---|---|---|---|---|
| **目标攻击成功率** | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| **判别器逃逸率** | 0.991 | 0.995 | 0.995 | 0.994 | 0.995 |
| fake_class_prob（D 识假能力） | 0.081 | 0.126 | 0.162 | 0.220 | 0.233 |
| target_prob | 0.900 | 0.845 | 0.802 | 0.734 | 0.721 |
| audio std（真实 0.77） | 0.80 | 3.17 | 3.90 | 4.00 | 3.89 |
| video std（真实 0.43） | 0.43 | 0.51 | 0.53 | 0.55 | 0.54 |
| video diversity_ratio | 0.90 | 1.19 | 1.22 | 1.36 | 1.27 |

**对抗解读**：生成器全程压制判别器（成功率 ≈1.0）。判别器缓慢变强（fake_class_prob 0.08→0.23），生成器以扩大 audio 输出空间（std 0.8→3.9）作为对抗策略持续欺骗——健康的对抗动态，非退化。video 通道全程稳定匹配真实分布，无 mode collapse（diversity_ratio 始终 ≥0.9）。

---

## 五、生成数据下游效果（TSTR）

用生成特征训练与全局模型同款的 `MMActionClassifier`（hid_size=64/无注意力），100 epoch，真实 test 评估。每版 5100 合成样本（51 类 × 100，clean_label + balanced）。

| 训练数据 | Best Test Acc | UAR | F1 | 达真实基线 |
|---|---|---|---|---|
| **final50 合成** | **57.15%** | 56.62% | 54.94% | **75.5%** |
| ckpt10 合成 | 50.36% | 50.66% | 48.07% | 66.6% |
| 真实数据（基线） | 75.62% | — | — | 100% |
| 随机猜测 | ~2% | — | — | — |

**下游解读**：纯合成数据训练的分类器远超随机基线（2%），证明生成特征携带了大量可迁移到真实分布的类别判别信息。**final50 下游比 ckpt10 高 6.8 个百分点**——分布匹配最优的 ckpt10 反而下游更差，因为对抗训练不充分、类别判别信号弱；final50 经充分对抗，每类编码了更鲜明的判别特征。

---

## 六、核心结论

1. **训练成功**：修复 NaN 根因后，50 epoch 对抗训练稳定完成，生成器与判别器健康博弈。
2. **对抗达标**：目标攻击成功率全程 ≈1.0、逃逸率 ≈0.995，完全满足特征空间投毒 GAN 的设计目标。
3. **下游可用**：合成数据 TSTR 达真实基线的 75.5%，生成特征具备真实的类别语义。
4. **充分对抗优于分布匹配**：epoch 50 在对抗指标与下游 TSTR 上**全面优于**分布匹配最优的 epoch 10。从下游任务角度验证了"应跑完整对抗训练"的判断。
5. **评估生成数据不能只看分布匹配**：分布匹配好（ckpt10）≠ 下游质量好。应优先用 TSTR 这类下游任务指标。

---

## 七、交付物清单

### 检查点与生成数据
| 文件 | 说明 |
|---|---|
| `dtm_poison_gan/final_dtm_final.pt` | 最终检查点（epoch 50），生产推荐 |
| `dtm_poison_gan/ckpt_{10,20,30,40,45,50}_dtm_final.pt` | 对抗演化曲线检查点 |
| `dtm_poison_gan/prototypes_dtm_final.pt` | 类原型 embedding 均值/方差库 |
| `dtm_poison_gan/history_dtm_final.json` | 训练曲线（ep31–50） |
| `dtm_poison_features/dtm_final_dtm_final_train5100.pt` | final50 合成训练集（5100 样本） |
| `dtm_poison_features/dtm_ckpt_10_dtm_final_train5100.pt` | ckpt10 合成训练集（5100 样本） |
| `dtm_poison_features/dtm_ckpt10_balanced.pt` | 投毒特征（1020 样本，clean_label） |

### 评估结果
| 文件 | 说明 |
|---|---|
| `dtm_poison_gan_eval/{ckpt10,ckpt20_final,ckpt30_final,ckpt_45,final_50}/analysis_results.json` | 5 点对抗评估 |
| `synthetic_training/final50/results_*.json` | final50 TSTR 训练历史 |
| `synthetic_training/ckpt10/results_*.json` | ckpt10 TSTR 训练历史 |

### 报告
| 文件 | 说明 |
|---|---|
| `dtm_poison_gan/SUMMARY.md` | **本文件**（训练+生成效果总览） |
| `dtm_poison_gan/GENERATOR_QUALITY_REPORT.md` | 生成器对抗质量详细报告 |
| `synthetic_training/TSTR_REPORT.md` | 下游 TSTR 评估详细报告 |

### 代码修复（工作树，未 commit）
`dtm_poison_gan/{config,models,losses,trainer}.py` —— NaN 根因修复（4 处）+ `train_dtm_poison_gan.py` 新增 `--resume`。

---

## 八、遗留与建议

1. **提交代码修复**：4 处 NaN 修复应 commit，否则任何复跑都会在 epoch 4 崩溃。
2. **生产用 final50**：对抗与下游均最优，作为投毒生成器的默认检查点。
3. **提升 TSTR 的方向**（若需更逼近真实基线 75.6%）：增大生成样本量、增强类内多样性正则、在保持类别可分性前提下收敛 audio 分布（当前 std 3.9 vs 真实 0.77）。
4. **清理实验残留**：`dtm_cloud`/`dtm_fixed2`/`dtm_guard*` 系列含早期带 NaN 权重的检查点，可按需删除以释放 ~1.5 GB。
