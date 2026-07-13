# GAN 实验自动化脚本使用指南

## 概述

本目录提供自动化脚本，用于简化 GAN 训练评估，以及基于 GAN 合成数据的下游训练流程：

1. **`run_gan_experiment.sh`** - 单个实验运行脚本
2. **`run_gan_experiments_batch.sh`** - 批量实验运行脚本

## 功能特性

✅ 自动化 GAN 训练和评估完整流程  
✅ 智能命名：根据实验参数自动生成文件名  
✅ 日志记录：保存所有命令和输出到 `results/logs/`  
✅ 结果组织：评估结果保存到 `results/gan_analysis/`  
✅ 批量运行：支持一次运行多组对比实验  
✅ 支持 fake_train：用 GAN 合成特征训练分类器，并在真实验证/测试集上检验可用性

---

## 1. 单个实验运行：`run_gan_experiment.sh`

### 基本用法

```bash
# 最简单的用法（使用所有默认参数）
bash run_gan_experiment.sh

# 指定实验名称
bash run_gan_experiment.sh --exp_name "MyExperiment"

# 完整参数示例
bash run_gan_experiment.sh \
  --exp_name "MoM0.5_FM0.5" \
  --gan_epochs 200 \
  --gan_mom_weight 0.5 \
  --gan_fm_weight 0.5
```

### 参数说明

#### 基础参数
- `--exp_name`: 实验名称（用于文件命名，若不指定则自动生成）
- `--mode`: `gan`（默认）/ `fake_train` / `fake_attack`
- `--model_path`: Teacher 模型路径（默认：`results/local_training/best_model.pt`）
- `--gan_epochs`: 训练轮数（默认：200）
- `--batch_size`: 批次大小（默认：32）
- `--audio_feat`: 音频特征类型（默认：`mfcc`）
- `--video_feat`: 视频特征类型（默认：`mobilenet_v2`）

#### GAN 超参数
- `--gan_lr_g`: 生成器学习率（默认：2e-4）
- `--gan_lr_d`: 判别器学习率（默认：1e-4）
- `--gan_rf_weight`: Real/Fake 对抗损失权重（默认：2.0）
- `--gan_aux_weight`: 辅助分类损失权重（默认：1.0）
- `--gan_cls_weight`: Teacher 分类损失权重（默认：0.1）
- `--gan_joint_weight`: Joint Critic 损失权重（默认：0.05）
- `--gan_fm_weight`: Feature Matching 损失权重（默认：0.05）
- `--gan_mom_weight`: Moment Matching 损失权重（默认：0.05）

#### 输出约束参数
- `--gan_audio_out_max`: 音频输出最大值（默认：1.0）
- `--gan_audio_scale_max`: 音频缩放最大值（默认：0.3）
- `--gan_video_out_max`: 视频输出最大值（默认：20.0）
- `--gan_video_scale_max`: 视频缩放最大值（默认：8.0）

#### 评估参数
- `--eval_num_batches`: 评估使用的批次数（默认：50）
- `--eval_no_tsne`: 跳过 t-SNE
- `--eval_no_extra_metrics`: 跳过 FID/MMD/Domain Classifier
- `--eval_no_domain_clf`: 仅跳过 Domain Classifier
- `--eval_max_metric_samples`: 额外指标的采样上限
- `--eval_use_train`: 使用 train 集评估

#### 合成特征训练参数
- `--gan_checkpoint`: 生成合成数据用的 GAN checkpoint
- `--fake_epochs` / `--num_epochs`: 合成数据训练轮数（默认：50）
- `--fake_output_dir`: 训练输出目录（可选）
- `--attack_src_label` / `--attack_dst_label` / `--attack_prob`: 可选标签扰动参数，只用于额外鲁棒性实验

> 提示：`fake_train` 需要 `--gan_checkpoint`，否则会默认尝试 `results/local_gan/ckpt_${GAN_EPOCHS}_${EXP_NAME}.pt`。

### 实验名称自动生成规则

如果不指定 `--exp_name`，脚本会根据参数自动生成名称：

- `Baseline_MMDD_HHMM`：FM=0, MoM=0
- `MoMOnly_0.5_MMDD_HHMM`：只使用 MoM
- `FMOnly_1.0_MMDD_HHMM`：只使用 FM
- `FM0.5_MoM0.5_MMDD_HHMM`：同时使用 FM 和 MoM

### 输出文件组织

执行实验后，会生成以下文件：

```
fed_multimodal/Local/
├── results/
│   ├── logs/
│   │   └── 25.1228.MoMOnly_0.5.txt          # 完整日志
│   ├── local_gan/
│   │   └── ckpt_200.pt                       # GAN checkpoint
│   ├── gan_analysis/
│       └── ckpt_200_MoMOnly_0.5/             # 评估结果目录
│           ├── analysis_results.json         # 数值结果
│           ├── tsne_audio.png                # t-SNE 可视化
│           ├── tsne_video.png
│           ├── dist_audio.png                # 分布对比图
│           └── dist_video.png
│   ├── fake_training/                         # 合成数据训练输出
│   └── fake_attack/                           # 可选标签扰动实验输出
```

---

## 2. 批量实验运行：`run_gan_experiments_batch.sh`

### 基本用法

```bash
# 运行所有预定义的实验
bash run_gan_experiments_batch.sh
```

### 配置实验列表

编辑 `run_gan_experiments_batch.sh` 文件中的 `experiments` 数组：

```bash
experiments=(
    # 实验名称|参数列表
    "Baseline|--gan_fm_weight 0.0 --gan_mom_weight 0.0"
    "MoMOnly_0.5|--gan_fm_weight 0.0 --gan_mom_weight 0.5"
    "FMOnly_1.0|--gan_fm_weight 1.0 --gan_mom_weight 0.0"
)
```

### 批量日志

批量运行会生成一个汇总日志：

```
results/logs/batch_251228_143022.log
```

记录所有实验的执行状态和结果汇总。

---

## 3. 实际使用示例

### 示例 1：复现日志中的 MoM 实验

```bash
bash run_gan_experiment.sh \
  --exp_name "MoMOnly_0.5" \
  --gan_epochs 200 \
  --gan_lr_g 2e-4 --gan_lr_d 1e-4 \
  --gan_rf_weight 2.0 --gan_aux_weight 1.0 \
  --gan_cls_weight 0.1 --gan_joint_weight 0.05 \
  --gan_fm_weight 0.0 --gan_mom_weight 0.5 \
  --gan_audio_out_max 1.0 --gan_audio_scale_max 0.3 \
  --gan_video_out_max 20.0 --gan_video_scale_max 8.0 \
  --batch_size 32 \
  --audio_feat mfcc --video_feat mobilenet_v2
```

### 示例 2：快速测试（少量 epoch）

```bash
bash run_gan_experiment.sh \
  --exp_name "QuickTest" \
  --gan_epochs 20 \
  --eval_num_batches 10
```

### 示例 3：用合成数据训练模型

```bash
bash run_gan_experiment.sh \
  --mode fake_train \
  --exp_name "FakeTrain_CLS_LOW" \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --fake_epochs 50
```

### 示例 4：可选标签扰动鲁棒性实验

```bash
bash run_gan_experiment.sh \
  --mode fake_attack \
  --exp_name "FakeAttack_3to7" \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --attack_src_label 3 \
  --attack_dst_label 7 \
  --attack_prob 0.7
```

### 示例 5：对比不同 MoM 权重

修改 `run_gan_experiments_batch.sh`：

```bash
experiments=(
    "MoM_0.0|--gan_mom_weight 0.0"
    "MoM_0.3|--gan_mom_weight 0.3"
    "MoM_0.5|--gan_mom_weight 0.5"
    "MoM_1.0|--gan_mom_weight 1.0"
)
```

然后运行：

```bash
bash run_gan_experiments_batch.sh
```

---

## 4. 日志文件格式

日志文件 (`results/logs/YY.MMDD.ExpName.txt`) 包含：

```
============================================================
GAN 训练和评估自动化实验
============================================================
实验名称: MoMOnly_0.5
日志文件: results/logs/25.1228.MoMOnly_0.5.txt
开始时间: 2025-12-28 18:38:44

训练参数:
  - Model Path: results/local_training/best_model.pt
  - Epochs: 200
  ...

【Step 1/2】开始 GAN 训练...
命令:
python train_local_gan.py ...

(训练输出)

【Step 2/2】开始 GAN 评估...
命令:
python eval_local_gan_quality.py ...

(评估输出)

============================================================
实验完成！
============================================================
```

---

## 5. 故障排除

### 问题 1：权限错误

```bash
# 添加执行权限
chmod +x run_gan_experiment.sh
chmod +x run_gan_experiments_batch.sh
```

### 问题 2：找不到 Teacher 模型

确保已经训练好 Teacher 模型：

```bash
python train_local.py
# 会生成 results/local_training/best_model.pt
```

### 问题 3：批量实验中某个失败

批量脚本会继续运行其他实验，并在最后汇总失败数量。查看对应的单独日志文件以排查问题。

---

## 6. 高级用法

### 自定义评估指标

修改 `eval_local_gan_quality.py` 后，脚本会自动使用新的评估逻辑。

### 多机并行运行

可以在不同机器上同时运行不同的实验（使用不同的 `--exp_name`），日志和结果互不冲突。

### 结果分析

所有实验结果以 JSON 格式保存在 `analysis_results.json`，方便后续统计分析：

```python
import json
import glob

# 读取所有实验结果
results = {}
for f in glob.glob('results/gan_analysis/*/analysis_results.json'):
    with open(f) as fp:
        results[f] = json.load(fp)

# 比较 ML Efficacy
for path, data in results.items():
    print(f"{path}: {data['ml_efficacy']:.4f}")
```

---

## 7. 与原始命令的对比

### 原始方式（手动）

```bash
# 训练
python train_local_gan.py \
  --model_path results/local_training/best_model.pt \
  --gan_epochs 200 \
  ... (很多参数)

# 评估
python eval_local_gan_quality.py \
  --checkpoint results/local_gan/ckpt_200.pt \
  --model_path results/local_training/best_model.pt \
  ... (更多参数)

# 手动复制输出到日志文件
```

### 自动化方式

```bash
# 一条命令完成全部流程
bash run_gan_experiment.sh --exp_name "MoMOnly_0.5" --gan_mom_weight 0.5

# 自动保存日志、自动命名、自动组织结果
```

---

## 8. 常见实验场景

### 场景 1：消融实验（Ablation Study）

```bash
experiments=(
    "NoTeacher_NoJoint|--gan_cls_weight 0.0 --gan_joint_weight 0.0"
    "WithTeacher_NoJoint|--gan_cls_weight 0.1 --gan_joint_weight 0.0"
    "NoTeacher_WithJoint|--gan_cls_weight 0.0 --gan_joint_weight 0.05"
    "Full|--gan_cls_weight 0.1 --gan_joint_weight 0.05"
)
```

### 场景 2：超参数搜索

```bash
for lr_g in 1e-4 2e-4 5e-4; do
    for cls_w in 0.05 0.1 0.2; do
        bash run_gan_experiment.sh \
            --exp_name "LR${lr_g}_Cls${cls_w}" \
            --gan_lr_g $lr_g \
            --gan_cls_weight $cls_w
    done
done
```

---

## 联系与反馈

如有问题或建议，请在项目 Issue 中提出。

**祝实验顺利！** 🚀
