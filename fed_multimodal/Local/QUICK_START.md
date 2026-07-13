# GAN 实验自动化脚本使用示例

## 快速上手

### 1. 最简单的使用方式

```bash
# 运行 MoM 实验（权重 0.5）
bash run_gan_experiment.sh --exp_name "MoMOnly_0.5" --gan_mom_weight 0.5
```

**会自动完成：**
1. 训练 GAN（200 epochs）
2. 评估生成质量
3. 保存日志到 `results/logs/25.1228.MoMOnly_0.5.txt`
4. 保存分析结果到 `results/gan_analysis/ckpt_200_MoMOnly_0.5/`

---

## 常用命令速查

### 单个实验

```bash
# MoM Only 实验
bash run_gan_experiment.sh --exp_name "MoMOnly_0.5" --gan_mom_weight 0.5

# FM Only 实验
bash run_gan_experiment.sh --exp_name "FMOnly_1.0" --gan_fm_weight 1.0

# Baseline（无 MoM 和 FM）
bash run_gan_experiment.sh --exp_name "Baseline" --gan_mom_weight 0.0 --gan_fm_weight 0.0

# 快速测试（10 epochs）
bash run_gan_experiment.sh --exp_name "QuickTest" --gan_epochs 10 --eval_num_batches 5
```

### 批量实验

```bash
# 运行预定义的多组实验
bash run_gan_experiments_batch.sh
```

---

## 与手动命令的对比

### 原始方式（需要多步操作）

```bash
# Step 1: 训练
python train_local_gan.py \
  --model_path results/local_training/best_model.pt \
  --gan_epochs 200 \
  --gan_lr_g 2e-4 --gan_lr_d 1e-4 \
  --gan_rf_weight 2.0 --gan_aux_weight 1.0 \
  --gan_cls_weight 0.1 --gan_joint_weight 0.05 \
  --gan_fm_weight 0.0 --gan_mom_weight 0.5 \
  --gan_audio_out_max 1.0 --gan_audio_scale_max 0.3 \
  --gan_video_out_max 20.0 --gan_video_scale_max 8.0 \
  --batch_size 32 \
  --audio_feat mfcc --video_feat mobilenet_v2

# Step 2: 评估
python eval_local_gan_quality.py \
  --checkpoint results/local_gan/ckpt_200.pt \
  --model_path results/local_training/best_model.pt \
  --num_batches 50 \
  --output_dir results/gan_analysis/ckpt_200_XXX

# Step 3: 手动复制粘贴终端输出到文本文件
# ...
```

### 自动化方式（一条命令）

```bash
bash run_gan_experiment.sh --exp_name "MoMOnly_0.5" --gan_mom_weight 0.5
```

✅ 自动训练  
✅ 自动评估  
✅ 自动保存日志  
✅ 自动组织结果文件  

---

## 输出文件结构

```
fed_multimodal/Local/results/
├── logs/
│   ├── 25.1228.MoMOnly_0.5.txt       ← 完整实验日志
│   ├── 25.1228.FMOnly_1.0.txt
│   └── batch_251228_143022.log       ← 批量实验汇总
├── local_gan/
│   ├── ckpt_200.pt                   ← GAN checkpoint
│   ├── ckpt_180.pt
│   └── ...
└── gan_analysis/
    ├── ckpt_200_MoMOnly_0.5/         ← 评估结果目录
    │   ├── analysis_results.json     ← 数值指标
    │   ├── tsne_audio.png            ← t-SNE 可视化
    │   ├── tsne_video.png
    │   ├── dist_audio.png            ← 分布对比
    │   └── dist_video.png
    └── ckpt_200_FMOnly_1.0/
        └── ...
```

---

## 实验参数说明

### 损失权重参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--gan_mom_weight` | Moment Matching 权重 | 0.05 | 0.0 - 1.0 |
| `--gan_fm_weight` | Feature Matching 权重 | 0.05 | 0.0 - 1.0 |
| `--gan_cls_weight` | Teacher 分类器权重 | 0.1 | 0.0 - 0.2 |
| `--gan_joint_weight` | Joint Critic 权重 | 0.05 | 0.0 - 0.5 |

### 约束参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gan_audio_out_max` | 音频输出最大值 | 1.0 |
| `--gan_audio_scale_max` | 音频缩放最大值 | 0.3 |
| `--gan_video_out_max` | 视频输出最大值 | 20.0 |
| `--gan_video_scale_max` | 视频缩放最大值 | 8.0 |

---

## 常见实验场景

### 场景 1：对比不同 MoM 权重

```bash
bash run_gan_experiment.sh --exp_name "MoM_0.0" --gan_mom_weight 0.0
bash run_gan_experiment.sh --exp_name "MoM_0.3" --gan_mom_weight 0.3
bash run_gan_experiment.sh --exp_name "MoM_0.5" --gan_mom_weight 0.5
bash run_gan_experiment.sh --exp_name "MoM_1.0" --gan_mom_weight 1.0
```

### 场景 2：消融实验

```bash
# 无 Teacher、无 Joint
bash run_gan_experiment.sh --exp_name "NoTeacher_NoJoint" \
  --gan_cls_weight 0.0 --gan_joint_weight 0.0

# 有 Teacher、无 Joint
bash run_gan_experiment.sh --exp_name "WithTeacher_NoJoint" \
  --gan_cls_weight 0.1 --gan_joint_weight 0.0

# 无 Teacher、有 Joint
bash run_gan_experiment.sh --exp_name "NoTeacher_WithJoint" \
  --gan_cls_weight 0.0 --gan_joint_weight 0.05

# 完整配置
bash run_gan_experiment.sh --exp_name "Full" \
  --gan_cls_weight 0.1 --gan_joint_weight 0.05
```

### 场景 3：批量运行对比实验

编辑 `run_gan_experiments_batch.sh`：

```bash
experiments=(
    "Baseline|--gan_fm_weight 0.0 --gan_mom_weight 0.0"
    "MoMOnly_0.5|--gan_fm_weight 0.0 --gan_mom_weight 0.5"
    "FMOnly_1.0|--gan_fm_weight 1.0 --gan_mom_weight 0.0"
    "Both|--gan_fm_weight 0.5 --gan_mom_weight 0.5"
)
```

运行：
```bash
bash run_gan_experiments_batch.sh
```

---

## 查看结果

### 1. 查看日志

```bash
# 最新的日志
tail -f results/logs/*.txt

# 特定实验的日志
cat results/logs/25.1228.MoMOnly_0.5.txt
```

### 2. 查看评估指标

```bash
# JSON 格式的数值结果
cat results/gan_analysis/ckpt_200_MoMOnly_0.5/analysis_results.json | jq .

# ML Efficacy 准确率
cat results/gan_analysis/ckpt_200_MoMOnly_0.5/analysis_results.json | jq .ml_efficacy
```

### 3. 评估新增相似度指标（FID / MMD / Domain Classifier）

`eval_local_gan_quality.py` 现默认输出更有说服力的分布相似性指标：

- FID（audio/video/joint）
- MMD (RBF)（audio/video/joint）
- Domain Classifier Accuracy（音视频可分性，越接近 50% 越好）

可选参数示例：

```bash
python eval_local_gan_quality.py \
  --checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --model_path results/local_training/best_model.pt \
  --output_dir results/gan_analysis/ckpt_200_CLS_LOW \
  --max_metric_samples 2000
```

跳过额外指标：
```bash
python eval_local_gan_quality.py \
  --checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --model_path results/local_training/best_model.pt \
  --no_extra_metrics
```

### 4. 可视化结果

```bash
# 在本地打开图片
eog results/gan_analysis/ckpt_200_MoMOnly_0.5/tsne_audio.png
```

---

## 使用合成数据训练模型（本地）

### 1) 用合成数据从零训练同结构模型（并用真实数据验证）

```bash
python train_with_fake.py \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --model_path results/local_training/best_model.pt \
  --exp_name FakeTrain_CLS_LOW \
  --num_epochs 50
```

或使用脚本（自动记录日志到 `results/logs`）：
```bash
bash run_gan_experiment.sh \
  --mode fake_train \
  --exp_name FakeTrain_CLS_LOW \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --fake_epochs 50
```

### 2) 可选标签扰动实验（先加载干净模型）

```bash
python train_with_fake.py \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --model_path results/local_training/best_model.pt \
  --init_from_model \
  --attack_src_label 3 \
  --attack_dst_label 7 \
  --attack_prob 0.7 \
  --exp_name FakeAttack_3to7
```

或使用脚本：
```bash
bash run_gan_experiment.sh \
  --mode fake_attack \
  --exp_name FakeAttack_3to7 \
  --gan_checkpoint results/local_gan/ckpt_200_CLS_LOW.pt \
  --attack_src_label 3 \
  --attack_dst_label 7 \
  --attack_prob 0.7
```

### 输出管理

默认输出目录：

```
fed_multimodal/Local/results/
├── fake_training/YYMMDD_HHMMSS_FakeTrain_CLS_LOW/
│   ├── train.log
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── training_history.json
│   └── summary.json
└── fake_attack/YYMMDD_HHMMSS_FakeAttack_3to7/
    ├── train.log
    ├── best_model.pt
    ├── final_model.pt
    ├── training_history.json
    └── summary.json
```

`summary.json` 会包含每类准确率，便于对比攻击前后差异。

---

## 故障排除

### 问题：权限不足

```bash
chmod +x run_gan_experiment.sh
chmod +x run_gan_experiments_batch.sh
```

### 问题：找不到 Teacher 模型

先训练 Teacher：
```bash
python train_local.py
```

### 问题：想修改默认参数

直接编辑脚本中的默认值，或通过命令行参数覆盖：

```bash
bash run_gan_experiment.sh \
  --gan_epochs 100 \           # 覆盖默认的 200
  --batch_size 64 \            # 覆盖默认的 32
  --exp_name "CustomExp"
```

---

## 进阶用法

### 循环运行多组实验

```bash
for weight in 0.0 0.3 0.5 0.7 1.0; do
    bash run_gan_experiment.sh \
        --exp_name "MoM_${weight}" \
        --gan_mom_weight $weight
done
```

### 并行运行（多个终端）

```bash
# 终端 1
bash run_gan_experiment.sh --exp_name "Exp1" --gan_mom_weight 0.3 &

# 终端 2
bash run_gan_experiment.sh --exp_name "Exp2" --gan_mom_weight 0.5 &

# 终端 3
bash run_gan_experiment.sh --exp_name "Exp3" --gan_mom_weight 0.7 &
```

---

## 总结

| 特性 | 手动方式 | 自动化脚本 |
|------|----------|------------|
| 命令数量 | 2+ | 1 |
| 日志保存 | 手动复制 | ✅ 自动 |
| 文件命名 | 手动管理 | ✅ 智能命名 |
| 批量运行 | ❌ 需要循环 | ✅ 内置支持 |
| 易出错 | 参数易遗漏 | ✅ 统一管理 |

**推荐优先使用自动化脚本！** 🎯
