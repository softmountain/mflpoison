# Fed-PoisonGAN Cloud Bundle

这个文件夹是 UCF101 Fed-PoisonGAN 的云端可运行代码包。它包含运行 feature-level Fed-PoisonGAN 所需的代码、已提取特征、UCF101 split 文件和本地 teacher/global 初始化权重。

## 内容

- `setup.py`, `requirements.txt`, `README.md`: 项目安装与依赖元信息。
- `fed_multimodal/Fed-PoisonGAN.md`: K+1 Fed-PoisonGAN 设计方案。
- `fed_multimodal/Local/GAN_ANALYSIS_REPORT.md`: 旧 GAN 分析报告。
- `fed_multimodal/poison_gan/`: 新增 Fed-PoisonGAN 核心实现。
- `fed_multimodal/Local/train_poison_gan.py`: 训练 K+1 discriminator PoisonGAN。
- `fed_multimodal/Local/eval_poison_gan.py`: 评估生成质量。
- `fed_multimodal/Local/generate_poison_features.py`: 生成可保存的 poison feature tensor 文件。
- `fed_multimodal/Local/train_with_poison_features.py`: 用 clean/poison/mixed feature 做本地分类器验证。
- `fed_multimodal/Local/run_poison_gan_smoke.sh`: 最小 smoke test。
- `fed_multimodal/Local/run_poison_gan_cloud.sh`: 云端正式训练示例。
- `fed_multimodal/Local/run_poison_pipeline.sh`: 训练、评估、生成 poison、分类器验证的一键流程示例。
- `fed_multimodal/model/`: `MMActionClassifier` 等模型定义。
- `fed_multimodal/Local/dataloader.py`: UCF101 feature-level 本地数据加载。
- `fed_multimodal/results/feature/audio/mfcc/ucf101/feature.pkl`: UCF101 MFCC 音频特征，约 675MB。
- `fed_multimodal/results/feature/video/mobilenet_v2/ucf101/feature.pkl`: UCF101 MobileNetV2 视频特征，约 230MB。
- `fed_multimodal/datasets/ucf101/ucfTrainTestlist/`: UCF101 官方 train/test 划分文件。
- `fed_multimodal/Local/results/local_training/best_model.pt`: 本地 teacher/global 初始化权重。
- `fed_multimodal/Local/results/local_training/final_model.pt`: 本地最终分类器权重。

## 云端安装

从 bundle 根目录执行：

```bash
python -m pip install --upgrade pip
pip install -e .
```

如果依赖缺失，先执行：

```bash
pip install -r requirements.txt
```

## 最小 smoke test

从 bundle 根目录执行：

```bash
bash fed_multimodal/Local/run_poison_gan_smoke.sh
```

等价命令：

```bash
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --max_batches 2 \
  --log_interval 1 \
  --save_interval 1 \
  --exp_name smoke
```

预期输出：

- 能加载 audio/video feature pickle。
- 能加载 K 类 teacher 权重并扩展为 K+1 判别器。
- 能完成至少 1 个 D step 和 G step。
- 写出 `fed_multimodal/Local/results/poison_gan/ckpt_1_smoke.pt`。

## 正式训练

```bash
bash fed_multimodal/Local/run_poison_gan_cloud.sh
```

或自行调整：

```bash
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 4 \
  --save_interval 10 \
  --target_strategy same_as_real \
  --freeze_d backbone \
  --exp_name cloud
```

主要结果保存在：

```text
fed_multimodal/Local/results/poison_gan/
```

## 评估 checkpoint

```bash
python fed_multimodal/Local/eval_poison_gan.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_batches 20 \
  --output_dir fed_multimodal/Local/results/poison_gan_eval/cloud
```

输出文件：

```text
fed_multimodal/Local/results/poison_gan_eval/cloud/analysis_results.json
```

重点指标：

- `target_success_rate`: fake 在前 K 类中是否被判为目标类。
- `fake_escape_rate`: fake 是否没有被 K+1 判别器判为 fake 类。
- `fake_class_prob`: fake 类概率，越低表示越能逃过 fake 类。
- `audio_diversity_ratio`, `video_diversity_ratio`: fake 类内多样性 / real 类内多样性，越接近 1 越好。
- `embedding_mean_l2_gap`, `embedding_var_l1_gap`: teacher/global embedding 空间的均值和方差差距。

## 生成 synthetic feature 文件

```bash
python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_samples 1000 \
  --target_strategy balanced \
  --attack_mode clean_label \
  --output_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt
```

保存的 `.pt` 包含：

- `audio`: `[N, 500, 80]`
- `video`: `[N, 9, 1280]`
- `len_a`, `len_v`
- `condition_label`: G 的条件目标类。
- `train_label`: 混入训练时使用的标签。
- `meta`: checkpoint、生成模式和配置。

## 本地 synthetic-training 验证

```bash
python fed_multimodal/Local/train_with_poison_features.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --poison_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt \
  --mode clean_plus_poison \
  --poison_ratio 0.2 \
  --init_from_model \
  --num_epochs 3 \
  --batch_size 32 \
  --num_workers 4 \
  --output_dir fed_multimodal/Local/results/poison_classifier_eval/cloud
```

## 重要说明

- 当前实现是 feature-level Fed-PoisonGAN，不生成原始 `.avi`、`.wav` 或 raw frames。
- 不需要上传原始 UCF101 视频、`audios/` 或 `rawframes/`。
- 没有复制旧 GAN 的大 checkpoint 目录 `fed_multimodal/Local/results/local_gan`，原目录约 11GB。
- 第一版联邦集成建议保持 FL 全局模型为 K 类，只在 synthetic-data 客户端内部使用 K+1 discriminator 训练 G；生成后的 synthetic feature 标签仍在 `[0, K-1]`。生成质量达标后，可继续用同一生成器输出做标签翻转中毒实验。
