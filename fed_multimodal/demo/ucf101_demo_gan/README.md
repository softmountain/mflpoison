# Demo UCF101 GAN pipeline

这个目录承接 `fed_multimodal/demo/ucf101_demo/` 的联邦训练结果：

- 使用 demo 联邦训练得到的全局模型 checkpoint 作为 Teacher
- 使用 demo 的 train client 聚合数据作为 GAN 训练素材
- 训练 feature-level multimodal GAN
- 输出评估 JSON 和可视化图像

## 主要脚本

- `dataloader.py`：从 demo packaged 数据聚合出 full_train/train/val/test
- `train_gan.py`：训练 demo GAN
- `eval_gan.py`：评估 demo GAN，并生成可视化
- `run_gan_experiment.sh`：一键训练 + 评估

## 运行示例

```bash
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo_gan.train_gan --fold_idx 1 --teacher_checkpoint /home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/training/fold1_fed_avg_sr02_ep200_lr005_glr001_hid128_best_model.pt --exp_name 0309BASE --gan_epochs 20
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo_gan.eval_gan --fold_idx 1 --teacher_checkpoint /home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/training/fold1_fed_avg_sr02_ep200_lr005_glr001_hid128_best_model.pt --checkpoint /home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/gan/checkpoints/ckpt_20_0309BASE.pt --num_batches 10
```

或：

```bash
bash fed_multimodal/demo/ucf101_demo_gan/run_gan_experiment.sh --fold_idx 1 --exp_name 0309BASE --gan_epochs 20 --num_batches 10
```

## 输出位置

- Teacher checkpoint：`fed_multimodal/results/demo/ucf101/training/fold{n}_{tag}_best_model.pt`
- GAN checkpoint：`fed_multimodal/results/demo/ucf101/gan/checkpoints/`
- GAN 训练日志：`fed_multimodal/results/demo/ucf101/gan/logs/<exp_name>.train_gan.log`
- GAN 分析结果：`fed_multimodal/results/demo/ucf101/gan_analysis/<checkpoint_name>/`
