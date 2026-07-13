#### gan 基准测试
```
python train_local_gan.py \
  --model_path results/local_training/best_model.pt\
  --gan_epochs 200 \
  --gan_lr_g 2e-4 \
  --gan_lr_d 1e-4 \
  --gan_rf_weight 2.0 \
  --gan_aux_weight 1.0 \
  --gan_cls_weight 0.1 \
  --gan_joint_weight 0.2 \
  --gan_fm_weight 0.0 \
  --gan_mom_weight 0.0 \
  --gan_audio_out_max 10.0 \
  --gan_audio_scale_max 5.0 \
  --gan_video_out_max 20.0 \
  --gan_video_scale_max 8.0 \
  --batch_size 32 \
  --audio_feat mfcc \
  --video_feat mobilenet_v2
```
#### 开启 mom_weight，收紧 Audio 参数

```
python train_local_gan.py \
  --model_path results/local_training/best_model.pt \
  --gan_epochs 200 \
  --gan_lr_g 2e-4 \
  --gan_lr_d 1e-4 \
  --gan_rf_weight 2.0 \
  --gan_aux_weight 1.0 \
  --gan_cls_weight 0.1 \
  --gan_joint_weight 0.2 \
  --gan_mom_weight 5.0 \
  --gan_fm_weight 0.0 \
  --gan_audio_out_max 2.0 \
  --gan_audio_scale_max 1.0 \
  --gan_video_out_max 20.0 \
  --gan_video_scale_max 8.0 \
  --batch_size 32 \
  --audio_feat mfcc \
  --video_feat mobilenet_v2
```

#### gan效果评估

```
python eval_local_gan_quality.py \
  --checkpoint results/local_gan/ckpt_200.pt \
  --model_path results/local_training/best_model.pt \
  --num_batches 50 \
  --output_dir results/gan_analysis/ckpt_200
```
