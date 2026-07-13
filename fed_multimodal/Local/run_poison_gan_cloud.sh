#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 50 \
  --batch_size 64 \
  --num_workers 4 \
  --save_interval 10 \
  --log_interval 20 \
  --target_strategy same_as_real \
  --freeze_d backbone \
  --lr_g 2e-4 \
  --lr_d 5e-5 \
  --lambda_d_fake 0.5 \
  --lambda_avoid 0.8 \
  --lambda_div 0.02 \
  --exp_name cloud
# python fed_multimodal/Local/train_poison_gan.py \
#   --model_path fed_multimodal/Local/results/local_training/best_model.pt \
#   --epochs 50 \
#   --batch_size 32 \
#   --num_workers 4 \
#   --log_interval 20 \
#   --save_interval 10 \
#   --target_strategy same_as_real \
#   --freeze_d backbone \
#   --exp_name cloud
