#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 5 \
  --batch_size 512 \
  --num_workers 0 \
  --max_batches 2 \
  --log_interval 1 \
  --save_interval 1 \
  --exp_name smoke
