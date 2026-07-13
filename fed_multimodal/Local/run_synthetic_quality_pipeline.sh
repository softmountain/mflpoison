#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME="${EXP_NAME:-synthetic_quality_epoch50}"
CHECKPOINT="${CHECKPOINT:-fed_multimodal/Local/results/poison_gan/ckpt_50_balance_g_escape_64.pt}"
MODEL_PATH="${MODEL_PATH:-fed_multimodal/Local/results/local_training/best_model.pt}"
NUM_SYNTHETIC="${NUM_SYNTHETIC:-4901}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

SYNTHETIC_PATH="fed_multimodal/Local/results/poison_features/${EXP_NAME}_synthetic_balanced.pt"
OUTPUT_DIR="fed_multimodal/Local/results/synthetic_quality_eval/$EXP_NAME"

python fed_multimodal/Local/eval_poison_gan.py \
  --checkpoint "$CHECKPOINT" \
  --model_path "$MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --num_batches 20 \
  --output_dir "fed_multimodal/Local/results/poison_gan_eval/$EXP_NAME"

python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint "$CHECKPOINT" \
  --model_path "$MODEL_PATH" \
  --num_samples "$NUM_SYNTHETIC" \
  --batch_size "$BATCH_SIZE" \
  --target_strategy balanced \
  --attack_mode clean_label \
  --output_path "$SYNTHETIC_PATH"

python fed_multimodal/Local/train_with_poison_features.py \
  --model_path "$MODEL_PATH" \
  --synthetic_path "$SYNTHETIC_PATH" \
  --mode poison_only \
  --from_scratch \
  --num_epochs "$CLASSIFIER_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --learning_rate "$LEARNING_RATE" \
  --eval_train \
  --output_dir "$OUTPUT_DIR"
