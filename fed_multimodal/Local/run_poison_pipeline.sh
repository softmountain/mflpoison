#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME="${EXP_NAME:-ckpt_50_cloud}"
CHECKPOINT="${CHECKPOINT:-fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt}"
MODEL_PATH="${MODEL_PATH:-fed_multimodal/Local/results/local_training/best_model.pt}"
POISON_RATIO="${POISON_RATIO:-0.2}"
NUM_POISON="${NUM_POISON:-1000}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

# python fed_multimodal/Local/eval_poison_gan.py \
#   --checkpoint "$CHECKPOINT" \
#   --model_path "$MODEL_PATH" \
#   --batch_size "$BATCH_SIZE" \
#   --num_workers "$NUM_WORKERS" \
#   --num_batches 20 \
#   --output_dir "fed_multimodal/Local/results/poison_gan_eval/$EXP_NAME"

python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint "$CHECKPOINT" \
  --model_path "$MODEL_PATH" \
  --num_samples "$NUM_POISON" \
  --batch_size "$BATCH_SIZE" \
  --target_strategy balanced \
  --attack_mode clean_label \
  --output_path "fed_multimodal/Local/results/poison_features/${EXP_NAME}_clean_label.pt"

python fed_multimodal/Local/train_with_poison_features.py \
  --model_path "$MODEL_PATH" \
  --poison_path "fed_multimodal/Local/results/poison_features/${EXP_NAME}_clean_label.pt" \
  --mode clean_only \
  --poison_ratio 0.0 \
  --init_from_model \
  --num_epochs "$CLASSIFIER_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --output_dir "fed_multimodal/Local/results/poison_classifier_eval/${EXP_NAME}_clean_baseline"

python fed_multimodal/Local/train_with_poison_features.py \
  --model_path "$MODEL_PATH" \
  --poison_path "fed_multimodal/Local/results/poison_features/${EXP_NAME}_clean_label.pt" \
  --mode clean_plus_poison \
  --poison_ratio "$POISON_RATIO" \
  --init_from_model \
  --num_epochs "$CLASSIFIER_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --output_dir "fed_multimodal/Local/results/poison_classifier_eval/${EXP_NAME}_poison_ratio_${POISON_RATIO}"
