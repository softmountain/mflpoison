#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="0309BASE"
FOLD_IDX=1
GAN_EPOCHS=200
NUM_BATCHES=30

while [[ $# -gt 0 ]]; do
  case $1 in
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --fold_idx) FOLD_IDX="$2"; shift 2 ;;
    --gan_epochs) GAN_EPOCHS="$2"; shift 2 ;;
    --num_batches) NUM_BATCHES="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift 1 ;;
  esac
done

conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo_gan.train_gan \
  --fold_idx "$FOLD_IDX" \
  --exp_name "$EXP_NAME" \
  --gan_epochs "$GAN_EPOCHS" \
  "${EXTRA_ARGS[@]:-}"

conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo_gan.eval_gan \
  --fold_idx "$FOLD_IDX" \
  --checkpoint "/home/xp/fed-multimodal/fed_multimodal/results/demo/ucf101/gan/checkpoints/ckpt_${GAN_EPOCHS}_${EXP_NAME}.pt" \
  --num_batches "$NUM_BATCHES"
