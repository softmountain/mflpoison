#!/usr/bin/env bash
# Generate the independent source-42 attack_pool (protocol §6.1): same DTM-GAN checkpoint as the
# TSTR pool, but an INDEPENDENT recorded generator seed, >=512 samples, condition==42, NOT used
# for TSTR/gap. The recorded seed makes it provably distinct from the frozen dtm_final_train5100.
#
# Run in the poigan env with PYTHONPATH=/home/xp/fedpoi (see memory: fedpoi-gan-run-env).
# Generation does NOT build a DataloadManager, so the --dataset_dir symlink trap does not apply.
set -euo pipefail

PY=/home/xp/anaconda3/envs/poigan/bin/python
ROOT=/home/xp/fedpoi/fed_multimodal
CKPT=$ROOT/Local/results/dtm_poison_gan/final_dtm_final.pt
SEED=${1:-424242}
N=${2:-1024}
OUT=$ROOT/Local/results/dtm_poison_features/attack_pool_s42_seed${SEED}.pt

export PYTHONPATH=/home/xp/fedpoi
"$PY" "$ROOT/Local/generate_dtm_poison_features.py" \
  --checkpoint "$CKPT" \
  --model_path "$ROOT/Local/results/local_training/best_model.pt" \
  --target_strategy fixed --target_class 42 --attack_mode clean_label \
  --num_samples "$N" --seed "$SEED" \
  --output_path "$OUT"

echo "attack_pool -> $OUT (seed=$SEED, N=$N, condition=42)"
