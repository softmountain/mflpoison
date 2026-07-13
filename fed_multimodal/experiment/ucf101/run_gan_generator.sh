#!/bin/bash

# UCF101 Improved GAN  Training Script
# Key improvements:
# 1. D learning rate = 1e-4 (half of G's 2e-4) - prevents D from dominating
# 2. Label smoothing (real=0.9, fake=0.1) - stabilizes training
# 3. Noise injection (std=0.1) - regularizes D
# 4. rf_weight=2.0, aux_weight=1.0 - focus on realism over classification

DATASET="ucf101"
FED_ALG="fed_avg"
SAMPLE_RATE=0.3
BATCH_SIZE=16
ALPHA=5.0  # -> alpha50

# FL Configuration
NUM_EPOCHS=200
GAN_START_EPOCH=50

# Improved GAN Configuration
GAN_EPOCHS=100
GAN_LR_G=0.0002      # Generator learning rate
GAN_LR_D=0.0001      # Discriminator learning rate (slower!)
GAN_RF_WEIGHT=2.0    # Focus on realism
GAN_AUX_WEIGHT=1.0   # Lower weight for classification
GAN_REAL_SMOOTH=0.9  # Label smoothing
GAN_FAKE_SMOOTH=0.1
GAN_NOISE_STD=0.1    # Noise injection

echo "=============================================="
echo "UCF101 Improved GAN Attack Training"
echo "=============================================="
echo "Key Improvements:"
echo "  - D LR ($GAN_LR_D) < G LR ($GAN_LR_G)"
echo "  - Label Smoothing: real=$GAN_REAL_SMOOTH, fake=$GAN_FAKE_SMOOTH"
echo "  - Noise Std: $GAN_NOISE_STD"
echo "  - RF Weight: $GAN_RF_WEIGHT, Aux Weight: $GAN_AUX_WEIGHT"
echo "=============================================="

python train_gan_generator.py \
    --dataset $DATASET \
    --fed_alg $FED_ALG \
    --num_epochs $NUM_EPOCHS \
    --sample_rate $SAMPLE_RATE \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --gan_start_epoch $GAN_START_EPOCH \
    --gan_epochs $GAN_EPOCHS \
    --gan_lr_g $GAN_LR_G \
    --gan_lr_d $GAN_LR_D \
    --gan_rf_weight $GAN_RF_WEIGHT \
    --gan_aux_weight $GAN_AUX_WEIGHT \
    --gan_real_smooth $GAN_REAL_SMOOTH \
    --gan_fake_smooth $GAN_FAKE_SMOOTH \
    --gan_noise_std $GAN_NOISE_STD \
    --gan_eval_interval 10 \
    --gan_save_interval 20 \
    --run_analysis

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
