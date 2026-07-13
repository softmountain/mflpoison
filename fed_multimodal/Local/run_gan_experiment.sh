#!/bin/bash
# =====================================================================
# GAN / Fake 训练与评估自动化脚本（主干精简版）
# =====================================================================
# 模式：
# 1) gan         - 训练 GAN + 评估（eval_local_gan_quality.py）
# 2) fake_train  - 用 GAN 生成特征训练分类器
# 3) fake_attack - 用 GAN 生成特征做可选标签扰动实验
# =====================================================================

set -e

# ========== 默认参数 ==========
EXP_NAME="BASE"
MODE="gan"
MODEL_PATH="results/local_training/best_model.pt"
GAN_EPOCHS=200
BATCH_SIZE=32
SEED=42
AUDIO_FEAT="mfcc"
VIDEO_FEAT="mobilenet_v2"

GAN_LR_G=2e-4
GAN_LR_D=1e-4

GAN_RF_WEIGHT=2.0
GAN_AUX_WEIGHT=1.0
GAN_CLS_WEIGHT=0.1
GAN_JOINT_WEIGHT=0.05
GAN_FM_WEIGHT=0.05
GAN_MOM_WEIGHT=0.05

GAN_JOINT_D_STEPS=3
GAN_JOINT_LR_MULT=2.0

GAN_AUDIO_OUT_MAX=1.0
GAN_AUDIO_SCALE_MAX=0.3
GAN_AUDIO_BIAS_MAX=0.1
GAN_VIDEO_OUT_MAX=20.0
GAN_VIDEO_SCALE_MAX=8.0

LOG_INTERVAL=5
SAVE_INTERVAL=0
EVAL_NUM_BATCHES=50
WARMUP_RATIO=0.2
RAMP_RATIO=0.2

# 合成特征训练参数
GAN_CHECKPOINT=""
FAKE_EPOCHS=50
FAKE_OUTPUT_DIR=""
ATTACK_SRC=-1
ATTACK_DST=-1
ATTACK_PROB=0.0

# 评估额外参数
EVAL_NO_TSNE=0
EVAL_NO_EXTRA_METRICS=0
EVAL_NO_DOMAIN_CLF=0
EVAL_MAX_METRIC_SAMPLES=""
EVAL_USE_TRAIN=0

# ========== 解析命令行参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --gan_epochs) GAN_EPOCHS="$2"; shift 2 ;;
        --num_epochs) FAKE_EPOCHS="$2"; shift 2 ;;
        --fake_epochs) FAKE_EPOCHS="$2"; shift 2 ;;
        --gan_checkpoint) GAN_CHECKPOINT="$2"; shift 2 ;;
        --fake_output_dir) FAKE_OUTPUT_DIR="$2"; shift 2 ;;
        --attack_src_label) ATTACK_SRC="$2"; shift 2 ;;
        --attack_dst_label) ATTACK_DST="$2"; shift 2 ;;
        --attack_prob) ATTACK_PROB="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --audio_feat) AUDIO_FEAT="$2"; shift 2 ;;
        --video_feat) VIDEO_FEAT="$2"; shift 2 ;;
        --gan_lr_g) GAN_LR_G="$2"; shift 2 ;;
        --gan_lr_d) GAN_LR_D="$2"; shift 2 ;;
        --gan_rf_weight) GAN_RF_WEIGHT="$2"; shift 2 ;;
        --gan_aux_weight) GAN_AUX_WEIGHT="$2"; shift 2 ;;
        --gan_cls_weight) GAN_CLS_WEIGHT="$2"; shift 2 ;;
        --gan_joint_weight) GAN_JOINT_WEIGHT="$2"; shift 2 ;;
        --gan_fm_weight) GAN_FM_WEIGHT="$2"; shift 2 ;;
        --gan_mom_weight) GAN_MOM_WEIGHT="$2"; shift 2 ;;
        --gan_joint_d_steps) GAN_JOINT_D_STEPS="$2"; shift 2 ;;
        --gan_joint_lr_mult) GAN_JOINT_LR_MULT="$2"; shift 2 ;;
        --gan_audio_out_max) GAN_AUDIO_OUT_MAX="$2"; shift 2 ;;
        --gan_audio_scale_max) GAN_AUDIO_SCALE_MAX="$2"; shift 2 ;;
        --gan_audio_bias_max) GAN_AUDIO_BIAS_MAX="$2"; shift 2 ;;
        --gan_video_out_max) GAN_VIDEO_OUT_MAX="$2"; shift 2 ;;
        --gan_video_scale_max) GAN_VIDEO_SCALE_MAX="$2"; shift 2 ;;
        --log_interval) LOG_INTERVAL="$2"; shift 2 ;;
        --save_interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --eval_num_batches) EVAL_NUM_BATCHES="$2"; shift 2 ;;
        --eval_no_tsne) EVAL_NO_TSNE=1; shift 1 ;;
        --eval_no_extra_metrics) EVAL_NO_EXTRA_METRICS=1; shift 1 ;;
        --eval_no_domain_clf) EVAL_NO_DOMAIN_CLF=1; shift 1 ;;
        --eval_max_metric_samples) EVAL_MAX_METRIC_SAMPLES="$2"; shift 2 ;;
        --eval_use_train) EVAL_USE_TRAIN=1; shift 1 ;;
        --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
        --ramp_ratio) RAMP_RATIO="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="BASE_$(date +"%m%d_%H%M")"
fi

MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"

LOG_DIR="results/logs"
GAN_CKPT_DIR="results/local_gan"
ANALYSIS_DIR="results/gan_analysis/ckpt_${GAN_EPOCHS}_${EXP_NAME}"

mkdir -p "$LOG_DIR" "$GAN_CKPT_DIR"

if [ "$MODE" = "gan" ]; then
    LOG_FILE="${LOG_DIR}/$(date +"%y.%m%d").${EXP_NAME}.txt"
else
    MODE_TAG="$(echo "$MODE" | tr '[:lower:]' '[:upper:]')"
    MODE_TAG="${MODE_TAG/FAKE_/}"
    LOG_FILE="${LOG_DIR}/$(date +"%y.%m%d").${EXP_NAME}.FAKE_${MODE_TAG}.txt"
fi

# ========== 打印实验信息 ==========
echo "============================================================" | tee "$LOG_FILE"
echo "GAN / Fake 实验自动化脚本（主干精简版）" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "模式: $MODE" | tee -a "$LOG_FILE"
echo "实验名称: $EXP_NAME" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$MODE" = "gan" ]; then
    # ========== Step 1: 训练 ==========
    echo "【Step 1/2】开始 GAN 训练..." | tee -a "$LOG_FILE"

    TRAIN_CMD="python train_local_gan.py \
      --model_path $MODEL_PATH \
      --exp_name $EXP_NAME \
      --seed $SEED \
      --gan_epochs $GAN_EPOCHS \
      --gan_lr_g $GAN_LR_G --gan_lr_d $GAN_LR_D \
      --gan_rf_weight $GAN_RF_WEIGHT --gan_aux_weight $GAN_AUX_WEIGHT \
      --gan_cls_weight $GAN_CLS_WEIGHT --gan_joint_weight $GAN_JOINT_WEIGHT \
      --gan_fm_weight $GAN_FM_WEIGHT --gan_mom_weight $GAN_MOM_WEIGHT \
      --gan_joint_d_steps $GAN_JOINT_D_STEPS --gan_joint_lr_mult $GAN_JOINT_LR_MULT \
      --gan_audio_out_max $GAN_AUDIO_OUT_MAX --gan_audio_scale_max $GAN_AUDIO_SCALE_MAX --gan_audio_bias_max $GAN_AUDIO_BIAS_MAX \
      --gan_video_out_max $GAN_VIDEO_OUT_MAX --gan_video_scale_max $GAN_VIDEO_SCALE_MAX \
      --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL \
      --warmup_ratio $WARMUP_RATIO --ramp_ratio $RAMP_RATIO \
      --batch_size $BATCH_SIZE \
      --audio_feat $AUDIO_FEAT --video_feat $VIDEO_FEAT"

    echo "$TRAIN_CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    echo "(fdmm) xp@W2L:~/fed-multimodal/fed_multimodal/Local$ $TRAIN_CMD" >> "$LOG_FILE"
    eval $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"

    # ========== Step 2: 评估 ==========
    echo "" | tee -a "$LOG_FILE"
    echo "【Step 2/2】开始 GAN 评估..." | tee -a "$LOG_FILE"

    CKPT_FILE="results/local_gan/ckpt_${GAN_EPOCHS}_${EXP_NAME}.pt"
    EVAL_CMD="python eval_local_gan_quality.py \
      --checkpoint $CKPT_FILE \
      --model_path $MODEL_PATH \
      --num_batches $EVAL_NUM_BATCHES \
      --output_dir $ANALYSIS_DIR"

    if [ "$EVAL_NO_TSNE" -eq 1 ]; then
        EVAL_CMD="$EVAL_CMD --no_tsne"
    fi
    if [ "$EVAL_NO_EXTRA_METRICS" -eq 1 ]; then
        EVAL_CMD="$EVAL_CMD --no_extra_metrics"
    fi
    if [ "$EVAL_NO_DOMAIN_CLF" -eq 1 ]; then
        EVAL_CMD="$EVAL_CMD --no_domain_clf"
    fi
    if [ -n "$EVAL_MAX_METRIC_SAMPLES" ]; then
        EVAL_CMD="$EVAL_CMD --max_metric_samples $EVAL_MAX_METRIC_SAMPLES"
    fi
    if [ "$EVAL_USE_TRAIN" -eq 1 ]; then
        EVAL_CMD="$EVAL_CMD --use_train"
    fi

    echo "$EVAL_CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    echo "(fdmm) xp@W2L:~/fed-multimodal/fed_multimodal/Local$ $EVAL_CMD" >> "$LOG_FILE"
    eval $EVAL_CMD 2>&1 | tee -a "$LOG_FILE"
elif [ "$MODE" = "fake_train" ] || [ "$MODE" = "fake_attack" ]; then
    ROOT_DIR="results/fake_training"
    if [ "$MODE" = "fake_attack" ]; then
        ROOT_DIR="results/fake_attack"
    fi

    if [ -z "$GAN_CHECKPOINT" ]; then
        GAN_CHECKPOINT="results/local_gan/ckpt_${GAN_EPOCHS}_${EXP_NAME}.pt"
    fi
    if [ ! -f "$GAN_CHECKPOINT" ]; then
        echo "错误: GAN checkpoint 不存在: $GAN_CHECKPOINT" | tee -a "$LOG_FILE"
        ls -la results/local_gan/ckpt_*.pt 2>/dev/null | tee -a "$LOG_FILE" || true
        exit 1
    fi
    if [ -z "$FAKE_OUTPUT_DIR" ]; then
        FAKE_OUTPUT_DIR="${ROOT_DIR}/$(date +"%y%m%d_%H%M%S")_${EXP_NAME}"
    fi

    mkdir -p "$ROOT_DIR"

    if [ "$MODE" = "fake_attack" ] && { [ "$ATTACK_SRC" -lt 0 ] || [ "$ATTACK_DST" -lt 0 ]; }; then
        echo "错误: fake_attack 需要设置 --attack_src_label / --attack_dst_label" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "【Fake】开始训练分类器..." | tee -a "$LOG_FILE"
    FAKE_CMD="python train_with_fake.py \
      --gan_checkpoint $GAN_CHECKPOINT \
      --model_path $MODEL_PATH \
      --num_epochs $FAKE_EPOCHS \
      --output_dir $FAKE_OUTPUT_DIR \
      --exp_name $EXP_NAME"

    if [ "$MODE" = "fake_attack" ]; then
        FAKE_CMD="$FAKE_CMD --init_from_model --attack_src_label $ATTACK_SRC --attack_dst_label $ATTACK_DST --attack_prob $ATTACK_PROB"
    fi

    echo "$FAKE_CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    eval $FAKE_CMD 2>&1 | tee -a "$LOG_FILE"
else
    echo "Unknown mode: $MODE (valid: gan, fake_train, fake_attack)" | tee -a "$LOG_FILE"
    exit 1
fi

# ========== 完成 ==========
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "实验完成！" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "输出文件位置:" | tee -a "$LOG_FILE"
echo "  - 日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
if [ "$MODE" = "gan" ]; then
    echo "  - GAN Checkpoint: $CKPT_FILE" | tee -a "$LOG_FILE"
    echo "  - 分析结果: $ANALYSIS_DIR" | tee -a "$LOG_FILE"
else
    echo "  - GAN Checkpoint: $GAN_CHECKPOINT" | tee -a "$LOG_FILE"
    echo "  - 训练输出: $FAKE_OUTPUT_DIR" | tee -a "$LOG_FILE"
fi
echo "============================================================" | tee -a "$LOG_FILE"
