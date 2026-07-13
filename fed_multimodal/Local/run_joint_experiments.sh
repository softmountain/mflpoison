#!/bin/bash
# ==============================================================================
# Joint Critic 验证实验脚本
# 
# 目标：验证新 JointCritic (两塔+LN+warmup/ramp) 是否能在不伤边缘分布的情况下
#       带来跨模态收益
#
# Phase-1: 验证不会破坏 OUTFIX 的边缘分布
#   E0: OUTFIX_J0 (joint=0, baseline)
#   E1: OUTFIX_JNEW_w0.05 (joint=0.05, 新结构+warmup/ramp)
#
# Phase-2: 小网格找安全的 joint 强度
#   target_joint ∈ {0.02, 0.05, 0.1}
#
# 判断标准：
#   红线1: audio fake std 不能比 E0 低太多 (跌 >20% 就算伤边缘)
#   红线2: audio cosine 不能长期显著下降
#   绿灯: joint gap 逐步缩小，video L2/cosine 不变差
# ==============================================================================

set -e

# 日期标记
DATE_TAG=$(date +"%y.%m%d")
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

# 公共参数 (基于 OUTFIX 配置)
COMMON_ARGS="
    --model_path results/local_training/best_model.pt
    --seed 42
    --gan_epochs 200
    --gan_lr_g 2e-4
    --gan_lr_d 1e-4
    --gan_rf_weight 2.0
    --gan_aux_weight 1.0
    --gan_cls_weight 0.1
    --gan_fm_weight 0.0
    --gan_mom_weight 0.0
    --gan_audio_out_max 30.0
    --gan_audio_scale_max 10.0
    --gan_audio_bias_max 5.0
    --log_interval 5
    --eval_num_batches 50
"

# ==============================================================================
# Phase-1: 验证边缘分布不被破坏
# ==============================================================================

echo "=============================================="
echo "Phase-1: 验证边缘分布不被破坏"
echo "=============================================="

# E0: OUTFIX baseline (joint=0)
echo ""
echo "[E0] OUTFIX_J0: joint=0 (baseline)"
echo "目的: 确认 OUTFIX 的 audio std/range 能稳定拉开"
bash run_gan_experiment.sh $COMMON_ARGS \
    --exp_name "OUTFIX_J0" \
    --gan_joint_weight 0.0

# E1: OUTFIX + joint (新结构 + warmup/ramp + target=0.05)
echo ""
echo "[E1] OUTFIX_JNEW_w0.05: joint=0.05 (新结构+warmup/ramp)"
echo "目的: 看引入 joint 后 audio 的 std/range 是否保持不被压回去"
bash run_gan_experiment.sh $COMMON_ARGS \
    --exp_name "OUTFIX_JNEW_w0.05" \
    --gan_joint_weight 0.05

echo ""
echo "=============================================="
echo "Phase-1 完成！请检查日志对比:"
echo "  - $LOG_DIR/${DATE_TAG}.OUTFIX_J0.txt"
echo "  - $LOG_DIR/${DATE_TAG}.OUTFIX_JNEW_w0.05.txt"
echo ""
echo "关键指标对比:"
echo "  1. [Edge-Audio] Fake std: E1 是否比 E0 低 >20%?"
echo "  2. [Joint-Critic] Gap: E1 的 gap 是否在 ramp 后逐步缩小?"
echo "  3. dist_audio 图: E1 的分布是否比 E0 差很多?"
echo "=============================================="

# ==============================================================================
# Phase-2: 小网格找安全的 joint 强度 (可选，根据 Phase-1 结果决定是否执行)
# ==============================================================================

# 取消下面的注释来运行 Phase-2
: '
echo ""
echo "=============================================="
echo "Phase-2: 小网格搜索 joint 强度"
echo "=============================================="

for JOINT_W in 0.02 0.1; do
    EXP_NAME="OUTFIX_JNEW_w${JOINT_W}"
    echo ""
    echo "[Phase-2] ${EXP_NAME}: joint=${JOINT_W}"
    bash run_gan_experiment.sh $COMMON_ARGS \
        --exp_name "$EXP_NAME" \
        --gan_joint_weight $JOINT_W
done

echo ""
echo "Phase-2 完成！"
'

# ==============================================================================
# Phase-3: 延后 warmup 实验 (如果 Phase-1 中 joint 仍伤 audio)
# ==============================================================================

# 取消下面的注释来运行 Phase-3 (需要先修改 train_local_gan.py 的 warmup 比例)
: '
echo ""
echo "=============================================="
echo "Phase-3: 延后 warmup (50%/20%)"
echo "如果 Phase-1 的 E1 仍伤 audio，尝试更保守的 warmup"
echo "=============================================="

# 注意：需要修改 train_local_gan.py 中的 warmup_epochs 和 ramp_epochs
# warmup_epochs = int(args.gan_epochs * 0.5)  # 改为 50%
# ramp_epochs = int(args.gan_epochs * 0.2)    # 保持 20%

EXP_NAME="OUTFIX_JNEW_w0.05_warmup50"
echo ""
echo "[Phase-3] ${EXP_NAME}: joint=0.05, warmup=50%"
bash run_gan_experiment.sh $COMMON_ARGS \
    --exp_name "$EXP_NAME" \
    --gan_joint_weight 0.05
'

echo ""
echo "=============================================="
echo "实验脚本执行完成！"
echo "=============================================="
