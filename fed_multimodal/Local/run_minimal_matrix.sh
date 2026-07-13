#!/bin/bash
# =====================================================================
# 最小实验矩阵 - Joint Critic 消融实验
# =====================================================================
# 三个实验在同一 seed、同一训练轮数下对比：
#   1. OUTFIX_J0: 基线（关闭 Joint）
#   2. JOINT_R02: Joint + 强 D_joint（R0-2 配置：3步+2xLR）
#   3. JOINT_R02_w0.02: Joint 权重更小（0.02），看是否更稳
#
# 每个 run 结束后自动跑评估（eval_local_gan_quality.py）
# =====================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ========== 公共配置 ==========
MODEL_PATH="results/local_training/best_model.pt"
GAN_EPOCHS=200
BATCH_SIZE=32
SEED=42

# R0-2 验证的配置
JOINT_D_STEPS=3
JOINT_LR_MULT=2.0

# 输出范围（OUTFIX 配置）
AUDIO_OUT_MAX=3.0
AUDIO_SCALE_MAX=2.0
AUDIO_BIAS_MAX=1.5

# ========== 实验定义 ==========
experiments=(
    # 格式: 实验名称|Joint权重|Joint D步数|Joint LR倍数|描述
    "OUTFIX_J0|0.0|1|1.0|基线：关闭 Joint Critic"
    "JOINT_R02|0.05|3|2.0|R0-2 配置：Joint=0.05 + 3步 + 2xLR"
    "JOINT_R02_w0.02|0.02|3|2.0|降低 Joint 权重到 0.02"
)

# ========== 运行 ==========
echo "============================================================"
echo "最小实验矩阵 - Joint Critic 消融"
echo "============================================================"
echo "公共配置:"
echo "  - Seed: $SEED"
echo "  - Epochs: $GAN_EPOCHS"
echo "  - OUTFIX 输出范围: audio_out=$AUDIO_OUT_MAX, scale=$AUDIO_SCALE_MAX"
echo "  - R0-2 Joint 增强: D_steps=$JOINT_D_STEPS, LR_mult=$JOINT_LR_MULT"
echo "============================================================"
echo ""

mkdir -p results/logs
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
BATCH_LOG="results/logs/minimal_matrix_${TIMESTAMP}.log"

echo "日志文件: $BATCH_LOG"
echo ""

# 记录实验信息
{
    echo "============================================================"
    echo "最小实验矩阵 - Joint Critic 消融"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo "公共配置:"
    echo "  - Seed: $SEED"
    echo "  - Epochs: $GAN_EPOCHS"
    echo "  - OUTFIX 输出范围: audio_out=$AUDIO_OUT_MAX, scale=$AUDIO_SCALE_MAX"
    echo ""
} > "$BATCH_LOG"

success_count=0
fail_count=0

for i in "${!experiments[@]}"; do
    IFS='|' read -r exp_name joint_weight joint_d_steps joint_lr_mult description <<< "${experiments[$i]}"
    
    exp_num=$((i + 1))
    echo "------------------------------------------------------------"
    echo "实验 $exp_num/${#experiments[@]}: $exp_name"
    echo "  描述: $description"
    echo "  Joint 配置: weight=$joint_weight, D_steps=$joint_d_steps, LR_mult=$joint_lr_mult"
    echo "------------------------------------------------------------"
    
    {
        echo ""
        echo "------------------------------------------------------------"
        echo "实验 $exp_num: $exp_name"
        echo "描述: $description"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"
    } >> "$BATCH_LOG"
    
    # 构建命令
    cmd="bash run_gan_experiment.sh \
        --exp_name $exp_name \
        --model_path $MODEL_PATH \
        --seed $SEED \
        --gan_epochs $GAN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gan_joint_weight $joint_weight \
        --gan_joint_d_steps $joint_d_steps \
        --gan_joint_lr_mult $joint_lr_mult \
        --gan_audio_out_max $AUDIO_OUT_MAX \
        --gan_audio_scale_max $AUDIO_SCALE_MAX \
        --gan_audio_bias_max $AUDIO_BIAS_MAX \
        --log_interval 10 \
        --eval_num_batches 50"
    
    echo "命令: $cmd"
    echo ""
    
    # 运行实验
    if $cmd 2>&1 | tee -a "$BATCH_LOG"; then
        success_count=$((success_count + 1))
        echo "✓ 实验 $exp_num ($exp_name) 完成" | tee -a "$BATCH_LOG"
    else
        fail_count=$((fail_count + 1))
        echo "✗ 实验 $exp_num ($exp_name) 失败" | tee -a "$BATCH_LOG"
    fi
    
    echo "" | tee -a "$BATCH_LOG"
done

# ========== 汇总结果 ==========
echo ""
echo "============================================================"
echo "实验完成！"
echo "============================================================"
echo "总实验数: ${#experiments[@]}"
echo "成功: $success_count"
echo "失败: $fail_count"
echo ""
echo "结果文件:"
echo "  - Checkpoints: results/local_gan/ckpt_${GAN_EPOCHS}_*.pt"
echo "  - Analysis: results/gan_analysis/ckpt_${GAN_EPOCHS}_<exp_name>/analysis_results.json"
echo "  - 批量日志: $BATCH_LOG"
echo "============================================================"

{
    echo ""
    echo "============================================================"
    echo "实验完成"
    echo "============================================================"
    echo "总实验数: ${#experiments[@]}"
    echo "成功: $success_count"
    echo "失败: $fail_count"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
} >> "$BATCH_LOG"

if [ $fail_count -gt 0 ]; then
    exit 1
fi
