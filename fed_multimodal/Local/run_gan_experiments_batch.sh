#!/bin/bash
# =====================================================================
# 批量 GAN 实验运行脚本 - 组件开关实验
# =====================================================================
# 功能：运行组件开关实验，验证每个模块是否在起作用
# 用法：bash run_gan_experiments_batch.sh
# =====================================================================

set -e

# ========== 配置 ==========
# 确保脚本在 Local 目录下执行
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 公共参数（Base Template）
MODEL_PATH="results/local_training/best_model.pt"
GAN_EPOCHS=200
BATCH_SIZE=32
EVAL_NUM_BATCHES=50

# ========== 组件开关实验配置 ==========
# 每个实验一行，格式：实验名称|参数列表
# 所有实验都基于 Base Template，只改动一个因素

experiments=(
    # ===== 第一轮：基线与正则强度 =====
    "BASE_NO_FM_MOM|--gan_fm_weight 0.0 --gan_mom_weight 0.0"
    "BASE|"
    "FM_MOM_0.1|--gan_fm_weight 0.1 --gan_mom_weight 0.1"

    # ===== 第二轮：Teacher 约束对比 =====
    "CLS0|--gan_cls_weight 0.0"
    "CLS_LOW|--gan_cls_weight 0.02"
    "CLS_DELAY|--gan_cls_weight 0.02 --warmup_ratio 0.4 --ramp_ratio 0.4"

    # ===== 第三轮：Joint 强度对比 =====
    "JOINT0|--gan_joint_weight 0.0"
    "JOINT_HIGH|--gan_joint_weight 0.1"

    # ===== 第四轮：输出范围调整 =====
    "OUTFIX|--gan_audio_out_max 3.0 --gan_audio_scale_max 1.0 --gan_audio_bias_max 0.5"
)

# ========== 运行实验 ==========
echo "============================================================"
echo "批量 GAN 实验"
echo "============================================================"
echo "总共 ${#experiments[@]} 个实验"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 创建日志目录
mkdir -p results/logs
BATCH_LOG="results/logs/batch_$(date +"%y%m%d_%H%M%S").log"

echo "批量日志文件: $BATCH_LOG"
echo "============================================================"
echo ""

# 记录批量实验信息
echo "============================================================" > "$BATCH_LOG"
echo "批量 GAN 实验" >> "$BATCH_LOG"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$BATCH_LOG"
echo "总共 ${#experiments[@]} 个实验" >> "$BATCH_LOG"
echo "============================================================" >> "$BATCH_LOG"
echo "" >> "$BATCH_LOG"

# 遍历实验
success_count=0
fail_count=0

for i in "${!experiments[@]}"; do
    # 解析实验配置
    IFS='|' read -r exp_name exp_params <<< "${experiments[$i]}"
    
    exp_num=$((i + 1))
    echo "------------------------------------------------------------"
    echo "实验 $exp_num/${#experiments[@]}: $exp_name"
    echo "参数: $exp_params"
    echo "开始时间: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"
    
    # 记录到批量日志
    echo "" >> "$BATCH_LOG"
    echo "------------------------------------------------------------" >> "$BATCH_LOG"
    echo "实验 $exp_num: $exp_name" >> "$BATCH_LOG"
    echo "参数: $exp_params" >> "$BATCH_LOG"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$BATCH_LOG"
    echo "------------------------------------------------------------" >> "$BATCH_LOG"
    
    # 运行实验
    if bash run_gan_experiment.sh \
        --exp_name "$exp_name" \
        --model_path "$MODEL_PATH" \
        --gan_epochs "$GAN_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --eval_num_batches "$EVAL_NUM_BATCHES" \
        $exp_params; then
        
        success_count=$((success_count + 1))
        echo "✓ 实验 $exp_num 完成" | tee -a "$BATCH_LOG"
    else
        fail_count=$((fail_count + 1))
        echo "✗ 实验 $exp_num 失败" | tee -a "$BATCH_LOG"
    fi
    
    echo "结束时间: $(date '+%H:%M:%S')" | tee -a "$BATCH_LOG"
    echo "" | tee -a "$BATCH_LOG"
done

# ========== 汇总结果 ==========
echo ""
echo "============================================================"
echo "批量实验完成！"
echo "============================================================"
echo "总实验数: ${#experiments[@]}"
echo "成功: $success_count"
echo "失败: $fail_count"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 记录汇总到批量日志
echo "" >> "$BATCH_LOG"
echo "============================================================" >> "$BATCH_LOG"
echo "批量实验完成" >> "$BATCH_LOG"
echo "============================================================" >> "$BATCH_LOG"
echo "总实验数: ${#experiments[@]}" >> "$BATCH_LOG"
echo "成功: $success_count" >> "$BATCH_LOG"
echo "失败: $fail_count" >> "$BATCH_LOG"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$BATCH_LOG"
echo "============================================================" >> "$BATCH_LOG"

# 如果有失败的实验，返回非零退出码
if [ $fail_count -gt 0 ]; then
    exit 1
fi
