# 实验分析工具

完整的联邦训练、中毒与防御实验统一使用：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml
```

本目录不提供第二套训练入口。新实验统一调用 runner，超参数差异写入 `configs/experiments/`。

多配置、多 GPU 正式实验使用：

```bash
bash scripts/run_experiments.sh \
  0:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-1_poison-20pct_generator-epochs-20.yaml:42 \
  1:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-2_poison-50pct_generator-epochs-20.yaml:42
```

runner 产物统一写入 `artifact/<config-group>/<config-name>/<YYYYMMDD-HHMMSS>_seed-N_git-<sha>/`，批处理状态写入 `artifact/batches/<YYYYMMDD-HHMMSS>/status.tsv`。直接位于 `configs/experiments/` 的配置省略 config-group 层。

## 旧 checkpoint 评估

```bash
python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --partition test --num_batches 20
```

支持 `teacher_guided`、`kplus1_legacy`、`dtm` 和 `temporal_adaptive`，实现位于 `fed_multimodal.legacy_evaluation`。

默认读取 FedMM `test`。训练侧评估必须显式使用 `--partition client --client-id ID`，不会合并多个客户端数据。

## 手工生成与 TSTR

```bash
python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --num_samples 5100 --target_strategy balanced \
  --attack_mode clean_label \
  --output artifact/manual/dtm-synthetic.pt

python experiments/evaluate_tstr.py \
  --synthetic_data artifact/manual/dtm-synthetic.pt -- \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --num_epochs 100
```

手工生成文件使用 canonical `SyntheticBatch` schema。TSTR 在同一 FedMM `dev` 上选模，模型确定后只在 `test` 上评估一次。

这些兼容工具的输出放在 `artifact/manual/` 或 `artifact/legacy_evaluation/`，不会合并进生产 runner 的 `summary.json`。完整调用关系和产物结构见 [当前流程与结果结构](../docs/CURRENT_PIPELINE_STRUCTURE.md)。
