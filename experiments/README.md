# 实验入口与兼容工具

完整的 UCF101 联邦生成式中毒与防御实验只有一个生产入口：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml

python experiments/run_scenario.py \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml \
  --artifact-root artifacts/my-run
```

`run_scenario.py` 只转发到 `mflpoison.runner.__main__`。`train_generator.py`、
`fed_multimodal/Local/train_dtm_poison_gan.py` 和
`train_temporal_adaptive_gan.py` 是旧命令名兼容 wrapper，同样要求完整八段场景配置；
它们不再提供集中式 `full_train` 流程。

## 旧 checkpoint 评估

```bash
python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --partition test --num_batches 20
```

支持 `teacher_guided`、`kplus1_legacy`、`dtm` 和 `temporal_adaptive`。
实现位于 `fed_multimodal.legacy_evaluation`；大写 `fed_multimodal/Local` 中的
评估文件只是旧路径 wrapper。默认读取 FedMM `test`；训练侧评估必须显式传入
`--partition client --client-id ID`，不会联合读取多个客户端。

## 手工生成与 TSTR

```bash
python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --num_samples 5100 --target_strategy balanced \
  --attack_mode clean_label --output artifacts/manual/dtm-synthetic.pt

python experiments/evaluate_tstr.py \
  --synthetic_data artifacts/manual/dtm-synthetic.pt -- \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --num_epochs 100
```

手工生成文件使用 canonical `SyntheticBatch` schema；TSTR 也能读取保留的历史
schema。TSTR 固定在同一 FedMM `dev` 上选模，模型确定后只在 `test` 上评估一次，
不再从集中训练数据随机切 validation，因此不承诺与旧 TSTR 数值等价。

这些工具的输出属于 `artifacts/manual/` 或 `artifacts/legacy_evaluation/`，不会进入
统一 runner 的 `summary.json`。生产调用链、标准结果树和当前实际结果状态见
[`docs/CURRENT_PIPELINE_STRUCTURE.md`](../docs/CURRENT_PIPELINE_STRUCTURE.md)。
