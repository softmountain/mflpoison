# DTM 生成器兼容实现

本包提供统一场景默认使用的 Distributional Temporal Matching 生成器、判别器、损失和训练器。它由 `mflpoison.adapters.fedmm.generator.FedMMGeneratorTrainer` 调用，每个恶意客户端只使用自己的 partition。

训练入口：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
```

旧 checkpoint 评估：

```bash
python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt
```

独立集中式训练和旧 shell pipeline 已删除。
