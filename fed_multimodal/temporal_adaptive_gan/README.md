# Temporal-adaptive 生成器兼容实现

本包保留 temporal-adaptive 的配置、模型、损失和训练器，并接入与 DTM 相同的客户端隔离 lifecycle。将完整场景配置中的 `generator.variant` 改为 `temporal_adaptive` 后通过统一 runner 训练。

```bash
python -m mflpoison.runner --config path/to/temporal-scenario.yaml

python experiments/evaluate_generator.py \
  --generator temporal_adaptive --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt
```

旧训练文件仅作为统一 runner 的兼容别名，独立生成和集中式训练入口已删除。
