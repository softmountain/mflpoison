# MFL-Poison 渐进式重构说明

## 边界

`fed_multimodal/` 继续作为 FDMM 基线和旧 checkpoint 的兼容实现。
`mflpoison/` 是新的研究框架，负责统一生成器、合成数据、联邦攻击和防御协议。
旧入口暂不删除，避免破坏已经训练完成的实验。

## GAN 家族

```text
teacher_guided
kplus1
  legacy
  temporal_adaptive
  dtm
```

统一加载入口：

```python
from mflpoison.generators import load_generator_backend

backend = load_generator_backend("dtm", "path/to/checkpoint.pt", "cuda")
synthetic = backend.generate(target_labels, train_labels=train_labels)
```

生成数据通过 `SyntheticBatch` 交换，显式区分：

- `condition_labels`：生成器条件类别；
- `train_labels`：受害模型训练时看到的标签；
- `source_labels`：定向攻击的来源类别；
- `metadata`：生成器变体、checkpoint、配置与随机种子。

## 标准流水线

```text
train_generator
  -> evaluate_generator
  -> generate_synthetic
  -> evaluate_tstr
  -> federated attack
  -> defense filter
  -> robust aggregation
  -> clean/attack evaluation
```

`experiments/` 中的入口负责兼容旧训练器。核心代码不应再调用
`fed_multimodal/Local` 下的具体脚本。

## 联邦插入点

`FederatedEngine` 只负责编排：

```text
client_runner
  -> ClientUpdate[]
  -> UpdateFilter[]
  -> Aggregator
  -> global_state
```

当前提供：

- `WeightedMean` / `FedAvg`；
- `CoordinateMedian`；
- `TrimmedMean`；
- `Krum`;
- `NormClipper`。

现有 FDMM Client 可通过一个 `client_runner` 适配器接入，不需要复制联邦训练循环。

## 兼容策略

1. 旧 `*.pt` checkpoint 路径保持不变；
2. 旧合成数据字典可由 `SyntheticBatch.from_dict()` 读取；
3. 新产物默认保存 schema version；
4. 旧指标名保留，同时新增语义明确的指标；
5. 旧 Local 命令在完成迁移和等价性测试前不删除。
