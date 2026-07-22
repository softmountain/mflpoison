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
ScenarioRunner
  -> FedMM UCF101 adapter
  -> clean FedAvg pretraining
  -> dev-only M* selection
  -> one generator lifecycle per malicious client partition
  -> clean / attack / defended branches on one client schedule
  -> validate / detect / decide / sanitize / aggregate
  -> test, attack, and detection evaluation
```

生产入口是：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
```

配置严格保留 `dataset/model/federation/generator/attack/defense/evaluation/artifacts`
八个 section。未知字段和当前版本不支持的组件在读取数据前失败，不做嵌套配置拍平或静默覆盖。

## 联邦插入点

`FedAvgCoordinator` 只负责编排：

```text
broadcast GlobalSnapshot
  -> selected ClientDataBundle
  -> ClientUpdate(delta)[]
  -> server protocol validation
  -> robust detectors and DefenseDecision[]
  -> optional NormClipper sanitizer
  -> Aggregator
  -> next GlobalSnapshot and RoundRecord
```

生成器只能接收绑定到一个 `client_id` 和 `partition_hash` 的本地 loader。
服务器防御只接收 `ClientUpdate`，不读取客户端 loader、合成样本或生成器对象。

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
5. DTM 与 temporal-adaptive 旧 checkpoint 继续由统一 backend 加载；
6. 旧集中式训练命令保留文件名，但作为场景 runner 的薄 wrapper。

## 可复现与恢复

- 预训练和三个分支使用预先生成并写入 manifest 的采样计划；
- `clean`、`attack`、`defended` 从同一个 M* 开始；
- resume state、snapshot、generator manifest 和 round-record bundle 都带内容哈希；
- resume 会恢复在线生成器 warm-start 状态和可选 EWMA reputation；
- 反序列化的 snapshot、update、decision 和 round record 会重新执行构造校验；
- test 只用于报告，M* 只由 dev 指标和 patience 规则选择。
