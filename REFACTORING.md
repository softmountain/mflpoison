# MFL-Poison 最终架构边界

## 唯一生产流程

`mflpoison/` 是当前研究框架，`python -m mflpoison.runner` 是唯一完整训练入口。配置严格保留 `dataset/model/federation/generator/attack/defense/evaluation/artifacts` 八个 section，未知字段或不支持组件会在读取数据前失败。

```text
ScenarioRunner
  -> FedMM UCF101 adapter
  -> clean FedAvg pretraining
  -> dev-only M* selection
  -> one generator lifecycle per malicious client partition
  -> clean / attack / defended branches on one schedule
  -> validate / detect / decide / sanitize / aggregate
  -> test, attack and detection evaluation
```

## FedMM 兼容边界

`fed_multimodal/` 不再作为多数据集 benchmark 或独立实验入口，只保留新流程真实依赖的内容：

- UCF101 常量、数据加载和预处理；
- `MMActionClassifier`；
- FedAvg 客户端训练与评估；
- teacher-guided、K+1、DTM 和 temporal-adaptive checkpoint 实现；
- 旧 checkpoint evaluator 与 TSTR evaluator。

旧 demo、其他数据集、集中式训练、FedRS、SCAFFOLD 和匿名 server loop 已删除。删除的 CLI 不再属于公共接口。

## 数据与隐私边界

- adapter 只读取 FedMM 已生成的 fold/alpha/client partition；
- 生成器只能接收绑定到一个 `client_id` 和 `partition_hash` 的本地 loader；
- 攻击只发生在恶意客户端的数据视图；
- 服务器只接收 `ClientUpdate`，不能读取客户端 loader、合成样本或生成器对象；
- M* 只由 dev 指标、patience 和 min-delta 选择，test 仅用于最终报告。

## 稳定契约

- `GlobalSnapshot`：CPU 模型状态、round、dev 指标、模型规格、partition hash 和内容哈希；
- `ClientUpdate`：明确 delta、base snapshot hash、clean/train 样本数、聚合权重和 artifact lineage；
- `GeneratorArtifact`：客户端、partition、父 snapshot、变体、随机种子和 checkpoint hash；
- `AttackSpec`：条件类、训练标签、受害评估类、目标预测类、预算和调度；
- `DefenseDecision`：检测分数、阈值、动作、原因和最终权重；
- `RoundRecord`：原始更新、防御决定、处理后更新、聚合诊断和评估。

这些契约继续支持旧 full-state update 和旧合成字典的读取，但新代码统一产生 delta 和 canonical schema。

## 生成器生命周期

- `offline_once`：M* 后每个恶意客户端训练一次；attack/defended 分支复用同一基础 artifact；
- `online_refresh`：按 `refresh_interval` 使用当前广播 snapshot warm-start，刷新状态写入 resume；
- DTM 和 temporal-adaptive 共享 lifecycle 和 adapter，不复制联邦循环；
- 旧 checkpoint 通过 backend 加载，不允许旧训练脚本绕过客户端 partition。

## 防御流水线

固定顺序为协议校验、特征提取、检测评分、决策、裁剪/变换、聚合。默认使用 norm MAD 与 cosine-to-robust-center MAD；两项异常拒绝、单项异常裁剪、其余接受。`NormClipper` 是 sanitizer。FedAvg/weighted mean 为默认聚合，coordinate median、trimmed mean 和 Krum 为实验选项。

## 产物与恢复

manifest、snapshot、generator manifest、round record bundle 和 resume state 均记录哈希或 lineage。预训练和三条分支使用预生成采样计划；clean、attack、defended 从相同 M* 开始。完整路径和指标说明见 `docs/CURRENT_PIPELINE_STRUCTURE.md`。
