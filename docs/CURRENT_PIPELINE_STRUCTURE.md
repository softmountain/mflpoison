# Fedpoi 当前流程、文件与结果结构说明

本文描述清理后的 `codex/ucf101-poison-pipeline` 工作区。它既是代码导航，也是运行 UCF101 联邦生成式中毒与服务器防御实验时的输入、过程、结果和分析文件规范。

## 1. 当前实际状态

- 唯一完整训练入口：`python -m mflpoison.runner --config configs/scenarios/ucf101_generative_poison_defense.yaml`。
- 当前保留 48 个 UCF101 分区特征文件，约 1.8 GB，覆盖 `alpha10/fold1` 和 `alpha50/fold1` 的音频/视频客户端、dev、test。
- 2026-07-23 对默认 `alpha10/fold1` 的 adapter 实测通过：10 个训练客户端、`dev/test` 完整，partition hash 为 `859e3a5fa58996c5d97ae3a64243bdebb95a2e5ad288126a4a5f527934abe744`。
- 本机 `fed_multimodal/datasets/ucf101/ucfTrainTestlist` 是被 Git 忽略的外部 symlink，目标包含三个官方 train/test split；其他机器需要自行准备或通过兼容 evaluator 的 `--dataset_dir` 指定。
- 旧 `fed_multimodal/Local/results`、`fed_multimodal/result`、旧日志、旧压缩包和缓存已删除，清理前约占 10 GB。
- 当前没有 `artifacts/` 目录，即尚未在清理后的统一结构中完成一次真实 50 轮场景运行。
- Git 不保存特征、checkpoint 或实验结果；它们均由 `.gitignore` 排除。

## 2. 清理后的工作区结构

```text
fedpoi/
├── README.md                         # 最短使用入口
├── REFACTORING.md                    # 最终架构与兼容边界
├── requirements.txt                 # Python 运行依赖
├── setup.py                          # fed-multimodal/mflpoison 安装配置
├── configs/scenarios/                # 唯一完整场景配置
├── experiments/                      # 统一入口和旧 checkpoint 工具
├── mflpoison/                        # 当前联邦攻防框架
├── fed_multimodal/                   # UCF101/FedMM 最小兼容实现
├── tests/                            # 自动化测试
├── docs/                             # 当前结构报告
├── artifacts/                        # 运行后生成；不进入 Git
└── fed_multimodal/results/feature/   # 本地 UCF101 输入；不进入 Git
```

### 2.1 根目录文件

| 文件 | 作用 |
|---|---|
| `.gitignore` | 排除 Python 缓存、环境、数据集、特征、checkpoint、日志和 `artifacts/`。 |
| `README.md` | 安装、输入、统一运行命令、结果入口和旧 checkpoint 兼容命令。 |
| `REFACTORING.md` | 规定客户端数据边界、服务器更新边界、核心契约、生成器生命周期和防御顺序。 |
| `requirements.txt` | PyTorch 1.13 系列、scikit-learn、SciPy、PyYAML 等依赖。 |
| `setup.py` | 安装 `fed_multimodal` 与 `mflpoison` Python 包。 |

### 2.2 `configs/`

`configs/scenarios/ucf101_generative_poison_defense.yaml` 是唯一生产配置，严格包含：

- `dataset`：特征根、fold、alpha、类别数和模态形状；
- `model`：`MMActionClassifier` 构造信息和可选初始 checkpoint；
- `federation`：预训练/攻击轮数、客户端采样、本地 epoch、学习率、seed、dev 收敛规则和设备；
- `generator`：DTM/temporal 变体、offline/online lifecycle、刷新间隔和训练参数；
- `attack`：恶意客户端、预算、replace/append、四个明确标签语义和调度；
- `defense`：检测器、sanitizer、聚合器、决策策略和可选 EWMA；
- `evaluation`：test、攻击与指标开关；
- `artifacts`：结果根、逐轮保存、manifest/snapshot/record 文件名。

旧分片 JSON 已删除，因为当前配置加载器不组合它们，也不允许拍平覆盖。

### 2.3 `experiments/`

| 文件 | 作用 |
|---|---|
| `run_scenario.py` | `mflpoison.runner` 的脚本兼容入口。 |
| `train_generator.py` | 兼容旧命令名，但实际要求完整场景并调用同一 runner。 |
| `evaluate_generator.py` | 根据 generator 名称分派到保留的旧 checkpoint evaluator。 |
| `generate_synthetic.py` | 通过统一 backend 生成 canonical `SyntheticBatch` 文件；不参与服务器主流程。 |
| `evaluate_tstr.py` | 分派到保留的 TSTR evaluator，验证合成数据训练效用。 |
| `_dispatch.py` | 维护 evaluator 路径映射并以相同 Python 环境启动兼容脚本。 |
| `README.md` | 说明统一入口与兼容工具。 |

### 2.4 `mflpoison/core/`

| 文件 | 作用 |
|---|---|
| `config.py` | 八段配置 dataclass、严格字段检查和 YAML/JSON 加载。 |
| `types.py` | `GlobalSnapshot`、`ClientUpdate`、`GeneratorArtifact`、`AttackSpec`、`DefenseDecision`、`RoundRecord` 等稳定契约。 |
| `protocols.py` | 客户端、聚合器、检测器和 sanitizer 的结构协议。 |
| `hashing.py` | 配置、映射、tensor map 和文件 SHA-256。 |
| `registry.py` | generator、aggregator 等命名组件注册表。 |
| `reproducibility.py` | Python/NumPy/PyTorch 随机状态捕获与恢复。 |
| `__init__.py` | 对外导出核心配置与类型。 |

### 2.5 `mflpoison/adapters/fedmm/`

| 文件 | 作用 |
|---|---|
| `ucf101.py` | 读取 FedMM fold/alpha/client pickle，校验音视频一一对应，建立 partition hash，构造模型和 dev/test evaluator。 |
| `client.py` | 将旧 `ClientFedAvg` 包装为确定性本地训练，输出 CPU `ClientUpdate(delta)`。 |
| `generator.py` | 每个恶意客户端训练 DTM/temporal 生成器，保存 teacher snapshot、checkpoint 和 lineage。 |
| `__init__.py` | 导出三个 adapter。 |

### 2.6 `mflpoison/federated/`

| 文件 | 作用 |
|---|---|
| `sampling.py` | 用固定 seed 预生成每轮客户端计划。 |
| `engine.py` | 单轮 broadcast、客户端训练、协议校验、防御和聚合。 |
| `coordinator.py` | 多轮训练、dev 评估、patience、best snapshot、恢复和 `RoundRecord` 编排。 |
| `__init__.py` | 对外导出联邦组件。 |

### 2.7 `mflpoison/generators/`

| 文件或目录 | 作用 |
|---|---|
| `base.py` | 统一生成 backend 接口。 |
| `registry.py` | generator 名称到 backend loader 的注册。 |
| `lifecycle.py` | 每客户端隔离的 offline-once/online-refresh 状态、seed、warm-start 和 resume。 |
| `kplus1/backends.py` | 加载 legacy、DTM 和 temporal-adaptive checkpoint 并生成 `SyntheticBatch`。 |
| `teacher_guided/backend.py` | 加载第一代 teacher-guided checkpoint。 |
| 各级 `__init__.py` | 注册并导出 backend。 |

### 2.8 `mflpoison/attacks/` 与 `mflpoison/data/`

| 文件 | 作用 |
|---|---|
| `client_selection.py` | 根据 seed 选择恶意客户端。 |
| `schedule.py` | 判断攻击轮次的 start/end/every 边界。 |
| `labels.py` | clean-label、label-flip 和平衡目标标签构造。 |
| `injector.py` | 将合成数据按预算注入 clean dataset。 |
| `strategy.py` | `GenerativeFeaturePoisoningStrategy`，统一生成、标签、预算、replace/append 和调度。 |
| `data/synthetic_dataset.py` | canonical synthetic dataset 和 clean/poison 混合视图。 |
| `data/fdmm_adapter.py` | 通用 FedMM batch 到 `MultimodalBatch` 的转换。 |
| 两个目录的 `__init__.py` | 对外导出攻击和数据 API。 |

### 2.9 `mflpoison/defenses/`

| 文件或目录 | 作用 |
|---|---|
| `validation.py` | 校验 snapshot hash、round、key、shape、dtype 和 NaN/Inf。 |
| `detection.py` | robust norm MAD、cosine MAD 和可选 EWMA reputation。 |
| `pipeline.py` | 固定执行 validate、score、decide、sanitize、aggregate。 |
| `common.py` | flatten、norm、稳健中心等公共更新操作。 |
| `registry.py` | 聚合器注册。 |
| `update_filter/norm_clipping.py` | 单项异常更新的 norm sanitizer。 |
| `robust_aggregation/weighted_mean.py` | 默认 FedAvg，权重使用 clean 客户端样本数。 |
| `coordinate_median.py` | 坐标中位数聚合。 |
| `trimmed_mean.py` | 截尾均值聚合。 |
| `krum.py` | 可选 Krum 实验聚合。 |
| `robust_aggregation/common.py` | 聚合输入校验和状态构造。 |
| 各级 `__init__.py` | 导出防御组件。 |

### 2.10 `mflpoison/artifacts/`、`evaluation/` 与 `training/`

| 文件 | 作用 |
|---|---|
| `artifacts/manifest.py` | 写配置、seed、Git commit、运行环境、partition 和采样计划。 |
| `artifacts/snapshot.py` | 保存/加载带内容哈希的 `GlobalSnapshot`。 |
| `artifacts/generator.py` | 保存/加载生成器 lineage JSON 并验证 checkpoint hash。 |
| `artifacts/round_record.py` | 保存单轮记录和多阶段 bundle，加载时重新校验契约及内容哈希。 |
| `artifacts/synthetic.py` | 保存/加载 canonical 或 legacy 合成批次。 |
| `evaluation/attack.py` | clean accuracy 和 targeted ASR 公共指标。 |
| `evaluation/detection.py` | precision、recall、FPR、FNR、AUROC 和混淆计数。 |
| `evaluation/intrinsic.py` | 生成特征的分布/多样性内在指标。 |
| `training/stability/finite.py` | tensor、loss 和梯度有限性检查。 |
| `training/stability/rnn.py` | RNN 二阶梯度稳定上下文。 |
| 各级 `__init__.py` | 导出 artifact、评估和稳定性 API。 |

### 2.11 `mflpoison/runner/`

| 文件 | 作用 |
|---|---|
| `__main__.py` | 解析 `--config`、`--artifact-root`，构建并运行场景。 |
| `scenario.py` | 完整生命周期、三分支公平比较、generator/defense 构建、resume 和结果持久化。 |
| `__init__.py` | 导出 runner 构建和结果类型。 |

### 2.12 `fed_multimodal/` 最小兼容层

| 路径 | 作用 |
|---|---|
| `constants/` | UCF101 类别数和 MFCC/MobileNetV2 特征维度。 |
| `dataloader/dataload_manager.py` | 读取 FedMM pickle、client simulation 配置并生成五 tensor batch。 |
| `model/mm_models.py` | 当前使用的 `MMActionClassifier` 及同文件内历史多模态模型定义。 |
| `trainers/fed_avg_trainer.py` | 真实 FedMM 客户端 SGD/FedAvg 更新。 |
| `trainers/evaluation.py` | acc、UAR、F1、top-5、truth/pred 等评估。 |
| `trainers/optimizer.py` | `ClientFedAvg` 保留的 FedProx optimizer 依赖。 |
| `generator/gan_generator.py` | teacher-guided 旧 checkpoint 模型。 |
| `generator/eval_gan_quality.py` | teacher-guided 兼容评估。 |
| `poison_gan/` | K+1 配置、模型、判别器扩展、loss、memory bank、metrics 和 legacy trainer。 |
| `dtm_poison_gan/` | DTM 配置、模型、loss 和 trainer。 |
| `temporal_adaptive_gan/` | temporal-adaptive 配置、模型、loss 和 trainer。 |
| `Local/dataloader.py` | 旧 evaluator 的集中读取兼容，不被生产训练调用。 |
| `Local/eval_*.py` | 四类旧 checkpoint evaluator；K+1 系列要求显式提供 legacy teacher，分析写入 `artifacts/legacy_evaluation/`。 |
| `Local/train_synthetic.py` | TSTR 兼容 evaluator。 |
| `Local/train_dtm_poison_gan.py` | DTM 旧文件名到统一 runner 的 wrapper。 |
| `Local/train_temporal_adaptive_gan.py` | temporal 旧文件名到统一 runner 的 wrapper。 |
| `features/` | 仅保留 UCF101 partition、simulation 和 feature extraction。 |
| `system.cfg` | teacher-guided evaluator 的兼容路径配置。 |
| `version.py` | 安装包版本。 |

`Local` evaluator 可以读取旧 checkpoint，但生产训练不能通过它访问 centralized `full_train`。
K+1/DTM/temporal evaluator 还需要调用者显式提供原 K 类 teacher checkpoint；依赖原始 UCF split metadata 的 evaluator 需要通过 `--dataset_dir` 指向外部数据目录。

### 2.13 `tests/`

| 测试文件 | 覆盖内容 |
|---|---|
| `test_core_types.py`、`test_core_contracts.py` | snapshot/update/artifact/decision/record 不变量、哈希和错误输入。 |
| `test_scenario_config.py` | 八段配置、未知字段和枚举校验。 |
| `test_fedavg_equivalence.py` | 新 delta FedAvg 与旧 full-state FedAvg 数值等价。 |
| `test_federated_engine.py`、`test_federated_coordinator.py` | 单轮/多轮、采样、收敛和恢复。 |
| `test_fedmm_client_adapter.py`、`test_fedmm_generator_adapter.py` | FedMM 客户端和每客户端生成器隔离。 |
| `test_generator_lifecycle.py`、`test_generator_backend_rng.py` | offline/online 刷新、warm-start、seed 与 RNG 隔离。 |
| `test_attacks.py`、`test_attack_strategy.py` | 标签方向、预算、replace/append 和调度。 |
| `test_defenses.py`、`test_defense_pipeline.py` | 检测器、裁剪、稳健聚合、决策和检测指标。 |
| `test_metrics.py` | 分类评估指标。 |
| `test_scenario_runner.py` | M*、三分支、artifact、summary、resume 和 dev/test 边界。 |
| `test_runner_builder_validation.py` | 默认 builder、配置交叉约束和 wrapper。 |
| `test_experiment_config.py` | 保留的 evaluator dispatch 路径。 |
| `test_training_stability.py` | NaN/Inf 与 RNN 稳定保护。 |

## 3. 输入特征结构

默认场景 `dataset.root: fed_multimodal/results`，`DataloadManager` 拼接以下路径：

```text
fed_multimodal/results/feature/
├── audio/mfcc/ucf101/
│   ├── alpha10/fold1/{0..9,dev,test}.pkl
│   └── alpha50/fold1/{0..9,dev,test}.pkl
└── video/mobilenet_v2/ucf101/
    ├── alpha10/fold1/{0..9,dev,test}.pkl
    └── alpha50/fold1/{0..9,dev,test}.pkl
```

- `alpha: 1.0` 转换为目录名 `alpha10`，`alpha: 5.0` 转换为 `alpha50`；
- `0.pkl` 至 `9.pkl` 是独立训练客户端；
- `dev.pkl` 只用于收敛和选择 M*；
- `test.pkl` 只用于 M* 和三条分支的最终报告；
- 音频 MFCC 的默认/最大形状为 `[500, 80]`，实际 batch 按其中最长样本动态 padding；视频为固定 MobileNetV2 序列 `[9, 1280]`；
- adapter 对每个客户端的音视频 key、标签、顺序和长度做配对检查，并由内容身份和 simulation 设置计算 partition hash。

## 4. 完整训练过程到代码和结果的映射

| 阶段 | 配置 | 主要代码 | 输入 | 输出/记录 | 关键失败条件 |
|---|---|---|---|---|---|
| 配置加载 | 全部八段 | `core/config.py`、`runner/__main__.py` | YAML | 内存中的 `ScenarioConfig` | 未知字段、缺 section、非法枚举立即失败 |
| 数据准备 | `dataset` | `adapters/fedmm/ucf101.py`、FedMM dataloader | client/dev/test pickle | `ClientDataBundle`、partition hash | 音视频不配对、dev/test 缺失、shape/类别不符 |
| 初始模型 | `model` | UCF adapter、`MMActionClassifier` | 随机状态或可选 checkpoint | `snapshots/initial.pt` | checkpoint hash、模型参数或 legacy args 不匹配 |
| 客户端计划 | `federation` | `federated/sampling.py` | client IDs、seed | manifest 中 pretrain/branch schedule | 每轮客户端数非法 |
| clean 预训练 | `federation.pretrain_rounds` | coordinator、engine、FedAvg client | initial snapshot、clean clients | pretrain round records | update 协议、NaN/Inf、snapshot lineage 错误 |
| M* 选择 | convergence 配置 | coordinator | 每轮 dev 指标 | `snapshots/m_star.pt`、`global_snapshot.pt` | dev 缺少 convergence metric |
| M* 测试 | `evaluation` | adapter evaluator | M*、test loader | `summary.json.m_star.test_metrics` | test 数据缺失或模型不兼容 |
| 基础生成器 | `generator`、`attack` | generator lifecycle、FedMM generator trainer | M*、恶意客户端自己的 loader | checkpoint、lineage JSON | client/partition/snapshot/hash 错配 |
| clean 分支 | `attack_rounds` | coordinator | M*、clean data、branch schedule | clean records/final snapshot/test metrics | 与公共 schedule 不一致会由 resume 校验失败 |
| attack 分支 | `attack` | attack strategy、synthetic dataset | M*、生成器、恶意客户端本地数据 | attack records/final snapshot/ASR | 预算、标签、调度或 artifact lineage 非法 |
| defended 分支 | `defense` | validation、detectors、pipeline、aggregator | 与 attack 相同更新 | decisions、processed updates、final snapshot | 更新 schema、检测配置或聚合条件非法 |
| 汇总 | `evaluation`、`artifacts` | runner persistence | 三分支结果 | manifest、summary、round bundle、resume | 写入前会重验 artifact/checkpoint hash |

三个分支从同一 M* 开始并使用完全相同的 branch schedule；差别只能来自攻击数据视图和服务器防御。

## 5. 标准结果目录

完整运行后的默认结构为：

```text
artifacts/ucf101_generative_poison_defense/
├── manifest.json
├── resume_state.pt
├── summary.json
├── global_snapshot.pt
├── round_records.pt
├── snapshots/
│   ├── initial.pt
│   ├── m_star.pt
│   ├── clean/final.pt
│   ├── attack/final.pt
│   └── defended/final.pt
├── round_records/
│   ├── pretrain/round-0000.pt ...
│   ├── clean/round-0000.pt ...
│   ├── attack/round-0000.pt ...
│   └── defended/round-0000.pt ...
├── generator_checkpoints/
│   └── <phase>/<client_id>/<snapshot_hash_prefix>/
│       ├── teacher_snapshot.pt
│       └── dtm.pt 或 temporal_adaptive.pt
└── generators/
    └── <phase>/<client_id>/<generator_artifact_hash>.json
```

### 5.1 `manifest.json`

记录 schema version、experiment/config hash、完整配置、seed、Git commit、Python/PyTorch/CUDA 环境、partition hash、客户端列表、恶意客户端、pretrain schedule、branch schedule，以及运行结束后的 M* 和三分支最终 snapshot hash。

### 5.2 snapshot 文件

每个 snapshot 包含 CPU `state`、round、dev metrics、`ModelSpec`、partition hash、metadata 和 content hash。`global_snapshot.pt` 是按配置文件名保存的 M* 副本；test 指标不写入 snapshot，避免参与选模。

### 5.3 generator 文件

- checkpoint 保存模型、优化器、epoch、训练 metrics 和 lineage；
- lineage JSON 保存 client ID、partition hash、父 snapshot hash、variant、seed、checkpoint path/hash、trained round 和 refresh index；
- JSON 和 checkpoint hash 不一致时加载失败；
- scenario 中合成样本只在恶意客户端 dataloader 内存视图中使用，不默认落盘；手工 `generate_synthetic.py` 的输出路径由调用者指定。

### 5.4 `round_records/` 与 `round_records.pt`

每个 `RoundRecord` 保存：

1. round 和 base snapshot hash；
2. 采样客户端；
3. 原始 `ClientUpdate(delta)`；
4. 每客户端 norm/cosine 分数、阈值、accept/clip/reject/quarantine、原因和最终权重；
5. 裁剪或过滤后的更新；
6. 聚合后状态及诊断；
7. 当轮 dev 评估。

单轮文件便于审计，`round_records.pt` 是按 `pretrain/clean/attack/defended` 分组的完整 bundle。读取时重新执行所有 constructor 校验并核对 record hash。

### 5.5 `summary.json` 与结果分析

`summary.json` 是主要机器可读结果：

- `m_star.dev_metrics`：选择 M* 的 dev acc/UAR/F1/loss；
- `m_star.test_metrics`：M* 固定后第一次 test 报告；
- `branches.<name>.dev_metrics/test_metrics`：三个最终模型指标；
- `attack_success_rate`：test 中 `victim_eval_class` 样本被预测为 `goal_prediction_class` 的比例；
- `detection_metrics`：defended 每轮决策汇总出的 precision、recall、FPR、FNR、AUROC 和 TP/FP/TN/FN；
- `clean_utility_drop`：clean 分支 test accuracy 减去 attack/defended 分支 test accuracy；
- `generator_artifacts`：各恶意客户端最终使用的 artifact hash。

建议分析顺序：

1. 用 clean/attack/defended test accuracy、UAR 和 F1 比较正常效用；
2. 用 attack 与 defended ASR 比较攻击效果和防御收益；
3. 用 detection metrics 判断服务器识别质量；
4. 从逐轮 decisions 检查误报集中在哪些客户端和轮次；
5. 从 aggregation diagnostics 检查拒绝/裁剪是否造成权重失衡；
6. 用 manifest、snapshot、generator 和 round hash 复核实验 lineage。

runner 当前不自动输出 PNG、CSV、Markdown 结论或可视化。任何人工报告必须注明所依据的 `summary.json` 和 `round_records.pt` 路径，不能把旧实验报告混入当前结果。

## 6. Offline 与 Online 生成器结果差别

### `offline_once`（默认）

- M* 后在 `generator_checkpoints/base/` 为每个恶意客户端训练一次；
- attack/defended lifecycle 从同一 base state 恢复，不重新训练；
- `generators/base` 保存原始 lineage，`generators/attack` 和 `generators/defended` 可保存引用同一 checkpoint 的阶段审计 JSON；
- resume 保存 base artifacts 和三分支使用状态。

### `online_refresh`

- 仍先建立 base artifact；
- attack 和 defended 分支按 `refresh_interval` 用当前广播 snapshot warm-start；
- 新 checkpoint 分别写入 `generator_checkpoints/attack/` 和 `generator_checkpoints/defended/`；
- 每次刷新产生新的父 snapshot hash、checkpoint hash、artifact hash 和 refresh index；
- `resume_state.pt` 同时恢复 branch lifecycle、warm-start artifact 和 EWMA reputation，刷新时点不会因恢复而漂移。

## 7. 已删除与保留原则

已删除内容包括其他数据集 benchmark、旧 UCF101 独立联邦脚本、demo、集中式 GAN/毒样本训练、旧 shell 矩阵、FedRS/SCAFFOLD/旧 server loop、历史 checkpoint、合成数据、分析图、报告、日志和打包文件。

保留内容必须至少满足一项：

- 被统一 runner 直接或传递 import；
- 用于生成当前 UCF101 输入特征；
- 用于读取已存在的四类旧 checkpoint；
- 验证当前核心契约和完整场景。

后续实验只应把结果写入 `artifacts/<run-name>/`。不得重新在 `fed_multimodal/Local/results`、`fed_multimodal/result` 或源码目录下创建结果树。
