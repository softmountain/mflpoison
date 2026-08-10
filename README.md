# MFL-Poison

MFL-Poison 用于研究 UCF101 音频/视频特征上的联邦生成式数据中毒和服务器侧异常更新防御。单次训练只有一个生产入口：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml
```

配置决定实验差异，Python 代码只实现可复用能力。需要改变恶意客户端数量、中毒比例或生成器轮数时，应新增或修改 `configs/experiments/` 下的配置，不复制训练代码；seed 和本次运行的阶段/分支可由命令行覆盖。BJMU 批量实验统一通过 `scripts/run_experiments.sh` 调度这个入口。

## 从配置到服务器防御

```text
实验配置
  -> mflpoison.runner.__main__：解析配置、seed、阶段/分支和结果目录
  -> runner.builder.build_default_runner：组装数据、模型、客户端训练器、攻击和防御
  -> runner.scenario.ScenarioRunner：生成或复用 M*、训练生成器、运行所选分支
  -> federated.coordinator.FedAvgCoordinator：逐轮选择客户端并收集更新
  -> attacks.GenerativeFeaturePoisoningStrategy：只为恶意客户端构造中毒数据
  -> defenses.DefensePipeline：校验、检测、决策、裁剪、聚合
  -> runner.persistence.ResultStore：保存逐轮记录和最终汇总
```

一次完整场景的执行顺序如下：

1. `mflpoison.core.config.load_scenario_config` 加载完整配置，或将 `base_config` 与 `overrides` 合并为完整配置。
2. `runner.builder` 根据配置创建 `UCF101FedMMAdapter`、`FedAvgClientTrainer`、DTM 生成器生命周期、攻击策略和可选服务器防御。
3. 未指定 `--m-star-path` 时，`ScenarioRunner` 执行 clean FedAvg 预训练，只用 dev 指标选择 M*；指定后则校验并复用共同 M*，跳过预训练。
4. `--m-star-only` 只生成并保存共同 M*，不运行任何实验分支。
5. 非 clean 分支中，每个恶意客户端只在自己的 partition 上训练生成器。
6. `ScenarioRunner` 从 M* 和固定客户端采样计划运行配置或 `--branch` 选择的 `clean`、`attack`、`defended` 分支。
7. 每轮中，`FedAvgCoordinator` 调用客户端本地训练；攻击分支先为恶意客户端替换或追加生成特征，再产生 `ClientUpdate`。
8. `defended` 分支把更新交给 `DefensePipeline`，依次完成合法性校验、异常检测、接受/裁剪/拒绝决策和服务器聚合。
9. runner 在 test 上计算效用、0→1 定向攻击和防御检测指标，并写入结果目录。attack-only/defended-only 运行可通过 `--canonical-clean` 引用共同 clean 基线并计算可比的 Delta ASR。

更细的文件级调用关系见 [当前流程与结果结构](docs/CURRENT_PIPELINE_STRUCTURE.md)。

## 配置

主配置位于 `configs/experiments/`：

- `ucf101_fdmm_dtm_poison_0to1.yaml`：clean 与 attack 分支；
- `ucf101_fdmm_dtm_poison_0to1_defense.yaml`：clean、attack 与 defended 分支；
- `ucf101_fdmm_dtm_poison_0to1_smoke.yaml`：短流程连通性测试；
- `ucf101_dtm_poison_strength/*.yaml`：使用人类可读文件名配置恶意客户端数、中毒比例和生成器 epoch，并明确声明为 attack-only；调度器为其注入共同 M* 和 canonical clean 路径。
- `ucf101_dtm_poison_strength_defense/*.yaml`：与攻击强度矩阵逐项对应，但启用服务器防御并声明为 defended-only。
- `ucf101_dtm_poison_strength_separate_gan_learning_rates/*.yaml`：沿用 11 组攻击者设置，显式使用 `lrG=3e-4`、`lrD=5e-5`，并在同一任务中配对运行 attack 与 defended。
- `ucf101_dtm_poison_strength_gan_step_ratios/*.yaml`：固定 20% 中毒、50 个生成器 epoch 和上述独立学习率，比较 1/2 个恶意客户端下每 batch 的生成器:鉴别器更新步数 10:1、20:1、40:1。

完整配置包含八部分：

- `dataset`：FedMM 特征根、fold、alpha、类别数和模态形状；
- `model`：分类模型构造参数和可选初始 checkpoint；
- `federation`：预训练/攻击轮数、客户端采样、本地训练、seed、分支以及 M* 生成/复用；
- `generator`：DTM 变体、生命周期和训练参数；`learning_rate` 对应 `lrG`，可选的 `discriminator_learning_rate` 对应 `lrD`，省略后兼容旧行为并使用与 `lrG` 相同的值；`generator_steps_per_batch` 和 `discriminator_steps_per_batch` 分别控制每个 dataloader batch 的生成器与鉴别器更新次数，默认 3:1；
- `attack`：恶意客户端、中毒预算、注入方式和 0→1 标签语义；
- `defense`：检测器、裁剪器、聚合器和决策策略；
- `evaluation`：test、攻击与检测指标开关，以及可选 canonical clean 产物路径；
- `artifact`：运行产物根目录。

命令行 `--seed` 会同时覆盖联邦训练与生成器 seed；`--run-dir` 可显式指定本次运行目录。`--branch` 可重复使用，只运行明确选择的分支：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --seed 42 \
  --branch clean \
  --branch attack \
  --run-dir artifact/ucf101_fdmm_dtm_poison_0to1/20260729-120000_seed-42_git-8d6f0057
```

canonical clean 协议使用四个低层阶段；正式批次由调度器自动完成，无需手工逐项启动：

```bash
# 1. 只生成共同 M*
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --seed 42 \
  --m-star-only \
  --run-dir artifact/canonical_clean/m_star/<mstar-run-id>

# 2. 从共同 M* 运行一份 clean replica；正式协议固定运行五份
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --seed 42 \
  --branch clean \
  --m-star-path artifact/canonical_clean/m_star/<mstar-run-id>/checkpoints/m_star.pt \
  --run-dir artifact/canonical_clean/clean-repeat-1/<clean-run-id>

# 3. 将恰好五份 clean summary 聚合为 canonical baseline
python -m mflpoison.runner.canonical_clean \
  --seed 42 \
  --m-star-path artifact/canonical_clean/m_star/<mstar-run-id>/checkpoints/m_star.pt \
  --output artifact/batches/<batch-id>/canonical_clean_seed-42.json \
  <clean-1-summary.json> <clean-2-summary.json> <clean-3-summary.json> \
  <clean-4-summary.json> <clean-5-summary.json>

# 4. attack-only 复用同一 M* 和五次 clean 的聚合基线
python -m mflpoison.runner \
  --config configs/experiments/ucf101_dtm_poison_strength/<attack-config>.yaml \
  --seed 42 \
  --branch attack \
  --m-star-path artifact/canonical_clean/m_star/<mstar-run-id>/checkpoints/m_star.pt \
  --canonical-clean artifact/batches/<batch-id>/canonical_clean_seed-42.json \
  --run-dir artifact/ucf101_dtm_poison_strength/<attack-config>/<attack-run-id>
```

`--m-star-only` 不能与 `--branch`、`--m-star-path` 或 `--canonical-clean` 组合。使用 `--canonical-clean` 时必须同时复用共同 M*；可单独运行 attack/defended，也可重复 `--branch` 配对运行两者，但不能再包含 clean 分支。

## 数据与安装

runner 直接读取 FedMM 生成的客户端特征：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/alpha10/fold1/
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/alpha10/fold1/
```

每种模态目录包含客户端 `0.pkl` 至 `9.pkl`，以及 `dev.pkl`、`test.pkl`。这些特征、模型 checkpoint 和实验结果均不进入 Git。

安装与测试：

```bash
pip install -e .
pip install -e ".[test]"
pytest -q
```

分析工具使用 `pip install -e ".[analysis]"`，旧特征提取和 checkpoint 评估工具使用 `pip install -e ".[legacy-features]"`。`requirements.txt` 保留 BJMU 核心运行环境的固定版本，完整服务器环境参考 `requirements/lock-py39-cu117.txt`。

BJMU 当前环境可直接运行：

```bash
conda run -p /mnt/sda/mtzh/xp/envs/fedpoi-py39 \
  python -c "import torch, fed_multimodal, mflpoison; print(torch.__version__)"
```

## 运行产物

`artifact/` 不需要预先创建。未指定 `--run-dir` 时，入口按配置位置创建：

```text
artifact/<config-name>/<YYYYMMDD-HHMMSS>_seed-<N>_git-<short-sha>/
artifact/<config-group>/<config-name>/<YYYYMMDD-HHMMSS>_seed-<N>_git-<short-sha>/
```

直接配置示例：

```text
artifact/ucf101_fdmm_dtm_poison_0to1_defense/20260729-120000_seed-42_git-8d6f0057/
├── config_resolved.yaml
├── run_manifest.json
├── summary.json
├── checkpoints/
│   ├── initial.pt
│   ├── m_star.pt
│   ├── clean_last.pt
│   ├── attack_last.pt
│   ├── defended_last.pt
│   └── generators/
├── generators/
└── round_records.pt
```

批处理脚本还会在每个 GPU 任务的运行目录中保存 `train.log`。

参数配置示例：

```text
artifact/ucf101_dtm_poison_strength/
└── malicious-clients-2_poison-50pct_generator-epochs-20/
    └── 20260729-120000_seed-42_git-8d6f0057/
```

`config_resolved.yaml` 保存全部实际参数，`run_manifest.json` 记录提交、工作区 dirty 状态、运行源码树 hash、运行环境、M* 来源和最终状态，逐轮客户端更新与服务器决策统一位于 `round_records.pt`。

`summary.json` 当前为 schema v3，保存 M*、实际选择的分支、分支 test/ASR、攻击暴露和防御检测结果。普通的同次 clean/attack 运行仍可计算运行内增量，此时 attack/defended 分支中有：

```text
delta_baseline = "run_clean"
delta_attack_success_rate
delta_asr_percentage_points
```

attack-only 或 defended-only 运行使用 `--canonical-clean` 后，summary 顶层增加 `canonical_clean` 来源与统计摘要，分支中改为：

```text
delta_baseline = "canonical_clean"
canonical_clean_attack_success_rate
canonical_clean_attack_success_rate_pct
delta_attack_success_rate
delta_asr_percentage_points
```

其中 `delta_attack_success_rate = ASR_attack - ASR_canonical_clean`，`delta_asr_percentage_points` 是相同增量的百分点表示。

批次级 canonical clean 产物位于：

```text
artifact/batches/<batch-id>/
├── status.tsv
├── canonical_clean_seed-<seed>.json
└── canonical_clean_seed-<seed>.log
```

`canonical_clean_seed-<seed>.json` 的 schema version 为 1。它记录共同 M*、partition、客户端日程、攻击源类/目标类、共同训练协议、源码身份、固定五份 clean replica 的 `summary_path`、run_dir、唯一 experiment ID、成功数、源类总数和 ASR，以及五个 ASR 的算术平均值 `asr_canonical_clean` 和总体标准差。聚合会核对 M* 生成运行的 completed manifest、seed、experiment ID、partition、checkpoint hash 与源码身份；加载时还会重新读取五份来源 summary/manifest 和共同 M* 并重建结果，因此 baseline 使用期间不能移动或删除这些来源产物。五份运行各自保留 `成功数/源类总数`，同一批测试样本不会合并成一个分母。

调度器会自动聚合该文件。需要单独复核时，可直接调用相同聚合入口；位置参数必须恰好给出正式协议的五个 clean-only `summary.json`：

```bash
python -m mflpoison.runner.canonical_clean \
  --seed 42 \
  --m-star-path artifact/canonical_clean/m_star/<mstar-run-id>/checkpoints/m_star.pt \
  --output artifact/batches/<batch-id>/canonical_clean_seed-42.json \
  <clean-1-summary.json> \
  <clean-2-summary.json> \
  <clean-3-summary.json> \
  <clean-4-summary.json> \
  <clean-5-summary.json>
```

## 多 GPU 批处理

`scripts/run_experiments.sh` 接收一个 GPU 池和多个 `CONFIG:SEED` 作业。`--experiment-branch` 选择统一运行 `attack`（默认）或 `defended`；`--experiment-branches attack,defended` 会让每个配置在同一个 runner 任务中运行严格配对的两条支线。作业不再预先绑定 GPU；脚本持续检查池内 GPU，只有在该卡没有计算进程且已用显存不高于空闲阈值时，才把下一个依赖已满足的任务分配给它。一张卡同时只运行一个调度任务。

```bash
PYTHON_BIN=/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python \
bash scripts/run_experiments.sh \
  --gpus 0,1,2,3 \
  --canonical-clean-config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --experiment-branches attack,defended \
  configs/experiments/ucf101_dtm_poison_strength_separate_gan_learning_rates/malicious-clients-1_poison-20pct_generator-epochs-5.yaml:42 \
  configs/experiments/ucf101_dtm_poison_strength_separate_gan_learning_rates/malicious-clients-2_poison-50pct_generator-epochs-20.yaml:42
```

`--gpus` 默认是 `0,1,2,3`，`--canonical-clean-config` 默认是上例的基准配置，`--experiment-branch` 默认是 `attack`。运行防御矩阵时可设置 `--experiment-branch defended`；需要配对结果时使用 `--experiment-branches attack,defended`，两种选项不能同时出现。还可用 `--monitor-interval` 设置轮询秒数、用 `--idle-memory-mib` 设置空闲显存阈值；默认分别为 30 秒和 1024 MiB。主机级全局锁保证同一主机只运行一个调度器实例；`SCHEDULER_LOCK_FILE` 仅供隔离测试或确认不会争用 GPU 的受控环境覆盖锁路径。

默认模式始终重新训练该 seed 的 M* 和五份 canonical clean。只有经明确批准复用旧基线时，才成对提供 `--reuse-m-star-path` 与 `--reuse-canonical-clean`；跨提交复用还必须显式提供 `--canonical-source-policy approved_reuse`。调度器会在预检后持续复核 clean Git 来源、canonical 严格重建结果以及 canonical/M* 文件哈希，变化时中止批次。

脚本会为输入中每个不同 seed 固定构造以下依赖图；同一 seed 的多个单支线或配对配置共享一份 M* 和 canonical clean：

```text
mstar
  -> clean-repeat-1 ┐
  -> clean-repeat-2 │
  -> clean-repeat-3 ├-> canonical-aggregate -> 所选分支配置 1
  -> clean-repeat-4 │                       └-> 所选分支配置 2 ...
  -> clean-repeat-5 ┘
```

- `mstar` 使用 `--m-star-only` 生成该 seed 的共同 M*；
- 五份 clean 使用相同 seed、partition、共同 M* 和客户端日程，并由 `--branch clean --m-star-path ...` 启动；四张卡时先运行四份，任一卡释放后自动补上第五份；
- `canonical-aggregate` 在五份 clean 全部成功后生成 `canonical_clean_seed-<seed>.json`；
- 每个实验配置随后由一个或两个 `--branch attack|defended` 配合 `--m-star-path ... --canonical-clean ...` 启动，不再重复训练 clean；配对模式的 `status.tsv` stage 为 `attack_defended`；
- 任一任务失败都会记为 `failed`；依赖它且尚未启动的任务同样停止，并记录 `failure_reason=dependency_failed:<job-id>`。
- 收到 `HUP`、`INT` 或 `TERM` 时，调度器停止派发、终止并回收已启动的独立进程组，把全部未完成任务写成 `failed` 和 `scheduler_interrupted:<signal>` 后释放主机锁。

每个 mstar、clean 和所选实验分支任务都有包含 stage、seed、repeat/序号的唯一 job ID 和运行目录，不会因同 seed 的并发任务在同一秒启动而碰撞。工作区有未提交候选时，run ID 的 Git 部分会附加 `-dirty`，并由 manifest 保存源码树 hash；正式实验仍应使用 clean 的里程碑提交。任务的 PID、GPU、退出码、依赖、时间戳、配置、run_dir 和失败原因持续写入：

```text
artifact/batches/<batch-id>/status.tsv
```

`status.tsv` 的状态为 `queued`、`running`、`completed` 或 `failed`，列为：

```text
job_id  stage  experiment  seed  repeat  depends_on  gpu  pid  status
exit_code  queued_at  started_at  finished_at  config  run_dir  failure_reason
```

GPU 任务的完整 stdout/stderr 位于各自 run_dir 的 `train.log`；canonical 聚合日志位于批次目录。队列未清空时，脚本会在任务结束后释放 GPU 并继续补位；批次结束后返回总体成功/失败状态，不删除正式 artifact。

队列配置必须位于仓库的 `configs/experiments/` 内，seed 必须在 NumPy 可接受的 `0..4294967295` 范围；runner 退出码为 0 后，调度器还会解析 `run_manifest.json` 并只接受顶层 `status=completed`。

## 目录边界

- `mflpoison/`：当前联邦攻防框架；
- `configs/experiments/`：完整实验和参数变体；
- `scripts/`：批处理和运行辅助脚本；
- `fed_multimodal/`：UCF101 数据、模型、FedAvg 与生成器兼容实现；
- `experiments/`：旧 checkpoint 的人工分析工具；
- `tests/`：自动化测试；
- `artifact/`：运行时按需创建的产物，不进入 Git；清理最后一次临时运行后不保留空目录。

本项目仅用于防御性安全研究和多模态联邦学习鲁棒性评估。
