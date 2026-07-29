# MFL-Poison

MFL-Poison 用于研究 UCF101 音频/视频特征上的联邦生成式数据中毒和服务器侧异常更新防御。完整实验只有一个入口：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml
```

配置决定实验差异，Python 代码只实现可复用能力。需要改变恶意客户端数量、中毒比例、生成器轮数或随机种子时，应新增或修改 `configs/experiments/` 下的配置，不复制训练代码。

## 从配置到服务器防御

```text
实验配置
  -> mflpoison.runner.__main__：解析配置、seed 和结果目录
  -> runner.builder.build_default_runner：组装数据、模型、客户端训练器、攻击和防御
  -> runner.scenario.ScenarioRunner：预训练、选择 M*、训练生成器、运行分支
  -> federated.coordinator.FedAvgCoordinator：逐轮选择客户端并收集更新
  -> attacks.GenerativeFeaturePoisoningStrategy：只为恶意客户端构造中毒数据
  -> defenses.DefensePipeline：校验、检测、决策、裁剪、聚合
  -> runner.persistence.ResultStore：保存逐轮记录和最终汇总
```

一次完整场景的执行顺序如下：

1. `mflpoison.core.config.load_scenario_config` 加载完整配置，或将 `base_config` 与 `overrides` 合并为完整配置。
2. `runner.builder` 根据配置创建 `UCF101FedMMAdapter`、`FedAvgClientTrainer`、DTM 生成器生命周期、攻击策略和可选服务器防御。
3. `ScenarioRunner` 先执行 clean FedAvg 预训练，只用 dev 指标选择 M*。
4. 每个恶意客户端只在自己的 partition 上训练生成器。
5. `ScenarioRunner` 从同一 M* 和同一客户端采样计划运行配置选择的 `clean`、`attack`、`defended` 分支。
6. 每轮中，`FedAvgCoordinator` 调用客户端本地训练；攻击分支先为恶意客户端替换或追加生成特征，再产生 `ClientUpdate`。
7. `defended` 分支把更新交给 `DefensePipeline`，依次完成合法性校验、异常检测、接受/裁剪/拒绝决策和服务器聚合。
8. runner 在 test 上计算效用、0→1 定向攻击和防御检测指标，并写入结果目录。

更细的文件级调用关系见 [当前流程与结果结构](docs/CURRENT_PIPELINE_STRUCTURE.md)。

## 配置

主配置位于 `configs/experiments/`：

- `ucf101_fdmm_dtm_poison_0to1.yaml`：clean 与 attack 分支；
- `ucf101_fdmm_dtm_poison_0to1_defense.yaml`：clean、attack 与 defended 分支；
- `ucf101_fdmm_dtm_poison_0to1_smoke.yaml`：短流程连通性测试；
- `ucf101_dtm_poison_strength/*.yaml`：使用人类可读文件名配置恶意客户端数、中毒比例和生成器 epoch。

完整配置包含八部分：

- `dataset`：FedMM 特征根、fold、alpha、类别数和模态形状；
- `model`：分类模型构造参数和可选初始 checkpoint；
- `federation`：预训练/攻击轮数、客户端采样、本地训练、seed 和分支；
- `generator`：DTM 变体、生命周期和训练参数；
- `attack`：恶意客户端、中毒预算、注入方式和 0→1 标签语义；
- `defense`：检测器、裁剪器、聚合器和决策策略；
- `evaluation`：test、攻击与检测指标开关；
- `artifact`：运行产物根目录。

命令行 `--seed` 会同时覆盖联邦训练与生成器 seed；`--run-dir` 可显式指定本次运行目录：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --seed 42 \
  --run-dir artifact/ucf101_fdmm_dtm_poison_0to1/20260729-120000_seed-42_git-8d6f0057
```

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

批处理脚本还会在每个运行目录中保存 `train.log`。

参数配置示例：

```text
artifact/ucf101_dtm_poison_strength/
└── malicious-clients-2_poison-50pct_generator-epochs-20/
    └── 20260729-120000_seed-42_git-8d6f0057/
```

`config_resolved.yaml` 保存全部实际参数，`run_manifest.json` 记录提交和运行环境，`summary.json` 汇总 M*、分支 test/ASR、攻击暴露和防御检测结果。逐轮客户端更新与服务器决策统一位于 `round_records.pt`。

## 多 GPU 批处理

`scripts/run_experiments.sh` 接收多个 `GPU:CONFIG:SEED` 作业，将每个配置绑定到指定 GPU，并持续更新批次状态：

```bash
PYTHON_BIN=/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python \
bash scripts/run_experiments.sh \
  0:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-1_poison-20pct_generator-epochs-20.yaml:42 \
  1:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-2_poison-50pct_generator-epochs-20.yaml:42
```

每个作业的日志写入自己的 `train.log`；批次监控表写入：

```text
artifact/batches/<YYYYMMDD-HHMMSS>/status.tsv
```

## 目录边界

- `mflpoison/`：当前联邦攻防框架；
- `configs/experiments/`：完整实验和参数变体；
- `scripts/`：批处理和运行辅助脚本；
- `fed_multimodal/`：UCF101 数据、模型、FedAvg 与生成器兼容实现；
- `experiments/`：旧 checkpoint 的人工分析工具；
- `tests/`：自动化测试；
- `artifact/`：运行时按需创建的产物，不进入 Git；清理最后一次临时运行后不保留空目录。

本项目仅用于防御性安全研究和多模态联邦学习鲁棒性评估。
