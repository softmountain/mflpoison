# MFL-Poison：UCF101 多模态联邦中毒与防御

本仓库研究 UCF101 音频/视频特征上的联邦生成式数据中毒和服务器侧异常更新防御。生产流程只有一个入口：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
```

完整代码、训练阶段、输入数据和结果文件映射见 [当前流程与结果结构说明](docs/CURRENT_PIPELINE_STRUCTURE.md)。架构约束和兼容边界见 [重构说明](REFACTORING.md)。

## 当前流程

```text
UCF101 FedMM 客户端特征
  -> clean FedAvg 预训练
  -> 仅由 dev 指标选择 M*
  -> 每个恶意客户端在自己的 partition 上训练生成器
  -> clean / attack / defended 使用相同客户端采样计划
  -> 服务器 validate / detect / decide / sanitize / aggregate
  -> test、攻击成功率和检测指标汇总
```

默认配置使用：

- 数据根：`fed_multimodal/results`
- fold：`1`
- Dirichlet alpha：`1.0`，对应磁盘目录 `alpha10/fold1`
- 模型：`MMActionClassifier`
- 联邦算法：FedAvg
- 生成器：DTM，默认 `offline_once`
- 中毒：生成式特征替换，比例 `0.2`
- 防御：norm MAD + cosine MAD + norm clipping + weighted mean
- 输出：`artifacts/ucf101_generative_poison_defense/`

## 安装

推荐使用已有的 `poigan` Conda 环境；重新安装时：

```bash
pip install -r requirements.txt
pip install -e .
```

验证：

```bash
conda run -n poigan python -c "import torch, mflpoison; print(torch.__version__)"
conda run -n poigan pytest -q
```

## 输入数据

当前 runner 读取已经按 FedMM 结构生成的 UCF101 特征，不重新集中划分数据：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/alpha10/fold1/
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/alpha10/fold1/
```

每个目录包含训练客户端 `0.pkl` 至 `9.pkl`，以及 `dev.pkl`、`test.pkl`。`alpha50/fold1` 也被保留，可通过修改场景中的 `dataset.alpha` 使用。

UCF101 特征提取、划分和缺失模态模拟工具位于 `fed_multimodal/features/`。数据和模型 checkpoint 均受 `.gitignore` 保护，不上传 Git。

## 运行和恢复

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml

python experiments/run_scenario.py \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml \
  --artifact-root artifacts/custom-run
```

恢复训练时在场景配置中设置 `federation.resume_from`。恢复状态包含预训练/分支进度、采样计划、生成器 lifecycle 和可选 EWMA reputation。

## 结果

统一流程运行后主要产生：

- `manifest.json`：完整配置、随机种子、Git commit、数据划分和采样计划；
- `snapshots/`：初始模型、M* 和 clean/attack/defended 最终模型；
- `generator_checkpoints/` 与 `generators/`：每客户端生成器 checkpoint 和 lineage manifest；
- `round_records/` 与 `round_records.pt`：逐轮客户端更新、防御决策和聚合审计；
- `resume_state.pt`：可复现恢复状态；
- `summary.json`：dev/test、攻击、检测和 clean utility drop 汇总。

仓库不会自动生成图表或主观分析报告；分析应以 `summary.json` 和 `round_records` 为依据。当前是否已经产生统一结果，以结构报告中的“当前实际状态”为准。

## 旧 checkpoint 兼容入口

旧 DTM、temporal-adaptive、K+1 和 teacher-guided checkpoint 仍可评估：

```bash
python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt \
  --dataset_dir /path/to/ucf101 --num_batches 20

python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --output artifacts/manual/synthetic.pt --num_samples 1000

python experiments/evaluate_tstr.py \
  --synthetic_data artifacts/manual/synthetic.pt -- \
  --dataset_dir /path/to/ucf101 --num_epochs 100
```

`experiments/train_generator.py`、`fed_multimodal/Local/train_dtm_poison_gan.py` 和 `train_temporal_adaptive_gan.py` 只是统一 runner 的兼容别名，不再提供集中式 `full_train` 训练。

## 项目边界

- `mflpoison/`：当前联邦攻防框架；
- `fed_multimodal/`：UCF101 数据、模型、FedAvg 和生成器的最小兼容实现；
- `experiments/`：统一入口与旧 checkpoint 工具；
- `tests/`：契约、攻击、防御、联邦引擎、生成器 lifecycle 和 runner 测试；
- `artifacts/`：统一实验结果，始终不进入 Git。

本项目仅用于防御性安全研究和多模态联邦学习鲁棒性评估。
