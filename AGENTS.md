# mflpoison 本地—BJMU 协作规范

## 当前有效信息

- 本地仓库：`C:\Users\86184\Desktop\mflpoison`
- BJMU SSH 别名：`bjmu4090`
- BJMU 仓库：`/mnt/sda/mtzh/xp/fedpoi`
- BJMU Python：`/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python`
- UCF101 特征根：`fed_multimodal/results`
- 唯一生产入口：`python -m mflpoison.runner`
- 实验配置：`configs/experiments/`
- 多 GPU 批处理：`scripts/run_experiments.sh`
- 结果目录：`results/YYYY-MM-DD/<config-name>/HH-MM-SS_seed-N/`

本地负责需求、暂存审查、验收和提交；BJMU 负责候选代码实施、测试和正式实验。环境、路径、入口或流程在批准后发生变化时，必须在同一批改动中更新两端 `AGENTS.md`；失效信息直接删除，不保留历史说明。

仓库不预建空目录。`results/`、缓存和临时目录都由实际命令按需创建，任务完成并清理内容后应一并删除空父目录。旧 `artifacts/` 工作流已经移除，不得重新使用该目录名。

## 日常改动流程

### 1. 确认基线

本地和 BJMU 分别运行：

```bash
git rev-parse HEAD
git status --short
```

开始正式改动前，两端 HEAD 应一致，BJMU 不应有未批准的 tracked 改动，也不应有旧实验占用目标 GPU。

### 2. BJMU 实施并做 smoke test

- 按本地给出的明确范围修改，不顺手增加兼容入口、安全包装或新的 Python 变体。
- 算法和可复用能力写入 `mflpoison/`；超参数组合只写 YAML。
- 先运行相关单测，再用 `ucf101_fdmm_dtm_poison_0to1_smoke.yaml` 验证完整调用链。
- smoke 输出只放在 `results/smoke/` 或明确的临时目录。确认日志后删除该次 smoke 结果、临时日志、缓存和清空后的父目录；正式实验目录不得当作测试目录。

### 3. 本地暂存审查

取得 BJMU 的候选 diff 后，本地只暂存候选文件：

```bash
git add <明确文件列表>
git diff --cached --stat
git diff --cached
```

暂存区就是待审快照，不再复制一套候选目录。审查代码、配置、测试、指标语义和结果命名：

- 不通过：`git restore --staged <文件>`，说明问题并由 BJMU 修正；
- 通过：保留暂存内容并提交；
- 不得使用 `git add .` 夹带图片、数据、checkpoint、结果或用户无关文件。

普通文本改动不计算逐文件哈希。只有二进制传输、patch 异常或两端内容确有疑问时才比较 SHA-256。

### 4. 提交和同步

本地提交批准的暂存内容后，将该提交同步到 BJMU。随后两端再次核对：

```bash
git rev-parse HEAD
git status --short
```

正式实验必须基于两端相同的批准提交，且 BJMU tracked 工作区干净。不得 reset、force-push、整目录覆盖或删除无关文件。

## 实验约定

### 配置而不是代码变体

- 基准实验使用完整语义配置名，例如 `ucf101_fdmm_dtm_poison_0to1.yaml`。
- 参数变体使用 `base_config + overrides`，例如 `poison_strength/clients2_poison50_gen20.yaml`。
- 禁止为恶意客户端数、中毒比例、generator epoch 或 seed 复制 Python 训练文件。
- 文件名使用小写 `snake_case`，不使用 `new`、`final`、`try2`、`artifact` 等含义不清的名称。

### 单次运行

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml \
  --seed 42
```

入口保存 `config_resolved.yaml`、`run_info.json`、`summary.json`、`checkpoints/`、`generators/` 和逐轮记录。分析结果时报告实际提交、解析后配置、seed、完成状态和关键样本计数；计划或旧结果不能写成新实验结论。

### 多 GPU 批处理

```bash
PYTHON_BIN=/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python \
bash scripts/run_experiments.sh \
  0:configs/experiments/poison_strength/clients1_poison20_gen20.yaml:42 \
  1:configs/experiments/poison_strength/clients2_poison50_gen20.yaml:42
```

每个作业绑定一个 `CUDA_VISIBLE_DEVICES`，日志写入各自 `train.log`；批次状态写入 `results/batches/YYYY-MM-DD/HH-MM-SS/status.tsv`。运行中不得切换提交、修改所用配置或清理正式结果。

## 只在异常时使用的规则

- 两端 diff 不一致：先比较 `git status` 和完整 diff；patch 传输先执行 `git apply --check`。
- 网络无法同步：确认批准提交已推送；必要时使用 HTTPS 或 Git bundle，但同步后仍以 HEAD 和 tracked 状态为准。
- 旧结果语义存疑：以当前代码、`config_resolved.yaml` 和原始逐轮记录为准。当前攻击方向为 `condition=0 / train_label=1 / victim=0 / goal=1`。
- 结果来源不明：缺少提交、配置、seed 或完成状态的目录不用于正式对比，不补写成已验证实验。
