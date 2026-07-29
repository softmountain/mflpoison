# mflpoison 本地—BJMU 协作规范

## 当前有效信息

- 本地仓库：`C:\Users\86184\Desktop\mflpoison`
- BJMU SSH 别名：`bjmu4090`
- BJMU 仓库：`/mnt/sda/mtzh/xp/fedpoi`
- BJMU Python：`/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python`
- UCF101 输入特征：`fed_multimodal/results`
- 唯一生产入口：`python -m mflpoison.runner`
- 实验配置：`configs/experiments/`
- 多 GPU 批处理：`scripts/run_experiments.sh`
- 运行产物：`artifact/<experiment>/<variant>/<YYYYMMDD-HHMMSS>_seed-<N>_git-<sha>/`

`fed_multimodal/results/` 是历史兼容的输入特征目录，不属于运行产物，不随 `artifact/` 重命名。

## 命名与文件准入

- Python 模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，名称表达职责或方法。
- 不使用 `new`、`final`、`v2`、`try2`、日期或单次超参数命名源码文件。
- 基准配置名表达数据集、模型、攻击和防御，例如 `ucf101_fdmm_dtm_poison_0to1_defense.yaml`。
- 参数配置名使用完整、固定的字段名，例如 `malicious-clients-2_poison-50pct_generator-epochs-20.yaml`；不使用含义不明的缩写。
- 超参数、seed 和客户端组合只写 YAML，不复制 Python 训练文件，也不为单次实验增加脚本。
- 只有明确新增算法、方法或独立可复用职责时才新增 Python 文件；其他需求优先修改已有模块。
- 不新增普通流程不需要的 preflight、hash、sweep、安全包装或重复确认程序。

## 运行产物

单次运行目录遵循：

```text
artifact/
└── <experiment-or-config-group>/
    └── <config-name>/
        └── <YYYYMMDD-HHMMSS>_seed-<N>_git-<short-sha>/
```

配置直接位于 `configs/experiments/` 时省略中间的 config-group 层。每次运行保存固定职责文件：

```text
config_resolved.yaml
run_manifest.json
summary.json
round_records.pt
checkpoints/
generators/
```

目录名只放实验语义、关键差异参数和运行标识；全部实际参数保存在 `config_resolved.yaml`，提交、运行环境和完成状态保存在 `run_manifest.json`。批处理任务另存 `train.log`。目录内部不再给每个文件重复拼接日期和超参数。

仓库不预建空的 `artifact/`、缓存或临时目录。命令按需创建，清理最后一个临时运行后同时删除空父目录。

## 改动、测试与本地审核

### 1. 开始

本地和 BJMU 各执行一次：

```bash
git rev-parse HEAD
git status --short
```

两端从同一批准提交开始，BJMU 没有未批准的 tracked 改动或占用目标 GPU 的旧实验即可实施。

### 2. BJMU 实施与测试

- 按明确范围修改现有代码和配置。
- 普通修改运行相关单测和一次 smoke；通过后默认软件调用链可靠，不再增加额外验证程序。
- smoke 只验证程序可运行，不代替正式实验或科学结论。
- smoke 输出放在 `artifact/smoke/`，验收日志后立即删除该次产物、临时日志、测试缓存和空父目录。

### 3. 本地暂存审核

将 BJMU 候选 diff 应用到本地后，只暂存明确文件：

```bash
git add <明确文件列表>
git diff --cached --stat
git diff --cached
```

暂存区就是待审版本，不复制候选目录：

- 通过：直接提交并推送；
- 不通过：`git restore --staged -- <文件>`，再撤销仅属于本次候选的本地工作区内容，BJMU 按意见继续修改；
- `git restore --staged` 只撤销暂存，不会删除工作区改动；
- 不使用 `git add .`，不夹带数据、图片、checkpoint、运行产物或用户无关文件。

普通文本不计算哈希。只有二进制传输异常、patch 异常或两端内容确实不一致时才比较 SHA-256。

### 4. 提交与同步

本地提交是批准版本。推送后 BJMU 丢弃仅属于候选 diff 的未提交副本，并以 fast-forward 同步批准提交。完成后两端各执行一次：

```bash
git rev-parse HEAD
git status --short
```

不再增加中间确认轮次。正式实验只使用已同步的批准提交。

## 实验执行

单次运行：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml \
  --seed 42
```

批量实验只使用统一脚本，不为参数组合增加脚本：

```bash
PYTHON_BIN=/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python \
bash scripts/run_experiments.sh \
  0:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-1_poison-20pct_generator-epochs-20.yaml:42 \
  1:configs/experiments/ucf101_dtm_poison_strength/malicious-clients-2_poison-50pct_generator-epochs-20.yaml:42
```

脚本负责 GPU 绑定、并行启动、独立日志、PID/退出状态监控和 `artifact/batches/<batch-id>/status.tsv`。运行中不修改所用代码或配置，也不清理正式产物。

## AGENTS.md 维护

- 本文件只保留当前仍有效的环境、路径、入口、目录结构和工作规则；失效内容直接更新或删除。
- 每批候选修改都检查是否改变上述信息，需要时把 `AGENTS.md` 纳入同一批暂存和审核。
- 两端不维护分叉版本。本地批准提交后，BJMU 同步并重新读取同一份 `AGENTS.md`。
- 常规安全检查直接在操作中完成，不为它们新建代码文件。不可恢复删除、正式实验目录覆盖和断点恢复仍需确认目标。
