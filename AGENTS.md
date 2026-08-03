# mflpoison 本地开发—BJMU 实验规范

## 当前有效信息

- 本地仓库：`C:\Users\86184\Desktop\mflpoison`
- GitHub 仓库：`https://github.com/softmountain/mflpoison.git`
- 本地与 BJMU Git origin：`https://github.com/softmountain/mflpoison.git`；BJMU 不使用 SSH origin，也不向 GitHub 推送。
- 唯一长期生产分支：`main`
- BJMU SSH 别名：`bjmu4090`
- BJMU 仓库：`/mnt/sda/mtzh/xp/fedpoi`
- BJMU Python：`/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python`
- UCF101 输入特征：`fed_multimodal/results`
- 唯一生产入口：`python -m mflpoison.runner`
- 实验配置：`configs/experiments/`
- 服务端批量调度：`scripts/run_experiments.sh`
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

## 本地改动、审核与发布

### 1. 本地开始

所有源码、配置、文档和 `AGENTS.md` 修改只在本地工作区完成。开始前执行：

```bash
git rev-parse HEAD
git status --short
```

保留并绕开与当前任务无关的 staged、unstaged 和 untracked 内容，不重置、不清理、不覆盖用户已有工作。

### 2. 本地实施与测试

- 按明确范围修改现有代码和配置。
- 普通修改先在本地运行当前环境能够执行的相关单测，不额外增加普通流程不需要的验证程序。
- 依赖 BJMU 数据或 CUDA 环境的测试不通过复制未提交 diff 执行；先完成本地审核、提交和推送，再由 BJMU 在同一批准提交上验证。
- 影响生产调用链的修改在 BJMU 同步后运行一次 smoke；smoke 只验证程序可运行，不代替正式实验或科学结论。
- smoke 输出放在 `artifact/smoke/`，验收日志后立即删除该次产物、临时日志、测试缓存和空父目录。

### 3. 本地暂存审核

只暂存本次明确文件：

```bash
git add <明确文件列表>
git diff --cached --stat
git diff --cached
```

暂存区就是待审版本，不复制候选目录：

- 通过：直接提交并推送；
- 不通过：`git restore --staged -- <文件>`，再撤销仅属于本次候选的本地工作区内容并继续修改；
- `git restore --staged` 只撤销暂存，不会删除工作区改动；
- 不使用 `git add .`，不夹带数据、图片、checkpoint、运行产物或用户无关文件。

普通文本不计算哈希。只有二进制传输异常、patch 异常或两端内容确实不一致时才比较 SHA-256。

### 4. GitHub 发布与 BJMU 同步

- 本地提交是唯一批准版本；审核通过后由本地推送到 GitHub `main`。
- `main` 是唯一长期生产分支。功能分支只在确有独立开发需要时创建，合入后删除；需要保留的历史快照使用归档标签，不长期保留已合入或混合实验产物的分支。
- 原 `temporal-adaptive-gan-evaluation` 分支保存在归档标签 `archive/temporal-adaptive-gan-evaluation-202607`；需要恢复历史时从标签读取，不重新建立长期分支。
- 不 force-push，不改写 `main` 历史，不把原始结果、图片、checkpoint、压缩包或正式运行产物提交到 Git。
- BJMU 不直接编辑 tracked 文件，不保存候选 diff，不创建候选提交，也不向 GitHub 推送。测试或实验前只执行：

```bash
git fetch --prune --tags origin
git merge --ff-only origin/main
```

- BJMU 若存在来源不明的 tracked 改动，先停止同步并查明归属，不擅自覆盖或清理。
- 同步完成后，本地与 BJMU 各执行一次：

```bash
git rev-parse HEAD
git status --short
```

不再增加双端候选同步或中间确认轮次。正式实验只使用已同步的批准提交。

## 实验执行

单次运行：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml \
  --seed 42
```

今后的批量实验只通过 BJMU 上的统一调度脚本执行，不在终端逐个启动，也不为单次批次或参数组合新增脚本。下一次批量启动前，在现有 `scripts/run_experiments.sh` 内实现并使用以下队列调度能力；接口落地时同步更新本节的可执行命令：

- 接收由配置路径和 seed 组成的实验队列，超参数仍只写入 YAML。
- 持续监控指定 GPU 池和任务状态；GPU 空闲时自动领取下一项实验，一张卡同时只运行一项任务。
- 为每项任务保存独立 `train.log`、PID、退出状态和运行目录，并在 `artifact/batches/<batch-id>/status.tsv` 中记录 `queued/running/completed/failed`。
- 队列未清空时继续调度；任务结束后及时释放 GPU，失败任务明确标记并停止依赖它的后续阶段。
- 队列结束后清理调度产生的临时 launcher 日志，不删除正式 artifact。

运行中不修改所用代码或配置，也不清理正式产物。

## 实验可比性与指标

- 每个 seed 只生成一份共同 M*；在相同 seed、数据分区、客户端日程和 M* 下运行五次 clean。BJMU 只有四张 GPU 时，调度器先启动四项，第五项在首张 GPU 空闲后自动启动。
- 五次 clean 的 ASR 算术平均值定义为该 seed 的 `ASR_canonical_clean`。canonical clean 是五次运行组成的指标基线，不表示单一 checkpoint。

```text
ASR_canonical_clean(seed) = mean(ASR_clean_1, ..., ASR_clean_5)
```

- 同一 seed 的所有攻击配置复用同一份 `ASR_canonical_clean`，不再为每个参数配置重复训练 clean；超参数筛选只运行 attack 分支。
- seeds 43 和 44 分别生成自己的共同 M* 和五次 clean 基线，不跨 seed 复用。
- 每个攻击配置按下式计算组内增量：

```text
Delta_ASR(seed, config) =
    ASR_attack(seed, config) - ASR_canonical_clean(seed)
```

- 每次 attack 都报告 `成功数/源类总数`；五次 clean 分别报告各自的 `成功数/源类总数`，再报告平均 ASR。当前 fold1 的源类总数为 44；由于五次使用同一批测试样本，不把它们合并为 `成功数/220`。

## AGENTS.md 维护

- 本文件只保留当前仍有效的环境、路径、入口、目录结构和工作规则；失效内容直接更新或删除。
- 每批本地候选修改都检查是否改变上述信息，需要时把 `AGENTS.md` 纳入同一批暂存和审核。
- `AGENTS.md` 只在本地编辑；本地批准提交并推送后，BJMU 通过 fast-forward 获得并重新读取同一版本。
- 常规安全检查直接在操作中完成，不为它们新建代码文件。不可恢复删除、正式实验目录覆盖和断点恢复仍需确认目标。

### “浓缩”口令

- 本项目的规则文件名统一为 `AGENTS.md`；不另建 `AGENT.md`。
- 当用户发送的消息去除首尾空白后恰好为“浓缩”时，回顾当前对话，提取对后续任务可重复使用且已经验证的信息，例如有效环境、路径、项目结构、稳定命令、工作规则、已确认决策和反复出现的注意事项。
- 不收录密码、令牌等敏感信息，不收录单次运行进度、临时 PID、短期 artifact 路径、未经验证的推测或已经失效的内容。
- 先向用户提交拟新增、更新或删除的条目清单，并说明建议放入 `AGENTS.md` 的位置；此时不得修改、暂存或同步文件。
- 只有收到用户明确审核意见和批准后，才按批准清单更新 `AGENTS.md`；若用户提出修订，先更新候选清单并再次审核。
- 批准后只在本地更新并暂存本次明确修改的 `AGENTS.md`，展示暂存 diff 供最终提交审核，并同时更新或删除与新信息冲突的旧条目；提交推送后由 BJMU 在下一次 fast-forward 时获取，不直接修改服务器副本。
