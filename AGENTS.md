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
- 单次运行产物：`artifact/<experiment>/<variant>/<YYYYMMDD-HHMMSS>_seed-<N>_git-<sha>/`
- 批次状态与 canonical 基线：`artifact/batches/<batch-id>/`

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

手工单次运行目录遵循：

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

目录名只放实验语义、关键差异参数和运行标识；全部实际参数保存在 `config_resolved.yaml`，提交、dirty 状态、运行源码树 hash、运行环境和完成状态保存在 `run_manifest.json`。批处理任务的运行标识还包含 `job_id`，保证同一 seed 的五次并发 clean 不会发生目录冲突；批调度检测到未提交候选时，其 run ID 的 Git 标识附加 `-dirty`，每项任务另存 `train.log`。手工单次入口的默认目录名仍只含 Git 短 SHA，源码身份以 manifest 为准。

动态调度批次另存：

```text
artifact/batches/<batch-id>/
├── status.tsv
├── canonical_clean_seed-<N>.json
└── canonical_clean_seed-<N>.log
```

`canonical_clean_seed-<N>.json` 保存五次 clean 各自的成功数、源类总数和 ASR，以及算术平均值、共同 M*、数据分区、完整客户端日程、攻击方向和来源运行目录。目录内部不再给每个文件重复拼接日期和超参数。

仓库不预建空的 `artifact/`、缓存或临时目录。命令按需创建，清理最后一个临时运行后同时删除空父目录。

## 本地终端、BJMU 候选与 Git 发布

### 1. 开始与职责边界

日常开发和服务器验证统一由本地终端发起。本地工作区是候选源码的权威副本；BJMU 保存与该候选对应的测试副本，但不独立产生另一套修改。开始前在本地和 BJMU 分别执行：

```bash
git rev-parse HEAD
git status --short
```

保留并绕开两端与当前任务无关的 staged、unstaged 和 untracked 内容，不重置、不清理、不覆盖用户已有工作。BJMU 若已有来源不明的 tracked 改动，先停止同步并查明归属。

### 2. 候选修改与测试

- 按明确范围修改现有代码和配置。
- 小批修改保持为未暂存、未提交候选；从本地终端只把本次明确文件同步到 BJMU，不用 GitHub 中转日常候选。
- BJMU 不独立编辑 tracked 文件，不创建候选提交，也不向 GitHub 推送；需要修订时先改本地权威副本，再重新同步明确文件。
- 普通修改先在本地运行当前环境能够执行的相关单测；依赖 BJMU 数据、Linux、CUDA 或服务器 Python 的测试，在两端候选内容一致后由本地终端登录 BJMU 执行。
- 影响生产调用链的修改在 BJMU 候选上运行一次 smoke；未提交候选仅用于单测和 smoke，不用于正式实验或科学结论。
- smoke 输出放在 `artifact/smoke/`，验收日志后立即删除该次产物、临时日志、测试缓存和空父目录。
- 同步或测试前后检查两端 `git status --short`；若同一候选文件内容不一致，先停止并查明原因，不用覆盖来源不明的内容来强行对齐。

### 3. 大版本里程碑审核

日常小改不要求逐批加入 Git。只有形成大版本更新或用户明确要求发布时，才在本地统一审核累计候选，并只暂存该里程碑的明确文件：

```bash
git add <明确文件列表>
git diff --cached --stat
git diff --cached
```

暂存区就是里程碑待审版本：

- 通过：直接提交并推送；
- 不通过：`git restore --staged -- <文件>`，再撤销仅属于本次候选的本地工作区内容并继续修改；
- `git restore --staged` 只撤销暂存，不会删除工作区改动；
- 不使用 `git add .`，不夹带数据、图片、checkpoint、运行产物或用户无关文件。

本地与 BJMU 做单文件传输核对时，普通文本不另算文件哈希；只有二进制传输异常、patch 异常或两端内容确实不一致时才比较文件 SHA-256。此规则不影响 runner 在 `run_manifest.json` 中自动记录用于实验来源比对的 `source_tree_hash`。

### 4. GitHub 发布与正式实验

- 大版本里程碑的本地提交是正式批准版本；审核通过后由本地推送到 GitHub `main`。GitHub 用于大版本归档和发布，不作为日常小改在本地与 BJMU 之间的中转站。
- `main` 是唯一长期生产分支。功能分支只在确有独立开发需要时创建，合入后删除；需要保留的历史快照使用归档标签，不长期保留已合入或混合实验产物的分支。
- 原 `temporal-adaptive-gan-evaluation` 分支保存在归档标签 `archive/temporal-adaptive-gan-evaluation-202607`；需要恢复历史时从标签读取，不重新建立长期分支。
- 不 force-push，不改写 `main` 历史，不把原始结果、图片、checkpoint、压缩包或正式运行产物提交到 Git。
- 里程碑提交推送后，先 fetch 并逐项确认 BJMU 明确候选文件与 `origin/main` 中的批准版本一致。只有内容一致且没有来源不明改动时，才从本地终端撤下这些可由 `origin/main` 恢复的明确候选工作副本，使 BJMU checkout 恢复 clean，再执行 fast-forward；BJMU 不推送：

```bash
git fetch --prune --tags origin
git merge --ff-only origin/main
```

- 收口前如 BJMU 仍有与里程碑提交不同的 tracked 或 untracked 候选，先比较归属并保留必要内容，不擅自覆盖或清理；不能确认完全一致时停止收口。同步完成后，本地与 BJMU 各执行一次：

```bash
git rev-parse HEAD
git status --short
```

正式实验只使用已经里程碑提交、推送且两端对齐的批准版本；`run_manifest.json` 中的 Git SHA 必须能够唯一代表所运行源码。未提交候选不得启动正式批次。

## 实验执行

单次运行：

```bash
python -m mflpoison.runner \
  --config configs/experiments/ucf101_fdmm_dtm_poison_0to1_defense.yaml \
  --seed 42
```

今后的批量实验只通过本地终端调用 BJMU 上的统一调度脚本执行，不在终端逐个启动，也不为单次批次或参数组合新增脚本。每个任务输入为 `CONFIG:SEED`，GPU 池单独指定：

```bash
PYTHON_BIN=/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python \
ARTIFACT_ROOT=/mnt/sda/mtzh/xp/experiments/fedpoi/artifact \
bash scripts/run_experiments.sh \
  --gpus 0,1,2,3 \
  --canonical-clean-config configs/experiments/ucf101_fdmm_dtm_poison_0to1.yaml \
  --experiment-branch attack \
  configs/experiments/<attack-config>.yaml:42 \
  configs/experiments/<attack-config>.yaml:43
```

- 超参数仍只写入 YAML；任务输入不再携带显式 GPU。
- 持续监控指定 GPU 池和任务状态；GPU 空闲时自动领取下一项实验，一张卡同时只运行一项任务。
- 每个 seed 固定建立 `mstar -> clean-1..clean-5 -> canonical-aggregate -> attack-only|defended-only` 依赖链；`--experiment-branch` 对一个批次统一选择 `attack` 或 `defended`。四卡环境先并发四次 clean，第五次在首张空闲卡上自动补位。
- 为每项任务保存独立 `train.log`、PID、退出状态和运行目录；`artifact/batches/<batch-id>/status.tsv` 固定记录 `job_id/stage/experiment/seed/repeat/depends_on/gpu/pid/status/exit_code/queued_at/started_at/finished_at/config/run_dir/failure_reason`。
- 状态只使用 `queued/running/completed/failed`。队列未清空时持续补位；任务结束后及时释放 GPU。上游失败时，下游不启动并记为 `failed`、退出码 `125`、`failure_reason=dependency_failed:<job_id>`。
- 调度器同时检查自身占卡、`nvidia-smi` compute 进程和显存占用，并使用主机级全局 `flock` 保证同一主机只运行一个调度器实例。`SCHEDULER_LOCK_FILE` 只允许在隔离测试或已确认不会争用 GPU 的受控环境中覆盖，不用于普通 BJMU 批次。
- 配置必须位于当前仓库的 `configs/experiments/`，seed 只允许 `0..4294967295`。runner 退出码为 0 后还必须确认 `run_manifest.json` 顶层 `status=completed`；收到 `HUP/INT/TERM` 时停止派发、终止并回收运行中进程组，把所有未完成任务标记为 `failed` 后再释放全局锁。
- 调度器不额外创建临时 launcher 日志；保留每项 `train.log`、canonical 聚合日志和正式 artifact。

运行中不修改所用代码或配置，也不清理正式产物。

## 实验可比性与指标

- 每个 seed 只生成一份共同 M*；在相同 seed、数据分区、客户端日程和 M* 下运行五次 clean。BJMU 只有四张 GPU 时，调度器先启动四项，第五项在首张 GPU 空闲后自动启动。
- 五次 clean 以 `repeat=1..5` 和唯一 `job_id` 分别记录；聚合阶段只有在五次均完成后才生成 `artifact/batches/<batch-id>/canonical_clean_seed-<N>.json`。
- 五次 clean 的 ASR 算术平均值定义为该 seed 的 `ASR_canonical_clean`。canonical clean 是五次运行组成的指标基线，不表示单一 checkpoint。

```text
ASR_canonical_clean(seed) = mean(ASR_clean_1, ..., ASR_clean_5)
```

- 同一 seed 的所有攻击或防御配置复用同一份共同 M* 和 `ASR_canonical_clean`，不再为每个参数配置重复训练 clean；`ucf101_dtm_poison_strength/*.yaml` 明确设置 `branches: [attack]`，`ucf101_dtm_poison_strength_defense/*.yaml` 设置 `branches: [defended]` 且启用服务器防御，调度器再注入共同 M* 与 canonical JSON 路径并写入 `config_resolved.yaml`。attack-only/defended-only 不允许省略 canonical baseline。
- seeds 43 和 44 分别生成自己的共同 M* 和五次 clean 基线，不跨 seed 复用。
- 每个攻击配置按下式计算组内增量：

```text
Delta_ASR(seed, config) =
    ASR_attack(seed, config) - ASR_canonical_clean(seed)
```

- 每次 attack 都报告 `成功数/源类总数`；五次 clean 分别报告各自的 `成功数/源类总数`，再报告平均 ASR。当前 fold1 的源类总数为 44；由于五次使用同一批测试样本，不把它们合并为 `成功数/220`。聚合和 attack 启动前必须校验 seed、共同 M* 的 completed m-star-only manifest、M* hash、数据分区、唯一 experiment ID、完整客户端日程、攻击方向、源类总数、共同训练协议和源码身份一致，并从五份明细重新核算 canonical 均值与标准差。

## AGENTS.md 维护

- 本文件只保留当前仍有效的环境、路径、入口、目录结构和工作规则；失效内容直接更新或删除。
- 每批本地候选修改都检查是否改变上述信息，需要时把 `AGENTS.md` 纳入同一候选并从本地终端同步到 BJMU。
- `AGENTS.md` 仍以本地副本为权威；日常更新作为未提交候选同步到 BJMU，形成大版本里程碑时再与源码一起审核、提交和推送。
- 常规安全检查直接在操作中完成，不为它们新建代码文件。不可恢复删除、正式实验目录覆盖和断点恢复仍需确认目标。

### “浓缩”口令

- 本项目的规则文件名统一为 `AGENTS.md`；不另建 `AGENT.md`。
- 当用户发送的消息去除首尾空白后恰好为“浓缩”时，回顾当前对话，提取对后续任务可重复使用且已经验证的信息，例如有效环境、路径、项目结构、稳定命令、工作规则、已确认决策和反复出现的注意事项。
- 不收录密码、令牌等敏感信息，不收录单次运行进度、临时 PID、短期 artifact 路径、未经验证的推测或已经失效的内容。
- 先向用户提交拟新增、更新或删除的条目清单，并说明建议放入 `AGENTS.md` 的位置；此时不得修改、暂存或同步文件。
- 只有收到用户明确审核意见和批准后，才按批准清单更新 `AGENTS.md`；若用户提出修订，先更新候选清单并再次审核。
- 批准后只在本地更新本次明确修改的 `AGENTS.md`，并同时更新或删除与新信息冲突的旧条目；作为普通未提交候选由本地终端同步到 BJMU。只有形成大版本里程碑时才暂存、展示 diff、提交并推送。
