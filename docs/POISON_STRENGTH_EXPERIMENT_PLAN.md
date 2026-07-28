# UCF101 0→1 攻击强度实验计划

本计划对应 `configs/sweeps/ucf101_poison_strength.yaml`。正式 GPU 实验必须在本地
审查候选 diff、提交批准版本并同步回 BJMU 后执行；当前服务器候选实现阶段只运行
单元测试、配置解析和轻量 smoke check。

## 固定语义和比较门禁

- 攻击方向固定为 `condition=0 / train_label=1 / victim=0 / goal=1`。
- 只运行 `clean` 和 `attack`；`defense.enabled=false`。
- `attack_rounds=20`、`clients_per_round=5`、`replace` 固定。
- 恶意客户端使用嵌套集合 M1=`["1"]`、M2=`["0","1"]`、
  M3=`["0","1","4"]`。
- 每个 run 使用独立的 `stage/experiment/seed-N` 目录；已完成目录跳过，配置哈希
  一致且 resume payload 完整的中断目录仅在 `--resume` 时继续，未知或冲突目录拒绝。
- 每个 seed 只由 `single_factor/B0` 训练一次 M*；其余单因素和全部组合通过
  `m_star_path + m_star_snapshot_hash` 只读复用，并校验模型规格与 partition。
- 同一种子各组完成后继续核对 M* hash、partition hash、branch schedule 和相对
  B0 的 clean 最终 snapshot；相同客户端且 generator epochs 相同时还核对 checkpoint
  SHA-256，任何漂移立即失败并写入/保留 sweep provenance。
- 正式执行要求 `--approved-commit`，当前 HEAD 必须等于该提交，tracked worktree
  和 staging 必须干净。

## 执行顺序

先只解析计划：

```bash
/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python experiments/run_sweep.py \
  --plan configs/sweeps/ucf101_poison_strength.yaml \
  --stage single_factor --seed 42
```

批准版本上先执行 B0 的 seed 42，生成共享 M*；再执行 M2、M3、P50、P100、
E20、E50。根据结果筛出
有提升的配置，再分别执行 `--experiment NAME --seed 43` 和
`--experiment NAME --seed 44` 补种子（每条正式执行命令只指定一个 seed）。

补 seed 43/44 时也必须先各自完成 B0，之后其他配置才会加载对应 seed 的共享 M*。

第二阶段按 C0–C4 组合运行 42、43、44，但必须在第一阶段分析完成后启动。C1
（M2 + P0.2 + E20）优先，C2 次之，C3 再次；C4 只作为压力测试。

正式执行必须显式给出一个 stage、一个 seed 和至少一个 experiment，例如：

```bash
/mnt/sda/mtzh/xp/envs/fedpoi-py39/bin/python experiments/run_sweep.py \
  --plan configs/sweeps/ucf101_poison_strength.yaml \
  --stage single_factor --experiment B0 --seed 42 \
  --approved-commit <FULL_COMMIT> --execute
```

合法中断在同一命令增加 `--resume`。跨 stage/seed 的大矩阵除 `--execute` 外还必须
显式增加 `--allow-full-matrix`；不要一次无筛选地启动全部 36 个 run。

## 报告要求

`summary.json` schema v2 自动记录：

- 0→1 ASR、百分比和源类样本数；
- ΔASR 百分点；
- 类0准确率/召回率、类1假阳性率；
- 非源类别准确率和 Macro-F1；
- Test Acc、UAR、F1及相对 clean 的下降；
- 实际恶意参与轮数、恶意席位、有效投毒更新和总投毒样本；
- 各恶意客户端生成器 checkpoint SHA-256。

筛选器将 ASR≥60%、ΔASR≥40 个百分点、全局准确率下降≤5 个百分点、非源类
准确率下降≤3 个百分点标记为 `strong_targeted_candidate`。P100 在 replace 语义下
是用类0条件生成、标签1训练的合成特征替换全部本地样本，不是对真实样本做全量
标签翻转；若全局准确率下降超过10个百分点，`sweep_run.json` 会标记为
`availability_or_model_collapse`。

这些 60/40/5/3/10 阈值属于本轮 `configs/sweeps/ucf101_poison_strength.yaml`
的筛选规则，只写入 sweep 分析 provenance；通用 `summary.json` 只保存原始指标和差值。

B0/E20/E50 使用不同且不可覆盖的 artifact 目录，因此分别保留 5、20、50 epoch
checkpoint。最终分析仍需检查类0条件生成的音频/视频多样性、均值/方差差距以及
跨种子联邦 ASR，不能只看生成器目标概率。
