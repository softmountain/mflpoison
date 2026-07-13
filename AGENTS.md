# AGENTS.md — 通用 Agent 规则

本文件供 Claude Code、Codex 及其他 AI 编码 Agent 在 `git pull` 后自动读取。
项目架构、环境配置等详细说明见 [`CLAUDE.md`](CLAUDE.md)。

---

## Push Policy / 推送规则

### 可以推送（Allowed）

- Python 源代码（`.py`）
- 配置文件（`.json`, `.yaml`, `.cfg`, `.toml`）
- Shell 脚本（`.sh`, `.bash`）
- **代码说明文档**：解释架构、API、模块职责、使用方法、命令行参数等
- 测试代码（`tests/`）
- 类型定义、协议接口、常量定义
- Package 配置文件

### 禁止推送（Forbidden）

- **实验方案文档**：包含研究假设、消融设计、对比策略、实验矩阵
- **实验结果/报告**：指标数值、训练曲线分析、质量评估报告、TSTR 结果、消融结论
- 训练产物（checkpoint, 特征文件, 日志）— 已被 `.gitignore` 覆盖
- 数据集文件、原始数据
- 个人工具配置（`.claude/`、`.vscode/`、`.idea/`）

### 判断标准

> 问自己：这个文件是「**解释程序怎么运行**」的，还是「**记录实验发现了什么**」的？
>
> - 前者 → 可以推送
> - 后者 → 禁止推送

### 示例

| 文件 | 判定 | 原因 |
|------|------|------|
| `experiments/README.md` | ✅ 可推 | 解释调度脚本用法 |
| `synthetic_data_quality_report_*.md` | ❌ 禁推 | 合成数据质量评估结果 |
| `trainer.py` 中新增 loss 的 docstring | ✅ 可推 | 代码说明 |
| 某次训练的 loss 曲线分析 | ❌ 禁推 | 实验结果 |
