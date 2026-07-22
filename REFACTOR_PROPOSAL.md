# FedPoisonGAN 代码重构提案

> 范围：基于 FDMM 基线，在多模态数据集（UCF101 为例）上训练 GAN 生成音视频合成特征，用于对联邦学习实施中毒攻击，并构建防御模块。
> 提案日期：2026-07-14
> 状态：设计稿，待评审

---

## 一、概述

### 1.1 现状
项目已实现两种 GAN 形式、三个 K+1 变体、中毒特征生成与集中式下游评估，联邦训练主流程（fed_avg/fed_rs/scaffold）可用。但代码随演进逐步分散，攻击逻辑散落三处，防御模块尚未建立。

### 1.2 判断
**应当重构，且现在正是时机。** 三个 GAN 变体已暴露"复制-改一点"的成本；防御模块尚未建立，若不先抽象，防御代码会重蹈攻击代码分散的覆辙。当前是阻塞最少、收益最高的重构窗口。

### 1.3 目标
- GAN 两种形式 + 三个变体归并为**正交的两维**（架构 × 正则），消除训练循环复制。
- `Local/` 脚本爆炸收敛为**配置驱动的统一入口**。
- 攻击与防御作为**联邦训练的可插拔模块**，独立组合实验。
- 数据格式与评估契约**统一**。

---

## 二、现状结构

### 2.1 两种 GAN 形式

| 形式 | 判别器结构 | 全局模型角色 | 代码位置 |
|---|---|---|---|
| 一、Teacher | 双生成器 + 双判别器 + Joint Critic | **Teacher**，提供语义分类损失（`cls_weight`） | `generator/gan_generator.py` + `Local/train_local_gan.py` |
| 二、K+1 | K+1 判别器（冻结骨干 + 可训头，`fake_class=K`） | **判别器骨干** | `poison_gan/`、`dtm_poison_gan/`、`temporal_adaptive_gan/` |

### 2.2 三个 K+1 变体

| 包 | 行数 (config/models/losses/trainer) | 复用 poison_gan | 差异本质 |
|---|---|---|---|
| `poison_gan` | 46/142/145/229 | — | K+1 基线 |
| `dtm_poison_gan` | 56/209/285/456 | kplus1/memory_bank/metrics/PoisonDiscriminator | 新增 MMD+var_floor+tail+mode_seek 正则 |
| `temporal_adaptive_gan` | 105/200/242/310 | 同上 | 新增 temporal 自适应正则 |

### 2.3 `Local/` 脚本分布（5 类职责混杂）
- GAN 训练：`train_poison_gan` / `train_dtm_poison_gan` / `train_temporal_adaptive_gan` / `train_local_gan`
- 特征生成：`generate_poison_features` / `generate_dtm_poison_features` / `generate_temporal_adaptive_features`
- GAN 评估：`eval_poison_gan` / `eval_dtm_poison_gan` / `eval_temporal_adaptive_gan` / `eval_local_gan_quality`
- 下游接入：`train_synthetic`(TSTR) / `train_with_poison_features` / `train_with_fake`
- 集中式训练：`train_local`

### 2.4 攻击/防御代码分布
- `generator/`：`label_flip_attack.py`、`gan_generator.py`、`eval_gan_quality.py`
- `Local/`：`train_with_poison_features.py`（集中式，非联邦注入）、`train_with_fake.py`
- `demo/attack/`：大量实验脚本（`robustness_client`、`poisoned_eval`、`label_shift` 等）
- `trainers/server_trainer.py`：**无中毒客户端注入接口、无防御 hook**
- 防御模块：**未建立**

---

## 三、问题诊断

### 问题 1：三个 K+1 包同构复制
每个变体各写一整套 config/models/losses/trainer，但差异仅在**正则项组合**。`dtm` 的 trainer 456 行、`temporal` 的 310 行，其中训练循环与 `poison_gan` 的 229 行高度重复。新增一个变体需复制 4 个文件 + 4 个 Local 脚本。

### 问题 2：`Local/` 脚本爆炸，职责混杂
每变体 × {train, generate, eval, run.sh} = 12+ 脚本，模式一致仅差一个 import。Local/ 同时承载 GAN 训练、特征生成、评估、联邦接入、集中式训练五类职责，无目录边界。

### 问题 3：两种 GAN 形式无统一抽象
Teacher 形式与 K+1 形式是完全独立的代码路径，无共同基类。生成、评估、下游接入各写一套，格式契约靠口头约定。

### 问题 4：攻击分散、防御缺失、联邦 trainer 无 hook
攻击逻辑散落三处；`train_with_poison_features` 是集中式训练而非联邦注入；`server_trainer` 没有攻击注入点与防御聚合 hook；防御模块完全未建。后续加防御若不先抽象，会继续分散。

---

## 四、目标架构

```
fed_multimodal/
├── gan/                          # 合并 generator/ + poison_gan/ + dtm + temporal
│   ├── base.py                   # BaseGANTrainer: train_epoch/generate/evaluate 抽象
│   ├── components.py             # ClassEmbeddingBank, build_kplus1, metrics（共享）
│   ├── losses.py                 # 共享损失库: mmd/var_floor/mode_seek/tail/raw_stat
│   ├── archs/
│   │   ├── teacher.py            # 形式一: 全局模型作 teacher (← generator/gan_generator)
│   │   └── kplus1.py             # 形式二: K+1 判别器 (← poison_gan/models+kplus1)
│   └── variants/
│       ├── base.py               # K+1 基线正则 (← poison_gan/trainer+config)
│       ├── dtm.py                # 仅 loss 组合 (← dtm_poison_gan)
│       └── temporal_adaptive.py  # 仅 loss 组合 (← temporal_adaptive_gan)
├── attack/                       # 收拢 generator/ + Local/train_with_* + demo/attack 核心
│   ├── poisoning.py              # PoisonTensorDataset, clean_label/label_flip/label_shift
│   ├── client_adapter.py         # 中毒数据注入联邦客户端
│   └── strategies.py             # 攻击策略枚举 + 工厂
├── defense/                      # 新建
│   ├── base.py                   # BaseDefense: on_aggregate(clients, aggregated) hook
│   ├── robust_agg.py             # Krum / Median / TrimmedMean
│   └── detection.py              # 异常客户端检测
├── trainers/                     # 现有，server_trainer 加两个 hook
│   └── server_trainer.py         #   attack_client_adapter + defense.on_aggregate
├── pipeline/                     # 统一入口，替代 Local/ 脚本爆炸
│   ├── train_gan.py              #   --arch {teacher,kplus1} --variant {base,dtm,temporal}
│   ├── generate_features.py      #   --checkpoint ... --attack_mode ...
│   ├── eval_gan.py               #   --mode {adversarial,tstr}
│   └── run_federated.py          #   --attack ... --defense ...
├── experiments/                  # demo/attack 的实验脚本归档于此
└── (dataloader / model / features / constants / experiment 保持)
```

---

## 五、关键设计

### 5.1 GAN 抽象：架构 × 正则 两维正交

```python
# gan/base.py
class BaseGANTrainer(ABC):
    @abstractmethod
    def train_epoch(self, epoch, max_batches=None): ...
    @abstractmethod
    def generate(self, num_samples, labels) -> dict: ...   # 统一输出契约
    @abstractmethod
    def evaluate(self, loader, mode: str) -> dict: ...     # mode = adversarial | tstr

# 架构维：决定判别器形式
class TeacherGANTrainer(BaseGANTrainer): ...   # 形式一
class KPlus1GANTrainer(BaseGANTrainer): ...    # 形式二

# 正则维：仅决定 loss 组合，通过 config 注入，不复制训练循环
@dataclass
class VariantConfig:
    losses: List[LossSpec]          # [{name, weight, kwargs}]
    diversity_start_epoch: int = 3
    ...

# dtm = KPlus1 + [mmd, var_floor, tail, mode_seek]
# temporal_adaptive = KPlus1 + [temporal_loss, ...]
# base = KPlus1 + [minimal]
```

**收益**：`dtm_poison_gan/trainer.py`（456 行）与 `temporal_adaptive_gan/trainer.py`（310 行）中重复的训练循环消失，各变体只剩 loss 组合的几十行 config。

### 5.2 统一生成契约

```python
# 所有 GAN.generate() 返回统一格式（已是现有事实格式，正式化为契约）
{
  "audio": Tensor[N, Ta, Da], "video": Tensor[N, Tv, Dv],
  "len_a": Tensor[N], "len_v": Tensor[N],
  "condition_label": Tensor[N], "train_label": Tensor[N],
  "meta": {"arch": ..., "variant": ..., "checkpoint": ..., "attack_mode": ...}
}
```
`train_with_poison_features`、`train_synthetic`、联邦客户端适配器统一消费此格式，不再各自硬编码。

### 5.3 攻击/防御作为 server_trainer 可插拔 hook

```python
# trainers/server_trainer.py（改造点）
class ServerTrainer:
    def __init__(self, ..., attack=None, defense=None):
        self.attack = attack       # attack.ClientAdapter | None
        self.defense = defense     # defense.BaseDefense | None

    def train_round(self):
        clients = self.sample_clients()
        if self.attack:
            clients = self.attack.inject(clients)          # 注入中毒客户端
        updates = [c.train() for c in clients]
        aggregated = self.aggregate(updates)
        if self.defense:
            aggregated = self.defense.on_aggregate(clients, updates, aggregated)
        self.apply(aggregated)
```

攻击与防御解耦，可任意组合（如 `--attack label_flip --defense krum`）。

---

## 六、文件迁移映射

| 现有文件 | 目标位置 | 处理 |
|---|---|---|
| `generator/gan_generator.py` | `gan/archs/teacher.py` | 迁移 |
| `poison_gan/models.py` + `poison_gan/kplus1.py` | `gan/archs/kplus1.py` | 合并 |
| `poison_gan/memory_bank.py` + `poison_gan/metrics.py` | `gan/components.py` | 合并 |
| `poison_gan/losses.py` + `dtm_poison_gan/losses.py` + `temporal_adaptive_gan/losses.py` | `gan/losses.py` | 合并去重 |
| `poison_gan/trainer.py` | `gan/variants/base.py` + `gan/base.py` | 拆分（循环→base，正则→variant） |
| `dtm_poison_gan/{trainer,config}.py` | `gan/variants/dtm.py` | 压缩为 loss 组合 |
| `temporal_adaptive_gan/{trainer,config}.py` | `gan/variants/temporal_adaptive.py` | 压缩为 loss 组合 |
| `Local/train_*_gan.py`（4 个） | `pipeline/train_gan.py` | 统一入口 |
| `Local/generate_*_features.py`（3 个） | `pipeline/generate_features.py` | 统一入口 |
| `Local/eval_*.py`（4 个） | `pipeline/eval_gan.py` | 统一入口（--mode） |
| `Local/train_synthetic.py` | `pipeline/eval_gan.py --mode tstr` | 合并 |
| `Local/train_with_poison_features.py` + `train_with_fake.py` | `attack/client_adapter.py` + `pipeline/run_federated.py` | 拆分 |
| `generator/label_flip_attack.py` | `attack/poisoning.py` | 迁移 |
| `generator/eval_gan_quality.py` | `pipeline/eval_gan.py` | 合并 |
| `Local/train_local.py` | 保留原位 | 集中式基线，独立功能 |
| `Local/run_*.sh` | `pipeline/run_*.sh` 或删除 | 收敛 |
| `demo/attack/*` | `experiments/attack/*` | 归档为实验脚本 |
| `trainers/server_trainer.py` | 原位改造 | 加 attack/defense hook |

---

## 七、迁移路径（3 阶段）

### 阶段 1：抽共享层（风险低，立即可做）
- 建 `gan/components.py`、`gan/losses.py`，上提 poison_gan 的 kplus1/memory_bank/metrics 与三包共有损失。
- 三个 GAN 包改为 import 共享层，删除各自重复实现。
- **验收**：三包训练/评估行为不变，现有 checkpoint 可正常加载评估。
- **风险**：低，纯抽取，行为不变。

### 阶段 2：统一入口（风险中，需回归）
- 建 `pipeline/{train_gan,generate_features,eval_gan}.py`，配置驱动。
- `Local/` 旧脚本保留为薄 wrapper（调用 pipeline）或标记 deprecated，不立即删除。
- 正式化生成数据格式契约。
- **验收**：`pipeline/train_gan.py --arch kplus1 --variant dtm` 等价于原 `train_dtm_poison_gan.py`。
- **风险**：中，需对每个变体做一轮回归（训练 + 评估指标对齐）。

### 阶段 3：attack/defense 框架（风险中，新功能）
- 建 `attack/` + `defense/`，给 `server_trainer` 加 hook。
- 将 `generator/label_flip_attack`、`Local/train_with_poison_features` 核心逻辑迁入 `attack/`。
- 实现 `pipeline/run_federated.py --attack ... --defense ...`。
- **验收**：能跑通"GAN 生成 → 注入中毒客户端 → 联邦训练 → 防御聚合 → 评估攻击成功率/防御鲁棒性"全链路。
- **风险**：中，属新功能，不破坏旧路径。

### 优先级
- **阶段 1 现在做**：收益大、风险低、为后续铺路。
- **阶段 2** 可与阶段 3 合并推进。
- **阶段 3 是接下来的主线**（防御模块），应先建框架再填实现，避免防御代码分散。

---

## 八、风险与回退

| 风险 | 缓解 |
|---|---|
| 重构期间破坏现有可跑的 GAN 训练 | 阶段 1 只抽取不改行为；旧包路径保留，新共享层通过 import 生效，可随时回退 |
| 三变体 loss 细节差异被合并后丢失 | 合并前用现有 checkpoint 做回归（各变体评估指标对齐），差异作为 `VariantConfig` 显式保留 |
| `Local/` 脚本被外部（shell/cron/文档）引用 | 旧脚本保留为 wrapper，不删除；迁移完成后再 deprecate |
| 联邦 trainer 加 hook 影响现有实验 | hook 默认 None（无攻击无防御时行为不变），现有实验脚本不受影响 |

---

## 九、结论

当前结构"能跑"但"不可持续扩展"：三个 GAN 变体已证明复制成本，攻击代码分散已造成维护负担，防御模块若不先抽象将继续恶化。

建议按三阶段重构：**先抽共享层（阶段 1，低风险高收益）→ 统一入口（阶段 2）→ 建 attack/defense 框架（阶段 3，主线）**。核心是把"架构 × 正则"两维正交化，把攻击/防御做成 `server_trainer` 的可插拔 hook，让后续的防御研究有干净的落点。

阶段 1 可立即启动，不破坏任何现有功能。
