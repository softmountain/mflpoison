# UCI-HAR 联邦学习与标签翻转攻击说明

## 数据集概览
- **来源**：Human Activity Recognition Using Smartphones（HAR）数据集，由 30 名 19~48 岁受试者佩戴三星 Galaxy S II 智能手机在腰部位置采集。
- **传感器**：三轴加速度计与三轴陀螺仪，采样率 50Hz。
- **活动类别**：6 类（WALKING、WALKING_UPSTAIRS、WALKING_DOWNSTAIRS、SITTING、STANDING、LAYING）。
- **预处理**：对信号进行去噪、分离重力/身体加速度，并使用 2.56 秒（128 帧）滑窗、50% overlap 切片；每个窗口提取 561 维时间/频域特征。
- **原始划分**：官方提供 70% 受试者为训练、30% 为测试；在此基础上我们进一步构建联邦客户端与 dev 集。

## 联邦学习设置
### 客户端与数据划分
- 每名受试者的数据切分为 **5 个子客户端**，共 150 个潜在客户端，保证每个客户端只含来自单一受试者/窗口的样本。
- 采用 Dirichlet 分布控制非独立同分布度：`alpha ∈ {0.1, 5.0}`；alpha 越小，客户端之间差异越大。
- `features/data_partitioning/uci-har/data_partition.py` 负责依据设定生成 `train/dev/test` JSON；`dev`/`test` 作为全局评估集合，不参与训练。
- 预处理（`features/feature_processing/uci-har/extract_feature.py`）对每个 alpha 生成对齐的加速度与陀螺仪特征文件。

### 训练超参数（与 `run_base.sh`、`train.py` 保持一致）
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fed_alg` | `fed_avg`, `fed_opt`, `fed_prox`（可扩展 `scaffold`, `fed_rs`） | 联邦聚合算法 |
| `sample_rate` | 0.1 | 每轮随机抽取 10% 客户端参与训练 |
| `num_epochs` | 200 | 全局训练轮数 |
| `local_epochs` | 1 | 每个客户端本地迭代次数 |
| `learning_rate` | 0.05 | 客户端优化器学习率（SGD） |
| `global_learning_rate` | 0.025 | 用于 FedOpt/FedProx 的全局步长 |
| `mu` | 0.01 | FedProx 正则系数（其它算法忽略） |
| `hid_size` | 128 | HARClassifier RNN 隐层维度 |
| `en_att`, `att_name` | `True`, `fuse_base` | 是否启用多模态自注意力融合 |
| `batch_size` | 16 | 本地 batch 大小 |
| `test_frequency` | 1 | 每轮都在 dev/test 评估 |
| 评价指标 | Macro F1 | `Server.log_epoch_result(metric='f1')` 依据 dev F1 选取 best epoch |
| `monitor_labels` | `None` | 逗号分隔的标签 ID，如 `--monitor_labels 1,2`，用于在 dev/test 日志与 `result.json` 中额外输出对应类别的准确率 |

### 模型与流程
1. `DataloadManager` 根据配置加载 acc/gyro 特征，构造 `dataloader_dict`（训练客户端随机打乱，dev/test 固定顺序）。
2. `HARClassifier` 同时处理加速度与陀螺仪模态，支持注意力与缺模模拟。
3. `Server` 负责：
   - 每轮抽样客户端 (`sample_rate`)，
   - 收集客户端上传的参数/控制变量，
   - `average_weights()` 聚合并广播，
   - `inference()` 在 `dev`/`test` 上评估并按 F1 保存最佳模型，
   - 将 `label.json` 与 `result.json` 输出到 `result/<fed_alg>/<dataset>/...`。

## 标签翻转攻击方法
### 总体思路
- 入口脚本：`experiment/uci-har/train_label_flip.py`，由 `run_label_flip.sh` 批量调用。
- 攻击类：`fed_multimodal/attacks/label_flip_attack.py`，在构建 dataloader 之前对目标客户端/数据副本动手脚，保证后续流程对攻击透明。
- 目的：在保持多模态特征结构不变的情况下，将指定标签（默认 WALKING_UPSTAIRS，id=1）更改为另一个标签（默认 WALKING_DOWNSTAIRS，id=2），制造定向 label noise。

### 训练阶段策略
- **目标客户端**：仅攻击 ID 以 `-1` 结尾的客户端（每个受试者的第一个切片），可通过 `--attack_client_suffix` 修改。
- **翻转概率**：满足条件的样本以 `--attack_prob`（默认 0.5）概率被替换为目标标签。
- **实现细节**：对 acc/gyro 列表进行深拷贝后同步修改标签字段，确保两模态一致；若某客户端缺少源标签则跳过。

### Dev/Test 处理
- **Dev 集**：为模拟评估偏差，从 dev 中筛选源标签样本，随机翻转 `total_dev_samples * attack_dev_ratio` 条（默认 0.2）。
- **Test 集**：始终保持干净，以衡量攻击对泛化性能的真实影响。

### 可配置参数（`train_label_flip.py` 解析）
| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--attack_src_label` | 1 | 被翻转的原始标签 ID |
| `--attack_dst_label` | 2 | 翻转后的目标标签 ID |
| `--attack_client_suffix` | `-1` | 仅匹配该后缀的客户端会被投毒；支持逗号分隔（如 `-1,-2`）同时攻击多个分区，留空表示所有客户端 |
| `--attack_prob` | 0.5 | 训练客户端中单样本翻转概率 |
| `--attack_dev_ratio` | 0.2 | dev 集按样本数的翻转比例 |
| `--attack_seed` | 42 | 控制随机性以便复现 |

### 运行示例
```bash
cd experiment/uci-har
bash run_label_flip.sh  # 批量对 alpha∈{0.1,5.0}、fed_alg∈{fed_avg,fed_opt,fed_prox} 执行攻击

# 或者单独配置：
python3 train_label_flip.py \
  --alpha 0.1 --fed_alg fed_avg --num_epochs 200 \
  --attack_src_label 3 --attack_dst_label 4 \
  --attack_client_suffix "-2" --attack_prob 0.6 --attack_dev_ratio 0.15

# 监控特定标签在 dev/test 的准确率
python3 train.py \
  --alpha 0.1 --fed_alg fed_avg --monitor_labels 1,2,5
```

训练完成后，结果存储在 `result/<fed_alg>_label_flip/uci-har/<modality>/<att_setting>/<model_setting>/`，便于与干净基线 (`result/<fed_alg>/...`) 对比。
