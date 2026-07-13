# UCF101 demo 数据处理管线

这个目录下是一个和原始 `features/data_partitioning/ucf101`、`experiment/ucf101` 分离的独立 demo。

目标：
- 删除 `dev`
- 用更接近均匀的客户端划分替代 Dirichlet
- 输出 demo 专用 manifest / feature cache / packaged client files

## 目录说明

- `build_partition.py`：根据官方 split 构建 demo manifest
- `extract_features.py`：提取 demo 专用 audio/video feature cache
- `package_clients.py`：把 manifest + cache 打包成每客户端 `.pkl`
- `loader.py`：demo 专用读取器，不依赖旧实验中的 `dev`
- `train.py`：demo 专用联邦训练入口，仅依赖 demo packaged 数据

## 输出目录

默认输出到：

- `fed_multimodal/results/demo/ucf101/partition/fold{n}/manifest.json`
- `fed_multimodal/results/demo/ucf101/feature_cache/...`
- `fed_multimodal/results/demo/ucf101/packaged/fold{n}/audio/*.pkl`
- `fed_multimodal/results/demo/ucf101/packaged/fold{n}/video/*.pkl`

## 运行顺序

### 1. 数据处理

```bash
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.build_partition --num_clients 15
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.extract_features
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.package_clients
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.verify_demo_ucf101 --folds 1 2 3
```

### 2. 联邦训练

单 fold smoke test：

```bash
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.train --folds 1 --num_epochs 10 --sample_rate 1.0 --local_epochs 1 --batch_size 8
```

3-fold 正式实验：

```bash
conda run -n fdmm python -m fed_multimodal.demo.ucf101_demo.train --folds 1 2 3 --num_epochs 10 --sample_rate 1.0 --local_epochs 1 --batch_size 8
```

或：

```bash
bash fed_multimodal/demo/ucf101_demo/run_train.sh --folds 1 2 3 --num_epochs 10 --sample_rate 1.0 --local_epochs 1 --batch_size 8
```

结果输出到：

- `fed_multimodal/results/demo/ucf101/training/result.json`

## 说明

- 不生成 `dev`
- `test` 保持官方 holdout
- 训练样本使用标签内打乱 + round-robin 分发到客户端
- packaged 记录格式保持为 `[key, file_path, label, feature]`
- 当前 demo 使用自己的 loader；原 `experiment/ucf101/train.py` 不能直接无修改复用
