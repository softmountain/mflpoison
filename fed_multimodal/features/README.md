# UCF101 特征准备

本目录只保留当前统一场景所需的 UCF101 上游数据工具：

1. `data_partitioning/partition_manager.py` 与 `ucf101/data_partition.py`：建立 fold/alpha/client 划分；
2. `simulation_features/simulation_manager.py` 与 `ucf101/`：生成缺失模态、标签噪声或缺失标签配置；
3. `feature_processing/feature_manager.py` 与 `ucf101/`：提取 MFCC 音频和 MobileNetV2 视频特征。

runner 本身不调用这些脚本，而是读取已经生成的结果：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/alpha10/fold1/
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/alpha10/fold1/
```

默认 `dataset.alpha: 1.0` 映射到 `alpha10`。训练客户端、dev 和 test 必须在音频与视频目录中一一对应。
