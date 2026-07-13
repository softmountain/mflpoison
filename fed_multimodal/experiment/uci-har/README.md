### Quick Start -- UCI-HAR (Acc. and Gyro)
Here we provide an example to quickly start with the experiments, and reproduce the UCI-HAR results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data: The data will be under data/uci-har by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_uci_har.sh
cd ..
```

#### 1. Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity. As each subject performs the same amount activities, we partition each subject data into 5 sub-clients.

```
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 5
```

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction

For UCI-HAR dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/uci-har
# output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/uci-har
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality
```
cd experiment/uci-har
bash run_mm.sh
```

#### 6. Run targeted label-flipping attack experiments (UCI-HAR only)
To reproduce the WALKING\_UPSTAIRS → WALKING\_DOWNSTAIRS (50%) label-flip attack, we provide a dedicated training entrypoint and shell script that mirror the base workflow but keep the attack logic separate from the clean pipeline:

```
cd experiment/uci-har
bash run_label_flip.sh
```

`run_label_flip.sh` calls `train_label_flip.py`, which injects the attack before dataloaders are built. By default we:

- target only clients whose id ends with `-1` (i.e., shard 1 of each subject) in the training split and flip their WALKING_UPSTAIRS labels to WALKING_DOWNSTAIRS with probability `--attack_prob` (default 0.5);
- randomly pick `--attack_dev_ratio` (default 0.2) of dev samples sharing the source label and flip them, while the test split remains untouched;
- keep the attack fully configurable through `--attack_src_label`, `--attack_dst_label`, `--attack_client_suffix`, `--attack_prob`, `--attack_dev_ratio`, and `--attack_seed`. `--attack_client_suffix` now accepts comma-separated values (e.g., `-1,-2`) to poison多个分区。

Example for remapping label `3` to `4` with a different poisoned shard and dev ratio:

```
python3 train_label_flip.py --attack_src_label 3 --attack_dst_label 4 \
	--attack_client_suffix "-2" --attack_prob 0.6 --attack_dev_ratio 0.15

# 毒化多个分区示例
python3 train_label_flip.py --attack_client_suffix "-1,-2,-3"
```

Results are stored under `result/<fed_alg>_label_flip/...` to avoid overwriting clean runs.

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 105 | Natural+Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 77.74% <br> 85.17% <br> 76.66% <br> 79.80% | 0.05 | 200 |

