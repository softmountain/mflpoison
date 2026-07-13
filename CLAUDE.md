# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FedMultimodal is a Python/PyTorch benchmark for multimodal federated learning. The package is installed editable with `pip install -e .` and experiment scripts are organized by dataset under `fed_multimodal/experiment/*`.

## Environment and setup

- Use Python 3.9 when reproducing the documented environment.
- Install the package from the repository root:
  ```bash
  pip install -e .
  ```
- Development extras are declared in `setup.py`:
  ```bash
  pip install -e '.[dev]'
  ```
- `fed_multimodal/system.cfg` controls the default dataset/output roots. `data_dir` is the raw/processed dataset root and `output_dir` is the default feature/result root used by training scripts.

## Common commands

There is no dedicated build step beyond editable installation.

- Run tests if/when tests are added:
  ```bash
  pytest
  pytest path/to/test_file.py
  pytest path/to/test_file.py::test_name
  ```
- Format/lint tools available via dev extras:
  ```bash
  black fed_multimodal
  isort fed_multimodal
  flake8 fed_multimodal
  ```
- UCI-HAR quick-start data download:
  ```bash
  cd fed_multimodal/data && bash download_uci_har.sh
  ```
- UCI-HAR partitioning and feature extraction from `fed_multimodal/`:
  ```bash
  python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
  python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 5
  python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
  python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0
  ```
- UCI-HAR base experiments from `fed_multimodal/experiment/uci-har/`:
  ```bash
  bash run_base.sh
  ```
  This script loops over `alpha in 0.1 5.0` and `fed_alg in fed_avg fed_opt fed_prox`, using `taskset -c 1-30`; remove or adjust `taskset` on machines where CPU affinity is not appropriate.
- Run one UCI-HAR training job directly from `fed_multimodal/experiment/uci-har/`:
  ```bash
  python3 train.py --alpha 0.1 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --fed_alg fed_avg --mu 0.01 --en_att --att_name fuse_base --hid_size 128
  ```

## Architecture

- `fed_multimodal/experiment/<dataset>/train.py` files are the main entry points. They parse dataset-specific arguments, read `system.cfg`, build a `DataloadManager`, choose the client trainer for `fed_alg`, construct the model, and run folds/rounds.
- `fed_multimodal/features/` is the preprocessing pipeline:
  - `data_partitioning/*` creates client partitions; Dirichlet `alpha` controls heterogeneity when there is no natural client split.
  - `feature_processing/*` extracts or normalizes pretrained modality features.
  - `simulation_features/*` generates optional missing-modality, label-noise, and missing-label metadata.
- Processed data follows the project convention under `feature/`, `partition/`, and `simulation_feature/`. Client examples are stored as lists like `[key, file_name, label, feature]`; simulation entries include missing modality/label-noise flags.
- `fed_multimodal/dataloader/dataload_manager.py` maps dataset/modality settings to feature paths, loads per-client pickle/JSON files, builds train/dev/test `DataLoader`s, pads variable-length sequences, and injects optional simulation metadata.
- `fed_multimodal/model/mm_models.py` contains the multimodal and unimodal PyTorch classifiers. Most models use GRU-based encoders with optional attention (`multihead`, `additive`, `base`, `fuse_base`); `fuse_base` changes the classifier input shape.
- `fed_multimodal/trainers/` contains federated learning logic:
  - `fed_avg_trainer.py`, `fed_rs_trainer.py`, and `scaffold_trainer.py` implement client-side local training variants.
  - `server_trainer.py` handles client sampling, aggregation/update logic, evaluation, TensorBoard logging, and JSON result summaries.
  - `evaluation.py` computes metrics such as accuracy, UAR, F1, and multilabel PTB-XL metrics.
- `fed_multimodal/constants/constants.py` defines dataset label counts and feature dimensions used when constructing models.
- `fed_multimodal/generator/` contains GAN/attack-related utilities such as GAN generation, label-flip attack code, and GAN quality evaluation.

## Dataset and output conventions

- Supported experiment folders include audio/video, audio/text, image/text, accelerometer/gyro, and ECG splits: `crema_d`, `meld`, `ucf101`, `mit10`, `mit51`, `uci-har`, `ku-har`, `extrasensory`, `extrasensory_watch`, `ptb-xl`, `crisis-mmd`, `hateful_memes`, and `ego4d-ttm`.
- Feature paths are built from modality names and feature names, e.g. `feature/{modality}/{feature_name}/{dataset}/...`; many datasets include `alpha{value}` and/or `foldK` subdirectories.
- Training logs are written under `{data_dir}/log/{fed_alg}/{dataset}/{feature}/{attention}/{setting}/foldK/raw_log`; result summaries are written to result/result-like directories depending on the script.
- Run scripts commonly assume CUDA and may bind CPU cores with `taskset`.

## Repository-specific working notes

- Preserve the feature and simulation directory naming conventions because `DataloadManager` discovers clients and folds from those paths.
- When adding a new dataset or modality pair, update the training entry point, `DataloadManager` path/loading branches, model construction in `mm_models.py` if needed, and dimensions/classes in `constants.py`.
- Avoid committing generated checkpoints, plots, TensorBoard logs, or large files under `fed_multimodal/result`, `fed_multimodal/results`, or `fed_multimodal/Local/results` unless explicitly requested.
- On this machine, run long experiments sequentially rather than launching multiple concurrent experiment jobs.
