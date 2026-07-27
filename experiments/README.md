# Unified UCF101 scenario entry point

The production poisoning flow has one entry point. It trains the clean global
model, selects M* on the development split, trains one generator per malicious
client partition, executes clean/attack/defended branches with the same client
schedule, and writes lineage and round audit records.

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml

python experiments/run_scenario.py \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml \
  --artifact-root artifacts/my-run
```

`experiments/train_generator.py`, `train_dtm_poison_gan.py`, and
`train_temporal_adaptive_gan.py` are temporary aliases for the same runner.
They require the complete eight-section scenario config; they no longer load a
centralized `full_train` dataset.

Generator checkpoints remain usable through the legacy-compatible inference
and evaluation entry points:

```bash
python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- \
  --model_path path/to/teacher.pt \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --partition test --num_batches 20

python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --num_samples 5100 --target_strategy balanced \
  --attack_mode clean_label --output artifacts/manual/dtm-synthetic.pt

python experiments/evaluate_tstr.py \
  --synthetic_data artifacts/manual/dtm-synthetic.pt -- \
  --data_dir fed_multimodal/results --alpha 1.0 --fold 1 \
  --num_epochs 100
```

Synthetic artifacts use the canonical SyntheticBatch schema by default. The
TSTR entry point reads canonical and historical artifact schemas, selects the
victim checkpoint on the fixed FedMM `dev` partition, and touches FedMM `test`
only for the final evaluation. This intentionally differs from the former
random 10% split of centralized training data and is not numerically equivalent.

Generator evaluators default to FedMM `test`. Train-side quality evaluation
must use `--partition client --client-id ID`; deprecated `--use_train` also
requires a client ID. Implementations live in
`fed_multimodal.legacy_evaluation`; capitalized `fed_multimodal/Local` files are
thin command wrappers only. K+1-family reports use timestamped JSON files and
record partition, client, alpha, fold, seed, and teacher-checkpoint lineage.
