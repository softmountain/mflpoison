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
  --generator dtm --checkpoint path/to/checkpoint.pt -- --num_batches 20

python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --num_samples 5100 --target_strategy balanced \
  --attack_mode clean_label --output outputs/dtm/synthetic.pt

python experiments/evaluate_tstr.py \
  --synthetic_data outputs/dtm/synthetic.pt --num_epochs 100
```

Synthetic artifacts use the canonical SyntheticBatch schema by default. The
TSTR entry point reads both canonical and legacy artifacts, selects the victim
checkpoint on a held-out validation split, and touches the real test split only
for the final evaluation.
