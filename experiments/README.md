# Unified experiment entry points

These commands are the compatibility layer for the gradual MFL-Poison
refactor.  Existing scripts under `fed_multimodal/Local` remain supported, but
new automation should call these entry points so generator variants can be
swapped without changing attack code.

```bash
python experiments/train_generator.py --generator dtm -- --epochs 50

# The variant and loss/schedule flags can also come from one versioned config.
python experiments/train_generator.py \
  --config configs/generators/temporal_div005_avoid050_start10.json -- --epochs 50

python experiments/evaluate_generator.py \
  --generator dtm --checkpoint path/to/checkpoint.pt -- --num_batches 20

python experiments/generate_synthetic.py \
  --generator dtm --checkpoint path/to/checkpoint.pt \
  --num_samples 5100 --target_strategy balanced \
  --attack_mode clean_label --output outputs/dtm/synthetic.pt

python experiments/evaluate_tstr.py \
  --synthetic_data outputs/dtm/synthetic.pt --num_epochs 100
```

Arguments after `--` are forwarded to the compatible legacy trainer or
evaluator where applicable.
Explicit forwarded arguments override values loaded from the config file.

Synthetic artifacts use the canonical SyntheticBatch schema by default. The
TSTR entry point reads both canonical and legacy artifacts, selects the victim
checkpoint on a held-out validation split, and touches the real test split only
for the final evaluation.
