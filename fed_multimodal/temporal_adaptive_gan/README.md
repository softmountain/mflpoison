# Temporal Adaptive Poison GAN

`temporal_adaptive_gan` is a K+1 generator objective used by the unified
scenario runner. Its model, losses, trainer, and checkpoint format remain
separate from the legacy implementation.

## Design

- Running real-audio statistics calibrate generated audio without per-sample
  z-normalization or hard audio clipping.
- Per-frame noise, positional embeddings, temporal convolution, and
  class-specific scale/bias model video diversity and temporal structure.
- Real sequence lengths mask generated features and distribution losses.
- The preset uses a 1:3 D/G schedule, decaying instance noise, lazy R1,
  feature/statistical matching, audio mean/std/kurtosis matching, and a
  diversity warm-up.
- A server-broadcast prototype bank can initialize missing-class targets.

## Entry points

Set `generator.variant: temporal_adaptive` in a complete scenario config, then
run the production entry point. The old training filename is a temporary alias
for this same command.

```bash
python -m mflpoison.runner --config path/to/temporal-scenario.yaml

python fed_multimodal/Local/eval_temporal_adaptive_gan.py --checkpoint path/to/checkpoint.pt

python fed_multimodal/Local/generate_temporal_adaptive_features.py \
  --checkpoint path/to/checkpoint.pt --num_samples 1000
```
