# DTM-GAN

`dtm_poison_gan` is the legacy-compatible implementation used by the unified
scenario runner for the Distributional Temporal Matching objective. Its
checkpoint format remains separate from the temporal-adaptive variant.

## Combined design

- The K+1 discriminator remains an auxiliary realism signal with a low weight.
- The primary objective is class-conditional teacher-embedding alignment:
  local classes use multi-scale RBF MMD; missing classes fall back to a
  server-broadcast class mean and diagonal variance.
- A VICReg-style variance floor prevents generated class spread from falling
  below real class spread.
- Audio is generated without per-sample z-normalization or hard tail clipping.
  Running real-data mean and standard deviation provide a learnable affine
  calibration, while skew and kurtosis are matched separately.
- Video uses per-frame noise, a GRU temporal decoder, and
  `sigmoid(gate) * softplus(magnitude)` to model sparse MobileNet features.
- Generated and real sequence lengths are aligned before distribution losses.

## Train

```bash
bash fed_multimodal/Local/run_dtm_poison_gan_cloud.sh
```

The script reads the complete scenario YAML from `SCENARIO_CONFIG` and defaults
to `configs/scenarios/ucf101_generative_poison_defense.yaml`. Generator
training is performed once per malicious client partition after M* selection;
there is no centralized `full_train` input. Override artifact output without
editing the config as follows:

```bash
ARTIFACT_ROOT=artifacts/dtm-run \
  bash fed_multimodal/Local/run_dtm_poison_gan_cloud.sh
```

## Evaluate or generate

```bash
python fed_multimodal/Local/eval_dtm_poison_gan.py \
  --checkpoint path/to/final_dtm_cloud.pt

python fed_multimodal/Local/generate_dtm_poison_features.py \
  --checkpoint path/to/final_dtm_cloud.pt \
  --num_samples 1000 \
  --target_strategy balanced \
  --attack_mode clean_label
```
