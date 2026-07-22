# Fragility-driven Attack — Baseline-corrected Attribution

baseline x3 (no-attack): drift_acc=+0.75, per-class recall drift std (median noise floor)=2.2 pts. A net effect is only meaningful if |net| exceeds this noise.

## group means (net Δt_recall, baseline-corrected)

| group | target | source | net Δt_recall | net ΔH_target |
|---|---|---|---|---|
| A (fragile × farthest) | top5 fragility | farthest | **-5.1** | +0.082 |
| B (fragile × nearest)  | top5 fragility | nearest  | **+1.3** | — |
| C (robust × farthest)  | bottom5       | farthest | **-0.5** | +0.005 |

## hypothesis tests

### fragility 假设 (A vs C)
- A (高 fragility) 净破坏 -5.1 vs C (低 fragility) -0.5 → 成立: fragile target 被破坏更重 (Δ=-4.6)
- fragility 与 net 破坏相关性 (A+C targets): pearson **-0.532**

### cosine 假设 (A vs B, 旧假设)
- A (远 source) -5.1 vs B (近 source) +1.3 → cosine 低破坏强(旧假设方向)

### softmax 熵变化 (target, net)
- A 组 target 熵变化 +0.082, C 组 +0.005 (攻击使模型对 fragile target 更困惑 → A 应更高)

## per-group detail

| group | target | source | cosine | Δt | drift | net Δt | net ΔH |
|---|---|---|---|---|---|---|---|
| A | 49 | 38 | -0.026 | -31.4 | -19.1 | -12.4 | +0.260 |
| A | 14 | 42 | -0.034 | +5.1 | +12.8 | -7.7 | +0.118 |
| A | 35 | 21 | -0.021 | -14.3 | -2.0 | -12.2 | +0.021 |
| A | 2 | 20 | 0.087 | +0.0 | -3.3 | +3.3 | -0.011 |
| A | 23 | 21 | 0.158 | +2.2 | -1.5 | +3.7 | +0.022 |
| B | 49 | 42 | 0.180 | -11.4 | -19.1 | +7.6 | -0.119 |
| B | 14 | 38 | 0.452 | +0.0 | +12.8 | -12.8 | +0.065 |
| B | 35 | 38 | 0.431 | -4.1 | -2.0 | -2.0 | -0.038 |
| B | 2 | 38 | 0.398 | +2.4 | -3.3 | +5.7 | -0.072 |
| B | 23 | 42 | 0.516 | +6.7 | -1.5 | +8.2 | +0.045 |
| C | 38 | 42 | 0.068 | +0.0 | +0.0 | +0.0 | -0.017 |
| C | 41 | 42 | 0.057 | -3.2 | +3.2 | -6.5 | +0.003 |
| C | 8 | 42 | -0.081 | +15.2 | +13.1 | +2.0 | +0.032 |
| C | 36 | 42 | 0.005 | +0.0 | -2.1 | +2.1 | +0.001 |
| C | 37 | 21 | 0.003 | +0.0 | +0.0 | +0.0 | +0.004 |