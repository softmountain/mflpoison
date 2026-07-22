"""P3: 生成 (source, target) 攻击组合。

方式1 (特征驱动): 每个 source s, 取 synth_centers[s] 与所有 real_centers[c] 余弦相似度最高的真实类 c (排除 s 自身) 为 target.
   -> 10 组 (每 source 一个最近真实类)
方式2 (TSTR 驱动): target ∈ target_worst5, 每个 source × 每个 target.
   -> 10 × 5 = 50 组 (排除 source==target)

输出 combos_method1.json / combos_method2.json, 含每组 cosine 相似度, 供 P4 调度和 P5 分析.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_centers", required=True)
    parser.add_argument("--synth_centers", required=True)
    parser.add_argument("--selection_json", required=True, help="P0 output: source_target_selection.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    real = torch.load(args.real_centers, map_location="cpu")["centers"]      # {cls: tensor}
    synth = torch.load(args.synth_centers, map_location="cpu")["centers"]    # {source: tensor}
    sel = json.load(open(args.selection_json))
    source_classes = sel["source_classes"]
    target_worst5 = sel["target_worst5"]

    # method 1: nearest real class per source
    combos1 = []
    print("=== method 1 (feature-driven: nearest real class per source) ===")
    for s in source_classes:
        if s not in synth:
            print(f"  source {s}: no synth center, skip")
            continue
        sims = {c: cos_sim(synth[s], real[c]) for c in real if c != s and c in real}
        if not sims:
            continue
        best_t = max(sims, key=sims.get)
        combos1.append({"source": int(s), "target": int(best_t), "cosine": sims[best_t]})
        print(f"  source {s:>3} -> target {best_t:>3}  cosine={sims[best_t]:.4f}")

    # method 2: all source x worst5 targets
    combos2 = []
    print("\n=== method 2 (TSTR-driven: source x worst5 targets) ===")
    for s in source_classes:
        for t in target_worst5:
            if s == t:
                continue
            c = cos_sim(synth[s], real[t]) if (s in synth and t in real) else float("nan")
            combos2.append({"source": int(s), "target": int(t), "cosine": c})
    print(f"  {len(combos2)} combos (10 sources x 5 targets, source!=target)")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(combos1, open(out / "combos_method1.json", "w"), indent=2)
    json.dump(combos2, open(out / "combos_method2.json", "w"), indent=2)
    print(f"\nSaved {len(combos1)} method1 combos and {len(combos2)} method2 combos to {out}")


if __name__ == "__main__":
    main()
