"""P10: fragility 实验归因分析 + 假设验证。

1. baseline×3 (seed 8/9/10) 取均值 drift, 量化噪声 (drift_std);
2. 15 攻击组扣除 drift 得净效果 (net Δt_recall / net ΔH / net Δacc);
3. A vs C 对比 (fragility 假设): 高 fragility target 净破坏应显著大于低 fragility;
4. A vs B 对比 (cosine 假设, 旧): 远 source vs 近 source 对同一 target 的破坏;
5. fragility(target) vs net Δt_recall 相关性 — 直接验证 fragility 预测攻击强度;
6. target softmax 熵变化 (net) — 攻击导致模型对 target 困惑的早期信号。

输入:
  --combos (combos_fragility.json, 含 A/B/C 分组 + cosine)
  --fragility_json (target fragility)
  --result_root (fed_multimodal/result/fed_avg_poison, 含 frag_*/result.json)
  --baseline_seeds 8 9 10
输出: fragility_attributed.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_result_json(result_root: Path, subdir: str):
    for f in result_root.rglob("result.json"):
        if f.parent.name == subdir:
            return f
    return None


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", required=True)
    p.add_argument("--fragility_json", required=True)
    p.add_argument("--result_root", default="/home/xp/fedpoi/fed_multimodal/result/fed_avg_poison")
    p.add_argument("--baseline_seeds", default="8,9,10")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    result_root = Path(args.result_root)
    combos = json.load(open(args.combos))
    frag = json.load(open(args.fragility_json))
    pc = frag["per_class"]
    num_classes = frag["num_classes"]
    fragility_of = lambda c: pc[str(c)]["fragility"]

    # --- baseline x3 drift (mean + std) ---
    base_pcr, r14_pcr, base_ent, r14_ent, base_acc, r14_acc = [], [], [], [], [], []
    for seed in args.baseline_seeds.split(","):
        f = find_result_json(result_root, f"frag_baseline_seed{seed}")
        if f is None:
            print(f"[warn] baseline seed{seed} result.json not found, skipping")
            continue
        r = json.load(open(f))["eval_history"]
        base_pcr.append(r["base"]["per_class_recall"])
        r14_pcr.append(r["round14"]["per_class_recall"])
        base_ent.append(r["base"]["per_class_entropy"])
        r14_ent.append(r["round14"]["per_class_entropy"])
        base_acc.append(r["base"]["acc"]); r14_acc.append(r["round14"]["acc"])
    if not base_pcr:
        raise SystemExit("no baseline results found; run baselines first")
    base_pcr = np.array(base_pcr); r14_pcr = np.array(r14_pcr)
    base_ent = np.array(base_ent); r14_ent = np.array(r14_ent)
    drift_recall = (r14_pcr - base_pcr).mean(axis=0)           # [C]
    drift_recall_std = (r14_pcr - base_pcr).std(axis=0)        # noise floor per class
    drift_entropy = (r14_ent - base_ent).mean(axis=0)
    drift_acc = float(np.mean(np.array(r14_acc) - np.array(base_acc)))
    noise_std_global = float(np.nanstd(r14_pcr - base_pcr))
    print(f"baseline x{len(base_pcr)}: drift_acc={drift_acc:+.2f}, "
          f"per-class recall drift std (median noise)={np.nanmedian(drift_recall_std):.1f}")

    # --- attack groups: net metrics ---
    def net_for(subdir, target, source, delta_t, delta_s, delta_acc, ev):
        d_t = drift_recall[target]; d_s = drift_recall[source]
        d_ent_t = drift_entropy[target]
        base_H = ev["base"]["per_class_entropy"][target]
        r14_H = ev["round14"]["per_class_entropy"][target]
        return {
            "subdir": subdir, "target": target, "source": source,
            "d_t_recall": delta_t, "drift_t": float(d_t),
            "net_d_t_recall": float(delta_t - d_t),
            "net_d_s_recall": float(delta_s - d_s),
            "net_d_acc": float(delta_acc - drift_acc),
            "d_H_target": float(r14_H - base_H) if not (np.isnan(base_H) or np.isnan(r14_H)) else float("nan"),
            "net_d_H_target": float((r14_H - base_H) - d_ent_t) if not (np.isnan(base_H) or np.isnan(r14_H)) else float("nan"),
        }

    groups = {}
    for gkey in ["combos_A", "combos_B", "combos_C"]:
        rows = []
        for row in combos.get(gkey, []):
            subdir = f"frag_{row['group']}_s{row['source']}_t{row['target']}"
            f = find_result_json(result_root, subdir)
            if f is None:
                print(f"[warn] {subdir} not found"); continue
            r = json.load(open(f))
            rows.append(net_for(subdir, row["target"], row["source"],
                                r["delta"]["target_recall"], r["delta"]["source_recall"],
                                r["delta"]["acc"], r["eval_history"]))
        groups[gkey[-1]] = rows  # 'A'/'B'/'C'

    def stat(rows, key):
        vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan"), float(np.std(vals)) if vals else float("nan"), len(vals)

    A, B, C = groups["A"], groups["B"], groups["C"]
    a_net, _, _ = stat(A, "net_d_t_recall")
    b_net, _, _ = stat(B, "net_d_t_recall")
    c_net, _, _ = stat(C, "net_d_t_recall")

    # fragility vs net destruction across all attack targets (A+C, distinct targets)
    frag_pts, net_pts = [], []
    for rows in [A, C]:
        seen = set()
        for r in rows:
            if r["target"] in seen: continue
            seen.add(r["target"])
            frag_pts.append(fragility_of(r["target"]))
            net_pts.append(r["net_d_t_recall"])
    corr_frag_net = pearson(frag_pts, net_pts)

    # entropy change A vs C
    a_dH, _, _ = stat(A, "net_d_H_target")
    c_dH, _, _ = stat(C, "net_d_H_target")

    print(f"\n=== A/B/C net Δt_recall (baseline-corrected) ===")
    print(f"  A (fragile×far):   {a_net:+.1f}  (n={len(A)})")
    print(f"  B (fragile×near):  {b_net:+.1f}  (n={len(B)})")
    print(f"  C (robust×far):    {c_net:+.1f}  (n={len(C)})")
    print(f"  fragility vs net-destruction pearson (A+C targets): {corr_frag_net:.3f}")
    print(f"  net entropy change: A={a_dH:+.3f}  C={c_dH:+.3f}")

    out = {
        "baseline": {"n_seeds": len(base_pcr), "drift_acc": drift_acc,
                     "noise_std_global": noise_std_global,
                     "drift_recall_median_std": float(np.nanmedian(drift_recall_std)),
                     "drift_recall": drift_recall.tolist(),
                     "drift_entropy": drift_entropy.tolist()},
        "groups": {g: rows for g, rows in groups.items()},
        "group_means": {
            "A_net_d_t_recall": a_net, "B_net_d_t_recall": b_net, "C_net_d_t_recall": c_net,
            "A_net_d_H_target": a_dH, "C_net_d_H_target": c_dH,
        },
        "hypothesis_tests": {
            "fragility (A vs C)": {"A_net": a_net, "C_net": c_net,
                                   "fragile_more_destroyed": a_net < c_net,
                                   "delta": a_net - c_net},
            "cosine (A vs B, same targets)": {"A_far_net": a_net, "B_near_net": b_net,
                                              "far_more_destroying": a_net < b_net},
            "fragility_corr_with_net_destruction": corr_frag_net,
        },
    }
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "fragility_attributed.json", "w"), indent=2, default=str)

    # markdown report
    md = ["# Fragility-driven Attack — Baseline-corrected Attribution\n",
          f"baseline x{len(base_pcr)} (no-attack): drift_acc={drift_acc:+.2f}, "
          f"per-class recall drift std (median noise floor)={np.nanmedian(drift_recall_std):.1f} pts. "
          f"A net effect is only meaningful if |net| exceeds this noise.\n",
          "## group means (net Δt_recall, baseline-corrected)\n",
          "| group | target | source | net Δt_recall | net ΔH_target |",
          "|---|---|---|---|---|",
          f"| A (fragile × farthest) | top5 fragility | farthest | **{a_net:+.1f}** | {a_dH:+.3f} |",
          f"| B (fragile × nearest)  | top5 fragility | nearest  | **{b_net:+.1f}** | — |",
          f"| C (robust × farthest)  | bottom5       | farthest | **{c_net:+.1f}** | {c_dH:+.3f} |",
          "\n## hypothesis tests\n",
          f"### fragility 假设 (A vs C)",
          f"- A (高 fragility) 净破坏 {a_net:+.1f} vs C (低 fragility) {c_net:+.1f} → "
          f"{'成立: fragile target 被破坏更重' if a_net < c_net else '不成立'} (Δ={a_net-c_net:+.1f})",
          f"- fragility 与 net 破坏相关性 (A+C targets): pearson **{corr_frag_net:.3f}**",
          f"\n### cosine 假设 (A vs B, 旧假设)",
          f"- A (远 source) {a_net:+.1f} vs B (近 source) {b_net:+.1f} → "
          f"{'cosine 低破坏强(旧假设方向)' if a_net < b_net else 'cosine 无关或反向'}",
          f"\n### softmax 熵变化 (target, net)",
          f"- A 组 target 熵变化 {a_dH:+.3f}, C 组 {c_dH:+.3f} "
          f"(攻击使模型对 fragile target 更困惑 → A 应更高)\n",
          "## per-group detail\n",
          "| group | target | source | cosine | Δt | drift | net Δt | net ΔH |",
          "|---|---|---|---|---|---|---|---|"]
    for gk in ["A", "B", "C"]:
        for r in groups[gk]:
            cos = next((c["cosine"] for c in combos[f"combos_{gk}"] if c["target"] == r["target"] and c["source"] == r["source"]), float("nan"))
            md.append(f"| {gk} | {r['target']} | {r['source']} | {cos:.3f} | {r['d_t_recall']:+.1f} | {r['drift_t']:+.1f} | {r['net_d_t_recall']:+.1f} | {r['net_d_H_target']:+.3f} |")
    (out_dir / "fragility_attributed.md").write_text("\n".join(md))
    print(f"\nSaved fragility_attributed.{{json,md}} to {out_dir}")


if __name__ == "__main__":
    main()
