"""P5: 汇总中毒攻击结果 + 相似度↔攻击效果相关性分析 + 报告生成。

读所有 P4 产出的 result.json (含 source/target/baseline/after/delta),
结合 P2 的 real/synth centers 算每组的 cosine 相似度,
计算相似度与攻击效果(-Δtarget_recall, -Δsource_recall, -Δacc)的 Pearson/Spearman 相关,
输出 summary.json + markdown 表 + (可选)散点图。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-8))


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True, help="dir to recursively search for result.json")
    parser.add_argument("--real_centers", required=True)
    parser.add_argument("--synth_centers", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot", action="store_true", help="generate scatter plots (needs matplotlib)")
    args = parser.parse_args()

    real = torch.load(args.real_centers, map_location="cpu")["centers"]
    synth = torch.load(args.synth_centers, map_location="cpu")["centers"]

    records = []
    for f in glob.glob(str(Path(args.result_root) / "**" / "result.json"), recursive=True):
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"skip {f}: {e}"); continue
        s, t = r["source_class"], r["target_class"]
        cos = cos_sim(synth[s], real[t]) if (s in synth and t in real) else float("nan")
        rec = {
            "source": s, "target": t, "cosine": cos,
            "method": Path(f).parent.name.split("_m")[-1] if "_m" in Path(f).parent.name else "?",
            "acc_base": r["baseline"]["acc"], "acc_after": r["after"]["acc"], "d_acc": r["delta"]["acc"],
            "t_recall_base": r["baseline"]["target_recall"], "t_recall_after": r["after"]["target_recall"],
            "d_t_recall": r["delta"]["target_recall"],
            "s_recall_base": r["baseline"]["source_recall"], "s_recall_after": r["after"]["source_recall"],
            "d_s_recall": r["delta"]["source_recall"],
            "s2t_base": r["baseline"].get("source_to_target", float("nan")),
            "s2t_after": r["after"].get("source_to_target", float("nan")),
            "d_s2t": r["delta"].get("source_to_target", float("nan")),
            "inject": sum(r.get("inject_counts", {}).values()),
        }
        records.append(rec)

    if not records:
        print("No result.json found under", args.result_root); return

    print(f"Collected {len(records)} attack experiments")

    # attack effectiveness metrics (higher = stronger attack)
    for rec in records:
        rec["eff_target破坏"] = -rec["d_t_recall"]      # target recall 下降越多越成功
        rec["eff_source误判"] = -rec["d_s_recall"]      # source recall 下降越多越成功
        rec["eff_acc下降"] = -rec["d_acc"]
        rec["eff_source_to_target"] = rec["d_s2t"]      # source→target 误判增幅（最直接攻击度量）

    valid = [r for r in records if not np.isnan(r["cosine"])]
    cosines = [r["cosine"] for r in valid]
    eff_t = [r["eff_target破坏"] for r in valid]
    eff_s = [r["eff_source误判"] for r in valid]
    eff_a = [r["eff_acc下降"] for r in valid]
    valid_s2t = [r for r in valid if not np.isnan(r["d_s2t"])]
    cosines_s2t = [r["cosine"] for r in valid_s2t]
    eff_s2t = [r["eff_source_to_target"] for r in valid_s2t]

    corr = {
        "n": len(cosines),
        "cosine_vs_sourceToTarget": {"pearson": pearson(cosines_s2t, eff_s2t), "spearman": spearman(cosines_s2t, eff_s2t), "n": len(cosines_s2t)},
        "cosine_vs_targetDestruction": {"pearson": pearson(cosines, eff_t), "spearman": spearman(cosines, eff_t)},
        "cosine_vs_sourceMisclassify": {"pearson": pearson(cosines, eff_s), "spearman": spearman(cosines, eff_s)},
        "cosine_vs_accDrop": {"pearson": pearson(cosines, eff_a), "spearman": spearman(cosines, eff_a)},
    }
    print("\n=== correlation: cosine similarity vs attack effectiveness ===")
    print(json.dumps(corr, indent=2))

    # sort by source→target misclassification gain (primary attack metric); nan -> -inf
    by_target = sorted(records, key=lambda r: (r["d_s2t"] if not np.isnan(r["d_s2t"]) else -1e9), reverse=True)
    print("\n=== top-10 by source→target misclassification gain ===")
    print(f"{'src':>4} {'tgt':>4} {'m':>3} {'cos':>6} {'s2t_b':>6} {'s2t_a':>6} {'Δs2t':>6} {'ΔtRcl':>7} {'Δacc':>7} {'inj':>5}")
    for r in by_target[:10]:
        print(f"{r['source']:>4} {r['target']:>4} {r['method']:>3} {r['cosine']:>6.3f} "
              f"{r['s2t_base']:>6.1f} {r['s2t_after']:>6.1f} {r['d_s2t']:>+6.1f} {r['d_t_recall']:>+7.2f} {r['d_acc']:>+7.2f} {r['inject']:>5}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"records": records, "correlation": corr}, open(out / "poison_summary.json", "w"), indent=2)

    # markdown table
    md = ["# Poison Attack Summary\n", f"共 {len(records)} 组实验\n",
          "## 相关性 (cosine 相似度 vs 攻击效果)\n",
          f"- source→target 误判(最直接): pearson={corr['cosine_vs_sourceToTarget']['pearson']:.3f}, spearman={corr['cosine_vs_sourceToTarget']['spearman']:.3f}",
          f"- target 类被破坏: pearson={corr['cosine_vs_targetDestruction']['pearson']:.3f}, spearman={corr['cosine_vs_targetDestruction']['spearman']:.3f}",
          f"- source 类被误判: pearson={corr['cosine_vs_sourceMisclassify']['pearson']:.3f}, spearman={corr['cosine_vs_sourceMisclassify']['spearman']:.3f}",
          f"- 全局 acc 下降:   pearson={corr['cosine_vs_accDrop']['pearson']:.3f}, spearman={corr['cosine_vs_accDrop']['spearman']:.3f}\n",
          "## 全部结果（按 source→target 误判增幅降序）\n",
          "| source | target | m | cosine | s2t base→after | Δs2t | Δt_recall | Δacc | inject |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in by_target:
        md.append(f"| {r['source']} | {r['target']} | m{r['method']} | {r['cosine']:.3f} | "
                  f"{r['s2t_base']:.1f}→{r['s2t_after']:.1f} | {r['d_s2t']:+.1f} | {r['d_t_recall']:+.1f} | {r['d_acc']:+.1f} | {r['inject']} |")
    Path(out / "poison_summary.md").write_text("\n".join(md))
    print(f"\nSaved poison_summary.json and poison_summary.md to {out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for eff_name, eff_vals, fname in [
                ("target-class destruction", [r["eff_target破坏"] for r in records], "scatter_target.png"),
                ("source-class misclassification", [r["eff_source误判"] for r in records], "scatter_source.png"),
            ]:
                xs = [r["cosine"] for r in records]; ys = eff_vals
                plt.figure(figsize=(6, 4))
                plt.scatter(xs, ys)
                plt.xlabel("cosine(synth_source, real_target)")
                plt.ylabel(f"attack effectiveness ({eff_name})")
                plt.title(f"cosine vs {eff_name}\nPearson={pearson(xs, ys):.3f}")
                plt.tight_layout(); plt.savefig(out / fname, dpi=120); plt.close()
            print(f"Saved scatter plots to {out}")
        except Exception as e:
            print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
