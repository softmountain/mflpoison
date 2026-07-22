"""P6: 用无攻击 baseline 校正攻击效果，归因真正的攻击破坏。

baseline = 相同 M*、15 轮、sample_rate 0.2、客户端划分，但 0 恶意客户端（natural drift）。
对每组攻击实验:
  net_Δt_recall = group.d_t_recall - baseline_drift_recall[target]   (扣除该 target 类的自然漂移)
  net_Δs_recall = group.d_s_recall - baseline_drift_recall[source]
  net_Δacc      = group.d_acc       - baseline_drift_acc
然后重做 cosine ↔ net 攻击效果相关性，对比未校正的旧相关性。

输入:
  - baseline result.json (eval_history 含 base/round4/9/14 的 per_class_recall + acc)
  - analysis_all/poison_summary.json (55 组 records，含 cosine + delta)
输出: analysis_all/poison_summary_attributed.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("/home/xp/fedpoi/fed_multimodal/Local/results/poison_attack")
RESULT_ROOT = Path("/home/xp/fedpoi/fed_multimodal/result/fed_avg_poison")


def find_baseline_json():
    for f in RESULT_ROOT.rglob("result.json"):
        if "baseline_noattack" in str(f):
            return f
    return None


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    bj = find_baseline_json()
    if bj is None:
        raise SystemExit("baseline result.json not found under result/fed_avg_poison/**/baseline_noattack/ — "
                         "has the baseline run finished?")
    b = json.load(open(bj))
    eh = b["eval_history"]
    base_pcr = eh["base"]["per_class_recall"]
    r14 = eh.get("round14") or list(eh.values())[-1]
    r14_pcr = r14["per_class_recall"]
    num_classes = len(base_pcr)
    drift_recall = [r14_pcr[t] - base_pcr[t] for t in range(num_classes)]
    drift_acc = r14["acc"] - eh["base"]["acc"]

    print(f"baseline_noattack: base acc={eh['base']['acc']:.2f} -> round14 acc={r14['acc']:.2f}  (drift_acc={drift_acc:+.2f})")
    print("\nper-class natural recall drift (biggest 12 drops):")
    for t in sorted(range(num_classes), key=lambda i: drift_recall[i])[:12]:
        print(f"  class {t:>2}: {drift_recall[t]:+6.1f}   ({base_pcr[t]:5.1f} -> {r14_pcr[t]:5.1f})")

    summary = json.load(open(BASE / "analysis_all" / "poison_summary.json"))
    sel = json.load(open(BASE / "source_target_selection.json"))
    names = sel.get("class_names", {})
    records = summary["records"]

    for r in records:
        t, s = r["target"], r["source"]
        r["drift_t"] = drift_recall[t]
        r["drift_s"] = drift_recall[s]
        r["net_d_t_recall"] = r["d_t_recall"] - drift_recall[t]
        r["net_d_s_recall"] = r["d_s_recall"] - drift_recall[s]
        r["net_d_acc"] = r["d_acc"] - drift_acc
        r["eff_net_target破坏"] = -r["net_d_t_recall"]

    valid = [r for r in records if not np.isnan(r["cosine"])]
    cos = [r["cosine"] for r in valid]
    old_eff_t = [-r["d_t_recall"] for r in valid]
    new_eff_t = [r["eff_net_target破坏"] for r in valid]
    old_eff_acc = [-r["d_acc"] for r in valid]
    new_eff_acc = [-r["net_d_acc"] for r in valid]

    corr = {
        "OLD_cosine_vs_targetDestruction": pearson(cos, old_eff_t),
        "NET_cosine_vs_targetDestruction": pearson(cos, new_eff_t),
        "OLD_cosine_vs_accDrop": pearson(cos, old_eff_acc),
        "NET_cosine_vs_accDrop": pearson(cos, new_eff_acc),
    }
    print("\n=== correlation: cosine vs attack effectiveness (OLD vs baseline-corrected NET) ===")
    print(json.dumps(corr, indent=2))

    print("\n=== per-target: avg Δt_recall OLD vs NET (baseline-corrected) ===")
    by_target = {}
    for r in records:
        by_target.setdefault(r["target"], []).append(r)
    print(f"{'tgt':>4} {'name':<20} {'drift':>7} {'oldΔt':>8} {'NETΔt':>8}")
    for t in sorted(by_target):
        rs = by_target[t]
        old = float(np.mean([r["d_t_recall"] for r in rs]))
        net = float(np.mean([r["net_d_t_recall"] for r in rs]))
        print(f"{t:>4} {names.get(str(t),''):<20} {drift_recall[t]:>+7.1f} {old:>+8.1f} {net:>+8.1f}")

    out = BASE / "analysis_all"
    json.dump({"records": records, "correlation": corr,
               "baseline": {"base_acc": eh["base"]["acc"], "round14_acc": r14["acc"],
                            "drift_acc": drift_acc, "drift_recall": drift_recall}},
              open(out / "poison_summary_attributed.json", "w"), indent=2)

    md = ["# Baseline-corrected Attack Attribution\n",
          f"baseline_noattack (same M\\*, 15 rounds, sr0.2, 0 malicious): "
          f"acc {eh['base']['acc']:.2f} -> {r14['acc']:.2f} (drift {drift_acc:+.2f})\n",
          "## per-class natural recall drift (baseline, biggest 15 drops)\n",
          "| class | name | base | round14 | drift |", "|---|---|---|---|---|"]
    for t in sorted(range(num_classes), key=lambda i: drift_recall[i])[:15]:
        md.append(f"| {t} | {names.get(str(t),'')} | {base_pcr[t]:.1f} | {r14_pcr[t]:.1f} | {drift_recall[t]:+.1f} |")
    md += ["\n## correlation: cosine vs attack effectiveness (OLD vs NET)\n",
           f"- OLD cosine vs target破坏: **{corr['OLD_cosine_vs_targetDestruction']:.3f}**",
           f"- NET cosine vs target破坏 (baseline-corrected): **{corr['NET_cosine_vs_targetDestruction']:.3f}**",
           f"- OLD cosine vs acc下降: {corr['OLD_cosine_vs_accDrop']:.3f}",
           f"- NET cosine vs acc下降: {corr['NET_cosine_vs_accDrop']:.3f}\n",
           "## per-target avg Δt_recall (OLD vs NET)\n",
           "| target | name | drift | old Δt | NET Δt |", "|---|---|---|---|---|"]
    for t in sorted(by_target):
        rs = by_target[t]
        old = float(np.mean([r["d_t_recall"] for r in rs]))
        net = float(np.mean([r["net_d_t_recall"] for r in rs]))
        md.append(f"| {t} | {names.get(str(t),'')} | {drift_recall[t]:+.1f} | {old:+.1f} | {net:+.1f} |")
    (out / "poison_summary_attributed.md").write_text("\n".join(md))
    print(f"\nSaved poison_summary_attributed.json/.md to {out}")


if __name__ == "__main__":
    main()
