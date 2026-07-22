"""为 55 组攻击实验每组绘制折线图 + 生成完整数据表。

每组折线: source 类 recall、target 类 recall、全局 acc 随攻击轮次 (baseline→4→9→14) 变化。
数据从 /tmp/poison_source{s}_target{t}_m{method}.log 解析（每轮 eval_every 5 的指标行）。
色盘 Okabe-Ito（dataviz validator PASS，CVD 安全）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"source": "#0072B2", "target": "#E69F00", "global": "#009E73"}  # Okabe-Ito
MARKERS = {"source": "o", "target": "s", "global": "^"}
BASE = Path("/home/xp/fedpoi/fed_multimodal/Local/results/poison_attack")


def parse_log(path: Path):
    base = None
    rounds = {}
    if not path.exists():
        return None, None
    for line in open(path):
        m = re.search(r"\[baseline M\*\] acc=([\d.]+) target_recall=([\d.]+) source_recall=([\d.]+)", line)
        if m:
            base = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        m = re.search(r"\[round (\d+)\] acc=([\d.]+) target_recall=([\d.]+) source_recall=([\d.]+)", line)
        if m:
            rounds[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return base, rounds


def main():
    summary = json.load(open(BASE / "analysis_all" / "poison_summary.json"))
    sel = json.load(open(BASE / "source_target_selection.json"))
    names = sel.get("class_names", {})
    recs = summary["records"]

    out_dir = BASE / "analysis_all" / "line_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    x_labels = ["base\n(M*)", "4", "9", "14"]
    x = [0, 1, 2, 3]

    rows = []
    plotted = 0
    missing = 0
    for r in sorted(recs, key=lambda r: (int(r["method"]), r["source"], r["target"])):
        s, t, m = r["source"], r["target"], r["method"]
        log = Path(f"/tmp/poison_source{s}_target{t}_m{m}.log")
        base, rounds = parse_log(log)
        if base is None:
            missing += 1
            rows.append((s, t, m, r, None))
            continue
        pts = {"global": [base[0]], "target": [base[1]], "source": [base[2]]}
        for rnd in [4, 9, 14]:
            d = rounds.get(rnd)
            for k in pts:
                pts[k].append(d[{"global": 0, "target": 1, "source": 2}[k]] if d else pts[k][-1])

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(x, pts["source"], "-", color=COLORS["source"], marker=MARKERS["source"], markersize=7, linewidth=2,
                label=f"source cls {s} ({names.get(str(s),'?')}) recall")
        ax.plot(x, pts["target"], "-", color=COLORS["target"], marker=MARKERS["target"], markersize=7, linewidth=2,
                label=f"target cls {t} ({names.get(str(t),'?')}) recall")
        ax.plot(x, pts["global"], "-", color=COLORS["global"], marker=MARKERS["global"], markersize=7, linewidth=2,
                label="global acc")
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_xlabel("attack round"); ax.set_ylabel("%")
        ax.set_title(f"method {m} | source {s} → target {t} | cosine(synth_s, real_t)={r['cosine']:.3f}", fontsize=10)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25); ax.set_ylim(-5, 105)
        png_name = f"line_s{s}_t{t}_m{m}.png"
        plt.tight_layout(); plt.savefig(out_dir / png_name, dpi=110); plt.close()
        plotted += 1
        rows.append((s, t, m, r, png_name))

    print(f"Plotted {plotted} groups, missing logs for {missing}")

    # full data table
    md = ["# 55 组特征碰撞攻击实验完整数据表\n",
          "每行一组实验。折线图列点开见该组的 source/target recall 与 global acc 随攻击轮次变化。",
          "`base` = M\*（攻击前），`4/9/14` = 攻击轮次。\n",
          "| method | source | target | cosine | s2t base→after | Δs2t | Δt_recall | Δs_recall | Δacc | inject | 折线图 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, t, m, r, png in rows:
        link = f"[📈](line_plots/{png})" if png else "—"
        md.append(f"| m{m} | {s} {names.get(str(s),'')} | {t} {names.get(str(t),'')} | {r['cosine']:.3f} | "
                  f"{r['s2t_base']:.1f}→{r['s2t_after']:.1f} | {r['d_s2t']:+.1f} | {r['d_t_recall']:+.1f} | "
                  f"{r['d_s_recall']:+.1f} | {r['d_acc']:+.1f} | {r['inject']} | {link} |")
    (BASE / "analysis_all" / "all_groups_table.md").write_text("\n".join(md))
    print(f"Saved all_groups_table.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
