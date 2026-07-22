"""P8: 三阶段脆弱 target 选择 + A/B/C 消融矩阵生成。

依据 docs/iid+稳定数据投毒+target多参数选择(7.20改进方案).md 的 pipeline:
  阶段1: M* recall >= 70% 的类 (候选)
  阶段2: fragility top10 (从阶段1候选); top5=A/B 的 target, 阶段1候选里 fragility 最低5=bottom5=C 的 target
  阶段3: 每个 target 配 penultimate cosine 最远(min)/最近(max)的 source

source 选择: TSTR 降序, 过滤 M* real_recall <= 90 (避免旧实验 best5 真实 recall 95+ 导致 s2t=0), 取前 5。

矩阵:
  A (5 组): top5 fragility target × 各自最远 source     — 主实验
  B (5 组): top5 fragility target × 各自最近 source     — 对照 cosine 效应 (A vs B)
  C (5 组): bottom5 fragility target × 各自最远 source  — 对照 fragility 效应 (A vs C)
  baseline ×3 在 run 阶段单独跑 (不同 seed), 不进 combos

输入:
  --fragility_json (compute_fragility 输出)
  --tstr_json (per-class TSTR recall dict)
  --real_centers, --synth_centers (extract_class_centers 用新 M* 重跑)
  --source_pool (可选, 限定 source 候选; 默认全部 51 类)
输出: combos_*.json (A/B/C 各 5 组) + 选择报告 md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fragility_json', required=True)
    p.add_argument('--tstr_json', required=True)
    p.add_argument('--real_centers', required=True)
    p.add_argument('--synth_centers', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--recall_threshold', type=float, default=70.0, help='stage-1 min recall')
    p.add_argument('--source_recall_cap', type=float, default=90.0, help='source real-recall cap (avoid 95+ solid classes)')
    p.add_argument('--topk', type=int, default=5)
    args = p.parse_args()

    frag = json.load(open(args.fragility_json))
    pc = frag['per_class']  # {str(c): {recall, H_entropy, tstr, gap, fragility}}
    num_classes = frag['num_classes']
    class_names = json.load(open(Path(args.fragility_json).with_name('source_target_selection.json'))).get('class_names', {}) \
        if (Path(args.fragility_json).parent / 'source_target_selection.json').exists() else {}

    tstr = {int(k): float(v) for k, v in json.load(open(args.tstr_json)).items()}

    real = torch.load(args.real_centers, map_location='cpu')['centers']
    synth = torch.load(args.synth_centers, map_location='cpu')['centers']

    recall_of = lambda c: pc[str(c)]['recall']
    frag_of = lambda c: pc[str(c)]['fragility']

    # --- source selection: TSTR desc, real_recall <= cap, take top K ---
    all_classes = list(range(num_classes))
    tstr_sorted = sorted(all_classes, key=lambda c: tstr.get(c, -1), reverse=True)
    source_candidates = [c for c in tstr_sorted if recall_of(c) is not None and recall_of(c) <= args.source_recall_cap]
    if len(source_candidates) < args.topk:
        # relax cap
        relaxed = [c for c in tstr_sorted if recall_of(c) is not None and recall_of(c) <= args.source_recall_cap + 5]
        source_candidates = relaxed
        print(f'[warn] source pool below {args.topk} at cap {args.source_recall_cap}; relaxed to {args.source_recall_cap + 5}')
    sources = source_candidates[:args.topk]
    print(f'selected sources (TSTR top, recall<={args.source_recall_cap}): {sources}')
    for s in sources:
        print(f'  source {s} {class_names.get(str(s),"")}: TSTR={tstr.get(s):.1f} recall={recall_of(s):.1f}')

    # only keep sources that have a synth center
    sources = [s for s in sources if s in synth]
    assert len(sources) >= 3, f'need >=3 sources with synth centers, got {len(sources)}'

    # --- stage 1: recall >= threshold ---
    stage1 = [c for c in all_classes if recall_of(c) is not None and recall_of(c) >= args.recall_threshold]
    print(f'\nstage-1 (recall>={args.recall_threshold}): {len(stage1)} classes')

    # --- stage 2: fragility top/bottom from stage-1 ---
    stage1_by_frag = sorted(stage1, key=lambda c: frag_of(c), reverse=True)
    target_top = stage1_by_frag[:args.topk]
    target_bottom = sorted(stage1, key=lambda c: frag_of(c))[:args.topk]
    print(f'target top5 (fragile):    {target_top}')
    print(f'target bottom5 (robust):  {target_bottom}')

    # --- stage 3: pair each target with farthest / nearest source (penultimate cosine) ---
    def pair(target, want_far):
        sims = [(s, cos_sim(synth[s], real[target])) for s in sources]
        sims.sort(key=lambda x: x[1], reverse=not want_far)  # far=ascending(min first); near=descending(max first)
        return sims[0]

    def build_group(targets, want_far, name):
        rows = []
        for t in targets:
            s, c = pair(t, want_far)
            rows.append({'source': int(s), 'target': int(t), 'cosine': float(c),
                         'source_name': class_names.get(str(s), ''), 'target_name': class_names.get(str(t), ''),
                         'role': 'farthest' if want_far else 'nearest'})
        return rows

    groupA = build_group(target_top, want_far=True, name='A')
    groupB = build_group(target_top, want_far=False, name='B')
    groupC = build_group(target_bottom, want_far=True, name='C')

    # tag group labels (A1..A5 etc) for result_subdir naming
    for label, grp in [('A', groupA), ('B', groupB), ('C', groupC)]:
        for i, row in enumerate(grp):
            row['group'] = f'{label}{i+1}'
            row['method'] = label

    out = {
        'config': {'recall_threshold': args.recall_threshold, 'source_recall_cap': args.source_recall_cap,
                   'topk': args.topk, 'm_star_acc': frag.get('m_star_acc')},
        'sources': sources,
        'target_top5_fragile': target_top,
        'target_bottom5_robust': target_bottom,
        'combos_A': groupA,   # fragile target × farthest source
        'combos_B': groupB,   # fragile target × nearest source
        'combos_C': groupC,   # robust target × farthest source
    }
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / 'combos_fragility.json', 'w'), indent=2)

    md = ['# Fragility-driven Target Selection (v2)\n',
          f"M* acc={frag.get('m_star_acc')}, recall>={args.recall_threshold}, source cap={args.source_recall_cap}\n",
          '## sources (TSTR top, real recall <= cap)\n',
          '| source | name | TSTR | real recall |', '|---|---|---|---|']
    for s in sources:
        md.append(f'| {s} | {class_names.get(str(s),"")} | {tstr.get(s):.1f} | {recall_of(s):.1f} |')
    md += ['\n## A: fragile target × farthest source (main)\n',
           '| group | target | name | fragility | recall | source | cosine |',
           '|---|---|---|---|---|---|---|']
    for r in groupA:
        md.append(f"| {r['group']} | {r['target']} | {r['target_name']} | {frag_of(r['target']):+.2f} | {recall_of(r['target']):.1f} | {r['source']} {r['source_name']} | {r['cosine']:.3f} |")
    md += ['\n## B: fragile target × nearest source (cosine control vs A)\n',
           '| group | target | source | cosine |', '|---|---|---|---|']
    for r in groupB:
        md.append(f"| {r['group']} | {r['target']} {r['target_name']} | {r['source']} {r['source_name']} | {r['cosine']:.3f} |")
    md += ['\n## C: robust target × farthest source (fragility control vs A)\n',
           '| group | target | name | fragility | recall | source | cosine |',
           '|---|---|---|---|---|---|---|']
    for r in groupC:
        md.append(f"| {r['group']} | {r['target']} | {r['target_name']} | {frag_of(r['target']):+.2f} | {recall_of(r['target']):.1f} | {r['source']} {r['source_name']} | {r['cosine']:.3f} |")
    (out_dir / 'combos_fragility.md').write_text("\n".join(md))
    print(f'\nSaved combos_fragility.{{json,md}} to {out_dir}')
    print(f'total attack groups: A{len(groupA)}+B{len(groupB)}+C{len(groupC)} = {len(groupA)+len(groupB)+len(groupC)}, plus baseline×3')


if __name__ == '__main__':
    main()
