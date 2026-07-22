"""P0: 从 TSTR per-class recall 排序选出 source 类与 target 候选。

source_best5  = recall 最高的 5 个类（GAN 合成质量最好）
source_worst5 = recall 最低的 5 个类（GAN 合成质量最差）
source_classes = best5 ∪ worst5（10 个，作为投毒内容来源）
target_worst5  = worst5（方式2 的 target 候选）

类名从 client pkl 的 key 前缀按字母序重建（与 partition/dataloader 一致）。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path


def build_class_names(client_pkl_dir: Path, num_clients: int):
    classes = set()
    for cid in range(num_clients):
        p = client_pkl_dir / f"{cid}.pkl"
        if not p.exists():
            continue
        for key, *_ in pickle.load(open(p, "rb")):
            classes.add(str(key).split("/")[0])
    sorted_classes = sorted(classes)
    return {idx: name for idx, name in enumerate(sorted_classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class_recall_json", required=True)
    parser.add_argument("--client_pkl_dir", required=True, help="audio client pkl dir (for class-name rebuild)")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    raw = json.load(open(args.per_class_recall_json))
    # filter NaN, keep (class_id, recall)
    items = []
    for k, v in raw.items():
        r = float(v)
        if math.isnan(r):
            continue
        items.append((int(k), r))
    items.sort(key=lambda x: x[1])  # ascending

    idx2name = build_class_names(Path(args.client_pkl_dir), args.num_clients)

    print("=== TSTR per-class recall ranking (low -> high) ===")
    print(f"{'rank':>4} {'cls':>4} {'recall':>8}  name")
    for rank, (cid, r) in enumerate(items):
        print(f"{rank:>4} {cid:>4} {r:>8.2f}  {idx2name.get(cid, '?')}")

    worst5 = [cid for cid, _ in items[:5]]
    best5 = [cid for cid, _ in reversed(items[-5:])]
    source_classes = sorted(set(best5 + worst5))
    target_worst5 = worst5

    print("\n=== selection ===")
    print(f"source_best5  (high recall): {best5}  <- {[idx2name[c] for c in best5]}")
    print(f"source_worst5 (low recall) : {worst5}  <- {[idx2name[c] for c in worst5]}")
    print(f"source_classes (10, union) : {source_classes}")
    print(f"target_worst5 (method 2)   : {target_worst5}")

    out = {
        "source_best5": best5,
        "source_worst5": worst5,
        "source_classes": source_classes,
        "target_worst5": target_worst5,
        "per_class_recall": {str(k): v for k, v in raw.items()},
        "class_names": {str(k): v for k, v in idx2name.items()},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output_json, "w"), indent=2)
    print(f"\nSaved selection to {args.output_json}")


if __name__ == "__main__":
    main()
