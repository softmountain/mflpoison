#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mflpoison.artifacts import save_synthetic
from mflpoison.attacks import balanced_targets, clean_label_labels, label_flip_labels
from mflpoison.generators import GENERATOR_REGISTRY, load_generator_backend


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a canonical synthetic artifact")
    parser.add_argument("--generator", required=True, choices=list(GENERATOR_REGISTRY.names()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_strategy", choices=["balanced", "fixed"], default="balanced")
    parser.add_argument("--target_class", type=int, default=-1)
    parser.add_argument("--attack_mode", choices=["clean_label", "label_flip"], default="clean_label")
    parser.add_argument("--source_class", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--legacy_format", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = load_generator_backend(args.generator, args.checkpoint, args.device)
    if args.target_strategy == "balanced":
        targets = balanced_targets(args.num_samples, backend.num_classes)
    else:
        if args.target_class < 0 or args.target_class >= backend.num_classes:
            raise ValueError("fixed target strategy requires a valid --target_class")
        targets = torch.full((args.num_samples,), args.target_class, dtype=torch.long)

    if args.attack_mode == "clean_label":
        train_labels = clean_label_labels(targets)
        source_labels = None
    else:
        train_labels = label_flip_labels(targets, source_class=args.source_class)
        source_labels = train_labels.clone()

    batch = backend.generate(
        targets,
        train_labels=train_labels,
        source_labels=source_labels,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    batch.metadata.update(
        {
            "attack_mode": args.attack_mode,
            "target_strategy": args.target_strategy,
            "target_class": args.target_class,
            "source_class": args.source_class,
        }
    )
    output = save_synthetic(batch, args.output, legacy=args.legacy_format)
    print(json.dumps({"output": str(output), "num_samples": batch.num_samples}, indent=2))


if __name__ == "__main__":
    main()
