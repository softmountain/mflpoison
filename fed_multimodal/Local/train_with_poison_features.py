#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fed_multimodal.Local.dataloader import UCF101LocalDataManager, collate_mm_fn_padd
from fed_multimodal.model.mm_models import MMActionClassifier


class PoisonTensorDataset(Dataset):
    def __init__(self, poison):
        self.poison = poison

    def __len__(self):
        return self.poison["audio"].size(0)

    def __getitem__(self, idx):
        return (
            self.poison["audio"][idx].float(),
            self.poison["video"][idx].float(),
            int(self.poison["len_a"][idx]),
            int(self.poison["len_v"][idx]),
            self.poison["train_label"][idx].long(),
        )


class MixedDataset(Dataset):
    def __init__(self, clean_dataset, poison_dataset, mode="clean_plus_poison", poison_ratio=0.2, train_length=None):
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.mode = mode
        self.poison_ratio = poison_ratio
        if mode == "clean_only":
            self.length = len(clean_dataset)
        elif mode == "poison_only":
            self.length = len(poison_dataset)
        else:
            self.length = len(clean_dataset)
        if train_length is not None:
            self.length = train_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == "clean_only":
            return self.clean_dataset[idx % len(self.clean_dataset)]
        if self.mode == "poison_only":
            return self.poison_dataset[idx % len(self.poison_dataset)]
        if torch.rand(1).item() < self.poison_ratio:
            return self.poison_dataset[idx % len(self.poison_dataset)]
        return self.clean_dataset[idx % len(self.clean_dataset)]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate classifiers trained with generated synthetic multimodal features")
    parser.add_argument("--model_path", type=str, default="fed_multimodal/Local/results/local_training/best_model.pt")
    parser.add_argument("--poison_path", "--synthetic_path", dest="poison_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="fed_multimodal/results")
    parser.add_argument("--dataset_dir", type=str, default="fed_multimodal/datasets/ucf101")
    parser.add_argument("--output_dir", type=str, default="fed_multimodal/Local/results/synthetic_classifier_eval/default")
    parser.add_argument("--mode", type=str, default="clean_plus_poison", choices=["clean_only", "poison_only", "clean_plus_poison"])
    parser.add_argument("--poison_ratio", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hid_size", type=int, default=128)
    parser.add_argument("--att", action="store_true")
    parser.add_argument("--att_name", type=str, default="")
    parser.add_argument("--init_from_model", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--train_length", type=int, default=None)
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (a, v, la, lv, y) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            a, v, la, lv, y = a.float().to(device), v.float().to(device), la.long().to(device), lv.long().to(device), y.long().to(device)
            logits, _ = model(a, v, la, lv)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.numel()
            loss_sum += float(loss.item()) * y.numel()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "total": total}


def main():
    args = parse_args()
    dm = UCF101LocalDataManager(args.data_dir, args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    loaders = dm.get_dataloaders()
    poison = torch.load(args.poison_path, map_location="cpu")
    poison_dataset = PoisonTensorDataset(poison)
    mixed_dataset = MixedDataset(loaders["train"].dataset, poison_dataset, args.mode, args.poison_ratio, args.train_length)
    train_loader = DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_mm_fn_padd)

    model = MMActionClassifier(dm.num_classes, dm.audio_feat_dim, dm.video_feat_dim, d_hid=args.hid_size, en_att=args.att, att_name=args.att_name).to(args.device)
    if args.from_scratch:
        args.init_from_model = False
    if args.init_from_model:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        model = MMActionClassifier(
            dm.num_classes,
            dm.audio_feat_dim,
            dm.video_feat_dim,
            d_hid=saved_args.get("hid_size", args.hid_size),
            en_att=saved_args.get("att", args.att),
            att_name=saved_args.get("att_name", args.att_name),
        ).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for batch_idx, (a, v, la, lv, y) in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break
            a, v, la, lv, y = a.float().to(args.device), v.float().to(args.device), la.long().to(args.device), lv.long().to(args.device), y.long().to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(a, v, la, lv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        metrics = {"epoch": epoch, "val": evaluate(model, loaders["val"], args.device), "test": evaluate(model, loaders["test"], args.device)}
        if args.eval_train:
            metrics["train"] = evaluate(model, loaders["train"], args.device)
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, output_dir / "final_model.pt")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"history": history, "poison_meta": poison.get("meta", {})}, f, indent=2)
    print(f"Saved classifier validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
