import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(metrics_path: str, save_path: str):
    with open(metrics_path, 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    plt.figure(figsize=(8, 5))
    for mode, values in metrics.items():
        plt.plot(values['rounds'], values['test_acc'], marker='o', label=mode)
    plt.xlabel('Round')
    plt.ylabel('Global Test Accuracy')
    plt.title('Robustness Evaluation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot robustness evaluation curves')
    parser.add_argument('--metrics', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.metrics, args.output)
