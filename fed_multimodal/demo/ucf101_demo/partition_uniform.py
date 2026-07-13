import random
from collections import defaultdict
from typing import Dict, Iterable, List


def uniform_round_robin_partition(
    keys: Iterable[str],
    key_to_label: Dict[str, int],
    num_clients: int,
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    label_buckets = defaultdict(list)
    for key in keys:
        label_buckets[key_to_label[key]].append(key)

    clients = {str(client_idx): [] for client_idx in range(num_clients)}
    client_order = list(range(num_clients))

    for label in sorted(label_buckets):
        bucket = list(label_buckets[label])
        rng.shuffle(bucket)
        rng.shuffle(client_order)
        for offset, key in enumerate(bucket):
            client_idx = client_order[offset % num_clients]
            clients[str(client_idx)].append(key)

    for client_keys in clients.values():
        client_keys.sort()
    return clients


def summarize_partition(clients: Dict[str, List[str]], key_to_label: Dict[str, int]) -> Dict[str, float]:
    sizes = [len(keys) for keys in clients.values()]
    label_coverages = []
    for keys in clients.values():
        labels = {key_to_label[key] for key in keys}
        label_coverages.append(len(labels))
    return {
        "min_client_size": min(sizes) if sizes else 0,
        "max_client_size": max(sizes) if sizes else 0,
        "avg_client_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "min_label_coverage": min(label_coverages) if label_coverages else 0,
        "max_label_coverage": max(label_coverages) if label_coverages else 0,
        "avg_label_coverage": (sum(label_coverages) / len(label_coverages)) if label_coverages else 0.0,
    }
