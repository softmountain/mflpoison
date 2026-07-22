from typing import Iterable, List, Sequence

import numpy as np


def build_client_schedule(
    client_ids: Iterable[str],
    rounds: int,
    sample_rate: float,
    seed: int,
) -> List[Sequence[str]]:
    clients = np.asarray(sorted(str(client_id) for client_id in client_ids))
    if clients.size == 0:
        raise ValueError("cannot sample an empty client population")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not 0.0 < float(sample_rate) <= 1.0:
        raise ValueError("sample_rate must be in (0, 1]")
    count = max(1, int(float(sample_rate) * clients.size))
    rng = np.random.RandomState(int(seed))
    return [
        tuple(str(value) for value in rng.choice(clients, count, replace=False))
        for _ in range(int(rounds))
    ]


def build_client_schedule_count(
    client_ids: Iterable[str],
    rounds: int,
    clients_per_round: int,
    seed: int,
) -> List[Sequence[str]]:
    clients = np.asarray(sorted(str(client_id) for client_id in client_ids))
    count = int(clients_per_round)
    if clients.size == 0:
        raise ValueError("cannot sample an empty client population")
    if rounds < 1 or count < 1 or count > clients.size:
        raise ValueError("invalid round count or clients_per_round")
    rng = np.random.RandomState(int(seed))
    return [
        tuple(str(value) for value in rng.choice(clients, count, replace=False))
        for _ in range(int(rounds))
    ]
