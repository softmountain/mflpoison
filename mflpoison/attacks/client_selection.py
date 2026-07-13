import random
from typing import Iterable, List


def select_malicious_clients(
    client_ids: Iterable[str], count: int, seed: int = 42
) -> List[str]:
    clients = sorted(str(client_id) for client_id in client_ids)
    count = int(count)
    if count < 0 or count > len(clients):
        raise ValueError("malicious client count is outside the client population")
    return sorted(random.Random(int(seed)).sample(clients, count))
