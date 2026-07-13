"""
UCI-HAR 联邦实验的标签翻转攻击实现

本模块提供标签翻转攻击的实现，用于模拟联邦学习中的恶意客户端行为。
"""
from __future__ import annotations

import copy
import random
from typing import List, Tuple, Optional, Sequence, Union


class UCILabelFlipAttack:
    """Configurable label-flipping attack tailored to UCI-HAR pipelines.

    By default the attack mutates WALKING_UPSTAIRS (1) to WALKING_DOWNSTAIRS (2).
    Only clients whose id ends with ``"-1"`` are poisoned in the training split,
    while the dev split receives label flips on a fixed ratio of its samples to
    mimic the same proportion of poisoned data (20% by default). Test data is
    never altered. All behaviour can be overridden via constructor arguments.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        seed: int = 42,
        src_label: int = 1,
        dst_label: int = 2,
        target_client_suffix: Union[str, Sequence[str], None] = '-1',
        dev_attack_ratio: float = 0.2,
    ) -> None:
        self.flip_prob = flip_prob
        self.seed = seed
        self.src_label = src_label
        self.dst_label = dst_label
        self.target_client_suffixes = self._normalize_suffixes(target_client_suffix)
        self.dev_attack_ratio = max(0.0, min(dev_attack_ratio, 1.0))
        self._rng = random.Random(seed)

    def reset_seed(self, new_seed: Optional[int] = None) -> None:
        """Reset the RNG seed to make the attack deterministic per run."""
        if new_seed is None:
            new_seed = self.seed
        self._rng = random.Random(new_seed)

    def apply(
        self,
        client_id: str,
        acc_dict: List[list],
        gyro_dict: List[list],
    ) -> Tuple[List[list], List[list]]:
        """Return attacked copies of the ACC/GYRO modality lists.

        Args:
            client_id: Federated client identifier.
            acc_dict: List with entries ``[..., label, feature]`` for acc data.
            gyro_dict: List with entries ``[..., label, feature]`` for gyro data.
        """
        if acc_dict is None or gyro_dict is None:
            return acc_dict, gyro_dict
        if len(acc_dict) != len(gyro_dict):
            raise ValueError("ACC and GYRO modality lengths must match for attack")
        if client_id == "test":
            return acc_dict, gyro_dict

        if client_id == "dev":
            return self._attack_dev(acc_dict, gyro_dict)

        if not self._should_attack_client(client_id):
            return acc_dict, gyro_dict
        return self._attack_train_client(acc_dict, gyro_dict)

    # ------------------------------------------------------------------
    # internal helpers
    @staticmethod
    def _normalize_suffixes(raw_suffix: Union[str, Sequence[str], None]) -> Optional[List[str]]:
        if raw_suffix is None:
            return None
        if isinstance(raw_suffix, str):
            tokens = [token.strip() for token in raw_suffix.split(',')]
        else:
            tokens = [str(token).strip() for token in raw_suffix]
        suffixes = [token for token in tokens if token]
        return suffixes if suffixes else None

    def _should_attack_client(self, client_id: str) -> bool:
        if client_id in {"dev", "test"}:
            return False
        if not self.target_client_suffixes:
            return True
        return any(client_id.endswith(suffix) for suffix in self.target_client_suffixes)

    def _attack_train_client(
        self,
        acc_dict: List[list],
        gyro_dict: List[list],
    ) -> Tuple[List[list], List[list]]:
        attacked_acc = copy.deepcopy(acc_dict)
        attacked_gyro = copy.deepcopy(gyro_dict)
        for idx in range(len(attacked_acc)):
            label = attacked_acc[idx][-2]
            if label != self.src_label:
                continue
            if self._rng.random() < self.flip_prob:
                attacked_acc[idx][-2] = self.dst_label
                attacked_gyro[idx][-2] = self.dst_label
        return attacked_acc, attacked_gyro

    def _attack_dev(
        self,
        acc_dict: List[list],
        gyro_dict: List[list],
    ) -> Tuple[List[list], List[list]]:
        if self.dev_attack_ratio <= 0:
            return acc_dict, gyro_dict
        candidate_idx = [idx for idx, entry in enumerate(acc_dict) if entry[-2] == self.src_label]
        if not candidate_idx:
            return acc_dict, gyro_dict
        total_len = len(acc_dict)
        num_to_flip = int(round(self.dev_attack_ratio * total_len))
        if num_to_flip <= 0:
            return acc_dict, gyro_dict
        num_to_flip = min(num_to_flip, len(candidate_idx))
        attacked_acc = copy.deepcopy(acc_dict)
        attacked_gyro = copy.deepcopy(gyro_dict)
        flip_idx = self._rng.sample(candidate_idx, num_to_flip)
        for idx in flip_idx:
            attacked_acc[idx][-2] = self.dst_label
            attacked_gyro[idx][-2] = self.dst_label
        return attacked_acc, attacked_gyro
