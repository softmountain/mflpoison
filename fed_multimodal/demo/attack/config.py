from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RobustnessConfig:
    exp_name: str = 'robustness_eval'
    fold_idx: int = 1
    num_epochs: int = 100
    eval_interval: int = 10
    clients_per_round: int = 5
    malicious_clients: int = 5
    clean_clients_per_round: int = 5
    stressed_clients_per_round: int = 5
    batch_size: int = 16
    learning_rate: float = 0.05
    global_learning_rate: float = 0.01
    local_epochs: int = 1
    hid_size: int = 128
    att: bool = True
    att_name: str = 'fuse_base'
    fed_alg: str = 'fed_avg'
    seed: int = 42
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    attack_output_dir: Optional[Path] = None
    attack_group_spec_path: Optional[Path] = None
    gan_checkpoint: Optional[Path] = None
    teacher_checkpoint: Optional[Path] = None


def resolve_robustness_config(**overrides) -> RobustnessConfig:
    normalized = {key: value for key, value in overrides.items() if value is not None}
    for key in ['data_dir', 'output_dir', 'attack_output_dir', 'attack_group_spec_path', 'gan_checkpoint', 'teacher_checkpoint']:
        if key in normalized:
            normalized[key] = Path(normalized[key])
    return RobustnessConfig(**normalized)
