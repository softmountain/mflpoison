from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AttackSchedule:
    start_round: int = 0
    end_round: Optional[int] = None
    every: int = 1

    def __post_init__(self):
        if self.start_round < 0 or self.every < 1:
            raise ValueError("invalid attack schedule")
        if self.end_round is not None and self.end_round < self.start_round:
            raise ValueError("end_round cannot precede start_round")

    def active(self, round_index: int) -> bool:
        round_index = int(round_index)
        if round_index < self.start_round:
            return False
        if self.end_round is not None and round_index > self.end_round:
            return False
        return (round_index - self.start_round) % self.every == 0
