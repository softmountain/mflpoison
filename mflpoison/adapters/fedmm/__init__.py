"""FedMultimodal adapters used by the UCF101 poisoning pipeline."""

from .client import FedAvgClientTrainer
from .generator import FedMMGeneratorTrainer
from .ucf101 import ClientDataBundle, UCF101FedMMAdapter

__all__ = [
    "ClientDataBundle",
    "FedAvgClientTrainer",
    "FedMMGeneratorTrainer",
    "UCF101FedMMAdapter",
]
