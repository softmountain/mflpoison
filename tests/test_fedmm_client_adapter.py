import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mflpoison.adapters.fedmm.client import FedAvgClientTrainer
from mflpoison.core.types import GlobalSnapshot, ModelSpec


class _TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        value = float(index) / 8.0
        return (
            torch.tensor([[value]]),
            torch.tensor([[1.0 - value]]),
            1,
            1,
            torch.tensor(index % 2),
        )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2, 2)

    def forward(self, audio, video, len_audio, len_video):
        del len_audio, len_video
        features = torch.cat([audio[:, 0], video[:, 0]], dim=1)
        logits = self.linear(self.dropout(features))
        return logits, features


class FedMMClientAdapterTest(unittest.TestCase):
    def test_round_client_rng_is_reproducible_and_isolated(self):
        initial_model = _TinyModel()
        snapshot = GlobalSnapshot(
            state=initial_model.state_dict(),
            round_index=3,
            dev_metrics={},
            model_spec=ModelSpec("tiny"),
            partition_hash="partition",
        )
        loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=True)
        trainer = FedAvgClientTrainer(
            lambda state: self._model_from_state(state),
            learning_rate=0.1,
            local_epochs=2,
            seed=19,
        )

        first = trainer.train("client-a", snapshot, loader, len(loader.dataset))
        torch.manual_seed(999)
        torch.randn(100)
        second = trainer.train("client-a", snapshot, loader, len(loader.dataset))

        for key in first.delta:
            self.assertTrue(torch.equal(first.delta[key], second.delta[key]))

    @staticmethod
    def _model_from_state(state):
        model = _TinyModel()
        model.load_state_dict(state)
        return model


if __name__ == "__main__":
    unittest.main()
