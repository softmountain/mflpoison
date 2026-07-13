import hashlib
from collections import Counter

import numpy as np
import torch

from fed_multimodal.trainers.evaluation import EvalMetric


def _stable_int(value) -> int:
    digest = hashlib.md5(str(value).encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


class ClientRealSameGroupLabelShift:
    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        attack_label_ids=None,
        same_group_target_map=None,
        same_group_candidate_map=None,
        client_id=None,
        round_idx=0,
        attack_seed=None,
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = False
        seed = int(getattr(args, 'seed', 42) if attack_seed is None else attack_seed)
        seed += int(round_idx or 0) * 1009 + _stable_int(client_id or 'unknown')
        self.rng = np.random.default_rng(seed)
        self.attack_label_ids = {int(x) for x in (attack_label_ids or [])}
        self.same_group_target_map = {int(k): int(v) for k, v in (same_group_target_map or {}).items()}
        self.same_group_candidate_map = {
            int(k): [int(candidate) for candidate in candidates]
            for k, candidates in (same_group_candidate_map or {}).items()
        }
        self.attack_trace = Counter()
        self.client_id = client_id

    def get_parameters(self):
        return self.model.state_dict()

    def _sample_same_group_target(self, source_label: int):
        candidates = [
            candidate
            for candidate in self.same_group_candidate_map.get(int(source_label), [])
            if candidate != int(source_label)
        ]
        if not candidates:
            target = self.same_group_target_map.get(int(source_label))
            if target is None or int(target) == int(source_label):
                return None
            return int(target)
        return int(self.rng.choice(candidates))

    def _apply_real_same_group_label_shift(self, y):
        train_labels = y.clone()
        for batch_idx, label in enumerate(y.tolist()):
            source_label = int(label)
            if source_label not in self.attack_label_ids:
                continue
            target_label = self._sample_same_group_target(source_label)
            if target_label is None:
                continue
            train_labels[batch_idx] = int(target_label)
            self.attack_trace[f'{source_label}->{target_label}'] += 1
        return train_labels

    def update_weights(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-5)
        for _ in range(int(self.args.local_epochs)):
            for batch_data in self.dataloader:
                optimizer.zero_grad()
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                train_labels = self._apply_real_same_group_label_shift(y)
                outputs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, train_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                self.eval.append_classification_results(train_labels, outputs, loss)
        self.result = self.eval.classification_summary()
        self.result['attack_trace'] = dict(self.attack_trace)
        self.result['client_id'] = self.client_id
