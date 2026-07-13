import hashlib
from collections import Counter

import numpy as np
import torch

from fed_multimodal.trainers.evaluation import EvalMetric

from .gan_source import generate_fake_multimodal_batch


def _stable_int(value) -> int:
    digest = hashlib.md5(str(value).encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


class ClientRobustnessShift:
    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        gan,
        mode='gan_same_group_shift',
        attack_label_ids=None,
        same_group_target_map=None,
        cross_group_target_map=None,
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
        self.gan = gan
        self.mode = mode
        self.multilabel = False
        seed = int(getattr(args, 'seed', 42) if attack_seed is None else attack_seed)
        seed += int(round_idx or 0) * 1009 + _stable_int(client_id or 'unknown')
        self.rng = np.random.default_rng(seed)
        self.attack_label_ids = {int(x) for x in (attack_label_ids or [])}
        self.same_group_target_map = {int(k): int(v) for k, v in (same_group_target_map or {}).items()}
        self.cross_group_target_map = {int(k): int(v) for k, v in (cross_group_target_map or {}).items()}
        self.same_group_candidate_map = {
            int(k): [int(candidate) for candidate in candidates]
            for k, candidates in (same_group_candidate_map or {}).items()
        }
        self.attack_trace = Counter()
        self.client_id = client_id

    def get_parameters(self):
        return self.model.state_dict()

    def _fit_generated_tensor(self, generated: torch.Tensor, target_len: int):
        current_len = generated.shape[1]
        if current_len > target_len:
            fitted = generated[:, :target_len, :]
        elif current_len < target_len:
            pad_shape = (generated.shape[0], target_len - current_len, generated.shape[2])
            pad = torch.zeros(pad_shape, device=generated.device, dtype=generated.dtype)
            fitted = torch.cat([generated, pad], dim=1)
        else:
            fitted = generated
        valid_len = min(current_len, target_len)
        return fitted, valid_len

    def _sample_same_group_target(self, source_label: int):
        candidates = [candidate for candidate in self.same_group_candidate_map.get(int(source_label), []) if candidate != int(source_label)]
        if not candidates:
            target = self.same_group_target_map.get(int(source_label))
            if target is None or int(target) == int(source_label):
                return None
            return int(target)
        return int(self.rng.choice(candidates))

    def _apply_same_group_shift(self, x_a, x_b, y, l_a, l_b):
        train_labels = y.clone()
        attacked_indices = []
        target_labels = []
        for batch_idx, label in enumerate(y.tolist()):
            source_label = int(label)
            if source_label not in self.attack_label_ids:
                continue
            target_label = self._sample_same_group_target(source_label)
            if target_label is None:
                continue
            attacked_indices.append(batch_idx)
            target_labels.append(target_label)
            self.attack_trace[f'{source_label}->{target_label}'] += 1

        if not attacked_indices:
            return x_a, x_b, l_a, l_b, train_labels

        index_tensor = torch.tensor(attacked_indices, device=self.device, dtype=torch.long)
        shifted_labels = torch.tensor(target_labels, device=self.device, dtype=torch.long)
        fake_a, fake_b = generate_fake_multimodal_batch(
            self.gan,
            shifted_labels,
            l_a[index_tensor],
            l_b[index_tensor],
            self.device,
        )
        x_a = x_a.clone()
        x_b = x_b.clone()
        l_a = l_a.clone()
        l_b = l_b.clone()
        train_labels = train_labels.clone()
        for local_idx, batch_idx in enumerate(attacked_indices):
            audio_target_len = x_a[batch_idx].shape[0]
            video_target_len = x_b[batch_idx].shape[0]
            fitted_audio, fitted_audio_len = self._fit_generated_tensor(fake_a[local_idx:local_idx + 1], audio_target_len)
            fitted_video, fitted_video_len = self._fit_generated_tensor(fake_b[local_idx:local_idx + 1], video_target_len)
            x_a[batch_idx] = fitted_audio[0]
            x_b[batch_idx] = fitted_video[0]
            l_a[batch_idx] = fitted_audio_len
            l_b[batch_idx] = fitted_video_len
            train_labels[batch_idx] = shifted_labels[local_idx]
        return x_a, x_b, l_a, l_b, train_labels

    def _apply_cross_modal_shift(self, x_a, x_b, y, l_a, l_b):
        attack_mask = torch.tensor(
            [int(label.item()) in self.attack_label_ids and int(label.item()) in self.cross_group_target_map for label in y],
            device=self.device,
            dtype=torch.bool,
        )
        if not attack_mask.any():
            return x_a, x_b, l_a, l_b, y
        shifted_labels = torch.tensor(
            [self.cross_group_target_map[int(label.item())] for label in y[attack_mask]],
            device=self.device,
            dtype=torch.long,
        )
        _, fake_video = generate_fake_multimodal_batch(
            self.gan,
            shifted_labels,
            l_a[attack_mask],
            l_b[attack_mask],
            self.device,
        )
        x_b = x_b.clone()
        l_b = l_b.clone()
        attacked_indices = attack_mask.nonzero(as_tuple=False).flatten().tolist()
        for local_idx, batch_idx in enumerate(attacked_indices):
            source_label = int(y[batch_idx].item())
            target_label = int(shifted_labels[local_idx].item())
            self.attack_trace[f'{source_label}->{target_label}'] += 1
            video_target_len = x_b[batch_idx].shape[0]
            fitted_video, fitted_video_len = self._fit_generated_tensor(fake_video[local_idx:local_idx + 1], video_target_len)
            x_b[batch_idx] = fitted_video[0]
            l_b[batch_idx] = fitted_video_len
        return x_a, x_b, l_a, l_b, y

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
                if self.mode == 'gan_same_group_shift':
                    x_a, x_b, l_a, l_b, train_labels = self._apply_same_group_shift(x_a, x_b, y, l_a, l_b)
                elif self.mode == 'cross_modal_mismatch_eval':
                    x_a, x_b, l_a, l_b, train_labels = self._apply_cross_modal_shift(x_a, x_b, y, l_a, l_b)
                else:
                    train_labels = y
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
