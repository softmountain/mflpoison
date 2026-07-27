import torch

from .evaluation import EvalMetric
from .optimizer import FedProxOptimizer


class ClientFedAvg:
    """UCF101 multimodal local trainer retained behind the typed adapter."""

    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None,
    ):
        del label_dict, num_class
        if args.dataset != "ucf101":
            raise ValueError("ClientFedAvg only supports ucf101")
        if args.modality != "multimodal":
            raise ValueError("ClientFedAvg only supports multimodal UCF101 batches")
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result

    def update_weights(self):
        self.model.train()
        evaluator = EvalMetric(multilabel=False)
        if self.args.fed_alg in {"fed_avg", "fed_opt"}:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu,
            )

        for _ in range(int(self.args.local_epochs)):
            for audio, video, audio_lengths, video_lengths, labels in self.dataloader:
                optimizer.zero_grad()
                audio = audio.float().to(self.device)
                video = video.float().to(self.device)
                audio_lengths = audio_lengths.to(self.device)
                video_lengths = video_lengths.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(
                    audio, video, audio_lengths, video_lengths
                )
                outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                evaluator.append_classification_results(labels, outputs, loss)
        self.result = evaluator.classification_summary()
