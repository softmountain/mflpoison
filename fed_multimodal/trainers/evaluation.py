import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


class EvalMetric:
    """Classification metrics used by the UCF101 adapter and local trainer."""

    def __init__(self, multilabel=False):
        if multilabel:
            raise ValueError("EvalMetric only supports UCF101 classification")
        self.pred_list = []
        self.truth_list = []
        self.top_k_list = []
        self.loss_list = []

    def append_classification_results(self, labels, outputs, loss):
        output_array = outputs.detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        predictions = np.argmax(output_array, axis=1)
        top_k_predictions = np.argsort(output_array, axis=1)[:, ::-1][:, :5]
        self.pred_list.extend(predictions.tolist())
        self.truth_list.extend(label_array.tolist())
        self.top_k_list.extend(top_k_predictions)
        self.loss_list.append(loss.item())

    def classification_summary(self, monitor_labels=None, return_auc=False):
        if return_auc:
            raise ValueError("UCF101 multiclass AUC requires probability scores")
        truth = np.asarray(self.truth_list)
        predictions = np.asarray(self.pred_list)
        top_k = np.asarray(self.top_k_list)
        result = {
            "acc": accuracy_score(truth, predictions) * 100,
            "uar": recall_score(truth, predictions, average="macro") * 100,
            "top5_acc": (
                np.sum(top_k == truth.reshape(-1, 1)) / len(truth)
            )
            * 100,
            "conf": np.round(
                confusion_matrix(truth, predictions, normalize="true") * 100,
                decimals=2,
            ),
            "loss": np.mean(self.loss_list),
            "sample": len(truth),
            "f1": f1_score(truth, predictions, average="macro") * 100,
        }
        if monitor_labels:
            per_label_accuracy = {}
            for label in monitor_labels:
                label = int(label)
                label_mask = truth == label
                per_label_accuracy[str(label)] = (
                    float("nan")
                    if label_mask.sum() == 0
                    else np.mean(predictions[label_mask] == label) * 100
                )
            result["monitored_label_acc"] = per_label_accuracy
        return result

    def classification_detailed_summary(
        self, monitor_labels=None, return_auc=False
    ):
        result = self.classification_summary(
            monitor_labels=monitor_labels,
            return_auc=return_auc,
        )
        truth = np.asarray(self.truth_list, dtype=int)
        predictions = np.asarray(self.pred_list, dtype=int)
        if truth.size == 0:
            result.update(
                {
                    "truth": [],
                    "pred": [],
                    "confusion_count": [],
                    "confusion_row_normalized": [],
                    "label_support": [],
                }
            )
            return result

        label_ids = np.unique(np.concatenate([truth, predictions]))
        counts = confusion_matrix(truth, predictions, labels=label_ids)
        support = counts.sum(axis=1)
        normalized = np.zeros_like(counts, dtype=np.float64)
        nonzero = support > 0
        normalized[nonzero] = counts[nonzero] / support[nonzero, None] * 100.0
        normalized = np.round(normalized, 4)
        result.update(
            {
                "truth": truth.tolist(),
                "pred": predictions.tolist(),
                "confusion_labels": label_ids.astype(int).tolist(),
                "confusion_count": counts.tolist(),
                "confusion_row_normalized": normalized.tolist(),
                "label_support": support.tolist(),
                "label_support_map": {
                    int(label): int(count)
                    for label, count in zip(label_ids, support)
                },
                "confusion_count_map": {
                    int(label): counts[index].tolist()
                    for index, label in enumerate(label_ids)
                },
                "confusion_row_normalized_map": {
                    int(label): normalized[index].tolist()
                    for index, label in enumerate(label_ids)
                },
            }
        )
        return result
