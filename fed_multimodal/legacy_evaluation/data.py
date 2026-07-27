import copy
import warnings
from types import SimpleNamespace

import numpy as np

from fed_multimodal.constants import constants
from fed_multimodal.dataloader.dataload_manager import DataloadManager


def _validate_pairs(audio, video, partition_name):
    if len(audio) != len(video):
        raise ValueError(
            f"audio/video length mismatch for partition {partition_name}"
        )
    for index, (audio_item, video_item) in enumerate(zip(audio, video)):
        if str(audio_item[0]) != str(video_item[0]):
            raise ValueError(
                f"audio/video key mismatch at {partition_name}[{index}]"
            )
        if int(audio_item[-2]) != int(video_item[-2]):
            raise ValueError(
                f"audio/video label mismatch at {partition_name}[{index}]"
            )


class UCF101EvaluationData:
    """Read one UCF101 client/dev/test partition for legacy evaluation.

    This class deliberately has no full-training-set view. A train-side
    evaluation must name one client explicitly so the original partition
    boundary remains visible.
    """

    audio_feat_dim = constants.feature_len_dict["mfcc"]
    video_feat_dim = constants.feature_len_dict["mobilenet_v2"]
    audio_seq_len = 500
    video_seq_len = 9
    num_classes = constants.num_class_dict["ucf101"]

    def __init__(
        self,
        data_dir,
        dataset_dir=None,
        audio_feat="mfcc",
        video_feat="mobilenet_v2",
        split_idx=1,
        batch_size=32,
        num_workers=0,
        alpha=1.0,
        fold=None,
        client_id=None,
    ):
        if dataset_dir is not None:
            warnings.warn(
                "dataset_dir is ignored; legacy evaluation now reads FedMM "
                "client/dev/test partitions",
                DeprecationWarning,
                stacklevel=2,
            )
        self.data_dir = str(data_dir)
        self.alpha = float(alpha)
        self.fold = int(split_idx if fold is None else fold)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.client_id = None if client_id is None else str(client_id)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self._args = SimpleNamespace(
            data_dir=self.data_dir,
            dataset="ucf101",
            audio_feat=str(audio_feat),
            video_feat=str(video_feat),
            alpha=self.alpha,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            num_workers=self.num_workers,
            missing_modality=False,
            missing_modailty_rate=0.0,
            missing_label=False,
            missing_label_rate=0.0,
            label_nosiy=False,
            label_nosiy_level=0.0,
        )
        self.manager = DataloadManager(self._args)
        self.manager.get_client_ids(fold_idx=self.fold)
        self.partition_ids = tuple(self.manager.client_ids)

    @property
    def client_ids(self):
        return tuple(
            partition_id
            for partition_id in self.partition_ids
            if partition_id not in {"dev", "test"}
        )

    def get_loader(self, partition="test", client_id=None, shuffle=False):
        partition = str(partition)
        if partition == "client":
            selected_client = self.client_id if client_id is None else str(client_id)
            if selected_client is None:
                raise ValueError("client partition requires an explicit client_id")
            partition_id = selected_client
            if partition_id in {"dev", "test"}:
                raise ValueError("client_id cannot name dev or test")
        elif partition in {"dev", "test"}:
            if client_id is not None:
                raise ValueError("client_id is only valid for the client partition")
            partition_id = partition
        else:
            raise ValueError("partition must be dev, test, or client")
        if partition_id not in self.partition_ids:
            raise KeyError(
                f"partition {partition_id!r} is unavailable for "
                f"alpha={self.alpha}, fold={self.fold}"
            )

        audio = self.manager.load_audio_feat(partition_id, self.fold)
        video = self.manager.load_video_feat(partition_id, self.fold)
        _validate_pairs(audio, video, partition_id)
        loader = self.manager.set_dataloader(
            copy.deepcopy(audio),
            copy.deepcopy(video),
            default_feat_shape_a=np.array(
                [self.audio_seq_len, self.audio_feat_dim]
            ),
            default_feat_shape_b=np.array(
                [self.video_seq_len, self.video_feat_dim]
            ),
            shuffle=bool(shuffle),
        )
        if loader is None:
            raise ValueError(f"partition {partition_id!r} is empty")
        return loader

    def get_dataloaders(self, val_split=0.1):
        """Compatibility surface without the former centralized full_train."""
        del val_split
        loaders = {
            "val": self.get_loader("dev"),
            "dev": self.get_loader("dev"),
            "test": self.get_loader("test"),
        }
        if self.client_id is not None:
            client_loader = self.get_loader("client", self.client_id)
            loaders["client"] = client_loader
            loaders["train"] = client_loader
        return loaders


class UCF101LocalDataManager(UCF101EvaluationData):
    """Deprecated import name retained for old external evaluator scripts."""


def add_partition_arguments(parser, allow_client=True):
    parser.add_argument("--data_dir", default="fed_multimodal/results")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Deprecated and ignored; partition pickles are read from --data_dir",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fold", "--split_idx", dest="fold", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    if allow_client:
        parser.add_argument("--partition", choices=("test", "dev", "client"), default="test")
        parser.add_argument("--client-id", "--client_id", dest="client_id")
        parser.add_argument(
            "--use_train",
            action="store_true",
            help="Deprecated; requires --client-id and selects that client partition",
        )


def evaluation_loader_from_args(args):
    partition = args.partition
    if args.use_train:
        if args.client_id is None:
            raise ValueError("--use_train now requires an explicit --client-id")
        if partition not in {"test", "client"}:
            raise ValueError("--use_train conflicts with --partition " + partition)
        warnings.warn(
            "--use_train is deprecated; use --partition client --client-id ID",
            DeprecationWarning,
            stacklevel=2,
        )
        partition = "client"
    elif partition == "client" and args.client_id is None:
        raise ValueError("--partition client requires --client-id")
    elif args.client_id is not None and partition != "client":
        raise ValueError("--client-id requires --partition client")

    data = UCF101EvaluationData(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        audio_feat=getattr(args, "audio_feat", "mfcc"),
        video_feat=getattr(args, "video_feat", "mobilenet_v2"),
        alpha=args.alpha,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        client_id=args.client_id,
    )
    data.selected_partition = partition
    return data, data.get_loader(partition, client_id=args.client_id)
