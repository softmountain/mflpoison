import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.features.data_partitioning.partition_manager import (
    PartitionManager,
)
from fed_multimodal.features.feature_processing.feature_manager import FeatureManager
from fed_multimodal.features.simulation_features.simulation_manager import (
    SimulationManager,
)


def _namespace(**overrides):
    values = {
        "dataset": "ucf101",
        "data_dir": ".",
        "output_dir": ".",
        "raw_data_dir": ".",
        "audio_feat": "mfcc",
        "video_feat": "mobilenet_v2",
        "alpha": 1.0,
        "num_clients": 2,
        "batch_size": 2,
        "missing_modality": False,
        "missing_modailty_rate": 0.5,
        "missing_label": False,
        "missing_label_rate": 0.5,
        "label_nosiy": False,
        "label_nosiy_level": 0.2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _record(key, label, feature):
    return [key, f"/raw/{key}", label, np.asarray(feature, dtype=np.float32)]


class DataloadManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.args = _namespace(data_dir=str(self.root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_partition(self, alpha, fold, client_id, audio, video):
        alpha_name = str(alpha).replace(".", "")
        audio_dir = (
            self.root
            / "feature/audio/mfcc/ucf101"
            / f"alpha{alpha_name}"
            / f"fold{fold}"
        )
        video_dir = (
            self.root
            / "feature/video/mobilenet_v2/ucf101"
            / f"alpha{alpha_name}"
            / f"fold{fold}"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        with (audio_dir / f"{client_id}.pkl").open("wb") as stream:
            pickle.dump(audio, stream)
        with (video_dir / f"{client_id}.pkl").open("wb") as stream:
            pickle.dump(video, stream)

    def test_ucf_paths_client_ids_and_pickle_loading(self):
        audio = [_record("sample", 3, np.ones((16, 80)))]
        video = [_record("sample", 3, np.ones((2, 1280)))]
        for client_id in ("2", "10", "dev", "test"):
            self._write_partition(1.0, 1, client_id, audio, video)

        manager = DataloadManager(self.args)
        self.assertEqual(
            manager.get_audio_feat_path(),
            self.root / "feature/audio/mfcc/ucf101",
        )
        self.assertEqual(
            manager.get_video_feat_path(),
            self.root / "feature/video/mobilenet_v2/ucf101",
        )
        manager.get_client_ids(fold_idx=1)
        self.assertEqual(manager.client_ids, ["10", "2", "dev", "test"])
        self.assertEqual(manager.load_audio_feat("2", 1)[0][0], "sample")
        self.assertEqual(manager.load_video_feat("2", 1)[0][-1].shape, (2, 1280))

        alpha_five = _namespace(data_dir=str(self.root), alpha=5.0)
        self._write_partition(5.0, 1, "0", audio, video)
        manager = DataloadManager(alpha_five)
        manager.get_client_ids(fold_idx=1)
        self.assertEqual(manager.client_ids, ["0"])

    def test_non_ucf_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports ucf101"):
            DataloadManager(_namespace(dataset="meld"))

    def test_multimodal_loader_pads_to_batch_max_and_preserves_metadata(self):
        audio = [
            _record("a", 4, np.ones((16, 80))),
            _record("b", 7, np.ones((24, 80)) * 2),
        ]
        video = [
            _record("a", 4, np.ones((2, 1280))),
            _record("b", 7, np.ones((3, 1280)) * 2),
        ]
        manager = DataloadManager(self.args)
        loader = manager.set_dataloader(audio, video, shuffle=False)
        batch = next(iter(loader))

        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(loader.batch_size, 64)
        self.assertEqual(tuple(batch[0].shape), (2, 24, 80))
        self.assertEqual(tuple(batch[1].shape), (2, 3, 1280))
        self.assertTrue(torch.equal(batch[2], torch.tensor([16, 24])))
        self.assertTrue(torch.equal(batch[3], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(batch[4], torch.tensor([4, 7])))

        train_loader = manager.set_dataloader(audio, video, shuffle=True)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertEqual(train_loader.batch_size, 2)

    def test_missing_modality_uses_default_shape_and_empty_data_returns_none(self):
        audio = [["a", "/raw/a", 4, None]]
        video = [_record("a", 4, np.ones((2, 1280)))]
        manager = DataloadManager(self.args)
        loader = manager.set_dataloader(
            audio,
            video,
            default_feat_shape_a=np.array([8, 80]),
            default_feat_shape_b=np.array([3, 1280]),
            shuffle=False,
        )
        x_audio, _, len_audio, len_video, _ = next(iter(loader))
        self.assertEqual(tuple(x_audio.shape), (1, 8, 80))
        self.assertEqual(len_audio.item(), 0)
        self.assertEqual(len_video.item(), 2)
        self.assertIsNone(manager.set_dataloader([], [], shuffle=False))
        self.assertIsNone(
            manager.set_dataloader(
                [["a", "/raw/a", 4, None]],
                [["a", "/raw/a", 4, None]],
                shuffle=False,
            )
        )

    def test_simulation_setting_and_json_pickle_loading(self):
        manager = DataloadManager(self.args)
        manager.get_simulation_setting(alpha=1.0)
        manager.load_sim_dict(fold_idx=1)
        self.assertEqual(manager.setting_str, "")
        self.assertIsNone(manager.sim_data)

        args = _namespace(
            data_dir=str(self.root),
            missing_modality=True,
            label_nosiy=True,
            missing_label=True,
        )
        manager = DataloadManager(args)
        manager.get_simulation_setting(alpha=1.0)
        self.assertEqual(manager.setting_str, "mm05_ln02_ml05_alpha10")
        sim_dir = self.root / "simulation_feature/ucf101/fold1"
        sim_dir.mkdir(parents=True)
        payload = {"0": [["a", "/raw/a", 1, [0, 0, 1, 0]]]}
        with (sim_dir / f"{manager.setting_str}.json").open("w") as stream:
            json.dump(payload, stream)
        manager.load_sim_dict(fold_idx=1)
        self.assertEqual(manager.get_client_sim_dict("0"), payload["0"])

        with (sim_dir / f"{manager.setting_str}.pkl").open("wb") as stream:
            pickle.dump(payload, stream)
        manager.load_sim_dict(fold_idx=1, ext="pkl")
        self.assertEqual(manager.sim_data, payload)


class PartitionManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ucf_file_discovery_labels_split_and_dirichlet_partition(self):
        for label in ("ApplyEyeMakeup", "Archery"):
            audio_dir = self.root / "ucf101/audios" / label
            audio_dir.mkdir(parents=True)
            for index in range(20):
                (audio_dir / f"{index}.wav").touch()
        manager = PartitionManager(
            _namespace(raw_data_dir=str(self.root), num_clients=2, alpha=1.0)
        )
        manager.fetch_filelist()
        manager.fetch_label_dict()
        self.assertEqual(len(manager.file_list), 40)
        self.assertEqual(
            manager.label_dict,
            {"ApplyEyeMakeup": 0, "Archery": 1},
        )

        first_train, first_dev = manager.split_train_dev(list(range(20)), seed=9)
        second_train, second_dev = manager.split_train_dev(list(range(20)), seed=9)
        self.assertEqual((first_train, first_dev), (second_train, second_dev))
        self.assertEqual(len(first_dev), 4)
        self.assertEqual(set(first_train) | set(first_dev), set(range(20)))

        labels = [0] * 20 + [1] * 20
        client_indices = manager.direchlet_partition(
            labels, seed=4, min_sample_size=5
        )
        flattened = [index for client in client_indices for index in client]
        self.assertEqual(sorted(flattened), list(range(40)))
        self.assertGreaterEqual(min(map(len, client_indices)), 5)

    def test_non_ucf_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports ucf101"):
            PartitionManager(_namespace(dataset="mit51"))


class FeatureAndSimulationManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_feature_manager_accepts_namespace_without_feature_type(self):
        manager = FeatureManager(
            argparse.Namespace(dataset="ucf101", output_dir=str(self.root))
        )
        self.assertEqual(manager.args.dataset, "ucf101")

    def test_feature_manager_reads_ucf_partition(self):
        partition_dir = self.root / "partition/ucf101/fold1"
        partition_dir.mkdir(parents=True)
        payload = {"0": [["sample", "/raw/sample.wav", 2]]}
        with (partition_dir / "partition_alpha10.json").open("w") as stream:
            json.dump(payload, stream)
        manager = FeatureManager(
            argparse.Namespace(dataset="ucf101", output_dir=str(self.root))
        )
        self.assertEqual(manager.fetch_partition(1, 1.0), payload)

    def test_simulation_partition_and_51_class_noise_matrix_are_valid(self):
        partition_dir = self.root / "partition/ucf101/fold1"
        partition_dir.mkdir(parents=True)
        payload = {"0": [["sample", "/raw/sample.wav", 2]]}
        with (partition_dir / "partition_alpha10.json").open("w") as stream:
            json.dump(payload, stream)
        manager = SimulationManager(_namespace(output_dir=str(self.root)))
        self.assertEqual(manager.fetch_partition(1, 1.0), payload)

        matrix = manager.label_noise_matrix(seed=13, class_num=51)
        self.assertEqual(matrix.shape, (51, 51))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix >= 0).all())
        self.assertTrue(np.allclose(matrix.sum(axis=1), np.ones(51)))
        self.assertTrue(np.allclose(np.diag(matrix), np.full(51, 0.8)))
        self.assertTrue(
            np.array_equal(matrix, manager.label_noise_matrix(seed=13, class_num=51))
        )

    def test_preprocessing_managers_reject_non_ucf_dataset(self):
        for manager_type in (FeatureManager, SimulationManager):
            with self.subTest(manager=manager_type.__name__):
                with self.assertRaisesRegex(ValueError, "only supports ucf101"):
                    manager_type(_namespace(dataset="crema_d"))


if __name__ == "__main__":
    unittest.main()
