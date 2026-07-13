#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Training Data Loader for UCF101

This module provides data loading utilities for local (non-federated) training
using the original UCF101 train/test splits.
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


def pad_tensor(vec: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad tensor to specified length"""
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    if pad_size[0] > 0:
        return torch.cat([vec, torch.zeros(*pad_size)], dim=0)
    return vec


def collate_mm_fn_padd(batch):
    """Collate function for multimodal data with padding"""
    max_a_len = max(map(lambda x: x[0].shape[0], batch))
    max_b_len = max(map(lambda x: x[1].shape[0], batch))

    x_a, x_b, len_a, len_b, ys = [], [], [], [], []
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))
        ys.append(batch[idx][-1])
    
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, len_a, len_b, ys


class UCF101LocalDataset(Dataset):
    """
    UCF101 Local Dataset for centralized training.
    Loads features from the complete feature.pkl file and splits by train/test list.
    """
    def __init__(
        self,
        audio_data: List,
        video_data: List,
        default_audio_shape: np.ndarray = np.array([500, 80]),
        default_video_shape: np.ndarray = np.array([9, 1280])
    ):
        self.audio_data = audio_data
        self.video_data = video_data
        self.default_audio_shape = default_audio_shape
        self.default_video_shape = default_video_shape
        
        assert len(audio_data) == len(video_data), \
            f"Audio and video data length mismatch: {len(audio_data)} vs {len(video_data)}"
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        # Audio data: [key, path, label, feature]
        audio_item = self.audio_data[idx]
        video_item = self.video_data[idx]
        
        # Extract features and labels
        audio_feat = audio_item[-1]  # feature array
        video_feat = video_item[-1]  # feature array
        label = torch.tensor(audio_item[-2])  # label
        
        # Process audio feature
        if audio_feat is not None:
            if len(audio_feat.shape) == 3:
                audio_feat = audio_feat[0]
            audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
            len_a = len(audio_feat)
        else:
            audio_feat = torch.zeros(self.default_audio_shape, dtype=torch.float32)
            len_a = 0
        
        # Process video feature
        if video_feat is not None:
            if len(video_feat.shape) == 3:
                video_feat = video_feat[0]
            video_feat = torch.tensor(video_feat, dtype=torch.float32)
            len_v = len(video_feat)
        else:
            video_feat = torch.zeros(self.default_video_shape, dtype=torch.float32)
            len_v = 0
        
        return audio_feat, video_feat, len_a, len_v, label


class UCF101LocalDataManager:
    """
    Data Manager for UCF101 Local Training.
    Loads complete feature files and splits based on the classes available in feature files.
    Note: The original experiment uses 51 classes (subset of UCF101), not all 101 classes.
    """
    def __init__(
        self,
        data_dir: str,
        dataset_dir: str,
        audio_feat: str = 'mfcc',
        video_feat: str = 'mobilenet_v2',
        split_idx: int = 1,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.dataset_dir = Path(dataset_dir)
        self.audio_feat = audio_feat
        self.video_feat = video_feat
        self.split_idx = split_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Feature dimensions
        self.audio_feat_dim = 80  # MFCC
        self.video_feat_dim = 1280  # MobileNet V2
        self.audio_seq_len = 500
        self.video_seq_len = 9
        
        # First load features to get available classes (51 classes in this experiment)
        self.audio_dict, self.video_dict = self._load_features()
        
        # Build class index from actual feature files
        self.class_to_idx = self._build_class_index_from_features()
        self.num_classes = len(self.class_to_idx)
        
        # Load train/test splits (filtered by available classes)
        self.train_list = self._load_split_list('train')
        self.test_list = self._load_split_list('test')
        
        print(f"UCF101 Local Data Manager initialized:")
        print(f"  - Number of classes: {self.num_classes} (subset of UCF101)")
        print(f"  - Train samples: {len(self.train_list)}")
        print(f"  - Test samples: {len(self.test_list)}")
    
    def _build_class_index_from_features(self) -> Dict[str, int]:
        """Build class name to index mapping from available features (51 classes)"""
        # Extract all unique classes from feature files
        classes = set()
        for key in self.video_dict.keys():
            class_name = key.split('/')[0]
            classes.add(class_name)
        
        # Sort and create index mapping
        sorted_classes = sorted(classes)
        class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
        return class_to_idx
    
    def _load_split_list(self, split: str) -> List[Tuple[str, int]]:
        """Load train or test split file list with labels, filtered by available classes"""
        split_file = self.dataset_dir / 'ucfTrainTestlist' / f'{split}list0{self.split_idx}.txt'
        file_list = []
        skipped = 0
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Get video path (remove .avi extension)
                video_path = parts[0].replace('.avi', '')  # e.g., ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01
                class_name = video_path.split('/')[0]
                # Only include if class is available in feature files
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    file_list.append((video_path, label))
                else:
                    skipped += 1
        if skipped > 0:
            print(f"  Skipped {skipped} samples from classes not in feature files")
        return file_list
    
    def _load_features(self) -> Tuple[Dict, Dict]:
        """Load complete audio and video features from pkl files"""
        # Load audio features - format is {key: feature_array}
        audio_feat_path = self.data_dir / 'feature' / 'audio' / self.audio_feat / 'ucf101' / 'feature.pkl'
        with open(audio_feat_path, 'rb') as f:
            audio_dict = pickle.load(f)
        
        # Load video features - format is {key: feature_array}
        video_feat_path = self.data_dir / 'feature' / 'video' / self.video_feat / 'ucf101' / 'feature.pkl'
        with open(video_feat_path, 'rb') as f:
            video_dict = pickle.load(f)
        
        print(f"Loaded {len(audio_dict)} audio features and {len(video_dict)} video features")
        return audio_dict, video_dict
    
    def _split_data(
        self, 
        audio_dict: Dict, 
        video_dict: Dict, 
        file_list: List[Tuple[str, int]]
    ) -> Tuple[List, List]:
        """Split data based on file list
        
        Returns:
            audio_data: List of [key, label, feature_array]
            video_data: List of [key, label, feature_array]
        """
        audio_data = []
        video_data = []
        missing_count = 0
        
        for video_key, label in file_list:
            if video_key in audio_dict and video_key in video_dict:
                # Format as [key, path, label, feature]
                audio_data.append([video_key, video_key, label, audio_dict[video_key]])
                video_data.append([video_key, video_key, label, video_dict[video_key]])
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count}/{len(file_list)} samples not found in feature files")
        
        return audio_data, video_data
    
    def get_dataloaders(self, val_split: float = 0.1) -> Dict[str, DataLoader]:
        """
        Get train, validation, and test dataloaders.
        
        Args:
            val_split: Fraction of training data to use for validation
            
        Returns:
            Dictionary with 'train', 'val', and 'test' dataloaders
        """
        # Use already loaded features
        audio_dict = self.audio_dict
        video_dict = self.video_dict
        
        # Split by train/test lists
        train_audio, train_video = self._split_data(audio_dict, video_dict, self.train_list)
        test_audio, test_video = self._split_data(audio_dict, video_dict, self.test_list)
        
        print(f"Train data: {len(train_audio)} samples")
        print(f"Test data: {len(test_audio)} samples")
        
        # Create validation split from training data
        val_size = int(len(train_audio) * val_split)
        indices = np.random.permutation(len(train_audio))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Split train into train and val
        final_train_audio = [train_audio[i] for i in train_indices]
        final_train_video = [train_video[i] for i in train_indices]
        val_audio = [train_audio[i] for i in val_indices]
        val_video = [train_video[i] for i in val_indices]
        
        print(f"After val split - Train: {len(final_train_audio)}, Val: {len(val_audio)}")
        
        # Default shapes
        default_audio_shape = np.array([self.audio_seq_len, self.audio_feat_dim])
        default_video_shape = np.array([self.video_seq_len, self.video_feat_dim])
        
        # Create datasets
        train_dataset = UCF101LocalDataset(
            final_train_audio, final_train_video,
            default_audio_shape, default_video_shape
        )
        val_dataset = UCF101LocalDataset(
            val_audio, val_video,
            default_audio_shape, default_video_shape
        )
        test_dataset = UCF101LocalDataset(
            test_audio, test_video,
            default_audio_shape, default_video_shape
        )
        
        # Also create a full training dataset (without val split) for GAN training
        full_train_dataset = UCF101LocalDataset(
            train_audio, train_video,
            default_audio_shape, default_video_shape
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_mm_fn_padd,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_mm_fn_padd,
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_mm_fn_padd,
                pin_memory=True
            ),
            'full_train': DataLoader(
                full_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_mm_fn_padd,
                pin_memory=True
            )
        }
        
        return dataloaders


if __name__ == '__main__':
    # Test the data loader
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    
    data_dir = Path(__file__).parents[1] / 'results'
    dataset_dir = Path(__file__).parents[1] / 'datasets' / 'ucf101'
    
    dm = UCF101LocalDataManager(
        data_dir=str(data_dir),
        dataset_dir=str(dataset_dir),
        batch_size=32
    )
    
    dataloaders = dm.get_dataloaders()
    
    # Test loading a batch
    for split, loader in dataloaders.items():
        if split == 'full_train':
            continue
        batch = next(iter(loader))
        x_a, x_v, len_a, len_v, y = batch
        print(f"{split}: audio={x_a.shape}, video={x_v.shape}, labels={y.shape}")
