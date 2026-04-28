from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MILBagDataset(Dataset):
    """Dataset that returns one bag (subregion) at a time."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        features_root: Path,
        max_bag_size: int = 0,
        enable_sampling: bool = True,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.features_root = features_root
        self.max_bag_size = max_bag_size
        self.enable_sampling = enable_sampling
        self.required_cols = {"bag_id", "label"}

        missing = self.required_cols - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_features(self, feature_path: Path) -> np.ndarray:
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features: {feature_path}")
        
        data = np.load(feature_path, allow_pickle=True)
        feats: np.ndarray = data["features"].astype(np.float32)

        # Randomly subsample patches
        if self.enable_sampling and self.max_bag_size > 0 and len(feats) > self.max_bag_size:
            idx = np.random.choice(len(feats), size=self.max_bag_size, replace=False)
            feats = feats[idx]
        
        return feats # (num_patches, feature_dim)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.manifest.iloc[idx]
        bag_id = row["bag_id"]
        label = float(row["label"])

        feature_path = self.features_root / f"{bag_id}.npz"
        feats = self._load_features(feature_path)

        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.float32), bag_id


def collate_bags(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    bags, labels, ids = zip(*batch)
    return list(bags), torch.stack(labels), list(ids)
