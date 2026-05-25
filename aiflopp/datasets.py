from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aiflopp.feature_utils import (
    AVAILABLE_FEATURE_MODES,
    BagFeatureData,
    HandcraftedFeatureScaler,
    load_feature_file,
    merge_aligned_features,
)


class MILBagDataset(Dataset):
    """Dataset that returns one bag (subregion) at a time."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        features_root: Path,
        handcrafted_features_root: Path | None = None,
        feature_mode: str = "deep",
        handcrafted_scaler: HandcraftedFeatureScaler | None = None,
        max_bag_size: int = 0,
        enable_sampling: bool = True,
    ):
        
        self.manifest = manifest.reset_index(drop=True)
        self.feature_mode = feature_mode

        self.features_root = features_root
        self.handcrafted_features_root = handcrafted_features_root
        self.handcrafted_scaler = handcrafted_scaler
        
        self.max_bag_size = max_bag_size
        self.enable_sampling = enable_sampling
        self.required_cols = {"bag_id", "label"}

        missing = self.required_cols - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        if self.feature_mode not in AVAILABLE_FEATURE_MODES:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")
        if self.feature_mode in {"handcrafted", "concat"} and self.handcrafted_features_root is None:
            raise ValueError("handcrafted_features_root is required for handcrafted or concat mode.")
        if self.feature_mode in {"handcrafted", "concat"} and self.handcrafted_scaler is None:
            raise ValueError("handcrafted_scaler is required for handcrafted or concat mode.")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_features(self, bag_id: str) -> BagFeatureData:
        """
        Based on the feature_mode,  load deep and/or handcrafted features.
        Return a merged BagFeatureData object
        """

        deep_feature_path = self.features_root / f"{bag_id}.npz"

        if self.feature_mode == "deep":
            bag_data = load_feature_file(deep_feature_path)

        elif self.feature_mode == "handcrafted":
            handcrafted_feature_path = self.handcrafted_features_root / f"{bag_id}.npz"
            bag_data = load_feature_file(handcrafted_feature_path)
            scaled_features = self.handcrafted_scaler.transform(bag_data.features)

            bag_data = BagFeatureData(
                features=scaled_features,
                coords=bag_data.coords,
                wsi_names=bag_data.wsi_names,
            )

        elif self.feature_mode == "concat":

            # Load deep
            deep_data = load_feature_file(deep_feature_path)

            # Load handcrafted and scale features
            handcrafted_feature_path = self.handcrafted_features_root / f"{bag_id}.npz"
            handcrafted_data = load_feature_file(handcrafted_feature_path)
            scaled_handcrafted_features = self.handcrafted_scaler.transform(handcrafted_data.features)
            handcrafted_data = BagFeatureData(
                features=scaled_handcrafted_features,
                coords=handcrafted_data.coords,
                wsi_names=handcrafted_data.wsi_names,
            )

            # Merge and filter
            bag_data = merge_aligned_features(deep_data, handcrafted_data, bag_id)
        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        # Randomly subsample patches after any concat/alignment step so all features stay synchronized.
        if self.enable_sampling and self.max_bag_size > 0 and len(bag_data.features) > self.max_bag_size:
            idx = np.random.choice(len(bag_data.features), size=self.max_bag_size, replace=False)
            bag_data = bag_data.subset(idx)

        return bag_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, dict | None]:
        row = self.manifest.iloc[idx]
        bag_id = row["bag_id"]
        label = float(row["label"])

        bag_data = self._load_features(bag_id)

        return (
            torch.from_numpy(bag_data.features),
            torch.tensor(label, dtype=torch.float32),
            bag_id,
            bag_data.metadata(),
        )


def collate_bags(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str, dict | None]]):
    bags, labels, ids, patch_metadata = zip(*batch)
    return list(bags), torch.stack(labels), list(ids), list(patch_metadata)
