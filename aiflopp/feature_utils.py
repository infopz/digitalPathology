from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


AVAILABLE_FEATURE_MODES = ("deep", "handcrafted", "concat")


@dataclass(frozen=True)
class HandcraftedFeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {features.shape}")
        if features.shape[1] != len(self.mean):
            raise ValueError(
                "Handcrafted feature dimension mismatch: "
                f"got {features.shape[1]}, expected {len(self.mean)}"
            )

        transformed = (features - self.mean) / self.scale
        return transformed.astype(np.float32, copy=False)

    def to_metadata(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_metadata(cls, metadata: dict | None):
        if metadata is None:
            return None

        return cls(
            mean=np.asarray(metadata["mean"], dtype=np.float32),
            scale=np.asarray(metadata["scale"], dtype=np.float32),
        )


@dataclass(frozen=True)
class BagFeatureData:
    features: np.ndarray
    coords: np.ndarray | None = None
    wsi_names: np.ndarray | None = None

    def subset(self, indices: np.ndarray) -> "BagFeatureData":
        coords = None if self.coords is None else self.coords[indices]
        wsi_names = None if self.wsi_names is None else self.wsi_names[indices]
        return BagFeatureData(
            features=self.features[indices],
            coords=coords,
            wsi_names=wsi_names,
        )

    def metadata(self) -> dict | None:
        if self.coords is None or self.wsi_names is None:
            return None

        return {
            "coords": self.coords,
            "wsi_names": self.wsi_names,
        }

    def to_patch_df(self, index_name: str) -> pd.DataFrame:
        if self.coords is None or self.wsi_names is None:
            raise KeyError("Patch metadata is required to build the patch dataframe.")

        return pd.DataFrame(
            {
                "wsi_name": self.wsi_names.astype(str),
                "x": self.coords[:, 0].astype(int),
                "y": self.coords[:, 1].astype(int),
                index_name: np.arange(len(self.features), dtype=int),
            }
        )


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    if coords.ndim == 1:
        if len(coords) % 2 != 0:
            raise ValueError(f"Invalid flattened coords shape {coords.shape}")
        coords = coords.reshape(-1, 2)
    elif coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unsupported coords shape {coords.shape}")

    return coords.astype(int)


def load_feature_file(feature_path: Path) -> BagFeatureData:
    """
    Open a .npz feature file and extract features, coords, and wsi_names. Coords are normalized to integer (x,y) pairs.
    """

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing features: {feature_path}")

    with np.load(feature_path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        coords = normalize_coords(np.asarray(data["coords"])) if "coords" in data.files else None
        wsi_names = np.asarray(data["wsi_names"]).astype(str) if "wsi_names" in data.files else None

    # Sanity checks on loaded data
    if (coords is None) != (wsi_names is None):
        raise KeyError(
            f"Feature file must contain both 'coords' and 'wsi_names' or neither: {feature_path}"
        )
    if coords is not None and len(coords) != len(features):
        raise ValueError(
            f"Mismatch between coords and features lengths in {feature_path}: "
            f"coords={len(coords)} features={len(features)}"
        )
    if wsi_names is not None and len(wsi_names) != len(features):
        raise ValueError(
            f"Mismatch between wsi_names and features lengths in {feature_path}: "
            f"wsi_names={len(wsi_names)} features={len(features)}"
        )

    return BagFeatureData(features=features, coords=coords, wsi_names=wsi_names)


def fit_handcrafted_scaler(
    manifest: pd.DataFrame,
    handcrafted_features_root: Path,
) -> HandcraftedFeatureScaler:
    scaler = StandardScaler()
    num_bags = 0

    for _, row in manifest.iterrows():
        bag_id = row["bag_id"]
        feature_path = handcrafted_features_root / f"{bag_id}.npz"
        bag_data = load_feature_file(feature_path)

        if len(bag_data.features) == 0:
            raise ValueError(f"Handcrafted features are empty for bag {bag_id}")

        scaler.partial_fit(bag_data.features)
        num_bags += 1

    if num_bags == 0:
        raise RuntimeError("Empty manifest; cannot fit handcrafted scaler.")

    return HandcraftedFeatureScaler(
        mean=np.asarray(scaler.mean_, dtype=np.float32),
        scale=np.asarray(scaler.scale_, dtype=np.float32),
    )


def infer_input_dim(
    manifest: pd.DataFrame,
    feature_mode: str,
    deep_features_root: Path,
    handcrafted_features_root: Path | None,
) -> int:
    """
    Given the selected feature_mode and the manifest, infer the input dimensionality of the features.
    """

    if feature_mode not in AVAILABLE_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    if feature_mode in {"handcrafted", "concat"} and handcrafted_features_root is None:
        raise ValueError("handcrafted_features_root is required for handcrafted or concat mode.")

    for _, row in manifest.iterrows():
        bag_id = row["bag_id"]
        input_dim = 0

        if feature_mode in {"deep", "concat"}:
            deep_data = load_feature_file(deep_features_root / f"{bag_id}.npz")
            input_dim += int(deep_data.features.shape[1])

        if feature_mode in {"handcrafted", "concat"}:
            handcrafted_data = load_feature_file(
                handcrafted_features_root / f"{bag_id}.npz"
            )
            input_dim += int(handcrafted_data.features.shape[1])

        return input_dim

    raise RuntimeError("Empty manifest; cannot infer feature dimensionality.")


def merge_aligned_features(
    deep_data: BagFeatureData,
    handcrafted_data: BagFeatureData,
    bag_id: str,
) -> BagFeatureData:
    """
    Compute matching patch between deep and handcrafted features using their metadata.
    Then merge and return a new BagFeatureData with concatenated features and aligned metadata.
    """

    deep_indices, handcrafted_indices = match_patch_indices(deep_data, handcrafted_data, bag_id)

    aligned_deep = deep_data.subset(deep_indices)
    aligned_handcrafted = handcrafted_data.subset(handcrafted_indices)
    combined_features = np.concatenate(
        [aligned_deep.features, aligned_handcrafted.features],
        axis=1,
    )

    return BagFeatureData(
        features=combined_features.astype(np.float32, copy=False),
        coords=aligned_deep.coords,
        wsi_names=aligned_deep.wsi_names,
    )


def match_patch_indices(
    deep_data: BagFeatureData,
    handcrafted_data: BagFeatureData,
    bag_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the patch metadata (wsi_name, x, y) to find the shared patches.
    Return two index arrays used to filter and align the features in each set.
    """
    
    # Extract patch metadata datafram e from both
    deep_df = deep_data.to_patch_df("deep_idx")
    handcrafted_df = handcrafted_data.to_patch_df("handcrafted_idx")
    key_cols = ["wsi_name", "x", "y"]

    # Check for duplicated patch keys
    if deep_df.duplicated(key_cols).any():
        duplicate_rows = deep_df.loc[deep_df.duplicated(key_cols, keep=False), key_cols]
        raise ValueError(
            f"Duplicate patch keys found in deep features for bag {bag_id}: "
            f"{duplicate_rows.drop_duplicates().to_dict('records')}"
        )
    if handcrafted_df.duplicated(key_cols).any():
        duplicate_rows = handcrafted_df.loc[
            handcrafted_df.duplicated(key_cols, keep=False),
            key_cols,
        ]
        raise ValueError(
            f"Duplicate patch keys found in handcrafted features for bag {bag_id}: "
            f"{duplicate_rows.drop_duplicates().to_dict('records')}"
        )

    # Left merge to keep handcrafted features
    merged_df = handcrafted_df.merge(
        deep_df,
        on=key_cols,
        how="left",
        sort=False,
    )
    # Filter in case of missing deep features (its not our case)
    matched_df = merged_df.dropna(subset=["deep_idx"]).copy()

    if matched_df.empty:
        raise ValueError(f"No overlapping patches between deep and handcrafted features for bag {bag_id}")

    return (
        matched_df["deep_idx"].astype(int).to_numpy(dtype=int),
        matched_df["handcrafted_idx"].astype(int).to_numpy(dtype=int),
    )
