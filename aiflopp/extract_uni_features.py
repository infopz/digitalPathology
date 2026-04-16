import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from her2_test.feature_extraction import load_model, extract_features


class PatchDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UNI-2 features for each subregion."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to split manifest.csv from split_ihc4bc.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root folder where feature files are saved.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "/work/bolelli_synthetic/reggio_data/model_weights/uni2-h/pytorch_model.bin"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def iter_image_paths(folder: Path, extensions: Iterable[str]) -> list[Path]:
    image_paths: list[Path] = []
    for ext in extensions:
        image_paths.extend(folder.rglob(f"*{ext}"))
    return sorted(image_paths)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, transform = load_model(device=device, model_path=str(args.weights))
    model.to(device)

    print(f"Using device: {device}")

    import pandas as pd

    manifest = pd.read_csv(args.manifest)
    required_cols = {"subregion_id", "subregion_path", "label", "patient_id"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    for _, row in tqdm(manifest.iterrows()):
        # Process each subregion

        subregion_id = row["subregion_id"]
        subregion_path = Path(row["subregion_path"])
        label = int(row["label"])
        patient_id = row["patient_id"]

        patient_dir = args.output_root / str(patient_id)
        patient_dir.mkdir(parents=True, exist_ok=True)
        feature_path = patient_dir / f"{subregion_id.replace('/', '__')}.npz"

        if feature_path.exists():
            print(f"Skipping {subregion_id}: features already exist.")
            continue

        image_paths = iter_image_paths(subregion_path, args.extensions)
        if not image_paths:
            print(f"Skipping {subregion_id}: no images found.")
            continue

        dataset = PatchDataset(image_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            shuffle=False,
        )

        all_features: list[np.ndarray] = []
        all_paths: list[str] = []
        for images, paths in loader:
            feats = extract_features(model=model, device=device, images=images)
            all_features.append(feats.cpu().numpy().astype(np.float32))
            all_paths.extend(paths)

        features = np.concatenate(all_features, axis=0)

        np.savez_compressed(
            feature_path,
            features=features,
            label=label,
            patch_paths=np.array(all_paths),
        )

        meta_path = feature_path.with_suffix(".json")
        meta = {
            "subregion_id": subregion_id,
            "patient_id": patient_id,
            "num_patches": len(all_paths),
            "feature_path": str(feature_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"Wrote {feature_path} ({features.shape})")


if __name__ == "__main__":
    main()
