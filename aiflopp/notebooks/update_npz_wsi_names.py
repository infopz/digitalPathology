import argparse
from pathlib import Path

import numpy as np

"""
SCRIPT FATTO PER AGGIORNARE I VECCHI NPZ GENERATI SENZA IL CAMPO wsi_names
"""

def parse_args() -> argparse.Namespace:
    default_features_dir = Path("/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_reamins")
    default_bags_dir = Path("/home/ubuntu/giodir/digitalPathology/data/alice/bag_remains")
    default_output_dir = Path("/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_reamins_w_names")

    parser = argparse.ArgumentParser(
        description="Create updated NPZ feature files with an added wsi_names key."
    )
    parser.add_argument("--features-dir", type=Path, default=default_features_dir)
    parser.add_argument("--bags-dir", type=Path, default=default_bags_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def parse_patch_filename(patch_path: Path) -> tuple[str, int, int]:
    parts = patch_path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid patch filename format: {patch_path.name}")

    try:
        x = int(parts[-2])
        y = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid coordinates in patch filename: {patch_path.name}") from exc

    wsi_name = patch_path.stem[: -len(f"_{x}_{y}")]
    return wsi_name, x, y


def normalize_coords(coords: np.ndarray, npz_path: Path) -> np.ndarray:
    coords = np.asarray(coords)
    if coords.ndim == 1:
        if len(coords) % 2 != 0:
            raise ValueError(f"Invalid flattened coords in {npz_path}")
        coords = coords.reshape(-1, 2)
    elif coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unsupported coords shape {coords.shape} in {npz_path}")
    return coords.astype(np.int64)


def extract_patch_metadata(bag_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    # Keep the same unsorted glob order used during the original feature extraction.
    patch_paths = list(bag_dir.glob("*.png"))
    if not patch_paths:
        raise FileNotFoundError(f"No patch files found in {bag_dir}")

    wsi_names: list[str] = []
    coords: list[tuple[int, int]] = []
    for patch_path in patch_paths:
        wsi_name, x, y = parse_patch_filename(patch_path)
        wsi_names.append(wsi_name)
        coords.append((x, y))

    return np.asarray(wsi_names), np.asarray(coords, dtype=np.int64)


def update_one_npz(npz_path: Path, bags_dir: Path, output_dir: Path) -> None:
    bag_id = npz_path.stem
    bag_dir = bags_dir / bag_id
    if not bag_dir.exists():
        raise FileNotFoundError(f"Missing bag directory for {bag_id}: {bag_dir}")

    patch_wsi_names, patch_coords = extract_patch_metadata(bag_dir)

    with np.load(npz_path, allow_pickle=True) as data:
        if "features" not in data.files or "coords" not in data.files:
            raise KeyError(f"NPZ missing required keys in {npz_path}: found {data.files}")

        payload = {key: data[key] for key in data.files}

    npz_coords = normalize_coords(payload["coords"], npz_path)
    features = np.asarray(payload["features"])

    if len(features) != len(npz_coords):
        raise ValueError(
            f"Features/coords length mismatch in {npz_path}: "
            f"features={len(features)} coords={len(npz_coords)}"
        )
    if len(patch_coords) != len(npz_coords):
        raise ValueError(
            f"Patch/coords length mismatch for {bag_id}: "
            f"patches={len(patch_coords)} coords={len(npz_coords)}"
        )
    if not np.array_equal(patch_coords, npz_coords):
        raise ValueError(f"Patch coordinate order does not match stored coords for {bag_id}")

    payload["coords"] = npz_coords
    payload["wsi_names"] = patch_wsi_names

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / npz_path.name
    np.savez(output_path, **payload)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(args.features_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files found in {args.features_dir}")

    updated = 0
    failures: list[tuple[str, str]] = []

    for npz_path in npz_paths:
        try:
            update_one_npz(npz_path, args.bags_dir, args.output_dir)
            updated += 1
        except Exception as exc:
            failures.append((npz_path.name, str(exc)))

    print(f"Updated NPZ files: {updated}")
    print(f"Failed NPZ files: {len(failures)}")
    for name, error in failures:
        print(f"FAILED {name}: {error}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
