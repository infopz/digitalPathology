import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create patient-level train/val/test splits and subregion-level labels "
            "from per-patient patch CSVs."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Folder with one CSV per patient (file_image, manual_annot).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root folder for IHC Her2 images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where to write split folders and manifests.",
    )
    parser.add_argument("--patch-path-col", default="file_image")
    parser.add_argument("--label-col", default=" manual_annot")
    parser.add_argument(
        "--patient-index",
        type=int,
        default=0,
        help="Path segment index (relative to images root) to use as patient id.",
    )
    parser.add_argument(
        "--subregion-index",
        type=int,
        default=1,
        help="Path segment index (relative to images root) to use as subregion id.",
    )
    parser.add_argument(
        "--fix-extension",
        action="store_true",
        help="If a patch path doesn't exist, try swapping .png/.jpg/.jpeg.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def _resolve_patch_path(images_root: Path, patch_path: str) -> Path:
    # Some CSVs contain Windows-style separators ("\\") which turn the
    # subregion folder + filename into a single path part on POSIX. Normalize
    # them to forward slashes before building the Path so patient/subregion
    # parsing works consistently.
    patch_path = patch_path.replace("\\", "/")

    patch_path_obj = Path(patch_path)
    if patch_path_obj.is_absolute():
        return patch_path_obj
    return (images_root / patch_path_obj).resolve()


def _relative_parts(images_root: Path, patch_path: Path) -> tuple[str, ...]:
    try:
        rel = patch_path.relative_to(images_root)
    except ValueError:
        rel = patch_path
    return rel.parts


def _swap_extension(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path.with_suffix(".jpg")
    if path.suffix.lower() == ".jpg":
        return path.with_suffix(".png")
    if path.suffix.lower() == ".jpeg":
        return path.with_suffix(".jpg")
    return path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels_dir.glob("*.csv"))
    if not label_files:
        raise FileNotFoundError(f"No CSV files found in {args.labels_dir}")

    frames: list[pd.DataFrame] = []
    for label_file in label_files:
        patient_id = label_file.stem
        df = pd.read_csv(label_file)
        df["patient_id"] = patient_id
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    print(df.columns.tolist())

    if df.empty:
        raise ValueError("No rows found in labels CSVs.")

    def resolve_path(patch_path: str) -> str:
        resolved = _resolve_patch_path(args.images_root, patch_path)
        if args.fix_extension and not resolved.exists():
            candidate = _swap_extension(resolved)
            if candidate.exists():
                resolved = candidate
        return str(resolved)

    df["patch_path_resolved"] = df[args.patch_path_col].apply(resolve_path)
    df["path_parts"] = df["patch_path_resolved"].apply(
        lambda p: _relative_parts(args.images_root, Path(p))
    )

    def extract_ids(parts: tuple[str, ...]) -> tuple[str, str]:
        if len(parts) <= max(args.patient_index, args.subregion_index):
            raise ValueError(
                f"Path {parts} does not have enough parts for indices "
                f"{args.patient_index}, {args.subregion_index}."
            )
        return parts[args.patient_index], parts[args.subregion_index]

    df[["patient_from_path", "subregion_name"]] = df["path_parts"].apply(
        lambda parts: pd.Series(extract_ids(parts))
    )
    df["subregion_id"] = df["patient_from_path"] + "/" + df["subregion_name"]

    grouped = df.groupby("subregion_id")
    subregion_df = grouped.agg(
        patient_id=("patient_from_path", "first"),
        subregion_name=("subregion_name", "first"),
        label=(args.label_col, "max"),
        num_patches=("patch_path_resolved", "count"),
        example_patch=("patch_path_resolved", "first"),
    ).reset_index()

    subregion_df["subregion_path"] = subregion_df["example_patch"].apply(
        lambda p: str(Path(p).parent)
    )

    patients = sorted(subregion_df["patient_id"].unique().tolist())
    random.Random(args.seed).shuffle(patients)

    n_patients = len(patients)
    n_train = int(n_patients * args.train_ratio)
    n_val = int(n_patients * args.val_ratio)
    n_test = n_patients - n_train - n_val

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    split_map = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }

    for split, patient_set in split_map.items():
        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = subregion_df[subregion_df["patient_id"].isin(patient_set)].copy()
        split_df.to_csv(split_dir / "manifest.csv", index=False)

    summary = (
        f"Patients: {n_patients} | train={len(train_patients)} | "
        f"val={len(val_patients)} | test={len(test_patients)}"
    )
    print(summary)


if __name__ == "__main__":
    main()
