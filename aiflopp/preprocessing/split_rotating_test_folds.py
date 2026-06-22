import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


RE_BAG_ID = re.compile(r"^RE_I_25_(\d+)_\d+_([A-Za-z]+)$")
TN_BAG_ID = re.compile(r"^TN_(\d+)_(\d+)_\d+$")
BAG_ID_PATTERN = {"RE": RE_BAG_ID, "TN": TN_BAG_ID}

DEFAULT_FEATURES_DIR = Path("data/features/uni_features_merged_RE_TN")
DEFAULT_N_FOLDS = 7
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create rotating-test CV folds with patient-level grouping. "
            "For each output fold, one base fold is test, the next base fold is validation, "
            "and the remaining base folds are training."
        )
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing one NPZ file per bag.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="CSV with columns bag_id,label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where fold subdirectories are written.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of base folds. Must be >= 3.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def validate_args(n_folds: int) -> None:
    if n_folds < 3:
        raise ValueError(f"--n-folds must be >= 3, got {n_folds}.")


def _parse_bag_id(bag_id: str) -> tuple[str, str]:
    match = None
    for prefix, pattern in BAG_ID_PATTERN.items():
        if bag_id.startswith(prefix):
            match = pattern.match(bag_id)
            break
    else:
        raise ValueError(
            f"Bag id '{bag_id}' does not start with a recognized prefix ({', '.join(BAG_ID_PATTERN.keys())})."
        )

    if not match:
        raise ValueError(f"Bag id '{bag_id}' does not match expected pattern.")
    return match.group(1), match.group(2)


def load_filtered_manifest(labels_csv: Path, features_dir: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    required_cols = {"bag_id", "label"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    if labels_df["bag_id"].duplicated().any():
        dupes = labels_df.loc[labels_df["bag_id"].duplicated(), "bag_id"].head(5).tolist()
        raise ValueError(f"Duplicate bag_id values found in labels CSV. Examples: {dupes}")

    labels_df = labels_df.dropna(subset=["bag_id", "label"])

    available_bags = {path.stem for path in features_dir.glob("*.npz")}
    if not available_bags:
        raise FileNotFoundError(f"No .npz files found in {features_dir}")

    manifest = labels_df[labels_df["bag_id"].isin(available_bags)].copy()
    if manifest.empty:
        raise ValueError(
            "No overlap between labels CSV bag_id values and available NPZ files in features-dir."
        )

    ids = manifest["bag_id"].apply(_parse_bag_id)
    manifest["patient_id"] = ids.str[0]
    manifest["subregion_id"] = ids.str[1]
    manifest["label"] = manifest["label"].astype(int)

    manifest = manifest[["bag_id", "patient_id", "subregion_id", "label"]]
    return manifest.sort_values("bag_id").reset_index(drop=True)


def build_base_patient_folds(
    manifest: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[set[str]]:
    n_patients = manifest["patient_id"].nunique()
    if n_patients < n_folds:
        raise ValueError(
            f"Need at least n_folds patients: got patients={n_patients}, n_folds={n_folds}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    y = manifest["label"].to_numpy()
    groups = manifest["patient_id"].to_numpy()

    base_folds = []
    try:
        for _, fold_idx in splitter.split(manifest, y, groups):
            patient_ids = set(manifest.iloc[fold_idx]["patient_id"].unique().tolist())
            base_folds.append(patient_ids)
    except ValueError as exc:
        raise ValueError(
            "StratifiedGroupKFold could not build the requested folds. "
            "Try lowering --n-folds."
        ) from exc

    return base_folds


def build_rotating_patient_assignments(base_folds: list[set[str]]) -> list[dict[str, set[str]]]:
    assignments = []
    all_patients = set().union(*base_folds)

    for fold_idx, test_patients in enumerate(base_folds):
        val_patients = base_folds[(fold_idx + 1) % len(base_folds)]
        train_patients = all_patients - test_patients - val_patients

        if train_patients & val_patients:
            raise RuntimeError("Leakage detected between train and val patients.")
        if train_patients & test_patients:
            raise RuntimeError("Leakage detected between train and test patients.")
        if val_patients & test_patients:
            raise RuntimeError("Leakage detected between val and test patients.")

        assignments.append(
            {
                "train": train_patients,
                "val": val_patients,
                "test": test_patients,
            }
        )

    return assignments


def _manifest_for_patients(manifest: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    split_df = manifest[manifest["patient_id"].isin(patient_ids)].copy()
    split_df = split_df[["bag_id", "patient_id", "subregion_id", "label"]]
    split_df = split_df.sort_values(["patient_id", "subregion_id", "bag_id"]).reset_index(drop=True)
    return split_df


def _write_fold_manifests(
    manifest: pd.DataFrame,
    split_patients: dict[str, set[str]],
    fold_dir: Path,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_df = _manifest_for_patients(manifest, split_patients[split_name])
        split_df.to_csv(fold_dir / f"{split_name}_manifest.csv", index=False)


def _print_fold_summary(
    manifest: pd.DataFrame,
    split_patients: dict[str, set[str]],
    fold_name: str,
) -> None:
    print(f"\n{fold_name} summary:")
    for split_name in ["train", "val", "test"]:
        split_df = _manifest_for_patients(manifest, split_patients[split_name])
        n_bags = len(split_df)
        n_patients = split_df["patient_id"].nunique()
        label_dist = split_df["label"].value_counts().sort_index().to_dict()
        print(
            f"  {split_name}: bags={n_bags}, patients={n_patients}, "
            f"label_dist={label_dist}, patient_ids={sorted(split_patients[split_name])}"
        )


def _write_assignment_csv(
    assignments: list[dict[str, set[str]]],
    output_path: Path,
) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "patient_id"])
        writer.writeheader()
        for fold_idx, split_patients in enumerate(assignments, start=1):
            fold_name = f"fold_{fold_idx:02d}"
            for split_name in ["train", "val", "test"]:
                for patient_id in sorted(split_patients[split_name]):
                    writer.writerow(
                        {
                            "fold": fold_name,
                            "split": split_name,
                            "patient_id": patient_id,
                        }
                    )


def _print_cv_summary(assignments: list[dict[str, set[str]]]) -> None:
    test_counts = {}
    val_counts = {}
    for split_patients in assignments:
        for patient_id in split_patients["test"]:
            test_counts[patient_id] = test_counts.get(patient_id, 0) + 1
        for patient_id in split_patients["val"]:
            val_counts[patient_id] = val_counts.get(patient_id, 0) + 1

    print("\nCross-validation coverage:")
    print(f"  patients tested exactly once: {all(count == 1 for count in test_counts.values())}")
    print(f"  patients validated exactly once: {all(count == 1 for count in val_counts.values())}")


def main() -> None:
    args = parse_args()
    validate_args(args.n_folds)

    manifest = load_filtered_manifest(args.labels_csv, args.features_dir)
    base_folds = build_base_patient_folds(
        manifest=manifest,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    fold_assignments = build_rotating_patient_assignments(base_folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, split_patients in enumerate(fold_assignments, start=1):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = args.output_dir / fold_name
        _write_fold_manifests(manifest, split_patients, fold_dir)
        _print_fold_summary(manifest, split_patients, fold_name=fold_name)

    assignment_path = args.output_dir / "fold_assignments.csv"
    _write_assignment_csv(fold_assignments, assignment_path)
    _print_cv_summary(fold_assignments)
    print(f"\nWrote {len(fold_assignments)} rotating-test folds to: {args.output_dir}")
    print(f"Wrote fold assignments to: {assignment_path}")


if __name__ == "__main__":
    main()
