import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


BAG_ID_PATTERN = re.compile(r"^RE_I_25_(\d+)_\d+_([A-Za-z]+)$")

DEFAULT_FEATURES_DIR = Path("data/uni_features_RE_common")
DEFAULT_LABELS_CSV = Path("data/alice/bag_labels.csv")
DEFAULT_OUTPUT_DIR = Path("aiflopp/manifest_folds")

DEFAULT_TEST_RATIO = 0.2
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create K folds of train/val manifests with a shared patient-level test split "
            "using group-stratified splitting."
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
        default=DEFAULT_LABELS_CSV,
        help="CSV with columns bag_id,label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where fold subdirectories are written.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Target ratio for the shared test split.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of train/val folds over the non-test patients.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def validate_args(test_ratio: float, n_folds: int) -> None:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in (0, 1), got {test_ratio}.")
    if n_folds < 2:
        raise ValueError(f"--n-folds must be >= 2, got {n_folds}.")


def _parse_bag_id(bag_id: str) -> tuple[str, str]:
    match = BAG_ID_PATTERN.match(bag_id)
    if not match:
        raise ValueError(
            f"Bag id '{bag_id}' does not match expected pattern RE_I_25_<patient_id>_<subregion_id>."
        )
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


def _class_distribution(data: pd.DataFrame, label_classes: list[int]) -> np.ndarray:
    counts = data["label"].value_counts().reindex(label_classes, fill_value=0).to_numpy(dtype=float)
    if counts.sum() == 0:
        return np.zeros(len(label_classes), dtype=float)
    return counts / counts.sum()


def _best_group_stratified_holdout(
    data: pd.DataFrame,
    holdout_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best group-stratified holdout split close to target holdout_ratio."""
    if data.empty:
        raise ValueError("Cannot split an empty dataframe.")

    n_patients = data["patient_id"].nunique()
    if n_patients < 2:
        raise ValueError("At least 2 patients are required for a grouped holdout split.")

    y = data["label"].to_numpy()
    groups = data["patient_id"].to_numpy()
    label_classes = sorted(data["label"].unique().tolist())
    global_label_dist = _class_distribution(data, label_classes)

    best_train_idx: np.ndarray | None = None
    best_holdout_idx: np.ndarray | None = None
    best_score = float("inf")

    max_splits = min(10, n_patients)
    for n_splits in range(2, max_splits + 1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        for train_idx, holdout_idx in splitter.split(data, y, groups):
            if len(train_idx) == 0 or len(holdout_idx) == 0:
                continue

            ratio = len(holdout_idx) / len(data)
            holdout_label_dist = _class_distribution(data.iloc[holdout_idx], label_classes)
            label_dist_error = float(np.abs(holdout_label_dist - global_label_dist).sum())

            # Prioritize matching target size, then label balance.
            score = abs(ratio - holdout_ratio) + 0.5 * label_dist_error
            if score < best_score:
                best_score = score
                best_train_idx = train_idx
                best_holdout_idx = holdout_idx

    if best_train_idx is None or best_holdout_idx is None:
        raise RuntimeError("Unable to compute a valid StratifiedGroupKFold holdout split.")

    return best_train_idx, best_holdout_idx


def build_fold_patient_assignments(
    manifest: pd.DataFrame,
    test_ratio: float,
    n_folds: int,
    seed: int,
) -> list[dict[str, set[str]]]:
    n_patients = manifest["patient_id"].nunique()
    if n_patients < n_folds + 1:
        raise ValueError(
            f"Need at least n_folds + 1 patients for shared-test CV: got patients={n_patients}, n_folds={n_folds}."
        )

    train_val_idx, test_idx = _best_group_stratified_holdout(
        data=manifest,
        holdout_ratio=test_ratio,
        seed=seed,
    )

    train_val_df = manifest.iloc[train_val_idx].reset_index(drop=True)
    test_patients = set(manifest.iloc[test_idx]["patient_id"].unique().tolist())

    n_train_val_patients = train_val_df["patient_id"].nunique()
    if n_train_val_patients < n_folds:
        raise ValueError(
            f"Not enough non-test patients for {n_folds} folds: have {n_train_val_patients}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed + 1,
    )

    y = train_val_df["label"].to_numpy()
    groups = train_val_df["patient_id"].to_numpy()

    split_patients_list: list[dict[str, set[str]]] = []
    try:
        for train_idx, val_idx in splitter.split(train_val_df, y, groups):
            train_patients = set(train_val_df.iloc[train_idx]["patient_id"].unique().tolist())
            val_patients = set(train_val_df.iloc[val_idx]["patient_id"].unique().tolist())

            if train_patients & val_patients:
                raise RuntimeError("Leakage detected between train and val patients within a fold.")
            if train_patients & test_patients:
                raise RuntimeError("Leakage detected between train and test patients.")
            if val_patients & test_patients:
                raise RuntimeError("Leakage detected between val and test patients.")

            split_patients_list.append(
                {
                    "train": train_patients,
                    "val": val_patients,
                    "test": test_patients,
                }
            )
    except ValueError as exc:
        raise ValueError(
            "StratifiedGroupKFold could not build the requested folds. "
            "Try lowering --n-folds or adjusting --test-ratio."
        ) from exc

    return split_patients_list


def _manifest_for_patients(manifest: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    split_df = manifest[manifest["patient_id"].isin(patient_ids)].copy()
    split_df = split_df[["bag_id", "patient_id", "subregion_id", "label"]]
    split_df = split_df.sort_values(["patient_id", "subregion_id"]).reset_index(drop=True)
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
        label_dist = split_df["label"].value_counts().to_dict()
        print(
            f"  {split_name}: bags={n_bags}, patients={n_patients}, "
            f"label_dist={label_dist}, patient_ids={sorted(split_patients[split_name])}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args.test_ratio, args.n_folds)

    manifest = load_filtered_manifest(args.labels_csv, args.features_dir)
    fold_assignments = build_fold_patient_assignments(
        manifest=manifest,
        test_ratio=args.test_ratio,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, split_patients in enumerate(fold_assignments, start=1):
        fold_dir = args.output_dir / f"fold_{fold_idx:02d}"
        _write_fold_manifests(manifest, split_patients, fold_dir)
        _print_fold_summary(manifest, split_patients, fold_name=f"fold_{fold_idx:02d}")

    print(f"\nWrote {len(fold_assignments)} folds to: {args.output_dir}")


if __name__ == "__main__":
    main()
