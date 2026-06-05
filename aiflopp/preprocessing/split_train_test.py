import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


BAG_ID_PATTERN = re.compile(r"^RE_I_25_(\d+)_\d+_([A-Za-z]+)$")

DEFAULT_FEATURES_DIR = Path("data/features/uni_features_RE_all")
DEFAULT_LABELS_CSV = Path("/home/ubuntu/giodir/digitalPathology/data/labels/all_cad_discordance_labels/binary_diff_labels.csv")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_cad_binary_diff")

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Split available bag feature files into patient-level train/val/test manifests "
			"with label-aware stratification."
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
		help="Directory where train/val/test manifest CSV files are written.",
	)
	parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
	parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
	parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
	if min(train_ratio, val_ratio, test_ratio) <= 0:
		raise ValueError("All split ratios must be > 0.")

	total = train_ratio + val_ratio + test_ratio
	if not np.isclose(total, 1.0, atol=1e-6):
		raise ValueError(
			f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}."
		)


def _parse_bag_id(bag_id: str) -> tuple[str, str]:
	match = BAG_ID_PATTERN.match(bag_id)
	if not match:
		raise ValueError(
			f"Bag id '{bag_id}' does not match expected pattern RE_I_25_<patient_id>_<subregion_id>."
		)
	return match.group(1), match.group(2)


def load_filtered_manifest(labels_csv: Path, features_dir: Path) -> pd.DataFrame:

	# Load the label CSV

	labels_df = pd.read_csv(labels_csv)
	required_cols = {"bag_id", "label"}
	missing = required_cols - set(labels_df.columns)
	if missing:
		raise ValueError(f"Labels CSV missing columns: {missing}")

	if labels_df["bag_id"].duplicated().any():
		dupes = labels_df.loc[labels_df["bag_id"].duplicated(), "bag_id"].head(5).tolist()
		raise ValueError(f"Duplicate bag_id values found in labels CSV. Examples: {dupes}")

	# Remove rows with missing values (for those bag_id without label)
	labels_df = labels_df.dropna(subset=["bag_id", "label"])

	# Check the available files and filter the manifest

	available_bags = {path.stem for path in features_dir.glob("*.npz")}
	if not available_bags:
		raise FileNotFoundError(f"No .npz files found in {features_dir}")

	manifest = labels_df[labels_df["bag_id"].isin(available_bags)].copy()
	if manifest.empty:
		raise ValueError(
			"No overlap between labels CSV bag_id values and available NPZ files in features-dir."
		)

	# Parse the manifest

	ids = manifest["bag_id"].apply(_parse_bag_id) # Extract patient_id and subregion_id
	manifest["patient_id"] = ids.str[0]
	manifest["subregion_id"] = ids.str[1]
	manifest["label"] = manifest["label"].astype(int)

	manifest = manifest[["bag_id", "patient_id", "subregion_id", "label"]]
	return manifest.sort_values("bag_id").reset_index(drop=True)


def _best_group_stratified_holdout(
	data: pd.DataFrame,
	holdout_ratio: float,
	seed: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Find the best group-stratified holdout split close to target holdout_ratio."""

	# TODO: capire un attimo meglio il codice qua sotto, ma funziona

	if data.empty:
		raise ValueError("Cannot split an empty dataframe.")

	n_patients = data["patient_id"].nunique()
	if n_patients < 2:
		raise ValueError("At least 2 patients are required for a grouped holdout split.")

	y = data["label"].to_numpy()
	groups = data["patient_id"].to_numpy()
	global_pos_ratio = float(data["label"].mean())

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
			holdout_pos_ratio = float(data.iloc[holdout_idx]["label"].mean())

			# Prioritize matching target size, then label balance.
			score = abs(ratio - holdout_ratio) + 0.5 * abs(holdout_pos_ratio - global_pos_ratio)
			if score < best_score:
				best_score = score
				best_train_idx = train_idx
				best_holdout_idx = holdout_idx

	if best_train_idx is None or best_holdout_idx is None:
		raise RuntimeError("Unable to compute a valid StratifiedGroupKFold holdout split.")

	return best_train_idx, best_holdout_idx


def find_best_patient_assignment(
	manifest: pd.DataFrame,
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
	seed: int,
) -> dict[str, set[str]]:
	patients = manifest["patient_id"].unique().tolist()

	if len(patients) < 3:
		raise ValueError(
			"At least 3 patients are required to produce train/val/test splits without leakage."
		)

	train_val_idx, test_idx = _best_group_stratified_holdout(
		data=manifest,
		holdout_ratio=test_ratio,
		seed=seed,
	)

	train_val_df = manifest.iloc[train_val_idx].reset_index(drop=True)
	val_share_in_train_val = val_ratio / (train_ratio + val_ratio)

	train_idx_local, val_idx_local = _best_group_stratified_holdout(
		data=train_val_df,
		holdout_ratio=val_share_in_train_val,
		seed=seed + 1,
	)

	train_patients = set(train_val_df.iloc[train_idx_local]["patient_id"].unique().tolist())
	val_patients = set(train_val_df.iloc[val_idx_local]["patient_id"].unique().tolist())
	test_patients = set(manifest.iloc[test_idx]["patient_id"].unique().tolist())

	split_patients = {
		"train": train_patients,
		"val": val_patients,
		"test": test_patients,
	}
	return split_patients


def _write_split_manifests(manifest: pd.DataFrame, split_patients: dict[str, set[str]], output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	for split_name in ["train", "val", "test"]:
		split_df = manifest[manifest["patient_id"].isin(split_patients[split_name])].copy()
		split_df = split_df[["bag_id", "patient_id", "subregion_id", "label"]]
		split_df = split_df.sort_values(["patient_id", "subregion_id"]).reset_index(drop=True)
		split_df.to_csv(output_dir / f"{split_name}_manifest.csv", index=False)


def _print_summary(manifest: pd.DataFrame, split_patients: dict[str, set[str]]) -> None:
	print("\nSplit summary (bags and patient-level leakage check):")
	for split_name in ["train", "val", "test"]:
		split_df = manifest[manifest["patient_id"].isin(split_patients[split_name])]
		n_bags = len(split_df)
		n_patients = split_df["patient_id"].nunique()
		label_dist = split_df["label"].value_counts().to_dict()
		print(
			f"  {split_name}: bags={n_bags}, patients={n_patients}, "
			f"label_dist={label_dist}, patient_ids={sorted(split_patients[split_name])}"
		)

	all_sets = [split_patients["train"], split_patients["val"], split_patients["test"]]
	has_overlap = bool((all_sets[0] & all_sets[1]) or (all_sets[0] & all_sets[2]) or (all_sets[1] & all_sets[2]))
	print(f"\nPatient overlap across splits: {has_overlap}")


def main() -> None:
	args = parse_args()
	validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

	manifest = load_filtered_manifest(args.labels_csv, args.features_dir)
	split_patients = find_best_patient_assignment(
		manifest=manifest,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		seed=args.seed,
	)

	_write_split_manifests(manifest, split_patients, args.output_dir)
	_print_summary(manifest, split_patients)
	print(f"\nWrote manifests to: {args.output_dir}")


if __name__ == "__main__":
	main()
