import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BAG_ID_PATTERN = re.compile(r"^RE_I_25_(\d+_\d+)_([A-Za-z0-9]+)$")


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
		default=Path("data/uni_features_RE_common"),
		help="Directory containing one NPZ file per bag.",
	)
	parser.add_argument(
		"--labels-csv",
		type=Path,
		default=Path("data/alice/bag_labels.csv"),
		help="CSV with columns bag_id,label.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data"),
		help="Directory where train/val/test manifest CSV files are written.",
	)
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	parser.add_argument(
		"--random-trials",
		type=int,
		default=3000,
		help="Number of random grouped assignments to evaluate.",
	)
	parser.add_argument("--seed", type=int, default=7)
	return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
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


def _load_filtered_manifest(labels_csv: Path, features_dir: Path) -> pd.DataFrame:
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


def _assignment_score(
	stats_by_patient: pd.DataFrame,
	assignment: dict[str, str],
	target_ratios: dict[str, float],
) -> float:
	total_bags = int(stats_by_patient["n_bags"].sum())
	total_pos = int(stats_by_patient["n_pos"].sum())
	global_pos_ratio = total_pos / total_bags

	score = 0.0
	for split_name, split_target in target_ratios.items():
		selected = stats_by_patient[stats_by_patient["patient_id"].map(assignment.get) == split_name]
		if selected.empty:
			return 1e9

		split_bags = int(selected["n_bags"].sum())
		split_pos = int(selected["n_pos"].sum())
		split_ratio = split_bags / total_bags
		split_pos_ratio = split_pos / split_bags

		score += abs(split_ratio - split_target)
		score += 0.7 * abs(split_pos_ratio - global_pos_ratio)

	return score


def _find_best_patient_assignment(
	manifest: pd.DataFrame,
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
	random_trials: int,
	seed: int,
) -> dict[str, set[str]]:
	stats_by_patient = (
		manifest.groupby("patient_id")
		.agg(n_bags=("bag_id", "count"), n_pos=("label", "sum"))
		.reset_index()
	)
	patients = stats_by_patient["patient_id"].tolist()

	if len(patients) < 3:
		raise ValueError(
			"At least 3 patients are required to produce train/val/test splits without leakage."
		)

	splits = ["train", "val", "test"]
	split_probs = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
	split_probs /= split_probs.sum()
	target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

	rng = np.random.default_rng(seed)
	best_score = float("inf")
	best_assignment: dict[str, str] | None = None

	for _ in range(random_trials):
		shuffled = patients.copy()
		rng.shuffle(shuffled)

		assignment: dict[str, str] = {}
		for patient_id, split_name in zip(shuffled[:3], splits):
			assignment[patient_id] = split_name

		for patient_id in shuffled[3:]:
			assignment[patient_id] = str(rng.choice(splits, p=split_probs))

		score = _assignment_score(stats_by_patient, assignment, target_ratios)
		if score < best_score:
			best_score = score
			best_assignment = assignment

	if best_assignment is None:
		raise RuntimeError("Failed to find a valid grouped split assignment.")

	split_patients = {
		split_name: {pid for pid, s in best_assignment.items() if s == split_name}
		for split_name in splits
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
	_validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

	manifest = _load_filtered_manifest(args.labels_csv, args.features_dir)
	split_patients = _find_best_patient_assignment(
		manifest=manifest,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		random_trials=args.random_trials,
		seed=args.seed,
	)

	_write_split_manifests(manifest, split_patients, args.output_dir)
	_print_summary(manifest, split_patients)
	print(f"\nWrote manifests to: {args.output_dir}")


if __name__ == "__main__":
	main()
