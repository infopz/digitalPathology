import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


NONFL_CHECKPOINT_ROOT = Path("/home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/trained_models/nonFL-gated_bs512_d05")
OUTPUT_ROOT = Path("/home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/evaluations/nonFL-gated_bs512_d05")
MANIFEST_NAME = "fl_cad_binary_diff_5cv"

MERGED_DATASET_PARENT = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/mergedRT")
CLIENT_DATASET_PARENTS = {
    "reggio": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/reggio_client"),
    "trento": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/trento_client"),
}

FEATURES_ROOT = Path("/home/ubuntu/giodir/digitalPathology/data/features/uni_features_merged_RE_TN")
SKIP_EXISTING = True
METRICS = ["balanced_acc", "precision", "recall", "recall_0", "auc", "f2", "acc"]


def parse_args() -> argparse.Namespace:
    # Parse simple overrides while keeping editable defaults above.
    parser = argparse.ArgumentParser(description="Evaluate non-FL CV checkpoints on merged and client datasets.")
    parser.add_argument("--checkpoint-root", type=Path, default=NONFL_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    return parser.parse_args()


def build_eval_targets(manifest_name: str) -> dict[str, Path]:
    # Resolve merged and client dataset folders from the selected manifest name.
    return {
        "merged": MERGED_DATASET_PARENT / manifest_name,
        **{
            client_name: dataset_parent / manifest_name
            for client_name, dataset_parent in CLIENT_DATASET_PARENTS.items()
        },
    }


def run_cv_evaluation(eval_name: str, folds_dir: Path, checkpoint_root: Path, output_root: Path) -> Path:
    # Call the shared single-CV evaluator for one dataset target.
    output_dir = output_root / eval_name
    command = [
        sys.executable,
        "-m",
        "aiflopp.evaluate_single_cv",
        "--checkpoint-root",
        str(checkpoint_root),
        "--folds-dir",
        str(folds_dir),
        "--output-dir",
        str(output_dir),
        "--features-root",
        str(FEATURES_ROOT),
    ]
    if SKIP_EXISTING:
        command.append("--skip-existing")
    subprocess.run(command, check=True)
    return output_dir


def round_float(value):
    # Round finite values for JSON output.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 4) if math.isfinite(float(value)) else None
    return value


def format_mean_std(mean, std) -> str:
    # Format one metric cell for compact CSV comparison.
    if mean is None or pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def read_fold_summary(output_dir: Path) -> tuple[dict, dict, int]:
    # Read avg/std rows and total tested bags from fold_metrics.csv.
    table = pd.read_csv(output_dir / "fold_metrics.csv")
    avg_row = table.loc[table["fold"] == "avg"]
    std_row = table.loc[table["fold"] == "std"]
    fold_rows = table.loc[~table["fold"].isin(["avg", "std"])]

    avg = avg_row.iloc[0].to_dict() if not avg_row.empty else {}
    std = std_row.iloc[0].to_dict() if not std_row.empty else {}
    num_bags = int(pd.to_numeric(fold_rows["num_bags"], errors="coerce").sum())
    return avg, std, num_bags


def build_comparison_row(model_name: str, eval_dataset: str, output_dir: Path) -> dict:
    # Convert fold summary metrics into one grouped JSON comparison row.
    avg, std, num_bags = read_fold_summary(output_dir)
    return {
        "model": model_name,
        "eval_dataset": eval_dataset,
        "num_bags": num_bags,
        "eval_output_dir": str(output_dir),
        "metrics": {
            metric_name: {
                "mean": round_float(avg.get(metric_name)),
                "std": round_float(std.get(metric_name)),
            }
            for metric_name in METRICS
        },
    }


def build_csv_row(json_row: dict) -> dict:
    # Convert one grouped JSON row into a compact CSV row.
    row = {
        "model": json_row["model"],
        "eval_dataset": json_row["eval_dataset"],
        "num_bags": json_row["num_bags"],
        "eval_output_dir": json_row["eval_output_dir"],
    }
    for metric_name in METRICS:
        metric = json_row["metrics"].get(metric_name, {})
        row[metric_name] = format_mean_std(metric.get("mean"), metric.get("std"))
    return row


def main() -> None:
    # Evaluate centralized CV checkpoints on merged and client-specific datasets.
    args = parse_args()
    eval_targets = build_eval_targets(args.manifest_name)
    json_rows = []
    for eval_name, folds_dir in eval_targets.items():
        output_dir = run_cv_evaluation(eval_name, folds_dir, args.checkpoint_root, args.output_root)
        json_rows.append(build_comparison_row("nonfl", eval_name, output_dir))

    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_root / "nonfl_cv_comparison.csv"
    comparison_json = args.output_root / "nonfl_cv_comparison.json"
    pd.DataFrame([build_csv_row(row) for row in json_rows]).to_csv(comparison_csv, index=False)
    with comparison_json.open("w") as f:
        json.dump(json_rows, f, indent=4)

    print(f"Saved non-FL comparison table to {comparison_csv}")
    print(f"Saved non-FL comparison JSON to {comparison_json}")


if __name__ == "__main__":
    main()
