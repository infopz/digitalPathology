import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


FL_RUNS_ROOT = Path("/home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/trained_models/fl")
FL_JOB_PREFIX = "fl_cad_binary_diff_5cv_gated_bs512_d05"
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04", "fold_05"]

EXPORT_CHECKPOINT_ROOT = Path("/home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/evaluations/fl_cv_exported_checkpoints")
EVAL_OUTPUT_ROOT = Path("/home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/evaluations/fl_cv_evaluation")
CONFIG_TEMPLATE = Path("/home/ubuntu/giodir/digitalPathology/flare_exp/configs/base_config_fed.yaml")

MERGED_DATASET = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/mergedRT/fl_cad_binary_diff_5cv")
CLIENTS = {
    "reggio": {
        "site_name": "reggio_client",
        "dataset": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/reggio_client/fl_cad_binary_diff_5cv"),
    },
    "trento": {
        "site_name": "trento_client",
        "dataset": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/trento_client/fl_cad_binary_diff_5cv"),
    },
}

FEATURES_ROOT = Path("/home/ubuntu/giodir/digitalPathology/data/features/uni_features_merged_RE_TN")
EVALUATE_GLOBAL_ON_MERGED = True
GLOBAL_MODEL_NAME = "best_FL_global_model.pt"
LOCAL_ROUND = None
SKIP_EXISTING = True
METRICS = ["balanced_acc", "precision", "recall", "recall_0", "auc", "f2", "acc"]


def parse_args() -> argparse.Namespace:
    # Parse config/path overrides for this FL CV metrics run.
    parser = argparse.ArgumentParser(description="Export and evaluate FL CV checkpoints with fold mean/std metrics.")
    parser.add_argument("--config", type=Path, default=CONFIG_TEMPLATE)
    parser.add_argument("--fl-runs-root", type=Path, default=None)
    parser.add_argument("--job-prefix", default=None)
    parser.add_argument("--export-checkpoint-root", type=Path, default=None)
    parser.add_argument("--eval-output-root", type=Path, default=None)
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    # Load the FL YAML used for training and exported checkpoint configs.
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def manifest_name_from_config(config: dict) -> str | None:
    # Extract the CV manifest folder name, removing a fold suffix if present.
    manifest_set = config.get("manifest_set")
    if manifest_set is None:
        return None
    manifest_path = Path(str(manifest_set))
    if manifest_path.name.startswith("fold_"):
        return str(manifest_path.parent)
    return str(manifest_path)


def configure_from_args(args: argparse.Namespace) -> None:
    # Apply CLI/config values to the simple globals used by this script.
    global CONFIG_TEMPLATE, FL_RUNS_ROOT, FL_JOB_PREFIX, FEATURES_ROOT
    global EXPORT_CHECKPOINT_ROOT, EVAL_OUTPUT_ROOT, MERGED_DATASET

    config = load_yaml_config(args.config)
    CONFIG_TEMPLATE = args.config
    FL_RUNS_ROOT = args.fl_runs_root or Path(config.get("output_dir", FL_RUNS_ROOT))
    FL_JOB_PREFIX = args.job_prefix or str(config.get("job_name", FL_JOB_PREFIX))
    FEATURES_ROOT = Path(config.get("features_root", FEATURES_ROOT))

    if args.export_checkpoint_root is not None:
        EXPORT_CHECKPOINT_ROOT = args.export_checkpoint_root
    if args.eval_output_root is not None:
        EVAL_OUTPUT_ROOT = args.eval_output_root

    manifest_name = manifest_name_from_config(config)
    if manifest_name is not None:
        MERGED_DATASET = MERGED_DATASET.parent / manifest_name
        for client in CLIENTS.values():
            client["dataset"] = client["dataset"].parent / manifest_name


def fl_job_dir(fold_name: str) -> Path:
    # Resolve the NVFlare simulator output folder for one CV fold.
    return FL_RUNS_ROOT / f"{FL_JOB_PREFIX}_{fold_name}"


def load_config_template() -> dict:
    # Load the model config template written beside exported checkpoints.
    config = load_yaml_config(CONFIG_TEMPLATE)
    config.setdefault("feature_mode", "deep")
    config["features_root"] = str(FEATURES_ROOT)
    return config


def write_config(config: dict, output_dir: Path) -> None:
    # Write the evaluator-compatible config.yaml file.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_state_dict(source_path: Path, output_path: Path) -> None:
    # Save a raw PyTorch state dict, unwrapping NVFlare global checkpoints if needed.
    if not source_path.exists():
        raise FileNotFoundError(f"Missing FL model artifact: {source_path}")
    state = torch.load(source_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def latest_local_model(job_dir: Path, site_name: str) -> Path:
    # Pick the configured local round, or the latest available local model round.
    results_dir = job_dir / site_name / "results"
    if LOCAL_ROUND is not None:
        return results_dir / f"round_{LOCAL_ROUND}" / "model.pth"

    round_dirs = [path for path in results_dir.glob("round_*") if path.is_dir() and (path / "model.pth").exists()]
    if not round_dirs:
        raise FileNotFoundError(f"No local round models found in {results_dir}")
    round_dirs = sorted(round_dirs, key=lambda path: int(path.name.removeprefix("round_")))
    return round_dirs[-1] / "model.pth"


def export_fl_checkpoints() -> None:
    # Export global and client-local FL artifacts to the shared evaluator format.
    config = load_config_template()
    for fold_name in FOLDS:
        job_dir = fl_job_dir(fold_name)

        global_output_dir = EXPORT_CHECKPOINT_ROOT / "global" / fold_name
        global_source = job_dir / "server" / "simulate_job" / "app_server" / GLOBAL_MODEL_NAME
        save_state_dict(global_source, global_output_dir / "best_model.pth")
        write_config(config, global_output_dir)

        for client_name, client in CLIENTS.items():
            local_output_dir = EXPORT_CHECKPOINT_ROOT / "local" / client_name / fold_name
            local_source = latest_local_model(job_dir, client["site_name"])
            local_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_source, local_output_dir / "best_model.pth")
            write_config(config, local_output_dir)


def run_cv_evaluation(model_name: str, checkpoint_root: Path, eval_name: str, folds_dir: Path) -> Path:
    # Call the shared single-CV evaluator for one FL model/dataset pair.
    output_dir = EVAL_OUTPUT_ROOT / model_name / eval_name
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
    # Export FL models, evaluate all FL targets, and write a comparison table.
    args = parse_args()
    configure_from_args(args)
    export_fl_checkpoints()

    json_rows = []
    global_checkpoint_root = EXPORT_CHECKPOINT_ROOT / "global"
    for client_name, client in CLIENTS.items():
        output_dir = run_cv_evaluation("fl_global", global_checkpoint_root, client_name, client["dataset"])
        json_rows.append(build_comparison_row("fl_global", client_name, output_dir))

    if EVALUATE_GLOBAL_ON_MERGED:
        output_dir = run_cv_evaluation("fl_global", global_checkpoint_root, "merged", MERGED_DATASET)
        json_rows.append(build_comparison_row("fl_global", "merged", output_dir))

    for client_name, client in CLIENTS.items():
        model_name = f"fl_local_{client_name}"
        checkpoint_root = EXPORT_CHECKPOINT_ROOT / "local" / client_name
        output_dir = run_cv_evaluation(model_name, checkpoint_root, client_name, client["dataset"])
        json_rows.append(build_comparison_row(model_name, client_name, output_dir))

    EVAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_csv = EVAL_OUTPUT_ROOT / "fl_cv_comparison.csv"
    comparison_json = EVAL_OUTPUT_ROOT / "fl_cv_comparison.json"
    pd.DataFrame([build_csv_row(row) for row in json_rows]).to_csv(comparison_csv, index=False)
    with comparison_json.open("w") as f:
        json.dump(json_rows, f, indent=4)

    print(f"Saved FL comparison table to {comparison_csv}")
    print(f"Saved FL comparison JSON to {comparison_json}")


if __name__ == "__main__":
    main()
