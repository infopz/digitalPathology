import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_DEEP_FEATURES_ROOT = Path("/home/ubuntu/giodir/digitalPathology/data/features/uni_features_merged_RE_TN")
METRIC_PRIORITY = ["balanced_acc", "precision", "recall", "recall_0", "auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with all fold models from a CV training output and aggregate by majority vote."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Directory containing one trained checkpoint subdirectory per fold.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest to predict with every fold model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-fold inference outputs and ensemble predictions are saved.",
    )
    parser.add_argument(
        "--fold-glob",
        default="fold_*",
        help="Glob pattern used to discover fold checkpoint directories inside --checkpoint-root.",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=DEFAULT_DEEP_FEATURES_ROOT,
        help="Optional feature root override passed to infer_mil_attention.",
    )
    parser.add_argument(
        "--handcrafted-features-root",
        type=Path,
        default=None,
        help="Optional handcrafted feature root override passed to infer_mil_attention.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-attention-scores",
        action="store_true",
        help="Save per-fold attention score CSV files. By default they are skipped to save space.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip fold inference runs that already have predictions.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fold inference commands without running them.",
    )
    return parser.parse_args()


def discover_fold_checkpoints(checkpoint_root: Path, fold_glob: str) -> list[Path]:
    fold_dirs = sorted(path for path in checkpoint_root.glob(fold_glob) if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold checkpoint directories found in {checkpoint_root} with pattern {fold_glob!r}."
        )

    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / "best_model.pth"
        config_path = fold_dir / "config.yaml"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing fold config: {config_path}")

    return fold_dirs


def build_inference_command(args: argparse.Namespace, fold_dir: Path) -> tuple[list[str], Path]:
    fold_output_dir = args.output_dir / fold_dir.name
    command = [
        sys.executable,
        "-m",
        "aiflopp.infer_mil_attention",
        "--checkpoint-dir",
        str(fold_dir),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(fold_output_dir),
        "--batch-size",
        str(args.batch_size),
    ]

    if args.features_root is not None:
        command.extend(["--features-root", str(args.features_root)])
    if args.handcrafted_features_root is not None:
        command.extend(["--handcrafted-features-root", str(args.handcrafted_features_root)])
    if not args.save_attention_scores:
        command.append("--no-attention-scores")

    return command, fold_output_dir


def load_fold_predictions(fold_outputs: list[tuple[str, Path]]) -> pd.DataFrame:
    prediction_dfs = []
    expected_bag_ids: set[str] | None = None

    for fold_name, fold_output_dir in fold_outputs:
        predictions_path = fold_output_dir / "predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing fold predictions: {predictions_path}")

        pred_df = pd.read_csv(predictions_path)
        required_cols = {"bag_id", "label", "pred_label"}
        missing = required_cols - set(pred_df.columns)
        if missing:
            raise ValueError(f"Predictions file {predictions_path} missing columns: {missing}")

        pred_df = pred_df.copy()
        pred_df["fold"] = fold_name
        bag_ids = set(pred_df["bag_id"].astype(str).tolist())
        if expected_bag_ids is None:
            expected_bag_ids = bag_ids
        elif bag_ids != expected_bag_ids:
            raise ValueError(f"Fold {fold_name} predictions have a different bag_id set.")

        prediction_dfs.append(pred_df)

    return pd.concat(prediction_dfs, ignore_index=True)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def order_metric_names(metric_names: list[str]) -> list[str]:
    priority = [name for name in METRIC_PRIORITY if name in metric_names]
    remaining = sorted(
        name
        for name in metric_names
        if name not in METRIC_PRIORITY
    )
    return priority + remaining


def order_metrics(metrics: dict) -> dict:
    ordered = {}
    for key in order_metric_names(list(metrics.keys())):
        ordered[key] = metrics[key]
    return ordered


def load_fold_metrics(fold_outputs: list[tuple[str, Path]]) -> list[dict]:
    fold_metrics = []
    for fold_name, fold_output_dir in fold_outputs:
        metrics_path = fold_output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {metrics_path}")
        with open(metrics_path) as f:
            metrics = json.load(f)
        fold_metrics.append({"fold": fold_name, **order_metrics(metrics)})
    return fold_metrics


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    values_by_metric: dict[str, list[float]] = {}
    for metrics in fold_metrics:
        for key, value in metrics.items():
            if key == "fold" or not is_number(value):
                continue
            values_by_metric.setdefault(key, []).append(float(value))

    summary = {}
    for metric_name in order_metric_names(list(values_by_metric.keys())):
        values = values_by_metric[metric_name]
        finite_values = [value for value in values if math.isfinite(value)]
        if finite_values:
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
        else:
            mean = float("nan")
            std = float("nan")
        summary[metric_name] = {
            "mean": mean,
            "std": std,
            "count": len(finite_values),
            "values": values,
        }

    return summary


def write_fold_metrics_csv(fold_metrics: list[dict], output_path: Path) -> None:
    metric_names = sorted(
        {
            key
            for metrics in fold_metrics
            for key, value in metrics.items()
            if key != "fold" and is_number(value)
        }
    )
    metric_names = order_metric_names(metric_names)
    rows = []
    for metrics in fold_metrics:
        row = {"fold": metrics["fold"]}
        for metric_name in metric_names:
            row[metric_name] = metrics.get(metric_name)
        rows.append(row)

    pd.DataFrame(rows, columns=["fold", *metric_names]).to_csv(output_path, index=False)


def write_fold_metrics_summary_csv(summary: dict, output_path: Path) -> None:
    rows = []
    for metric_name, stats in summary.items():
        rows.append(
            {
                "metric": metric_name,
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
                "values": json.dumps(stats["values"]),
            }
        )
    pd.DataFrame(rows, columns=["metric", "mean", "std", "count", "values"]).to_csv(output_path, index=False)


def _probability_for_label(group: pd.DataFrame, label: int) -> float | None:
    prob_col = f"prob_class_{label}"
    if prob_col in group.columns:
        return float(group[prob_col].mean())
    if "pred_prob" in group.columns and label in {0, 1}:
        mean_prob = float(group["pred_prob"].mean())
        return mean_prob if label == 1 else 1.0 - mean_prob
    return None


def _choose_majority_label(group: pd.DataFrame) -> tuple[int, int, bool]:
    vote_counts = group["pred_label"].astype(int).value_counts().to_dict()
    max_votes = max(vote_counts.values())
    tied_labels = sorted(label for label, count in vote_counts.items() if count == max_votes)

    if len(tied_labels) == 1:
        return tied_labels[0], max_votes, False

    scored_labels = []
    for label in tied_labels:
        probability = _probability_for_label(group, label)
        scored_labels.append((label, -math.inf if probability is None else probability))

    best_score = max(score for _, score in scored_labels)
    best_labels = [label for label, score in scored_labels if np.isclose(score, best_score)]
    return min(best_labels), max_votes, True


def aggregate_majority_predictions(all_predictions: pd.DataFrame) -> pd.DataFrame:
    all_labels = sorted(all_predictions["pred_label"].astype(int).unique().tolist())
    prob_class_cols = sorted(
        col for col in all_predictions.columns if col.startswith("prob_class_")
    )

    rows = []
    for bag_id, group in all_predictions.groupby("bag_id", sort=True):
        labels = group["label"].astype(int).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels across folds for bag_id={bag_id}: {labels}")

        pred_label, vote_count, tie = _choose_majority_label(group)
        row = {
            "bag_id": bag_id,
            "label": int(labels[0]),
            "pred_label": int(pred_label),
            "vote_count": int(vote_count),
            "num_models": int(len(group)),
            "vote_fraction": float(vote_count / len(group)),
            "tie": bool(tie),
        }

        vote_counts = group["pred_label"].astype(int).value_counts().to_dict()
        for label in all_labels:
            row[f"votes_class_{label}"] = int(vote_counts.get(label, 0))

        if "pred_prob" in group.columns:
            row["mean_pred_prob"] = float(group["pred_prob"].mean())
        for prob_col in prob_class_cols:
            row[f"mean_{prob_col}"] = float(group[prob_col].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def compute_majority_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["label"].to_numpy(dtype=int)
    y_pred = predictions["pred_label"].to_numpy(dtype=int)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    num_classes = max(labels) + 1 if labels else 0
    metric_labels = list(range(num_classes))

    metrics = {
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Multiclass Metrics
    if num_classes > 2:
        metrics["precision"] = float(precision_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0))
        prob_cols = [f"mean_prob_class_{class_idx}" for class_idx in range(num_classes)]
        if all(col in predictions.columns for col in prob_cols):
            try:
                metrics["auc"] = float(
                    roc_auc_score(
                        y_true,
                        predictions[prob_cols].to_numpy(),
                        labels=list(range(num_classes)),
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except ValueError:
                metrics["auc"] = float("nan")
        metrics.update(
            {
                "acc": float(accuracy_score(y_true, y_pred)),
                "f2": float(fbeta_score(y_true, y_pred, labels=metric_labels, beta=2, average="macro", zero_division=0)),
                "num_models": int(predictions["num_models"].iloc[0]) if len(predictions) else 0,
                "num_bags": int(len(predictions)),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=metric_labels).tolist(),
            }
        )
        return order_metrics(metrics)

    # Binary Metrics
    metrics.update(
        {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "recall_0": float(recall_score(y_true, y_pred, zero_division=0, pos_label=0)),
        }
    )
    if "mean_pred_prob" in predictions.columns:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, predictions["mean_pred_prob"].to_numpy()))
        except ValueError:
            metrics["auc"] = float("nan")

    metrics.update(
        {
            "acc": float(accuracy_score(y_true, y_pred)),
            "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
            "num_models": int(predictions["num_models"].iloc[0]) if len(predictions) else 0,
            "num_bags": int(len(predictions)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=metric_labels).tolist(),
        }
    )

    return order_metrics(metrics)


def print_metric_block(title: str, metrics: dict) -> None:
    print(title)
    for metric_name, value in metrics.items():
        if metric_name == "confusion_matrix":
            continue
        if is_number(value):
            print(f"  {metric_name}: {value:.4f}")


def print_metric_summary(title: str, summary: dict) -> None:
    print(title)
    for metric_name, stats in summary.items():
        print(
            f"  {metric_name}: mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, count={stats['count']}"
        )


def write_json(data, output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def main() -> None:
    args = parse_args()
    fold_dirs = discover_fold_checkpoints(args.checkpoint_root, args.fold_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_outputs = []
    for fold_dir in fold_dirs:
        command, fold_output_dir = build_inference_command(args, fold_dir)
        predictions_path = fold_output_dir / "predictions.csv"

        if args.skip_existing and predictions_path.exists():
            print(f"Skipping {fold_dir.name}; found existing {predictions_path}")
        else:
            if fold_output_dir.exists() and any(fold_output_dir.iterdir()):
                raise FileExistsError(
                    f"Fold inference output directory is not empty: {fold_output_dir}. "
                    "Use a new --output-dir or --skip-existing for completed folds."
                )
            print(f"Running {fold_dir.name}: {' '.join(command)}")
            if args.dry_run:
                continue
            subprocess.run(command, check=True)

        if not args.dry_run:
            fold_outputs.append((fold_dir.name, fold_output_dir))

    if args.dry_run:
        return

    all_predictions = load_fold_predictions(fold_outputs)
    all_predictions_path = args.output_dir / "fold_predictions.csv"
    all_predictions.to_csv(all_predictions_path, index=False)

    fold_metrics = load_fold_metrics(fold_outputs)
    fold_metrics_path = args.output_dir / "fold_metrics.json"
    write_json(fold_metrics, fold_metrics_path)

    fold_metrics_csv_path = args.output_dir / "fold_metrics.csv"
    write_fold_metrics_csv(fold_metrics, fold_metrics_csv_path)

    fold_metrics_summary = summarize_fold_metrics(fold_metrics)
    fold_metrics_summary_path = args.output_dir / "fold_metrics_summary.json"
    write_json(fold_metrics_summary, fold_metrics_summary_path)

    fold_metrics_summary_csv_path = args.output_dir / "fold_metrics_summary.csv"
    write_fold_metrics_summary_csv(fold_metrics_summary, fold_metrics_summary_csv_path)

    majority_predictions = aggregate_majority_predictions(all_predictions)
    majority_predictions_path = args.output_dir / "majority_predictions.csv"
    majority_predictions.to_csv(majority_predictions_path, index=False)

    majority_metrics = compute_majority_metrics(majority_predictions)
    metrics_path = args.output_dir / "majority_metrics.json"
    write_json(majority_metrics, metrics_path)

    print(f"Saved fold predictions to {all_predictions_path}")
    print(f"Saved per-model metrics to {fold_metrics_path}")
    print(f"Saved per-model metrics table to {fold_metrics_csv_path}")
    print(f"Saved per-model metric summary to {fold_metrics_summary_path}")
    print(f"Saved per-model metric summary table to {fold_metrics_summary_csv_path}")
    print(f"Saved majority-vote predictions to {majority_predictions_path}")
    print(f"Saved majority-vote metrics to {metrics_path}")

    print_metric_summary("\nPer-model metric summary:", fold_metrics_summary)
    print_metric_block("\nMajority-vote metrics:", majority_metrics)


if __name__ == "__main__":
    main()
