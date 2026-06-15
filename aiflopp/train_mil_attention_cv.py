import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MIL attention training across manifest folds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base YAML config passed to each train_mil_attention run.",
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per fold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-fold outputs and CV summary files are saved.",
    )
    parser.add_argument(
        "--fold-glob",
        default="fold_*",
        help="Glob pattern used to discover fold directories inside --folds-dir.",
    )
    parser.add_argument("--train-manifest-name", default="train_manifest.csv")
    parser.add_argument("--val-manifest-name", default="val_manifest.csv")
    parser.add_argument("--test-manifest-name", default="test_manifest.csv")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folds that already have metrics.json in their output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fold training commands without running them.",
    )

    args, train_args = parser.parse_known_args()
    forbidden_train_args = {
        "--train-manifest",
        "--val-manifest",
        "--test-manifest",
        "--output-dir",
    }
    forbidden = [arg for arg in train_args if arg in forbidden_train_args]
    if forbidden:
        parser.error(
            "These arguments are controlled by the CV wrapper and cannot be forwarded: "
            f"{', '.join(forbidden)}"
        )

    return args, train_args


def discover_folds(args: argparse.Namespace) -> list[Path]:
    fold_dirs = sorted(
        path for path in args.folds_dir.glob(args.fold_glob) if path.is_dir()
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold directories found in {args.folds_dir} with pattern {args.fold_glob!r}."
        )

    for fold_dir in fold_dirs:
        for manifest_name in (
            args.train_manifest_name,
            args.val_manifest_name,
            args.test_manifest_name,
        ):
            manifest_path = fold_dir / manifest_name
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")

    return fold_dirs


def build_train_command(
    args: argparse.Namespace,
    train_args: list[str],
    fold_dir: Path,
) -> tuple[list[str], Path]:
    fold_output_dir = args.output_dir / fold_dir.name
    command = [sys.executable, "-m", "aiflopp.train_mil_attention"]

    if args.config is not None:
        command.extend(["--config", str(args.config)])

    command.extend(train_args)
    command.extend(
        [
            "--train-manifest",
            str(fold_dir / args.train_manifest_name),
            "--val-manifest",
            str(fold_dir / args.val_manifest_name),
            "--test-manifest",
            str(fold_dir / args.test_manifest_name),
            "--output-dir",
            str(fold_output_dir),
        ]
    )

    return command, fold_output_dir


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def collect_numeric_metrics(data: dict, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            metrics.update(collect_numeric_metrics(value, path))
        elif is_number(value):
            metrics[".".join(path)] = float(value)

    return metrics


def aggregate_metrics(fold_metrics: list[dict]) -> dict[str, dict]:
    values_by_metric: dict[str, list[float]] = {}
    for metrics in fold_metrics:
        for metric_name, value in collect_numeric_metrics(metrics).items():
            values_by_metric.setdefault(metric_name, []).append(value)

    aggregate = {}
    for metric_name, values in sorted(values_by_metric.items()):
        aggregate[metric_name] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "count": len(values),
            "values": values,
        }

    return aggregate


def write_summary_csv(aggregate: dict[str, dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "std", "count", "values"],
        )
        writer.writeheader()
        for metric_name, stats in aggregate.items():
            writer.writerow(
                {
                    "metric": metric_name,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "count": stats["count"],
                    "values": json.dumps(stats["values"]),
                }
            )


def main() -> None:
    args, train_args = parse_args()
    fold_dirs = discover_folds(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    fold_metrics = []

    for fold_dir in fold_dirs:
        command, fold_output_dir = build_train_command(args, train_args, fold_dir)
        metrics_path = fold_output_dir / "metrics.json"

        if args.skip_existing and metrics_path.exists():
            print(f"Skipping {fold_dir.name}; found existing {metrics_path}")
        else:
            if fold_output_dir.exists() and any(fold_output_dir.iterdir()):
                raise FileExistsError(
                    f"Fold output directory is not empty: {fold_output_dir}. "
                    "Use a new --output-dir or --skip-existing for completed folds."
                )
            print(f"Running {fold_dir.name}: {' '.join(command)}")
            if args.dry_run:
                continue
            subprocess.run(command, check=True)

        if args.dry_run:
            continue

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics after fold run: {metrics_path}")
        with open(metrics_path) as f:
            metrics = json.load(f)
        fold_metrics.append(metrics)
        fold_results.append(
            {
                "fold": fold_dir.name,
                "fold_dir": str(fold_dir),
                "output_dir": str(fold_output_dir),
                "metrics_path": str(metrics_path),
            }
        )

    if args.dry_run:
        return

    aggregate = aggregate_metrics(fold_metrics)
    summary = {
        "folds_dir": str(args.folds_dir),
        "output_dir": str(args.output_dir),
        "num_folds": len(fold_results),
        "folds": fold_results,
        "aggregate": aggregate,
    }

    summary_json_path = args.output_dir / "cv_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    summary_csv_path = args.output_dir / "cv_summary.csv"
    write_summary_csv(aggregate, summary_csv_path)

    print(f"Saved CV summary to {summary_json_path}")
    print(f"Saved CV summary table to {summary_csv_path}")


if __name__ == "__main__":
    main()
