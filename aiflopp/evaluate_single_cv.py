import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.feature_utils import infer_input_dim
from aiflopp.infer_mil_attention import load_handcrafted_scaler
from aiflopp.models import MODEL_REGISTRY
from aiflopp.train_mil_attention import (
    collect_predictions,
    compute_metrics,
    is_multiclass_task,
    search_best_threshold,
)


METRICS = ["balanced_acc", "precision", "recall", "recall_0", "auc", "f2", "acc"]
FOLD_METRIC_COLUMNS = ["fold", "threshold", *METRICS, "num_bags"]


def parse_args() -> argparse.Namespace:
    # Parse one checkpoint root and one CV manifest folder to evaluate.
    parser = argparse.ArgumentParser(description="Evaluate one CV model set with fold mean/std metrics.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--handcrafted-features-root", type=Path, default=None)
    parser.add_argument("--fold-glob", default="fold_*")
    parser.add_argument("--threshold-metric", default="balanced_acc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def discover_folds(folds_dir: Path, checkpoint_root: Path, fold_glob: str) -> list[str]:
    # Validate that each fold has val/test manifests and a checkpoint folder.
    fold_dirs = sorted(path for path in folds_dir.glob(fold_glob) if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {folds_dir} with pattern {fold_glob!r}.")

    fold_names = [path.name for path in fold_dirs]
    for fold_name in fold_names:
        fold_dir = folds_dir / fold_name
        checkpoint_dir = checkpoint_root / fold_name
        for split_name in ("val", "test"):
            manifest_path = fold_dir / f"{split_name}_manifest.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not (checkpoint_dir / "best_model.pth").exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir / 'best_model.pth'}")
        if not (checkpoint_dir / "config.yaml").exists():
            raise FileNotFoundError(f"Missing checkpoint config: {checkpoint_dir / 'config.yaml'}")
    return fold_names


def load_config(checkpoint_dir: Path) -> argparse.Namespace:
    # Load checkpoint config and fill defaults needed by exported FL checkpoints.
    with (checkpoint_dir / "config.yaml").open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {checkpoint_dir / 'config.yaml'}")

    config.setdefault("feature_mode", "deep")
    config.setdefault("max_bag_size", 0)
    config.setdefault("num_classes", 2)
    config.setdefault("output_dim", 1 if int(config["num_classes"]) <= 2 else int(config["num_classes"]))
    return argparse.Namespace(**config)


def resolve_feature_paths(
    args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    # Resolve deep and optional handcrafted feature roots.
    features_root = args.features_root or getattr(model_args, "features_root", None)
    if features_root is None:
        raise ValueError("features_root is missing from both CLI args and checkpoint config.")

    handcrafted_features_root = args.handcrafted_features_root
    if handcrafted_features_root is None and getattr(model_args, "handcrafted_features_root", None):
        handcrafted_features_root = Path(model_args.handcrafted_features_root)
    return Path(features_root), handcrafted_features_root


def build_loader(
    manifest_path: Path,
    model_args: argparse.Namespace,
    features_root: Path,
    handcrafted_features_root: Path | None,
    handcrafted_scaler,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, DataLoader]:
    # Build a deterministic dataloader for one manifest split.
    manifest = pd.read_csv(manifest_path)
    dataset = MILBagDataset(
        manifest,
        features_root,
        handcrafted_features_root=handcrafted_features_root,
        feature_mode=getattr(model_args, "feature_mode", "deep"),
        handcrafted_scaler=handcrafted_scaler,
        max_bag_size=0,
        enable_sampling=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
        drop_last=False,
    )
    return manifest, loader


def load_model(
    checkpoint_dir: Path,
    manifest: pd.DataFrame,
    model_args: argparse.Namespace,
    features_root: Path,
    handcrafted_features_root: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    # Build the model from config and load raw or NVFlare-nested weights.
    model_args.input_dim = infer_input_dim(
        manifest,
        feature_mode=getattr(model_args, "feature_mode", "deep"),
        deep_features_root=features_root,
        handcrafted_features_root=handcrafted_features_root,
    )
    num_classes = int(model_args.num_classes)
    model_args.output_dim = 1 if not is_multiclass_task(num_classes) else num_classes

    model_entry = MODEL_REGISTRY[model_args.model_type]
    model_entry["validate"](model_args)
    model = model_entry["build"](model_args).to(device)

    state = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    return model


def evaluate_fold(args: argparse.Namespace, fold_name: str) -> dict:
    # Tune the threshold on validation and compute metrics on the fold test split.
    checkpoint_dir = args.checkpoint_root / fold_name
    fold_manifest_dir = args.folds_dir / fold_name
    model_args = load_config(checkpoint_dir)
    num_classes = int(model_args.num_classes)
    features_root, handcrafted_features_root = resolve_feature_paths(args, model_args)
    handcrafted_scaler = load_handcrafted_scaler(checkpoint_dir, getattr(model_args, "feature_mode", "deep"))

    val_manifest, val_loader = build_loader(
        fold_manifest_dir / "val_manifest.csv",
        model_args,
        features_root,
        handcrafted_features_root,
        handcrafted_scaler,
        args.batch_size,
        args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_dir, val_manifest, model_args, features_root, handcrafted_features_root, device)
    _, val_y_true, val_y_prob, _ = collect_predictions(model, val_loader, device, num_classes)

    if is_multiclass_task(num_classes):
        threshold = None
    else:
        threshold, _, _ = search_best_threshold(val_y_true, val_y_prob, objective=args.threshold_metric)

    _, test_loader = build_loader(
        fold_manifest_dir / "test_manifest.csv",
        model_args,
        features_root,
        handcrafted_features_root,
        handcrafted_scaler,
        args.batch_size,
        args.num_workers,
    )
    _, test_y_true, test_y_prob, _ = collect_predictions(model, test_loader, device, num_classes)
    test_metrics = compute_metrics(test_y_true, test_y_prob, threshold=threshold, num_classes=num_classes)

    row = {
        "fold": fold_name,
        "threshold": threshold,
        "num_bags": int(len(test_y_true)),
    }
    row.update({metric_name: test_metrics.get(metric_name) for metric_name in METRICS})
    return row


def round_float(value):
    # Round finite floats and leave missing/non-finite values blank.
    if value is None:
        return ""
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        return round(float(value), 4) if math.isfinite(float(value)) else ""
    return value


def summarize_fold_rows(fold_rows: list[dict]) -> tuple[dict, dict]:
    # Compute mean and std rows for fold metrics only.
    avg_row = {"fold": "avg", "threshold": "", "num_bags": ""}
    std_row = {"fold": "std", "threshold": "", "num_bags": ""}
    for metric_name in METRICS:
        values = [float(row[metric_name]) for row in fold_rows if isinstance(row.get(metric_name), (int, float)) and math.isfinite(float(row[metric_name]))]
        avg_row[metric_name] = round_float(float(np.mean(values))) if values else ""
        std_row[metric_name] = round_float(float(np.std(values, ddof=1))) if len(values) > 1 else 0.0 if values else ""
    return avg_row, std_row


def write_fold_metrics_csv(fold_rows: list[dict], output_path: Path) -> None:
    # Write one fold metrics table with avg and std rows.
    avg_row, std_row = summarize_fold_rows(fold_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FOLD_METRIC_COLUMNS)
        writer.writeheader()
        for row in fold_rows:
            writer.writerow({key: round_float(row.get(key)) for key in FOLD_METRIC_COLUMNS})
        writer.writerow(avg_row)
        writer.writerow(std_row)


def main() -> None:
    # Evaluate all CV folds and write fold mean/std metrics.
    args = parse_args()
    output_path = args.output_dir / "fold_metrics.csv"
    if args.skip_existing and output_path.exists():
        print(f"Skipping evaluation; found {output_path}")
        return

    fold_names = discover_folds(args.folds_dir, args.checkpoint_root, args.fold_glob)
    fold_rows = [evaluate_fold(args, fold_name) for fold_name in fold_names]
    write_fold_metrics_csv(fold_rows, output_path)
    print(f"Saved fold metrics to {output_path}")


if __name__ == "__main__":
    main()
