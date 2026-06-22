import argparse
import json
from pathlib import Path
import sys

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.feature_utils import HandcraftedFeatureScaler, infer_input_dim
from aiflopp.models import MODEL_REGISTRY
from aiflopp.train_mil_attention import print_metrics


def is_multiclass_task(num_classes: int) -> bool:
    return num_classes > 2


def parse_args() -> argparse.Namespace:

    default_manifest = Path(
        "data/manifests/afpp_manifest_base_diff/test_manifest.csv"
    )
    default_output_dir = Path("/home/ubuntu/giodir/digitalPathology/aiflopp/outputs_inference/test_tn_base_on_re")

    parser = argparse.ArgumentParser(
        description="Run MIL inference from a saved checkpoint folder."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Optional override for the deep feature root saved in checkpoint config.",
    )
    parser.add_argument(
        "--handcrafted-features-root",
        type=Path,
        default=None,
        help="Optional override for the handcrafted feature root saved in checkpoint config.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-attention-scores",
        action="store_true",
        help="Do not save per-bag attention score CSV files.",
    )
    return parser.parse_args()


def load_checkpoint_config(checkpoint_dir: Path) -> tuple[argparse.Namespace, dict]:
    """
    Load the checkpoint config.yaml file and extract the model configuration.
    
    Returns:
        model_args (argparse.Namespace): model configuration parameters used to build the model
        config (dict): the full checkpoint config
    """

    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config.yaml must contain a mapping of argument names to values.")

    model_type = config.get("model_type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type in config: {model_type}")

    model_args = argparse.Namespace(**config)

    return model_args, config


def save_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_attention_scores(
    bag_id: str,
    attention_weights: torch.Tensor,
    patch_metadata: dict | None,
    attention_dir: Path,
) -> None:
    if patch_metadata is None:
        raise ValueError(f"Patch metadata is required to save attention scores for bag {bag_id}")

    coords = np.asarray(patch_metadata["coords"])
    wsi_names = np.asarray(patch_metadata["wsi_names"])
    weights = attention_weights.detach().cpu().numpy()

    if len(coords) != len(weights):
        raise ValueError(
            f"metadata/features length mismatch for {bag_id}: metadata={len(coords)} attention={len(weights)}"
        )

    # Include wsi_name in patch_id because the same coordinates can appear across WSIs.
    patch_ids = [
        f"{str(wsi_name)}__patch_{int(x)}_{int(y)}"
        for wsi_name, (x, y) in zip(wsi_names, coords)
    ]
    attention_df = pd.DataFrame(
        {
            "bag_id": bag_id,
            "patch_id": patch_ids,
            "wsi_name": wsi_names,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "attention_score": weights,
        }
    )

    # Save the csv
    attention_dir.mkdir(parents=True, exist_ok=True)
    attention_df.to_csv(attention_dir / f"{bag_id}.csv", index=False)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = 0.5,
    num_classes: int = 2,
) -> dict:
    labels = list(range(num_classes))

    if is_multiclass_task(num_classes):
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        recall_0 = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0, pos_label=0)
        f2 = fbeta_score(y_true, y_pred, labels=labels, beta=2, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        try:
            auc = roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            auc = float("nan")

        return {
            "threshold": None,
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "recall_0": float(recall_0),
            "f2": float(f2),
            "balanced_acc": float(balanced_acc),
            "auc": float(auc),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f2": float(f2),
            "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "weighted_f2": float(fbeta_score(y_true, y_pred, labels=labels, beta=2, average="weighted", zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }

    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, zero_division=0, pos_label=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "threshold": float(threshold),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "recall_0": float(recall_0),
        "f2": float(f2),
        "balanced_acc": float(balanced_acc),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None,
    num_classes: int,
    output_dir: Path,
    save_attention: bool = True,
) -> dict:
    model.eval()

    attention_dir = output_dir / "attention_scores" if save_attention else None
    prediction_rows: list[dict] = []

    for bags, labels, bag_ids, patch_metadata_list in tqdm(loader):

        # Process one batch of bags

        bags = [b.to(device) for b in bags]
        labels = labels.to(device)

        logits, attn_weights = model(bags) # logits shape (batch_size,) or (batch_size, num_classes)
        if is_multiclass_task(num_classes):
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        else:
            if threshold is None:
                threshold = 0.5
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
        labels_np = labels.cpu().numpy().astype(int)

        for row_idx, (bag_id, label, pred, weights, patch_metadata) in enumerate(zip(
            bag_ids, labels_np, preds, attn_weights, patch_metadata_list
        )):
            prediction_row = {
                "bag_id": bag_id,
                "label": int(label),
                "pred_label": int(pred),
            }
            if is_multiclass_task(num_classes):
                for class_idx in range(num_classes):
                    prediction_row[f"prob_class_{class_idx}"] = float(probs[row_idx, class_idx])
            else:
                prediction_row["pred_prob"] = float(probs[row_idx])

            prediction_rows.append(prediction_row)
            if save_attention:
                save_attention_scores(
                    bag_id=bag_id,
                    attention_weights=weights,
                    patch_metadata=patch_metadata,
                    attention_dir=attention_dir,
                )

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    y_true = pred_df["label"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    if is_multiclass_task(num_classes):
        prob_cols = [f"prob_class_{class_idx}" for class_idx in range(num_classes)]
        y_prob = pred_df[prob_cols].to_numpy()
    else:
        y_prob = pred_df["pred_prob"].to_numpy()

    return compute_metrics(y_true, y_prob, threshold=threshold, num_classes=num_classes)


def resolve_inference_data_config(
    args: argparse.Namespace,
    config: dict,
) -> tuple[str, Path, Path | None, HandcraftedFeatureScaler | None, float | None]:
    feature_mode = config.get("feature_mode", "deep")
    features_root_value = args.features_root or config.get("features_root")
    if features_root_value is None:
        raise ValueError("Deep feature root is missing from both CLI arguments and checkpoint config.")

    handcrafted_root_value = args.handcrafted_features_root
    if handcrafted_root_value is None:
        handcrafted_root_value = config.get("handcrafted_features_root")

    handcrafted_scaler = load_handcrafted_scaler(args.checkpoint_dir, feature_mode)
    decision_threshold = load_decision_threshold(args.checkpoint_dir, int(config.get("num_classes", 2)))

    return (
        feature_mode,
        Path(features_root_value),
        Path(handcrafted_root_value) if handcrafted_root_value is not None else None,
        handcrafted_scaler,
        decision_threshold,
    )


def load_handcrafted_scaler(
    checkpoint_dir: Path,
    feature_mode: str,
) -> HandcraftedFeatureScaler | None:
    if feature_mode not in {"handcrafted", "concat"}:
        return None

    scaler_path = checkpoint_dir / "handcrafted_scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing handcrafted scaler file: {scaler_path}")

    with np.load(scaler_path) as data:
        return HandcraftedFeatureScaler(
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
        )


def load_decision_threshold(checkpoint_dir: Path, num_classes: int) -> float | None:
    """
    Loads from the metrics file the best decision threshold for binary classification.
    """

    if is_multiclass_task(num_classes):
        return None

    metrics_path = checkpoint_dir / "metrics.json"
    if not metrics_path.exists():
        return 0.5

    with open(metrics_path) as f:
        metrics = json.load(f)

    threshold_value = metrics.get("decision_threshold", 0.5)
    return 0.5 if threshold_value is None else float(threshold_value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = pd.read_csv(args.manifest)
    required_cols = {"bag_id", "label"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    # Load and validate model config from checkpoint
    model_args, config = load_checkpoint_config(args.checkpoint_dir)
    num_classes = int(getattr(model_args, "num_classes", 2))
    model_entry = MODEL_REGISTRY[model_args.model_type]

    (
        feature_mode,
        features_root,
        handcrafted_features_root,
        handcrafted_scaler,
        decision_threshold,
    ) = resolve_inference_data_config(args, config)

    model_args.input_dim = infer_input_dim(
        manifest,
        feature_mode=feature_mode,
        deep_features_root=features_root,
        handcrafted_features_root=handcrafted_features_root,
    )
    model_args.output_dim = 1 if not is_multiclass_task(num_classes) else num_classes
    model_entry["validate"](model_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    # Load and build model
    model = model_entry["build"](model_args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    num_workers = 0
    dataset = MILBagDataset(
        manifest,
        features_root,
        handcrafted_features_root=handcrafted_features_root,
        feature_mode=feature_mode,
        handcrafted_scaler=handcrafted_scaler,
        max_bag_size=0,
        enable_sampling=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
        drop_last=False,
    )

    metrics = run_inference(
        model=model,
        loader=loader,
        device=device,
        threshold=decision_threshold,
        num_classes=num_classes,
        output_dir=args.output_dir,
        save_attention=not args.no_attention_scores,
    )

    metrics["threshold"] = decision_threshold
    metrics["num_classes"] = num_classes
    metrics["model_folder"] = str(args.checkpoint_dir)
    metrics["test_manifest"] = str(args.manifest)
    save_metrics(metrics, args.output_dir)

    print_metrics("inference_set", metrics)

    print(f"Saved predictions to {args.output_dir / 'predictions.csv'}")
    if args.no_attention_scores:
        print("Skipped attention score export")
    else:
        print(f"Saved attention scores to {args.output_dir / 'attention_scores'}")


if __name__ == "__main__":
    main()
