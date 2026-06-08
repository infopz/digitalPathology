import argparse
import json
from pathlib import Path
import sys

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
from aiflopp.feature_utils import HandcraftedFeatureScaler
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
        help="Optional override for the deep feature root saved in checkpoint metadata.",
    )
    parser.add_argument(
        "--handcrafted-features-root",
        type=Path,
        default=None,
        help="Optional override for the handcrafted feature root saved in checkpoint metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_checkpoint_config(checkpoint_dir: Path) -> tuple[argparse.Namespace, dict]:
    """
    Load the checkoint metadata.json file and extract the model_details that contains the model configurations.
    
    Returns:
        model_args (argparse.Namespace): model configuration parameters used to build the model
        metadata (dict): the full metadata from the checkpoint
    """

    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    model_details = metadata.get("model_details", {})
    model_type = model_details.get("model_type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type in metadata: {model_type}")

    model_args = argparse.Namespace(**model_details)

    return model_args, metadata


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
) -> dict:
    model.eval()

    attention_dir = output_dir / "attention_scores"
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
    metadata: dict,
) -> tuple[str, Path, Path | None, HandcraftedFeatureScaler | None, float | None]:
    data_details = metadata.get("data_details", {})
    training_details = metadata.get("training_details", {})

    feature_mode = data_details.get("feature_mode", "deep")
    features_root_value = args.features_root or data_details.get("features_root")
    if features_root_value is None:
        raise ValueError("Deep feature root is missing from both CLI arguments and checkpoint metadata.")

    handcrafted_root_value = args.handcrafted_features_root
    if handcrafted_root_value is None:
        handcrafted_root_value = data_details.get("handcrafted_features_root")

    handcrafted_scaler = HandcraftedFeatureScaler.from_metadata(
        data_details.get("handcrafted_scaler")
    )
    threshold_value = training_details.get("decision_threshold", 0.5)
    decision_threshold = None if threshold_value is None else float(threshold_value)

    return (
        feature_mode,
        Path(features_root_value),
        Path(handcrafted_root_value) if handcrafted_root_value is not None else None,
        handcrafted_scaler,
        decision_threshold,
    )


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
    model_args, metadata = load_checkpoint_config(args.checkpoint_dir)
    training_details = metadata.get("training_details", {})
    num_classes = int(getattr(model_args, "num_classes", training_details.get("num_classes", 2)))
    model_entry = MODEL_REGISTRY[model_args.model_type]
    model_entry["validate"](model_args)

    (
        feature_mode,
        features_root,
        handcrafted_features_root,
        handcrafted_scaler,
        decision_threshold,
    ) = resolve_inference_data_config(args, metadata)

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
    )

    metrics["threshold"] = decision_threshold
    metrics["num_classes"] = num_classes
    metrics["model_folder"] = str(args.checkpoint_dir)
    metrics["test_manifest"] = str(args.manifest)
    save_metrics(metrics, args.output_dir)

    print_metrics("inference_set", metrics)

    print(f"Saved predictions to {args.output_dir / 'predictions.csv'}")
    print(f"Saved attention scores to {args.output_dir / 'attention_scores'}")


if __name__ == "__main__":
    main()
