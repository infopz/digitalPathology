import argparse
import json
from datetime import datetime
from pathlib import Path
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
from tqdm import tqdm

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.feature_utils import (
    AVAILABLE_FEATURE_MODES,
    HandcraftedFeatureScaler,
    fit_handcrafted_scaler,
    infer_input_dim,
)
from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    default_train_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_binary_diff/train_manifest.csv"
    )
    default_val_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_binary_diff/val_manifest.csv"
    )
    default_test_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_binary_diff/test_manifest.csv"
    )

    default_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/features/uni_features_RE_all"
    )
    default_handcrafted_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/features/ali_handcraft_RE_common_w_names"
    )
    default_output_dir = Path("aiflopp/outputs/all_data/binary_diff_first")

    parser = argparse.ArgumentParser(
        description="Train a MIL attention model on subregion patch features."
    )

    # Input/output paths
    parser.add_argument("--train-manifest", type=Path, default=default_train_manifest)
    parser.add_argument("--val-manifest", type=Path, default=default_val_manifest)
    parser.add_argument("--test-manifest", type=Path, default=default_test_manifest)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=default_features_root,
        help="Root folder containing deep per-bag feature npz files.",
    )
    parser.add_argument(
        "--handcrafted-features-root",
        type=Path,
        default=default_handcrafted_features_root,
        help="Optional root folder containing handcrafted per-bag feature npz files.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to save model checkpoints and logs.")

    # Model hyperparameters
    parser.add_argument(
        "--model-type",
        type=str,
        choices=AVAILABLE_MODEL_TYPES,
        default="base_mil",
        help="Type of MIL model to train.",
    )
    parser.add_argument(
        "--attention-dim", type=int, default=128, help="Hidden size for attention MLP."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden size for final classifier."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Number of label classes. If 0, infer from train/val/test manifests.",
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=AVAILABLE_FEATURE_MODES,
        default="deep",
        help="Which feature source to use: deep, handcrafted, or concat.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs.")
    parser.add_argument(
        "--max-bag-size",
        type=int,
        default=0,
        help="If >0, randomly subsample each bag to this many patches to stabilize batches.",
    )

    # Other settings
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--threshold-metric",
        type=str,
        choices=("acc", "precision", "recall", "f2", "balanced_acc"),
        default="balanced_acc",
        help="Validation metric used to choose the final decision threshold.",
    )
    
    return parser.parse_args()


def validate_output_dir(path: Path) -> Path:
    """
    Ensure output directory exists and is empty, or create a timestamped directory to avoid overwriting.
    Returns the path that should be used for output.
    """

    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = path.parent / f"{path.stem}_{timestamp}"
    print(
        f"Output directory {path} already exists and is not empty. "
        f"Using {timestamped_path} instead to avoid overwriting."
    )
    timestamped_path.mkdir(parents=True, exist_ok=True)

    return timestamped_path     


def is_multiclass_task(num_classes: int) -> bool:
    return num_classes > 2


def prepare_label_space(manifests: list[pd.DataFrame], requested_num_classes: int) -> int:
    labels: list[np.ndarray] = []

    # Parse all labels from manifests
    for manifest in manifests:
        if "label" not in manifest.columns:
            raise ValueError("Manifest missing required column: label")

        # Read the labels as numeric values
        label_values = pd.to_numeric(manifest["label"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(label_values).all():
            raise ValueError("Labels must be finite numeric values.")

        # Convert them to int
        label_ints = label_values.astype(int)
        if not np.allclose(label_values, label_ints):
            raise ValueError("Labels must be integer class ids.")

        # Re-assing the int labels
        manifest["label"] = label_ints
        labels.append(label_ints)

    # Validate the labels
    all_labels = np.concatenate(labels)
    if len(all_labels) == 0:
        raise ValueError("Cannot infer label space from empty manifests.")
    if all_labels.min() < 0:
        raise ValueError("Labels must be non-negative integer class ids.")

    # Check class consitency (contiguous)
    observed_classes = set(np.unique(all_labels).tolist())
    if requested_num_classes > 0:
        num_classes = requested_num_classes
        expected_classes = set(range(num_classes))
        unexpected = observed_classes - expected_classes
        if unexpected:
            raise ValueError(
                f"Observed labels outside --num-classes={num_classes}: {sorted(unexpected)}"
            )
    else:
        num_classes = int(all_labels.max()) + 1
        expected_classes = set(range(num_classes))
        missing = expected_classes - observed_classes
        if missing:
            raise ValueError(
                "Labels must be contiguous from 0 to num_classes - 1. "
                f"Missing classes: {sorted(missing)}"
            )

    if num_classes < 2:
        raise ValueError(f"At least 2 classes are required, got {num_classes}.")

    return num_classes


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Run the model on the given DataLoader
    """
    model.eval()

    all_bag_ids: list[str] = []
    all_probs: list[float] | list[list[float]] = []
    all_labels: list[int] = []

    for bags, labels, bag_ids, _ in loader:
        bags = [b.to(device) for b in bags]
        labels = labels.to(device)

        logits, _ = model(bags)
        if is_multiclass_task(num_classes):
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        all_bag_ids.extend(bag_ids)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    return all_bag_ids, np.array(all_labels), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = 0.5,
    num_classes: int = 2,
) -> dict:
    labels = list(range(num_classes))

    # Multiclass metrics
    if is_multiclass_task(num_classes):
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
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

    # Binary metrics

    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    metrics = {
        "threshold": float(threshold),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "balanced_acc": float(balanced_acc),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def search_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    objective: str = "f2",
) -> tuple[float, dict, list[dict]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_metrics: dict | None = None
    all_results: list[dict] = []

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))

        if objective not in metrics:
            raise ValueError(f"Objective metric '{objective}' not found in computed metrics.")

        all_results.append(metrics)

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        current_score = metrics[objective]
        best_score = best_metrics[objective]
        same_score = np.isclose(current_score, best_score)

        # Check the best metric first, than use recall and precision as tie-breakers
        if current_score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif same_score and metrics["recall"] > best_metrics["recall"]:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif (
            same_score
            and np.isclose(metrics["recall"], best_metrics["recall"])
            and metrics["precision"] > best_metrics["precision"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Threshold search did not evaluate any candidate thresholds.")

    return best_threshold, best_metrics, all_results


def save_predictions(
    bag_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None,
    num_classes: int,
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if is_multiclass_task(num_classes):
        pred_data = {
            "bag_id": bag_ids,
            "label": y_true,
            "pred_label": np.argmax(y_prob, axis=1),
        }
        for class_idx in range(num_classes):
            pred_data[f"prob_class_{class_idx}"] = y_prob[:, class_idx]
        pred_df = pd.DataFrame(pred_data)
    else:
        if threshold is None:
            threshold = 0.5
        pred_df = pd.DataFrame(
            {
                "bag_id": bag_ids,
                "label": y_true,
                "pred_prob": y_prob,
                "pred_label": (y_prob >= threshold).astype(int),
            }
        )

    pred_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = 0.5,
    num_classes: int = 2,
    output_csv_path: Path = None,
) -> dict:
    bag_ids, y_true, y_prob = collect_predictions(model, loader, device, num_classes)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold, num_classes=num_classes)

    # Optionally save predictions for error analysis
    if output_csv_path is not None:
        save_predictions(bag_ids, y_true, y_prob, threshold, num_classes, output_csv_path)

    return metrics


def compute_pos_weight(train_manifest: pd.DataFrame, device: torch.device) -> torch.Tensor:
    # Compute the labels distribution to calculate pos_weight for BCEWithLogitsLoss
    
    label_counts = train_manifest["label"].value_counts().to_dict()
    n_pos = int(label_counts.get(1, 0))
    n_neg = int(label_counts.get(0, 0))

    if n_pos == 0:
        raise ValueError("Training manifest has no positive bags; cannot compute pos_weight.")
    if n_neg == 0:
        raise ValueError("Training manifest has no negative bags; cannot compute pos_weight.")

    pos_weight = n_neg / n_pos
    print(
        f"Training class distribution: negatives={n_neg}, positives={n_pos}, "
        f"pos_weight={pos_weight:.4f}"
    )
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def compute_class_weights(
    train_manifest: pd.DataFrame,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    # Compute the labels distribution to calculate class weights for CrossEntropyLoss

    label_counts = train_manifest["label"].value_counts().to_dict()
    missing_classes = [class_idx for class_idx in range(num_classes) if class_idx not in label_counts]
    if missing_classes:
        raise ValueError(
            "Training manifest is missing classes required for multiclass training: "
            f"{missing_classes}"
        )

    counts = np.array([label_counts[class_idx] for class_idx in range(num_classes)], dtype=np.float32)
    weights = counts.sum() / (num_classes * counts)
    print(
        "Training class distribution: "
        f"{dict(zip(range(num_classes), counts.astype(int).tolist()))}, "
        f"class_weights={weights.tolist()}"
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    loss_weight: torch.Tensor,
    best_metric: str = "auc",
):
    if is_multiclass_task(args.num_classes):
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    patience_max = args.patience

    best_val = -float("inf")
    best_state = None
    best_epoch = -1
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for bags, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            bags = [b.to(device) for b in bags]
            labels = labels.to(device)

            logits, _ = model(bags)
            if is_multiclass_task(args.num_classes):
                loss = criterion(logits, labels.long())
            else:
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bags)

        avg_loss = epoch_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, num_classes=args.num_classes)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_bal_acc={val_metrics['balanced_acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
        )

        if val_metrics[best_metric] > best_val:
            # Save new best model state
            best_val = val_metrics[best_metric]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            # Check early stopping
            if epoch - best_epoch >= patience_max:
                print(
                    f"No improvement in {best_metric} for {patience_max} epochs. "
                    f"Stopping early at epoch {epoch}."
                )
                break

    if best_state is not None:
        print(f"Best validation epoch {best_epoch} with {best_metric}: {best_val:.4f}. Loading best model state.")
        model.load_state_dict(best_state)
    return model


def save_model_and_metadata(
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    model_config: dict,
    handcrafted_scaler: HandcraftedFeatureScaler | None,
) -> None:
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_dict = {
        "model_details": {
            "model_type": args.model_type,
            **model_config,
        },
        "training_details": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_bag_size": args.max_bag_size,
            "num_classes": args.num_classes,
            "pos_weight": args.pos_weight,
            "class_weights": args.class_weights,
            "threshold_metric": args.threshold_metric,
            "decision_threshold": args.decision_threshold,
        },
        "data_details": {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "test_manifest": str(args.test_manifest),
            "features_root": str(args.features_root),
            "feature_mode": args.feature_mode,
            "handcrafted_features_root": (
                str(args.handcrafted_features_root)
                if args.handcrafted_features_root is not None
                else None
            ),
            "handcrafted_scaler": (
                handcrafted_scaler.to_metadata() if handcrafted_scaler is not None else None
            ),
        }
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"Saved training metadata to {metadata_path}")

    # Save model checkpoint
    save_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")


def save_model_metrics(metrics: dict, output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved evaluation metrics to {metrics_path}")


def print_metrics(split_name: str, metrics: dict) -> None:
    print(f"Final {split_name} metrics:")
    if metrics["threshold"] is None:
        print("  threshold: None")
    else:
        print(f"  threshold: {metrics['threshold']:.4f}")
    print(f"  accuracy: {metrics['acc']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f2: {metrics['f2']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_acc']:.4f}")
    print(f"  auc: {metrics['auc']:.4f}")
    print("  confusion_matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")


def main() -> None:
    args = parse_args()
    args.output_dir = validate_output_dir(args.output_dir)

    seed_everything(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_manifest = pd.read_csv(args.train_manifest)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)

    args.num_classes = prepare_label_space(
        [train_manifest, val_manifest, test_manifest],
        requested_num_classes=args.num_classes,
    )
    args.output_dim = 1 if not is_multiclass_task(args.num_classes) else args.num_classes
    print(f"Using num_classes={args.num_classes}, model output_dim={args.output_dim}")

    # Select model_type and validate args
    model_entry = MODEL_REGISTRY[args.model_type]
    model_entry["validate"](args)

    # If using handcrafted features, fit a scaler on the training set to apply to all splits
    handcrafted_scaler = None
    if args.feature_mode in {"handcrafted", "concat"}:
        handcrafted_scaler = fit_handcrafted_scaler(
            train_manifest,
            args.handcrafted_features_root
        )
        print(
            "Fitted handcrafted scaler "
            f"for {len(handcrafted_scaler.mean)} handcrafted features"
        )

    input_dim = infer_input_dim(
        train_manifest,
        feature_mode=args.feature_mode,
        deep_features_root=args.features_root,
        handcrafted_features_root=args.handcrafted_features_root,
    )
    args.input_dim = input_dim
    print(f"Using feature mode: {args.feature_mode}")
    print(f"Inferred feature dim: {input_dim}")

    # Compute class weights
    if is_multiclass_task(args.num_classes):
        loss_weight = compute_class_weights(train_manifest, args.num_classes, device)
        args.pos_weight = None
        args.class_weights = loss_weight.detach().cpu().numpy().tolist()
        args.decision_threshold = None
    else:
        loss_weight = compute_pos_weight(train_manifest, device)
        args.pos_weight = float(loss_weight.item())
        args.class_weights = None
        args.decision_threshold = 0.5

    # Filter args to keep only those used by model initialization
    model_config = model_entry["config"](args)

    train_ds = MILBagDataset(
        train_manifest,
        args.features_root,
        handcrafted_features_root=args.handcrafted_features_root,
        feature_mode=args.feature_mode,
        handcrafted_scaler=handcrafted_scaler,
        max_bag_size=args.max_bag_size,
        enable_sampling=True,
    )
    val_ds = MILBagDataset(
        val_manifest,
        args.features_root,
        handcrafted_features_root=args.handcrafted_features_root,
        feature_mode=args.feature_mode,
        handcrafted_scaler=handcrafted_scaler,
        max_bag_size=args.max_bag_size,
        enable_sampling=False,
    )
    test_ds = MILBagDataset(
        test_manifest,
        args.features_root,
        handcrafted_features_root=args.handcrafted_features_root,
        feature_mode=args.feature_mode,
        handcrafted_scaler=handcrafted_scaler,
        max_bag_size=args.max_bag_size,
        enable_sampling=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
        drop_last=False,
    )

    # Build model based on specified model_type
    model = model_entry["build"](args).to(device)

    # Train model
    model = train(model, train_loader, val_loader, args, device, loss_weight, best_metric=args.threshold_metric)

    val_bag_ids, val_y_true, val_y_prob = collect_predictions(
        model,
        val_loader,
        device,
        args.num_classes,
    )
    if is_multiclass_task(args.num_classes):
        best_threshold = None
        val_threshold_search = []
        val_metrics = compute_metrics(
            val_y_true,
            val_y_prob,
            threshold=None,
            num_classes=args.num_classes,
        )
        args.decision_threshold = None
    else:
        # Tune the threshold on the validation predictions
        best_threshold, val_metrics, val_threshold_search = search_best_threshold(
            val_y_true,
            val_y_prob,
            objective=args.threshold_metric,
        )
        args.decision_threshold = float(best_threshold)

    val_csv_path = args.output_dir / "val_predictions.csv"
    save_predictions(
        val_bag_ids,
        val_y_true,
        val_y_prob,
        threshold=best_threshold,
        num_classes=args.num_classes,
        output_csv_path=val_csv_path,
    )
    if best_threshold is None:
        print(
            "Using argmax predictions on validation set "
            f"with {args.threshold_metric}={val_metrics[args.threshold_metric]:.4f}"
        )
    else:
        print(
            f"Selected validation threshold={best_threshold:.4f} "
            f"using {args.threshold_metric}={val_metrics[args.threshold_metric]:.4f}"
        )

    # Evaluate the model on test set using the selected threshold
    test_csv_path = args.output_dir / "test_predictions.csv"
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=best_threshold,
        num_classes=args.num_classes,
        output_csv_path=test_csv_path,
    )

    print_metrics("val", val_metrics)
    print_metrics("test", test_metrics)

    # Save model and metrics
    metrics = {
        "num_classes": args.num_classes,
        "decision_threshold": best_threshold,
        "threshold_metric": args.threshold_metric,
        "val_threshold_search": val_threshold_search,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_model_metrics(metrics, args.output_dir)
    save_model_and_metadata(
        model,
        args.output_dir,
        args,
        model_config,
        handcrafted_scaler=handcrafted_scaler,
    )


if __name__ == "__main__":
    main()
