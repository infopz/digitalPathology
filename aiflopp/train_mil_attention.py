import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
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

METRIC_CHOICES = ("acc", "precision", "recall", "recall_0", "f2", "balanced_acc", "auc")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    config_args, _ = config_parser.parse_known_args()

    default_train_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_mRT_binary_diff/train_manifest.csv"
    )
    default_val_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_mRT_binary_diff/val_manifest.csv"
    )
    default_test_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_mRT_binary_diff/test_manifest.csv"
    )

    default_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/features/uni_features_merged_RE_TN"
    )
    default_handcrafted_features_root = None
    default_output_dir = Path("aiflopp/outputs/merged_RT/binary_diff_first")

    parser = argparse.ArgumentParser(
        description="Train a MIL attention model on subregion patch features.",
        parents=[config_parser],
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

    # Metrics and evaluation
    parser.add_argument(
        "--epoch-selection-metric",
        type=str,
        choices=METRIC_CHOICES,
        default="auc",
        help="Metric used to select the best epoch during training.",
    )
    parser.add_argument(
        "--epoch-selection-secondary-metric",
        type=str,
        choices=METRIC_CHOICES,
        default="balanced_acc",
        help="Secondary metric used to break ties when selecting the best epoch.",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        choices=METRIC_CHOICES,
        default="balanced_acc",
        help="Validation metric used to choose the final decision threshold.",
    )

    # Other settings
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    
    if config_args.config is not None:
        with config_args.config.open("r") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError("YAML config must contain a mapping of argument names to values.")
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config option(s): {', '.join(unknown_keys)}")
        parser.set_defaults(**config)

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
    # Validate the manifest labels
    # Return the number of classes to use for training and evaluation
    
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
        recall_0 = recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0, pos_label=0)
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
            "balanced_acc": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "recall_0": float(recall_0),
            "auc": float(auc),
            "f2": float(f2),
            "acc": float(acc),
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
    recall_0 = recall_score(y_true, y_pred, zero_division=0, pos_label=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    metrics = {
        "threshold": float(threshold),
        "balanced_acc": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "recall_0": float(recall_0),
        "auc": float(auc),
        "acc": float(acc),
        "f2": float(f2),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def search_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    objective: str = "balanced_acc",
    objective_secondary: str = "auc",
) -> tuple[float, dict, list[dict]]:
    """
    Given a set of true and predicted labels (as probs),
    search the best threshold that maximizes the given objective metric.

    Returns:
        best_threshold (float): The best threshold value.
        best_metrics (dict): Metrics for the best threshold.
        all_results (list[dict]): Metrics for all evaluated thresholds.
    """

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_metrics: dict | None = None
    all_results: list[dict] = []

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))

        if objective not in metrics:
            raise ValueError(f"Objective metric '{objective}' not found in computed metrics.")
        if objective_secondary not in metrics:
            raise ValueError(f"Secondary objective metric '{objective_secondary}' not found in computed metrics.")

        all_results.append(metrics)

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        current_score = metrics[objective]
        best_score = best_metrics[objective]
        same_score = np.isclose(current_score, best_score)

        # Check the best metric first, than use the secondary metric to break ties
        if current_score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif same_score and metrics[objective_secondary] > best_metrics[objective_secondary]:
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
    best_metric: str | None = None, # balanced_acc
    secondary_metric: str = "auc"
):
    if is_multiclass_task(args.num_classes):
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    patience_max = args.patience

    best_val_primary = -float("inf")
    best_val_secondary = -float("inf")
    best_state = None
    best_epoch = -1
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        train_predictions = []
        train_labels = []

        for bags, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            bags = [b.to(device) for b in bags]
            labels = labels.to(device)

            logits, _ = model(bags)

            if is_multiclass_task(args.num_classes):
                train_probs = torch.softmax(logits.detach(), dim=1)
            else:
                train_probs = torch.sigmoid(logits.detach())
            train_predictions.append(train_probs.cpu())
            train_labels.append(labels.detach().cpu())

            if is_multiclass_task(args.num_classes):
                loss = criterion(logits, labels.long())
            else:
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bags)

        avg_loss = epoch_loss / len(train_loader.dataset)

        train_metrics = compute_metrics(
            y_true=torch.cat(train_labels).numpy(),
            y_prob=torch.cat(train_predictions).numpy(),
            num_classes=args.num_classes,
        )
        print(
            f"Epoch {epoch}: loss={avg_loss:.3f} "
            f"train_recall={train_metrics['recall']:.3f} "
            f"train_bal_acc={train_metrics['balanced_acc']:.3f} "
            f"train_auc={train_metrics['auc']:.3f} "
        )
        val_metrics = evaluate(model, val_loader, device, num_classes=args.num_classes)
        print(
            f"Epoch {epoch}:             "
            f" val_recall={val_metrics['recall']:.3f} "
            f"  val_bal_acc={val_metrics['balanced_acc']:.3f} "
            f"  val_auc={val_metrics['auc']:.3f} "
        )

        if best_metric is None:
            # Skip epoch selection and early stopping if no best_metric is specified
            continue

        if val_metrics[best_metric] >= best_val_primary:
            secondary_score = val_metrics.get(secondary_metric, -float("inf"))
            if val_metrics[best_metric] > best_val_primary or secondary_score > best_val_secondary:
                # If the primary improved or (the primary is same and) secondary improved, update the best state
                best_val_primary = val_metrics[best_metric]
                best_val_secondary = secondary_score
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
        print(f"Best validation epoch {best_epoch} with {best_metric}: {best_val_primary:.4f}. Loading best model state.")
        model.load_state_dict(best_state)
    return model


def save_model_and_metadata(
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    handcrafted_scaler: HandcraftedFeatureScaler | None = None,
) -> None:
    
    output_dir.mkdir(parents=True, exist_ok=True)

    def yaml_safe(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: yaml_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [yaml_safe(item) for item in value]
        return value

    config_keys = [
        "train_manifest",
        "val_manifest",
        "test_manifest",
        "features_root",
        "handcrafted_features_root",
        "output_dir",
        "model_type",
        "attention_dim",
        "hidden_dim",
        "num_classes",
        "dropout",
        "feature_mode",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "patience",
        "max_bag_size",
        "epoch_selection_metric",
        "epoch_selection_secondary_metric",
        "threshold_metric",
        "device",
        "num_workers",
        "seed",
    ]
    used_config = {key: yaml_safe(getattr(args, key)) for key in config_keys}
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(used_config, f, sort_keys=False)
    print(f"Saved resolved training config to {config_path}")

    if handcrafted_scaler is not None:
        scaler_path = output_dir / "handcrafted_scaler.npz"
        np.savez(
            scaler_path,
            mean=handcrafted_scaler.mean,
            scale=handcrafted_scaler.scale,
        )
        print(f"Saved handcrafted scaler to {scaler_path}")

    # Save model checkpoint
    save_path = output_dir / "best_model.pth"
    save_model(model, save_path)


def save_model(model: nn.Module, output_path: Path) -> None:
    # Save model checkpoint to the given folder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


def save_model_metrics(metrics: dict, output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved evaluation metrics to {metrics_path}")


def print_metrics(metrics: dict, split_name: str = "test", compact=False) -> None:
    if compact:
        print(f"{split_name} metrics: bal_acc: {metrics['balanced_acc']:.3f}, pr: {metrics['precision']:.3f}, auc: {metrics['auc']:.3f}")
        return

    print(f"Final {split_name} metrics:")
    if metrics["threshold"] is None:
        print("  threshold: None")
    else:
        print(f"  threshold: {metrics['threshold']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_acc']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  recall_0: {metrics['recall_0']:.4f}")
    print(f"  auc: {metrics['auc']:.4f}")
    print(f"  accuracy: {metrics['acc']:.4f}")
    print(f"  f2: {metrics['f2']:.4f}")
    print("  confusion_matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")

    print(f"  For Export: {metrics['balanced_acc']:.3f}\t{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['recall_0']:.3f}\t{metrics['auc']:.3f}")


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
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        loss_weight=loss_weight,
        best_metric=args.epoch_selection_metric,
        secondary_metric=args.epoch_selection_secondary_metric,
    )

    val_bag_ids, val_y_true, val_y_prob = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=args.num_classes,
    )
    if is_multiclass_task(args.num_classes):
        best_threshold = None
        val_threshold_search = []
        val_metrics = compute_metrics(
            y_true=val_y_true,
            y_prob=val_y_prob,
            threshold=None,
            num_classes=args.num_classes,
        )
        args.decision_threshold = None
    else:
        # Tune the threshold on the validation predictions
        # For tie break, use the epoch selection metric if its different from the threshold one, otherwise use the secondary
        tie_breaking_metric = (
            args.epoch_selection_metric
            if args.epoch_selection_metric != args.threshold_metric
            else args.epoch_selection_secondary_metric
        )
        best_threshold, val_metrics, val_threshold_search = search_best_threshold(
            y_true=val_y_true,
            y_prob=val_y_prob,
            objective=args.threshold_metric,
            objective_secondary=tie_breaking_metric
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
        model=model,
        loader=test_loader,
        device=device,
        threshold=best_threshold,
        num_classes=args.num_classes,
        output_csv_path=test_csv_path,
    )

    print_metrics(val_metrics, "val")
    print_metrics(test_metrics, "test")

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
        handcrafted_scaler=handcrafted_scaler,
    )


if __name__ == "__main__":
    main()
