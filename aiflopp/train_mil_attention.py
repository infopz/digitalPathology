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
from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    default_train_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/manifests/afpp_manifest_base/train_manifest.csv"
    )
    default_val_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/manifests/afpp_manifest_base/val_manifest.csv"
    )
    default_test_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/manifests/afpp_manifest_base/test_manifest.csv"
    )

    default_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_common"
    )
    default_output_dir = Path("aiflopp/outputs")

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
        help="Root folder containing per-patient feature npz files.",
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
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()

    all_bag_ids: list[str] = []
    all_probs: list[float] = []
    all_labels: list[int] = []

    for bags, labels, bag_ids in loader:
        bags = [b.to(device) for b in bags]
        labels = labels.to(device)

        logits, _ = model(bags)
        probs = torch.sigmoid(logits)

        all_bag_ids.extend(bag_ids)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    return all_bag_ids, np.array(all_labels), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
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
    threshold: float,
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

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
    threshold: float = 0.5,
    output_csv_path: Path = None,
) -> dict:
    bag_ids, y_true, y_prob = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    # Optionally save predictions for error analysis
    if output_csv_path is not None:
        save_predictions(bag_ids, y_true, y_prob, threshold, output_csv_path)

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


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_input_dim(manifest: pd.DataFrame, features_root: Path) -> int:
    """Inspect the first bag to deduce feature dimensionality."""
    for _, row in manifest.iterrows():
        bag_id = row["bag_id"]
        feature_path = features_root / f"{bag_id}.npz"
        data = np.load(feature_path, allow_pickle=True)
        feats: np.ndarray = data["features"].astype(np.float32)
        return int(feats.shape[1])
    raise RuntimeError("Empty manifest; cannot infer feature dimensionality.")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    pos_weight: torch.Tensor,
    best_metric: str = "auc",
):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for bags, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            bags = [b.to(device) for b in bags]
            labels = labels.to(device)

            logits, _ = model(bags)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bags)

        avg_loss = epoch_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f2={val_metrics['f2']:.4f} "
            f"val_bal_acc={val_metrics['balanced_acc']:.4f} "
        )

        if val_metrics[best_metric] > best_val:
            best_val = val_metrics[best_metric]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save_model_and_metadata(
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    model_config: dict,
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
            "pos_weight": args.pos_weight,
            "threshold_metric": args.threshold_metric,
            "decision_threshold": args.decision_threshold,
        },
        "data_details": {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "test_manifest": str(args.test_manifest),
            "features_root": str(args.features_root),
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
    print(f"  threshold: {metrics['threshold']:.4f}")
    print(f"  accuracy: {metrics['acc']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f2: {metrics['f2']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_acc']:.4f}")
    print(f"  auc: {metrics['auc']:.4f}")
    print("  confusion_matrix:")
    print(f"    {metrics['confusion_matrix'][0]}")
    print(f"    {metrics['confusion_matrix'][1]}")


def main() -> None:
    args = parse_args()
    args.output_dir = validate_output_dir(args.output_dir)

    # Select model_type and validate args
    model_entry = MODEL_REGISTRY[args.model_type]
    model_entry["validate"](args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_manifest = pd.read_csv(args.train_manifest)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)

    input_dim = infer_input_dim(train_manifest, args.features_root)
    args.input_dim = input_dim
    print(f"Inferred feature dim: {input_dim}")
    pos_weight = compute_pos_weight(train_manifest, device)
    args.pos_weight = float(pos_weight.item())
    args.threshold_metric = "f2"
    args.decision_threshold = 0.5

    # Filter args to keep only those used by model initialization
    model_config = model_entry["config"](args)

    train_ds = MILBagDataset(
        train_manifest,
        args.features_root,
        args.max_bag_size,
        enable_sampling=True,
    )
    val_ds = MILBagDataset(
        val_manifest,
        args.features_root,
        args.max_bag_size,
        enable_sampling=False,
    )
    test_ds = MILBagDataset(
        test_manifest,
        args.features_root,
        args.max_bag_size,
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
    model = train(model, train_loader, val_loader, args, device, pos_weight)

    # Tune the threshold on the validation predictions
    val_bag_ids, val_y_true, val_y_prob = collect_predictions(model, val_loader, device)
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
        output_csv_path=val_csv_path,
    )
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
        output_csv_path=test_csv_path,
    )

    print_metrics("val", val_metrics)
    print_metrics("test", test_metrics)

    # Save model and metrics
    metrics = {
        "decision_threshold": best_threshold,
        "threshold_metric": args.threshold_metric,
        "val_threshold_search": val_threshold_search,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_model_metrics(metrics, args.output_dir)
    save_model_and_metadata(model, args.output_dir, args, model_config)


if __name__ == "__main__":
    main()
