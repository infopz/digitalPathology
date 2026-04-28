import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple


import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
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
    parser.add_argument("--epochs", type=int, default=40)
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, output_csv_path: Path = None) -> Tuple[float, float]:
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

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    # Optionally save predictions for error analysis
    if output_csv_path is not None:

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame({
            "bag_id": all_bag_ids,
            "label": y_true,
            "pred_prob": y_prob,
            "pred_label": y_pred,
        })
        pred_df.to_csv(output_csv_path, index=False)
        
        print(f"Saved predictions to {output_csv_path}")
    
    return acc, auc


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


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_auc = -float("inf")
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
        val_acc, val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
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
    model = train(model, train_loader, val_loader, args, device)

    # Evaluate final model and save predictions
    val_csv_path = args.output_dir / "val_predictions.csv"
    test_csv_path = args.output_dir / "test_predictions.csv"
    val_acc, val_auc = evaluate(model, val_loader, device, output_csv_path=val_csv_path)
    test_acc, test_auc = evaluate(model, test_loader, device, output_csv_path=test_csv_path)

    print(f"Final val_acc={val_acc:.4f} val_auc={val_auc:.4f}")
    print(f"Final test_acc={test_acc:.4f} test_auc={test_auc:.4f}")

    # Save model and metrics
    metrics = {
        "val_acc": val_acc,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }
    save_model_metrics(metrics, args.output_dir)
    save_model_and_metadata(model, args.output_dir, args, model_config)


if __name__ == "__main__":
    main()
