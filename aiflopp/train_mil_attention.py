import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    default_train_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/train_manifest.csv"
    )
    default_val_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/val_manifest.csv"
    )
    default_test_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/test_manifest.csv"
    )

    default_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_common"
    )

    parser = argparse.ArgumentParser(
        description="Train a MIL attention model on subregion patch features."
    )
    parser.add_argument("--train-manifest", type=Path, default=default_train_manifest)
    parser.add_argument("--val-manifest", type=Path, default=default_val_manifest)
    parser.add_argument("--test-manifest", type=Path, default=default_test_manifest)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=default_features_root,
        help="Root folder containing per-patient feature npz files.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--attention-dim", type=int, default=128, help="Hidden size for attention MLP."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden size for final classifier."
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-bag-size",
        type=int,
        default=0,
        help="If >0, randomly subsample each bag to this many patches to stabilize batches.",
    )
    return parser.parse_args()


class MILBagDataset(Dataset):
    """Dataset that returns one bag (subregion) at a time."""

    def __init__(self, manifest: pd.DataFrame, features_root: Path, max_bag_size: int = 0):
        self.manifest = manifest.reset_index(drop=True)
        self.features_root = features_root
        self.max_bag_size = max_bag_size
        self.required_cols = {"bag_id", "label"}

        missing = self.required_cols - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_features(self, feature_path: Path) -> np.ndarray:
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features: {feature_path}")
        data = np.load(feature_path, allow_pickle=True)
        feats: np.ndarray = data["features"].astype(np.float32)

        # Randomly subsample patches
        # This can be improved
        if self.max_bag_size > 0 and len(feats) > self.max_bag_size:
            idx = np.random.choice(len(feats), size=self.max_bag_size, replace=False)
            feats = feats[idx]
        return feats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.manifest.iloc[idx]
        bag_id = row["bag_id"]
        label = float(row["label"])

        feature_path = self.features_root / f"{bag_id}.npz"
        feats = self._load_features(feature_path)

        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.float32), bag_id


def collate_bags(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    bags, labels, ids = zip(*batch)
    return list(bags), torch.stack(labels), list(ids)


class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        # E' la versione piu basic del MIL
        # W*tan(V*h) come attenzione, quindi una matrice V di peso per ogni patch che porta da una dimension attention_dim
        # poi W che la porta a 1 (seguita poi dalla softmax nella forward)
        
        # TODO: la si potrebbe migliorare con la Gated attention aggiungendo un'altra matrice U 
        #       e moltiplicando elemento per elemento V*h e U*h prima di passare a W
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bags: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits: list[torch.Tensor] = []
        attn_weights: list[torch.Tensor] = []

        for bag in bags:
            scores = self.attention(bag).squeeze(-1)  # (num_patches,)
            weights = torch.softmax(scores, dim=0)
            bag_repr = torch.sum(weights.unsqueeze(-1) * bag, dim=0)
            logit = self.classifier(bag_repr)

            logits.append(logit.squeeze(-1))
            attn_weights.append(weights)

        return torch.stack(logits), attn_weights


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    for bags, labels, _ in loader:
        bags = [b.to(device) for b in bags]
        labels = labels.to(device)

        logits, _ = model(bags)
        probs = torch.sigmoid(logits)

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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_manifest = pd.read_csv(args.train_manifest)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)

    input_dim = infer_input_dim(train_manifest, args.features_root)
    print(f"Inferred feature dim: {input_dim}")

    train_ds = MILBagDataset(train_manifest, args.features_root, args.max_bag_size)
    val_ds = MILBagDataset(val_manifest, args.features_root, args.max_bag_size)
    test_ds = MILBagDataset(test_manifest, args.features_root, args.max_bag_size)

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

    model = AttentionMIL(
        input_dim=input_dim,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    model = train(model, train_loader, val_loader, args, device)

    val_acc, val_auc = evaluate(model, val_loader, device)
    test_acc, test_auc = evaluate(model, test_loader, device)

    print(f"Final val_acc={val_acc:.4f} val_auc={val_auc:.4f}")
    print(f"Final test_acc={test_acc:.4f} test_auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
