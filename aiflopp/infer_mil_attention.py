import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.models import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:

    default_manifest = Path(
        "/home/ubuntu/giodir/digitalPathology/manifests/afpp_manifest_base/test_manifest.csv"
    )
    default_features_root = Path(
        "/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_common"
    )
    default_output_dir = Path("aiflopp/inference_outputs")

    parser = argparse.ArgumentParser(
        description="Run MIL inference from a saved checkpoint folder."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--features-root", type=Path, default=default_features_root)
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


def load_coords(feature_path: Path) -> np.ndarray:
    """
    Load patch coordinates from the feature file. Supports both flattened and 2D formats.
    """

    data = np.load(feature_path, allow_pickle=True)
    if "coords" not in data.files:
        raise KeyError(f"Feature file does not contain coords: {feature_path}")

    coords = np.asarray(data["coords"])
    if coords.ndim == 1:
        if len(coords) % 2 != 0:
            raise ValueError(f"Invalid flattened coords in {feature_path}")
        coords = coords.reshape(-1, 2)
    elif coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unsupported coords shape {coords.shape} in {feature_path}")
    return coords.astype(int)


def save_attention_scores(
    bag_id: str,
    attention_weights: torch.Tensor,
    features_root: Path,
    attention_dir: Path,
) -> None:
    
    # Re-loads the features files to get the patch coords
    feature_path = features_root / f"{bag_id}.npz"
    coords = load_coords(feature_path)
    weights = attention_weights.detach().cpu().numpy()

    if len(coords) != len(weights):
        raise ValueError(
            f"Coords/features length mismatch for {bag_id}: coords={len(coords)} attention={len(weights)}"
        )

    # Create a DataFrame with patch_id, x, y, and attention_score columns
    patch_ids = [f"patch_{int(x)}_{int(y)}" for x, y in coords]
    attention_df = pd.DataFrame(
        {
            "bag_id": bag_id,
            "patch_id": patch_ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "attention_score": weights,
        }
    )

    # Save the csv
    attention_dir.mkdir(parents=True, exist_ok=True)
    attention_df.to_csv(attention_dir / f"{bag_id}.csv", index=False)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    features_root: Path,
    output_dir: Path,
) -> tuple[float, float]:
    model.eval()

    attention_dir = output_dir / "attention_scores"
    prediction_rows: list[dict] = []

    for bags, labels, bag_ids in loader:

        # Process one batch of bags

        bags = [b.to(device) for b in bags]
        labels = labels.to(device)

        logits, attn_weights = model(bags) # logits shape (batch_size,), attn_weights is list of (num_patches,) tensors for each bag
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy().astype(int)
        preds = (probs >= 0.5).astype(int)

        for bag_id, label, prob, pred, weights in zip(
            bag_ids, labels_np, probs, preds, attn_weights
        ):
            prediction_rows.append(
                {
                    "bag_id": bag_id,
                    "label": int(label),
                    "pred_prob": float(prob),
                    "pred_label": int(pred),
                }
            )
            save_attention_scores(
                bag_id=bag_id,
                attention_weights=weights,
                features_root=features_root,
                attention_dir=attention_dir,
            )

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    y_true = pred_df["label"].to_numpy()
    y_prob = pred_df["pred_prob"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return acc, auc


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
    model_args, _ = load_checkpoint_config(args.checkpoint_dir)
    model_entry = MODEL_REGISTRY[model_args.model_type]
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
        args.features_root,
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

    acc, auc = run_inference(
        model=model,
        loader=loader,
        device=device,
        features_root=args.features_root,
        output_dir=args.output_dir,
    )

    metrics = {
        "accuracy": acc,
        "auc": auc,
    }
    save_metrics(metrics, args.output_dir)

    print(f"Final acc={acc:.4f} auc={auc:.4f}")
    print(f"Saved predictions to {args.output_dir / 'predictions.csv'}")
    print(f"Saved attention scores to {args.output_dir / 'attention_scores'}")


if __name__ == "__main__":
    main()
