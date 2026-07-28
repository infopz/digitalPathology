import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:

    default_train_maifest = Path("/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/train_manifest.csv")
    default_val_maifest = Path("/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/val_manifest.csv")
    default_test_maifest = Path("/home/ubuntu/giodir/digitalPathology/aiflopp/manifest/test_manifest.csv")

    default_features_root = Path("data/uni_features_RE_common")

    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on pooled subregion features."
    )
    parser.add_argument("--train-manifest", type=Path, default=default_train_maifest, required=False)
    parser.add_argument("--val-manifest", type=Path, default=default_val_maifest, required=False)
    parser.add_argument("--test-manifest", type=Path, default=default_test_maifest, required=False)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=default_features_root,
        help="Root folder containing per-patient feature npz files.",
    )
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="If > 0, apply PCA to this number of components.",
    )
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="XGBoost max_depth.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="XGBoost n_estimators.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning_rate.",
    )
    return parser.parse_args()


def load_features(manifest: pd.DataFrame, features_root: Path, pooling: str):
    X: list[np.ndarray] = []
    y: list[int] = []
    ids: list[str] = []

    for _, row in manifest.iterrows():

        bag_id = row["bag_id"]

        label = int(row["label"])
        feature_path = (
            features_root / f"{bag_id}.npz"
        )

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features: {feature_path}")

        data = np.load(feature_path, allow_pickle=True)
        feats = data["features"]

        if pooling == "mean":
            pooled = feats.mean(axis=0)
        else:
            pooled = feats.max(axis=0)

        X.append(pooled.astype(np.float32))
        y.append(label)
        ids.append(bag_id)

    return np.vstack(X), np.array(y), ids


def build_model(args):

    return XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
    )


def evaluate(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, auc


def main() -> None:
    args = parse_args()

    train_manifest = pd.read_csv(args.train_manifest)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)

    required_cols = {"bag_id", "label"}
    for name, manifest in [
        ("train", train_manifest),
        ("val", val_manifest),
        ("test", test_manifest),
    ]:
        missing = required_cols - set(manifest.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {missing}")

    X_train, y_train, _ = load_features(train_manifest, args.features_root, args.pooling)
    X_val, y_val, _ = load_features(val_manifest, args.features_root, args.pooling)
    X_test, y_test, _ = load_features(test_manifest, args.features_root, args.pooling)

    if args.pca_components > 0:
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

    model = build_model(args)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_acc, val_auc = evaluate(y_val, val_prob)
    test_acc, test_auc = evaluate(y_test, test_prob)

    print(f"Val acc={val_acc:.4f} auc={val_auc:.4f}")
    print(f"Test acc={test_acc:.4f} auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
