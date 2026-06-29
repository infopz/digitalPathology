from pathlib import Path
import argparse
import json
import yaml

from torch.utils.data import DataLoader
import nvflare.client as flare
import pandas as pd
import torch

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY
from aiflopp.train_mil_attention import (
    collect_predictions,
    compute_pos_weight,
    evaluate,
    print_metrics,
    save_predictions,
    save_model_and_metadata,
    search_best_threshold,
    seed_everything,
    train,
    validate_output_dir
) 


MANIFEST_PATH = {
    "reggio_client": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/reggio_only/afpp_manifest_all_base"),
    "trento_client": Path("/home/ubuntu/giodir/digitalPathology/data/manifests/trento_only/afpp_manifest_tn_base")
}


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/client_config.yaml"),
        help="Path to the client configuration file.",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train a MIL attention model on subregion patch features.",
        parents=[config_parser],
    )

    # Input/output paths
    parser.add_argument(
        "--features-root",
        type=Path,
        help="Root folder containing deep per-bag feature npz files.",
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        help="Directory to save model checkpoints and logs."
    )

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=float, default=10, help="Number of epochs to wait for improvement.")
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


def evaluate_given_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    json_out_path: Path | None = None,
    csv_out_folder: Path | None = None
):
    """
    Evaluate the given model on the validation and test sets.
    First, search for the best decision threshold using the validation set, 
    then evaluate on the test set using that threshold.
    
    If json_out_path is provided, save the evaluation metrics to a JSON file.
    If csv_out_folder is provided, save the predictions for both validation and test sets as CSV files in that folder.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the evaluation on.
        args (argparse.Namespace): Command-line arguments containing evaluation settings.
        json_out_path (Path | None): Optional path to save evaluation metrics as JSON.
        csv_out_folder (Path | None): Optional folder to save predictions as CSV files.
    
    Returns:
        test_metrics (dict): Evaluation metrics on the test set.
    """

    # Search for best threshold using val set
    val_bag_ids, val_y_true, val_y_prob = collect_predictions(
        model,
        val_loader,
        device,
        args.num_classes,
    )

    best_threshold, val_metrics, val_threshold_search = search_best_threshold(
        val_y_true,
        val_y_prob,
        objective=args.threshold_metric,
    )

    # If csv_out_folder is provided, save predictions for val
    # and set test_csv_path for saving test predictions later
    test_csv_path = None
    if csv_out_folder is not None:
        csv_out_folder.mkdir(parents=True, exist_ok=True)

        val_csv_path = csv_out_folder / "val_predictions.csv"
        save_predictions(
            val_bag_ids,
            val_y_true,
            val_y_prob,
            threshold=best_threshold,
            num_classes=args.num_classes,
            output_csv_path=val_csv_path,
        )

        test_csv_path = csv_out_folder / "test_predictions.csv"
        
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=best_threshold,
        num_classes=args.num_classes,
        output_csv_path=test_csv_path
    )

    if json_out_path is not None:
        metrics = {
            "num_classes": args.num_classes,
            "decision_threshold": best_threshold,
            "threshold_metric": args.threshold_metric,
            "val": val_metrics,
            "test": test_metrics,
        }
        with open(json_out_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved evaluation metrics to {json_out_path}")
    
    return test_metrics


def main():
    args = parse_args()
    args.output_dir = validate_output_dir(args.output_dir)

    # Fixed args
    args.num_classes = 2
    args.output_dim = 1
    args.input_dim = 1536

    seed_everything(args.seed)

    device = torch.device(args.device)

    # Initalize Flare and get client_name
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    print(f"{client_name} - Using device: {device}")

    args.output_dir = args.output_dir / client_name

    # Load manifest based on client_name
    base_manifest_path = MANIFEST_PATH.get(client_name)
    train_manifest = pd.read_csv(base_manifest_path / "train_manifest.csv")
    val_manifest = pd.read_csv(base_manifest_path / "val_manifest.csv")
    test_manifest = pd.read_csv(base_manifest_path / "test_manifest.csv")

    # Select model_type and validate args
    model_entry = MODEL_REGISTRY[args.model_type]
    model_entry["validate"](args)

    # Compute pos weight
    loss_weight = compute_pos_weight(train_manifest, device)
    args.pos_weight = float(loss_weight.item())
    args.class_weights = None
    args.decision_threshold = 0.5

    # Build model based on specified model_type
    model = model_entry["build"](args).to(device)

    # Create datasets and dataloaders
    train_ds = MILBagDataset(
        train_manifest,
        args.features_root,
        max_bag_size=args.max_bag_size,
        enable_sampling=True,
    )
    val_ds = MILBagDataset(
        val_manifest,
        args.features_root,
        max_bag_size=args.max_bag_size,
        enable_sampling=False,
    )
    test_ds = MILBagDataset(
        test_manifest,
        args.features_root,
        max_bag_size=args.max_bag_size,
        enable_sampling=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
    )

    while flare.is_running():

        if flare.is_submit_model():
            # Submit model to server
            # TODO: probabilmente da fixare perche non gli devo dare l'ultimo (che potrebbe avere dei pesi random)
            #       ma caricarmi il migliore dal pth
            flare.submit_model(flare.FLModel(params=model.cpu().state_dict()))
            continue

        # Receive and load the model
        input_model = flare.receive()
        model.load_state_dict(input_model.params)
        model.to(device)

        round_num = input_model.current_round
        print(f"{client_name} - Received model for round {round_num}")

        if flare.is_evaluate():
            # Evaluate the received model
            test_metrics = evaluate_given_model(
                model,
                val_loader,
                test_loader,
                device,
                args,
            )
            output_model = flare.FLModel(metrics={"balanced_accuracy": test_metrics["balanced_accuracy"]})
            flare.send(output_model)
            continue

        # TODO: reuse the last lr and other hyperparameters from the previous round, or reset them?

        # Train model
        model = train(model, train_loader, val_loader, args, device, loss_weight, best_metric=args.threshold_metric)

        # Evaluate
        round_out_dir = args.output_dir / f"round_{round_num}"
        round_out_dir.mkdir(parents=True, exist_ok=True)

        test_metrics = evaluate_given_model(
            model,
            val_loader,
            test_loader,
            device,
            args,
            json_out_path=round_out_dir / "evaluation_metrics.json",
            csv_out_folder=round_out_dir / "predictions",
        )

        print(f"{client_name} - Round {round_num} evaluation metrics:")
        print_metrics(test_metrics)

        # Save model and metrics
        save_model_and_metadata(model, round_out_dir, args)

        # Send the model and metrics back to the server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"balanced_accuracy": test_metrics["balanced_accuracy"]},
        )
        flare.send(output_model)

    
if __name__ == "__main__":
    main()
