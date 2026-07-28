import argparse
from functools import partial
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import DataLoader
import nvflare.client as flare
from nvflare.client.tracking import WandBWriter
import pandas as pd
import torch

from aiflopp.datasets import MILBagDataset, collate_bags
from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY
from aiflopp.train_mil_attention import (
    collect_predictions,
    compute_pos_weight,
    evaluate,
    is_multiclass_task,
    print_metrics,
    save_predictions,
    save_model,
    search_best_threshold,
    seed_everything,
    train,
    METRIC_CHOICES
) 
from flare_exp.fl_utils import (
    optional_metric, 
    parse_bool, 
    prefix_prints, 
    prepare_tracking_metrics, 
    log_metrics,
    train_log_callback,
)

CLIENT_ENV_FILE = "/home/ubuntu/giodir/digitalPathology/flare_exp/client_secrets.env"
load_dotenv(CLIENT_ENV_FILE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MIL attention model on subregion patch features.",
    )

    # Input/output paths
    parser.add_argument(
        "--features-root",
        type=Path,
        required=True,
        help="Root folder containing deep per-bag feature npz files.",
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory to save model checkpoints and logs."
    )
    parser.add_argument(
        "--manifest-set",
        type=str,
        required=True,
        help="Name of the manifest set to use.",
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
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output-dim", type=int, default=1)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement.")
    parser.add_argument(
        "--max-bag-size",
        type=int,
        default=0,
        help="If >0, randomly subsample each bag to this many patches to stabilize batches.",
    )

    # Other settings
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--key-metric",
        type=str,
        choices=METRIC_CHOICES,
        default="balanced_acc",
        help="Metric sent to the server at the end of each round.",
    )
    parser.add_argument(
        "--eval-threshold-metric",
        type=str,
        choices=METRIC_CHOICES,
        default="balanced_acc",
        help="Validation metric used to choose the final decision threshold.",
    )
    parser.add_argument(
        "--epoch-selection-metric",
        type=optional_metric,
        default=None,
        help="Metric used to select the best local epoch; null disables epoch selection.",
    )
    parser.add_argument("--enable-tracking", type=parse_bool, default=True)
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
    val_bag_ids, val_y_true, val_y_prob, _ = collect_predictions(
        model,
        val_loader,
        device,
        args.num_classes,
    )

    best_threshold, val_metrics, val_threshold_search = search_best_threshold(
        val_y_true,
        val_y_prob,
        objective=args.eval_threshold_metric,
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
            "eval_threshold_metric": args.eval_threshold_metric,
            "val": val_metrics,
            "test": test_metrics,
        }
        with open(json_out_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved evaluation metrics to {json_out_path}")
    
    return test_metrics, val_metrics


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initalize Flare and get client_name
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    prefix_prints(client_name)

    # Initialize WanDB tracking
    tracking_writer = None
    if args.enable_tracking:
        print("Initializing WandB tracking...")
        tracking_writer = WandBWriter()

    print(f"Using device: {device}")

    args.output_dir = args.output_dir / client_name / "results"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load manifest based on client_name and manifest_set
    manifest_root_path = Path(os.getenv("MANIFEST_ROOT_PATH"))
    base_manifest_path = manifest_root_path / client_name / args.manifest_set
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
    if is_multiclass_task(args.num_classes):
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

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

        # Receive the model/state from the server
        input_model = flare.receive()

        if flare.is_submit_model():
            # Submit model to server
            print("Received submit model request from server.")
            flare.send(flare.FLModel(params=model.cpu().state_dict()))
            continue

        # Load received model
        model.load_state_dict(input_model.params)
        model.to(device)

        if flare.is_evaluate():
            print("Received evaluate request from server.")
            # Evaluate the received model
            test_metrics, val_metrics = evaluate_given_model(
                model,
                val_loader,
                test_loader,
                device,
                args,
            )
            print(f"Requested evaluation metrics:")
            print_metrics(val_metrics, split_name="val", compact=True)
            print_metrics(test_metrics, split_name="test", compact=True)

            full_metrics = {
                "val": val_metrics,
                "test": test_metrics,
            }
            output_model = flare.FLModel(metrics=full_metrics)
            flare.send(output_model)
            continue

        round_num = input_model.current_round
        print(f"Received model for round {round_num}")

        # Evaluate the received global model before local updates to track site shift.
        global_val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            threshold=0.5,
            num_classes=args.num_classes,
        )
        print(f"Round {round_num} received global model metrics:")
        print_metrics(global_val_metrics, split_name="global_val", compact=True)
        round_step_base = round_num * (args.epochs + 2)
        tracking_metrics = {
            "round": round_num,
            **prepare_tracking_metrics(
                "global_received/val",
                global_val_metrics,
            ),
        }
        log_metrics(tracking_writer, tracking_metrics, step=round_step_base)

        # TODO: implement a server-side scheduler for the learning rate, weight decay, and other hyperparameters if needed.

        # Train model
        model = train(
            model, 
            train_loader, 
            val_loader, 
            args, 
            device, 
            criterion, 
            best_metric=args.epoch_selection_metric,
            hide_progress=True,
            log_callback=partial(
                train_log_callback,
                round_step_base=round_step_base,
                round_num=round_num,
                tracking_writer=tracking_writer,
            ),
        )

        # Evaluate
        round_out_dir = args.output_dir / f"round_{round_num}"
        round_out_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate the model on the val test with fixed threshold
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            threshold=0.5,
            num_classes=args.num_classes,
        )

        print(f"Round {round_num} evaluation metrics:")
        print_metrics(val_metrics, split_name="val", compact=True)
        log_metrics(
            tracking_writer,
            {
                "round": round_num,
                **prepare_tracking_metrics("local_post/val", val_metrics),
            },
            step=round_step_base + args.epochs + 1,
        )

        # Save model and metrics
        save_model(model, round_out_dir / "model.pth")

        # Send the model and metrics back to the server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics=val_metrics,
        )
        flare.send(output_model)

    
if __name__ == "__main__":
    main()
