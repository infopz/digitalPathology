import argparse
import builtins
import json
import math
from pathlib import Path
from typing import Any

from nvflare.client.tracking import WandBWriter
import wandb

from aiflopp.train_mil_attention import METRIC_CHOICES


def optional_metric(value: str | None) -> str | None:
    # Parse a metric arg that can be None or one of the valid metric choices
    if value is None or value.lower() in {"none", "null"}:
        return None
    if value not in METRIC_CHOICES:
        raise argparse.ArgumentTypeError(
            f"invalid metric {value!r}; expected one of {', '.join(METRIC_CHOICES)} or null"
        )
    return value


def parse_bool(value: str | bool) -> bool:
    # Parse boolean args
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def prefix_prints(prefix: str) -> None:
    # Override built-in print func to add client name prefix to all prints
    base_print = builtins.print

    def prefixed_print(*args, **kwargs):
        base_print(f"[{prefix}]", *args, **kwargs)

    builtins.print = prefixed_print


# HELPER FUNCTIONS FOR TRACKING METRICS AND CONF

def prepare_tracking_metrics(prefix: str, metrics: dict) -> dict:
    # Given the metrics dict and the prefix, prepare a new dict with the prefixed metrics
    log_metrics = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
            log_metrics[f"{prefix}/{name}"] = float(value)
    return log_metrics


def log_metrics(writer: WandBWriter | None, metrics: dict, step: int) -> None:
    # Call the writer's log method if available
    if writer is not None and metrics:
        writer.log(metrics, step=step)


def train_log_callback(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    round_step_base: int,
    round_num: int,
    tracking_writer: WandBWriter | None
) -> None:
    # Callback function passed to train loop func to log metrics at the end of each epoch

    step = round_step_base + epoch
    metrics = {
        "round": round_num,
        "local_epoch": epoch,
        **prepare_tracking_metrics("local/train", train_metrics),
        **prepare_tracking_metrics("local/val", val_metrics),
    }
    log_metrics(tracking_writer, metrics, step=step)


def yaml_safe(value: Any) -> Any:
    # Convert Path to str, recursively
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: yaml_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(item) for item in value]
    return value


def build_resolved_config(args: argparse.Namespace) -> dict:
    # Build a dict of the resolved config values from the argparse.Namespace, excluding the "config" key
    # Used to export the resolved config to WandB
    return {
        key: yaml_safe(value)
        for key, value in vars(args).items()
        if key != "config"
    }


CROSS_SITE_TABLE_METRICS = (
    "balanced_acc",
    "auc",
    "precision",
    "recall",
    "recall_0",
    "f2",
    "acc",
    "threshold",
)

def build_wandb_table_rows(site_results: dict) -> list[list]:
    # Convert the cross-site results dict to list of rows for WandB logging
    rows = []
    for model_name, model_results in site_results.items():
        row = [model_name]
        for split_name in ("val", "test"):
            split_metrics = model_results.get(split_name, {})
            for metric_name in CROSS_SITE_TABLE_METRICS:
                value = split_metrics.get(metric_name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    row.append(float(value))
                else:
                    row.append(None)
        rows.append(row)
    return rows


def log_cross_site_tables_to_wandb(
    args: argparse.Namespace,
    resolved_config: dict,
    result_dir: Path,
    wandb_group: str,
) -> None:
    # Called at the end of the training,
    # logs the cross-site evalutation to wandb as summary table + config
    
    if not args.enable_tracking:
        return

    cross_val_path = result_dir / "server" / "simulate_job" / "cross_site_val" / "cross_val_results.json"
    if not cross_val_path.exists():
        print(f"Cross-site results not found at {cross_val_path}; skipping W&B summary tables.")
        return

    with cross_val_path.open("r") as f:
        cross_val_results = json.load(f)

    wandb.login(timeout=1, verify=True)

    wandb_args = {
        "project": args.wandb_project,
        "name": f"{args.job_name}-cross-site-summary",
        "group": wandb_group,
        "job_type": "cross_site_summary",
        "config": resolved_config,
        "mode": "online",
    }

    columns = ["model"]
    for split_name in ("val", "test"):
        columns.extend(f"{split_name}_{metric_name}" for metric_name in CROSS_SITE_TABLE_METRICS)

    try:
        with wandb.init(**wandb_args) as run:
            for site_name, site_results in cross_val_results.items():
                table = wandb.Table(
                    columns=columns,
                    data=build_wandb_table_rows(site_results),
                )
                run.log({f"cross_site/{site_name}": table})
            run.save(str(cross_val_path), policy="now")
    except Exception as exc:
        print(f"Failed to log W&B summary tables: {exc}")
