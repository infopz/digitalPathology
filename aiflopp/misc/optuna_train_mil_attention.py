import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd


#####
# Given a fixed train/val/test split, this script performs hyperparameter optimization for the MIL attention model using Optuna.
# It runs a grid search over the specified hyperparameter search space, training the model for each combination of hyperparameters and evaluating it on the validation set.
#####


TRAIN_SCRIPT = Path("aiflopp/train_mil_attention.py")
OUTPUT_DIR = Path("aiflopp/outputs/optuna_mil_attention")

TRAIN_MANIFEST = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_cad_binary_diff/train_manifest.csv")
VAL_MANIFEST = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_cad_binary_diff/val_manifest.csv")
TEST_MANIFEST = Path("/home/ubuntu/giodir/digitalPathology/data/manifests/afpp_manifest_all_cad_binary_diff/test_manifest.csv")
FEATURES_ROOT = Path("/home/ubuntu/giodir/digitalPathology/data/features/uni_features_RE_all")
HANDCRAFTED_FEATURES_ROOT = Path("/home/ubuntu/giodir/digitalPathology/data/features/ali_handcraft_RE_common_w_names")

FEATURE_MODE = "deep"
EPOCHS = 50
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 7
DEVICE = None # cuda if available, else CPU
PATIENCE = 10
THRESHOLD_METRIC = "balanced_acc"
NUM_CLASSES = 0

OBJECTIVE_SPLIT = "val"
OBJECTIVE_METRIC = "balanced_acc"
DIRECTION = "maximize"


SEARCH_SPACE = {
    "model_type": ["base_mil", "gated_mil"],
    "lr": [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "max_bag_size": [0, 512, 1024, 2048],
    "attention_dim": [64, 128, 256],
    "hidden_dim": [32, 64, 128],
    "dropout": [0.0, 0.25, 0.5],
}


def n_grid_trials(search_space: dict[str, list]) -> int:
    size = 1
    for choices in search_space.values():
        size *= len(choices)
    return int(size)


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        name: trial.suggest_categorical(name, choices)
        for name, choices in SEARCH_SPACE.items()
    }


def add_optional_arg(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_command(params: dict, trial_output_dir: Path, trial_number: int) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-manifest",
        str(TRAIN_MANIFEST),
        "--val-manifest",
        str(VAL_MANIFEST),
        "--test-manifest",
        str(TEST_MANIFEST),
        "--features-root",
        str(FEATURES_ROOT),
        "--feature-mode",
        FEATURE_MODE,
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--num-workers",
        str(NUM_WORKERS),
        "--seed",
        str(SEED + trial_number),
        "--threshold-metric",
        THRESHOLD_METRIC,
        "--num-classes",
        str(NUM_CLASSES),
        "--output-dir",
        str(trial_output_dir),
    ]

    add_optional_arg(command, "--handcrafted-features-root", HANDCRAFTED_FEATURES_ROOT)
    add_optional_arg(command, "--device", DEVICE)
    add_optional_arg(command, "--patience", PATIENCE)

    for name, value in params.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])

    return command


def read_objective(metrics_path: Path) -> float:
    with open(metrics_path) as f:
        metrics = json.load(f)
    return float(metrics[OBJECTIVE_SPLIT][OBJECTIVE_METRIC])


def write_results_csv(study: optuna.Study) -> None:
    rows = []

    for trial in study.trials:
        output_dir = trial.user_attrs.get("output_dir")
        if output_dir is None:
            continue

        metrics_path = Path(output_dir) / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {
            "trial_number": trial.number,
            "objective_value": trial.value,
            "output_dir": output_dir,
            **trial.params,
            "val_balanced_acc": metrics["val"].get("balanced_acc"),
            "val_precision": metrics["val"].get("precision"),
            "val_recall": metrics["val"].get("recall"),
            "val_auc": metrics["val"].get("auc"),
            "test_balanced_acc": metrics["test"].get("balanced_acc"),
            "test_precision": metrics["test"].get("precision"),
            "test_recall": metrics["test"].get("recall"),
            "test_auc": metrics["test"].get("auc"),
        }
        rows.append(row)

    results_path = OUTPUT_DIR / "optuna_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved trial results to {results_path}")


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    total_trials = n_grid_trials(SEARCH_SPACE)
    trial_output_dir = OUTPUT_DIR / f"trial_{trial.number:04d}"
    stdout_path = OUTPUT_DIR / f"trial_{trial.number:04d}_stdout.log"
    stderr_path = OUTPUT_DIR / f"trial_{trial.number:04d}_stderr.log"
    command = build_command(params, trial_output_dir, trial.number)

    print(f"\n[{timestamp()}] Starting trial {trial.number + 1}/{total_trials}")
    print(f"  params: {params}")
    print(f"  output_dir: {trial_output_dir}")

    with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
        result = subprocess.run(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    trial.set_user_attr("output_dir", str(trial_output_dir))
    trial.set_user_attr("stdout_log", str(stdout_path))
    trial.set_user_attr("stderr_log", str(stderr_path))

    if result.returncode != 0:
        print(f"[{timestamp()}] Finished trial {trial.number + 1}/{total_trials}: FAILED")
        raise RuntimeError(f"Training failed. See {stderr_path}")

    value = read_objective(trial_output_dir / "metrics.json")
    print(
        f"[{timestamp()}] Finished trial {trial.number + 1}/{total_trials}: "
        f"{OBJECTIVE_SPLIT}.{OBJECTIVE_METRIC}={value:.6f}"
    )
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.GridSampler(SEARCH_SPACE)
    study = optuna.create_study(direction=DIRECTION, sampler=sampler)
    study.optimize(objective, n_trials=n_grid_trials(SEARCH_SPACE))

    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value: {study.best_trial.value:.6f}")
    print(f"  params: {study.best_trial.params}")
    print(f"  output_dir: {study.best_trial.user_attrs.get('output_dir')}")
    write_results_csv(study)


if __name__ == "__main__":
    main()
