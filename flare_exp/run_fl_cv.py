import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("/home/ubuntu/giodir/digitalPathology/flare_exp/configs/base_config_fed.yaml")
DEFAULT_MANIFEST_NAME = "fl_cad_binary_diff_5cv"
DEFAULT_JOB_NAME = "fl_cad_binary_diff_5cv"
DEFAULT_FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04", "fold_05"]
COMPLETE_MARKER = Path("server/simulate_job/app_server/best_FL_global_model.pt")


def parse_args() -> argparse.Namespace:
    # Parse simple CV launcher options; model/training details stay in the YAML config.
    parser = argparse.ArgumentParser(description="Run one NVFlare MIL job per CV fold.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    # Load the base YAML only to resolve defaults such as output_dir.
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")
    return config


def resolve_output_dir(args: argparse.Namespace, config: dict) -> Path:
    # Resolve the FL output root from CLI first, then config.
    if args.output_dir is not None:
        return args.output_dir
    output_dir = config.get("output_dir")
    if output_dir is None:
        raise ValueError("--output-dir is required when the YAML config has no output_dir.")
    return Path(output_dir)


def resolve_manifest_name(args: argparse.Namespace, config: dict) -> str:
    # Resolve the CV manifest-set parent, stripping a fold suffix if present in the YAML.
    if args.manifest_name is not None:
        return args.manifest_name
    manifest_set = config.get("manifest_set")
    if manifest_set is None:
        return DEFAULT_MANIFEST_NAME
    manifest_path = Path(str(manifest_set))
    if manifest_path.name.startswith("fold_"):
        return str(manifest_path.parent)
    return str(manifest_path)


def resolve_job_name(args: argparse.Namespace, config: dict) -> str:
    # Resolve the shared CV job prefix and W&B group name.
    if args.job_name is not None:
        return args.job_name
    return str(config.get("job_name") or DEFAULT_JOB_NAME)


def validate_expected_output(job_dir: Path, skip_existing: bool) -> bool:
    # Return True if the job should be skipped; otherwise guard incomplete existing outputs.
    if skip_existing and (job_dir / COMPLETE_MARKER).exists():
        print(f"Skipping {job_dir.name}; found {job_dir / COMPLETE_MARKER}")
        return True
    if job_dir.exists() and any(job_dir.iterdir()):
        raise FileExistsError(
            f"Expected FL job output is not empty and is incomplete: {job_dir}. "
            "Use a new --job-name/--output-dir, remove the folder, or use --skip-existing for completed jobs."
        )
    return False


def build_job_command(
    args: argparse.Namespace,
    output_dir: Path,
    manifest_name: str,
    job_name: str,
    fold_name: str,
) -> list[str]:
    # Build one flare_exp.job command with fold-specific manifest and job name.
    fold_job_name = f"{job_name}_{fold_name}"
    return [
        sys.executable,
        "-m",
        "flare_exp.job",
        "--config",
        str(args.config),
        "--manifest-set",
        f"{manifest_name}/{fold_name}",
        "--job-name",
        fold_job_name,
        "--output-dir",
        str(output_dir),
        "--wandb-group",
        job_name,
    ]


def main() -> None:
    # Run all CV folds sequentially.
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_output_dir(args, config)
    manifest_name = resolve_manifest_name(args, config)
    job_name = resolve_job_name(args, config)

    for fold_name in args.folds:
        fold_job_name = f"{job_name}_{fold_name}"
        job_dir = output_dir / fold_job_name
        if validate_expected_output(job_dir, args.skip_existing):
            continue

        command = build_job_command(args, output_dir, manifest_name, job_name, fold_name)
        print(f"Running {fold_name}: {' '.join(command)}")
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
