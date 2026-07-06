import argparse
import shlex
from pathlib import Path

import yaml

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_cross_site_evaluation

from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY
from aiflopp.train_mil_attention import validate_output_dir


CLIENT_ARGS = (
    "features_root",
    "output_dir",
    "model_type",
    "attention_dim",
    "hidden_dim",
    "dropout",
    "input_dim",
    "num_classes",
    "output_dim",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "max_bag_size",
    "num_workers",
    "seed",
    "threshold_metric"
)


def load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a mapping of argument names to values.")
    return config


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file.",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Create the NVFlare job for MIL attention training.",
        parents=[config_parser],
    )

    # Job/server settings
    parser.add_argument(
        "--client-list",
        nargs="+",
        default=["trento_client", "reggio_client"],
        help="Client site names expected to participate in each round.",
    )
    parser.add_argument("--num-rounds", "--num_rounds", dest="num_rounds", type=int, default=8)
    parser.add_argument("--train-script", type=str, default="flare-exp/client.py")
    parser.add_argument("--job-name", type=str, default="mil-fedavg")
    parser.add_argument(
        "--cross-site-eval",
        dest="cross_site_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--key-metric",
        type=str,
        default="balanced_accuracy",
        help="Metric name sent by clients and used by the server to select the best global model.",
    )

    # Client settings resolved here and forwarded as explicit CLI args.
    parser.add_argument("--features-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=AVAILABLE_MODEL_TYPES,
        default="base_mil",
    )
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-bag-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--threshold-metric",
        type=str,
        choices=("acc", "precision", "recall", "f2", "balanced_acc", "auc"),
        default="balanced_acc",
    )

    config = load_config(config_args.config)
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        parser.error(f"Unknown config option(s): {', '.join(unknown_keys)}")
    parser.set_defaults(**config)

    args = parser.parse_args()
    if args.features_root is None:
        parser.error("--features-root is required, either in YAML as features_root or on the CLI.")
    if args.output_dir is None:
        parser.error("--output-dir is required, either in YAML as output_dir or on the CLI.")
    return args


def build_client_train_args(args: argparse.Namespace) -> str:
    # Build the list of client training args to be passed to the client.py script via CLI
    client_args = []
    for attr_name in CLIENT_ARGS:
        cli_name = f"--{attr_name.replace('_', '-')}"
        value = getattr(args, attr_name)

        if cli_name == "--output-dir":
            # Append job name to output dir for each client
            value = Path(value) / args.job_name

        client_args.extend([cli_name, str(value)])
    return shlex.join(client_args)


def build_model_config(args: argparse.Namespace) -> dict:
    # Parse model args and return the model config dict to be passed to the NVFlare recipe.
    if args.model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type {args.model_type!r}. "
            f"Expected one of: {', '.join(AVAILABLE_MODEL_TYPES)}"
        )

    # Validate model args and get the model class
    model_entry = MODEL_REGISTRY[args.model_type]
    model_class = model_entry["class"]
    required_args = model_entry["validate"](args)

    # Build a dict with the required model args, used later to build the model
    model_args = {arg_name: getattr(args, arg_name) 
                  for arg_name in required_args}

    return {
        "class_path": f"{model_class.__module__}.{model_class.__qualname__}",
        "args": model_args,
    }


def main() -> FedAvgRecipe:
    args = parse_args()

    args.output_dir = validate_output_dir(args.output_dir)
     
    # Generate the client training args
    client_train_args = build_client_train_args(args)

    print(f"Client training args: {client_train_args}")

    # Parse other args to extract model config
    model_config = build_model_config(args)

    recipe = FedAvgRecipe(
        name=args.job_name,
        min_clients=len(args.client_list),
        num_rounds=args.num_rounds,
        model=model_config,
        train_script=args.train_script,
        train_args=client_train_args,
        key_metric=args.key_metric,
    )

    if args.cross_site_eval:
        add_cross_site_evaluation(recipe, args.client_list)

    env = SimEnv(
        clients=args.client_list,
        workspace_root=str(args.output_dir)
    )
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()



if __name__ == "__main__":
    main()
