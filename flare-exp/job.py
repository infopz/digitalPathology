import argparse
import shlex
from pathlib import Path

import yaml

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_cross_site_evaluation

from aiflopp.models import AVAILABLE_MODEL_TYPES, MODEL_REGISTRY


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Create the NVFlare FedAvg job for MIL attention training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config used locally for the initial model and forwarded to clients.",
    )
    parser.add_argument(
        "--client-list",
        nargs="+",
        default=["trento_client", "reggio_client"],
        help="Client site names expected to participate in each round.",
    )
    parser.add_argument("--num-rounds", "--num_rounds", dest="num_rounds", type=int, default=8)
    parser.add_argument("--train-script", type=str, default="flare-exp/client.py")
    parser.add_argument("--job-name", type=str, default="mil-fedavg")
    parser.add_argument("--cross-site-eval", "--cross_site_eval", action="store_true")

    # client_args contains the arguments that will be forwarded to the client.py script
    args, client_args = parser.parse_known_args()
    return args, client_args


def parse_model_args(config: dict, client_args: list[str]) -> argparse.Namespace:
    """
    Parse the model-related arguments from the config and client_args.
    The config is a dictionary loaded from the YAML file, and client_args is a list of
    command-line arguments that will be forwarded to the client.py script.

    Return full args with defaults from config and overrides from client_args.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=AVAILABLE_MODEL_TYPES,
        default="base_mil",
    )
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)

    model_keys = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in config.items() if key in model_keys})
    args, _ = parser.parse_known_args(client_args)

    if args.model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type {args.model_type!r}. "
            f"Expected one of: {', '.join(AVAILABLE_MODEL_TYPES)}"
        )

    # Keep the server-side initial model shape aligned with the current client setup.
    args.input_dim = 1536
    args.num_classes = 2
    args.output_dim = 1
    return args


def build_initial_model(model_args: argparse.Namespace) -> object:
    model_entry = MODEL_REGISTRY[model_args.model_type]
    model_entry["validate"](model_args)
    model = model_entry["build"](model_args)

    return model


def build_nvflare_model_config(model: object, model_args: argparse.Namespace) -> dict:
    # TODO: fix momentaneo. data la classe instanziata, splitta il nome della classe dai parametri
    #       cosi da poterlo inizializzare nuovamente. capire come gestire meglio questa cosa senza dover usare questa funzione
    model_class = model.__class__
    return {
        "class_path": f"{model_class.__module__}.{model_class.__qualname__}",
        "args": {
            "input_dim": model_args.input_dim,
            "attention_dim": model_args.attention_dim,
            "hidden_dim": model_args.hidden_dim,
            "dropout": model_args.dropout,
            "output_dim": model_args.output_dim,
        },
    }


def build_client_train_args(config_path: Path, client_args: list[str]) -> str:
    return shlex.join(["--config", str(config_path), *client_args])


def main() -> FedAvgRecipe:
    args, client_args = parse_args()

    with args.config.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a mapping of argument names to values.")
    
    # Generate the client training args
    client_train_args = build_client_train_args(args.config, client_args)

    print(f"Client training args: {client_train_args}")

    # Parse other args to build the initial model
    full_args = parse_model_args(config, client_args)
    initial_model = build_initial_model(full_args)
    nvflare_model_config = build_nvflare_model_config(initial_model, full_args)

    recipe = FedAvgRecipe(
        name=args.job_name,
        min_clients=len(args.client_list),
        num_rounds=args.num_rounds,
        model=nvflare_model_config,
        train_script=args.train_script,
        train_args=client_train_args,
    )

    if args.cross_site_eval:
        add_cross_site_evaluation(recipe, args.client_list)

    env = SimEnv(clients=args.client_list)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()



if __name__ == "__main__":
    main()
