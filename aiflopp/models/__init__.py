from .base_mil import build_base_mil, get_base_mil_config, validate_base_mil_args
from .gated_mil import build_gated_mil, get_gated_mil_config, validate_gated_mil_args


MODEL_REGISTRY = {
    "base_mil": {
        "build": build_base_mil,
        "config": get_base_mil_config,
        "validate": validate_base_mil_args,
    },
    "gated_mil": {
        "build": build_gated_mil,
        "config": get_gated_mil_config,
        "validate": validate_gated_mil_args,
    },
}


AVAILABLE_MODEL_TYPES = tuple(sorted(MODEL_REGISTRY))
