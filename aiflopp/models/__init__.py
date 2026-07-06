from .base_mil import build_base_mil, validate_base_mil_args, AttentionMILBase
from .gated_mil import build_gated_mil, validate_gated_mil_args, AttentionMILGated


MODEL_REGISTRY = {
    "base_mil": {
        "build": build_base_mil,
        "validate": validate_base_mil_args,
        "class": AttentionMILBase,
    },
    "gated_mil": {
        "build": build_gated_mil,
        "validate": validate_gated_mil_args,
        "class": AttentionMILGated,
    },
}


AVAILABLE_MODEL_TYPES = tuple(sorted(MODEL_REGISTRY))
