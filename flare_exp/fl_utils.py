import argparse
import builtins

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


def prefix_prints(prefix: str) -> None:
    # Override built-in print func to add client name prefix to all prints
    base_print = builtins.print

    def prefixed_print(*args, **kwargs):
        base_print(f"[{prefix}]", *args, **kwargs)

    builtins.print = prefixed_print