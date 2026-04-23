#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

for fold_dir in "$ROOT_DIR"/aiflopp/manifest_folds/fold_*; do
    echo "================================"
    echo "Running fold: $fold_dir"
    python "$ROOT_DIR/aiflopp/train_mil_attention.py" \
        --train-manifest "$fold_dir/train_manifest.csv" \
        --val-manifest "$fold_dir/val_manifest.csv" \
        --test-manifest "$fold_dir/test_manifest.csv" \
        "$@"
done
