# Digital Pathology Project Recap

This project implements a digital pathology workflow for bag-level classification from whole-slide image patch features. The main active package is `aiflopp`, which focuses on Multiple Instance Learning (MIL) over bags of patch embeddings, with optional handcrafted features.

## Main Goal

The project trains and evaluates models that predict slide/subregion-level labels from pre-extracted patch features. Each bag corresponds to a pathology region or slide-derived unit, and each bag contains multiple patch-level feature vectors.

The pipeline is designed to avoid patient leakage by splitting data at patient level rather than at bag level.

## Implemented Components

### Data Preparation

The preprocessing utilities are in `aiflopp/preprocessing`.

- `extract_bag_features.py` extracts or organizes patch-level features into bag-level `.npz` files.
- `split_train_test.py` creates patient-level train/validation/test manifests.
- `split_shared_test_folds.py` creates multiple train/validation folds with a shared patient-level test set for cross-validation.

Generated manifest files contain:

- `bag_id`
- `patient_id`
- `subregion_id`
- `label`

### Feature Loading

Feature utilities are implemented in `aiflopp/feature_utils.py`.

Supported feature modes:

- `deep`: use deep patch embeddings only.
- `handcrafted`: use handcrafted patch features only.
- `concat`: concatenate deep and handcrafted features.

When handcrafted features are used, a `StandardScaler` is fitted on the training split and reused for validation, test, and inference.

### MIL Models

MIL models are implemented in `aiflopp/models`.

Currently available model types:

- `base_mil`: attention-based MIL model.
- `gated_mil`: gated-attention MIL model.

Both models aggregate patch-level features into a bag representation using learned attention weights, then classify the bag.

### Training

The main training script is:

```bash
aiflopp/train_mil_attention.py
```

It supports:

- Binary and multiclass classification.
- Class weighting for imbalanced labels.
- Early stopping.
- Validation-based threshold tuning for binary classification.
- Saving predictions for validation and test splits.
- Saving metrics to `metrics.json`.
- Saving the trained model to `best_model.pth`.
- Saving the resolved reusable training config to `config.yaml`.
- Saving `handcrafted_scaler.npz` when handcrafted features are used.

The script accepts either named CLI arguments or a YAML config file:

```bash
uv run python -m aiflopp.train_mil_attention \
  --config aiflopp/configs/base_config.yaml \
  --epochs 50 \
  --lr 5e-4
```

CLI arguments override values loaded from the YAML config.

### Cross-Validation

Cross-validation is implemented in:

```bash
aiflopp/train_mil_attention_cv.py
```

This wrapper runs `train_mil_attention.py` once per fold. Each fold directory is expected to contain:

```text
train_manifest.csv
val_manifest.csv
test_manifest.csv
```

The test manifest can be repeated inside each fold folder. The wrapper aggregates fold metrics and writes:

- `cv_summary.json`
- `cv_summary.csv`

Example:

```bash
uv run python -m aiflopp.train_mil_attention_cv \
  --config aiflopp/configs/base_config.yaml \
  --folds-dir data/manifests/reggio_only/afpp_manifest_folds \
  --output-dir aiflopp/outputs/reggio_cv_run
```

### Inference

Inference is implemented in:

```bash
aiflopp/infer_mil_attention.py
```

It loads a trained checkpoint folder containing:

```text
best_model.pth
config.yaml
metrics.json
handcrafted_scaler.npz  # only for handcrafted/concat models
```

Inference outputs:

- `predictions.csv`
- `metrics.json`
- per-bag attention score CSV files under `attention_scores/`

The inference script reads the trained config from `config.yaml`, infers the feature input dimension from the inference manifest/features, and loads `handcrafted_scaler.npz` when needed.

### Attention Visualization

Attention heatmap plotting is implemented in:

```bash
aiflopp/plot_attention_heatmap.py
```

It overlays attention scores on WSI thumbnails.

Supported modes:

- Plot one bag with `--bag-id`.
- Randomly sample bags with `--num-sampled-bag`.
- Plot all bags if neither option is provided.

Example:

```bash
uv run python -m aiflopp.plot_attention_heatmap \
  --attention-dir aiflopp/outputs_inference/test_model_wnames/attention_scores \
  --num-sampled-bag 5
```

## Config Files

Reusable YAML configs are stored in:

```text
aiflopp/configs/
```

Configs are intentionally flat and contain only arguments accepted by `train_mil_attention.py`. Computed artifacts such as `input_dim`, `output_dim`, decision thresholds, class weights, and handcrafted scaler values are not stored in the config.

## Typical Workflow

1. Extract or prepare bag-level `.npz` feature files.
2. Create patient-level manifests with `split_train_test.py` or `split_shared_test_folds.py`.
3. Train a MIL model with `train_mil_attention.py` or run cross-validation with `train_mil_attention_cv.py`.
4. Evaluate saved metrics and predictions.
5. Run inference on new manifests with `infer_mil_attention.py`.
6. Visualize attention maps with `plot_attention_heatmap.py`.

## Current Output Artifacts

Single training runs produce:

```text
best_model.pth
config.yaml
metrics.json
val_predictions.csv
test_predictions.csv
handcrafted_scaler.npz  # when needed
```

Cross-validation runs produce one such folder per fold plus aggregate summaries.
