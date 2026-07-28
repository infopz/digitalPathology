# FL vs. Non-FL Evaluation Protocol - Decisions & Rationale

Context document for the federated MIL / inter-observer discordance project.
Records the current evaluation design and the reasoning behind it.

---

## 1. Problem Being Solved

Data splits are highly sensitive: metrics computed on a single split are biased by that
particular draw. The non-FL baseline handled this with k-fold CV over pooled data. Moving to a
federated setup, currently with **2 clients**, the goal is an evaluation protocol that is:

1. Robust to split choice.
2. Fully comparable between FL and non-FL models.
3. Explicit about whether a model is evaluated on merged data or client-specific data.

The central comparison is not only "FL vs. centralized on merged data". Since the FL setup is
intended to model client-specific adaptation, the evaluation must also report performance on each
client separately.

---

## 2. Cross-Validation Structure

The current setup uses **5 folds**.

Each fold has its own train, validation, and test split. Test folds rotate, so each case appears
in the test split exactly once across the 5 folds for a given dataset view.

The reduced 5-fold design was chosen deliberately: fewer folds than 7, but larger validation and
test slices per fold. This matters because threshold selection is done on validation data and the
per-client validation sets can be small.

---

## 3. Fold Construction

Folds are built per client, then merged upward to obtain the centralized/global view.

```text
global fold k = client A fold k + client B fold k
```

This property is required for a fair comparison:

```text
centralized train fold k = union of client train fold k
centralized val fold k   = union of client val fold k
centralized test fold k  = union of client test fold k
```

The current client CV folders are:

```text
data/manifests/reggio_client/fl_cad_binary_diff_5cv
data/manifests/trento_client/fl_cad_binary_diff_5cv
```

The merged/global CV folder is:

```text
data/manifests/mergedRT/fl_cad_binary_diff_5cv
```

The merged folder is created by:

```text
aiflopp/preprocessing/merge_client_cv_folds.py
```

The merge script concatenates matching `fold_XX/train_manifest.csv`, `val_manifest.csv`, and
`test_manifest.csv` files across clients, preserving one CSV header.

Adding a third client should only require adding another client fold directory to the merge script
and to the evaluation dictionaries.

---

## 4. Comparison Design

The evaluation matrix is:

| Model | Trained On | Evaluated On |
|---|---|---|
| non-FL centralized | merged train folds | merged val/test folds |
| non-FL centralized | merged train folds | Reggio val/test folds |
| non-FL centralized | merged train folds | Trento val/test folds |
| FL global | federated client train folds | Reggio val/test folds |
| FL global | federated client train folds | Trento val/test folds |
| FL global | federated client train folds | merged val/test folds, optional |
| FL local Reggio | Reggio local FL model | Reggio val/test folds |
| FL local Trento | Trento local FL model | Trento val/test folds |

The client-specific rows are the fair FL-vs-non-FL comparison:

| Dataset | Non-FL Comparator | FL Comparator |
|---|---|---|
| Reggio | centralized model evaluated on Reggio folds | FL global and/or Reggio local evaluated on Reggio folds |
| Trento | centralized model evaluated on Trento folds | FL global and/or Trento local evaluated on Trento folds |

The merged non-FL row is still useful as the centralized global reference. The merged FL-global row
can be reported as an optional diagnostic, but it is not a substitute for client-specific
evaluation.

---

## 5. Metric Computation

The current reporting rule is:

```text
compute each metric independently on each fold test set
report mean and standard deviation across the 5 folds
```

The active reported metrics are:

```text
balanced_acc
precision
recall
recall_0
auc
f2
acc
```

The current evaluation code does **not** compute pooled out-of-fold metrics and does **not** compute
bootstrap confidence intervals. Earlier versions considered pooled OOF metrics and bootstrap CIs,
but these have been removed to keep the evaluation simpler and easier to inspect.

For each model/dataset evaluation, the main output is:

```text
fold_metrics.csv
```

Format:

```text
fold,threshold,balanced_acc,precision,recall,recall_0,auc,f2,acc,num_bags
fold_01,...
fold_02,...
fold_03,...
fold_04,...
fold_05,...
avg,,...
std,,...
```

The comparison CSV files report metric cells as:

```text
mean +/- std
```

Example:

```text
0.7812 +/- 0.0350
```

---

## 6. Threshold Handling

Threshold handling is identical for FL and non-FL evaluation.

For every model, dataset target, and CV fold:

```text
fold k validation predictions -> search threshold -> freeze threshold
fold k test predictions       -> compute metrics with frozen threshold
```

The threshold is always searched on the same dataset view that is being evaluated.

Examples:

```text
non-FL on Reggio: threshold from Reggio fold k val, metrics on Reggio fold k test
non-FL on Trento: threshold from Trento fold k val, metrics on Trento fold k test
FL global on Reggio: threshold from Reggio fold k val, metrics on Reggio fold k test
FL local Trento: threshold from Trento fold k val, metrics on Trento fold k test
```

This avoids giving the centralized model a better operating point by tuning it on pooled validation
data while reporting only on a client subset.

---

## 7. Scripts And Responsibilities

### Single CV Evaluator

```text
aiflopp/evaluate_single_cv.py
```

Evaluates one checkpoint root against one CV dataset folder.

It expects:

```text
checkpoint_root/fold_01/best_model.pth
checkpoint_root/fold_01/config.yaml
checkpoint_root/fold_02/best_model.pth
...

folds_dir/fold_01/val_manifest.csv
folds_dir/fold_01/test_manifest.csv
folds_dir/fold_02/val_manifest.csv
...
```

It writes only:

```text
fold_metrics.csv
```

### Non-FL Client Evaluation

```text
aiflopp/evaluate_nonfl_on_clients.py
```

Evaluates one centralized non-FL CV checkpoint root on:

```text
merged
reggio
trento
```

It calls `aiflopp.evaluate_single_cv` once per target dataset and writes:

```text
nonfl_cv_comparison.csv
nonfl_cv_comparison.json
```

The script is client-extensible through `CLIENT_DATASET_PARENTS`.

### FL CV Launcher

```text
flare_exp/run_fl_cv.py
```

Runs one NVFlare job per CV fold by calling `flare_exp.job` sequentially.

It derives defaults from the FL YAML config:

```text
output_dir
manifest_set
job_name
```

All fold runs share the same W&B group, while each fold still has a unique job name.

### FL Metrics Evaluation

```text
flare_exp/evaluate_fl_cv_metrics.py
```

Exports FL checkpoints into a standard evaluator format and evaluates all FL targets.

It exports:

```text
global/fold_XX/best_model.pth
local/reggio/fold_XX/best_model.pth
local/trento/fold_XX/best_model.pth
```

Then it evaluates:

```text
FL global -> Reggio
FL global -> Trento
FL global -> merged, optional
FL local Reggio -> Reggio
FL local Trento -> Trento
```

It writes:

```text
fl_cv_comparison.csv
fl_cv_comparison.json
```

---

## 8. Recipe Parity Between FL And Non-FL

The comparison is only meaningful if the non-data choices remain aligned:

- Same fold definitions.
- Same model architecture when comparing a specific experiment.
- Same feature root.
- Same threshold handling.
- Same metric set.
- Same fold-level mean/std reporting.
- Same random seed policy as much as possible.

For the current gated CAD-diff experiment, the FL config is:

```text
flare_exp/configs/gated_bs512_d05_CAD_DIFF_fed.yaml
```

It mirrors the non-FL gated model settings where applicable.

---

## 9. Run Structure

1. Create per-client CV folds.
2. Merge per-client folds into `mergedRT/fl_cad_binary_diff_5cv`.
3. Train the centralized non-FL model over the merged CV folds.
4. Train the FL model once per fold using the FL CV launcher.
5. Evaluate non-FL on merged and client-specific folds.
6. Evaluate FL global/local models on the configured target datasets.
7. Compare the final CSV files, which report mean +/- std across folds.

---

## 10. Open Points

1. Whether FL-global or FL-local is the main headline comparator remains a scientific/reporting
   decision.
2. If a third client is added, update the client dictionaries in the merge and evaluation scripts.
3. Merged FL-global evaluation is optional and should not replace the client-specific rows.

---

## 11. Commands

### Merge Client CV Folds

Run this only when the merged global folds need to be created or refreshed:

```bash
PYTHONPATH=. python3 aiflopp/preprocessing/merge_client_cv_folds.py
```

### Train FL Across The 5 CV Folds

```bash
PYTHONPATH=. python3 -m flare_exp.run_fl_cv \
  --config flare_exp/configs/gated_bs512_d05_CAD_DIFF_fed.yaml \
  --skip-existing
```

### Evaluate Non-FL On Merged And Client Datasets

```bash
PYTHONPATH=. python3 -m aiflopp.evaluate_nonfl_on_clients \
  --checkpoint-root /home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/trained_models/nonFL-gated_bs512_d05 \
  --output-root /home/ubuntu/giodir/digitalPathology/outputs/cad_binary_diff/evaluations/nonFL-gated_bs512_d05 \
  --manifest-name fl_cad_binary_diff_5cv
```

### Evaluate FL CV Metrics

```bash
PYTHONPATH=. python3 flare_exp/evaluate_fl_cv_metrics.py \
  --config flare_exp/configs/gated_bs512_d05_CAD_DIFF_fed.yaml
```

### Evaluate A Single Model/Dataset CV Pair Manually

Use this for debugging or custom comparisons:

```bash
PYTHONPATH=. python3 -m aiflopp.evaluate_single_cv \
  --checkpoint-root /path/to/checkpoints \
  --folds-dir /path/to/folds \
  --output-dir /path/to/evaluation_output \
  --features-root /home/ubuntu/giodir/digitalPathology/data/features/uni_features_merged_RE_TN
```
