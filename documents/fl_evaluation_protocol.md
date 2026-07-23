# FL vs. Non-FL Evaluation Protocol — Decisions & Rationale

Context document for future work on the federated MIL / inter-observer discordance project.
Records the evaluation design agreed on, and *why*, so the reasoning doesn't have to be rederived.

---

## 1. Problem being solved

Data splits are highly sensitive: metrics computed on a single split are biased by that
particular draw. The non-FL baseline handled this with k-fold CV over pooled data. Moving to a
federated setup (currently **2 clients**), the goal is an evaluation protocol that is:

1. Robust to split choice.
2. **Fully comparable** between the FL and non-FL versions — same data, same recipe, differing
   only in whether it is partitioned across clients.

---

## 2. Cross-validation structure

**5 folds** (reduced from 7). Trade-off accepted deliberately: coarser fold-to-fold spread
(5 points instead of 7) in exchange for larger validation slices per client per fold, which
stabilises the per-client threshold search — the weakest link in this setup.

### Rotating test sets (chosen over a fixed shared test set)

Each fold has its own test slice; every slide appears in test **exactly once** across the 5 folds.

- **Why:** a single fixed test set can be unrepresentative (unusual pathologist pair, batch of
  hard cases). With a fixed test set that bias hits all folds identically and the fold-to-fold
  std cannot see it. Rotating gives a lower-variance, less biased generalisation estimate.
- **Cost, accepted knowingly:** the fold-to-fold spread is no longer interpretable as pure
  *training-split* sensitivity — it now confounds training-split variation with test-slice
  variation. This is acceptable because the bootstrap CI (§5) supplies the uncertainty that
  matters for headline claims.
- **Fairness is preserved** as long as FL and non-FL use the *same* rotation: identical
  slide → fold assignment, so both pipelines are tested on the same slides in the same fold.

---

## 3. Fold construction

**Build folds per client, then merge upward to obtain the global view.**

```
global fold k  =  (client A's fold k)  ∪  (client B's fold k)
```

This satisfies the property that actually matters for comparability: for every fold `k`, the
union of the clients' train partitions equals the centralized train partition, and likewise for
val and test.

### Why per-client and not pooled-then-split

Both directions are mathematically equivalent *provided the folds are stratified by site*.
Building per client guarantees that stratification by construction. Pooled random splitting does
not, and can produce a fold where one site is underrepresented in validation — precisely the
concept-shift-sensitive failure to avoid — while also making per-client validation slice sizes
vary across folds, which destabilises the threshold search.

### Additional requirements

- **Stratify within each client on the label** as well. Label distribution is skewed, so this
  keeps pos/neg ratios stable across folds.
- **Assign fold indices once and persist them.** A CSV of `slide_id → fold_index → site`, read by
  *both* the FL and the centralized pipeline.
  - *Failure mode being guarded against:* regenerating folds with a different seed in one
    pipeline and silently comparing "fold 3" against a different fold 3.
- **Verify per-fold, per-client class counts before committing to a full run matrix.** With two
  clients and rotating test sets, per-fold test slices get small. If a fold leaves one client
  with ~15 test slides, that per-fold number is near noise (pooled OOF, §5, rescues the headline
  metric, but the per-fold report degrades).

---

## 4. Comparison design

Metrics are **not merged across clients** — the two clients' data are intrinsically different,
which is the whole reason two models are being trained in the FL setup.

Everything is evaluated **per client**. The non-FL model is additionally run in *"client mode"*:
evaluated on a single client's val/test partition. This yields, for each site, a like-for-like
set of cells:

| Model | Evaluated on |
|---|---|
| non-FL (client mode) | client *k*'s val/test |
| FL global | client *k*'s val/test |
| FL local / personalized | client *k*'s val/test |

**Open point:** which of FL-global vs. FL-local is the headline number against the non-FL
baseline is not yet settled. Both are evaluable per client under the same rules.

### If a pooled FL number is ever reported

Not currently planned. If it is: compute it on **concatenated predictions across sites** — one
metric on the pooled prediction vector — never an average of per-site metrics. Averaging
per-site balanced accuracy weights a small site equally with a large one and will not match the
centralized computation.

---

## 5. Metric computation

Two distinct quantities, both reported, answering different questions.

### 5a. Pooled OOF metric — the headline number

Each slide has exactly one out-of-fold prediction. Concatenate all predictions across the 5 folds
into a single vector and compute the metric **once** on that vector. One number, not five
averaged.

**Why not average the per-fold metrics:** with small per-fold test slices each per-fold metric is
a noisy estimate, and averaging noisy estimates ≠ computing the metric once on the full set.
This matters especially for balanced accuracy, which depends on class-conditional rates that are
unstable when a fold contains only a handful of positives. Pooled computation uses every slide's
contribution directly. Consistent with the OOF convention already used elsewhere in this project.

### 5b. Bootstrap CI — uncertainty on the headline

Resample the N slides in the pooled OOF prediction set **with replacement**, ~1000 times;
recompute the pooled metric on each resample; take the 2.5th and 97.5th percentiles as a 95%
confidence interval.

- Measures how much the metric would move given a different *test sample* of the same size.
- Distinct from the fold-to-fold std, which reflects sensitivity to a different *training* split.
- With small per-client test sets this interval may be wide enough to affect any claim that FL
  matches or beats centralized — which is exactly why it is reported.

### 5c. Per-fold mean ± std — secondary report

Five metrics, one per fold's test slice; report mean and std. Under the rotating-test design this
std confounds training-split and test-slice variation (see §2). Kept as a secondary descriptive
figure, not the headline.

### Reporting format

```
pooled OOF balanced accuracy = 0.71  [95% CI 0.65 – 0.77]
per-fold: 0.68 ± 0.05
```

---

## 6. Threshold handling

**Rule: threshold search and metric evaluation always use the same client's data.**

```
client k's validation predictions  →  search threshold  →  freeze
client k's test predictions        →  compute metrics with that frozen threshold
```

Applied uniformly to FL-global, FL-local, and non-FL-in-client-mode.

### Why not tune on pooled validation

For the non-FL model evaluated in client mode, tuning the threshold on the *pooled* validation
set and only reporting metrics on client *k*'s slice would give the non-FL model a different
operating point — better calibrated, fit on more data — than the FL model receives. That biases
the comparison in favour of centralized. Each FL model gets a threshold tuned on the site it is
deployed at; the non-FL counterpart must go through the identical recipe.

### Known caveat

Per-client validation splits are small, so the tuned threshold is noisy. This noise hits FL and
non-FL **equally**, so the comparison stays fair — but it is a further reason the bootstrap CI is
necessary, since threshold variability is part of what the interval should reflect.

### Never tune and report on the same split

Client val → threshold. Client test → metrics. Never the same slides for both.

---

## 7. Recipe parity between FL and non-FL

For the comparison to mean anything, every non-data choice must mirror across the two pipelines:

- **Selection metric:** FL round-selection must mirror centralized epoch-selection — balanced
  accuracy at threshold 0.5 (per existing project convention; AUC retained as a diagnostic only,
  since frozen UNI2-h features are strongly linearly separable from the start and AUC degrades
  monotonically).
- **Threshold handling:** structurally identical (§6).
- **Fold assignment:** the same persisted CSV (§3).
- **Randomness:** fix all FL-side randomness — seeds, deterministic client order, any client
  sampling. Otherwise the fold-to-fold std conflates split sensitivity with FL stochasticity
  (aggregation order, round-reset optimizer noise) and no longer means the same thing as the
  centralized std. The fold should be the only factor varying across the 5 runs.

---

## 8. Run structure

- Define the 5 folds once, per client, persist to CSV.
- Call the FL algorithm **5 times**, once per fold, with identical fold definitions on both
  clients.
- Run the centralized baseline 5 times on the merged folds, plus in client mode per site.
- Collect per-slide OOF test predictions from every run.
- Compute: pooled OOF metric + bootstrap CI (headline), per-fold mean ± std (secondary), per
  client, per model type.

---

## 9. Open points

1. **FL-global vs. FL-local** as the headline comparison target — not settled.
2. **Per-fold, per-client class counts** under rotating test sets — verify empirically that
   enough positives remain per client per fold before committing to the full run matrix.
3. **Pooled-across-sites FL number** — deliberately deferred; if ever needed, use concatenated
   predictions, not averaged per-site metrics.
