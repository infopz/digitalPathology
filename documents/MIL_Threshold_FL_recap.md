# MIL Pipeline — Threshold Selection, Overfitting & FL Transition: Working Notes

Context recap from an extended design discussion. Scope: ABMIL / gated-attention MIL on frozen UNI2-h (1536-dim) WSI features, ~1000 WSIs, binary classification, transitioning from centralized (with CV) to federated learning (NVIDIA FLARE, ~2–3 sites). Intended as reusable context for future work — reflects decisions reached, not just options.

---

## 1. Threshold selection — the core problem and the resolution

**Problem.** Tuning the decision threshold on the validation set and then reporting metrics on that same set gives an optimistic-by-construction estimate. The set used to *choose* the threshold cannot also serve as an honest performance read-out.

**Rejected idea — moving threshold search to the training set.** Tempting (larger sample → lower-variance threshold), but wrong. Train-set score distributions are systematically different from held-out ones: BCE/CE loss pushes confidence toward 0/1 on the exact bags the optimizer sees, so scores on train are more separated than on unseen data. More train samples reduces *variance* of the threshold estimate but not this *bias* (distributional mismatch) — you'd get a very stable, systematically wrong threshold. Worse in FL, where repeated local fine-tuning + aggregation compounds the overconfidence round over round.

**Adopted approach.**
- **During training (every epoch / every FL round): evaluate at a fixed 0.5 threshold.** A fixed cutoff is never "tuned" on anything, so there's no leakage — this is the honest per-round monitoring signal.
- **Search the threshold exactly once, at the very end of training**, on val predictions from the final/best checkpoint. Freeze it. Apply it to compute the final test metrics.
- **Report val metrics from the search step with an explicit caveat** that they are optimistic; **test is the only fully clean number.**
- This is a single, isolated leakage event (one final report) instead of leakage on every round reported to the FL server.

**Note on the existing centralized code.** It already behaved this way — `train()` calls `evaluate()` without a threshold (defaults to 0.5) for per-epoch monitoring, and only `search_best_threshold` at the end touches val. The FL design generalizes this existing pattern rather than inventing something new.

**Better-than-a-threshold option (on the horizon).** Fit calibration (Platt / temperature scaling) on train via cross-fitting so raw probabilities are well-calibrated and 0.5 becomes principled — avoids "searching" a threshold on any set at all.

---

## 2. FL-specific constraints that drove the design

- **Cannot report test metrics to the server** — test must stay untouched (it's the final read-out; the server would otherwise use it for aggregation/selection, contaminating it).
- **Cannot afford a 4th "calibration" split** — data is too limited; only train/val/test per site.
- **No CV in FL (at least initially)** — reliance on single splits per site, which are demonstrably noisy at this sample size (see §5).
- **Per-round server metric should be threshold-free where possible.** AUC needs no cutoff and can be computed honestly every round with zero tuning bias. Good candidate for server-side model selection / cross-site eval tracking (which is what FLARE's cross-site eval workflow is built around).
- **Freeze the threshold once**, not per round. Options, cheapest first: (a) one-off local/pooled out-of-fold threshold search on train only, before FL, frozen for the whole run; (b) single end-of-FL search using out-of-fold train predictions from the *final* global model; (c) calibration instead of a raw threshold.
- **Key principle for FL: monitor the trend, freeze one threshold, trust the pooled/test number** — do not react to any single round's val metric.

---

## 3. Epoch/round selection metric — AUC vs balanced accuracy

This went back and forth and landed on evidence, not priors:

- **AUC** measures ranking/discrimination (threshold-invariant, tail-sensitive). **Balanced accuracy @ 0.5** measures classification at a fixed cutoff (calibration-sensitive, blind to tail-ranking changes until mass crosses 0.5).
- **Initial instinct** was to switch to AUC as primary (threshold-invariant → no leakage worry, isolates representation quality from per-round calibration drift). But:
- **Empirical finding killed this for this setup:** with frozen UNI2-h features (already strongly separable — validated by the mean-pool + XGBoost baseline), val AUC is *highest in epochs 1–2* and **degrades monotonically thereafter**, while balanced_acc@0.5 peaks later (~epoch 7–13). Selecting on AUC would grab epoch 1–2 — a model whose attention hasn't moved from near-uniform init (≈ mean pooling), i.e. before any of the MIL-specific learning the ablation is meant to measure.
- **Decision: keep balanced_acc@0.5 as the primary selection metric**, log AUC as a diagnostic. The leakage concern that motivated switching to AUC is *already fully solved* by fixing the threshold at 0.5 during training — no metric change needed to fix leakage.

**Why AUC degrades while bal_acc@0.5 holds longer:** overfitting pushes confident predictions further toward 0/1, scrambling the ranking of borderline cases in the tails (hurts AUC) while the mass of cases near the 0.5 boundary stays correctly classified (bal_acc lags). bal_acc eventually decays too, just later.

**Code hygiene:** split the single `threshold_metric` arg (which did double duty as both `best_metric` in `train()` and `objective` in `search_best_threshold`) into two independent fields — e.g. `epoch_selection_metric` (default balanced_acc) and `threshold_metric`/`threshold_objective` (default balanced_acc or f2). Otherwise one CLI flag silently couples both stages — the exact coupling being designed away.

---

## 4. Overfitting — diagnosis and mitigation

**Confirmed, not hypothesized.** Printing train *and* val metrics per epoch showed the textbook signature: **train AUC climbs monotonically (→0.98+) while val AUC falls monotonically (0.93→0.87) over the same epochs** — a generalization gap widening every epoch. Train balanced_acc and recall also march up toward ~1.0. Concentrated largely in the attention module (the most expressive part with only ~700 train bags to constrain it).

**Mechanism.** Attention starts near-uniform (≈ mean pooling on already-good frozen features → decent from epoch 1), then specializes onto bag-specific patches that don't generalize.

**Mitigations applied (worked):**
- `dropout` 0.25 → **0.5**
- `max_bag_size` → **512** (random patch subsampling per epoch acts as data augmentation, if re-drawn each forward pass — worth confirming it's re-drawn, not a static truncation)
- `weight_decay` engaged
- `batch_size` 8 → **32** (noisier averaged gradients → mild regularization; fewer, more stable updates given ~1000 bags)

Result: train/val AUC divergence largely gone — train AUC now plateaus ~0.96–0.98 instead of running to 1.0; val AUC stays flat instead of collapsing.

**Further levers, prioritized by how directly they target the observed mechanism:**
1. **Verify dropout is applied *inside* the gated-attention branches (tanh V + sigmoid U), not just the final classifier head** — otherwise the overfitting-prone part isn't regularized. (Top priority: cheap, targeted.)
2. **Attention entropy regularization** — penalize peaked attention (`loss = bce - λ·attention_entropy`, entropy per-bag). Directly targets specialization-driven overfitting; doubles as an interpretability diagnostic.
3. **Instance-level feature dropout / Gaussian noise** on the 1536-dim patch features before attention (used in CLAM / DTFD-MIL variants; frozen-feature MIL overfits on exact feature values).
4. **Gradient clipping** (`clip_grad_norm_` 1.0–5.0) — stabilizes early large steps that push premature attention specialization.
5. **LR schedule** (cosine/step decay) — dampens "keep fitting train harder" in late epochs without guessing a lower constant LR.
6. **Reduce `attention_dim` 128 → 64** — modest expected effect (128 isn't large for gated attention); try after the targeted options.

- `lr` reduction alone: not a fix — just stretches the same overfitting shape over more epochs. Cheap to try, low expectation.
- **Tighten early-stopping patience** (10 → 7–8): best epochs are consistently early (see §5); extra epochs waste compute per FL round *and* actively degrade ranking quality.

---

## 5. What the 7-fold CV revealed (final config, ~1000 WSIs)

Config: `gated_mil`, attention_dim 128, hidden_dim 64, dropout 0.5, max_bag_size 512, bs 32, lr 5e-4, wd 1e-4, patience 10, pos_weight ≈3.7–4.0.

Headline: mean test balanced_acc > 0.90. But the per-fold detail matters more than the mean:

1. **Test bal_acc is variable and mean-inflating.** Per fold: 0.876 / 0.970 / 0.887 / 0.900 / 0.909 / 0.890 / 0.893. Fold 2 (0.970) is a high outlier; the rest cluster ~0.88–0.91. **Report median alongside mean, and prefer a bootstrap CI on pooled out-of-fold predictions** over mean±std (std is inflated by one fold).

2. **Val and test disagree in direction, sometimes strongly** (fold 1: val 0.980 ≫ test 0.876; fold 2: val 0.895 ≪ test 0.970). At ~130 bags / ~30 positives per split, each single split is a high-variance estimate. Washes out over 7 folds → pooled estimate trustworthy. **But any single-split decision (= FL per-round monitoring) is on shaky ground** — strongest empirical backing for "monitor the trend, don't over-trust a single round's val."

3. **Precision consistently low (~0.54–0.68), recall very high (~0.93–1.0)** — every fold's confusion matrix has the same shape (catch nearly all positives, many false positives). Direct consequence of `pos_weight`≈3.7. **Whether this is right is a clinical decision, not a metric one:** correct for screening/triage (missing a positive costlier than a false alarm); reconsider if false positives carry real cost. Controlled by the end-of-training threshold objective (`f2` vs `balanced_acc`).

4. **Selected thresholds vary widely: 0.50/0.50/0.50/0.45/0.60/0.45/0.35.** Confirms per-fold calibration instability (raw score distribution shifts fold to fold). Implications: (a) a single *frozen* threshold for FL will be a compromise — expect it to underperform per-fold-tuned numbers, that's fine and expected; (b) strong motivation for **calibration analysis (reliability diagrams)** — the model *ranks* well (AUC 0.89–0.98) but is not consistently *calibrated*. Classic AUC-good / calibration-variable situation.

5. **Best epoch consistently early** (3–13, no fold's best in late training; e.g. fold 7 ran to 23, best 13). Confirms patience=10 is generous → tighten to 7–8, saves FL round cost.

**Must-verify:** confirm CV folds are **patient-grouped** (no patient's slides split across train/val/test). Directory `cv_binary_pat_diff` suggests it's intended — if confirmed, numbers are trustworthy; any patient leakage would make the tight ~0.9 test cluster optimistic.

---

## 6. Standing principles distilled

- Fixed-threshold metrics during training carry **no leakage** — leakage only ever attaches to the *searched* threshold, so search it once at the end.
- **AUC = ranking; accuracy@fixed-threshold = calibrated classification.** High AUC + mediocre bal_acc@0.5 = well-ranking but poorly-calibrated model. Don't conflate.
- With strong frozen features, early-epoch AUC can reflect near-uniform-attention (≈ mean pooling), **not** learned MIL behavior — don't select on it.
- Confirm overfitting by logging **train *and* val** metrics per epoch; a widening train/val AUC gap is the definitive signal.
- At small per-split sample sizes, single-split val/test estimates are high-variance and can disagree in direction — **pool out-of-fold predictions + bootstrap CI** for the defensible headline number.
- Always save patch coordinates with features (attention heatmaps, resolution normalization) — unchanged standing rule.
