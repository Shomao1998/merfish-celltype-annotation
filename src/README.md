# `src/` — modeling line & validation

Paths in these scripts assume the competition data is available under `data/`
(see [`../data/README.md`](../data/README.md)).

## Models

| File | What it is |
|---|---|
| `model_v031_robust_5model.py` | **The final line.** Equal-probability ensemble of LightGBM · LogReg · MLP · KNN · CatBoost, with the E/I hard-constraint decoder and fold-safe target encoding. Experiment IDs are excluded from features and used only for grouped validation. |
| `model_v04_full_centroid_knn_bucket.py` | **Most complete pipeline.** v0.31 plus three extra generalizable ideas — class-centroid cosine, expression-kNN true-label vote, and a Region×E/I×Segment bucket mask. It did **not** beat v0.31 on group CV, so v0.31 stayed the submission; kept here for transparency and as the base for future scaling. |

Common commands:

```bash
# Grouped out-of-fold evaluation (leave-one-mouse-out and leave-one-batch-out)
python src/model_v031_robust_5model.py --mode oof --schemes mouse dataset --seeds 0 1

# Train on all training rows and predict the test set
python src/model_v031_robust_5model.py --mode predict --seeds 0 1 2 3 4
```

## `validation/` — the rigor

| File | What it checks |
|---|---|
| `group_cv_audit.py` | Nested leave-one-mouse-out / leave-one-batch-out audit of the model — the check that exposed the CV→leaderboard gap and drove the ID-free redesign. |
| `group_cv_ablation.py` | Feature ablations under grouped CV (e.g., adding fold-safe target encoding) — what actually generalizes vs. what only helps random CV. |
| `reference_classifiers_group_cv.py` | Simple reference classifiers under the same grouped protocol, as sanity baselines. |

**Why grouped CV matters here:** random CV shares tissue sections between train and validation,
so it is optimistic. Holding out a whole mouse (or a whole batch) simulates a genuinely new
experiment — and that is where fragile, ID-like features are exposed.
