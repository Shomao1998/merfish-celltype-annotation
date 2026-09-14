# MERFISH Spatial Transcriptomics — Cell-Type Annotation

**English | [中文](README.zh-CN.md)**

> Classifying 60 neuronal cell types from sparse spatial gene expression.
> **🥈 2nd Prize — University of Rochester Biomedical Data Science Hackathon, Summer 2026** (Team **XDRKAMOA**).

This repository is **my modeling line** for the team's 2nd-place entry, plus the validation
harness that drove our decisions and a post-competition study of what would push the model
further. The emphasis throughout is **honest generalization**, not leaderboard chasing.

---

## TL;DR

- **Task:** predict the cell type (1 of 60) for each cell in a mouse spinal-cord **MERFISH**
  dataset — ~10,000 cells, **200 genes**, ~**93% zeros**. Metric: overall accuracy.
- **What worked:** a **biology-constrained, 5-model probability ensemble** with fold-safe
  **target encoding**, validated by **leave-one-mouse-out** GroupKFold.
- **The interesting part:** I diagnosed a **~0.02 CV→leaderboard gap** (random CV was
  optimistic), rebuilt the model to be **ID-free and group-validated**, and then ran a
  **post-competition scaling study** showing the real ceiling is **data, not method**
  (leave-one-mouse-out accuracy **0.77 → 0.92** as labeled cells grow 5K → 118K).
- **Integrity:** the submitted model uses **zero external data**. I also found (and did not
  exploit) a leaderboard **data leak** — the public source atlas contains the test cells.

---

## Results

Leave-one-mouse-out accuracy (the honest, group-validated metric):

| Model | LOMO accuracy |
|---|---|
| KNN | 0.688 |
| MLP | 0.734 |
| Logistic Regression | 0.735 |
| CatBoost | 0.743 |
| LightGBM (baseline) | 0.760 |
| **5-model ensemble (final)** | **0.769** |

![Ensemble vs. single models](reports/figures/ensemble_vs_single.png)

The gain is small but real, and it comes entirely from **diversity** — five models that make
uncorrelated errors, averaged.

---

## Approach

**Baseline → ensemble.** I started from **LightGBM** (strong on tabular data). Because the
data is only 5K cells over 200 very sparse genes, I added **logistic regression** (linear
models handle sparse, high-dimensional data well and fail differently from trees), then fused
an **MLP** (non-linear structure), a **KNN** (neighborhood signal), and **CatBoost** (a
differently-behaved tree). The five are combined by **equal-probability averaging**.

**Biology-constrained decoding.** EDA showed the *excitatory / inhibitory / non-neuronal*
groups **never share a cell type**, so a hard mask forbids any prediction from crossing its
E/I group — a whole class of errors removed for free.

**Fold-safe target encoding.** Anatomical groups (Region, Segment, and their joint with E/I)
are encoded into per-group cell-type priors — an out-of-fold "who lives here" signal that the
200 sparse genes can't supply on their own. This was the single most effective feature.

**Generalization-first design.** Experiment-specific IDs (Mouse / Section / Dataset) are kept
**out of the features** and used only for grouped validation.

---

## What makes this rigorous (the part I'm proud of)

- **Double-seed discipline** — a change counts only if two seeds agree in sign (noise floor ≈ ±0.005).
- **Leave-one-mouse-out & leave-one-batch-out GroupKFold** — simulates a genuinely new
  animal/experiment. Random CV shares sections between train and validation and is optimistic.
- **CV→leaderboard gap, diagnosed and fixed.** An earlier version scored ~0.777 in random CV
  but ~0.75 online. The cause was random-CV optimism plus ID-like crutches; the fix was the
  ID-free, group-validated design above.

---

## Post-competition study: the ceiling was data, not method

Using a larger **public** spinal-cord MERFISH atlas (~147K cells, 10 mice) purely as a
**scaling testbed** — holding out **whole mice** so no held-out cell's neighbor is ever in
training — the *same* pipeline goes from ~0.77 to ~**0.92** leave-one-mouse-out as labeled
cells grow 5K → 118K.

![Data scaling](reports/figures/data_scaling.png)

A caution I learned the hard way: an **under-regularized GBDT silently collapses whole
held-out mice** to ~0.06 accuracy while random CV still reads ~0.9. **Watch fold variance,
not just the mean.** Code: [`experiments/atlas_loso_testbed.py`](experiments/atlas_loso_testbed.py).

**This is why several ideas we dropped are validation-limited, not wrong** — see
[`reports/lessons_learned.md`](reports/lessons_learned.md).

---

## Integrity note

The competition banned external data. Our **submitted model uses none**. During the
competition I found that the public source atlas (Zenodo `MERFISH_spinal_cord_resolved_0718`)
**contains the competition's own cells**, making labels recoverable — a real leaderboard leak.
We **did not exploit it**; this repo ships **no answer-key lookup code**. The atlas appears
here only as a *post-competition* scaling testbed with whole-mouse holdout, and any
LLM component (below) was kept as **research only, never submitted**, because a pretrained
model's knowledge could itself count as external information.

---

## Repository layout

```
src/
  model_v031_robust_5model.py          # final line: 5-model ensemble + E/I decoding + target encoding
  model_v04_full_centroid_knn_bucket.py# most complete pipeline (adds centroid/kNN/bucket features)*
  validation/                          # GroupKFold audits & ablations (the rigor)
experiments/
  atlas_loso_testbed.py                # ★ post-competition data-scaling study (0.77 → 0.92)
  llm_reranker/                        # Qwen3 OOF reranker — research only, never submitted
reports/
  lessons_learned.md                   # 3 things that worked + 4 promising, data-limited directions
  figures/
```
\* v0.4 is the most complete pipeline but did **not** beat v0.31 on group CV, so v0.31 remained
the submission line — kept here for transparency.

See [`src/README.md`](src/README.md) and [`experiments/llm_reranker/README.md`](experiments/llm_reranker/README.md) for details.

---

## Reproduce

```bash
pip install -r requirements.txt
# Data is NOT redistributed here — see data/README.md to obtain it, then:
python src/model_v031_robust_5model.py --mode oof   --schemes mouse dataset --seeds 0 1   # group-CV
python src/model_v031_robust_5model.py --mode predict --seeds 0 1 2 3 4                    # submission
# Post-competition scaling study (needs the public atlas — see data/README.md):
python experiments/atlas_loso_testbed.py --mode dev
```

## Data

Not included (it belongs to the organizers / is a separately-licensed public dataset).
See [`data/README.md`](data/README.md) for how to obtain the competition data and the atlas.

## Team & my role

Team **XDRKAMOA** (4 members) placed 2nd. **My contribution** (this repo): the modeling and
ensemble line, the generalization-first validation harness, and the post-competition scaling
and LLM-reranker research. Teammates led EDA/workflow, final-model write-up, and biological
interpretation.

## Author

M.S. Data Science, University of Rochester · `@<your-github-handle>`

## License

Code released under the [MIT License](LICENSE). Data and the third-party atlas are **not**
covered by this license and are not redistributed here.
