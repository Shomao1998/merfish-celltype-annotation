# Lessons Learned

My section of the team presentation: what we kept, what we dropped, and why.

## What moved the needle

| Direction | What it is | Why we added it | Result |
|---|---|---|---|
| **Biology-constrained decoding** | A hard mask keeps each prediction within its E/I group's allowed cell types. | E/I is fixed biology — a cross-boundary label is always wrong, so this rules out impossible classes for free. | Predictions never cross the E/I boundary — one error class gone. |
| **Five-model ensemble** | Equal-probability mean of LightGBM · LogReg · MLP · KNN · CatBoost. | Diverse models make uncorrelated errors that cancel when averaged. | **LOMO 0.760 → 0.769** — our largest single architectural gain. |
| **Target encoding (Region × Segment × E/I)** | Fold-safe encoding of anatomical groups into per-group cell-type priors, fed to the linear models. | With only 200 sparse genes, it injects an anatomical prior the raw counts can't supply. | Clear lift for LogReg & MLP — our most effective single feature. |

## Tried, dropped — but high potential

Our training set was only ~5,000 cells. Real problems in this field can be 10⁵–10⁶ cells, and at
that scale these become winners. **They are validation-limited, not wrong.**

| Direction | What we did | Why no gain yet | Why it's promising |
|---|---|---|---|
| **Expression label features** | Score each cell by cosine to the 60 class-mean profiles + a true-label vote from its nearest neighbors in expression space. | Class centroids are noisy at 5K cells and don't transfer across mice. | Stabilize as reference cells grow. |
| **Anatomical bucket constraint** | Split cells into 28 Region×E/I×Segment buckets (15 map to one cell type) and restrict predictions to types seen in that bucket. | Held-out mice hit buckets unseen in training, so the constraint misfires. | A much finer successor to the E/I mask once bucket coverage fills in. |
| **Deep & spatial models** | CNNs / DNNs and semi-supervised learning over full tissue sections. | Data-hungry; 5K cells cause heavy overfitting. | ST benchmarks show they overtake with more data. |
| **LLM reranker (Qwen)** | Re-score the hardest, most-uncertain cells among our own top-3, gated on marker-gene evidence (never submitted). | Weak signal — +0.0014 on the batch axis (p ≈ 0.04). | More calibration data sharpens both the scores and the gating. |

## The overarching lesson

**The ceiling was data, not method.** On a larger public atlas with whole-mouse holdout, the
same pipeline goes from ~0.77 to ~0.92 (5K → 118K cells). And watch **fold variance, not the
mean** — an under-regularized model silently collapsed whole held-out mice to 0.06 while random
CV still read 0.9. Most of our "failed" ideas are our roadmap, not our failures.
