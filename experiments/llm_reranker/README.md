# LLM reranker (Qwen3) — research only, **never submitted**

> ⚠️ Kept as a **post-competition experiment**. A pretrained model's knowledge could count as
> external information, so this was never connected to the competition submission
> (`submission_approved: false` in the run manifest).

## Design motivation

Before building anything, I looked at how LLMs and foundation models are currently used for
cell-type annotation. Two families dominate, and **both lean on external knowledge or data**:

- **Reference / atlas matching.** Map query cells onto a large *annotated* reference and transfer
  the labels — e.g. **CellTypist**, **Azimuth**, **SingleR** — or ask a general-purpose LLM to
  name a cluster from its top marker genes using the model's own biological knowledge and marker
  databases (e.g. **GPTCelltype** with GPT-4). Fast, but the answer rides entirely on an external
  reference or the LLM's outside knowledge.
- **Single-cell foundation models.** Transformers pretrained on tens of millions of cells —
  e.g. **Geneformer**, **scGPT**, **scBERT**, **scFoundation** — then fine-tuned to classify cell
  types. Powerful, but they require a massive external pretraining corpus.

Both were off the table here — the competition banned external data. So in a *vibe-coding*
brainstorm we asked a different question: **can an LLM help without any external data or reference,
purely as an arbiter inside our own model's decision flow?** That reframes the LLM from a
*knowledge source* into a **decision-flow interceptor** — it never labels a cell from scratch and
never sees an outside atlas. It only **intercepts** the handful of cells our own ensemble is
unsure about and **votes/scores** among the candidates our model already produced, grounded solely
in marker-gene evidence computed from our own training folds, and gated so it can rarely overrule
us.

*(The tools above are representative examples of each category; see each project's paper for details.)*

## What this is (and what it is *not*)

Concretely, the LLM only **re-ranks our own model's output**, for a small set of uncertain cells,
under a chain of conservative gates — and it may only choose among candidates our model already
proposed. It runs a **local** `qwen3:8b` via Ollama; nothing leaves the machine.

## Which cells it touches (target set)

- Base ensemble is **unsure**: top-1 probability ≤ **0.80**.
- Among those, the **lowest-margin** cells (smallest top-1 − top-2 gap), capped at **100 per
  scheme**, balanced across folds.

## What the LLM sees

- The ensemble's **top-3 candidate labels** only (it cannot invent a class).
- **Anatomical context** — E/I, Region, Segment, AP position. Experiment IDs
  (Mouse / Section / Dataset) are **withheld**.
- **Fold-derived evidence** — for each candidate vs. the baseline, the top discriminative
  marker genes (≤8 per direction) computed from **this fold's training data**, with each gene's
  mean expression, detection rate and pairwise score; plus nearest-neighbor support
  (15 neighbors, 5 per class).

## When a label is actually changed (all must hold)

1. **Order-invariant:** the model is asked **twice with candidate order shuffled**; both answers
   must agree.
2. **Confident:** LLM self-reported confidence ≥ **0.80**.
3. **Grounded:** it cites ≥ **2 marker genes** that *both* passes cite **and** that each clear a
   fold-training pairwise discriminative score ≥ **0.25** — i.e. the gene genuinely separates
   those two types in our data, not just in the model's opinion.
4. **Cross-family only** (`switch_policy=cross_family`): a switch is allowed only *between* broad
   families (astrocyte / oligodendrocyte-lineage / meninges / …); flips *within* a fine subtype
   cluster are blocked.

Otherwise the ensemble's original prediction is kept.

## Result

Marginal: **+0.0014 on the batch axis (p ≈ 0.04), 0 on the mouse axis** — real but tiny, and
gated behind everything above. Promising with more calibration data; not worth submitting today.

## Files

| File | Role |
|---|---|
| `qwen3_oof_reranker.py` | The reranker (selection → prompt → two-pass → gated switch). |
| `qwen3_dataset_confirm.py` | Dataset-axis independent-confirmation run. |
| `qwen3_gated_fusion.py` | Gated-fusion variant. |

> These are the **as-run research scripts**. Their internal cross-references and I/O paths
> point to the original experiment filenames/layout; adjust them for this repo's structure
> before running. They are included to document the approach, not as a turnkey pipeline.
