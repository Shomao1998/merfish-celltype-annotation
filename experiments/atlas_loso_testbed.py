#!/usr/bin/env python3
"""Leave-one-mouse-out testbed on the MERFISH spinal-cord atlas.

Why this exists
---------------
A post-competition sandbox for the honest question the small 5,000-cell set could
not answer: *does more labelled data (and which methods) actually improve
cross-mouse generalisation?* The atlas has ~147k cells across 10 mice, so it is a
~30x larger pool to develop methods on.

Protocol (the important part)
-----------------------------
* ONE mouse is set aside as the FINAL JUDGE and is never touched while you develop
  a method. Look at its score at most a handful of times, ideally once.
* The other 9 mice are the DEV pool. Method development uses leave-one-mouse-out
  CV over these 9 (rotating folds): train on 8 mice, validate on the 1 held out,
  rotate through all 9, and read the mean +/- std. A change is only "real" if it
  holds across folds, not on a single lucky mouse.
* When you have chosen a method, retrain on all 9 dev mice ONCE and score the FINAL
  JUDGE mouse. That single number is the honest generalisation estimate.

Why holding out a whole MOUSE is leakage-clean
----------------------------------------------
A tissue section belongs to exactly one mouse (Section ID is nested in Mouse ID,
asserted at load). Removing a whole mouse removes all of its sections and every
spatial neighbour of its cells, so expression-kNN / spatial / centroid-style
features cannot see a near-twin of a held-out cell. This is the clean, scaled-up
version of the GroupKFold(leave-one-mouse-out) validation used in v0.31.

Two knobs you will actually edit are marked with `# >>> PLUG` banners:
  1. build_features(...)  -- add / change the design matrix (fold-honest).
  2. fit_predict(...)     -- swap the model.
Everything else is the harness that keeps the evaluation honest.

Usage
-----
    # fast smoke test (subsample cells + few trees) to prove the pipeline runs:
    python src/atlas_loso_testbed.py --mode dev --quick

    # full rotating-fold dev evaluation (this is your everyday testbed):
    python src/atlas_loso_testbed.py --mode dev

    # ONE-TIME honest read on the untouched judge mouse, once a method is chosen:
    python src/atlas_loso_testbed.py --mode final

First finding (why this testbed earns its keep)
-----------------------------------------------
The very first run exposed a trap random CV cannot see. On the ~118k-cell training
pool a *properly regularised* model generalises across mice at ~0.89 (logistic
regression) / ~0.92 (LightGBM) leave-one-mouse-out -- a large jump over the ~0.77
of the 5,000-cell competition regime, so more labelled data genuinely helped. But
an *under-regularised* GBDT collapses an entire held-out mouse to ~majority-class
accuracy (0.06-0.13) while other mice score ~0.91, and which mouse collapses shifts
with the training subsample. That is a model covariate-shift artifact, not a batch
curse (see the CAUTION on fit_predict). Lesson: on this data the lever is model
regularisation, and the metric to trust is fold_std across the rotating folds, not
the mean alone.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
ATLAS_PATH = ROOT / "reference" / "MERFISH_spinal_cord_resolved_0718.h5ad"
OUT_DIR = ROOT / "outputs" / "atlas_loso_testbed"

LABEL_COL = "MERFISH cell type annotation"
MOUSE_COL = "Mouse ID"
SECTION_COL = "Section ID"
# Anatomical categoricals used as baseline metadata features. All are *properties*
# (they describe the tissue, not the animal's identity), so they transfer across
# mice. "Gender" is deliberately excluded from the default: with 5F/5M mice it can
# act as a weak mouse-identity proxy. Segment is absent in the atlas; "Axial level"
# (cervical/lumbar/sacral/thoracic) is the nearest rostro-caudal analogue.
META_CATEGORICAL = ["Region", "Excitatory_vs_Inhibitory", "Axial level"]

# The mouse held out as the untouched final judge. None -> chosen deterministically
# from SEED so the choice is reproducible; set e.g. "M5" to pin it.
FINAL_JUDGE_MOUSE: str | None = None
SEED = 0

# Library-size normalisation target before log1p (MERFISH cells carry only tens of
# counts, so a small target like 500 is appropriate, matching the reference teams).
NORM_TARGET = 500.0

# Optional: restrict to the competition's 200-gene panel by pointing at its
# counts_train.csv header. None -> use all atlas genes (more features than the
# competition; fine for a testbed, but not apples-to-apples with the 200-gene runs).
GENE_PANEL_FROM: str | None = None


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Atlas:
    counts: np.ndarray          # (n_cells, n_genes) raw integer counts
    genes: list[str]
    labels: np.ndarray          # (n_cells,) str cell-type
    mouse: np.ndarray           # (n_cells,) str Mouse ID
    section: np.ndarray         # (n_cells,) str Section ID
    meta: pd.DataFrame          # META_CATEGORICAL columns, str, index 0..n-1


def load_atlas(gene_panel_from: str | None = GENE_PANEL_FROM) -> Atlas:
    import anndata as ad

    adata = ad.read_h5ad(ATLAS_PATH)
    obs = adata.obs

    labels = obs[LABEL_COL].astype(str).to_numpy()
    keep = ~pd.isna(obs[LABEL_COL].to_numpy()) & (labels != "nan")
    adata = adata[keep].copy()
    obs = adata.obs
    labels = obs[LABEL_COL].astype(str).to_numpy()

    x = adata.X
    counts = np.asarray(x.todense() if hasattr(x, "todense") else x)
    counts = np.rint(counts).astype(np.int32)
    genes = [str(g) for g in adata.var_names]

    if gene_panel_from:
        panel = list(pd.read_csv(gene_panel_from, index_col=0, nrows=1).columns)
        cols = [genes.index(g) for g in panel if g in genes]
        missing = [g for g in panel if g not in genes]
        if missing:
            print(f"  [genes] {len(missing)} panel genes absent from atlas: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        counts = counts[:, cols]
        genes = [genes[i] for i in cols]

    meta = pd.DataFrame(
        {c: obs[c].astype(str).to_numpy() for c in META_CATEGORICAL}
    ).reset_index(drop=True)

    atlas = Atlas(
        counts=counts,
        genes=genes,
        labels=labels,
        mouse=obs[MOUSE_COL].astype(str).to_numpy(),
        section=obs[SECTION_COL].astype(str).to_numpy(),
        meta=meta,
    )
    _assert_sections_nested(atlas)
    print(f"Loaded atlas: {counts.shape[0]} cells x {counts.shape[1]} genes, "
          f"{len(np.unique(atlas.labels))} classes, "
          f"{len(np.unique(atlas.mouse))} mice.")
    return atlas


def _assert_sections_nested(atlas: Atlas) -> None:
    """Each Section ID must map to exactly one Mouse ID; otherwise holding out a
    mouse would not remove all of a held-out cell's spatial neighbours."""
    df = pd.DataFrame({"section": atlas.section, "mouse": atlas.mouse})
    per_section = df.groupby("section")["mouse"].nunique()
    bad = per_section[per_section > 1]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} sections span multiple mice; leave-one-mouse-out would "
            f"leak spatial neighbours. Examples: {list(bad.index[:5])}"
        )


def lognorm(counts: np.ndarray, target: float = NORM_TARGET) -> np.ndarray:
    """Library-size normalise to `target` counts, then log1p. Unsupervised, so it
    may be fit on train+val together without leakage."""
    totals = counts.sum(axis=1, keepdims=True)
    scaled = counts / np.maximum(totals, 1) * target
    return np.log1p(scaled).astype(np.float32)


# --------------------------------------------------------------------------- #
# >>> PLUG 1 of 2:  FEATURES  ------------------------------------------------ #
# Build the design matrix for one fold. `train_idx` are the only rows whose      #
# labels you may use (reference set); `eval_idx` are the rows being featurised.  #
# Anything derived from labels (class centroids, expression-kNN votes, ...) must #
# be computed with train_idx as reference and left-one-out for train rows, so it #
# stays honest for the rows you later score. The baseline below uses no labels.  #
# Return (X_train, X_eval) as float arrays with matching column order.           #
# --------------------------------------------------------------------------- #
def build_features(
    atlas: Atlas,
    logx: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Block A: log-normalised expression (fit-free, no leakage).
    expr_train, expr_eval = logx[train_idx], logx[eval_idx]

    # Block B: one-hot of anatomical metadata, encoder fit on train only.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    meta_train = encoder.fit_transform(atlas.meta.iloc[train_idx])
    meta_eval = encoder.transform(atlas.meta.iloc[eval_idx])

    x_train = np.hstack([expr_train, meta_train]).astype(np.float32)
    x_eval = np.hstack([expr_eval, meta_eval]).astype(np.float32)

    # --- add extra blocks here, e.g. fold-honest label features -------------- #
    # from label_feats import label_features                                    #
    # y_train_idx = np.searchsorted(classes, atlas.labels[train_idx])           #
    # extra = label_features(logx, train_idx, y_train_idx, len(classes), seed)  #
    # x_train = np.hstack([x_train, extra[train_idx]])                          #
    # x_eval  = np.hstack([x_eval,  extra[eval_idx]])                           #
    # ------------------------------------------------------------------------ #
    return x_train, x_eval


# --------------------------------------------------------------------------- #
# >>> PLUG 2 of 2:  MODEL  --------------------------------------------------- #
# Train on (X_train, y_train), return predicted string labels for X_eval.        #
# Swap this body for CatBoost / an MLP / a stack to test a different learner.    #
#                                                                                #
# CAUTION -- REGULARISATION MATTERS UNDER LEAVE-ONE-MOUSE-OUT. An aggressive     #
# GBDT (row subsampling + deep trees + no leaf floor + no L2) overfits a         #
# training-specific expression direction and then routes an ENTIRE slightly      #
# shifted held-out mouse into one leaf, collapsing that mouse to ~majority-class #
# accuracy (0.06-0.13) while OTHER held-out mice still score ~0.91. Random CV     #
# hides this completely. Which mouse collapses even changes with the training    #
# subsample -- a classic covariate-shift artifact, NOT a data/batch problem      #
# (plain logistic regression is stable at ~0.89 on the same folds). The params   #
# below (no subsample, num_leaves=31, min_child_samples=200, reg_lambda=5) are    #
# the regularisation that removed the collapse and lifted the collapsing mice to  #
# ~0.92. If you retune, watch fold_std, not just the mean.                        #
# --------------------------------------------------------------------------- #
def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        reg_lambda=5.0,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model.predict(x_eval)


# --------------------------------------------------------------------------- #
# Harness (keeps evaluation honest -- normally no need to edit below)          #
# --------------------------------------------------------------------------- #
def _subsample_mask(atlas: Atlas, cap_per_mouse: int, seed: int) -> np.ndarray:
    """Deterministic per-mouse cap for quick smoke tests. Returns a boolean mask."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(atlas.mouse), dtype=bool)
    for m in np.unique(atlas.mouse):
        rows = np.where(atlas.mouse == m)[0]
        if len(rows) > cap_per_mouse:
            rows = rng.choice(rows, cap_per_mouse, replace=False)
        keep[rows] = True
    return keep


def run_one_fold(
    atlas: Atlas, logx: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
    classes: np.ndarray, seed: int, n_estimators: int,
) -> tuple[float, np.ndarray]:
    # Hard leakage guard: the validation mouse must be absent from training.
    if set(atlas.mouse[train_idx]) & set(atlas.mouse[eval_idx]):
        raise RuntimeError("A validation mouse leaked into the training rows.")
    x_train, x_eval = build_features(
        atlas, logx, train_idx, eval_idx, classes, seed
    )
    pred = fit_predict(
        x_train, atlas.labels[train_idx], x_eval, seed, n_estimators
    )
    return accuracy_score(atlas.labels[eval_idx], pred), pred


def pick_judge_mouse(mice: np.ndarray, seed: int) -> str:
    if FINAL_JUDGE_MOUSE is not None:
        if FINAL_JUDGE_MOUSE not in set(mice):
            raise ValueError(f"FINAL_JUDGE_MOUSE {FINAL_JUDGE_MOUSE!r} not in atlas")
        return FINAL_JUDGE_MOUSE
    return str(np.random.default_rng(seed).choice(sorted(mice)))


def dev_cv(atlas: Atlas, logx: np.ndarray, dev_mice: list[str], classes: np.ndarray,
           seed: int, n_estimators: int) -> pd.DataFrame:
    print(f"\n=== DEV leave-one-mouse-out over {len(dev_mice)} mice "
          f"(rotating folds) ===")
    rows = []
    for held in dev_mice:
        start = time.time()
        eval_idx = np.where(atlas.mouse == held)[0]
        train_idx = np.where(
            np.isin(atlas.mouse, [m for m in dev_mice if m != held])
        )[0]
        acc, _ = run_one_fold(
            atlas, logx, train_idx, eval_idx, classes, seed, n_estimators
        )
        rows.append({"held_mouse": held, "n_val": len(eval_idx),
                     "n_train": len(train_idx), "accuracy": acc})
        print(f"  held={held:>3}  n_val={len(eval_idx):6d}  "
              f"acc={acc:.4f}  ({time.time() - start:.0f}s)")
    df = pd.DataFrame(rows)
    print(f"\n  DEV CV accuracy = {df.accuracy.mean():.4f} "
          f"+/- {df.accuracy.std():.4f}  "
          f"(min {df.accuracy.min():.4f} / max {df.accuracy.max():.4f})")
    return df


def final_eval(atlas: Atlas, logx: np.ndarray, dev_mice: list[str], judge: str,
               classes: np.ndarray, seed: int, n_estimators: int) -> float:
    print(f"\n=== FINAL judge: train on {len(dev_mice)} dev mice, score {judge} ===")
    print("  (Look at this number rarely -- ideally once per chosen method.)")
    eval_idx = np.where(atlas.mouse == judge)[0]
    train_idx = np.where(np.isin(atlas.mouse, dev_mice))[0]
    start = time.time()
    acc, pred = run_one_fold(
        atlas, logx, train_idx, eval_idx, classes, seed, n_estimators
    )
    print(f"  judge={judge}  n={len(eval_idx)}  FINAL accuracy = {acc:.4f}  "
          f"({time.time() - start:.0f}s)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Cell_row": eval_idx,
        "true": atlas.labels[eval_idx],
        "pred": pred,
    }).to_csv(OUT_DIR / f"final_judge_{judge}_predictions.csv", index=False)
    return acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["dev", "final", "all"], default="dev")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--final-mouse", default=None,
                   help="Override the judge mouse (else deterministic from seed).")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: cap 2000 cells/mouse and 80 trees.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    atlas = load_atlas()

    if args.quick:
        mask = _subsample_mask(atlas, cap_per_mouse=2000, seed=args.seed)
        atlas = Atlas(
            counts=atlas.counts[mask], genes=atlas.genes,
            labels=atlas.labels[mask], mouse=atlas.mouse[mask],
            section=atlas.section[mask], meta=atlas.meta.iloc[mask].reset_index(drop=True),
        )
        args.n_estimators = min(args.n_estimators, 80)
        print(f"[quick] subsampled to {len(atlas.mouse)} cells, "
              f"{args.n_estimators} trees.")

    logx = lognorm(atlas.counts)
    classes = np.unique(atlas.labels)
    mice = np.unique(atlas.mouse)

    judge = args.final_mouse or pick_judge_mouse(mice, args.seed)
    dev_mice = sorted(m for m in mice if m != judge)
    print(f"Final judge mouse (held out entirely): {judge}")
    print(f"Dev mice ({len(dev_mice)}): {dev_mice}")

    if args.mode in {"dev", "all"}:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df = dev_cv(atlas, logx, dev_mice, classes, args.seed, args.n_estimators)
        df.to_csv(OUT_DIR / "dev_loso_folds.csv", index=False)
    if args.mode in {"final", "all"}:
        final_eval(atlas, logx, dev_mice, judge, classes, args.seed, args.n_estimators)


if __name__ == "__main__":
    main()
