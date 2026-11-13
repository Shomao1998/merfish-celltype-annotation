#!/usr/bin/env python3
"""SHAP explainability for the LightGBM cell-type classifier.

Why: the ensemble is strong but opaque. SHAP shows *which inputs drive the model* —
a sanity check that it leans on biologically sensible signals (marker genes, E/I,
anatomy) rather than artifacts, and a readable story for non-ML reviewers.

What it does, faithfully to the LightGBM path of the final model:
  * features = library-size-normalized log expression (200 genes, named) +
    one-hot of the anatomical metadata (E/I, Region, Segment, AP position, Gender);
  * trains one LightGBM on the training cells;
  * runs shap.TreeExplainer on a subsample and writes two figures to reports/figures/:
      - shap_global_importance.png  (top features overall, genes vs. metadata)
      - shap_beeswarm.png           (per-cell SHAP for the most abundant class)

Usage:
    python src/shap_analysis.py [DATA_DIR]      # DATA_DIR defaults to ./data
See data/README.md to obtain the competition data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "MERFISH_cell_type_annotation"
CATEGORICAL = ["Excitatory_vs_Inhibitory", "Region", "Segment", "AP_position", "Gender"]
SEED = 0
SHAP_SAMPLE = 2000

NAVY, TEAL, ORANGE = "#0E2841", "#156082", "#E97132"


def lognorm(counts: np.ndarray, target: float = 500.0) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / np.maximum(totals, 1) * target).astype(np.float32)


def main() -> None:
    counts = pd.read_csv(DATA_DIR / "counts_train.csv", index_col=0)
    meta = pd.read_csv(DATA_DIR / "meta_train.csv", index_col=0).loc[counts.index]
    genes = [str(g) for g in counts.columns]
    gene_set = set(genes)

    x_expr = lognorm(counts.values.astype(np.float32))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_cat = encoder.fit_transform(meta[CATEGORICAL].fillna("NA").astype(str))
    cat_names = list(encoder.get_feature_names_out(CATEGORICAL))
    x = np.hstack([x_expr, x_cat]).astype(np.float32)
    feat_names = genes + cat_names
    y = meta[TARGET].astype(str).to_numpy()
    print(f"features: {x.shape[1]} ({len(genes)} genes + {len(cat_names)} metadata one-hot); "
          f"cells: {x.shape[0]}; classes: {len(np.unique(y))}")

    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    ).fit(x, y)

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x), min(SHAP_SAMPLE, len(x)), replace=False)
    xs = x[sample]
    print(f"computing SHAP on {len(xs)} cells ...")
    sv = shap.TreeExplainer(model).shap_values(xs)
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (n, feat, classes)
    if arr.ndim != 3:
        raise RuntimeError(f"unexpected SHAP shape {arr.shape}")

    # ---- Figure 1: global importance (mean |SHAP| over cells & classes) ----
    glob = np.abs(arr).mean(axis=(0, 2))
    top = np.argsort(-glob)[:18][::-1]
    names = [feat_names[i] for i in top]
    vals = glob[top]
    colors = [TEAL if feat_names[i] in gene_set else ORANGE for i in top]
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    ax.barh(range(len(top)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("mean |SHAP value|  (impact on model output)")
    ax.set_title("SHAP global feature importance — what the model relies on",
                 color=NAVY, fontweight="bold", fontsize=12.5, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#E4E9ED", linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label="gene expression"),
                       Patch(color=ORANGE, label="anatomical metadata")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_global_importance.png", bbox_inches="tight")
    print("saved shap_global_importance.png")

    # ---- Figure 2: beeswarm for the most abundant class ----
    classes = model.classes_
    cls_name = pd.Series(y).value_counts().index[0]
    cls = int(np.where(classes == cls_name)[0][0])
    plt.figure()
    shap.summary_plot(arr[:, :, cls], xs, feature_names=feat_names,
                      max_display=14, show=False, plot_size=(7.4, 5.2))
    plt.title(f"SHAP beeswarm — drivers of class '{cls_name}'",
              color=NAVY, fontweight="bold", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    print(f"saved shap_beeswarm.png  (class = {cls_name})")

    print("\nTop 15 global features:")
    for i in np.argsort(-glob)[:15]:
        kind = "gene" if feat_names[i] in gene_set else "meta"
        print(f"  {glob[i]:.4f}  [{kind}]  {feat_names[i]}")


if __name__ == "__main__":
    main()
