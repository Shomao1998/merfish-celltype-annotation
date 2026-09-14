#!/usr/bin/env python3
"""Train-only reference classifiers under Mouse and Dataset Group CV.

This audit intentionally uses no external atlas or pretrained model. For every
outer fold, the labelled reference profiles are rebuilt from only the official
training rows in that fold.

Models
------
SciBet-style
    A smoothed class multinomial. Each class template is its pooled raw count
    vector with Laplace smoothing. A query is assigned to the template with
    the largest multinomial log-likelihood (equivalently, smallest KL distance
    up to query-only constants).

SingleR-style
    A class-centroid rank-correlation classifier. Each class template is the
    mean library-normalized log-expression profile. Queries are compared with
    all templates using Spearman correlation across the 200 measured genes.

Both gene-only predictions and predictions with the existing fold-safe E/I
allowed-class mask are reported so the value of the reference classifier is
not confused with the value of known E/I metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, confusion_matrix

from v025_group_cv_ablation import TARGET, fold_allowed, make_splits


LAPLACE_ALPHA = 1.0


def normalized_log_expression(raw: np.ndarray) -> np.ndarray:
    total = raw.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(raw / total * 100.0)


def row_standardize(values: np.ndarray) -> np.ndarray:
    """Center and L2-normalize rows for vectorized Pearson correlation."""
    centered = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return centered / norm


def fit_scibet_scores(
    raw: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    alpha: float = LAPLACE_ALPHA,
) -> np.ndarray:
    """Return class log-likelihood scores from fold-specific templates."""
    scores = np.full((len(valid), len(classes)), -np.inf, dtype=float)
    query = raw[valid]
    query_total = query.sum(axis=1, keepdims=True)
    query_total[query_total == 0] = 1.0
    query_composition = query / query_total

    for class_no, label in enumerate(classes):
        members = train[y[train] == label]
        if len(members) == 0:
            continue
        # This is the direct Laplace-smoothed multinomial estimate. Pooling
        # counts is important here: in this dataset the median gene mean is
        # only 0.041, so adding one to a per-cell mean would flatten almost the
        # entire class template toward a uniform distribution.
        pooled_counts = raw[members].sum(axis=0)
        probability = (pooled_counts + alpha) / (
            pooled_counts.sum() + alpha * raw.shape[1]
        )
        scores[:, class_no] = query_composition @ np.log(probability)
    return scores


def fit_singler_scores(
    log_expression: np.ndarray,
    ranked_queries: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return Spearman scores to fold-specific mean class centroids."""
    centroids = np.zeros((len(classes), log_expression.shape[1]), dtype=float)
    present = np.zeros(len(classes), dtype=bool)
    for class_no, label in enumerate(classes):
        members = train[y[train] == label]
        if len(members) == 0:
            continue
        centroids[class_no] = log_expression[members].mean(axis=0)
        present[class_no] = True

    ranked_centroids = rankdata(centroids[present], axis=1, method="average")
    query_z = row_standardize(ranked_queries[valid])
    centroid_z = row_standardize(ranked_centroids)
    scores = np.full((len(valid), len(classes)), -np.inf, dtype=float)
    scores[:, present] = query_z @ centroid_z.T
    return scores


def fit_centroid_pearson_scores(
    log_expression: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Sensitivity check: Pearson rather than rank correlation to centroids."""
    centroids = np.zeros((len(classes), log_expression.shape[1]), dtype=float)
    present = np.zeros(len(classes), dtype=bool)
    for class_no, label in enumerate(classes):
        members = train[y[train] == label]
        if len(members) == 0:
            continue
        centroids[class_no] = log_expression[members].mean(axis=0)
        present[class_no] = True
    scores = np.full((len(valid), len(classes)), -np.inf, dtype=float)
    scores[:, present] = (
        row_standardize(log_expression[valid])
        @ row_standardize(centroids[present]).T
    )
    return scores


def predict_from_scores(
    scores: np.ndarray,
    classes: np.ndarray,
    ei_rows: np.ndarray | None = None,
    allowed: dict[str, set[str]] | None = None,
) -> np.ndarray:
    if ei_rows is None or allowed is None:
        return classes[np.argmax(scores, axis=1)]
    mask = np.array(
        [
            [label in allowed.get(group, set(classes)) for label in classes]
            for group in ei_rows
        ],
        dtype=bool,
    )
    return classes[np.where(mask, scores, -np.inf).argmax(axis=1)]


def run_group_audit(
    raw: np.ndarray,
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_expression = normalized_log_expression(raw)
    ranked_queries = rankdata(log_expression, axis=1, method="average")
    ei = (
        meta["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for scheme, group_col in [("mouse", "Mouse_ID"), ("dataset", "Datasets")]:
        groups = meta[group_col].astype(str).to_numpy()
        folds = make_splits(scheme, y, groups)
        prediction = {
            "scibet_gene_only": np.empty(len(y), dtype=object),
            "scibet_ei_mask": np.empty(len(y), dtype=object),
            "singler_gene_only": np.empty(len(y), dtype=object),
            "singler_ei_mask": np.empty(len(y), dtype=object),
            "centroid_pearson_gene_only": np.empty(len(y), dtype=object),
            "centroid_pearson_ei_mask": np.empty(len(y), dtype=object),
        }
        all_scibet_scores = np.full((len(y), len(classes)), -np.inf, dtype=float)
        all_singler_scores = np.full_like(all_scibet_scores, -np.inf)
        all_pearson_scores = np.full_like(all_scibet_scores, -np.inf)
        print(f"\n=== {scheme}: {len(folds)} held-group folds ===", flush=True)
        start = time.time()

        for fold_no, (train, valid) in enumerate(folds):
            fold_start = time.time()
            scibet_scores = fit_scibet_scores(
                raw, y, classes, train, valid
            )
            singler_scores = fit_singler_scores(
                log_expression,
                ranked_queries,
                y,
                classes,
                train,
                valid,
            )
            pearson_scores = fit_centroid_pearson_scores(
                log_expression, y, classes, train, valid
            )
            all_scibet_scores[valid] = scibet_scores
            all_singler_scores[valid] = singler_scores
            all_pearson_scores[valid] = pearson_scores
            allowed = fold_allowed(y, ei, train)
            fold_predictions = {
                "scibet_gene_only": predict_from_scores(scibet_scores, classes),
                "scibet_ei_mask": predict_from_scores(
                    scibet_scores, classes, ei[valid], allowed
                ),
                "singler_gene_only": predict_from_scores(singler_scores, classes),
                "singler_ei_mask": predict_from_scores(
                    singler_scores, classes, ei[valid], allowed
                ),
                "centroid_pearson_gene_only": predict_from_scores(
                    pearson_scores, classes
                ),
                "centroid_pearson_ei_mask": predict_from_scores(
                    pearson_scores, classes, ei[valid], allowed
                ),
            }
            held = ",".join(sorted(np.unique(groups[valid]).astype(str)))
            for model, fold_pred in fold_predictions.items():
                prediction[model][valid] = fold_pred
                fold_rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "fold": fold_no,
                        "held_groups": held,
                        "n_valid": len(valid),
                        "accuracy": accuracy_score(y[valid], fold_pred),
                    }
                )
            print(
                f"  fold {fold_no + 1}/{len(folds)} held={held}: "
                f"SciBet={accuracy_score(y[valid], fold_predictions['scibet_ei_mask']):.4f}, "
                f"SingleR={accuracy_score(y[valid], fold_predictions['singler_ei_mask']):.4f} "
                f"({time.time() - fold_start:.2f}s)",
                flush=True,
            )

        for model, pred in prediction.items():
            relevant = [
                row
                for row in fold_rows
                if row["scheme"] == scheme and row["model"] == model
            ]
            fold_accuracy = np.array(
                [row["accuracy"] for row in relevant], dtype=float
            )
            summary_rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "pooled_accuracy": accuracy_score(y, pred),
                    "fold_mean": fold_accuracy.mean(),
                    "fold_std": fold_accuracy.std(),
                    "fold_min": fold_accuracy.min(),
                    "fold_max": fold_accuracy.max(),
                    "n_folds": len(folds),
                    "elapsed_seconds_for_scheme": time.time() - start,
                }
            )

        pd.DataFrame(
            {
                "cell_id": meta.index.astype(str),
                "true_label": y,
                **{model: pred for model, pred in prediction.items()},
            }
        ).to_csv(out_dir / f"reference_predictions_{scheme}.csv", index=False)
        np.savez_compressed(
            out_dir / f"reference_scores_{scheme}.npz",
            scibet_scores=all_scibet_scores,
            singler_scores=all_singler_scores,
            pearson_scores=all_pearson_scores,
        )

        # Save per-class recall for error analysis without selecting on test data.
        recall_rows: list[dict[str, object]] = []
        for model, pred in prediction.items():
            cm = confusion_matrix(y, pred, labels=classes)
            support = cm.sum(axis=1)
            recall = np.divide(
                np.diag(cm),
                support,
                out=np.zeros_like(support, dtype=float),
                where=support > 0,
            )
            recall_rows.extend(
                {
                    "scheme": scheme,
                    "model": model,
                    "cell_type": label,
                    "support": int(n),
                    "recall": value,
                }
                for label, n, value in zip(classes, support, recall)
            )
        pd.DataFrame(recall_rows).to_csv(
            out_dir / f"reference_class_recall_{scheme}.csv", index=False
        )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(fold_rows)
    summary.to_csv(out_dir / "reference_group_summary.csv", index=False)
    detail.to_csv(out_dir / "reference_group_folds.csv", index=False)
    return summary, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(data_dir / "counts_train.csv", index_col=0)
    meta = pd.read_csv(data_dir / "meta_train.csv", index_col=0).loc[counts.index]
    raw = counts.to_numpy(dtype=float)
    y = meta[TARGET].astype(str).to_numpy()
    classes = np.unique(y)

    manifest = {
        "data_dir": str(data_dir.resolve()),
        "n_cells": len(y),
        "n_genes": raw.shape[1],
        "n_classes": len(classes),
        "external_reference_data": False,
        "scibet": {
            "template": "pooled raw counts per class",
            "laplace_alpha_per_gene": LAPLACE_ALPHA,
            "query": "per-cell raw count composition",
            "similarity": "multinomial log-likelihood / negative cross-entropy",
        },
        "singler": {
            "template": "mean normalized log-expression per class",
            "query": "normalized log-expression",
            "similarity": "Spearman correlation across all measured genes",
        },
        "centroid_pearson_sensitivity": {
            "template": "mean normalized log-expression per class",
            "query": "normalized log-expression",
            "similarity": "Pearson correlation across all measured genes",
            "purpose": "checks whether sparse rank ties explain SingleR-style failure",
        },
    }
    (out_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary, _ = run_group_audit(raw, meta, y, classes, out_dir)
    print("\n=== SUMMARY ===", flush=True)
    print(
        summary[
            ["scheme", "model", "pooled_accuracy", "fold_mean", "fold_std", "fold_min"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
