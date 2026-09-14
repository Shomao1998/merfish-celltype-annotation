#!/usr/bin/env python3
"""v0.31 robust five-model MERFISH classifier.

Purpose
-------
This version is designed for the organizer's unseen-dataset evaluation. It
removes Mouse_ID, Section_ID and Datasets from the predictive feature matrix,
while retaining them for grouped validation and within-section transforms.

The primary prediction is an equal probability mean of five heterogeneous
models: LightGBM, logistic regression, MLP, KNN and CatBoost. The learned
stacker from v0.3 is intentionally removed because true nested Group CV did
not show a stable generalization benefit.

Robust features
---------------
* log-normalized expression for the 200 measured genes;
* E/I, Region, Segment, AP_position and Gender one-hot features;
* four within-Section_ID QC percentile ranks;
* fold-safe Segment/Region/joint target encodings for LR and MLP only.

Direct model features intentionally excluded
--------------------------------------------
* Mouse_ID, Section_ID and Datasets;
* raw center_x/center_y;
* raw depth, volume and density.

Examples
--------
Create grouped OOF probabilities for Qwen (audited one-seed default):

    python src/v0.31-Robust-NoID-5ModelMean.py \
      --mode oof --schemes mouse dataset --seeds 0

Train on all official training rows and predict the current test files:

    python src/v0.31-Robust-NoID-5ModelMean.py \
      --mode predict --seeds 0 1 2 3 4

Run both stages with seed 0:

    python src/v0.31-Robust-NoID-5ModelMean.py --mode all

The OOF NPZ files contain a stable, ID-validated contract for the separate
Qwen reranker. This script never uses external reference data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VERSION = "v0.31"
TARGET = "MERFISH_cell_type_annotation"
PREDICTION_COLUMN = "MERFISH_cell_type_annotation.y"
BASE_MODELS = ("lgbm", "lr", "mlp", "knn", "cb")
STABLE_CATEGORICAL = (
    "Excitatory_vs_Inhibitory",
    "Region",
    "Segment",
    "AP_position",
    "Gender",
)
TE_COLUMNS = (
    ("Segment",),
    ("Region",),
    ("Segment", "Region", "Excitatory_vs_Inhibitory"),
)
GROUP_COLUMN = {"mouse": "Mouse_ID", "dataset": "Datasets"}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs" / "v031_robust_noid_5model"

LR_C = 0.3
TE_SMOOTH = 10.0
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "max_bin": 63,
    "n_jobs": -1,
    "verbose": -1,
}
CATBOOST_PARAMS = {
    "iterations": 400,
    "learning_rate": 0.06,
    "depth": 6,
    "subsample": 0.8,
    "rsm": 0.6,
    "bootstrap_type": "Bernoulli",
    "border_count": 63,
    "loss_function": "MultiClass",
    "thread_count": -1,
    "verbose": 0,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class DataBundle:
    counts_train: pd.DataFrame
    meta_train: pd.DataFrame
    counts_test: pd.DataFrame | None
    meta_test: pd.DataFrame | None


@dataclass(frozen=True)
class FeatureStore:
    expression: np.ndarray
    section_qc: np.ndarray
    meta: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust no-ID five-model classifier for unseen datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--submission-out",
        type=Path,
        default=None,
        help="Defaults to <out-dir>/v0.31-Prediction-Robust5Mean.csv",
    )
    parser.add_argument(
        "--mode", choices=["predict", "oof", "all"], default="predict"
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["mouse", "dataset"],
        default=["mouse", "dataset"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "Seed 0 matches the grouped audit. Optional 0 1 2 3 4 bagging is "
            "supported but its grouped gain has not been established."
        ),
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicates")
    if args.submission_out is None:
        args.submission_out = (
            args.out_dir / "v0.31-Prediction-Robust5Mean.csv"
        )
    return args


def read_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    index_name = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(path, index_col=0, dtype={index_name: str})
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate primary keys in {path}")
    return frame


def align_metadata(counts: pd.DataFrame, meta: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = counts.index.difference(meta.index)
    extra = meta.index.difference(counts.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"{name} counts/meta ID mismatch: missing_meta={len(missing)}, "
            f"extra_meta={len(extra)}"
        )
    return meta.loc[counts.index].copy()


def load_data(data_dir: Path, need_test: bool) -> DataBundle:
    counts_train = read_indexed_csv(data_dir / "counts_train.csv")
    meta_train = align_metadata(
        counts_train, read_indexed_csv(data_dir / "meta_train.csv"), "train"
    )
    if TARGET not in meta_train or meta_train[TARGET].isna().any():
        raise ValueError(f"meta_train.csv must contain complete {TARGET!r} labels")
    if counts_train.columns.duplicated().any():
        raise ValueError("counts_train.csv contains duplicate gene columns")

    required_meta = set(STABLE_CATEGORICAL) | {
        "volume",
        "Section_ID",
        "Mouse_ID",
        "Datasets",
    }
    missing_meta = required_meta.difference(meta_train.columns)
    if missing_meta:
        raise ValueError(f"meta_train.csv is missing columns: {sorted(missing_meta)}")

    if not need_test:
        return DataBundle(counts_train, meta_train, None, None)

    counts_test = read_indexed_csv(data_dir / "counts_test.csv")
    meta_test = align_metadata(
        counts_test, read_indexed_csv(data_dir / "meta_test.csv"), "test"
    )
    missing_genes = counts_train.columns.difference(counts_test.columns)
    extra_genes = counts_test.columns.difference(counts_train.columns)
    if len(missing_genes) or len(extra_genes):
        raise ValueError(
            "Train/test gene panels differ: "
            f"missing_in_test={missing_genes.tolist()}, "
            f"extra_in_test={extra_genes.tolist()}"
        )
    counts_test = counts_test.loc[:, counts_train.columns]
    missing_test_meta = required_meta.difference(meta_test.columns)
    if missing_test_meta:
        raise ValueError(f"meta_test.csv is missing columns: {sorted(missing_test_meta)}")
    return DataBundle(counts_train, meta_train, counts_test, meta_test)


def array_fingerprint(counts: pd.DataFrame, meta: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update("\n".join(map(str, counts.index)).encode())
    digest.update("\n".join(map(str, counts.columns)).encode())
    digest.update("\n".join(meta[TARGET].astype(str)).encode())
    return digest.hexdigest()


def expression_and_qc(counts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = counts.to_numpy(dtype=float)
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Count matrix contains negative or non-finite values")
    totals = raw.sum(axis=1, keepdims=True)
    safe_totals = totals.copy()
    safe_totals[safe_totals == 0] = 1.0
    expression = np.log1p(raw / safe_totals * 100.0)
    return expression, totals.ravel()


def within_section_qc(
    counts: pd.DataFrame, totals: np.ndarray, meta: pd.DataFrame
) -> np.ndarray:
    raw = counts.to_numpy(dtype=float)
    volume = np.clip(meta["volume"].to_numpy(dtype=float), 1e-12, None)
    qc = pd.DataFrame(
        {
            "log_depth": np.log1p(totals),
            "log_volume": np.log1p(volume),
            "log_density": np.log1p(totals / volume),
            "detected_genes": (raw > 0).sum(axis=1).astype(float),
        },
        index=meta.index,
    )
    sections = meta["Section_ID"].fillna("MISSING").astype(str)
    ranked = qc.groupby(sections, sort=False).rank(method="average", pct=True)
    return ranked.fillna(0.5).to_numpy(dtype=float)


def build_store(counts: pd.DataFrame, meta: pd.DataFrame) -> FeatureStore:
    expression, totals = expression_and_qc(counts)
    section_qc = within_section_qc(counts, totals, meta)
    if not np.isfinite(expression).all() or not np.isfinite(section_qc).all():
        raise ValueError("Engineered numeric features contain non-finite values")
    return FeatureStore(expression, section_qc, meta)


def assert_sections_nested(meta: pd.DataFrame) -> None:
    """Prevent label-free section transforms from crossing held-out groups."""
    sections = meta["Section_ID"].fillna("MISSING").astype(str)
    for group_column in ("Mouse_ID", "Datasets"):
        group_values = meta[group_column].fillna("MISSING").astype(str)
        crossing = (
            pd.DataFrame({"section": sections, "group": group_values})
            .groupby("section", sort=False)["group"]
            .nunique()
        )
        crossing = crossing[crossing > 1]
        if len(crossing):
            raise ValueError(
                f"{len(crossing)} Section_ID values cross {group_column} groups; "
                "within-section QC must be recomputed inside each fold"
            )


def categorical_frame(meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta.loc[:, STABLE_CATEGORICAL].copy()
    for column in frame:
        frame[column] = frame[column].fillna("MISSING").astype(str)
    return frame


def make_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float64,
    }
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def fit_static_features(
    store: FeatureStore,
    fit_rows: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, OneHotEncoder]:
    categories = categorical_frame(store.meta)
    encoder = make_encoder().fit(categories.iloc[fit_rows])
    encoded = encoder.transform(categories.iloc[rows])
    matrix = np.hstack(
        [store.expression[rows], store.section_qc[rows], encoded]
    ).astype(float, copy=False)
    return matrix, encoder


def transform_static_features(
    store: FeatureStore,
    rows: np.ndarray,
    encoder: OneHotEncoder,
) -> np.ndarray:
    categories = categorical_frame(store.meta)
    encoded = encoder.transform(categories.iloc[rows])
    return np.hstack(
        [store.expression[rows], store.section_qc[rows], encoded]
    ).astype(float, copy=False)


def target_encode(
    meta: pd.DataFrame,
    columns: tuple[str, ...],
    fit_rows: np.ndarray,
    y_fit: np.ndarray,
    classes: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    keys = (
        meta.loc[:, list(columns)]
        .fillna("MISSING")
        .astype(str)
        .agg("|".join, axis=1)
        .to_numpy()
    )
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    encoded_y = np.array([class_to_idx[label] for label in y_fit], dtype=int)
    prior = np.bincount(encoded_y, minlength=len(classes)).astype(float)
    prior /= prior.sum()
    group_counts: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(classes), dtype=float)
    )
    for key, label_index in zip(keys[fit_rows], encoded_y):
        group_counts[key][label_index] += 1.0
    posteriors = {
        key: (counts + TE_SMOOTH * prior) / (counts.sum() + TE_SMOOTH)
        for key, counts in group_counts.items()
    }
    return np.array([posteriors.get(key, prior) for key in keys[rows]], dtype=float)


def add_target_encodings(
    static: np.ndarray,
    meta: pd.DataFrame,
    fit_rows: np.ndarray,
    y_fit: np.ndarray,
    classes: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    blocks = [static]
    blocks.extend(
        target_encode(meta, columns, fit_rows, y_fit, classes, rows)
        for columns in TE_COLUMNS
    )
    return np.hstack(blocks)


def place_probabilities(
    destination: np.ndarray,
    model_classes: np.ndarray,
    probabilities: np.ndarray,
    class_to_idx: dict[str, int],
) -> None:
    columns = np.array(
        [class_to_idx[str(label)] for label in model_classes], dtype=int
    )
    destination[:, columns] = probabilities


def five_model_probabilities(
    store: FeatureStore,
    y: np.ndarray,
    classes: np.ndarray,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_train = y[train_rows]
    encoded_y = np.array([class_to_idx[label] for label in y_train], dtype=int)
    n_eval = len(eval_rows)
    result = {
        model: np.zeros((n_eval, len(classes)), dtype=float)
        for model in BASE_MODELS
    }

    static_train, encoder = fit_static_features(store, train_rows, train_rows)
    static_eval = transform_static_features(store, eval_rows, encoder)
    linear_train = add_target_encodings(
        static_train, store.meta, train_rows, y_train, classes, train_rows
    )
    linear_eval = add_target_encodings(
        static_eval, store.meta, train_rows, y_train, classes, eval_rows
    )

    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=seed).fit(
        static_train, y_train
    )
    place_probabilities(
        result["lgbm"],
        lgbm.classes_,
        lgbm.predict_proba(static_eval),
        class_to_idx,
    )

    linear_scaler = StandardScaler().fit(linear_train)
    z_train = linear_scaler.transform(linear_train)
    z_eval = linear_scaler.transform(linear_eval)
    lr = LogisticRegression(
        max_iter=2000, C=LR_C, n_jobs=-1, random_state=seed
    ).fit(z_train, y_train)
    place_probabilities(
        result["lr"], lr.classes_, lr.predict_proba(z_eval), class_to_idx
    )

    mlp_kwargs = {
        "hidden_layer_sizes": (128,),
        "alpha": 1e-3,
        "max_iter": 300,
        "early_stopping": True,
        "random_state": seed,
    }
    try:
        mlp = MLPClassifier(**mlp_kwargs).fit(z_train, encoded_y)
    except ValueError as error:
        warnings.warn(
            f"MLP early stopping failed ({error}); retrying without early stopping"
        )
        mlp_kwargs["early_stopping"] = False
        mlp = MLPClassifier(**mlp_kwargs).fit(z_train, encoded_y)
    result["mlp"][:, mlp.classes_.astype(int)] = mlp.predict_proba(z_eval)

    knn_scaler = StandardScaler().fit(static_train)
    knn_train = knn_scaler.transform(static_train)
    knn_eval = knn_scaler.transform(static_eval)
    n_components = max(1, min(50, knn_train.shape[0] - 1, knn_train.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed).fit(knn_train)
    knn = KNeighborsClassifier(
        n_neighbors=min(30, len(train_rows)), weights="distance", n_jobs=-1
    ).fit(pca.transform(knn_train), y_train)
    place_probabilities(
        result["knn"],
        knn.classes_,
        knn.predict_proba(pca.transform(knn_eval)),
        class_to_idx,
    )

    # CatBoost is trained with local contiguous class IDs so folds remain valid
    # even when a globally rare class is absent from the training groups.
    local_labels, local_y = np.unique(y_train, return_inverse=True)
    catboost = CatBoostClassifier(
        **CATBOOST_PARAMS, random_seed=seed
    ).fit(static_train, local_y)
    cat_probabilities = catboost.predict_proba(static_eval)
    cat_class_indices = np.asarray(catboost.classes_, dtype=int)
    if cat_probabilities.shape[1] != len(cat_class_indices):
        raise RuntimeError("CatBoost class/probability width mismatch")
    place_probabilities(
        result["cb"],
        local_labels[cat_class_indices],
        cat_probabilities,
        class_to_idx,
    )

    for model, probabilities in result.items():
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise RuntimeError(f"Invalid probabilities from {model}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"Probability rows from {model} do not sum to one")
    return result


def make_group_splits(
    scheme: str, y: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros((len(y), 1), dtype=float)
    if scheme == "mouse":
        splitter = GroupKFold(n_splits=5)
    elif scheme == "dataset":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(scheme)
    return list(splitter.split(dummy, y, groups))


def allowed_by_ei(
    y: np.ndarray, ei: np.ndarray, train_rows: np.ndarray
) -> dict[str, set[str]]:
    return {
        value: set(y[train_rows][ei[train_rows] == value])
        for value in np.unique(ei[train_rows])
    }


def mask_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    ei_rows: np.ndarray,
    allowed: dict[str, set[str]],
) -> np.ndarray:
    mask = np.array(
        [
            [label in allowed.get(value, set(classes)) for label in classes]
            for value in ei_rows
        ],
        dtype=bool,
    )
    masked = np.where(mask, probabilities, 0.0)
    row_sums = masked.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("E/I mask removed all labels for at least one cell")
    return masked / row_sums


def probability_variants(
    model_probabilities: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        "blend2": 0.5 * (
            model_probabilities["lgbm"] + model_probabilities["lr"]
        ),
        "mean5": np.mean(
            [model_probabilities[name] for name in BASE_MODELS], axis=0
        ),
    }


def topk_recall(y: np.ndarray, classes: np.ndarray, p: np.ndarray, k: int) -> float:
    columns = np.argsort(-p, axis=1)[:, : min(k, len(classes))]
    return float(
        np.mean(
            [truth in set(classes[row_columns]) for truth, row_columns in zip(y, columns)]
        )
    )


def margin_mean(p: np.ndarray) -> float:
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return float(np.mean(np.max(top2, axis=1) - np.min(top2, axis=1)))


def run_oof_scheme(
    args: argparse.Namespace,
    bundle: DataBundle,
    store: FeatureStore,
    scheme: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = bundle.meta_train[TARGET].astype(str).to_numpy()
    classes = np.unique(y)
    group_column = GROUP_COLUMN[scheme]
    groups = bundle.meta_train[group_column].fillna("MISSING").astype(str).to_numpy()
    ei = (
        bundle.meta_train["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    splits = make_group_splits(scheme, y, groups)
    fold_id = np.full(len(y), -1, dtype=int)
    held_group = np.full(len(y), "", dtype=f"<U{max(16, max(map(len, groups)))}")
    seed_oof: list[dict[str, np.ndarray]] = []

    for seed in args.seeds:
        print(f"\n[{scheme}] seed={seed}", flush=True)
        oof = {
            model: np.zeros((len(y), len(classes)), dtype=float)
            for model in BASE_MODELS
        }
        for fold, (train_rows, valid_rows) in enumerate(splits):
            start = time.time()
            fold_probabilities = five_model_probabilities(
                store, y, classes, train_rows, valid_rows, seed
            )
            for model in BASE_MODELS:
                oof[model][valid_rows] = fold_probabilities[model]
            fold_id[valid_rows] = fold
            held = ",".join(sorted(np.unique(groups[valid_rows])))
            held_group[valid_rows] = held
            print(
                f"  fold {fold + 1}/{len(splits)} held={held} "
                f"n={len(valid_rows)} elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        seed_oof.append(oof)

    if np.any(fold_id < 0) or np.any(held_group == ""):
        raise RuntimeError(f"Incomplete {scheme} OOF assignment")
    averaged = {
        model: np.mean([item[model] for item in seed_oof], axis=0)
        for model in BASE_MODELS
    }
    variants = probability_variants(averaged)

    masked_mean5 = np.zeros_like(variants["mean5"])
    masked_blend2 = np.zeros_like(variants["blend2"])
    masked_base = {
        model: np.zeros_like(averaged[model]) for model in BASE_MODELS
    }
    predictions = {
        model: np.empty(len(y), dtype=object)
        for model in (*BASE_MODELS, "blend2", "mean5")
    }
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_rows, valid_rows) in enumerate(splits):
        allowed = allowed_by_ei(y, ei, train_rows)
        for model in BASE_MODELS:
            p_masked = mask_probabilities(
                averaged[model][valid_rows], classes, ei[valid_rows], allowed
            )
            masked_base[model][valid_rows] = p_masked
            predictions[model][valid_rows] = classes[p_masked.argmax(axis=1)]
        for name, target in [
            ("blend2", masked_blend2),
            ("mean5", masked_mean5),
        ]:
            p_masked = mask_probabilities(
                variants[name][valid_rows], classes, ei[valid_rows], allowed
            )
            target[valid_rows] = p_masked
            predictions[name][valid_rows] = classes[p_masked.argmax(axis=1)]
            fold_rows.append(
                {
                    "scheme": scheme,
                    "model": name,
                    "fold": fold,
                    "held_groups": ",".join(sorted(np.unique(groups[valid_rows]))),
                    "n_valid": len(valid_rows),
                    "accuracy": accuracy_score(y[valid_rows], predictions[name][valid_rows]),
                    "top3_recall": topk_recall(
                        y[valid_rows], classes, p_masked, 3
                    ),
                    "mean_margin": margin_mean(p_masked),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for model in (*BASE_MODELS, "blend2", "mean5"):
        model_fold_rows = [
            row for row in fold_rows if row["model"] == model
        ]
        if model in BASE_MODELS:
            # Base-model fold details were not appended above; calculate them here.
            model_fold_rows = []
            for fold, (_, valid_rows) in enumerate(splits):
                model_fold_rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "fold": fold,
                        "held_groups": ",".join(
                            sorted(np.unique(groups[valid_rows]))
                        ),
                        "n_valid": len(valid_rows),
                        "accuracy": accuracy_score(
                            y[valid_rows], predictions[model][valid_rows]
                        ),
                    }
                )
            fold_rows.extend(model_fold_rows)
        fold_accuracies = [row["accuracy"] for row in model_fold_rows]
        p_for_diagnostics = (
            masked_mean5
            if model == "mean5"
            else masked_blend2
            if model == "blend2"
            else masked_base[model]
        )
        summary_rows.append(
            {
                "version": VERSION,
                "scheme": scheme,
                "model": model,
                "seeds": "|".join(map(str, args.seeds)),
                "pooled_accuracy": accuracy_score(y, predictions[model]),
                "fold_mean": float(np.mean(fold_accuracies)),
                "fold_std": float(np.std(fold_accuracies)),
                "fold_min": float(np.min(fold_accuracies)),
                "fold_max": float(np.max(fold_accuracies)),
                "top3_recall": topk_recall(y, classes, p_for_diagnostics, 3),
                "mean_margin": margin_mean(p_for_diagnostics),
            }
        )

    artifact = args.out_dir / f"v031_oof_{scheme}.npz"
    np.savez_compressed(
        artifact,
        p_lgbm=averaged["lgbm"],
        p_lr=averaged["lr"],
        p_mlp=averaged["mlp"],
        p_knn=averaged["knn"],
        p_cb=averaged["cb"],
        p_blend2=variants["blend2"],
        p_mean5=variants["mean5"],
        p_mean5_masked=masked_mean5,
        classes=classes.astype(str),
        # Use fixed-width Unicode rather than object dtype so downstream tools
        # can load the artifact safely with allow_pickle=False.
        cell_ids=np.asarray(bundle.counts_train.index.astype(str).tolist(), dtype=str),
        fold_id=fold_id,
        held_group=held_group.astype(str),
        scheme=np.array(scheme),
        group_column=np.array(group_column),
    )
    print(f"Saved Qwen-compatible OOF artifact: {artifact}", flush=True)
    return summary_rows, fold_rows


def run_oof(args: argparse.Namespace, bundle: DataBundle) -> None:
    assert_sections_nested(bundle.meta_train)
    store = build_store(bundle.counts_train, bundle.meta_train)
    all_summary: list[dict[str, Any]] = []
    all_folds: list[dict[str, Any]] = []
    for scheme in args.schemes:
        summary, folds = run_oof_scheme(args, bundle, store, scheme)
        all_summary.extend(summary)
        all_folds.extend(folds)
        pd.DataFrame(all_summary).to_csv(
            args.out_dir / "v031_oof_summary.csv", index=False
        )
        pd.DataFrame(all_folds).to_csv(
            args.out_dir / "v031_oof_folds.csv", index=False
        )
    print("\n=== v0.31 GROUPED OOF SUMMARY ===")
    print(pd.DataFrame(all_summary).to_string(index=False))


def run_prediction(args: argparse.Namespace, bundle: DataBundle) -> None:
    if bundle.counts_test is None or bundle.meta_test is None:
        raise RuntimeError("Test data were not loaded")
    n_train = len(bundle.counts_train)
    combined_meta = pd.concat([bundle.meta_train, bundle.meta_test], axis=0)
    # Rank QC independently in train and test so a reused Section_ID cannot make
    # the new evaluation batch alter training features (or vice versa).
    train_store = build_store(bundle.counts_train, bundle.meta_train)
    test_store = build_store(bundle.counts_test, bundle.meta_test)
    store = FeatureStore(
        expression=np.vstack([train_store.expression, test_store.expression]),
        section_qc=np.vstack([train_store.section_qc, test_store.section_qc]),
        meta=combined_meta,
    )
    y = bundle.meta_train[TARGET].astype(str).to_numpy()
    classes = np.unique(y)
    train_rows = np.arange(n_train, dtype=int)
    test_rows = np.arange(
        n_train, n_train + len(bundle.counts_test), dtype=int
    )
    seed_predictions: list[dict[str, np.ndarray]] = []
    for seed in args.seeds:
        start = time.time()
        seed_predictions.append(
            five_model_probabilities(
                store, y, classes, train_rows, test_rows, seed
            )
        )
        print(
            f"[predict] seed={seed} complete in {time.time() - start:.1f}s",
            flush=True,
        )
    averaged = {
        model: np.mean([item[model] for item in seed_predictions], axis=0)
        for model in BASE_MODELS
    }
    variants = probability_variants(averaged)
    train_ei = (
        bundle.meta_train["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    test_ei = (
        bundle.meta_test["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    allowed = allowed_by_ei(y, train_ei, np.arange(n_train))
    masked_mean5 = mask_probabilities(
        variants["mean5"], classes, test_ei, allowed
    )
    predictions = classes[masked_mean5.argmax(axis=1)]
    submission = pd.DataFrame(
        {
            "Cell_ID": bundle.counts_test.index,
            PREDICTION_COLUMN: predictions,
        }
    )
    if not np.array_equal(
        submission["Cell_ID"].astype(str).to_numpy(),
        bundle.meta_test.index.astype(str).to_numpy(),
    ):
        raise RuntimeError("Submission order does not match meta_test.csv")
    if not submission[PREDICTION_COLUMN].isin(classes).all():
        raise RuntimeError("Submission contains unknown labels")
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_out, index=False)
    np.savez_compressed(
        args.out_dir / "v031_test_probabilities.npz",
        p_lgbm=averaged["lgbm"],
        p_lr=averaged["lr"],
        p_mlp=averaged["mlp"],
        p_knn=averaged["knn"],
        p_cb=averaged["cb"],
        p_blend2=variants["blend2"],
        p_mean5=variants["mean5"],
        p_mean5_masked=masked_mean5,
        classes=classes.astype(str),
        cell_ids=np.asarray(bundle.counts_test.index.astype(str).tolist(), dtype=str),
    )
    print(f"Wrote {len(submission)} predictions -> {args.submission_out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    need_test = args.mode in {"predict", "all"}
    bundle = load_data(args.data_dir, need_test=need_test)
    fingerprint = array_fingerprint(bundle.counts_train, bundle.meta_train)
    manifest = {
        "version": VERSION,
        "created_unix_time": time.time(),
        "data_fingerprint": fingerprint,
        "mode": args.mode,
        "schemes": args.schemes,
        "seeds": args.seeds,
        "base_models": list(BASE_MODELS),
        "fusion": "equal probability mean",
        "stacker": None,
        "direct_identifier_features": [],
        "identifier_uses": {
            "Mouse_ID": "Mouse Group CV only",
            "Datasets": "Dataset Group CV only",
            "Section_ID": "within-section QC percentile ranks only",
        },
        "stable_categorical_features": list(STABLE_CATEGORICAL),
        "target_encoding_columns": [list(columns) for columns in TE_COLUMNS],
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "submission_out": str(args.submission_out.resolve()),
    }
    (args.out_dir / "v031_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.mode in {"oof", "all"}:
        run_oof(args, bundle)
    if args.mode in {"predict", "all"}:
        run_prediction(args, bundle)


if __name__ == "__main__":
    main()
