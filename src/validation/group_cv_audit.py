#!/usr/bin/env python3
"""Grouped-CV generalization audit for the v0.3 five-model stacker.

The LGBM and LR implementations in v0.3 are identical to v0.25, so this
script reuses their saved outer-fold probabilities.  It trains the three new
v0.3 base learners (MLP, KNN and CatBoost) on the same held-out-mouse and
held-out-dataset folds, then evaluates:

* each of the five base models;
* the original two-model LGBM/LR mean;
* an equal five-model mean; and
* a true nested grouped-CV logistic-regression stacker.

The nested stacker is genuinely nested: for every outer validation group, its
meta-training matrix is created only from inner group-held-out predictions on
the outer-training rows.  It never trains the meta learner on predictions from
models that saw those same rows.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from v025_group_cv_ablation import (
    STACKER_C,
    TARGET,
    FeatureConfig,
    build_configs,
    decode,
    fit_base_fold,
    fold_allowed,
    inner_splits,
    lr_rows,
    make_splits,
    place_probabilities,
)


SEED = 0
BASE_ORDER = ["lgbm", "lr", "mlp", "knn", "cb"]
AUDIT_CONFIGS = ["00_v025_baseline", "05_add_fold_safe_te"]
ROBUST_CONFIG = "05_add_fold_safe_te"
CB_PARAMS = dict(
    iterations=400,
    learning_rate=0.06,
    depth=6,
    subsample=0.8,
    rsm=0.6,
    bootstrap_type="Bernoulli",
    border_count=63,
    loss_function="MultiClass",
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)


def fit_extra_fold(
    config: FeatureConfig,
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    class_to_idx: dict[str, int],
    train: np.ndarray,
    valid: np.ndarray,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Fit the MLP, KNN and CatBoost exactly as in v0.3 for one fold."""
    n_valid = len(valid)
    n_classes = len(classes)
    yi = np.array([class_to_idx[c] for c in y[train]], dtype=int)

    # v0.3 MLP uses the same complete (including fold-safe TE) feature matrix
    # and scaler as LR.  For no-TE ablations, lr_rows returns static features.
    x_train = lr_rows(config, meta, train, y[train], classes, train)
    x_valid = lr_rows(config, meta, train, y[train], classes, valid)
    mlp_scaler = StandardScaler().fit(x_train)
    z_train = mlp_scaler.transform(x_train)
    z_valid = mlp_scaler.transform(x_valid)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        alpha=1e-3,
        max_iter=300,
        early_stopping=True,
        random_state=seed,
    ).fit(z_train, yi)
    p_mlp = np.zeros((n_valid, n_classes), dtype=float)
    p_mlp[:, mlp.classes_.astype(int)] = mlp.predict_proba(z_valid)

    # v0.3 KNN intentionally uses only static LR features, then scaled PCA-50.
    knn_scaler = StandardScaler().fit(config.lr[train])
    knn_train_scaled = knn_scaler.transform(config.lr[train])
    pca = PCA(50, random_state=seed).fit(knn_train_scaled)
    knn = KNeighborsClassifier(
        30, weights="distance", n_jobs=-1
    ).fit(pca.transform(knn_train_scaled), y[train])
    p_knn = np.zeros((n_valid, n_classes), dtype=float)
    place_probabilities(
        p_knn,
        np.arange(n_valid),
        knn.classes_,
        knn.predict_proba(
            pca.transform(knn_scaler.transform(config.lr[valid]))
        ),
        class_to_idx,
    )

    cb = CatBoostClassifier(**CB_PARAMS, random_seed=seed).fit(
        config.lgb[train], yi
    )
    p_cb = np.zeros((n_valid, n_classes), dtype=float)
    p_cb[:, cb.classes_.astype(int)] = cb.predict_proba(config.lgb[valid])
    return {"mlp": p_mlp, "knn": p_knn, "cb": p_cb}


def decode_foldwise(
    probabilities: np.ndarray,
    config: FeatureConfig,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    pred = np.empty(len(y), dtype=object)
    fold_accuracy: list[float] = []
    for train, valid in folds:
        allowed = fold_allowed(y, ei, train)
        fold_pred = decode(
            probabilities[valid],
            classes,
            ei[valid],
            allowed,
            config.use_ei_decode,
        )
        pred[valid] = fold_pred
        fold_accuracy.append(accuracy_score(y[valid], fold_pred))
    return pred, fold_accuracy


def metric_rows(
    config: FeatureConfig,
    scheme: str,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    variants = dict(probabilities)
    variants["blend2_lgbm_lr"] = 0.5 * (
        probabilities["lgbm"] + probabilities["lr"]
    )
    variants["mean5"] = np.mean(
        [probabilities[name] for name in BASE_ORDER], axis=0
    )

    for model, proba in variants.items():
        pred, fold_acc = decode_foldwise(
            proba, config, y, classes, ei, folds
        )
        summary.append(
            {
                "config": config.name,
                "description": config.description,
                "scheme": scheme,
                "model": model,
                "pooled_accuracy": accuracy_score(y, pred),
                "fold_mean": float(np.mean(fold_acc)),
                "fold_std": float(np.std(fold_acc)),
                "fold_min": float(np.min(fold_acc)),
                "fold_max": float(np.max(fold_acc)),
                "n_folds": len(folds),
                "elapsed_seconds_for_extra_models": elapsed_seconds,
            }
        )
        for fold_no, ((_, valid), acc) in enumerate(zip(folds, fold_acc)):
            detail.append(
                {
                    "config": config.name,
                    "scheme": scheme,
                    "model": model,
                    "fold": fold_no,
                    "held_groups": ",".join(
                        sorted(np.unique(groups[valid]).astype(str))
                    ),
                    "n_valid": len(valid),
                    "accuracy": acc,
                }
            )
    return summary, detail


def load_v025_base(
    base_dir: Path, scheme: str, config_name: str
) -> dict[str, np.ndarray]:
    path = base_dir / f"base_probs_{scheme}_{config_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run v025_group_cv_ablation.py --stage ablation first."
        )
    cached = np.load(path)
    return {"lgbm": cached["p_lgb"], "lr": cached["p_lr"]}


def run_outer_audit(
    configs: dict[str, FeatureConfig],
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    base_dir: Path,
    out_dir: Path,
) -> None:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_summary: list[dict[str, object]] = []
    all_detail: list[dict[str, object]] = []
    for scheme, group_col in [("mouse", "Mouse_ID"), ("dataset", "Datasets")]:
        groups = meta[group_col].astype(str).to_numpy()
        folds = make_splits(scheme, y, groups)
        print(
            f"\n=== v0.3 OUTER AUDIT: {scheme}, {len(folds)} folds ===",
            flush=True,
        )
        for config_name in AUDIT_CONFIGS:
            config = configs[config_name]
            cache_path = out_dir / f"v03_extra_probs_{scheme}_{config_name}.npz"
            start = time.time()
            if cache_path.exists():
                cache = np.load(cache_path)
                extra = {name: cache[f"p_{name}"] for name in ["mlp", "knn", "cb"]}
                print(f"[{scheme}] {config_name}: loaded extra-model cache", flush=True)
            else:
                extra = {
                    name: np.zeros((len(y), len(classes)), dtype=float)
                    for name in ["mlp", "knn", "cb"]
                }
                print(f"[{scheme}] {config_name}: training new v0.3 models", flush=True)
                for fold_no, (train, valid) in enumerate(folds):
                    fold_start = time.time()
                    fold_probs = fit_extra_fold(
                        config,
                        meta,
                        y,
                        classes,
                        class_to_idx,
                        train,
                        valid,
                    )
                    for model, proba in fold_probs.items():
                        extra[model][valid] = proba
                    held = ",".join(sorted(np.unique(groups[valid]).astype(str)))
                    print(
                        f"  fold {fold_no + 1}/{len(folds)} held={held} "
                        f"({time.time() - fold_start:.1f}s)",
                        flush=True,
                    )
                np.savez_compressed(
                    cache_path,
                    p_mlp=extra["mlp"],
                    p_knn=extra["knn"],
                    p_cb=extra["cb"],
                )

            probabilities = load_v025_base(base_dir, scheme, config_name)
            probabilities.update(extra)
            elapsed = time.time() - start
            summary, detail = metric_rows(
                config,
                scheme,
                y,
                classes,
                ei,
                folds,
                groups,
                probabilities,
                elapsed,
            )
            all_summary.extend(summary)
            all_detail.extend(detail)
            mean5 = next(row for row in summary if row["model"] == "mean5")
            blend2 = next(
                row for row in summary if row["model"] == "blend2_lgbm_lr"
            )
            print(
                f"  -> blend2={blend2['pooled_accuracy']:.4f}, "
                f"mean5={mean5['pooled_accuracy']:.4f}",
                flush=True,
            )
            pd.DataFrame(all_summary).to_csv(
                out_dir / "v03_group_summary.csv", index=False
            )
            pd.DataFrame(all_detail).to_csv(
                out_dir / "v03_group_folds.csv", index=False
            )


def load_outer_probabilities(
    base_dir: Path,
    out_dir: Path,
    scheme: str,
    config_name: str,
) -> dict[str, np.ndarray]:
    probabilities = load_v025_base(base_dir, scheme, config_name)
    extra_path = out_dir / f"v03_extra_probs_{scheme}_{config_name}.npz"
    if not extra_path.exists():
        raise FileNotFoundError(
            f"Missing {extra_path}. Run this script with --stage outer first."
        )
    extra = np.load(extra_path)
    probabilities.update(
        {name: extra[f"p_{name}"] for name in ["mlp", "knn", "cb"]}
    )
    return probabilities


def run_true_nested_stacking(
    config: FeatureConfig,
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    base_dir: Path,
    outer_prob_dir: Path,
    out_dir: Path,
    schemes: list[str],
) -> None:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    group_columns = {"mouse": "Mouse_ID", "dataset": "Datasets"}
    for scheme in schemes:
        group_col = group_columns[scheme]
        groups = meta[group_col].astype(str).to_numpy()
        outer_folds = make_splits(scheme, y, groups)
        outer_probs = load_outer_probabilities(
            base_dir, outer_prob_dir, scheme, config.name
        )
        p_stack = np.zeros((len(y), len(classes)), dtype=float)
        p_mean5 = np.mean([outer_probs[name] for name in BASE_ORDER], axis=0)
        p_blend2 = 0.5 * (outer_probs["lgbm"] + outer_probs["lr"])
        start = time.time()
        print(
            f"\n=== TRUE NESTED v0.3 STACKER: {scheme} / {config.name} ===",
            flush=True,
        )

        for outer_no, (outer_train, outer_valid) in enumerate(outer_folds):
            checkpoint = out_dir / (
                f"v03_nested_meta_{scheme}_{config.name}_outer{outer_no}.npz"
            )
            if checkpoint.exists():
                saved = np.load(checkpoint)
                meta_probs = {
                    name: saved[f"p_{name}"] for name in BASE_ORDER
                }
                print(
                    f"  outer {outer_no + 1}/{len(outer_folds)}: "
                    "loaded inner-OOF checkpoint",
                    flush=True,
                )
            else:
                meta_probs = {
                    name: np.zeros(
                        (len(outer_train), len(classes)), dtype=float
                    )
                    for name in BASE_ORDER
                }
                lookup = {
                    global_pos: local_pos
                    for local_pos, global_pos in enumerate(outer_train)
                }
                inner_folds = inner_splits(outer_train, groups)
                for inner_no, (inner_train, inner_valid) in enumerate(inner_folds):
                    inner_start = time.time()
                    p_lgb, p_lr = fit_base_fold(
                        config,
                        meta,
                        y,
                        classes,
                        class_to_idx,
                        inner_train,
                        inner_valid,
                    )
                    extra = fit_extra_fold(
                        config,
                        meta,
                        y,
                        classes,
                        class_to_idx,
                        inner_train,
                        inner_valid,
                    )
                    local_valid = np.array(
                        [lookup[pos] for pos in inner_valid], dtype=int
                    )
                    meta_probs["lgbm"][local_valid] = p_lgb
                    meta_probs["lr"][local_valid] = p_lr
                    for model, proba in extra.items():
                        meta_probs[model][local_valid] = proba
                    print(
                        f"  outer {outer_no + 1}/{len(outer_folds)} inner "
                        f"{inner_no + 1}/{len(inner_folds)} "
                        f"({time.time() - inner_start:.1f}s)",
                        flush=True,
                    )
                np.savez_compressed(
                    checkpoint,
                    **{f"p_{name}": meta_probs[name] for name in BASE_ORDER},
                )

            meta_x = np.hstack([meta_probs[name] for name in BASE_ORDER])
            meta_scaler = StandardScaler().fit(meta_x)
            stacker = LogisticRegression(
                max_iter=3000, C=STACKER_C, n_jobs=-1
            ).fit(meta_scaler.transform(meta_x), y[outer_train])
            outer_x = np.hstack(
                [outer_probs[name][outer_valid] for name in BASE_ORDER]
            )
            fold_stack = np.zeros(
                (len(outer_valid), len(classes)), dtype=float
            )
            place_probabilities(
                fold_stack,
                np.arange(len(outer_valid)),
                stacker.classes_,
                stacker.predict_proba(meta_scaler.transform(outer_x)),
                class_to_idx,
            )
            p_stack[outer_valid] = fold_stack

            allowed = fold_allowed(y, ei, outer_train)
            predictions: dict[str, np.ndarray] = {}
            for name, proba in {
                "blend2": p_blend2[outer_valid],
                "mean5": p_mean5[outer_valid],
                "stack": fold_stack,
            }.items():
                predictions[name] = decode(
                    proba,
                    classes,
                    ei[outer_valid],
                    allowed,
                    config.use_ei_decode,
                )
            row = {
                "config": config.name,
                "scheme": scheme,
                "fold": outer_no,
                "held_groups": ",".join(
                    sorted(np.unique(groups[outer_valid]).astype(str))
                ),
                "n_valid": len(outer_valid),
                "blend2_accuracy": accuracy_score(
                    y[outer_valid], predictions["blend2"]
                ),
                "mean5_accuracy": accuracy_score(
                    y[outer_valid], predictions["mean5"]
                ),
                "stack_accuracy": accuracy_score(
                    y[outer_valid], predictions["stack"]
                ),
            }
            row["stack_minus_blend2"] = (
                row["stack_accuracy"] - row["blend2_accuracy"]
            )
            row["stack_minus_mean5"] = (
                row["stack_accuracy"] - row["mean5_accuracy"]
            )
            fold_rows.append(row)
            print(
                f"  outer held={row['held_groups']}: "
                f"blend2={row['blend2_accuracy']:.4f}, "
                f"mean5={row['mean5_accuracy']:.4f}, "
                f"stack={row['stack_accuracy']:.4f}",
                flush=True,
            )
            pd.DataFrame(fold_rows).to_csv(
                out_dir / "v03_nested_folds.csv", index=False
            )

        decoded: dict[str, np.ndarray] = {}
        for name, proba in {
            "blend2": p_blend2,
            "mean5": p_mean5,
            "stack": p_stack,
        }.items():
            decoded[name], _ = decode_foldwise(
                proba, config, y, classes, ei, outer_folds
            )
        scheme_rows = [row for row in fold_rows if row["scheme"] == scheme]
        summary_rows.append(
            {
                "config": config.name,
                "scheme": scheme,
                "blend2_pooled_accuracy": accuracy_score(y, decoded["blend2"]),
                "mean5_pooled_accuracy": accuracy_score(y, decoded["mean5"]),
                "stack_pooled_accuracy": accuracy_score(y, decoded["stack"]),
                "stack_minus_blend2": accuracy_score(y, decoded["stack"])
                - accuracy_score(y, decoded["blend2"]),
                "stack_minus_mean5": accuracy_score(y, decoded["stack"])
                - accuracy_score(y, decoded["mean5"]),
                "blend2_fold_mean": float(
                    np.mean([row["blend2_accuracy"] for row in scheme_rows])
                ),
                "mean5_fold_mean": float(
                    np.mean([row["mean5_accuracy"] for row in scheme_rows])
                ),
                "stack_fold_mean": float(
                    np.mean([row["stack_accuracy"] for row in scheme_rows])
                ),
                "stack_fold_std": float(
                    np.std([row["stack_accuracy"] for row in scheme_rows])
                ),
                "stack_fold_min": float(
                    np.min([row["stack_accuracy"] for row in scheme_rows])
                ),
                "elapsed_seconds": time.time() - start,
            }
        )
        np.savez_compressed(
            out_dir / f"v03_nested_probs_{scheme}_{config.name}.npz",
            p_blend2=p_blend2,
            p_mean5=p_mean5,
            p_stack=p_stack,
        )
        pd.DataFrame(summary_rows).to_csv(
            out_dir / "v03_nested_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--v025-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--outer-prob-dir",
        help="Directory containing v03_extra_probs caches; defaults to --out-dir",
    )
    parser.add_argument(
        "--stage", choices=["outer", "nested", "all"], default="all"
    )
    parser.add_argument("--stack-config", default=ROBUST_CONFIG)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["mouse", "dataset"],
        default=["mouse", "dataset"],
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore", message="The least populated class in y has only"
    )
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names"
    )
    data_dir = Path(args.data_dir)
    base_dir = Path(args.v025_out_dir)
    out_dir = Path(args.out_dir)
    outer_prob_dir = (
        Path(args.outer_prob_dir) if args.outer_prob_dir else out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(data_dir / "counts_train.csv", index_col=0)
    meta = pd.read_csv(data_dir / "meta_train.csv", index_col=0).loc[counts.index]
    y = meta[TARGET].astype(str).to_numpy()
    classes = np.unique(y)
    ei = (
        meta["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    configs = build_configs(counts, meta)
    if args.stack_config not in configs:
        raise ValueError(
            f"Unknown stack config {args.stack_config}; choose from {list(configs)}"
        )

    manifest = {
        "source_model": "v0.3 five-model stacker",
        "data_dir": str(data_dir.resolve()),
        "v025_probability_cache": str(base_dir.resolve()),
        "n_cells": len(y),
        "n_genes": counts.shape[1],
        "n_classes": len(classes),
        "base_order": BASE_ORDER,
        "seed": SEED,
        "stacker_c": STACKER_C,
        "outer_audit_configs": AUDIT_CONFIGS,
        "nested_stack_config": args.stack_config,
        "catboost_params": CB_PARAMS,
    }
    (out_dir / "v03_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.stage in {"outer", "all"}:
        run_outer_audit(
            configs, meta, y, classes, ei, base_dir, out_dir
        )
    if args.stage in {"nested", "all"}:
        run_true_nested_stacking(
            configs[args.stack_config],
            meta,
            y,
            classes,
            ei,
            base_dir,
            outer_prob_dir,
            out_dir,
            args.schemes,
        )


if __name__ == "__main__":
    main()
