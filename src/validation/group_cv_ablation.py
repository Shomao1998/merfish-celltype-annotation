#!/usr/bin/env python3
"""Grouped-CV generalization audit for the v0.25 LGBM + LR pipeline.

The public leaderboard split mixes cells from the same experiments.  This script
instead holds out complete mice or complete acquisition datasets, and evaluates
feature blocks in a fixed order:

1. v0.25 feature baseline
2. gene-only LR and LGBM
3. genes + E/I + stable anatomical metadata
4. section-normalized QC
5. experimental identifiers added back as an ablation
6. fold-safe target encoding
7. true nested grouped-CV stacking (separate --stage nested run)

Only training labels are used.  Target encodings and E/I allowed-class masks are
fit inside each fold.  The nested stacker builds its meta-training probabilities
using inner group folds that never contain the outer validation groups.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


TARGET = "MERFISH_cell_type_annotation"
SEED = 0
W = 0.5
LR_C = 0.3
STACKER_C = 0.01
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.06,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    max_bin=63,
    n_jobs=-1,
    verbose=-1,
    random_state=SEED,
)
RATIO_PAIRS = [
    ("Slc17a7", "Gad2"),
    ("Chat", "Slc17a7"),
    ("Pvalb", "Sst"),
    ("Slc18a3", "Gad2"),
    ("Trem2", "Slc17a7"),
]
TE_COLS = {
    "te_segment": ["Segment"],
    "te_region": ["Region"],
    "te_joint_sre": ["Segment", "Region", "Excitatory_vs_Inhibitory"],
}


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    lgb: np.ndarray
    lr: np.ndarray
    use_te: bool
    use_ei_decode: bool
    description: str


def onehot(meta: pd.DataFrame, col: str, fill: object = "MISSING") -> np.ndarray:
    return pd.get_dummies(meta[col].fillna(fill)).values.astype(float)


def expression_and_qc(counts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = counts.values.astype(float)
    total = raw.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    expr = np.log1p(raw / total * 100.0)
    return expr, raw, total.ravel()


def section_normalized_qc(
    raw: np.ndarray, depth: np.ndarray, meta: pd.DataFrame
) -> np.ndarray:
    volume = np.clip(meta["volume"].to_numpy(dtype=float), 1e-12, None)
    qc = pd.DataFrame(
        {
            "log_depth": np.log1p(depth),
            "log_volume": np.log1p(volume),
            "log_density": np.log1p(depth / volume),
            "detected_genes": (raw > 0).sum(axis=1).astype(float),
        },
        index=meta.index,
    )
    # Rank is computed without labels and is invariant to section-specific scale.
    ranked = qc.groupby(meta["Section_ID"].astype(str), sort=False).rank(pct=True)
    return ranked.fillna(0.5).to_numpy(dtype=float)


def build_configs(counts: pd.DataFrame, meta: pd.DataFrame) -> dict[str, FeatureConfig]:
    expr, raw, depth = expression_and_qc(counts)
    volume = np.clip(meta["volume"].to_numpy(dtype=float), 1e-12, None)
    raw_qc = np.column_stack(
        [
            np.log1p(depth),
            np.log1p(volume),
            depth / volume,
            (raw > 0).sum(axis=1).astype(float),
        ]
    )
    sec_qc = section_normalized_qc(raw, depth, meta)

    ei = onehot(meta, "Excitatory_vs_Inhibitory")
    region = onehot(meta, "Region", -1)
    segment = onehot(meta, "Segment", -1)
    ap = onehot(meta, "AP_position", -1)
    gender = onehot(meta, "Gender")
    stable_anatomy = np.hstack([ei, region, segment, ap, gender])

    mouse = onehot(meta, "Mouse_ID")
    section = onehot(meta, "Section_ID")
    dataset = onehot(meta, "Datasets")
    identifiers = np.hstack([mouse, section, dataset])

    genes = list(counts.columns)
    ratios = np.column_stack(
        [
            np.log((raw[:, genes.index(a)] + 1) / (raw[:, genes.index(b)] + 1))
            for a, b in RATIO_PAIRS
            if a in genes and b in genes
        ]
    )

    # Exact v0.25 feature blocks for the grouped-CV baseline.
    v025_lgb = np.hstack(
        [
            expr,
            raw_qc,
            meta[["center_x", "center_y", "AP_position"]].to_numpy(dtype=float),
            ei,
            region,
            segment,
            gender,
            mouse,
            section,
        ]
    )
    v025_lr = np.hstack(
        [expr, raw_qc, ei, region, segment, gender, ap, dataset, ratios]
    )

    gene_anatomy = np.hstack([expr, stable_anatomy])
    robust = np.hstack([gene_anatomy, sec_qc])

    configs = [
        FeatureConfig(
            "00_v025_baseline",
            v025_lgb,
            v025_lr,
            True,
            True,
            "Exact v0.25 features, including raw QC/coordinates and experiment IDs",
        ),
        FeatureConfig(
            "01_gene_only",
            expr,
            expr,
            False,
            False,
            "Normalized expression only; no E/I feature or E/I decoding",
        ),
        FeatureConfig(
            "02_gene_ei_anatomy",
            gene_anatomy,
            gene_anatomy,
            False,
            True,
            "Genes + E/I + Region + Segment + AP position + Gender",
        ),
        FeatureConfig(
            "03_add_section_qc",
            robust,
            robust,
            False,
            True,
            "Stable anatomy plus four within-section QC percentile ranks",
        ),
        FeatureConfig(
            "04_add_experiment_ids",
            np.hstack([robust, identifiers]),
            np.hstack([robust, identifiers]),
            False,
            True,
            "Robust features plus Mouse_ID, Section_ID, and Datasets one-hot",
        ),
        FeatureConfig(
            "05_add_fold_safe_te",
            robust,
            robust,
            True,
            True,
            "Robust no-ID features plus fold-safe Segment/Region/joint target encoding",
        ),
    ]
    return {c.name: c for c in configs}


def te_transform(
    meta: pd.DataFrame,
    columns: list[str],
    fit_pos: np.ndarray,
    y_fit: np.ndarray,
    classes: np.ndarray,
    rows: np.ndarray,
    smooth: float = 10.0,
) -> np.ndarray:
    key = meta[columns].fillna("NA").astype(str).agg("|".join, axis=1).to_numpy()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([class_to_idx[c] for c in y_fit], dtype=int)
    prior = np.bincount(yi, minlength=len(classes)).astype(float)
    prior /= prior.sum()
    counts: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(classes), dtype=float)
    )
    for k, c in zip(key[fit_pos], yi):
        counts[k][c] += 1
    encoded = {
        k: (v + smooth * prior) / (v.sum() + smooth) for k, v in counts.items()
    }
    return np.array([encoded.get(k, prior) for k in key[rows]])


def lr_rows(
    config: FeatureConfig,
    meta: pd.DataFrame,
    fit_pos: np.ndarray,
    y_fit: np.ndarray,
    classes: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    blocks = [config.lr[rows]]
    if config.use_te:
        blocks.extend(
            te_transform(meta, cols, fit_pos, y_fit, classes, rows)
            for cols in TE_COLS.values()
        )
    return np.hstack(blocks)


def place_probabilities(
    target: np.ndarray,
    rows: np.ndarray,
    model_classes: np.ndarray,
    probabilities: np.ndarray,
    class_to_idx: dict[str, int],
) -> None:
    cols = np.array([class_to_idx[c] for c in model_classes], dtype=int)
    target[np.ix_(rows, cols)] = probabilities


def fold_allowed(y: np.ndarray, ei: np.ndarray, train: np.ndarray) -> dict[str, set[str]]:
    return {
        group: set(y[train][ei[train] == group])
        for group in np.unique(ei[train])
    }


def decode(
    probabilities: np.ndarray,
    classes: np.ndarray,
    ei_rows: np.ndarray,
    allowed: dict[str, set[str]],
    use_ei_decode: bool,
) -> np.ndarray:
    if not use_ei_decode:
        return classes[probabilities.argmax(axis=1)]
    mask = np.array(
        [[c in allowed.get(group, set(classes)) for c in classes] for group in ei_rows]
    )
    return classes[np.where(mask, probabilities, -np.inf).argmax(axis=1)]


def make_splits(
    scheme: str, y: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros((len(y), 1))
    if scheme == "mouse":
        splitter = GroupKFold(n_splits=5)
    elif scheme == "dataset":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return [(tr, va) for tr, va in splitter.split(dummy, y, groups)]


def fit_base_fold(
    config: FeatureConfig,
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    class_to_idx: dict[str, int],
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lgb = LGBMClassifier(**LGBM_PARAMS).fit(config.lgb[train], y[train])
    p_lgb = np.zeros((len(valid), len(classes)), dtype=float)
    place_probabilities(
        p_lgb,
        np.arange(len(valid)),
        lgb.classes_,
        lgb.predict_proba(config.lgb[valid]),
        class_to_idx,
    )

    x_train = lr_rows(config, meta, train, y[train], classes, train)
    x_valid = lr_rows(config, meta, train, y[train], classes, valid)
    scaler = StandardScaler().fit(x_train)
    lr = LogisticRegression(max_iter=2000, C=LR_C, n_jobs=-1).fit(
        scaler.transform(x_train), y[train]
    )
    p_lr = np.zeros((len(valid), len(classes)), dtype=float)
    place_probabilities(
        p_lr,
        np.arange(len(valid)),
        lr.classes_,
        lr.predict_proba(scaler.transform(x_valid)),
        class_to_idx,
    )
    return p_lgb, p_lr


def metric_rows(
    config: FeatureConfig,
    scheme: str,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    p_lgb: np.ndarray,
    p_lr: np.ndarray,
    groups: np.ndarray,
    elapsed_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    variants = {
        "lgbm": p_lgb,
        "lr": p_lr,
        "blend_0.5": W * p_lgb + (1 - W) * p_lr,
    }
    for model_name, probabilities in variants.items():
        pooled_pred = np.empty(len(y), dtype=object)
        fold_acc: list[float] = []
        for fold_no, (tr, va) in enumerate(folds):
            allowed = fold_allowed(y, ei, tr)
            pred = decode(
                probabilities[va], classes, ei[va], allowed, config.use_ei_decode
            )
            pooled_pred[va] = pred
            acc = accuracy_score(y[va], pred)
            fold_acc.append(acc)
            detail.append(
                {
                    "config": config.name,
                    "scheme": scheme,
                    "model": model_name,
                    "fold": fold_no,
                    "held_groups": ",".join(sorted(np.unique(groups[va]).astype(str))),
                    "n_valid": len(va),
                    "accuracy": acc,
                }
            )
        summary.append(
            {
                "config": config.name,
                "description": config.description,
                "scheme": scheme,
                "model": model_name,
                "pooled_accuracy": accuracy_score(y, pooled_pred),
                "fold_mean": float(np.mean(fold_acc)),
                "fold_std": float(np.std(fold_acc)),
                "fold_min": float(np.min(fold_acc)),
                "fold_max": float(np.max(fold_acc)),
                "n_folds": len(folds),
                "elapsed_seconds_for_config": elapsed_seconds,
            }
        )
    return summary, detail


def run_ablation(
    configs: dict[str, FeatureConfig],
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    out_dir: Path,
) -> None:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_summary: list[dict[str, object]] = []
    all_detail: list[dict[str, object]] = []
    for scheme, group_col in [("mouse", "Mouse_ID"), ("dataset", "Datasets")]:
        groups = meta[group_col].astype(str).to_numpy()
        folds = make_splits(scheme, y, groups)
        print(
            f"\n=== {scheme}: {len(np.unique(groups))} groups, {len(folds)} outer folds ===",
            flush=True,
        )
        for config in configs.values():
            start = time.time()
            p_lgb = np.zeros((len(y), len(classes)), dtype=float)
            p_lr = np.zeros((len(y), len(classes)), dtype=float)
            print(f"[{scheme}] {config.name}", flush=True)
            for fold_no, (tr, va) in enumerate(folds):
                fold_start = time.time()
                fold_lgb, fold_lr = fit_base_fold(
                    config, meta, y, classes, class_to_idx, tr, va
                )
                p_lgb[va] = fold_lgb
                p_lr[va] = fold_lr
                held = ",".join(sorted(np.unique(groups[va]).astype(str)))
                print(
                    f"  fold {fold_no + 1}/{len(folds)} held={held} "
                    f"({time.time() - fold_start:.1f}s)",
                    flush=True,
                )
            elapsed = time.time() - start
            summary, detail = metric_rows(
                config,
                scheme,
                y,
                classes,
                ei,
                folds,
                p_lgb,
                p_lr,
                groups,
                elapsed,
            )
            all_summary.extend(summary)
            all_detail.extend(detail)
            cache = out_dir / f"base_probs_{scheme}_{config.name}.npz"
            np.savez_compressed(cache, p_lgb=p_lgb, p_lr=p_lr)
            blend = next(row for row in summary if row["model"] == "blend_0.5")
            print(
                f"  -> blend pooled={blend['pooled_accuracy']:.4f}, "
                f"fold mean={blend['fold_mean']:.4f}±{blend['fold_std']:.4f}",
                flush=True,
            )
            pd.DataFrame(all_summary).to_csv(out_dir / "ablation_summary.csv", index=False)
            pd.DataFrame(all_detail).to_csv(out_dir / "ablation_folds.csv", index=False)


def inner_splits(
    outer_train: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    local_groups = groups[outer_train]
    n_groups = len(np.unique(local_groups))
    n_splits = min(5, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(outer_train), 1))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_tr, inner_va in splitter.split(dummy, groups=local_groups):
        result.append((outer_train[inner_tr], outer_train[inner_va]))
    return result


def run_nested_stacking(
    config: FeatureConfig,
    meta: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    ei: np.ndarray,
    out_dir: Path,
) -> None:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for scheme, group_col in [("mouse", "Mouse_ID"), ("dataset", "Datasets")]:
        groups = meta[group_col].astype(str).to_numpy()
        outer = make_splits(scheme, y, groups)
        outer_cache_path = out_dir / f"base_probs_{scheme}_{config.name}.npz"
        if outer_cache_path.exists():
            outer_cache = np.load(outer_cache_path)
            cached_outer_lgb = outer_cache["p_lgb"]
            cached_outer_lr = outer_cache["p_lr"]
        else:
            cached_outer_lgb = cached_outer_lr = None
        p_stack = np.zeros((len(y), len(classes)), dtype=float)
        p_blend = np.zeros_like(p_stack)
        stack_pred = np.empty(len(y), dtype=object)
        blend_pred = np.empty(len(y), dtype=object)
        start = time.time()
        print(f"\n=== TRUE NESTED STACKING: {scheme} / {config.name} ===", flush=True)
        for fold_no, (outer_tr, outer_va) in enumerate(outer):
            inner = inner_splits(outer_tr, groups)
            meta_lgb = np.zeros((len(outer_tr), len(classes)), dtype=float)
            meta_lr = np.zeros_like(meta_lgb)
            outer_lookup = {global_pos: local_pos for local_pos, global_pos in enumerate(outer_tr)}

            for inner_no, (inner_tr, inner_va) in enumerate(inner):
                fold_lgb, fold_lr = fit_base_fold(
                    config, meta, y, classes, class_to_idx, inner_tr, inner_va
                )
                local_va = np.array([outer_lookup[pos] for pos in inner_va], dtype=int)
                meta_lgb[local_va] = fold_lgb
                meta_lr[local_va] = fold_lr
                print(
                    f"  outer {fold_no + 1}/{len(outer)} inner "
                    f"{inner_no + 1}/{len(inner)}",
                    flush=True,
                )

            # Meta learner sees only inner-OOF probabilities from outer training groups.
            meta_x = np.hstack([meta_lgb, meta_lr])
            meta_scaler = StandardScaler().fit(meta_x)
            stacker = LogisticRegression(
                max_iter=3000, C=STACKER_C, n_jobs=-1
            ).fit(meta_scaler.transform(meta_x), y[outer_tr])

            # Base predictions for the untouched outer validation groups.
            if cached_outer_lgb is None:
                outer_lgb, outer_lr = fit_base_fold(
                    config, meta, y, classes, class_to_idx, outer_tr, outer_va
                )
            else:
                outer_lgb = cached_outer_lgb[outer_va]
                outer_lr = cached_outer_lr[outer_va]
            outer_meta_x = np.hstack([outer_lgb, outer_lr])
            fold_stack = np.zeros((len(outer_va), len(classes)), dtype=float)
            place_probabilities(
                fold_stack,
                np.arange(len(outer_va)),
                stacker.classes_,
                stacker.predict_proba(meta_scaler.transform(outer_meta_x)),
                class_to_idx,
            )
            fold_blend = W * outer_lgb + (1 - W) * outer_lr
            p_stack[outer_va] = fold_stack
            p_blend[outer_va] = fold_blend
            allowed = fold_allowed(y, ei, outer_tr)
            pred_stack = decode(
                fold_stack, classes, ei[outer_va], allowed, config.use_ei_decode
            )
            pred_blend = decode(
                fold_blend, classes, ei[outer_va], allowed, config.use_ei_decode
            )
            stack_pred[outer_va] = pred_stack
            blend_pred[outer_va] = pred_blend
            held = ",".join(sorted(np.unique(groups[outer_va]).astype(str)))
            row = {
                "config": config.name,
                "scheme": scheme,
                "fold": fold_no,
                "held_groups": held,
                "n_valid": len(outer_va),
                "blend_accuracy": accuracy_score(y[outer_va], pred_blend),
                "stack_accuracy": accuracy_score(y[outer_va], pred_stack),
            }
            row["delta"] = row["stack_accuracy"] - row["blend_accuracy"]
            fold_rows.append(row)
            print(
                f"  outer held={held}: blend={row['blend_accuracy']:.4f}, "
                f"stack={row['stack_accuracy']:.4f}, delta={row['delta']:+.4f}",
                flush=True,
            )

        scheme_folds = [row for row in fold_rows if row["scheme"] == scheme]
        summary_rows.append(
            {
                "config": config.name,
                "scheme": scheme,
                "blend_pooled_accuracy": accuracy_score(y, blend_pred),
                "stack_pooled_accuracy": accuracy_score(y, stack_pred),
                "pooled_delta": accuracy_score(y, stack_pred)
                - accuracy_score(y, blend_pred),
                "blend_fold_mean": float(
                    np.mean([row["blend_accuracy"] for row in scheme_folds])
                ),
                "stack_fold_mean": float(
                    np.mean([row["stack_accuracy"] for row in scheme_folds])
                ),
                "stack_fold_std": float(
                    np.std([row["stack_accuracy"] for row in scheme_folds])
                ),
                "stack_fold_min": float(
                    np.min([row["stack_accuracy"] for row in scheme_folds])
                ),
                "elapsed_seconds": time.time() - start,
            }
        )
        np.savez_compressed(
            out_dir / f"nested_probs_{scheme}_{config.name}.npz",
            p_stack=p_stack,
            p_blend=p_blend,
        )
        pd.DataFrame(fold_rows).to_csv(out_dir / "nested_stacking_folds.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(
            out_dir / "nested_stacking_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", choices=["ablation", "nested", "all"], default="all")
    parser.add_argument("--stack-config", default="03_add_section_qc")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore", message="The least populated class in y has only"
    )
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(data_dir / "counts_train.csv", index_col=0)
    meta = pd.read_csv(data_dir / "meta_train.csv", index_col=0).loc[counts.index]
    y = meta[TARGET].astype(str).to_numpy()
    classes = np.unique(y)
    ei = meta["Excitatory_vs_Inhibitory"].fillna("MISSING").astype(str).to_numpy()
    configs = build_configs(counts, meta)

    manifest = {
        "data_dir": str(data_dir.resolve()),
        "n_cells": len(y),
        "n_genes": counts.shape[1],
        "n_classes": len(classes),
        "mouse_groups": sorted(meta["Mouse_ID"].astype(str).unique().tolist()),
        "dataset_groups": sorted(meta["Datasets"].astype(str).unique().tolist()),
        "configs": {name: cfg.description for name, cfg in configs.items()},
        "lgbm_params": LGBM_PARAMS,
        "lr_c": LR_C,
        "stacker_c": STACKER_C,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.stage in {"ablation", "all"}:
        run_ablation(configs, meta, y, classes, ei, out_dir)
    if args.stage in {"nested", "all"}:
        if args.stack_config not in configs:
            raise ValueError(
                f"Unknown stack config {args.stack_config}; choose from {list(configs)}"
            )
        run_nested_stacking(
            configs[args.stack_config], meta, y, classes, ei, out_dir
        )


if __name__ == "__main__":
    main()
