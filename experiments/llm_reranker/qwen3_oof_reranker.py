#!/usr/bin/env python3
"""Fold-safe local Qwen3 reranker for v0.31 grouped-OOF probabilities.

This script is an evaluation-only companion to v0.31.  It consumes one or
more explicit v0.31 OOF ``.npz`` artifacts, selects fresh low-margin cells,
builds evidence using only each cell's outer-training fold, and asks a local
Ollama model to rerank the existing top-k candidates twice with different
candidate orders.  A label is changed only when the two calls agree and every
conservative gate passes.

Required v0.31 NPZ contract
---------------------------
Each input NPZ must contain:

``p_mean5_masked``
    Float array shaped ``(n_cells, n_classes)``.  Rows must be finite,
    non-negative, sum to one, and already include fold-safe E/I decoding masks.
``classes``
    String array in exactly the probability-column order.
``cell_ids``
    String array in exactly the probability-row order.
``fold_id``
    Integer array assigning every row to one outer validation fold.
``held_group``
    String array containing the comma-joined held-out group name(s) for each
    row's validation fold; values must be nonempty and constant within a fold.
``group_column``
    Scalar string naming the metadata column used for grouped CV.
``scheme``
    Scalar string, normally ``mouse`` or ``dataset``.

No test data are read, no submission file is written, and no data are sent to
a remote service.  Ollama must be listening on localhost.  The supplied model
is pretrained, so this experiment must remain outside the final competition
pipeline unless the organizers explicitly approve it in writing.

Example
-------
python src/v0.32-Qwen3-v0.31-OOF-Reranker.py \
  --oof-dir outputs/v031_robust_noid_5model \
  --max-cells-per-scheme 100

Inspect fresh selections and prompts without calling Ollama:

python src/v0.32-Qwen3-v0.31-OOF-Reranker.py \
  --schemes dataset --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "v0.32"
TARGET = "MERFISH_cell_type_annotation"
DEFAULT_MODEL = "qwen3:8b"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OOF_DIR = PROJECT_DIR / "outputs" / "v031_robust_noid_5model"
DEFAULT_PILOT_DIR = (
    PROJECT_DIR
    / "generalization"
    / "outputs"
    / "v04_local_llm_oof_reranker_qwen3_8b"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "generalization"
    / "outputs"
    / "v032_qwen3_v031_oof_reranker"
)
SAFE_META_COLUMNS = [
    "Excitatory_vs_Inhibitory",
    "Region",
    "Segment",
    "AP_position",
]
REQUIRED_OOF_KEYS = {
    "p_mean5_masked",
    "classes",
    "cell_ids",
    "fold_id",
    "held_group",
    "group_column",
    "scheme",
}


@dataclass(frozen=True)
class OOFArtifact:
    path: Path
    probabilities: np.ndarray
    classes: np.ndarray
    cell_ids: np.ndarray
    fold_ids: np.ndarray
    held_groups: np.ndarray
    group_column: str
    scheme: str


@dataclass(frozen=True)
class SelectedCell:
    scheme: str
    row_position: int
    fold: int
    held_groups: str
    cell_id: str
    baseline_label: str
    baseline_probability: float
    second_probability: float
    margin: float
    entropy: float
    candidates: tuple[str, ...]
    candidate_probabilities: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fold-safe local Qwen3 reranker using explicit v0.31 "
            "grouped-OOF artifacts."
        )
    )
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=DEFAULT_OOF_DIR,
        help=(
            "Directory containing v031_oof_mouse.npz and/or "
            "v031_oof_dataset.npz."
        ),
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["mouse", "dataset"],
        default=["mouse", "dataset"],
        help="Conventional OOF artifacts to load from --oof-dir.",
    )
    parser.add_argument(
        "--oof-npz",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional direct NPZ paths. When provided, these override the "
            "conventional paths selected through --oof-dir/--schemes."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument(
        "--include-pilot",
        action="store_true",
        help="Permit cells evaluated in v0.4. Off by default.",
    )
    parser.add_argument(
        "--pilot-fallback-skip",
        type=int,
        default=40,
        help=(
            "If no v0.4 pilot CSV is available, reserve this many lowest-margin "
            "cells per scheme before selecting the fresh evaluation sample."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-k", type=int, default=3)
    parser.add_argument("--max-cells-per-scheme", type=int, default=100)
    parser.add_argument("--max-base-confidence", type=float, default=0.80)
    parser.add_argument("--cell-genes", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--class-neighbors", type=int, default=5)
    parser.add_argument("--pairwise-genes-per-direction", type=int, default=8)
    parser.add_argument(
        "--min-pairwise-support-score",
        type=float,
        default=0.25,
        help="Minimum fold-training pairwise score for a gene to support a switch.",
    )
    parser.add_argument("--min-llm-confidence", type=float, default=0.80)
    parser.add_argument(
        "--min-supported-evidence",
        type=int,
        default=2,
        help=(
            "Minimum genes that both candidate-order calls cite and that are "
            "detected fold-safe pairwise supporters of the agreed label."
        ),
    )
    parser.add_argument(
        "--switch-policy",
        choices=["cross_family", "all"],
        default="cross_family",
        help=(
            "cross_family blocks within-family/fine-subtype changes; all is an "
            "explicit research-only override."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.candidate_k < 2:
        parser.error("--candidate-k must be at least 2")
    if args.max_cells_per_scheme < 1:
        parser.error("--max-cells-per-scheme must be positive")
    if args.pilot_fallback_skip < 0:
        parser.error("--pilot-fallback-skip cannot be negative")
    if args.cell_genes < 1 or args.neighbors < 1 or args.class_neighbors < 1:
        parser.error("gene and neighbor counts must be positive")
    if args.pairwise_genes_per_direction < 1:
        parser.error("--pairwise-genes-per-direction must be positive")
    if args.min_supported_evidence < 2:
        parser.error("--min-supported-evidence must be at least 2")
    if not 0.0 <= args.max_base_confidence <= 1.0:
        parser.error("--max-base-confidence must be between 0 and 1")
    if not 0.0 <= args.min_llm_confidence <= 1.0:
        parser.error("--min-llm-confidence must be between 0 and 1")
    if args.min_pairwise_support_score < 0:
        parser.error("--min-pairwise-support-score cannot be negative")
    return args


def scalar_string(value: np.ndarray, key: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {key!r} must be a scalar string")
    return str(array.reshape(-1)[0])


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_path = data_dir / "counts_train.csv"
    meta_path = data_dir / "meta_train.csv"
    if not counts_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Expected {counts_path} and {meta_path}; pass --data-dir explicitly."
        )
    counts_index_name = pd.read_csv(counts_path, nrows=0).columns[0]
    meta_index_name = pd.read_csv(meta_path, nrows=0).columns[0]
    counts = pd.read_csv(
        counts_path, index_col=0, dtype={counts_index_name: str}
    )
    meta = pd.read_csv(meta_path, index_col=0, dtype={meta_index_name: str})
    counts.index = counts.index.astype(str)
    meta.index = meta.index.astype(str)
    if not counts.index.is_unique or not meta.index.is_unique:
        raise ValueError("Training cell IDs must be unique")
    missing = counts.index.difference(meta.index)
    extra = meta.index.difference(counts.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"Counts/meta ID mismatch: {len(missing)} missing and {len(extra)} extra"
        )
    meta = meta.loc[counts.index]
    if TARGET not in meta.columns or meta[TARGET].isna().any():
        raise ValueError(f"Training target {TARGET!r} is absent or contains NA")
    if counts.columns.duplicated().any():
        raise ValueError("counts_train.csv contains duplicate gene columns")
    return counts, meta


def load_oof_artifact(
    path: Path,
    counts: pd.DataFrame,
    meta: pd.DataFrame,
) -> OOFArtifact:
    if not path.exists():
        raise FileNotFoundError(f"OOF artifact does not exist: {path}")

    def read_contract(allow_pickle: bool) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=allow_pickle) as saved:
            missing = REQUIRED_OOF_KEYS.difference(saved.files)
            if missing:
                raise KeyError(
                    f"{path}: missing v0.31 OOF keys {sorted(missing)}; "
                    f"available keys are {sorted(saved.files)}"
                )
            return {key: np.asarray(saved[key]) for key in REQUIRED_OOF_KEYS}

    try:
        contract = read_contract(allow_pickle=False)
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
        # Compatibility for the first locally generated v0.31 artifacts, in
        # which only cell_ids was accidentally stored as object dtype. Never
        # use this fallback with an NPZ obtained from another person or source.
        contract = read_contract(allow_pickle=True)
        object_keys = {
            key for key, value in contract.items() if value.dtype == object
        }
        if object_keys != {"cell_ids"}:
            raise ValueError(
                f"{path}: unsafe legacy object arrays {sorted(object_keys)}; "
                "only the known local cell_ids compatibility case is allowed"
            ) from error
        contract["cell_ids"] = np.asarray(
            [str(value) for value in contract["cell_ids"]], dtype=str
        )

    probabilities = np.asarray(contract["p_mean5_masked"], dtype=float)
    classes = np.asarray(contract["classes"]).astype(str)
    cell_ids = np.asarray(contract["cell_ids"]).astype(str)
    fold_ids_raw = np.asarray(contract["fold_id"])
    held_groups = np.asarray(contract["held_group"]).astype(str)
    group_column = scalar_string(contract["group_column"], "group_column", path)
    scheme = scalar_string(contract["scheme"], "scheme", path).lower()

    expected_group_columns = {"mouse": "Mouse_ID", "dataset": "Datasets"}
    if scheme not in expected_group_columns:
        raise ValueError(f"{path}: unsupported scheme {scheme!r}")
    if group_column != expected_group_columns[scheme]:
        raise ValueError(
            f"{path}: scheme {scheme!r} requires group_column "
            f"{expected_group_columns[scheme]!r}, found {group_column!r}"
        )

    n_rows = len(counts)
    if probabilities.ndim != 2:
        raise ValueError(f"{path}: p_mean5_masked must be two-dimensional")
    if probabilities.shape != (n_rows, len(classes)):
        raise ValueError(
            f"{path}: probability shape {probabilities.shape} does not match "
            f"({n_rows}, {len(classes)})"
        )
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes):
        raise ValueError(f"{path}: classes must be a unique one-dimensional array")
    if cell_ids.ndim != 1 or len(cell_ids) != n_rows:
        raise ValueError(f"{path}: cell_ids must have one entry per probability row")
    expected_ids = counts.index.astype(str).to_numpy()
    if not np.array_equal(cell_ids, expected_ids):
        mismatch = np.flatnonzero(cell_ids != expected_ids)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            f"{path}: cell_ids are not in counts_train row order; first mismatch "
            f"at row {first}"
        )
    y_classes = np.unique(meta[TARGET].astype(str).to_numpy())
    if set(classes) != set(y_classes):
        missing_labels = sorted(set(y_classes).difference(classes))
        extra_labels = sorted(set(classes).difference(y_classes))
        raise ValueError(
            f"{path}: class-set mismatch; missing={missing_labels}, extra={extra_labels}"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-12):
        raise ValueError(f"{path}: probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{path}: p_mean5_masked rows do not sum to one")

    if fold_ids_raw.ndim != 1 or len(fold_ids_raw) != n_rows:
        raise ValueError(f"{path}: fold_id must have one entry per row")
    if not np.all(np.isfinite(fold_ids_raw.astype(float))):
        raise ValueError(f"{path}: fold_id contains non-finite values")
    fold_ids = fold_ids_raw.astype(int)
    if not np.array_equal(fold_ids_raw.astype(float), fold_ids.astype(float)):
        raise ValueError(f"{path}: fold_id must contain integers")
    unique_folds = np.unique(fold_ids)
    if unique_folds[0] != 0 or not np.array_equal(
        unique_folds, np.arange(unique_folds[-1] + 1)
    ):
        raise ValueError(f"{path}: fold_id values must be contiguous from zero")
    if group_column not in meta.columns:
        raise ValueError(f"{path}: group column {group_column!r} is absent from metadata")
    groups = meta[group_column].fillna("MISSING").astype(str).to_numpy()
    if held_groups.ndim != 1 or len(held_groups) != n_rows:
        raise ValueError(f"{path}: held_group must have one entry per row")
    for fold in unique_folds:
        valid_mask = fold_ids == fold
        valid_groups = set(groups[valid_mask])
        train_groups = set(groups[fold_ids != fold])
        overlap = valid_groups.intersection(train_groups)
        if overlap:
            raise ValueError(
                f"{path}: fold {fold} is not group-disjoint for {group_column}; "
                f"overlap={sorted(overlap)[:5]}"
            )
        fold_held_values = set(held_groups[valid_mask])
        if len(fold_held_values) != 1 or not next(iter(fold_held_values), "").strip():
            raise ValueError(
                f"{path}: held_group must be one nonempty value within fold {fold}"
            )
        expected_held = ",".join(sorted(valid_groups))
        actual_held = next(iter(fold_held_values))
        if actual_held != expected_held:
            raise ValueError(
                f"{path}: held_group mismatch in fold {fold}; "
                f"expected {expected_held!r}, found {actual_held!r}"
            )

    return OOFArtifact(
        path=path,
        probabilities=np.clip(probabilities, 0.0, 1.0),
        classes=classes,
        cell_ids=cell_ids,
        fold_ids=fold_ids,
        held_groups=held_groups,
        group_column=group_column,
        scheme=scheme,
    )


def load_artifacts(
    paths: Iterable[Path], counts: pd.DataFrame, meta: pd.DataFrame
) -> list[OOFArtifact]:
    artifacts = [load_oof_artifact(path, counts, meta) for path in paths]
    schemes = [item.scheme for item in artifacts]
    duplicates = [name for name, count in Counter(schemes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Only one OOF artifact per scheme is allowed: {duplicates}")
    return artifacts


def normalized_expression(counts: pd.DataFrame) -> np.ndarray:
    values = counts.to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(values / totals * 100.0)


def row_entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def candidate_arrays(
    probabilities: np.ndarray, candidate_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :candidate_k]
    top_probabilities = np.take_along_axis(probabilities, order, axis=1)
    margins = top_probabilities[:, 0] - top_probabilities[:, 1]
    entropies = row_entropy(probabilities)
    return order, top_probabilities, margins, entropies


def balanced_pick(
    eligible: np.ndarray,
    fold_ids: np.ndarray,
    margins: np.ndarray,
    n_pick: int,
) -> list[int]:
    if n_pick <= 0:
        return []
    folds = np.unique(fold_ids)
    base_quota, extra = divmod(n_pick, len(folds))
    picked: list[int] = []
    for ordinal, fold in enumerate(folds):
        quota = base_quota + int(ordinal < extra)
        fold_rows = np.flatnonzero(eligible & (fold_ids == fold))
        ranked = fold_rows[
            np.lexsort((fold_rows, margins[fold_rows]))
        ]
        picked.extend(ranked[:quota].tolist())
    if len(picked) < n_pick:
        already = np.zeros(len(eligible), dtype=bool)
        already[picked] = True
        remaining = np.flatnonzero(eligible & ~already)
        ranked = remaining[np.lexsort((remaining, margins[remaining]))]
        picked.extend(ranked[: n_pick - len(picked)].tolist())
    return picked


def load_pilot_positions(
    pilot_dir: Path,
    expected_cell_ids: np.ndarray,
) -> tuple[set[int], list[str]]:
    positions: set[int] = set()
    sources: list[str] = []
    for scheme in ("mouse", "dataset"):
        path = pilot_dir / f"v04_llm_reranker_cells_{scheme}.csv"
        if not path.exists():
            continue
        sources.append(str(path.resolve()))
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"row_position", "cell_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Pilot file lacks {sorted(required)}: {path}")
            for row in reader:
                position = int(row["row_position"])
                if position < 0 or position >= len(expected_cell_ids):
                    raise ValueError(f"Pilot row {position} is outside training data: {path}")
                if str(expected_cell_ids[position]) != str(row["cell_id"]):
                    raise ValueError(
                        f"Pilot cell ID mismatch at row {position}: {path}"
                    )
                positions.add(position)
    return positions, sources


def select_fresh_cells(
    artifact: OOFArtifact,
    candidate_k: int,
    max_cells: int,
    max_base_confidence: float,
    pilot_positions: set[int],
    include_pilot: bool,
    fallback_skip: int,
) -> tuple[list[SelectedCell], dict[str, Any]]:
    probabilities = artifact.probabilities
    order, top_probs, margins, entropies = candidate_arrays(
        probabilities, min(candidate_k, probabilities.shape[1])
    )
    eligible = top_probs[:, 0] <= max_base_confidence
    exclusion_method = "none"
    excluded: set[int] = set()
    if not include_pilot and pilot_positions:
        excluded = set(pilot_positions)
        eligible[np.fromiter(sorted(excluded), dtype=int)] = False
        exclusion_method = "v0.4_union_row_positions"
    elif not include_pilot and fallback_skip:
        fallback = balanced_pick(
            eligible.copy(), artifact.fold_ids, margins, fallback_skip
        )
        excluded = set(fallback)
        eligible[np.asarray(fallback, dtype=int)] = False
        exclusion_method = "label_free_low_margin_fallback"

    picked = balanced_pick(eligible, artifact.fold_ids, margins, max_cells)
    if len(picked) < max_cells:
        raise RuntimeError(
            f"{artifact.scheme}: only {len(picked)} fresh eligible cells are "
            f"available, fewer than requested {max_cells}"
        )
    result: list[SelectedCell] = []
    for row in sorted(picked, key=lambda pos: (artifact.fold_ids[pos], margins[pos], pos)):
        columns = order[row]
        fold = int(artifact.fold_ids[row])
        held = str(artifact.held_groups[row])
        result.append(
            SelectedCell(
                scheme=artifact.scheme,
                row_position=int(row),
                fold=fold,
                held_groups=held,
                cell_id=str(artifact.cell_ids[row]),
                baseline_label=str(artifact.classes[columns[0]]),
                baseline_probability=float(top_probs[row, 0]),
                second_probability=float(top_probs[row, 1]),
                margin=float(margins[row]),
                entropy=float(entropies[row]),
                candidates=tuple(str(artifact.classes[col]) for col in columns),
                candidate_probabilities=tuple(float(probabilities[row, col]) for col in columns),
            )
        )
    selection_manifest = {
        "exclusion_method": exclusion_method,
        "n_excluded_union": len(excluded),
        "excluded_row_positions_sha256": hashlib.sha256(
            json.dumps(sorted(excluded), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "n_eligible_after_exclusion": int(eligible.sum()),
        "n_selected": len(result),
    }
    return result, selection_manifest


def broad_family(label: str) -> str:
    if label.startswith("astrocyte_"):
        return "astrocyte"
    if label.startswith("oligodendrocyte_"):
        return "oligodendrocyte_lineage"
    if label.startswith("meninges_"):
        return "meninges"
    if label.startswith(("DH_ex_", "DM_ex_", "MV_ex_", "M_ex_")):
        return "excitatory_neuron"
    if label.startswith(("DH_in_", "MV_in_", "M_in_", "VH_in_")):
        return "inhibitory_neuron"
    if label in {
        "alpha_motoneuron",
        "beta_motoneuron",
        "gamma_motoneuron",
        "visceral_motoneuron",
    }:
        return "motoneuron"
    return label


def clean_meta_value(value: Any) -> Any:
    if pd.isna(value):
        return "MISSING"
    if isinstance(value, np.generic):
        return value.item()
    return value


class FoldEvidenceCache:
    """Fold-safe cosine neighbors and pairwise class statistics."""

    def __init__(
        self,
        normalized: np.ndarray,
        raw_counts: np.ndarray,
        genes: np.ndarray,
        y: np.ndarray,
        fold_ids: np.ndarray,
        ei: np.ndarray,
        neighbors: int,
        class_neighbors: int,
        pairwise_genes: int,
        min_pairwise_score: float,
    ) -> None:
        self.normalized = normalized
        self.raw_counts = raw_counts
        self.genes = genes
        self.y = y
        self.fold_ids = fold_ids
        self.ei = ei
        self.neighbors = neighbors
        self.class_neighbors = class_neighbors
        self.pairwise_genes = pairwise_genes
        self.min_pairwise_score = min_pairwise_score
        self._fold_matrix: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._class_stats: dict[tuple[int, str], dict[str, np.ndarray | int]] = {}

    def training_rows(self, fold: int, query_row: int | None = None) -> np.ndarray:
        rows = np.flatnonzero(self.fold_ids != fold)
        if query_row is not None and query_row in rows:
            raise RuntimeError("Validation query appeared in its outer-training fold")
        return rows

    def _normalized_fold_matrix(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        if fold not in self._fold_matrix:
            rows = self.training_rows(fold)
            matrix = self.normalized[rows].astype(float, copy=True)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms
            self._fold_matrix[fold] = (rows, matrix)
        return self._fold_matrix[fold]

    def neighbor_evidence(
        self,
        row: int,
        fold: int,
        candidates: tuple[str, ...],
    ) -> dict[str, Any]:
        train_rows, matrix = self._normalized_fold_matrix(fold)
        query = self.normalized[row].astype(float, copy=True)
        query_norm = float(np.linalg.norm(query))
        if query_norm > 0:
            query /= query_norm
        similarities = matrix @ query

        same_ei = self.ei[train_rows] == self.ei[row]
        if same_ei.any():
            pool_local = np.flatnonzero(same_ei)
        else:
            pool_local = np.arange(len(train_rows))
        pool_sim = similarities[pool_local]
        order = np.argsort(-pool_sim, kind="stable")
        top_local = pool_local[order[: min(self.neighbors, len(order))]]
        top_rows = train_rows[top_local]
        top_sim = np.maximum(similarities[top_local], 0.0)
        top_labels = self.y[top_rows]

        counts = Counter(top_labels.tolist())
        candidate_weight = {
            label: float(top_sim[top_labels == label].sum()) for label in candidates
        }
        total_candidate_weight = sum(candidate_weight.values())
        support: list[dict[str, Any]] = []
        for label in candidates:
            label_local = pool_local[self.y[train_rows[pool_local]] == label]
            label_sim = np.sort(similarities[label_local])[::-1]
            best = label_sim[: min(self.class_neighbors, len(label_sim))]
            support.append(
                {
                    "label": label,
                    "top_neighbor_count": int(counts.get(label, 0)),
                    "top_neighbor_vote_share": round(
                        candidate_weight[label] / total_candidate_weight, 4
                    )
                    if total_candidate_weight > 0
                    else 0.0,
                    "max_class_cosine": round(float(best[0]), 4) if len(best) else None,
                    "mean_top_class_cosine": round(float(best.mean()), 4)
                    if len(best)
                    else None,
                    "training_cells_in_search_pool": int(len(label_local)),
                }
            )
        neighbor_label = max(
            support,
            key=lambda item: (
                item["top_neighbor_vote_share"],
                -math.inf
                if item["mean_top_class_cosine"] is None
                else item["mean_top_class_cosine"],
                -candidates.index(item["label"]),
            ),
        )["label"]
        return {
            "query_ei_group": str(self.ei[row]),
            "search_pool_cells": int(len(pool_local)),
            "n_top_neighbors": int(len(top_rows)),
            "top_neighbor_label_counts": [
                {"label": str(label), "count": int(count)}
                for label, count in counts.most_common()
            ],
            "candidate_support": support,
            "neighbor_candidate_label": neighbor_label,
        }

    def _stats(self, fold: int, label: str) -> dict[str, np.ndarray | int]:
        key = (fold, label)
        if key not in self._class_stats:
            train = self.training_rows(fold)
            rows = train[self.y[train] == label]
            if len(rows) == 0:
                raise RuntimeError(
                    f"Class {label!r} is absent from outer-training fold {fold}"
                )
            expression = self.normalized[rows]
            self._class_stats[key] = {
                "n": int(len(rows)),
                "mean": expression.mean(axis=0),
                "variance": expression.var(axis=0),
                "detection": (self.raw_counts[rows] > 0).mean(axis=0),
            }
        return self._class_stats[key]

    def pairwise_evidence(
        self,
        row: int,
        fold: int,
        baseline: str,
        candidates: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
        detected = np.flatnonzero(self.raw_counts[row] > 0)
        baseline_stats = self._stats(fold, baseline)
        profiles: list[dict[str, Any]] = []
        support_sets: dict[str, set[str]] = {baseline: set()}

        for candidate in candidates:
            if candidate == baseline:
                continue
            candidate_stats = self._stats(fold, candidate)
            mean_new = np.asarray(candidate_stats["mean"])
            mean_base = np.asarray(baseline_stats["mean"])
            var_new = np.asarray(candidate_stats["variance"])
            var_base = np.asarray(baseline_stats["variance"])
            detect_new = np.asarray(candidate_stats["detection"])
            detect_base = np.asarray(baseline_stats["detection"])
            pooled_scale = np.sqrt(0.5 * (var_new + var_base)) + 0.10
            standardized_delta = (mean_new - mean_base) / pooled_scale
            detection_delta = detect_new - detect_base
            score = standardized_delta + detection_delta

            candidate_support = detected[
                (score[detected] >= self.min_pairwise_score)
                & (mean_new[detected] > mean_base[detected])
                & (detection_delta[detected] >= 0)
            ]
            baseline_support = detected[
                (score[detected] <= -self.min_pairwise_score)
                & (mean_new[detected] < mean_base[detected])
                & (detection_delta[detected] <= 0)
            ]
            candidate_ranked = candidate_support[
                np.argsort(-score[candidate_support], kind="stable")
            ][: self.pairwise_genes]
            baseline_ranked = baseline_support[
                np.argsort(score[baseline_support], kind="stable")
            ][: self.pairwise_genes]
            support_sets[candidate] = {
                str(self.genes[col]) for col in candidate_support
            }

            def records(columns: np.ndarray) -> list[dict[str, Any]]:
                return [
                    {
                        "gene": str(self.genes[col]),
                        "cell_count": round(float(self.raw_counts[row, col]), 3),
                        "cell_normalized": round(float(self.normalized[row, col]), 3),
                        "candidate_mean": round(float(mean_new[col]), 3),
                        "baseline_mean": round(float(mean_base[col]), 3),
                        "candidate_detection_rate": round(float(detect_new[col]), 3),
                        "baseline_detection_rate": round(float(detect_base[col]), 3),
                        "pairwise_score": round(float(score[col]), 3),
                    }
                    for col in columns
                ]

            profiles.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_training_cells": int(candidate_stats["n"]),
                    "baseline_training_cells": int(baseline_stats["n"]),
                    "observed_genes_supporting_candidate": records(candidate_ranked),
                    "observed_genes_supporting_baseline": records(baseline_ranked),
                }
            )
        return profiles, support_sets


def observed_gene_records(
    row: int,
    raw_counts: np.ndarray,
    normalized: np.ndarray,
    genes: np.ndarray,
    top_n: int,
    pairwise_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detected = np.flatnonzero(raw_counts[row] > 0)
    ranked = detected[np.argsort(-normalized[row, detected], kind="stable")]
    selected = list(ranked[:top_n])
    gene_to_col = {str(gene): col for col, gene in enumerate(genes)}
    for profile in pairwise_profiles:
        for direction in (
            "observed_genes_supporting_candidate",
            "observed_genes_supporting_baseline",
        ):
            for record in profile[direction]:
                selected.append(gene_to_col[record["gene"]])
    unique = list(dict.fromkeys(selected))
    return [
        {
            "gene": str(genes[col]),
            "count": round(float(raw_counts[row, col]), 3),
            "normalized_log_expression": round(float(normalized[row, col]), 3),
        }
        for col in unique
    ]


def candidate_orders(candidates: tuple[str, ...]) -> list[tuple[str, ...]]:
    canonical = tuple(candidates)
    rotated = tuple(candidates[1:] + candidates[:1])
    if rotated == canonical:
        rotated = tuple(reversed(canonical))
    return [canonical, rotated]


def build_prompt(
    cell: SelectedCell,
    order: tuple[str, ...],
    meta: pd.DataFrame,
    observed_genes: list[dict[str, Any]],
    neighbor: dict[str, Any],
    pairwise_profiles: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    probability_lookup = dict(zip(cell.candidates, cell.candidate_probabilities))
    neighbor_lookup = {
        item["label"]: item for item in neighbor["candidate_support"]
    }
    pairwise_lookup = {
        item["candidate"]: item for item in pairwise_profiles
    }
    candidates = []
    for label in order:
        candidates.append(
            {
                "label": label,
                "is_supervised_baseline": label == cell.baseline_label,
                "supervised_probability": round(probability_lookup[label], 6),
                "fold_training_neighbor_support": neighbor_lookup[label],
                "pairwise_vs_baseline": pairwise_lookup.get(label),
            }
        )
    safe_metadata = {
        column: clean_meta_value(meta.iloc[cell.row_position][column])
        for column in SAFE_META_COLUMNS
        if column in meta.columns
    }
    case = {
        "baseline_label": cell.baseline_label,
        "base_model_margin": round(cell.margin, 6),
        "candidates_in_presented_order": candidates,
        "cell_observed_genes": observed_genes,
        "fold_training_neighbor_summary": {
            key: value
            for key, value in neighbor.items()
            if key not in {"candidate_support"}
        },
        "safe_metadata": safe_metadata,
    }
    prompt = f"""You are a conservative cell-type candidate reranker.

Use only the supplied supervised probabilities and evidence calculated from
the outer-training fold. Do not introduce external marker knowledge. Candidate
display order is arbitrary and must not affect your answer. The supervised
baseline is identified explicitly by is_supervised_baseline. Choose the
baseline unless BOTH the cosine-neighbor summary and the observed pairwise
gene evidence clearly support one different candidate. Missing genes are weak
evidence because this is a sparse 200-gene panel.

Return only the requested JSON object. The label must be one supplied candidate.
Every evidence_genes entry must occur in cell_observed_genes and must positively
support the chosen label against the supervised baseline according to the
supplied fold-training pairwise evidence. Confidence is only a self-assessment.

Case:
{json.dumps(case, ensure_ascii=False, separators=(",", ":"))}
"""
    return prompt, case


def response_schema(
    candidates: tuple[str, ...], observed_gene_names: list[str]
) -> dict[str, Any]:
    gene_items: dict[str, Any] = {"type": "string"}
    if observed_gene_names:
        gene_items["enum"] = observed_gene_names
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(candidates)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_genes": {
                "type": "array",
                "items": gene_items,
                "uniqueItems": True,
                "maxItems": 6,
            },
            "reason_code": {
                "type": "string",
                "enum": [
                    "keep_supervised_prior",
                    "neighbor_and_pairwise_support",
                    "insufficient_switch_evidence",
                ],
            },
        },
        "required": ["label", "confidence", "evidence_genes", "reason_code"],
        "additionalProperties": False,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def http_json(
    url: str, payload: dict[str, Any] | None, timeout_seconds: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama(base_url: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    version = http_json(f"{base_url}/api/version", None, timeout_seconds)
    tags = http_json(f"{base_url}/api/tags", None, timeout_seconds)
    installed = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    aliases = installed | {name.split(":")[0] for name in installed}
    if model not in installed and model.split(":")[0] not in aliases:
        raise RuntimeError(
            f"Model {model!r} is not installed; installed models: {sorted(installed)}"
        )
    return {"version": version, "installed_models": sorted(installed)}


def validate_response(
    parsed: dict[str, Any],
    candidates: tuple[str, ...],
    observed_gene_names: set[str],
) -> dict[str, Any]:
    required = {"label", "confidence", "evidence_genes", "reason_code"}
    if not isinstance(parsed, dict) or not required.issubset(parsed):
        raise ValueError(f"Response lacks required fields: {parsed}")
    if parsed["label"] not in candidates:
        raise ValueError(f"Response label is outside candidates: {parsed}")
    confidence = float(parsed["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Response confidence is outside [0,1]: {parsed}")
    evidence = parsed["evidence_genes"]
    if (
        not isinstance(evidence, list)
        or len(evidence) > 6
        or len(evidence) != len(set(evidence))
    ):
        raise ValueError(f"Response evidence_genes must be a unique list: {parsed}")
    if not set(evidence).issubset(observed_gene_names):
        raise ValueError(f"Response cited a gene absent from supplied evidence: {parsed}")
    allowed_reason_codes = {
        "keep_supervised_prior",
        "neighbor_and_pairwise_support",
        "insufficient_switch_evidence",
    }
    if parsed["reason_code"] not in allowed_reason_codes:
        raise ValueError(f"Response has an invalid reason_code: {parsed}")
    parsed = dict(parsed)
    parsed["confidence"] = confidence
    parsed["evidence_genes"] = [str(gene) for gene in evidence]
    return parsed


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    candidates: tuple[str, ...],
    observed_gene_names: list[str],
    seed: int,
    timeout_seconds: float,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any], float, str]:
    schema = response_schema(candidates, observed_gene_names)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": schema,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": seed, "num_predict": 180},
        "keep_alive": "10m",
    }
    digest = request_hash(payload)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            raw = http_json(
                f"{base_url.rstrip('/')}/api/chat", payload, timeout_seconds
            )
            elapsed = time.perf_counter() - start
            content = raw.get("message", {}).get("content", "")
            parsed = validate_response(
                json.loads(content), candidates, set(observed_gene_names)
            )
            return parsed, raw, elapsed, digest
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts") from last_error


def load_response_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cache[item["request_hash"]] = item
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid cache line {line_number} in {path}") from error
    return cache


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def preview_request_hash(
    model: str,
    prompt: str,
    candidates: tuple[str, ...],
    observed_gene_names: list[str],
    seed: int,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": response_schema(candidates, observed_gene_names),
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": seed, "num_predict": 180},
        "keep_alive": "10m",
    }
    return request_hash(payload)


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def exact_paired_error_pvalue(corrected: int, introduced: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired label changes."""
    discordant = corrected + introduced
    if discordant == 0:
        return 1.0
    lower = min(corrected, introduced)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def run_request(
    args: argparse.Namespace,
    scheme: str,
    cell: SelectedCell,
    pass_number: int,
    order: tuple[str, ...],
    prompt: str,
    observed_names: list[str],
    cache: dict[str, dict[str, Any]],
    response_path: Path,
) -> dict[str, Any]:
    digest = preview_request_hash(args.model, prompt, order, observed_names, args.seed)
    result: dict[str, Any] | None = None
    error_message = ""
    latency = float("nan")
    source = "dry_run"
    if not args.dry_run:
        cached = cache.get(digest) if not args.no_resume else None
        if cached and cached.get("parsed_response") is not None and not cached.get("error"):
            result = validate_response(
                cached["parsed_response"], order, set(observed_names)
            )
            latency = float(cached.get("latency_seconds", float("nan")))
            source = "cache"
        else:
            source = "ollama"
            raw: dict[str, Any] | None = None
            try:
                result, raw, latency, actual_digest = call_ollama(
                    args.ollama_url,
                    args.model,
                    prompt,
                    order,
                    observed_names,
                    args.seed,
                    args.timeout_seconds,
                    args.retries,
                )
                if actual_digest != digest:
                    raise RuntimeError("Internal request-hash mismatch")
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
            cache_item = {
                "request_hash": digest,
                "version": VERSION,
                "scheme": scheme,
                "row_position": cell.row_position,
                "cell_id": cell.cell_id,
                "candidate_order_pass": pass_number,
                "candidate_order": list(order),
                "model": args.model,
                "parsed_response": result,
                "error": error_message,
                "latency_seconds": latency,
                "ollama_eval_count": raw.get("eval_count") if raw else None,
                "ollama_eval_duration": raw.get("eval_duration") if raw else None,
            }
            append_jsonl(response_path, cache_item)
            cache[digest] = cache_item
    return {
        "result": result,
        "success": result is not None and not error_message,
        "error": error_message,
        "latency": latency,
        "source": source,
        "request_hash": digest,
        "order": order,
    }


def gate_decision(
    cell: SelectedCell,
    requests: list[dict[str, Any]],
    neighbor: dict[str, Any],
    pairwise_support: dict[str, set[str]],
    min_confidence: float,
    min_evidence: int,
    switch_policy: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    both_success = len(requests) == 2 and all(item["success"] for item in requests)
    if not both_success:
        reasons.append("request_failure")
        labels: list[str] = []
        confidences: list[float] = []
        evidence_sets: list[set[str]] = []
    else:
        labels = [str(item["result"]["label"]) for item in requests]
        confidences = [float(item["result"]["confidence"]) for item in requests]
        evidence_sets = [set(item["result"]["evidence_genes"]) for item in requests]

    consensus = bool(both_success and labels[0] == labels[1])
    if both_success and not consensus:
        reasons.append("candidate_order_disagreement")
    consensus_label = labels[0] if consensus else cell.baseline_label
    is_switch = consensus_label != cell.baseline_label
    if consensus and not is_switch:
        reasons.append("consensus_keep")

    confidence_min = min(confidences) if confidences else float("nan")
    if is_switch and confidence_min < min_confidence:
        reasons.append("confidence_below_threshold")

    common_evidence = set.intersection(*evidence_sets) if evidence_sets else set()
    supporting = pairwise_support.get(consensus_label, set())
    common_supported = sorted(common_evidence.intersection(supporting))
    if is_switch and len(common_supported) < min_evidence:
        reasons.append("insufficient_common_pairwise_evidence")

    neighbor_candidate_label = str(neighbor["neighbor_candidate_label"])
    neighbor_agrees = bool(
        is_switch and neighbor_candidate_label == consensus_label
    )
    if is_switch and not neighbor_agrees:
        reasons.append("neighbor_disagrees_with_switch")

    baseline_family = broad_family(cell.baseline_label)
    consensus_family = broad_family(consensus_label)
    fine_switch_blocked = bool(
        is_switch
        and switch_policy == "cross_family"
        and baseline_family == consensus_family
    )
    if fine_switch_blocked:
        reasons.append("within_family_switch_blocked")

    gate_passed = bool(
        both_success
        and consensus
        and is_switch
        and confidence_min >= min_confidence
        and len(common_supported) >= min_evidence
        and neighbor_agrees
        and not fine_switch_blocked
    )
    if gate_passed:
        reasons = ["passed"]
    return {
        "both_requests_success": both_success,
        "candidate_order_consensus": consensus,
        "consensus_label": consensus_label,
        "derived_action": "switch" if is_switch else "keep",
        "minimum_confidence": confidence_min,
        "common_evidence_genes": sorted(common_evidence),
        "common_supported_evidence_genes": common_supported,
        "n_common_supported_evidence": len(common_supported),
        "neighbor_candidate_label": neighbor_candidate_label,
        "neighbor_agrees_with_switch": neighbor_agrees,
        "baseline_family": baseline_family,
        "consensus_family": consensus_family,
        "fine_switch_blocked": fine_switch_blocked,
        "gate_passed": gate_passed,
        "gate_reasons": reasons,
        "gated_label": consensus_label if gate_passed else cell.baseline_label,
    }


def evaluate_scheme(
    args: argparse.Namespace,
    artifact: OOFArtifact,
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    y: np.ndarray,
    raw_counts: np.ndarray,
    normalized: np.ndarray,
    genes: np.ndarray,
    pilot_positions: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selected, selection_manifest = select_fresh_cells(
        artifact,
        args.candidate_k,
        args.max_cells_per_scheme,
        args.max_base_confidence,
        pilot_positions,
        args.include_pilot,
        args.pilot_fallback_skip,
    )
    ei = (
        meta["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    evidence_cache = FoldEvidenceCache(
        normalized,
        raw_counts,
        genes,
        y,
        artifact.fold_ids,
        ei,
        args.neighbors,
        args.class_neighbors,
        args.pairwise_genes_per_direction,
        args.min_pairwise_support_score,
    )
    response_path = args.out_dir / f"v032_llm_responses_{artifact.scheme}.jsonl"
    prompt_path = args.out_dir / f"v032_llm_prompts_{artifact.scheme}.jsonl"
    cache = {} if args.no_resume else load_response_cache(response_path)
    rows: list[dict[str, Any]] = []

    with prompt_path.open("w", encoding="utf-8") as prompt_handle:
        for number, cell in enumerate(selected, start=1):
            neighbor = evidence_cache.neighbor_evidence(
                cell.row_position, cell.fold, cell.candidates
            )
            pairwise_profiles, pairwise_support = evidence_cache.pairwise_evidence(
                cell.row_position,
                cell.fold,
                cell.baseline_label,
                cell.candidates,
            )
            observed = observed_gene_records(
                cell.row_position,
                raw_counts,
                normalized,
                genes,
                args.cell_genes,
                pairwise_profiles,
            )
            observed_names = [item["gene"] for item in observed]
            requests: list[dict[str, Any]] = []
            for pass_number, order in enumerate(candidate_orders(cell.candidates), start=1):
                prompt, case = build_prompt(
                    cell, order, meta, observed, neighbor, pairwise_profiles
                )
                digest = preview_request_hash(
                    args.model, prompt, order, observed_names, args.seed
                )
                prompt_handle.write(
                    json.dumps(
                        {
                            "version": VERSION,
                            "scheme": artifact.scheme,
                            "row_position": cell.row_position,
                            "cell_id": cell.cell_id,
                            "fold": cell.fold,
                            "candidate_order_pass": pass_number,
                            "candidate_order": list(order),
                            "request_hash": digest,
                            "case": case,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                requests.append(
                    run_request(
                        args,
                        artifact.scheme,
                        cell,
                        pass_number,
                        order,
                        prompt,
                        observed_names,
                        cache,
                        response_path,
                    )
                )

            gate = gate_decision(
                cell,
                requests,
                neighbor,
                pairwise_support,
                args.min_llm_confidence,
                args.min_supported_evidence,
                args.switch_policy,
            )
            # True labels are accessed only now, after label-free selection,
            # fold-safe evidence construction, both LLM calls, and gating.
            true_label = str(y[cell.row_position])
            gated_label = str(gate["gated_label"])
            neighbor_label = str(neighbor["neighbor_candidate_label"])
            row = asdict(cell)
            row.update(
                {
                    "candidates": "|".join(cell.candidates),
                    "candidate_probabilities": "|".join(
                        f"{value:.8f}" for value in cell.candidate_probabilities
                    ),
                    "neighbor_candidate_label": neighbor_label,
                    "neighbor_candidate_support_json": canonical_json(
                        neighbor["candidate_support"]
                    ),
                    "pairwise_support_counts_json": canonical_json(
                        {label: len(values) for label, values in pairwise_support.items()}
                    ),
                }
            )
            for pass_number, request in enumerate(requests, start=1):
                result = request["result"] if request["success"] else {}
                row.update(
                    {
                        f"pass{pass_number}_order": "|".join(request["order"]),
                        f"pass{pass_number}_success": request["success"],
                        f"pass{pass_number}_label": result.get("label", ""),
                        f"pass{pass_number}_confidence": result.get(
                            "confidence", float("nan")
                        ),
                        f"pass{pass_number}_evidence_genes": "|".join(
                            result.get("evidence_genes", [])
                        ),
                        f"pass{pass_number}_reason_code": result.get("reason_code", ""),
                        f"pass{pass_number}_source": request["source"],
                        f"pass{pass_number}_latency_seconds": request["latency"],
                        f"pass{pass_number}_error": request["error"],
                        f"pass{pass_number}_request_hash": request["request_hash"],
                    }
                )
            row.update(
                {
                    "both_requests_success": gate["both_requests_success"],
                    "candidate_order_consensus": gate["candidate_order_consensus"],
                    "consensus_label": gate["consensus_label"],
                    "derived_action": gate["derived_action"],
                    "minimum_confidence": gate["minimum_confidence"],
                    "common_evidence_genes": "|".join(gate["common_evidence_genes"]),
                    "common_supported_evidence_genes": "|".join(
                        gate["common_supported_evidence_genes"]
                    ),
                    "n_common_supported_evidence": gate[
                        "n_common_supported_evidence"
                    ],
                    "neighbor_agrees_with_switch": gate[
                        "neighbor_agrees_with_switch"
                    ],
                    "baseline_family": gate["baseline_family"],
                    "consensus_family": gate["consensus_family"],
                    "fine_switch_blocked": gate["fine_switch_blocked"],
                    "gate_passed": gate["gate_passed"],
                    "gate_reasons": "|".join(gate["gate_reasons"]),
                    "gated_label": gated_label,
                    "true_label": true_label,
                    "truth_in_candidates": true_label in cell.candidates,
                    "baseline_correct": cell.baseline_label == true_label,
                    "neighbor_correct": neighbor_label == true_label,
                    "consensus_correct": gate["consensus_label"] == true_label,
                    "gated_correct": gated_label == true_label,
                    "corrected_error": (
                        cell.baseline_label != true_label and gated_label == true_label
                    ),
                    "introduced_error": (
                        cell.baseline_label == true_label and gated_label != true_label
                    ),
                    "wrong_to_wrong": (
                        cell.baseline_label != true_label
                        and gated_label != true_label
                        and gated_label != cell.baseline_label
                    ),
                }
            )
            rows.append(row)
            print(
                f"[{artifact.scheme}] {number:>3}/{len(selected)} "
                f"row={cell.row_position} base={cell.baseline_label} "
                f"consensus={gate['consensus_label']} gated={gated_label} "
                f"pass={gate['gate_passed']}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    cell_path = args.out_dir / f"v032_llm_reranker_cells_{artifact.scheme}.csv"
    frame.to_csv(cell_path, index=False)

    baseline = artifact.classes[artifact.probabilities.argmax(axis=1)]
    gated_hybrid = baseline.copy()
    neighbor_hybrid = baseline.copy()
    selected_positions = frame["row_position"].to_numpy(dtype=int)
    gated_hybrid[selected_positions] = frame["gated_label"].astype(str).to_numpy()
    neighbor_hybrid[selected_positions] = frame["neighbor_candidate_label"].astype(str).to_numpy()
    baseline_accuracy = safe_mean(baseline == y)
    gated_accuracy = safe_mean(gated_hybrid == y)
    neighbor_accuracy = safe_mean(neighbor_hybrid == y)

    fold_rows: list[dict[str, Any]] = []
    for fold in np.unique(artifact.fold_ids):
        fold_mask = artifact.fold_ids == fold
        selected_fold = frame[frame["fold"] == fold]
        fold_rows.append(
            {
                "version": VERSION,
                "scheme": artifact.scheme,
                "fold": int(fold),
                "held_groups": ",".join(
                    sorted(
                        meta.loc[fold_mask, artifact.group_column]
                        .fillna("MISSING")
                        .astype(str)
                        .unique()
                    )
                ),
                "n_fold_cells": int(fold_mask.sum()),
                "n_selected": len(selected_fold),
                "baseline_accuracy": safe_mean(baseline[fold_mask] == y[fold_mask]),
                "gated_accuracy": safe_mean(
                    gated_hybrid[fold_mask] == y[fold_mask]
                ),
                "gated_delta": safe_mean(gated_hybrid[fold_mask] == y[fold_mask])
                - safe_mean(baseline[fold_mask] == y[fold_mask]),
                "n_gate_switches": int(selected_fold["gate_passed"].sum()),
                "n_corrected_errors": int(selected_fold["corrected_error"].sum()),
                "n_introduced_errors": int(selected_fold["introduced_error"].sum()),
                "net_corrections": int(selected_fold["corrected_error"].sum())
                - int(selected_fold["introduced_error"].sum()),
            }
        )

    summary = {
        "version": VERSION,
        "scheme": artifact.scheme,
        "group_column": artifact.group_column,
        "oof_npz": str(artifact.path.resolve()),
        "model": args.model,
        "switch_policy": args.switch_policy,
        "n_total_cells": len(y),
        "n_selected_fresh": len(frame),
        "coverage": len(frame) / len(y),
        "n_both_requests_success": int(frame["both_requests_success"].sum()),
        "n_candidate_order_consensus": int(frame["candidate_order_consensus"].sum()),
        "baseline_overall_accuracy": baseline_accuracy,
        "baseline_selected_accuracy": safe_mean(
            frame["baseline_correct"].to_numpy(dtype=bool)
        ),
        "candidate_topk_ceiling_selected": safe_mean(
            frame["truth_in_candidates"].to_numpy(dtype=bool)
        ),
        "neighbor_selected_accuracy": safe_mean(
            frame["neighbor_correct"].to_numpy(dtype=bool)
        ),
        "neighbor_hybrid_overall_accuracy": neighbor_accuracy,
        "neighbor_hybrid_delta": neighbor_accuracy - baseline_accuracy,
        "consensus_selected_accuracy": safe_mean(
            frame["consensus_correct"].to_numpy(dtype=bool)
        ),
        "gated_selected_accuracy": safe_mean(
            frame["gated_correct"].to_numpy(dtype=bool)
        ),
        "gated_hybrid_overall_accuracy": gated_accuracy,
        "gated_hybrid_delta": gated_accuracy - baseline_accuracy,
        "n_gate_switches": int(frame["gate_passed"].sum()),
        "n_corrected_errors": int(frame["corrected_error"].sum()),
        "n_introduced_errors": int(frame["introduced_error"].sum()),
        "n_wrong_to_wrong": int(frame["wrong_to_wrong"].sum()),
        "net_corrections": int(frame["corrected_error"].sum())
        - int(frame["introduced_error"].sum()),
        "paired_exact_mcnemar_p": exact_paired_error_pvalue(
            int(frame["corrected_error"].sum()),
            int(frame["introduced_error"].sum()),
        ),
        "n_fine_switches_blocked": int(frame["fine_switch_blocked"].sum()),
        "worst_fold_gated_delta": min(row["gated_delta"] for row in fold_rows),
        "mean_uncached_latency_seconds_per_request": safe_mean(
            np.concatenate(
                [
                    frame.loc[
                        frame[f"pass{pass_number}_source"] == "ollama",
                        f"pass{pass_number}_latency_seconds",
                    ]
                    .dropna()
                    .to_numpy(dtype=float)
                    for pass_number in (1, 2)
                ]
            )
        ),
    }
    return summary, fold_rows, selection_manifest


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts, meta = load_training_data(args.data_dir)
    y = meta[TARGET].astype(str).to_numpy()
    raw_counts = counts.to_numpy(dtype=float)
    normalized = normalized_expression(counts)
    genes = counts.columns.astype(str).to_numpy()
    if args.oof_npz:
        oof_paths = args.oof_npz
    else:
        oof_paths = [
            args.oof_dir / f"v031_oof_{scheme}.npz" for scheme in args.schemes
        ]
    artifacts = load_artifacts(oof_paths, counts, meta)

    pilot_positions: set[int] = set()
    pilot_sources: list[str] = []
    if not args.include_pilot:
        pilot_positions, pilot_sources = load_pilot_positions(
            args.pilot_dir, counts.index.astype(str).to_numpy()
        )

    ollama_info: dict[str, Any] | None = None
    if not args.dry_run:
        try:
            ollama_info = check_ollama(
                args.ollama_url, args.model, args.timeout_seconds
            )
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {args.ollama_url}; start `ollama serve`."
            ) from error

    manifest: dict[str, Any] = {
        "version": VERSION,
        "purpose": "fresh fold-safe local LLM OOF reranker evaluation",
        "submission_approved": False,
        "warning": (
            "Pretrained model knowledge may count as external information. Do not "
            "connect this experiment to final prediction code without written "
            "organizer approval."
        ),
        "created_unix_time": time.time(),
        "data_dir": str(args.data_dir.resolve()),
        "oof_dir": str(args.oof_dir.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "oof_artifacts": [
            {
                "path": str(item.path.resolve()),
                "scheme": item.scheme,
                "group_column": item.group_column,
                "sha256": hashlib.sha256(item.path.read_bytes()).hexdigest(),
            }
            for item in artifacts
        ],
        "model": args.model,
        "candidate_k": args.candidate_k,
        "max_cells_per_scheme": args.max_cells_per_scheme,
        "max_base_confidence": args.max_base_confidence,
        "neighbors": args.neighbors,
        "class_neighbors": args.class_neighbors,
        "pairwise_genes_per_direction": args.pairwise_genes_per_direction,
        "min_pairwise_support_score": args.min_pairwise_support_score,
        "min_llm_confidence": args.min_llm_confidence,
        "min_supported_evidence": args.min_supported_evidence,
        "switch_policy": args.switch_policy,
        "candidate_order_passes": 2,
        "pilot_dir": str(args.pilot_dir.resolve()),
        "pilot_sources": pilot_sources,
        "pilot_union_positions_loaded": len(pilot_positions),
        "include_pilot": args.include_pilot,
        "pilot_fallback_skip": args.pilot_fallback_skip,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "safe_prompt_metadata": SAFE_META_COLUMNS,
        "excluded_prompt_metadata": ["Mouse_ID", "Section_ID", "Datasets"],
        "ollama": ollama_info,
        "status": "started",
    }
    manifest_path = args.out_dir / "v032_llm_reranker_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    selection_manifests: dict[str, Any] = {}
    for artifact in artifacts:
        summary, fold_rows, selection_manifest = evaluate_scheme(
            args,
            artifact,
            counts,
            meta,
            y,
            raw_counts,
            normalized,
            genes,
            pilot_positions,
        )
        summaries.append(summary)
        all_fold_rows.extend(fold_rows)
        selection_manifests[artifact.scheme] = selection_manifest

    summary_frame = pd.DataFrame(summaries)
    summary_path = args.out_dir / "v032_llm_reranker_summary.csv"
    fold_path = args.out_dir / "v032_llm_reranker_folds.csv"
    summary_frame.to_csv(summary_path, index=False)
    pd.DataFrame(all_fold_rows).to_csv(fold_path, index=False)
    manifest["status"] = "complete"
    manifest["completed_unix_time"] = time.time()
    manifest["selection"] = selection_manifests
    manifest["output_contract"] = {
        "per_cell": "v032_llm_reranker_cells_<scheme>.csv",
        "summary": summary_path.name,
        "folds": fold_path.name,
        "prompts_without_truth": "v032_llm_prompts_<scheme>.jsonl",
        "resumable_response_cache": "v032_llm_responses_<scheme>.jsonl",
        "manifest": manifest_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== v0.32 LOCAL QWEN3 FRESH OOF RERANKER ===")
    print(summary_frame.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
