#!/usr/bin/env python3
"""v0.34: v0.31 five-model prediction with an optional local Qwen gate.

The default path is deliberately conservative: Qwen is OFF unless
``--enable-qwen`` is supplied, so the resulting labels are exactly the v0.31
masked equal mean of LGBM, LR, MLP, KNN and CatBoost.  When enabled, Qwen may
rerank only a small, low-margin subset of the v0.31 top-3 candidates.  Every
accepted change must pass two candidate-order calls, full-training neighbor
agreement, common observed pairwise-gene evidence, a cross-family rule, and a
vascular-family lock.

No OOF labels or test labels are read by the Qwen decision path.  All labeled
evidence is calculated from the official training rows; test rows are used
only as unlabeled queries.  Mouse_ID, Section_ID and Datasets never enter a
prompt.  The pretrained Qwen model may nevertheless count as external
information, so every manifest records ``submission_approved=false``.

Examples
--------
Safe v0.31-only prediction (Qwen disabled):

    python src/v0.34-v0.31-Qwen3-Gated-Fusion.py

Explicit local Qwen experiment/fusion:

    python src/v0.34-v0.31-Qwen3-Gated-Fusion.py --enable-qwen

Validate configuration and input files without fitting models or calling
Ollama:

    python src/v0.34-v0.31-Qwen3-Gated-Fusion.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


VERSION = "v0.34"
TARGET = "MERFISH_cell_type_annotation"
PREDICTION_COLUMN = "MERFISH_cell_type_annotation.y"
FORBIDDEN_PROMPT_COLUMNS = {"Mouse_ID", "Section_ID", "Datasets"}
SAFE_PROMPT_COLUMNS = (
    "Excitatory_vs_Inhibitory",
    "Region",
    "Segment",
    "AP_position",
)
VASCULAR_LABELS = {"endothelial", "pericyte"}

ROOT = Path(__file__).resolve().parent.parent
V031_PATH = ROOT / "src" / "v0.31-Robust-NoID-5ModelMean.py"
V032_PATH = ROOT / "src" / "v0.32-Qwen3-v0.31-OOF-Reranker.py"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs" / "v034_v031_qwen3_gated_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v0.31 robust five-model predictor, optionally followed "
            "by a conservative local Qwen3 top-3 gate."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--submission-out",
        type=Path,
        default=None,
        help="Defaults to <out-dir>/prediction/prediction.csv.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help=(
            "v0.31 model seeds; defaults to the validated five-seed baseline. "
            "Use --seeds 0 explicitly for a faster one-seed run."
        ),
    )
    parser.add_argument(
        "--enable-qwen",
        action="store_true",
        help="Explicitly enable the experimental local Qwen fusion layer.",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-k", type=int, default=3)
    parser.add_argument("--max-qwen-cells", type=int, default=100)
    parser.add_argument(
        "--qwen-coverage",
        type=float,
        default=0.02,
        help=(
            "Maximum fraction of test rows sent to Qwen. The actual limit is "
            "min(ceil(n_test * coverage), --max-qwen-cells) when the latter is "
            "positive."
        ),
    )
    parser.add_argument("--max-base-confidence", type=float, default=0.80)
    parser.add_argument(
        "--max-margin",
        type=float,
        default=0.05,
        help="Qwen is considered only when top1 - top2 is at most this value.",
    )
    parser.add_argument("--cell-genes", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--class-neighbors", type=int, default=5)
    parser.add_argument("--pairwise-genes-per-direction", type=int, default=8)
    parser.add_argument("--min-pairwise-support-score", type=float, default=0.25)
    parser.add_argument("--min-llm-confidence", type=float, default=0.80)
    parser.add_argument("--min-supported-evidence", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Qwen decoding seed.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--base-probabilities-npz",
        type=Path,
        default=None,
        help=(
            "Optional explicit v0.31-compatible test NPZ. IDs/classes/shape are "
            "validated, but supplying the correct current-data file is the user's "
            "responsibility."
        ),
    )
    parser.add_argument(
        "--allow-legacy-object-npz",
        action="store_true",
        help=(
            "Opt in to pickle-loading an explicitly supplied trusted legacy "
            "v0.31 NPZ whose cell_ids were stored as object dtype. Never use "
            "this for an untrusted artifact."
        ),
    )
    parser.add_argument(
        "--no-base-cache",
        action="store_true",
        help="Do not read or write the fingerprinted v0.34 base-probability cache.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching Ollama response-cache entries and call again.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Permit overwriting completed derived outputs; caches remain append-only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/imports and print the plan; do not fit or call Ollama.",
    )
    args = parser.parse_args()
    if args.submission_out is None:
        args.submission_out = args.out_dir / "prediction" / "prediction.csv"
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain one or more unique integers")
    if args.candidate_k != 3:
        parser.error("v0.34 intentionally requires --candidate-k 3")
    if args.max_qwen_cells < 0:
        parser.error("--max-qwen-cells cannot be negative")
    if not 0.0 <= args.qwen_coverage <= 1.0:
        parser.error("--qwen-coverage must be in [0,1]")
    if args.cell_genes < 1 or args.neighbors < 1 or args.class_neighbors < 1:
        parser.error("gene and neighbor counts must be positive")
    if args.pairwise_genes_per_direction < 1:
        parser.error("--pairwise-genes-per-direction must be positive")
    if args.min_supported_evidence < 2:
        parser.error("--min-supported-evidence must be at least 2")
    for name in ("max_base_confidence", "min_llm_confidence"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.max_margin < 0:
        parser.error("--max-margin cannot be negative")
    if args.min_pairwise_support_score < 0:
        parser.error("--min-pairwise-support-score cannot be negative")
    return args


def load_local_module(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import local module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_ollama_with_model_provenance(
    v032: ModuleType,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Validate Ollama and retain the resolved local model digest for audit."""
    info = v032.check_ollama(base_url, model, timeout_seconds)
    tags = v032.http_json(
        f"{base_url.rstrip('/')}/api/tags", None, timeout_seconds
    )
    records = [
        item
        for item in tags.get("models", [])
        if item.get("name") == model
        or (
            ":" not in model
            and str(item.get("name", "")).split(":", 1)[0] == model
        )
    ]
    if not records:
        raise RuntimeError(f"Could not resolve Ollama provenance for {model!r}")
    selected = records[0]
    info["selected_model"] = {
        key: selected.get(key)
        for key in ("name", "model", "digest", "size", "modified_at", "details")
        if key in selected
    }
    return info


def data_hashes(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("counts_train.csv", "meta_train.csv", "counts_test.csv", "meta_test.csv"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = sha256_file(path)
    return result


def combined_fingerprint(hashes: dict[str, str], seeds: list[int]) -> str:
    payload = {"data": hashes, "seeds": seeds, "v031_sha256": sha256_file(V031_PATH)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scalar_text(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {name} must be scalar")
    return str(array.reshape(-1)[0])


def validate_base_arrays(
    probabilities: np.ndarray,
    classes: np.ndarray,
    cell_ids: np.ndarray,
    bundle: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes).astype(str)
    cell_ids = np.asarray(cell_ids).astype(str)
    expected_ids = bundle.counts_test.index.astype(str).to_numpy()
    expected_classes = np.unique(bundle.meta_train[TARGET].astype(str).to_numpy())
    if probabilities.shape != (len(expected_ids), len(expected_classes)):
        raise ValueError(
            f"Base probability shape {probabilities.shape} != "
            f"{(len(expected_ids), len(expected_classes))}"
        )
    if not np.array_equal(cell_ids, expected_ids):
        raise ValueError("Base NPZ cell_ids do not match counts_test.csv row order")
    if not np.array_equal(classes, expected_classes):
        raise ValueError("Base NPZ classes do not match sorted training classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-12):
        raise ValueError("Base probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Base probability rows do not sum to one")
    return np.clip(probabilities, 0.0, 1.0), classes, cell_ids


def load_base_npz(
    path: Path,
    bundle: Any,
    allow_legacy_object: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    def read(allow_pickle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=allow_pickle) as saved:
            required = {"p_mean5_masked", "classes", "cell_ids"}
            missing = required.difference(saved.files)
            if missing:
                raise KeyError(f"{path}: missing keys {sorted(missing)}")
            return validate_base_arrays(
                saved["p_mean5_masked"], saved["classes"], saved["cell_ids"], bundle
            )

    try:
        return read(False)
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
        if not allow_legacy_object:
            raise ValueError(
                f"{path} contains a legacy object-dtype array. Recreate it with "
                "the corrected v0.31, or explicitly opt in for this trusted local "
                "artifact with --allow-legacy-object-npz."
            ) from error
        return read(True)


def compute_v031_base(
    v031: ModuleType,
    bundle: Any,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle.counts_test is None or bundle.meta_test is None:
        raise RuntimeError("Test data are unavailable")
    n_train = len(bundle.counts_train)
    # Match the corrected v0.31 path exactly: section ranks are computed
    # independently in train and test, while categorical encoders/TE fit on train.
    train_store = v031.build_store(bundle.counts_train, bundle.meta_train)
    test_store = v031.build_store(bundle.counts_test, bundle.meta_test)
    combined_meta = pd.concat([bundle.meta_train, bundle.meta_test], axis=0)
    store = v031.FeatureStore(
        expression=np.vstack([train_store.expression, test_store.expression]),
        section_qc=np.vstack([train_store.section_qc, test_store.section_qc]),
        meta=combined_meta,
    )
    y = bundle.meta_train[TARGET].astype(str).to_numpy()
    classes = np.unique(y)
    train_rows = np.arange(n_train, dtype=int)
    test_rows = np.arange(n_train, n_train + len(bundle.counts_test), dtype=int)
    per_seed = []
    for seed in seeds:
        start = time.time()
        per_seed.append(
            v031.five_model_probabilities(
                store, y, classes, train_rows, test_rows, seed
            )
        )
        print(f"[v0.31] seed={seed} complete in {time.time() - start:.1f}s", flush=True)
    averaged = {
        model: np.mean([item[model] for item in per_seed], axis=0)
        for model in v031.BASE_MODELS
    }
    mean5 = v031.probability_variants(averaged)["mean5"]
    train_ei = (
        bundle.meta_train["Excitatory_vs_Inhibitory"]
        .fillna("MISSING").astype(str).to_numpy()
    )
    test_ei = (
        bundle.meta_test["Excitatory_vs_Inhibitory"]
        .fillna("MISSING").astype(str).to_numpy()
    )
    allowed = v031.allowed_by_ei(y, train_ei, train_rows)
    masked = v031.mask_probabilities(mean5, classes, test_ei, allowed)
    cell_ids = bundle.counts_test.index.astype(str).to_numpy()
    return validate_base_arrays(masked, classes, cell_ids, bundle)


def load_or_compute_base(
    args: argparse.Namespace,
    v031: ModuleType,
    bundle: Any,
    fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, Path | None]:
    if args.base_probabilities_npz is not None:
        p, classes, ids = load_base_npz(
            args.base_probabilities_npz,
            bundle,
            allow_legacy_object=args.allow_legacy_object_npz,
        )
        return p, classes, ids, "explicit_npz", args.base_probabilities_npz

    cache_path = args.out_dir / "v034_base_probabilities.npz"
    if cache_path.exists() and not args.no_base_cache:
        with np.load(cache_path, allow_pickle=False) as saved:
            if "data_fingerprint" not in saved.files:
                raise ValueError(f"Unversioned base cache: {cache_path}")
            cached_fingerprint = scalar_text(
                saved["data_fingerprint"], "data_fingerprint", cache_path
            )
            if cached_fingerprint != fingerprint:
                raise ValueError(
                    f"Stale base cache {cache_path}; use a new --out-dir or "
                    "--no-base-cache"
                )
            p, classes, ids = validate_base_arrays(
                saved["p_mean5_masked"], saved["classes"], saved["cell_ids"], bundle
            )
        print(f"Loaded fingerprinted v0.31 base cache: {cache_path}", flush=True)
        return p, classes, ids, "v034_cache", cache_path

    p, classes, ids = compute_v031_base(v031, bundle, args.seeds)
    if not args.no_base_cache:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            p_mean5_masked=p,
            classes=np.asarray(classes, dtype=str),
            cell_ids=np.asarray(ids, dtype=str),
            data_fingerprint=np.array(fingerprint),
            seeds=np.asarray(args.seeds, dtype=int),
        )
        print(f"Saved fingerprinted v0.31 base cache: {cache_path}", flush=True)
        return p, classes, ids, "computed_and_cached", cache_path
    return p, classes, ids, "computed", None


def select_test_cells(
    v032: ModuleType,
    probabilities: np.ndarray,
    classes: np.ndarray,
    cell_ids: np.ndarray,
    n_train: int,
    args: argparse.Namespace,
) -> list[Any]:
    order, top_probabilities, margins, entropies = v032.candidate_arrays(
        probabilities, 3
    )
    eligible = (
        (top_probabilities[:, 0] <= args.max_base_confidence)
        & (margins <= args.max_margin)
    )
    rows = np.flatnonzero(eligible)
    rows = rows[np.lexsort((rows, margins[rows]))]
    coverage_limit = int(math.ceil(len(probabilities) * args.qwen_coverage))
    actual_limit = coverage_limit
    if args.max_qwen_cells > 0:
        actual_limit = min(actual_limit, args.max_qwen_cells)
    rows = rows[:actual_limit]
    selected = []
    for row in rows:
        columns = order[row]
        selected.append(
            v032.SelectedCell(
                scheme="test",
                row_position=n_train + int(row),
                fold=1,
                held_groups="UNLABELED_TEST",
                cell_id=str(cell_ids[row]),
                baseline_label=str(classes[columns[0]]),
                baseline_probability=float(top_probabilities[row, 0]),
                second_probability=float(top_probabilities[row, 1]),
                margin=float(margins[row]),
                entropy=float(entropies[row]),
                candidates=tuple(str(classes[column]) for column in columns),
                candidate_probabilities=tuple(
                    float(probabilities[row, column]) for column in columns
                ),
            )
        )
    return selected


def clean_meta_value(value: Any) -> Any:
    if pd.isna(value):
        return "MISSING"
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_full_train_prompt(
    cell: Any,
    order: tuple[str, ...],
    meta_all: pd.DataFrame,
    observed_genes: list[dict[str, Any]],
    neighbor: dict[str, Any],
    pairwise_profiles: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    probability_lookup = dict(zip(cell.candidates, cell.candidate_probabilities))
    neighbor_lookup = {item["label"]: item for item in neighbor["candidate_support"]}
    pairwise_lookup = {item["candidate"]: item for item in pairwise_profiles}
    candidates = [
        {
            "label": label,
            "is_supervised_baseline": label == cell.baseline_label,
            "supervised_probability": round(probability_lookup[label], 6),
            "full_train_neighbor_support": neighbor_lookup[label],
            "pairwise_vs_baseline": pairwise_lookup.get(label),
        }
        for label in order
    ]
    safe_metadata = {
        column: clean_meta_value(meta_all.iloc[cell.row_position][column])
        for column in SAFE_PROMPT_COLUMNS
        if column in meta_all.columns
    }
    if FORBIDDEN_PROMPT_COLUMNS.intersection(safe_metadata):
        raise RuntimeError("Forbidden identifier metadata entered the prompt")
    case = {
        "baseline_label": cell.baseline_label,
        "base_model_margin": round(cell.margin, 6),
        "candidates_in_presented_order": candidates,
        "cell_observed_genes": observed_genes,
        "full_training_neighbor_summary": {
            key: value for key, value in neighbor.items() if key != "candidate_support"
        },
        "safe_metadata": safe_metadata,
    }
    prompt = f"""You are a conservative cell-type candidate reranker.

Use only the supplied supervised probabilities and aggregate evidence computed
from the official labeled training rows. Do not introduce external marker
knowledge. Candidate order is arbitrary. Keep the supervised baseline unless
BOTH cosine-neighbor evidence and observed pairwise-gene evidence clearly
support one different candidate. Missing genes are weak evidence in this
sparse 200-gene panel.

Return only the requested JSON object. The label must be one supplied candidate.
Every evidence_genes entry must occur in cell_observed_genes and positively
support the chosen label against the baseline in the supplied pairwise evidence.

Case:
{json.dumps(case, ensure_ascii=False, separators=(",", ":"))}
"""
    # Defense in depth: metadata keys are controlled above; this check catches
    # accidental future prompt-template additions of identifier column names.
    for forbidden in FORBIDDEN_PROMPT_COLUMNS:
        if forbidden in prompt:
            raise RuntimeError(f"Forbidden prompt column name detected: {forbidden}")
    return prompt, case


def cached_qwen_request(
    args: argparse.Namespace,
    v032: ModuleType,
    cell: Any,
    pass_number: int,
    order: tuple[str, ...],
    prompt: str,
    observed_names: list[str],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    digest = v032.preview_request_hash(args.model, prompt, order, observed_names, args.seed)
    cached = cache.get(digest) if not args.no_resume else None
    if cached and cached.get("parsed_response") is not None and not cached.get("error"):
        parsed = v032.validate_response(
            cached["parsed_response"], order, set(observed_names)
        )
        return {
            "result": parsed,
            "success": True,
            "error": "",
            "latency": float(cached.get("latency_seconds", float("nan"))),
            "source": "cache",
            "request_hash": digest,
            "order": order,
        }

    result: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    latency = float("nan")
    error_message = ""
    try:
        result, raw, latency, actual_digest = v032.call_ollama(
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
            raise RuntimeError("Internal Qwen request-hash mismatch")
    except Exception as error:  # Failure is safe: the v0.31 label is retained.
        error_message = f"{type(error).__name__}: {error}"
    item = {
        "request_hash": digest,
        "version": VERSION,
        "scheme": "test",
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
    v032.append_jsonl(cache_path, item)
    cache[digest] = item
    return {
        "result": result,
        "success": result is not None and not error_message,
        "error": error_message,
        "latency": latency,
        "source": "ollama",
        "request_hash": digest,
        "order": order,
    }


def apply_extra_gates(
    gate: dict[str, Any], cell: Any, max_margin: float
) -> dict[str, Any]:
    result = dict(gate)
    reasons = list(result["gate_reasons"])
    margin_blocked = cell.margin > max_margin
    consensus = str(result["consensus_label"])
    vascular_lock_blocked = bool(
        consensus != cell.baseline_label
        and cell.baseline_label in VASCULAR_LABELS
        and consensus in VASCULAR_LABELS
    )
    passed = bool(result["gate_passed"] and not margin_blocked and not vascular_lock_blocked)
    if margin_blocked:
        reasons.append("margin_above_threshold")
    if vascular_lock_blocked:
        reasons.append("vascular_family_switch_blocked")
    if passed:
        reasons = ["passed"]
    result.update(
        {
            "margin_gate_passed": not margin_blocked,
            "vascular_lock_blocked": vascular_lock_blocked,
            "gate_passed": passed,
            "gate_reasons": reasons,
            "gated_label": consensus if passed else cell.baseline_label,
        }
    )
    return result


def run_qwen_layer(
    args: argparse.Namespace,
    v032: ModuleType,
    bundle: Any,
    probabilities: np.ndarray,
    classes: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if bundle.counts_test is None or bundle.meta_test is None:
        raise RuntimeError("Test data are unavailable")
    n_train = len(bundle.counts_train)
    selected = select_test_cells(
        v032, probabilities, classes, cell_ids, n_train, args
    )
    print(
        f"[Qwen] selected {len(selected)} cells with confidence <= "
        f"{args.max_base_confidence} and margin <= {args.max_margin}",
        flush=True,
    )
    raw_train = bundle.counts_train.to_numpy(dtype=float)
    raw_test = bundle.counts_test.to_numpy(dtype=float)
    raw_all = np.vstack([raw_train, raw_test])
    normalized_all = np.vstack(
        [
            v032.normalized_expression(bundle.counts_train),
            v032.normalized_expression(bundle.counts_test),
        ]
    )
    genes = bundle.counts_train.columns.astype(str).to_numpy()
    y_train = bundle.meta_train[TARGET].astype(str).to_numpy()
    y_all = np.concatenate(
        [y_train, np.full(len(bundle.counts_test), "__UNLABELED_TEST__", dtype=object)]
    )
    fold_ids = np.concatenate(
        [np.zeros(n_train, dtype=int), np.ones(len(bundle.counts_test), dtype=int)]
    )
    meta_all = pd.concat([bundle.meta_train, bundle.meta_test], axis=0)
    ei_all = (
        meta_all["Excitatory_vs_Inhibitory"]
        .fillna("MISSING").astype(str).to_numpy()
    )
    evidence = v032.FoldEvidenceCache(
        normalized_all,
        raw_all,
        genes,
        y_all,
        fold_ids,
        ei_all,
        args.neighbors,
        args.class_neighbors,
        args.pairwise_genes_per_direction,
        args.min_pairwise_support_score,
    )
    cache_path = args.out_dir / "v034_qwen_responses_test.jsonl"
    prompt_path = args.out_dir / "v034_qwen_prompts_test.jsonl"
    cache = {} if args.no_resume else v032.load_response_cache(cache_path)
    prompt_tmp = prompt_path.with_suffix(prompt_path.suffix + ".tmp")
    rows: list[dict[str, Any]] = []

    with prompt_tmp.open("w", encoding="utf-8") as prompt_handle:
        for number, cell in enumerate(selected, start=1):
            neighbor = evidence.neighbor_evidence(cell.row_position, cell.fold, cell.candidates)
            pairwise_profiles, pairwise_support = evidence.pairwise_evidence(
                cell.row_position, cell.fold, cell.baseline_label, cell.candidates
            )
            observed = v032.observed_gene_records(
                cell.row_position,
                raw_all,
                normalized_all,
                genes,
                args.cell_genes,
                pairwise_profiles,
            )
            observed_names = [item["gene"] for item in observed]
            requests = []
            for pass_number, order in enumerate(v032.candidate_orders(cell.candidates), start=1):
                prompt, case = build_full_train_prompt(
                    cell, order, meta_all, observed, neighbor, pairwise_profiles
                )
                digest = v032.preview_request_hash(
                    args.model, prompt, order, observed_names, args.seed
                )
                prompt_handle.write(
                    json.dumps(
                        {
                            "version": VERSION,
                            "cell_id": cell.cell_id,
                            "test_row_position": cell.row_position - n_train,
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
                    cached_qwen_request(
                        args,
                        v032,
                        cell,
                        pass_number,
                        order,
                        prompt,
                        observed_names,
                        cache,
                        cache_path,
                    )
                )
            gate = v032.gate_decision(
                cell,
                requests,
                neighbor,
                pairwise_support,
                args.min_llm_confidence,
                args.min_supported_evidence,
                "cross_family",
            )
            gate = apply_extra_gates(gate, cell, args.max_margin)
            row = asdict(cell)
            row["row_position"] = cell.row_position - n_train
            row["candidates"] = "|".join(cell.candidates)
            row["candidate_probabilities"] = "|".join(
                f"{value:.8f}" for value in cell.candidate_probabilities
            )
            row.update(
                {
                    "neighbor_candidate_label": neighbor["neighbor_candidate_label"],
                    "pairwise_support_counts_json": v032.canonical_json(
                        {label: len(values) for label, values in pairwise_support.items()}
                    ),
                    "both_requests_success": gate["both_requests_success"],
                    "candidate_order_consensus": gate["candidate_order_consensus"],
                    "consensus_label": gate["consensus_label"],
                    "minimum_confidence": gate["minimum_confidence"],
                    "common_supported_evidence_genes": "|".join(
                        gate["common_supported_evidence_genes"]
                    ),
                    "n_common_supported_evidence": gate["n_common_supported_evidence"],
                    "neighbor_agrees_with_switch": gate["neighbor_agrees_with_switch"],
                    "fine_switch_blocked": gate["fine_switch_blocked"],
                    "vascular_lock_blocked": gate["vascular_lock_blocked"],
                    "margin_gate_passed": gate["margin_gate_passed"],
                    "gate_passed": gate["gate_passed"],
                    "gate_reasons": "|".join(gate["gate_reasons"]),
                    "gated_label": gate["gated_label"],
                }
            )
            for pass_number, request in enumerate(requests, start=1):
                parsed = request["result"] if request["success"] else {}
                row.update(
                    {
                        f"pass{pass_number}_success": request["success"],
                        f"pass{pass_number}_label": parsed.get("label", ""),
                        f"pass{pass_number}_confidence": parsed.get("confidence", math.nan),
                        f"pass{pass_number}_evidence_genes": "|".join(
                            parsed.get("evidence_genes", [])
                        ),
                        f"pass{pass_number}_source": request["source"],
                        f"pass{pass_number}_latency_seconds": request["latency"],
                        f"pass{pass_number}_error": request["error"],
                        f"pass{pass_number}_request_hash": request["request_hash"],
                    }
                )
            rows.append(row)
            print(
                f"[Qwen] {number:>3}/{len(selected)} cell={cell.cell_id} "
                f"base={cell.baseline_label} final={gate['gated_label']} "
                f"pass={gate['gate_passed']}",
                flush=True,
            )
    prompt_tmp.replace(prompt_path)
    frame = pd.DataFrame(rows)
    final = classes[probabilities.argmax(axis=1)].copy()
    for row in rows:
        final[int(row["row_position"])] = str(row["gated_label"])
    summary = {
        "n_test_cells": len(final),
        "n_qwen_selected": len(selected),
        "n_both_requests_success": int(frame["both_requests_success"].sum()) if len(frame) else 0,
        "n_candidate_order_consensus": int(frame["candidate_order_consensus"].sum()) if len(frame) else 0,
        "n_gate_switches": int(frame["gate_passed"].sum()) if len(frame) else 0,
        "n_vascular_switches_blocked": int(frame["vascular_lock_blocked"].sum()) if len(frame) else 0,
    }
    return final, frame, summary


def ensure_output_safety(args: argparse.Namespace) -> None:
    completed = [
        args.submission_out,
        args.out_dir / "v034_manifest.json",
        args.out_dir / "v034_qwen_test_cells.csv",
    ]
    existing = [path for path in completed if path.exists()]
    if existing and not args.overwrite_output:
        raise FileExistsError(
            "Refusing to overwrite completed outputs: "
            + ", ".join(map(str, existing))
            + ". Use a new --out-dir or explicitly pass --overwrite-output."
        )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    v031 = load_local_module(V031_PATH, "hackathon_v031")
    v032 = load_local_module(V032_PATH, "hackathon_v032")
    bundle = v031.load_data(args.data_dir, need_test=True)
    hashes = data_hashes(args.data_dir)
    fingerprint = combined_fingerprint(hashes, args.seeds)
    plan = {
        "version": VERSION,
        "qwen_enabled": args.enable_qwen,
        "seeds": args.seeds,
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "submission_out": str(args.submission_out.resolve()),
        "n_train": len(bundle.counts_train),
        "n_test": len(bundle.counts_test),
        "n_genes": bundle.counts_train.shape[1],
        "max_qwen_cells": args.max_qwen_cells,
        "qwen_coverage": args.qwen_coverage,
        "max_base_confidence": args.max_base_confidence,
        "max_margin": args.max_margin,
        "submission_approved": False,
        "allow_legacy_object_npz": args.allow_legacy_object_npz,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("DRY RUN: no model training, Ollama call, cache write, or prediction write.")
        return

    ensure_output_safety(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probabilities, classes, cell_ids, base_source, base_path = load_or_compute_base(
        args, v031, bundle, fingerprint
    )
    base_predictions = classes[probabilities.argmax(axis=1)]
    qwen_summary = {
        "n_test_cells": len(base_predictions),
        "n_qwen_selected": 0,
        "n_both_requests_success": 0,
        "n_candidate_order_consensus": 0,
        "n_gate_switches": 0,
        "n_vascular_switches_blocked": 0,
    }
    # Keep the auxiliary output a valid, parseable CSV even when Qwen is off.
    cell_frame = pd.DataFrame(
        columns=[
            "row_position",
            "cell_id",
            "baseline_label",
            "gated_label",
            "gate_passed",
        ]
    )
    ollama_info: dict[str, Any] | None = None
    if args.enable_qwen:
        ollama_info = check_ollama_with_model_provenance(
            v032, args.ollama_url, args.model, args.timeout_seconds
        )
        final_predictions, cell_frame, qwen_summary = run_qwen_layer(
            args, v032, bundle, probabilities, classes, cell_ids
        )
    else:
        final_predictions = base_predictions
        print("Qwen disabled: returning the exact v0.31 five-model prediction.")

    submission = pd.DataFrame(
        {"Cell_ID": cell_ids, PREDICTION_COLUMN: final_predictions}
    )
    expected_ids = bundle.counts_test.index.astype(str).to_numpy()
    if not np.array_equal(submission["Cell_ID"].astype(str).to_numpy(), expected_ids):
        raise RuntimeError("Submission row order differs from counts_test.csv")
    if not submission[PREDICTION_COLUMN].isin(classes).all():
        raise RuntimeError("Submission contains an unknown label")
    atomic_csv(submission, args.submission_out)
    cell_path = args.out_dir / "v034_qwen_test_cells.csv"
    atomic_csv(cell_frame, cell_path)

    manifest = {
        **plan,
        "status": "complete",
        "completed_unix_time": time.time(),
        "submission_approved": False,
        "warning": (
            "Pretrained Qwen knowledge may count as external information; this "
            "manifest does not claim organizer approval."
        ),
        "data_sha256": hashes,
        "data_fingerprint": fingerprint,
        "source_code_sha256": {
            "v0.31": sha256_file(V031_PATH),
            "v0.32": sha256_file(V032_PATH),
            "v0.34": sha256_file(Path(__file__).resolve()),
        },
        "base_probability_source": base_source,
        "base_probability_path": str(base_path.resolve()) if base_path else None,
        "base_probability_sha256": sha256_file(base_path) if base_path else None,
        "base_probability_validation_warning": (
            None
            if base_source != "explicit_npz"
            else (
                "Explicit NPZ provenance, training seeds, and data fingerprint "
                "cannot be verified when the supplied file lacks those fields; "
                "v0.34 validated row IDs, class order, shape, finiteness, and row "
                "sums. Legacy pickle loading was explicitly enabled."
                if args.allow_legacy_object_npz
                else "Explicit artifact validated without pickle; provenance and "
                "training seeds remain the user's responsibility."
            )
        ),
        "model": args.model if args.enable_qwen else None,
        "ollama": ollama_info,
        "rules": {
            "candidate_k": 3,
            "candidate_order_passes": 2,
            "max_base_confidence": args.max_base_confidence,
            "max_margin": args.max_margin,
            "max_qwen_cells": args.max_qwen_cells,
            "qwen_coverage": args.qwen_coverage,
            "min_llm_confidence": args.min_llm_confidence,
            "min_supported_common_pairwise_genes": args.min_supported_evidence,
            "min_pairwise_support_score": args.min_pairwise_support_score,
            "neighbor_must_agree": True,
            "switch_policy": "cross_family",
            "vascular_family_lock": sorted(VASCULAR_LABELS),
        },
        "evidence_source": "official labeled training rows only",
        "prompt_metadata": list(SAFE_PROMPT_COLUMNS),
        "excluded_prompt_metadata": sorted(FORBIDDEN_PROMPT_COLUMNS),
        "qwen_summary": qwen_summary,
        "output": {
            "prediction_csv": str(args.submission_out.resolve()),
            "prediction_sha256": sha256_file(args.submission_out),
            "qwen_cells_csv": str(cell_path.resolve()),
            "qwen_response_cache": str(
                (args.out_dir / "v034_qwen_responses_test.jsonl").resolve()
            ) if args.enable_qwen else None,
        },
    }
    atomic_json(args.out_dir / "v034_manifest.json", manifest)
    print(f"Wrote {len(submission)} predictions -> {args.submission_out}")
    print(
        f"Qwen enabled={args.enable_qwen}; accepted switches="
        f"{qwen_summary['n_gate_switches']}"
    )


if __name__ == "__main__":
    main()
