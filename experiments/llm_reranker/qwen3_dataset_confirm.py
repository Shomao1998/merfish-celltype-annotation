#!/usr/bin/env python3
"""Independent Dataset-Group confirmation of the frozen v0.32 Qwen protocol.

This is a confirmatory, evaluation-only runner.  It deliberately reuses the
exact prompt construction, two candidate orders, fold-safe evidence, response
schema, and conservative gate from the reviewed v0.32 source.  A SHA-256 guard
stops the experiment if that source changes, preventing accidental protocol
drift after the confirmation sample has been chosen.

The experiment uses Dataset Group OOF probabilities only.  Selection is
label-free, balanced across Dataset folds, and excludes both:

1. every row evaluated in either v0.4 pilot CSV; and
2. every row evaluated in either v0.32 Mouse or Dataset confirmation CSV.

The default sample contains 100 new low-margin cells.  No test data or final
prediction file are read or written.  Ollama requests remain on localhost.

Default run
-----------
python src/v0.33-Qwen3-Dataset-Independent-Confirm.py

Small prompt/output check without Ollama
----------------------------------------
python src/v0.33-Qwen3-Dataset-Independent-Confirm.py \
  --dry-run --max-cells-per-scheme 6 --out-dir /private/tmp/v033_dry_run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


VERSION = "v0.33"
PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_DIR / "src" / "v0.32-Qwen3-v0.31-OOF-Reranker.py"
FROZEN_BASE_SHA256 = "1f2b0eb0ecc41845466557871a3c3b714953ddfb03edc2fb6ed6e8e80c42ddb0"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OOF_NPZ = (
    PROJECT_DIR
    / "outputs"
    / "v031_robust_noid_5model"
    / "v031_oof_dataset.npz"
)
DEFAULT_V04_DIR = (
    PROJECT_DIR
    / "generalization"
    / "outputs"
    / "v04_local_llm_oof_reranker_qwen3_8b"
)
DEFAULT_V032_DIR = (
    PROJECT_DIR
    / "generalization"
    / "outputs"
    / "v032_qwen3_v031_oof_reranker"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "generalization"
    / "outputs"
    / "v033_qwen3_dataset_independent_confirm"
)

# Frozen scientific protocol.  These are constants rather than tunable CLI
# options so this confirmation cannot silently optimize thresholds again.
MODEL = "qwen3:8b"
CANDIDATE_K = 3
MAX_BASE_CONFIDENCE = 0.80
CELL_GENES = 24
NEIGHBORS = 15
CLASS_NEIGHBORS = 5
PAIRWISE_GENES_PER_DIRECTION = 8
MIN_PAIRWISE_SUPPORT_SCORE = 0.25
MIN_LLM_CONFIDENCE = 0.80
MIN_SUPPORTED_EVIDENCE = 2
SWITCH_POLICY = "cross_family"
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen v0.32 protocol on a fresh Dataset-Group sample."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--oof-npz", type=Path, default=DEFAULT_OOF_NPZ)
    parser.add_argument("--v04-dir", type=Path, default=DEFAULT_V04_DIR)
    parser.add_argument("--v032-dir", type=Path, default=DEFAULT_V032_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-cells-per-scheme",
        type=int,
        default=100,
        help="Fresh Dataset cells to evaluate; the confirmatory default is 100.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.max_cells_per_scheme < 1:
        parser.error("--max-cells-per-scheme must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_v032() -> ModuleType:
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Frozen v0.32 source is missing: {BASE_SCRIPT}")
    actual_hash = sha256_file(BASE_SCRIPT)
    if actual_hash != FROZEN_BASE_SHA256:
        raise RuntimeError(
            "The v0.32 source changed after this confirmation protocol was "
            f"frozen. Expected SHA-256 {FROZEN_BASE_SHA256}, found {actual_hash}. "
            "Do not run until the change is audited explicitly."
        )
    spec = importlib.util.spec_from_file_location("frozen_v032_protocol", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen protocol: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Only cache metadata uses VERSION; prompt and gate logic remain untouched.
    module.VERSION = VERSION
    return module


def load_exclusion_csv(
    path: Path,
    expected_cell_ids: np.ndarray,
) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required prior evaluation CSV is missing: {path}. Confirmation "
            "must not proceed without the complete exclusion ledger."
        )
    positions: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"row_position", "cell_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns {sorted(required)}")
        for record in reader:
            position = int(record["row_position"])
            if position < 0 or position >= len(expected_cell_ids):
                raise ValueError(f"{path}: row_position {position} is out of range")
            if str(record["cell_id"]) != str(expected_cell_ids[position]):
                raise ValueError(f"{path}: cell_id mismatch at row {position}")
            positions.add(position)
    if not positions:
        raise ValueError(f"Required exclusion CSV is empty: {path}")
    return positions


def build_exclusion_ledger(
    args: argparse.Namespace,
    expected_cell_ids: np.ndarray,
) -> tuple[set[int], list[dict[str, Any]]]:
    sources = [
        args.v04_dir / "v04_llm_reranker_cells_mouse.csv",
        args.v04_dir / "v04_llm_reranker_cells_dataset.csv",
        args.v032_dir / "v032_llm_reranker_cells_mouse.csv",
        args.v032_dir / "v032_llm_reranker_cells_dataset.csv",
    ]
    union: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in sources:
        positions = load_exclusion_csv(path, expected_cell_ids)
        union.update(positions)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "n_unique_rows": len(positions),
            }
        )
    return union, records


def frozen_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Namespace expected by the reused frozen request helper."""
    return argparse.Namespace(
        model=MODEL,
        no_resume=args.no_resume,
        dry_run=args.dry_run,
        ollama_url=args.ollama_url,
        seed=SEED,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )


def evaluate_dataset(
    args: argparse.Namespace,
    base: ModuleType,
    artifact: Any,
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    y: np.ndarray,
    raw_counts: np.ndarray,
    normalized: np.ndarray,
    genes: np.ndarray,
    excluded_positions: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selected, selection_manifest = base.select_fresh_cells(
        artifact,
        CANDIDATE_K,
        args.max_cells_per_scheme,
        MAX_BASE_CONFIDENCE,
        excluded_positions,
        False,
        0,
    )
    selection_manifest["exclusion_method"] = "v04_and_v032_all_scheme_union"
    ei = (
        meta["Excitatory_vs_Inhibitory"]
        .fillna("MISSING")
        .astype(str)
        .to_numpy()
    )
    evidence_cache = base.FoldEvidenceCache(
        normalized,
        raw_counts,
        genes,
        y,
        artifact.fold_ids,
        ei,
        NEIGHBORS,
        CLASS_NEIGHBORS,
        PAIRWISE_GENES_PER_DIRECTION,
        MIN_PAIRWISE_SUPPORT_SCORE,
    )
    response_path = args.out_dir / "v033_llm_responses_dataset.jsonl"
    prompt_path = args.out_dir / "v033_llm_prompts_dataset.jsonl"
    cache = {} if args.no_resume else base.load_response_cache(response_path)
    request_args = frozen_runtime_args(args)
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
            observed = base.observed_gene_records(
                cell.row_position,
                raw_counts,
                normalized,
                genes,
                CELL_GENES,
                pairwise_profiles,
            )
            observed_names = [item["gene"] for item in observed]
            requests: list[dict[str, Any]] = []
            for pass_number, order in enumerate(
                base.candidate_orders(cell.candidates), start=1
            ):
                prompt, case = base.build_prompt(
                    cell, order, meta, observed, neighbor, pairwise_profiles
                )
                digest = base.preview_request_hash(
                    MODEL, prompt, order, observed_names, SEED
                )
                prompt_handle.write(
                    json.dumps(
                        {
                            "version": VERSION,
                            "frozen_protocol_version": "v0.32",
                            "scheme": "dataset",
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
                    base.run_request(
                        request_args,
                        "dataset",
                        cell,
                        pass_number,
                        order,
                        prompt,
                        observed_names,
                        cache,
                        response_path,
                    )
                )

            gate = base.gate_decision(
                cell,
                requests,
                neighbor,
                pairwise_support,
                MIN_LLM_CONFIDENCE,
                MIN_SUPPORTED_EVIDENCE,
                SWITCH_POLICY,
            )
            # The held-out truth is first accessed after selection, evidence,
            # both LLM calls, consensus, and all frozen gates are complete.
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
                    "neighbor_candidate_support_json": base.canonical_json(
                        neighbor["candidate_support"]
                    ),
                    "pairwise_support_counts_json": base.canonical_json(
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
                f"[dataset-confirm] {number:>3}/{len(selected)} "
                f"row={cell.row_position} base={cell.baseline_label} "
                f"consensus={gate['consensus_label']} gated={gated_label} "
                f"pass={gate['gate_passed']}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    cell_path = args.out_dir / "v033_llm_reranker_cells_dataset.csv"
    frame.to_csv(cell_path, index=False)
    baseline = artifact.classes[artifact.probabilities.argmax(axis=1)]
    gated_hybrid = baseline.copy()
    neighbor_hybrid = baseline.copy()
    selected_positions = frame["row_position"].to_numpy(dtype=int)
    gated_hybrid[selected_positions] = frame["gated_label"].astype(str).to_numpy()
    neighbor_hybrid[selected_positions] = frame[
        "neighbor_candidate_label"
    ].astype(str).to_numpy()
    baseline_accuracy = base.safe_mean(baseline == y)
    gated_accuracy = base.safe_mean(gated_hybrid == y)
    neighbor_accuracy = base.safe_mean(neighbor_hybrid == y)

    fold_rows: list[dict[str, Any]] = []
    for fold in np.unique(artifact.fold_ids):
        fold_mask = artifact.fold_ids == fold
        selected_fold = frame[frame["fold"] == fold]
        fold_rows.append(
            {
                "version": VERSION,
                "scheme": "dataset",
                "fold": int(fold),
                "held_groups": str(artifact.held_groups[np.flatnonzero(fold_mask)[0]]),
                "n_fold_cells": int(fold_mask.sum()),
                "n_selected": len(selected_fold),
                "baseline_accuracy": base.safe_mean(
                    baseline[fold_mask] == y[fold_mask]
                ),
                "gated_accuracy": base.safe_mean(
                    gated_hybrid[fold_mask] == y[fold_mask]
                ),
                "gated_delta": base.safe_mean(
                    gated_hybrid[fold_mask] == y[fold_mask]
                )
                - base.safe_mean(baseline[fold_mask] == y[fold_mask]),
                "n_gate_switches": int(selected_fold["gate_passed"].sum()),
                "n_corrected_errors": int(
                    selected_fold["corrected_error"].sum()
                ),
                "n_introduced_errors": int(
                    selected_fold["introduced_error"].sum()
                ),
                "net_corrections": int(selected_fold["corrected_error"].sum())
                - int(selected_fold["introduced_error"].sum()),
            }
        )

    corrected = int(frame["corrected_error"].sum())
    introduced = int(frame["introduced_error"].sum())
    summary = {
        "version": VERSION,
        "frozen_protocol_version": "v0.32",
        "scheme": "dataset",
        "group_column": artifact.group_column,
        "oof_npz": str(artifact.path.resolve()),
        "model": MODEL,
        "switch_policy": SWITCH_POLICY,
        "n_total_cells": len(y),
        "n_selected_fresh": len(frame),
        "coverage": len(frame) / len(y),
        "n_both_requests_success": int(frame["both_requests_success"].sum()),
        "n_candidate_order_consensus": int(
            frame["candidate_order_consensus"].sum()
        ),
        "baseline_overall_accuracy": baseline_accuracy,
        "baseline_selected_accuracy": base.safe_mean(
            frame["baseline_correct"].to_numpy(dtype=bool)
        ),
        "candidate_topk_ceiling_selected": base.safe_mean(
            frame["truth_in_candidates"].to_numpy(dtype=bool)
        ),
        "neighbor_selected_accuracy": base.safe_mean(
            frame["neighbor_correct"].to_numpy(dtype=bool)
        ),
        "neighbor_hybrid_overall_accuracy": neighbor_accuracy,
        "neighbor_hybrid_delta": neighbor_accuracy - baseline_accuracy,
        "consensus_selected_accuracy": base.safe_mean(
            frame["consensus_correct"].to_numpy(dtype=bool)
        ),
        "gated_selected_accuracy": base.safe_mean(
            frame["gated_correct"].to_numpy(dtype=bool)
        ),
        "gated_hybrid_overall_accuracy": gated_accuracy,
        "gated_hybrid_delta": gated_accuracy - baseline_accuracy,
        "n_gate_switches": int(frame["gate_passed"].sum()),
        "n_corrected_errors": corrected,
        "n_introduced_errors": introduced,
        "n_wrong_to_wrong": int(frame["wrong_to_wrong"].sum()),
        "net_corrections": corrected - introduced,
        "paired_exact_mcnemar_p": base.exact_paired_error_pvalue(
            corrected, introduced
        ),
        "n_fine_switches_blocked": int(frame["fine_switch_blocked"].sum()),
        "worst_fold_gated_delta": min(row["gated_delta"] for row in fold_rows),
        "mean_uncached_latency_seconds_per_request": base.safe_mean(
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
    base = load_frozen_v032()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts, meta = base.load_training_data(args.data_dir)
    y = meta[base.TARGET].astype(str).to_numpy()
    raw_counts = counts.to_numpy(dtype=float)
    normalized = base.normalized_expression(counts)
    genes = counts.columns.astype(str).to_numpy()
    artifact = base.load_oof_artifact(args.oof_npz, counts, meta)
    if artifact.scheme != "dataset" or artifact.group_column != "Datasets":
        raise ValueError(
            "v0.33 accepts Dataset Group OOF only; found "
            f"scheme={artifact.scheme!r}, group_column={artifact.group_column!r}"
        )

    expected_ids = counts.index.astype(str).to_numpy()
    excluded_positions, exclusion_sources = build_exclusion_ledger(
        args, expected_ids
    )
    exclusion_hash = hashlib.sha256(
        json.dumps(
            sorted(excluded_positions), separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    ollama_info: dict[str, Any] | None = None
    if not args.dry_run:
        try:
            ollama_info = base.check_ollama(
                args.ollama_url, MODEL, args.timeout_seconds
            )
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {args.ollama_url}; start `ollama serve`."
            ) from error

    manifest_path = args.out_dir / "v033_llm_reranker_manifest.json"
    manifest: dict[str, Any] = {
        "version": VERSION,
        "purpose": "independent Dataset-Group confirmation of frozen v0.32",
        "confirmation_only": True,
        "submission_approved": False,
        "created_unix_time": time.time(),
        "status": "started",
        "dry_run": args.dry_run,
        "data_dir": str(args.data_dir.resolve()),
        "oof_npz": str(args.oof_npz.resolve()),
        "oof_npz_sha256": sha256_file(args.oof_npz),
        "out_dir": str(args.out_dir.resolve()),
        "frozen_base_script": str(BASE_SCRIPT.resolve()),
        "frozen_base_sha256": FROZEN_BASE_SHA256,
        "scheme": "dataset",
        "group_column": "Datasets",
        "model": MODEL,
        "frozen_protocol": {
            "candidate_k": CANDIDATE_K,
            "max_base_confidence": MAX_BASE_CONFIDENCE,
            "cell_genes": CELL_GENES,
            "neighbors": NEIGHBORS,
            "class_neighbors": CLASS_NEIGHBORS,
            "pairwise_genes_per_direction": PAIRWISE_GENES_PER_DIRECTION,
            "min_pairwise_support_score": MIN_PAIRWISE_SUPPORT_SCORE,
            "min_llm_confidence": MIN_LLM_CONFIDENCE,
            "min_supported_evidence": MIN_SUPPORTED_EVIDENCE,
            "switch_policy": SWITCH_POLICY,
            "candidate_order_passes": 2,
            "seed": SEED,
        },
        "max_cells_per_scheme": args.max_cells_per_scheme,
        "exclusion_sources": exclusion_sources,
        "n_excluded_union": len(excluded_positions),
        "excluded_row_positions_sha256": exclusion_hash,
        "ollama": ollama_info,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary, fold_rows, selection_manifest = evaluate_dataset(
        args,
        base,
        artifact,
        counts,
        meta,
        y,
        raw_counts,
        normalized,
        genes,
        excluded_positions,
    )
    summary_path = args.out_dir / "v033_llm_reranker_summary.csv"
    fold_path = args.out_dir / "v033_llm_reranker_folds.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    manifest["status"] = "complete"
    manifest["completed_unix_time"] = time.time()
    manifest["selection"] = selection_manifest
    manifest["output_contract"] = {
        "per_cell": "v033_llm_reranker_cells_dataset.csv",
        "summary": summary_path.name,
        "folds": fold_path.name,
        "prompts_without_truth": "v033_llm_prompts_dataset.jsonl",
        "resumable_response_cache": "v033_llm_responses_dataset.jsonl",
        "manifest": manifest_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== v0.33 DATASET INDEPENDENT CONFIRMATION ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
