#!/usr/bin/env python3
"""
EVoC Pipeline Evaluation Runner
================================

Standalone evaluation pipeline that runs alongside clustering for any account.

Usage:
    # Full eval (no gold set, no LLM judge — unsupervised metrics only)
    python -m eval.runner data/evoc_200_refined.duckdb

    # With gold set
    python -m eval.runner data/evoc_200_refined.duckdb --gold eval/gold_standard.json

    # With LLM judge for intent quality
    python -m eval.runner data/evoc_200_refined.duckdb --judge --judge-sample 30

    # Compare against a baseline
    python -m eval.runner data/evoc_200_refined.duckdb --baseline eval/baseline_results.json

    # Save current run as new baseline
    python -m eval.runner data/evoc_200_refined.duckdb --save-baseline eval/baseline_results.json

    # Full eval with everything
    python -m eval.runner data/evoc_200_refined.duckdb \\
        --gold eval/gold_standard.json \\
        --baseline eval/baseline_results.json \\
        --judge --judge-sample 50 \\
        --save-baseline eval/baseline_results.json \\
        --output eval/report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent to path so eval.* and core.* both resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.component_metrics import (
    ClauseNamingEval,
    ClusteringEval,
    FieldSuggestionEval,
    KGEmbeddingEval,
)
from eval.intent_judge import IntentGoldEval, IntentJudge
from eval.regression import RegressionTracker
from eval.system_metrics import SystemEval, compute_health_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.runner")


# ═══════════════════════════════════════════════════════════════════════════════
# GOLD SET LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_gold_standard(path: str | Path) -> dict:
    """
    Load gold_standard.json and return parsed sections.

    Returns:
        {
            "cluster_ground_truth": {clause_id: label, ...},
            "field_ground_truth": {"agr_X::Field Name": "value", ...},
            "intent_ground_truth": {clause_id: [{label, summary, ...}, ...]},
        }
    """
    with open(path) as f:
        data = json.load(f)

    cluster_gt = data.get("cluster_ground_truth", {})
    field_gt = data.get("field_ground_truth", {})

    # Build intent ground truth from documents section
    intent_gt: dict[str, list[dict]] = {}
    for doc in data.get("documents", []):
        for clause in doc.get("clauses", []):
            cid = clause.get("clause_id")
            gold_intents = clause.get("gold_intents", [])
            if cid and gold_intents:
                intent_gt[cid] = gold_intents

    # Strip comment keys
    cluster_gt = {k: v for k, v in cluster_gt.items() if not k.startswith("_")}
    field_gt = {k: v for k, v in field_gt.items() if not k.startswith("_")}

    return {
        "cluster_ground_truth": cluster_gt,
        "field_ground_truth": field_gt,
        "intent_ground_truth": intent_gt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    db_path: str,
    gold_path: str | None = None,
    baseline_path: str | None = None,
    save_baseline_path: str | None = None,
    run_judge: bool = False,
    judge_sample: int = 50,
    output_path: str | None = None,
    embedding_dim: int = 768,
) -> dict[str, Any]:
    """
    Run the full evaluation suite.

    Can be called programmatically (from pipeline) or via CLI.
    Returns the full metrics dict.
    """
    t0 = time.time()
    all_metrics: dict[str, Any] = {}
    all_metrics["_meta.db_path"] = db_path
    all_metrics["_meta.timestamp"] = datetime.now(timezone.utc).isoformat()

    # Load gold set if provided
    gold = {}
    if gold_path and Path(gold_path).exists():
        gold = load_gold_standard(gold_path)
        logger.info(
            f"Gold set loaded: {len(gold['cluster_ground_truth'])} clusters, "
            f"{len(gold['field_ground_truth'])} fields, "
            f"{len(gold['intent_ground_truth'])} intent annotations"
        )

    # ── 1. Component-wise metrics (always run) ────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 1: Component-wise metrics")
    logger.info("═" * 60)

    # 1a. Clustering
    logger.info("  ▸ Clustering metrics...")
    clustering_eval = ClusteringEval(db_path=db_path, embedding_dim=embedding_dim)
    all_metrics.update(clustering_eval.evaluate())

    # 1b. KG Embedding neighborhood consistency
    logger.info("  ▸ KG embedding neighborhood consistency...")
    kg_eval = KGEmbeddingEval(db_path=db_path, embedding_dim=embedding_dim)
    all_metrics.update(kg_eval.evaluate())

    # 1c. Field suggestion (if gold set has fields)
    if gold.get("field_ground_truth"):
        logger.info("  ▸ Field suggestion F1 (vs gold set)...")
        field_eval = FieldSuggestionEval(
            db_path=db_path,
            gold_fields=gold["field_ground_truth"],
        )
        all_metrics.update(field_eval.evaluate())

    # 1d. Clause naming (if gold set has cluster labels + embedder available)
    if gold.get("cluster_ground_truth"):
        logger.info("  ▸ Clause naming semantic similarity (vs gold set)...")
        embedder = _get_embedder()
        naming_eval = ClauseNamingEval(
            db_path=db_path,
            gold_clusters=gold["cluster_ground_truth"],
            embedder=embedder,
        )
        all_metrics.update(naming_eval.evaluate())

    # ── 2. System-wide metrics (always run) ───────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 2: System-wide metrics")
    logger.info("═" * 60)

    system_eval = SystemEval(db_path=db_path)
    all_metrics.update(system_eval.evaluate())

    # Health score — the single "gold metric"
    health = compute_health_score(all_metrics)
    all_metrics.update(health)
    logger.info(
        f"  ★ Pipeline Health Score: {health['health.score']:.4f} "
        f"({health['health.grade']})"
    )

    # ── 3. Intent evaluation ──────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 3: Intent evaluation")
    logger.info("═" * 60)

    # 3a. Gold comparison (embedding-based, no LLM cost)
    if gold.get("intent_ground_truth"):
        logger.info("  ▸ Intent gold comparison (embedding similarity)...")
        embedder = _get_embedder()
        intent_gold = IntentGoldEval(
            db_path=db_path,
            gold_intents=gold["intent_ground_truth"],
            embedder=embedder,
        )
        all_metrics.update(intent_gold.evaluate())

    # 3b. LLM-as-a-Judge (opt-in, costs money)
    if run_judge:
        logger.info(f"  ▸ LLM-as-a-Judge (sampling {judge_sample} clauses)...")
        llm = _get_llm_client()
        judge = IntentJudge(
            db_path=db_path,
            llm_client=llm,
            sample_size=judge_sample,
        )
        all_metrics.update(judge.evaluate())

    # ── 4. Regression tracking ────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    all_metrics["_meta.eval_seconds"] = elapsed

    logger.info("═" * 60)
    logger.info("STAGE 4: Regression comparison")
    logger.info("═" * 60)

    tracker = RegressionTracker(baseline_path=baseline_path)
    rows = tracker.compare(all_metrics)
    report = tracker.format_markdown(rows)

    # Print to console
    print()
    print(report)

    # Save report
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    # Save metrics as JSON alongside the report
    metrics_json_path = (
        Path(output_path).with_suffix(".json")
        if output_path
        else Path(db_path).parent / "eval_results.json"
    )
    with open(metrics_json_path, "w") as f:
        json.dump(
            {k: v for k, v in all_metrics.items() if not k.startswith("_meta")},
            f, indent=2,
        )
    logger.info(f"Metrics JSON saved to {metrics_json_path}")

    # Save as new baseline if requested
    if save_baseline_path:
        tracker.save_baseline(all_metrics, save_baseline_path)

    logger.info(f"Evaluation complete in {elapsed}s")
    return all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY LOADERS (avoid importing heavy modules unless needed)
# ═══════════════════════════════════════════════════════════════════════════════

_embedder_cache = None
_llm_cache = None


def _get_embedder():
    global _embedder_cache
    if _embedder_cache is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from core.nomic_embedder import NomicEmbedder
        _embedder_cache = NomicEmbedder(device=device)
        logger.info(f"Embedder loaded: {_embedder_cache.model_name} on {device}")
    return _embedder_cache


def _get_llm_client():
    global _llm_cache
    if _llm_cache is None:
        from core.llm_client import LLMClient
        _llm_cache = LLMClient()
        logger.info(f"LLM client loaded: {_llm_cache.backend}/{_llm_cache.model}")
    return _llm_cache


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EVoC Pipeline Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick unsupervised eval
  python -m eval.runner data/evoc_200_refined.duckdb

  # Full eval with gold set and LLM judge
  python -m eval.runner data/evoc_200_refined.duckdb \\
      --gold eval/gold_standard.json \\
      --judge --judge-sample 30 \\
      --baseline eval/baseline_results.json \\
      --save-baseline eval/baseline_results.json \\
      --output eval/report.md
        """,
    )
    parser.add_argument(
        "db_path",
        help="Path to the DuckDB database to evaluate",
    )
    parser.add_argument(
        "--gold", metavar="PATH",
        help="Path to gold_standard.json for supervised metrics",
    )
    parser.add_argument(
        "--baseline", metavar="PATH",
        help="Path to baseline_results.json for regression comparison",
    )
    parser.add_argument(
        "--save-baseline", metavar="PATH",
        help="Save current results as the new baseline",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-a-Judge for intent quality (costs LLM tokens)",
    )
    parser.add_argument(
        "--judge-sample", type=int, default=50,
        help="Number of clauses to sample for LLM judge (default: 50)",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Save Markdown report to this path",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=768,
        help="Embedding dimension (default: 768 for Nomic)",
    )

    args = parser.parse_args()

    run_evaluation(
        db_path=args.db_path,
        gold_path=args.gold,
        baseline_path=args.baseline,
        save_baseline_path=args.save_baseline,
        run_judge=args.judge,
        judge_sample=args.judge_sample,
        output_path=args.output,
        embedding_dim=args.embedding_dim,
    )


if __name__ == "__main__":
    main()
