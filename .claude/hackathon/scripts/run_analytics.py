#!/usr/bin/env python3
"""Run all 4 post-pipeline analytics modules on a DuckDB database."""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analytics")


def main():
    parser = argparse.ArgumentParser(description="Run post-pipeline analytics")
    parser.add_argument("db_path", help="Path to refined DuckDB database")
    parser.add_argument("--skip-risk", action="store_true", help="Skip risk scoring")
    parser.add_argument("--skip-deviation", action="store_true", help="Skip clause deviation")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip field normalization")
    parser.add_argument("--skip-playbook", action="store_true", help="Skip playbook benchmarks")
    parser.add_argument("--use-llm-risk", action="store_true",
                        help="Use LLM for risk scoring (default: heuristic)")
    parser.add_argument("--min-instances", type=int, default=10,
                        help="Min clauses per cluster for deviation/playbook (default: 10)")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Risk Scoring ──
    if not args.skip_risk:
        logger.info("=" * 60)
        logger.info("STAGE 1: Risk Scoring")
        logger.info("=" * 60)
        from core.risk_scorer import RiskScorer
        llm = None
        if args.use_llm_risk:
            from core.llm_client import LLMClient
            llm = LLMClient()
        scorer = RiskScorer(args.db_path, llm=llm)
        t1 = time.time()
        cluster_risks = scorer.score_all_clusters(use_llm=args.use_llm_risk)
        agr_risks = scorer.score_all_agreements()
        print()
        print(scorer.summary())
        print(f"\n[Risk scoring: {time.time()-t1:.1f}s]")
        scorer.close()

    # ── 2. Clause Deviation Scoring ──
    if not args.skip_deviation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 2: Clause Deviation Scoring")
        logger.info("=" * 60)
        from core.clause_deviation import ClauseDeviationScorer
        dev_scorer = ClauseDeviationScorer(args.db_path, min_instances=args.min_instances)
        t2 = time.time()
        dev_results = dev_scorer.score_all()
        print()
        print(dev_scorer.summary())
        print(f"\n[Deviation scoring: {time.time()-t2:.1f}s]")
        dev_scorer.close()

    # ── 3. Field Normalization ──
    if not args.skip_normalize:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 3: Field Normalization")
        logger.info("=" * 60)
        from core.field_normalizer import FieldNormalizer
        normalizer = FieldNormalizer(args.db_path)
        t3 = time.time()
        norm_counts = normalizer.normalize_all()
        print()
        print(normalizer.summary())
        print(f"\n[Field normalization: {time.time()-t3:.1f}s]")
        normalizer.close()

    # ── 4. Playbook / Benchmark Library ──
    if not args.skip_playbook:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 4: Playbook / Benchmark Library")
        logger.info("=" * 60)
        from core.playbook import PlaybookLibrary
        playbook = PlaybookLibrary(args.db_path)
        t4 = time.time()
        benchmarks = playbook.build_benchmarks(min_instances=args.min_instances)
        print()
        print(playbook.summary())
        print(f"\n[Playbook: {time.time()-t4:.1f}s]")
        playbook.close()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"ALL ANALYTICS COMPLETE — {elapsed:.1f}s total")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
