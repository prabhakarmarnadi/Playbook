#!/usr/bin/env python3
"""
Bootstrap a gold_standard.json from existing pipeline output.

Exports a stratified sample of documents with their current pipeline predictions
pre-filled, so a human annotator only needs to correct — not author from scratch.

Usage:
    PYTHONPATH=. python -m eval.bootstrap data/evoc_200_refined.duckdb --n-docs 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval.bootstrap")


def bootstrap_gold_set(db_path: str, n_docs: int = 50, output: str | None = None):
    conn = duckdb.connect(db_path, read_only=True)

    # ── Stratified sampling ───────────────────────────────────────────────
    # Get cluster quality scores
    cluster_quality = {}
    for cid, qs in conn.execute(
        "SELECT cluster_id, quality_score FROM clusters WHERE quality_score IS NOT NULL"
    ).fetchall():
        cluster_quality[cid] = qs

    # Get agreement clause counts
    agr_clause_counts = {}
    for aid, cnt in conn.execute(
        "SELECT agreement_id, count(*) FROM clauses GROUP BY agreement_id"
    ).fetchall():
        agr_clause_counts[aid] = cnt

    # Pick agreements: prefer diversity
    all_agreements = conn.execute("""
        SELECT agreement_id, filename, domain_id
        FROM agreements
        ORDER BY agreement_id
    """).fetchall()

    # Simple stratification: sort by clause count, pick evenly
    sorted_agrs = sorted(all_agreements, key=lambda r: agr_clause_counts.get(r[0], 0))
    step = max(1, len(sorted_agrs) // n_docs)
    selected = [sorted_agrs[i] for i in range(0, len(sorted_agrs), step)][:n_docs]

    logger.info(f"Selected {len(selected)} agreements from {len(all_agreements)} total")

    # ── Build gold set structure ──────────────────────────────────────────
    documents = []
    cluster_gt = {}
    field_gt = {}

    for aid, filename, domain_id in selected:
        # Get domain label
        domain_label = ""
        if domain_id:
            row = conn.execute(
                "SELECT label FROM domains WHERE domain_id = ?", [domain_id]
            ).fetchone()
            if row:
                domain_label = row[0]

        # Get clauses with predictions
        clause_rows = conn.execute("""
            SELECT cl.clause_id, cl.clause_title, cl.clause_type_id,
                   c.label as cluster_label
            FROM clauses cl
            LEFT JOIN clusters c ON cl.clause_type_id = c.cluster_id
            WHERE cl.agreement_id = ?
            ORDER BY cl.clause_index
        """, [aid]).fetchall()

        clauses = []
        for cid, title, type_id, cluster_label in clause_rows:
            # Get predicted intents
            intent_rows = conn.execute("""
                SELECT intent_label, summary, party_from, party_to
                FROM clause_intents
                WHERE clause_id = ?
            """, [cid]).fetchall()

            gold_intents = [
                {
                    "label": il,
                    "summary": s or "",
                    "party_from": pf or "unspecified",
                    "party_to": pt or "unspecified",
                    "is_actionable": True,   # default to true, annotator corrects
                    "is_faithful": True,      # default to true, annotator corrects
                }
                for il, s, pf, pt in intent_rows
            ]

            # Get predicted fields for this clause's cluster
            field_rows = []
            if type_id:
                field_rows = conn.execute("""
                    SELECT fd.name, fd.field_type, e.value
                    FROM field_definitions fd
                    JOIN extractions e ON fd.field_id = e.field_id
                    WHERE fd.cluster_id = ? AND e.agreement_id = ?
                """, [type_id, aid]).fetchall()

            gold_fields = [
                {
                    "field_name": name,
                    "field_type": ftype,
                    "expected_value": val or "",
                }
                for name, ftype, val in field_rows
            ]

            clauses.append({
                "clause_id": cid,
                "clause_title": title or "",
                "gold_cluster_label": cluster_label or "",
                "gold_intents": gold_intents,
                "gold_fields": gold_fields,
            })

            if cluster_label:
                cluster_gt[cid] = cluster_label

            for gf in gold_fields:
                fkey = f"{aid}::{gf['field_name']}"
                field_gt[fkey] = gf["expected_value"]

        documents.append({
            "agreement_id": aid,
            "filename": filename or "",
            "domain_label": domain_label,
            "clauses": clauses,
        })

    conn.close()

    gold = {
        "_meta": {
            "version": "1.0",
            "created": "2026-04-03",
            "description": (
                f"Auto-bootstrapped gold set from {Path(db_path).name}. "
                f"Review and correct predictions before using as ground truth."
            ),
            "annotator": "NEEDS_REVIEW",
            "pipeline_version": Path(db_path).stem,
        },
        "documents": documents,
        "cluster_ground_truth": cluster_gt,
        "field_ground_truth": field_gt,
    }

    out_path = output or str(
        Path(db_path).parent.parent / "eval" / "gold_standard_draft.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gold, f, indent=2)

    logger.info(f"Gold set draft saved to {out_path}")
    logger.info(
        f"  {len(documents)} documents, "
        f"{len(cluster_gt)} cluster annotations, "
        f"{len(field_gt)} field annotations"
    )
    logger.info("  → Review and correct before using as ground truth!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap gold_standard.json")
    parser.add_argument("db_path", help="Path to DuckDB database")
    parser.add_argument("--n-docs", type=int, default=50, help="Number of documents")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()
    bootstrap_gold_set(args.db_path, args.n_docs, args.output)
