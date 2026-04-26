"""
System-wide metrics — integration testing for insights.

Cross-component metrics that measure the pipeline as a whole:
- Coverage cascade (how many docs get full treatment end-to-end)
- Ontology completeness (domains → clusters → fields → intents chain)
- Cross-component consistency
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SystemEval:
    """End-to-end system quality metrics."""

    db_path: str

    def evaluate(self) -> dict[str, Any]:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            metrics = {}
            metrics.update(self._coverage_cascade(conn))
            metrics.update(self._ontology_completeness(conn))
            metrics.update(self._intent_coverage(conn))
            metrics.update(self._extraction_coverage(conn))
            metrics.update(self._cross_component_consistency(conn))
            return metrics
        finally:
            conn.close()

    # ── Coverage cascade ──────────────────────────────────────────────────

    def _coverage_cascade(self, conn: duckdb.DuckDBPyConnection) -> dict:
        """Track the drop-off at each pipeline stage."""
        n_agreements = conn.execute("SELECT count(*) FROM agreements").fetchone()[0]
        n_with_domain = conn.execute(
            "SELECT count(*) FROM agreements WHERE domain_id IS NOT NULL"
        ).fetchone()[0]
        n_clauses = conn.execute("SELECT count(*) FROM clauses").fetchone()[0]
        n_clauses_typed = conn.execute(
            "SELECT count(*) FROM clauses WHERE clause_type_id IS NOT NULL"
        ).fetchone()[0]
        n_with_intents = conn.execute(
            "SELECT count(DISTINCT clause_id) FROM clause_intents"
        ).fetchone()[0]
        n_with_extractions = conn.execute(
            "SELECT count(DISTINCT agreement_id) FROM extractions"
        ).fetchone()[0]

        return {
            "system.total_agreements": n_agreements,
            "system.agreements_with_domain_pct": round(
                n_with_domain / n_agreements * 100, 2
            ) if n_agreements > 0 else 0.0,
            "system.clauses_typed_pct": round(
                n_clauses_typed / n_clauses * 100, 2
            ) if n_clauses > 0 else 0.0,
            "system.clauses_with_intents_pct": round(
                n_with_intents / n_clauses * 100, 2
            ) if n_clauses > 0 else 0.0,
            "system.agreements_with_extractions_pct": round(
                n_with_extractions / n_agreements * 100, 2
            ) if n_agreements > 0 else 0.0,
        }

    # ── Ontology completeness ─────────────────────────────────────────────

    def _ontology_completeness(self, conn: duckdb.DuckDBPyConnection) -> dict:
        """Does every cluster have fields and labels?"""
        n_clusters = conn.execute("SELECT count(*) FROM clusters").fetchone()[0]
        clusters_with_fields = conn.execute("""
            SELECT count(DISTINCT cluster_id) FROM field_definitions
        """).fetchone()[0]
        clusters_with_intents = conn.execute("""
            SELECT count(DISTINCT clause_type_id) FROM clause_intents
            WHERE clause_type_id IS NOT NULL
        """).fetchone()[0]
        # Clusters with actual labels (not null/empty)
        clusters_labeled = conn.execute("""
            SELECT count(*) FROM clusters
            WHERE label IS NOT NULL AND LENGTH(label) > 0
        """).fetchone()[0]

        return {
            "system.clusters_with_fields_pct": round(
                clusters_with_fields / n_clusters * 100, 2
            ) if n_clusters > 0 else 0.0,
            "system.clusters_with_intents_pct": round(
                clusters_with_intents / n_clusters * 100, 2
            ) if n_clusters > 0 else 0.0,
            "system.clusters_labeled_pct": round(
                clusters_labeled / n_clusters * 100, 2
            ) if n_clusters > 0 else 0.0,
        }

    # ── Intent coverage ───────────────────────────────────────────────────

    def _intent_coverage(self, conn: duckdb.DuckDBPyConnection) -> dict:
        n_intent_types = conn.execute(
            "SELECT count(*) FROM intent_types"
        ).fetchone()[0]
        n_intent_instances = conn.execute(
            "SELECT count(*) FROM clause_intents"
        ).fetchone()[0]
        avg_intents_per_clause = conn.execute("""
            SELECT AVG(cnt) FROM (
                SELECT count(*) as cnt FROM clause_intents GROUP BY clause_id
            )
        """).fetchone()[0] or 0.0
        # Unique intent labels vs total — measures deduplication quality
        n_unique_labels = conn.execute(
            "SELECT count(DISTINCT intent_label) FROM clause_intents"
        ).fetchone()[0]

        return {
            "system.intent_types_discovered": n_intent_types,
            "system.intent_instances": n_intent_instances,
            "system.avg_intents_per_clause": round(float(avg_intents_per_clause), 2),
            "system.intent_dedup_ratio": round(
                n_unique_labels / n_intent_types, 4
            ) if n_intent_types > 0 else 0.0,
        }

    # ── Extraction coverage ───────────────────────────────────────────────

    def _extraction_coverage(self, conn: duckdb.DuckDBPyConnection) -> dict:
        n_fields = conn.execute(
            "SELECT count(*) FROM field_definitions"
        ).fetchone()[0]
        n_extractions = conn.execute(
            "SELECT count(*) FROM extractions"
        ).fetchone()[0]
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM extractions"
        ).fetchone()[0] or 0.0
        low_confidence = conn.execute(
            "SELECT count(*) FROM extractions WHERE confidence < 0.5"
        ).fetchone()[0]

        return {
            "system.field_definitions": n_fields,
            "system.total_extractions": n_extractions,
            "system.avg_extraction_confidence": round(float(avg_confidence), 4),
            "system.low_confidence_extraction_pct": round(
                low_confidence / n_extractions * 100, 2
            ) if n_extractions > 0 else 0.0,
        }

    # ── Cross-component consistency ───────────────────────────────────────

    def _cross_component_consistency(self, conn: duckdb.DuckDBPyConnection) -> dict:
        """
        Do clusters that share extracted fields also share intents?
        Measures coherence across pipeline stages.
        """
        # For each cluster: get set of intent labels and set of field names
        cluster_intents: dict[str, set] = defaultdict(set)
        for cid, label in conn.execute("""
            SELECT clause_type_id, intent_label
            FROM clause_intents WHERE clause_type_id IS NOT NULL
        """).fetchall():
            cluster_intents[cid].add(label)

        cluster_fields: dict[str, set] = defaultdict(set)
        for cid, name in conn.execute("""
            SELECT cluster_id, name FROM field_definitions
        """).fetchall():
            cluster_fields[cid].add(name)

        # Clusters that have BOTH intents and fields
        both = set(cluster_intents.keys()) & set(cluster_fields.keys())
        all_clusters = conn.execute("SELECT count(*) FROM clusters").fetchone()[0]

        return {
            "system.clusters_with_both_intents_and_fields_pct": round(
                len(both) / all_clusters * 100, 2
            ) if all_clusters > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HEALTH SCORE — the single "gold metric"
# ═══════════════════════════════════════════════════════════════════════════════

def compute_health_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a single 0-1 Pipeline Health Score from all collected metrics.

    This is the "gold metric" — the one number that answers
    "is the pipeline output good enough to ship?"

    Components (weighted by impact on downstream usability):
    ──────────────────────────────────────────────────────────
    Quality   (40%): Intra-cluster cosine coherence.
                     Are chunks within a cluster actually similar?
                     This is the RIGHT metric for semantic clustering
                     (unlike silhouette/DBCV which penalize non-spherical shapes).

    Coverage  (30%): What fraction of the corpus gets full pipeline treatment?
                     Weighted average of clauses typed, intents discovered,
                     extractions generated.

    Structure (20%): Is the cluster distribution healthy?
                     - Outlier health: fewer unassigned = better
                     - Dominance health: no single cluster hogs everything

    Consistency (10%): Do embedding neighbors agree with cluster assignments?
                       KG neighborhood overlap.

    Interpretation:
        0.85+ = Excellent (ship it)
        0.70+ = Good (minor issues, review flagged areas)
        0.50+ = Degraded (investigate before using)
        <0.50 = Broken (do not deploy)
    """
    # ── Quality (40%) ─────────────────────────────────────────────────
    quality = metrics.get("clustering.avg_coherence", 0.0)

    # ── Coverage (30%) ────────────────────────────────────────────────
    clauses_typed = metrics.get("system.clauses_typed_pct", 0.0) / 100.0
    intents_coverage = metrics.get("system.clauses_with_intents_pct", 0.0) / 100.0
    extraction_coverage = metrics.get(
        "system.agreements_with_extractions_pct", 0.0
    ) / 100.0
    coverage = clauses_typed * 0.4 + intents_coverage * 0.3 + extraction_coverage * 0.3

    # ── Structure (20%) ───────────────────────────────────────────────
    outlier_pct = metrics.get("clustering.chunk_outlier_pct", 0.0) / 100.0
    outlier_health = 1.0 - outlier_pct

    largest_pct = metrics.get("clustering.largest_cluster_pct", 0.0) / 100.0
    # Fully healthy if no cluster > 10%; degrades linearly to 0 at 30%
    dominance_health = min(1.0, max(0.0, 1.0 - max(0.0, largest_pct - 0.10) * 5.0))

    structure = outlier_health * 0.5 + dominance_health * 0.5

    # ── Consistency (10%) ─────────────────────────────────────────────
    consistency = metrics.get("kg.neighborhood_cluster_overlap", 0.0)

    # ── Composite ─────────────────────────────────────────────────────
    health = quality * 0.4 + coverage * 0.3 + structure * 0.2 + consistency * 0.1

    # Grade
    if health >= 0.85:
        grade = "Excellent"
    elif health >= 0.70:
        grade = "Good"
    elif health >= 0.50:
        grade = "Degraded"
    else:
        grade = "Broken"

    return {
        "health.score": round(health, 4),
        "health.grade": grade,
        "health.quality_component": round(quality, 4),
        "health.coverage_component": round(coverage, 4),
        "health.structure_component": round(structure, 4),
        "health.consistency_component": round(consistency, 4),
    }
