#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ONTOLOGY EVOLUTION ENGINE — OPEN-WORLD STABILITY MONITORING & ADAPTATION  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Solves the fundamental problem: you never "finish" clustering.  New agreements
arrive continuously with no guarantee on type distribution.  This engine
monitors ontology health and takes corrective action automatically.

THREE SIGNALS THAT TRIGGER ACTION:
  1. Outlier rate trending up         → ontology doesn't cover new data
  2. Centroid drift                   → clusters are silently changing meaning
  3. Extraction confidence dropping   → field definitions no longer match content

THREE ACTIONS:
  1. Buffer   — park novel docs for later re-clustering
  2. Recluster — EVoC on the buffer, merge results into main ontology
  3. Split    — subdivide a drifted cluster

STABILITY METRIC:
  S = 1 - (novel_docs_30d / total_docs_30d)
    S ≥ 0.90  → Stable      — no action
    S ∈ [0.75, 0.90) → Mild drift  — recluster buffer if large enough
    S < 0.75  → Major drift — full recluster recommended

USAGE:
  # As a CLI
  python scripts/ontology_evolution.py check  data/my_account.duckdb
  python scripts/ontology_evolution.py evolve data/my_account.duckdb
  python scripts/ontology_evolution.py report data/my_account.duckdb

  # Programmatic
  from scripts.ontology_evolution import OntologyEvolutionEngine
  engine = OntologyEvolutionEngine(store, embedder)
  health = engine.compute_health()
  if health.action_needed:
      result = engine.evolve()
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ClusterHealth:
    """Per-cluster health metrics."""
    cluster_id: str
    label: str
    chunk_count: int
    centroid_drift: float          # cosine distance from original to current centroid
    mean_assignment_confidence: float  # mean cosine sim of recent assignments
    mean_extraction_confidence: float  # mean extraction confidence (from extractions table)
    outlier_rate_30d: float        # fraction of recent chunks assigned with low confidence
    field_hit_rate: float          # fraction of fields with ≥1 extraction
    status: str = "healthy"        # healthy | drifting | degraded | splitting


@dataclass
class OntologyHealth:
    """Global ontology health report."""
    timestamp: str = ""
    total_clusters: int = 0
    total_agreements: int = 0
    total_chunks: int = 0

    # ── Signal 1: Outlier rate ──
    novel_docs_30d: int = 0
    total_docs_30d: int = 0
    stability_score: float = 1.0   # S = 1 - novel/total

    # ── Signal 2: Centroid drift ──
    clusters_drifting: int = 0     # centroid shift > threshold
    max_centroid_drift: float = 0.0
    mean_centroid_drift: float = 0.0

    # ── Signal 3: Confidence drop ──
    clusters_confidence_drop: int = 0  # extraction confidence below gate
    mean_extraction_confidence: float = 0.0

    # ── Derived ──
    stability_grade: str = "stable"   # stable | mild_drift | major_drift
    action_needed: bool = False
    recommended_action: str = "none"  # none | recluster_buffer | full_recluster | split_clusters
    buffer_size: int = 0

    # ── Per-cluster detail ──
    cluster_health: list[ClusterHealth] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionResult:
    """Result of an ontology evolution action."""
    action: str                    # recluster_buffer | split_cluster | merge_clusters
    clusters_added: int = 0
    clusters_merged: int = 0
    clusters_split: int = 0
    docs_reassigned: int = 0
    buffer_cleared: int = 0
    new_fields_discovered: int = 0
    details: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═════════════════════════════════════════════════════════════════════════════


class OntologyEvolutionEngine:
    """Monitors ontology stability and performs incremental evolution."""

    # ── Thresholds ──
    STABILITY_STABLE = 0.90
    STABILITY_MILD = 0.75
    CENTROID_DRIFT_WARN = 0.15     # cosine distance
    CENTROID_DRIFT_SPLIT = 0.25    # cosine distance → recommend split
    CONFIDENCE_DROP_THRESHOLD = 0.55
    BUFFER_RECLUSTER_MIN = 200     # minimum novel docs before reclustering
    MERGE_SIMILARITY = 0.88        # cosine sim to merge new cluster into existing
    MIN_NEW_CLUSTER_DOCS = 5       # min docs for a new cluster to be "real"
    MIN_NEW_CLUSTER_AGREEMENTS = 3 # min distinct agreements

    def __init__(self, store, embedder=None, lance_store=None):
        """
        Args:
            store: ClusteringStore instance (DuckDB)
            embedder: NomicEmbedder or compatible (for re-embedding)
            lance_store: LanceVectorStore (optional, for vector updates)
        """
        self.store = store
        self.embedder = embedder
        self.lance_store = lance_store
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_counts: dict[str, int] = {}
        self._original_centroids: dict[str, np.ndarray] = {}
        self._load_centroids()

    def _load_centroids(self):
        """Load current centroids from DB."""
        try:
            rows = self.store.conn.execute("""
                SELECT cc.cluster_id, cc.centroid, cc.chunk_count
                FROM cluster_centroids cc
            """).fetchall()
            for cid, blob, count in rows:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._centroids[cid] = vec
                self._centroid_counts[cid] = count
                self._original_centroids[cid] = vec.copy()
        except Exception as e:
            logger.warning(f"Failed to load centroids: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # SIGNAL 1: Outlier Rate (Stability Score)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_stability_score(self) -> tuple[float, int, int]:
        """Compute S = 1 - (novel_docs / total_docs) over last 30 days."""
        try:
            novel = self.store.conn.execute("""
                SELECT COUNT(*) FROM novel_documents
                WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            """).fetchone()[0]
        except Exception:
            novel = self.store.conn.execute(
                "SELECT COUNT(*) FROM novel_documents"
            ).fetchone()[0]

        try:
            total_recent = self.store.conn.execute("""
                SELECT COUNT(*) FROM agreements
            """).fetchone()[0]
        except Exception:
            total_recent = 1

        # Novel docs / total docs (use total if no timestamp filtering available)
        total = max(total_recent, 1)
        s = 1.0 - (novel / total)
        return max(0.0, min(1.0, s)), novel, total

    # ─────────────────────────────────────────────────────────────────────
    # SIGNAL 2: Centroid Drift
    # ─────────────────────────────────────────────────────────────────────

    def _compute_centroid_drift(self) -> dict[str, float]:
        """Measure how far each cluster's current centroid has moved.

        Computes cosine distance (1 - cosine_sim) between the stored centroid
        and the actual mean embedding of chunks currently in the cluster.
        """
        drift_map = {}
        try:
            # For each cluster, compute the actual mean of its chunk embeddings
            rows = self.store.conn.execute("""
                SELECT ca.cluster_id, ch.embedding
                FROM cluster_assignments ca
                JOIN chunks ch ON ca.chunk_id = ch.chunk_id
                WHERE ca.cluster_id IN (SELECT cluster_id FROM cluster_centroids)
            """).fetchall()
        except Exception:
            # Fallback: just compare stored centroids to themselves (no drift)
            for cid in self._centroids:
                drift_map[cid] = 0.0
            return drift_map

        cluster_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
        for cid, blob in rows:
            if blob is not None:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                cluster_embeddings[cid].append(vec)

        for cid, stored_centroid in self._centroids.items():
            if cid in cluster_embeddings and len(cluster_embeddings[cid]) > 0:
                actual_mean = np.mean(cluster_embeddings[cid], axis=0)
                norm = np.linalg.norm(actual_mean)
                if norm > 0:
                    actual_mean = actual_mean / norm
                cos_sim = float(np.dot(stored_centroid, actual_mean))
                drift_map[cid] = 1.0 - cos_sim
            else:
                drift_map[cid] = 0.0

        return drift_map

    # ─────────────────────────────────────────────────────────────────────
    # SIGNAL 3: Extraction Confidence
    # ─────────────────────────────────────────────────────────────────────

    def _compute_extraction_confidence_per_cluster(self) -> dict[str, float]:
        """Mean extraction confidence per cluster."""
        try:
            rows = self.store.conn.execute("""
                SELECT fd.cluster_id, AVG(e.confidence) as mean_conf
                FROM extractions e
                JOIN field_definitions fd ON e.field_id = fd.field_id
                WHERE e.confidence IS NOT NULL
                GROUP BY fd.cluster_id
            """).fetchall()
            return {r[0]: float(r[1]) for r in rows}
        except Exception:
            return {}

    def _compute_field_hit_rate_per_cluster(self) -> dict[str, float]:
        """Fraction of fields with ≥1 extraction per cluster."""
        try:
            rows = self.store.conn.execute("""
                SELECT fd.cluster_id,
                       COUNT(DISTINCT CASE WHEN e.extraction_id IS NOT NULL
                             THEN fd.field_id END)::FLOAT
                       / NULLIF(COUNT(DISTINCT fd.field_id), 0) as hit_rate
                FROM field_definitions fd
                LEFT JOIN extractions e ON e.field_id = fd.field_id
                GROUP BY fd.cluster_id
            """).fetchall()
            return {r[0]: float(r[1]) if r[1] else 0.0 for r in rows}
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────────────────────
    # HEALTH REPORT
    # ─────────────────────────────────────────────────────────────────────

    def compute_health(self) -> OntologyHealth:
        """Compute comprehensive ontology health report."""
        health = OntologyHealth()

        # Basic counts
        try:
            health.total_clusters = self.store.conn.execute(
                "SELECT COUNT(*) FROM clusters").fetchone()[0]
            health.total_agreements = self.store.conn.execute(
                "SELECT COUNT(*) FROM agreements").fetchone()[0]
            health.total_chunks = self.store.conn.execute(
                "SELECT COUNT(*) FROM chunks").fetchone()[0]
        except Exception:
            pass

        # Signal 1: Stability score
        health.stability_score, health.novel_docs_30d, health.total_docs_30d = \
            self._compute_stability_score()

        # Signal 2: Centroid drift
        drift_map = self._compute_centroid_drift()
        drift_values = list(drift_map.values())
        if drift_values:
            health.max_centroid_drift = max(drift_values)
            health.mean_centroid_drift = sum(drift_values) / len(drift_values)
            health.clusters_drifting = sum(
                1 for d in drift_values if d > self.CENTROID_DRIFT_WARN)

        # Signal 3: Extraction confidence
        conf_map = self._compute_extraction_confidence_per_cluster()
        if conf_map:
            health.mean_extraction_confidence = sum(conf_map.values()) / len(conf_map)
            health.clusters_confidence_drop = sum(
                1 for c in conf_map.values() if c < self.CONFIDENCE_DROP_THRESHOLD)

        # Field hit rate
        fhr_map = self._compute_field_hit_rate_per_cluster()

        # Buffer size
        try:
            health.buffer_size = self.store.conn.execute(
                "SELECT COUNT(*) FROM novel_documents").fetchone()[0]
        except Exception:
            pass

        # Per-cluster health
        cluster_labels = {}
        cluster_chunk_counts = {}
        try:
            rows = self.store.conn.execute(
                "SELECT cluster_id, label, chunk_count FROM clusters"
            ).fetchall()
            for r in rows:
                cluster_labels[r[0]] = r[1]
                cluster_chunk_counts[r[0]] = r[2] or 0
        except Exception:
            pass

        for cid in self._centroids:
            drift = drift_map.get(cid, 0.0)
            ext_conf = conf_map.get(cid, 0.0)
            fhr = fhr_map.get(cid, 0.0)

            status = "healthy"
            if drift > self.CENTROID_DRIFT_SPLIT:
                status = "splitting"
            elif drift > self.CENTROID_DRIFT_WARN:
                status = "drifting"
            elif ext_conf > 0 and ext_conf < self.CONFIDENCE_DROP_THRESHOLD:
                status = "degraded"

            health.cluster_health.append(ClusterHealth(
                cluster_id=cid,
                label=cluster_labels.get(cid, "?"),
                chunk_count=cluster_chunk_counts.get(cid, 0),
                centroid_drift=round(drift, 4),
                mean_assignment_confidence=0.0,  # populated if assignment log exists
                mean_extraction_confidence=round(ext_conf, 3),
                outlier_rate_30d=0.0,
                field_hit_rate=round(fhr, 3),
                status=status,
            ))

        # Determine grade and recommended action
        if health.stability_score >= self.STABILITY_STABLE:
            health.stability_grade = "stable"
            health.recommended_action = "none"
        elif health.stability_score >= self.STABILITY_MILD:
            health.stability_grade = "mild_drift"
            if health.buffer_size >= self.BUFFER_RECLUSTER_MIN:
                health.action_needed = True
                health.recommended_action = "recluster_buffer"
            else:
                health.recommended_action = "monitor"
        else:
            health.stability_grade = "major_drift"
            health.action_needed = True
            health.recommended_action = "full_recluster"

        # Override: if many clusters are splitting, recommend splits
        n_splitting = sum(1 for ch in health.cluster_health if ch.status == "splitting")
        if n_splitting > 0 and health.stability_grade != "major_drift":
            health.action_needed = True
            health.recommended_action = "split_clusters"

        return health

    # ─────────────────────────────────────────────────────────────────────
    # EVOLUTION: RECLUSTER BUFFER
    # ─────────────────────────────────────────────────────────────────────

    def recluster_buffer(self) -> EvolutionResult:
        """Recluster the novel document buffer and merge results into the ontology.

        1. Load buffered novel docs + their embeddings
        2. Run EVoC clustering on just the buffer
        3. For each new cluster:
           a. If centroid cosine ≥ MERGE_SIMILARITY to existing → merge
           b. Else if ≥ MIN_NEW_CLUSTER_DOCS from ≥ MIN_NEW_CLUSTER_AGREEMENTS → add as new
           c. Else → discard (noise)
        4. Clear the buffer
        """
        result = EvolutionResult(action="recluster_buffer")

        novel_docs = self.store.get_novel_documents()
        if len(novel_docs) < self.BUFFER_RECLUSTER_MIN:
            logger.info(f"Buffer too small ({len(novel_docs)} < {self.BUFFER_RECLUSTER_MIN}), skipping")
            return result

        logger.info(f"Reclustering {len(novel_docs)} buffered novel documents")

        # Extract embeddings
        embeddings = []
        doc_ids = []
        for doc in novel_docs:
            emb = doc.get("embedding")
            if emb is not None:
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32)
                embeddings.append(emb)
                doc_ids.append(doc.get("doc_id", ""))

        if len(embeddings) < 5:
            logger.info("Too few valid embeddings in buffer, skipping")
            return result

        embeddings = np.array(embeddings)

        # Run EVoC on the buffer
        try:
            from core.evoc_pipeline import evoc_cluster
            cluster_labels_arr = evoc_cluster(
                embeddings,
                min_cluster_size=max(3, len(embeddings) // 20),
                noise_level=0.3,
            )
        except Exception as e:
            logger.warning(f"EVoC on buffer failed: {e}, falling back to KMeans")
            from sklearn.cluster import KMeans
            n_clusters = max(2, len(embeddings) // 50)
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
            cluster_labels_arr = km.fit_predict(embeddings)

        # Group by new cluster
        new_clusters: dict[int, list[int]] = defaultdict(list)
        for idx, cl in enumerate(cluster_labels_arr):
            if cl >= 0:  # skip noise (-1)
                new_clusters[cl].append(idx)

        # Build centroid matrix for existing clusters
        existing_ids = list(self._centroids.keys())
        if existing_ids:
            existing_matrix = np.stack([self._centroids[cid] for cid in existing_ids])
        else:
            existing_matrix = np.zeros((0, embeddings.shape[1]))

        merged = 0
        added = 0
        for cl_id, member_indices in new_clusters.items():
            member_embs = embeddings[member_indices]
            new_centroid = np.mean(member_embs, axis=0)
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm

            member_doc_ids = [doc_ids[i] for i in member_indices]

            # Check similarity to existing clusters
            if len(existing_ids) > 0:
                sims = existing_matrix @ new_centroid
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
            else:
                best_sim = 0.0
                best_idx = -1

            if best_sim >= self.MERGE_SIMILARITY and best_idx >= 0:
                # Merge into existing cluster
                existing_cid = existing_ids[best_idx]
                logger.info(f"  Merging new cluster ({len(member_indices)} docs) "
                           f"into '{existing_cid}' (sim={best_sim:.3f})")
                self._merge_into_existing(existing_cid, member_embs, member_doc_ids)
                merged += 1
            elif (len(member_indices) >= self.MIN_NEW_CLUSTER_DOCS and
                  len(set(member_doc_ids)) >= self.MIN_NEW_CLUSTER_AGREEMENTS):
                # Genuinely new cluster
                new_cid = f"ct_{uuid.uuid4().hex[:8]}"
                logger.info(f"  New cluster '{new_cid}' with {len(member_indices)} docs")
                self._create_new_cluster(new_cid, new_centroid, member_embs, member_doc_ids)
                added += 1
            else:
                logger.info(f"  Discarding noise cluster ({len(member_indices)} docs)")

        # Clear the buffer
        self.store.clear_novel_buffer()

        result.clusters_merged = merged
        result.clusters_added = added
        result.buffer_cleared = len(novel_docs)
        result.docs_reassigned = sum(len(m) for m in new_clusters.values())
        return result

    def _merge_into_existing(self, cluster_id: str, embeddings: np.ndarray,
                              doc_ids: list[str]):
        """Merge new documents into an existing cluster (update centroid)."""
        old_count = self._centroid_counts.get(cluster_id, 1)
        old_centroid = self._centroids[cluster_id]
        new_mean = np.mean(embeddings, axis=0)
        n_new = len(embeddings)

        # Weighted running mean
        updated = (old_count * old_centroid + n_new * new_mean) / (old_count + n_new)
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        self._centroids[cluster_id] = updated
        self._centroid_counts[cluster_id] = old_count + n_new

        # Persist updated centroid
        self.store.upsert_centroid(cluster_id, updated, old_count + n_new)

    def _create_new_cluster(self, cluster_id: str, centroid: np.ndarray,
                             embeddings: np.ndarray, doc_ids: list[str]):
        """Create a genuinely new cluster in the ontology."""
        # Insert cluster record
        self.store.conn.execute("""
            INSERT INTO clusters (cluster_id, label, description, chunk_count, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [cluster_id, f"[New] Cluster {cluster_id[:8]}", "Auto-discovered from novel buffer",
              len(embeddings)])

        # Insert centroid
        self.store.upsert_centroid(cluster_id, centroid, len(embeddings))

        # Update in-memory
        self._centroids[cluster_id] = centroid
        self._centroid_counts[cluster_id] = len(embeddings)

        logger.info(f"  Created new cluster {cluster_id} ({len(embeddings)} chunks, "
                    f"{len(set(doc_ids))} agreements)")

    # ─────────────────────────────────────────────────────────────────────
    # EVOLUTION: SPLIT DRIFTED CLUSTER
    # ─────────────────────────────────────────────────────────────────────

    def split_cluster(self, cluster_id: str, n_splits: int = 2) -> EvolutionResult:
        """Split a drifted cluster by re-running EVoC on just its members.

        Used when centroid drift exceeds CENTROID_DRIFT_SPLIT threshold,
        indicating the cluster has absorbed heterogeneous content.
        """
        result = EvolutionResult(action="split_cluster")

        # Get all chunks in this cluster
        try:
            rows = self.store.conn.execute("""
                SELECT ca.chunk_id, ch.embedding
                FROM cluster_assignments ca
                JOIN chunks ch ON ca.chunk_id = ch.chunk_id
                WHERE ca.cluster_id = ?
            """, [cluster_id]).fetchall()
        except Exception as e:
            logger.warning(f"Failed to load chunks for {cluster_id}: {e}")
            return result

        if len(rows) < 10:
            logger.info(f"Cluster {cluster_id} too small to split ({len(rows)} chunks)")
            return result

        chunk_ids = [r[0] for r in rows]
        embeddings = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows
                               if r[1] is not None])

        if len(embeddings) < 10:
            return result

        # Re-cluster
        try:
            from core.evoc_pipeline import evoc_cluster
            sub_labels = evoc_cluster(
                embeddings,
                min_cluster_size=max(3, len(embeddings) // (n_splits * 3)),
                noise_level=0.2,
            )
        except Exception:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_splits, n_init=3, random_state=42)
            sub_labels = km.fit_predict(embeddings)

        # Create sub-clusters
        sub_groups: dict[int, list[int]] = defaultdict(list)
        for idx, sl in enumerate(sub_labels):
            if sl >= 0:
                sub_groups[sl].append(idx)

        if len(sub_groups) < 2:
            logger.info(f"Split produced only {len(sub_groups)} group(s), skipping")
            return result

        # Get original label
        old_label = self.store.conn.execute(
            "SELECT label FROM clusters WHERE cluster_id = ?", [cluster_id]
        ).fetchone()
        old_label = old_label[0] if old_label else "Unknown"

        for sub_idx, (sub_id, members) in enumerate(sub_groups.items()):
            sub_embs = embeddings[members]
            sub_centroid = np.mean(sub_embs, axis=0)
            norm = np.linalg.norm(sub_centroid)
            if norm > 0:
                sub_centroid = sub_centroid / norm

            new_cid = f"ct_{uuid.uuid4().hex[:8]}"

            self.store.conn.execute("""
                INSERT INTO clusters (cluster_id, label, description, chunk_count, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [new_cid, f"{old_label} (split {sub_idx+1}/{len(sub_groups)})",
                  f"Split from {cluster_id}", len(members)])

            self.store.upsert_centroid(new_cid, sub_centroid, len(members))

            # Reassign chunks
            for midx in members:
                cid_for_chunk = chunk_ids[midx] if midx < len(chunk_ids) else None
                if cid_for_chunk:
                    try:
                        self.store.conn.execute("""
                            UPDATE cluster_assignments SET cluster_id = ?
                            WHERE chunk_id = ? AND cluster_id = ?
                        """, [new_cid, cid_for_chunk, cluster_id])
                    except Exception:
                        pass

            result.clusters_added += 1

        # Mark old cluster as split (don't delete — preserve history)
        try:
            self.store.conn.execute("""
                UPDATE clusters SET label = label || ' [SPLIT]',
                       description = 'Split into sub-clusters on ' || CURRENT_TIMESTAMP
                WHERE cluster_id = ?
            """, [cluster_id])
        except Exception:
            pass

        result.clusters_split = 1
        result.docs_reassigned = sum(len(m) for m in sub_groups.values())
        return result

    # ─────────────────────────────────────────────────────────────────────
    # ORCHESTRATOR: EVOLVE
    # ─────────────────────────────────────────────────────────────────────

    def evolve(self, health: OntologyHealth | None = None) -> list[EvolutionResult]:
        """Run the appropriate evolution action based on ontology health.

        1. Compute health if not provided
        2. Split any clusters above CENTROID_DRIFT_SPLIT
        3. If stability < STABILITY_STABLE and buffer ≥ threshold → recluster buffer
        4. Return all results
        """
        if health is None:
            health = self.compute_health()

        results = []

        # Phase 1: Split drifted clusters
        splitting = [ch for ch in health.cluster_health if ch.status == "splitting"]
        if splitting:
            logger.info(f"Splitting {len(splitting)} drifted cluster(s)")
            for ch in splitting:
                r = self.split_cluster(ch.cluster_id)
                if r.clusters_split > 0:
                    results.append(r)

        # Phase 2: Recluster buffer if stability is concerning
        if (health.stability_score < self.STABILITY_STABLE and
                health.buffer_size >= self.BUFFER_RECLUSTER_MIN):
            logger.info(f"Reclustering buffer ({health.buffer_size} docs, "
                       f"S={health.stability_score:.3f})")
            results.append(self.recluster_buffer())
        elif health.stability_score < self.STABILITY_MILD:
            logger.warning(f"Major drift detected (S={health.stability_score:.3f}). "
                          f"Full recluster recommended but not auto-triggered — "
                          f"requires manual confirmation.")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────

    def format_report(self, health: OntologyHealth) -> str:
        """Format health report as human-readable text."""
        lines = []
        def p(s=""):
            lines.append(s)

        p("=" * 80)
        p("  ONTOLOGY HEALTH REPORT")
        p(f"  {health.timestamp}")
        p("=" * 80)
        p()
        p(f"  Clusters:    {health.total_clusters}")
        p(f"  Agreements:  {health.total_agreements:,}")
        p(f"  Chunks:      {health.total_chunks:,}")
        p()
        p("  ── Signal 1: Stability Score ──")
        p(f"  S = {health.stability_score:.3f}  "
          f"({health.novel_docs_30d} novel / {health.total_docs_30d} total)")
        grade_emoji = {"stable": "✅", "mild_drift": "⚠️", "major_drift": "🔴"}
        p(f"  Grade: {health.stability_grade.upper()} "
          f"{grade_emoji.get(health.stability_grade, '')}")
        p()
        p("  ── Signal 2: Centroid Drift ──")
        p(f"  Max drift:     {health.max_centroid_drift:.4f}")
        p(f"  Mean drift:    {health.mean_centroid_drift:.4f}")
        p(f"  Drifting:      {health.clusters_drifting} clusters "
          f"(>{self.CENTROID_DRIFT_WARN})")
        p()
        p("  ── Signal 3: Extraction Confidence ──")
        p(f"  Mean:          {health.mean_extraction_confidence:.3f}")
        p(f"  Below gate:    {health.clusters_confidence_drop} clusters "
          f"(<{self.CONFIDENCE_DROP_THRESHOLD})")
        p()
        p(f"  Novel buffer:  {health.buffer_size} documents")
        p(f"  Action needed: {'YES' if health.action_needed else 'No'}")
        p(f"  Recommended:   {health.recommended_action}")
        p()

        # Per-cluster breakdown (show unhealthy first)
        unhealthy = [ch for ch in health.cluster_health if ch.status != "healthy"]
        if unhealthy:
            p("  ── Unhealthy Clusters ──")
            p(f"  {'Cluster':<40s} {'Status':<12s} {'Drift':>8s} {'Conf':>6s} {'FHR':>6s}")
            p("  " + "─" * 72)
            for ch in sorted(unhealthy, key=lambda c: -c.centroid_drift):
                p(f"  {ch.label[:40]:<40s} {ch.status:<12s} "
                  f"{ch.centroid_drift:>8.4f} "
                  f"{ch.mean_extraction_confidence:>6.3f} "
                  f"{ch.field_hit_rate:>6.3f}")
            p()

        p("=" * 80)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ontology Evolution Engine")
    parser.add_argument("command", choices=["check", "evolve", "report"],
                       help="check: health report | evolve: auto-fix | report: save JSON")
    parser.add_argument("db_path", help="Path to DuckDB database")
    parser.add_argument("--lance", default=None, help="LanceDB path")
    parser.add_argument("--output", default=None, help="Output path for report JSON")
    parser.add_argument("--force", action="store_true",
                       help="Force evolution even if health is good")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s | %(levelname)-5s | %(message)s")

    from core.store import ClusteringStore
    store = ClusteringStore(args.db_path)
    engine = OntologyEvolutionEngine(store)

    health = engine.compute_health()

    if args.command == "check":
        print(engine.format_report(health))

    elif args.command == "evolve":
        print(engine.format_report(health))
        if health.action_needed or args.force:
            print("\nRunning evolution...")
            results = engine.evolve(health)
            for r in results:
                print(f"  {r.action}: +{r.clusters_added} clusters, "
                      f"{r.clusters_merged} merged, {r.clusters_split} split, "
                      f"{r.docs_reassigned} docs reassigned")
        else:
            print("\nOntology is healthy — no evolution needed.")

    elif args.command == "report":
        report = {
            "health": asdict(health),
            "cluster_health": [asdict(ch) for ch in health.cluster_health],
        }
        # Remove the nested cluster_health from health to avoid duplication
        report["health"].pop("cluster_health", None)

        out_path = args.output or str(
            Path(args.db_path).parent / "ontology_health.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {out_path}")
        print(engine.format_report(health))


if __name__ == "__main__":
    main()
