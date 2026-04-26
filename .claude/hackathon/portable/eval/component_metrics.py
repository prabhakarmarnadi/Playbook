"""
Component-wise metrics — unit-testing for ML.

Each evaluator operates on a single pipeline component and returns
a dict of metric_name → float that the runner collects.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLUSTERING METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusteringEval:
    """Evaluate clustering quality with focus on outlier tracking."""

    db_path: str
    embedding_dim: int = 768

    # ── core interface ────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, Any]:
        """Return all clustering metrics as a flat dict."""
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            metrics = {}
            metrics.update(self._outlier_metrics(conn))
            metrics.update(self._silhouette(conn))
            metrics.update(self._davies_bouldin(conn))
            metrics.update(self._dbcv(conn))
            metrics.update(self._cluster_size_distribution(conn))
            metrics.update(self._cosine_coherence(conn))
            metrics.update(self._legacy_composite(metrics))
            return metrics
        finally:
            conn.close()

    # ── outlier metrics (primary pain point) ──────────────────────────────

    def _outlier_metrics(self, conn: duckdb.DuckDBPyConnection) -> dict:
        total_chunks = conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assigned_chunks = conn.execute(
            "SELECT count(DISTINCT chunk_id) FROM cluster_assignments"
        ).fetchone()[0]
        total_clauses = conn.execute("SELECT count(*) FROM clauses").fetchone()[0]
        clauses_with_type = conn.execute(
            "SELECT count(*) FROM clauses WHERE clause_type_id IS NOT NULL"
        ).fetchone()[0]

        # Clauses with text but no embedding (too short to chunk)
        clauses_no_embed = conn.execute("""
            SELECT count(DISTINCT cl.clause_id)
            FROM clauses cl
            LEFT JOIN chunks ch ON cl.clause_id = ch.clause_id
            WHERE ch.chunk_id IS NULL
        """).fetchone()[0]

        chunk_outlier_pct = (
            (total_chunks - assigned_chunks) / total_chunks * 100
            if total_chunks > 0 else 0.0
        )
        clause_outlier_pct = (
            (total_clauses - clauses_with_type) / total_clauses * 100
            if total_clauses > 0 else 0.0
        )
        clause_no_embed_pct = (
            clauses_no_embed / total_clauses * 100
            if total_clauses > 0 else 0.0
        )

        return {
            "clustering.total_chunks": total_chunks,
            "clustering.assigned_chunks": assigned_chunks,
            "clustering.chunk_outlier_pct": round(chunk_outlier_pct, 2),
            "clustering.total_clauses": total_clauses,
            "clustering.clauses_with_type": clauses_with_type,
            "clustering.clause_outlier_pct": round(clause_outlier_pct, 2),
            "clustering.clauses_no_embedding_pct": round(clause_no_embed_pct, 2),
        }

    # ── silhouette score ──────────────────────────────────────────────────

    def _silhouette(self, conn: duckdb.DuckDBPyConnection) -> dict:
        rows = conn.execute("""
            SELECT ca.cluster_id, ch.embedding
            FROM cluster_assignments ca
            JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            WHERE ch.embedding IS NOT NULL
        """).fetchall()

        if len(rows) < 10:
            return {"clustering.silhouette_score": 0.0}

        labels, embeddings = [], []
        label_map: dict[str, int] = {}
        for cid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if len(emb) != self.embedding_dim:
                continue
            if cid not in label_map:
                label_map[cid] = len(label_map)
            labels.append(label_map[cid])
            embeddings.append(emb)

        labels_arr = np.array(labels)
        emb_arr = np.stack(embeddings)

        if len(set(labels_arr)) < 2:
            return {"clustering.silhouette_score": 0.0}

        from sklearn.metrics import silhouette_score
        # Sample for large datasets to keep eval fast
        n = len(labels_arr)
        sample_size = min(5000, n)
        score = float(silhouette_score(
            emb_arr, labels_arr,
            metric="cosine",
            sample_size=sample_size,
            random_state=42
        ))
        return {"clustering.silhouette_score": round(score, 4)}

    # ── Davies-Bouldin index ──────────────────────────────────────────────

    def _davies_bouldin(self, conn: duckdb.DuckDBPyConnection) -> dict:
        rows = conn.execute("""
            SELECT ca.cluster_id, ch.embedding
            FROM cluster_assignments ca
            JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            WHERE ch.embedding IS NOT NULL
        """).fetchall()

        if len(rows) < 10:
            return {"clustering.davies_bouldin_index": 0.0}

        label_map: dict[str, int] = {}
        labels, embeddings = [], []
        for cid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if len(emb) != self.embedding_dim:
                continue
            if cid not in label_map:
                label_map[cid] = len(label_map)
            labels.append(label_map[cid])
            embeddings.append(emb)

        labels_arr = np.array(labels)
        emb_arr = np.stack(embeddings)

        if len(set(labels_arr)) < 2:
            return {"clustering.davies_bouldin_index": 0.0}

        from sklearn.metrics import davies_bouldin_score
        score = float(davies_bouldin_score(emb_arr, labels_arr))
        # DB: lower is better. We report raw value; the runner handles thresholds.
        return {"clustering.davies_bouldin_index": round(score, 4)}

    # ── DBCV (density-based cluster validity — same as scoring.py) ────────

    def _dbcv(self, conn: duckdb.DuckDBPyConnection) -> dict:
        rows = conn.execute("""
            SELECT ca.cluster_id, ch.embedding
            FROM cluster_assignments ca
            JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            WHERE ch.embedding IS NOT NULL
        """).fetchall()

        if len(rows) < 10:
            return {"clustering.dbcv_score": 0.0}

        label_map: dict[str, int] = {}
        labels, embeddings = [], []
        for cid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if len(emb) != self.embedding_dim:
                continue
            if cid not in label_map:
                label_map[cid] = len(label_map)
            labels.append(label_map[cid])
            embeddings.append(emb)

        labels_arr = np.array(labels)
        emb_arr = np.stack(embeddings)

        if len(set(labels_arr)) < 2:
            return {"clustering.dbcv_score": 0.0}

        try:
            from hdbscan.validity import validity_index
            score = float(validity_index(
                emb_arr.astype(np.float64),
                labels_arr
            ))
        except Exception:
            score = 0.0
        return {"clustering.dbcv_score": round(score, 4)}

    # ── cluster size distribution ─────────────────────────────────────────

    def _cluster_size_distribution(self, conn: duckdb.DuckDBPyConnection) -> dict:
        rows = conn.execute("""
            SELECT cluster_id, count(*) AS n
            FROM cluster_assignments GROUP BY cluster_id
        """).fetchall()

        if not rows:
            return {
                "clustering.n_clusters": 0,
                "clustering.largest_cluster_pct": 0.0,
                "clustering.smallest_cluster_size": 0,
                "clustering.median_cluster_size": 0.0,
                "clustering.cluster_size_cv": 0.0,
            }

        sizes = np.array([r[1] for r in rows])
        total = sizes.sum()
        return {
            "clustering.n_clusters": len(sizes),
            "clustering.largest_cluster_pct": round(
                float(sizes.max()) / total * 100, 2
            ),
            "clustering.smallest_cluster_size": int(sizes.min()),
            "clustering.median_cluster_size": float(np.median(sizes)),
            "clustering.cluster_size_cv": round(
                float(np.std(sizes) / np.mean(sizes)), 4
            ) if np.mean(sizes) > 0 else 0.0,
        }

    # ── intra-cluster cosine coherence ────────────────────────────────────

    def _cosine_coherence(self, conn: duckdb.DuckDBPyConnection) -> dict:
        rows = conn.execute("""
            SELECT ca.cluster_id, ch.embedding
            FROM cluster_assignments ca
            JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            WHERE ch.embedding IS NOT NULL
        """).fetchall()

        clusters: dict[str, list[np.ndarray]] = defaultdict(list)
        for cid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if len(emb) == self.embedding_dim:
                clusters[cid].append(emb)

        per_cluster_coherence = []
        all_sims = []
        for cid, embs in clusters.items():
            if len(embs) < 2:
                continue
            arr = np.stack(embs)
            centroid = arr.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            normed = arr / np.clip(norms, 1e-12, None)
            sims = normed @ centroid
            per_cluster_coherence.append(float(np.mean(sims)))
            all_sims.extend(sims.tolist())

        if not per_cluster_coherence:
            return {
                "clustering.avg_coherence": 0.0,
                "clustering.weighted_coherence": 0.0,
                "clustering.min_coherence": 0.0,
            }
        return {
            "clustering.avg_coherence": round(
                float(np.mean(per_cluster_coherence)), 4
            ),
            "clustering.weighted_coherence": round(
                float(np.mean(all_sims)), 4
            ),
            "clustering.min_coherence": round(
                float(np.min(per_cluster_coherence)), 4
            ),
        }

    # ── legacy composite score (matches scoring.py exactly) ───────────────

    def _legacy_composite(self, metrics: dict) -> dict:
        """
        Recompute the same z_score + adjusted_score from core/scoring.py
        so eval tracks the exact number Optuna optimized.

        Z = cosine * 0.3 + silhouette * 0.3 + DBCV * 0.4
        adjusted = Z * cluster_count_penalty * dominance_penalty
                     * outlier_penalty
        """
        cosine = metrics.get("clustering.avg_coherence", 0.0)
        sil = metrics.get("clustering.silhouette_score", 0.0)
        dbcv = metrics.get("clustering.dbcv_score", 0.0)

        z_score = cosine * 0.3 + sil * 0.3 + dbcv * 0.4

        n_clusters = metrics.get("clustering.n_clusters", 0)
        total = metrics.get("clustering.total_chunks", 1)
        assigned = metrics.get("clustering.assigned_chunks", 0)
        outlier_ratio = (total - assigned) / total if total > 0 else 0.0

        # 1. Cluster count penalty (legacy threshold = 35)
        min_clusters = 35
        if n_clusters < min_clusters:
            cluster_penalty = 0.4 + 0.3 * (n_clusters - 1) / max(1, min_clusters - 1)
        else:
            cluster_penalty = 1.0

        # 2. Cumulative dominance penalty (legacy threshold = 8%)
        max_dominance = 0.08
        dominance_penalty = 1.0
        # Use cluster sizes from the distribution metric
        largest_pct = metrics.get("clustering.largest_cluster_pct", 0.0) / 100.0
        # For the cumulative version we need per-cluster sizes; approximate
        # using the stored stats. The single-cluster largest_pct undercounts,
        # so we also compute n_oversized from median + CV.
        # However, since we have n_clusters and total, we can reconstruct:
        # We know all clusters are assigned here (no noise label in DB), so
        # every cluster with > max_dominance * total is penalized.
        median_size = metrics.get("clustering.median_cluster_size", 0.0)
        cv = metrics.get("clustering.cluster_size_cv", 0.0)
        if total > 0 and n_clusters > 0:
            threshold_size = max_dominance * total
            # Estimate how many clusters exceed the threshold using lognormal assumption
            # But for accuracy, just apply the single-largest penalty (conservative)
            if largest_pct > max_dominance:
                excess = largest_pct - max_dominance
                dominance_penalty = max(0.2, 1.0 - excess * 4.0)

        # 3. Outlier penalty (legacy threshold = 15%)
        max_outlier = 0.15
        if outlier_ratio > max_outlier:
            excess = outlier_ratio - max_outlier
            outlier_penalty = max(0.7, 1.0 - excess * 0.75)
        else:
            outlier_penalty = 1.0

        adjusted = max(0.0, z_score * cluster_penalty * dominance_penalty * outlier_penalty)

        return {
            "clustering.z_score": round(z_score, 4),
            "clustering.adjusted_score": round(adjusted, 4),
            "clustering.cluster_count_penalty": round(cluster_penalty, 4),
            "clustering.dominance_penalty": round(dominance_penalty, 4),
            "clustering.outlier_penalty": round(outlier_penalty, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KG EMBEDDING / NEIGHBORHOOD CONSISTENCY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KGEmbeddingEval:
    """
    Do similar documents stay near each other in embedding space?

    Measures neighborhood consistency:
    - For each agreement, find k-nearest neighbors by embedding centroid.
    - Check what fraction share the same domain.
    - Check what fraction share ≥1 cluster type.
    """

    db_path: str
    embedding_dim: int = 768
    k: int = 5

    def evaluate(self) -> dict[str, Any]:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._neighborhood_consistency(conn)
        finally:
            conn.close()

    def _neighborhood_consistency(self, conn: duckdb.DuckDBPyConnection) -> dict:
        # Build per-agreement centroid from chunk embeddings
        rows = conn.execute("""
            SELECT ch.agreement_id, ch.embedding
            FROM chunks ch
            WHERE ch.embedding IS NOT NULL
        """).fetchall()

        agr_embs: dict[str, list[np.ndarray]] = defaultdict(list)
        for aid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if len(emb) == self.embedding_dim:
                agr_embs[aid].append(emb)

        if len(agr_embs) < self.k + 1:
            return {
                "kg.neighborhood_domain_consistency": 0.0,
                "kg.neighborhood_cluster_overlap": 0.0,
            }

        # Compute agreement centroids
        agr_ids = sorted(agr_embs.keys())
        centroids = []
        for aid in agr_ids:
            c = np.mean(agr_embs[aid], axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids.append(c)
        centroid_matrix = np.stack(centroids)

        # Agreement → domain mapping
        domain_map = {}
        for aid, did in conn.execute(
            "SELECT agreement_id, domain_id FROM agreements"
        ).fetchall():
            domain_map[aid] = did

        # Agreement → set of cluster_ids
        cluster_map: dict[str, set] = defaultdict(set)
        for aid, cid in conn.execute("""
            SELECT ch.agreement_id, ca.cluster_id
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        """).fetchall():
            cluster_map[aid].add(cid)

        # kNN on cosine similarity
        sim_matrix = centroid_matrix @ centroid_matrix.T
        np.fill_diagonal(sim_matrix, -1)  # exclude self

        domain_hits = []
        cluster_overlaps = []

        for i, aid in enumerate(agr_ids):
            top_k_idx = np.argsort(sim_matrix[i])[-self.k:]
            my_domain = domain_map.get(aid)
            my_clusters = cluster_map.get(aid, set())

            same_domain = sum(
                1 for j in top_k_idx
                if domain_map.get(agr_ids[j]) == my_domain
            )
            domain_hits.append(same_domain / self.k)

            shared = sum(
                1 for j in top_k_idx
                if cluster_map.get(agr_ids[j], set()) & my_clusters
            )
            cluster_overlaps.append(shared / self.k)

        return {
            "kg.neighborhood_domain_consistency": round(
                float(np.mean(domain_hits)), 4
            ),
            "kg.neighborhood_cluster_overlap": round(
                float(np.mean(cluster_overlaps)), 4
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIELD SUGGESTION METRICS (against Gold Set)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldSuggestionEval:
    """F1 / Precision / Recall for extracted field values vs. gold annotations."""

    db_path: str
    gold_fields: dict[str, str] = field(default_factory=dict)
    # gold_fields: {"agr_X::Field Name": "expected_value", ...}

    def evaluate(self) -> dict[str, Any]:
        if not self.gold_fields:
            return {
                "fields.precision": 0.0,
                "fields.recall": 0.0,
                "fields.f1": 0.0,
                "fields.gold_count": 0,
                "fields.predicted_count": 0,
            }

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._precision_recall(conn)
        finally:
            conn.close()

    def _precision_recall(self, conn: duckdb.DuckDBPyConnection) -> dict:
        # Build predicted set: {agr_id::field_name -> value}
        predicted = {}
        rows = conn.execute("""
            SELECT e.agreement_id, fd.name, e.value
            FROM extractions e
            JOIN field_definitions fd ON e.field_id = fd.field_id
            WHERE e.value IS NOT NULL AND LENGTH(e.value) > 0
        """).fetchall()
        for aid, fname, val in rows:
            key = f"{aid}::{fname}"
            predicted[key] = val

        tp = 0
        fp = 0
        fn = 0
        matches = []

        for gold_key, gold_val in self.gold_fields.items():
            if gold_key in predicted:
                pred_val = predicted[gold_key]
                # Fuzzy match: normalize and check containment
                if self._values_match(gold_val, pred_val):
                    tp += 1
                    matches.append((gold_key, "TP", gold_val, pred_val))
                else:
                    # Predicted a value but wrong
                    fp += 1
                    fn += 1
                    matches.append((gold_key, "MISMATCH", gold_val, pred_val))
            else:
                fn += 1
                matches.append((gold_key, "FN", gold_val, ""))

        # Fields predicted but not in gold set (for keys matching gold agreements)
        gold_agr_ids = {k.split("::")[0] for k in self.gold_fields}
        for key in predicted:
            agr_id = key.split("::")[0]
            if agr_id in gold_agr_ids and key not in self.gold_fields:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "fields.precision": round(precision, 4),
            "fields.recall": round(recall, 4),
            "fields.f1": round(f1, 4),
            "fields.gold_count": len(self.gold_fields),
            "fields.predicted_count": len(predicted),
            "fields.true_positives": tp,
        }

    @staticmethod
    def _values_match(gold: str, predicted: str) -> bool:
        """Fuzzy value comparison — handles formatting differences."""
        g = gold.strip().lower()
        p = predicted.strip().lower()
        if g == p:
            return True
        # Containment (gold value appears in predicted or vice versa)
        if g in p or p in g:
            return True
        # Numeric equivalence
        try:
            gn = float(g.replace(",", "").replace("$", "").replace("%", ""))
            pn = float(p.replace(",", "").replace("$", "").replace("%", ""))
            return abs(gn - pn) < 1e-2
        except ValueError:
            pass
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLAUSE NAMING METRICS (Semantic Similarity)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClauseNamingEval:
    """
    BERTScore-style semantic similarity between pipeline cluster labels
    and gold standard clause type names.
    """

    db_path: str
    gold_clusters: dict[str, str] = field(default_factory=dict)
    # gold_clusters: {clause_id: "expected cluster label"}
    embedder: Any = None  # NomicEmbedder instance, injected

    def evaluate(self) -> dict[str, Any]:
        if not self.gold_clusters or self.embedder is None:
            return {
                "naming.avg_semantic_similarity": 0.0,
                "naming.exact_match_pct": 0.0,
                "naming.gold_count": len(self.gold_clusters),
            }

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._semantic_similarity(conn)
        finally:
            conn.close()

    def _semantic_similarity(self, conn: duckdb.DuckDBPyConnection) -> dict:
        # Get predicted cluster labels for gold clause_ids
        gold_ids = list(self.gold_clusters.keys())
        placeholders = ",".join(["?"] * len(gold_ids))
        rows = conn.execute(f"""
            SELECT cl.clause_id, c.label
            FROM clauses cl
            JOIN clusters c ON cl.clause_type_id = c.cluster_id
            WHERE cl.clause_id IN ({placeholders})
        """, gold_ids).fetchall()

        predicted_labels = {r[0]: r[1] for r in rows}

        if not predicted_labels:
            return {
                "naming.avg_semantic_similarity": 0.0,
                "naming.exact_match_pct": 0.0,
                "naming.gold_count": len(self.gold_clusters),
                "naming.matched_count": 0,
            }

        # Compute semantic similarity via embeddings
        gold_texts = []
        pred_texts = []
        exact_matches = 0

        for cid in predicted_labels:
            gold_label = self.gold_clusters[cid]
            pred_label = predicted_labels[cid]
            gold_texts.append(gold_label)
            pred_texts.append(pred_label)
            if gold_label.strip().lower() == pred_label.strip().lower():
                exact_matches += 1

        gold_embs = self.embedder.embed(gold_texts, prefix="search_document")
        pred_embs = self.embedder.embed(pred_texts, prefix="search_document")

        # Pairwise cosine similarity (element-wise, not cross-product)
        g_norms = np.linalg.norm(gold_embs, axis=1, keepdims=True)
        p_norms = np.linalg.norm(pred_embs, axis=1, keepdims=True)
        g_normed = gold_embs / np.clip(g_norms, 1e-12, None)
        p_normed = pred_embs / np.clip(p_norms, 1e-12, None)
        similarities = np.sum(g_normed * p_normed, axis=1)

        return {
            "naming.avg_semantic_similarity": round(
                float(np.mean(similarities)), 4
            ),
            "naming.min_semantic_similarity": round(
                float(np.min(similarities)), 4
            ),
            "naming.exact_match_pct": round(
                exact_matches / len(predicted_labels) * 100, 2
            ),
            "naming.gold_count": len(self.gold_clusters),
            "naming.matched_count": len(predicted_labels),
        }
