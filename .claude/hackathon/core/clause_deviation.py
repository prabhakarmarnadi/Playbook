"""
Clause Deviation Scoring
========================
For each clause type with ≥N instances, computes the centroid embedding
and scores each individual clause by cosine distance to its centroid.

Clauses far from the centroid = non-standard / unusual language.
This enables:
  - Flagging outlier clauses for review
  - Identifying clause variants across agreements
  - Measuring standardization per clause type
"""

import json
import logging
from dataclasses import dataclass

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterDeviation:
    cluster_id: str
    cluster_label: str
    n_clauses: int
    mean_similarity: float
    std_similarity: float
    n_outliers: int
    outlier_pct: float


@dataclass
class ClauseScore:
    clause_id: str
    cluster_id: str
    agreement_id: str
    similarity_to_centroid: float
    z_score: float
    is_outlier: bool
    percentile: float


class ClauseDeviationScorer:
    """Scores each clause by distance to its cluster centroid."""

    def __init__(self, db_path: str, min_instances: int = 10,
                 outlier_threshold_z: float = 2.0, embedding_dim: int = 768):
        self.db = duckdb.connect(db_path, read_only=False)
        self.min_instances = min_instances
        self.outlier_z = outlier_threshold_z
        self.dim = embedding_dim
        self._ensure_tables()

    def _ensure_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS clause_deviations (
                clause_id VARCHAR,
                cluster_id VARCHAR,
                agreement_id VARCHAR,
                similarity_to_centroid DOUBLE,
                z_score DOUBLE,
                is_outlier BOOLEAN,
                percentile DOUBLE,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (clause_id, cluster_id)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS cluster_deviation_stats (
                cluster_id VARCHAR PRIMARY KEY,
                n_clauses INTEGER,
                mean_similarity DOUBLE,
                std_similarity DOUBLE,
                n_outliers INTEGER,
                outlier_pct DOUBLE,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def score_all(self) -> list[ClusterDeviation]:
        """Score all clauses across all qualifying clusters."""
        # Get clusters with enough instances
        clusters = self.db.execute(f"""
            SELECT c.cluster_id, c.label, c.chunk_count
            FROM clusters c
            WHERE c.chunk_count >= {self.min_instances}
            ORDER BY c.chunk_count DESC
        """).fetchall()

        logger.info(f"Scoring deviations for {len(clusters)} clusters "
                     f"(min {self.min_instances} instances)")

        results = []
        total_outliers = 0
        total_clauses = 0

        for cluster_id, label, chunk_count in clusters:
            dev = self._score_cluster(cluster_id, label)
            if dev:
                results.append(dev)
                total_outliers += dev.n_outliers
                total_clauses += dev.n_clauses

        logger.info(f"Deviation scoring complete: {total_clauses} clauses, "
                     f"{total_outliers} outliers ({100*total_outliers/max(total_clauses,1):.1f}%)")
        return results

    def _score_cluster(self, cluster_id: str, label: str) -> ClusterDeviation | None:
        """Score all clauses in a single cluster."""
        # Get all clause embeddings for this cluster
        rows = self.db.execute("""
            SELECT DISTINCT cl.clause_id, cl.agreement_id, ch.embedding
            FROM clauses cl
            JOIN chunks ch ON cl.clause_id = ch.clause_id
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            AND ch.embedding IS NOT NULL
        """, [cluster_id]).fetchall()

        if len(rows) < self.min_instances:
            return None

        # Aggregate chunk embeddings per clause (mean pool)
        clause_data = {}
        for clause_id, agreement_id, emb_blob in rows:
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if len(emb) != self.dim:
                # Try reshape for multi-chunk
                if len(emb) % self.dim == 0:
                    emb = emb.reshape(-1, self.dim).mean(axis=0)
                else:
                    continue
            if clause_id not in clause_data:
                clause_data[clause_id] = {"agreement_id": agreement_id, "embeddings": []}
            clause_data[clause_id]["embeddings"].append(emb)

        if len(clause_data) < self.min_instances:
            return None

        # Mean-pool per clause, then stack
        clause_ids = list(clause_data.keys())
        clause_embeddings = []
        agreement_ids = []
        for cid in clause_ids:
            embs = clause_data[cid]["embeddings"]
            mean_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            clause_embeddings.append(mean_emb)
            agreement_ids.append(clause_data[cid]["agreement_id"])

        matrix = np.stack(clause_embeddings)  # (n_clauses, dim)

        # Compute centroid
        centroid = matrix.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Cosine similarity to centroid
        similarities = matrix @ centroid  # (n_clauses,)

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        if std_sim < 1e-8:
            std_sim = 1e-8

        # Z-scores and outlier detection
        z_scores = (similarities - mean_sim) / std_sim
        # Outliers = clauses that are unusually FAR from centroid (low similarity = high negative z)
        is_outlier = z_scores < -self.outlier_z

        # Percentiles
        sorted_sims = np.sort(similarities)
        percentiles = np.searchsorted(sorted_sims, similarities) / len(similarities) * 100

        # Persist clause-level scores
        n_outliers = 0
        for i, clause_id in enumerate(clause_ids):
            outlier = bool(is_outlier[i])
            if outlier:
                n_outliers += 1

            self.db.execute("""
                INSERT OR REPLACE INTO clause_deviations
                (clause_id, cluster_id, agreement_id, similarity_to_centroid,
                 z_score, is_outlier, percentile)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [clause_id, cluster_id, agreement_ids[i],
                  round(float(similarities[i]), 4),
                  round(float(z_scores[i]), 4),
                  outlier,
                  round(float(percentiles[i]), 1)])

        outlier_pct = round(100 * n_outliers / len(clause_ids), 1)

        # Persist cluster-level stats
        self.db.execute("""
            INSERT OR REPLACE INTO cluster_deviation_stats
            (cluster_id, n_clauses, mean_similarity, std_similarity, n_outliers, outlier_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [cluster_id, len(clause_ids), round(mean_sim, 4),
              round(std_sim, 4), n_outliers, outlier_pct])

        logger.info(f"  {label}: {len(clause_ids)} clauses, "
                     f"mean_sim={mean_sim:.3f}, {n_outliers} outliers ({outlier_pct}%)")

        return ClusterDeviation(
            cluster_id=cluster_id, cluster_label=label,
            n_clauses=len(clause_ids), mean_similarity=round(mean_sim, 4),
            std_similarity=round(std_sim, 4), n_outliers=n_outliers,
            outlier_pct=outlier_pct,
        )

    def get_outliers(self, cluster_id: str = None, limit: int = 50) -> list[dict]:
        """Get the most deviant clauses, optionally filtered by cluster."""
        where = "WHERE cd.is_outlier = true"
        params = []
        if cluster_id:
            where += " AND cd.cluster_id = ?"
            params.append(cluster_id)

        rows = self.db.execute(f"""
            SELECT cd.clause_id, cd.cluster_id, c.label as cluster_label,
                   cd.agreement_id, a.filename,
                   cd.similarity_to_centroid, cd.z_score, cd.percentile,
                   cl.clause_title, LEFT(cl.full_text, 300) as snippet
            FROM clause_deviations cd
            JOIN clusters c ON cd.cluster_id = c.cluster_id
            JOIN agreements a ON cd.agreement_id = a.agreement_id
            JOIN clauses cl ON cd.clause_id = cl.clause_id
            {where}
            ORDER BY cd.z_score ASC
            LIMIT {limit}
        """, params).fetchall()

        return [
            {
                "clause_id": r[0], "cluster_id": r[1], "cluster_label": r[2],
                "agreement_id": r[3], "filename": r[4],
                "similarity": r[5], "z_score": r[6], "percentile": r[7],
                "clause_title": r[8], "snippet": r[9],
            }
            for r in rows
        ]

    def get_cluster_stats(self) -> list[dict]:
        """Get deviation stats for all scored clusters."""
        rows = self.db.execute("""
            SELECT cds.cluster_id, c.label, cds.n_clauses,
                   cds.mean_similarity, cds.std_similarity,
                   cds.n_outliers, cds.outlier_pct
            FROM cluster_deviation_stats cds
            JOIN clusters c ON cds.cluster_id = c.cluster_id
            ORDER BY cds.outlier_pct DESC
        """).fetchall()

        return [
            {
                "cluster_id": r[0], "label": r[1], "n_clauses": r[2],
                "mean_similarity": r[3], "std_similarity": r[4],
                "n_outliers": r[5], "outlier_pct": r[6],
            }
            for r in rows
        ]

    def summary(self) -> str:
        """Print deviation scoring summary."""
        total = self.db.execute("SELECT COUNT(*) FROM clause_deviations").fetchone()[0]
        outliers = self.db.execute("SELECT COUNT(*) FROM clause_deviations WHERE is_outlier").fetchone()[0]
        clusters = self.db.execute("SELECT COUNT(*) FROM cluster_deviation_stats").fetchone()[0]

        if total == 0:
            return "No deviation scores computed yet."

        worst = self.db.execute("""
            SELECT c.label, cds.n_clauses, cds.outlier_pct, cds.mean_similarity
            FROM cluster_deviation_stats cds
            JOIN clusters c ON cds.cluster_id = c.cluster_id
            ORDER BY cds.outlier_pct DESC
            LIMIT 10
        """).fetchall()

        lines = [f"Clause Deviation Summary ({total} clauses, {clusters} clusters)"]
        lines.append("─" * 70)
        lines.append(f"Total outliers: {outliers} ({100*outliers/max(total,1):.1f}%)")
        lines.append(f"Outlier threshold: z < -{self.outlier_z}")
        lines.append("")
        lines.append(f"{'Cluster':<40} {'Clauses':>8} {'Outlier%':>9} {'MeanSim':>8}")
        lines.append("─" * 70)
        for label, n, pct, mean in worst:
            display = label[:38] if len(label) > 38 else label
            lines.append(f"{display:<40} {n:>8} {pct:>8.1f}% {mean:>8.3f}")
        return "\n".join(lines)

    def close(self):
        self.db.close()
