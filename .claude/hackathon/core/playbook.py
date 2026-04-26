"""
Playbook / Benchmark Library
=============================
Defines "standard" clause text per clause type, then scores every clause
against its benchmark. Enables:
  - Automatic deviation scoring for new clauses
  - Portfolio standardization metrics
  - Clause comparison against organizational playbook

Benchmark selection strategies:
  1. Centroid clause   — clause nearest to cluster centroid (most "average")
  2. Highest-confidence — clause from highest-confidence extraction
  3. Curated           — user-selected gold standard (manual override)
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Benchmark:
    cluster_id: str
    cluster_label: str
    benchmark_clause_id: str
    benchmark_text: str
    source: str  # "centroid" | "highest_confidence" | "curated"
    mean_deviation: float
    std_deviation: float
    n_clauses: int


class PlaybookLibrary:
    """Manages benchmark clauses and scores new clauses against them."""

    def __init__(self, db_path: str, embedding_dim: int = 768):
        self.db = duckdb.connect(db_path, read_only=False)
        self.dim = embedding_dim
        self._ensure_tables()

    def _ensure_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS playbook_benchmarks (
                cluster_id VARCHAR PRIMARY KEY,
                benchmark_clause_id VARCHAR,
                benchmark_text VARCHAR,
                benchmark_embedding BLOB,
                source VARCHAR DEFAULT 'centroid',
                mean_deviation DOUBLE,
                std_deviation DOUBLE,
                n_clauses INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS playbook_scores (
                clause_id VARCHAR,
                cluster_id VARCHAR,
                agreement_id VARCHAR,
                similarity_to_benchmark DOUBLE,
                deviation_score DOUBLE,
                deviation_label VARCHAR,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (clause_id, cluster_id)
            )
        """)

    def build_benchmarks(self, min_instances: int = 5,
                          strategy: str = "centroid") -> list[Benchmark]:
        """
        Build benchmark clauses for all qualifying clusters.
        
        strategy: "centroid" — pick clause nearest to centroid
                  "highest_confidence" — pick clause with highest avg extraction confidence
        """
        clusters = self.db.execute(f"""
            SELECT c.cluster_id, c.label, c.chunk_count
            FROM clusters c
            WHERE c.chunk_count >= {min_instances}
            ORDER BY c.chunk_count DESC
        """).fetchall()

        logger.info(f"Building benchmarks for {len(clusters)} clusters (strategy={strategy})")
        results = []

        for cluster_id, label, chunk_count in clusters:
            bench = self._build_benchmark(cluster_id, label, strategy)
            if bench:
                results.append(bench)

        logger.info(f"Built {len(results)} benchmarks")
        return results

    def _build_benchmark(self, cluster_id: str, label: str,
                          strategy: str) -> Optional[Benchmark]:
        """Build a single benchmark for a cluster."""
        # Get all clause embeddings + texts
        rows = self.db.execute("""
            SELECT DISTINCT cl.clause_id, cl.full_text, ch.embedding, cl.agreement_id
            FROM clauses cl
            JOIN chunks ch ON cl.clause_id = ch.clause_id
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            AND ch.embedding IS NOT NULL
            AND cl.full_text IS NOT NULL
        """, [cluster_id]).fetchall()

        if len(rows) < 2:
            return None

        # Aggregate per clause
        clause_data = {}
        for clause_id, full_text, emb_blob, agreement_id in rows:
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if len(emb) != self.dim:
                if len(emb) % self.dim == 0:
                    emb = emb.reshape(-1, self.dim).mean(axis=0)
                else:
                    continue
            if clause_id not in clause_data:
                clause_data[clause_id] = {
                    "text": full_text, "embeddings": [],
                    "agreement_id": agreement_id,
                }
            clause_data[clause_id]["embeddings"].append(emb)

        if len(clause_data) < 2:
            return None

        clause_ids = list(clause_data.keys())
        embeddings = []
        for cid in clause_ids:
            embs = clause_data[cid]["embeddings"]
            mean_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            embeddings.append(mean_emb)

        matrix = np.stack(embeddings)
        centroid = matrix.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        sims = matrix @ centroid

        if strategy == "centroid":
            # Pick clause nearest to centroid
            best_idx = int(np.argmax(sims))
        elif strategy == "highest_confidence":
            # Pick clause with highest avg extraction confidence
            best_idx = self._pick_highest_confidence(clause_ids) or int(np.argmax(sims))
        else:
            best_idx = int(np.argmax(sims))

        benchmark_clause_id = clause_ids[best_idx]
        benchmark_text = clause_data[benchmark_clause_id]["text"]
        benchmark_embedding = embeddings[best_idx]

        # Score all clauses against benchmark
        bench_sims = matrix @ benchmark_embedding
        mean_dev = float(np.mean(bench_sims))
        std_dev = float(np.std(bench_sims))

        # Persist benchmark
        self.db.execute("""
            INSERT OR REPLACE INTO playbook_benchmarks
            (cluster_id, benchmark_clause_id, benchmark_text, benchmark_embedding,
             source, mean_deviation, std_deviation, n_clauses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [cluster_id, benchmark_clause_id,
              benchmark_text[:5000],  # Truncate for storage
              benchmark_embedding.astype(np.float32).tobytes(),
              strategy,
              round(mean_dev, 4), round(std_dev, 4), len(clause_ids)])

        # Score each clause
        if std_dev < 1e-8:
            std_dev = 1e-8

        for i, cid in enumerate(clause_ids):
            sim = float(bench_sims[i])
            z = (sim - mean_dev) / std_dev
            # Deviation label
            if sim >= 0.95:
                label_str = "standard"
            elif sim >= 0.85:
                label_str = "minor_deviation"
            elif sim >= 0.70:
                label_str = "moderate_deviation"
            else:
                label_str = "major_deviation"

            self.db.execute("""
                INSERT OR REPLACE INTO playbook_scores
                (clause_id, cluster_id, agreement_id,
                 similarity_to_benchmark, deviation_score, deviation_label)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [cid, cluster_id, clause_data[cid]["agreement_id"],
                  round(sim, 4), round(float(z), 4), label_str])

        logger.info(f"  {label}: benchmark={benchmark_clause_id[:12]}…, "
                     f"mean_sim={mean_dev:.3f}, n={len(clause_ids)}")

        return Benchmark(
            cluster_id=cluster_id, cluster_label=label,
            benchmark_clause_id=benchmark_clause_id,
            benchmark_text=benchmark_text[:200],
            source=strategy,
            mean_deviation=round(mean_dev, 4),
            std_deviation=round(std_dev, 4),
            n_clauses=len(clause_ids),
        )

    def _pick_highest_confidence(self, clause_ids: list[str]) -> Optional[int]:
        """Pick clause with highest avg extraction confidence."""
        if not clause_ids:
            return None
        placeholders = ",".join(["?"] * len(clause_ids))
        rows = self.db.execute(f"""
            SELECT ch.clause_id, AVG(e.confidence) as avg_conf
            FROM chunks ch
            JOIN extractions e ON ch.chunk_id = e.source_chunk_id
            WHERE ch.clause_id IN ({placeholders})
            GROUP BY ch.clause_id
            ORDER BY avg_conf DESC
            LIMIT 1
        """, clause_ids).fetchall()
        if rows:
            best_clause = rows[0][0]
            if best_clause in clause_ids:
                return clause_ids.index(best_clause)
        return None

    def set_curated_benchmark(self, cluster_id: str, clause_id: str) -> bool:
        """Manually set a curated benchmark clause for a cluster type."""
        row = self.db.execute("""
            SELECT cl.full_text, ch.embedding
            FROM clauses cl
            JOIN chunks ch ON cl.clause_id = ch.clause_id
            WHERE cl.clause_id = ?
            LIMIT 1
        """, [clause_id]).fetchone()
        if not row:
            logger.error(f"Clause {clause_id} not found")
            return False

        text, emb_blob = row
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        if len(emb) != self.dim:
            if len(emb) % self.dim == 0:
                emb = emb.reshape(-1, self.dim).mean(axis=0)
            else:
                return False
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        self.db.execute("""
            INSERT OR REPLACE INTO playbook_benchmarks
            (cluster_id, benchmark_clause_id, benchmark_text, benchmark_embedding, source)
            VALUES (?, ?, ?, ?, 'curated')
        """, [cluster_id, clause_id, text[:5000], emb.astype(np.float32).tobytes()])

        logger.info(f"Set curated benchmark for {cluster_id}: {clause_id}")
        return True

    def score_clause(self, clause_text: str, cluster_id: str) -> Optional[dict]:
        """Score a single clause against its cluster benchmark."""
        row = self.db.execute("""
            SELECT benchmark_embedding, mean_deviation, std_deviation
            FROM playbook_benchmarks
            WHERE cluster_id = ?
        """, [cluster_id]).fetchone()
        if not row:
            return None

        bench_emb = np.frombuffer(row[0], dtype=np.float32)
        mean_dev, std_dev = row[1], row[2]

        # Embed the clause (import lazily to avoid circular deps)
        from core.nomic_embedder import NomicEmbedder
        embedder = NomicEmbedder.get_instance()
        clause_emb = embedder.embed_single(clause_text, prefix="search_document")

        sim = float(clause_emb @ bench_emb)
        z = (sim - mean_dev) / max(std_dev, 1e-8)

        if sim >= 0.95:
            label = "standard"
        elif sim >= 0.85:
            label = "minor_deviation"
        elif sim >= 0.70:
            label = "moderate_deviation"
        else:
            label = "major_deviation"

        return {
            "similarity_to_benchmark": round(sim, 4),
            "deviation_score": round(z, 4),
            "deviation_label": label,
            "mean_deviation": mean_dev,
            "std_deviation": std_dev,
        }

    def get_deviations(self, cluster_id: str = None,
                        label: str = None, limit: int = 50) -> list[dict]:
        """Get clauses ranked by deviation from benchmark."""
        where_parts = []
        params = []
        if cluster_id:
            where_parts.append("ps.cluster_id = ?")
            params.append(cluster_id)
        if label:
            where_parts.append("ps.deviation_label = ?")
            params.append(label)
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        rows = self.db.execute(f"""
            SELECT ps.clause_id, ps.cluster_id, c.label as cluster_label,
                   ps.agreement_id, a.filename,
                   ps.similarity_to_benchmark, ps.deviation_score, ps.deviation_label,
                   cl.clause_title, LEFT(cl.full_text, 300) as snippet
            FROM playbook_scores ps
            JOIN clusters c ON ps.cluster_id = c.cluster_id
            JOIN agreements a ON ps.agreement_id = a.agreement_id
            JOIN clauses cl ON ps.clause_id = cl.clause_id
            {where_clause}
            ORDER BY ps.similarity_to_benchmark ASC
            LIMIT {limit}
        """, params).fetchall()

        return [
            {
                "clause_id": r[0], "cluster_id": r[1], "cluster_label": r[2],
                "agreement_id": r[3], "filename": r[4],
                "similarity": r[5], "deviation_score": r[6], "deviation_label": r[7],
                "clause_title": r[8], "snippet": r[9],
            }
            for r in rows
        ]

    def get_standardization_report(self) -> list[dict]:
        """Per-cluster standardization metrics."""
        rows = self.db.execute("""
            SELECT pb.cluster_id, c.label,
                   pb.n_clauses, pb.mean_deviation, pb.std_deviation,
                   pb.source,
                   COUNT(CASE WHEN ps.deviation_label = 'standard' THEN 1 END) as standard_count,
                   COUNT(CASE WHEN ps.deviation_label = 'minor_deviation' THEN 1 END) as minor_count,
                   COUNT(CASE WHEN ps.deviation_label = 'moderate_deviation' THEN 1 END) as moderate_count,
                   COUNT(CASE WHEN ps.deviation_label = 'major_deviation' THEN 1 END) as major_count
            FROM playbook_benchmarks pb
            JOIN clusters c ON pb.cluster_id = c.cluster_id
            LEFT JOIN playbook_scores ps ON pb.cluster_id = ps.cluster_id
            GROUP BY pb.cluster_id, c.label, pb.n_clauses, pb.mean_deviation,
                     pb.std_deviation, pb.source
            ORDER BY pb.mean_deviation ASC
        """).fetchall()

        return [
            {
                "cluster_id": r[0], "label": r[1], "n_clauses": r[2],
                "mean_similarity": r[3], "std_similarity": r[4], "source": r[5],
                "standard": r[6], "minor": r[7], "moderate": r[8], "major": r[9],
                "standardization_pct": round(100 * r[6] / max(r[2], 1), 1),
            }
            for r in rows
        ]

    def summary(self) -> str:
        """Print playbook summary."""
        bench_count = self.db.execute("SELECT COUNT(*) FROM playbook_benchmarks").fetchone()[0]
        score_count = self.db.execute("SELECT COUNT(*) FROM playbook_scores").fetchone()[0]

        if bench_count == 0:
            return "No benchmarks built yet."

        labels = self.db.execute("""
            SELECT deviation_label, COUNT(*)
            FROM playbook_scores
            GROUP BY deviation_label
            ORDER BY COUNT(*) DESC
        """).fetchall()

        report = self.get_standardization_report()

        lines = [f"Playbook Library Summary ({bench_count} benchmarks, {score_count} scored clauses)"]
        lines.append("─" * 70)
        lines.append("Deviation Distribution:")
        for label, count in labels:
            pct = 100 * count / max(score_count, 1)
            bar = "█" * int(pct / 2)
            lines.append(f"  {label:<20} {count:>8} ({pct:>5.1f}%) {bar}")

        # Least standardized clusters
        lines.append("")
        lines.append(f"{'Cluster':<35} {'N':>5} {'Std%':>6} {'Major':>6} {'Source':>10}")
        lines.append("─" * 70)
        for r in sorted(report, key=lambda x: x["standardization_pct"])[:10]:
            display = r["label"][:33] if len(r["label"]) > 33 else r["label"]
            lines.append(f"{display:<35} {r['n_clauses']:>5} {r['standardization_pct']:>5.1f}% "
                        f"{r['major']:>6} {r['source']:>10}")

        return "\n".join(lines)

    def close(self):
        self.db.close()
