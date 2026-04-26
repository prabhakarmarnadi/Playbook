#!/usr/bin/env python3
"""
Standalone EVoC Clustering Quality Evaluator
=============================================

Computes clustering quality metrics for an EVoC-clustered DuckDB database.
No pipeline dependencies — reads directly from the DB and computes scores.

Metrics computed:
  - Silhouette Score (cosine): cluster separation [-1, 1]
  - DBCV: density-based cluster validity [-1, 1]
  - Cosine Coherence: avg intra-cluster similarity to centroid [0, 1]
  - Davies-Bouldin Index: cluster overlap (lower is better)
  - Outlier ratio: fraction of unassigned points
  - Cluster size distribution: CV, min/max/median
  - Composite Z-score: weighted combination

Usage:
    python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb
    python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb --embedding-dim 768
    python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb --json results.json
    python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb --per-cluster
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterMetrics:
    """Per-cluster quality metrics."""
    cluster_id: str
    label: str
    size: int
    coherence: float  # avg cosine sim to centroid
    silhouette_mean: float  # avg silhouette of members
    radius: float  # max distance from centroid


@dataclass
class GlobalMetrics:
    """Summary metrics for the entire clustering."""
    # Core quality scores
    silhouette_score: float
    dbcv_score: float
    davies_bouldin_index: float
    avg_coherence: float
    weighted_coherence: float  # size-weighted

    # Composite
    z_score: float  # weighted combination

    # Distribution
    n_clusters: int
    n_points: int
    n_outliers: int
    outlier_pct: float
    min_cluster_size: int
    max_cluster_size: int
    median_cluster_size: float
    cluster_size_cv: float  # coefficient of variation

    # Per-cluster breakdown (optional)
    per_cluster: list[ClusterMetrics] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_embeddings_and_labels(db_path: str, embedding_dim: int = 768) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load chunk embeddings and cluster assignments from DuckDB.

    Returns:
        (embeddings, labels, cluster_info)
        - embeddings: (N, dim) float32 array
        - labels: (N,) int array (-1 for outliers)
        - cluster_info: {cluster_id: {"label": str, "idx": int}}
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # Get chunks with embeddings
        chunks = conn.execute("""
            SELECT ch.chunk_id, ch.embedding, ca.cluster_id
            FROM chunks ch
            LEFT JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ch.embedding IS NOT NULL
            ORDER BY ch.chunk_id
        """).fetchall()

        if not chunks:
            raise ValueError("No chunks with embeddings found in database")

        # Get cluster labels
        clusters = conn.execute("""
            SELECT cluster_id, label FROM clusters
        """).fetchall()
        cluster_labels = {cid: label for cid, label in clusters}

        # Build arrays
        embeddings = []
        labels = []
        cluster_id_to_idx = {}
        next_idx = 0

        for chunk_id, emb_blob, cluster_id in chunks:
            if emb_blob is None:
                continue

            # Decode embedding from blob
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if len(emb) != embedding_dim:
                # Try float64
                emb = np.frombuffer(emb_blob, dtype=np.float64).astype(np.float32)
            if len(emb) != embedding_dim:
                continue

            embeddings.append(emb)

            if cluster_id is None:
                labels.append(-1)
            else:
                if cluster_id not in cluster_id_to_idx:
                    cluster_id_to_idx[cluster_id] = next_idx
                    next_idx += 1
                labels.append(cluster_id_to_idx[cluster_id])

        cluster_info = {}
        for cid, idx in cluster_id_to_idx.items():
            cluster_info[cid] = {
                "label": cluster_labels.get(cid, f"Cluster_{idx}"),
                "idx": idx,
            }

        return (
            np.array(embeddings, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            cluster_info,
        )
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute cosine silhouette score (excluding outliers).
    
    Returns: (global_score, per_sample_scores)
    """
    mask = labels != -1
    unique = set(labels[mask])
    if len(unique) < 2 or mask.sum() < 3:
        return 0.0, np.zeros(len(labels))

    sample_scores = np.zeros(len(labels))
    try:
        n_samples = min(5000, int(mask.sum()))
        global_score = silhouette_score(
            embeddings[mask], labels[mask],
            metric="cosine", sample_size=n_samples, random_state=42,
        )
        per_sample = silhouette_samples(
            embeddings[mask], labels[mask], metric="cosine",
        )
        sample_scores[mask] = per_sample
        return float(global_score), sample_scores
    except Exception:
        return 0.0, sample_scores


def compute_dbcv(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute DBCV (density-based cluster validity index).
    
    DBCV measures how well clusters define density-connected regions.
    Range: [-1, 1], higher is better.
    """
    mask = labels != -1
    unique = set(labels[mask])
    if len(unique) < 2 or mask.sum() < 10:
        return 0.0

    try:
        from hdbscan.validity import validity_index
        score = validity_index(
            embeddings[mask].astype(np.float64),
            labels[mask].astype(np.int32),
        )
        return float(score)
    except ImportError:
        print("  [warn] hdbscan not installed — DBCV unavailable")
        return 0.0
    except Exception as e:
        print(f"  [warn] DBCV computation failed: {e}")
        return 0.0


def compute_davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower is better, 0 = perfect)."""
    mask = labels != -1
    unique = set(labels[mask])
    if len(unique) < 2:
        return float("inf")

    try:
        return float(davies_bouldin_score(embeddings[mask], labels[mask]))
    except Exception:
        return float("inf")


def compute_coherence(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Per-cluster cosine coherence (avg cosine similarity to centroid)."""
    coherence = {}
    for tid in set(labels):
        if tid == -1:
            continue
        mask = labels == tid
        cluster_embs = embeddings[mask]
        if len(cluster_embs) < 2:
            coherence[tid] = 1.0
            continue

        centroid = cluster_embs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-12:
            coherence[tid] = 0.0
            continue

        centroid_unit = centroid / centroid_norm
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        normed = cluster_embs / (norms + 1e-12)
        sims = normed @ centroid_unit
        coherence[tid] = float(np.mean(sims))

    return coherence


def compute_radius(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Per-cluster radius (max cosine distance from centroid)."""
    radii = {}
    for tid in set(labels):
        if tid == -1:
            continue
        mask = labels == tid
        cluster_embs = embeddings[mask]
        if len(cluster_embs) < 2:
            radii[tid] = 0.0
            continue

        centroid = cluster_embs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-12:
            radii[tid] = 1.0
            continue

        centroid_unit = centroid / centroid_norm
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        normed = cluster_embs / (norms + 1e-12)
        sims = normed @ centroid_unit
        # Cosine distance = 1 - cosine similarity
        max_dist = float(1.0 - np.min(sims))
        radii[tid] = max_dist

    return radii


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_clustering(
    db_path: str,
    embedding_dim: int = 768,
    compute_per_cluster: bool = False,
) -> GlobalMetrics:
    """Run full clustering quality evaluation."""

    print(f"Loading embeddings from {db_path}...")
    embeddings, labels, cluster_info = load_embeddings_and_labels(db_path, embedding_dim)
    print(f"  {len(embeddings)} points, {len(cluster_info)} clusters, "
          f"{int(np.sum(labels == -1))} outliers")

    # Cluster sizes
    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        raise ValueError("No assigned clusters found — nothing to evaluate")

    sizes = np.bincount(valid_labels)
    sizes = sizes[sizes > 0]  # remove empty bins

    n_clusters = len(sizes)
    n_points = len(labels)
    n_outliers = int(np.sum(labels == -1))

    print(f"\nComputing metrics...")
    t0 = time.time()

    # 1. Silhouette
    print("  [1/4] Silhouette score (cosine)...")
    sil_score, sil_samples = compute_silhouette(embeddings, labels)

    # 2. DBCV
    print("  [2/4] DBCV (density-based validity)...")
    dbcv = compute_dbcv(embeddings, labels)

    # 3. Davies-Bouldin
    print("  [3/4] Davies-Bouldin index...")
    dbi = compute_davies_bouldin(embeddings, labels)

    # 4. Coherence
    print("  [4/4] Cosine coherence...")
    coherence_per_cluster = compute_coherence(embeddings, labels)
    avg_coherence = float(np.mean(list(coherence_per_cluster.values()))) if coherence_per_cluster else 0.0

    # Weighted coherence (by cluster size)
    total_assigned = n_points - n_outliers
    weighted_coh = 0.0
    for tid, coh in coherence_per_cluster.items():
        cluster_size = int(np.sum(labels == tid))
        weighted_coh += coh * (cluster_size / total_assigned)

    # Composite Z-score (weighted combination)
    z_score = (
        avg_coherence * 0.3
        + sil_score * 0.3
        + dbcv * 0.4
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Per-cluster breakdown
    per_cluster_metrics = []
    if compute_per_cluster:
        radii = compute_radius(embeddings, labels)
        idx_to_cid = {info["idx"]: cid for cid, info in cluster_info.items()}

        for tid in sorted(set(labels)):
            if tid == -1:
                continue
            cid = idx_to_cid.get(tid, f"cluster_{tid}")
            info = cluster_info.get(cid, {"label": f"Cluster_{tid}"})
            mask = labels == tid

            # Per-cluster silhouette
            cluster_sil = float(np.mean(sil_samples[mask])) if mask.sum() > 0 else 0.0

            per_cluster_metrics.append(ClusterMetrics(
                cluster_id=cid,
                label=info["label"],
                size=int(mask.sum()),
                coherence=coherence_per_cluster.get(tid, 0.0),
                silhouette_mean=cluster_sil,
                radius=radii.get(tid, 0.0),
            ))

        # Sort by coherence (best first)
        per_cluster_metrics.sort(key=lambda x: x.coherence, reverse=True)

    return GlobalMetrics(
        silhouette_score=sil_score,
        dbcv_score=dbcv,
        davies_bouldin_index=dbi,
        avg_coherence=avg_coherence,
        weighted_coherence=weighted_coh,
        z_score=z_score,
        n_clusters=n_clusters,
        n_points=n_points,
        n_outliers=n_outliers,
        outlier_pct=round(100.0 * n_outliers / n_points, 2) if n_points > 0 else 0.0,
        min_cluster_size=int(sizes.min()),
        max_cluster_size=int(sizes.max()),
        median_cluster_size=float(np.median(sizes)),
        cluster_size_cv=float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0.0,
        per_cluster=per_cluster_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: GlobalMetrics):
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 64)
    print("  EVoC CLUSTERING QUALITY REPORT")
    print("=" * 64)

    # Grade the clustering
    grade = _grade(metrics)

    print(f"\n  Grade: {grade}")
    print(f"  Composite Z-Score: {metrics.z_score:.4f}")
    print()

    print("  ┌─────────────────────────────────┬──────────┬─────────────────────┐")
    print("  │ Metric                          │ Value    │ Interpretation      │")
    print("  ├─────────────────────────────────┼──────────┼─────────────────────┤")
    print(f"  │ Silhouette Score (cosine)       │ {metrics.silhouette_score:>8.4f} │ {_interpret_sil(metrics.silhouette_score):<19} │")
    print(f"  │ DBCV Score                      │ {metrics.dbcv_score:>8.4f} │ {_interpret_dbcv(metrics.dbcv_score):<19} │")
    print(f"  │ Davies-Bouldin Index            │ {metrics.davies_bouldin_index:>8.4f} │ {_interpret_dbi(metrics.davies_bouldin_index):<19} │")
    print(f"  │ Avg Coherence                   │ {metrics.avg_coherence:>8.4f} │ {_interpret_coh(metrics.avg_coherence):<19} │")
    print(f"  │ Weighted Coherence              │ {metrics.weighted_coherence:>8.4f} │ {_interpret_coh(metrics.weighted_coherence):<19} │")
    print("  └─────────────────────────────────┴──────────┴─────────────────────┘")

    print(f"\n  Distribution:")
    print(f"    Clusters:         {metrics.n_clusters}")
    print(f"    Points:           {metrics.n_points}")
    print(f"    Outliers:         {metrics.n_outliers} ({metrics.outlier_pct:.1f}%)")
    print(f"    Size range:       [{metrics.min_cluster_size}, {metrics.max_cluster_size}]")
    print(f"    Median size:      {metrics.median_cluster_size:.0f}")
    print(f"    Size CV:          {metrics.cluster_size_cv:.3f}")

    if metrics.per_cluster:
        print(f"\n  Per-Cluster Breakdown (top 15 / {len(metrics.per_cluster)}):")
        print("  ┌────┬────────────────────────────────┬──────┬──────────┬──────────┬────────┐")
        print("  │  # │ Label                          │ Size │ Coherence│Silhouette│ Radius │")
        print("  ├────┼────────────────────────────────┼──────┼──────────┼──────────┼────────┤")
        for i, cm in enumerate(metrics.per_cluster[:15]):
            label_disp = cm.label[:28] if len(cm.label) <= 28 else cm.label[:25] + "..."
            print(f"  │{i+1:>3} │ {label_disp:<30} │ {cm.size:>4} │ {cm.coherence:>8.4f} │ {cm.silhouette_mean:>8.4f} │ {cm.radius:>6.4f} │")
        print("  └────┴────────────────────────────────┴──────┴──────────┴──────────┴────────┘")

        # Worst clusters
        worst = sorted(metrics.per_cluster, key=lambda x: x.coherence)[:5]
        print(f"\n  Worst 5 Clusters (by coherence):")
        for cm in worst:
            print(f"    {cm.label[:40]:<40} coh={cm.coherence:.4f} sil={cm.silhouette_mean:.4f} size={cm.size}")

    print("\n" + "=" * 64)


def _grade(m: GlobalMetrics) -> str:
    """Assign a letter grade based on composite metrics."""
    # Weighted formula
    score = (
        m.silhouette_score * 25
        + m.dbcv_score * 20
        + m.avg_coherence * 30
        + (1.0 - min(m.outlier_pct / 100.0, 1.0)) * 15
        + (1.0 - min(m.davies_bouldin_index / 5.0, 1.0)) * 10
    )
    if score >= 80:
        return "A — Excellent"
    elif score >= 65:
        return "B — Good"
    elif score >= 50:
        return "C — Acceptable"
    elif score >= 35:
        return "D — Poor"
    else:
        return "F — Bad"


def _interpret_sil(v: float) -> str:
    if v > 0.5: return "strong structure"
    if v > 0.25: return "reasonable"
    if v > 0.05: return "weak structure"
    return "no structure"


def _interpret_dbcv(v: float) -> str:
    if v > 0.5: return "excellent density"
    if v > 0.2: return "good density"
    if v > 0.0: return "marginal"
    return "poor density"


def _interpret_dbi(v: float) -> str:
    if v < 1.0: return "well separated"
    if v < 2.0: return "moderate overlap"
    if v < 3.0: return "significant overlap"
    return "heavy overlap"


def _interpret_coh(v: float) -> str:
    if v > 0.95: return "very tight"
    if v > 0.90: return "tight"
    if v > 0.80: return "reasonable"
    if v > 0.70: return "loose"
    return "very loose"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone EVoC clustering quality evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb
  python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb --per-cluster
  python scripts/eval_evoc_clustering.py data/cuad_eval.duckdb --json results.json
""",
    )
    parser.add_argument("db_path", help="Path to DuckDB database with clustered data")
    parser.add_argument("--embedding-dim", type=int, default=768,
                        help="Embedding dimension (default: 768)")
    parser.add_argument("--per-cluster", action="store_true",
                        help="Compute and display per-cluster breakdown")
    parser.add_argument("--json", metavar="PATH",
                        help="Save metrics to JSON file")

    args = parser.parse_args()

    if not Path(args.db_path).exists():
        print(f"Error: {args.db_path} not found")
        sys.exit(1)

    metrics = evaluate_clustering(
        db_path=args.db_path,
        embedding_dim=args.embedding_dim,
        compute_per_cluster=args.per_cluster,
    )

    print_report(metrics)

    if args.json:
        out = asdict(metrics)
        # Convert per_cluster dataclasses to dicts
        out["per_cluster"] = [asdict(c) for c in metrics.per_cluster]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nMetrics saved to {args.json}")


if __name__ == "__main__":
    main()
