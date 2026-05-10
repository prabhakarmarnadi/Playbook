"""
Scoring engine — ported from legacy scoring.py + scores.py + constants.py.

Computes a composite Z-score from 3 unsupervised metrics:
  - Average cosine similarity (intra-cluster coherence) — weight 0.3
  - Silhouette score (cluster separation) — weight 0.3
  - DBCV score (density-based cluster validity) — weight 0.4

Plus 4 penalty terms from the legacy Bayesian optimizer:
  - Cluster count penalty (too few clusters)
  - Dominance penalty (one cluster hogs all docs)
  - Outlier ratio penalty (too many unclustered docs)
  - Keyword diversity penalty (generic/overlapping topics)

Legacy source: apr_agreement_clustering.modules.evaluation.scoring + constants.py
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# ── Default metric weights ─────────────────────────────────────────────────────
# DBCV is unreliable on high-dimensional embeddings (768d) — it computes mutual
# reachability distances that become meaningless in high dimensions.  Cosine
# coherence and silhouette are the primary quality signals for EVoC.
DEFAULT_WEIGHTS = {
    "average_cosine_similarity": 0.40,
    "silhouette_score": 0.40,
    "dbcv_score": 0.20,
}

# ── Penalty constants (from legacy constants.py) ──────────────────────────────
CLUSTER_COUNT_PENALTY_FLOOR = 0.4
CLUSTER_COUNT_PENALTY_SLOPE = 0.3
DOMINANCE_PENALTY_FLOOR = 0.2
DOMINANCE_PENALTY_SCALE = 4.0
OUTLIER_PENALTY_FLOOR = 0.7
OUTLIER_PENALTY_SCALE = 0.75
KEYWORD_DIVERSITY_PENALTY_SCALE = 0.75

# ── Default thresholds ─────────────────────────────────────────────────────────
# EVoC naturally produces more outliers than UMAP+HDBSCAN (20-35% is normal).
# Legacy thresholds (0.15 outlier, 0.08 dominance) were calibrated for HDBSCAN
# and over-penalize healthy EVoC results.
DEFAULT_MIN_CLUSTERS = 20
DEFAULT_MAX_DOMINANCE = 0.12
DEFAULT_MAX_OUTLIER_RATIO = 0.30
DEFAULT_KEYWORD_DIVERSITY_THRESHOLD = 0.5


@dataclass
class ClusterScores:
    """All scores for a single clustering run."""

    average_cosine_similarity: float = 0.0
    silhouette_score: float = 0.0
    dbcv_score: float = 0.0
    z_score: float = 0.0
    adjusted_score: float = 0.0

    # Penalty breakdown
    cluster_count_penalty: float = 1.0
    dominance_penalty: float = 1.0
    outlier_penalty: float = 1.0
    keyword_diversity_penalty: float = 1.0

    # Raw stats
    n_clusters: int = 0
    n_outliers: int = 0
    n_total: int = 0
    max_cluster_fraction: float = 0.0
    outlier_ratio: float = 0.0
    topic_counts: dict = field(default_factory=dict)  # topic_id -> count

    params: dict = field(default_factory=dict)


def _cosine_similarity_per_cluster(
    topics: np.ndarray,
    embeddings: np.ndarray,
) -> dict[int, float]:
    """Per-cluster average cosine similarity to centroid."""
    sims = {}
    for tid in set(topics):
        if tid == -1:
            continue
        mask = topics == tid
        cluster_embs = embeddings[mask]
        if len(cluster_embs) < 2:
            sims[tid] = 1.0
            continue
        centroid = cluster_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        normed = cluster_embs / (norms + 1e-12)
        sims[tid] = float(np.mean(normed @ centroid_norm))
    return sims


def _safe_silhouette(topics: np.ndarray, embeddings: np.ndarray) -> float:
    """Silhouette score with edge-case handling."""
    non_outlier_mask = topics != -1
    unique_labels = set(topics[non_outlier_mask])
    if len(unique_labels) < 2 or non_outlier_mask.sum() < 3:
        return 0.0
    try:
        return float(silhouette_score(
            embeddings[non_outlier_mask],
            topics[non_outlier_mask],
            metric="cosine",
            sample_size=min(5000, int(non_outlier_mask.sum())),
            random_state=42,
        ))
    except Exception:
        return 0.0


def _safe_dbcv(topics: np.ndarray, embeddings: np.ndarray) -> float:
    """DBCV score with fallback on import/computation failure.

    DBCV computes a pairwise distance matrix — O(n²) time and memory.
    For >5000 non-outlier points this takes minutes per call and the metric
    is unreliable on high-dimensional embeddings anyway. We skip it for
    large datasets and use a fast sample-based estimate for medium ones.

    On high-dimensional input (>100d) DBCV's 1/distance^dim term overflows
    to inf for small distances, producing noisy RuntimeWarnings and a
    degraded score. We skip entirely above that threshold.
    """
    import warnings

    non_outlier_mask = topics != -1
    unique_labels = set(topics[non_outlier_mask])
    if len(unique_labels) < 2:
        return 0.0

    n_valid = int(non_outlier_mask.sum())

    # Skip entirely for large datasets — DBCV is O(n²) and unreliable on 768d
    if n_valid > 2000:
        return 0.0

    # Skip when embedding dim > 100 — DBCV's 1/d^dim term overflows; use silhouette/cosine instead.
    if embeddings.ndim == 2 and embeddings.shape[1] > 100:
        return 0.0

    def _call_validity(embs: np.ndarray, tps: np.ndarray) -> float:
        # Silence DBCV's intrinsic numpy warnings (overflow / underflow / invalid)
        # so the pipeline log stays clean. The function's own try/except handles
        # actual failures.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from hdbscan.validity import validity_index
            return float(validity_index(embs.astype(np.float64), np.asarray(tps)))

    # Sample-based estimate for medium datasets (500-2000)
    if n_valid > 500:
        rng = np.random.RandomState(42)
        sample_size = min(1000, n_valid)
        sample_idx = rng.choice(np.where(non_outlier_mask)[0], size=sample_size, replace=False)
        sample_embs = embeddings[sample_idx]
        sample_topics = topics[sample_idx]
        if len(set(sample_topics)) < 2:
            return 0.0
        try:
            return _call_validity(sample_embs, sample_topics)
        except Exception:
            return 0.0

    try:
        return _call_validity(embeddings[non_outlier_mask], topics[non_outlier_mask])
    except Exception:
        return 0.0


def calculate_scores(
    topics: np.ndarray,
    embeddings: np.ndarray,
    umap_embeddings: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> ClusterScores:
    """
    Calculate all clustering quality metrics.

    Ported from legacy ScoringEngine.calculate_all_scores().
    Uses UMAP embeddings for distance-based metrics when available.
    """
    topics = np.asarray(topics)
    w = weights or DEFAULT_WEIGHTS

    distance_embs = umap_embeddings if umap_embeddings is not None else embeddings

    # Metrics
    per_cluster_sim = _cosine_similarity_per_cluster(topics, embeddings)
    avg_cosine = float(np.mean(list(per_cluster_sim.values()))) if per_cluster_sim else 0.0
    sil = _safe_silhouette(topics, distance_embs)
    dbcv = _safe_dbcv(topics, distance_embs)

    z = (
        avg_cosine * w.get("average_cosine_similarity", 0.3)
        + sil * w.get("silhouette_score", 0.3)
        + dbcv * w.get("dbcv_score", 0.4)
    )

    n_clusters = len(set(t for t in topics if t != -1))
    n_outliers = int(np.sum(topics == -1))
    n_total = len(topics)
    counts = np.bincount(topics[topics >= 0]) if np.any(topics >= 0) else np.array([])
    max_frac = float(counts.max() / n_total) if len(counts) > 0 and n_total > 0 else 0.0
    outlier_ratio = n_outliers / n_total if n_total > 0 else 0.0

    # Per-topic counts for cumulative dominance penalty
    topic_counts_dict = {}
    for tid in set(topics):
        if tid != -1:
            topic_counts_dict[int(tid)] = int(np.sum(topics == tid))

    scores = ClusterScores(
        average_cosine_similarity=avg_cosine,
        silhouette_score=sil,
        dbcv_score=dbcv,
        z_score=z,
        n_clusters=n_clusters,
        n_outliers=n_outliers,
        n_total=n_total,
        max_cluster_fraction=max_frac,
        outlier_ratio=outlier_ratio,
        topic_counts=topic_counts_dict,
    )
    return scores


def apply_penalties(
    scores: ClusterScores,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_dominance: float = DEFAULT_MAX_DOMINANCE,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
    keyword_diversity: float | None = None,
    keyword_diversity_threshold: float = DEFAULT_KEYWORD_DIVERSITY_THRESHOLD,
) -> ClusterScores:
    """
    Apply the 4 legacy penalty terms to the raw z_score.

    Ported exactly from legacy BayesianOptimizer._calculate_adjusted_score():
    1. Cluster count: linear scaling FLOOR + SLOPE*(n-1)/(threshold-1)
    2. Dominance: cumulative per-oversized-cluster penalty
    3. Outlier ratio: excess * scale
    4. Keyword diversity: excess * scale

    adjusted_score = z_score * all_penalties (floored at 0.0)
    """
    penalty_multiplier = 1.0

    # 1. Cluster count penalty — linear interpolation (legacy formula)
    if scores.n_clusters < min_clusters:
        cluster_factor = CLUSTER_COUNT_PENALTY_FLOOR + CLUSTER_COUNT_PENALTY_SLOPE * (
            scores.n_clusters - 1
        ) / max(1, min_clusters - 1)
        scores.cluster_count_penalty = cluster_factor
        penalty_multiplier *= cluster_factor
    else:
        scores.cluster_count_penalty = 1.0

    # 2. Dominance penalty — cumulative per oversized cluster (legacy formula)
    #    Each cluster exceeding max_dominance gets its own penalty multiplied in
    if scores.topic_counts and scores.n_clusters > 0:
        non_outlier_docs = scores.n_total - scores.n_outliers
        if non_outlier_docs > 0:
            dominance_product = 1.0
            for _tid, count in scores.topic_counts.items():
                ratio = count / non_outlier_docs
                if ratio > max_dominance:
                    excess = ratio - max_dominance
                    factor = max(DOMINANCE_PENALTY_FLOOR, 1.0 - excess * DOMINANCE_PENALTY_SCALE)
                    dominance_product *= factor
            scores.dominance_penalty = dominance_product
            penalty_multiplier *= dominance_product
    else:
        scores.dominance_penalty = 1.0

    # 3. Outlier ratio penalty
    if scores.outlier_ratio > max_outlier_ratio:
        excess = scores.outlier_ratio - max_outlier_ratio
        scores.outlier_penalty = max(
            OUTLIER_PENALTY_FLOOR,
            1.0 - excess * OUTLIER_PENALTY_SCALE,
        )
        penalty_multiplier *= scores.outlier_penalty
    else:
        scores.outlier_penalty = 1.0

    # 4. Keyword diversity penalty
    if keyword_diversity is not None and keyword_diversity < keyword_diversity_threshold:
        excess = keyword_diversity_threshold - keyword_diversity
        scores.keyword_diversity_penalty = max(
            0.3,
            1.0 - excess * KEYWORD_DIVERSITY_PENALTY_SCALE,
        )
        penalty_multiplier *= scores.keyword_diversity_penalty
    else:
        scores.keyword_diversity_penalty = 1.0

    # Composite
    scores.adjusted_score = max(0.0, scores.z_score * penalty_multiplier)

    return scores
