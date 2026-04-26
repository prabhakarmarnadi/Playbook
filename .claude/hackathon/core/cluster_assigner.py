"""
Incremental cluster assigner — FIELD-801 REQUIRED deliverable.

Legacy pain point: V1 requires full re-clustering for every new batch of documents.
This is expensive (UMAP + HDBSCAN recomputation) and disruptive (cluster IDs change,
breaking downstream field assignments).

V2 approach: Once clusters are formed, new documents are assigned to the nearest
centroid in <500ms. Only trigger re-clustering when the novel buffer is full.
Centroids are updated with a running mean to adapt over time.

THREE-TIER CONFIDENCE MODEL:
    Tier A  HIGH_CONFIDENCE (≥0.65): Hard-assign + update centroid.
            These are clear cluster members — they improve coverage
            without degrading coherence.
    Tier B  TENTATIVE (0.45-0.65): Soft-assign with top-K memberships.
            Extracted against best-match cluster but recorded as soft
            member so they don't pull the centroid or affect quality.
    Tier C  NOVEL (<0.45): Buffer for re-clustering.  Genuinely different
            content — tagging them as any cluster would degrade quality.

Running centroid update: only Tier A (high-confidence) assignments update
the centroid via exponential running mean.  This prevents soft/novel
assignments from silently shifting cluster meaning.
"""
import logging
from enum import Enum

import numpy as np

from config import (
    HIGH_CONFIDENCE_THRESHOLD, TENTATIVE_THRESHOLD, NOVEL_BUFFER_SIZE,
)
from core.embedder import Embedder

logger = logging.getLogger(__name__)


class AssignmentType(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    TENTATIVE = "tentative"
    NOVEL = "novel"


class IncrementalClusterAssigner:
    """
    Assign new documents to existing clusters without re-clustering.

    Three-tier model:
      Tier A  (≥ high_threshold):  hard assign + centroid update
      Tier B  (≥ tentative_threshold): soft assign, top-K memberships
      Tier C  (< tentative_threshold): buffer for re-clustering

    Tier B documents get soft_memberships: {cluster_id: probability}
    for top-K clusters, computed via softmax over cosine similarities.
    This mirrors the pipeline's soft membership computation so soft
    members can be extracted against their best cluster without
    structurally polluting it.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        high_confidence_threshold: float | None = None,
        tentative_threshold: float | None = None,
        novel_buffer_size: int | None = None,
        soft_membership_k: int = 3,
        soft_temperature: float = 5.0,
    ):
        self.embedder = embedder or Embedder.get_instance()
        self.high_threshold = high_confidence_threshold or HIGH_CONFIDENCE_THRESHOLD
        self.tentative_threshold = tentative_threshold or TENTATIVE_THRESHOLD
        self.novel_buffer_size = novel_buffer_size or NOVEL_BUFFER_SIZE
        self.soft_k = soft_membership_k
        self.soft_temperature = soft_temperature

        self.cluster_centroids: dict[str, np.ndarray] = {}
        self.cluster_counts: dict[str, int] = {}
        self.novel_buffer: list[dict] = []

        # Track assignment history for drift monitoring
        self._assignment_log: list[dict] = []

    def load_centroids(self, centroids: dict[str, tuple[np.ndarray, int]]):
        """Load cluster centroids from store. centroids = {cluster_id: (centroid, count)}"""
        for cid, (centroid, count) in centroids.items():
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.cluster_centroids[cid] = centroid
            self.cluster_counts[cid] = count

    def assign_document(self, text: str, doc_id: str = "") -> dict:
        """
        Assign a new document/chunk to the best matching cluster.

        Returns:
            {
                "cluster_id": str or None,
                "confidence": float,
                "assignment_type": AssignmentType,
                "top_matches": [(cluster_id, similarity), ...],
                "soft_memberships": {cluster_id: prob} or {},
            }

        Tier A (≥ high_threshold): Hard assign + centroid update
        Tier B (≥ tentative_threshold): Soft assign with top-K memberships
        Tier C (< tentative_threshold): Buffer as novel

        Latency target: <500ms (embedding + cosine).
        """
        if not self.cluster_centroids:
            return {
                "cluster_id": None,
                "confidence": 0.0,
                "assignment_type": AssignmentType.NOVEL,
                "top_matches": [],
                "soft_memberships": {},
            }

        doc_embedding = self.embedder.embed_single(text, normalize=True)

        # Compute similarity to all centroids
        similarities = []
        for cluster_id, centroid in self.cluster_centroids.items():
            sim = float(np.dot(doc_embedding, centroid))
            similarities.append((cluster_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)

        top_matches = similarities[:5]
        best_id, best_sim = top_matches[0] if top_matches else (None, 0.0)

        soft_memberships = {}

        if best_sim >= self.high_threshold:
            # ── Tier A: Hard assign + centroid update ──
            assignment_type = AssignmentType.HIGH_CONFIDENCE
            self._update_centroid(best_id, doc_embedding)
        elif best_sim >= self.tentative_threshold:
            # ── Tier B: Soft assign with top-K memberships ──
            assignment_type = AssignmentType.TENTATIVE
            soft_memberships = self._compute_soft_memberships(similarities)
            # Do NOT update centroid — soft members don't shape the cluster
        else:
            # ── Tier C: Buffer as novel ──
            assignment_type = AssignmentType.NOVEL
            best_id = None
            self.novel_buffer.append({
                "doc_id": doc_id,
                "text": text,
                "embedding": doc_embedding,
            })

        if len(self.novel_buffer) >= self.novel_buffer_size:
            logger.warning(
                f"Novel buffer has {len(self.novel_buffer)} docs — re-clustering recommended"
            )

        # Log for drift monitoring
        self._assignment_log.append({
            "doc_id": doc_id,
            "cluster_id": best_id,
            "confidence": best_sim,
            "assignment_type": assignment_type.value,
        })

        return {
            "cluster_id": best_id,
            "confidence": best_sim,
            "assignment_type": assignment_type,
            "top_matches": top_matches,
            "soft_memberships": soft_memberships,
        }

    def _compute_soft_memberships(
        self, similarities: list[tuple[str, float]]
    ) -> dict[str, float]:
        """Compute softmax probabilities over top-K cluster similarities.

        Uses temperature scaling: prob_i = exp(sim_i * T) / Σ exp(sim_j * T)
        Higher temperature → sharper distribution (more confident top pick).
        """
        top_k = similarities[:self.soft_k]
        if not top_k:
            return {}

        sims = np.array([s for _, s in top_k])
        # Temperature-scaled softmax
        scaled = sims * self.soft_temperature
        scaled -= scaled.max()  # numerical stability
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum()

        return {cid: float(p) for (cid, _), p in zip(top_k, probs)}

    def _update_centroid(self, cluster_id: str, new_embedding: np.ndarray):
        """Update cluster centroid with running mean (Tier A only)."""
        n = self.cluster_counts.get(cluster_id, 1)
        old_centroid = self.cluster_centroids[cluster_id]
        new_centroid = (n * old_centroid + new_embedding) / (n + 1)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self.cluster_centroids[cluster_id] = new_centroid
        self.cluster_counts[cluster_id] = n + 1

    def get_buffer_size(self) -> int:
        return len(self.novel_buffer)

    def needs_recluster(self) -> bool:
        return len(self.novel_buffer) >= self.novel_buffer_size

    def get_assignment_stats(self) -> dict:
        """Return assignment statistics for drift monitoring."""
        if not self._assignment_log:
            return {"total": 0, "high": 0, "tentative": 0, "novel": 0,
                    "mean_confidence": 0.0, "novel_rate": 0.0}
        total = len(self._assignment_log)
        high = sum(1 for a in self._assignment_log
                   if a["assignment_type"] == "high_confidence")
        tentative = sum(1 for a in self._assignment_log
                        if a["assignment_type"] == "tentative")
        novel = sum(1 for a in self._assignment_log
                    if a["assignment_type"] == "novel")
        mean_conf = sum(a["confidence"] for a in self._assignment_log) / total
        return {
            "total": total,
            "high": high,
            "tentative": tentative,
            "novel": novel,
            "mean_confidence": round(mean_conf, 4),
            "novel_rate": round(novel / total, 4) if total else 0.0,
        }
