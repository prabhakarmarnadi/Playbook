"""
Incremental cluster assigner — FIELD-801 REQUIRED deliverable.

Legacy pain point: V1 requires full re-clustering for every new batch of documents.
This is expensive (UMAP + HDBSCAN recomputation) and disruptive (cluster IDs change,
breaking downstream field assignments).

V2 approach: Once clusters are formed, new documents are assigned to the nearest
centroid in <500ms. Only trigger re-clustering when the novel buffer is full.
Centroids are updated with a running mean to adapt over time.

Flow:
    New Doc → Embed → Query centroids → Assign/Flag/Buffer
    HIGH_CONFIDENCE (>0.85): Assign + update centroid
    TENTATIVE (0.60-0.85): Assign + flag for review
    NOVEL (<0.60): Buffer for periodic re-clustering
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

    Key insight: Once clusters are formed, their centroids are stable.
    New docs can be assigned by finding the nearest centroid.
    Only trigger re-clustering when the novel buffer exceeds threshold.

    Legacy equivalent: None — V1 requires full re-clustering for every batch.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        high_confidence_threshold: float | None = None,
        tentative_threshold: float | None = None,
        novel_buffer_size: int | None = None,
    ):
        self.embedder = embedder or Embedder.get_instance()
        self.high_threshold = high_confidence_threshold or HIGH_CONFIDENCE_THRESHOLD
        self.tentative_threshold = tentative_threshold or TENTATIVE_THRESHOLD
        self.novel_buffer_size = novel_buffer_size or NOVEL_BUFFER_SIZE

        self.cluster_centroids: dict[str, np.ndarray] = {}
        self.cluster_counts: dict[str, int] = {}
        self.novel_buffer: list[dict] = []

    def load_centroids(self, centroids: dict[str, tuple[np.ndarray, int]]):
        """Load cluster centroids from store. centroids = {cluster_id: (centroid, count)}"""
        for cid, (centroid, count) in centroids.items():
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
            }

        Latency target: <500ms (embedding + cosine).
        """
        if not self.cluster_centroids:
            return {
                "cluster_id": None,
                "confidence": 0.0,
                "assignment_type": AssignmentType.NOVEL,
                "top_matches": [],
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

        if best_sim >= self.high_threshold:
            assignment_type = AssignmentType.HIGH_CONFIDENCE
            self._update_centroid(best_id, doc_embedding)
        elif best_sim >= self.tentative_threshold:
            assignment_type = AssignmentType.TENTATIVE
        else:
            assignment_type = AssignmentType.NOVEL
            best_id = None
            self.novel_buffer.append({"doc_id": doc_id, "text": text, "embedding": doc_embedding})

        if len(self.novel_buffer) >= self.novel_buffer_size:
            logger.warning(
                f"Novel buffer has {len(self.novel_buffer)} docs — re-clustering recommended"
            )

        return {
            "cluster_id": best_id,
            "confidence": best_sim,
            "assignment_type": assignment_type,
            "top_matches": top_matches,
        }

    def _update_centroid(self, cluster_id: str, new_embedding: np.ndarray):
        """Update cluster centroid with running mean."""
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
