"""
Hierarchical topic merger + synonym resolution — FIELD-310 REQUIRED deliverable.

Legacy pain point: BERTopic consistently produces near-duplicate clusters:
  - "Payment Terms" and "Net Payment Terms" as separate clusters
  - "Confidentiality" and "Non-Disclosure" as separate clusters
V1 treats these as distinct, fragmenting the ontology and creating duplicate extraction work.

V2 approach: Two-phase post-processing.
  Phase 1: Merge clusters with high centroid cosine similarity (single-linkage).
  Phase 2: Resolve field name synonyms across (merged) clusters.
"""
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import MERGE_SIMILARITY_THRESHOLD, SYNONYM_SIMILARITY_THRESHOLD
from core.embedder import Embedder

logger = logging.getLogger(__name__)


class HierarchicalTopicMerger:
    """Merge semantically similar topics and resolve field synonyms.

    Uses a two-phase approach:
    1. Topic merging: Combine clusters with high centroid similarity
    2. Synonym resolution: Unify field names across merged clusters
    """

    def __init__(
        self,
        merge_threshold: float | None = None,
        synonym_threshold: float | None = None,
    ):
        self.merge_threshold = merge_threshold or MERGE_SIMILARITY_THRESHOLD
        self.synonym_threshold = synonym_threshold or SYNONYM_SIMILARITY_THRESHOLD

    def merge_similar_topics(
        self,
        clusters: list[dict],
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[dict]:
        """
        Hierarchically merge clusters that are semantically equivalent.

        If embeddings dict is not provided, uses centroids from cluster records.
        Returns merged cluster list with parent-child relationships.
        """
        if len(clusters) < 2:
            return clusters

        # Build centroid matrix
        cluster_ids = [c["cluster_id"] for c in clusters]
        if embeddings:
            centroids = np.array([embeddings[cid] for cid in cluster_ids])
        else:
            centroids = np.array([c["centroid"] for c in clusters])

        # Compute pairwise similarity
        sim_matrix = cosine_similarity(centroids)

        # Find merge groups (complete-linkage to prevent chaining)
        merge_groups = self._complete_linkage_merge(cluster_ids, sim_matrix)

        # Build merged clusters
        merged = []
        for group in merge_groups:
            group_clusters = [c for c in clusters if c["cluster_id"] in group]
            if len(group_clusters) == 1:
                merged.append(group_clusters[0])
            else:
                merged.append(self._merge_cluster_group(group_clusters))
                logger.info(f"  Merged {len(group_clusters)} clusters: "
                           f"{[c.get('label','?') for c in group_clusters]} → "
                           f"{merged[-1].get('label','?')}")

        return merged

    def _complete_linkage_merge(self, ids: list[str], sim_matrix: np.ndarray) -> list[list[str]]:
        """Complete-linkage merging: merge two clusters only if ALL pairwise
        similarities exceed threshold.  Prevents the chaining problem where
        single-linkage drags unrelated clusters together through bridge pairs."""
        n = len(ids)
        # Start with each item in its own group
        groups: list[list[int]] = [[i] for i in range(n)]

        changed = True
        while changed:
            changed = False
            best_pair = None
            best_min_sim = -1.0
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    # Complete linkage: minimum pairwise similarity between groups
                    min_sim = min(
                        sim_matrix[a, b]
                        for a in groups[gi] for b in groups[gj]
                    )
                    if min_sim >= self.merge_threshold and min_sim > best_min_sim:
                        best_min_sim = min_sim
                        best_pair = (gi, gj)
            if best_pair is not None:
                gi, gj = best_pair
                groups[gi].extend(groups[gj])
                groups.pop(gj)
                changed = True

        return [[ids[i] for i in g] for g in groups]

    def _merge_cluster_group(self, clusters: list[dict]) -> dict:
        """Merge multiple clusters into one, preserving provenance."""
        # Pick cluster with highest quality score as primary
        primary = max(clusters, key=lambda c: c.get("quality_score", 0))

        # Combine keywords, deduplicated, capped at 15
        all_kw: list[str] = []
        for c in clusters:
            all_kw.extend(c.get("keywords", []))
        combined_kw = list(dict.fromkeys(all_kw))[:15]

        # Record aliases
        aliases = [c["label"] for c in clusters if c.get("label") != primary.get("label")]

        merged = {**primary}
        merged.update({
            "keywords": combined_kw,
            "aliases": aliases,
            "merged_from": [c["cluster_id"] for c in clusters],
            "chunk_count": sum(c.get("chunk_count", 0) for c in clusters),
            "agreement_count": sum(c.get("agreement_count", 0) for c in clusters),
            # Recalculate centroid as weighted average
            "centroid": self._weighted_centroid(clusters),
        })
        # Merge chunk_ids if present
        if any("chunk_ids" in c for c in clusters):
            merged["chunk_ids"] = []
            for c in clusters:
                merged["chunk_ids"].extend(c.get("chunk_ids", []))
        if any("chunk_indices" in c for c in clusters):
            merged["chunk_indices"] = []
            for c in clusters:
                merged["chunk_indices"].extend(c.get("chunk_indices", []))

        return merged

    @staticmethod
    def _weighted_centroid(clusters: list[dict]) -> np.ndarray:
        """Compute chunk-count-weighted centroid across merged clusters."""
        total = sum(c.get("chunk_count", 1) for c in clusters)
        if total == 0:
            return clusters[0].get("centroid", np.zeros(384))
        weighted = sum(
            c.get("centroid", np.zeros(384)) * (c.get("chunk_count", 1) / total)
            for c in clusters
        )
        norm = np.linalg.norm(weighted)
        return weighted / norm if norm > 0 else weighted

    def resolve_field_synonyms(
        self,
        fields: list[dict],
        embedder: Embedder,
    ) -> dict[str, str]:
        """
        Map field name variants to canonical names using embedding similarity.

        Returns: {variant_name: canonical_name}
        """
        if len(fields) < 2:
            return {}

        # Embed field names + descriptions
        texts = [f"{f['name']}: {f.get('description', '')}" for f in fields]
        embs = embedder.embed(texts)
        sim_matrix = cosine_similarity(embs)

        names = [f["name"] for f in fields]
        groups = self._single_linkage_merge(names, sim_matrix)

        mapping = {}
        for group in groups:
            if len(group) > 1:
                canonical = min(group)  # Alphabetical for determinism
                for name in group:
                    if name != canonical:
                        mapping[name] = canonical
                        logger.info(f"  Synonym: '{name}' → '{canonical}'")

        return mapping


class SynonymResolver:
    """Resolve synonyms using known legal field synonym lists + embedding similarity."""

    KNOWN_SYNONYMS = {
        "payment_days": ["net_terms", "payment_period", "days_to_pay", "payment_terms_days"],
        "effective_date": ["start_date", "commencement_date", "agreement_date"],
        "expiration_date": ["end_date", "termination_date", "term_end"],
        "governing_law": ["applicable_law", "choice_of_law", "jurisdiction"],
        "liability_cap": ["limitation_of_liability", "max_liability", "liability_limit"],
        "notice_period": ["notice_days", "termination_notice", "advance_notice"],
        "auto_renewal": ["auto_renew", "automatic_renewal", "evergreen"],
        "confidentiality_period": ["nda_term", "confidentiality_term", "secrecy_period"],
    }

    @classmethod
    def resolve(cls, field_name: str) -> str:
        """Resolve a field name to its canonical form."""
        normalized = field_name.lower().strip().replace(" ", "_").replace("-", "_")
        for canonical, synonyms in cls.KNOWN_SYNONYMS.items():
            if normalized == canonical or normalized in synonyms:
                return canonical
        return normalized
