"""
Personalization Injector — Runtime Pipeline Integration
========================================================

Injects personalization signals into the pipeline at every stage.
This is the "glue" between PersonalizationEngine and the core pipeline
components (cluster_assigner, intent_extractor, knowledge_graph, ontology_refiner).

The injector intercepts pipeline calls and overlays tenant-specific
behavior WITHOUT modifying core pipeline code.

Usage:
    injector = PersonalizationInjector(tenant_id="acme_corp", main_db_path="data/acme.duckdb")

    # Before cluster assignment:
    result = injector.personalized_assign(chunk_text, chunk_id, assigner)

    # Before intent extraction:
    prompt = injector.personalized_intent_prompt(clause_text, clause_type_id, base_prompt)

    # After intent extraction:
    intents = injector.post_process_intents(raw_intents)

    # During KG build:
    weight = injector.get_edge_weight(from_id, to_id, base_weight)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.feedback_store import FeedbackStore
from core.personalization import PersonalizationContext, PersonalizationEngine

logger = logging.getLogger(__name__)


class PersonalizationInjector:
    """
    Runtime injector — the single integration point for the pipeline.

    Lifecycle:
      1. Pipeline creates injector at start of run
      2. Each component calls injector methods before/after its work
      3. Injector reads from sidecar DB, never writes to main DB
      4. User corrections come in via API → FeedbackStore → sidecar DB
    """

    def __init__(
        self,
        tenant_id: str,
        main_db_path: str | None = None,
        data_dir: str | Path = "data",
    ):
        self.tenant_id = tenant_id
        self.store = FeedbackStore(tenant_id=tenant_id, data_dir=data_dir)
        self.engine = PersonalizationEngine(
            feedback_store=self.store,
            main_db_path=main_db_path,
        )
        self._ctx: PersonalizationContext | None = None

    @property
    def ctx(self) -> PersonalizationContext:
        """Lazy-load the personalization context."""
        if self._ctx is None:
            self._ctx = self.engine.get_context()
        return self._ctx

    def invalidate_cache(self):
        """Force reload of personalization context (after new feedback)."""
        self._ctx = None

    # ══════════════════════════════════════════════════════════════════════
    # INJECTION POINT 1: CLUSTER ASSIGNMENT
    # ══════════════════════════════════════════════════════════════════════

    def personalized_assign(
        self,
        embedding: np.ndarray,
        chunk_id: str,
        global_centroids: dict[str, np.ndarray],
    ) -> dict:
        """
        Personalized cluster assignment that considers:
        1. Tenant-specific confidence thresholds
        2. Virtual cluster centroids (added to candidate pool)
        3. Cluster weight adjustments (boost/penalize certain clusters)

        Called INSTEAD of default assignment logic when personalization is active.

        Returns:
            {
                "cluster_id": str,
                "similarity": float,
                "assignment_type": "high_confidence" | "tentative" | "novel" | "virtual",
                "is_virtual": bool,
                "top_matches": [(cluster_id, similarity), ...]
            }
        """
        ctx = self.ctx

        # Merge virtual cluster centroids into candidate pool
        all_centroids = dict(global_centroids)
        for vc in ctx.virtual_clusters:
            if vc.get("centroid"):
                vc_emb = np.frombuffer(vc["centroid"], dtype=np.float32)
                if len(vc_emb) == len(embedding):
                    all_centroids[vc["cluster_id"]] = vc_emb

        if not all_centroids:
            return {
                "cluster_id": None,
                "similarity": 0.0,
                "assignment_type": "novel",
                "is_virtual": False,
                "top_matches": [],
            }

        # Compute similarities with weight adjustments
        cids = list(all_centroids.keys())
        centroid_matrix = np.stack([all_centroids[c] for c in cids])
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_matrix = centroid_matrix / np.clip(norms, 1e-12, None)

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        similarities = centroid_matrix @ emb_norm

        # Apply personalization weights (boost approved, penalize corrected)
        for i, cid in enumerate(cids):
            weight = ctx.cluster_weights.get(cid, 1.0)
            similarities[i] *= weight

        # Sort by adjusted similarity
        top_indices = np.argsort(similarities)[::-1][:5]
        top_matches = [(cids[i], float(similarities[i])) for i in top_indices]

        best_cid = top_matches[0][0]
        best_sim = top_matches[0][1]
        is_virtual = best_cid.startswith("vc_")

        # Apply tenant-specific thresholds
        if best_sim >= ctx.high_confidence_threshold:
            assignment_type = "high_confidence"
        elif best_sim >= ctx.tentative_threshold:
            assignment_type = "tentative"
        else:
            assignment_type = "novel"
            best_cid = None

        if is_virtual and assignment_type != "novel":
            assignment_type = "virtual"

        return {
            "cluster_id": best_cid,
            "similarity": best_sim,
            "assignment_type": assignment_type,
            "is_virtual": is_virtual,
            "top_matches": top_matches,
        }

    # ══════════════════════════════════════════════════════════════════════
    # INJECTION POINT 2: INTENT EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    def personalized_intent_prompt(
        self,
        clause_text: str,
        clause_type_id: str | None,
        base_prompt: str,
    ) -> str:
        """
        Augment the intent discovery prompt with tenant-specific context.

        Appends:
        - Correction examples (few-shot)
        - Organization vocabulary (label mappings)
        - Temperature override (via config, applied outside this method)
        """
        addendum = self.engine.build_personalized_prompt_addendum(
            clause_type_id=clause_type_id,
        )
        if addendum:
            return base_prompt + "\n" + addendum
        return base_prompt

    def post_process_intents(self, intents: list[dict]) -> list[dict]:
        """Apply tenant intent label overrides after LLM extraction."""
        return self.engine.apply_intent_overrides(intents)

    def get_intent_temperature(self) -> float:
        """Get tenant-preferred LLM temperature for intent extraction."""
        return self.ctx.intent_temperature

    def get_intent_match_threshold(self) -> float:
        """Get tenant-preferred intent registry match threshold."""
        return self.ctx.intent_match_threshold

    # ══════════════════════════════════════════════════════════════════════
    # INJECTION POINT 3: KNOWLEDGE GRAPH
    # ══════════════════════════════════════════════════════════════════════

    def get_edge_weight(
        self,
        from_entity: str,
        to_entity: str,
        base_weight: float = 1.0,
    ) -> float:
        """
        Get personalized edge weight for KG construction.

        Called by knowledge_graph.py when adding edges.
        """
        edge_key = (from_entity, to_entity)
        multiplier = self.ctx.edge_weights.get(edge_key, 1.0)
        # Also check reverse direction for undirected edges
        if multiplier == 1.0:
            reverse_key = (to_entity, from_entity)
            multiplier = self.ctx.edge_weights.get(reverse_key, 1.0)
        return base_weight * multiplier

    # ══════════════════════════════════════════════════════════════════════
    # INJECTION POINT 4: OUTLIER HANDLING
    # ══════════════════════════════════════════════════════════════════════

    def get_outlier_threshold(self) -> float:
        """Get tenant-specific outlier assignment threshold."""
        return self.ctx.outlier_min_similarity

    def resolve_cluster_for_tenant(
        self,
        chunk_id: str,
        global_cluster_id: str,
    ) -> str:
        """
        Resolve effective cluster: returns virtual override if exists,
        global otherwise.
        """
        return self.engine.resolve_cluster(chunk_id, global_cluster_id)

    # ══════════════════════════════════════════════════════════════════════
    # FEEDBACK RECORDING (called by API layer)
    # ══════════════════════════════════════════════════════════════════════

    def record_cluster_correction(
        self,
        user_id: str,
        chunk_id: str,
        old_cluster_id: str,
        new_cluster_id: str | None = None,
        new_cluster_label: str | None = None,
        clause_id: str | None = None,
    ) -> dict:
        """User corrects a cluster assignment."""
        result = self.engine.correct_cluster_assignment(
            user_id=user_id,
            chunk_id=chunk_id,
            old_cluster_id=old_cluster_id,
            new_cluster_id=new_cluster_id,
            new_cluster_label=new_cluster_label,
            clause_id=clause_id,
        )
        self.invalidate_cache()
        return result

    def record_intent_correction(
        self,
        user_id: str,
        clause_id: str,
        old_label: str,
        new_label: str,
        new_summary: str = "",
        clause_type_id: str | None = None,
    ):
        """User corrects an intent label."""
        self.engine.correct_intent(
            user_id=user_id,
            clause_id=clause_id,
            old_intent_label=old_label,
            new_intent_label=new_label,
            new_summary=new_summary,
            clause_type_id=clause_type_id,
        )
        self.invalidate_cache()

    def record_edge_correction(
        self,
        user_id: str,
        from_entity: str,
        to_entity: str,
        action: str,
        reason: str = "",
    ) -> float:
        """User strengthens/weakens a KG edge."""
        weight = self.engine.adjust_kg_edge(
            user_id=user_id,
            from_entity=from_entity,
            to_entity=to_entity,
            action=action,
            reason=reason,
        )
        self.invalidate_cache()
        return weight

    def record_approval(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
    ):
        """User confirms a pipeline output is correct (positive signal)."""
        self.store.record_feedback(
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            action="approve",
        )
        self.invalidate_cache()

    def auto_tune(self) -> dict[str, float]:
        """Trigger adaptive threshold tuning based on accumulated feedback."""
        result = self.engine.auto_tune_thresholds()
        self.invalidate_cache()
        return result

    def close(self):
        self.store.close()
