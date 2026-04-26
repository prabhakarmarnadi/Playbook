"""
Personalization Engine — Active Feedback Loop
==============================================

Consumes feedback events from FeedbackStore and produces runtime
personalization signals that inject into every pipeline component.

Three active subsystems:
  1. KG Weight Adjuster  — re-weight edges based on correction patterns
  2. Virtual Cluster Mgr — create/manage tenant-specific outlier clusters
  3. Prompt Personalizer  — adapt LLM prompts using tenant's correction history

Usage:
    from core.feedback_store import FeedbackStore
    from core.personalization import PersonalizationEngine

    store = FeedbackStore(tenant_id="acme_corp")
    engine = PersonalizationEngine(store, main_db_path="data/acme.duckdb")

    # At inference time:
    ctx = engine.get_context(clause_type_id="ct_abc123")
    # ctx.edge_weights, ctx.intent_overrides, ctx.virtual_clusters, ctx.thresholds
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from core.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSONALIZATION CONTEXT — returned at inference time
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonalizationContext:
    """Everything a pipeline component needs to personalize its output."""

    tenant_id: str = ""

    # Threshold overrides
    high_confidence_threshold: float = 0.85
    tentative_threshold: float = 0.60
    outlier_min_similarity: float = 0.35
    intent_match_threshold: float = 0.85
    intent_temperature: float = 0.0

    # KG edge weight adjustments: {(from_id, to_id): weight_multiplier}
    edge_weights: dict[tuple[str, str], float] = field(default_factory=dict)

    # Cluster weight adjustments: {cluster_id: weight_multiplier}
    cluster_weights: dict[str, float] = field(default_factory=dict)

    # Virtual clusters available for this tenant
    virtual_clusters: list[dict] = field(default_factory=list)

    # Intent label overrides: {global_label: {tenant_label, summary}}
    intent_overrides: dict[str, dict] = field(default_factory=dict)

    # Few-shot correction examples for LLM prompt injection
    correction_examples: list[dict] = field(default_factory=list)

    # Approval signals: entities the user confirmed are correct
    approved_entities: set[str] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSONALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PersonalizationEngine:
    """
    Orchestrates all personalization subsystems.

    Sits between the pipeline and the FeedbackStore, transforming
    raw feedback into actionable runtime signals.
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        main_db_path: str | None = None,
    ):
        self.store = feedback_store
        self.tenant_id = feedback_store.tenant_id
        self.main_db_path = main_db_path

    # ── Primary interface ─────────────────────────────────────────────────

    def get_context(
        self,
        clause_type_id: str | None = None,
        agreement_id: str | None = None,
    ) -> PersonalizationContext:
        """
        Build a complete PersonalizationContext for the current request.

        This is the single call a pipeline component makes to get
        all personalization signals.
        """
        config = self.store.get_tenant_config()
        ctx = PersonalizationContext(
            tenant_id=self.tenant_id,
            high_confidence_threshold=config.get(
                "high_confidence_threshold", 0.85
            ),
            tentative_threshold=config.get("tentative_threshold", 0.60),
            outlier_min_similarity=config.get("outlier_min_similarity", 0.35),
            intent_match_threshold=config.get("intent_match_threshold", 0.85),
            intent_temperature=config.get("intent_temperature", 0.0),
        )

        # Load all personalization layers
        ctx.cluster_weights = self.store.get_weights_batch("cluster")
        ctx.edge_weights = self._load_edge_weights()
        ctx.virtual_clusters = self.store.get_virtual_clusters()
        ctx.intent_overrides = self.store.get_intent_overrides()
        ctx.correction_examples = self._load_correction_examples(
            clause_type_id
        )
        ctx.approved_entities = self._load_approved_entities()

        return ctx

    # ══════════════════════════════════════════════════════════════════════
    # 1. KG WEIGHT ADJUSTER
    # ══════════════════════════════════════════════════════════════════════

    def adjust_kg_edge(
        self,
        user_id: str,
        from_entity: str,
        to_entity: str,
        action: str,
        reason: str = "",
    ) -> float:
        """
        User says an edge is wrong (unlink) or missing (link).

        Adjusts the personalization weight for this specific edge.
        The KG builder reads these weights at graph-build time.

        Returns the new weight.
        """
        edge_id = f"{from_entity}::{to_entity}"
        self.store.record_feedback(
            user_id=user_id,
            entity_type="edge",
            entity_id=edge_id,
            action=action,
            old_value={"from": from_entity, "to": to_entity},
            new_value={"reason": reason},
        )
        return self.store.get_weight("edge", edge_id)

    def get_adjusted_edge_weight(
        self,
        from_entity: str,
        to_entity: str,
        base_weight: float = 1.0,
    ) -> float:
        """
        Called by KG builder: returns base_weight * personalization_multiplier.

        If no feedback exists, returns base_weight unchanged.
        """
        edge_id = f"{from_entity}::{to_entity}"
        multiplier = self.store.get_weight("edge", edge_id)
        return base_weight * multiplier

    def _load_edge_weights(self) -> dict[tuple[str, str], float]:
        """Load all edge weights as {(from, to): multiplier}."""
        raw = self.store.get_weights_batch("edge")
        result = {}
        for edge_id, weight in raw.items():
            parts = edge_id.split("::", 1)
            if len(parts) == 2:
                result[(parts[0], parts[1])] = weight
        return result

    # ══════════════════════════════════════════════════════════════════════
    # 2. VIRTUAL CLUSTER MANAGER
    # ══════════════════════════════════════════════════════════════════════

    def correct_cluster_assignment(
        self,
        user_id: str,
        chunk_id: str,
        old_cluster_id: str,
        new_cluster_id: str | None = None,
        new_cluster_label: str | None = None,
        clause_id: str | None = None,
    ) -> dict:
        """
        User says this chunk belongs to a different cluster.

        Two modes:
        1. new_cluster_id provided → reassign to existing (global or virtual) cluster
        2. new_cluster_label provided → create a new virtual cluster for this tenant

        Returns: {"action": str, "virtual_cluster_id": str | None}
        """
        result = {"action": "reassigned", "virtual_cluster_id": None}

        if new_cluster_label and not new_cluster_id:
            # Create a virtual cluster for this tenant
            vc_id = self.store.create_virtual_cluster(
                label=new_cluster_label,
                user_id=user_id,
                description=f"Created from correction on chunk {chunk_id}",
            )
            new_cluster_id = vc_id
            result["action"] = "virtual_cluster_created"
            result["virtual_cluster_id"] = vc_id
            logger.info(
                f"Virtual cluster created: {vc_id} ({new_cluster_label}) "
                f"for tenant={self.tenant_id}"
            )

        # Record the assignment
        if new_cluster_id and new_cluster_id.startswith("vc_"):
            self.store.assign_to_virtual_cluster(
                virtual_cluster_id=new_cluster_id,
                user_id=user_id,
                chunk_id=chunk_id,
                clause_id=clause_id,
                global_cluster_id=old_cluster_id,
            )

        # Record feedback event
        self.store.record_feedback(
            user_id=user_id,
            entity_type="cluster",
            entity_id=old_cluster_id,
            action="correct",
            old_value={"cluster_id": old_cluster_id, "chunk_id": chunk_id},
            new_value={
                "cluster_id": new_cluster_id,
                "label": new_cluster_label or "",
            },
        )

        return result

    def resolve_cluster(
        self,
        chunk_id: str,
        global_cluster_id: str,
    ) -> str:
        """
        At inference time: check if this chunk has a virtual cluster override.

        Returns the effective cluster_id (virtual if overridden, global otherwise).
        """
        row = self.store.conn.execute(
            """SELECT virtual_cluster_id FROM virtual_assignments
               WHERE tenant_id = ? AND chunk_id = ?
               ORDER BY assigned_at DESC LIMIT 1""",
            [self.tenant_id, chunk_id],
        ).fetchone()
        if row:
            return row[0]
        return global_cluster_id

    def promote_virtual_cluster(
        self,
        virtual_cluster_id: str,
        embedder=None,
    ) -> dict:
        """
        When a virtual cluster has enough members, compute its centroid
        so it can participate in incremental assignment.

        Returns: {"member_count": int, "centroid_computed": bool}
        """
        members = self.store.conn.execute(
            """SELECT chunk_id FROM virtual_assignments
               WHERE virtual_cluster_id = ?""",
            [virtual_cluster_id],
        ).fetchall()

        if not members or not self.main_db_path or not embedder:
            return {"member_count": len(members), "centroid_computed": False}

        # Fetch embeddings from main DB
        chunk_ids = [m[0] for m in members if m[0]]
        if not chunk_ids:
            return {"member_count": len(members), "centroid_computed": False}

        main_conn = duckdb.connect(self.main_db_path, read_only=True)
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = main_conn.execute(
            f"SELECT embedding FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        main_conn.close()

        embeddings = []
        for (blob,) in rows:
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
                embeddings.append(emb)

        if not embeddings:
            return {"member_count": len(members), "centroid_computed": False}

        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroid_blob = centroid.astype(np.float32).tobytes()

        self.store.conn.execute(
            "UPDATE virtual_clusters SET centroid = ? WHERE cluster_id = ?",
            [centroid_blob, virtual_cluster_id],
        )

        return {"member_count": len(members), "centroid_computed": True}

    # ══════════════════════════════════════════════════════════════════════
    # 3. PROMPT PERSONALIZER
    # ══════════════════════════════════════════════════════════════════════

    def correct_intent(
        self,
        user_id: str,
        clause_id: str,
        old_intent_label: str,
        new_intent_label: str,
        new_summary: str = "",
        clause_type_id: str | None = None,
    ):
        """
        User corrects an intent label or summary.

        Two effects:
        1. Records a correction example (used for few-shot prompt injection)
        2. Creates an intent override if label differs significantly
        """
        self.store.record_feedback(
            user_id=user_id,
            entity_type="intent",
            entity_id=f"{clause_id}::{old_intent_label}",
            action="correct",
            old_value={"label": old_intent_label},
            new_value={"label": new_intent_label, "summary": new_summary},
            context={"clause_type_id": clause_type_id or ""},
        )

        # If labels differ, create a persistent override
        if old_intent_label.strip().lower() != new_intent_label.strip().lower():
            clause_types = [clause_type_id] if clause_type_id else []
            self.store.set_intent_override(
                global_label=old_intent_label,
                tenant_label=new_intent_label,
                user_id=user_id,
                summary=new_summary,
                clause_types=clause_types,
            )

    def build_personalized_prompt_addendum(
        self,
        clause_type_id: str | None = None,
        max_examples: int = 5,
    ) -> str:
        """
        Build a prompt addendum that injects tenant's correction history
        as few-shot examples into the intent discovery prompt.

        Returns empty string if no relevant corrections exist.
        """
        examples = self._load_correction_examples(clause_type_id, max_examples)
        if not examples:
            return ""

        lines = [
            "\n## Tenant-Specific Naming Conventions",
            "The following corrections were made by the user for this organization. "
            "Apply these naming preferences when discovering intents:\n",
        ]
        for ex in examples:
            old = ex.get("old_label", "")
            new = ex.get("new_label", "")
            summary = ex.get("new_summary", "")
            lines.append(f"- Instead of \"{old}\", use \"{new}\"")
            if summary:
                lines.append(f"  Summary: {summary}")

        # Add intent overrides as global rules
        overrides = self.store.get_intent_overrides()
        if overrides:
            lines.append("\n## Organization Intent Vocabulary")
            lines.append(
                "Always use these canonical intent names for this organization:\n"
            )
            for gl, ov in list(overrides.items())[:10]:
                lines.append(
                    f"- \"{gl}\" → \"{ov['tenant_label']}\""
                )

        return "\n".join(lines)

    def apply_intent_overrides(
        self,
        intents: list[dict],
    ) -> list[dict]:
        """
        Post-process LLM-extracted intents: apply tenant overrides.

        Replaces global labels with tenant-preferred labels.
        """
        overrides = self.store.get_intent_overrides()
        if not overrides:
            return intents

        for intent in intents:
            label = intent.get("label", "")
            if label in overrides:
                ov = overrides[label]
                intent["label"] = ov["tenant_label"]
                if ov.get("summary"):
                    intent["summary"] = ov["summary"]
                intent["_override_applied"] = True

        return intents

    # ══════════════════════════════════════════════════════════════════════
    # 4. ADAPTIVE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════

    def auto_tune_thresholds(self) -> dict[str, float]:
        """
        Analyze feedback patterns and suggest threshold adjustments.

        Logic:
        - High TENTATIVE→correct rate → lower high_confidence_threshold
        - High TENTATIVE→reject rate → raise tentative_threshold
        - Many virtual clusters → lower outlier_min_similarity
        """
        events = self.store.conn.execute(
            """SELECT entity_type, action, count(*) as cnt
               FROM feedback_events
               WHERE tenant_id = ?
               GROUP BY entity_type, action""",
            [self.tenant_id],
        ).fetchall()

        stats = {}
        for etype, action, cnt in events:
            stats[f"{etype}.{action}"] = cnt

        config = self.store.get_tenant_config()
        suggestions = {}

        # Cluster corrections → adjust confidence thresholds
        cluster_corrects = stats.get("cluster.correct", 0)
        cluster_approves = stats.get("cluster.approve", 0)
        total_cluster = cluster_corrects + cluster_approves
        if total_cluster >= 10:
            accuracy = cluster_approves / total_cluster
            if accuracy > 0.9:
                # Users rarely correct → we can be more aggressive
                suggestions["high_confidence_threshold"] = max(
                    0.70, config.get("high_confidence_threshold", 0.85) - 0.05
                )
            elif accuracy < 0.5:
                # Users frequently correct → be more conservative
                suggestions["high_confidence_threshold"] = min(
                    0.95, config.get("high_confidence_threshold", 0.85) + 0.05
                )

        # Many virtual clusters → lower outlier threshold
        n_virtual = len(self.store.get_virtual_clusters())
        if n_virtual >= 5:
            suggestions["outlier_min_similarity"] = max(
                0.25, config.get("outlier_min_similarity", 0.35) - 0.05
            )

        # Intent corrections → adjust match threshold
        intent_corrects = stats.get("intent.correct", 0)
        if intent_corrects >= 20:
            suggestions["intent_match_threshold"] = min(
                0.95, config.get("intent_match_threshold", 0.85) + 0.05
            )

        if suggestions:
            self.store.update_tenant_config(**suggestions)
            logger.info(
                f"Auto-tuned thresholds for {self.tenant_id}: {suggestions}"
            )

        return suggestions

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _load_correction_examples(
        self,
        clause_type_id: str | None = None,
        max_examples: int = 5,
    ) -> list[dict]:
        """Load recent intent correction examples for few-shot injection."""
        query = """
            SELECT old_value, new_value, context
            FROM feedback_events
            WHERE tenant_id = ? AND entity_type = 'intent' AND action = 'correct'
        """
        params: list = [self.tenant_id]
        if clause_type_id:
            query += " AND json_extract_string(context, '$.clause_type_id') = ?"
            params.append(clause_type_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max_examples)

        rows = self.store.conn.execute(query, params).fetchall()
        examples = []
        for old_json, new_json, ctx_json in rows:
            old = json.loads(old_json) if isinstance(old_json, str) else old_json
            new = json.loads(new_json) if isinstance(new_json, str) else new_json
            examples.append({
                "old_label": old.get("label", ""),
                "new_label": new.get("label", ""),
                "new_summary": new.get("summary", ""),
            })
        return examples

    def _load_approved_entities(self) -> set[str]:
        """Load entities the user has explicitly approved."""
        rows = self.store.conn.execute(
            """SELECT entity_id FROM feedback_events
               WHERE tenant_id = ? AND action = 'approve'
               ORDER BY created_at DESC LIMIT 500""",
            [self.tenant_id],
        ).fetchall()
        return {r[0] for r in rows}

    def summary(self) -> dict:
        """Full state summary for debugging."""
        store_summary = self.store.summary()
        config = self.store.get_tenant_config()
        return {
            **store_summary,
            "config": config,
        }
