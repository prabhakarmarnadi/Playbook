"""
Feedback & Personalization Engine — Sidecar Storage Layer
=========================================================

A tenant-scoped sidecar database that sits alongside the main pipeline DB.
Never pollutes the core ontology — all personalization is overlay.

Architecture:
  main DB (evoc_200.duckdb)     ← pipeline writes here (global truth)
  sidecar DB (<tenant>.feedback.duckdb) ← feedback writes here (per-tenant overlay)

Tables:
  feedback_events       — unified feedback log (all entity types)
  personalization_weights — per-tenant weight overrides for KG edges, clusters, thresholds
  virtual_clusters      — tenant-specific clusters for outliers that don't fit global taxonomy
  virtual_assignments   — maps chunks/clauses to virtual clusters
  intent_overrides      — tenant-preferred intent labels/mappings
  tenant_config         — per-tenant threshold overrides
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK EVENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_TYPES = (
    "cluster",          # cluster assignment correction
    "edge",             # KG edge strength correction
    "intent",           # intent label/attribute correction
    "field",            # field extraction correction
    "clause_name",      # clause type label correction
    "domain",           # domain assignment correction
)

ACTION_TYPES = (
    "correct",          # user provides the right answer
    "reject",           # user says this is wrong, no alternative given
    "refine",           # user tweaks (e.g., edits intent summary)
    "link",             # user says these two entities should be connected
    "unlink",           # user says these two entities should NOT be connected
    "merge",            # user says two clusters/intents are the same
    "split",            # user says one cluster should be two
    "approve",          # user confirms pipeline output is correct (positive signal)
)

# Temporal decay: half-life in days for feedback weight
DEFAULT_HALF_LIFE_DAYS = 90


# ═══════════════════════════════════════════════════════════════════════════════
# SIDECAR DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackStore:
    """Sidecar DuckDB database for feedback and personalization state."""

    def __init__(self, tenant_id: str, data_dir: str | Path = "data"):
        self.tenant_id = tenant_id
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / f"{tenant_id}.feedback.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                event_id        VARCHAR PRIMARY KEY,
                tenant_id       VARCHAR NOT NULL,
                user_id         VARCHAR NOT NULL,
                entity_type     VARCHAR NOT NULL,
                entity_id       VARCHAR NOT NULL,
                action          VARCHAR NOT NULL,
                context         JSON DEFAULT '{}',
                old_value       JSON DEFAULT '{}',
                new_value       JSON DEFAULT '{}',
                source_doc_id   VARCHAR,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id      VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS personalization_weights (
                weight_id       VARCHAR PRIMARY KEY,
                tenant_id       VARCHAR NOT NULL,
                entity_type     VARCHAR NOT NULL,
                entity_id       VARCHAR NOT NULL,
                weight          DOUBLE DEFAULT 1.0,
                adjustment      DOUBLE DEFAULT 0.0,
                feedback_count  INTEGER DEFAULT 0,
                last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS virtual_clusters (
                cluster_id      VARCHAR PRIMARY KEY,
                tenant_id       VARCHAR NOT NULL,
                label           VARCHAR NOT NULL,
                description     VARCHAR DEFAULT '',
                centroid        BLOB,
                source_action   VARCHAR DEFAULT 'user_created',
                member_count    INTEGER DEFAULT 0,
                created_by      VARCHAR,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS virtual_assignments (
                assignment_id   VARCHAR PRIMARY KEY,
                tenant_id       VARCHAR NOT NULL,
                chunk_id        VARCHAR,
                clause_id       VARCHAR,
                global_cluster_id   VARCHAR,
                virtual_cluster_id  VARCHAR NOT NULL,
                similarity      DOUBLE,
                assigned_by     VARCHAR,
                assigned_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (virtual_cluster_id) REFERENCES virtual_clusters(cluster_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_overrides (
                override_id     VARCHAR PRIMARY KEY,
                tenant_id       VARCHAR NOT NULL,
                global_intent_label     VARCHAR NOT NULL,
                tenant_intent_label     VARCHAR NOT NULL,
                tenant_summary          VARCHAR DEFAULT '',
                apply_to_clause_types   JSON DEFAULT '[]',
                created_by      VARCHAR,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_config (
                tenant_id               VARCHAR PRIMARY KEY,
                high_confidence_threshold   DOUBLE DEFAULT 0.85,
                tentative_threshold         DOUBLE DEFAULT 0.60,
                outlier_min_similarity      DOUBLE DEFAULT 0.35,
                intent_match_threshold      DOUBLE DEFAULT 0.85,
                intent_temperature          DOUBLE DEFAULT 0.0,
                edge_decay_half_life_days   INTEGER DEFAULT 90,
                auto_apply_corrections      BOOLEAN DEFAULT FALSE,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure tenant_config row exists
        existing = self.conn.execute(
            "SELECT 1 FROM tenant_config WHERE tenant_id = ?",
            [self.tenant_id]
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO tenant_config (tenant_id) VALUES (?)",
                [self.tenant_id]
            )

    # ── Feedback Events ───────────────────────────────────────────────────

    def record_feedback(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        old_value: dict | None = None,
        new_value: dict | None = None,
        context: dict | None = None,
        source_doc_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Record a feedback event and update personalization weights."""
        event_id = f"fb_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO feedback_events
               (event_id, tenant_id, user_id, entity_type, entity_id, action,
                context, old_value, new_value, source_doc_id, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                event_id, self.tenant_id, user_id,
                entity_type, entity_id, action,
                json.dumps(context or {}),
                json.dumps(old_value or {}),
                json.dumps(new_value or {}),
                source_doc_id, session_id,
            ],
        )
        # Update the weight matrix
        self._update_weight(entity_type, entity_id, action)
        logger.info(
            f"Feedback recorded: {action} on {entity_type}/{entity_id} "
            f"by user={user_id}"
        )
        return event_id

    def get_feedback_history(
        self,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query feedback events with optional filters."""
        query = "SELECT * FROM feedback_events WHERE tenant_id = ?"
        params: list = [self.tenant_id]
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).df().to_dict("records")

    # ── Weight Matrix ─────────────────────────────────────────────────────

    def _update_weight(self, entity_type: str, entity_id: str, action: str):
        """Update personalization weight based on feedback action."""
        delta = {
            "correct": -0.15,   # user corrected → reduce confidence in pipeline output
            "reject": -0.25,    # user rejected → stronger negative signal
            "refine": -0.05,    # minor tweak → small reduction
            "approve": +0.10,   # user confirmed → boost confidence
            "link": +0.20,      # user linked → strengthen relationship
            "unlink": -0.30,    # user unlinked → weaken relationship
            "merge": 0.0,       # handled separately
            "split": 0.0,       # handled separately
        }.get(action, 0.0)

        weight_id = f"w_{entity_type}_{entity_id}"
        existing = self.conn.execute(
            "SELECT weight, feedback_count FROM personalization_weights "
            "WHERE weight_id = ?",
            [weight_id],
        ).fetchone()

        if existing:
            old_weight, count = existing
            new_weight = max(0.0, min(2.0, old_weight + delta))
            self.conn.execute(
                """UPDATE personalization_weights
                   SET weight = ?, adjustment = adjustment + ?,
                       feedback_count = feedback_count + 1,
                       last_updated = CURRENT_TIMESTAMP
                   WHERE weight_id = ?""",
                [new_weight, delta, weight_id],
            )
        else:
            new_weight = max(0.0, min(2.0, 1.0 + delta))
            self.conn.execute(
                """INSERT INTO personalization_weights
                   (weight_id, tenant_id, entity_type, entity_id,
                    weight, adjustment, feedback_count)
                   VALUES (?, ?, ?, ?, ?, ?, 1)""",
                [weight_id, self.tenant_id, entity_type, entity_id,
                 new_weight, delta],
            )

    def get_weight(self, entity_type: str, entity_id: str) -> float:
        """Get the personalization weight for an entity (1.0 = neutral)."""
        weight_id = f"w_{entity_type}_{entity_id}"
        row = self.conn.execute(
            "SELECT weight, last_updated FROM personalization_weights "
            "WHERE weight_id = ?",
            [weight_id],
        ).fetchone()
        if not row:
            return 1.0
        weight, last_updated = row
        # Apply temporal decay
        return self._apply_decay(weight, last_updated)

    def get_weights_batch(self, entity_type: str) -> dict[str, float]:
        """Get all weights for an entity type."""
        rows = self.conn.execute(
            """SELECT entity_id, weight, last_updated
               FROM personalization_weights
               WHERE tenant_id = ? AND entity_type = ?""",
            [self.tenant_id, entity_type],
        ).fetchall()
        return {
            eid: self._apply_decay(w, ts)
            for eid, w, ts in rows
        }

    def _apply_decay(self, weight: float, last_updated) -> float:
        """Apply temporal decay — recent feedback weighs more."""
        config = self.get_tenant_config()
        half_life = config.get("edge_decay_half_life_days", DEFAULT_HALF_LIFE_DAYS)

        if last_updated is None:
            return weight

        now = datetime.now(timezone.utc)
        if hasattr(last_updated, "tzinfo") and last_updated.tzinfo is None:
            from datetime import timezone as tz
            last_updated = last_updated.replace(tzinfo=tz.utc)
        elif not hasattr(last_updated, "tzinfo"):
            return weight

        days_elapsed = (now - last_updated).total_seconds() / 86400.0
        # Exponential decay toward neutral (1.0)
        deviation = weight - 1.0
        decay_factor = math.pow(0.5, days_elapsed / half_life)
        return 1.0 + deviation * decay_factor

    # ── Tenant Config ─────────────────────────────────────────────────────

    def get_tenant_config(self) -> dict:
        row = self.conn.execute(
            "SELECT * FROM tenant_config WHERE tenant_id = ?",
            [self.tenant_id],
        ).fetchone()
        if not row:
            return {}
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, row))

    def update_tenant_config(self, **kwargs):
        """Update tenant-specific thresholds."""
        allowed = {
            "high_confidence_threshold", "tentative_threshold",
            "outlier_min_similarity", "intent_match_threshold",
            "intent_temperature", "edge_decay_half_life_days",
            "auto_apply_corrections",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [self.tenant_id]
        self.conn.execute(
            f"UPDATE tenant_config SET {set_clause}, "
            f"updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ?",
            vals,
        )

    # ── Virtual Clusters ──────────────────────────────────────────────────

    def create_virtual_cluster(
        self,
        label: str,
        user_id: str,
        description: str = "",
        centroid: bytes | None = None,
    ) -> str:
        cluster_id = f"vc_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """INSERT INTO virtual_clusters
               (cluster_id, tenant_id, label, description, centroid,
                created_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [cluster_id, self.tenant_id, label, description,
             centroid, user_id],
        )
        return cluster_id

    def assign_to_virtual_cluster(
        self,
        virtual_cluster_id: str,
        user_id: str,
        chunk_id: str | None = None,
        clause_id: str | None = None,
        global_cluster_id: str | None = None,
        similarity: float | None = None,
    ) -> str:
        assignment_id = f"va_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """INSERT INTO virtual_assignments
               (assignment_id, tenant_id, chunk_id, clause_id,
                global_cluster_id, virtual_cluster_id, similarity, assigned_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [assignment_id, self.tenant_id, chunk_id, clause_id,
             global_cluster_id, virtual_cluster_id, similarity, user_id],
        )
        # Update member count
        self.conn.execute(
            """UPDATE virtual_clusters
               SET member_count = member_count + 1
               WHERE cluster_id = ?""",
            [virtual_cluster_id],
        )
        return assignment_id

    def get_virtual_clusters(self) -> list[dict]:
        return self.conn.execute(
            "SELECT * FROM virtual_clusters WHERE tenant_id = ? "
            "ORDER BY member_count DESC",
            [self.tenant_id],
        ).df().to_dict("records")

    # ── Intent Overrides ──────────────────────────────────────────────────

    def set_intent_override(
        self,
        global_label: str,
        tenant_label: str,
        user_id: str,
        summary: str = "",
        clause_types: list[str] | None = None,
    ) -> str:
        override_id = f"io_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """INSERT INTO intent_overrides
               (override_id, tenant_id, global_intent_label,
                tenant_intent_label, tenant_summary,
                apply_to_clause_types, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [override_id, self.tenant_id, global_label, tenant_label,
             summary, json.dumps(clause_types or []), user_id],
        )
        return override_id

    def get_intent_overrides(self) -> dict[str, dict]:
        """Returns {global_label: {tenant_label, summary, clause_types}}."""
        rows = self.conn.execute(
            """SELECT global_intent_label, tenant_intent_label,
                      tenant_summary, apply_to_clause_types
               FROM intent_overrides
               WHERE tenant_id = ?
               ORDER BY created_at DESC""",
            [self.tenant_id],
        ).fetchall()
        overrides = {}
        for gl, tl, ts, ct in rows:
            if gl not in overrides:  # most recent wins
                overrides[gl] = {
                    "tenant_label": tl,
                    "summary": ts,
                    "clause_types": json.loads(ct) if ct else [],
                }
        return overrides

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        events = self.conn.execute(
            "SELECT count(*) FROM feedback_events WHERE tenant_id = ?",
            [self.tenant_id],
        ).fetchone()[0]
        weights = self.conn.execute(
            "SELECT count(*) FROM personalization_weights WHERE tenant_id = ?",
            [self.tenant_id],
        ).fetchone()[0]
        vclusters = self.conn.execute(
            "SELECT count(*) FROM virtual_clusters WHERE tenant_id = ?",
            [self.tenant_id],
        ).fetchone()[0]
        overrides = self.conn.execute(
            "SELECT count(*) FROM intent_overrides WHERE tenant_id = ?",
            [self.tenant_id],
        ).fetchone()[0]
        return {
            "tenant_id": self.tenant_id,
            "feedback_events": events,
            "personalization_weights": weights,
            "virtual_clusters": vclusters,
            "intent_overrides": overrides,
        }

    def close(self):
        self.conn.close()
