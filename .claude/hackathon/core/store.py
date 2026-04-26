"""
DuckDB storage layer — replaces legacy AIDB gRPC + Azure Blob + Redis.

Legacy pain points solved:
- AIDB gRPC: 4+ network hops for embedding storage/retrieval → DuckDB local reads
- Azure Blob: Artifact persistence with marker files → DuckDB tables with status columns
- Redis: Job status tracking → DuckDB transaction log
- Field summary embeddings (1 embedding/field/agreement) → clause-level embeddings stored locally

This is the integration contract between Person A (pipeline) and Person B (UI).
Person A writes. Person B reads. Schema is agreed up front.
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np


def _generate_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class ClusteringStore:
    """Local DuckDB store for the entire V2 pipeline state."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        """Create all tables. Idempotent — safe to call on every startup."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS domains (
                domain_id       VARCHAR PRIMARY KEY,
                label           VARCHAR NOT NULL,
                description     VARCHAR,
                agreement_count INTEGER DEFAULT 0,
                confidence      DOUBLE DEFAULT 0.0,
                centroid        BLOB,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agreements (
                agreement_id      VARCHAR PRIMARY KEY,
                filename          VARCHAR NOT NULL,
                domain_id         VARCHAR,
                raw_text          VARCHAR,
                summary_text      VARCHAR,
                domain_confidence DOUBLE,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      VARCHAR PRIMARY KEY,
                agreement_id  VARCHAR,
                clause_id     VARCHAR,
                chunk_text    VARCHAR NOT NULL,
                chunk_index   INTEGER,
                embedding     BLOB,
                token_count   INTEGER DEFAULT 0
            )
        """)

        # ── nupunkt structural metadata (Phase 1) ────────────────────────
        try:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS block_id VARCHAR")
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sentence_ids VARCHAR")
        except Exception:
            pass  # Column already exists

        # ── KeyBERT clause-type metadata (Phase 2) ───────────────────────
        try:
            self.conn.execute("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS dominant_clause_type VARCHAR")
            self.conn.execute("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS clause_type_distribution VARCHAR")
        except Exception:
            pass
        try:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS keybert_scores VARCHAR")
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS augmented_text VARCHAR")
        except Exception:
            pass

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clauses (
                clause_id       VARCHAR PRIMARY KEY,
                agreement_id    VARCHAR NOT NULL,
                clause_number   VARCHAR,
                clause_title    VARCHAR,
                heading_level   INTEGER DEFAULT 0,
                clause_index    INTEGER DEFAULT 0,
                clause_type_id  VARCHAR,
                full_text       VARCHAR,
                token_count     INTEGER DEFAULT 0
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id    VARCHAR PRIMARY KEY,
                domain_id     VARCHAR,
                label         VARCHAR,
                description   VARCHAR,
                keywords      VARCHAR,    -- JSON array
                aliases       VARCHAR,    -- JSON array of merged alias labels
                chunk_count   INTEGER DEFAULT 0,
                agreement_count INTEGER DEFAULT 0,
                quality_score DOUBLE,
                centroid      BLOB,
                merged_from   VARCHAR,    -- JSON array of original cluster_ids
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_assignments (
                chunk_id      VARCHAR,
                cluster_id    VARCHAR,
                probability   DOUBLE,
                is_outlier    BOOLEAN DEFAULT false,
                PRIMARY KEY (chunk_id, cluster_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS field_definitions (
                field_id    VARCHAR PRIMARY KEY,
                cluster_id  VARCHAR,
                name        VARCHAR NOT NULL,
                field_type  VARCHAR NOT NULL,
                description VARCHAR,
                source      VARCHAR DEFAULT 'auto_discovered',
                examples    VARCHAR,    -- JSON array of gold examples
                confidence  DOUBLE DEFAULT 0.0,
                span_text   VARCHAR,    -- single evidence span
                span_texts  VARCHAR,    -- JSON array of evidence spans (composite fields)
                derivation  VARCHAR,    -- formula for derived fields
                iteration   INTEGER DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS domain_field_definitions (
                field_id    VARCHAR PRIMARY KEY,
                domain_id   VARCHAR NOT NULL,
                name        VARCHAR NOT NULL,
                field_type  VARCHAR NOT NULL,
                description VARCHAR,
                source      VARCHAR DEFAULT 'domain_discovery',
                examples    VARCHAR,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                extraction_id   VARCHAR PRIMARY KEY,
                agreement_id    VARCHAR,
                field_id        VARCHAR,
                value           VARCHAR,
                confidence      DOUBLE,
                source_chunk_id VARCHAR,
                was_corrected   BOOLEAN DEFAULT FALSE,
                original_value  VARCHAR,
                extracted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_extractions (
                extraction_id   VARCHAR PRIMARY KEY,
                intent_id       VARCHAR NOT NULL,
                agreement_id    VARCHAR,
                value           VARCHAR,
                confidence      DOUBLE,
                source_chunk_id VARCHAR,
                extracted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clause_extractions (
                extraction_id   VARCHAR PRIMARY KEY,
                clause_type     VARCHAR NOT NULL,
                cluster_id      VARCHAR,
                agreement_id    VARCHAR,
                value           VARCHAR,
                confidence      DOUBLE,
                source_chunk_id VARCHAR,
                extracted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS obligation_extractions (
                extraction_id   VARCHAR PRIMARY KEY,
                obligation_key  VARCHAR NOT NULL,
                cluster_id      VARCHAR,
                agreement_id    VARCHAR,
                value           VARCHAR,
                confidence      DOUBLE,
                source_chunk_id VARCHAR,
                extracted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clause type definitions — persists RLM-discovered clause types
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clause_type_definitions (
                clause_type_def_id  VARCHAR PRIMARY KEY,
                cluster_id          VARCHAR NOT NULL,
                name                VARCHAR NOT NULL,
                emergent_description VARCHAR,
                span_text           VARCHAR,
                span_texts          VARCHAR DEFAULT '[]',   -- JSON array
                confidence          DOUBLE DEFAULT 0.0,
                source              VARCHAR DEFAULT 'agent',
                iteration           INTEGER DEFAULT 0,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Obligation definitions — persists RLM-discovered obligations
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS obligation_definitions (
                obligation_def_id   VARCHAR PRIMARY KEY,
                cluster_id          VARCHAR NOT NULL,
                obligor             VARCHAR,
                action              VARCHAR NOT NULL,
                trigger_condition   VARCHAR,
                deadline            VARCHAR,
                clause_ref          VARCHAR,
                clause_refs         VARCHAR DEFAULT '[]',   -- JSON array
                confidence          DOUBLE DEFAULT 0.0,
                source              VARCHAR DEFAULT 'agent',
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS composite_definitions (
                field_id         VARCHAR PRIMARY KEY,
                name             VARCHAR NOT NULL,
                expression       VARCHAR NOT NULL,
                source_field_ids VARCHAR NOT NULL,   -- JSON array
                result_type      VARCHAR DEFAULT 'float',
                cluster_id       VARCHAR,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                correction_id VARCHAR PRIMARY KEY,
                extraction_id VARCHAR,
                field_id      VARCHAR,
                old_value     VARCHAR,
                new_value     VARCHAR,
                corrected_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # V2-specific: cluster centroids for incremental assignment (FIELD-801)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_centroids (
                cluster_id   VARCHAR PRIMARY KEY,
                centroid     BLOB NOT NULL,
                chunk_count  INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # V2-specific: novel document buffer for incremental assignment
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS novel_documents (
                doc_id     VARCHAR PRIMARY KEY,
                raw_text   VARCHAR,
                embedding  BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pipeline run log (replaces Redis job tracking + Azure Blob markers)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id     VARCHAR PRIMARY KEY,
                status     VARCHAR DEFAULT 'running',
                stage      VARCHAR,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at   TIMESTAMP,
                metadata   VARCHAR    -- JSON
            )
        """)

        # Intent type registry — growing catalog of discovered legal intents.
        # Each type represents a distinct legal concept (e.g., "obligation_to_indemnify",
        # "governing_law_choice"). Types are discovered dynamically, not predefined.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_types (
                intent_type_id       VARCHAR PRIMARY KEY,
                label                VARCHAR NOT NULL,
                description          VARCHAR,
                sample_attributes    VARCHAR,
                aliases              VARCHAR DEFAULT '[]',
                occurrence_count     INTEGER DEFAULT 1,
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clause intent instances — each row is one intent found in one clause.
        # A clause can have 0 intents (table of contents) or many (omnibus section).
        # Attributes are a flexible JSON dict of intent-specific properties.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS clause_intents (
                intent_id            VARCHAR PRIMARY KEY,
                clause_id            VARCHAR NOT NULL,
                agreement_id         VARCHAR NOT NULL,
                clause_type_id       VARCHAR,
                intent_type_id       VARCHAR,
                intent_label         VARCHAR,
                summary              VARCHAR,
                party_from           VARCHAR,
                party_to             VARCHAR,
                attributes           VARCHAR DEFAULT '{}',
                clause_ref           VARCHAR,
                clause_refs          VARCHAR DEFAULT '[]',    -- JSON array
                chunk_idx            INTEGER DEFAULT -1,
                chunk_ids            VARCHAR DEFAULT '[]',    -- JSON array
                confidence           DOUBLE DEFAULT 0.0,
                source               VARCHAR DEFAULT 'agent',
                intent_hash          VARCHAR,
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # EVoC native cluster layers — every resolution EVoC produces,
        # not just the selected one.  One row per (layer, item).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_layers (
                clustering_run  VARCHAR NOT NULL,     -- 'clause' | 'macro' | 'micro_{domain}'
                layer_index     INTEGER NOT NULL,
                item_index      INTEGER NOT NULL,     -- clause/doc ordinal
                item_id         VARCHAR,              -- clause_id / agreement_id
                cluster_label   INTEGER NOT NULL,     -- raw EVoC label (-1 = outlier)
                PRIMARY KEY (clustering_run, layer_index, item_index)
            )
        """)

        # Layer metadata — one row per (clustering_run, layer_index)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_layer_meta (
                clustering_run    VARCHAR NOT NULL,
                layer_index       INTEGER NOT NULL,
                n_clusters        INTEGER,
                n_outliers        INTEGER,
                persistence_score DOUBLE,
                silhouette_score  DOUBLE,
                cosine_score      DOUBLE,
                composite_score   DOUBLE,
                is_selected       BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (clustering_run, layer_index)
            )
        """)

        # ── ARM tables (Phase 3) ─────────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arm_rules (
                rule_id          VARCHAR PRIMARY KEY,
                antecedent       VARCHAR NOT NULL,
                consequent       VARCHAR NOT NULL,
                rule_type        VARCHAR NOT NULL,
                support          DOUBLE,
                confidence       DOUBLE,
                lift             DOUBLE,
                cluster_id       VARCHAR,
                pipeline_run_id  VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arm_term_packages (
                package_id       VARCHAR PRIMARY KEY,
                clause_types     VARCHAR NOT NULL,
                support          DOUBLE,
                size             INTEGER,
                cluster_id       VARCHAR,
                pipeline_run_id  VARCHAR
            )
        """)

    # ── Serialization helpers ──────────────────────────────────────────────────

    @staticmethod
    def _serialize_embedding(arr: np.ndarray) -> bytes:
        return arr.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(-1, dim) if blob else np.array([])

    @staticmethod
    def _deserialize_single_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32) if blob else np.array([])

    # ── Domain CRUD ────────────────────────────────────────────────────────────

    def upsert_domain(self, domain_id: str, label: str, description: str = "",
                      agreement_count: int = 0, confidence: float = 0.0,
                      centroid: np.ndarray | None = None):
        centroid_blob = self._serialize_embedding(centroid) if centroid is not None else None
        self.conn.execute("""
            INSERT OR REPLACE INTO domains (domain_id, label, description, agreement_count, confidence, centroid)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [domain_id, label, description, agreement_count, confidence, centroid_blob])

    def get_domains(self) -> list[dict]:
        return self.conn.execute("SELECT * FROM domains ORDER BY agreement_count DESC").fetchdf().to_dict("records")

    # ── Agreement CRUD ─────────────────────────────────────────────────────────

    def insert_agreement(self, agreement_id: str, filename: str, raw_text: str,
                         summary_text: str = "", domain_id: str | None = None,
                         domain_confidence: float | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO agreements
            (agreement_id, filename, domain_id, raw_text, summary_text, domain_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [agreement_id, filename, domain_id, raw_text, summary_text, domain_confidence])

    def get_agreements(self, domain_id: str | None = None) -> list[dict]:
        if domain_id:
            return self.conn.execute(
                "SELECT * FROM agreements WHERE domain_id = ? ORDER BY filename",
                [domain_id]
            ).fetchdf().to_dict("records")
        return self.conn.execute("SELECT * FROM agreements ORDER BY filename").fetchdf().to_dict("records")

    def get_agreement(self, agreement_id: str) -> dict | None:
        rows = self.conn.execute(
            "SELECT * FROM agreements WHERE agreement_id = ?", [agreement_id]
        ).fetchdf().to_dict("records")
        return rows[0] if rows else None

    def update_agreement_domain(self, agreement_id: str, domain_id: str, confidence: float):
        self.conn.execute(
            "UPDATE agreements SET domain_id = ?, domain_confidence = ? WHERE agreement_id = ?",
            [domain_id, confidence, agreement_id]
        )

    # ── Chunk CRUD ─────────────────────────────────────────────────────────────

    def insert_chunk(self, chunk_id: str, agreement_id: str, chunk_text: str,
                     chunk_index: int, embedding: np.ndarray | None = None,
                     token_count: int = 0, clause_id: str | None = None):
        emb_blob = self._serialize_embedding(embedding) if embedding is not None else None
        self.conn.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, agreement_id, clause_id, chunk_text, chunk_index, embedding, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [chunk_id, agreement_id, clause_id, chunk_text, chunk_index, emb_blob, token_count])

    def get_chunks(self, agreement_id: str | None = None, cluster_id: str | None = None,
                   include_outliers: bool = True) -> list[dict]:
        if cluster_id:
            outlier_filter = "" if include_outliers else " AND (ca.is_outlier = false OR ca.is_outlier IS NULL)"
            rows = self.conn.execute(f"""
                SELECT c.*, ca.is_outlier FROM chunks c
                JOIN cluster_assignments ca ON c.chunk_id = ca.chunk_id
                WHERE ca.cluster_id = ?{outlier_filter}
                ORDER BY c.chunk_index
            """, [cluster_id]).fetchdf().to_dict("records")
        elif agreement_id:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE agreement_id = ? ORDER BY chunk_index",
                [agreement_id]
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute("SELECT * FROM chunks ORDER BY agreement_id, chunk_index").fetchdf().to_dict("records")
        return rows

    def get_outlier_chunks(self, cluster_id: str | None = None) -> list[dict]:
        """Return chunks assigned as outliers, optionally filtered to a cluster."""
        if cluster_id:
            return self.conn.execute("""
                SELECT c.*, ca.probability as outlier_similarity, ca.cluster_id as nearest_cluster_id
                FROM chunks c
                JOIN cluster_assignments ca ON c.chunk_id = ca.chunk_id
                WHERE ca.is_outlier = true AND ca.cluster_id = ?
                ORDER BY ca.probability ASC
            """, [cluster_id]).fetchdf().to_dict("records")
        return self.conn.execute("""
            SELECT c.*, ca.probability as outlier_similarity, ca.cluster_id as nearest_cluster_id
            FROM chunks c
            JOIN cluster_assignments ca ON c.chunk_id = ca.chunk_id
            WHERE ca.is_outlier = true
            ORDER BY ca.probability ASC
        """).fetchdf().to_dict("records")

    def get_all_chunk_embeddings(self, domain_id: str | None = None) -> tuple[list[str], np.ndarray]:
        """Return (chunk_ids, embeddings_matrix) for all chunks, optionally filtered by domain."""
        if domain_id:
            rows = self.conn.execute("""
                SELECT c.chunk_id, c.embedding FROM chunks c
                JOIN agreements a ON c.agreement_id = a.agreement_id
                WHERE a.domain_id = ? AND c.embedding IS NOT NULL
            """, [domain_id]).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows:
            return [], np.array([])
        chunk_ids = [r[0] for r in rows]
        embeddings = np.array([self._deserialize_single_embedding(r[1]) for r in rows])
        return chunk_ids, embeddings

    # ── Clause CRUD ────────────────────────────────────────────────────────────

    def insert_clause(self, clause_id: str, agreement_id: str,
                      clause_number: str = "", clause_title: str = "",
                      heading_level: int = 0, clause_index: int = 0,
                      full_text: str = "", token_count: int = 0):
        self.conn.execute("""
            INSERT OR REPLACE INTO clauses
            (clause_id, agreement_id, clause_number, clause_title, heading_level,
             clause_index, full_text, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [clause_id, agreement_id, clause_number, clause_title,
              heading_level, clause_index, full_text, token_count])

    def update_clause_type(self, clause_id: str, clause_type_id: str):
        self.conn.execute(
            "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ?",
            [clause_type_id, clause_id]
        )

    def get_clauses(self, agreement_id: str | None = None) -> list[dict]:
        if agreement_id:
            return self.conn.execute(
                "SELECT * FROM clauses WHERE agreement_id = ? ORDER BY clause_index",
                [agreement_id]
            ).fetchdf().to_dict("records")
        return self.conn.execute(
            "SELECT * FROM clauses ORDER BY agreement_id, clause_index"
        ).fetchdf().to_dict("records")

    # ── Cluster CRUD ───────────────────────────────────────────────────────────

    def upsert_cluster(self, cluster_id: str, domain_id: str, label: str = "",
                       description: str = "", keywords: list[str] | None = None,
                       chunk_count: int = 0, agreement_count: int = 0,
                       quality_score: float | None = None, centroid: np.ndarray | None = None,
                       aliases: list[str] | None = None, merged_from: list[str] | None = None):
        centroid_blob = self._serialize_embedding(centroid) if centroid is not None else None
        self.conn.execute("""
            INSERT OR REPLACE INTO clusters
            (cluster_id, domain_id, label, description, keywords, chunk_count,
             agreement_count, quality_score, centroid, aliases, merged_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            cluster_id, domain_id, label, description,
            json.dumps(keywords or []), chunk_count, agreement_count,
            quality_score, centroid_blob,
            json.dumps(aliases or []), json.dumps(merged_from or [])
        ])

    def get_clusters(self, domain_id: str | None = None) -> list[dict]:
        if domain_id:
            rows = self.conn.execute(
                "SELECT * FROM clusters WHERE domain_id = ? ORDER BY chunk_count DESC",
                [domain_id]
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute("SELECT * FROM clusters ORDER BY domain_id, chunk_count DESC").fetchdf().to_dict("records")
        for r in rows:
            r["keywords"] = json.loads(r["keywords"]) if r.get("keywords") else []
            r["aliases"] = json.loads(r["aliases"]) if r.get("aliases") else []
            r["merged_from"] = json.loads(r["merged_from"]) if r.get("merged_from") else []
        return rows

    def get_cluster(self, cluster_id: str) -> dict | None:
        rows = self.conn.execute("SELECT * FROM clusters WHERE cluster_id = ?", [cluster_id]).fetchdf().to_dict("records")
        if not rows:
            return None
        r = rows[0]
        r["keywords"] = json.loads(r["keywords"]) if r.get("keywords") else []
        r["aliases"] = json.loads(r["aliases"]) if r.get("aliases") else []
        r["merged_from"] = json.loads(r["merged_from"]) if r.get("merged_from") else []
        return r

    # ── Cluster Assignments ────────────────────────────────────────────────────

    def insert_cluster_assignment(self, chunk_id: str, cluster_id: str,
                                   probability: float = 1.0, is_outlier: bool = False):
        self.conn.execute("""
            INSERT OR REPLACE INTO cluster_assignments (chunk_id, cluster_id, probability, is_outlier)
            VALUES (?, ?, ?, ?)
        """, [chunk_id, cluster_id, probability, is_outlier])

    # ── Field Definitions ──────────────────────────────────────────────────────

    def upsert_field(self, field_id: str, cluster_id: str, name: str, field_type: str,
                     description: str = "", source: str = "auto_discovered",
                     examples: list | None = None, confidence: float = 0.0,
                     span_text: str = "", span_texts: list | None = None,
                     derivation: str = "", iteration: int = 0):
        self.conn.execute("""
            INSERT OR REPLACE INTO field_definitions
            (field_id, cluster_id, name, field_type, description, source, examples,
             confidence, span_text, span_texts, derivation, iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [field_id, cluster_id, name, field_type, description, source,
              json.dumps(examples or []), confidence, span_text,
              json.dumps(span_texts or []), derivation, iteration])

    def get_fields(self, cluster_id: str | None = None) -> list[dict]:
        if cluster_id:
            rows = self.conn.execute(
                "SELECT * FROM field_definitions WHERE cluster_id = ? ORDER BY name", [cluster_id]
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute("SELECT * FROM field_definitions ORDER BY cluster_id, name").fetchdf().to_dict("records")
        for r in rows:
            r["examples"] = json.loads(r["examples"]) if r.get("examples") else []
        return rows

    def update_field(self, field_id: str, **kwargs):
        """Update field definition attributes."""
        allowed = {"name", "field_type", "description", "examples", "source",
                   "confidence", "span_text", "span_texts", "derivation", "iteration"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "examples" in updates and isinstance(updates["examples"], list):
            updates["examples"] = json.dumps(updates["examples"])
        if "span_texts" in updates and isinstance(updates["span_texts"], list):
            updates["span_texts"] = json.dumps(updates["span_texts"])
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [field_id]
        self.conn.execute(f"UPDATE field_definitions SET {set_clause} WHERE field_id = ?", values)

    def delete_field(self, field_id: str):
        self.conn.execute("DELETE FROM field_definitions WHERE field_id = ?", [field_id])

    # ── Domain Field Definitions (two-tier architecture) ───────────────────────

    def upsert_domain_field(self, field_id: str, domain_id: str, name: str,
                            field_type: str, description: str = "",
                            source: str = "domain_discovery",
                            examples: list | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO domain_field_definitions
            (field_id, domain_id, name, field_type, description, source, examples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [field_id, domain_id, name, field_type, description, source,
              json.dumps(examples or [])])

    def get_domain_fields(self, domain_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM domain_field_definitions WHERE domain_id = ? ORDER BY name",
            [domain_id],
        ).fetchdf().to_dict("records")
        for r in rows:
            r["examples"] = json.loads(r["examples"]) if r.get("examples") else []
        return rows

    # ── Extractions ────────────────────────────────────────────────────────────

    def insert_extraction(self, extraction_id: str, agreement_id: str, field_id: str,
                          value: str | None, confidence: float = 0.0,
                          source_chunk_id: str | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO extractions
            (extraction_id, agreement_id, field_id, value, confidence, source_chunk_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [extraction_id, agreement_id, field_id, value, confidence, source_chunk_id])

    def get_extractions(self, agreement_id: str | None = None,
                        field_id: str | None = None,
                        cluster_id: str | None = None) -> list[dict]:
        if cluster_id:
            return self.conn.execute("""
                SELECT e.* FROM extractions e
                JOIN field_definitions f ON e.field_id = f.field_id
                WHERE f.cluster_id = ?
                ORDER BY e.agreement_id
            """, [cluster_id]).fetchdf().to_dict("records")
        conditions, params = [], []
        if agreement_id:
            conditions.append("agreement_id = ?")
            params.append(agreement_id)
        if field_id:
            conditions.append("field_id = ?")
            params.append(field_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return self.conn.execute(
            f"SELECT * FROM extractions {where} ORDER BY agreement_id", params
        ).fetchdf().to_dict("records")

    def correct_extraction(self, extraction_id: str, new_value: str):
        """Apply a user correction and log it."""
        row = self.conn.execute(
            "SELECT value, field_id FROM extractions WHERE extraction_id = ?", [extraction_id]
        ).fetchone()
        if not row:
            return
        old_value, field_id = row
        self.conn.execute("""
            UPDATE extractions SET value = ?, was_corrected = TRUE, original_value = ?
            WHERE extraction_id = ?
        """, [new_value, old_value, extraction_id])
        self.conn.execute("""
            INSERT INTO corrections (correction_id, extraction_id, field_id, old_value, new_value)
            VALUES (?, ?, ?, ?, ?)
        """, [_generate_id("corr_"), extraction_id, field_id, old_value, new_value])

    # ── Intent Extractions ─────────────────────────────────────────────────

    def insert_intent_extraction(self, extraction_id: str, intent_id: str,
                                 agreement_id: str, value: str | None,
                                 confidence: float = 0.0,
                                 source_chunk_id: str | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO intent_extractions
            (extraction_id, intent_id, agreement_id, value, confidence, source_chunk_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [extraction_id, intent_id, agreement_id, value, confidence, source_chunk_id])

    # ── Clause Extractions ──────────────────────────────────────────────────

    def insert_clause_extraction(self, extraction_id: str, clause_type: str,
                                 cluster_id: str, agreement_id: str,
                                 value: str | None, confidence: float = 0.0,
                                 source_chunk_id: str | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO clause_extractions
            (extraction_id, clause_type, cluster_id, agreement_id, value, confidence, source_chunk_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [extraction_id, clause_type, cluster_id, agreement_id, value, confidence, source_chunk_id])

    # ── Obligation Extractions ─────────────────────────────────────────────

    def insert_obligation_extraction(self, extraction_id: str, obligation_key: str,
                                     cluster_id: str, agreement_id: str,
                                     value: str | None, confidence: float = 0.0,
                                     source_chunk_id: str | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO obligation_extractions
            (extraction_id, obligation_key, cluster_id, agreement_id, value, confidence, source_chunk_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [extraction_id, obligation_key, cluster_id, agreement_id, value, confidence, source_chunk_id])

    # ── Clause Type Definitions ────────────────────────────────────────────

    def insert_clause_type_definition(self, clause_type_def_id: str, cluster_id: str,
                                      name: str, emergent_description: str = "",
                                      span_text: str = "", span_texts: list | None = None,
                                      confidence: float = 0.0, source: str = "agent",
                                      iteration: int = 0):
        self.conn.execute("""
            INSERT OR REPLACE INTO clause_type_definitions
            (clause_type_def_id, cluster_id, name, emergent_description,
             span_text, span_texts, confidence, source, iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [clause_type_def_id, cluster_id, name, emergent_description,
              span_text, json.dumps(span_texts or []), confidence, source, iteration])

    def get_clause_type_definitions(self, cluster_id: str | None = None) -> list[dict]:
        if cluster_id:
            rows = self.conn.execute(
                "SELECT * FROM clause_type_definitions WHERE cluster_id = ? ORDER BY name",
                [cluster_id],
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute(
                "SELECT * FROM clause_type_definitions ORDER BY cluster_id, name"
            ).fetchdf().to_dict("records")
        for r in rows:
            r["span_texts"] = json.loads(r["span_texts"]) if r.get("span_texts") else []
        return rows

    # ── Obligation Definitions ─────────────────────────────────────────────

    def insert_obligation_definition(self, obligation_def_id: str, cluster_id: str,
                                     obligor: str = "", action: str = "",
                                     trigger_condition: str | None = None,
                                     deadline: str | None = None,
                                     clause_ref: str = "",
                                     clause_refs: list | None = None,
                                     confidence: float = 0.0,
                                     source: str = "agent"):
        self.conn.execute("""
            INSERT OR REPLACE INTO obligation_definitions
            (obligation_def_id, cluster_id, obligor, action,
             trigger_condition, deadline, clause_ref, clause_refs,
             confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [obligation_def_id, cluster_id, obligor, action,
              trigger_condition, deadline, clause_ref,
              json.dumps(clause_refs or []), confidence, source])

    def get_obligation_definitions(self, cluster_id: str | None = None) -> list[dict]:
        if cluster_id:
            rows = self.conn.execute(
                "SELECT * FROM obligation_definitions WHERE cluster_id = ? ORDER BY obligor, action",
                [cluster_id],
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute(
                "SELECT * FROM obligation_definitions ORDER BY cluster_id, obligor"
            ).fetchdf().to_dict("records")
        for r in rows:
            r["clause_refs"] = json.loads(r["clause_refs"]) if r.get("clause_refs") else []
        return rows

    # ── Composite Definitions ──────────────────────────────────────────────────

    def upsert_composite(self, field_id: str, name: str, expression: str,
                         source_field_ids: list[str], result_type: str = "float",
                         cluster_id: str | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO composite_definitions
            (field_id, name, expression, source_field_ids, result_type, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [field_id, name, expression, json.dumps(source_field_ids), result_type, cluster_id])

    def get_composites(self, cluster_id: str | None = None) -> list[dict]:
        if cluster_id:
            rows = self.conn.execute(
                "SELECT * FROM composite_definitions WHERE cluster_id = ? ORDER BY name",
                [cluster_id]
            ).fetchdf().to_dict("records")
        else:
            rows = self.conn.execute(
                "SELECT * FROM composite_definitions ORDER BY name"
            ).fetchdf().to_dict("records")
        for r in rows:
            r["source_field_ids"] = json.loads(r["source_field_ids"]) if r.get("source_field_ids") else []
        return rows

    def delete_composite(self, field_id: str):
        self.conn.execute("DELETE FROM composite_definitions WHERE field_id = ?", [field_id])

    # ── Corrections / Feedback ─────────────────────────────────────────────────

    def get_corrections(self, field_id: str | None = None) -> list[dict]:
        if field_id:
            return self.conn.execute(
                "SELECT * FROM corrections WHERE field_id = ? ORDER BY corrected_at DESC",
                [field_id]
            ).fetchdf().to_dict("records")
        return self.conn.execute("SELECT * FROM corrections ORDER BY corrected_at DESC").fetchdf().to_dict("records")

    # ── Cluster Centroids (FIELD-801) ──────────────────────────────────────────

    def save_centroid(self, cluster_id: str, centroid: np.ndarray, chunk_count: int):
        self.conn.execute("""
            INSERT OR REPLACE INTO cluster_centroids (cluster_id, centroid, chunk_count, last_updated)
            VALUES (?, ?, ?, ?)
        """, [cluster_id, self._serialize_embedding(centroid), chunk_count, datetime.now()])

    def get_all_centroids(self) -> dict[str, tuple[np.ndarray, int]]:
        rows = self.conn.execute("SELECT cluster_id, centroid, chunk_count FROM cluster_centroids").fetchall()
        return {r[0]: (self._deserialize_single_embedding(r[1]), r[2]) for r in rows}

    # ── Novel Documents ────────────────────────────────────────────────────────

    def insert_novel_document(self, doc_id: str, raw_text: str, embedding: np.ndarray):
        self.conn.execute("""
            INSERT OR REPLACE INTO novel_documents (doc_id, raw_text, embedding)
            VALUES (?, ?, ?)
        """, [doc_id, raw_text, self._serialize_embedding(embedding)])

    def get_novel_documents(self) -> list[dict]:
        return self.conn.execute("SELECT * FROM novel_documents ORDER BY created_at").fetchdf().to_dict("records")

    def clear_novel_buffer(self):
        self.conn.execute("DELETE FROM novel_documents")

    # ── Intent Types Registry ──────────────────────────────────────────────────

    def upsert_intent_type(self, intent_type_id: str, label: str,
                           description: str = "",
                           sample_attributes: list[str] | None = None):
        self.conn.execute("""
            INSERT OR REPLACE INTO intent_types
            (intent_type_id, label, description, sample_attributes)
            VALUES (?, ?, ?, ?)
        """, [intent_type_id, label, description,
              json.dumps(sample_attributes or [])])

    def get_intent_types(self) -> list[dict]:
        return self.conn.execute(
            "SELECT * FROM intent_types ORDER BY occurrence_count DESC"
        ).fetchdf().to_dict("records")

    def increment_intent_type_count(self, intent_type_id: str):
        self.conn.execute(
            "UPDATE intent_types SET occurrence_count = occurrence_count + 1 "
            "WHERE intent_type_id = ?", [intent_type_id])

    def add_intent_type_alias(self, intent_type_id: str, alias: str):
        row = self.conn.execute(
            "SELECT aliases FROM intent_types WHERE intent_type_id = ?",
            [intent_type_id]).fetchone()
        if row:
            aliases = json.loads(row[0] or "[]")
            if alias not in aliases:
                aliases.append(alias)
                self.conn.execute(
                    "UPDATE intent_types SET aliases = ? WHERE intent_type_id = ?",
                    [json.dumps(aliases), intent_type_id])

    # ── Clause Intents ─────────────────────────────────────────────────────────

    def insert_clause_intent(self, intent_id: str, clause_id: str,
                             agreement_id: str, clause_type_id: str | None,
                             intent_type_id: str | None, intent_label: str,
                             summary: str = "", party_from: str = "none",
                             party_to: str = "none",
                             attributes: dict | None = None,
                             clause_ref: str = "",
                             clause_refs: list | None = None,
                             chunk_idx: int = -1,
                             chunk_ids: list | None = None,
                             confidence: float = 0.0,
                             source: str = "agent",
                             intent_hash: str = ""):
        self.conn.execute("""
            INSERT OR REPLACE INTO clause_intents
            (intent_id, clause_id, agreement_id, clause_type_id,
             intent_type_id, intent_label, summary, party_from, party_to,
             attributes, clause_ref, clause_refs, chunk_idx, chunk_ids,
             confidence, source, intent_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [intent_id, clause_id, agreement_id, clause_type_id,
              intent_type_id, intent_label, summary, party_from, party_to,
              json.dumps(attributes or {}), clause_ref,
              json.dumps(clause_refs or []), chunk_idx,
              json.dumps(chunk_ids or []), confidence, source, intent_hash])

    def get_clause_intents(self, clause_type_id: str | None = None,
                           agreement_id: str | None = None,
                           clause_id: str | None = None,
                           intent_type_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM clause_intents WHERE 1=1"
        params = []
        if clause_type_id:
            query += " AND clause_type_id = ?"
            params.append(clause_type_id)
        if agreement_id:
            query += " AND agreement_id = ?"
            params.append(agreement_id)
        if clause_id:
            query += " AND clause_id = ?"
            params.append(clause_id)
        if intent_type_id:
            query += " AND intent_type_id = ?"
            params.append(intent_type_id)
        query += " ORDER BY clause_id, intent_label"
        rows = self.conn.execute(query, params).fetchdf().to_dict("records")
        for r in rows:
            try:
                r["attributes"] = json.loads(r.get("attributes", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                r["attributes"] = {}
        return rows

    # ── Pipeline Runs ──────────────────────────────────────────────────────────

    def start_run(self, run_id: str, stage: str = "init"):
        self.conn.execute("""
            INSERT INTO pipeline_runs (run_id, status, stage) VALUES (?, 'running', ?)
        """, [run_id, stage])

    def update_run(self, run_id: str, stage: str, status: str = "running"):
        self.conn.execute("""
            UPDATE pipeline_runs SET stage = ?, status = ?, ended_at = CASE WHEN ? IN ('completed','failed') THEN CURRENT_TIMESTAMP ELSE ended_at END
            WHERE run_id = ?
        """, [stage, status, status, run_id])

    # ── Stats / Dashboard ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        domain_count = self.conn.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
        agreement_count = self.conn.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
        cluster_count = self.conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        field_count = self.conn.execute("SELECT COUNT(*) FROM field_definitions").fetchone()[0]
        extraction_count = self.conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
        correction_count = self.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        composite_count = self.conn.execute("SELECT COUNT(*) FROM composite_definitions").fetchone()[0]
        avg_coverage = self.conn.execute("""
            SELECT AVG(coverage) FROM (
                SELECT e.agreement_id, 
                       COUNT(CASE WHEN e.value IS NOT NULL AND e.value != '' THEN 1 END) * 1.0 / COUNT(*) as coverage
                FROM extractions e GROUP BY e.agreement_id
            )
        """).fetchone()[0]
        return {
            "domains": domain_count,
            "agreements": agreement_count,
            "clusters": cluster_count,
            "fields": field_count,
            "extractions": extraction_count,
            "corrections": correction_count,
            "composites": composite_count,
            "avg_coverage": round(avg_coverage, 3) if avg_coverage else 0.0,
        }

    # ── Backup / Restore ──────────────────────────────────────────────────

    def checkpoint(self):
        """Force WAL flush to disk. Called automatically before backup."""
        self.conn.execute("CHECKPOINT")

    def backup(self, backup_path: str | Path | None = None) -> Path:
        """
        Create a consistent snapshot of the database.

        Uses DuckDB EXPORT DATABASE to dump all tables as Parquet files,
        which can be re-imported into a fresh database.

        Args:
            backup_path: Directory to write backup. Defaults to
                         <db_dir>/backups/<timestamp>/.
        Returns:
            Path to the backup directory.
        """
        self.checkpoint()
        if backup_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(self.db_path).parent / "backups" / ts
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        self.conn.execute(f"EXPORT DATABASE '{backup_path}' (FORMAT PARQUET)")
        return backup_path

    @classmethod
    def restore(cls, backup_path: str | Path, db_path: str | Path) -> "ClusteringStore":
        """
        Restore a database from a backup created by `backup()`.

        Args:
            backup_path: Directory produced by backup().
            db_path: Path for the restored database file.
        Returns:
            A new ClusteringStore connected to the restored database.
        """
        backup_path = Path(backup_path)
        db_path = Path(db_path)
        if db_path.exists():
            db_path.unlink()
        conn = duckdb.connect(str(db_path))
        conn.execute(f"IMPORT DATABASE '{backup_path}'")
        conn.close()
        return cls(db_path)

    # ── Cluster Layers CRUD ────────────────────────────────────────────────────

    def save_cluster_layers(
        self,
        clustering_run: str,
        layers: list,            # list[np.ndarray]
        item_ids: list[str],
        persistence_scores: list[float],
        selected_layer_idx: int,
        layer_scores: list[dict] | None = None,
    ):
        """Persist all EVoC cluster layers for a clustering run."""
        # Clear previous data for this run
        self.conn.execute(
            "DELETE FROM cluster_layers WHERE clustering_run = ?", [clustering_run])
        self.conn.execute(
            "DELETE FROM cluster_layer_meta WHERE clustering_run = ?", [clustering_run])

        for li, layer in enumerate(layers):
            labels = list(layer)
            n_types = len(set(labels) - {-1})
            n_out = sum(1 for t in labels if t == -1)

            scores = (layer_scores[li] if layer_scores and li < len(layer_scores)
                      else {})

            self.conn.execute("""
                INSERT INTO cluster_layer_meta
                (clustering_run, layer_index, n_clusters, n_outliers,
                 persistence_score, silhouette_score, cosine_score,
                 composite_score, is_selected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [clustering_run, li, n_types, n_out,
                  float(persistence_scores[li]) if li < len(persistence_scores) else None,
                  scores.get("silhouette"),
                  scores.get("cosine"),
                  scores.get("composite"),
                  li == selected_layer_idx])

            # Batch insert layer assignments
            rows = [(clustering_run, li, idx,
                     item_ids[idx] if idx < len(item_ids) else None,
                     int(labels[idx]))
                    for idx in range(len(labels))]
            self.conn.executemany("""
                INSERT INTO cluster_layers
                (clustering_run, layer_index, item_index, item_id, cluster_label)
                VALUES (?, ?, ?, ?, ?)
            """, rows)

    def get_cluster_layer_meta(self, clustering_run: str | None = None) -> list[dict]:
        """Get layer metadata. If clustering_run is None, return all."""
        if clustering_run:
            return self.conn.execute(
                "SELECT * FROM cluster_layer_meta WHERE clustering_run = ? ORDER BY layer_index",
                [clustering_run]).fetchdf().to_dict("records")
        return self.conn.execute(
            "SELECT * FROM cluster_layer_meta ORDER BY clustering_run, layer_index"
        ).fetchdf().to_dict("records")

    def get_cluster_layer(self, clustering_run: str, layer_index: int) -> list[dict]:
        """Get all assignments for a specific layer."""
        return self.conn.execute("""
            SELECT * FROM cluster_layers
            WHERE clustering_run = ? AND layer_index = ?
            ORDER BY item_index
        """, [clustering_run, layer_index]).fetchdf().to_dict("records")

    def get_cluster_layer_runs(self) -> list[str]:
        """Get distinct clustering run names."""
        rows = self.conn.execute(
            "SELECT DISTINCT clustering_run FROM cluster_layer_meta ORDER BY clustering_run"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self):
        self.conn.close()
