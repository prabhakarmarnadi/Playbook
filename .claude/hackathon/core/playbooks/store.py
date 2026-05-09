"""DuckDB store for the playbook subsystem."""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import duckdb

logger = logging.getLogger(__name__)


class PlaybookStore:
    """CRUD + schema init for playbooks. Idempotent."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        # DuckDB 1.5+ rejects existing zero-byte files (e.g. from NamedTemporaryFile).
        p = Path(self.db_path)
        if p.exists() and p.stat().st_size == 0:
            p.unlink()
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def close(self):
        self.conn.close()

    # ── schema ────────────────────────────────────────────────────────────
    def _init_schema(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS playbooks (
            playbook_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            owner_org VARCHAR,
            description TEXT,
            source_file VARCHAR,
            status VARCHAR DEFAULT 'draft',
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT current_timestamp,
            updated_at TIMESTAMP DEFAULT current_timestamp
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS playbook_rules (
            rule_id VARCHAR PRIMARY KEY,
            playbook_id VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            description TEXT,
            applies_to VARCHAR NOT NULL,
            predicate JSON,
            nl_assertion TEXT,
            reference_text TEXT,
            similarity_threshold DOUBLE,
            answer_type VARCHAR,
            answer_options JSON,
            severity VARCHAR NOT NULL,
            escalation_owner VARCHAR,
            preferred_language TEXT,
            fallback_language JSON,
            walkaway_language TEXT,
            rationale TEXT,
            tags JSON,
            source_provenance JSON,
            status VARCHAR DEFAULT 'draft',
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp,
            updated_at TIMESTAMP DEFAULT current_timestamp
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS rule_bindings (
            binding_id VARCHAR PRIMARY KEY,
            rule_id VARCHAR NOT NULL,
            entity_kind VARCHAR NOT NULL,
            entity_id VARCHAR NOT NULL,
            label_text VARCHAR,
            label_embedding BLOB,
            confidence DOUBLE
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS rule_revisions (
            revision_id VARCHAR PRIMARY KEY,
            rule_id VARCHAR NOT NULL,
            revision_no INTEGER NOT NULL,
            diff JSON,
            changed_by VARCHAR,
            change_reason TEXT,
            changed_at TIMESTAMP DEFAULT current_timestamp
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS rule_evaluations (
            eval_id VARCHAR PRIMARY KEY,
            rule_id VARCHAR NOT NULL,
            agreement_id VARCHAR NOT NULL,
            run_id VARCHAR,
            outcome VARCHAR NOT NULL,
            answer_value JSON,
            severity VARCHAR NOT NULL,
            deviation DOUBLE,
            evidence JSON,
            rationale TEXT,
            evaluator_used JSON,
            evaluated_at TIMESTAMP DEFAULT current_timestamp
        )""")

    # ── playbooks ─────────────────────────────────────────────────────────
    def create_playbook(self, *, name: str, owner_org: str = "",
                         description: str = "", source_file: str = "") -> str:
        pid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO playbooks (playbook_id,name,owner_org,description,source_file)
               VALUES (?,?,?,?,?)""",
            [pid, name, owner_org, description, source_file],
        )
        return pid

    def get_playbook(self, playbook_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT * FROM playbooks WHERE playbook_id=?", [playbook_id]
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip(cols, row))

    def list_playbooks(self, status: Optional[str] = None) -> list[dict]:
        if status:
            cur = self.conn.execute(
                "SELECT * FROM playbooks WHERE status=? ORDER BY name", [status]
            )
        else:
            cur = self.conn.execute("SELECT * FROM playbooks ORDER BY name")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ── rules ─────────────────────────────────────────────────────────────
    def create_rule(self, *, playbook_id: str, title: str, applies_to: str,
                     severity: str, **kwargs: Any) -> str:
        rid = str(uuid.uuid4())
        cols = ["rule_id", "playbook_id", "title", "applies_to", "severity"]
        vals: list[Any] = [rid, playbook_id, title, applies_to, severity]
        json_cols = {"predicate", "answer_options", "fallback_language",
                     "tags", "source_provenance"}
        for k, v in kwargs.items():
            cols.append(k)
            vals.append(json.dumps(v) if k in json_cols and v is not None else v)
        placeholders = ",".join(["?"] * len(vals))
        self.conn.execute(
            f"INSERT INTO playbook_rules ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        return rid

    def list_rules(self, playbook_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM playbook_rules WHERE playbook_id=? ORDER BY title",
            [playbook_id],
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            for k in ("predicate", "answer_options", "fallback_language",
                       "tags", "source_provenance"):
                if d.get(k):
                    d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
            out.append(d)
        return out

    def update_rule_status(self, rule_id: str, status: str):
        self.conn.execute(
            "UPDATE playbook_rules SET status=?, updated_at=current_timestamp WHERE rule_id=?",
            [status, rule_id],
        )

    # ── bindings ──────────────────────────────────────────────────────────
    def add_binding(self, *, rule_id: str, entity_kind: str, entity_id: str,
                     label_text: str = "", label_embedding: bytes = b"",
                     confidence: float = 1.0) -> str:
        bid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO rule_bindings
               (binding_id,rule_id,entity_kind,entity_id,label_text,label_embedding,confidence)
               VALUES (?,?,?,?,?,?,?)""",
            [bid, rule_id, entity_kind, entity_id, label_text, label_embedding, confidence],
        )
        return bid

    def bindings_for(self, rule_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM rule_bindings WHERE rule_id=?", [rule_id]
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ── evaluations ───────────────────────────────────────────────────────
    def record_eval(self, *, rule_id: str, agreement_id: str, run_id: str,
                     outcome: str, severity: str, deviation: float = 0.0,
                     answer_value: Optional[dict] = None,
                     evidence: Optional[list] = None, rationale: str = "",
                     evaluator_used: Optional[dict] = None) -> str:
        eid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO rule_evaluations
               (eval_id,rule_id,agreement_id,run_id,outcome,answer_value,severity,
                deviation,evidence,rationale,evaluator_used)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [eid, rule_id, agreement_id, run_id, outcome,
             json.dumps(answer_value) if answer_value is not None else None,
             severity, deviation,
             json.dumps(evidence) if evidence is not None else None,
             rationale,
             json.dumps(evaluator_used) if evaluator_used is not None else None],
        )
        return eid

    def findings_for_run(self, run_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM rule_evaluations WHERE run_id=? ORDER BY severity DESC",
            [run_id],
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            for k in ("answer_value", "evidence", "evaluator_used"):
                if d.get(k):
                    d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
            out.append(d)
        return out
