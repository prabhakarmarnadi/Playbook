# Playbook Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a playbook subsystem that imports xlsx/docx playbooks, mines rule candidates from the existing clustering corpus, lets a human edit and approve them, and aligns new contracts against the published rules — emitting findings with severity, evidence, and pre-approved redline language.

**Architecture:** New package `core/playbooks/` (store · DSL · importers · miner · aligner) plus three Streamlit pages. Five new DuckDB tables (`playbooks`, `playbook_rules`, `rule_bindings`, `rule_revisions`, `rule_evaluations`). Rules bind to existing ontology IDs (`domain_id`, `cluster_id`, `field_id`); soft-rebinding via stored label embeddings survives re-clustering. Hybrid evaluator: deterministic JSON predicate DSL + similarity-to-reference + LLM NL-judge.

**Tech Stack:** Python 3.10+, DuckDB (existing), Pydantic v2, openpyxl, python-docx, Streamlit (existing), FastAPI (existing), `core/embedder.py` (MiniLM 384-d), `core/llm_client.py` (existing).

**Project root:** `.claude/hackathon/` (per `.claude/CLAUDE.md`). All paths below are relative to that root unless noted.

**Test harness:** This project has no pytest suite by convention. Tests live in `scripts/validate_playbook.py` (mirroring `scripts/validate_pipeline.py`): each check is a `def test_*()` function that asserts and prints `[PASS]`. The TDD cycle for every task is:

1. Add the failing check in `scripts/validate_playbook.py`,
2. Run `poetry run python scripts/validate_playbook.py` from `.claude/hackathon/` — see the new check fail,
3. Write the implementation,
4. Re-run — see the check pass,
5. Commit.

**Spec:** `docs/superpowers/specs/2026-05-09-playbook-rules-design.md`

---

## File Structure

**New files (created):**

```
.claude/hackathon/
├── core/playbooks/
│   ├── __init__.py
│   ├── store.py                       # DuckDB schema + CRUD
│   ├── models.py                      # Pydantic mirrors of tables
│   ├── predicates/
│   │   ├── __init__.py                # Registry + decorator
│   │   ├── evaluator.py               # JSON tree walker
│   │   ├── field_ops.py               # field.* operators
│   │   ├── clause_ops.py              # clause.* operators
│   │   ├── logical_ops.py             # and/or/not/if_then/any_of/all_of/count
│   │   └── context_ops.py             # domain.is, doc.* operators
│   ├── importers/
│   │   ├── __init__.py                # Header auto-detection dispatcher
│   │   ├── tabular.py                 # Cloudera + Walmart xlsx
│   │   ├── walmart_logic.py           # "IF X_SP IS SET --> Answer = Standard" parser
│   │   ├── narrative.py               # Docusign xlsx
│   │   └── desirable.py               # Sample AI Playbook docx
│   ├── miner.py                       # Coverage / distribution / outlier candidates
│   └── aligner.py                     # Predicate + similarity + NL-judge + combiner
├── ui/
│   ├── playbook_authoring_page.py
│   ├── playbook_mining_page.py
│   └── playbook_findings_page.py
├── scripts/
│   ├── validate_playbook.py           # Test harness
│   └── migrate_benchmarks_to_playbook.py  # Promote existing playbook_benchmarks → new schema
└── eval/
    └── playbook_eval.py               # Findings precision/recall vs gold set
```

**Existing files (modified):**

- `app.py` — add a `"📜 Playbooks"` section to the `SECTIONS` dict.
- `api_server.py` — new endpoints: `POST /playbooks/import`, `GET /playbooks/{id}`, `POST /playbooks/{id}/run`, `GET /findings/{run_id}`.
- `core/evoc_pipeline.py` — accept `--playbook <playbook_id>` and call the aligner as the final stage.
- `pyproject.toml` — add `python-docx` (the only new runtime dep; `openpyxl` is already a transitive of `pandas`).

**Conventions to honor (from `.claude/CLAUDE.md`):**

- Idempotent schema init (`CREATE TABLE IF NOT EXISTS`).
- Graceful CPU fallback (no GPU-only code on a code path that runs in tests).
- No pytest — extend `scripts/validate_*.py`.
- `data/` is gitignored — fixtures live under `tests_fixtures/playbooks/` (also gitignored, populated from the existing `data/` files at test time).

---

### Task 1: Bootstrap test harness

**Files:**

- Create: `.claude/hackathon/scripts/validate_playbook.py`
- Create: `.claude/hackathon/core/playbooks/__init__.py` (empty)

- [ ] **Step 1: Write the failing harness skeleton + first check.**

Create `scripts/validate_playbook.py`:

```python
"""
Validation script for the playbook subsystem (mirror of validate_pipeline.py).
Each test_* function asserts and prints [PASS]; main() runs them and reports.

Usage:  poetry run python scripts/validate_playbook.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_importable():
    """Validate: core.playbooks package exists and imports cleanly."""
    import core.playbooks  # noqa: F401
    print("  [PASS] Package import")


CHECKS = [
    ("package_importable", test_package_importable),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only checks whose name contains this substring")
    args = parser.parse_args()

    failed = 0
    for name, fn in CHECKS:
        if args.only and args.only not in name:
            continue
        print(f"\n— {name} —")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}\n{len(CHECKS)-failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it. Expect FAIL on import.**

```
cd .claude/hackathon
poetry run python scripts/validate_playbook.py
```

Expected: `[ERROR] ModuleNotFoundError: No module named 'core.playbooks'`.

- [ ] **Step 3: Create the empty package.**

Create `core/playbooks/__init__.py` with a single line:

```python
"""Playbook subsystem: rules, importers, miner, aligner."""
```

- [ ] **Step 4: Re-run. Expect PASS.**

```
poetry run python scripts/validate_playbook.py
```

Expected: `[PASS] Package import` and `1/1 passed`.

- [ ] **Step 5: Commit.**

```
git add scripts/validate_playbook.py core/playbooks/__init__.py
git commit -m "test(playbooks): bootstrap validate_playbook harness + empty package"
```

---

### Task 2: Schema + store

**Files:**

- Create: `core/playbooks/store.py`
- Modify: `scripts/validate_playbook.py` (add 2 checks)

- [ ] **Step 1: Add failing schema check.**

Append to `scripts/validate_playbook.py` above `CHECKS = [...]`:

```python
def test_store_schema_idempotent():
    """Validate: PlaybookStore creates 5 tables and is safe to re-init."""
    from core.playbooks.store import PlaybookStore
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s1 = PlaybookStore(tf.name)
        s2 = PlaybookStore(tf.name)  # second init must not fail
        tables = {r[0] for r in s1.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        for required in ("playbooks", "playbook_rules", "rule_bindings",
                          "rule_revisions", "rule_evaluations"):
            assert required in tables, f"missing table: {required}"
        s1.close(); s2.close()
    print("  [PASS] Schema idempotent (5 tables)")


def test_store_crud_roundtrip():
    """Validate: insert + fetch a playbook and a rule round-trips fields."""
    from core.playbooks.store import PlaybookStore
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="Demo", owner_org="Acme")
        rid = s.create_rule(playbook_id=pid, title="LoL cap",
                             applies_to="field", severity="warn",
                             predicate={"op": "field.gte", "args": ["cap", 250000]})
        rules = s.list_rules(pid)
        assert len(rules) == 1 and rules[0]["rule_id"] == rid
        assert rules[0]["predicate"]["op"] == "field.gte"
        s.close()
    print("  [PASS] CRUD round-trip")
```

Add to `CHECKS` list:

```python
    ("store_schema_idempotent", test_store_schema_idempotent),
    ("store_crud_roundtrip",    test_store_crud_roundtrip),
```

- [ ] **Step 2: Run. Expect FAIL.**

```
poetry run python scripts/validate_playbook.py
```

Expected: `ModuleNotFoundError: No module named 'core.playbooks.store'`.

- [ ] **Step 3: Implement `core/playbooks/store.py`.**

```python
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
    def create_playbook(self, name: str, owner_org: str = "",
                         description: str = "", source_file: str = "") -> str:
        pid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO playbooks (playbook_id,name,owner_org,description,source_file)
               VALUES (?,?,?,?,?)""",
            [pid, name, owner_org, description, source_file],
        )
        return pid

    def get_playbook(self, playbook_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM playbooks WHERE playbook_id=?", [playbook_id]
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, row))

    def list_playbooks(self, status: Optional[str] = None) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM playbooks WHERE status=? ORDER BY name", [status]
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM playbooks ORDER BY name").fetchall()
        cols = [d[0] for d in self.conn.description]
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
        rows = self.conn.execute(
            "SELECT * FROM playbook_rules WHERE playbook_id=? ORDER BY title",
            [playbook_id],
        ).fetchall()
        cols = [d[0] for d in self.conn.description]
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
        rows = self.conn.execute(
            "SELECT * FROM rule_bindings WHERE rule_id=?", [rule_id]
        ).fetchall()
        cols = [d[0] for d in self.conn.description]
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
             json.dumps(answer_value) if answer_value else None,
             severity, deviation,
             json.dumps(evidence) if evidence else None,
             rationale,
             json.dumps(evaluator_used) if evaluator_used else None],
        )
        return eid

    def findings_for_run(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM rule_evaluations WHERE run_id=? ORDER BY severity DESC",
            [run_id],
        ).fetchall()
        cols = [d[0] for d in self.conn.description]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            for k in ("answer_value", "evidence", "evaluator_used"):
                if d.get(k):
                    d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
            out.append(d)
        return out
```

- [ ] **Step 4: Run. Expect 3/3 PASS.**

```
poetry run python scripts/validate_playbook.py
```

- [ ] **Step 5: Commit.**

```
git add core/playbooks/store.py scripts/validate_playbook.py
git commit -m "feat(playbooks): DuckDB store with 5 idempotent tables and CRUD"
```

---

### Task 3: Pydantic models

**Files:**

- Create: `core/playbooks/models.py`
- Modify: `scripts/validate_playbook.py` (add 1 check)

- [ ] **Step 1: Add failing check.**

Append to validate_playbook.py:

```python
def test_models_roundtrip():
    """Validate: Pydantic models round-trip a Rule with predicate JSON."""
    from core.playbooks.models import Rule, Severity, AppliesTo
    r = Rule(
        rule_id="r1", playbook_id="p1", title="cap",
        applies_to=AppliesTo.field, severity=Severity.warn,
        predicate={"op": "field.gte", "args": ["cap", 250000]},
    )
    j = r.model_dump_json()
    r2 = Rule.model_validate_json(j)
    assert r2.predicate == r.predicate
    assert r2.severity == Severity.warn
    print("  [PASS] Pydantic round-trip")
```

Add `("models_roundtrip", test_models_roundtrip)` to `CHECKS`.

- [ ] **Step 2: Run. Expect FAIL (no module).**

```
poetry run python scripts/validate_playbook.py --only models_roundtrip
```

- [ ] **Step 3: Implement `core/playbooks/models.py`.**

```python
"""Pydantic v2 mirrors of playbook DuckDB tables."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    approval_required = "approval_required"
    blocker = "blocker"


class AppliesTo(str, Enum):
    document = "document"
    domain = "domain"
    cluster = "cluster"
    field = "field"
    composite = "composite"
    cross_field = "cross_field"


class RuleStatus(str, Enum):
    draft = "draft"
    active = "active"
    retired = "retired"


class Outcome(str, Enum):
    pass_ = "pass"
    fail = "fail"
    na = "n/a"
    needs_human = "needs_human"


class Playbook(BaseModel):
    playbook_id: str
    name: str
    owner_org: Optional[str] = None
    description: Optional[str] = None
    source_file: Optional[str] = None
    status: str = "draft"
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Rule(BaseModel):
    rule_id: str
    playbook_id: str
    title: str
    description: Optional[str] = None
    applies_to: AppliesTo
    predicate: Optional[dict[str, Any]] = None
    nl_assertion: Optional[str] = None
    reference_text: Optional[str] = None
    similarity_threshold: Optional[float] = None
    answer_type: Optional[str] = None
    answer_options: Optional[list[str]] = None
    severity: Severity
    escalation_owner: Optional[str] = None
    preferred_language: Optional[str] = None
    fallback_language: Optional[list[str]] = None
    walkaway_language: Optional[str] = None
    rationale: Optional[str] = None
    tags: Optional[list[str]] = None
    source_provenance: Optional[dict[str, Any]] = None
    status: RuleStatus = RuleStatus.draft


class Binding(BaseModel):
    binding_id: str
    rule_id: str
    entity_kind: str   # domain | cluster | field | composite
    entity_id: str
    label_text: Optional[str] = None
    confidence: float = 1.0


class Finding(BaseModel):
    eval_id: str
    rule_id: str
    agreement_id: str
    run_id: str
    outcome: Outcome
    severity: Severity
    deviation: float = 0.0
    answer_value: Optional[dict[str, Any]] = None
    evidence: Optional[list[dict[str, Any]]] = None
    rationale: Optional[str] = None
    evaluator_used: Optional[dict[str, bool]] = None
```

- [ ] **Step 4: Run. Expect PASS.**

```
poetry run python scripts/validate_playbook.py --only models_roundtrip
```

- [ ] **Step 5: Commit.**

```
git add core/playbooks/models.py scripts/validate_playbook.py
git commit -m "feat(playbooks): Pydantic v2 models mirroring schema"
```

---

### Task 4: Predicate DSL

**Files:**

- Create: `core/playbooks/predicates/__init__.py`
- Create: `core/playbooks/predicates/evaluator.py`
- Create: `core/playbooks/predicates/field_ops.py`
- Create: `core/playbooks/predicates/clause_ops.py`
- Create: `core/playbooks/predicates/logical_ops.py`
- Create: `core/playbooks/predicates/context_ops.py`
- Modify: `scripts/validate_playbook.py` (add 1 check covering 5 sub-cases)

- [ ] **Step 1: Add failing check covering each operator group.**

```python
def test_predicate_dsl_examples():
    """Validate: each spec §5 example evaluates correctly."""
    from core.playbooks.predicates.evaluator import evaluate

    # Walmart-style OR
    ctx = {"fields": {"Indemnification_Standard_SP": "yes"}}
    p = {"op": "or", "args": [
        {"op": "field.exists", "args": ["Indemnification_Standard_SP"]},
        {"op": "field.exists", "args": ["Indemnification_Standard_SS"]},
    ]}
    assert evaluate(p, ctx) is True

    # Cloudera-style regex
    ctx = {"fields": {"late_payment_penalty": "1.5%"}}
    p = {"op": "field.matches",
         "args": ["late_payment_penalty", "regex", r"\d+(\.\d+)?%"]}
    assert evaluate(p, ctx) is True

    # AI Playbook AND/OR with numeric gte
    ctx = {"fields": {"liability_cap_amount": 300000, "liability_cap_ratio_fees": 0.5}}
    p = {"op": "and", "args": [
        {"op": "field.exists", "args": ["liability_cap_amount"]},
        {"op": "or", "args": [
            {"op": "field.gte", "args": ["liability_cap_amount", 250000]},
            {"op": "field.gte", "args": ["liability_cap_ratio_fees", 1.0]},
        ]},
    ]}
    assert evaluate(p, ctx) is True

    # if_then with violated consequent
    ctx = {"fields": {"governing_law": "New York", "arbitration_venue": "Delaware"}}
    p = {"op": "if_then", "args": [
        {"op": "field.eq", "args": ["governing_law", "New York"]},
        {"op": "field.eq", "args": ["arbitration_venue", "New York"]},
    ]}
    assert evaluate(p, ctx) is False

    # any_of over clauses
    ctx = {"clauses": [{"label": "Indemnification"}, {"label": "Limitation of Liability"}]}
    p = {"op": "any_of", "args": [
        {"op": "clause.classified_as", "args": ["Indemnification"]},
    ]}
    assert evaluate(p, ctx) is True

    print("  [PASS] DSL examples (5 sub-cases)")
```

Register in `CHECKS`.

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement registry + evaluator + 4 op modules.**

`core/playbooks/predicates/__init__.py`:

```python
"""Predicate registry. Operators register via @predicate('name')."""
from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def predicate(name: str):
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown predicate operator: {name}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY.keys())
```

`core/playbooks/predicates/evaluator.py`:

```python
"""JSON s-expression evaluator. Each node = {op, args}.

ctx schema (loose):
  {
    "fields":   {field_name: value, ...}     # for field.* ops
    "clauses":  [{"label": str, "text": str, "embedding": list[float]?}, ...]
    "domain":   {"id": str, "name": str}
    "doc":      {"value": float, "party_role": str, ...}
  }
"""
from __future__ import annotations
from typing import Any

# Side-effect imports register operators
from . import field_ops, clause_ops, logical_ops, context_ops  # noqa: F401
from . import get


def evaluate(node: Any, ctx: dict) -> Any:
    if not isinstance(node, dict) or "op" not in node:
        return node  # literal pass-through (numbers, strings, lists)
    op = node["op"]
    args = node.get("args", [])
    return get(op)(args, ctx, evaluate)
```

`core/playbooks/predicates/field_ops.py`:

```python
import re
from . import predicate


def _val(args, ctx):
    name = args[0]
    return ctx.get("fields", {}).get(name)


@predicate("field.exists")
def op_exists(args, ctx, _ev):
    return _val(args, ctx) is not None


@predicate("field.eq")
def op_eq(args, ctx, _ev):
    return _val(args, ctx) == args[1]


@predicate("field.in")
def op_in(args, ctx, _ev):
    return _val(args, ctx) in args[1]


@predicate("field.gte")
def op_gte(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(v) >= float(args[1])
    except (TypeError, ValueError):
        return False


@predicate("field.lte")
def op_lte(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(v) <= float(args[1])
    except (TypeError, ValueError):
        return False


@predicate("field.between")
def op_between(args, ctx, _ev):
    v = _val(args, ctx)
    try:
        return v is not None and float(args[1]) <= float(v) <= float(args[2])
    except (TypeError, ValueError):
        return False


@predicate("field.matches")
def op_matches(args, ctx, _ev):
    """args = [field_name, "regex"|"glob", pattern]"""
    v = _val(args, ctx)
    if v is None:
        return False
    kind, pattern = args[1], args[2]
    if kind == "regex":
        return re.search(pattern, str(v)) is not None
    if kind == "glob":
        import fnmatch
        return fnmatch.fnmatch(str(v), pattern)
    raise ValueError(f"unknown match kind: {kind}")


@predicate("field.value")
def op_value(args, ctx, _ev):
    return _val(args, ctx)


@predicate("field.semantic_match")
def op_sem(args, ctx, _ev):
    """args = [field_name, target_text, threshold]
    Requires ctx['embed'] callable (text -> vector). Returns False if absent."""
    v = _val(args, ctx)
    embed = ctx.get("embed")
    if v is None or embed is None:
        return False
    import numpy as np
    a = np.asarray(embed(str(v)))
    b = np.asarray(embed(args[1]))
    if a.size == 0 or b.size == 0:
        return False
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return cos >= float(args[2])
```

`core/playbooks/predicates/clause_ops.py`:

```python
from . import predicate


@predicate("clause.classified_as")
def op_classified_as(args, ctx, _ev):
    target = args[0]
    for c in ctx.get("clauses", []):
        if c.get("label") == target:
            return True
    return False


@predicate("clause.contains")
def op_contains(args, ctx, _ev):
    needle = args[0].lower()
    for c in ctx.get("clauses", []):
        if needle in (c.get("text") or "").lower():
            return True
    return False


@predicate("clause.absent")
def op_absent(args, ctx, _ev):
    target = args[0]
    return not any(c.get("label") == target for c in ctx.get("clauses", []))


@predicate("clause.similar_to")
def op_similar(args, ctx, _ev):
    """args = [reference_text, threshold]. Uses ctx['embed']."""
    embed = ctx.get("embed")
    if embed is None:
        return False
    import numpy as np
    ref = np.asarray(embed(args[0]))
    thr = float(args[1])
    for c in ctx.get("clauses", []):
        text = c.get("text")
        if not text:
            continue
        v = np.asarray(embed(text))
        if v.size == 0 or ref.size == 0:
            continue
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
        if cos >= thr:
            return True
    return False
```

`core/playbooks/predicates/logical_ops.py`:

```python
from . import predicate


@predicate("and")
def op_and(args, ctx, ev):
    return all(ev(a, ctx) for a in args)


@predicate("or")
def op_or(args, ctx, ev):
    return any(ev(a, ctx) for a in args)


@predicate("not")
def op_not(args, ctx, ev):
    return not ev(args[0], ctx)


@predicate("if_then")
def op_if_then(args, ctx, ev):
    return (not ev(args[0], ctx)) or ev(args[1], ctx)


@predicate("any_of")
def op_any_of(args, ctx, ev):
    return any(ev(a, ctx) for a in args)


@predicate("all_of")
def op_all_of(args, ctx, ev):
    return all(ev(a, ctx) for a in args)


@predicate("count")
def op_count(args, ctx, ev):
    """args = [sub_predicate, op, n]   e.g. [..., '>=', 2]"""
    sub, cmp_op, n = args
    items = ctx.get("clauses", [])
    hits = sum(1 for it in items if ev(sub, {**ctx, "clauses": [it]}))
    return {">=": hits >= n, ">": hits > n, "<=": hits <= n,
            "<": hits < n, "==": hits == n}[cmp_op]
```

`core/playbooks/predicates/context_ops.py`:

```python
from . import predicate


@predicate("domain.is")
def op_domain_is(args, ctx, _ev):
    return (ctx.get("domain") or {}).get("name") == args[0]


@predicate("doc.party_role.is")
def op_role_is(args, ctx, _ev):
    return (ctx.get("doc") or {}).get("party_role") == args[0]


@predicate("doc.value.gte")
def op_value_gte(args, ctx, _ev):
    v = (ctx.get("doc") or {}).get("value")
    try:
        return v is not None and float(v) >= float(args[0])
    except (TypeError, ValueError):
        return False
```

- [ ] **Step 4: Run. Expect PASS on the 5 sub-cases.**

- [ ] **Step 5: Commit.**

```
git add core/playbooks/predicates/ scripts/validate_playbook.py
git commit -m "feat(playbooks): predicate DSL (registry + evaluator + 4 op modules)"
```

---

### Task 5: Tabular importer (Cloudera + Walmart)

**Files:**

- Create: `core/playbooks/importers/__init__.py`
- Create: `core/playbooks/importers/tabular.py`
- Create: `core/playbooks/importers/walmart_logic.py`
- Modify: `pyproject.toml` (add `python-docx`; openpyxl already available)
- Modify: `scripts/validate_playbook.py` (add 2 checks)

- [ ] **Step 1: Set up fixtures + add failing checks.**

Create `tests_fixtures/playbooks/` and copy the two source files in (gitignored):

```bash
mkdir -p .claude/hackathon/tests_fixtures/playbooks
cp "../../data/Cloudera Insight Ruelbook.xlsx" .claude/hackathon/tests_fixtures/playbooks/cloudera.xlsx
cp "../../data/Walmart Custom Extractions (1).xlsx" .claude/hackathon/tests_fixtures/playbooks/walmart.xlsx
```

(Note: working dir is `.claude/hackathon`, the source xlsx live two levels up at the repo-level `data/` folder. Adjust paths if running from repo root.)

Append to validate_playbook.py:

```python
FIX = Path(__file__).parent.parent / "tests_fixtures" / "playbooks"


def test_walmart_logic_parser():
    """Validate: 'IF X_SP IS SET --> Answer = Standard' parses to predicate JSON."""
    from core.playbooks.importers.walmart_logic import parse
    src = "1) IF X_Std_SP IS SET --> Answer = Standard\n2) IF X_NonStd_SP IS SET --> Answer = Non-Standard"
    rules = parse(src)
    assert len(rules) >= 2
    std = next(r for r in rules if r["answer"] == "Standard")
    assert std["predicate"]["op"] == "field.exists"
    assert std["predicate"]["args"] == ["X_Std_SP"]
    print("  [PASS] Walmart logic parser")


def test_tabular_importer_cloudera():
    """Validate: tabular importer ingests Cloudera xlsx and produces ≥10 rules."""
    if not (FIX / "cloudera.xlsx").exists():
        print("  [SKIP] cloudera fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.tabular import import_xlsx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_xlsx(s, str(FIX / "cloudera.xlsx"), name="Cloudera")
        rules = s.list_rules(pid)
        assert len(rules) >= 10, f"too few rules: {len(rules)}"
        # round-trip provenance
        prov = rules[0]["source_provenance"]
        assert prov and "row" in prov
        s.close()
    print(f"  [PASS] Cloudera importer ({len(rules)} rules)")
```

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement Walmart parser + tabular importer + dispatcher.**

`core/playbooks/importers/walmart_logic.py`:

```python
"""Parse Walmart-style 'Deployment Logic' strings into predicate JSON.

Grammar covered (per Walmart Custom Extractions xlsx):
  '<n>) IF <expr> IS SET --> Answer = <ans>'
  '<n>) IF (<a> OR <b>) IS SET --> Answer = <ans>'
  '<n>) IF <a> AND <b> IS SET --> Answer = <ans>'
  '<n>) Else --> Answer = <ans>'

The :* suffix on field names is stripped (it's a wildcard for sub-properties).
"""
from __future__ import annotations
import re

CLAUSE_RE = re.compile(
    r"(?:\d+\)\s*)?(?:IF\s+(?P<expr>.+?)\s+IS\s+SET|Else)\s*-->?\s*Answer\s*=\s*(?P<ans>[^\n]+)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?::\*)?")


def _to_predicate(expr: str) -> dict:
    expr = expr.strip().strip("()").strip()
    upper = expr.upper()
    # OR has lower precedence than AND
    if " OR " in upper:
        parts = re.split(r"\s+OR\s+", expr, flags=re.IGNORECASE)
        return {"op": "or", "args": [_to_predicate(p) for p in parts]}
    if " AND " in upper:
        parts = re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)
        return {"op": "and", "args": [_to_predicate(p) for p in parts]}
    m = TOKEN_RE.match(expr)
    if not m:
        return {"op": "field.exists", "args": [expr]}
    name = m.group(0).rstrip(":*")
    return {"op": "field.exists", "args": [name]}


def parse(text: str) -> list[dict]:
    out = []
    for m in CLAUSE_RE.finditer(text or ""):
        ans = (m.group("ans") or "").strip()
        expr = m.group("expr")
        pred = _to_predicate(expr) if expr else {"op": "and", "args": []}  # else
        out.append({"predicate": pred, "answer": ans, "is_else": expr is None})
    return out
```

`core/playbooks/importers/tabular.py`:

```python
"""Cloudera/Walmart tabular xlsx importer.

Header detection: contains Topic AND (Sub-Topic|Subtopic) AND
'Prompt Question'|'Subtopic Question' AND 'Answer Type'|'Answer Options'.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import openpyxl

from ..store import PlaybookStore
from . import walmart_logic


def _header_map(row: tuple) -> Optional[dict]:
    cells = [str(c).strip() if c else "" for c in row]
    norm = [c.lower() for c in cells]

    def find(*names) -> Optional[int]:
        for i, c in enumerate(norm):
            if c in [n.lower() for n in names]:
                return i
        return None

    topic = find("topic")
    sub = find("sub-topic", "subtopic")
    q = find("prompt question", "subtopic question")
    a = find("answer type", "answer options")
    if None in (topic, sub, q, a):
        return None
    return {
        "topic": topic, "sub": sub, "question": q, "answer_type": a,
        "example": find("customer example languge", "customer example language"),
        "logic": find("deployment logic"),
    }


def _split_options(s: str) -> list[str]:
    if not s:
        return []
    return [t.strip() for t in s.replace(":", ";").split(";") if t.strip()]


def import_xlsx(store: PlaybookStore, path: str, *,
                 name: Optional[str] = None,
                 owner_org: str = "") -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    pid = store.create_playbook(
        name=name or Path(path).stem, owner_org=owner_org, source_file=path
    )

    n_rules = 0
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        # find header row
        hmap = None
        header_idx = -1
        for idx, row in enumerate(rows[:5]):
            hmap = _header_map(row)
            if hmap:
                header_idx = idx
                break
        if not hmap:
            continue

        for ri, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
            topic = row[hmap["topic"]] if hmap["topic"] is not None else None
            sub = row[hmap["sub"]] if hmap["sub"] is not None else None
            q = row[hmap["question"]] if hmap["question"] is not None else None
            if not (topic and sub and q):
                continue
            ans_type_raw = row[hmap["answer_type"]] if hmap["answer_type"] is not None else ""
            options = _split_options(str(ans_type_raw or ""))
            example = row[hmap["example"]] if hmap.get("example") is not None else None
            logic = row[hmap["logic"]] if hmap.get("logic") is not None else None

            predicate = None
            if logic:
                parsed = walmart_logic.parse(str(logic))
                # Use the FIRST non-else branch as the rule's predicate; alternatives go to tags
                primary = next((p for p in parsed if not p["is_else"]), None)
                if primary:
                    predicate = primary["predicate"]

            rid = store.create_rule(
                playbook_id=pid,
                title=f"{topic} :: {sub}",
                description=str(q),
                applies_to="cluster",
                severity="warn",
                predicate=predicate,
                reference_text=str(example) if example else None,
                answer_type="standard_nonstandard" if "standard" in str(ans_type_raw).lower() else "yes_no",
                answer_options=options,
                tags=[str(topic), str(sub)],
                source_provenance={"file": path, "sheet": sheet_name, "row": ri},
                status="draft",
            )
            store.add_binding(
                rule_id=rid, entity_kind="cluster",
                entity_id=f"label:{topic}",   # soft binding placeholder; resolver fills cluster_id
                label_text=str(topic), confidence=0.5,
            )
            n_rules += 1

    store.conn.execute(
        "UPDATE playbooks SET description=? WHERE playbook_id=?",
        [f"Imported {n_rules} rules from {Path(path).name}", pid],
    )
    return pid
```

`core/playbooks/importers/__init__.py`:

```python
"""Playbook importers. Auto-dispatch by file extension + header detection."""
from pathlib import Path
from ..store import PlaybookStore


def import_file(store: PlaybookStore, path: str, **kwargs) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".xlsx":
        from . import tabular, narrative
        # try tabular first; if it produces 0 rules, retry as narrative
        pid = tabular.import_xlsx(store, path, **kwargs)
        rules = store.list_rules(pid)
        if rules:
            return pid
        # narrative fallback (Docusign)
        return narrative.import_xlsx(store, path, **kwargs)
    if ext == ".docx":
        from . import desirable
        return desirable.import_docx(store, path, **kwargs)
    raise ValueError(f"unsupported playbook file extension: {ext}")
```

- [ ] **Step 4: Add `python-docx` to deps and run.**

```
poetry add python-docx
poetry run python scripts/validate_playbook.py
```

Expect: walmart_logic + cloudera importer PASS. (The fallback in `__init__.py` references `narrative` and `desirable` which aren't implemented yet — leave the import inside the conditional so it doesn't break Task 5; tasks 6 + 7 land them.)

- [ ] **Step 5: Commit.**

```
git add core/playbooks/importers/__init__.py core/playbooks/importers/tabular.py \
        core/playbooks/importers/walmart_logic.py scripts/validate_playbook.py pyproject.toml poetry.lock
git commit -m "feat(playbooks): tabular xlsx importer + Walmart deployment-logic parser"
```

---

### Task 6: Narrative importer (Docusign)

**Files:**

- Create: `core/playbooks/importers/narrative.py`
- Modify: `scripts/validate_playbook.py` (add 1 check)

- [ ] **Step 1: Add failing check.**

```python
def test_narrative_importer_docusign():
    """Validate: narrative importer ingests Docusign MSA sheet without crashing."""
    src = FIX / "docusign.xlsx"
    if not src.exists():
        print("  [SKIP] docusign fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.narrative import import_xlsx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_xlsx(s, str(src), name="Docusign")
        rules = s.list_rules(pid)
        assert len(rules) >= 5, f"too few rules: {len(rules)}"
        # at least one with escalation_owner set
        assert any(r.get("escalation_owner") for r in rules)
        s.close()
    print(f"  [PASS] Docusign narrative importer ({len(rules)} rules)")
```

Also copy fixture: `cp "../../data/Docusign Commercial Playbooks - DS Restricted - Not for External Use .xlsx" tests_fixtures/playbooks/docusign.xlsx`. Register in `CHECKS`.

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement.**

`core/playbooks/importers/narrative.py`:

```python
"""Docusign-style narrative xlsx importer.

Each playbook sheet has columns:
  Section/Topic | Customer Issue/Request | Approvals/Notes/Responses | Escalation Owner(s) | Edits

Rows are grouped by Section/Topic header (column A). A header row contains text
in col A and 'Clause' or None in col B. Subsequent rows with col A blank are
sub-rules within that section. Each non-empty 'Customer Issue/Request' row
becomes one rule.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import openpyxl

from ..store import PlaybookStore


HEADERS = ("section/topic", "customer issue/request",
           "approvals/notes/responses", "escalation owner(s)")


def _is_header(row: tuple) -> Optional[dict]:
    cells = [str(c).strip().lower() if c else "" for c in row]
    found = {h: i for i, c in enumerate(cells) if c in HEADERS}
    if len(found) >= 3:  # tolerate missing "Edits"
        return {
            "topic": found.get("section/topic"),
            "issue": found.get("customer issue/request"),
            "response": found.get("approvals/notes/responses"),
            "owner": found.get("escalation owner(s)"),
            "edits": next((i for i, c in enumerate(cells) if c.startswith("edits")), None),
        }
    return None


def import_xlsx(store: PlaybookStore, path: str, *,
                 name: Optional[str] = None, owner_org: str = "Docusign") -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    pid = store.create_playbook(
        name=name or Path(path).stem, owner_org=owner_org, source_file=path
    )
    n_rules = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        # locate header
        hmap = None
        header_idx = -1
        for idx, row in enumerate(rows[:6]):
            hmap = _is_header(row)
            if hmap:
                header_idx = idx
                break
        if not hmap:
            continue

        current_section = None
        for ri, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
            topic = row[hmap["topic"]] if hmap["topic"] is not None else None
            issue = row[hmap["issue"]] if hmap["issue"] is not None else None
            resp = row[hmap["response"]] if hmap["response"] is not None else None
            owner = row[hmap["owner"]] if hmap["owner"] is not None else None
            edits = row[hmap["edits"]] if hmap.get("edits") is not None else None

            if topic and not issue:  # section header
                current_section = str(topic)
                continue
            if not issue:
                continue
            if str(issue).strip().lower() == "clause":
                continue

            severity = "approval_required" if resp and "approval required" in str(resp).lower() else "warn"

            rid = store.create_rule(
                playbook_id=pid,
                title=f"{current_section or sheet_name} :: {str(issue)[:80]}",
                description=str(issue),
                applies_to="cluster",
                severity=severity,
                nl_assertion=str(issue),
                preferred_language=str(edits) if edits else None,
                escalation_owner=str(owner) if owner else None,
                rationale=str(resp) if resp else None,
                tags=[sheet_name, current_section or ""],
                source_provenance={"file": path, "sheet": sheet_name, "row": ri},
                status="draft",
            )
            if current_section:
                store.add_binding(
                    rule_id=rid, entity_kind="cluster",
                    entity_id=f"label:{current_section}",
                    label_text=current_section, confidence=0.5,
                )
            n_rules += 1

    store.conn.execute(
        "UPDATE playbooks SET description=? WHERE playbook_id=?",
        [f"Imported {n_rules} rules from {Path(path).name}", pid],
    )
    return pid
```

- [ ] **Step 4: Run. Expect PASS.**

- [ ] **Step 5: Commit.**

```
git add core/playbooks/importers/narrative.py scripts/validate_playbook.py
git commit -m "feat(playbooks): narrative importer for Docusign-style playbooks"
```

---

### Task 7: Desirable/undesirable importer

**Files:**

- Create: `core/playbooks/importers/desirable.py`
- Modify: `scripts/validate_playbook.py` (add 1 check)

- [ ] **Step 1: Copy fixture, add failing check.**

```bash
cp "../../data/Sample AI playbook 1.1.docx" tests_fixtures/playbooks/sample_ai.docx
```

Append:

```python
def test_desirable_importer_ai_playbook():
    """Validate: desirable/undesirable docx importer extracts rule + ref + walkaway."""
    src = FIX / "sample_ai.docx"
    if not src.exists():
        print("  [SKIP] sample_ai fixture missing"); return
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.importers.desirable import import_docx
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = import_docx(s, str(src), name="Sample AI")
        rules = s.list_rules(pid)
        assert len(rules) >= 3
        assert any(r.get("reference_text") for r in rules)
        assert any(r.get("walkaway_language") for r in rules)
        s.close()
    print(f"  [PASS] AI Playbook docx importer ({len(rules)} rules)")
```

Register.

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement.**

`core/playbooks/importers/desirable.py`:

```python
"""Sample AI Playbook docx importer.

Detection: docx with a single 5-column table whose header row is
  Topic | Sources | AI Playbook Rule | Desirable Sample Clause | Undesirable Sample Clause
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from docx import Document   # python-docx

from ..store import PlaybookStore


HEADERS = ("topic", "sources", "ai playbook rule",
           "desirable sample clause", "undesirable sample clause")


def import_docx(store: PlaybookStore, path: str, *,
                 name: Optional[str] = None, owner_org: str = "") -> str:
    doc = Document(path)
    pid = store.create_playbook(
        name=name or Path(path).stem, owner_org=owner_org, source_file=path
    )
    n_rules = 0

    for ti, table in enumerate(doc.tables):
        if not table.rows:
            continue
        header = [c.text.strip().lower() for c in table.rows[0].cells]
        if not all(any(h.startswith(needle) for h in header) for needle in HEADERS):
            continue
        idx = {needle: next(i for i, h in enumerate(header) if h.startswith(needle))
                for needle in HEADERS}

        for ri, row in enumerate(table.rows[1:], start=2):
            cells = [c.text.strip() for c in row.cells]
            topic = cells[idx["topic"]]
            sources = cells[idx["sources"]]
            rule_txt = cells[idx["ai playbook rule"]]
            desirable = cells[idx["desirable sample clause"]]
            undesirable = cells[idx["undesirable sample clause"]]
            if not rule_txt:
                continue

            rid = store.create_rule(
                playbook_id=pid,
                title=topic[:80] or f"Rule {ri}",
                description=rule_txt,
                applies_to="cluster",
                severity="warn",
                nl_assertion=rule_txt,
                reference_text=desirable or None,
                walkaway_language=undesirable or None,
                rationale=sources or None,
                similarity_threshold=0.85,
                tags=[topic],
                source_provenance={"file": path, "table": ti, "row": ri},
                status="draft",
            )
            store.add_binding(
                rule_id=rid, entity_kind="cluster",
                entity_id=f"label:{topic}",
                label_text=topic, confidence=0.5,
            )
            n_rules += 1

    store.conn.execute(
        "UPDATE playbooks SET description=? WHERE playbook_id=?",
        [f"Imported {n_rules} rules from {Path(path).name}", pid],
    )
    return pid
```

- [ ] **Step 4: Run. Expect PASS.**

- [ ] **Step 5: Commit.**

```
git add core/playbooks/importers/desirable.py scripts/validate_playbook.py
git commit -m "feat(playbooks): desirable/undesirable docx importer (AI Playbook)"
```

---

### Task 8: Miner

**Files:**

- Create: `core/playbooks/miner.py`
- Modify: `scripts/validate_playbook.py` (add 1 check on a synthetic mini-corpus)

- [ ] **Step 1: Add failing check.**

```python
def test_miner_proposes_candidates():
    """Validate: miner emits ≥1 coverage and ≥1 distribution candidate from a fake corpus."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.miner import mine_candidates

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        # synthetic corpus: 20 MSAs, 19 of which contain the "Indemnification" cluster
        corpus = {
            "domain_clusters": {
                "msa": {"Indemnification": 19, "Other": 5},
            },
            "field_values": {
                "msa::Limitation of Liability::cap_amount":
                    [250_000] * 18 + [None, 500_000],
            },
        }
        pid = s.create_playbook(name="mined")
        cands = mine_candidates(s, pid, corpus, n_total_per_domain=20)
        coverage = [c for c in cands if c["kind"] == "coverage"]
        distribution = [c for c in cands if c["kind"] == "distribution"]
        assert coverage, "no coverage candidates"
        assert distribution, "no distribution candidates"
        # all should be inserted as draft rules
        rules = s.list_rules(pid)
        assert all(r["status"] == "draft" for r in rules)
        s.close()
    print(f"  [PASS] Miner produced {len(cands)} candidates")
```

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement.**

`core/playbooks/miner.py`:

```python
"""Rule candidate miner.

Reads coverage + field distribution stats and inserts draft rules. The caller
prepares a `corpus` dict with two keys:

  domain_clusters: {domain_name: {cluster_label: count, ...}}
  field_values:    {"<domain>::<cluster>::<field>": [v1, v2, ...]}    # values may be None

In production the caller assembles this from the existing DuckDB
(`agreements`, `domains`, `clusters`, `cluster_assignments`, `extractions`).
The seam is intentional so the miner is unit-testable without a clustering run.
"""
from __future__ import annotations
import statistics
from typing import Any

from .store import PlaybookStore


COVERAGE_THRESHOLD = 0.90       # cluster present in ≥90% of domain → expected
NUMERIC_QUANTILE_LO = 0.10      # use p10/p90 as suggested rule bounds
NUMERIC_QUANTILE_HI = 0.90
MIN_OBSERVATIONS = 5


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
    return xs[k]


def mine_candidates(store: PlaybookStore, playbook_id: str,
                     corpus: dict[str, Any], *,
                     n_total_per_domain: int) -> list[dict]:
    cands: list[dict] = []

    # ── coverage candidates ──────────────────────────────────────────────
    for domain, clusters in (corpus.get("domain_clusters") or {}).items():
        for cluster_label, n in clusters.items():
            ratio = n / max(1, n_total_per_domain)
            if ratio < COVERAGE_THRESHOLD:
                continue
            rid = store.create_rule(
                playbook_id=playbook_id,
                title=f"{domain.upper()} must contain {cluster_label}",
                description=(f"{ratio:.0%} of {domain} agreements in the corpus "
                             f"contain a {cluster_label} clause."),
                applies_to="domain",
                severity="warn",
                predicate={"op": "any_of", "args": [
                    {"op": "clause.classified_as", "args": [cluster_label]}
                ]},
                tags=[domain, "coverage"],
                source_provenance={"miner": "coverage", "ratio": ratio, "n": n},
                status="draft",
            )
            store.add_binding(rule_id=rid, entity_kind="domain",
                               entity_id=f"label:{domain}", label_text=domain, confidence=0.5)
            cands.append({"kind": "coverage", "rule_id": rid, "ratio": ratio})

    # ── distribution candidates (numeric only for now) ───────────────────
    for key, values in (corpus.get("field_values") or {}).items():
        nums = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        if len(nums) < MIN_OBSERVATIONS:
            continue
        domain, cluster_label, field_name = key.split("::", 2)
        lo = _quantile(nums, NUMERIC_QUANTILE_LO)
        hi = _quantile(nums, NUMERIC_QUANTILE_HI)
        median = statistics.median(nums)
        rid = store.create_rule(
            playbook_id=playbook_id,
            title=f"{cluster_label}::{field_name} between p10 and p90",
            description=(f"Corpus distribution: median={median}, "
                         f"p10={lo}, p90={hi} ({len(nums)} samples)."),
            applies_to="field",
            severity="info",
            predicate={"op": "field.between", "args": [field_name, lo, hi]},
            answer_type="numeric_range",
            tags=[domain, cluster_label, "distribution"],
            source_provenance={"miner": "distribution", "n": len(nums),
                                "median": median, "lo": lo, "hi": hi},
            status="draft",
        )
        store.add_binding(rule_id=rid, entity_kind="field",
                           entity_id=f"label:{field_name}", label_text=field_name, confidence=0.5)
        cands.append({"kind": "distribution", "rule_id": rid,
                       "median": median, "lo": lo, "hi": hi})

    return cands


def soft_rebind(store: PlaybookStore, *, embed,
                 cluster_centroids: dict[str, dict],
                 threshold: float = 0.85) -> int:
    """For each binding with confidence < 1.0 (or unresolved 'label:' entity_id),
    find the best-matching cluster centroid by cosine on label embeddings.

    cluster_centroids = {cluster_id: {"label": str, "embedding": list[float]}}
    Returns number of bindings updated.
    """
    import numpy as np
    updated = 0
    rows = store.conn.execute(
        "SELECT binding_id, label_text FROM rule_bindings "
        "WHERE entity_id LIKE 'label:%' AND label_text IS NOT NULL"
    ).fetchall()
    if not rows or not cluster_centroids:
        return 0
    centroid_items = list(cluster_centroids.items())
    centroid_vecs = np.asarray([embed(c["label"]) for _, c in centroid_items])
    centroid_norms = np.linalg.norm(centroid_vecs, axis=1) + 1e-9

    for bid, label in rows:
        v = np.asarray(embed(label))
        if v.size == 0:
            continue
        cos = (centroid_vecs @ v) / (centroid_norms * (np.linalg.norm(v) + 1e-9))
        best = int(np.argmax(cos))
        if float(cos[best]) >= threshold:
            cid, _ = centroid_items[best]
            store.conn.execute(
                "UPDATE rule_bindings SET entity_id=?, confidence=? WHERE binding_id=?",
                [cid, float(cos[best]), bid],
            )
            updated += 1
    return updated
```

- [ ] **Step 4: Run. Expect PASS.**

- [ ] **Step 5: Commit.**

```
git add core/playbooks/miner.py scripts/validate_playbook.py
git commit -m "feat(playbooks): miner + soft-rebinding (coverage and field-distribution candidates)"
```

---

### Task 9: Alignment engine

**Files:**

- Create: `core/playbooks/aligner.py`
- Modify: `scripts/validate_playbook.py` (add 2 checks)

- [ ] **Step 1: Add failing checks.**

```python
def test_aligner_predicate_only():
    """Validate: aligner runs a predicate-only rule and writes a Finding."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(
            playbook_id=pid, title="cap >= 250k",
            applies_to="field", severity="warn",
            predicate={"op": "field.gte", "args": ["cap", 250000]},
            status="active",
        )
        ctx = {"agreement_id": "a1", "fields": {"cap": 100000}}
        findings = align(s, pid, ctx)
        assert len(findings) == 1
        assert findings[0]["outcome"] == "fail"
        # second run identical → determinism
        findings2 = align(s, pid, ctx)
        assert findings2[0]["outcome"] == findings[0]["outcome"]
        s.close()
    print("  [PASS] Aligner predicate-only + determinism")


def test_aligner_combiner_severity():
    """Validate: when predicate fails AND nl_judge stub passes, outcome is fail (combiner rule)."""
    import tempfile
    from core.playbooks.store import PlaybookStore
    from core.playbooks.aligner import align

    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        s = PlaybookStore(tf.name)
        pid = s.create_playbook(name="t")
        rid = s.create_rule(
            playbook_id=pid, title="combo",
            applies_to="field", severity="approval_required",
            predicate={"op": "field.gte", "args": ["cap", 999999]},  # fails
            nl_assertion="liability cap is reasonable",  # stub will say pass
            status="active",
        )
        ctx = {"agreement_id": "a1", "fields": {"cap": 100000},
                "nl_judge": lambda assertion, evidence: ("pass", "stubbed")}
        findings = align(s, pid, ctx)
        assert findings[0]["outcome"] == "fail"  # predicate fail dominates
        assert findings[0]["severity"] == "approval_required"
        s.close()
    print("  [PASS] Aligner combiner")
```

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement.**

`core/playbooks/aligner.py`:

```python
"""Playbook alignment engine.

Inputs:  PlaybookStore, playbook_id, ctx
Output:  list of Findings dicts (also persisted to rule_evaluations)

ctx is the same shape consumed by the predicate evaluator, plus:
  agreement_id, run_id (auto-generated if absent), embed (callable, optional),
  nl_judge (callable[(assertion:str, evidence:str) -> ("pass"|"fail"|"n/a", rationale)],
            optional).

Combiner policy:
  - If any evaluator says fail   → outcome = fail
  - If all set evaluators pass   → outcome = pass
  - Otherwise                    → outcome = needs_human
"""
from __future__ import annotations
import uuid
from typing import Any, Callable, Optional

import numpy as np

from .predicates.evaluator import evaluate
from .store import PlaybookStore


def _evaluator_used(rule: dict) -> dict:
    return {
        "predicate": rule.get("predicate") is not None,
        "similarity": bool(rule.get("reference_text")),
        "nl": bool(rule.get("nl_assertion")),
    }


def _eval_predicate(rule: dict, ctx: dict) -> Optional[bool]:
    pred = rule.get("predicate")
    if not pred:
        return None
    try:
        return bool(evaluate(pred, ctx))
    except Exception:
        return None


def _eval_similarity(rule: dict, ctx: dict) -> Optional[float]:
    """Returns deviation in [0,1] (0=aligned to reference, 1=most deviating).
    None if the rule has no reference or ctx has no embed callable."""
    ref = rule.get("reference_text")
    embed: Optional[Callable] = ctx.get("embed")
    if not ref or embed is None:
        return None
    target = ctx.get("clause_text") or " ".join(
        c.get("text", "") for c in ctx.get("clauses", [])[:3]
    )
    if not target:
        return None
    a = np.asarray(embed(target))
    b = np.asarray(embed(ref))
    if a.size == 0 or b.size == 0:
        return None
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return max(0.0, 1.0 - cos)


def _eval_nl(rule: dict, ctx: dict) -> Optional[tuple[str, str]]:
    assertion = rule.get("nl_assertion")
    judge: Optional[Callable] = ctx.get("nl_judge")
    if not assertion or judge is None:
        return None
    evidence = ctx.get("clause_text") or "\n".join(
        c.get("text", "") for c in ctx.get("clauses", [])[:5]
    )
    return judge(assertion, evidence)


def align(store: PlaybookStore, playbook_id: str, ctx: dict) -> list[dict]:
    run_id = ctx.get("run_id") or str(uuid.uuid4())
    agreement_id = ctx["agreement_id"]
    out: list[dict] = []

    rules = [r for r in store.list_rules(playbook_id) if r["status"] == "active"]
    for rule in rules:
        evaluator_used = _evaluator_used(rule)
        outcomes: list[str] = []
        rationale_parts: list[str] = []
        deviation = 0.0
        answer_value: dict[str, Any] = {}

        # 1. predicate
        if evaluator_used["predicate"]:
            ok = _eval_predicate(rule, ctx)
            if ok is True:
                outcomes.append("pass")
            elif ok is False:
                outcomes.append("fail")
            else:
                outcomes.append("needs_human")
                rationale_parts.append("predicate inconclusive")

        # 2. similarity
        if evaluator_used["similarity"]:
            d = _eval_similarity(rule, ctx)
            if d is None:
                pass
            else:
                deviation = max(deviation, d)
                thr = rule.get("similarity_threshold") or 0.15
                if d > (1.0 - thr):
                    outcomes.append("fail")
                else:
                    outcomes.append("pass")

        # 3. NL judge — only if predicate inconclusive or absent
        run_nl = evaluator_used["nl"] and (
            (not evaluator_used["predicate"]) or "needs_human" in outcomes
        )
        if run_nl:
            res = _eval_nl(rule, ctx)
            if res is not None:
                verdict, why = res
                outcomes.append("fail" if verdict == "fail"
                                  else "pass" if verdict == "pass" else "needs_human")
                rationale_parts.append(why)

        # combiner
        if "fail" in outcomes:
            outcome = "fail"
        elif outcomes and all(o == "pass" for o in outcomes):
            outcome = "pass"
        else:
            outcome = "needs_human"

        evidence = [{"clause_id": c.get("id"),
                     "text": (c.get("text") or "")[:300]}
                    for c in ctx.get("clauses", [])[:3]]

        eid = store.record_eval(
            rule_id=rule["rule_id"], agreement_id=agreement_id, run_id=run_id,
            outcome=outcome, severity=rule["severity"], deviation=deviation,
            answer_value=answer_value or None, evidence=evidence,
            rationale=" / ".join(rationale_parts) or None,
            evaluator_used=evaluator_used,
        )
        out.append({"eval_id": eid, "rule_id": rule["rule_id"],
                     "outcome": outcome, "severity": rule["severity"],
                     "deviation": deviation, "evidence": evidence})

    return out
```

- [ ] **Step 4: Run. Expect PASS.**

- [ ] **Step 5: Commit.**

```
git add core/playbooks/aligner.py scripts/validate_playbook.py
git commit -m "feat(playbooks): alignment engine (predicate + similarity + NL-judge + combiner)"
```

---

### Task 10: Streamlit pages + app wiring

**Files:**

- Create: `ui/playbook_authoring_page.py`
- Create: `ui/playbook_mining_page.py`
- Create: `ui/playbook_findings_page.py`
- Modify: `app.py` (add `📜 Playbooks` section)

- [ ] **Step 1: Add a manual smoke check (no automated UI test).**

Append:

```python
def test_ui_modules_import():
    """Validate: UI modules import without errors (manual visual check still required)."""
    import importlib
    for mod in ("ui.playbook_authoring_page",
                "ui.playbook_mining_page",
                "ui.playbook_findings_page"):
        importlib.import_module(mod)
    print("  [PASS] UI modules importable")
```

Register.

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement the three pages.**

`ui/playbook_authoring_page.py`:

```python
"""Playbook authoring: list playbooks, view rules, edit predicate JSON, publish."""
import json
import streamlit as st

from config import DB_PATH
from core.playbooks.store import PlaybookStore
from core.playbooks.importers import import_file


@st.cache_resource
def _store():
    return PlaybookStore(DB_PATH)


def render(store=None):  # signature kept compatible with app.py convention
    st.header("📜 Playbook authoring")
    pb = _store()
    playbooks = pb.list_playbooks()

    with st.expander("Import a playbook (xlsx/docx)"):
        path = st.text_input("Absolute path to playbook file")
        name = st.text_input("Playbook name (optional)")
        if st.button("Import", disabled=not path):
            try:
                pid = import_file(pb, path, name=name or None)
                st.success(f"Imported as {pid}")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if not playbooks:
        st.info("No playbooks yet. Import or mine some.")
        return

    pb_choice = st.selectbox(
        "Playbook",
        options=[p["playbook_id"] for p in playbooks],
        format_func=lambda i: next(p["name"] for p in playbooks if p["playbook_id"] == i),
    )
    rules = pb.list_rules(pb_choice)
    st.caption(f"{len(rules)} rules")

    for r in rules:
        with st.expander(f"[{r['severity']}] {r['title']}"):
            st.markdown(f"**Applies to:** `{r['applies_to']}` · **Status:** `{r['status']}`")
            if r.get("description"):
                st.markdown(r["description"])
            if r.get("predicate"):
                st.code(json.dumps(r["predicate"], indent=2), language="json")
            if r.get("nl_assertion"):
                st.markdown(f"_NL assertion:_ {r['nl_assertion']}")
            if r.get("preferred_language"):
                st.markdown("**Preferred language:**")
                st.write(r["preferred_language"])
            cols = st.columns(2)
            if cols[0].button("Activate", key=f"act_{r['rule_id']}", disabled=r["status"] == "active"):
                pb.update_rule_status(r["rule_id"], "active"); st.rerun()
            if cols[1].button("Retire", key=f"ret_{r['rule_id']}", disabled=r["status"] == "retired"):
                pb.update_rule_status(r["rule_id"], "retired"); st.rerun()
```

`ui/playbook_mining_page.py`:

```python
"""Mining page: shows draft candidates from imports/miner, with accept/reject."""
import streamlit as st

from config import DB_PATH
from core.playbooks.store import PlaybookStore


@st.cache_resource
def _store():
    return PlaybookStore(DB_PATH)


def render(store=None):
    st.header("⛏️ Playbook mining")
    pb = _store()
    playbooks = pb.list_playbooks()
    if not playbooks:
        st.info("No playbooks yet."); return

    pid = st.selectbox(
        "Playbook",
        options=[p["playbook_id"] for p in playbooks],
        format_func=lambda i: next(p["name"] for p in playbooks if p["playbook_id"] == i),
    )
    drafts = [r for r in pb.list_rules(pid) if r["status"] == "draft"]
    st.caption(f"{len(drafts)} draft candidates")

    for r in drafts:
        st.divider()
        st.markdown(f"### [{r['severity']}] {r['title']}")
        st.markdown(f"_{r.get('description') or ''}_")
        if r.get("source_provenance"):
            st.caption(f"source: {r['source_provenance']}")
        cols = st.columns(3)
        if cols[0].button("Accept", key=f"a_{r['rule_id']}"):
            pb.update_rule_status(r["rule_id"], "active"); st.rerun()
        if cols[1].button("Retire", key=f"r_{r['rule_id']}"):
            pb.update_rule_status(r["rule_id"], "retired"); st.rerun()
        cols[2].write("")  # spacer
```

`ui/playbook_findings_page.py`:

```python
"""Findings page: pick agreement + playbook → run alignment → render findings."""
import streamlit as st

from config import DB_PATH
from core.playbooks.store import PlaybookStore
from core.playbooks.aligner import align


@st.cache_resource
def _store():
    return PlaybookStore(DB_PATH)


def render(store=None):
    st.header("🚨 Playbook findings")
    pb = _store()
    playbooks = [p for p in pb.list_playbooks() if p.get("status") != "archived"]
    if not playbooks:
        st.info("No playbooks yet."); return

    pid = st.selectbox(
        "Playbook",
        options=[p["playbook_id"] for p in playbooks],
        format_func=lambda i: next(p["name"] for p in playbooks if p["playbook_id"] == i),
    )
    agreement_id = st.text_input("Agreement ID")
    if st.button("Run alignment", disabled=not agreement_id):
        # Minimal ctx — extend with field/clause loaders in production
        ctx = {"agreement_id": agreement_id, "fields": {}, "clauses": []}
        findings = align(pb, pid, ctx)
        st.success(f"{len(findings)} rules evaluated")

    # Show recent findings for this agreement
    if agreement_id:
        rows = pb.conn.execute(
            "SELECT * FROM rule_evaluations WHERE agreement_id=? ORDER BY evaluated_at DESC LIMIT 50",
            [agreement_id],
        ).fetchall()
        if rows:
            cols = [d[0] for d in pb.conn.description]
            for r in rows:
                d = dict(zip(cols, r))
                st.write(f"**{d['severity']}** · {d['outcome']} · rule `{d['rule_id'][:8]}` "
                         f"· deviation {d['deviation']:.2f}")
```

Modify `app.py` — find the `SECTIONS` dict and add a new entry between Author and Analyze:

```python
    "📜 Playbooks": [
        ("📜 Authoring", "ui.playbook_authoring_page"),
        ("⛏️ Mining",   "ui.playbook_mining_page"),
        ("🚨 Findings",  "ui.playbook_findings_page"),
    ],
```

- [ ] **Step 4: Run validation, then smoke-test the UI manually.**

```
poetry run python scripts/validate_playbook.py
poetry run streamlit run app.py --server.port 8501
```

Expected validation: `[PASS] UI modules importable`. Visit the app, click into the new `📜 Playbooks` section, verify all three pages render without errors. Import the Cloudera fixture by absolute path on the Authoring page — should show ≥10 rules.

- [ ] **Step 5: Commit.**

```
git add ui/playbook_authoring_page.py ui/playbook_mining_page.py \
        ui/playbook_findings_page.py app.py scripts/validate_playbook.py
git commit -m "feat(playbooks): three Streamlit pages (authoring/mining/findings) + app wiring"
```

---

### Task 11: API endpoints + pipeline flag + benchmark migration

**Files:**

- Modify: `api_server.py` (add 4 endpoints)
- Modify: `core/evoc_pipeline.py` (add `--playbook` flag + final-stage call)
- Create: `scripts/migrate_benchmarks_to_playbook.py`

- [ ] **Step 1: Add failing check covering migration only.**

```python
def test_benchmark_migration():
    """Validate: migration script lifts existing playbook_benchmarks rows into a playbook."""
    import tempfile
    import duckdb
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tf:
        # seed legacy benchmark
        c = duckdb.connect(tf.name)
        c.execute("""CREATE TABLE IF NOT EXISTS playbook_benchmarks (
            cluster_id VARCHAR, cluster_label VARCHAR, benchmark_clause_id VARCHAR,
            benchmark_text TEXT, source VARCHAR, mean_deviation DOUBLE,
            std_deviation DOUBLE, n_clauses INTEGER)""")
        c.execute("""INSERT INTO playbook_benchmarks VALUES
                     ('c1','Indemnification','b1','standard text','centroid',0.1,0.05,42)""")
        c.close()

        from scripts.migrate_benchmarks_to_playbook import migrate
        pid = migrate(tf.name, name="Auto-mined baseline")

        from core.playbooks.store import PlaybookStore
        s = PlaybookStore(tf.name)
        rules = s.list_rules(pid)
        assert len(rules) == 1
        assert rules[0]["reference_text"] == "standard text"
        s.close()
    print("  [PASS] Benchmark migration")
```

- [ ] **Step 2: Run. Expect FAIL.**

- [ ] **Step 3: Implement migration + endpoints + pipeline flag.**

`scripts/migrate_benchmarks_to_playbook.py`:

```python
"""One-off migration: promote core/playbook.py's playbook_benchmarks rows
into the new playbook subsystem as a default 'Auto-mined baseline' playbook.

Each row → one rule with applies_to=cluster, similarity_threshold=0.85,
reference_text=benchmark_text. Idempotent: re-running creates a new playbook
each time (caller should retire the old one).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.store import PlaybookStore


def migrate(db_path: str, *, name: str = "Auto-mined baseline") -> str:
    s = PlaybookStore(db_path)
    rows = s.conn.execute(
        "SELECT cluster_id, cluster_label, benchmark_text, source, mean_deviation "
        "FROM playbook_benchmarks"
    ).fetchall()
    pid = s.create_playbook(
        name=name, owner_org="auto",
        description=f"Migrated {len(rows)} cluster benchmarks from playbook_benchmarks",
    )
    for cluster_id, cluster_label, text, source, mean_dev in rows:
        rid = s.create_rule(
            playbook_id=pid,
            title=f"{cluster_label} similarity to benchmark",
            applies_to="cluster",
            severity="info",
            reference_text=text,
            similarity_threshold=max(0.5, 1.0 - 2.0 * (mean_dev or 0.0)),
            tags=[cluster_label, source or "unknown", "auto"],
            source_provenance={"migrated_from": "playbook_benchmarks", "cluster_id": cluster_id},
            status="draft",
        )
        s.add_binding(rule_id=rid, entity_kind="cluster",
                       entity_id=cluster_id, label_text=cluster_label, confidence=1.0)
    s.close()
    return pid


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("db_path")
    p.add_argument("--name", default="Auto-mined baseline")
    args = p.parse_args()
    pid = migrate(args.db_path, name=args.name)
    print(f"Created playbook: {pid}")
```

`api_server.py` — append after existing endpoints (insert imports at top of file alongside existing imports):

```python
from core.playbooks.store import PlaybookStore
from core.playbooks.importers import import_file as _pb_import_file
from core.playbooks.aligner import align as _pb_align


class PlaybookImportRequest(BaseModel):
    path: str = Field(..., description="Absolute path to xlsx or docx playbook")
    name: str | None = None
    owner_org: str | None = None


@app.post("/playbooks/import")
def playbooks_import(req: PlaybookImportRequest):
    s = PlaybookStore(os.environ.get("DB_PATH", "data/evoc_200_refined.duckdb"))
    pid = _pb_import_file(s, req.path, name=req.name, owner_org=req.owner_org or "")
    s.close()
    return {"playbook_id": pid}


@app.get("/playbooks/{playbook_id}")
def playbooks_get(playbook_id: str):
    s = PlaybookStore(os.environ.get("DB_PATH", "data/evoc_200_refined.duckdb"))
    pb = s.get_playbook(playbook_id)
    if not pb:
        s.close()
        raise HTTPException(status_code=404, detail="not found")
    rules = s.list_rules(playbook_id)
    s.close()
    return {"playbook": pb, "rules": rules}


class AlignRequest(BaseModel):
    agreement_id: str
    fields: dict | None = None
    clauses: list[dict] | None = None


@app.post("/playbooks/{playbook_id}/run")
def playbooks_run(playbook_id: str, req: AlignRequest):
    s = PlaybookStore(os.environ.get("DB_PATH", "data/evoc_200_refined.duckdb"))
    ctx = {"agreement_id": req.agreement_id,
           "fields": req.fields or {}, "clauses": req.clauses or []}
    findings = _pb_align(s, playbook_id, ctx)
    s.close()
    return {"findings": findings}


@app.get("/findings/{run_id}")
def findings_get(run_id: str):
    s = PlaybookStore(os.environ.get("DB_PATH", "data/evoc_200_refined.duckdb"))
    rows = s.findings_for_run(run_id)
    s.close()
    return {"findings": rows}
```

`core/evoc_pipeline.py` — locate the argparse setup in `main()` (search for `add_argument(`) and add:

```python
    parser.add_argument("--playbook", default=None,
                        help="Playbook ID to evaluate against after extraction (optional)")
```

At the end of the run, before logging the run summary, add:

```python
    if args.playbook:
        from core.playbooks.store import PlaybookStore
        from core.playbooks.aligner import align
        pb = PlaybookStore(db_path)
        all_findings = []
        for ag in store.get_agreements():
            extractions = store.get_extractions(agreement_id=ag["agreement_id"])
            fields = {e.get("field_name") or e.get("field_id"): e.get("value")
                      for e in extractions}
            clauses_rows = store.get_clauses(agreement_id=ag["agreement_id"])
            clauses = [{"id": c.get("clause_id"),
                         "label": c.get("clause_type") or c.get("clause_type_id"),
                         "text": c.get("text")} for c in clauses_rows]
            ctx = {"agreement_id": ag["agreement_id"],
                    "fields": fields, "clauses": clauses}
            all_findings.extend(align(pb, args.playbook, ctx))
        pb.close()
        logger.info(f"Playbook alignment: {len(all_findings)} findings")
```

(`store.get_agreements`, `store.get_extractions(agreement_id=...)`, and `store.get_clauses(agreement_id=...)` are the existing methods in `core/store.py`; the aligner accepts an empty `fields`/`clauses` so a missing extraction set still keeps the pipeline running.)

- [ ] **Step 4: Run validation; smoke-test API.**

```
poetry run python scripts/validate_playbook.py
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000 &
sleep 2
curl -X POST http://localhost:8000/playbooks/import \
     -H "Content-Type: application/json" \
     -d '{"path":"'"$(pwd)"'/tests_fixtures/playbooks/cloudera.xlsx","name":"smoke"}'
kill %1
```

Expected: `{"playbook_id": "..."}`.

- [ ] **Step 5: Commit.**

```
git add api_server.py core/evoc_pipeline.py scripts/migrate_benchmarks_to_playbook.py \
        scripts/validate_playbook.py
git commit -m "feat(playbooks): API endpoints + evoc_pipeline --playbook flag + benchmark migration"
```

---

### Task 12: Eval harness

**Files:**

- Create: `eval/playbook_eval.py`

- [ ] **Step 1: Write the harness skeleton + a self-check.**

`eval/playbook_eval.py`:

```python
"""Evaluation harness: measure findings precision/recall vs a small gold set.

Gold set JSON shape (eval/playbook_gold.json):
  [
    {"agreement_id": "a1", "playbook_id": "p1",
     "expected": [{"rule_id": "r1", "outcome": "fail"},
                   {"rule_id": "r2", "outcome": "pass"}]},
    ...
  ]

Usage:
  poetry run python -m eval.playbook_eval data/evoc_200_refined.duckdb p1 \\
      --gold eval/playbook_gold.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.aligner import align
from core.playbooks.store import PlaybookStore


def evaluate(db_path: str, playbook_id: str, gold_path: str) -> dict:
    gold = json.loads(Path(gold_path).read_text())
    pb = PlaybookStore(db_path)
    tp = fp = fn = 0
    for entry in gold:
        ctx = {"agreement_id": entry["agreement_id"],
                "fields": entry.get("fields", {}),
                "clauses": entry.get("clauses", [])}
        actual = {f["rule_id"]: f["outcome"]
                   for f in align(pb, playbook_id, ctx)}
        for exp in entry["expected"]:
            rid, want = exp["rule_id"], exp["outcome"]
            got = actual.get(rid)
            if got == want and want == "fail":
                tp += 1
            elif got == "fail" and want != "fail":
                fp += 1
            elif got != "fail" and want == "fail":
                fn += 1
    pb.close()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db_path")
    p.add_argument("playbook_id")
    p.add_argument("--gold", default="eval/playbook_gold.json")
    args = p.parse_args()
    result = evaluate(args.db_path, args.playbook_id, args.gold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-run on an empty gold file.**

```bash
echo "[]" > eval/playbook_gold.json
poetry run python -m eval.playbook_eval data/evoc_200_refined.duckdb dummy --gold eval/playbook_gold.json
```

Expected: `{"tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,"f1":0.0}` — and exit code 0.

- [ ] **Step 3: Run the full validate harness one last time.**

```
poetry run python scripts/validate_playbook.py
```

Expected: all checks PASS (≈11 total).

- [ ] **Step 4: Commit.**

```
git add eval/playbook_eval.py eval/playbook_gold.json
git commit -m "feat(playbooks): evaluation harness measuring findings precision/recall"
```

- [ ] **Step 5: Final summary commit / PR-ready state.**

Tag a final commit summarizing the slice of work that landed (no code change in this step):

```
git log --oneline -n 12
```

---

## Spec coverage cross-check

| Spec section                                                            | Implemented in task                                         |
| ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| §3 Architecture (3 units)                                               | Tasks 2, 5–7, 8, 9 (one task per unit at minimum)           |
| §4 Schema (5 tables)                                                    | Task 2                                                      |
| §5 Predicate DSL                                                        | Task 4 (registry + 4 op modules + evaluator)                |
| §6 HITL miner (coverage / distribution / outlier / soft-rebind)         | Task 8                                                      |
| §7.1 Tabular importer (Cloudera + Walmart deployment-logic parser)      | Task 5                                                      |
| §7.2 Narrative importer (Docusign)                                      | Task 6                                                      |
| §7.3 Desirable/undesirable importer (Sample AI Playbook)                | Task 7                                                      |
| §8 Alignment engine (predicate → similarity → NL judge → combiner)      | Task 9                                                      |
| §9 Streamlit pages (authoring / mining / findings)                      | Task 10                                                     |
| §10 evoc_pipeline `--playbook` flag, API endpoints, benchmark migration | Task 11                                                     |
| §11 Tests (`scripts/validate_playbook.py`)                              | Tasks 1–10 incrementally; covers the 8 checks listed in §11 |
| §11 Eval harness (`eval/playbook_eval.py`)                              | Task 12                                                     |

**Outlier-only candidates from §6** (cluster-centroid based) are partially covered: the benchmark migration in Task 11 lifts existing centroid-based benchmarks into similarity-only rules, which is the equivalent shape. A pure miner-side outlier candidate generator can be added as a future increment (signal source #4 in §6's table).

## Manual setup notes

- Fixtures (Task 5–7) live under `.claude/hackathon/tests_fixtures/playbooks/` (gitignored). Expect repository-relative copies of the three files in the top-level `data/` folder. If running the project from a different machine, re-copy or skip those tests (each has a `[SKIP]` guard).
- `python-docx` is added in Task 5 via `poetry add python-docx`.
- Streamlit smoke test (Task 10) requires manually running the app and clicking through.
- API smoke test (Task 11) starts the server with `&`; remember to `kill %1`.
