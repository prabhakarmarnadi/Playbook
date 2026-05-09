# Playbook Rules — Design Spec

**Date:** 2026-05-09
**Status:** Draft for review
**Related project:** `.claude/hackathon/` (Clustering V2)

---

## 1. Problem & goals

During contract review (negotiation phase or portfolio audit), users want to identify
contracts that **don't align with their organization's playbook**. Today playbooks
exist as ad-hoc xlsx/docx (see `data/`): tabular question-and-answer rulebooks
(Cloudera, Walmart), narrative escalation-driven playbooks (Docusign), and
desirable/undesirable language pairs (Sample AI Playbook). None of these are
machine-evaluable, none reference a shared ontology, and none close the loop with
the corpus statistics that the existing clustering pipeline already produces.

This spec proposes a **playbook subsystem** that:

1. Defines an extensible schema for representing playbook rules at multiple
   dimensions — document, domain, legal pattern (cluster), field, composite, and
   cross-field.
2. Mines rule candidates from the corpus (cluster coverage, field value
   distributions, outlier scoring) and presents them through a human-in-the-loop
   review UI.
3. Imports the three sample playbook formats found in `data/` so existing
   organizational knowledge becomes a starting point, not a throwaway.
4. Evaluates a published playbook against a contract and emits negotiation-ready
   findings with severity, evidence, and pre-approved redline language.

Non-goals (this spec): full CLM workflow integration, e-signature, redline insertion
into Word, multi-tenant authentication. Those belong to later phases.

## 2. Context: what already exists

The hackathon project (`.claude/hackathon/`) already produces, per run:

- `domains` — macro clusters of agreement types (MSA, NDA, …).
- `clusters` — micro clusters of clause/legal patterns within each domain.
- `field_definitions` and `extractions` — per-cluster fields with extracted values.
- `composite_definitions` — derived fields with cascade propagation.
- `cluster_centroids` — embedding centroid per cluster, used for outlier scoring.
- A thin `core/playbook.py` that records a single benchmark clause per cluster and
  scores clauses by similarity to that benchmark. Useful but single-dimensional.

The new subsystem **layers on top of these IDs** (`domain_id`, `cluster_id`,
`field_id`, `composite_id`) rather than redefining a parallel taxonomy. Rules bind
to ontology entities; the binding survives re-clustering via stored label
embeddings (soft rebinding).

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Existing pipeline (cluster_pipeline / evoc_pipeline)                  │
│   PDF → chunks → clauses → domain → cluster → field_definitions        │
│   → extractions  (DuckDB + LanceDB)                                    │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │  ontology IDs (domain_id, cluster_id, field_id)
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Playbook subsystem (new — core/playbooks/)                            │
│                                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐  │
│  │ Importers    │──▶│ Rule store   │◀──│ Miner (HITL suggester)     │  │
│  │ (xlsx/docx)  │   │ (DuckDB)     │   │ — reads cluster + field    │  │
│  └──────────────┘   └──────┬───────┘   │   distributions, proposes  │  │
│                            │           │   rule drafts              │  │
│                            ▼           └────────────────────────────┘  │
│                    ┌────────────────┐                                  │
│                    │ Alignment eng. │                                  │
│                    │ (predicate +   │  ──▶  rule_evaluations           │
│                    │  NL judge +    │       findings: {rule_id,        │
│                    │  similarity)   │        contract_id, severity,   │
│                    └────────────────┘        evidence, deviation}     │
│                            │                                          │
│                            ▼                                          │
│           Streamlit HITL pages: Authoring · Mining · Findings         │
└────────────────────────────────────────────────────────────────────────┘
```

Three discrete units, each with one purpose:

- **Rule store + DSL** (`core/playbooks/store.py`, `core/playbooks/predicates/`):
  CRUD over rules, JSON predicate evaluation, no I/O outside DuckDB.
- **Importers + Miner** (`core/playbooks/importers/`, `core/playbooks/miner.py`):
  produce _draft_ rules. Never publishes.
- **Alignment engine** (`core/playbooks/aligner.py`): reads published rules,
  evaluates against a single agreement, writes `rule_evaluations`.

Each unit is independently testable.

## 4. Schema (DuckDB)

Five new tables. All use `CREATE TABLE IF NOT EXISTS` and idempotent
`ALTER … ADD COLUMN IF NOT EXISTS` per project convention. Schema init runs from
`core/playbooks/store.py::PlaybookStore._init_schema()`.

```sql
-- 4.1 Playbooks: an authored policy collection.
CREATE TABLE IF NOT EXISTS playbooks (
    playbook_id   VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    owner_org     VARCHAR,                  -- e.g. "Cloudera", "Docusign"
    description   TEXT,
    source_file   VARCHAR,                  -- import provenance
    status        VARCHAR DEFAULT 'draft',  -- draft | published | archived
    version       INTEGER DEFAULT 1,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    updated_at    TIMESTAMP DEFAULT current_timestamp
);

-- 4.2 Rules: the unit of evaluation. Hybrid representation: any combination of
--     predicate / nl_assertion / reference_text. Engine runs whichever are populated.
CREATE TABLE IF NOT EXISTS playbook_rules (
    rule_id              VARCHAR PRIMARY KEY,
    playbook_id          VARCHAR NOT NULL,
    title                VARCHAR NOT NULL,
    description          TEXT,
    applies_to           VARCHAR NOT NULL,   -- document | domain | cluster | field | composite | cross_field
    predicate            JSON,               -- {op, args} expression tree (see §5)
    nl_assertion         TEXT,               -- LLM-judge assertion
    reference_text       TEXT,               -- exemplar standard clause for similarity
    similarity_threshold DOUBLE,             -- below = non-standard
    answer_type          VARCHAR,            -- yes_no | standard_nonstandard | numeric_range | enum | presence
    answer_options       JSON,               -- e.g. ["Standard","Non-Standard","Missing"]
    severity             VARCHAR NOT NULL,   -- info | warn | approval_required | blocker
    escalation_owner     VARCHAR,
    preferred_language   TEXT,
    fallback_language    JSON,               -- ordered array of fallback clauses
    walkaway_language    TEXT,
    rationale            TEXT,
    tags                 JSON,
    source_provenance    JSON,               -- {file, sheet, row}
    status               VARCHAR DEFAULT 'draft',  -- draft | active | retired
    created_by           VARCHAR,
    created_at           TIMESTAMP DEFAULT current_timestamp,
    updated_at           TIMESTAMP DEFAULT current_timestamp
);

-- 4.3 Bindings: which ontology entities the rule targets. A rule may bind to
--     multiple entities. Soft-rebinding via embedding similarity if entity_id
--     becomes stale after re-clustering.
CREATE TABLE IF NOT EXISTS rule_bindings (
    binding_id      VARCHAR PRIMARY KEY,
    rule_id         VARCHAR NOT NULL,
    entity_kind     VARCHAR NOT NULL,    -- domain | cluster | field | composite
    entity_id       VARCHAR NOT NULL,
    label_text      VARCHAR,             -- original taxonomy label from source
    label_embedding BLOB,                -- for soft rebind
    confidence      DOUBLE               -- 1.0 = explicit; <1.0 = inferred
);

-- 4.4 Revisions: every edit is recorded. Audit + rollback.
CREATE TABLE IF NOT EXISTS rule_revisions (
    revision_id   VARCHAR PRIMARY KEY,
    rule_id       VARCHAR NOT NULL,
    revision_no   INTEGER NOT NULL,
    diff          JSON,
    changed_by    VARCHAR,
    change_reason TEXT,
    changed_at    TIMESTAMP DEFAULT current_timestamp
);

-- 4.5 Evaluations: alignment results.
CREATE TABLE IF NOT EXISTS rule_evaluations (
    eval_id        VARCHAR PRIMARY KEY,
    rule_id        VARCHAR NOT NULL,
    agreement_id   VARCHAR NOT NULL,
    run_id         VARCHAR,             -- groups a single playbook run
    outcome        VARCHAR NOT NULL,    -- pass | fail | n/a | needs_human
    answer_value   JSON,
    severity       VARCHAR NOT NULL,
    deviation      DOUBLE,              -- 0.0 = aligned, 1.0 = max
    evidence       JSON,                -- [{chunk_id, clause_id, span, score}]
    rationale      TEXT,
    evaluator_used JSON,                -- {"predicate":true,"nl":false,"sim":true}
    evaluated_at   TIMESTAMP DEFAULT current_timestamp
);
```

### Mirror Pydantic models

Pydantic models in `core/playbooks/models.py` mirror these tables and double as
JSON Schema source for an HTTP API and for round-tripping rules to YAML files (so
playbooks can be version-controlled outside DuckDB if a customer prefers).

## 5. The predicate DSL

Predicates are a small s-expression-style JSON tree. Every node is `{op, args}`.
Operators are registered in `core/playbooks/predicates/` via decorator and grouped:

```
field operators:   field.value, field.exists, field.in, field.matches,
                   field.between, field.gte, field.lte, field.eq,
                   field.semantic_match
clause operators:  clause.contains, clause.absent, clause.similar_to,
                   clause.classified_as
logical:           and, or, not, if_then, count, all_of, any_of
context:           domain.is, doc.party_role.is, doc.value.gte
```

### Worked examples (covering all three sample playbooks)

```json
// Walmart deployment logic: "Indemnification Standard if z_..._Std_SP OR z_..._Std_SS is set"
{"op": "or", "args": [
  {"op": "field.exists", "args": ["Indemnification_Standard_SP"]},
  {"op": "field.exists", "args": ["Indemnification_Standard_SS"]}
]}

// Cloudera "Late payment penalty must be a percentage" — answer_type=enum
{"op": "field.matches", "args": ["late_payment_penalty", "regex", "\\d+(\\.\\d+)?%"]}

// Sample AI Playbook: "Liability cap ≥ greater of $250k or fees paid"
{"op": "and", "args": [
  {"op": "field.exists", "args": ["liability_cap_amount"]},
  {"op": "or", "args": [
    {"op": "field.gte", "args": ["liability_cap_amount", 250000]},
    {"op": "field.gte", "args": ["liability_cap_ratio_fees", 1.0]}
  ]}
]}

// Cross-field: "If governing_law=NY then arbitration_venue must be NY"
{"op": "if_then", "args": [
  {"op": "field.eq", "args": ["governing_law", "New York"]},
  {"op": "field.eq", "args": ["arbitration_venue", "New York"]}
]}

// Domain-level: "MSA must contain an Indemnification clause"
// rule.applies_to = "domain"; bound via rule_bindings(entity_kind="domain",
// entity_id=<msa_domain_id>). Predicate is evaluated existentially across the
// document's clauses in that domain (see §8 step 2).
{"op": "any_of", "args": [
  {"op": "clause.classified_as", "args": ["Indemnification"]}
]}
```

### Why a DSL rather than Python lambdas

- Rules round-trip as data — exportable, diffable, version-controllable.
- Storage in DuckDB JSON column is trivial.
- Evaluator is small (≈200 lines); auditable per operator.
- A future "predicate builder" UI can render the same JSON without code-gen.

### Adding a new operator

1. Implement `def op_field_eq(args, ctx) -> bool` in
   `core/playbooks/predicates/field.py`.
2. Register with `@predicate("field.eq")`.
3. Document semantics in the operator's docstring (one short line).

That is the only required change. No DuckDB schema migration; predicates remain
opaque JSON to the storage layer.

## 6. HITL rule mining

The miner runs after a clustering pipeline run and produces **rule candidates**
in `draft` status. Mining never auto-publishes.

| Signal source                                                                                 | Candidate rule produced                                                           |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Cluster coverage per domain (e.g. 97% of MSAs have an Indemnification cluster)                | `applies_to=domain`, presence rule                                                |
| Field value distribution per cluster (e.g. `liability_cap` median = 12 × ARR, p95 = 24 × ARR) | `applies_to=field`, numeric_range rule with thresholds at distribution boundaries |
| Field categorical mode (e.g. `governing_law` = "Delaware" in 89% of corpus)                   | `applies_to=field`, enum rule with default = mode                                 |
| Cluster centroid + outlier scoring (`clause_deviation.py`)                                    | `applies_to=cluster`, similarity rule                                             |
| Imported playbook xlsx/docx                                                                   | `applies_to` inferred from Topic→cluster + Sub-topic→sub-cluster mapping          |

Each candidate is scored on:

- **lift** — fraction of corpus contracts that would be flagged by this rule.
- **stability** — agreement rate of the rule across resamples of the corpus.

Candidates surface in `ui/playbook_mining_page.py` ranked by `lift × stability`,
with a side-by-side preview: rule predicate + the 3 most-aligned and 3 most-
deviating clauses. The reviewer:

1. Edits the rule (predicate / NL / threshold / severity / escalation),
2. Adds preferred + fallback language (auto-suggested from cluster centroid clause),
3. Approves → rule moves from `draft` → `active`.

**Soft rebinding.** When clustering is re-run and `cluster_id` values change, the
binding's `label_embedding` is matched against the new cluster centroids; if the
top match exceeds 0.85 cosine similarity, the binding is updated automatically
(`confidence < 1.0`). Below 0.85, the rule is parked in a "needs rebind" queue
visible in the authoring page.

## 7. Importers

One importer per playbook shape, all output `playbook_rules` + `rule_bindings` in
`draft` status. Auto-detection by sheet headers; explicit override via CLI flag.

### 7.1 Tabular importer (Cloudera, Walmart)

Header detection: contains `Topic` AND (`Sub-Topic` OR `Subtopic`) AND `Question`
AND `Answer Type`. One row → one rule.

- `Topic` → soft-bind to cluster label via embedding similarity over existing
  cluster labels.
- `Sub-Topic` → adds tag and refines binding (sub-cluster if available).
- `Question` → `description`.
- `Answer Type` (e.g. `Standard;\nNon-Standard;\nNo Extraction`) → parses to
  `answer_type` + `answer_options`.
- Walmart-only: `Deployment Logic` (e.g.
  `IF X_Std_SP IS SET --> Answer = Standard`) parses into a `predicate` JSON tree
  via a small regex-driven parser at `core/playbooks/importers/walmart_logic.py`.
- `Customer Example Languge` (Cloudera typo preserved) → `reference_text`.

### 7.2 Narrative importer (Docusign)

Header detection: contains `Section/Topic` AND `Customer Issue/Request` AND
`Approvals/Notes/Responses` AND `Escalation Owner(s)`. The importer groups
consecutive rows by section header (column A) so a `Clause` row plus its
sub-rows become one rule.

- `Section/Topic` → soft-bind to cluster.
- `Customer Issue/Request` → `nl_assertion` (LLM-judge target).
- `Edits (if any)` → `preferred_language` (the row's "standard" text) or
  `fallback_language` if multiple consecutive rows exist for one section.
- `Approvals/Notes/Responses` → parsed for "Approval Required:" prefix; if
  present, `severity = approval_required`, otherwise `warn`.
- `Escalation Owner(s)` → `escalation_owner`.

### 7.3 Desirable/Undesirable importer (Sample AI Playbook docx)

Detection: docx with a single 5-column table whose header row is
`Topic / Sources / AI Playbook Rule / Desirable Sample Clause / Undesirable Sample Clause`.

- `AI Playbook Rule` → `nl_assertion`.
- `Desirable Sample Clause` → `reference_text` (positive exemplar).
- `Undesirable Sample Clause` → `walkaway_language` (used as anti-pattern
  reference; alignment engine penalizes similarity to this text).
- `Sources` → `rationale`.

Provenance preserved in `source_provenance JSON` so revert/diff is trivial.

## 8. Alignment engine

Given an `agreement_id` and a `playbook_id`:

1. Resolve all active `playbook_rules` × `rule_bindings` for the playbook.
2. For each rule, fetch the relevant evidence by `applies_to`:
   - `document` → full document summary + chunks.
   - `domain` → domain assignment + summary.
   - `cluster` → matching clauses for that `cluster_id`.
   - `field` → extracted value from `extractions` (or compute on-demand if missing).
   - `composite` → composite value from `composite_definitions`.
   - `cross_field` → all referenced field values.
3. Run evaluators in order:
   - **predicate** (cheap, deterministic) — runs first if `predicate` is set.
   - **similarity** (cached) — if `reference_text` is set, score embedding cosine
     between bound clause text and `reference_text`. Below `similarity_threshold`
     contributes to `deviation`.
   - **nl_judge** (LLM) — runs only if `nl_assertion` is set AND
     (predicate is unset OR predicate result was inconclusive). Uses the project's
     `llm_client.py` with a templated prompt; result is `pass | fail | n/a`.
4. Combine outcomes per rule into a single
   `Finding(rule_id, outcome, severity, deviation, evidence, rationale)`.
   Combination policy: `fail` if any evaluator says `fail`; `pass` if all set
   evaluators say `pass`; otherwise `needs_human`.
5. Write rows to `rule_evaluations`; emit a `Findings` JSON for the UI.

For domain-scoped rules (`applies_to=domain`), step 2's "matching evidence" is
the set of clauses assigned to the bound domain in this document; predicates
that need set-level semantics use `any_of` / `all_of` / `count` over that set.
For document-scoped rules, the evidence set is the whole document.

### Cost shape

For a corpus of N contracts × R rules:

- Predicate-only rules: O(N × R) DuckDB scans, sub-second per rule.
- Similarity rules: O(N × R) cosine over already-cached embeddings.
- NL-judge rules: O(N × R_nl) LLM calls — bounded by gating on predicate result.
  Expected R_nl ≪ R because most imported rules will have a predicate companion
  after the HITL author session.

## 9. HITL UI pages

Three new Streamlit pages added to `ui/`, each exposing `render(store)` and
registered in `app.py`'s `SECTIONS` dict under a new "Playbooks" section:

- `ui/playbook_authoring_page.py` — list playbooks, create/edit rules, predicate
  builder (form + JSON view), bind to ontology entities, preview matching clauses
  from corpus, manage 3-tier language. Surfaces the "needs rebind" queue.
- `ui/playbook_mining_page.py` — ranked candidate list, accept/edit/reject.
  Imported rules also land here for review before publishing.
- `ui/playbook_findings_page.py` — pick agreement + playbook → run alignment →
  render findings grouped by severity, with evidence highlighted and 3-tier
  language ready to copy as redlines.

## 10. Integration points

- `core/evoc_pipeline.py`: optional `--playbook <playbook_id>` flag wires the
  alignment engine as a final stage so a single CLI run produces extractions
  **and** alignment findings.
- `api_server.py`: new endpoints `POST /playbooks/import`,
  `GET /playbooks/{id}`, `POST /playbooks/{id}/run?agreement_id=…`,
  `GET /findings/{run_id}`. Long-running operations follow the existing
  background-task + in-memory `_jobs` pattern.
- `core/playbook.py` (existing) is **not removed** — the existing
  `playbook_benchmarks` table and similarity scores become one specific rule
  _type_ (`applies_to=cluster`, similarity-only) inside the new system. A
  one-off migration script promotes existing benchmarks into a default
  "Auto-mined baseline" playbook.

## 11. Testing & evaluation

No pytest is introduced (project convention). Tests live in
`scripts/validate_playbook.py` with 8 checks:

1. `PlaybookStore._init_schema()` is idempotent.
2. Tabular importer round-trips every row in `data/Cloudera Insight Ruelbook.xlsx`
   without loss.
3. Narrative importer parses Docusign MSA sheet without crashing on merged cells
   or row-grouping.
4. Desirable/Undesirable importer extracts all rows from
   `data/Sample AI playbook 1.1.docx`.
5. Predicate DSL evaluates each example in §5 deterministically.
6. Miner produces ≥1 candidate per cluster on a 5-PDF mini-corpus.
7. Alignment engine returns identical results across two consecutive runs
   (determinism for predicate + similarity; NL judge fixtures stubbed).
8. Severity → escalation routing matches the `escalation_owner` field.

`eval/playbook_eval.py` measures findings precision/recall against a small
lawyer-annotated gold set (start with 20 contracts × 10 rules; reuses
`eval/runner` patterns).

## 12. Open questions

- **Gold set source.** We need ≈20 lawyer-annotated alignment results to
  benchmark. Initial proposal: sample 20 from `~/maf/Public_Contracts` (e.g.
  `Cloud_Hosting`, `Consulting_Agreements`) and annotate against the imported
  Cloudera playbook. Alternative: use CUAD subset already referenced by
  `scripts/resume_cuad.py`.
- **NL-judge model choice.** Default to whatever `LLM_BACKEND` is configured
  (OpenAI / Azure OpenAI / Ollama). For deterministic tests, stub at the
  `llm_client.py` boundary.
- **Multi-tenant.** Phase-1 spec is single-tenant DuckDB. Multi-tenant follows
  the roadmap's per-customer-DuckDB pattern with no schema change.

## 13. Sources & prior art

- ContractEval benchmark for LLM clause-level legal risk identification —
  https://arxiv.org/abs/2508.03080
- Industry playbook structure (3-tier language, escalation tiers, rule engine
  with AI suggestions) — Sirion, Spellbook, Summize, Malbek, HyperStart product
  documentation, surveyed May 2026.
- Project-internal: `.claude/clustering_v2_roadmap.md` Phase 2 (corpus analytics
  & feedback loop) and Phase 3 (playbook integration), this spec realizes the
  rule-engine half of those phases.
