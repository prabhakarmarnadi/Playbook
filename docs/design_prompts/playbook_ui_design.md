# Design prompt — Playbook intelligence UI

> A copy-and-paste prompt for handing to Claude (Sonnet/Opus) or another design-
> capable LLM to generate UX exploration, wireframes, or production component
> code for the Playbook subsystem.
>
> Two delivery modes are described at the end — pick one when you paste:
> **(A) Wireframe + design rationale** (early exploration), or
> **(B) Production-ready React/Streamlit components** (build-out).

---

## How to use this file

1. Copy everything from `## PROMPT BEGINS` down to the bottom into a Claude chat.
2. Optionally append a **few sample rows** from the actual DuckDB (an export from
   `scripts/export_playbook.py data/demo.duckdb` works well — keep it short).
3. Append your chosen `MODE: A` or `MODE: B` line at the end.
4. Iterate: ask for specific views to be redesigned, ask for accessibility audits,
   ask for a Tailwind component reference.

---

## PROMPT BEGINS

You are a senior product designer and front-end engineer with deep experience
in legal-tech tools (think DocuSign CLM, Ironclad, Spellbook, Sirion).

Your job: design the **Playbook Intelligence** UI — a workspace where a contract
lawyer reviews AI-discovered legal patterns, validates draft rules mined from a
corpus, and aligns new contracts against an approved playbook.

You will design for three personas and four visualization domains. Treat each
visualization as a self-contained component that can be combined into pages.

---

### 1. The product context (you don't need to redesign this part)

The underlying system already exists. Its job is to:

1. Ingest a corpus of contract PDFs.
2. Cluster them into **domains** (NDA, MSA, Consulting, etc.) and within each
   domain into **clause types** (Indemnification, Limitation of Liability,
   Termination, etc.).
3. Use an LLM to discover **fields** per clause type (e.g. Indemnification has
   `indemnified_party`, `notice_requirement`, `settlement_consent_required`).
4. Extract **values** for each field from each contract.
5. Mine **playbook rule candidates** from corpus statistics — coverage,
   field-value distribution, categorical-mode, and outlier candidates — all in
   `status="draft"` awaiting human review.
6. Let the human review drafts, edit/accept/retire, then **align** new
   contracts against the published rules to produce **findings**.

Today, all of this is exposed through a Streamlit app with three barebones
pages (Authoring, Mining, Findings). That's our starting point. We want
something purpose-built that a lawyer can actually live in.

---

### 2. The three personas

**P1. Contract Lawyer (primary)** — reviews AI-discovered patterns and
validates draft rules. Spends 60–90 min/week on this. Needs to:

- Quickly confirm clusters represent real legal concepts (not garbage).
- See examples and value ranges before accepting/editing thresholds.
- Edit predicate, severity, escalation owner, preferred / fallback language.
- Promote drafts to `active`.

**P2. Compliance Reviewer** — scans a portfolio of contracts for non-conforming
provisions. Needs to:

- See a heatmap-style risk overview of N contracts against an active playbook.
- Drill into any flagged contract to see specific findings + evidence spans.
- Export findings to a CSV/PDF for the contracts team.

**P3. Deal Negotiator** — uses the playbook during active negotiation. Needs:

- One contract → instant findings → recommended redline language for each fail.
- Side-by-side: client's clause vs. playbook's preferred clause.
- Escalation routing visible: who must approve this deviation?

---

### 3. The four visualization domains

You're designing UI for these four concept-families. Each comes with the
underlying data schema and what insights matter most.

#### 3.1 Common legal constructs (cluster taxonomy)

**What it is:** The hierarchical map of what kinds of clauses exist in the
corpus, per domain.

**Data schema:**

```
domains
  - domain_id: str (e.g. "d_msa")
  - label: str    (e.g. "MSA", "NDA", "Consulting")
  - agreement_count: int
  - confidence: float

clusters
  - cluster_id: str (e.g. "c_indem")
  - domain_id: str   (parent)
  - label: str       (e.g. "Indemnification", "Limitation of Liability")
  - keywords: list[str]      (top KeyBERT terms)
  - chunk_count: int         (how many text chunks belong to this cluster)
  - quality_score: float     (cluster cohesion, 0..1, higher better)
  - description: str         (LLM-generated one-liner)
```

**Sample rows (from a real run on 5 Consulting Agreements):**

```
clusters (top 8 by chunk_count):
  cluster_id   | label           | chunk_count | quality | keywords
  ct_618ea282  | Confidentiality | 20          | 0.93    | confidential, information, disclose, party
  ct_e0bcad7d  | Arbitration     | 14          | 0.94    | arbitration, dispute, binding, JAMS
  ct_4dba6dc2  | Governing Law   | 14          | 0.90    | governing, law, jurisdiction, state
  ct_83f4481a  | Indemnification | 12          | 0.93    | indemnify, hold harmless, defend
  ct_750e0fd2  | Compensation    | 11          | 0.92    | fee, payment, compensation, hourly
  ct_0a8f09b8  | Payment Terms   | 10          | 0.95    | invoice, days, due, payment
  ct_d67c78db  | Term            | 9           | 0.92    | term, expiration, effective date
  ct_ff4565a9  | Compliance      | 8           | 0.91    | comply, laws, regulations
```

**Insights this view must surface:**

- Which clause types are common vs rare (size visualization).
- How clusters relate hierarchically (domain → cluster).
- Cluster quality at a glance (low-quality ones need lawyer review).
- Top keywords per cluster as a quick smell test.
- Click-through to the underlying clauses + their source PDFs.

**Visualization options to consider (pick the strongest one for the lawyer
persona):**

- **Treemap** with domains as outer rectangles, clusters as inner rectangles
  sized by chunk_count, colored by quality_score.
- **Sunburst** with domain → cluster radial hierarchy.
- **Force-directed graph** where cluster nodes are sized by chunk_count and
  edges represent intent/association strength.
- **Indented hierarchical table** with sparklines for quality and chunk_count.

---

#### 3.2 Schema (fields per cluster)

**What it is:** For each cluster, the set of fields the LLM discovered, with
type and discovery confidence.

**Data schema:**

```
field_definitions
  - field_id: str
  - cluster_id: str   (parent — must always render in context)
  - name: str         (e.g. "indemnified_party", "cure_period_days")
  - field_type: str   (entity_name | text | boolean | integer | date |
                       percentage | duration | monetary)
  - description: str  (LLM-generated explanation)
  - source: str       ("auto_discovered" | "manual")
  - examples: list    (concrete values seen during discovery)
```

**Sample rows:**

```
cluster_id   cluster_label    field                        field_type    n_extractions
ct_618ea282  Confidentiality  confidentiality_duration     duration      3
ct_618ea282  Confidentiality  intellectual_property_owner  entity_name   14
ct_618ea282  Confidentiality  return_of_materials_deadline duration      2
ct_e0bcad7d  Arbitration      arbitration_rules            text          4
ct_e0bcad7d  Arbitration      governing_law                text          3
ct_e0bcad7d  Arbitration      injunctive_relief_permitted  boolean       4
ct_750e0fd2  Compensation     consultant_name              entity_name   7
ct_750e0fd2  Compensation     consulting_fee               monetary      3
ct_750e0fd2  Compensation     payment_frequency            text          3
```

**Insights this view must surface:**

- Schema completeness per cluster (clusters with <3 fields → lawyer attention).
- Field type distribution (mostly boolean? mostly numeric? Mixed?).
- Field discovery confidence (low n_extractions → field may be hallucinated).
- Ability to add/rename/retire fields manually.

**Visualization options:**

- **Per-cluster expandable card** — cluster header + field list with type badges.
- **Matrix view** — clusters × field-types heatmap, count of fields in each cell.
- **Bento layout** — one cluster per tile, fields as chips inside.

---

#### 3.3 Range of values (per field across the corpus)

**What it is:** For each field, the actual values extracted from contracts,
with summary statistics and example values.

**Data schema:**

```
extractions
  - extraction_id: str
  - agreement_id: str (which contract this came from)
  - field_id: str
  - value: str        (LLM returns everything as strings — even numbers)
  - confidence: float (0..1)
  - source_chunk_id: str (for evidence-jump UI)
```

**Sample distribution stats (computed at view time):**

```
Field: consulting_fee  (monetary)
  N=3 extractions:
    "$450"
    "$200,000"
    "$1,950"
  After numeric coercion: [450, 200000, 1950]
  min=450, max=200000, median=1950, mean=$67,466
  Outlier flag: $200,000 (p10/p90 disagree by 100x)

Field: notice_period_days (duration)
  N=4 extractions, all numeric:
    [10, 15, 30, 30]
  min=10, max=30, p10=10, p25=12.5, median=22.5, p75=30, p90=30
  Suggested rule: field.between(notice_period_days, 10, 30) — severity=info

Field: governing_law (text)
  N=5 extractions:
    "Delaware" (3 occurrences, 60%)
    "California" (1 occurrence, 20%)
    "New York" (1 occurrence, 20%)
  Mode: Delaware (60%)
  Below CATEGORICAL_MODE_THRESHOLD (0.70) → no categorical rule emitted
```

**Insights this view must surface:**

- Numeric range distribution (histogram, box plot, violin).
- Categorical mode + frequency (bar chart, treemap of values).
- Outlier values (visually highlighted — they're often extraction errors).
- The actual extracted text + jump-to-evidence in the source PDF.
- Whether this field is currently driving a mined rule (and which one).

**Visualization options:**

- **Violin or box plot** for numeric fields, **horizontal bar chart** for
  categorical, **calendar heatmap** for dates.
- **Card-stack** — one card per unique value, sized by frequency, with the
  agreement_id badge.
- **Combined view** — distribution chart + a "raw extractions" table beside it,
  sortable/filterable.

---

#### 3.4 Playbook rules (suggested + derived)

**What it is:** The mined playbook drafts (and any derived rules) that a lawyer
must review, edit, accept, or retire. The bridge between corpus statistics and
contract-review policy.

**Data schema:**

```
playbook_rules
  - rule_id: str
  - playbook_id: str (parent playbook)
  - title: str       (e.g. "MSA must contain Indemnification clause")
  - description: str (one-liner of the corpus signal)
  - applies_to: enum (document | domain | cluster | field | composite | cross_field)
  - predicate: dict  (s-expression-style JSON predicate — see below)
  - nl_assertion: str | null  (LLM-judge-evaluated text rule)
  - reference_text: str | null (exemplar standard clause for similarity)
  - similarity_threshold: float | null
  - answer_type: enum (yes_no | standard_nonstandard | numeric_range | enum | presence)
  - answer_options: list[str] | null
  - severity: enum (info | warn | approval_required | blocker)
  - escalation_owner: str | null  (e.g. "Commercial Legal")
  - preferred_language: str | null  (the standard clause)
  - fallback_language: list[str] | null
  - walkaway_language: str | null   (anti-pattern reference)
  - rationale: str | null
  - tags: list[str]
  - source_provenance: dict (where this rule came from)
  - status: enum (draft | active | retired)

rule_bindings
  - rule_id: str
  - entity_kind: enum (cluster | domain | field | composite)
  - entity_id: str    (resolved — points to a real ontology entity)
  - label_text: str   (original label, e.g. "Indemnification")
  - confidence: float (1.0 = explicit binding, <1.0 = soft-rebound)
```

**Predicate examples (the JSON tree that runs at alignment time):**

```json
// Coverage rule: "Contract must contain Indemnification clause"
{"op": "any_of", "args": [
    {"op": "clause.classified_as", "args": ["Indemnification"]}
]}

// Distribution rule: "Notice period must be between 10 and 30 days"
{"op": "field.between", "args": ["notice_period_days", 10, 30]}

// Categorical rule: "Governing law is typically Delaware"
{"op": "field.eq", "args": ["governing_law", "Delaware"]}

// Cross-field rule: "If governing_law=NY then arbitration_venue=NY"
{"op": "if_then", "args": [
    {"op": "field.eq", "args": ["governing_law", "New York"]},
    {"op": "field.eq", "args": ["arbitration_venue", "New York"]}
]}
```

**Sample mined drafts (from a real run):**

```
Title                                              | applies_to | severity | source            | confidence
"ALL DOCUMENTS must contain Confidentiality"       | domain     | warn     | coverage(0.80)    | 1.0
"ALL DOCUMENTS must contain Arbitration"           | domain     | warn     | coverage(0.80)    | 1.0
"governing_law is typically Delaware"              | field      | info     | categorical(0.60) | 1.0
"consultant_name is typically 'Consultant'"        | field      | info     | categorical(0.86) | 1.0
"payment_deadline between 10 and 15 days"          | field      | info     | distribution      | 1.0
"Confidentiality similarity to benchmark"          | cluster    | info     | legacy_benchmark  | 1.0
```

**Source provenance flavors:**

- `{"miner": "coverage", "ratio": 0.80, "n": 4}` — "X clusters present in Y% of corpus"
- `{"miner": "distribution", "median": 15, "lo": 10, "hi": 30, "n": 4}`
- `{"miner": "categorical", "mode": "Delaware", "frequency": 0.60, "n": 5}`
- `{"miner": "outlier", "outlier_pct": 0.15, "n": 42}` — "15% of this cluster's clauses deviate from the reference"
- `{"miner": "contrastive", "cluster_lift": 2.8, "global_lift": 1.2, "lift_ratio": 2.3}`
- `{"migrated_from": "playbook_benchmarks"}` — legacy similarity rule

**Insights this view must surface:**

- A ranked feed of drafts with lift × stability scoring (which to review first).
- Visual distinction between mined / migrated / derived rules.
- Source-provenance "trust" indicators (n, ratio, frequency).
- Inline editing of severity, escalation owner, preferred language.
- Predicate JSON in a structured form + raw JSON toggle for advanced users.
- Side-by-side example matches: 3 clauses that pass, 3 that fail.
- Accept / Retire / Edit-then-Accept actions.

**Visualization options:**

- **Stacked card feed** — one card per rule, signal indicators (lift, support,
  confidence) as small chips, examples in expandable section.
- **Inbox/triage queue** — like email triage, with severity color-bands and
  bulk actions.
- **Kanban** — columns: Draft / In Review / Approved / Retired.
- **Tree view** — playbook → applies_to → bindings → predicate sub-tree.

---

### 4. The four pages to design

Combine the visualization components above into these four pages:

#### Page 1: Corpus Overview (P1 + P3)

_"What is in my corpus?"_

- Treemap of domain → clusters (legal constructs, §3.1)
- Sidebar: schema completeness summary (§3.2 — small bento)
- Top fields by extraction count
- "Mine a fresh playbook" call-to-action

#### Page 2: Field Explorer (P1)

_"What values does this field actually take?"_

- Field picker (cluster → field cascading dropdown)
- Distribution chart for selected field (§3.3 — histogram or bar)
- Raw extractions table with confidence + jump-to-evidence
- Existing rules that reference this field (sidebar)

#### Page 3: Playbook Studio (P1)

_"Review and approve draft rules."_

- Triage queue of drafts (§3.4 — card feed or kanban)
- Filter by severity / applies_to / source kind / confidence
- Detail panel with predicate builder + example matches + edit fields
- Bulk Accept / Retire actions
- "Migrate legacy benchmarks" expander

#### Page 4: Contract Alignment (P2 + P3)

_"How does this contract compare to the playbook?"_

- Pick a contract → run alignment
- Severity-grouped finding list (blocker > approval_required > warn > info > n/a)
- For each finding: rule title, evidence snippet, suggested redline language
- Side-by-side compare: client's clause vs. preferred language
- Escalation badge per finding (who must approve?)
- Export findings (CSV, PDF)
- Portfolio view: same playbook × N contracts → heatmap of compliance

---

### 5. Design system constraints

- **Framework:** React + Tailwind CSS (preferred) or Streamlit + Plotly
  (current). State which you're choosing.
- **Components:** Reuse shadcn/ui or Radix if React; reuse st.columns / st.cards
  if Streamlit.
- **Charts:** Plotly (interactive) for distributions, Recharts (React) or
  Plotly (Streamlit) for everything else. **No D3 unless required.**
- **Color palette:** Severity colors must be unambiguous and accessible —
  recommend a specific palette with WCAG AA contrast against your background.
  Suggested mapping:
  - blocker → red 600
  - approval_required → amber 500
  - warn → yellow 400
  - info → blue 500
  - n/a → gray 400
  - pass → green 500
- **Typography:** Inter or system-ui. Legal text should be slightly larger than
  metadata.
- **Density:** Dense by default (this is a power-user tool), with whitespace
  available on detail panels.
- **Empty states:** Critical. Every view should have a real empty state because
  early-stage accounts genuinely have no fields/extractions/rules yet.
- **Loading states:** Required for alignment runs (can take 10-60s).
- **Mobile:** Out of scope. Desktop-first.
- **Dark mode:** Bonus, not required.

---

### 6. Hard rules

1. **Show provenance, always.** Every mined rule must visibly show what corpus
   signal produced it. Lawyers must be able to validate the math before
   trusting the rule. No "trust the model" black boxes.
2. **Show examples, always.** Every field stat and every rule must show at
   least 3 concrete value examples. Aggregates without examples lose trust.
3. **Make severity legible at a glance** — color + icon + label, never just
   color or just icon.
4. **Editing is paramount.** Every mined rule must be editable in-place —
   severity, threshold, escalation owner, language tiers. A lawyer must be able
   to convert "info" → "approval_required" with two clicks.
5. **Evidence-driven** — every finding (and every example value) should have a
   clickable evidence span that opens the source PDF at the right page.
6. **N=5 problem** — design for the case where the user's corpus has only 5
   contracts. The miner produces sparse, low-confidence drafts. The UI must not
   collapse, must show clear "we need more data" affordances, and must not
   intimidate the user with a sea of low-confidence rules.

---

### 7. Anti-patterns to avoid

- Tooltips as the primary information channel. Lawyers read in slow scans, not
  hovers.
- Unbounded modal dialogs. Use side panels or inline expansion instead.
- "Loading..." spinners without indication of progress for alignment runs.
- Hiding the predicate JSON from users who want it. Show it as a toggle.
- Conflating "mined draft" with "active rule" — visual distinction must be
  immediate.
- Treating clusters with `chunk_count < 5` as first-class. Surface them but
  visually deprioritize them.
- Tables wider than the viewport without horizontal scroll affordance.

---

### 8. Deliverables

**MODE: A — Wireframe + design rationale**

For each of the four pages, produce:

1. ASCII wireframe (mark sections, components, calls-to-action). Keep it under
   60 lines per page.
2. Component inventory list with brief descriptions.
3. Data-binding annotations: which DuckDB table/field powers which UI element.
4. Interaction notes: what happens on click, hover (sparingly), drag.
5. One paragraph of design rationale per page explaining the key decisions and
   tradeoffs vs the alternatives in §3.

After the four pages, produce a short component-library summary listing every
reusable component (Card, FieldDistributionChart, ClauseTreemap, SeverityBadge,
PredicateRenderer, etc.).

**MODE: B — Production component code**

For each of the four pages, produce:

1. A single React + TypeScript + Tailwind file per page that compiles and runs
   against the data schema in §3.
2. Use mock data inline at the top of each file matching the sample rows in §3.
3. Import any external libs (recharts, lucide-react, etc.) — no specific
   restrictions.
4. Include a minimal type definitions block at the top of each file.
5. Include accessibility (ARIA labels, semantic HTML).
6. Each component should be self-contained — ready to drop into a Next.js app.

In either mode, end with a short "open questions" list — things you'd want to
clarify with the product team before final implementation.

---

### 9. Inspiration to draw from (not copy from)

- **Ironclad** for the rule-editor + escalation-routing pattern.
- **Spellbook** for the inline-redline-suggestion UX.
- **DocuSign Insight** for the field-distribution explorer.
- **Linear** for the triage-queue card-feed pattern (P3 Playbook Studio).
- **Mixpanel / Amplitude** for the cohort/segmentation card design.
- **shadcn/ui dashboard examples** for the Bento layout.

---

### 10. What you are NOT designing

- Authentication / org-switching / role management — out of scope.
- The PDF parser or clustering pipeline — they're upstream.
- The Streamlit page registration system — out of scope, that's plumbing.
- Anything mobile.

---

## PROMPT ENDS

---

## Appendix A — Real DuckDB query snippets to paste alongside the prompt

If you want the LLM to have _actual_ data to bind to, run these against
`data/demo.duckdb` and paste the results as additional context:

```bash
# Cluster taxonomy (Page 1 — Corpus Overview)
.venv_playbooks/bin/python -c "
import duckdb, json
c = duckdb.connect('data/demo.duckdb', read_only=True)
rows = c.execute('''
  SELECT d.label AS domain, c.label AS cluster, c.chunk_count, c.quality_score
  FROM clusters c JOIN domains d ON c.domain_id = d.domain_id
  ORDER BY c.chunk_count DESC LIMIT 12
''').fetchdf()
print(rows.to_string(index=False))
"

# Field schema (Page 2 — Field Explorer)
.venv_playbooks/bin/python -c "
import duckdb
c = duckdb.connect('data/demo.duckdb', read_only=True)
rows = c.execute('''
  SELECT cl.label AS cluster, f.name AS field, f.field_type,
         COUNT(e.extraction_id) AS n
  FROM field_definitions f
  LEFT JOIN clusters cl ON f.cluster_id = cl.cluster_id
  LEFT JOIN extractions e ON e.field_id = f.field_id
  GROUP BY cluster, field, field_type
  ORDER BY n DESC LIMIT 20
''').fetchdf()
print(rows.to_string(index=False))
"

# Mined playbook rules (Page 3 — Playbook Studio)
.venv_playbooks/bin/python scripts/export_playbook.py data/demo.duckdb \
    --playbook-id <pid> --compact | jq '.playbooks[].rules[:5]'

# Sample alignment finding (Page 4 — Contract Alignment)
.venv_playbooks/bin/python -c "
from core.store import ClusteringStore
from core.playbooks.store import PlaybookStore
from core.playbooks.integration import agreement_ctx
from core.playbooks.aligner import align
import json
cs = ClusteringStore('data/demo.duckdb')
pb = PlaybookStore('data/demo.duckdb')
# Promote drafts to active first
for r in pb.list_rules(pb.list_playbooks()[1]['playbook_id']):
    pb.update_rule_status(r['rule_id'], 'active')
ctx = agreement_ctx(cs, 'agr_996f4094')
findings = align(pb, pb.list_playbooks()[1]['playbook_id'], ctx)
print(json.dumps(findings[:5], indent=2, default=str))
"
```

---

## Appendix B — Iteration prompts

Once Claude returns the first design pass, drive iteration with focused
follow-ups:

- _"Page 3 looks too dense for a lawyer doing 60-90 min/week. Redesign with
  fewer rules per fold and clearer call-to-action."_
- _"The treemap on Page 1 doesn't communicate quality_score. Show me three
  alternatives that encode size AND quality."_
- _"Add a 'Why is this rule here?' explanation popover that shows the corpus
  signal (n, ratio, mode, lift_ratio) in one paragraph."_
- _"How should we handle the case where a contract has 47 'info'-level findings
  but 0 blockers? Today they're noisy and demoralizing."_
- _"What does the empty state look like for a brand-new account that's
  uploaded only 3 PDFs and has no fields yet?"_
- _"Redesign Page 4's portfolio-view heatmap. Today it's contracts (rows) ×
  rules (columns). Some teams want to invert."_

---

## Appendix C — File:line bindings (for engineers implementing)

When you (or Claude in MODE B) starts producing component code, point at these
files for the data shapes:

- DuckDB schema: `core/store.py` + `core/playbooks/store.py`
- Predicate operators: `core/playbooks/predicates/` (`__init__.py` lists them all)
- Aligner output: `core/playbooks/aligner.py::align()` returns the finding shape
- Existing Streamlit pages: `ui/playbook_authoring_page.py`,
  `ui/playbook_mining_page.py`, `ui/playbook_findings_page.py`
- Export utility: `scripts/export_playbook.py` produces JSON that any UI can
  consume directly
- Spec: `docs/superpowers/specs/2026-05-09-playbook-rules-design.md`
