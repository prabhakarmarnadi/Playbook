# Clustering V2 — Long-Term Roadmap
## From Contract Processing Engine to Agreement Intelligence Platform

---

## Vision

Build the **legal knowledge infrastructure layer** — not another document processing tool, but the system that understands what contracts mean, how they relate, how they change, and what risks they carry. Every contract processed makes the system smarter. Every organization's patterns become queryable institutional knowledge.

The moat is compounding data intelligence: after processing 10M agreements across hundreds of customers, no competitor can replicate the corpus understanding, field distributions, clause co-occurrence models, and negotiation pattern libraries that this platform accumulates.

---

## Current State (Delivered)

| Capability | Status |
|---|---|
| Two-level hierarchical clustering (macro domains → micro clause types) | ✅ Shipped |
| Semantic chunking with legal structural awareness | ✅ Shipped |
| Field-aware document summaries (legacy fields_processor parity) | ✅ Shipped |
| GPU-accelerated pipeline (Nomic 768-dim, cuML UMAP/HDBSCAN, Ray optimization) | ✅ Shipped |
| Cluster-conditioned extraction (40% token reduction, 0 hallucinated fields) | ✅ Shipped |
| Automated ontology construction (domains → clause types → fields → composites) | ✅ Shipped |
| Hierarchical topic merging / FIELD-310 | ✅ Shipped |
| Incremental cluster assignment / FIELD-801 (<500ms) | ✅ Shipped |
| Clause-level semantic embeddings / FIELD-110 | ✅ Shipped |
| Hybrid retrieval (BM25 + Dense + CrossEncoder + RRF) | ✅ Shipped |
| Composite/derived field engine with cascade propagation | ✅ Shipped |
| Example Set Review → Scale UX | ✅ Shipped |
| DuckDB (12 tables) + LanceDB (persistent ANN) + Azure Blob snapshots | ✅ Shipped |
| 10-page Streamlit UI | ✅ Shipped |
| 60 static field definitions (ported from legacy 222 predefined configs) | ✅ Shipped |

**Release gates met:** 8/8 infrastructure tests, 7/7 GPU tests passing. Smoke tested on synthetic documents.

---

## Phase 1: Foundation & Production Readiness (0–3 months)

**Goal:** Get V2 running on real customer data at scale. Prove clustering quality meets release gates (purity ≥0.75, label precision ≥0.85, extraction F1 ≥0.80).

### 1.1 Evaluation Framework
- Gold set construction: 500 agreements across 10 accounts, lawyer-annotated clause types + field values
- Automated accuracy pipeline: clustering purity, ARI stability, extraction F1, latency P95
- Regression detection: run on every commit, block merge if metrics degrade
- A/B comparison harness: V1 flat clustering vs V2 hierarchical on same corpus

### 1.2 Data Contract Migration (from eng doc §5)
- Phase 1: Field-summary embeddings preserved (AIDB compatibility)
- Phase 2: Dual-write (field-summary + clause-level embeddings)
- Phase 3: Cutover to clause-level only
- Migration cost estimate: ~$500–$1,000 for 10M agreements, +80GB storage, <1hr re-clustering

### 1.3 Multi-Tenant Isolation
- Per-customer DuckDB + LanceDB instances (not row-level security — full isolation)
- Tenant-scoped Blob snapshot backup/restore
- Customer onboarding automation: provision storage → ingest → cluster → review → publish

### 1.4 Streaming Ingestion
- Document arrival → chunk → embed → incremental assignment (FIELD-801) for known clusters
- Batch re-clustering triggered on threshold (e.g., 500 new docs or weekly)
- Webhook/event integration for CLM platforms (upload triggers processing)

### 1.5 Scale Validation
- Load test: 10K documents, measure end-to-end latency and memory
- Optimize chunker + embedder for streaming (avoid loading full corpus into memory)
- LanceDB index tuning: IVF_PQ parameters for 1M+ vectors

**Release gate:** Purity ≥0.75, label precision ≥0.85, extraction F1 ≥0.80 on gold set.

---

## Phase 2: Corpus Analytics & Feedback Loop (3–6 months)

**Goal:** The system understands its own data. Every processed agreement enriches a statistical model of the portfolio. Users get quantitative answers, not just extracted values.

### 2.1 Field Distribution Engine
- Per field per domain: value histograms, median, mode, percentiles, coverage %
- Automatic type detection refinement (field labeled "string" but 95% of values are dates)
- Queryable: "What's the distribution of payment_days in SaaS agreements?"
- Surfaces in UI as sparklines per field in the Cluster Explorer

### 2.2 Extraction Reliability Rollup
- Aggregate confidence scores per field per domain across the corpus
- Flag low-reliability fields: "payment_days extracts at 94% confidence, price_adjustment at 58%"
- Auto-prioritize gold example collection for worst-performing fields
- Track reliability over time (weekly trend)

### 2.3 Clause Co-occurrence Matrix
- Conditional probabilities: P(auto_renewal | payment_terms), P(liability_cap | indemnification)
- Identify structural patterns: which clauses always appear together, which are mutually exclusive
- Anomaly detection: "This SaaS agreement has payment_terms but no auto_renewal — unusual (only 3% of corpus)"

### 2.4 Per-Document Anomaly Flagging
- Expected field coverage per domain (95% of SaaS agreements have X, this one doesn't)
- Centroid distance outlier detection (doc is in SaaS domain but far from centroid — likely misclassified)
- Missing clause alerts surfaced in Dashboard and per-document view

### 2.5 Feedback & Learning Loop
- Track correction rates per field per domain over time
- Auto-trigger re-extraction when gold example count crosses threshold for a field
- Detect extraction regression: confidence drops after prompt/model changes
- Inter-annotator agreement tracking (when multiple reviewers correct same field)
- Weekly "system health" digest: what improved, what regressed, where to focus

**Release gate:** Field distribution available for 100% of discovered fields. Anomaly detection FPR ≤15%.

---

## Phase 3: Compliance & Policy Engine (6–9 months)

**Goal:** Move from "what's in this contract" to "what's wrong with this contract." This is the highest-revenue capability — it drives action, not just awareness.

### 3.1 Policy Definition Framework
- Admin UI: define rules per domain per field
- Rule types: range (liability_cap must be ≤ 2x ACV), presence (NDA must have return_of_materials), pattern (governing_law must be "Delaware" or "New York")
- Rules stored as structured data (not LLM prompts) — deterministic, auditable, fast

### 3.2 Violation Detection
- Evaluate all extracted field values against applicable policies
- Severity levels: critical (blocks signature), warning (requires review), info (logged)
- Batch mode: scan full portfolio on policy change
- Real-time mode: evaluate on new document ingestion

### 3.3 Risk Dashboard
- Per-agreement risk score (0–10 scale, from product doc §7.5)
- Portfolio-level risk heatmap: which domains, which fields, which vendors carry most risk
- Drill-down: risk score → contributing violations → specific clause → document

### 3.4 Governance & Audit Trail (from eng doc §11)
- All policy changes versioned with author + timestamp
- Violation resolutions tracked: accepted risk, renegotiated, escalated
- 3-year retention for compliance reporting
- Rollback: any policy version revertible within 5 minutes

**Release gate:** FPR ≤15%, FNR ≤10% on 200-agreement audit sample. Legal team sign-off.

---

## Phase 4: Smart Recommendations & Explainability (6–12 months, parallel with Phase 3)

**Goal:** The system suggests actions, not just reports facts. Every suggestion includes structured evidence so lawyers trust it.

### 4.1 Clause Recommendations
- Given a document's domain + existing clauses, suggest missing clauses from corpus patterns
- "92% of SaaS agreements in your portfolio include a data_protection clause — this one doesn't"
- Grounded in co-occurrence matrix (Phase 2.3) + domain norms

### 4.2 Field Value Suggestions
- For each field, suggest values based on domain distribution + org preferences
- "Your typical payment_days for SaaS is 30 — this agreement says 60"
- Confidence thresholds configurable per org

### 4.3 Structured Reasoning Traces (from product doc §4.5)
- Every recommendation includes: source evidence, cluster frequency, org historical pattern, policy match
- Never LLM-generated explanations — always from stored data
- "Why?" link on every suggestion opens evidence panel

### 4.4 Org-Level Preference Learning
- Capture: clause accept/reject/edit patterns per org
- Build org preference profile: which clauses they always negotiate, which they accept as-is
- Use preferences to rank recommendations (suggest what this org typically cares about)

**Release gate:** Recommendation accept rate ≥2x random baseline in A/B test. Reasoning traces for 100% of suggestions.

---

## Phase 5: Contract Graph & Temporal Intelligence (9–15 months)

**Goal:** Model contracts as a connected graph, not independent documents. Enable temporal analysis that reveals how deals evolve.

### 5.1 Agreement Relationship Graph
- Link amendments to parent agreements (detect via "Amendment to Agreement dated...")
- MSA → SOW hierarchy (detect via cross-references)
- NDA → LOI → Definitive agreement chains
- Renewal chains: same parties + same subject matter + sequential dates
- Store as graph edges in DuckDB (agreement_id, related_agreement_id, relationship_type)

### 5.2 Amendment-Aware Extraction
- When processing an amendment, inherit base agreement's field values
- Only extract modified fields (diff-based extraction — massive token savings)
- Show field version history: "Liability cap changed from $1M (2022) → $2M (2023) → $5M (2025)"

### 5.3 Temporal Trend Analysis
- Time-series on field values: "Average liability cap in SaaS increased 40% YoY"
- Clause adoption/deprecation tracking: "AI governance clauses appear in 42% of 2025 contracts vs 3% in 2023"
- Per-vendor trajectory: "This vendor's payment terms worsened each renewal: 30 → 45 → 60 days"

### 5.4 Portfolio Benchmarking (Internal)
- Domain norms computed from org's own portfolio: "Your median SaaS liability cap is $2.3M"
- Per-document comparison: "This agreement's cap ($50K) is in the 5th percentile for your SaaS portfolio"
- Cross-vendor comparison: "Vendor A's terms are 15% more favorable than Vendor B on average"

### 5.5 Cluster Stability Tracking
- Track cluster centroids across re-clustering runs
- Detect drift: centroid shift >0.1 between runs → alert
- Detect emerging clusters: new clause patterns that didn't exist 6 months ago

**Release gate:** Graph relationships detected for ≥80% of amendment/renewal pairs. Temporal trends validated by legal team on 3 customer portfolios.

---

## Phase 6: Platform Intelligence & Market Moat (12–24 months)

**Goal:** Cross-customer intelligence that no single-tenant tool can replicate. This is the defensible moat.

### 6.1 Cross-Customer Benchmarking (from product doc §7)
- Anonymized, aggregated field distributions across customers (same industry, same domain)
- "Your SaaS liability cap is in the 30th percentile compared to similar companies"
- Strict privacy: minimum 20 orgs per cohort, differential privacy (ε=1.0), opt-in only, signed data-sharing agreement

### 6.2 Legal-Domain Embedding Fine-Tuning
- Contrastive learning on clause pairs from same vs different clusters (labeled by production clustering)
- Domain adapter on top of Nomic/MiniLM base
- Expected: 10–15% improvement in clustering purity and field scoring accuracy
- Re-train quarterly as corpus grows

### 6.3 Intent Modeling (from product doc §4.7)
- Move from "what the clause says" to "why it exists"
- Intent taxonomy: risk allocation, cash flow optimization, IP protection, regulatory compliance
- "Payment Terms (45 days) → moderate cash flow optimization + low vendor friction"
- Enables intent-based search: "Find all clauses optimizing for cash flow"

### 6.4 Negotiation Intelligence
- Track clause evolution across contract versions with same counterparty
- Identify negotiation leverage: "You conceded on liability cap in 3 of 4 deals with this vendor"
- Suggest negotiation strategy based on historical outcomes

### 6.5 Predictive Analytics
- Predict renewal probability based on clause composition + vendor history
- Estimate renegotiation effort based on deviation from org norms
- Risk forecasting: identify agreements likely to trigger violations before they do

### 6.6 Agent-Based Evaluation (from product doc §7)
- Self-play agent evaluation for extraction quality
- Critic agent validates outputs before surfacing to users
- Continuous automated quality monitoring without human annotation

**Release gate:** A/B validation showing measurable uplift vs Phase 4 baseline. Benchmarking available for ≥3 industry verticals.

---

## Engineering Infrastructure (Cross-Cutting, Continuous)

| Investment | When | Why |
|---|---|---|
| **Evaluation framework + gold sets** | Phase 1, then continuous | Can't improve what you can't measure. Blocks every release. |
| **Streaming pipeline** | Phase 1 | Batch-only doesn't work for CLM integration. |
| **Multi-tenant isolation** | Phase 1 | Prerequisite for any production deployment. |
| **Distributed extraction (Ray parallel LLM)** | Phase 2 | 500+ doc batches need parallel LLM calls with rate limiting. |
| **Embedding fine-tuning pipeline** | Phase 6, prep in Phase 4 | Legal-specific embeddings improve everything downstream. |
| **Feature store** | Phase 2 | Pre-computed field distributions, cluster stats, org preferences. |
| **Schema versioning** | Phase 2 | Fields evolve; need rollback + lineage tracking. |
| **Incremental UMAP** | Phase 3 | Project new points without re-training the full model. |
| **GPU memory optimization** | Phase 3 | Streaming ANN search for 10M+ vectors, not full memory load. |
| **Graph database (or DuckDB graph extension)** | Phase 5 | Agreement relationship queries at scale. |

---

## Success Metrics

### Phase 1 (Foundation)
- Clustering purity ≥0.75 on 500-sample gold set
- Label precision ≥0.85 on 200 cluster labels
- Extraction F1 ≥0.80 on 100-doc gold set
- Pipeline handles 10K docs in <30 min (GPU)

### Phase 2 (Analytics)
- Field distributions available for 100% of discovered fields
- Anomaly detection FPR ≤15%
- Extraction reliability tracked and trended weekly
- ≥1 auto-triggered re-extraction based on gold example threshold

### Phase 3 (Compliance)
- Violation FPR ≤15%, FNR ≤10%
- ≥80% of violations confirmed actionable by legal review
- Policy definition turnaround <1 hour per domain

### Phase 4 (Recommendations)
- Recommendation accept rate ≥2x random baseline
- Reasoning traces on 100% of suggestions
- >40% MAU engaging with recommendations

### Phase 5 (Graph + Temporal)
- ≥80% amendment/renewal pairs linked correctly
- Temporal trends validated on 3 customer portfolios
- Centroid drift detection with <1 day latency

### Phase 6 (Market Moat)
- Cross-customer benchmarking for ≥3 industry verticals
- Embedding fine-tuning shows ≥10% purity improvement
- Negotiation intelligence validated on ≥5 multi-contract vendor relationships

---

## Cost Model (at scale — 10M agreements)

| Component | Cost | Notes |
|---|---|---|
| Embedding compute (one-time) | $500–$1,000 | Nomic GPU, ~768-dim |
| Storage delta | +80 GB | DuckDB + LanceDB |
| Cluster re-training | <1 hour | cuML GPU |
| Ontology re-labeling | ~$50 | 5–10 LLM calls per domain |
| Extraction (per agreement) | ≤$0.02 | Cluster-conditioned, 40% fewer tokens |
| Compliance check (per agreement) | ≤$0.005 | Deterministic rules, no LLM |
| Recommendation retrieval (per query) | ≤$0.01 | Hybrid search + rerank |

---

## What Makes This "The Next Big AI Thing"

1. **Compounding data moat:** Every agreement processed enriches field distributions, co-occurrence models, domain norms, and negotiation patterns. After 10M agreements, the corpus intelligence is unreplicable.

2. **Knowledge infrastructure, not a feature:** Other tools extract fields or summarize documents. This builds a queryable legal knowledge graph with clause-level semantics, temporal evolution, and cross-portfolio benchmarking.

3. **Trust through explainability:** Every output has a structured reasoning trace from stored data — never LLM-generated explanations. Lawyers trust data provenance, not model confidence scores.

4. **Clause DNA:** Each organization develops a unique legal fingerprint — preferred clauses, negotiation patterns, risk tolerances. This becomes the personalization layer that makes switching costs prohibitive.

5. **Network effects via benchmarking:** Cross-customer anonymized benchmarking creates value that increases with every new customer. A single-tenant tool can never replicate "your liability cap is in the 30th percentile for SaaS companies your size."
