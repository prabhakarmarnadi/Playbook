# Clustering V2: Agreement Intelligence Platform — Product Document

---

## Glossary of Key Terms

| Term | Plain-Language Definition |
|------|---------------------------|
| Ontology | A structured catalog of all the clause types the system knows about — what they contain, how they relate, and what fields can be extracted from each. Think of it as the system's "vocabulary" for understanding contracts. Built automatically from clustering, not manually. |
| Cluster | A group of agreement sections that the system identified as semantically similar (e.g., all "Payment Terms" paragraphs across contracts). |
| Clause Type | A named category derived from a cluster (e.g., "Payment Terms", "Liability Limitation"). The building block of the ontology. |
| Field | A specific data point extracted from a clause type (e.g., `payment_days = 45`, `liability_cap = 2x fees`). |
| Derived Field | A computed field defined from other fields (e.g., `last_opt_out_date = expiration_date - notice_period`). |
| Domain | A high-level agreement family (e.g., SaaS, NDA, Employment). Agreements are classified into domains before clause-level analysis. |
| Concept Profile | The set of clause types, fields, and norms associated with a domain — an agreement's "DNA." |
| Example Set | A representative sample of documents from a cluster, shown to users for review before extraction runs on the full corpus. |
| Confidence Score | A numeric measure (0–1) of how certain the system is about a classification, extraction, or finding. Low-confidence results are flagged or suppressed. |
| Reasoning Trace | Structured evidence explaining *why* a recommendation was made — always based on data, never invented by the AI. |
| Intent | An abstraction of *why* a clause exists and what business tradeoff it encodes (e.g., cash flow optimization vs vendor relationship). |
| Cold Start | The challenge of making useful suggestions for a new customer with no prior interaction history. |

---

## 1. Executive Summary

We propose evolving the current agreement clustering platform from a **static topic discovery tool** into a full **Agreement Intelligence Platform** — a system that automatically discovers clause types, generates schemas, recommends clauses, enforces compliance, and continuously learns from user behavior.

The current system (V1) uses BERTopic with MiniLM-384 embeddings to cluster agreement field summaries. V2 extends this foundation with:

- **Ontology construction** from clusters (clause types + fields + relationships)
- **Intent detection** (understanding *why* a clause exists, not just *what* it says)
- **Policy-aware compliance engine** (real-time violation detection)
- **Personalization** (organization + user preference learning)
- **Explainable reasoning traces** (structured evidence for every recommendation)
- **Hybrid retrieval** (dense recall + multi-vector precision)
- **Continuous learning loops** (system improves with every interaction)

### Why This Matters

Most legal AI competitors build: `LLM + RAG + vector DB`

This system builds: `Structure + Memory + Learning + Intent`

That is the difference between a **temporary feature** and an **enduring platform**.

### Why Now / Why Us

This is not speculative greenfield work. We already have:

- **BERTopic clustering pipeline** — production-grade, optimized with Bayesian search (30 Optuna trials), quality-gated
- **MiniLM-384 embeddings** — pre-computed and served via AIDB at scale
- **SchematizeHandler** — LLM-based field extraction with HyDE retrieval and Cochran sampling
- **Service Bus orchestration** — async job management, status tracking, lock renewal
- **AIDB gRPC infrastructure** — embeddings, snippets, and write APIs already operational

The V2 proposal layers new capabilities on top of proven infrastructure. Every Phase 1–2 deliverable extends existing services rather than replacing them.

### First Wedge

**Policy-aware clause intelligence for enterprise agreement review.**

One sentence: "Auto-detect clause types in uploaded agreements, extract key fields, and flag the top 3 policy violations — with structured evidence for every finding."

### Target ROI (Phase 1–2)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Agreement review time reduction | 30% | A/B comparison: assisted vs unassisted review |
| Policy violations auto-detected | Top 3 violation types surfaced with ≥ 80% precision | Manual audit of flagged vs missed violations |
| Clause-level outlier detection | Surface non-standard clauses deviating > 2σ from org norm | Deviation from org field-value distributions |
| Manual schema creation effort | 50% reduction | Compare time-to-first-field: manual vs auto-suggested |

---

## 2. Current State (V1)

### What Exists Today

```
DocuSign Agreements (via AIDB)
        ↓
Field Summary Embeddings (MiniLM-v3-384)
        ↓
BERTopic Clustering (UMAP → HDBSCAN → c-TF-IDF)
        ↓
Topic Clusters with Keywords
        ↓
Field Schema Generation (LLM-based extraction)
        ↓
Auto-Suggested Fields for Users
```

**Current capabilities:**
- Bayesian / grid search hyperparameter optimization for UMAP + HDBSCAN
- Outlier reduction via c-TF-IDF cosine matching
- BM25-weighted representation for domain-specific term amplification
- HyDE-based snippet retrieval for field extraction
- Cochran statistical sampling for document subset selection
- Optional GPT-based topic naming

**Current limitations:**
- Static clustering — doesn't learn or evolve after initial run
- Flat topic structure — no hierarchy, no clause relationships
- No compliance enforcement — insights without action
- No personalization — same output regardless of user/org context
- No reasoning traces — users don't know *why* suggestions are made
- No intent modeling — structure without purpose

---

## 3. Product Vision

### From → To

| Dimension | V1 (Current) | V2 (Target) |
|-----------|-------------|-------------|
| Intelligence | Topic keywords | Clause ontology + intent |
| Output | Suggested fields | Recommendations + compliance alerts |
| Learning | Static (train once) | Continuous (improves with usage) |
| Personalization | None | Org + user + industry preferences |
| Explainability | None | Structured reasoning traces |
| Scope | Discovery only | Discovery + compliance + recommendations |

### Positioning

DocuSign evolves from:

> **Agreement execution platform** → **Agreement intelligence + compliance + recommendation engine**

---

## 4. Core Product Capabilities

### 4.1 Domain-Aware Clustering

Automatically organize agreements into meaningful domains before extraction. Instead of treating all contracts identically, the system clusters agreements into domains using **embedding-based summary clustering** (embed title + opening text → UMAP + HDBSCAN → coarse clusters → LLM labels only the resulting clusters), then discovers clause patterns within each domain via micro-clustering.

**Why embedding-based clustering, not per-document LLM classification:** Per-document LLM classification (calling GPT-4o-mini once per uploaded document) is cost-prohibitive at scale. With 500 documents, that's 500 LLM calls just for domain assignment. Embedding-based clustering achieves the same result with only 5–10 LLM calls (one per discovered cluster) — a 50–100x cost reduction. It's also faster (runs locally in seconds), more consistent (deterministic cluster boundaries), and can discover unexpected domains that weren't in a predefined taxonomy.

**Processing timeline (what users see):**

| Time after upload | 100 docs | 500 docs | What appears |
|-------------------|----------|----------|-------------|
| ~30s | ✅ | ✅ | Domain distribution chart with cluster labels |
| ~90s | ✅ | ✅ | Micro clusters within first domain |
| ~3 min | ✅ | — | Full ontology: all domains, clusters, fields |
| ~5–8 min | — | ✅ | Full ontology for 500 docs |

**User experience:**
```
Your agreements organized by domain:
  ┌─ SaaS Agreements (423 docs)
  │   ├─ Payment Terms (cluster)
  │   ├─ Liability Limitation (cluster)
  │   ├─ Auto-Renewal (cluster)
  │   └─ Data Processing (cluster)
  ├─ NDAs (187 docs)
  │   ├─ Confidentiality Scope (cluster)
  │   └─ Non-Solicitation (cluster)
  └─ Vendor Agreements (312 docs)
      ├─ Payment Terms (cluster)
      └─ Indemnification (cluster)
```

Note: "Payment Terms" in SaaS vs Vendor agreements may have different norms, fields, and policy implications. Domain-aware clustering keeps them separate.

**Day-1 personalization:** New customers or agreement types immediately inherit domain-specific intelligence from existing clusters. No bespoke rule-building required.

**Per-document metadata enrichment (beyond V1):**

Each document gets a rich profile that V1 never produced:

| Metadata | Example |
|----------|---------|
| Macro domain + confidence | `SaaS, 0.92` |
| Clause type assignments | `[Payment Terms, Liability, Auto-Renewal]` |
| Per-clause extracted fields | `Payment: {payment_days: 30, method: "wire"}` |
| Universal fields | `{parties: ["Acme", "Widget"], governing_law: "Delaware"}` |
| Composite values | `{TCV: $4,500,000}` |
| Cluster typicality score | `0.87` — how representative of its cluster |
| Coverage % | `85%` — fields found vs expected |

**Data format — hierarchical tree:**

The output is a tree structure (Account → Domains → Clause Types → Fields → Composites), not flat topics. See the engineering doc §4.2 for the full data model.

### 4.2 Clause Intelligence

Automatically detect clause types and extract structured fields from agreements.

**User experience:**
```
Detected clauses:
  - Payment Terms
  - Termination
  - Liability

Fields extracted:
  - payment_days = 30
  - liability_cap = unlimited
  - termination_notice = 90 days
```

Clickable clauses highlight relevant sections in the document.

### 4.3 Compliance Engine

Enforce organizational policies against detected clauses in real-time.

**User experience:**
```
⚠️ Violation detected:
  Liability cap = unlimited
  Policy = max 2x contract value

Suggested alternative (requires human review):
  Cap liability at 2x fees
  Confidence: 0.87 — based on 2,300 similar clauses
```

**Product guardrail:** All compliance findings and suggested fixes are **assistive, not autonomous**. The system surfaces findings with confidence scores and evidence; a human reviewer must approve any action. No clause is modified, accepted, or rejected without explicit user confirmation. This boundary is enforced at the API level — there is no "auto-fix" mode.

**Dashboard view (for legal ops):**
```
Contracts with policy violations:
  - 23 with unlimited liability
  - 17 with payment > 60 days
  - 8 missing confidentiality clause
```

### 4.4 Smart Recommendations

Suggest clauses and field values during drafting, grounded in data.

**User experience:**
```
Suggested Payment Terms: 45 days

Because:
  • Most common in your contracts (45%)
  • Within policy limits
  • Industry median for SaaS = 30 days
```

**Product guardrails:**
- Recommendations are **suggestions, not legal advice**. Provenance (cluster evidence, org history, policy match) is always displayed.
- Confidence thresholds are configurable per organization. Below threshold, recommendations are suppressed rather than shown with low confidence.
- All recommendations include a "Why this suggestion?" link that expands to the full reasoning trace.
- Organizations can configure mandatory human approval for specific clause types (e.g., indemnification, liability).

### 4.5 Reasoning & Explainability

Every recommendation includes structured evidence — never black-box.

**User experience:**
```
Recommended: Liability Cap at 2x fees

Reasoning:
  • Appears in 72% of SaaS contracts
  • Used in 80% of your organization's contracts
  • Similar contracts in your repository use this clause
  • Cluster similarity score = 0.87
```

### 4.6 Personalization & Field Lifecycle Management

Learn preferences at user, organization, and industry levels. Give users full control over auto-discovered field definitions.

**Signals captured:**
- Clause accepted / rejected / modified
- Field values changed (e.g., `payment_days: 30 → 45`)
- Clause types consistently added or removed
- **Field definitions refined** (renamed, retyped, redefined with examples)

**Field lifecycle management:**

Auto-discovered fields are starting points, not final answers. Users can:

| Action | Example |
|--------|----------|
| Rename a field | `payment_days` → `Net Payment Terms` |
| Change return type | `string` → `integer` |
| Update description + examples | "Number of days until payment is due from invoice date" + gold examples |
| Create derived/composite field | `last_opt_out_date = expiration_date - notice_period` |
| Scope to specific clusters | `auto_renew` applies to Renewal cluster only; `governing_law` applies globally |
| Delete/hide a field | Remove irrelevant auto-suggested field from cluster schema |

**Derived fields** unlock clause-level intelligence:
```
Clause: Auto-Renewal
  Detected fields: renewal_term, notice_period, auto_renew
  Derived: last_opt_out_date = expiration - notice_period
  Derived: renewal_type = auto | manual (from auto_renew)
```

All changes trigger re-extraction on the **sample set** (see Section 4.8) so users see immediate impact before publishing to the full corpus.

**Output:**
```
Your company profile:
  - Payment Terms: 45 days (70%)
  - Liability Cap: 2x (80%)
  - Arbitration: avoided (90%)
```

### 4.7 Intent-Aware Intelligence

Move beyond "what the clause says" to "what it's trying to achieve."

**User experience:**
```
Intent analysis:
  Payment Terms (45 days)
  → moderate cash flow optimization
  → low vendor friction
  → aligned with your organization's risk profile
```

### 4.8 Example Set Review & Graduated Rollout

Extraction doesn’t run blindly across all documents. Instead, it follows a **sample → review → publish** pattern:

**User experience:**
```
Cluster: Payment Terms (SaaS)
  423 agreements in cluster

Step 1: Sample set (30 representative docs) → auto-extracted
  │ payment_days: 30 (conf: 0.92), 45 (conf: 0.88), ...
  │ liability_cap: 2x fees (conf: 0.85), ...
  └─ Review: ✅ Accept  ❌ Reject  ✏️ Edit field definition

Step 2: User reviews, refines field definitions, re-runs on sample
  │ Feedback: rename payment_days → Net Payment Terms
  │ Feedback: add example {"net thirty" → 30}
  └─ Re-run on sample → improved extraction

Step 3: Confidence met → Publish to full cluster
  └─ Extraction runs across all 423 docs with refined definitions
```

**Why this matters:**
- Users validate before committing compute to the full corpus
- Low-confidence fields surface early for targeted improvement
- Feedback directly improves extraction quality (not just preferences)
- Reduces wasted extraction cost by ~60–80% (catch definition errors on 30 docs, not 423)

### 4.9 Learning Loop: Timing & User Effort

**How long does the learning loop take?**

| Step | User action | Time | System action |
|------|------------|------|---------------|
| **1. View example set** | Click cluster → see 20 representative docs with extracted values | Instant | Pre-selected nearest-to-centroid docs |
| **2. Quick review** | Scan values. Mark 2–3 as incorrect, correct them | ~2–3 min | Store corrections, flag unreliable fields |
| **3. Refine schema** (optional) | Rename a field, add an example, change a type | ~1 min | Rebuild extraction prompt with new examples |
| **4. Re-extract on sample** | Click "Re-run on sample" | ~30–60s | Re-extract 20 docs with improved prompt, show diff |
| **5. Confirm** | Review improvements | ~1 min | Lock in corrections + schema changes |
| **6. Scale** | Click "Scale to all 147 docs" | ~2–5 min (async) | Extract remaining docs with refined schema, show progress bar |

**Total active user time: ~5–7 minutes per cluster.** The system handles compute asynchronously.

**What compounds over time:** Gold examples from corrections become few-shot examples. Type corrections (string → int) produce cleaner values. Composites automatically benefit. No re-clustering needed — corrections improve extraction within stable clusters.

---

## 5. Product Problems Solved

### Problem 1: "We don't know what's inside our contracts"

Enterprises have millions of contracts with no structured visibility. Current tools give keyword search which fails for nuanced clauses.

**Solution:** Clause-level ontology with structured field extraction surfaces what's actually in every agreement.

### Problem 2: "We can't enforce internal policies consistently"

Companies have policies (liability cap ≤ 2x, payment ≤ 45 days, no arbitration in EU) but enforcement is manual — lawyers forget, contracts vary.

**Solution:** Policy-aware detection engine flags violations automatically and suggests compliant alternatives.

### Problem 3: "We don't know what 'good' looks like"

Legal teams lack benchmarks — what's standard, what do peers use, what is market practice?

**Solution:** Cluster statistics across agreements provide benchmarking: clause frequency, field distributions, industry norms.

### Problem 4: "Contract review is slow and inconsistent"

Manual review depends on lawyer experience and varies across team members.

**Solution:** Automated clause detection + recommendations + risk scoring make review faster and more consistent.

### Problem 5: "Our system doesn't learn"

Current tools produce the same output regardless of how much the organization uses them.

**Solution:** Continuous learning loop: user interactions → feature updates → better recommendations.

---

## 6. Competitive Landscape & Gaps

### Market Segments

| Segment | Examples | What They Build |
|---------|----------|----------------|
| Legal Research AI | Westlaw AI, Lexis+ AI | Case law search + summarization |
| Contract Lifecycle Mgmt | Ironclad, Icertis, Evisort | Workflow automation + storage |
| AI Drafting Assistants | Harvey, Legora, LegalOn | LLM-powered research + drafting |
| Litigation AI | Relativity, Everlaw | eDiscovery + document review |
| Practice Management | Clio, Filevine | Billing + client management |

### Critical Market Gaps

1. **Legal knowledge graphs** — almost no systems build ontologies automatically. Most use only vector embeddings.
2. **Clause-level intelligence** — most platforms treat contracts as documents, not structured clause collections.
3. **Personalization** — no major platform learns company policies, negotiation style, or risk tolerance.
4. **Long-term learning systems** — most legal AI tools are stateless (documents → LLM → output) with no memory.
5. **Schema discovery** — current systems rely on manually defined schemas vs automatic discovery.

### Strategic Positioning

Most competitors build: **legal productivity tools**

This architecture builds: **legal knowledge infrastructure**

That's a fundamentally different category — and the foundation for a lasting moat.

---

## 7. Differentiation Strategy

### 7.1 Clause DNA (Legal Fingerprint)

Each organization has a unique legal identity — preferred clause structures, field values, and negotiation patterns. The system computes this as a "Clause DNA" profile.

### 7.2 Market Benchmarking

Aggregate anonymized cluster statistics across customers to provide industry benchmarking:

```
Payment Terms:
  You: 60 days
  Market median: 30 days
  → More aggressive than 82% of peers
```

**Cross-Customer Data Governance:**

Benchmarking requires strict guardrails. Cross-customer data sharing is a potential legal/privacy blocker in the legal compliance domain.

| Guardrail | Requirement |
|-----------|-------------|
| Tenant isolation | Raw clause text, field values, and agreement content are NEVER shared across tenants. Only aggregate statistics (counts, distributions, percentiles) cross tenant boundaries. |
| Minimum cohort size | No benchmark statistic is computed or displayed for cohorts with fewer than **20 organizations**. This prevents re-identification. |
| Opt-in only | Cross-customer benchmarking requires explicit opt-in per organization via admin settings. Disabled by default. |
| Aggregation method | Differential privacy noise (ε=1.0) applied to all cross-tenant statistics before display. |
| Data-sharing agreement | Benchmarking participation requires a signed data-sharing addendum reviewed by legal. |
| Audit trail | All benchmark queries logged with requesting org, cohort used, and statistics returned. |
| Regulatory review | Benchmarking feature requires privacy/legal sign-off before launch in each jurisdiction. |

**Cold-start fallback constraints:** When using "similar organization profiles" for cold-start personalization, the same cohort-size and opt-in rules apply. If no qualifying cohort exists, the system falls back to global (all-customer) statistics only.

### 7.3 Negotiation Intelligence

Track clause evolution across contract versions:

```
This clause was negotiated 3 times:
  - liability cap: unlimited → 3x → 2x

Insight:
  Most vendors push for unlimited, final settlement = 2x
```

### 7.4 Clause Evolution Tracking

Detect emerging clause patterns:

```
Emerging trend:
  AI governance clauses appearing in 42% of new contracts (+300% in 6 months)
```

### 7.5 Risk Scoring

Quantified risk per clause:

```
Risk Score: 8.2/10

Reasons:
  • Unlimited liability (high risk)
  • Rare in similar contracts (top 5%)
  • Violates company policy
```

### 7.6 Counterfactual Explanations

```
If liability remains unlimited:
  • 92% of peers would reject this
  • Increases negotiation time by ~30%
```

---

## 8. Product Roadmap

### Phase 1: Cluster → Clause System (0–3 months)

**Goal:** Turn existing BERTopic clusters into usable legal primitives with domain awareness.

- **Domain clustering:** Automatically cluster agreements into macro domains (SaaS, NDA, Employment, Vendor) using **embedding-based summary clustering** (BERTopic on document-level summary embeddings) — NOT per-document LLM classification. LLM is used only to label the resulting 5–10 clusters, reducing cost by 50–100x vs per-doc classification.
- **Clause discovery:** Run BERTopic within each domain to discover clause-level micro clusters
- LLM-based cluster labeling (Topic_17 → "Payment Terms")
- Cluster → ClauseType proto-ontology mapping
- Clause view UI: detected clauses highlighted in document, organized by domain
- **Universal field extraction:** Extract baseline fields (parties, effective date, governing law) across all agreements
- **Domain-specific extraction:** Cluster-conditioned extraction for top 3–5 fields per cluster (payment_days, liability_cap, termination_notice)
- **Example Set review:** Present sample set (~30 docs) per cluster for user validation before full-corpus extraction
- **Field lifecycle management:** Users can rename, retype, redefine, add examples, create derived fields
- **Minimum viable clustering:** Target ~100 docs per domain for stable clustering; below threshold, use LLM-only field discovery on sample set
- **Prerequisite:** 500-sample gold set prepared and reviewed by legal domain expert
- **Scope:** SaaS/vendor contracts only
- **Release gate:** Cluster purity ≥ 0.75, label precision ≥ 0.85 on gold set

### Phase 2: Policy & Compliance Engine (3–6 months)

**Goal:** Turn intelligence into actionable compliance insights.

- Policy definition UI (simple rules: `liability_cap ≤ 2x contract_value`)
- Map policies to cluster/clause types
- Real-time violation detection on upload (with confidence gating)
- Risk dashboard for legal ops
- **Governance:** All violations require human review. Policy rules require admin approval.
- **Release gate:** Compliance false-positive rate ≤ 15% on audit sample. Legal team sign-off.

### Phase 3: Smart Recommendations (6–9 months)

**Goal:** Move from detection to suggestion.

- Cluster frequency modeling → suggest most common variant
- Clause alternative suggestions (with provenance and confidence)
- Similar contract retrieval (vector search within clusters)
- **Product guardrail:** All recommendations labeled as suggestions with "Why?" link to reasoning trace

### Phase 4: Explainability + Light Personalization (6–9 months)

**Goal:** Build trust and stickiness.

- Structured reasoning traces for every recommendation
- Organization-level preference tracking (most common field values)
- Feedback capture (accept/reject/edit signals)

### Phase 5: Advanced Intelligence (12–18 months)

**Goal:** Deep personalization and intent modeling.

- User + org contextual preference embeddings (by contract type, jurisdiction, counterparty)
- Intent layer — **validation required:** must show measurable uplift vs Phase 3–4 baseline in A/B test before full rollout
- Negotiation intelligence (cross-version clause tracking)
- Market benchmarking (anonymized cross-customer stats, with privacy guardrails)
- Risk scoring engine

### Phase 6: Platform Evolution (18+ months)

**Goal:** Full agreement intelligence platform.

- Graph-based clause relationships
- Contrastive learning for embedding improvement
- Self-play agent evaluation (generator vs critic)
- Schema versioning and ontology evolution
- Predictive analytics (negotiation outcome prediction)

---

## 9. Success Metrics & Evaluation Framework

### Product Metrics

| Category | Metric | Target | Measurement Method |
|----------|--------|--------|--------------------|
| Adoption | % agreements analyzed | > 60% within 6 months of launch | Platform telemetry |
| Adoption | % users engaging with clause insights | > 40% MAU | UI event tracking |
| Value | Policy violations auto-detected (precision) | ≥ 80% | Quarterly manual audit sample (n=200) |
| Value | Recommendation accept rate | > 35% (vs 0% baseline) | Accept/reject/edit event tracking |
| Value | Review time reduction | 30% | A/B cohort comparison |
| Stickiness | Repeat usage rate | > 50% week-over-week | Session analytics |
| Stickiness | Feedback loop participation (accept/reject/edit) | > 20% of recommendations receive feedback | Event tracking |

### ML Evaluation Framework

Each layer has independent quality gates. No layer ships without meeting its acceptance threshold.

| Layer | Metric | Acceptance Threshold | Gold Set | Cadence |
|-------|--------|---------------------|----------|---------|
| Clustering | Cluster purity (fraction of dominant clause type per cluster) | ≥ 0.75 | 500 manually labeled summaries across 10 accounts | Per re-clustering run |
| Clustering | Cluster stability (Adjusted Rand Index across consecutive runs) | ≥ 0.85 | Same input corpus, different seeds | Per re-clustering run |
| Ontology | Clause-type label precision | ≥ 0.85 | 200 cluster-label pairs, lawyer-reviewed | Per labeling batch |
| Ontology | Clause-type label recall (coverage of known clause families) | ≥ 0.70 for top 20 clause types | Curated taxonomy of 30 clause families | Quarterly |
| Extraction | Field extraction F1 (per top clause type) | ≥ 0.80 for payment_days, liability_cap, termination_notice | 100 agreements with manual field annotations | Per model/prompt update |
| Compliance | False-positive rate (incorrect violation flags) | ≤ 15% | 200 agreements with known policy status | Monthly audit sample |
| Compliance | False-negative rate (missed real violations) | ≤ 10% | Same audit set | Monthly audit sample |
| Recommendations | Recommendation relevance (accept rate uplift vs random baseline) | ≥ 2× random | Online A/B or interleaving experiment | Quarterly |
| Personalization | Preference prediction accuracy (predicted vs actual edit) | ≥ 0.60 | Holdout set of recent edits | Monthly |

### Error Budget & Release Gates

- **Clustering:** If cluster purity drops below 0.70 OR stability below 0.80 on the gold set, re-clustering is blocked until root cause is resolved.
- **Compliance:** If false-positive rate exceeds 20% in any monthly audit, compliance engine is switched to "audit-only" mode (findings visible to admins only, not end users) until precision is restored.
- **Ontology:** New clause-type labels require reviewer approval before entering production ontology. No auto-promotion.
- **Recommendations:** If accept rate is statistically indistinguishable from random in any quarterly evaluation, recommendation engine is deprioritized pending investigation.

### Cost & Latency Targets

| Workflow | Latency SLO | Cost Target | Measurement |
|----------|-------------|-------------|-------------|
| Clause detection (per agreement) | ≤ 5s (P95) | ≤ $0.02 per agreement | Cluster assignment + field extraction |
| Compliance check (per agreement) | ≤ 2s (P95) | ≤ $0.005 per agreement | Rule evaluation against extracted fields |
| Recommendation retrieval | ≤ 3s (P95) | ≤ $0.01 per query | ANN search + scoring |
| Reasoning trace generation | ≤ 1s (P95) | ≤ $0.005 per trace | Template rendering (no LLM call for standard traces) |
| Full clustering run (batch, per account) | ≤ 30 min | ≤ $5.00 per account | UMAP + HDBSCAN + labeling |
| LLM cluster labeling (batch) | ≤ 10 min for 50 clusters | ≤ $1.00 per labeling run | GPT-4o-mini, temperature=0 |
| Reranking (ColBERT, Phase 5+) | ≤ 500ms (P95) for top-100 | ≤ $0.03 per query | Precomputed token embeddings |

**Storage growth assumptions:**
- Feature store: ~1 KB per org × clause_type pair → manageable at 10K orgs × 50 clause types = 500 MB
- Reasoning traces: ~2 KB per trace → 10M traces/year = 20 GB/year
- Vector index (FAISS): ~1.5 KB per embedding × 100M embeddings = 150 GB (on-disk, memory-mapped)

---

## 10. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cluster noise | Bad ontology → bad recommendations | Keep top 20–50 clusters; prune long tail; human-in-the-loop validation |
| Agent hallucinations | Schema/field drift | Constrain agent outputs; critic agent validation; human review |
| Lawyer trust | Low adoption | Structured reasoning traces; never generate explanations from LLM alone |
| Overbuilding infra | Delayed product-market fit | Ship value per phase; avoid full ontology/GNN/graph DB before demonstrating product value |
| Embedding limitations | "shall indemnify" ≈ "may indemnify" | Domain-specific model refinement; hybrid retrieval with multi-vector reranking |
| Schema explosion | User confusion from too many fields | Start narrow (3–5 fields per clause type); expand based on demand |
| Legal-advice liability | Recommendations misinterpreted as legal judgment | All outputs are assistive; provenance always shown; mandatory human approval for high-severity findings |
| Cross-customer data leakage | Privacy/regulatory violation | Tenant-isolated aggregation; minimum cohort size; opt-in only; differential privacy; legal review |

---

## 11. Governance & Human-Review Operating Model

This system produces outputs that affect legal compliance decisions. Autonomous operation without governance is unacceptable. The following operating model is a **first-class design requirement**, not an afterthought.

### 11.1 Ontology Governance

| Event | Governance Requirement |
|-------|------------------------|
| New ClauseType created (from cluster labeling) | Enters ontology as "provisional". Promoted to "active" only after reviewer approval. |
| ClauseType label changed | Requires reviewer approval. Old label retained in version history. |
| ClauseType merged or split | Requires reviewer approval + downstream impact assessment (which policies, preferences, and features reference this type). |
| Field added to ClauseType | Enters as "provisional". Promoted after extraction F1 meets threshold on gold set. |
| Field deprecated | Requires reviewer approval + migration path documented. |

**Reviewer queue:** All pending ontology changes surface in a dedicated admin workflow. If the reviewer queue exceeds 50 items or items age beyond 7 days, alerts fire to the ML platform team.

### 11.2 Compliance Governance

| Rule | Requirement |
|------|-------------|
| Violation findings (≥ "violation" severity) | Always require human review before any downstream action. No auto-remediation. |
| Warning findings | Visible to users but clearly labeled as suggestions. |
| Confidence below combined threshold | Finding suppressed from user view; logged for audit. |
| Policy rule creation/modification | Requires admin approval. Changes logged with author, timestamp, and diff. |

### 11.3 Model & Pipeline Release Gates

| Component | Release Gate |
|-----------|-------------|
| BERTopic re-clustering | Must pass purity (≥ 0.75) and stability (≥ 0.85) on gold set. Reviewed by ML lead. |
| LLM cluster labeling prompt change | Must pass label precision (≥ 0.85) on gold set. Reviewed by ML lead. |
| Field extraction prompt change | Must pass F1 (≥ 0.80) on gold set per clause type. Reviewed by ML lead. |
| Recommendation scoring weight change | Must pass accept-rate uplift test. Reviewed by product lead. |
| Compliance rule engine change | Requires legal/compliance team sign-off. |

### 11.4 Rollback Semantics

- Ontology: Version-controlled. Any version can be rolled back to within 5 minutes. Downstream caches (feature store, reasoning store) are invalidated on rollback.
- Clustering: Previous cluster model artifact retained in Azure Blob. Rollback = re-point cluster assignment to previous artifact.
- Compliance rules: Version-controlled in AIDB. Rollback = revert to prior version.

### 11.5 Audit Trail

All system decisions that influence user-facing outputs are logged:

```
AuditEvent {
    event_type: enum[cluster_assignment, clause_labeling, field_extraction,
                     policy_evaluation, recommendation, ontology_change]
    entity_id: string          // agreement_id, clause_type_id, etc.
    input_summary: string      // what went in
    output_summary: string     // what came out
    confidence: float
    model_version: string
    timestamp: timestamp
    actor: enum[system, reviewer, admin]
}
```

Retention: 3 years minimum (aligned with legal compliance retention requirements).
