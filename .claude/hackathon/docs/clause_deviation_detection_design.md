# Clause Deviation Detection Pipeline — Technical Design Document

**Version:** 1.0  
**Date:** April 2026  
**Author:** Agreement Intelligence Platform Team  
**Status:** Draft  
**Audience:** ML Engineers, Backend Engineers, Product Leadership

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Overview](#2-system-overview)
3. [Architecture Principles](#3-architecture-principles)
4. [Workflow 1: Playbook Hydration](#4-workflow-1-playbook-hydration)
5. [Workflow 2: PDF Processing](#5-workflow-2-pdf-processing)
6. [Workflow 3: Deviation Detection](#6-workflow-3-deviation-detection)
7. [Data Model](#7-data-model)
8. [Inference Infrastructure](#8-inference-infrastructure)
9. [Scoring & Severity Classification](#9-scoring--severity-classification)
10. [Borrowed Components from Clustering Pipeline](#10-borrowed-components-from-clustering-pipeline)
11. [Failure Modes & Mitigations](#11-failure-modes--mitigations)
12. [Scalability Analysis](#12-scalability-analysis)
13. [Evaluation Strategy](#13-evaluation-strategy)
14. [Deployment Topology](#14-deployment-topology)

---

## 1. Problem Statement

Enterprise legal teams maintain playbooks — prescriptive rule sets that define acceptable clause language for each contract type. Today, deviation detection is manual: a reviewer reads each clause, mentally recalls the playbook rule, and judges whether the language conforms. This process takes 2-4 hours per contract and misses subtle deviations in 15-30% of cases.

**Goal:** Given a contract PDF and a tenant's playbook, automatically identify every clause that deviates from playbook expectations, classify the severity, detect missing mandatory clauses, and produce a structured deviation report — in under 60 seconds per contract.

### Requirements

| Requirement | Target |
|---|---|
| Latency (per contract) | < 60s for 30-page contract |
| Deviation precision | ≥ 0.90 (avoid false alarms) |
| Deviation recall | ≥ 0.85 (catch real deviations) |
| Missing clause detection | 100% coverage of required types |
| Structured output | 100% valid JSON (no parse failures) |
| Concurrent throughput | ≥ 50 contracts/minute |
| Playbook onboarding | < 10 minutes per new playbook |
| Cost per contract | < $0.15 at steady state |

---

## 2. System Overview

The pipeline decomposes into three sequential workflows executed via Temporal task queues:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WF1: Playbook Hydration (once per tenant)                              │
│  Playbook XLSX → parse → hydrate rules → embed → store to PostgreSQL    │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HydratedRules in DB
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WF2: PDF Processing (per contract)                                     │
│  PDF → extract text → filter blocks → classify clauses → extract fields │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ TypedClause[]
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WF3: Deviation Detection (per contract)                                │
│  TypedClauses + HydratedRules → match → 3-tier cascade → report        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                   ContractDeviationReport
```

Each workflow runs on dedicated Temporal task queues, enabling independent scaling of CPU-bound preprocessing, GPU-bound inference, and I/O-bound database operations.

---

## 3. Architecture Principles

| # | Principle | Rationale |
|---|---|---|
| 1 | **Tiered cost escalation** | Resolve deviations with the cheapest signal first (regex < embedding < LLM). Only escalate when uncertain. |
| 2 | **Constrained generation** | All LLM outputs pass through XGrammar schemas. Zero tolerance for malformed JSON. |
| 3 | **Type safety at boundaries** | Clause type conflicts are detected and rejected before deviation scoring, not after. |
| 4 | **Playbook-as-code** | Playbook rules compile into executable artifacts (CEL expressions, JSONLogic rules, regex patterns) at hydration time, not at scoring time. |
| 5 | **Embedding polarity** | Every rule carries both a desirable and undesirable embedding. HyDE generates synthetic negatives when the playbook lacks explicit undesirable language. |
| 6 | **Separation of discovery and enforcement** | The clustering pipeline discovers *what exists*; this pipeline enforces *what should exist*. Neither replaces the other. |

---

## 4. Workflow 1: Playbook Hydration

**Trigger:** Manual — run once when a tenant onboards a new playbook.  
**Frequency:** Re-run on playbook revision (expected quarterly).  
**Duration:** 3-8 minutes depending on playbook size (50-500 rules).

### 4.1 Activity 1: `parse_playbook_Vx`

**Task Queue:** `playbook-ops` (CPU, low priority)

Ingests the playbook document and extracts structured rules.

**Input:** Playbook file (XLSX, CSV, or structured PDF)

**Processing:**
1. Detect format by inspecting column headers
2. Load tenant configuration (jurisdiction defaults, clause type taxonomy)
3. Extract raw rules with field-level granularity

**Output artifact — `RawPlaybookRule[]`:**

```json
{
  "rule_id": "R001",
  "topic": "Limitation of Liability",
  "subtopic": "Cap Amount",
  "field_rules": {
    "cap_amount": {"type": "currency", "operator": "<=", "threshold": 5000000},
    "cap_type": {"type": "enum", "allowed": ["aggregate", "per-incident"]}
  },
  "desirable_language": "Liability shall not exceed the total fees paid...",
  "undesirable_language": null,
  "deployment_logic": "apply_when: clause_type IN ['limitation_of_liability']",
  "model_refs": ["qwen3-8b-lol-v2"]
}
```

### 4.2 Activity 2: `hydrate_rules_batch_v4`

**Task Queue:** `hydration-compute` (CPU/GPU, medium priority)

This is the core compilation step. Each raw rule is hydrated into three executable artifact types plus XGrammar schemas.

**Processing per rule:**

**a) Extraction Pattern Generation**

Compile field rules into extraction-time instructions:

| Pattern Type | Example | Use |
|---|---|---|
| `digit_regex` | `r"\$[\d,]+(?:\.\d{2})?"` | Extract monetary values from clause text |
| `duckling` | `{"dim": "amount-of-money"}` | Facebook Duckling entity extraction |
| `cfg` | Context-free grammar fragment | Structured field boundaries |

**b) Comparison Rule Pre-compilation**

Generate executable comparison expressions:

```
CEL:    clause.cap_amount > rule.threshold  →  is_deviation = true
CEL:    is_deviation && cap_amount > 10M    →  severity = "critical"
JSONLogic: {">" : [{"var": "cap_amount"}, 5000000]}
```

CEL (Common Expression Language) is chosen for its sub-millisecond evaluation, sandboxed execution, and human readability. JSONLogic serves as a portable fallback for environments without a CEL runtime.

**c) XGrammar Schema Construction**

Build JSON Schema objects that constrain downstream LLM extraction output:

```json
{
  "type": "object",
  "properties": {
    "cap_amount": {"type": "number"},
    "cap_type": {"type": "string", "enum": ["aggregate", "per-incident", "none"]},
    "cap_period": {"type": "string"}
  },
  "required": ["cap_amount", "cap_type"]
}
```

These schemas are compiled into XGrammar finite-state automata at serving time, guaranteeing 100% structurally valid JSON from the extraction LLM.

**d) HyDE Negative Generation**

When `undesirable_language` is null (common — playbooks often only specify *what they want*, not *what they don't want*):

1. Prompt GPT-4o with the rule's desirable language, topic, and field constraints
2. Request 3 diverse synthetic negatives:
   - A clause that superficially resembles the desirable language but violates a field constraint
   - A clause from a different jurisdiction that changes the legal meaning
   - A clause with missing key protections
3. Embed all three negatives → average into `undesirable_embedding`

**Rationale:** Tier 2 deviation detection relies on bipolar embedding comparison (`desirable_sim` vs. `undesirable_sim`). Without a negative pole, Tier 2 degenerates to simple similarity-to-standard, which cannot distinguish "compliant but differently worded" from "non-compliant."

### 4.3 Activity 3: `generate_embeddings_v4`

**Task Queue:** `embedding-ops` (CPU, I/O bound)

| Parameter | Value |
|---|---|
| Model | Azure OpenAI `text-embedding-3-large` |
| Dimensions | 3072 |
| Batch size | 64 texts per API call |
| Rate limit | 500 RPM (Azure tier) |

Two embeddings per rule:
- `desirable_embedding`: From `desirable_language` text
- `undesirable_embedding`: From `undesirable_language` or HyDE-generated text

### 4.4 Activity 4: `store_hydrated_rules_v4`

**Task Queue:** `db-ops` (CPU, I/O bound)

Batch upsert all hydrated rules into PostgreSQL with pgvector:

```sql
INSERT INTO playbook_rules_v4 (
    rule_id, tenant_id, playbook_id, topic, subtopic,
    extraction_patterns,     -- JSONB
    comparison_rules,        -- JSONB
    xgrammar_schema,         -- JSONB
    desirable_embedding,     -- vector(3072)
    undesirable_embedding,   -- vector(3072)
    field_rules,             -- JSONB
    desirable_language,      -- TEXT
    undesirable_language,    -- TEXT
    hydrated_at
) VALUES (...)
ON CONFLICT (rule_id, tenant_id) DO UPDATE SET ...;
```

**Indexes:**

| Index | Type | Purpose |
|---|---|---|
| `idx_topic_trgm` | GIN trigram | Fuzzy topic matching in WF3 Activity 2 |
| `idx_emb_desirable_ivfflat` | IVFFlat (pgvector) | ANN search for Tier 2 semantic matching |
| `idx_emb_undesirable_ivfflat` | IVFFlat (pgvector) | ANN search for pole scoring |
| `idx_tenant_playbook` | B-tree | Partition rules by tenant + playbook |

---

## 5. Workflow 2: PDF Processing

**Trigger:** Automatic — fires when a new contract PDF is uploaded.  
**Duration:** 8-15 seconds per 30-page contract.

### 5.1 Stage 0: `extract_pdf_text`

**Task Queue:** `pdf-processing` (CPU)

| Component | Technology |
|---|---|
| PDF parser | muPDF via `fitz.open()` |
| Output granularity | Text blocks with page number and bounding box coordinates |

**Output artifact — `PDFBlock[]`:**

```json
{
  "block_id": "b_001",
  "page": 3,
  "text": "The Contractor shall indemnify and hold harmless...",
  "bbox": [72.0, 144.5, 540.0, 198.2]
}
```

Bounding boxes enable downstream UI features: clause highlighting, PDF annotation overlays, and spatial-aware chunking (detecting multi-column layouts, headers, footers).

### 5.2 Stage 0.5: `score_blocks_batch`

**Task Queue:** `pdf-processing` (CPU)

Pre-filters noise blocks (headers, footers, page numbers, watermarks, tables of contents) before they reach the GPU-bound classification stage.

**Three-signal relevance scoring:**

| Signal | Method | Weight |
|---|---|---|
| Lexical overlap | RapidFuzz `partial_ratio` against exemplar clauses | 0.3 |
| Semantic similarity | Model2Vec cosine similarity to clause-type centroids | 0.4 |
| Term importance | c-TF-IDF score against exemplar database | 0.3 |

**Threshold:** Blocks scoring below 0.60 combined relevance are discarded.

**Impact:** Filters approximately 35-45% of raw blocks, reducing GPU inference cost proportionally.

**Output artifact — `SemanticSample[]`:**

```json
{
  "block_id": "b_001",
  "relevance_score": 0.82,
  "candidates": ["indemnification", "liability"],
  "pass": true
}
```

### 5.3 Stage 1: `classify_clauses_batch`

**Task Queue:** `gpu-inference`

**Model:** Qwen3-8B with QLoRA adapter (rank=16, alpha=32)  
**Serving:** vLLM with continuous batching  
**Task:** Multi-label classification across 32 clause types  
**Training data:** 12,000 labeled clauses from production contracts

**Clause type taxonomy (32 types):**

The taxonomy covers the standard enterprise legal clause categories: `limitation_of_liability`, `indemnification`, `confidentiality`, `termination`, `governing_law`, `ip_ownership`, `force_majeure`, `non_compete`, `non_solicitation`, `warranty`, `representations`, `payment_terms`, `insurance`, `data_protection`, `anti_corruption`, `assignment`, `notice`, `amendment`, `waiver`, `severability`, `entire_agreement`, `survival`, `dispute_resolution`, `arbitration`, `audit_rights`, `subcontracting`, `change_of_control`, `exclusivity`, `most_favored_nation`, `price_escalation`, `service_levels`, `acceptance_criteria`.

**Output per block:**

```json
{
  "block_id": "b_001",
  "clause_types": ["indemnification", "liability"],
  "confidence": [0.94, 0.72],
  "multi_label": true
}
```

### 5.4 Stage 1.5: `augment_classifications`

**Task Queue:** `pdf-processing` (CPU)

A safety net for high-stakes clause types where false negatives are unacceptable.

**Problem:** The QLoRA classifier may miss edge cases for clause types that have legally significant consequences (e.g., `anti_corruption`, `non_compete`, `change_of_control`). A missed anti-corruption clause could result in regulatory exposure.

**Solution:** For a predefined `PROBLEM_CLAUSES` set, run a secondary check:

1. RapidFuzz matching against canonical clause headings
2. Model2Vec cosine similarity against per-type exemplar embeddings
3. If either signal exceeds threshold and the classifier missed it, append the classification

**Output artifact — `ClassifiedClause[]`:**

```json
{
  "clause_id": "CL_003",
  "block_ids": ["b_001", "b_002"],
  "clause_types": ["indemnification"],
  "confidence": 0.94,
  "augmented": false
}
```

### 5.5 Stage 4: `extract_fields_batch`

**Task Queue:** `gpu-inference`

**Model:** Qwen3-8B with SLOT LoRA adapters (one adapter per clause type family)  
**Constraint:** XGrammar-enforced JSON schema (compiled from WF1 hydrated schemas)  
**Guarantee:** 100% structurally valid JSON output — no post-hoc parsing, no retry loops

**SLOT LoRA architecture:**

The base Qwen3-8B model stays loaded in GPU memory. Different LoRA adapter weights are hot-swapped per request based on the clause type, without reloading the base model. This achieves per-type specialization at near-zero switching cost.

| LoRA Slot | Clause Types | Rank | Training Samples |
|---|---|---|---|
| `slot_financial` | payment_terms, price_escalation, mfn | 16 | 2,400 |
| `slot_liability` | lol, indemnification, warranty | 16 | 3,100 |
| `slot_governance` | governing_law, dispute, arbitration | 8 | 1,800 |
| `slot_operational` | sla, acceptance, force_majeure | 8 | 1,500 |
| `slot_general` | All other types (fallback) | 16 | 3,200 |

**Output — `TypedClause[]`:**

```json
{
  "clause_id": "CL_003",
  "clause_types": ["indemnification"],
  "extracted_fields": {
    "indemnifying_party": "Contractor",
    "indemnified_party": "Client",
    "cap_amount": 5000000,
    "cap_type": "aggregate",
    "carve_outs": ["gross negligence", "willful misconduct"],
    "survival_period": {"value": 24, "unit": "months"}
  },
  "embedding": [0.012, -0.034, ...],  // 3072-dim
  "raw_text": "The Contractor shall indemnify..."
}
```

---

## 6. Workflow 3: Deviation Detection

**Trigger:** Automatic — fires after WF2 completes for a contract.  
**Duration:** 2-8 seconds per contract (dominated by Tier 3 LLM calls, if any).

### 6.1 Activity 1: `load_playbook_rules`

**Task Queue:** `db-ops`

```sql
SELECT * FROM playbook_rules_v4
WHERE tenant_id = $1
  AND playbook_id = $2
  AND active = true;
```

Loads all hydrated rules for the tenant's active playbook into memory. Typical size: 50-200 rules, ~50 MB with embeddings.

### 6.2 Activity 2: `match_clause_topics`

**Task Queue:** `tier-processing`

Maps each classified clause to its applicable playbook rules via a 4-tier matching cascade:

| Tier | Method | Example | Resolves |
|---|---|---|---|
| **Exact** | Normalized string equality | `"indemnification" == "indemnification"` | ~60% |
| **Synonym** | Lookup in curated synonym table | `"hold harmless" → "indemnification"` | ~15% |
| **Fuzzy** | RapidFuzz `token_sort_ratio ≥ 80` | `"limitation on liability" → "limitation_of_liability"` | ~15% |
| **Semantic** | Cosine similarity ≥ 0.75 on topic embeddings | Novel clause language | ~10% |

**Output artifact — `TopicMatch[]`:**

```json
{
  "clause_id": "CL_003",
  "clause_types": ["indemnification"],
  "matched_rules": ["R001", "R002"],
  "match_method": "exact",
  "match_confidence": 1.0
}
```

### 6.3 Type Conflict Guard

A decision gate that prevents nonsensical clause-rule pairings.

**Problem:** Multi-label classification means a clause may carry types `["anti_corruption", "compliance"]`. If the matched rule targets `limitation_of_liability`, scoring this pairing produces meaningless deviation results and erodes user trust.

**Logic:**

```
IF clause.clause_types ∩ rule.applicable_types == ∅:
    SKIP this (clause, rule) pair
    LOG: "Type mismatch: clause {CL_003} types {anti_corruption} 
          vs rule {R001} target {limitation_of_liability}"
ELSE:
    PROCEED to deviation scoring
```

**Confusable type pairs** requiring stricter enforcement:
- `anti_corruption` ↔ `compliance` (topically adjacent but legally distinct)
- `limitation_of_liability` ↔ `indemnification` (often co-located in contracts)
- `non_compete` ↔ `non_solicitation` (different scope, different enforceability)

### 6.4 Activity 3: `process_deviation_tiers`

**Task Queue:** `tier-processing`

The core deviation detection engine. Implements a 3-tier cost-escalation cascade where each tier is more expensive but more capable than the last. A clause-rule pair exits the cascade as soon as any tier reaches a confident verdict.

#### Tier 1: Deterministic — Regex + CEL

**Cost:** ~0.01 ms per clause-rule pair  
**Resolution rate:** ~60% of cases

For rules with field-level constraints (numeric thresholds, enumerated values, date comparisons), apply the pre-compiled extraction patterns and CEL expressions:

```
EXAMPLE — Rule R001: "Liability cap must not exceed $5M"

Step 1: Extract field via regex
  Input:  "total liability shall not exceed Ten Million Dollars ($10,000,000)"
  Regex:  r"\$[\d,]+" → "$10,000,000"
  Parse:  10000000 (integer)

Step 2: Evaluate CEL expression
  Expression: clause.cap_amount > rule.threshold
  Evaluate:   10000000 > 5000000 → true
  Result:     is_deviation = true

Step 3: Compute severity via CEL
  Expression: cap_amount > threshold * 2
  Evaluate:   10000000 > 10000000 → false
  Severity:   "major" (not "critical" since not >2x)
```

**Exit condition:** Field extracted AND CEL evaluates without error → emit verdict, skip Tiers 2-3.

**Fallthrough condition:** No regex match, ambiguous parse, or rule lacks field constraints → escalate to Tier 2.

#### Tier 2: Statistical — Embedding Similarity

**Cost:** ~2 ms per clause-rule pair (vector dot products)  
**Resolution rate:** ~30% of remaining cases

Computes bipolar similarity between the clause embedding and the rule's desirable/undesirable embeddings:

```
desirable_sim   = cosine(clause_embedding, rule.desirable_embedding)
undesirable_sim = cosine(clause_embedding, rule.undesirable_embedding)
pole_score      = desirable_sim - undesirable_sim

EXAMPLE:
  desirable_sim:   0.42  (weak alignment with expected language)
  undesirable_sim: 0.78  (strong alignment with deviation patterns)
  pole_score:     -0.36  (negative = closer to undesirable pole)
  → DEVIATION
```

##### Dynamic Thresholds

A fixed global threshold fails for clause types with overlapping embedding spaces. Confusable types require stricter boundaries:

| Clause Type | Threshold | Rationale |
|---|---|---|
| Limitation of Liability | 0.72 | High confusion with Indemnification |
| Indemnification | 0.72 | High confusion with Limitation of Liability |
| Anti-Corruption | 0.75 | Confusable with general Compliance clauses |
| Non-Compete | 0.70 | Confusable with Non-Solicitation |
| General (all other types) | 0.60 | Default |

Thresholds are calibrated empirically: for each clause type, compute the similarity distribution of known-compliant and known-deviant clauses, then set the threshold at the EER (Equal Error Rate) point.

**Exit condition:** `|pole_score| > dynamic_threshold` → emit verdict, skip Tier 3.

**Fallthrough condition:** `|pole_score| ≤ dynamic_threshold` (ambiguous zone) → escalate to Tier 3.

#### Tier 3: Generative — LLM Analysis

**Cost:** ~$0.003 per clause-rule pair (GPT-4o)  
**Resolution rate:** Handles all remaining edge cases

Reserved for genuinely ambiguous cases — typically complex conditional language, jurisdiction-specific nuance, or novel clause structures.

**Prompt structure:**

```
You are a senior contract reviewer. Compare the following clause 
against the playbook rule and determine if the clause deviates.

PLAYBOOK RULE:
  Topic: {rule.topic}
  Expected: {rule.desirable_language}
  Field constraints: {rule.field_rules}

CONTRACT CLAUSE:
  {clause.raw_text}

EXTRACTED FIELDS:
  {clause.extracted_fields}

Respond with:
  classification: "Standard" | "Non-Standard" | "Not-Applicable"
  severity: "critical" | "major" | "attention" | "standard"
  explanation: <2-sentence rationale>
  confidence: 0.0-1.0
```

**Cost guard:** Tier 3 invocations are capped at a per-tenant budget (configurable, default: 200 calls/hour). When the budget is exhausted, the clause-rule pair is flagged as `"needs_review"` instead of escalating.

### 6.5 Activity 4: `run_pattern_evaluators`

Eight specialized evaluators that detect structural and policy-level deviations beyond field-value comparison:

| Evaluator | Detection Target | Example |
|---|---|---|
| `forbidden_phrase` | Banned language patterns | "best efforts" in a jurisdiction requiring "reasonable efforts" |
| `compliance_checklist` | Required sub-clauses present | GDPR clause must have data transfer, DPA, and breach notification |
| `stakeholder_routing` | Conditional approval rules | Deals > $10M require VP Legal sign-off |
| `tiered_approval` | Threshold-based escalation | Indemnity > $5M → Board approval |
| `mutual_obligation` | Reciprocity verification | Confidentiality must be mutual, not one-sided |
| `temporal_validity` | Date/duration constraint enforcement | Non-compete > 2 years is unenforceable in California |
| `jurisdiction_conflict` | Governing law compatibility | Delaware corp using UK governing law |
| `cross_reference` | Inter-clause dependency | Termination clause must reference Survival clause |

### 6.6 Activity 5: `detect_missing_clauses`

Compares the set of clause types found in the contract against the tenant's required clause type list:

```
required_types   = {t for r in playbook_rules if r.mandatory}
found_types      = {t for c in typed_clauses for t in c.clause_types}
missing          = required_types - found_types
```

**Output per missing type:**

```json
{
  "clause_type": "confidentiality",
  "severity": "critical",
  "playbook_rules": ["R015", "R016"],
  "recommendation": "Add a mutual confidentiality clause covering..."
}
```

### 6.7 Output: `ContractDeviationReport`

```json
{
  "contract_id": "CTR-2026-0451",
  "tenant_id": "tenant_acme",
  "playbook_id": "pb_enterprise_saas_v4",
  "processed_at": "2026-04-06T18:30:00Z",
  "processing_time_ms": 4200,
  
  "summary": {
    "total_clauses": 28,
    "total_rules_applied": 142,
    "deviations": {"critical": 2, "major": 3, "attention": 5, "standard": 18},
    "missing_clauses": ["confidentiality", "ip_ownership"],
    "tier_resolution": {"tier1": 85, "tier2": 42, "tier3": 15},
    "overall_risk": "high"
  },
  
  "deviations": [
    {
      "clause_id": "CL_003",
      "clause_type": "indemnification",
      "rule_id": "R001",
      "severity": "critical",
      "tier_resolved": 1,
      "deviation_detail": {
        "field": "cap_amount",
        "expected": "≤ $5,000,000",
        "found": "$10,000,000",
        "explanation": "Indemnification cap exceeds playbook maximum by 2x."
      },
      "clause_text": "The Contractor shall indemnify...",
      "page": 12,
      "bbox": [72.0, 144.5, 540.0, 198.2]
    }
  ],
  
  "missing_clauses": [
    {
      "clause_type": "confidentiality",
      "severity": "critical",
      "recommendation": "Add mutual confidentiality clause per Rule R015."
    }
  ]
}
```

---

## 7. Data Model

### 7.1 Primary Store: PostgreSQL + pgvector

| Table | Purpose | Key Columns |
|---|---|---|
| `playbook_rules_v4` | Hydrated playbook rules | `rule_id`, `tenant_id`, `topic`, `extraction_patterns` (JSONB), `comparison_rules` (JSONB), `xgrammar_schema` (JSONB), `desirable_embedding` vector(3072), `undesirable_embedding` vector(3072) |
| `deviation_results` | Per-clause deviation verdicts | `result_id`, `contract_id`, `clause_id`, `rule_id`, `severity`, `tier_resolved`, `deviation_detail` (JSONB) |
| `contract_reports` | Aggregate reports | `report_id`, `contract_id`, `tenant_id`, `summary` (JSONB), `created_at` |
| `missing_clauses` | Detected absences | `contract_id`, `clause_type`, `severity`, `recommendation` |
| `tier_metrics` | Per-tier resolution tracking | `contract_id`, `tier`, `invocations`, `resolutions`, `avg_latency_ms` |
| `playbook_versions` | Hydration version control | `version_id`, `playbook_id`, `tenant_id`, `hydrated_at`, `rule_count`, `validation_hash`, `is_active` |

### 7.2 Index Strategy

```sql
-- Topic matching (fuzzy)
CREATE INDEX idx_topic_trgm ON playbook_rules_v4 
  USING GIN (topic gin_trgm_ops);

-- Embedding ANN (Tier 2)
CREATE INDEX idx_emb_desirable ON playbook_rules_v4 
  USING ivfflat (desirable_embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_emb_undesirable ON playbook_rules_v4 
  USING ivfflat (undesirable_embedding vector_cosine_ops) WITH (lists = 100);

-- Tenant isolation
CREATE INDEX idx_tenant_playbook ON playbook_rules_v4 (tenant_id, playbook_id);

-- Result queries
CREATE INDEX idx_results_contract ON deviation_results (contract_id);
CREATE INDEX idx_results_severity ON deviation_results (severity) 
  WHERE severity IN ('critical', 'major');
```

---

## 8. Inference Infrastructure

### 8.1 vLLM Deployment

| Parameter | Value |
|---|---|
| Base model | Qwen3-8B (BF16, ~16 GB VRAM) |
| Serving framework | vLLM with continuous batching |
| LoRA management | SLOT LoRA — hot-swap adapters without model reload |
| GPU | NVIDIA A100 80GB (supports 4 concurrent LoRA slots) |
| Max batch size | 64 sequences |
| Max sequence length | 4096 tokens |
| Tensor parallelism | 1 (single-GPU per instance) |

### 8.2 LoRA Adapter Configuration

```yaml
base_model: Qwen/Qwen3-8B
lora_slots:
  - name: classification_qlora
    path: /models/adapters/classify_v2
    rank: 16
    alpha: 32
    target_modules: [q_proj, k_proj, v_proj, o_proj]
    task: multi_label_classification
    
  - name: extraction_financial
    path: /models/adapters/extract_financial_v1
    rank: 16
    alpha: 32
    task: field_extraction
    clause_types: [payment_terms, price_escalation, mfn]
    
  - name: extraction_liability
    path: /models/adapters/extract_liability_v1
    rank: 16
    alpha: 32
    task: field_extraction
    clause_types: [lol, indemnification, warranty]
    
  - name: extraction_general
    path: /models/adapters/extract_general_v1
    rank: 16
    alpha: 32
    task: field_extraction
    clause_types: [all_other]
```

### 8.3 XGrammar Integration

XGrammar compiles JSON Schemas into finite-state automata that constrain the LLM's token sampling at generation time:

```
JSON Schema (from hydrated rule)
        ↓
  XGrammar Compiler
        ↓
  Finite-State Automaton
        ↓
  Token Mask at each decoding step
        ↓
  Guaranteed valid JSON
```

**Impact:** Eliminates the `json.loads()` → retry → re-prompt loop that otherwise consumes 5-15% of inference budget on malformed outputs.

---

## 9. Scoring & Severity Classification

### 9.1 Multi-Signal Combined Score

Each clause-rule pair produces a combined score from three independent signals:

$$\text{combined\_score} = 0.3 \times S_{\text{field}} + 0.3 \times S_{\text{regex}} + 0.4 \times S_{\text{embedding}}$$

| Signal | Computation | Range |
|---|---|---|
| $S_{\text{field}}$ | Fraction of field constraints satisfied | [0, 1] |
| $S_{\text{regex}}$ | Fraction of regex patterns matched | [0, 1] |
| $S_{\text{embedding}}$ | Normalized pole score: $(d_{\text{sim}} - u_{\text{sim}} + 1) / 2$ | [0, 1] |

Weights calibrated via grid search over 15 manually-labeled production contracts (450 clause-rule pairs).

### 9.2 Severity Classification Flow

A three-level cascade determines the final severity label:

```
┌─────────────────────────────────────────────────┐
│  LEVEL 1: LLM Classification Field              │
│                                                  │
│  IF tier3_response.classification exists:        │
│    "Standard"       → severity = STANDARD        │
│    "Not-Applicable" → FILTER OUT (skip clause)   │
│    "Non-Standard"   → fall through to Level 3    │
└────────────────────────┬────────────────────────┘
                         │ (Non-Standard or no LLM)
                         ▼
┌─────────────────────────────────────────────────┐
│  LEVEL 2: Explanation Text Scan                  │
│                                                  │
│  Scan LLM explanation for sentiment signals:     │
│                                                  │
│  Standard phrases:                               │
│    "aligns with", "matches", "complies with",    │
│    "consistent with", "in accordance"            │
│    → severity = STANDARD                         │
│                                                  │
│  Deviation phrases:                              │
│    "deviates from", "differs from",              │
│    "fails to", "does not include",               │
│    "exceeds", "below minimum"                    │
│    → fall through to Level 3                     │
└────────────────────────┬────────────────────────┘
                         │ (no clear signal)
                         ▼
┌─────────────────────────────────────────────────┐
│  LEVEL 3: Combined Score Thresholds              │
│                                                  │
│  combined_score ≥ 0.70  →  STANDARD              │
│  combined_score ≥ 0.50  →  ATTENTION             │
│  combined_score ≥ 0.30  →  MAJOR                 │
│  combined_score <  0.30  →  CRITICAL             │
└─────────────────────────────────────────────────┘
```

**Why this matters:** Before this cascade, all clauses that failed Tier 1 regex matching defaulted to "attention" — even when the LLM clearly stated "this clause aligns with the playbook." The three-level cascade ensures the LLM's semantic understanding is respected first.

---

## 10. Borrowed Components from Clustering Pipeline

Several components are adopted or adapted from the existing clustering pipeline (Architecture A). Each is listed with the rationale for inclusion.

### 10.1 Legal-Aware Semantic Chunker

**Source:** `core/chunker.py` from the clustering pipeline  
**Used in:** WF2 Stage 0 post-processing (splitting raw PDF blocks into clause-level segments)

**Why required:** muPDF extracts text at the *block* level — rectangular regions on the page. Legal contracts have deeply nested structures (Article → Section → Subsection → Paragraph) where a single block may contain multiple distinct clauses. The clustering pipeline's chunker handles this via:

- **Neighbor-smoothed breakpoint detection:** Reduces false splits from single outlier sentences. Smoothing formula: $d'_i = (1 - w) \cdot d_i + w \cdot \text{mean}(d_{i-1}, d_i, d_{i+1})$ where $w = 0.3$
- **Legal structural bonuses:** Adds +0.15 to the distance signal at patterns like `ARTICLE`, `Section X.Y`, `WHEREAS` — ensuring splits align with legal structure rather than arbitrary embedding distances
- **Cohesion-based sub-splitting:** For omnibus clauses with multiple topics in one paragraph, recursively splits when intra-segment cosine similarity drops below 0.75

**Adaptation for this pipeline:** Use the chunker as a pre-processing step between Stage 0 (block extraction) and Stage 0.5 (relevance filtering). The clustering pipeline's 100-2000 character constraint maps well to the clause granularity needed for classification.

### 10.2 Field Normalization

**Source:** `core/field_normalizer.py` from the clustering pipeline  
**Used in:** WF2 Stage 4 post-processing and WF3 Tier 1 pre-comparison

**Why required:** XGrammar guarantees structurally valid JSON, but it does not normalize semantic values. A clause containing "Ten Million Dollars ($10,000,000)" may be extracted as either `"cap_amount": "Ten Million Dollars ($10,000,000)"` (string) or `"cap_amount": 10000000` (integer), depending on the LLM's interpretation. Tier 1 CEL expressions require canonical numeric representations.

**Normalization patterns adopted:**

| Type | Input → Output | Pattern |
|---|---|---|
| Currency | `"$500K"` → `{"amount": 500000, "currency": "USD"}` | Regex + scale suffix handling (K/M/B) |
| Duration | `"thirty (30) days"` → `{"value": 30, "unit": "days"}` | Written number dict + regex |
| Date | `"January 1, 2026"` → `{"date": "2026-01-01"}` | 4 canonical date patterns |
| Percentage | `"5.5 percent"` → `{"value": 5.5}` | Regex |
| Boolean | `"mutual"` → `true` | Semantic word list lookup |

**Adaptation:** Used *after* XGrammar extraction and *before* CEL evaluation, ensuring the CEL runtime receives canonical types.

### 10.3 Risk Scoring Framework

**Source:** `core/risk_scorer.py` from the clustering pipeline  
**Used in:** Overall contract risk classification (enriching the `ContractDeviationReport`)

**Why required:** The deviation pipeline produces per-clause verdicts, but product stakeholders need an aggregate contract-level risk signal for triage ("which of my 200 contracts should a human review first?"). The clustering pipeline's risk scorer provides this through:

- **7 risk categories:** liability, IP, termination, compliance, financial, confidentiality, operational
- **Dual-mode scoring:** Fast heuristic (keyword matching, microsecond latency) for bulk scoring; LLM-based for high-precision classification
- **Contract-level aggregation:** Per-agreement risk profiles with portfolio-level heatmaps

**Adaptation:** The deviation pipeline adds its own deviations as additional risk signals:

```python
# Merged risk signal
base_risk = risk_scorer.score_agreement(clause_texts)  # from clustering pipeline
deviation_risk = sum(
    severity_weight[d.severity] 
    for d in deviations
) / len(clauses)

final_risk = 0.4 * base_risk + 0.6 * deviation_risk
```

This weighted merge ensures that a contract with no deviations but inherently risky clause language (e.g., uncapped indemnification that happens to match the playbook) still surfaces as high-risk.

### 10.4 Feedback-Driven Threshold Calibration

**Source:** `core/feedback_store.py` and `core/personalization.py` from the clustering pipeline  
**Used in:** Dynamic threshold refinement for Tier 2, and severity re-calibration for Tier 3

**Why required:** The initial dynamic thresholds (0.60-0.75) are calibrated from 15 contracts. As the system processes more contracts and reviewers correct false positives / false negatives, the thresholds should adapt per tenant.

**Mechanism:**

| Feedback Event | Impact on Pipeline |
|---|---|
| Reviewer marks deviation as "false positive" | Raise Tier 2 threshold for that clause type by 0.02 (reduce sensitivity) |
| Reviewer marks standard as "missed deviation" | Lower Tier 2 threshold for that clause type by 0.02 (increase sensitivity) |
| Reviewer overrides severity | Adjusts combined score → severity mapping thresholds |
| Reviewer adds custom rule | Creates new entry in `playbook_rules_v4`, triggers re-hydration |

**Temporal decay:** Feedback weight halves every 90 days, preventing stale corrections from permanently biasing thresholds.

### 10.5 Incremental Playbook Rule Assignment

**Source:** `core/cluster_assigner.py` (FIELD-801) from the clustering pipeline  
**Used in:** WF3 Activity 2 when encountering clause types not in the playbook taxonomy

**Why required:** Contracts may contain clause types that don't exactly match any playbook rule topic. Rather than dropping these clauses, the centroid-based assignment strategy from the clustering pipeline maps novel clause types to the nearest-matching rule:

```python
# From cluster_assigner.py logic, adapted for playbook rules
similarities = cosine(clause_embedding, all_rule_topic_embeddings)
best_sim = max(similarities)

if best_sim >= 0.85:    # HIGH_CONFIDENCE
    → Apply matched rule normally
elif best_sim >= 0.60:  # TENTATIVE
    → Apply matched rule but flag as "approximate_match"
else:                   # NOVEL
    → Skip rule matching, flag as "unmatched_clause" for human review
```

**Why not just rely on WF3 Activity 2's semantic matching?** The activity's matching cascade is topic-string focused. This component adds embedding-level fallback for completely novel clause language with no lexical overlap to any rule topic.

---

## 11. Failure Modes & Mitigations

### 11.1 vLLM Server Failure

| Scenario | Impact | Mitigation |
|---|---|---|
| vLLM OOM on large batch | WF2 Stages 1 & 4 halt | Configure `max_batch_size=64` and `max_model_len=4096`; implement request queuing |
| vLLM crash / restart | In-flight requests lost | Temporal auto-retries (3 attempts with exponential backoff); idempotent processing |
| LoRA adapter corruption | Garbage classifications | MD5 checksum validation on adapter load; fallback to `extraction_general` slot |

**Recommended:** Deploy vLLM behind a load balancer with 2+ replicas and `/health` probe.

### 11.2 Embedding Model Unavailability

| Scenario | Impact | Mitigation |
|---|---|---|
| Azure OpenAI rate limit (429) | WF1 hydration stalls | Exponential backoff with jitter; batch embeddings (64/call) |
| Azure OpenAI outage | No new playbook hydration | Cache embeddings in PostgreSQL; hydration can wait hours |
| Embedding model version change | Silent accuracy degradation | Pin model version in config; validate cosine sim distribution against known-good benchmarks |

### 11.3 Tier Escalation Runaway

| Scenario | Impact | Mitigation |
|---|---|---|
| Poor HyDE negatives → Tier 2 unreliable → 40%+ Tier 3 escalation | $200+/day LLM cost spike | Per-tenant Tier 3 budget cap (200 calls/hr); Tier 2.5 ensemble vote (2-of-3 signals agree → resolve without LLM) |
| GPT-4o returns ambiguous classification | Severity defaults to "attention" | Explanation text scan (Level 2); combined score threshold fallback (Level 3) |
| Cascading failures across all 3 tiers | No verdict produced | Final fallback: flag as `"needs_review"` with all intermediate signals attached |

### 11.4 Playbook Hydration Corruption

| Scenario | Impact | Mitigation |
|---|---|---|
| Bad CEL expression compiled | Tier 1 false positives/negatives | Post-hydration validation gate: run against 10 gold-labeled contracts before promoting to production |
| XGrammar schema mismatch | Extraction produces wrong fields | Schema diff check against previous version; alert on >20% field changes |
| HyDE generates degenerate negatives | Tier 2 pole scoring fails | Validate: `cosine(desirable_emb, undesirable_emb) < 0.85` — reject if poles are too similar |

---

## 12. Scalability Analysis

### 12.1 Throughput Model

For a 30-page contract (~28 clauses, ~140 rules applied):

| Stage | Latency | Parallelism | Bottleneck |
|---|---|---|---|
| WF2 Stage 0 (PDF extract) | 200 ms | Per-contract | CPU-bound |
| WF2 Stage 0.5 (filter) | 100 ms | Per-block | CPU-bound |
| WF2 Stage 1 (classify) | 3 sec | Batched on GPU | **GPU-bound** |
| WF2 Stage 4 (extract) | 5 sec | Batched on GPU | **GPU-bound** |
| WF3 Activity 2 (match) | 50 ms | Per-clause | CPU-bound |
| WF3 Activity 3 (tiers) | 1.5 sec | Per-clause-rule | Mix (Tier 3 = API) |
| WF3 Activity 4 (evaluators) | 200 ms | Per-clause | CPU-bound |
| **Total** | **~11 sec** | | |

### 12.2 Scaling Strategy

| Load | GPU Instances | Temporal Workers | PostgreSQL | Throughput |
|---|---|---|---|---|
| Low (< 100 contracts/day) | 1× A100 | 4 CPU workers | Single primary | ~8 contracts/min |
| Medium (1K contracts/day) | 2× A100 | 8 CPU workers | Primary + read replica | ~50 contracts/min |
| High (10K contracts/day) | 4× A100 (2 classify + 2 extract) | 16 CPU workers | Primary + 2 replicas | ~200 contracts/min |

### 12.3 Cost Model (Steady State, Medium Load)

| Component | Monthly Cost | Notes |
|---|---|---|
| GPU inference (2× A100 spot) | $4,800 | ~$3.50/hr spot pricing |
| Azure OpenAI embeddings | $450 | Hydration only (amortized) |
| GPT-4o Tier 3 calls | $900 | ~10% escalation rate, 1K contracts/day |
| PostgreSQL (64GB, pgvector) | $600 | Azure Flexible Server |
| Temporal Cloud | $200 | Action-based pricing |
| **Total** | **~$6,950/mo** | **~$0.23/contract** |

---

## 13. Evaluation Strategy

### 13.1 Offline Evaluation

| Metric | Target | Dataset |
|---|---|---|
| Classification accuracy (32 types) | ≥ 0.93 F1 | 2,400 held-out labeled clauses |
| Field extraction accuracy | ≥ 0.88 exact match | 800 gold clause-field pairs |
| Deviation precision | ≥ 0.90 | 500 labeled deviations |
| Deviation recall | ≥ 0.85 | 500 labeled deviations |
| Missing clause detection | 100% | 100 contracts with known missing clauses |
| Tier resolution rate | T1 ≥ 55%, T2 ≥ 25%, T3 ≤ 20% | All test contracts |

### 13.2 Online Monitoring

| Signal | Alert Threshold | Action |
|---|---|---|
| Tier 3 escalation rate | > 25% (rolling 1hr) | Page on-call; investigate Tier 2 threshold drift |
| XGrammar schema violations | > 0% | Impossible by design — indicates library bug |
| Average latency per contract | > 30 sec (p95) | Scale GPU inference workers |
| Reviewer correction rate | > 15% (rolling 1wk) | Trigger threshold recalibration |
| HyDE negative similarity | desirable/undesirable cosine > 0.85 | Flag rule for manual undesirable language authoring |

### 13.3 A/B Testing Framework

Each tier is independently toggleable per tenant:

```yaml
tenant_acme:
  tier1_enabled: true
  tier2_enabled: true
  tier3_enabled: true       # Can disable to measure T1+T2-only accuracy
  tier3_model: "gpt-4o"     # Can swap to "claude-sonnet" for comparison
  dynamic_thresholds: true  # Can disable to measure global-threshold baseline
```

---

## 14. Deployment Topology

```
                        ┌──────────────────┐
                        │   API Gateway     │
                        │   (Contract Upload│
                        │    + Report Query)│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Temporal Server  │
                        │  (Workflow Engine)│
                        └──┬─────┬─────┬───┘
                           │     │     │
              ┌────────────┘     │     └────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
     │  CPU Workers   │ │  GPU Workers  │ │   DB Workers   │
     │  (Task Queues: │ │  (Task Queue: │ │  (Task Queue:  │
     │  playbook-ops, │ │  gpu-inference│ │   db-ops)      │
     │  pdf-processing│ │              )│ │                │
     │  tier-processing│ │  vLLM Server │ │                │
     │              ) │ │  Qwen3-8B    │ │                │
     └────────────────┘ │  + LoRA Slots│ └───────┬────────┘
                        └──────────────┘         │
                                                 ▼
                                        ┌────────────────┐
                                        │  PostgreSQL    │
                                        │  + pgvector    │
                                        │                │
                                        │  Tables:       │
                                        │  playbook_rules│
                                        │  deviation_res │
                                        │  tier_metrics  │
                                        └────────────────┘

External Dependencies:
  ├── Azure OpenAI (text-embedding-3-large) — WF1 hydration only
  └── GPT-4o API — WF3 Tier 3 (≤20% of clause-rule pairs)
```

---

*End of document.*
