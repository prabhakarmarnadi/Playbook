# Field Quality: Confidence Gating & Phantom Field Pruning

## Overview

The pipeline uses a multi-layer quality system to ensure extracted fields are real (grounded in document text) and not hallucinated by the LLM. Two mechanisms work together:

1. **Confidence Gating** — a cost-saving sample-then-scale extraction strategy
2. **Phantom Field Pruning** — three-layer defense against fields with zero extractions

---

## Confidence Gating

### Problem

Running full RAG extraction across all agreements for every cluster is expensive (~100-500 LLM calls per cluster). Some clusters produce consistently low-quality extractions — the LLM can't reliably ground values in the source text (e.g., highly templated boilerplate, or clusters with vague/short chunks).

### Solution

Extract a **statistically representative sample** first. If quality is too low, skip the full run.

```
Cluster: "Indemnification" (850 chunks, 10 fields)
                    │
         ┌──────────┴──────────┐
         │  Phase 1: SAMPLE     │
         │  Cochran formula →   │
         │  select 25 chunks    │
         │  Run RAG extraction  │
         └──────────┬──────────┘
                    │
              mean confidence
              of sample extractions
                    │
          ┌─────────┴─────────┐
     ≥ threshold          < threshold
     GATE PASSED          GATE FAILED
          │                    │
   ┌──────┴──────┐    ┌───────┴───────┐
   │ Phase 2:    │    │ Skip cluster  │
   │ Full RAG on │    │ Prune phantom │
   │ all 850     │    │ fields        │
   │ chunks      │    │ Save LLM cost │
   └─────────────┘    └───────────────┘
```

### Sample Size: Cochran Formula

Instead of a fixed sample, we use the Cochran formula for statistical representativeness:

```
sample = clamp(cluster_chunks × cochran_fraction, cochran_min, cochran_max)
```

| Parameter | Default | Effect |
|-----------|:-------:|--------|
| `cochran_sample_fraction` | 0.3 | 30% of cluster chunks |
| `cochran_min_sample` | 5 | Never sample fewer than 5 |
| `cochran_max_sample` | 30 | Cap at 30 (diminishing returns) |

Chunks are selected via **diversity sampling** — spread across different agreements to avoid bias toward any single document.

### Gate Decision

```python
mean_confidence = average(extraction.confidence for extraction in sample_results)

if mean_confidence >= confidence_gate_threshold:
    # PASSED — proceed with full extraction
else:
    # FAILED — skip this cluster, prune its fields
```

### Threshold Tuning Guide

| Threshold | Behavior | Use Case |
|:---------:|----------|----------|
| `0.70` | Aggressive — skip anything mediocre | Production with strict quality requirements |
| `0.60` | Balanced — skip clearly bad clusters | Default production |
| `0.55` | Permissive — maximize coverage | Hackathon / exploratory analysis |
| `0.40` | Very permissive — almost nothing is skipped | When you want everything, accept noise |
| `off` | `enable_confidence_gate=False` | Full extraction regardless of quality |

### Config

```python
MinimalPipelineConfig(
    enable_confidence_gate=True,        # on/off
    confidence_gate_threshold=0.55,     # minimum mean confidence to proceed
    cochran_sample_fraction=0.3,        # sample proportion
    cochran_min_sample=5,
    cochran_max_sample=30,
)
```

---

## Phantom Field Pruning

### What Is a Phantom Field?

A field definition that exists in the `field_definitions` table but has **zero rows** in the `extractions` table. The LLM "discovered" the field but could never actually extract a value from any document.

Example for a "Non-Compete" cluster:

| Field | Extractions | Status |
|-------|:-----------:|--------|
| `non_compete_duration` | 12 | ✅ Active |
| `geographic_scope` | 8 | ✅ Active |
| `competitor_definition` | 0 | ❌ Phantom |
| `damages_cap` | 0 | ❌ Phantom |

### Why Phantoms Are Harmful

- **Misleading metrics** — field counts look impressive (3,000 fields!) but half do nothing
- **Wasted LLM cost** — the RAG extractor attempts extraction for every field × every agreement; phantoms burn tokens with zero return
- **Noisy ontology** — knowledge graph, cluster profiles, and dashboards show fields that don't actually exist in the data
- **False promises for downstream consumers** — APIs and UIs suggest a field is available when it has no values

### Why They Happen

| Cause | % of Phantoms | Mechanism |
|-------|:------------:|-----------|
| **LLM hallucination** | ~30% | Agent reasons "this clause type should have X" — domain knowledge, not grounded evidence |
| **Confidence gate skip** | ~40% | Fields are persisted during discovery, then the cluster fails the gate — no extraction ever runs |
| **RAG retrieval miss** | ~30% | Field exists in documents but the hybrid retriever doesn't surface the right chunk — the LLM sees irrelevant context and returns null |

### Three-Layer Defense

Each layer catches a different wave of phantoms:

```
Discovery (RLM V3)
    │
    │  "10 fields discovered"
    ▼
┌────────────────────────────────────┐
│ Layer 1: Quick-Sample Validation   │  ← After discovery, before full extraction
│                                    │
│ Extract from 3 random docs.        │
│ Drop fields with 0/3 hits.         │
│                                    │
│ Catches: ~30-50% of hallucinated   │
│ fields before any real cost.       │
└─────────────┬──────────────────────┘
              │  "7 fields survive"
              ▼
┌────────────────────────────────────┐
│ Layer 2: Confidence Gate Pruning   │  ← When the gate fails for this cluster
│                                    │
│ If mean_confidence < threshold:    │
│   DELETE all field_definitions     │
│   for this cluster.               │
│                                    │
│ Catches: all phantom fields from   │
│ clusters where extraction quality  │
│ is too low to proceed.             │
└─────────────┬──────────────────────┘
              │  (only if gate passes)
              ▼
┌────────────────────────────────────┐
│ Layer 3: Post-Extraction Pruning   │  ← After full RAG extraction completes
│                                    │
│ For each persisted field:          │
│   COUNT extractions.               │
│   If 0 → DELETE from               │
│   field_definitions.               │
│                                    │
│ Catches: fields that survived      │
│ validation but still got nothing   │
│ in production extraction.          │
└────────────────────────────────────┘
```

### Expected Phantom Rates

| Rate | Quality Level |
|:----:|---------------|
| **< 10%** | Excellent — nearly every field has real values |
| **10-20%** | Good — some fields are legitimately sparse (exist in clause type but not in every document) |
| **20-30%** | Acceptable — room for improvement in retrieval or discovery prompts |
| **30-50%** | Problem — indicates retrieval failures, hallucination, or missing pruning |
| **> 50%** | Broken — most "discovered" fields are noise |

### Diagnosing High Phantom Rates

```sql
-- Phantom rate by source
WITH fc AS (
    SELECT fd.field_id, fd.source, COUNT(e.extraction_id) as n
    FROM field_definitions fd
    LEFT JOIN extractions e ON e.field_id = fd.field_id
    GROUP BY fd.field_id, fd.source
)
SELECT source,
       COUNT(*) as total,
       SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) as phantom,
       ROUND(100.0 * SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct
FROM fc
GROUP BY source;
```

| Source | What it means if phantom rate is high |
|--------|--------------------------------------|
| `agent` | RLM agent is hallucinating fields — tighten the discovery prompt or increase `critique_fields()` frequency |
| `domain_seed` | Domain-level fields don't apply to this specific cluster — quick-sample validation should catch these |
| `rag_extractor` | RAG retrieval is failing — check chunk quality, embedding model, or BM25 index |

---

## Interaction Between the Two Systems

```
                    Cluster arrives
                          │
                   ┌──────┴──────┐
                   │ RLM V3      │
                   │ Discovery   │
                   │ (10 fields) │
                   └──────┬──────┘
                          │
                   ┌──────┴──────┐
                   │ Layer 1:    │
                   │ Quick-sample│──→ 3 fields dropped (0/3 hits)
                   │ Validation  │
                   └──────┬──────┘
                          │
                   "7 fields persisted to DB"
                          │
                   ┌──────┴──────┐
                   │ Confidence  │
                   │ Gate Sample │
                   └──────┬──────┘
                          │
              ┌───────────┴───────────┐
         GATE PASSED              GATE FAILED
              │                       │
              ▼                       ▼
       Full RAG extraction      Layer 2: Delete all 7
              │                 field_definitions
              ▼                 (0 phantom fields remain)
       Layer 3: Post-prune
       Delete fields with
       0 extractions
       (e.g., 2 more pruned)
              │
              ▼
       "5 active fields,
        each with real values"
```

The net effect: every field that survives in the database has been validated at multiple stages and has at least one real extraction grounded in document text.
