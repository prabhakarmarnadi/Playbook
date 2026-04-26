# Quality & Latency Levers — RLM V3 Production Tuning

**Date:** April 2026  
**Context:** Clustering V2 pipeline with RLM V3 (agentic REPL + structured state + dual-mode workers)

---

## 1. The Latency Problem

Measured discovery time per cluster (136 clusters, 198 docs):

| Approach | Discovery/Cluster | LLM Calls/Cluster | Total Discovery (5 workers) | Multiplier vs Standard |
|----------|------------------:|-------------------:|----------------------------:|-----------------------:|
| Standard | ~3.5s | 1 | ~1.6 min | 1× |
| RLM+LB V1 | ~35s | 10–25 | ~16 min | 10× |
| V2 (structured) | ~21s | 6–10 (fixed) | ~10 min | 6× |
| **V3 (projected)** | **~25–40s** | **8–25 (agent)** | **~11–18 min** | **7–11×** |

Intent extraction remains the pipeline bottleneck at scale (API rate limits), but discovery becomes the bottleneck _per cluster_ with RLM approaches.

---

## 2. Latency Levers

### Lever 1 — Tiered Discovery (highest impact, zero quality loss)

Run standard single-shot on ALL clusters first, then V3 only on under-discovered clusters.

```
Standard on 136 clusters     →  3.5s each  →  ~1.6 min total
  ↓ Score quality: field count, type diversity, description length
V3 on bottom ~20% (~27)      →  25s each   →  ~2.3 min total
                                               ─────────────
                                               ~3.9 min total (vs ~14 min naive V3)
```

**Scoring heuristic for "needs deeper discovery":**
- `field_count < 3` → definitely re-run
- `field_type_diversity < 0.5` (all fields are "text") → re-run
- `avg_description_length < 20 chars` → re-run
- Cluster has `chunk_count > 50` but `field_count < 5` → complex content, under-discovered

### Lever 2 — Reduce RLM Budget

| Setting | Hackathon | Production | Aggressive | Quality Impact |
|---------|-----------|-----------|------------|---------------|
| `max_iterations` | 25 | 12 | 8 | Minimal — most converge by iter 8 |
| `max_llm_calls` | 50 | 25 | 15 | Moderate — limits exploration depth |
| `optuna_trials` | 30 | 10 | 5 | Small — top-3 trials capture 90% of quality |
| `adaptive_stop_patience` | 3 | 2 | 2 | Faster convergence for easy clusters |

**Expected time savings:** ~40–60% reduction in V3 discovery time.

### Lever 3 — Prompt Caching (Azure OpenAI)

Azure automatically caches identical prompt prefixes. Cached input tokens cost **$0.03/1M** vs **$0.25/1M** (88% cheaper).

**How to exploit:** Structure prompts so the system instruction + field schema + state brief is a shared prefix across all `llm_query()` / `llm_explore()` calls within a cluster. Only the chunk text varies.

```
┌─────────────────────────────────────────┐  ← CACHED (identical across calls)
│ System: You are a legal contract...     │
│ Schema: fields, clause_types, ...       │
│ State brief: {"already_found": ...}     │
├─────────────────────────────────────────┤
│ [Excerpt 3]                             │  ← VARIES (only this part is new)
│ The Licensee shall maintain...          │
└─────────────────────────────────────────┘
```

**Estimated saving:** ~75% of input tokens → each RLM call drops from ~$0.0002 to ~$0.00005.

### Lever 4 — Batch API (50% off, non-real-time)

Azure Batch API delivers results within 24h at 50% off all token costs. V3's fallback mode (structured grounded + exploratory passes) produces independent requests that batch perfectly.

| Mode | Latency | Cost |
|------|---------|------|
| Real-time | ~250ms/call | $0.25/$2.00 per 1M (in/out) |
| Batch | ≤24h | $0.125/$1.00 per 1M (in/out) |

**Use case:** Bulk onboarding, nightly re-processing, retroactive extraction on archive.

### Lever 5 — Split Models (Orchestrator vs Workers)

V3 supports separate `lm` (orchestrator) and `sub_lm` (worker). Workers handle 80%+ of token volume.

| Role | Model | Cost (input/1M) | Justification |
|------|-------|----------------:|---------------|
| Orchestrator (REPL agent) | GPT-5.4 | $1.25 | Needs reasoning to decide strategy |
| Workers (grounded + explore) | GPT-5.4-mini | $0.25 | Structured extraction, schema-guided |

**Saving:** ~5× reduction on the bulk of token spend.

### Lever 6 — Field Cache Across Runs

Many clusters are stable across document batches. If a cluster exists from a prior run and its centroid hasn't shifted (cosine sim > 0.95 to previous), skip re-discovery entirely.

```python
# Pseudocode
if cluster_exists_in_db(cluster_id):
    old_centroid = load_centroid(cluster_id)
    if cosine_sim(old_centroid, new_centroid) > 0.95:
        fields = load_cached_fields(cluster_id)  # skip discovery
        return fields
```

**Impact:** At steady state (incremental document ingestion), most clusters are stable → discovery drops to near-zero for 80–90% of clusters.

### Lever 7 — V3 Adaptive Stopping (already built)

The `coverage_report()` tool lets the REPL agent see convergence in real-time. Simple clusters ("Governing Law" — 2–3 fields) converge in 2–3 iterations (~8s). Complex clusters ("Indemnification" — 12+ fields) use the full budget. Average should be well under max.

---

## 3. Quality Levers

### The Quality Measurements (from eval)

| Metric | Standard | RLM+LB | V2 | V3 (projected) |
|--------|----------|--------|-----|----------------|
| Fill rate (macro) | 0.312 | **0.421** | 0.395 | 0.42–0.45 |
| Fill rate (clause) | 0.298 | **0.445** | 0.410 | 0.44–0.47 |
| Grounding score | 0.845 | **0.894** | 0.872 | 0.88–0.90 |
| Hallucination rate | 0.045 | 0.038 | **0.030** | 0.028–0.035 |
| Field count/cluster | 5.2 | 8.1 | 7.8 | 8–10 |
| Niche field ratio | 0.05 | 0.12 | **0.18** | 0.18–0.22 |

### Quality Lever A — Dual-Mode Exploration Ratio

Control how much creative exploration vs precise grounding:

| Profile | Grounded Temp | Exploratory Temp | Niche Yield | Hallucination Risk |
|---------|-------------:|----------------:|------------:|-------------------:|
| Conservative | 0.1 | 0.5 | Low | Lowest |
| **Balanced (default)** | **0.2** | **0.8** | **Medium** | **Low** |
| Aggressive | 0.3 | 1.0 | High | Medium |

### Quality Lever B — Overlap Merge Threshold

Controls how aggressively exploratory items are deduplicated against grounded items:

| Threshold | Exploratory Kept | Diversity | Duplication Risk |
|----------:|----------------:|----------:|-----------------:|
| 0.75 | Many | High | Higher — some near-duplicates |
| **0.85 (default)** | **Balanced** | **Medium** | **Low** |
| 0.92 | Few | Low | Minimal — only truly novel items |

### Quality Lever C — Canonicalization Strictness

| Embedding Sim Threshold | Fields Removed | Risk |
|------------------------:|---------------:|------|
| 0.75 | Many | Over-merging distinct but similar fields |
| **0.82 (default)** | Balanced | Good dedup with niche preservation |
| 0.90 | Few | More duplicates but no false merges |

### Quality Lever D — Max Fields per Cluster

| Max Fields | Coverage | Processing Time | Downstream Extraction |
|-----------:|---------:|----------------:|----------------------:|
| 8 | Basic | Fast | Faster, less discovery |
| **15 (default)** | Good | Medium | Balanced |
| 25 | Exhaustive | Slow | More null extractions |

---

## 4. Recommended Profiles

### Profile: Real-Time (API serving)

```python
RLMV3Config(
    max_iterations=8,
    max_llm_calls=15,
    max_fields=12,
    adaptive_stop_patience=2,
    grounded_temp=0.2,
    exploratory_temp=0.6,
)
# + Tiered discovery (standard first, V3 on gaps)
# + Field cache across runs
# Expected: ~4 min discovery for 136 clusters
```

### Profile: Quality (bulk processing)

```python
RLMV3Config(
    max_iterations=20,
    max_llm_calls=40,
    max_fields=20,
    adaptive_stop_patience=3,
    grounded_temp=0.2,
    exploratory_temp=0.8,
    overlap_threshold=0.82,
)
# + Batch API for cost savings
# + Full GPT-5.4 for orchestrator
# Expected: ~18 min discovery, highest quality
```

### Profile: Cost-Optimized (high volume)

```python
RLMV3Config(
    max_iterations=6,
    max_llm_calls=12,
    max_fields=10,
    adaptive_stop_patience=2,
    grounded_temp=0.15,
    exploratory_temp=0.5,
)
# + GPT-5.4-mini for all roles
# + Batch API + prompt caching
# + Field cache (skip stable clusters)
# Expected: ~$0.018/doc, ~3 min discovery
```

---

## 5. Comparison: RLM V3 vs Agentic Extraction Experiments

The "Crazy Experiments" explored a fundamentally different bottleneck from what V3 addresses. Here's where they sit:

```
                    DISCOVERY                          EXTRACTION
            "What fields exist?"                "What values fill them?"
            ═══════════════════                 ═══════════════════════
    V3 operates here ──►                │
                                        │
                ┌─ Standard (1 shot)     │   ┌─ Standard (1 shot)
                ├─ RLM+LB (agentic)     │   ├─ Cluster-conditioned
    DISCOVERY ──┤                        │   │
                ├─ V2 (structured)       │   │    Crazy Experiments
                └─ V3 (hybrid) ◄────    │   │    operate here ──►
                                        │   │
                                        │   ├─ Multi-Step Snippet Selection
                                        │   ├─ Reviewer Agent
                                EXTRACTION──┤
                                        │   ├─ Document NL Index
                                        │   └─ Agentic Snippet Search
```

**V3 and the Crazy Experiments are complementary, not competing.** V3 improves _what to look for_ (field schemas); the experiments improve _how to find values_ (snippet selection + validation). They compose.

### 5.1 Multi-Step Snippet Selection

**What it does:** When extraction fails due to insufficient context, the LLM outputs "Insufficient Context: Let's search: {signature...}" and retries with targeted retrieval, up to 3 rounds.

**Result:** +3% F1 (71% → 74%) on GPT-5.4-mini.

**How it compares to V3:**

| Dimension | V3 Discovery | Multi-Step Snippet |
|-----------|-------------|-------------------|
| Stage | Discovery (schema) | Extraction (values) |
| Problem solved | "What fields exist?" | "Can't find the value in this chunk" |
| Mechanism | Agentic REPL + dual-mode workers | Agentic retry with targeted re-retrieval |
| LLM cost | ~10× standard discovery | ~1–3× per extraction (most succeed on attempt 1) |
| Quality gain | +35% fill rate, +22% niche fields | +3% F1 on extraction accuracy |

**Synergy:** V3 discovers richer field schemas → Multi-Step Snippet finds values for those fields that would otherwise return null. This is multiplicative:

```
V3 discovers 10 fields (vs 5 standard)  ×  Snippet Selection fills 74% (vs 71%)
= 7.4 extracted values per cluster  vs  3.55 baseline
= ~2.1× combined improvement
```

**Recommendation:** High priority integration. The retry loop is cheap (most extractions succeed on attempt 1; only ~15–20% need retry) and directly addresses the #1 extraction failure mode. Wire it into `_extract_chunk()` in `production_pipeline.py`:

```python
async def _extract_chunk_with_retry(llm, cluster_label, chunk_text, fields, 
                                     lance_store, agreement_id, max_retries=3):
    for attempt in range(max_retries):
        result = await _extract_chunk(llm, cluster_label, chunk_text, fields)
        
        # Check for insufficient context signals
        retry_queries = [v for k, (v, c) in result.items() 
                        if c < 0.3 or "insufficient" in str(v).lower()]
        if not retry_queries or attempt == max_retries - 1:
            return result
        
        # Retrieve additional context via LanceDB vector search
        additional = lance_store.search(
            query_text=retry_queries[0], 
            agreement_id=agreement_id, top_k=3
        )
        chunk_text = chunk_text + "\n\nADDITIONAL CONTEXT:\n" + "\n".join(additional)
    return result
```

### 5.2 Reviewer Agent

**What it does:** A second model validates extracted values against the source evidence. Input is minimal (value + 1–2 sentences) so cost is trivial with caching.

**How it compares to V3:**

| Dimension | V3 | Reviewer Agent |
|-----------|-----|---------------|
| Stage | Discovery | Post-extraction validation |
| Problem solved | Finding the right schema | Catching hallucinated values |
| Mechanism | Agentic exploration + canonicalization | Adversarial verification |
| Quality signal | Overlap merge + niche preservation | Binary accept/reject per extraction |

**Synergy:** V3's `_confidence_score()` already does rule-based grounding checks (substring match, numeric detection). A reviewer agent would be a **learned replacement** for that heuristic:

```
Current:  extracted value → rule-based confidence → threshold filter
Proposed: extracted value → reviewer LLM (value + evidence span) → accept/reject
```

**Cost estimate:** With prompt caching, the reviewer sees ~100 tokens per extraction. At 23K extractions × $0.03/1M cached input = **$0.07 per full pipeline run**. Negligible.

**Recommendation:** Medium priority. Implement as an optional post-extraction pass. The V3 span_text from discovery provides the evidence anchor — the reviewer checks `(field_name, extracted_value, span_text)` triples.

### 5.3 Document NL Index

**What it does:** Generate a TOC-like structure index per document. When snippet selection fails, an agent reasons over the index to locate relevant sections — mimicking how a human expert would navigate a 172-page agreement.

**How it compares to V3:**

| Dimension | V3 | NL Index |
|-----------|-----|----------|
| Scope | Cross-document (cluster-level) | Within-document (navigation) |
| Problem solved | Schema discovery across corpus | "Where in THIS document is field X?" |
| Mechanism | Rolling state + trajectory memory | Structural TOC + heading detection |
| When useful | Always (discovery) | Long documents where retrieval fails |

**Synergy:** This is the missing piece in the full pipeline. V3 discovers fields. Multi-Step Snippet tries to extract. When snippet search fails for long documents, the NL Index provides a fallback navigation layer:

```
V3 discovers: "recycling_fee" field
  → Extraction attempt on nearest chunk: null (no match)
  → Multi-Step retry: vector search retrieves wrong section
  → NL Index fallback: "recycling_fee → Section: Franchise Payments and Fees (pg 71)"
  → Re-extract from page 71 context: "$2,500/month" ✓
```

**Cost estimate:** One-time per document using GPT-5.4-nano (~$0.001/doc). Amortized over all extractions.

**Recommendation:** High priority for long-document corpora (50+ pages). Can be generated during the chunking stage and stored in DuckDB. Short documents (<10 pages) don't need it — the chunker already captures full context.

### 5.4 Combined Architecture (V3 + All Experiments)

```
STAGE 1: DISCOVERY (V3)                    STAGE 2: EXTRACTION (Experiments)
═══════════════════                        ════════════════════════════════

PDF → Chunk → Embed → Cluster              For each (cluster, chunk, fields):
         │                                  ┌──────────────────────────────┐
         ▼                                  │ Attempt 1: Direct extraction │
┌─────────────────┐                         │   confidence ≥ 0.5 → accept │
│ Tiered Discovery│                         │   confidence < 0.3 → retry  │
│  Standard first │                         ├──────────────────────────────┤
│  V3 on gaps     │──► field schemas        │ Attempt 2: Snippet Search   │
│  + cache reuse  │    per cluster          │   LanceDB vector retrieval  │
└─────────────────┘                         │   Re-extract with context   │
         │                                  ├──────────────────────────────┤
         │                                  │ Attempt 3: NL Index Lookup  │
         │                                  │   TOC navigation for long   │
         │                                  │   docs, section-targeted    │
         │                                  │   re-extraction             │
         │                                  ├──────────────────────────────┤
         │                                  │ Reviewer Agent (post-pass)  │
         │                                  │   Validate (value, evidence)│
         │                                  │   Accept / Reject / Flag    │
         ▼                                  └──────────────────────────────┘
  Intent Discovery                                      │
  Obligation Detection                                   ▼
  Knowledge Graph                               Final extractions
                                                with confidence scores
```

### 5.5 Expected Combined Impact

| Metric | Standard Baseline | V3 Only | V3 + Experiments |
|--------|------------------:|--------:|-----------------:|
| Fields discovered/cluster | 5.2 | 8–10 | 8–10 |
| Extraction F1 | 71% | 71% (same extractor) | **~77–80%** |
| Fill rate | 0.31 | 0.42–0.45 | **0.50–0.55** |
| Hallucination rate | 4.5% | 2.8–3.5% | **1.5–2.5%** |
| Cost/doc (mini) | $0.020 | $0.025 | ~$0.030 |
| Discovery latency (136 clusters) | 1.6 min | 4 min (tiered) | 4 min |
| Extraction latency (1871 clauses) | 5 min | 5 min | ~7 min (retries) |

**The V3 + Experiments stack is the clear production path.** Discovery gains (V3) and extraction gains (experiments) are orthogonal and compose multiplicatively. Total cost increase is marginal (~$0.01/doc) for a projected ~2× improvement in usable extractions.

---

## 6. Implementation Priority

| # | Component | Effort | Impact | Dependencies |
|---|-----------|--------|--------|-------------|
| 1 | **Tiered Discovery** | Small | High | V3 already built |
| 2 | **Multi-Step Snippet Selection** | Medium | High | LanceDB search API |
| 3 | **Reduced RLM Budgets** | Trivial | Medium | Config change only |
| 4 | **Field Cache Across Runs** | Small | High at steady-state | Centroid similarity check |
| 5 | **Prompt Caching** | Small | Medium | Prompt restructuring |
| 6 | **NL Index Generation** | Medium | High for long docs | Chunker integration |
| 7 | **Reviewer Agent** | Medium | Medium | Post-extraction pass |
| 8 | **Batch API Integration** | Small | Medium (cost only) | Async result polling |
| 9 | **Split Models** | Trivial | Medium (cost) | Config change only |
