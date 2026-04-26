# Pipeline Run Results — Tracking Document

## Run History

| Run | Date | Clusters | Fields | Extractions | Agreement Coverage | Phantom Rate | Confidence | Wall Time |
|-----|------|:--------:|:------:|:-----------:|:-----------------:|:------------:|:----------:|:---------:|
| Baseline (30 clusters) | Apr 22 | 30 | 496 | 2,717 | 13.9% (401/2889) | 47.0% | 0.881 | ~15 min |
| Full-458 (all improvements) | Apr 23 | 188 | 3,027 | 7,450 | 28.7% (829/2889) | 48.4% | 0.836 | 75.6 min |
| Full-v2 (coverage fixes) | Apr 24 | 2,904 | 10,878 | 23,483 | 58.4% (1686/2889) | 55.5% | 0.852 | 130 min |
| Full-v3 (lazy persist) | Apr 24 | 2,904 | 4,734 | 23,360 | 58.2% (1681/2889) | 0.1% | 0.852 | 120 min |
| **Full-v4 (ColBERT + wide scope)** | **Apr 24** | **2,904** | **5,497** | **28,328** | **68.3% (1973/2889)** | **0.1%** | **0.837** | **164 min** |
| **Full-v5 (quality + schema fixes)** | **Apr 24** | — | — | — | — | — | — | *running* |

---

## Domain-Seed Phantom Failure & Fix — Cross-Run Tracking

The domain-seed phantom problem was one of the most instructive failures in the pipeline. Here's how it evolved:

| Run | domain_seed Fields | Phantoms | Phantom Rate | What Happened |
|-----|:------------------:|:--------:|:------------:|---------------|
| Full-458 | 396 | 135 | **34.1%** ⚠️ | Domain seeds persisted blindly — no cluster-level validation |
| v2 | 931 | 15 | **1.6%** ✅ | Quick-sample validation: test 3 docs before persisting |
| v3 | 888 | 4 | **0.5%** ✅ | Same fix + lazy-persist catches remaining edge cases |

**The failure:** Stage 3.5 discovers universal fields at the *domain* level — "effective_date", "governing_law", "party_names" — and seeds them into every cluster in that domain. But domains are broad topic groups (120 total). A domain might contain clusters for "Governing Law", "Jurisdiction", and "Venue Selection" — all related, but each with different extractable fields. Seeding "venue_city" into a "Governing Law" cluster creates a phantom because that clause type never mentions a venue city.

**The root cause:** No validation between domain-level discovery and cluster-level persistence. The pipeline trusted that if a field was valid at the domain level, it would be valid for every cluster in that domain. This was wrong — domains are too coarse a grouping.

**The fix:** Quick-sample validation (added in v2). Before persisting any field — including domain seeds — the pipeline extracts from 3 sample documents for that cluster. If 0/3 yield a value, the field is pruned. This single check dropped domain_seed phantom rate from 34.1% → 1.6%.

**The lesson:** Domain-level discovery is valuable for seeding *candidates*, but cluster-level validation is mandatory before persistence. The two-tier architecture (Tier 1 domain seeds + Tier 2 agent deltas) only works when the tier boundary includes a quality gate.

---

## Run 1: Baseline (30 Clusters)

**Date:** April 22, 2026
**Config:** `max_clusters=30`, `min_cluster_chunks=20`, `merge_threshold=0.88`

| Metric | Value |
|--------|:-----:|
| Documents | 2,889 |
| Chunks | 144,464 |
| Raw clusters | 8,936 |
| Clusters processed | 30 (capped) |
| Domains | 120 |
| Field definitions | 496 |
| Extractions | 2,717 |
| Mean confidence | 0.881 |
| Clause intents | 55 |
| Intent extractions | 339 |
| Agreement coverage | 13.9% (401/2889) |
| Field hit rate | 53% |
| Phantom field rate | 47% |
| Grounding rate | 79% |

**Notes:** First end-to-end run. Capped at 30 clusters to validate pipeline. High confidence but low coverage.

---

## Run 2: Full-458 (All Quality Improvements)

**Date:** April 23, 2026
**Config:** `max_clusters=None`, `min_cluster_chunks=20`, `merge_threshold=0.88`, `max_fields=10`, `resume_from_stage=4`
**Run ID:** `run_f68320d0`

### Headline Metrics

| Metric | Value | vs Baseline |
|--------|:-----:|:-----------:|
| Documents | 2,889 | — |
| Chunks | 144,464 | — |
| Raw clusters (≥20 chunks) | 458 | — |
| After merge consolidation | 188 | 6.3× baseline |
| Domain field definitions | 1,111 | new |
| Field definitions | 3,027 | 6.1× |
| Extractions | 7,450 | 2.7× |
| Mean confidence | 0.836 | -5% |
| Clause intents | 665 | 12× |
| Intent extractions | 1,461 | 4.3× |
| Agreement coverage | 28.7% (829/2889) | 2× |
| Field hit rate | 51.6% (1561/3027) | -1.4pp |
| Phantom field rate | 48.4% (1466/3027) | +1.4pp |

### Quality Improvements Applied

1. `max_fields_per_cluster`: 15 → 10
2. Field dedup by embedding cosine > 0.85
3. Quick-sample validation (3 docs)
4. Post-extraction pruning (0 extractions → delete)
5. Type enforcement in extraction prompt
6. Multi-chunk evidence window (±500 chars)
7. Post-extraction type validation
8. Cross-domain universal fields (Stage 3.5b)
9. Remove max_clusters cap

### Stage Timing

| Stage | Time | % |
|-------|:----:|:-:|
| Domain field discovery (3.5 + 3.5b) | 2,009s | 44% |
| Discovery + extraction (Stage 4) | 2,522s | 56% |
| Assigner init (Stage 5) | 0.1s | 0% |
| **Total** | **4,537s (75.6 min)** | |

### LLM Usage

| Metric | Value |
|--------|:-----:|
| Total LLM calls | 33,421 |
| Total LLM latency | 67,000s |
| Avg latency per call | 2.0s |

### Extraction Confidence Distribution

| Threshold | Count | % |
|:---------:|:-----:|:-:|
| ≥ 0.9 | 3,492 | 46.9% |
| ≥ 0.8 | 4,442 | 59.6% |
| ≥ 0.7 | 6,241 | 83.8% |
| ≥ 0.6 | 7,096 | 95.2% |
| ≥ 0.5 | 7,347 | 98.6% |

### Phantom Field Analysis by Source

| Source | Total | Phantom | Phantom % |
|--------|:-----:|:-------:|:---------:|
| rag_extractor | 2,319 | 1,331 | 57.4% |
| **domain_seed** | **396** | **135** | **34.1%** ⚠️ |
| agent | 312 | 0 | 0.0% |

> **⚠️ Domain-Seed Failure:** One in three domain-seeded fields was phantom. Stage 3.5 discovers universal fields at the *domain* level (e.g., "effective_date" for all contract clusters), but many of these fields don't apply to every cluster's clause type. A "governing_law" cluster doesn't contain an "earnout_period" even though both belong to the same domain. Domain seeds were persisted without any cluster-level validation — they inflated field_definitions with fields that could never extract values from the cluster's actual text.
>
> **Root cause:** No quick-sample validation on seeded fields before persisting. The pipeline trusted domain-level discovery blindly.
>
> **Fix applied in v2:** Quick-sample validation (3 docs) now runs on ALL fields — including domain seeds — before they're persisted. Result: domain_seed phantom rate dropped from **34.1% → 1.6%** (-32.5pp).

### Coverage Gap Analysis

| Cluster Bucket | Clusters | Chunks | % of Total | Processed? |
|:--------------|:--------:|:------:|:----------:|:----------:|
| ≥ 20 chunks | 458 | 11,902 | 14.4% | ✅ (merged → 188) |
| 10-19 chunks | 2,446 | 31,307 | 37.9% | ❌ skipped |
| 5-9 chunks | 6,032 | 39,334 | 47.6% | ❌ skipped |
| **Total** | **8,936** | **82,543** | | |

### Critique

1. **85.6% of chunks are in clusters <20 and never processed.** Lowering `min_cluster_chunks` to 10 would recover 31K+ chunks (37.9%).
2. **Phantom rate unchanged at ~48%.** Domain-seeded fields inflate field_definitions but many don't apply to the cluster's clause type. Need quick-sample validation on seeded fields.
3. **Domain seeds were the second-worst phantom source (34.1%)** — see analysis above. Universal fields seeded from domain-level discovery often don't match the specific clause type of individual clusters.
4. **Agreement coverage at 28.7%** — limited by cluster coverage. With cluster processing expanding to 10+ chunk clusters, coverage should jump to ~60%+.
5. **Extractions per field median = 2** — most fields are barely extracted, suggesting many clusters have thin agreement overlap.

### Fixes for Next Run (v2)

- [x] Lower `min_cluster_chunks` from 20 → 10
- [x] Validate domain-seeded fields before persisting (quick-sample check)
- [x] Batch small clusters (5-9 chunks) by domain in single RLM sessions

---

## Run 3: Full-v2 (Coverage Fixes + Confidence Gating)

**Date:** April 24, 2026
**Config:** `min_cluster_chunks=10`, `batch_small_clusters=True`, `small_cluster_min=5`, `confidence_gate_threshold=0.55`
**Script:** `scripts/run_full_v2.py`
**PID:** 922087 (completed)

### Headline Metrics

| Metric | Value | vs Full-458 |
|--------|:-----:|:-----------:|
| Documents | 2,889 | — |
| Chunks | 144,464 | — |
| Extraction-targets (≥10 chunks) | 2,904 | 15.4× |
| Fields discovered (total) | 10,878 | 3.6× |
| Active fields (non-phantom) | 4,837 | 3.1× |
| Extractions | 23,483 | 3.2× |
| Mean confidence | 0.852 | +1.6pp |
| ClusterProfiles built | 789 | 4.2× |
| RLM intents discovered | 1,620 | 2.4× |
| Agreement coverage | 58.4% (1686/2889) | **2.0×** |
| Overall phantom rate | 55.5% (6041/10878) | +7.1pp |

### Confidence Gate Performance

| Metric | Value |
|--------|:-----:|
| Gates decided | 649 |
| Passed | 421 (64.8%) |
| Failed | 228 (35.2%) |
| Est. LLM cost saved | ~35% |

### Extraction Confidence Distribution

| Bucket | Count | % |
|:---------:|:-----:|:-:|
| ≥ 0.90 | 12,012 | 51.2% |
| 0.70 – 0.89 | 8,179 | 34.8% |
| 0.50 – 0.69 | 2,862 | 12.2% |
| < 0.50 | 430 | 1.8% |

86% of extractions at ≥0.70 confidence. Mean confidence improved from 0.836 → 0.852.

### Phantom Field Analysis by Source

| Source | Total | Phantom | Phantom % | vs Full-458 |
|--------|:-----:|:-------:|:---------:|:-----------:|
| rag_extractor | 8,962 | 6,026 | **67.2%** | +9.8pp |
| domain_seed | 931 | 15 | 1.6% ✅ | **-32.5pp** |
| agent | 984 | 0 | 0.0% | unchanged |

> **✅ Domain-Seed Fix Validated:** Quick-sample validation on seeded fields dropped domain_seed phantom rate from **34.1% → 1.6%** (-32.5pp). The fix works: before persisting any domain-seeded field into a cluster, the pipeline now tests it against 3 sample documents from that cluster. If 0/3 documents yield a value, the field is pruned as inapplicable to this clause type. This eliminated 120+ phantom fields per run.

**Key finding:** Domain seed phantom rate collapsed (34.1% → 1.6%) thanks to quick-sample validation. RAG extractor phantom rate worsened because it now processes 6× more clusters (many marginal). Agent fields remain perfect at 0%.

### Stage 4b: Batch Small Clusters

| Metric | Value |
|--------|:-----:|
| Small clusters batched (5-9 chunks) | 6,032 |
| Fields from batch stage | 249 |
| Extractions from batch stage | 1,221 |

### LLM Usage

| Metric | Value |
|--------|:-----:|
| Total LLM calls | 99,630 |
| Wall time | 130 min (2.2 hours) |

### Critique

1. **Coverage doubled** (28.7% → 58.4%) — the primary goal was achieved. Lowering min_cluster_chunks from 20 → 10 was the biggest lever.
2. **Phantom rate worsened** (48.4% → 55.5%) — but this is misleading. The rag_extractor now processes marginal clusters that are harder to extract from. Agent and domain_seed sources are both improved or stable.
3. **Domain seed validation worked** — phantom rate dropped from 34.1% to 1.6%, proving quick-sample validation on seeded fields is effective.
4. **RAG retriever is the bottleneck** — 67.2% of rag_extractor fields are phantom. Root cause: cluster-scoped retrieval misses values in adjacent clauses. Fix implemented (agreement-wide scope + ColBERT), not yet run at scale.

### Fixes for Next Run (v3)
- [x] Enable `use_late_interaction=True` (ColBERT MaxSim retrieval)
- [x] Enable `widen_retrieval_scope=True` (agreement-wide + cluster boost)
- [x] Lazy-persist: don't persist rag_extractor field_definitions until ≥1 extraction succeeds

---

## Run 4: Full-v3 (Lazy Persist — Phantom Elimination)

**Date:** April 24, 2026
**Config:** Same as v2, plus `use_late_interaction=True`, `widen_retrieval_scope=True`, lazy-persist in rag_extractor
**Script:** `scripts/run_full_v3.py`
**PID:** 1036744 (completed)

**Bug discovered:** `shared_rag_config` was missing `use_late_interaction` and `widen_retrieval_scope` — ColBERT and wide scope did NOT activate. V3 results measure lazy-persist only. Bug fixed for v4.

### Headline Metrics

| Metric | Value | vs v2 |
|--------|:-----:|:-----:|
| Documents | 2,889 | — |
| Extraction-targets | 2,904 | same |
| Fields in DB | 4,734 | -56% (6,144 fewer phantom rows) |
| Active fields (non-phantom) | 4,730 | -2.2% |
| Extractions | 23,360 | -0.5% |
| Mean confidence | 0.852 | same |
| ClusterProfiles built | 789 | same |
| RLM intents | 1,550 | -4.3% |
| Agreement coverage | 58.2% (1681/2889) | -0.2pp |
| **Phantom rate** | **0.1% (4/4734)** | **-55.4pp** |
| Wall time | 120 min (2.0 hrs) | -8% faster |
| LLM calls | 97,313 | -2.3% |

### Phantom Field Resolution

| Source | v2 Rate | v3 Rate | Change |
|--------|:-------:|:-------:|:------:|
| rag_extractor | 67.2% (6,026/8,962) | **0.0%** (0/2,846) | **eliminated** |
| agent | 0.0% (0/984) | 0.0% (0/997) | unchanged |
| domain_seed | 1.6% (15/931) | 0.5% (4/888) | -1.1pp |
| **Overall** | **55.5%** | **0.1%** | **-55.4pp** |

### Lazy-Persist Field Drop Stats (aggregated from logs)

| Metric | Value |
|--------|------:|
| Fields kept (had extractions) | 2,030 |
| **Fields dropped (total)** | **6,044** |
| — Retrieval miss (no snippets found) | 6,044 |
| — Low confidence (all null) | 0 |
| — No extractions (other) | 0 |

100% of dropped fields were retrieval_miss — the RAG retriever couldn't find any relevant snippets for those fields. These were always phantom; now they're never persisted.

### Extraction Value Quality

| Metric | Value |
|--------|-------|
| High-value extractions (>20 chars, ≥0.70 conf) | 12,609 (54.0%) |
| Source chunk grounding | 19,226 (82.3%) |
| Unique field names | 2,002 |
| Avg extractions per covered agreement | 13.9 (median: 7) |

### Value Length Distribution

| Length | Count | % | Avg Confidence |
|--------|:-----:|:---:|:-:|
| 1-5 chars | 2,781 | 11.9% | 0.903 |
| 6-20 chars | 5,550 | 23.8% | 0.876 |
| 21-100 chars | 6,995 | 29.9% | 0.848 |
| 101-500 chars | 7,499 | 32.1% | 0.821 |
| 500+ chars | 535 | 2.3% | 0.821 |

### Field Source Contribution

| Source | Fields | Extractions | Avg Confidence | Agreements |
|--------|:------:|:-----------:|:-:|:-:|
| agent | 997 | 8,419 | 0.928 | 899 |
| domain_seed | 884 | 7,998 | 0.804 | 1,242 |
| rag_extractor | 2,846 | 6,932 | 0.815 | 1,161 |

### Confidence Distribution

| Bucket | Count | % |
|:-----------:|:-----:|:---:|
| ≥ 0.90 | 11,857 | 50.8% |
| 0.70 – 0.89 | 8,225 | 35.2% |
| 0.50 – 0.69 | 2,891 | 12.4% |
| < 0.50 | 387 | 1.7% |

### Gate Performance

| Metric | Value |
|--------|:-----:|
| Gates decided | 651 |
| Passed | 414 (63.6%) |
| Failed | 237 (36.4%) |

### Critique

1. **Phantom problem solved.** 55.5% → 0.1% with lazy-persist alone. Zero code change to the extraction logic — just don't persist fields that never get values.
2. **Extraction count preserved.** 23,360 vs 23,483 (-0.5%). The 6,044 dropped fields contributed zero extractions by definition — no real value was lost.
3. **All drop reasons are retrieval_miss** — the retriever found no relevant snippets for those 6,044 fields. This confirms the root cause: fields discovered by the agent don't always have extractable content in every agreement's cluster-scoped chunks.
4. **ColBERT + wide scope didn't activate** due to config wiring bug in `shared_rag_config`. The fix is applied — next run will test all three changes together.
5. **Run was 8% faster** (120 vs 130 min) — fewer fields to persist = less DB I/O.

### Fixes for Next Run (v4)

- [x] Verify ColBERT + wide scope activate (config bug fixed in `shared_rag_config`)
- [x] Measure retrieval_miss rate with wide scope (expect significant reduction from 6,044)
- [x] Compare extractions per field between cluster-scoped and agreement-wide retrieval

---

## Run 5: Full-v4 (ColBERT + Wide Scope — Config Fix)

**Date:** April 24, 2026
**Config:** Same as v3, plus `use_late_interaction=True` and `widen_retrieval_scope=True` now confirmed active (config wiring bug fixed in `shared_rag_config`)
**Script:** `scripts/run_full_v4.py`
**PID:** 1091817 (completed)

### Headline Metrics

| Metric | Value | vs v3 |
|--------|:-----:|:-----:|
| Documents | 2,889 | — |
| Extraction-targets | 2,904 | same |
| Fields in DB | 5,497 | +16.1% (+763) |
| Active fields (non-phantom) | 5,493 | +16.1% |
| Extractions | 28,328 | **+21.3%** (+4,968) |
| Mean confidence | 0.837 | -1.5pp |
| ClusterProfiles built | 789 | same |
| RLM intents | 1,532 | -1.2% |
| Agreement coverage | 68.3% (1973/2889) | **+10.1pp** |
| Phantom rate | 0.1% (4/5497) | same (0.1%) |
| Wall time | 164 min (2.7 hrs) | +37% slower |
| LLM calls | 111,729 | +14.8% |

### ColBERT Activation Confirmed

| Evidence | v3 | v4 |
|----------|:--:|:--:|
| `colbert`/`late_interaction` log mentions | 1 | **15,070** |
| Source-grounded extractions | 82.3% | **100.0%** |
| Confidence gate decisions | 651 | 652 |

ColBERT late-interaction retrieval is confirmed active in v4 (15,070 log mentions vs 1 in v3). Every extraction now has a `source_chunk_id` — 100% provenance, up from 82.3% in v3.

### Phantom Field Resolution

| Source | v3 Rate | v4 Rate | Change |
|--------|:-------:|:-------:|:------:|
| rag_extractor | 0.0% (0/2,846) | 0.0% (0/3,099) | stable |
| agent | 0.0% (0/997) | 0.0% (0/984) | stable |
| domain_seed | 0.5% (4/888) | 0.3% (4/1,412) | -0.2pp |
| **Overall** | **0.1%** | **0.1%** | **stable** |

Phantom rate remains at 0.1% — lazy-persist continues to hold. The 4 remaining phantoms are all domain seeds.

### Lazy-Persist Field Drop Stats

| Metric | v3 | v4 | Change |
|--------|---:|---:|-------:|
| Lazy-persist decisions | 826 | 839 | +13 |
| Fields kept | 2,030 | 2,138 | **+5.3%** |
| Fields dropped | 6,044 | 5,963 | **-1.3%** |
| — retrieval_miss | 6,043 | 5,962 | -1.3% |
| Keep rate | 25.1% | 26.4% | **+1.3pp** |

ColBERT + wide scope improved the keep rate from 25.1% → 26.4%. The retriever now finds relevant snippets for 108 additional fields that were retrieval-misses in v3. While the improvement is modest in absolute terms, it translates to +763 active fields and +4,968 extractions.

### Extraction Source Breakdown

| Source | Fields | Extractions | Avg Confidence | Agreements |
|--------|:------:|:-----------:|:-:|:-:|
| domain_seed | 1,408 | 11,607 | 0.799 | 1,661 |
| rag_extractor | 3,099 | 8,581 | 0.816 | 1,318 |
| agent | 984 | 8,110 | 0.915 | 879 |
| repl_output | 2 | 30 | 0.795 | 24 |

**Key shift:** Domain seeds now produce the most extractions (11,607), overtaking agent (8,110). This is because wide-scope retrieval allows domain-seeded fields to find values in agreement-wide chunks that were previously invisible to cluster-scoped retrieval. Domain seed fields went from 7,998 (v3) → 11,607 (v4) extractions, a **45% increase**.

| Source | v3 Extractions | v4 Extractions | Change |
|--------|:-:|:-:|:-:|
| domain_seed | 7,998 | 11,607 | **+45.1%** |
| rag_extractor | 6,932 | 8,581 | **+23.8%** |
| agent | 8,419 | 8,110 | -3.7% |

### Confidence Distribution

| Bucket | v3 Count | v3 % | v4 Count | v4 % | Change |
|:-----------:|:-----:|:---:|:-----:|:---:|:---:|
| ≥ 0.90 | 11,857 | 50.8% | 13,392 | 47.3% | -3.5pp |
| 0.70 – 0.89 | 8,225 | 35.2% | 10,015 | 35.4% | +0.2pp |
| 0.50 – 0.69 | 2,891 | 12.4% | 4,492 | 15.9% | +3.5pp |
| < 0.50 | 387 | 1.7% | 429 | 1.5% | -0.2pp |

The ≥0.90 share dropped from 50.8% → 47.3%. The additional 4,968 extractions come predominantly from harder retrieval targets (wide-scope agreement chunks rather than cluster-scoped chunks), which naturally score lower. Mean confidence shifted from 0.852 → 0.837 (-1.5pp) — a reasonable tradeoff for +21% more extractions and +10pp more agreement coverage.

### Value Length Distribution

| Length | Count | % | Avg Confidence |
|--------|:-----:|:---:|:-:|
| 1-5 chars | 3,221 | 11.4% | 0.886 |
| 6-20 chars | 6,161 | 21.8% | 0.861 |
| 21-100 chars | 7,863 | 27.8% | 0.831 |
| 101-500 chars | 9,900 | 34.9% | 0.815 |
| 500+ chars | 1,183 | 4.2% | 0.808 |

The 500+ chars bucket nearly doubled (535 → 1,183), confirming that wide-scope retrieval recovers longer clause texts that cluster-scoped retrieval truncated or missed.

### Extraction Quality

| Metric | v3 | v4 | Change |
|--------|:---:|:---:|:------:|
| High-value (>20ch, ≥0.70) | 12,609 (54.0%) | 15,142 (53.5%) | +2,533 (+20.1%) |
| Source-grounded | 19,226 (82.3%) | 28,328 (100.0%) | **+17.7pp** |
| Unique field names (active) | 2,002 | 2,241 | +12.0% |
| Active fields per cluster | median 5 | median 5 | same |
| Avg extractions/covered agreement | 13.9 | 14.4 | +3.6% |

### Agreement Coverage Deep-Dive

| Extractions per Agreement | Agreements | Total Extractions |
|:-------------------------:|:----------:|:-----------------:|
| 1–5 | 695 | 2,023 |
| 6–15 | 744 | 7,167 |
| 16–30 | 384 | 8,275 |
| 31+ | 150 | 10,863 |

1,973 agreements now have at least one extraction. The median agreement has 8 extractions (P25=4, P75=16, max=4,105).

### Field Type Distribution (Active Fields)

| Type | Count |
|------|:-----:|
| text | 4,928 |
| boolean | 288 |
| enum | 141 |
| duration | 48 |
| date | 43 |
| currency | 18 |
| integer | 16 |
| percentage | 11 |

### Domain Coverage

117/120 domains have at least one extraction. The 3 uncovered domains are highly specialized categories with minimal cluster overlap.

**Top 5 domains by extractions:**

| Domain | Extractions | Agreements |
|--------|:-----------:|:----------:|
| Severability Clauses | 2,173 | 304 |
| Contract Definitions | 1,461 | 73 |
| Entire Agreement Clauses | 1,319 | 101 |
| Lease Term Provisions | 1,249 | 104 |
| Representations and Warranties | 1,082 | 212 |

### Per-Stage Timing

| Stage | v3 | v4 | Change |
|-------|:--:|:--:|:------:|
| Discovery + extraction | 6,830s (95%) | 9,677s (98%) | +41.7% |
| Domain field discovery | 338s (5%) | 179s (2%) | -47.0% |
| Assigner init | 0.1s | 0.1s | — |
| **Total** | **7,174s (120 min)** | **9,863s (164 min)** | **+37.5%** |

Wall time increased from 120 → 164 min (+37%). Root cause: ColBERT retrieval adds latency per query (MaxSim scoring over wider document scope), and more fields survive lazy-persist, generating more extraction LLM calls. The per-call latency also increased (2.02s → 2.44s), likely due to wider context windows from agreement-level retrieval.

### LLM Usage & Cost Estimate

| Metric | v3 | v4 | Change |
|--------|:--:|:--:|:------:|
| Total LLM calls | 97,313 | 111,729 | +14.8% |
| Total LLM latency | 196,236s | 272,417s | +38.8% |
| Avg latency per call | 2.02s | 2.44s | +20.8% |

**Estimated cost (2K input / 500 output tokens per call):**

| Model | v3 | v4 | Change |
|-------|:--:|:--:|:------:|
| gpt-5.4-mini ($0.30/$1.20) | ~$117 | ~$134 | +$17 |
| gpt-5.4 ($5.00/$15.00) | ~$1,703 | ~$1,955 | +$252 |

### Critique

1. **ColBERT + wide scope delivered.** +21.3% extractions, +10.1pp agreement coverage, 100% source grounding. The config wiring bug was the only thing holding this back — the retrieval architecture was sound from v2.

2. **Agreement coverage broke the two-thirds mark** (58.2% → 68.3%). The remaining 31.7% (916 agreements) likely have insufficient chunk representation in the processed clusters. Further gains require either processing more clusters or cross-cluster extraction.

3. **Phantom rate held at 0.1%.** Lazy-persist continues to work perfectly. Adding ColBERT didn't introduce new phantom modes — the extra fields it enables all produce real extractions.

4. **Confidence traded for coverage.** Mean confidence dropped 0.852 → 0.837 (-1.5pp). This is expected: wide-scope retrieval pulls in more distant chunks that are harder to extract from precisely. The tradeoff is favorable — 4,968 additional extractions at slightly lower confidence is better than not having them.

5. **Domain seeds were the biggest winner.** +45% more domain_seed extractions thanks to wide scope. Domain-level fields like "effective_date" and "party_names" now find values across the full agreement rather than just the cluster's chunks. This validates the two-tier (domain seed + cluster-specific) architecture.

6. **Wall time regression is the main cost.** +44 min (+37%) from ColBERT MaxSim overhead and more surviving fields. If cost/latency becomes critical, we can introduce ColBERT only for fields that fail cluster-scoped BM25 first (fallback strategy), rather than using it for all queries.

7. **Keep rate improvement was modest (25.1% → 26.4%).** Wide scope helped the retriever find snippets for ~108 more fields, but ~74% of proposed fields still fail retrieval. This confirms that the discovery hallucination problem (§9 of FEEDBACK_INTEGRATION_PLAN.md) is the larger lever — fixing the prompts to propose fewer phantom fields will have more impact than further retrieval improvements.

---

## Cross-Run Assessment: Baseline → v4

### Quantitative Evolution

| Metric | Baseline | Full-458 | v2 | v3 | **v4** | Baseline→v4 |
|--------|:--------:|:--------:|:--:|:--:|:------:|:-----------:|
| Clusters processed | 30 | 188 | 2,904 | 2,904 | 2,904 | **97×** |
| Agreement coverage | 13.9% | 28.7% | 58.4% | 58.2% | **68.3%** | **+54.4pp** |
| Extractions | 2,717 | 7,450 | 23,483 | 23,360 | **28,328** | **10.4×** |
| Active fields | 263 | 1,561 | 4,837 | 4,730 | **5,493** | **20.9×** |
| Phantom rate | 47.0% | 48.4% | 55.5% | 0.1% | **0.1%** | **-46.9pp** |
| Confidence (mean) | 0.881 | 0.836 | 0.852 | 0.852 | **0.837** | -4.4pp |
| Source grounding | 79% | ~80% | ~82% | 82.3% | **100%** | **+21pp** |
| LLM calls | ~4K | 33K | 100K | 97K | **112K** | 28× |
| Wall time | 15 min | 76 min | 130 min | 120 min | **164 min** | 11× |

### What Each Run Proved

| Run | Primary Question | Answer |
|-----|-----------------|--------|
| **Baseline** | Does the pipeline work end-to-end? | Yes — 30 clusters, real extractions, 0.881 confidence. But 47% phantom rate and 14% coverage. |
| **Full-458** | Do quality improvements help? | Mixed — domain seed phantoms discovered (34.1%), overall phantom rate unchanged. Coverage doubled but still only 29%. |
| **v2** | Does processing more clusters scale? | Yes — 2,904 clusters, 58% coverage, 3.2× extractions. But phantom rate worsened to 55.5% because marginal clusters are harder. |
| **v3** | Can we eliminate phantom fields? | Yes — lazy-persist dropped phantom rate from 55.5% → 0.1% with zero extraction loss. The simplest fix was the most effective. |
| **v4** | Does better retrieval (ColBERT) improve extraction? | Yes — +21% extractions, +10pp coverage, 100% grounding. Config bug was the only barrier. |

### The Three Levers

Across all five runs, extraction quality was controlled by three levers:

**1. Coverage (which clusters get processed)**
- Baseline → v2: min_cluster_chunks 20→10, uncapped clusters. Coverage 14% → 58%.
- v4: Wide-scope retrieval added another 10pp. Now at 68%.
- **Remaining gap:** 916 agreements (32%) have no extractions. Further coverage requires processing clusters <10 chunks or cross-cluster extraction.

**2. Precision (which fields survive)**
- Baseline–v2: Phantom rate ~47-56%. Fields persisted blindly.
- v3: Lazy-persist eliminated phantoms (0.1%). Zero cost to extraction count.
- **Remaining gap:** 74% of proposed fields are still dropped at the retrieval stage. Discovery hallucination (proposing too many speculative fields) is the primary cost driver.

**3. Retrieval quality (finding the right text)**
- Baseline–v3: BM25 + cluster-scoped. 82% grounding, many retrieval misses.
- v4: ColBERT MaxSim + agreement-wide + cluster boost. 100% grounding, +21% extractions.
- **Remaining gap:** Per-query latency increased 20%. ColBERT is powerful but expensive. A hybrid BM25-first → ColBERT-fallback strategy could reduce cost.

### Confidence vs. Coverage Tradeoff

```
Confidence:  0.881  →  0.836  →  0.852  →  0.852  →  0.837
Coverage:    13.9%  →  28.7%  →  58.4%  →  58.2%  →  68.3%
              ↑         ↑         ↑         ↑         ↑
           Baseline   458       v2        v3        v4
```

Confidence declined modestly from 0.881 → 0.837 (-4.4pp) across all runs while coverage increased 5× (13.9% → 68.3%). This is the expected signature of scaling: easy agreements extract at high confidence, and each coverage increment brings in harder agreements that score slightly lower. The 0.837 average is still strong — 83% of extractions are ≥0.70.

### Cost Efficiency

| Run | Extractions | LLM Calls | Extractions/Call | Est. Cost (mini) | Cost/Extraction |
|-----|:-----------:|:---------:|:----------------:|:----------------:|:---------------:|
| Baseline | 2,717 | ~4K | 0.68 | ~$5 | $0.002 |
| Full-458 | 7,450 | 33K | 0.22 | ~$40 | $0.005 |
| v2 | 23,483 | 100K | 0.24 | ~$120 | $0.005 |
| v3 | 23,360 | 97K | 0.24 | ~$117 | $0.005 |
| **v4** | **28,328** | **112K** | **0.25** | **~$134** | **$0.005** |

Cost per extraction has stabilized at ~$0.005 (gpt-5.4-mini). The extraction/call ratio improved slightly in v4 (0.25 vs 0.24) because ColBERT's better retrieval means fewer wasted extraction attempts. The baseline was artificially efficient (0.68) because it only processed the easiest 30 clusters.

### Extraction Quality — LLM Judge Evaluation

To measure actual extraction correctness (not just confidence scores), we ran a **stratified LLM judge evaluation** on 300 random extractions (100 per source) using gpt-4o-mini as an independent judge. The judge evaluates each extraction against its source chunk on three dimensions: faithfulness, relevance, and completeness.

#### Overall Results

| Dimension | Score | Interpretation |
|-----------|:-----:|----------------|
| **Precision (overall correct)** | **70.3%** | 7 in 10 extractions are fully correct |
| Faithfulness | 79.7% | Value is present in or supported by source text |
| Relevance | 97.3% | Value answers what the field description asks |
| Completeness | 75.7% | Value is not truncated or partial |

#### By Source

| Source | Correct | Faithful | Relevant | Complete | n |
|--------|:-------:|:--------:|:--------:|:--------:|:-:|
| **agent** | **86.0%** | 91.0% | 100.0% | 89.0% | 100 |
| rag_extractor | 72.0% | 83.0% | 99.0% | 73.0% | 100 |
| domain_seed | 53.0% | 65.0% | 93.0% | 65.0% | 100 |

Agent-sourced extractions are the most reliable (86% correct). Domain seeds are the weakest (53% correct) — they're broad fields applied to clusters where they may not fit well, leading to faithfulness issues (35% are hallucinated or unsupported).

#### By Confidence Bucket

| Confidence | Correct | Faithful | Relevant | n |
|:----------:|:-------:|:--------:|:--------:|:-:|
| ≥ 0.90 | 74.1% | 83.7% | 98.5% | 135 |
| 0.70–0.89 | 65.4% | 74.0% | 96.2% | 104 |
| 0.50–0.69 | 72.4% | 82.8% | 98.3% | 58 |
| < 0.50 | 33.3% | 33.3% | 66.7% | 3 |

Confidence is moderately calibrated — higher confidence extractions are somewhat more likely to be correct, but the 0.50–0.69 bucket (72.4%) outperforms 0.70–0.89 (65.4%), suggesting the confidence signal has noise. The self-reported grounding boost (+0.1 for substring match) may over-inflate scores for partial matches.

#### Failure Analysis (89 incorrect extractions)

| Failure Mode | Count | % of Failures |
|:------------|:-----:|:-------------:|
| Incomplete (truncated) | 73 | 82% |
| Unfaithful (hallucinated) | 61 | 69% |
| Irrelevant (wrong field) | 8 | 9% |

> **Key finding:** The dominant failure mode is **incomplete extraction** (82% of failures), not hallucination. Many extractions capture the right concept but truncate the clause text — likely due to chunk boundaries or max-token limits in the extraction prompt. The second mode is **unfaithfulness** (69%), where the extracted value doesn't appear in the source chunk — this overlaps with incompleteness (a truncated value can appear unfaithful).
>
> Relevance is excellent (97.3%) — the pipeline almost always extracts from the correct field. The problem is at the value level, not the field level.

---

### Clause Intent Quality — LLM Judge Evaluation

The pipeline discovers **clause intents** (what a clause is trying to accomplish) and then extracts **intent values** (the specific text expressing that intent). These are evaluated separately.

#### Clause Intent Definitions (150 samples)

The clause intent definitions (intent_label + summary + party identification) produced by the RLM discovery stage:

| Dimension | Score |
|-----------|:-----:|
| **Overall correct** | **86.0%** |
| Label accurate | 89.3% |
| Summary quality | 88.7% |
| Parties correct | 98.0% |
| Specificity | 93.3% |

Clause intent definitions are significantly higher quality than field extractions (86.0% vs 70.3%). Party identification is near-perfect (98.0%). The main failure modes are label inaccuracy (11%) — the intent label oversimplifies or mischaracterizes the clause — and summary errors (11%) where the summary misrepresents specific clause details.

**Sample failures:**
- `use_of_proceeds_restriction` applied to a clause that *allows* general corporate purposes (label suggests restriction where none exists)
- `define_term` applied to a clause defining multiple terms, not just the one named
- `service_modification_right` applied to a Windows app terms clause (wrong scope)

#### Intent Extractions (150 samples)

The intent extraction values — the actual clause text extracted to represent each intent:

| Dimension | Score |
|-----------|:-----:|
| **Overall correct** | **71.3%** |
| Faithfulness | 91.3% |
| Relevance | 99.3% |
| Completeness | 71.3% |
| Specificity | 98.0% |

**By confidence:**

| Confidence | Correct | n |
|:----------:|:-------:|:-:|
| ≥ 0.90 | 80.3% | 66 |
| 0.70–0.89 | 64.6% | 65 |
| 0.50–0.69 | 63.2% | 19 |

Intent extractions are highly faithful (91.3%) and relevant (99.3%), but **completeness is the limiting factor** — 100% of failures involve truncated/incomplete values. Intent extractions tend to be longer texts (78% are >50 chars, 53% are >200 chars) and frequently hit chunk boundaries, causing mid-sentence truncation. Only 30% of failures are unfaithful; the vast majority capture the right text but cut it short.

**Failure analysis (43 incorrect):**

| Failure Mode | Count | % of Failures |
|:------------|:-----:|:-------------:|
| Incomplete | 43 | 100% |
| Unfaithful | 13 | 30% |
| Non-specific | 3 | 7% |
| Irrelevant | 1 | 2% |

---

### Domain-Seed Impact Analysis

#### Quality: Domain-Seed vs Non-Seed (200 samples, 100 each)

| Source | Correct | Faithful | Relevant | Complete | n |
|--------|:-------:|:--------:|:--------:|:--------:|:-:|
| **Non-domain-seed** (rag_extractor + agent) | **73.0%** | 85.0% | 97.0% | 77.0% | 100 |
| **Domain-seed** | **58.0%** | 67.0% | 91.0% | 70.0% | 100 |
| **Gap** | **-15.0pp** | -18.0pp | -6.0pp | -7.0pp | |

Domain seeds are **15 percentage points worse** than non-seed extractions on overall correctness. The primary gap is faithfulness (-18pp): domain seed fields are broad definitions (e.g., "effective_date", "governing_law") applied across many clusters, and the wide-scope retriever finds text fragments that *mention* the concept but don't contain the actual extractable value.

#### Failure Breakdown

| Failure Mode | Domain-Seed (42 failures) | Non-Seed (27 failures) |
|:------------|:-------------------------:|:----------------------:|
| Unfaithful (hallucinated) | 33 (79%) | 15 (56%) |
| Incomplete (truncated) | 30 (71%) | 23 (85%) |

Domain-seed failures are dominated by **unfaithfulness** (79%) — the retrieved chunk mentions the concept tangentially but the extraction doesn't reflect what's actually in the text. Non-seed failures are dominated by **incompleteness** (85%) — the right text was found but truncated.

#### Domain-Seed Confidence Calibration

| Confidence | Correct | n |
|:----------:|:-------:|:-:|
| ≥ 0.90 | 62.5% | 40 |
| 0.70–0.89 | 69.2% | 39 |
| 0.50–0.69 | 28.6% | 21 |

Domain-seed confidence is **poorly calibrated**. High-confidence (≥0.90) extractions are only 62.5% correct — the grounding boost (+0.1 for substring match) over-inflates confidence for partial matches. The 0.50–0.69 bucket contains the weakest extractions (28.6% correct) — these should likely be filtered.

#### Volume vs Quality Tradeoff

| Metric | Domain-Seed | Non-Seed | Note |
|--------|:-----------:|:--------:|------|
| Extractions | 11,607 | 16,721 | 41% of total |
| Unique fields | 1,408 | 4,083 | 26% of total |
| Agreements covered | 1,661 | 1,610 | DS covers 57% of all covered agreements |
| Correctness (judge) | 58.0% | 73.0% | -15pp gap |
| Est. correct extractions | ~6,732 | ~12,206 | |
| Est. incorrect | ~4,875 | ~4,515 | DS adds similar # wrong extractions |

Domain seeds add ~6,732 correct extractions (valuable) but also ~4,875 incorrect ones. Non-seed sources add more correct extractions (12,206) with fewer incorrect ones (4,515). Domain seeds contribute **breadth** (cover 1,661 agreements) at the cost of per-extraction quality.

#### Field Name Overlap

| Overlap | Count |
|---------|:-----:|
| Domain-seed unique field names | 478 |
| RAG-extractor unique field names | 2,184 |
| Agent unique field names | 855 |
| DS ∩ RAG overlap | 421 (88% of DS names appear in RAG too) |
| DS ∩ Agent overlap | 41 |

88% of domain-seed field names also appear as rag_extractor fields. This means domain seeds are largely **redundant with per-cluster discovery** — they seed the same fields that the RAG extractor would discover independently. The value of domain seeds is that they ensure coverage when per-cluster discovery fails, but the quality cost is significant.

#### Cluster Quality: With vs Without Domain Seeds

| Status | Clusters | Avg Extractions | Avg Confidence | Total Extractions |
|--------|:--------:|:---------------:|:--------------:|:-----------------:|
| With domain seeds | 540 | 35.5 | 0.788 | 19,181 |
| Without domain seeds | 219 | 41.8 | 0.833 | 9,147 |

Clusters without domain seeds have **higher average confidence** (+4.5pp) and **more extractions per cluster** (+6.3). This confirms that domain seeds dilute quality — they add low-confidence extractions that drag down the cluster average. Clusters that succeed on purely per-cluster discovery (agent + rag_extractor) are inherently higher quality.

---

### Coverage Funnel Analysis

The 68.3% agreement coverage (1,973/2,889) results from a 4-level funnel where each level loses agreements:

```
Level 1: Chunk Assignment
  144,464 chunks → 82,543 assigned to clusters (57.1%)
  61,921 chunks (42.9%) are outliers — never assigned to any cluster

Level 2: Cluster Field Survival
  8,936 clusters → only 760 produce surviving field definitions (8.5%)
  ~2,904 clusters attempted (≥10 chunks), 2,144 had ALL fields dropped
  • 10–19 chunk clusters: 18.9% survival (463/2,446)
  • 20+ chunk clusters: 39.7% survival (182/458)

Level 3: Agreement Reachability
  2,889 agreements → 2,110 have chunks in the 760 productive clusters (73%)
  779 agreements only have chunks in unproductive clusters

Level 4: Extraction Match
  2,110 reachable → 1,973 get at least one extraction (93.5%)
  138 reachable agreements had <5% chunk overlap — too sparse to extract from
```

The single biggest bottleneck is **Level 2**: only 8.5% of clusters produce surviving fields. The pipeline processes all 2,904 clusters (≥10 chunks), but in 74% of them, every proposed field is a retrieval miss — the LLM proposes fields for concepts mentioned in the text that don't have extractable structured values.

**Uncovered agreements (917)** break down as:
- **779** only have chunks in unproductive clusters (zero surviving fields)
- **138** have chunks in productive clusters but <5% overlap — the agreement contributes 1–2 chunks to a cluster but isn't the cluster's primary agreement set

---

### Domain Seeding Clarification

Domain seeding (Stage 3.5) has been **active in every run** — it was never turned off. There is no toggle to disable it; `min_domain_chunks=50` is the only implicit control (domains with <50 total chunks are skipped). Stage 3.5 runs unconditionally even when `resume_from_stage=4` because it sits outside the stage-3 resume guard.

| Run | Domain Fields Discovered | Domain Seed Extractions | Phantom Rate |
|-----|:------------------------:|:-----------------------:|:------------:|
| Full-458 | 1,111 | — | 34.1% |
| v2 | 1,112 | — | 1.6% |
| v3 | 1,111 | 7,998 | 0.5% |
| **v4** | **1,107** | **11,607** | **0.3%** |

The slight variation (1,107–1,112) is LLM non-determinism. Quick-sample validation (added in v2) dropped domain_seed phantoms from 34.1% → 1.6%, and lazy-persist (v3) brought them to 0.3%.

**What changed in v4:** ColBERT + wide-scope retrieval was the game-changer for domain seeds. Domain-level fields like "effective_date" and "governing_law" are broad — they apply across many clause types but the value might not appear in the cluster's specific chunks. With agreement-wide retrieval, the extractor can now search the entire agreement instead of just cluster chunks. This drove domain_seed extractions from 7,998 → 11,607 (**+45%**), making domain seeds the single largest extraction source in v4.

**The quality tradeoff:** Domain seeds have the lowest judge-correctness score (53.0% vs 86.0% for agent, 72.0% for rag_extractor). Their broad applicability means they're often applied to clusters where the field exists conceptually but the extracted value is incomplete or poorly matched. The faithfulness gap (65.0%) confirms that wider retrieval recovers more values but at lower fidelity — the retriever finds something that looks like "governing_law" in a distant chunk, but it may be a tangential reference rather than the actual governing law clause.

---
#### Quality Summary Table

| Component | Precision | Faithfulness | Relevance | Completeness | Primary Issue |
|-----------|:---------:|:------------:|:---------:|:------------:|---------------|
| Field extractions (all) | 70.3% | 79.7% | 97.3% | 75.7% | Truncation + DS faithfulness |
| — Agent source | 86.0% | 91.0% | 100.0% | 89.0% | Minor truncation |
| — RAG extractor | 72.0% | 83.0% | 99.0% | 73.0% | Truncation |
| — Domain seed | 58.0% | 67.0% | 91.0% | 70.0% | **Faithfulness** |
| Intent extractions | 71.3% | 91.3% | 99.3% | 71.3% | **Truncation** |
| Clause intent defs | 86.0% | — | 89.3% (label) | — | Label mischaracterization |

---

## Run 6: Full-v5 (Quality + Schema Fixes)

**Date:** April 24, 2026
**Config:** Same infrastructure as v4 (ColBERT + wide scope), all quality and schema fixes applied
**Script:** `scripts/run_full_v5.py`
**Status:** *Running*

### V4 → V5 Changes Summary

This run applies 5 categories of improvements focused on extraction quality, schema completeness, and data pipeline integrity. No infrastructure parameters (cluster count, retrieval strategy, concurrency) changed — this is a pure quality run.

#### 1. RLM Source Quality — Grounding & Precision Rules

All 4 deposit functions (`deposit_fields`, `deposit_clause_types`, `deposit_intents`, `deposit_obligations`) received:

- **Universal precision rules**: "Only deposit what you can PROVE from the text" added to every DSPy signature docstring. Explicit instruction to prefer fewer high-quality deposits over many speculative ones.
- **Grounding rules**: Each deposit requires `span_text` with a direct quote. Penalty system: grounding score < 0.5 triggers critique, < 0.3 forces rejection.
- **Few-shot examples**: 3 GOOD + 2 BAD examples per deposit function, drawn from domain-diverse contracts (real estate, IP, employment, SaaS, construction) to prevent overfitting.
- **Positive reinforcement**: Examples include notes like "GOOD: high-value, well-grounded extraction" to reward quality.
- **Runtime negative reinforcement**: Deposits with empty span_text get confidence capped at 0.3; deposits with span_text < 10 chars get -0.2 penalty.

**Expected impact**: Fewer hallucinated fields at the source → fewer phantom fields → higher precision. Agent-source correctness should improve from 86% → 90%+.

#### 2. Context Window & RAG Improvements (5 Judge-Identified Fixes)

- **Fix #1 — Context window expansion**: RAG extractor chunk window ±500 → ±1000 chars, max_tokens 500 → 1000. Addresses the #1 failure mode (82% of failures = truncation).
- **Fix #2 — Domain-seed confidence floor**: Domain-seed extractions with confidence < 0.55 are dropped. Addresses the 28.6% correctness in the 0.50–0.69 DS bucket.
- **Fix #3 — Grounding boost calibration**: Reduced from +0.10 → +0.05 for substring match. Addresses the over-inflated confidence in domain seeds (62.5% correct at ≥0.90).
- **Fix #4 — Intent `clause.full_text`**: Completeness prompt now explicitly requests full clause text, not just the first sentence.

**Expected impact**: Completeness should improve 75.7% → 82%+. Domain-seed correctness should improve 53% → 60%+ from confidence floor filtering.

#### 3. Derived/Composite Field Support (Multi-Evidence)

All entity models now support composite/derived fields:

- **FieldV3**: Added `span_texts: list[str]` (multiple source quotes) and `derivation: str` (explains how field was inferred from 2+ sources).
- **ClauseTypeV3**: Added `span_texts: list[str]` for multi-clause evidence.
- **IntentV3**: Added `clause_refs: list[str]` and `chunk_ids: list[int]` for cross-clause intent linking.
- **ObligationV3**: Added `clause_refs: list[str]` for multi-clause obligation evidence.

Grounding logic updated: a field/clause is considered grounded if it has EITHER a single `span_text` OR multiple `span_texts`. Critique scoring counts multi-span fields.

**Expected impact**: Better coverage for complex fields (e.g., "net payment = gross - deductions" derived from 2 clauses). Marginal coverage improvement (+1-2pp).

#### 4. Structural/Schema Gaps Fixed

- **Confidence on all entities**: Added `confidence: float` to `ClauseTypeV3` and `ObligationV3` (fields and intents already had it). All entities now have consistent confidence scoring.
- **Dedup for intents**: New `_dedup_intents()` in phase 2 postprocess — deduplicates by normalized label, keeps the intent with the longer summary.
- **Dedup for obligations**: New `_dedup_obligations()` in phase 2 postprocess — deduplicates by normalized (obligor, action[:60]) key, keeps longer action text.
- **DSPy signature fixes**: Both `V3DiscoverySig` and `V3DiscoverExtractSig` now list all 11 available tools (was missing `critique_fields()` and `budget()`). Strategy instructions rewritten with proper ordering (deposit → critique → explore → budget). Output field description fixed from `{name, type, description, examples}` → `{name, value_type, description, span_text, confidence}`.

**Expected impact**: Fewer duplicate intents/obligations → cleaner data. Agents can now use critique and budget tools → better resource allocation → higher quality per LLM call.

#### 5. Data Pipeline — End-to-End Persistence

Previously, models collected rich metadata (span_texts, derivation, confidence, clause_refs, chunk_ids) but the pipeline discarded it before DB persistence. Now everything flows through:

- **`field_definitions` table**: New columns — `confidence`, `span_text`, `span_texts` (JSON), `derivation`, `iteration`.
- **`clause_intents` table**: New columns — `clause_ref`, `clause_refs` (JSON), `chunk_idx`, `chunk_ids` (JSON), `confidence`, `source`.
- **NEW `clause_type_definitions` table**: Persists RLM-discovered clause sub-types with full metadata.
- **NEW `obligation_definitions` table**: Persists RLM-discovered obligations with full metadata.
- **`_normalize_field()`**: Now passes through all metadata (previously discarded).
- **`_to_field_dicts()`**: Rewritten to pass ALL metadata from Pydantic models to output dicts.

**Expected impact**: No direct quality change, but enables downstream analysis and future feedback loops.

### Expected Metrics — V5 vs V4 Baselines

| Metric | V4 Baseline | V5 Expected | Rationale |
|--------|:-----------:|:-----------:|-----------|
| **Overall precision** | 70.3% | **75–78%** | Grounding rules + fewer hallucinated fields at source |
| **Agent correctness** | 86.0% | **89–92%** | Precision rules + critique tool + grounding penalties |
| **RAG correctness** | 72.0% | **76–80%** | Context window ±1000 fixes truncation |
| **Domain-seed correctness** | 53.0% | **58–63%** | Confidence floor (0.55) drops worst extractions |
| **Faithfulness** | 79.7% | **83–86%** | Grounding rules require span_text proof |
| **Completeness** | 75.7% | **81–85%** | Context window expansion + full_text prompt |
| **Phantom rate** | 0.1% | **≤0.1%** | Should hold — grounding rules may further reduce |
| **Agreement coverage** | 68.3% | **67–70%** | Slight risk from stricter filters; upside from multi-span |
| **Extractions** | 28,328 | **25K–29K** | Confidence floor drops weak DS; better grounding may recover elsewhere |
| **Intent defs correct** | 86.0% | **88–91%** | Dedup removes noisy duplicates |
| **Intent extractions** | 71.3% | **76–80%** | Full-text prompt + context window expansion |
| **Clause type defs** | (not persisted) | **NEW** | Now tracked in DB |
| **Obligation defs** | (not persisted) | **NEW** | Now tracked in DB |
| **Mean confidence** | 0.837 | **0.845–0.860** | Grounding boost calibration + confidence floor |
| **Wall time** | 164 min | **160–180 min** | Similar — no infra changes |

### Key Hypotheses to Validate

1. **Grounding rules reduce hallucination**: If agent faithfulness improves 91% → 95%+, grounding rules work. If flat, LLM was already doing its best.
2. **Context window expansion fixes truncation**: If completeness improves 75.7% → 82%+, the problem was config. If flat, it's structural (chunk boundaries).
3. **Domain-seed confidence floor trades volume for quality**: If DS correctness 53% → 60%+ but count drops 11,607 → ~9,000, the floor works.
4. **Dedup cleans intents/obligations**: V4 had 1,532 intents. If V5 has ~1,200–1,400 with higher judge scores, dedup removed noise.
5. **Coverage holds**: If coverage drops below 66%, the stricter filters are too aggressive.
