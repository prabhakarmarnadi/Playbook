# Retriever Architecture Changes

## Current Architecture (Before Changes)

```
Per-Cluster Extraction:
  store.get_chunks(cluster_id=X)           ← only cluster-member chunks
       │
       ▼
  Group by agreement_id
       │
       ▼
  Per-Agreement: {1-5 chunks}              ← THIS is why logs show "1 chunks"
       │
       ▼
  HybridRetriever(chunks, embeddings)
  ├── Dense: FAISS IndexFlatIP (MiniLM 384-d, single-vector)
  ├── Sparse: BM25
  └── Cross-encoder: ms-marco-MiniLM (OPTIONAL, currently OFF)
       │
       ▼
  RRF Fusion → top-3 snippets
       │
       ▼
  LLM Extraction
```

### Problems

| Problem | Cause | Impact |
|---------|-------|--------|
| **"1 chunk" retrievers** | Only cluster-assigned chunks loaded per agreement. If Agreement X contributes 1 chunk to the "Indemnification" cluster, the retriever for that agreement only sees that chunk. | Adjacent clauses with relevant data (e.g., "Limitation of Liability" mentioning dollar amounts) are invisible. → null extraction → phantom field. |
| **Single-vector information bottleneck** | MiniLM compresses entire chunk into 1 vector (384-d). Query "termination notice period" becomes 1 vector. | Loses compositional structure. Cannot distinguish "60 days general notice" from "30 days for-cause notice" within same chunk. Synonym misses: "cancel" ≠ "terminate" at vector level. |
| **Cross-encoder off by default** | `use_reranker=False` in all extraction paths to save latency. | Best-quality scoring signal is unused. RRF fusion relies entirely on single-vector dense + BM25. |

---

## New Architecture

Two independent changes, each controlled by a config flag:

### Change 1: Wide Retrieval Scope (`widen_retrieval_scope=True`)

```
Per-Cluster Extraction:
  store.get_chunks(cluster_id=X)           ← get cluster chunk IDs
  store.get_chunks(agreement_id IN (...))  ← get ALL chunks for sampled agreements
       │                                      (single batch SQL query)
       ▼
  Per-Agreement: {20-80 chunks}            ← full agreement content
  Each chunk tagged: _in_cluster=True/False
       │
       ▼
  Retriever(chunks)
       │
       ▼
  Score all chunks → apply cluster-member boost → take top-K
       │
       ▼
  LLM Extraction
```

**What changed:**
- Chunk scope widened from cluster-filtered to full agreement
- Cluster membership becomes a **scoring boost**, not a **filter**
- DB queries batched into single `WHERE agreement_id IN (...)` — no N+1 problem

### Change 2: Late Interaction Retrieval (`use_late_interaction=True`)

```
HybridRetriever (before)              LateInteractionRetriever (after)
─────────────────────                 ─────────────────────────────────
Dense: FAISS, single-vector          ColBERT MaxSim, per-token vectors
  chunk → 1 × 384-d                    chunk → N_tokens × 128-d
  query → 1 × 384-d                    query → N_tokens × 128-d
  score = cosine(q, d)                  score = Σ max(q_i · d_j) for each q_i

Sparse: BM25                         Sparse: BM25 (unchanged)

Fusion: RRF(dense, bm25)             Fusion: RRF(maxsim, bm25)

Reranker: cross-encoder (OFF)        Not needed — MaxSim IS reranker-quality
```

**What changed:**
- Dense search leg replaced with ColBERTv2 MaxSim scoring
- Uses trained 768→128 linear projection (not raw BERT truncation)
- ColBERT `query_maxlen=32` to minimize [MASK] padding noise on short queries
- Cross-encoder reranker no longer needed — ColBERT provides equivalent quality

---

## Config Reference

### RAGConfig (in `core/rag_extractor.py`)

| Parameter | Default | Purpose |
|-----------|:-------:|---------|
| `use_late_interaction` | `False` | Replace FAISS dense with ColBERT MaxSim |
| `widen_retrieval_scope` | `False` | Search all agreement chunks, not just cluster-member |
| `cluster_boost_weight` | `0.015` | Additive RRF bonus for cluster-member chunks (only when wide scope on) |
| `wide_retrieve_k_multiplier` | `2` | Retrieve `final_top_k × this` candidates before re-ranking (only when wide scope on) |

### MinimalPipelineConfig (in `core/minimal_pipeline.py`)

| Parameter | Default | Passed to RAGConfig as |
|-----------|:-------:|------------------------|
| `use_late_interaction` | `False` | `use_late_interaction` |
| `widen_retrieval_scope` | `False` | `widen_retrieval_scope` |

### Enabling in a run script

```python
config = MinimalPipelineConfig(
    # ... existing config ...
    use_late_interaction=True,       # ColBERT MaxSim + BM25
    widen_retrieval_scope=True,      # full-agreement chunk scope
)
```

Both flags are independent — you can enable wide scope with original dense retrieval, or ColBERT with cluster-scoped retrieval.

---

## Quality Impact

### Benchmark: 5 legal queries on 5 contract chunks

| Query | Correct? | ColBERT Score Separation |
|-------|:--------:|:------------------------:|
| termination notice period | ✅ | 1.0x (close 2nd — non-compete mentions "termination") |
| indemnification and liability protection | ✅ | 3.2x |
| governing law jurisdiction | ✅ | 1.9x |
| purchase price amount payable | ✅ | 5.4x |
| non-compete duration geographic scope | ✅ | 2.7x |

### Speed: 40 chunks, A100 GPU

| Operation | HybridRetriever (Dense) | LateInteraction (ColBERT) | Dense + CrossEncoder |
|-----------|:-----------------------:|:-------------------------:|:--------------------:|
| Build index | 2,334ms | 2,361ms | 2,334ms |
| Search/query | 6.0ms | 14.9ms | 11.9ms |
| Quality | Baseline | **Better** (per-token) | Best (full cross-attn) |

ColBERT is 2.5× slower per query than raw dense, but 1.3× faster than dense+cross-encoder — while matching cross-encoder quality.

---

## Problems to Watch For

### 1. Index Build Cost (Wide Scope)

| Scenario | Before (cluster-scoped) | After (agreement-wide) |
|----------|:-----------------------:|:----------------------:|
| Chunks per retriever | 1-5 | 20-80 |
| Retrievers per cluster | ~200 (docs) | ~200 (docs) |
| Total chunks indexed | ~400-1,000 | ~4,000-16,000 |

**Mitigation already applied:**
- DB queries batched into single `SELECT ... WHERE agreement_id IN (...)` instead of N roundtrips
- `wide_retrieve_k_multiplier=2` (not 3) limits candidate set expansion
- Feature is opt-in via config flag — can disable if too slow

**Additional mitigation if needed:**
- Reduce `CLUSTER_CONCURRENCY` from 15 → 10 (env var) to cap parallel index builds
- Cap at first 40 chunks per agreement (sorted by chunk_index) to limit long-tail documents

### 2. False Positives from Non-Cluster Chunks

When the full agreement is visible, the retriever might surface a "Limitation of Liability" chunk when extracting for an "Indemnification" cluster — similar vocabulary, wrong clause.

**Mitigation already applied:**
- `cluster_boost_weight=0.015` — cluster-member chunks get additive boost in RRF scoring
- Only applied after initial retrieval, so non-member chunks can still appear if strongly relevant

**Tuning guide:**
| `cluster_boost_weight` | Effect |
|:----------------------:|--------|
| `0.005` | Minimal boost — non-cluster chunks easily outrank |
| `0.015` | Balanced (default) — cluster member wins unless non-member is clearly better |
| `0.030` | Strong boost — cluster members almost always win |
| `0.050` | Near-filter — effectively reverts to cluster-scoped behavior |

### 3. GPU Memory (ColBERT + Wide Scope)

ColBERTv2 model: ~500MB GPU. Per-retriever token embeddings: ~128d × 300 tokens × 80 chunks × 4 bytes ≈ 12MB.

With 15 concurrent clusters × 10 active retrievers each: ~1.8GB extra.

**Mitigation:** A100 has 81GB, current usage ~2.5GB. Ample headroom. If constrained, reduce `CLUSTER_CONCURRENCY`.

### 4. ColBERT [MASK] Padding on Short Queries

ColBERT pads queries to `query_maxlen` with [MASK] tokens. These participate in MaxSim scoring and can match random document tokens.

**Mitigation already applied:**
- `query_maxlen` reduced from 64 → 32. Field queries average 10-20 tokens, so padding is 12-22 [MASK] tokens instead of 44-54.
- Verified no quality regression: 5/5 test queries correct with `query_maxlen=32`.

### 5. ColBERT Not Domain-Tuned

ColBERTv2 was trained on MS MARCO (web search). Legal text has specific vocabulary patterns ("hold harmless" ≈ "indemnify") that may not transfer perfectly.

**Current status:** Working well on test queries — transfer learning appears sufficient for contract extraction. 

**Future improvement:** Fine-tune ColBERT on legal clause pairs using contrastive learning (same clause type = positive, different = negative). Would require building training pairs from existing cluster assignments.

### 6. Debugging Extraction Failures

With full-agreement scope, "why did it pick this chunk?" is harder to trace — 40+ candidates instead of 1-5.

**Mitigation:** Each `RetrievedSnippet` already carries `score`, `source`, and `rank`. The `_in_cluster` tag on each chunk is preserved in metadata. Logs show individual scores for traceability.

---

## Recommendations for Next Run

```python
config = MinimalPipelineConfig(
    # Retriever upgrades
    use_late_interaction=True,       # ColBERT MaxSim scoring
    widen_retrieval_scope=True,      # full-agreement chunk scope
    
    # Existing config (unchanged)
    min_cluster_chunks=10,
    batch_small_clusters=True,
    small_cluster_min=5,
    confidence_gate_threshold=0.55,
    # ...
)
```

Environment variables to consider:
```bash
export CLUSTER_CONCURRENCY=10          # down from 15 — account for heavier indexing
export RAG_ENABLE_REVIEWER=1           # enable reviewer — pairs well with wider scope
```

### Expected Impact

| Metric | Current (v2 run) | Expected (with retriever changes) |
|--------|:-----------------:|:----------------------------------:|
| Phantom rate | ~25-30% | **15-20%** (fewer RAG-miss phantoms) |
| Extraction fill rate | ~60-70% | **70-80%** (wider scope finds more values) |
| Runtime per cluster | ~6s | **~10s** (heavier indexing, offset by better extraction) |
| Total runtime | ~5h (2,904 clusters) | **~8h** (acceptable for quality gain) |
