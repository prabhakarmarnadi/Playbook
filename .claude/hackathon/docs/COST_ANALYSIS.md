# Cost Analysis: CLM Contract Intelligence Pipeline

**Date:** April 2026  
**Model:** GPT-5.4-mini (Azure OpenAI, East US 2)  
**Infrastructure:** 2× NVIDIA A100 80GB PCIe, Azure VM  
**Baseline:** 96 documents → 3,335 clauses → 1,871 clustered → 3,034 chunks  

All prices in USD. Token prices per 1M tokens (Azure OpenAI Global Standard).

---

## 1. LLM Compute Costs

### 1.1 Pricing (per 1M tokens)

| Model | Input | Cached Input | Output | Batch (50% off) |
|-------|------:|-------------:|-------:|-----------------:|
| GPT-5.4-mini¹ | $0.25 | $0.03 | $2.00 | $0.125 / $1.00 |
| GPT-5-mini (reference) | $0.25 | $0.03 | $2.00 | $0.125 / $1.00 |
| GPT-5 (full, reference) | $1.25 | $0.13 | $10.00 | $0.63 / $5.00 |
| GPT-5.1 (reference) | $1.25 | $0.13 | $10.00 | N/A |

¹ *GPT-5.4-mini pricing estimated from GPT-5-mini tier (same class).*

### 1.2 Token Budget Per Pipeline Stage

| Stage | LLM Calls | Avg Input Tokens/Call | Avg Output Tokens/Call | Notes |
|-------|----------:|----------------------:|-----------------------:|-------|
| **Field Discovery** | ~79 (1 per cluster) | ~800 | ~200 | Schema discovery per clause type |
| **Extraction** | ~1,871 (1 per clause) | ~500 | ~150 | Field value extraction |
| **Intent Extraction** | ~1,845 (1 per clause) | ~360² | ~250 | Dynamic intent discovery |
| **Knowledge Graph** | 0 | 0 | 0 | Pure compute, no LLM |
| **Embedding** | 0 | 0 | 0 | Local Nomic on GPU |

² *360 = 289 avg clause tokens + 150 prompt template tokens (capped at 3000 chars).*

### 1.3 Cost Per Run (96 Documents)

| Stage | Input Tokens | Output Tokens | Cost (GPT-5.4-mini) | Cost (GPT-5 full) |
|-------|-------------:|--------------:|---------------------:|-------------------:|
| Field Discovery | 63K | 16K | $0.05 | $0.24 |
| Extraction | 936K | 281K | $0.80 | $1.73 |
| Intent Extraction | 660K | 461K | $1.09 | $5.44 |
| **Total** | **1.66M** | **758K** | **$1.93** | **$7.41** |

**Cost per document: $0.020 (mini) / $0.077 (full)**

### 1.4 Scaling Projections

| Documents | Clauses (est.) | Input Tokens | Output Tokens | GPT-5.4-mini | GPT-5 full | Batch API (mini) |
|----------:|---------------:|-------------:|--------------:|-------------:|-----------:|-----------------:|
| 100 | 1,921 | 1.73M | 790K | $2.01 | $7.72 | $1.01 |
| 500 | 9,609 | 8.65M | 3.95M | $10.06 | $38.61 | $5.03 |
| 1,000 | 19,218 | 17.30M | 7.90M | $20.13 | $77.22 | $10.06 |
| 5,000 | 96,093 | 86.50M | 39.50M | $100.63 | $386.10 | $50.31 |
| 10,000 | 192,187 | 173.00M | 79.00M | $201.25 | $772.20 | $100.63 |
| 50,000 | 960,937 | 865.00M | 395.00M | $1,006.25 | $3,861.00 | $503.13 |

> **Batch API** delivers results within 24h at 50% discount — viable for bulk processing.

### 1.5 Monthly Operational Cost (Steady-State)

Assuming 500 new documents/month, with 10% re-extraction for corrections:

| Scenario | New Docs | Re-extractions | Monthly LLM Cost (mini) | Annual |
|----------|----------|----------------|------------------------:|-------:|
| Pilot | 100 | 10% | $2.21 | $27 |
| Production | 500 | 10% | $11.07 | $133 |
| Scale | 5,000 | 10% | $110.69 | $1,328 |
| Enterprise | 50,000 | 5% | $1,056.56 | $12,679 |

---

## 2. Compute / GPU Costs

### 2.1 Pipeline Stage Timing (Measured — 96 docs)

| Stage | Wall Time | Notes |
|-------|----------:|-------|
| Ingest + Parse | ~60s | PDF parsing, clause splitting |
| Embedding | ~30s | 3,034 chunks × 768-dim, Nomic on A100 |
| Clustering (EVoC) | ~600s | Optuna 30 trials + HDBSCAN + EVoC |
| Domain Detection | ~120s | Hierarchical clustering |
| Field Discovery | ~180s | LLM calls (5 workers) |
| Extraction | ~300s | LLM calls (5 workers) |
| Intent Extraction | ~99s | 350 intents, 3.5 intents/sec (5 workers) |
| Knowledge Graph | <1s | NetworkX construction + analysis |
| Export | ~5s | JSON artifacts |
| **Total** | **~1,239s (20m39s)** | |

### 2.2 GPU Utilization

| Component | GPU Used | VRAM | Duration |
|-----------|----------|-----:|----------|
| Nomic Embedder (nomic-embed-text-v1.5) | A100 #1 | ~2 GB | ~30s for 3K chunks |
| EVoC clustering (cuML HDBSCAN) | A100 #1 | ~4 GB | ~600s |
| Intent similarity (pairwise cosine) | A100 #1 | <1 GB | ~1.6s for 319 intents |
| **A100 #2** | **Idle** | 0 | — |

> One A100 is sufficient. Second GPU is unused by the pipeline.

### 2.3 Azure VM Cost

| VM SKU | GPUs | Price/hr | Price/month (730h) | Needed? |
|--------|------|----------|-------------------:|---------|
| NC24ads A100 v4 (1×A100) | 1 | $3.67 | $2,679 | Sufficient for pipeline |
| NC48ads A100 v4 (2×A100) | 2 | $7.35 | $5,366 | Current (overprovisioned) |
| NC96ads A100 v4 (4×A100) | 4 | $14.69 | $10,723 | Overkill |

**Recommendation:** Downgrade to 1× A100 VM. Saves **$2,687/month**.

### 2.4 Compute Cost Per Document

| Scale | Pipeline Time (est.) | GPU-hours | VM Cost |
|------:|---------------------:|----------:|--------:|
| 96 docs | 20m39s | 0.34h | $1.27 |
| 500 docs | ~105min | 1.75h | $6.42 |
| 1,000 docs | ~210min | 3.5h | $12.85 |
| 5,000 docs | ~17.5h | 17.5h | $64.23 |

**Cost per document (compute): $0.013**

---

## 3. Storage Costs

### 3.1 Storage Per Document (Measured)

| Component | Size (96 docs) | Per Document | What's Stored |
|-----------|---------------:|-------------:|---------------|
| DuckDB | 28.0 MB | 0.29 MB | Clauses, clusters, intents, extractions, relations |
| LanceDB | 12.0 MB | 0.12 MB | Vector index for similarity search |
| Embeddings (NPZ) | 8.5 MB | 0.09 MB | 768-dim float32 arrays |
| JSON Artifacts | ~5.0 MB | 0.05 MB | Fields, chunks, clause_types, KG exports |
| Knowledge Graph JSONs | ~2.0 MB | 0.02 MB | Rules, strategies, archetypes, recommendations |
| **Total** | **55.5 MB** | **0.57 MB** | |

### 3.2 Scaling Projections

| Documents | DuckDB | LanceDB | Embeddings | Artifacts | Total |
|----------:|-------:|--------:|-----------:|----------:|------:|
| 100 | 29 MB | 13 MB | 9 MB | 7 MB | 58 MB |
| 500 | 146 MB | 63 MB | 44 MB | 36 MB | 289 MB |
| 1,000 | 292 MB | 125 MB | 89 MB | 73 MB | 579 MB |
| 5,000 | 1.4 GB | 625 MB | 443 MB | 365 MB | 2.8 GB |
| 10,000 | 2.9 GB | 1.3 GB | 886 MB | 729 MB | 5.7 GB |
| 50,000 | 14.3 GB | 6.3 GB | 4.3 GB | 3.6 GB | 28.5 GB |
| 100,000 | 28.6 GB | 12.5 GB | 8.6 GB | 7.3 GB | 57.0 GB |

### 3.3 Azure Storage Pricing

| Tier | Price/GB/month | 10K docs | 50K docs | 100K docs |
|------|---------------:|---------:|---------:|----------:|
| Premium SSD (attached) | $0.132 | $0.75 | $3.76 | $7.52 |
| Standard SSD | $0.08 | $0.46 | $2.28 | $4.56 |
| Blob Storage (Hot) | $0.018 | $0.10 | $0.51 | $1.03 |

> **Storage is negligible.** Even at 100K documents, storage is <$8/month.

---

## 4. Throughput & Latency

### 4.1 Observed Throughput (96-doc run)

| Metric | Value | Conditions |
|--------|------:|------------|
| **End-to-end pipeline** | 4.6 docs/min | Full pipeline, 96 docs |
| **PDF parse + chunk** | ~96 docs/min | CPU only |
| **Embedding** | ~6,000 chunks/min | Nomic, A100, batch_size=64 |
| **EVoC clustering** | ~5 chunks/sec | With Optuna 30 trials |
| **Intent extraction** | 3.5 intents/sec | GPT-5.4-mini, 5 workers |
| **Intent extraction** | 212 intents/min | Per-minute sustained rate |
| **Knowledge graph build** | <1s total | NetworkX, 1,807 edges |
| **Similarity embedding** | 319 intents in 1.6s | Nomic on A100 |

### 4.2 Per-Call LLM Latency (GPT-5.4-mini, Azure East US 2)

| Operation | Avg Latency | P50 | P95 | Max |
|-----------|------------:|----:|----:|----:|
| Intent extraction (1 clause) | ~283ms³ | ~250ms | ~500ms | ~1.5s |
| Field extraction (1 clause) | ~300ms | ~250ms | ~600ms | ~2s |
| Field discovery (1 cluster) | ~400ms | ~350ms | ~800ms | ~2.5s |

³ *283ms = 99s / 350 intents with 5 concurrent workers. Single-threaded would be ~1.4s.*

### 4.3 Throughput Bottlenecks

| Stage | Bottleneck | Mitigation |
|-------|-----------|------------|
| Clustering (EVoC) | CPU-bound Optuna trials | Reduce trials (30→10), cache results |
| Intent extraction | API rate limits + latency | Increase workers (5→10), use Batch API |
| Extraction | API rate limits | Same as above |
| Embedding | GPU memory bandwidth | Already fast (~200 chunks/sec) |

### 4.4 Projected Throughput at Scale

| Documents | Est. Pipeline Time | Docs/Hour | Bottleneck |
|----------:|-------------------:|----------:|------------|
| 100 | ~22 min | 273 | Clustering |
| 500 | ~1.75 hr | 286 | Intent extraction |
| 1,000 | ~3.5 hr | 286 | Intent extraction |
| 5,000 | ~17.5 hr | 286 | Intent extraction |
| 10,000 | ~35 hr | 286 | Intent extraction |

> At scale, **intent extraction becomes the bottleneck** (API rate limits).
> With Batch API (async, 24hr delivery), 50K docs can be processed in a single batch.

---

## 5. Total Cost Summary

### 5.1 Per-Document Cost Breakdown

| Component | GPT-5.4-mini | GPT-5 full |
|-----------|-------------:|-----------:|
| LLM (intent + extraction + fields) | $0.020 | $0.077 |
| GPU compute | $0.013 | $0.013 |
| Storage | $0.0001 | $0.0001 |
| **Total per document** | **$0.033** | **$0.090** |

### 5.2 Monthly Cost Scenarios

| | Pilot (100 docs/mo) | Production (500/mo) | Scale (5K/mo) | Enterprise (50K/mo) |
|---|---:|---:|---:|---:|
| **LLM (mini)** | $2 | $11 | $111 | $1,057 |
| **LLM (full GPT-5)** | $8 | $39 | $386 | $3,861 |
| **VM (1× A100)** | $2,679 | $2,679 | $2,679 | $2,679 |
| **Storage** | <$1 | <$1 | <$1 | $4 |
| **Total (mini)** | **$2,682** | **$2,691** | **$2,791** | **$3,740** |
| **Total (full)** | **$2,688** | **$2,719** | **$3,066** | **$6,544** |

### 5.3 Cost Optimization Levers

| Optimization | Savings | Trade-off |
|-------------|--------:|-----------|
| Downgrade to 1× A100 VM | $2,687/mo | None — pipeline only uses 1 GPU |
| Use Batch API (50% off LLM) | 50% of LLM cost | Results in 24h instead of real-time |
| Prompt caching (same clause type = cached prefix) | ~75% of input tokens | Requires structured prompt design |
| Reduce Optuna trials (30→10) | ~60% of clustering time | Slightly lower cluster quality |
| Skip intent extraction for ToC/Signatures | ~15% fewer LLM calls | Already filtered by min_clause_chars=80 |
| Spot/Low-priority VM | ~60% VM savings | May be preempted |

### 5.4 Break-Even: Spot VM + Batch API (Aggressive)

| Component | Cost |
|-----------|-----:|
| Spot VM (1× A100, ~60% off) | $1,072/mo |
| Batch API, 5K docs/mo (50% off) | $50/mo |
| Storage | <$1/mo |
| **Total** | **$1,123/mo** |
| **Per document** | **$0.022** |

---

## 6. Comparison: GPT-5.4-mini vs GPT-5 (Full)

| Dimension | GPT-5.4-mini | GPT-5 (full) |
|-----------|:-------------|:-------------|
| **Input price** | $0.25/1M | $1.25/1M (5× more) |
| **Output price** | $2.00/1M | $10.00/1M (5× more) |
| **Latency** | ~250ms/call | ~400ms/call |
| **Intent quality** | Good for structured extraction | Better for nuanced legal reasoning |
| **Cost at 5K docs/mo** | $111 LLM | $386 LLM |
| **Recommendation** | ✅ Use for production | Use for quality-critical audits |

> GPT-5.4-mini at $0.020/doc is the sweet spot. Upgrade to full GPT-5 only for
> high-stakes compliance reviews where intent nuance matters.

---

## 7. Key Takeaways

1. **LLM costs are trivial** — $0.020/doc with GPT-5.4-mini. At 5K docs/month it's $111.
2. **The VM is the real expense** — $2,679/month for a GPU VM dwarfs everything else.
3. **You're overprovisioned** — pipeline uses 1 GPU; second A100 is idle. Save $2,687/mo.
4. **Storage is free-adjacent** — 100K documents = 57 GB = $8/month.
5. **Intent extraction is the bottleneck** at scale — not cost, but throughput (API rate limits). Batch API solves this.
6. **At 50K docs/month** the total system costs ~$3,740/mo (mini) — well within enterprise SaaS margins.
