# EVoC Evaluation Framework — Gold Set Strategy
## How to Build a 50-Document Gold Standard

### Selection Criteria (Stratified Sampling)

Pick 50 documents from the pipeline output that cover:

| Stratum | Count | Why |
|---------|-------|-----|
| **High-coherence clusters** (top 10 by quality_score) | 15 docs | Validate that good clusters stay good |
| **Low-coherence clusters** (bottom 10 by quality_score) | 10 docs | Hardest cases — most likely to regress |
| **Multi-domain agreements** (appear in >1 cluster type) | 10 docs | Test cross-domain consistency |
| **Short/edge-case documents** (<5 clauses) | 5 docs | Boundary conditions |
| **Long complex documents** (>50 clauses) | 5 docs | Stress test chunking + intent discovery |
| **Recent outlier-squashed** (newly assigned by refiner) | 5 docs | Validate outlier assignment quality |

### Annotation Instructions

For each of the 50 documents, a domain expert (legal ops or contract analyst) annotates:

#### 1. Clause-Level Ground Truth
For each clause:
- **Expected cluster label** — what "type" is this clause? (e.g., "Indemnification", "Payment Terms")
- **Expected intents** — what obligations/rights/constraints does this clause create?
  - Label (snake_case)
  - One-line summary
  - Party from / party to
  - Is it actionable? (yes/no — would you set a reminder/trigger a workflow?)
  - Is it faithful? (yes/no — is this actually in the clause text?)

#### 2. Field-Level Ground Truth
For each agreement:
- **Expected extracted fields** — what key-value pairs should the pipeline find?
  - Field name
  - Field type (date, currency, duration, etc.)
  - Expected value

### File Format

See `eval/gold_standard.json` for the schema. The structure is:

```json
{
  "_meta": {
    "version": "1.0",
    "created": "2026-04-03",
    "description": "...",
    "annotator": "Name",
    "pipeline_version": "evoc_200_refined"
  },
  "documents": [
    {
      "agreement_id": "agr_...",
      "filename": "...",
      "domain_label": "...",
      "clauses": [
        {
          "clause_id": "cls_...",
          "clause_title": "...",
          "gold_cluster_label": "...",
          "gold_intents": [...],
          "gold_fields": [...]
        }
      ]
    }
  ],
  "cluster_ground_truth": { "cls_X": "Label", ... },
  "field_ground_truth": { "agr_X::Field Name": "value", ... }
}
```

### Bootstrap Workflow (Semi-Automated)

To accelerate annotation, bootstrap from existing pipeline output:

```bash
# 1. Export candidate gold set from current DB
python -m eval.bootstrap data/evoc_200_refined.duckdb --n-docs 50

# 2. Review & correct in the JSON file
#    → Fix wrong cluster labels
#    → Remove hallucinated intents
#    → Add missing fields

# 3. Run eval against the corrected gold set
python -m eval.runner data/evoc_200_refined.duckdb \
    --gold eval/gold_standard.json \
    --save-baseline eval/baseline_results.json
```

### What Each Metric Tests

| Metric | What It Catches | When It Fires |
|--------|----------------|---------------|
| `clustering.silhouette_score` | Cluster separation degraded | Changed embeddings or chunk size |
| `clustering.davies_bouldin_index` | Clusters overlapping | Changed HDBSCAN params |
| `clustering.chunk_outlier_pct` | Too many unassigned chunks | Embedding model change |
| `clustering.clause_outlier_pct` | Clauses without types | Chunking strategy change |
| `kg.neighborhood_domain_consistency` | Similar docs drifting apart | Embedding model change |
| `kg.neighborhood_cluster_overlap` | KNN neighbors share fewer types | Clustering params change |
| `naming.avg_semantic_similarity` | Cluster labels drifted | Label synthesis LLM change |
| `fields.f1` | Extraction regression | Field discovery prompt change |
| `judge.healthy_insight_score` | Intent quality dropped | Intent extraction prompt change |
| `judge.hallucination_rate` | LLM making things up | Model version upgrade |
| `intent_gold.recall_at_0_8` | Missing known intents | Temperature / prompt change |

### Running the Eval Suite

```bash
# Alongside any pipeline run — add to your CLI command:
python -m core.evoc_pipeline /path/to/pdfs --account-id my_acct \
    --mode hybrid --optuna-trials 30 --refine --analytics

# Then immediately run eval:
python -m eval.runner data/my_acct_refined.duckdb \
    --gold eval/gold_standard.json \
    --baseline eval/baseline_results.json \
    --judge --judge-sample 50 \
    --output eval/report_$(date +%Y%m%d_%H%M).md \
    --save-baseline eval/baseline_results.json
```

### Regression Thresholds

Default: 5% relative degradation triggers ❌. Hard thresholds:

| Metric | Hard Threshold | Direction |
|--------|---------------|-----------|
| `chunk_outlier_pct` | > 20% | ❌ |
| `clause_outlier_pct` | > 50% | ❌ |
| `silhouette_score` | < 0.05 | ❌ |
| `davies_bouldin_index` | > 5.0 | ❌ |
| `fields.f1` | < 0.10 | ❌ |
