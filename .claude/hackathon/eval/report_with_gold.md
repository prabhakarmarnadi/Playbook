## Evaluation Report

**Summary:** 37 ✅  0 ❌  15 🆕  (52 total metrics)

### _META

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| eval_seconds | 24.9000 | 1.3000 | +23.6000 | +1815.4% | ✅ |

### CLUSTERING

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| assigned_chunks | 6184 | 6184 | +0.0000 | +0.0% | ✅ |
| avg_coherence | 0.8939 | 0.8939 | +0.0000 | +0.0% | ✅ |
| chunk_outlier_pct | 0.0000 | 0.0000 | +0.0000 | +0.0% | ✅ |
| clause_outlier_pct | 0.1200 | 0.1200 | +0.0000 | +0.0% | ✅ |
| clauses_no_embedding_pct | 38.4000 | 38.4000 | +0.0000 | +0.0% | ✅ |
| clauses_with_type | 7351 | 7351 | +0.0000 | +0.0% | ✅ |
| cluster_size_cv | 0.8470 | 0.8470 | +0.0000 | +0.0% | ✅ |
| davies_bouldin_index | 2.8378 | 2.8378 | +0.0000 | +0.0% | ✅ |
| largest_cluster_pct | 4.0400 | 4.0400 | +0.0000 | +0.0% | ✅ |
| median_cluster_size | 32.0000 | 32.0000 | +0.0000 | +0.0% | ✅ |
| min_coherence | 0.8313 | 0.8313 | +0.0000 | +0.0% | ✅ |
| n_clusters | 136 | 136 | +0.0000 | +0.0% | ✅ |
| silhouette_score | 0.0852 | 0.0852 | +0.0000 | +0.0% | ✅ |
| smallest_cluster_size | 9 | 9 | +0.0000 | +0.0% | ✅ |
| total_chunks | 6184 | 6184 | +0.0000 | +0.0% | ✅ |
| total_clauses | 7360 | 7360 | +0.0000 | +0.0% | ✅ |
| weighted_coherence | 0.8821 | 0.8821 | +0.0000 | +0.0% | ✅ |

### FIELDS

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| f1 | 0.9963 | — | — | — | 🆕 |
| gold_count | 1072 | — | — | — | 🆕 |
| precision | 0.9963 | — | — | — | 🆕 |
| predicted_count | 5888 | — | — | — | 🆕 |
| recall | 0.9963 | — | — | — | 🆕 |
| true_positives | 1068 | — | — | — | 🆕 |

### INTENT_GOLD

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| avg_best_match_sim | 1.0000 | — | — | — | 🆕 |
| gold_count | 2710 | — | — | — | 🆕 |
| matched_count | 2710 | — | — | — | 🆕 |
| recall_at_0_8 | 100.00 | — | — | — | 🆕 |

### KG

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| neighborhood_cluster_overlap | 0.7707 | 0.7707 | +0.0000 | +0.0% | ✅ |
| neighborhood_domain_consistency | 0.4535 | 0.4535 | +0.0000 | +0.0% | ✅ |

### NAMING

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| avg_semantic_similarity | 1.0000 | — | — | — | 🆕 |
| exact_match_pct | 100.00 | — | — | — | 🆕 |
| gold_count | 603 | — | — | — | 🆕 |
| matched_count | 603 | — | — | — | 🆕 |
| min_semantic_similarity | 1.0000 | — | — | — | 🆕 |

### SYSTEM

| Metric | Current | Baseline | Delta | Δ% | Status |
|--------|---------|----------|-------|-----|--------|
| agreements_with_domain_pct | 100.00 | 100.00 | +0.0000 | +0.0% | ✅ |
| agreements_with_extractions_pct | 100.00 | 100.00 | +0.0000 | +0.0% | ✅ |
| avg_extraction_confidence | 0.7196 | 0.7196 | +0.0000 | +0.0% | ✅ |
| avg_intents_per_clause | 4.0900 | 4.0900 | +0.0000 | +0.0% | ✅ |
| clauses_typed_pct | 99.8800 | 99.8800 | +0.0000 | +0.0% | ✅ |
| clauses_with_intents_pct | 56.5900 | 56.5900 | +0.0000 | +0.0% | ✅ |
| clusters_labeled_pct | 100.00 | 100.00 | +0.0000 | +0.0% | ✅ |
| clusters_with_both_intents_and_fields_pct | 98.5300 | 98.5300 | +0.0000 | +0.0% | ✅ |
| clusters_with_fields_pct | 100.00 | 100.00 | +0.0000 | +0.0% | ✅ |
| clusters_with_intents_pct | 98.5300 | 98.5300 | +0.0000 | +0.0% | ✅ |
| field_definitions | 1523 | 1523 | +0.0000 | +0.0% | ✅ |
| intent_dedup_ratio | 2.2312 | 2.2312 | +0.0000 | +0.0% | ✅ |
| intent_instances | 17024 | 17024 | +0.0000 | +0.0% | ✅ |
| intent_types_discovered | 7008 | 7008 | +0.0000 | +0.0% | ✅ |
| low_confidence_extraction_pct | 0.0000 | 0.0000 | +0.0000 | +0.0% | ✅ |
| total_agreements | 198 | 198 | +0.0000 | +0.0% | ✅ |
| total_extractions | 23225 | 23225 | +0.0000 | +0.0% | ✅ |
