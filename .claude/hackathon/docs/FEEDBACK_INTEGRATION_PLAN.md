# Feedback Integration Plan: Self-Improving Contract Extraction Pipeline

> **Status:** Proposal  
> **Date:** April 24, 2026  
> **Scope:** How user feedback should flow back into every pipeline stage — extraction, discovery, clustering, and assignment — to create a continuously improving system.

---

## 1. Current State

### What's Built

We have a complete feedback infrastructure that mostly sits idle:

| Component | Location | Status |
|-----------|----------|--------|
| FeedbackStore (6 tables) | `core/feedback_store.py` | ✅ Fully implemented |
| PersonalizationEngine (4 subsystems) | `core/personalization.py` | ✅ Fully implemented |
| Weight matrix with temporal decay | `feedback_store.get_weights_batch()` | ✅ Ready to query |
| Virtual clusters | `feedback_store.create_virtual_cluster()` | ✅ Built, never auto-used |
| Intent overrides | `feedback_store.get/set_intent_override()` | ✅ Built, partially wired |
| Tenant config (adaptive thresholds) | `feedback_store.tenant_config` | ✅ Built, wired to assignment |
| Auto-threshold tuner | `personalization.auto_tune_thresholds()` | ✅ Built, never called |
| FastAPI endpoints for all feedback ops | `server.py` (26 endpoints) | ✅ Tested, live |

### What's Wired (3 injection points)

| Point | Stage | What it does | Code location |
|-------|-------|-------------|---------------|
| Correction few-shot injection | Stage 4 (Extraction) | Appends up to 5 `"Instead of X, use Y"` lines to extraction prompt | `minimal_pipeline.py` L2263–2271 |
| Threshold override for confidence gate | Stage 4 (Extraction) | `tentative_threshold` from feedback overrides the Cochran gate | `minimal_pipeline.py` L2089–2090 |
| Assignment tier routing | Stage 5 (Assignment) | `high_confidence_threshold` / `tentative_threshold` from feedback set A/B/C tiers | `minimal_pipeline.py` L3618–3640 |

### What's NOT Wired (the gaps)

| Signal | Built? | Consumed? | Gap |
|--------|--------|-----------|-----|
| `cluster_weights` (per-cluster quality scores) | ✅ | ❌ | RAG retrieval re-ranking ignores them |
| `edge_weights` (entity relationship weights) | ✅ | ❌ | KG traversal never reads them |
| `virtual_clusters` (user-created categories) | ✅ | ❌ | Never injected into IncrementalClusterAssigner |
| `approved_entities` (user-confirmed outputs) | ✅ | ❌ | Never gated on during extraction |
| Discovery feedback | ❌ | N/A | Field naming corrections never reach discovery prompts |
| Clustering feedback (split/merge/reassign) | Partial | ❌ | Only stored as events, never reshape topology |
| `auto_tune_thresholds()` | ✅ | ❌ | Never called from pipeline or API |

---

## 2. Feedback Signal Taxonomy

Every feedback action should route to the pipeline component it can improve:

| Feedback action | Extraction (prompts) | Discovery (prompts) | Clustering (topology) | Assignment (thresholds) |
|---|---|---|---|---|
| `correct` (extraction value) | **yes** — KNN few-shot demo | **yes** — field naming pref | | |
| `reject` (extraction) | **yes** — negative demo | | | **yes** — raise gate |
| `approve` (extraction) | **yes** — positive demo | | | **yes** — lower gate |
| `reassign` (chunk → cluster) | | | **yes** — must/cannot link | **yes** — centroid update |
| `merge` (two clusters) | | | **yes** — must-link constraint | **yes** — granularity signal |
| `split` (cluster) | | | **yes** — cannot-link constraint | **yes** — granularity signal |
| `relabel` (cluster name) | | **yes** — vocabulary pref | | |
| `correct_intent` (intent label) | | **yes** — intent override | | |
| `refine` (minor edit) | **yes** — soft negative | | | |

---

## 3. Continuous Improvement Architecture

### Dual-Loop Design

Separate **fast** feedback (applied immediately, no pipeline restart) from **slow** feedback (batched, requires re-run):

```
FAST LOOP (real-time, per-request)                SLOW LOOP (batched, per-run)
─────────────────────────────────                 ─────────────────────────────
• KNN few-shot bank updates                       • Constraint-based re-clustering
• Weight matrix updates (automatic)               • Optuna objective reweighting
• Threshold adjustments                           • Virtual cluster promotion
• Intent override application                     • Centroid drift correction
• Correction injection into prompts               • Discovery prompt evolution
                                                  • Few-shot bank pruning/curation
Trigger: every feedback event                     Trigger: N corrections OR T days
Latency: <1s (next extraction sees it)            Latency: hours (full re-run)
```

### Signal Flow Diagram

```
                 ┌──────────────────────────────────────────┐
                 │            User Feedback Events           │
                 │    correct / reject / approve / merge     │
                 │    split / relabel / reassign / refine    │
                 └──────────────────┬───────────────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │     Feedback Router      │
                       │   (route by action type  │
                       │    to affected stages)   │
                       └──┬────┬────┬────┬───────┘
                          │    │    │    │
           ┌──────────────┘    │    │    └──────────────────┐
           │           ┌───────┘    └────────┐              │
           ▼           ▼                     ▼              ▼
  ┌─────────────┐ ┌──────────┐     ┌──────────────┐ ┌────────────┐
  │   Cluster   │ │ Discovery│     │  Extraction   │ │ Threshold  │
  │  Topology   │ │  Prompt  │     │   Prompt      │ │ Auto-Tuner │
  │  Refiner    │ │ Enricher │     │  Optimizer     │ │            │
  │             │ │          │     │               │ │ Adjusts:   │
  │ Produces:   │ │ Injects: │     │ Updates:      │ │ - conf gate│
  │ - must/can't│ │ - naming │     │ - KNN few-    │ │ - assign   │
  │   link pairs│ │   prefs  │     │   shot bank   │ │   tiers    │
  │ - Optuna    │ │ - field  │     │ - correction  │ │ - outlier  │
  │   obj tweak │ │   vocab  │     │   examples    │ │   thresh   │
  │ - virtual   │ │ - intent │     │ - approved    │ │ - Optuna   │
  │   cluster   │ │   vocab  │     │   entity set  │ │   obj wt   │
  │   promotion │ │          │     │               │ │            │
  │ - centroid  │ │ Affects: │     │ Affects:      │ │ Affects:   │
  │   correction│ │ Stage 3.5│     │ Stage 4       │ │ Stages 2-5 │
  │             │ │ + Stage 4│     │               │ │            │
  │ Affects:    │ │ (discov) │     │               │ │            │
  │ Stage 2     │ │          │     │               │ │            │
  │ (cluster)   │ │          │     │               │ │            │
  └──────┬──────┘ └────┬─────┘     └───────┬───────┘ └─────┬──────┘
         │             │                   │               │
         └─────────────┴─────────┬─────────┴───────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Recompile Trigger   │
                      │  (N corrections OR   │
                      │   T days OR          │
                      │   novel_buffer_full) │
                      └──────────┬──────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │     Next Pipeline Run      │
                   │                            │
                   │  1. Cluster with           │
                   │     constraints +          │
                   │     reweighted Optuna obj  │
                   │  2. Discover with          │
                   │     enriched prompts       │
                   │  3. Extract with           │
                   │     KNN few-shot +         │
                   │     cluster-weighted       │
                   │     retrieval              │
                   │  4. Assign with tuned      │
                   │     thresholds + promoted  │
                   │     virtual clusters       │
                   └───────────────────────────┘
```

---

## 4. Detailed Design per Stage

### 4.1 Extraction Stage — Feedback-Weighted Retrieval

**Problem:** When a user repeatedly corrects extractions from cluster X, that's a signal that cluster X's chunks are noisy or mis-assigned. Currently, all clusters get equal retrieval treatment.

**Solution:** Multiply chunk RRF scores by the cluster's feedback weight during retrieval re-ranking.

**How it works:**
1. `feedback_store.get_weights_batch("cluster")` returns `{cluster_id: weight}` where:
   - `1.0` = neutral (no feedback)
   - `< 1.0` = penalized (user corrected outputs from this cluster)
   - `> 1.0` = boosted (user approved outputs from this cluster)
2. During `_hybrid_retrieve()`, after RRF fusion, scale each chunk's score: `rrf_score *= cluster_weight`
3. Weights decay toward 1.0 over time (90-day half-life already implemented)

**Impact:** Chunks from high-correction clusters rank lower → less likely to be selected as context → fewer bad extractions.

### 4.2 Extraction Stage — KNN Few-Shot Demonstrations

**Problem:** Currently, we inject the same 5 correction examples into every extraction prompt, regardless of what's being extracted. A correction for "Notice Period" fields shouldn't influence "Payment Terms" extraction.

**Solution:** Use nearest-neighbor retrieval over the correction store to select the most relevant demonstrations per extraction call.

**How it works:**
1. Every `correct_extraction` event stores: `(chunk_text_snippet, field_name, wrong_value, correct_value)`
2. Embed these corrections using the same Nomic embedder (768-dim)
3. At extraction time, embed the current chunk + field name, retrieve K=3 nearest correction examples
4. Format them as few-shot demonstrations in the extraction prompt:
   ```
   ## Similar Corrections (apply these patterns):
   Field: termination_notice_days | Chunk: "...either party may terminate upon 30 days..."
   Wrong: "either party may terminate" | Correct: "30 days"
   ```
5. These replace the current generic 5-correction injection

**Impact:** Field-specific corrections reach only relevant extractions. A user correcting one field type doesn't pollute prompts for unrelated fields.

### 4.3 Discovery Stage — Correction-Aware Field Naming

**Problem:** Field discovery has zero feedback awareness. Even after a user renames "Notice Period" → "termination_notice_period" 10 times, the next discovery run will rediscover it with the wrong name.

**Solution:** Inject accumulated naming corrections and intent overrides into the discovery prompt.

**How it works:**
1. Before discovery, gather `correction_examples` and `intent_overrides` from PersonalizationContext
2. Append a section to `FIELD_DISCOVERY_PROMPT`:
   ```
   ## Naming Preferences (from prior user feedback):
   - Use "termination_notice_period" not "Notice Period"
   - Use "governing_law_state" not "jurisdiction"
   - Intent "liability_cap" should be called "monetary_limitation"
   ```
3. The discovery model respects these preferences when generating field names

**Impact:** Fields come out pre-aligned with user vocabulary. Fewer post-discovery corrections needed.

### 4.4 Clustering Stage — Constraint-Based Refinement

**Problem:** When users reassign chunks, merge clusters, or split clusters, they're giving direct supervision about what the cluster topology should look like. This signal is stored as events but never reshapes the actual clusters.

**Solution:** Convert cluster feedback into constraints that influence the next clustering run.

**How it works:**
1. Extract constraints from feedback events:
   - `reassign(chunk_A, C1→C2)` → `must_link(A, C2)`, `cannot_link(A, C1)`
   - `merge(C1, C2)` → `must_link(sample_from_C1, sample_from_C2)` for N pairs
   - `split(C)` → `cannot_link` across the two halves
2. During EVoC optimization (Stage 2), add constraint satisfaction as a term in the Optuna objective:
   ```
   score = α * silhouette + β * cosine_coherence + γ * constraint_satisfaction
   ```
   where `constraint_satisfaction` = fraction of must-link/cannot-link pairs satisfied
3. The optimizer naturally finds a resolution that respects user feedback

**Impact:** Cluster boundaries physically move to match what users want. Downstream discovery and extraction improve because clusters are more semantically coherent.

### 4.5 Clustering Stage — Granularity Signal from Split/Merge Ratio

**Problem:** If users consistently split clusters, the EVoC resolution is too coarse. If they merge, it's too fine. The Optuna objective doesn't know this.

**Solution:** Track `split_count / merge_count` ratio and adjust the Optuna objective weighting.

**How it works:**
1. From feedback_events, compute: `ratio = split_events / max(merge_events, 1)`
2. If `ratio > 2.0` → users want finer clusters → add penalty for large clusters to Optuna objective
3. If `ratio < 0.5` → users want coarser clusters → add penalty for too many clusters
4. Persist this preference in tenant_config for the next run

**Impact:** Each run automatically calibrates the "right" number of clusters for this tenant's mental model.

### 4.6 Assignment Stage — Virtual Cluster Auto-Promotion

**Problem:** Users create virtual clusters (outlier categories), but these never get injected into the assigner. New documents that match a virtual cluster still get assigned to the nearest global cluster or flagged as "novel."

**Solution:** When a virtual cluster reaches a member threshold, automatically promote it into the assigner's centroid set.

**How it works:**
1. On pipeline startup, check all virtual clusters for the tenant
2. If `member_count >= N` (default 5) and `promote_virtual_cluster()` has been called (centroid exists):
   - Load the virtual cluster's centroid into `IncrementalClusterAssigner.centroids`
   - New documents can now match against virtual clusters
3. Assignments to virtual clusters get tagged `assignment_type = "virtual_match"`

**Impact:** User-defined categories become real routing targets without a full re-clustering run.

### 4.7 Assignment Stage — Auto-Threshold Tuning

**Problem:** `auto_tune_thresholds()` is fully implemented but never called. It analyzes feedback patterns to suggest threshold adjustments.

**Solution:** Call it periodically (every N feedback events or when `/personalization/auto-tune` is hit) and apply its suggestions.

**How it works (already implemented):**
1. Analyze feedback: `approve_rate = cluster_approves / (cluster_approves + cluster_corrects)`
2. If `approve_rate > 0.9` → lower `high_confidence_threshold` (pipeline is good, be more aggressive)
3. If `approve_rate < 0.5` → raise `high_confidence_threshold` (pipeline makes mistakes, be conservative)
4. If `len(virtual_clusters) >= 5` → lower `outlier_min_similarity` (user sees many outlier categories)
5. Write updated thresholds to `tenant_config` → next run uses them

**Impact:** Thresholds self-calibrate based on user behavior. No manual tuning required.

---

## 5. DSPy-Inspired Structures (No Model Training)

These ideas borrow DSPy's optimization patterns without training any models:

### 5.1 KNN Few-Shot (from `dspy.KNNFewShot`)

Already described in §4.2. The core DSPy insight: instead of fixed demonstration examples, retrieve the **nearest** examples from a bank at inference time. We implement this using our existing Nomic embedder + a small LanceDB table of corrections.

### 5.2 SIMBA-Style Reflective Rules (from `dspy.SIMBA`)

**Concept:** Identify fields with high extraction variance (inconsistent outputs), then use a meta-LLM call to generate self-reflective improvement rules.

**How it works:**
1. After a run, compute per-field confidence standard deviation. Fields with stddev > 0.2 are "unstable."
2. For unstable fields, gather: 5 approved extractions + 5 corrected extractions
3. Meta-prompt: "Here are 10 extractions for field X. Some were correct (user approved), some were wrong (user corrected). What pattern explains the mistakes? Write one rule to avoid them."
4. The meta-LLM produces rules like: `"When extracting 'Governing Law', look for state names in jurisdiction clauses, not in the header address block"`
5. Persist these rules per-field and inject them into the extraction prompt as a `## Field-Specific Rules` section

**Cost:** One meta-LLM call per unstable field per run. At $0.02/call × 50 unstable fields = $1 per run.

**Impact:** Targeted prompt improvement for exactly the fields that need it, driven by user correction patterns.

### 5.3 GEPA-Style Prompt Evolution (from `dspy.GEPA`)

**Concept:** After accumulating enough feedback trajectories, use an LLM to reflect on what went wrong and propose an improved prompt.

**How it works:**
1. Log extraction trajectories: `(prompt_used, chunk_text, model_output, user_feedback)`
2. When `correction_count >= 50` for a cluster type, run GEPA:
   - Gather 20 trajectories (mix of approved + corrected)
   - Meta-prompt: "Here's the current extraction instruction and 20 extraction attempts. X were approved, Y were corrected. Analyze the failures and write an improved extraction instruction."
3. A/B test: run the new instruction on a held-out sample (10 chunks from the cluster), compare against the old instruction using the judge metric
4. If the new instruction wins, promote it as the cluster's extraction instruction

**Cost:** One meta-LLM call + 20 re-extractions per optimization. ~$2 per cluster optimization.

**Impact:** Prompts evolve based on actual user behavior. The system literally learns how to ask better questions.

---

## 6. What Can We Implement NOW (Minimal Changes)

These are the changes that take the least code and wire into the most existing infrastructure:

### Quick Win 1: Cluster-Weighted Retrieval Re-Ranking

**Effort:** ~20 lines changed across 2 files

**Changes:**
1. **`core/rag_extractor.py`** — In `RAGConfig`, add `cluster_weight_map: dict[str, float] = field(default_factory=dict)`
2. **`core/rag_extractor.py` line ~1098** — In `_hybrid_retrieve()`, after the cluster-boost line, multiply by feedback weight:
   ```python
   # Existing:
   r["rrf_score"] += boost
   # Add:
   fb_weight = self.config.cluster_weight_map.get(self._current_cluster_id, 1.0)
   r["rrf_score"] *= fb_weight
   ```
3. **`core/minimal_pipeline.py` line ~3402** — When building `shared_rag_config`, load and pass cluster weights:
   ```python
   cluster_weights = {}
   if personalization_ctx:
       cluster_weights = personalization_ctx.cluster_weights
   shared_rag_config = RAGConfig(
       ...,
       cluster_weight_map=cluster_weights,
   )
   ```

**How it helps:** Clusters that users frequently correct get their retrieval scores demoted. Clusters that users approve get boosted. Self-regulating quality per cluster.

### Quick Win 2: Correction-Aware Discovery Prompts

**Effort:** ~15 lines changed across 2 files

**Changes:**
1. **`core/field_discovery.py` line ~96** — Add `{feedback_section}` to `FIELD_DISCOVERY_PROMPT`:
   ```
   Keywords: {keywords}
   {feedback_section}
   CLAUSE EXCERPTS:
   ```
2. **`core/field_discovery.py` line ~191** — In `discover_fields_for_cluster()`, add parameter `feedback_section: str = ""` and pass it to `.format()`:
   ```python
   prompt = FIELD_DISCOVERY_PROMPT.format(
       cluster_label=cluster_label,
       keywords=", ".join(keywords[:10]),
       feedback_section=feedback_section,
       chunks_text=chunks_text,
       max_fields=config.max_fields_per_cluster,
   )
   ```
3. **`core/minimal_pipeline.py`** — Where discovery is called, build the feedback section from `personalization_ctx.correction_examples` and `personalization_ctx.intent_overrides` (same pattern as extraction injection at L2263).

**How it helps:** Discovery models use the user's preferred vocabulary from day one of a re-run. Field names come out consistent with prior corrections.

### Quick Win 3: Virtual Cluster Auto-Promotion

**Effort:** ~20 lines in 1 file

**Changes:**
1. **`core/minimal_pipeline.py`** — In `assign_new_documents()`, after loading centroids and before building the assigner, check for promotable virtual clusters:
   ```python
   if feedback_store:
       vcs = feedback_store.get_virtual_clusters()
       for vc in vcs:
           if vc["member_count"] >= 5 and vc.get("centroid"):
               centroid = deserialize(vc["centroid"])
               all_centroids[vc["cluster_id"]] = (centroid, vc["member_count"])
               cluster_labels[vc["cluster_id"]] = vc["label"]
   ```

**How it helps:** User-created categories automatically start receiving new documents. Closes the virtual cluster loop.

### Quick Win 4: Auto-Tune Trigger

**Effort:** ~10 lines in 1 file + 1 new endpoint

**Changes:**
1. **`server.py`** — Add `POST /personalization/auto-tune` endpoint:
   ```python
   @app.post("/personalization/auto-tune")
   async def auto_tune(tenant_id: str = Query(...), account_id: str | None = Query(None)):
       pe = _get_personalization_engine(tenant_id, account_id)
       result = pe.auto_tune_thresholds()
       pe.store.close()
       return result
   ```
2. **`core/minimal_pipeline.py`** — At pipeline startup, if feedback is enabled and `events >= 20`, auto-call `auto_tune_thresholds()`:
   ```python
   if feedback_store and feedback_store.summary()["feedback_events"] >= 20:
       personalization_engine.auto_tune_thresholds()
   ```

**How it helps:** Thresholds self-calibrate. No human tuning. Already fully implemented, just needs the trigger.

### Quick Win 5: Approved Entity Fast-Path

**Effort:** ~10 lines in 1 file

**Changes:**
1. **`core/minimal_pipeline.py`** — In the confidence gate (Cochran sampling), if a cluster_id is in `approved_entities`, reduce the sample fraction:
   ```python
   sample_fraction = config.cochran_sample_fraction
   if personalization_ctx and cluster_id in personalization_ctx.approved_entities:
       sample_fraction = max(0.1, sample_fraction * 0.5)  # half the sampling for approved clusters
   ```

**How it helps:** Clusters the user has confirmed as good get processed faster (less sampling overhead) and are less likely to be accidentally gated out.

---

## 7. Implementation Priority

| # | Change | Effort | Files | Impact | Loop type |
|---|--------|--------|-------|--------|-----------|
| **1** | Cluster-weighted retrieval re-ranking | ~20 lines | 2 | High — reduces repeat errors | Fast |
| **2** | Auto-tune trigger | ~10 lines | 2 | Medium — thresholds self-calibrate | Fast |
| **3** | Virtual cluster auto-promotion | ~20 lines | 1 | Medium — closes virtual cluster loop | Fast |
| **4** | Correction-aware discovery prompts | ~15 lines | 2 | Medium — naming consistency | Slow |
| **5** | Approved entity fast-path | ~10 lines | 1 | Low — performance optimization | Fast |
| **6** | KNN few-shot extraction | ~60 lines | 2 | High — precise per-field corrections | Fast |
| **7** | SIMBA reflective rules | ~80 lines | 2 | High — automated prompt debugging | Slow |
| **8** | GEPA prompt evolution | ~120 lines | 3 | Very high — prompts self-improve | Slow |
| **9** | Constraint-based re-clustering | ~100 lines | 2 | High — structural improvement | Slow |
| **10** | Granularity signal (split/merge ratio) | ~30 lines | 1 | Medium — cluster count self-calibrates | Slow |

**Suggested order:** 1 → 2 → 3 → 4 → 5 (all quick wins, total ~75 lines) → then 6 → 7 if we want DSPy-inspired behaviors.

---

## 8. Open Questions

1. **Cold start:** For new tenants with zero feedback, should we bootstrap from the "default" tenant's corrections? Or keep the pipeline fully unsupervised until feedback accumulates?

2. **Feedback volume thresholds:** How many corrections per cluster before we trust the weight? (Proposal: `min_feedback_count = 3` before applying weight, otherwise default to 1.0)

3. **Evaluation:** We need a held-out evaluation set to measure whether feedback actually improves extraction quality. The quality judge from v3 could serve as this — run it before and after feedback integration.

4. **Conflict resolution:** What if User A approves an extraction that User B rejects? (Proposal: recency wins with decay, or majority vote if >2 users)

5. **GEPA A/B testing:** How do we safely A/B test evolved prompts in production without regressing live extractions? (Proposal: shadow mode — run both prompts, surface the evolved one only if it scores higher on the judge metric)

---

## 9. Discovery Hallucination Problem

### Observation

In the v3 run, the pipeline proposed ~10 fields per cluster but only ~3.6 survived validation, giving a **64% waste rate**. Each phantom field costs a full validation cycle (sample 3 docs × RAG retrieval × LLM extraction × judge evaluation) before being discarded — pure wasted spend.

### Root Causes

Three layered issues create the phantom-field problem:

1. **Prompt encourages over-discovery.** The `FieldDiscoverySig` docstring (`field_discovery.py` L262) instructs `"aim for 10-15 fields per cluster. Miss nothing."` This explicit quantity target and "miss nothing" framing pushes the LLM to pad its output with speculative or marginal fields.

2. **Grounding criterion checks recurrence, not extractability.** Quick-sample validation (`minimal_pipeline.py` L1973, `_validate_fields_quick_sample`) asks "is this concept present?" in 3 sampled documents. A concept can be *mentioned* in prose ("this agreement relates to intellectual property") without being *extractable* as a structured value. The grounding test conflates presence with extractability.

3. **Validation is expensive and reactive.** Each proposed field is tested by running full RAG extraction + judge scoring on 3 documents. With ~10 proposals per cluster and ~8,900 clusters, the pipeline runs ~267K validation LLM calls just to discard ~64% of them. The validation works — it catches phantoms — but only after incurring the cost.

### Proposed Fixes (No Feedback Required)

**Fix A — Require evidence spans in the discovery prompt.** Change the prompt to require that each proposed field include a verbatim quote from the input text showing an extractable value. Fields without concrete evidence are flagged as hypotheses rather than confirmed discoveries. This is a prompt-only change to `FIELD_DISCOVERY_PROMPT` (L96) and `FieldDiscoverySig` docstring (L262).

**Fix B — Add few-shot contrast pairs to the discovery prompt.** Include 2–3 examples showing the distinction between a real field (with an extractable value) and a phantom field (concept mentioned but no discrete value). This teaches the LLM the boundary without additional infrastructure.

**Fix C — Set realistic count expectations.** Replace "aim for 10-15" with "propose only fields you can back with a direct quote; 3-7 high-quality fields are better than 15 speculative ones." Remove the "miss nothing" framing.

**Fix D — Deterministic span validation before LLM validation.** Before spending LLM calls on validation, run a zero-cost keyword/regex scan: for each proposed field, check whether a plausible value pattern appears in at least 2 of the sampled documents. Fields that fail the keyword check are pruned immediately — no LLM call needed.

### Expected Impact

Fixes A–C are prompt-only changes (~15 lines total in `field_discovery.py`). Fix D adds a lightweight pre-filter (~25 lines in `minimal_pipeline.py`). Together, they should reduce the phantom rate from ~64% to ~20-30%, cutting validation LLM spend roughly in half.

---

## 10. Tiered Discovery: Balancing Precision with Niche Field Recall

### The Tension

Tightening discovery (§9 above) risks losing **niche-but-real fields** — the unusual clause that matters to a specific contract type (e.g., "ROFR Exercise Period" in real estate, "Clawback Trigger Percentage" in executive comp). These fields are genuinely rare: they appear in few documents, have low recurrence, and look identical to phantom fields at proposal time after applying stricter prompts.

### Tiered Discovery Design

Rather than a single pass/fail threshold, split discovery into two tiers:

**Tier 1 — Grounded Fields (High-Confidence)**
- The LLM must provide a verbatim evidence span from the cluster's source text.
- These fields proceed directly to full extraction — no additional validation needed.
- Expected: 3–5 fields per cluster (the "obvious" ones).

**Tier 2 — Scout Fields (Exploratory Hypotheses)**
- The LLM flags fields it suspects exist but can't back with a direct quote.
- These do NOT go through the expensive full validation cycle.
- Instead, a **zero-cost deterministic keyword scan** runs across all documents in the cluster:
  - Build a simple pattern (field name + likely value formats: dates, percentages, monetary amounts, proper nouns).
  - If ≥2 documents contain matching patterns → promote to Tier 1 and extract.
  - If 0–1 documents match → **park** the field (don't discard, don't validate).
- Expected: 5–10 scout proposals per cluster, ~30% promoted, ~70% parked.

**Parked Fields**
- Stored in the cluster metadata as "suggested fields" visible via the API.
- Surfaced to users as "We noticed these might exist — would you like us to look for them?"
- When a user approves a parked field, it's promoted to Tier 1 and extracted on the next run.
- When a user rejects, the field is suppressed for future discovery in that cluster.
- This creates a **low-cost feedback channel** — users train the system by triaging cheap suggestions rather than correcting expensive extraction errors.

### Why This Works

The key insight: **phantom fields and niche fields look identical at proposal time** (both lack strong evidence), but they're **trivially separable** by a keyword scan. A phantom field like "Strategic Alliance Framework" has no extractable value anywhere in the documents — the keyword scan finds nothing. A niche field like "ROFR Exercise Period" will have specific values (dates, day counts) in the 2–3 documents where it genuinely applies — the keyword scan catches it.

### Feedback Amplification

The tiered system creates a natural feedback loop that improves over time:

1. **Promoted scouts** that yield good extractions → their weight increases → they're proposed as Tier 1 in similar clusters next run.
2. **User-approved parked fields** → added to the approved entities list → fast-tracked in future runs.
3. **User-rejected parked fields** → added to the "known phantom" list → suppressed in discovery for similar clusters.
4. **Zero user action** → parked fields remain parked, costing nothing.

This means the system starts conservative (only grounded fields extracted, speculative ones parked) and broadens coverage as feedback accumulates — the opposite of the current approach, which starts broad and wastes spend pruning.
