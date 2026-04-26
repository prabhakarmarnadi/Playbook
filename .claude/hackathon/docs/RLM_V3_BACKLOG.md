# RLM V3 — Backlog Improvements

> Created 2026-04-21 after implementing core improvements informed by three empirical RLM studies.
> All 7 architecture concerns from the articles are addressed in `core/rlm_v3.py`.

---

## Completed

| # | Improvement | File(s) | Concern |
|---|------------|---------|---------|
| ✅ | State budget + pruning (`_prune_state`, `state_field_budget` config) | `rlm_v3.py` | 1 — State Shape > State Size |
| ✅ | Dense self-critique tool (`critique_fields()` with 5-axis rubric) | `rlm_v3.py` | 2 — Dense Feedback for RLM |
| ✅ | Task-aligned critic (scores extraction-ready signals, not style) | `rlm_v3.py` | 3 — Judge ≠ Task Success |
| ✅ | Token budget + efficiency early-exit (metering, `extraction_only_mode`) | `rlm_v3.py` | 4 — Efficiency and Early Exit |
| ✅ | Architecture doc as top-level script comment (7 concerns mapped) | `rlm_v3.py` | All |
| ✅ | MIPRO compilation support (`compile_with_mipro`, `load_compiled`) | `rlm_with_memory.py` | 7 — Prompt Optimization Guard |

---

## High Impact — Wire Existing Features End-to-End

### 1. Surface V3 efficiency metrics in production pipeline
**Status:** Not started
**Files:** `production_pipeline.py` → `_discover_fields_rlm_v3`, `_discover_and_extract_rlm_v3`
**Description:** The V3 RLM now tracks token consumption, efficiency warnings, pruning counts, and extraction-only mode — but none of these flow into the pipeline manifest. The `rlm_meta` dict returned by V3 contains all the data; it just needs to be captured in `_process_cluster()` stats and surfaced in the final `pipeline_manifest.json`.
**Effort:** Small — plumb existing data through existing stats dict.

### 2. MIPRO compilation CLI integration
**Status:** Not started
**Files:** `rlm_with_memory.py`, `production_pipeline.py` (or new standalone script)
**Description:** `compile_with_mipro()` is implemented but has no entry point. Need:
- A `--compile` CLI flag on the pipeline (or a `scripts/compile_rlm.py`)
- A path to a gold standard JSON file (`eval/gold_standard.json` or CUAD-derived)
- Persistence of compiled weights (`.pkl` or DSPy checkpoint)
- A `--load-compiled` flag to use pre-optimized prompts at runtime
**Effort:** Medium — needs CLI args, file I/O for weights, and integration testing.

### 3. Extraction failure → discovery re-weight
**Status:** Not started
**Files:** `production_pipeline.py` → `_process_cluster()`, `rlm_v3.py` → `critique_fields()`
**Description:** After extraction, compute per-field null rates. Fields with >70% null across chunks are likely badly specified. Options:
- Auto-demote confidence in `RollingStateV3` and re-run `_prune_state()`
- Flag in `critique_fields()` so the agent can self-correct
- Log to pipeline manifest for human review
This closes the discovery↔extraction feedback loop — currently the two stages are fire-and-forget.
**Effort:** Medium — needs post-extraction analysis + feedback channel back to state.

---

## Medium Impact — New Capabilities

### 4. Dynamic temperature scheduling
**Status:** Not started
**Files:** `rlm_v3.py` → `_V3RLM._make_llm_tools()` → `llm_explore()`
**Description:** `llm_explore()` always uses temperature 0.8. Early iterations benefit from high recall (high temp), but later iterations should converge (low temp). Simple schedule:
```python
temp = max(0.4, 0.8 - 0.03 * iteration)
```
Could also tie temperature to the convergence delta — if delta is large (lots of new fields), keep high; if delta is small, drop.
**Effort:** Small — one-line change + config param.

### 5. Cluster-aware field inheritance
**Status:** Not started
**Files:** `production_pipeline.py`, `rlm_v3.py`
**Description:** Sibling clusters in the same domain (e.g., "Termination" and "Renewal" under "Service Agreements") likely share common fields like `effective_date`, `governing_law`, `party_names`. Instead of discovering these from scratch per cluster:
1. After the first few clusters in a domain finish, identify shared fields (appear in ≥2 clusters)
2. Inject these as a "warm start" field set into subsequent clusters' `RollingStateV3`
3. The agent still validates/prunes them, but saves exploration tokens
**Effort:** Medium — needs cross-cluster state sharing in the pipeline loop.

### 6. Cross-cluster batch critique
**Status:** Not started
**Files:** `production_pipeline.py` (new post-discovery stage), `ontology_refiner.py`
**Description:** After all clusters finish discovery, run a global consistency pass:
- Flag field name collisions with different types (e.g., `notice_period` as both `duration` and `text`)
- Suggest standardized names across clusters
- Detect orphan fields that appear in only 1 cluster with low confidence
This is complementary to `OntologyRefiner.deduplicate_fields()` which handles embedding-level dedup.
**Effort:** Medium — new pipeline stage, possibly LLM-assisted.

### 7. Adaptive max_iterations
**Status:** Not started
**Files:** `rlm_v3.py` → `_V3RLM._execute_iteration()`
**Description:** The `state_deltas` list already tracks convergence. Instead of running to a fixed `max_iterations=25`, dynamically stop when:
```python
if len(state_deltas) >= 3 and all(d < 0.05 for d in state_deltas[-3:]):
    # Signal early stop — no new fields for 3 iterations
```
The coverage_report() already computes `converging` but the REPL agent decides when to stop. Could inject a stronger stop signal or reduce remaining iterations.
**Effort:** Small — logic exists, just needs wiring into the iteration cap.

---

## Lower Impact — Polish

### 8. Cost tracking per cluster
**Status:** Not started
**Files:** `production_pipeline.py`, `rlm_v3.py`
**Description:** Use `state.tokens_consumed` from V3 + model pricing table to estimate $/cluster. Surface in pipeline manifest as `estimated_cost_usd`. Helps with budgeting and identifying expensive clusters.
**Effort:** Small — arithmetic on existing counters.

### 9. RAG-enhanced discovery
**Status:** Not started
**Files:** `rlm_v3.py`, `rag_extractor.py`
**Description:** Currently RAG is extraction-only (`extraction_backend: "rag"`). For sparse clusters with very few sample chunks (<5), use LanceDB retrieval to pull similar chunks from other clusters/agreements to augment the discovery sample. This increases the signal for rare clause types.
**Effort:** Medium — needs retrieval integration into V3 discovery, careful filtering.

### 10. Agreement-level field registry
**Status:** Not started
**Files:** `production_pipeline.py`, `store.py` (new table)
**Description:** After extraction, compute field frequency across agreements:
- **Universal fields** (>80% of agreements): `effective_date`, `governing_law`
- **Common fields** (30-80%): `termination_notice_period`, `renewal_term`
- **Rare fields** (<30%): agreement-specific provisions
This informs field importance scoring and helps downstream consumers prioritize which fields to surface in UIs.
**Effort:** Medium — needs a new DB table + post-extraction aggregation query.

---

## Source Articles

These improvements are grounded in findings from:

1. **"Sparse Signal Loop"** (stochi0, Apr 2026) — 2×2 chat vs RLM × dense vs sparse feedback
2. **"RLMs Are the New Reasoning Models"** (Weitekamp, Apr 2026) — Survey of RLM results across benchmarks
3. **"Recursive Language Models as Memory Systems"** (Weitekamp, Feb 2026) — DSPy.RLM + Pydantic on LongMemEval
