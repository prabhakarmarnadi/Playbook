# Next improvements — borrowed patterns from `keyBERTClause`

**Date:** 2026-05-11
**Source codebase analyzed:** `/mnt/data/code/keyBERTClause`
**Target codebase:** `/mnt/data/code/clustering/.claude/hackathon/` (this repo)
**Status:** Backlog. No code changes landed from this document yet.

---

## TL;DR

`keyBERTClause` is the more mature predecessor of our ARM stack. Same library
(`mlxtend.fpgrowth`), same rule_id conventions, same threshold structure. The
delta is keyBERTClause has built out the field analytics, contrastive mining,
deviation scoring, and visualization layers we never finished. This file is a
prioritized backlog of patterns to borrow, with file:line citations and effort
estimates.

---

## A. Repo orientation

`keyBERTClause` is a fully-realized legal-contract analysis pipeline operating
on ~1,588 contracts and 32 hand-curated CUAD clause types. It is a mature
single-author research codebase organized in 6 phases driven by
`scripts/run_pipeline.py` and shell scripts. Code organization:

- `src/core/` — loader, encoder, clustering, deviation scorer
- `src/arm/` — two FP-Growth miners (clause-level + field-level), `field_value_analyzer.py`
- `src/keybert/` — zero-shot fingerprints
- `src/extraction/` — regex + context fallback extractor, end-to-end inference pipeline
- `src/classification/` — optional DeBERTa classifier
- `src/graph/` — dataclass-typed knowledge graph + Cypher export

The receiving project (this repo) already has a stripped-down clone of the ARM
portion under `core/arm/{arm_miner,field_arm_miner,configs}.py`, lifted from
this codebase. The patterns below are the parts that didn't make the cut and
that are now worth reconsidering.

---

## B. High-ROI borrows (ranked by effort/value)

### 1. Numeric-bounds field registry — **S**

**Source:** `src/arm/field_value_analyzer.py:22-123`

A 60+ entry dict `NUMERIC_FIELD_BOUNDS` mapping field names to `(min, max)`
tuples, plus `validate_numeric_value()` with suffix-based fallback bounds:

```python
NUMERIC_FIELD_BOUNDS = {
    "notice_period_days":          (0, 365),
    "cure_period_days":            (0, 180),
    "survival_period_years":       (0, 25),
    "ownership_threshold_percent": (0, 100),
    "number_of_arbitrators":       (1, 9),
    ...
}
# Pattern-based fallback for unrecognized fields:
if any(p in field_name for p in ["_days", "_period_days"]):
    return 0 <= value <= 730
if "_hours" in field_name:  return 0 <= value <= 720    # max 30 days
if "_months" in field_name: return 0 <= value <= 120    # max 10 years
if "_years" in field_name:  return 0 <= value <= 50
if "_percent" in field_name: return 0 <= value <= 100
```

**Why:** Pre-filters LLM extraction errors (numbers in thousands when the field
should be in days; non-numeric junk for numeric fields) before they pollute
bucketing or ARM. We currently have nothing — bad extractions silently propagate.

**Where to graft:**

- New module `core/playbooks/field_validation.py`, OR
- Fold into existing `core/field_normalizer.py`
- Call before quantile computation in `core/playbooks/miner.py:87` so p10/p90
  aren't skewed by outliers.

---

### 2. Range-derived numeric bucketing — **S**

**Source:** `src/arm/field_value_analyzer.py:126-145`

```python
def create_numeric_buckets(min_val, max_val):
    if max_val <= 10:    return [(0,1),(2,3),(4,5),(6,10)]
    elif max_val <= 30:  return [(0,7),(8,14),(15,30)]
    elif max_val <= 100: return [(0,25),(26,50),(51,75),(76,100)]
    elif max_val <= 365: return [(0,30),(31,90),(91,180),(181,365)]
    elif max_val <= 1000:return [(0,100),(101,250),(251,500),(501,1000)]
    else:                return [(0,100),(101,1000),(1001,10000),(10001,inf)]
```

**Why:** Our `core/arm/field_arm_miner.py` uses one fixed bin set
`[0,7,30,60,90,180,365,inf]` regardless of field (`core/arm/configs.py:42-43`).
That's sensible for "\_days" fields but wrong for years, percents, or counts.
Range-derived buckets are a 1-day port and immediately improve cross-field
reasoning.

**Where to graft:**

- `core/arm/field_arm_miner.py::normalize_field_value` — swap fixed bins for
  `create_numeric_buckets(min_val, max_val)` keyed off the observed range.
- `core/playbooks/miner.py` — emit distribution rules with categorical buckets
  in addition to the existing numeric `field.between` predicate.

---

### 3. Contrastive rules: cluster-lift vs global-lift — **M**

**Source:** `src/arm/arm_miner.py:319-350`

```python
def find_contrastive_rules(global_rules, cluster_rules):
    for cid, rules in cluster_rules.items():
        for rule in rules:
            cluster_lift = rule["lift"]
            global_lift = global_rules.get(rule_key, {}).get("lift", 0.0)
            if cluster_lift > 2.0 and (
                global_lift < 1.5 or cluster_lift / max(global_lift, 0.01) > 2.0
            ):
                contrastive.append({
                    "cluster_id": cid,
                    "antecedent": rule["antecedent"],
                    "consequent": rule["consequent"],
                    "cluster_lift": cluster_lift,
                    "global_lift": global_lift,
                    "lift_ratio": round(cluster_lift / max(global_lift, 0.01), 2),
                    ...
                })
```

**Why:** Direct upgrade for our playbook miner. We currently emit per-cluster
rules but have no way to surface "this rule is special to this cluster vs the
corpus baseline." Cluster-discriminative rules are exactly the kind of playbook
material a lawyer wants — it tells them what's unique about a contract family.
We already produce both global + per-cluster ARM rules; just need the
comparison.

**Where to graft:**

- New method `core/arm/arm_miner.py::find_contrastive_rules(global, per_cluster)`.
- `core/playbooks/miner.py::mine_candidates` adds a `"contrastive"` candidate
  kind alongside coverage/distribution/categorical/outlier.
- Maps to `predicate={"op":"clause.classified_as", ...}` with
  `tags=["contrastive", cluster_label]` and `source_provenance={"lift_ratio":...}`.

---

### 4. Cross-clause field rules at lower thresholds — **M**

**Source:** `src/arm/field_arm_miner.py:522-583`

Dedicated FP-Growth pass at much looser thresholds:

```json
{
  "global": { "support": 0.01, "conf": 0.5, "lift": 1.2, "max_rules": 1000 },
  "cross_clause": {
    "support": 0.005,
    "conf": 0.4,
    "lift": 1.3,
    "max_rules": 500
  }
}
```

Post-filters to rules where `cons_clauses - ant_clauses` is non-empty (the
consequent introduces a clause type not in the antecedent). Output rule_id
prefix `XC` and adds `antecedent_clauses` / `consequent_clauses` sets.

**Why:** Our `mine_field_rules` (`core/arm/field_arm_miner.py:68-107`) lumps
both same-clause and cross-clause rules into one output, distinguishable only
via the `rule_type` tag. The keyBERTClause approach is cleaner: dedicated
mining at lower thresholds (because cross-clause patterns are rarer),
filtered post-hoc to those that genuinely cross clause boundaries.

The README justifies the lower numbers: "Field extractions are sparse (only
0.89% matrix fill rate) and cross-clause patterns are inherently rare."
(`CLAUDE.md:498`).

**Where to graft:**

- Add `mine_cross_clause_rules()` next to `mine_field_rules` in
  `core/arm/field_arm_miner.py`.
- Wire to `core/playbooks/miner.py` as a new `"cross_clause"` candidate kind.

---

### 5. Field-value type detection + diverse-quartile examples — **M**

**Source:** `src/arm/field_value_analyzer.py:257-359`

`analyze_value_distribution()` heuristically classifies a field as
bool/numeric/text by value-population ratio:

```python
bool_values = [v for v in values if isinstance(v, bool)]
if len(bool_values) > len(values) * 0.8:
    return analyze_boolean_distribution(values)
numeric_values = [...]
if len(numeric_values) > len(values) * 0.5:
    return analyze_numeric_distribution(numeric_values, values)
return analyze_text_distribution(values)
```

`collect_diverse_examples()` samples from different parts of the distribution
— for numeric it picks indices `[0, n/4, n/2, 3n/4, n-1]`; for text it groups
by value and takes one example from each top group; for bool it picks 2 of each
truth value.

**Why:** Our playbook miner candidates carry `source_provenance` with only `n`,
`median`, `lo`, `hi`. A lawyer reviewing a draft rule can't validate a
threshold without seeing 5 representative values. Sampling diverse examples
avoids the "all 5 examples are 30 days" failure mode.

**Where to graft:**

- Augment `core/playbooks/miner.py::mine_candidates` to attach
  `examples: [{doc_id, value, display}]` to each candidate's
  `source_provenance`.
- Reuse `collect_diverse_examples` logic.
- Persist either in `playbook_rules.source_provenance` JSON or a sidecar
  column.

---

## C. Secondary borrows

### 6. Lift-based cluster naming — S

**Source:** `src/core/clause_clustering.py:88-106`

```python
ratios = {}
for ct in ALL_CLAUSE_TYPES:
    cp = cluster_prevalence.get(ct, 0)
    gp = global_prevalence.get(ct, 0.01)
    if cp >= 0.3:                # only consider clauses present in ≥30% of cluster
        ratios[ct] = cp / gp     # lift over global baseline
top = sorted(ratios.items(), key=lambda x: -x[1])[:3]
return " + ".join(ct for ct, _ in top)
```

Produces names like `"Payment Terms"`, `"SLA + Payment Terms"`,
`"Data Breach + Security + Audit"`.

**Why:** Deterministic fallback when LLM labeling is unavailable, AND a useful
prompt augmentation for the LLM labeler ("name this cluster; hint:
distinguishing features are X, Y, Z").

**Where:** `core/macro_clusterer.py` or `core/micro_clusterer.py` — add a
`compute_distinguishing_features(cluster_id, all_clauses)` helper, call before
LLM labeling.

---

### 7. Prevalence categorization — S

**Source:** `src/core/clause_clustering.py:109-118`

```python
if rate >= 0.8: return "expected"
elif rate >= 0.5: return "common"
elif rate >= 0.2: return "occasional"
else:             return "rare"
```

Used to drive missing-clause anomaly severity (`arm_miner.py:215`).

**Why:** Our miner has `COVERAGE_THRESHOLD=0.90` (with auto-scaling
`_coverage_threshold_for(n_total)`), but no intermediate tiers. A 4-tier
categorization lets us emit different candidate severities:

- `expected` → `severity=warn` if missing
- `common` → `severity=info` if missing
- `occasional` → no rule
- `rare` → `severity=info` if present (unusual-combination flag)

**Where:** `core/playbooks/miner.py::mine_candidates` — emit the unusual-
combination kind alongside coverage.

---

### 8. Externalized deviation weights + portfolio report — M

**Source:** `src/core/deviation_scorer.py:108-318` + `configs/deviation_weights.json`

Per-flag weighted score with multipliers:

```json
{
  "missing_arm_consequent": 3.0,
  "missing_expected": 2.0,
  "unexpected_clause": 1.5,
  "rare_combination": 1.0,
  "prevalence_gap_multiplier": 2.0,
  "confidence_multiplier": 1.5,
  "lift_multiplier": 0.5
}
```

Severity tiers: `score ≥ 5.0 → high`, `≥ 2.5 → medium`, `> 0 → low`. Portfolio
report (`generate_portfolio_report`, `deviation_scorer.py:249-318`) summarizes
by severity bucket, top-10% outliers, most-commonly-missing clauses, and per-
cluster mean/max/high-count.

**Why:** Every score point traces to a flag with a `rule_id`, making findings
explainable. UI can show "doc scored 7.3 because rule R47 (lift 2.8) + rule
R12 (lift 1.9) + ...".

**Where:** Extend `core/playbooks/aligner.py` or `core/risk_scorer.py` to
attach `severity_weight` per flag and emit a portfolio-report-shaped JSON.

---

### 9. Single-file HTML dashboard — S/M

**Source:** `src/arm/field_arm_miner.py:625-1151`
(`generate_field_matrix_viewer`)

500-line f-string that bakes a Tailwind + Plotly dashboard combining summary
stats, clause coverage bars, top-20 field-value items, co-occurrence pair lists
(Jaccard), top association rules, per-clause expandable field distributions,
and a Plotly bar chart of cluster sizes. Zero build step.

**Why:** Our subsystem has Streamlit (`app.py`) for interactive use, but the
keyBERTClause approach (static single-file HTML) is great for point-in-time
playbook artifacts that ship as deliverables — emailable, archivable,
re-renderable from a JSON snapshot.

**Where:** New `core/playbooks/export_html.py`. If we adopt it, refactor to
Jinja first — the literal f-string is hard to maintain.

---

### 10. Pattern + anti-keyword + bounds regex extractor — L (low priority)

**Source:** `src/extraction/context_aware_extractor.py:67-528`

Each of 22 field configs has multiple ranked regex patterns, `context_keywords`
for fallback, `anti_keywords` to reject false positives, `unit`, and `bounds`
for value-validity. Plus `WORD_TO_NUM` map ("twelve"→12 etc., lines 22-32).

**Why:** Our extraction is LLM-first, so the regex registry is low priority.
BUT two sub-patterns are worth pulling:

1. **`anti_keywords` cross-validator** — defensive post-extraction sanity check
   ("the field is `notice_period_days` but the evidence span contains
   'effective date'") rejects LLM hallucinations.
2. **`WORD_TO_NUM` normalizer** — useful in `core/field_normalizer.py` as
   a defensive layer for fields like `"twelve months"` → `12`.

**Where:** `core/field_normalizer.py` — pull just `WORD_TO_NUM` and
`_check_anti_keywords()`. Skip the rest of the regex registry.

---

### 11. End-to-end inference report shape — S

**Source:** `src/extraction/inference_pipeline.py:330-422`

`InferencePipeline.process_contract` emits one JSON per document combining
`clause_types`, `clause_confidences`, `cluster_assignment`, `deviation_score`,
`severity`, `violations`, `violation_count`, `paragraphs_processed`,
`total_paragraphs`.

**Why:** Good template for our `core/playbooks/aligner.py` per-document
evaluation output.

**Where:** `core/playbooks/aligner.py` output format.

---

### 12. Cluster-conditional contrastive rule structure — M

Combines patterns 3 + 7. Synthesize cluster-aware coverage candidates whose
`source_provenance` includes `global_baseline_pct` so a single rule record
knows it's BOTH cluster-coverage-significant AND cluster-distinguishing.

**Why:** Lawyers want "this clause matters here because it's both common in
this contract type AND unusual elsewhere."

**Where:** `core/playbooks/miner.py` — extend the coverage candidate path.

---

## D. Patterns to AVOID porting

1. **`ALL_CLAUSE_TYPES` as a frozen Python constant** —
   `src/core/data_loader.py:30-43` hard-codes the 32 CUAD types. Our entire
   subsystem is built on _discovering_ clause types per account; importing the
   literal list would be a regression. Only the _idea_ of having a per-account
   canonical list (which we already have via `clusters` table) is borrowable.

2. **`FIELD_TO_CLAUSE_TYPE` hard-coded mapping** —
   `src/arm/field_arm_miner.py:36-78`. We have per-cluster field discovery via
   `field_definitions.cluster_id` already; reintroducing the dictionary would
   break per-account discovery.

3. **Their `min_support=0.005` ARM thresholds** —
   `configs/field_arm_thresholds.json`. Works on 1,588-doc corpora; would
   re-introduce the OOM hang we just fixed for 5-doc corpora. Keep our
   `ARM_MIN_TRANSACTIONS=20` floor (`core/arm/arm_miner.py:28`).

4. **The 500-line f-string HTML generator (as-is)** —
   `field_arm_miner.py:625-1151`. Useful pattern, but the literal HTML uses
   Tailwind CDN, inline Plotly, and hard-coded color cycling. If adopted,
   refactor to Jinja template + extract the color/CSS constants.

5. **Their `detect_anomalies` flag dedup logic** —
   `arm_miner.py:213-281`. Same missing clause gets flagged by overlapping
   rules; partial dedup only via `already_flagged` (between ARM and baseline
   flags, not across overlapping ARM rules). If we port, dedupe per
   `(missing_clause, doc_id)` at flag generation, not at scoring.

6. **DeBERTa classifier path** — `src/classification/clause_classifier.py`
   referenced from `inference_pipeline.py`. Assumes a fine-tuned multi-label
   classifier exists. We replace this with cluster assignment via cosine
   similarity + LLM labeling, which is the right call for unknown taxonomies.
   Don't port the classifier hook.

7. **Bool detection by simple ratio** — `field_value_analyzer.py:264`.
   Python `True/False` are also `int` (1/0), so the analysis order matters.
   keyBERTClause handles it by checking bool _before_ numeric. Reuse but
   preserve the check order.

---

## E. Recommended sequencing

### Phase 1 — pure correctness/quality wins, no architectural change (≈1 day)

- [ ] **#1** — Land `NUMERIC_FIELD_BOUNDS` + suffix-based fallback as a
      `core/field_normalizer.py` extension. Call from
      `core/playbooks/miner.py::mine_candidates` before p10/p90 computation.
- [ ] **#2** — Replace the fixed-bin scheme in
      `core/arm/field_arm_miner.py::normalize_field_value` with
      `create_numeric_buckets()`.
- [ ] **#6** — Add lift-based cluster-naming helper
      `compute_distinguishing_features()` in `core/macro_clusterer.py`.

### Phase 2 — playbook miner upgrades (≈2-3 days)

- [ ] **#3** — `mine_contrastive_rules()` in `core/arm/arm_miner.py` +
      `"contrastive"` candidate kind in `core/playbooks/miner.py`.
- [ ] **#4** — `mine_cross_clause_rules()` in `core/arm/field_arm_miner.py` +
      `"cross_clause"` candidate kind in `core/playbooks/miner.py`.
- [ ] **#7** — Prevalence categorization (`expected/common/occasional/rare`)
      for differentiated coverage rule severities.

### Phase 3 — observability + HITL polish (≈2 days)

- [ ] **#5** — Attach `examples: [{doc_id, value, display}]` to every miner
      candidate via `collect_diverse_examples` logic. Critical for HITL review.
- [ ] **#8** — Externalize deviation weights to
      `configs/playbook_weights.json` so the Findings page can show per-point
      traceability.
- [ ] **#11** — Adopt the per-document inference report shape in
      `core/playbooks/aligner.py` output.

### Phase 4 — optional polish (≈1 day each)

- [ ] **#9** — `core/playbooks/export_html.py` (Jinja-templated, not f-string).
- [ ] **#10** — Pull `WORD_TO_NUM` + `_check_anti_keywords()` defensive layer
      into `core/field_normalizer.py`.
- [ ] **#12** — Cluster-conditional contrastive rule structure (synthesis of
      #3 + #7).

**Total backlog:** ~5-7 days of borrows that meaningfully strengthen the
playbook subsystem without architectural churn.

---

## F. References

- Spec for the existing playbook subsystem:
  `docs/superpowers/specs/2026-05-09-playbook-rules-design.md`
- Implementation plan:
  `docs/superpowers/plans/2026-05-09-playbook-rules.md`
- Current ARM code: `core/arm/{arm_miner,field_arm_miner,configs}.py`
- Current playbook miner: `core/playbooks/{miner,miner_runner,aligner}.py`
- Source codebase analyzed: `/mnt/data/code/keyBERTClause`
