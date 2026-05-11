"""Rule candidate miner.

Reads coverage + field distribution stats and inserts draft rules. The caller
prepares a `corpus` dict with two keys:

  domain_clusters: {domain_name: {cluster_label: count, ...}}
  field_values:    {"<domain>::<cluster>::<field>": [v1, v2, ...]}    # values may be None

In production the caller assembles this from the existing DuckDB
(`agreements`, `domains`, `clusters`, `cluster_assignments`, `extractions`).
The seam is intentional so the miner is unit-testable without a clustering run.
"""
from __future__ import annotations
import statistics
from typing import Any

from .store import PlaybookStore


COVERAGE_THRESHOLD = 0.90       # cluster present in ≥90% of domain → expected (large N)
NUMERIC_QUANTILE_LO = 0.10      # use p10/p90 as suggested rule bounds
NUMERIC_QUANTILE_HI = 0.90
MIN_OBSERVATIONS = 5
CATEGORICAL_MODE_THRESHOLD = 0.70  # dominant string value must appear in ≥70% of non-null obs
OUTLIER_THRESHOLD = 0.10           # cluster outlier_pct must be ≥10% to emit a rule


def _coverage_threshold_for(n_total: int) -> float:
    """Auto-scale coverage threshold by corpus size.

    With small N a strict 90% floor is impractical (5 docs → must hit 5/5).
    For tiny corpora we relax to a majority floor; the threshold approaches
    COVERAGE_THRESHOLD asymptotically as N grows.
      N=1..5:   0.60      (majority of a handful)
      N=6..10:  0.70
      N=11..30: 0.80
      N>=31:    0.90      (statistical confidence)
    """
    if n_total <= 5:
        return 0.60
    if n_total <= 10:
        return 0.70
    if n_total <= 30:
        return 0.80
    return COVERAGE_THRESHOLD


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
    return xs[k]


def mine_candidates(store: PlaybookStore, playbook_id: str,
                     corpus: dict[str, Any], *,
                     n_total_per_domain: int) -> list[dict]:
    cands: list[dict] = []

    # ── coverage candidates ──────────────────────────────────────────────
    cov_threshold = _coverage_threshold_for(n_total_per_domain)
    for domain, clusters in (corpus.get("domain_clusters") or {}).items():
        for cluster_label, n in clusters.items():
            ratio = n / max(1, n_total_per_domain)
            if ratio < cov_threshold:
                continue
            rid = store.create_rule(
                playbook_id=playbook_id,
                title=f"{domain.upper()} must contain {cluster_label}",
                description=(f"{ratio:.0%} of {domain} agreements in the corpus "
                             f"contain a {cluster_label} clause."),
                applies_to="domain",
                severity="warn",
                predicate={"op": "any_of", "args": [
                    {"op": "clause.classified_as", "args": [cluster_label]}
                ]},
                tags=[domain, "coverage"],
                source_provenance={"miner": "coverage", "ratio": ratio, "n": n},
                status="draft",
            )
            store.add_binding(rule_id=rid, entity_kind="domain",
                               entity_id=f"label:{domain}", label_text=domain, confidence=0.5)
            cands.append({"kind": "coverage", "rule_id": rid, "ratio": ratio})

    # ── distribution candidates (numeric only for now) ───────────────────
    for key, values in (corpus.get("field_values") or {}).items():
        nums = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        if len(nums) < MIN_OBSERVATIONS:
            continue
        domain, cluster_label, field_name = key.split("::", 2)
        lo = _quantile(nums, NUMERIC_QUANTILE_LO)
        hi = _quantile(nums, NUMERIC_QUANTILE_HI)
        median = statistics.median(nums)
        rid = store.create_rule(
            playbook_id=playbook_id,
            title=f"{cluster_label}::{field_name} between p10 and p90",
            description=(f"Corpus distribution: median={median}, "
                         f"p10={lo}, p90={hi} ({len(nums)} samples)."),
            applies_to="field",
            severity="info",
            predicate={"op": "field.between", "args": [field_name, lo, hi]},
            answer_type="numeric_range",
            tags=[domain, cluster_label, "distribution"],
            source_provenance={"miner": "distribution", "n": len(nums),
                                "median": median, "lo": lo, "hi": hi},
            status="draft",
        )
        store.add_binding(rule_id=rid, entity_kind="field",
                           entity_id=f"label:{field_name}", label_text=field_name, confidence=0.5)
        cands.append({"kind": "distribution", "rule_id": rid,
                       "median": median, "lo": lo, "hi": hi})

    # ── categorical mode candidates ──────────────────────────────────────
    for key, values in (corpus.get("field_values") or {}).items():
        non_null = [v for v in values if v is not None]
        if len(non_null) < MIN_OBSERVATIONS:
            continue
        # Predominantly strings: ≥80% non-numeric
        str_vals = [v for v in non_null if isinstance(v, str)]
        if len(str_vals) / len(non_null) < 0.80:
            continue
        # Find mode
        freq_map: dict = {}
        for v in str_vals:
            freq_map[v] = freq_map.get(v, 0) + 1
        mode_value = max(freq_map, key=lambda k: freq_map[k])
        freq = freq_map[mode_value] / len(non_null)
        if freq < CATEGORICAL_MODE_THRESHOLD:
            continue
        domain, cluster_label, field_name = key.split("::", 2)
        n_non_null = len(non_null)
        rid = store.create_rule(
            playbook_id=playbook_id,
            title=f"{cluster_label}::{field_name} is typically '{mode_value}'",
            description=(f"'{mode_value}' appears in {freq:.0%} of observations "
                         f"({n_non_null} samples)."),
            applies_to="field",
            severity="info",
            predicate={"op": "field.eq", "args": [field_name, mode_value]},
            answer_type="enum",
            answer_options=[mode_value, "Other"],
            tags=[domain, cluster_label, "categorical"],
            source_provenance={"miner": "categorical", "mode": mode_value,
                                "frequency": freq, "n": n_non_null},
            status="draft",
        )
        store.add_binding(rule_id=rid, entity_kind="field",
                           entity_id=f"label:{field_name}", label_text=field_name, confidence=0.5)
        cands.append({"kind": "categorical", "rule_id": rid,
                       "mode": mode_value, "frequency": freq})

    # ── outlier candidates ───────────────────────────────────────────────
    for cluster_label, entry in (corpus.get("cluster_outliers") or {}).items():
        outlier_pct = entry.get("outlier_pct", 0.0)
        n = entry.get("n", 0)
        if outlier_pct < OUTLIER_THRESHOLD or n < MIN_OBSERVATIONS:
            continue
        reference_text = entry.get("reference_text", "")
        rid = store.create_rule(
            playbook_id=playbook_id,
            title=f"{cluster_label}: outlier language detected",
            description=(f"{outlier_pct:.0%} of {n} clauses deviate from the "
                         f"reference pattern for {cluster_label}."),
            applies_to="cluster",
            severity="info",
            reference_text=reference_text,
            similarity_threshold=0.85,
            tags=[cluster_label, "outlier"],
            source_provenance={"miner": "outlier", "outlier_pct": outlier_pct, "n": n},
            status="draft",
        )
        store.add_binding(rule_id=rid, entity_kind="cluster",
                           entity_id=f"label:{cluster_label}", label_text=cluster_label,
                           confidence=0.5)
        cands.append({"kind": "outlier", "rule_id": rid,
                       "outlier_pct": outlier_pct, "n": n})

    return cands


def soft_rebind(store: PlaybookStore, *, embed,
                 cluster_centroids: dict[str, dict],
                 threshold: float = 0.85,
                 entity_kind: str | None = None) -> int:
    """For each binding with unresolved 'label:' entity_id (and matching entity_kind
    if given), find the best-matching candidate by cosine similarity on label
    embeddings.

    cluster_centroids = {entity_id: {"label": str, "embedding": list[float]?}}
      — "embedding" is optional; if absent we embed(label) on the fly.
    entity_kind: filter to bindings of this kind (e.g. "cluster", "domain", "field").
      None matches all kinds (legacy behavior).
    threshold: minimum cosine to accept. The updated binding's confidence is set
      to the matched cosine value.
    Returns number of bindings updated.
    """
    import numpy as np
    updated = 0
    if entity_kind:
        rows = store.conn.execute(
            "SELECT binding_id, label_text FROM rule_bindings "
            "WHERE entity_id LIKE 'label:%' AND label_text IS NOT NULL AND entity_kind=?",
            [entity_kind],
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT binding_id, label_text FROM rule_bindings "
            "WHERE entity_id LIKE 'label:%' AND label_text IS NOT NULL"
        ).fetchall()
    if not rows or not cluster_centroids:
        return 0
    candidate_items = list(cluster_centroids.items())
    # Cache candidate embeddings — use provided "embedding" if present, else embed(label).
    candidate_vecs = []
    for _, c in candidate_items:
        if c.get("embedding") is not None:
            candidate_vecs.append(np.asarray(c["embedding"]))
        else:
            candidate_vecs.append(np.asarray(embed(c["label"])))
    candidate_vecs = np.asarray(candidate_vecs)
    if candidate_vecs.size == 0:
        return 0
    candidate_norms = np.linalg.norm(candidate_vecs, axis=1) + 1e-9

    for bid, label in rows:
        v = np.asarray(embed(label))
        if v.size == 0:
            continue
        cos = (candidate_vecs @ v) / (candidate_norms * (np.linalg.norm(v) + 1e-9))
        best = int(np.argmax(cos))
        if float(cos[best]) >= threshold:
            cid, _ = candidate_items[best]
            store.conn.execute(
                "UPDATE rule_bindings SET entity_id=?, confidence=? WHERE binding_id=?",
                [cid, float(cos[best]), bid],
            )
            updated += 1
    return updated
