"""Aggregates ClusteringStore tables into a `corpus` dict that mine_candidates consumes."""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any

from core.playbooks.miner import mine_candidates
from core.playbooks.store import PlaybookStore


# Match the first signed-number literal in a string (handles "$1,000", "30 days",
# "12.5%", "USD 250000", etc.). Returns None if no numeric literal is found.
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _coerce_numeric(value):
    """Return (kind, value) where kind ∈ {'numeric','string','none'}.
    LLM extractions arrive as strings — even for numbers (e.g. '30 days', '$1,000').
    We pull the first numeric literal so distribution mining can use them."""
    if value is None:
        return "none", None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "numeric", float(value)
    if isinstance(value, bool):
        return "string", str(value)
    s = str(value).strip()
    if not s:
        return "none", None
    # Try direct float first (avoids matching just the integer part of "30.5 days")
    try:
        return "numeric", float(s.replace(",", ""))
    except ValueError:
        pass
    m = _NUM_RE.search(s)
    if m:
        try:
            return "numeric", float(m.group(0).replace(",", ""))
        except ValueError:
            pass
    return "string", s


def build_corpus(clustering_store) -> tuple[dict, dict[str, int]]:
    """Returns (corpus, domain_totals). Domain totals = {domain_label: count_agreements}."""
    conn = clustering_store.conn

    # Per-domain agreement count
    rows = conn.execute(
        "SELECT d.label, COUNT(*) "
        "FROM agreements a JOIN domains d ON a.domain_id = d.domain_id "
        "GROUP BY d.label"
    ).fetchall()
    domain_totals = {label: int(n) for label, n in rows if label}

    # Per-(domain, cluster) clause count via clauses → agreements → domains → clusters.
    # Fall back to per-domain × clusters if cluster_assignments table has no data.
    rows = conn.execute(
        "SELECT d.label AS domain_label, c.label AS cluster_label, COUNT(DISTINCT a.agreement_id) "
        "FROM clauses cl "
        "JOIN agreements a ON cl.agreement_id = a.agreement_id "
        "JOIN domains d ON a.domain_id = d.domain_id "
        "JOIN clusters c ON cl.clause_type_id = c.cluster_id "
        "GROUP BY d.label, c.label"
    ).fetchall()
    domain_clusters: dict[str, dict[str, int]] = defaultdict(dict)
    for d_label, c_label, n in rows:
        if d_label and c_label:
            domain_clusters[d_label][c_label] = int(n)

    # Per-(domain, cluster, field) value lists
    rows = conn.execute(
        "SELECT d.label AS domain_label, c.label AS cluster_label, "
        "       f.name AS field_name, e.value "
        "FROM extractions e "
        "JOIN field_definitions f ON e.field_id = f.field_id "
        "JOIN clusters c ON f.cluster_id = c.cluster_id "
        "JOIN agreements a ON e.agreement_id = a.agreement_id "
        "JOIN domains d ON a.domain_id = d.domain_id"
    ).fetchall()
    field_values: dict[str, list] = defaultdict(list)
    for d_label, c_label, fname, value in rows:
        if not (d_label and c_label and fname):
            continue
        key = f"{d_label}::{c_label}::{fname}"
        kind, coerced = _coerce_numeric(value)
        if kind == "none":
            field_values[key].append(None)
        else:
            field_values[key].append(coerced)

    corpus: dict[str, Any] = {
        "domain_clusters": dict(domain_clusters),
        "field_values": dict(field_values),
    }
    return corpus, domain_totals


def _build_rebind_candidates(clustering_store) -> tuple[dict, dict, dict]:
    """Read the ontology and produce candidate maps keyed by real entity_id for
    soft_rebind to match against. Values are {label: str} (embedding computed
    lazily by soft_rebind via the passed-in `embed` callable)."""
    conn = clustering_store.conn

    clusters: dict[str, dict] = {}
    for cid, lbl in conn.execute("SELECT cluster_id, label FROM clusters").fetchall():
        if cid and lbl:
            clusters[cid] = {"label": lbl}

    domains: dict[str, dict] = {}
    for did, lbl in conn.execute("SELECT domain_id, label FROM domains").fetchall():
        if did and lbl:
            domains[did] = {"label": lbl}

    fields: dict[str, dict] = {}
    for fid, name in conn.execute("SELECT field_id, name FROM field_definitions").fetchall():
        if fid and name:
            fields[fid] = {"label": name}

    return clusters, domains, fields


def _default_embedder():
    """Returns a (text)->np.ndarray callable using the project's MiniLM CPU
    embedder. Cached at module level so repeat run_miner calls don't reload."""
    from core.embedder import Embedder
    if not hasattr(_default_embedder, "_emb"):
        _default_embedder._emb = Embedder()
    emb = _default_embedder._emb

    def embed(text: str):
        import numpy as np
        if not text:
            return np.zeros(0, dtype="float32")
        v = emb.embed([str(text)])
        return v[0]
    return embed


def run_miner(clustering_store, playbook_store: PlaybookStore, *,
              playbook_name: str = "Auto-mined draft",
              owner_org: str = "auto",
              rebind: bool = True,
              rebind_threshold: float = 0.55,
              embed_callable=None) -> tuple[str, list[dict]]:
    """End-to-end: build corpus → create draft playbook → mine candidates →
    soft-rebind label:* bindings to real ontology IDs.

    Returns (playbook_id, candidates). The number of rebound bindings is also
    logged; call store.bindings_for(rule_id) to inspect.
    """
    from core.playbooks.miner import soft_rebind

    corpus, domain_totals = build_corpus(clustering_store)
    pid = playbook_store.create_playbook(name=playbook_name, owner_org=owner_org,
                                          description="Generated by miner_runner")
    n_total = max(domain_totals.values()) if domain_totals else 1
    cands = mine_candidates(playbook_store, pid, corpus, n_total_per_domain=n_total)

    if rebind:
        # Resolve label:* bindings to real entity_ids via embedding similarity.
        embed = embed_callable or _default_embedder()
        clusters, domains, fields = _build_rebind_candidates(clustering_store)
        n_cluster = soft_rebind(playbook_store, embed=embed,
                                  cluster_centroids=clusters,
                                  threshold=rebind_threshold,
                                  entity_kind="cluster")
        n_domain = soft_rebind(playbook_store, embed=embed,
                                 cluster_centroids=domains,
                                 threshold=rebind_threshold,
                                 entity_kind="domain")
        n_field = soft_rebind(playbook_store, embed=embed,
                                cluster_centroids=fields,
                                threshold=rebind_threshold,
                                entity_kind="field")
        total_rebound = n_cluster + n_domain + n_field
        import logging
        logging.getLogger(__name__).info(
            f"soft_rebind: cluster={n_cluster} domain={n_domain} field={n_field} "
            f"(total {total_rebound})"
        )

    return pid, cands
