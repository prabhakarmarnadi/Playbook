"""Bridge between the existing ClusteringStore and the playbook subsystem.

Provides:
  agreement_ctx(clustering_store, agreement_id) -> dict
    Returns a ctx dict ready to feed core.playbooks.aligner.align(...).
"""
from __future__ import annotations

from typing import Any


def agreement_ctx(clustering_store, agreement_id: str) -> dict:
    """Build an alignment ctx for one agreement.

    Reads from the existing ClusteringStore tables:
      - agreements (for domain_id)
      - extractions JOIN field_definitions (for field_name -> value)
      - clauses (for {cluster_id, label, text})
      - domains (for domain name)
    """
    conn = clustering_store.conn

    cur = conn.execute(
        "SELECT a.agreement_id, a.domain_id, d.label AS domain_name "
        "FROM agreements a LEFT JOIN domains d ON a.domain_id = d.domain_id "
        "WHERE a.agreement_id = ?",
        [agreement_id],
    )
    row = cur.fetchone()
    if row is None:
        return {"agreement_id": agreement_id, "domain": None, "fields": {}, "clauses": []}
    _, domain_id, domain_name = row

    # extractions JOIN field_definitions to recover the human field name
    cur = conn.execute(
        "SELECT f.name AS field_name, e.value "
        "FROM extractions e JOIN field_definitions f ON e.field_id = f.field_id "
        "WHERE e.agreement_id = ?",
        [agreement_id],
    )
    fields: dict[str, Any] = {}
    for name, value in cur.fetchall():
        if name is not None:
            fields[name] = value

    # clauses — full_text carries the clause body in ClusteringStore
    cur = conn.execute(
        "SELECT clause_id, clause_type_id, full_text FROM clauses WHERE agreement_id = ?",
        [agreement_id],
    )
    clauses = []
    for clause_id, clause_type_id, full_text in cur.fetchall():
        clauses.append({
            "id": clause_id,
            "cluster_id": clause_type_id,  # legacy column name
            "label": clause_type_id,
            "text": full_text,
        })

    return {
        "agreement_id": agreement_id,
        "domain": {"id": domain_id, "name": domain_name} if domain_id else None,
        "fields": fields,
        "clauses": clauses,
    }
