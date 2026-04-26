"""Ontology Tree page — Demo Act 2/8: the "wow" visual.

Interactive tree showing: Domains → Clause Types → Fields → Composites.
"See the full knowledge graph. All discovered automatically."
"""
import json
import streamlit as st
from core.store import ClusteringStore


def render(store: ClusteringStore):
    st.header("🌳 Ontology Tree")
    st.caption("Full knowledge graph: domains → clause types → fields → composites. "
               "All discovered automatically, refined by humans, compounding over time.")

    domains = store.get_domains()
    if not domains:
        st.info("No ontology data. Run the pipeline first.")
        return

    # ── Render tree ────────────────────────────────────────────────────────
    for domain in domains:
        domain_clusters = store.get_clusters(domain["domain_id"])
        cluster_count = len(domain_clusters)
        total_fields = sum(len(store.get_fields(c["cluster_id"])) for c in domain_clusters)

        with st.expander(
            f"📁 **{domain['label']}** — {domain['agreement_count']} agreements, "
            f"{cluster_count} clause types, {total_fields} fields",
            expanded=True,
        ):
            if domain.get("description"):
                st.caption(domain["description"])

            for cluster in domain_clusters:
                merge_badge = " 🔗 (merged)" if cluster.get("aliases") else ""
                st.markdown(
                    f"&emsp; 💼 **{cluster['label']}**{merge_badge} — "
                    f"{cluster['chunk_count']} chunks | "
                    f"Quality: {cluster.get('quality_score', 0):.2f}"
                )

                if cluster.get("aliases"):
                    st.markdown(
                        f"&emsp;&emsp; *Also known as: {', '.join(cluster['aliases'])}*"
                    )

                # Fields
                fields = store.get_fields(cluster["cluster_id"])
                for f in fields:
                    # Compute coverage
                    extractions = store.get_extractions(field_id=f["field_id"])
                    filled = sum(1 for e in extractions if e.get("value"))
                    total = len(extractions) if extractions else 0
                    coverage = f"{filled}/{total}" if total > 0 else "—"

                    st.markdown(
                        f"&emsp;&emsp;&emsp; 📋 `{f['name']}`: {f['field_type']} "
                        f"(coverage: {coverage})"
                    )

                # Composites
                composites = store.get_composites(cluster["cluster_id"])
                for comp in composites:
                    st.markdown(
                        f"&emsp;&emsp;&emsp; 🧮 `{comp['name']}` = {comp['expression']}"
                    )

    # ── JSON export ────────────────────────────────────────────────────────
    st.divider()
    if st.checkbox("Show raw JSON ontology"):
        ontology = _build_ontology_json(store, domains)
        st.json(ontology)


def _build_ontology_json(store: ClusteringStore, domains: list[dict]) -> dict:
    """Build the full ontology as a nested JSON structure."""
    result = {"domains": []}
    for d in domains:
        domain_obj = {
            "domain_id": d["domain_id"],
            "label": d["label"],
            "description": d.get("description", ""),
            "agreement_count": d["agreement_count"],
            "clusters": [],
        }
        for c in store.get_clusters(d["domain_id"]):
            cluster_obj = {
                "cluster_id": c["cluster_id"],
                "label": c["label"],
                "keywords": c.get("keywords", []),
                "chunk_count": c["chunk_count"],
                "quality_score": c.get("quality_score"),
                "aliases": c.get("aliases", []),
                "fields": [],
                "composites": [],
            }
            for f in store.get_fields(c["cluster_id"]):
                cluster_obj["fields"].append({
                    "name": f["name"],
                    "type": f["field_type"],
                    "description": f.get("description", ""),
                    "source": f.get("source", "auto_discovered"),
                })
            for comp in store.get_composites(c["cluster_id"]):
                cluster_obj["composites"].append({
                    "name": comp["name"],
                    "expression": comp["expression"],
                })
            domain_obj["clusters"].append(cluster_obj)
        result["domains"].append(domain_obj)
    return result
