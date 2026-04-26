"""Cluster Explorer page — Demo Act 2: Auto Ontology.

Shows micro clusters within a selected domain. UMAP scatter, keywords, fields.
"The system built this ontology on its own. No predefined schema."
"""
import json

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from core.store import ClusteringStore


def render(store: ClusteringStore):
    st.header("💼 Cluster Explorer")

    domains = store.get_domains()
    if not domains:
        st.info("No domains found. Run the pipeline first.")
        return

    # Domain selector
    domain_options = {d["label"]: d["domain_id"] for d in domains if d["label"] != "Uncategorized"}
    if not domain_options:
        st.warning("No categorized domains found.")
        return

    selected_label = st.selectbox("Select Domain", list(domain_options.keys()))
    domain_id = domain_options[selected_label]

    clusters = store.get_clusters(domain_id)
    if not clusters:
        st.info(f"No clusters found in {selected_label}.")
        return

    # ── Cluster overview metrics ───────────────────────────────────────────
    # Count outliers for this domain's clusters
    total_outliers = 0
    for c in clusters:
        outliers = store.get_outlier_chunks(cluster_id=c["cluster_id"])
        total_outliers += len(outliers)

    cols = st.columns(5)
    cols[0].metric("Clause Types", len(clusters))
    cols[1].metric("Total Chunks", sum(c.get("chunk_count", 0) for c in clusters))
    cols[2].metric("Avg Quality", f"{np.mean([c.get('quality_score', 0) or 0 for c in clusters]):.2f}")
    merged_count = sum(1 for c in clusters if c.get("aliases"))
    cols[3].metric("Merged Topics", merged_count)
    cols[4].metric("⚠️ Outliers", total_outliers)

    # ── Cluster cards ──────────────────────────────────────────────────────
    st.subheader(f"Clause Types in {selected_label}")
    for cluster in clusters:
        with st.expander(
            f"💼 {cluster['label']} — {cluster['chunk_count']} chunks | "
            f"Quality: {cluster.get('quality_score', 0):.2f}",
            expanded=False,
        ):
            if cluster.get("aliases"):
                st.caption(f"🔗 Also known as: {', '.join(cluster['aliases'])}")
            if cluster.get("description"):
                st.write(cluster["description"])

            # Keywords
            kw = cluster.get("keywords", [])
            if kw:
                st.markdown("**Keywords:** " + ", ".join(f"`{w}`" for w in kw[:10]))

            # Fields
            fields = store.get_fields(cluster["cluster_id"])
            if fields:
                st.markdown("**Auto-discovered fields:**")
                for f in fields:
                    st.markdown(f"  - 📋 `{f['name']}` ({f['field_type']}) — {f.get('description', '')}")

            # Sample chunks
            chunks = store.get_chunks(cluster_id=cluster["cluster_id"], include_outliers=False)
            if chunks:
                st.markdown(f"**Sample clauses** ({min(3, len(chunks))} of {len(chunks)}):")
                for c in chunks[:3]:
                    st.text_area(
                        f"Chunk {c.get('chunk_index', '?')}",
                        c["chunk_text"][:500],
                        height=80,
                        disabled=True,
                        key=f"chunk_{cluster['cluster_id']}_{c['chunk_id']}",
                    )

            # Outlier chunks attached to this cluster
            outlier_chunks = store.get_outlier_chunks(cluster_id=cluster["cluster_id"])
            if outlier_chunks:
                st.markdown(
                    f"**⚠️ Outliers** ({len(outlier_chunks)} chunks — nearest to this cluster "
                    f"but not a confident member):"
                )
                for oc in outlier_chunks[:5]:
                    sim = oc.get("outlier_similarity", 0)
                    color = "#f85149" if sim < 0.5 else "#d29922"
                    st.markdown(
                        f'<div style="border-left: 3px solid {color}; padding-left: 8px; '
                        f'margin-bottom: 6px; opacity: 0.8;">'
                        f'<small>sim={sim:.3f}</small><br>'
                        f'{oc["chunk_text"][:300]}</div>',
                        unsafe_allow_html=True,
                    )
                if len(outlier_chunks) > 5:
                    st.caption(f"... and {len(outlier_chunks) - 5} more outliers")
