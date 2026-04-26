"""Dashboard page — Demo Act 8: The Big Picture.

Cluster quality scores, extraction coverage, correction counts, system health.
"300 contracts → 5 domains → 25 clause types → 100 fields → 10 computed metrics."
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from core.store import ClusteringStore


def render(store: ClusteringStore):
    st.header("📊 Dashboard")
    st.caption("Live cluster health, extraction coverage, and system metrics.")

    stats = store.get_stats()

    # ── Top-level metrics ──────────────────────────────────────────────
    cols = st.columns(4)
    cols[0].metric("📁 Domains", stats["domains"])
    cols[1].metric("📄 Agreements", stats["agreements"])
    cols[2].metric("💼 Clusters", stats["clusters"])
    cols[3].metric("📋 Fields", stats["fields"])

    cols2 = st.columns(4)
    cols2[0].metric("🔍 Extractions", stats["extractions"])
    cols2[1].metric("✏️ Corrections", stats["corrections"])
    cols2[2].metric("🧮 Composites", stats["composites"])
    cols2[3].metric("📈 Avg Coverage", f"{stats['avg_coverage']:.1%}")

    st.divider()

    # ── Domain distribution ────────────────────────────────────────────
    domains = store.get_domains()
    if domains:
        st.subheader("Domain Distribution")
        df_domains = pd.DataFrame(domains)
        fig = px.pie(
            df_domains, values="agreement_count", names="label",
            title="Documents by Domain",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster quality ────────────────────────────────────────────────
    clusters = store.get_clusters()
    if clusters:
        st.subheader("Cluster Quality Scores")
        df_clusters = pd.DataFrame([
            {
                "Cluster": c["label"],
                "Domain": next((d["label"] for d in domains if d["domain_id"] == c.get("domain_id")), "?"),
                "Chunks": c["chunk_count"],
                "Quality": c.get("quality_score", 0) or 0,
                "Merged": "Yes" if c.get("aliases") else "No",
            }
            for c in clusters
        ])

        fig = px.bar(
            df_clusters, x="Cluster", y="Quality",
            color="Domain",
            title="Cluster Quality Scores (cosine similarity to centroid)",
            hover_data=["Chunks", "Merged"],
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                      annotation_text="Quality threshold (0.8)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Quality table
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)

    # ── Field coverage ─────────────────────────────────────────────────
    if clusters:
        st.subheader("Field Coverage by Cluster")
        coverage_data = []
        for c in clusters:
            fields = store.get_fields(c["cluster_id"])
            for f in fields:
                extractions = store.get_extractions(field_id=f["field_id"])
                filled = sum(1 for e in extractions if e.get("value"))
                total = len(extractions) if extractions else 0
                coverage_data.append({
                    "Cluster": c["label"],
                    "Field": f["name"],
                    "Coverage": filled / total if total > 0 else 0,
                    "Filled": filled,
                    "Total": total,
                })

        if coverage_data:
            df_cov = pd.DataFrame(coverage_data)
            fig = px.bar(
                df_cov, x="Field", y="Coverage", color="Cluster",
                barmode="group",
                title="Extraction Coverage by Field",
            )
            fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # ── Corrections log ────────────────────────────────────────────────
    corrections = store.get_corrections()
    if corrections:
        st.subheader(f"Recent Corrections ({len(corrections)})")
        df_corr = pd.DataFrame(corrections[:20])
        st.dataframe(df_corr, use_container_width=True, hide_index=True)

    # ── Summary statement ──────────────────────────────────────────────
    st.divider()
    st.success(
        f"**{stats['agreements']} contracts → {stats['domains']} domains → "
        f"{stats['clusters']} clause types → {stats['fields']} fields → "
        f"{stats['composites']} computed metrics.** "
        f"All discovered automatically, refined by humans, compounding over time."
    )
