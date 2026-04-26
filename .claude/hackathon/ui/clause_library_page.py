"""Clause Library & Deviation Analysis page — Demo Act 10.

Two interconnected features:
  1. Deviation Analysis — how far is each chunk/document from its cluster centroid?
  2. Clause Library — browsable, searchable clause template library for authoring.

"Know exactly how standard — or unusual — every clause is."
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.store import ClusteringStore


# ═══════════════════════════════════════════════════════════════════════════
# DEVIATION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_chunk_deviations(store: ClusteringStore) -> pd.DataFrame:
    """
    For every chunk with an embedding and cluster assignment, compute:
      - cosine similarity to its cluster centroid
      - deviation = 1 - similarity  (0 = perfect match, 1 = orthogonal)
    """
    centroids = store.get_all_centroids()
    if not centroids:
        return pd.DataFrame()

    rows = store.conn.execute("""
        SELECT c.chunk_id, c.agreement_id, c.clause_id,
               c.chunk_text, c.chunk_index, c.embedding,
               ca.cluster_id, ca.probability
        FROM chunks c
        JOIN cluster_assignments ca ON c.chunk_id = ca.chunk_id
        WHERE c.embedding IS NOT NULL
    """).fetchall()

    if not rows:
        return pd.DataFrame()

    # Get cluster labels
    clusters = store.get_clusters()
    cluster_label_map = {c["cluster_id"]: c["label"] for c in clusters}

    # Get domain info per cluster
    cluster_domain_map = {c["cluster_id"]: c.get("domain_id", "") for c in clusters}
    domains = store.get_domains()
    domain_label_map = {d["domain_id"]: d["label"] for d in domains}

    # Get agreement filenames
    agreements = store.conn.execute(
        "SELECT agreement_id, filename FROM agreements"
    ).fetchall()
    agreement_name_map = {a[0]: a[1] for a in agreements}

    records = []
    for chunk_id, agreement_id, clause_id, text, idx, emb_blob, cluster_id, prob in rows:
        if cluster_id not in centroids:
            continue
        centroid_vec, _ = centroids[cluster_id]
        chunk_vec = store._deserialize_single_embedding(emb_blob)

        # Cosine similarity (embeddings are L2-normalized)
        sim = float(np.dot(chunk_vec, centroid_vec))
        sim = max(-1.0, min(1.0, sim))  # clamp for floating point
        deviation = 1.0 - sim

        domain_id = cluster_domain_map.get(cluster_id, "")
        records.append({
            "chunk_id": chunk_id,
            "agreement_id": agreement_id,
            "agreement": agreement_name_map.get(agreement_id, agreement_id)[:60],
            "clause_id": clause_id or "",
            "chunk_index": idx,
            "cluster_id": cluster_id,
            "cluster": cluster_label_map.get(cluster_id, cluster_id),
            "domain": domain_label_map.get(domain_id, "Unknown"),
            "similarity": round(sim, 4),
            "deviation": round(deviation, 4),
            "probability": prob,
            "text_preview": (text or "")[:120],
            "text": text or "",
        })

    return pd.DataFrame(records)


def _compute_document_deviations(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chunk-level deviations to per-document metrics:
      - mean deviation across all chunks in the document
      - max deviation (the most unusual clause)
      - std deviation (consistency)
      - outlier count (chunks with deviation > 0.3)
    """
    if chunk_df.empty:
        return pd.DataFrame()

    agg = chunk_df.groupby(["agreement_id", "agreement"]).agg(
        chunks=("chunk_id", "count"),
        clusters=("cluster_id", "nunique"),
        mean_similarity=("similarity", "mean"),
        min_similarity=("similarity", "min"),
        max_deviation=("deviation", "max"),
        mean_deviation=("deviation", "mean"),
        std_deviation=("deviation", "std"),
    ).reset_index()

    agg["std_deviation"] = agg["std_deviation"].fillna(0)
    agg["outlier_chunks"] = chunk_df[chunk_df["deviation"] > 0.3].groupby(
        "agreement_id"
    ).size().reindex(agg["agreement_id"]).fillna(0).astype(int).values

    # Conformity score: how "standard" is this document overall
    agg["conformity"] = (agg["mean_similarity"] * 100).round(1)

    return agg.sort_values("mean_deviation", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLAUSE LIBRARY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _build_clause_library(store: ClusteringStore, chunk_df: pd.DataFrame) -> list[dict]:
    """
    Build a clause library: for each cluster, find the most representative
    (closest to centroid) and most deviant clauses.
    """
    clusters = store.get_clusters()
    if not clusters or chunk_df.empty:
        return []

    # Get intents if available
    all_intents = {}
    try:
        intent_rows = store.conn.execute(
            "SELECT clause_id, intent_label, summary FROM clause_intents"
        ).fetchall()
        for cid, label, summary in intent_rows:
            all_intents.setdefault(cid, []).append({
                "label": label, "summary": summary or ""
            })
    except Exception:
        pass

    # Get fields per cluster
    all_fields = {}
    for c in clusters:
        fields = store.get_fields(c["cluster_id"])
        all_fields[c["cluster_id"]] = fields

    library = []
    for c in clusters:
        cdf = chunk_df[chunk_df["cluster_id"] == c["cluster_id"]]
        if cdf.empty:
            continue

        # Representative: closest to centroid (min deviation)
        best = cdf.loc[cdf["deviation"].idxmin()]
        # Most deviant
        worst = cdf.loc[cdf["deviation"].idxmax()]

        # Median exemplar
        median_idx = (cdf["deviation"] - cdf["deviation"].median()).abs().idxmin()
        median_row = cdf.loc[median_idx]

        library.append({
            "cluster_id": c["cluster_id"],
            "cluster_label": c["label"],
            "domain": cdf["domain"].iloc[0],
            "description": c.get("description", ""),
            "keywords": c.get("keywords", []),
            "chunk_count": len(cdf),
            "mean_similarity": round(cdf["similarity"].mean(), 4),
            "std_similarity": round(cdf["similarity"].std(), 4) if len(cdf) > 1 else 0.0,
            "fields": all_fields.get(c["cluster_id"], []),
            "intents": list({
                i["label"]
                for clause_id in cdf["clause_id"].unique()
                if clause_id
                for i in all_intents.get(clause_id, [])
            }),
            "exemplar": {
                "text": best["text"],
                "similarity": best["similarity"],
                "agreement": best["agreement"],
            },
            "median": {
                "text": median_row["text"],
                "similarity": median_row["similarity"],
                "agreement": median_row["agreement"],
            },
            "outlier": {
                "text": worst["text"],
                "similarity": worst["similarity"],
                "deviation": worst["deviation"],
                "agreement": worst["agreement"],
            },
        })

    library.sort(key=lambda x: x["chunk_count"], reverse=True)
    return library


# ═══════════════════════════════════════════════════════════════════════════
# UI RENDER
# ═══════════════════════════════════════════════════════════════════════════

def render(store: ClusteringStore):
    st.header("📏 Clause Library & Deviation Analysis")
    st.caption(
        "Measure how standard or unusual every clause and document is. "
        "Browse the clause template library for authoring."
    )

    # Compute deviations (cached per session)
    if "chunk_deviations" not in st.session_state:
        with st.spinner("Computing centroid deviations for all chunks..."):
            st.session_state["chunk_deviations"] = _compute_chunk_deviations(store)

    chunk_df = st.session_state["chunk_deviations"]
    if chunk_df.empty:
        st.warning("No embeddings or centroids found. Run the pipeline first.")
        return

    tab_dev, tab_docs, tab_library = st.tabs([
        "📊 Clause Deviations",
        "📄 Document Conformity",
        "📚 Clause Library",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1: CLAUSE-LEVEL DEVIATION
    # ════════════════════════════════════════════════════════════════════
    with tab_dev:
        _render_clause_deviations(chunk_df, store)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2: DOCUMENT CONFORMITY
    # ════════════════════════════════════════════════════════════════════
    with tab_docs:
        _render_document_conformity(chunk_df)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3: CLAUSE LIBRARY
    # ════════════════════════════════════════════════════════════════════
    with tab_library:
        _render_clause_library(store, chunk_df)


def _render_clause_deviations(chunk_df: pd.DataFrame, store: ClusteringStore):
    """Visualize deviation of every clause from its cluster centroid."""

    # Top metrics
    cols = st.columns(4)
    cols[0].metric("Total Chunks", len(chunk_df))
    cols[1].metric("Avg Similarity", f"{chunk_df['similarity'].mean():.3f}")
    cols[2].metric("Avg Deviation", f"{chunk_df['deviation'].mean():.3f}")
    outlier_count = (chunk_df["deviation"] > 0.3).sum()
    cols[3].metric("Outliers (>0.3)", outlier_count,
                   delta=f"{outlier_count / len(chunk_df):.1%} of total",
                   delta_color="inverse")

    st.divider()

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    domains = sorted(chunk_df["domain"].unique())
    selected_domain = col_f1.selectbox("Filter by Domain", ["All"] + domains,
                                       key="dev_domain")
    clusters_in_view = chunk_df if selected_domain == "All" else \
        chunk_df[chunk_df["domain"] == selected_domain]
    cluster_labels = sorted(clusters_in_view["cluster"].unique())
    selected_cluster = col_f2.selectbox("Filter by Cluster", ["All"] + cluster_labels,
                                        key="dev_cluster")

    df = chunk_df.copy()
    if selected_domain != "All":
        df = df[df["domain"] == selected_domain]
    if selected_cluster != "All":
        df = df[df["cluster"] == selected_cluster]

    # Deviation distribution
    st.subheader("Deviation Distribution")
    fig = px.histogram(
        df, x="deviation", color="domain" if selected_domain == "All" else "cluster",
        nbins=50,
        title="Chunk Deviation from Cluster Centroid",
        labels={"deviation": "Deviation (1 − cosine similarity)"},
    )
    fig.add_vline(x=0.3, line_dash="dash", line_color="red",
                  annotation_text="Outlier threshold")
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: similarity vs cluster
    st.subheader("Similarity by Cluster")
    fig2 = px.strip(
        df, x="cluster", y="similarity", color="domain",
        hover_data=["agreement", "text_preview"],
        title="Chunk Similarity to Cluster Centroid (each dot = one chunk)",
    )
    fig2.add_hline(y=0.7, line_dash="dash", line_color="orange",
                   annotation_text="Low similarity")
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # Table of most deviant chunks
    st.subheader("Most Deviant Chunks")
    deviant = df.nlargest(20, "deviation")[
        ["agreement", "cluster", "similarity", "deviation", "text_preview"]
    ]
    st.dataframe(deviant, use_container_width=True, hide_index=True)

    # Expandable details for top outliers
    top5 = df.nlargest(5, "deviation")
    for _, row in top5.iterrows():
        with st.expander(
            f"⚠️ Deviation {row['deviation']:.3f} — {row['cluster']} "
            f"({row['agreement']})"
        ):
            st.text_area("Full text", row["text"][:1500], height=120,
                         disabled=True, key=f"dev_{row['chunk_id']}")
            st.caption(
                f"Similarity: {row['similarity']:.4f} | "
                f"Cluster: {row['cluster']} | "
                f"Document: {row['agreement']}"
            )


def _render_document_conformity(chunk_df: pd.DataFrame):
    """Aggregate deviations to document level."""
    doc_df = _compute_document_deviations(chunk_df)
    if doc_df.empty:
        st.info("No documents to analyze.")
        return

    # Top metrics
    cols = st.columns(4)
    cols[0].metric("Documents", len(doc_df))
    cols[1].metric("Avg Conformity", f"{doc_df['conformity'].mean():.1f}%")
    high_conformity = (doc_df["conformity"] >= 85).sum()
    cols[2].metric("High Conformity (≥85%)", high_conformity)
    low_conformity = (doc_df["conformity"] < 70).sum()
    cols[3].metric("Low Conformity (<70%)", low_conformity,
                   delta_color="inverse")

    st.divider()

    # Conformity bar chart
    st.subheader("Document Conformity Scores")
    fig = px.bar(
        doc_df.sort_values("conformity"),
        x="agreement", y="conformity",
        color="conformity",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        range_color=[50, 100],
        hover_data=["chunks", "clusters", "mean_deviation", "outlier_chunks"],
        title="How standard is each document? (100% = perfectly typical)",
    )
    fig.add_hline(y=85, line_dash="dash", line_color="green",
                  annotation_text="High conformity")
    fig.add_hline(y=70, line_dash="dash", line_color="orange",
                  annotation_text="Review threshold")
    fig.update_layout(xaxis_tickangle=-45, coloraxis_colorbar_title="Conformity %")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: chunks vs conformity
    st.subheader("Document Size vs Conformity")
    fig2 = px.scatter(
        doc_df, x="chunks", y="conformity",
        size="outlier_chunks", color="mean_deviation",
        hover_data=["agreement", "clusters", "max_deviation"],
        color_continuous_scale="RdYlGn_r",
        title="Larger documents tend to have more clause diversity",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.subheader("Document Details")
    display_df = doc_df[[
        "agreement", "chunks", "clusters", "conformity",
        "mean_similarity", "min_similarity", "max_deviation",
        "std_deviation", "outlier_chunks",
    ]].rename(columns={
        "agreement": "Document",
        "chunks": "Chunks",
        "clusters": "Clusters",
        "conformity": "Conformity %",
        "mean_similarity": "Avg Sim",
        "min_similarity": "Min Sim",
        "max_deviation": "Max Dev",
        "std_deviation": "Std Dev",
        "outlier_chunks": "Outliers",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Drill-down: select a document to see its clause deviations
    st.divider()
    st.subheader("Drill Down: Document Clause Map")
    doc_options = doc_df["agreement"].tolist()
    selected_doc = st.selectbox("Select Document", doc_options, key="doc_drill")
    selected_aid = doc_df[doc_df["agreement"] == selected_doc]["agreement_id"].iloc[0]

    doc_chunks = chunk_df[chunk_df["agreement_id"] == selected_aid].sort_values("chunk_index")
    if not doc_chunks.empty:
        fig3 = px.bar(
            doc_chunks, x="chunk_index", y="similarity",
            color="cluster",
            hover_data=["text_preview", "deviation"],
            title=f"Clause similarity map — {selected_doc}",
            labels={"chunk_index": "Clause Position", "similarity": "Centroid Similarity"},
        )
        fig3.add_hline(y=0.7, line_dash="dash", line_color="orange")
        fig3.update_layout(bargap=0.1)
        st.plotly_chart(fig3, use_container_width=True)


def _render_clause_library(store: ClusteringStore, chunk_df: pd.DataFrame):
    """Browsable clause template library with exemplars and deviation stats."""

    if "clause_library" not in st.session_state:
        with st.spinner("Building clause library..."):
            st.session_state["clause_library"] = _build_clause_library(store, chunk_df)

    library = st.session_state["clause_library"]
    if not library:
        st.info("No clusters to build library from.")
        return

    # Top metrics
    cols = st.columns(4)
    cols[0].metric("Clause Types", len(library))
    cols[1].metric("Total Templates", sum(c["chunk_count"] for c in library))
    avg_sim = np.mean([c["mean_similarity"] for c in library])
    cols[2].metric("Avg Cluster Cohesion", f"{avg_sim:.3f}")
    with_intents = sum(1 for c in library if c["intents"])
    cols[3].metric("With Intents", with_intents)

    st.divider()

    # Filter
    col_f1, col_f2 = st.columns(2)
    lib_domains = sorted(set(c["domain"] for c in library))
    selected_dom = col_f1.selectbox("Filter Domain", ["All"] + lib_domains,
                                    key="lib_domain")
    search_term = col_f2.text_input("Search clauses", placeholder="e.g., indemnification",
                                     key="lib_search")

    filtered = library
    if selected_dom != "All":
        filtered = [c for c in filtered if c["domain"] == selected_dom]
    if search_term:
        term = search_term.lower()
        filtered = [
            c for c in filtered
            if term in c["cluster_label"].lower()
            or term in c.get("description", "").lower()
            or any(term in k.lower() for k in c.get("keywords", []))
            or term in c["exemplar"]["text"].lower()
        ]

    # Library overview
    st.subheader(f"Clause Type Library ({len(filtered)} types)")

    # Summary table
    summary_data = [{
        "Clause Type": c["cluster_label"],
        "Domain": c["domain"],
        "Templates": c["chunk_count"],
        "Cohesion": f"{c['mean_similarity']:.3f}",
        "Spread": f"±{c['std_similarity']:.3f}",
        "Fields": len(c["fields"]),
        "Intents": len(c["intents"]),
    } for c in filtered]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.divider()

    # Clause cards
    for entry in filtered:
        with st.expander(
            f"📋 {entry['cluster_label']} — {entry['chunk_count']} templates | "
            f"Cohesion: {entry['mean_similarity']:.3f} | "
            f"{entry['domain']}",
            expanded=False,
        ):
            # Description and keywords
            if entry["description"]:
                st.write(entry["description"])
            if entry["keywords"]:
                st.markdown("**Keywords:** " + ", ".join(
                    f"`{k}`" for k in entry["keywords"][:12]
                ))

            # Intents
            if entry["intents"]:
                st.markdown("**Discovered Intents:** " + ", ".join(
                    f"🎯 {i}" for i in sorted(entry["intents"])[:8]
                ))

            # Fields
            if entry["fields"]:
                st.markdown("**Extractable Fields:**")
                field_cols = st.columns(min(3, len(entry["fields"])))
                for i, f in enumerate(entry["fields"][:9]):
                    field_cols[i % 3].markdown(
                        f"  `{f['name']}` ({f['field_type']})"
                    )

            st.markdown("---")

            # Three exemplars: best, median, worst
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**✅ Most Representative**")
                st.caption(f"Similarity: {entry['exemplar']['similarity']:.4f}")
                st.text_area(
                    f"From: {entry['exemplar']['agreement']}",
                    entry["exemplar"]["text"][:800],
                    height=180, disabled=True,
                    key=f"lib_best_{entry['cluster_id']}",
                )

            with col2:
                st.markdown("**📐 Median Example**")
                st.caption(f"Similarity: {entry['median']['similarity']:.4f}")
                st.text_area(
                    f"From: {entry['median']['agreement']}",
                    entry["median"]["text"][:800],
                    height=180, disabled=True,
                    key=f"lib_med_{entry['cluster_id']}",
                )

            with col3:
                st.markdown("**⚠️ Most Unusual**")
                st.caption(
                    f"Similarity: {entry['outlier']['similarity']:.4f} | "
                    f"Deviation: {entry['outlier']['deviation']:.4f}"
                )
                st.text_area(
                    f"From: {entry['outlier']['agreement']}",
                    entry["outlier"]["text"][:800],
                    height=180, disabled=True,
                    key=f"lib_out_{entry['cluster_id']}",
                )
