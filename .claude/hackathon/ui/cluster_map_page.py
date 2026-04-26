"""
Hierarchical Cluster Visualization — Treemap, Sunburst, and Micro-Cluster Explorer.

Scales to millions of documents. Three visualization modes:
1. Treemap: Where's the mass? Domain → Clause Types sized by clause count
2. Sunburst: Full taxonomy hierarchy with drill-down
3. Micro-Cluster Explorer: 2D UMAP projection showing what clustered together,
   sub-groupings within clusters, and outliers
"""
import logging
from collections import Counter, defaultdict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.store import ClusteringStore

logger = logging.getLogger(__name__)


# ── Color palette ─────────────────────────────────────────────────────────
DOMAIN_COLORS = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
    "#f778ba", "#79c0ff", "#56d364", "#e3b341", "#ff7b72",
    "#d2a8ff", "#ff9bce", "#a5d6ff", "#7ee787", "#f0c94d",
]


def _build_hierarchy(store: ClusteringStore):
    """Build domain → cluster hierarchy data (many-to-many).

    In hybrid mode, clusters span all documents, so a single cluster may
    have chunks from agreements in *multiple* domains.  We compute
    per-(domain, cluster) chunk counts so the treemap / sunburst shows each
    cluster under every domain it touches, sized by that domain's contribution.
    """
    domains = store.get_domains()
    clusters = store.get_clusters()

    # Separate real domains from placeholder
    placeholder_ids = set()
    real_domains = []
    for d in domains:
        lbl = (d.get("label") or "").lower()
        if lbl in ("all documents", "uncategorized", ""):
            placeholder_ids.add(d["domain_id"])
        else:
            real_domains.append(d)

    domain_map = {d["domain_id"]: d for d in real_domains}

    # Build agreement → real domain lookup
    agreements = store.get_agreements()
    agr_domain = {}
    for a in agreements:
        did = a.get("domain_id", "")
        if did and did not in placeholder_ids:
            agr_domain[a["agreement_id"]] = did

    # For each cluster, count chunks per real domain
    # domain_clusters[domain_id] = list of (cluster_dict, chunk_count_in_domain)
    domain_clusters: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    orphan_clusters = []

    for c in clusters:
        chunks = store.get_chunks(cluster_id=c["cluster_id"], include_outliers=False)
        per_domain: dict[str, int] = Counter()
        unassigned = 0
        for ch in chunks:
            aid = ch.get("agreement_id", "")
            did = agr_domain.get(aid)
            if did:
                per_domain[did] += 1
            else:
                unassigned += 1

        if per_domain:
            for did, cnt in per_domain.items():
                if did in domain_map:
                    domain_clusters[did].append((c, cnt))
        else:
            orphan_clusters.append(c)

    return real_domains, clusters, domain_map, domain_clusters, orphan_clusters


def _render_treemap(store: ClusteringStore):
    """Treemap: Domain → Clause Types, sized by clause count."""
    domains, clusters, domain_map, domain_clusters, orphan_clusters = _build_hierarchy(store)

    if not domains and not orphan_clusters:
        st.info("No domain or cluster data found.")
        return

    # Build treemap data
    ids, labels, parents, values, colors, hovers = [], [], [], [], [], []

    # Root
    ids.append("Portfolio")
    labels.append("All Contracts")
    parents.append("")
    values.append(0)
    colors.append("")
    hovers.append("")

    # Domains — each cluster can appear under multiple domains
    for i, d in enumerate(domains):
        did = d["domain_id"]
        dlabel = d.get("label", did)
        entries = domain_clusters.get(did, [])  # list of (cluster_dict, chunk_count)
        n_clusters = len(entries)
        n_chunks = sum(cnt for _, cnt in entries)
        color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]

        ids.append(did)
        labels.append(dlabel)
        parents.append("Portfolio")
        values.append(0)
        colors.append(color)
        hovers.append(f"{n_clusters} clause types, {n_chunks:,} clauses")

        # Clusters within domain — use did:cid as unique id since a cluster
        # can appear under multiple domains
        for c, chunk_count in sorted(entries, key=lambda x: x[1], reverse=True):
            cid = c["cluster_id"]
            clabel = c.get("label", cid)
            chunk_count = chunk_count or 1
            quality = c.get("quality_score", 0) or 0
            total_chunks = c.get("chunk_count", 0) or chunk_count

            ids.append(f"{did}:{cid}")
            labels.append(clabel)
            parents.append(did)
            values.append(chunk_count)
            colors.append(color)
            hovers.append(
                f"<b>{clabel}</b><br>"
                f"In this domain: {chunk_count} clauses<br>"
                f"Total across all domains: {total_chunks}<br>"
                f"Quality: {quality:.3f}<br>"
                f"Domain: {dlabel}"
            )

    # Orphan clusters (no real domain) — group under "Other"
    if orphan_clusters:
        ids.append("__orphan__")
        labels.append("Other")
        parents.append("Portfolio")
        values.append(0)
        colors.append("#8b949e")
        hovers.append(f"{len(orphan_clusters)} clause types without a domain")

        for c in sorted(orphan_clusters,
                        key=lambda x: x.get("chunk_count", 0), reverse=True):
            cid = c["cluster_id"]
            clabel = c.get("label", cid)
            chunk_count = c.get("chunk_count", 0) or 1

            ids.append(cid)
            labels.append(clabel)
            parents.append("__orphan__")
            values.append(chunk_count)
            colors.append("#8b949e")
            hovers.append(f"<b>{clabel}</b><br>Clauses: {chunk_count}")

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=1, color="#0d1117")),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        textinfo="label+value",
        textfont=dict(size=12),
        branchvalues="remainder",
        maxdepth=3,
    ))

    fig.update_layout(
        height=700,
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor="#0d1117",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, key="treemap")


def _render_sunburst(store: ClusteringStore):
    """Sunburst: Domain → Clause Types → Rarity tier."""
    domains, clusters, domain_map, domain_clusters, _ = _build_hierarchy(store)

    n_agreements = len(store.get_agreements())

    ids, labels, parents, values, colors_list = [], [], [], [], []

    # Root
    ids.append("root")
    labels.append("Portfolio")
    parents.append("")
    values.append(0)
    colors_list.append("#161b22")

    for i, d in enumerate(domains):
        did = d["domain_id"]
        dlabel = d.get("label", did)
        color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]

        ids.append(did)
        labels.append(dlabel)
        parents.append("root")
        values.append(0)
        colors_list.append(color)

        # Classify clusters by rarity
        for c, chunk_count in domain_clusters.get(did, []):
            cid = c["cluster_id"]
            clabel = c.get("label", cid)
            chunk_count = chunk_count or 1
            agr_count = c.get("agreement_count", 0) or 0
            pct = (agr_count / n_agreements * 100) if n_agreements else 0

            if pct >= 15:
                rarity = "Boilerplate"
            elif pct >= 5:
                rarity = "Common"
            elif pct >= 2:
                rarity = "Specialized"
            else:
                rarity = "Unique"

            rarity_color = {
                "Boilerplate": "#3fb950",
                "Common": "#58a6ff",
                "Specialized": "#d29922",
                "Unique": "#f85149",
            }.get(rarity, color)

            ids.append(f"{did}:{cid}")
            labels.append(f"{clabel} ({rarity[0]})")
            parents.append(did)
            values.append(chunk_count)
            colors_list.append(rarity_color)

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors_list, line=dict(width=1, color="#0d1117")),
        branchvalues="remainder",
        maxdepth=3,
        insidetextorientation="radial",
    ))

    fig.update_layout(
        height=700,
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor="#0d1117",
        font=dict(color="white", size=11),
    )

    st.plotly_chart(fig, key="sunburst")

    # Legend
    st.markdown(
        "**Rarity:** "
        '<span style="color:#3fb950">● Boilerplate (≥15%)</span> · '
        '<span style="color:#58a6ff">● Common (5-15%)</span> · '
        '<span style="color:#d29922">● Specialized (2-5%)</span> · '
        '<span style="color:#f85149">● Unique (<2%)</span>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=600, show_spinner="Computing 2D projection...")
def _compute_umap(db_path: str, domain_id: str | None, sample_limit: int):
    """Compute UMAP 2D projection of chunk embeddings."""
    from sklearn.decomposition import PCA

    store = ClusteringStore(db_path)
    chunk_ids, embeddings = store.get_all_chunk_embeddings(domain_id=domain_id)

    if len(chunk_ids) == 0:
        store.close()
        return None

    # Sample if too many
    if len(chunk_ids) > sample_limit:
        indices = np.random.RandomState(42).choice(len(chunk_ids), sample_limit, replace=False)
        chunk_ids = [chunk_ids[i] for i in indices]
        embeddings = embeddings[indices]

    # Get cluster assignments
    assignments = {}
    rows = store.conn.execute(
        "SELECT chunk_id, cluster_id FROM cluster_assignments"
    ).fetchall()
    for cid, clid in rows:
        assignments[cid] = clid

    # Get cluster labels
    cluster_labels = {}
    for c in store.get_clusters():
        cluster_labels[c["cluster_id"]] = c.get("label", c["cluster_id"])

    # Get chunk texts (first 200 chars for hover)
    chunk_texts = {}
    chunk_agrs = {}
    placeholders = ",".join(["?"] * len(chunk_ids))
    text_rows = store.conn.execute(
        f"SELECT chunk_id, chunk_text, agreement_id FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    ).fetchall()
    for cid, txt, agr in text_rows:
        chunk_texts[cid] = (txt or "")[:200]
        chunk_agrs[cid] = agr

    # Get agreement filenames
    agr_names = {}
    for a in store.get_agreements():
        agr_names[a["agreement_id"]] = a.get("filename", a["agreement_id"])

    store.close()

    # UMAP or PCA
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="cosine", random_state=42)
        coords = reducer.fit_transform(embeddings)
    except ImportError:
        # Fallback to PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

    # Build result
    data = []
    for i, cid in enumerate(chunk_ids):
        cluster_id = assignments.get(cid, "unclustered")
        cluster_label = cluster_labels.get(cluster_id, "Unclustered")
        agr_id = chunk_agrs.get(cid, "")
        data.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "chunk_id": cid,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label if len(cluster_label) <= 35
                             else cluster_label[:32] + "...",
            "agreement": agr_names.get(agr_id, agr_id)[:40],
            "text": chunk_texts.get(cid, ""),
        })

    return data


@st.cache_data(ttl=600, show_spinner="Computing cluster internals...")
def _compute_cluster_internals(db_path: str, cluster_id: str):
    """Compute within-cluster similarity structure."""
    from sklearn.decomposition import PCA

    store = ClusteringStore(db_path)
    chunks = store.get_chunks(cluster_id=cluster_id)
    if not chunks:
        store.close()
        return None

    # Get embeddings for these chunks
    chunk_ids = [c["chunk_id"] for c in chunks][:500]  # cap for performance
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = store.conn.execute(
        f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({placeholders}) AND embedding IS NOT NULL",
        chunk_ids
    ).fetchall()

    if len(rows) < 5:
        store.close()
        return None

    ids_with_emb = [r[0] for r in rows]
    embs = np.array([store._deserialize_single_embedding(r[1]) for r in rows])

    # Get centroid
    centroid_data = store.get_all_centroids()
    centroid_entry = centroid_data.get(cluster_id)
    if centroid_entry:
        centroid = centroid_entry[0]
    else:
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    # Compute similarity to centroid
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs_normed = embs / norms
    centroid_normed = centroid / (np.linalg.norm(centroid) + 1e-9)
    similarities = embs_normed @ centroid_normed

    # Get chunk metadata
    chunk_map = {c["chunk_id"]: c for c in chunks}
    agr_names = {}
    for a in store.get_agreements():
        agr_names[a["agreement_id"]] = a.get("filename", a["agreement_id"])

    # PCA 2D for within-cluster view
    try:
        from umap import UMAP
        if len(embs) >= 15:
            reducer = UMAP(n_components=2, n_neighbors=min(10, len(embs)-1),
                           min_dist=0.05, metric="cosine", random_state=42)
            coords = reducer.fit_transform(embs)
        else:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embs)
    except ImportError:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embs)

    # Sub-cluster detection via HDBSCAN or simple k-means
    sub_labels = ["main"] * len(ids_with_emb)
    try:
        from hdbscan import HDBSCAN
        if len(embs) >= 10:
            sub_clusterer = HDBSCAN(min_cluster_size=max(3, len(embs) // 10),
                                     min_samples=2, metric="cosine")
            sub_labels_raw = sub_clusterer.fit_predict(embs)
            sub_labels = [f"sub-{l}" if l >= 0 else "outlier" for l in sub_labels_raw]
    except ImportError:
        pass

    data = []
    for i, cid in enumerate(ids_with_emb):
        ch = chunk_map.get(cid, {})
        agr_id = ch.get("agreement_id", "")
        data.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "chunk_id": cid,
            "similarity": round(float(similarities[i]), 4),
            "sub_cluster": sub_labels[i],
            "agreement": agr_names.get(agr_id, agr_id)[:40],
            "text": (ch.get("chunk_text", "") or "")[:200],
        })

    store.close()
    return data


def _render_micro_explorer(store: ClusteringStore):
    """Micro-cluster explorer: 2D scatter of chunks within clusters."""
    st.markdown("#### Micro-Cluster Explorer")
    st.caption("See what clustered together, sub-groupings within clusters, and outliers")

    col1, col2 = st.columns([1, 3])

    with col1:
        domains = store.get_domains()
        domains = [d for d in domains if d.get("label", "").lower() not in
                   ("all documents", "uncategorized")]
        domain_options = ["All Domains"] + [d.get("label", d["domain_id"]) for d in domains]
        selected_domain = st.selectbox("Domain", domain_options, key="micro_domain")

        domain_id = None
        if selected_domain != "All Domains":
            domain_id = next(
                (d["domain_id"] for d in domains if d.get("label") == selected_domain), None
            )

        clusters = store.get_clusters(domain_id=domain_id)
        clusters = [c for c in clusters if c.get("chunk_count", 0) >= 5]
        clusters.sort(key=lambda c: c.get("chunk_count", 0), reverse=True)

        view_mode = st.radio("View", ["Overview (all clusters)", "Drill into cluster"],
                             key="micro_view")

        sample_limit = st.slider("Sample size", 500, 10000, 3000, 500,
                                 key="micro_sample",
                                 help="Max chunks to project (higher = slower)")

    with col2:
        if view_mode == "Overview (all clusters)":
            data = _compute_umap(str(store.db_path), domain_id, sample_limit)
            if data is None:
                st.warning("No embeddings found.")
                return

            import pandas as pd
            df = pd.DataFrame(data)

            # Count clusters for color ordering
            cluster_counts = df["cluster_label"].value_counts()
            top_clusters = cluster_counts.head(20).index.tolist()
            df["display_cluster"] = df["cluster_label"].apply(
                lambda x: x if x in top_clusters else "Other"
            )

            fig = px.scatter(
                df, x="x", y="y", color="display_cluster",
                hover_data={"text": True, "agreement": True,
                            "cluster_label": True, "x": False, "y": False},
                title=f"2D Projection — {len(df):,} chunks, {df['cluster_label'].nunique()} clusters",
                color_discrete_sequence=DOMAIN_COLORS + px.colors.qualitative.Set3,
            )
            fig.update_traces(marker=dict(size=3, opacity=0.7))
            fig.update_layout(
                height=650,
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9"),
                legend=dict(font=dict(size=10), itemsizing="constant"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            )
            st.plotly_chart(fig, key="umap_overview")

            # Summary stats
            unclustered = (df["cluster_label"] == "Unclustered").sum()
            st.markdown(
                f"**{df['cluster_label'].nunique()}** clusters · "
                f"**{len(df):,}** chunks shown · "
                f"**{unclustered}** unclustered"
            )

        else:
            # Drill-down into a single cluster
            cluster_options = [
                f"{c.get('label', c['cluster_id'])} ({c.get('chunk_count', 0)} chunks)"
                for c in clusters
            ]
            if not cluster_options:
                st.info("No clusters with ≥5 chunks found.")
                return

            selected_idx = st.selectbox(
                "Select cluster", range(len(cluster_options)),
                format_func=lambda i: cluster_options[i],
                key="micro_cluster_select"
            )

            cluster = clusters[selected_idx]
            cluster_id = cluster["cluster_id"]
            cluster_label = cluster.get("label", cluster_id)

            data = _compute_cluster_internals(str(store.db_path), cluster_id)
            if data is None:
                st.warning("Not enough data for this cluster.")
                return

            import pandas as pd
            df = pd.DataFrame(data)

            # Color by sub-cluster, size by similarity
            fig = px.scatter(
                df, x="x", y="y", color="sub_cluster",
                size=np.clip(df["similarity"], 0.3, 1.0),
                hover_data={"text": True, "agreement": True,
                            "similarity": True, "sub_cluster": True,
                            "x": False, "y": False},
                title=f"{cluster_label} — {len(df)} chunks, sub-structure analysis",
                color_discrete_map={
                    "main": "#58a6ff",
                    "outlier": "#f85149",
                    **{f"sub-{i}": DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
                       for i in range(20)},
                },
            )
            fig.update_traces(marker=dict(opacity=0.8))
            fig.update_layout(
                height=600,
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            )
            st.plotly_chart(fig, key="cluster_drill")

            # Similarity distribution
            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = px.histogram(
                    df, x="similarity", nbins=30,
                    title="Similarity to Centroid",
                    color_discrete_sequence=["#58a6ff"],
                )
                fig_hist.update_layout(
                    height=250, margin=dict(t=30, b=20),
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#c9d1d9", size=11),
                )
                st.plotly_chart(fig_hist, key="sim_hist")

            with col_b:
                # Sub-cluster breakdown
                sub_counts = df["sub_cluster"].value_counts()
                n_subs = (sub_counts.index != "outlier").sum()
                n_outliers = sub_counts.get("outlier", 0)
                avg_sim = df["similarity"].mean()
                min_sim = df["similarity"].min()

                st.markdown(f"""
                **Cluster Stats**
                - **Chunks:** {len(df)}
                - **Sub-groups found:** {n_subs}
                - **Outliers:** {n_outliers} ({n_outliers/len(df)*100:.1f}%)
                - **Avg similarity:** {avg_sim:.3f}
                - **Min similarity:** {min_sim:.3f}
                - **Agreements:** {df['agreement'].nunique()}
                """)

            # Outlier table
            outliers = df[df["similarity"] < df["similarity"].quantile(0.05)].sort_values("similarity")
            if len(outliers) > 0:
                st.markdown("##### ⚠️ Lowest Similarity Members (potential misassignments)")
                for _, row in outliers.head(10).iterrows():
                    with st.expander(
                        f"sim={row['similarity']:.3f} | {row['agreement']} | {row['sub_cluster']}"
                    ):
                        st.text(row["text"])


def render(store: ClusteringStore):
    st.title("🗺️ Cluster Map")

    tab1, tab2, tab3 = st.tabs(["📦 Treemap", "🍩 Sunburst", "🔬 Micro-Clusters"])

    with tab1:
        st.markdown("#### Hierarchical Treemap")
        st.caption("Size = clause count. Click to drill into domains → clause types.")
        _render_treemap(store)

    with tab2:
        st.markdown("#### Sunburst — Taxonomy & Rarity")
        st.caption("Ring 1: Domains. Ring 2: Clause types colored by rarity. Click to zoom.")
        _render_sunburst(store)

    with tab3:
        _render_micro_explorer(store)
