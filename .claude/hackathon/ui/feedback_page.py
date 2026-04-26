"""Feedback & Personalization page — Demo Act 9.

Interactive playground for the Feedback & Personalization Engine.
Record corrections, manage virtual clusters, tune thresholds, view feedback history.
"Every correction makes the system smarter — for YOUR organization."
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.store import ClusteringStore


def _get_injector(tenant_id: str, main_db_path: str, data_dir: str):
    """Create or retrieve a cached PersonalizationInjector."""
    from core.personalization_injector import PersonalizationInjector
    return PersonalizationInjector(
        tenant_id=tenant_id,
        main_db_path=main_db_path,
        data_dir=data_dir,
    )


def render(store: ClusteringStore):
    st.header("🔄 Feedback & Personalization")
    st.caption(
        "Record corrections, manage virtual clusters, override intent labels, "
        "and tune thresholds. All changes are tenant-scoped overlays — "
        "the core ontology stays clean."
    )

    data_dir = str(Path(store.db_path).parent)
    main_db = store.db_path

    # ── Tenant selector ─────────────────────────────────────────────────
    col_t1, col_t2 = st.columns([2, 1])
    tenant_id = col_t1.text_input(
        "Tenant ID",
        value=st.session_state.get("feedback_tenant", "demo_tenant"),
        help="Each tenant gets an isolated sidecar database.",
    )
    st.session_state["feedback_tenant"] = tenant_id
    user_id = col_t2.text_input("User ID", value="demo_user")

    injector = _get_injector(tenant_id, main_db, data_dir)

    # ── Tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "💼 Cluster Corrections",
        "🎯 Intent Overrides",
        "🔗 KG Edge Tuning",
        "🏗️ Virtual Clusters",
        "⚙️ Thresholds",
        "📜 Feedback Log",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 0: OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    with tabs[0]:
        _render_overview(injector, store)

    # ════════════════════════════════════════════════════════════════════
    # TAB 1: CLUSTER CORRECTIONS
    # ════════════════════════════════════════════════════════════════════
    with tabs[1]:
        _render_cluster_corrections(injector, store, user_id)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2: INTENT OVERRIDES
    # ════════════════════════════════════════════════════════════════════
    with tabs[2]:
        _render_intent_overrides(injector, store, user_id)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3: KG EDGE TUNING
    # ════════════════════════════════════════════════════════════════════
    with tabs[3]:
        _render_edge_tuning(injector, store, user_id)

    # ════════════════════════════════════════════════════════════════════
    # TAB 4: VIRTUAL CLUSTERS
    # ════════════════════════════════════════════════════════════════════
    with tabs[4]:
        _render_virtual_clusters(injector, store, user_id)

    # ════════════════════════════════════════════════════════════════════
    # TAB 5: THRESHOLDS
    # ════════════════════════════════════════════════════════════════════
    with tabs[5]:
        _render_thresholds(injector)

    # ════════════════════════════════════════════════════════════════════
    # TAB 6: FEEDBACK LOG
    # ════════════════════════════════════════════════════════════════════
    with tabs[6]:
        _render_feedback_log(injector)


# ═══════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═══════════════════════════════════════════════════════════════════════════


def _render_overview(injector, store: ClusteringStore):
    """Dashboard overview of personalization state."""
    summary = injector.store.summary()
    ctx = injector.ctx

    # Top metrics
    cols = st.columns(4)
    cols[0].metric("📝 Feedback Events", summary["feedback_events"])
    cols[1].metric("🏗️ Virtual Clusters", summary["virtual_clusters"])
    cols[2].metric("🎯 Intent Overrides", summary["intent_overrides"])
    cols[3].metric("⚖️ Weight Adjustments", summary["personalization_weights"])

    st.divider()

    # Context snapshot
    st.subheader("Personalization Context (Runtime Snapshot)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Thresholds**")
        st.json({
            "high_confidence_threshold": ctx.high_confidence_threshold,
            "tentative_threshold": ctx.tentative_threshold,
            "outlier_min_similarity": ctx.outlier_min_similarity,
            "intent_match_threshold": ctx.intent_match_threshold,
            "intent_temperature": ctx.intent_temperature,
        })

    with col2:
        st.markdown("**Active Overrides**")
        st.json({
            "edge_weight_overrides": len(ctx.edge_weights),
            "cluster_weight_overrides": len(ctx.cluster_weights),
            "correction_examples": len(ctx.correction_examples),
            "virtual_clusters": len(ctx.virtual_clusters),
            "intent_overrides": len(ctx.intent_overrides),
            "approved_entities": len(ctx.approved_entities),
        })

    # Feedback distribution chart
    history = injector.store.get_feedback_history(limit=500)
    if history:
        df = pd.DataFrame(history)
        if "action" in df.columns:
            st.subheader("Feedback Distribution")
            fig = px.histogram(
                df, x="action", color="entity_type",
                barmode="group",
                title="Feedback Actions by Entity Type",
                category_orders={"action": [
                    "approve", "correct", "reject", "refine",
                    "link", "unlink", "merge", "split",
                ]},
            )
            fig.update_layout(xaxis_title="Action", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)


def _render_cluster_corrections(injector, store: ClusteringStore, user_id: str):
    """Correct cluster assignments for individual chunks."""
    st.subheader("Correct Cluster Assignments")
    st.caption(
        "Move a chunk from one cluster to another, or create a new virtual cluster."
    )

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found. Run the pipeline first.")
        return

    cluster_map = {c["cluster_id"]: c["label"] for c in clusters}
    cluster_options = {f"{c['label']} ({c['chunk_count']} chunks)": c["cluster_id"]
                       for c in clusters}

    # Select source cluster
    src_label = st.selectbox(
        "Source Cluster",
        list(cluster_options.keys()),
        key="fb_src_cluster",
    )
    src_cluster_id = cluster_options[src_label]

    # Show chunks in this cluster
    chunks = store.get_chunks(cluster_id=src_cluster_id)
    if not chunks:
        st.info("No chunks in this cluster.")
        return

    chunk_options = {
        f"[{c.get('chunk_index', '?')}] {c['chunk_text'][:80]}…": c["chunk_id"]
        for c in chunks[:50]
    }
    selected_chunk_label = st.selectbox(
        f"Select Chunk ({len(chunks)} total, showing first 50)",
        list(chunk_options.keys()),
        key="fb_chunk_select",
    )
    selected_chunk_id = chunk_options[selected_chunk_label]

    # Show full chunk text
    chunk_obj = next((c for c in chunks if c["chunk_id"] == selected_chunk_id), None)
    if chunk_obj:
        with st.expander("📄 Full chunk text", expanded=False):
            st.text(chunk_obj["chunk_text"][:1000])

    st.markdown("---")

    # Target: existing cluster or new virtual cluster
    correction_type = st.radio(
        "Correct to:",
        ["Existing Cluster", "New Virtual Cluster"],
        horizontal=True,
        key="fb_correction_type",
    )

    if correction_type == "Existing Cluster":
        # Remove source from target options
        target_options = {k: v for k, v in cluster_options.items()
                         if v != src_cluster_id}
        target_label = st.selectbox("Target Cluster", list(target_options.keys()),
                                    key="fb_tgt_cluster")
        target_id = target_options[target_label]

        if st.button("✅ Apply Cluster Correction", key="fb_apply_cluster"):
            injector.record_cluster_correction(
                user_id=user_id,
                chunk_id=selected_chunk_id,
                old_cluster_id=src_cluster_id,
                new_cluster_id=target_id,
            )
            st.success(
                f"Moved chunk to **{cluster_map.get(target_id, target_id)}**. "
                f"Weight for {cluster_map.get(src_cluster_id, src_cluster_id)} "
                f"reduced, weight for target boosted."
            )
            injector.invalidate_cache()
            st.rerun()
    else:
        new_label = st.text_input("Virtual Cluster Label",
                                  placeholder="e.g., Custom NDA Clause",
                                  key="fb_vc_label")
        if st.button("🏗️ Create & Assign to Virtual Cluster",
                     key="fb_apply_virtual"):
            if new_label:
                result = injector.record_cluster_correction(
                    user_id=user_id,
                    chunk_id=selected_chunk_id,
                    old_cluster_id=src_cluster_id,
                    new_cluster_label=new_label,
                )
                st.success(
                    f"Created virtual cluster **{new_label}** "
                    f"(ID: {result.get('virtual_cluster_id', '?')}) "
                    f"and assigned chunk."
                )
                injector.invalidate_cache()
                st.rerun()
            else:
                st.warning("Please provide a label for the virtual cluster.")


def _render_intent_overrides(injector, store: ClusteringStore, user_id: str):
    """Manage intent label overrides."""
    st.subheader("Intent Label Overrides")
    st.caption(
        "Map pipeline-generated intent labels to your organization's vocabulary."
    )

    # Existing overrides
    overrides = injector.store.get_intent_overrides()
    if overrides:
        st.markdown("**Active Overrides**")
        override_data = [
            {
                "Pipeline Label": gl,
                "Your Label": info["tenant_label"],
                "Summary": info["summary"],
                "Clause Types": ", ".join(info["clause_types"]) or "All",
            }
            for gl, info in overrides.items()
        ]
        st.dataframe(pd.DataFrame(override_data), use_container_width=True,
                      hide_index=True)
        st.divider()

    # Get existing intent labels from the pipeline
    intents = store.get_intent_types() if hasattr(store, "get_intent_types") else []
    intent_labels = [it.get("label", "") for it in intents if it.get("label")]

    st.markdown("**Add New Override**")
    with st.form("intent_override_form"):
        col1, col2 = st.columns(2)

        if intent_labels:
            old_label = col1.selectbox(
                "Pipeline Intent Label",
                intent_labels,
                key="fb_old_intent",
            )
        else:
            old_label = col1.text_input(
                "Pipeline Intent Label",
                placeholder="e.g., Limit Liability",
                key="fb_old_intent_text",
            )

        new_label = col2.text_input(
            "Your Organization's Label",
            placeholder="e.g., Liability Cap",
            key="fb_new_intent",
        )
        new_summary = st.text_input(
            "Description (optional)",
            placeholder="e.g., Sets maximum monetary exposure",
            key="fb_intent_summary",
        )

        # Optional clause type filter
        clusters = store.get_clusters()
        clause_type_ids = [c["cluster_id"] for c in clusters]
        clause_type_labels = [c["label"] for c in clusters]
        selected_ct = st.multiselect(
            "Apply to clause types (optional — leave empty for all)",
            clause_type_labels,
            key="fb_intent_clause_types",
        )
        selected_ct_ids = [
            clause_type_ids[clause_type_labels.index(l)]
            for l in selected_ct
        ] if selected_ct else []

        if st.form_submit_button("🎯 Add Override"):
            if old_label and new_label:
                injector.record_intent_correction(
                    user_id=user_id,
                    clause_id="global",
                    old_label=old_label,
                    new_label=new_label,
                    new_summary=new_summary,
                    clause_type_id=selected_ct_ids[0] if selected_ct_ids else None,
                )
                st.success(f"Override added: **{old_label}** → **{new_label}**")
                injector.invalidate_cache()
                st.rerun()
            else:
                st.warning("Both labels are required.")


def _render_edge_tuning(injector, store: ClusteringStore, user_id: str):
    """Tune Knowledge Graph edge weights."""
    st.subheader("Knowledge Graph Edge Tuning")
    st.caption("Strengthen or weaken relationships between clause types.")

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found.")
        return

    cluster_labels = [c["label"] for c in clusters]
    cluster_id_map = {c["label"]: c["cluster_id"] for c in clusters}

    col1, col2 = st.columns(2)
    from_label = col1.selectbox("From Clause Type", cluster_labels,
                                key="fb_edge_from")
    to_label = col2.selectbox("To Clause Type", cluster_labels,
                              index=min(1, len(cluster_labels) - 1),
                              key="fb_edge_to")

    from_id = cluster_id_map[from_label]
    to_id = cluster_id_map[to_label]

    # Show current personalized weight
    current_weight = injector.get_edge_weight(from_id, to_id, base_weight=1.0)
    st.metric("Current Personalized Weight", f"{current_weight:.3f}",
              delta=f"{current_weight - 1.0:+.3f}" if current_weight != 1.0 else None)

    action = st.radio(
        "Action",
        ["link (strengthen +0.20)", "unlink (weaken −0.30)"],
        horizontal=True,
        key="fb_edge_action",
    )
    action_val = "link" if "link" in action else "unlink"

    reason = st.text_input("Reason (optional)",
                           placeholder="These always co-occur in our contracts",
                           key="fb_edge_reason")

    if st.button("⚡ Apply Edge Adjustment", key="fb_apply_edge"):
        new_weight = injector.record_edge_correction(
            user_id=user_id,
            from_entity=from_id,
            to_entity=to_id,
            action=action_val,
            reason=reason,
        )
        verb = "Strengthened" if action_val == "link" else "Weakened"
        st.success(
            f"{verb} edge **{from_label}** ↔ **{to_label}**. "
            f"New weight: {new_weight:.3f}"
        )
        st.rerun()

    # Show all adjusted edges
    edge_weights = injector.ctx.edge_weights
    if edge_weights:
        st.divider()
        st.markdown("**All Adjusted Edges**")
        edge_data = []
        for (f_id, t_id), w in edge_weights.items():
            edge_data.append({
                "From": next((c["label"] for c in clusters
                              if c["cluster_id"] == f_id), f_id),
                "To": next((c["label"] for c in clusters
                            if c["cluster_id"] == t_id), t_id),
                "Multiplier": f"{w:.3f}",
                "Direction": "↑ Boosted" if w > 1.0 else "↓ Weakened",
            })
        if edge_data:
            st.dataframe(pd.DataFrame(edge_data), use_container_width=True,
                         hide_index=True)


def _render_virtual_clusters(injector, store: ClusteringStore, user_id: str):
    """Manage tenant-specific virtual clusters."""
    st.subheader("Virtual Clusters")
    st.caption(
        "Virtual clusters are tenant-specific groupings for chunks "
        "that don't fit the global taxonomy."
    )

    # Existing virtual clusters
    vcs = injector.store.get_virtual_clusters()
    if vcs:
        st.markdown("**Existing Virtual Clusters**")
        vc_data = [
            {
                "ID": vc["cluster_id"],
                "Label": vc["label"],
                "Members": vc.get("member_count", 0),
                "Created By": vc.get("created_by", "?"),
                "Description": vc.get("description", ""),
            }
            for vc in vcs
        ]
        st.dataframe(pd.DataFrame(vc_data), use_container_width=True,
                      hide_index=True)

        # Promotion check
        for vc in vcs:
            if vc.get("member_count", 0) >= 3 and not vc.get("centroid"):
                st.info(
                    f"💡 **{vc['label']}** has {vc['member_count']} members. "
                    f"Consider promoting it to compute a centroid for "
                    f"automatic assignment."
                )
                if st.button(f"🚀 Promote {vc['label']}",
                             key=f"promote_{vc['cluster_id']}"):
                    injector.engine.promote_virtual_cluster(vc["cluster_id"])
                    st.success(f"Promoted **{vc['label']}** — centroid computed.")
                    injector.invalidate_cache()
                    st.rerun()
    else:
        st.info("No virtual clusters yet. Create one from the Cluster Corrections tab.")

    # Manual creation
    st.divider()
    st.markdown("**Create Virtual Cluster Manually**")
    with st.form("create_vc_form"):
        vc_label = st.text_input("Label", placeholder="e.g., GDPR Data Processing")
        vc_desc = st.text_input("Description",
                                placeholder="Clauses specific to EU data processing requirements")
        if st.form_submit_button("🏗️ Create"):
            if vc_label:
                vc_id = injector.store.create_virtual_cluster(
                    label=vc_label,
                    user_id=user_id,
                    description=vc_desc,
                )
                st.success(f"Created virtual cluster **{vc_label}** (ID: {vc_id})")
                injector.invalidate_cache()
                st.rerun()
            else:
                st.warning("Label is required.")


def _render_thresholds(injector):
    """View and tune tenant-specific thresholds."""
    st.subheader("Threshold Configuration")
    st.caption("Adjust pipeline thresholds for your organization.")

    config = injector.store.get_tenant_config()

    with st.form("threshold_form"):
        col1, col2 = st.columns(2)

        hc = col1.slider(
            "High Confidence Threshold",
            min_value=0.50, max_value=1.00,
            value=float(config.get("high_confidence_threshold", 0.85)),
            step=0.01,
            help="Chunks above this similarity are assigned with high confidence.",
        )
        tt = col2.slider(
            "Tentative Threshold",
            min_value=0.30, max_value=0.90,
            value=float(config.get("tentative_threshold", 0.60)),
            step=0.01,
            help="Chunks between tentative and high confidence get tentative assignment.",
        )
        oms = col1.slider(
            "Outlier Min Similarity",
            min_value=0.10, max_value=0.60,
            value=float(config.get("outlier_min_similarity", 0.35)),
            step=0.01,
            help="Minimum similarity for outlier recovery in ontology refiner.",
        )
        imt = col2.slider(
            "Intent Match Threshold",
            min_value=0.50, max_value=1.00,
            value=float(config.get("intent_match_threshold", 0.85)),
            step=0.01,
            help="Minimum similarity for matching intents to registry.",
        )
        it = col1.slider(
            "Intent Temperature",
            min_value=0.0, max_value=1.0,
            value=float(config.get("intent_temperature", 0.0)),
            step=0.05,
            help="LLM temperature for intent extraction. 0 = deterministic.",
        )
        half_life = col2.number_input(
            "Feedback Decay Half-life (days)",
            min_value=7, max_value=365,
            value=int(config.get("edge_decay_half_life_days", 90)),
            help="Older feedback decays toward neutral. Half-life in days.",
        )

        col_btn1, col_btn2 = st.columns(2)
        save = col_btn1.form_submit_button("💾 Save Thresholds")
        auto = col_btn2.form_submit_button("🤖 Auto-Tune from Feedback")

    if save:
        injector.store.update_tenant_config(
            high_confidence_threshold=hc,
            tentative_threshold=tt,
            outlier_min_similarity=oms,
            intent_match_threshold=imt,
            intent_temperature=it,
            edge_decay_half_life_days=half_life,
        )
        st.success("Thresholds saved.")
        injector.invalidate_cache()
        st.rerun()

    if auto:
        result = injector.auto_tune()
        if result:
            st.success("Auto-tune applied:")
            st.json(result)
            injector.invalidate_cache()
            st.rerun()
        else:
            st.info("Not enough feedback data to auto-tune yet.")


def _render_feedback_log(injector):
    """Show raw feedback event log."""
    st.subheader("Feedback Event Log")

    col1, col2, col3 = st.columns(3)
    entity_filter = col1.selectbox(
        "Entity Type",
        ["All", "cluster", "edge", "intent", "field", "clause_name", "domain"],
        key="fb_log_entity",
    )
    limit = col2.number_input("Limit", value=50, min_value=10, max_value=500,
                              step=10, key="fb_log_limit")

    entity_type = None if entity_filter == "All" else entity_filter
    history = injector.store.get_feedback_history(
        entity_type=entity_type,
        limit=limit,
    )

    if history:
        df = pd.DataFrame(history)
        # Clean up display
        display_cols = [
            c for c in [
                "created_at", "user_id", "entity_type", "entity_id",
                "action", "old_value", "new_value",
            ]
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols] if display_cols else df,
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv = df.to_csv(index=False)
        col3.download_button(
            "📥 Export CSV",
            csv,
            f"feedback_{injector.tenant_id}.csv",
            "text/csv",
            key="fb_export",
        )
    else:
        st.info("No feedback events yet. Start by making corrections in the other tabs.")
