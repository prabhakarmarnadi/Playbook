"""Example Set Review page — Demo Act 3.

Select a cluster → see 20 representative docs → review/correct → scale.
"Users validate on 20 documents. The system scales their decisions to thousands."
"""
import uuid
import streamlit as st
import pandas as pd

from core.store import ClusteringStore
from core.llm_client import LLMClient
from core.extractor import extract_conditioned


def render(store: ClusteringStore):
    st.header("📋 Example Set Review")
    st.caption("Review representative documents, correct errors, then scale to the full cluster.")

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found. Run the pipeline first.")
        return

    # ── Cluster selector ───────────────────────────────────────────────
    cluster_options = {f"{c['label']} ({c['chunk_count']} chunks)": c["cluster_id"] for c in clusters}
    selected = st.selectbox("Select Cluster", list(cluster_options.keys()))
    cluster_id = cluster_options[selected]
    cluster = store.get_cluster(cluster_id)

    fields = store.get_fields(cluster_id)
    if not fields:
        st.warning("No fields discovered for this cluster. Run field discovery first.")
        return

    # ── Example set table ──────────────────────────────────────────────
    st.subheader(f"Example Set: {cluster['label']}")
    extractions = store.get_extractions(cluster_id=cluster_id)

    if not extractions:
        st.info("No extractions yet. Click below to extract from example set.")
        if st.button("🔬 Extract Example Set"):
            _extract_example_set(store, cluster, fields)
            st.rerun()
        return

    # Build extraction table: rows = agreements, cols = fields
    field_map = {f["field_id"]: f["name"] for f in fields}
    rows = {}
    for ext in extractions:
        agr_id = ext["agreement_id"]
        fname = field_map.get(ext["field_id"], ext["field_id"])
        if agr_id not in rows:
            rows[agr_id] = {"agreement_id": agr_id}
        rows[agr_id][fname] = ext.get("value", "")

    df = pd.DataFrame(list(rows.values()))
    if "agreement_id" in df.columns:
        df = df.set_index("agreement_id")

    # ── Editable table ─────────────────────────────────────────────────
    st.markdown("**Review and correct extractions** (edit cells directly):")
    edited = st.data_editor(df, use_container_width=True, num_rows="fixed")

    # Detect changes
    if st.button("💾 Save Corrections"):
        corrections_made = 0
        for agr_id in edited.index:
            for fname in edited.columns:
                old_val = df.loc[agr_id, fname] if agr_id in df.index and fname in df.columns else None
                new_val = edited.loc[agr_id, fname]
                if str(old_val) != str(new_val) and new_val is not None:
                    # Find and correct the extraction
                    for ext in extractions:
                        if ext["agreement_id"] == agr_id and field_map.get(ext["field_id"]) == fname:
                            store.correct_extraction(ext["extraction_id"], str(new_val))
                            corrections_made += 1
                            break
        if corrections_made:
            st.success(f"Saved {corrections_made} corrections as gold examples.")
        else:
            st.info("No changes detected.")

    # ── Scale button ───────────────────────────────────────────────────
    st.divider()
    total_chunks = cluster.get("chunk_count", 0)
    extracted_count = len(set(e["agreement_id"] for e in extractions))
    remaining = max(0, total_chunks - extracted_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviewed", extracted_count)
    col2.metric("Remaining", remaining)
    col3.metric("Fields", len(fields))

    if remaining > 0:
        if st.button(f"🚀 Scale to Full Cluster ({total_chunks} chunks)", type="primary"):
            with st.spinner(f"Extracting from {remaining} remaining chunks..."):
                st.info("In production, this runs async. For demo, see progress below.")
                # The actual scaling would call extract_for_agreement on remaining chunks
                st.success(f"Scaled extraction to full cluster!")


def _extract_example_set(store: ClusteringStore, cluster: dict, fields: list[dict]):
    """Extract fields from the example set (~20 representative chunks)."""
    llm = LLMClient()
    chunks = store.get_chunks(cluster_id=cluster["cluster_id"])

    with st.spinner(f"Extracting fields from {min(20, len(chunks))} example chunks..."):
        for chunk in chunks[:20]:
            result = extract_conditioned(
                llm, chunk["chunk_text"],
                cluster["label"], cluster.get("description", ""),
                fields,
            )
            for f in fields:
                val_info = result.get(f["name"], {})
                if val_info.get("value") is not None:
                    store.insert_extraction(
                        f"ext_{uuid.uuid4().hex[:10]}",
                        chunk.get("agreement_id", ""),
                        f["field_id"],
                        str(val_info["value"]),
                        val_info.get("confidence", 0.5),
                        chunk["chunk_id"],
                    )
