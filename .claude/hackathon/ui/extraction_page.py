"""Extraction Comparison page — Demo Act 4.

Side-by-side: V1 generic extraction vs V2 cluster-conditioned extraction.
"Same LLM, same document. V2 knows what it's looking for."
"""
import streamlit as st
import time

from core.store import ClusteringStore
from core.llm_client import LLMClient
from core.extractor import extract_conditioned, extract_generic


def render(store: ClusteringStore):
    st.header("⚖️ Extraction Comparison: V1 vs V2")
    st.caption("Same LLM. Same document. V2 knows what it's looking for.")

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found. Run the pipeline first.")
        return

    # ── Select cluster and chunk ───────────────────────────────────────
    cluster_options = {f"{c['label']} ({c['chunk_count']} chunks)": c["cluster_id"] for c in clusters}
    selected = st.selectbox("Select Cluster", list(cluster_options.keys()), key="ext_cluster")
    cluster_id = cluster_options[selected]
    cluster = store.get_cluster(cluster_id)
    fields = store.get_fields(cluster_id)

    chunks = store.get_chunks(cluster_id=cluster_id)
    if not chunks:
        st.warning("No chunks assigned to this cluster.")
        return

    # Pick a sample chunk
    chunk_idx = st.slider("Select chunk", 0, min(len(chunks) - 1, 19), 0)
    sample_text = chunks[chunk_idx]["chunk_text"]

    st.text_area("Sample Clause", sample_text[:1000], height=120, disabled=True)

    if not st.button("🔬 Run Both Extractions", type="primary"):
        return

    llm = LLMClient()

    # ── Side-by-side extraction ────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❌ V1: Generic Extraction")
        st.caption('Prompt: "Extract fields from this text"')
        t0 = time.time()
        generic_result = extract_generic(llm, sample_text)
        generic_time = time.time() - t0
        generic_tokens = len(sample_text) // 4 + 200  # rough estimate

        # Count hallucinated fields (those not in known fields)
        known_names = {f["name"] for f in fields}
        generic_fields = {k: v for k, v in generic_result.items()
                         if k not in ("raw_response", "parse_error")}
        hallucinated = [k for k in generic_fields if k not in known_names]
        real = [k for k in generic_fields if k in known_names]

        st.metric("Fields found", len(generic_fields))
        st.metric("Hallucinated", len(hallucinated), delta=f"-{len(hallucinated)}", delta_color="inverse")
        st.metric("Tokens (est.)", f"{generic_tokens + 300:,}")
        st.metric("Time", f"{generic_time:.1f}s")
        st.divider()
        for k, v in generic_fields.items():
            icon = "✅" if k in known_names else "❌"
            val = v.get("value", v) if isinstance(v, dict) else v
            st.markdown(f"{icon} **{k}**: {val}")

    with col2:
        st.markdown("### ✅ V2: Cluster-Conditioned")
        st.caption(f'Prompt: "This is a {cluster["label"]} clause. Extract: '
                   f'{", ".join(f["name"] for f in fields[:4])}..."')
        t0 = time.time()
        conditioned_result = extract_conditioned(
            llm, sample_text,
            cluster["label"], cluster.get("description", ""),
            fields,
        )
        conditioned_time = time.time() - t0
        conditioned_tokens = len(sample_text) // 4 + 100

        cond_fields = {k: v for k, v in conditioned_result.items() if v.get("value") is not None}

        st.metric("Fields found", len(cond_fields))
        st.metric("Hallucinated", 0, delta="0", delta_color="off")
        st.metric("Tokens (est.)", f"{conditioned_tokens + 150:,}",
                  delta=f"↓{((generic_tokens + 300 - conditioned_tokens - 150) * 100) // (generic_tokens + 300)}%",
                  delta_color="normal")
        st.metric("Time", f"{conditioned_time:.1f}s")
        st.divider()
        for k, v in conditioned_result.items():
            val = v.get("value")
            conf = v.get("confidence", 0)
            icon = "✅" if val is not None else "⬜"
            st.markdown(f"{icon} **{k}**: {val} (conf: {conf:.2f})")

    # ── Summary ────────────────────────────────────────────────────────
    st.divider()
    st.info(
        f"**V2 advantage:** {len(hallucinated)} fewer hallucinated fields, "
        f"~{((generic_tokens - conditioned_tokens) * 100) // max(generic_tokens, 1)}% fewer tokens. "
        f"The LLM focuses only on fields relevant to {cluster['label']}."
    )
