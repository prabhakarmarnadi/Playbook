"""Hybrid Search page — Demo Act 6.

BM25 + Dense + Cross-Encoder + RRF with score visualization.
"Dense search understands meaning. BM25 catches exact terms. Together, they miss nothing."
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.store import ClusteringStore
from core.embedder import Embedder
from core.retrieval import HybridRetriever


@st.cache_resource
def build_retriever(_store: ClusteringStore):
    """Build hybrid retriever from all chunks in the store."""
    chunk_ids, embeddings = _store.get_all_chunk_embeddings()
    if len(chunk_ids) == 0:
        return None

    # Get chunk texts
    all_chunks = _store.get_chunks()
    text_map = {c["chunk_id"]: c["chunk_text"] for c in all_chunks}
    meta_map = {c["chunk_id"]: {"agreement_id": c.get("agreement_id", ""), "idx": c.get("chunk_index", 0)} for c in all_chunks}

    texts = [text_map.get(cid, "") for cid in chunk_ids]
    metadata = [meta_map.get(cid, {}) for cid in chunk_ids]

    embedder = Embedder.get_instance()
    return HybridRetriever(
        chunks=texts,
        embeddings=embeddings,
        embed_fn=lambda q: embedder.embed_single(q),
        chunk_metadata=metadata,
    )


def render(store: ClusteringStore):
    st.header("🔍 Hybrid Search")
    st.caption("Dense (semantic) + BM25 (keyword) + Cross-Encoder reranking with score breakdown.")

    retriever = build_retriever(store)
    if retriever is None:
        st.info("No chunks indexed. Run the pipeline first.")
        return

    # ── Search input ───────────────────────────────────────────────────
    query = st.text_input(
        "Search query",
        placeholder='e.g., "liability cap exceeding $5M in SaaS agreements"',
    )

    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Results", 5, 50, 10)
    use_reranker = col2.checkbox("Cross-Encoder Reranking", value=False)
    show_scores = col3.checkbox("Show Score Breakdown", value=True)

    if not query:
        st.info("Enter a search query above.")
        return

    # ── Search ─────────────────────────────────────────────────────────
    with st.spinner("Searching..."):
        results = retriever.search(query, top_k=top_k, use_reranker=use_reranker)

    if not results:
        st.warning("No results found.")
        return

    st.success(f"Found {len(results)} results" +
               (" (reranked by cross-encoder)" if use_reranker else ""))

    # ── Score breakdown table ──────────────────────────────────────────
    if show_scores:
        score_data = []
        for r in results:
            row = {
                "Rank": len(score_data) + 1,
                "Text Preview": r["chunk"][:100] + "...",
                "Dense": f"{r.get('dense_score', 0):.3f}",
                "BM25": f"{r.get('bm25_score', 0):.3f}",
                "RRF": f"{r.get('rrf_score', 0):.4f}",
            }
            if use_reranker:
                row["CrossEncoder"] = f"{r.get('cross_encoder_score', 0):.3f}"
            score_data.append(row)
        st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

    # ── Individual results ─────────────────────────────────────────────
    st.subheader("Results")
    for i, r in enumerate(results):
        with st.expander(f"#{i+1} — Score: {r.get('rrf_score', 0):.4f}", expanded=(i < 3)):
            # Score badges
            badge_cols = st.columns(4 if use_reranker else 3)
            badge_cols[0].metric("Dense", f"{r.get('dense_score', 0):.3f}")
            badge_cols[1].metric("BM25", f"{r.get('bm25_score', 0):.3f}")
            badge_cols[2].metric("RRF Combined", f"{r.get('rrf_score', 0):.4f}")
            if use_reranker:
                badge_cols[3].metric("Cross-Encoder", f"{r.get('cross_encoder_score', 0):.3f}")

            # Highlight which signal contributed more
            dense = r.get("dense_score", 0)
            bm25 = r.get("bm25_score", 0)
            if bm25 > dense * 1.5:
                st.caption("📌 BM25 dominant — likely an exact keyword match")
            elif dense > bm25 * 1.5:
                st.caption("🧠 Dense dominant — semantic meaning match")
            else:
                st.caption("⚖️ Balanced — both signals contributed")

            st.text_area("Full text", r["chunk"], height=120, disabled=True, key=f"res_{i}")
            if r.get("metadata", {}).get("agreement_id"):
                st.caption(f"Agreement: {r['metadata']['agreement_id']}")
