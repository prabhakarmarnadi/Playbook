"""
Hybrid retrieval — BM25 + Dense + Cross-Encoder + RRF fusion.
NEW V2 capability (doesn't exist in V1).

Legacy pain point: V1 uses HyDE → single ANN search via AIDB.
This misses exact keyword matches (statute numbers, dollar amounts, specific terms).

V2 approach: Three-signal retrieval:
  1. Dense search (FAISS) — semantic similarity
  2. BM25 (sparse) — exact keyword matching
  3. Cross-encoder reranking — fine-grained relevance scoring
  Combined with Reciprocal Rank Fusion (RRF).

Demo impact: "Dense search understands meaning. BM25 catches exact terms. Together, they miss nothing."
"""
import logging

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, RRF_K, SEARCH_TOP_K

logger = logging.getLogger(__name__)


# Module-level cross-encoder cache — avoids reloading 250MB model per retriever instance
_cross_encoder_cache: dict[str, CrossEncoder] = {}


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder, loading it only on first request."""
    if model_name not in _cross_encoder_cache:
        logger.info(f"Loading CrossEncoder: {model_name}")
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


class HybridRetriever:
    """Three-signal hybrid retrieval with RRF fusion and optional cross-encoder reranking."""

    def __init__(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        embed_fn,
        chunk_metadata: list[dict] | None = None,
        reranker_model: str | None = None,
    ):
        self.chunks = chunks
        self.embed_fn = embed_fn
        self.metadata = chunk_metadata or [{} for _ in chunks]

        # Load cross-encoder reranker (cached)
        model_name = reranker_model or CROSS_ENCODER_MODEL
        self.reranker = _get_cross_encoder(model_name)

        # Dense index (FAISS Inner Product on normalized vectors)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        normed = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normed)
        self.faiss_index.add(normed)

        # Sparse index (BM25)
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(f"HybridRetriever initialized: {len(chunks)} chunks, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = False,
        rrf_k: int | None = None,
    ) -> list[dict]:
        """
        Hybrid search: Dense + BM25 + RRF fusion + optional cross-encoder reranking.

        Returns list of result dicts with individual scores for visualization:
            {chunk, idx, rrf_score, dense_score, bm25_score, [cross_encoder_score], metadata}
        """
        k = top_k or SEARCH_TOP_K
        rk = rrf_k or RRF_K

        # ── Dense search ───────────────────────────────────────────────────
        q_vec = self.embed_fn(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)
        dense_scores, dense_ids = self.faiss_index.search(q_vec, min(k, len(self.chunks)))

        # ── BM25 search ───────────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[-k:][::-1]

        # ── RRF fusion ─────────────────────────────────────────────────────
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_ids[0]):
            idx_int = int(idx)
            if idx_int >= 0:
                rrf[idx_int] = rrf.get(idx_int, 0) + 1.0 / (rk + rank)
        for rank, idx in enumerate(bm25_top):
            idx_int = int(idx)
            rrf[idx_int] = rrf.get(idx_int, 0) + 1.0 / (rk + rank)

        # Build results with per-signal scores
        dense_id_set = set(int(x) for x in dense_ids[0])
        results = []
        for idx, rrf_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]:
            # Get dense score for this idx
            if idx in dense_id_set:
                dense_rank_pos = list(dense_ids[0]).index(idx)
                d_score = float(dense_scores[0][dense_rank_pos])
            else:
                d_score = 0.0

            results.append({
                "chunk": self.chunks[idx],
                "idx": idx,
                "rrf_score": rrf_score,
                "dense_score": d_score,
                "bm25_score": float(bm25_scores[idx]),
                "metadata": self.metadata[idx],
            })

        # ── Optional cross-encoder reranking ───────────────────────────────
        if use_reranker and results:
            pairs = [(query, r["chunk"]) for r in results]
            ce_scores = self.reranker.predict(pairs)
            for i, r in enumerate(results):
                r["cross_encoder_score"] = float(ce_scores[i])
            results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        return results
