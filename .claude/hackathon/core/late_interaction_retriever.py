"""
Late-interaction retrieval using ColBERT MaxSim scoring.

Replaces single-vector dense search with per-token interaction:
  - Each chunk → matrix of token embeddings (not one collapsed vector)
  - Query → matrix of token embeddings
  - Score = sum of per-query-token MaxSim across chunk tokens

Combined with BM25 via RRF fusion (same as existing HybridRetriever).

Why this matters for legal extraction:
  - Queries are compositional: "termination notice period in days"
  - Single-vector collapses "termination" + "notice" + "period" + "days" into one point
  - MaxSim matches each query token independently against the best chunk token
  - Legal synonyms handled per-token: "terminate"↔"cancel", "notice"↔"notification"

Usage:
    retriever = LateInteractionRetriever(chunks, embed_fn, chunk_metadata)
    results = retriever.search("termination notice period", top_k=5)
"""

import logging
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from config import RRF_K, SEARCH_TOP_K

logger = logging.getLogger(__name__)

# ── ColBERT checkpoint cache (singleton) ─────────────────────────────────────
_colbert_checkpoint = None

COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERT_DIM = 128  # ColBERTv2 projects 768 → 128 via trained linear layer


def _load_colbert():
    """Load the ColBERTv2 checkpoint once with its trained 768→128 linear projection."""
    global _colbert_checkpoint
    if _colbert_checkpoint is not None:
        return _colbert_checkpoint

    from colbert.infra.config import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    logger.info(f"Loading ColBERT model: {COLBERT_MODEL_NAME}")
    # query_maxlen=32: our field queries are short (~5-15 tokens).
    # ColBERT pads to query_maxlen with [MASK] tokens that participate in
    # MaxSim scoring — too many masks add noise.  32 is ample for
    # "field_name field_description examples: ..." style queries.
    config = ColBERTConfig(doc_maxlen=512, query_maxlen=32, dim=COLBERT_DIM)
    _colbert_checkpoint = Checkpoint(COLBERT_MODEL_NAME, colbert_config=config)
    logger.info("ColBERT loaded with trained linear projection (768→128)")
    return _colbert_checkpoint


def _encode_tokens(texts: list[str], is_query: bool = False) -> list[np.ndarray]:
    """
    Encode texts into per-token ColBERT embeddings using the trained checkpoint.

    Uses ColBERTv2's own encoding pipeline:
      BERT → 768-d token embeddings → linear projection → 128-d → L2 normalize

    Returns list of (n_tokens, 128) arrays — one per text.
    """
    ckpt = _load_colbert()

    all_embeddings = []

    if is_query:
        # ColBERT query encoding: pads to query_maxlen with [MASK] tokens
        # for query augmentation (allows learning what to attend to)
        Q = ckpt.queryFromText(texts, bsize=len(texts))
        # Q shape: (batch, query_maxlen, dim) — already projected + normalized
        for i in range(Q.shape[0]):
            all_embeddings.append(Q[i].cpu().numpy())
    else:
        # ColBERT document encoding: no padding augmentation
        D_out = ckpt.docFromText(texts, bsize=min(len(texts), 32))
        # docFromText returns a tuple: (tensor, ...) — take the tensor
        D = D_out[0] if isinstance(D_out, tuple) else D_out
        # D shape: (batch, seq_len, dim) — already projected + normalized
        for i in range(D.shape[0]):
            all_embeddings.append(D[i].cpu().numpy())

    return all_embeddings


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """
    ColBERT MaxSim scoring.

    For each query token, find the max cosine similarity to any document token.
    Sum these maxima to get the final relevance score.

    query_tokens: (n_q, dim)
    doc_tokens: (n_d, dim)
    """
    # (n_q, n_d) similarity matrix
    sim_matrix = query_tokens @ doc_tokens.T
    # Per-query-token maximum similarity
    max_sims = sim_matrix.max(axis=1)  # (n_q,)
    return float(max_sims.sum())


class LateInteractionRetriever:
    """
    Late-interaction hybrid retrieval: ColBERT MaxSim + BM25 + RRF fusion.

    Replaces the dense FAISS leg of HybridRetriever with ColBERT per-token
    matching. BM25 leg is identical. Cross-encoder reranker is unnecessary
    since ColBERT already provides reranker-quality scoring.
    """

    def __init__(
        self,
        chunks: list[str],
        embed_fn: Callable | None = None,  # kept for interface compat, not used
        chunk_metadata: list[dict] | None = None,
    ):
        self.chunks = chunks
        self.metadata = chunk_metadata or [{} for _ in chunks]

        # Pre-compute per-token ColBERT embeddings for all chunks
        if chunks:
            self._doc_token_embeds = _encode_tokens(chunks, is_query=False)
        else:
            self._doc_token_embeds = []

        # BM25 sparse index (same as HybridRetriever)
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if chunks else None

        logger.info(
            f"LateInteractionRetriever initialized: {len(chunks)} chunks, "
            f"ColBERT {COLBERT_DIM}-d per-token"
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = False,  # ignored — ColBERT IS the reranker
        rrf_k: int | None = None,
    ) -> list[dict]:
        """
        Late-interaction hybrid search: ColBERT MaxSim + BM25 + RRF.

        Returns list of result dicts (same interface as HybridRetriever):
            {chunk, idx, rrf_score, dense_score, bm25_score, metadata}

        dense_score here is the ColBERT MaxSim score (not single-vector cosine).
        """
        if not self.chunks:
            return []

        k = top_k or SEARCH_TOP_K
        rk = rrf_k or RRF_K

        # ── ColBERT MaxSim scoring ─────────────────────────────────────────
        query_tokens = _encode_tokens([query], is_query=True)[0]  # (n_q, dim)

        colbert_scores = []
        for doc_tokens in self._doc_token_embeds:
            score = maxsim_score(query_tokens, doc_tokens)
            colbert_scores.append(score)

        colbert_ranked = np.argsort(colbert_scores)[::-1][:k]

        # ── BM25 search ───────────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked = np.argsort(bm25_scores)[::-1][:k]

        # ── RRF fusion ─────────────────────────────────────────────────────
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(colbert_ranked):
            idx_int = int(idx)
            rrf[idx_int] = rrf.get(idx_int, 0) + 1.0 / (rk + rank)
        for rank, idx in enumerate(bm25_ranked):
            idx_int = int(idx)
            rrf[idx_int] = rrf.get(idx_int, 0) + 1.0 / (rk + rank)

        # ── Build results ──────────────────────────────────────────────────
        results = []
        for idx, rrf_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]:
            results.append({
                "chunk": self.chunks[idx],
                "idx": idx,
                "rrf_score": rrf_score,
                "dense_score": colbert_scores[idx],  # MaxSim score
                "bm25_score": float(bm25_scores[idx]),
                "metadata": self.metadata[idx],
            })

        return results
