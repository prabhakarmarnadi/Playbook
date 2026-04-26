"""
Clause-level embedder — FIELD-110 REQUIRED deliverable.

Legacy pain point: V1 embeds "field summaries" — averaged representations that
lose clause-specific signals. A liability clause and a payment clause within the
same agreement get blurred when summarized together.

V2 approach: Each chunk/clause gets its own embedding. No averaging.
Optionally embeds with surrounding context for short clauses.
This is the architectural foundation for V2's superior clustering quality.
"""
import logging

import numpy as np

from core.embedder import Embedder

logger = logging.getLogger(__name__)


class ClauseEmbedder:
    """Generate clause-level embeddings that preserve semantic granularity.

    Key insight: Field summaries average away critical clause-specific signals.
    A liability clause and a payment clause get blurred when summarized together.
    Clause-level embeddings preserve these distinctions.
    """

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder.get_instance()

    def embed_clauses(self, chunks: list[dict]) -> np.ndarray:
        """
        Embed each clause/chunk independently.

        Args:
            chunks: List of dicts with at least a "text" key.

        Returns:
            Embeddings matrix (n_chunks × embedding_dim)
        """
        texts = [c["text"] for c in chunks]
        if not texts:
            return np.array([])
        logger.info(f"Embedding {len(texts)} clauses (independent mode)")
        return self.embedder.embed(texts, show_progress=True)

    def embed_with_context(self, chunks: list[dict], context_window: int = 1) -> np.ndarray:
        """
        Embed each clause with surrounding context for better semantic anchoring.

        This helps with short clauses that lack sufficient context on their own.
        Prepends/appends neighboring chunk text before embedding.

        Legacy: No equivalent — V1 never embeds at clause level.
        """
        if not chunks:
            return np.array([])

        contextualized = []
        for i, chunk in enumerate(chunks):
            parts = []
            # Context before
            for j in range(max(0, i - context_window), i):
                parts.append(chunks[j]["text"])
            # Current chunk
            parts.append(chunk["text"])
            # Context after
            for j in range(i + 1, min(len(chunks), i + 1 + context_window)):
                parts.append(chunks[j]["text"])
            contextualized.append(" ".join(parts))

        logger.info(f"Embedding {len(contextualized)} clauses (context window={context_window})")
        return self.embedder.embed(contextualized, show_progress=True)
