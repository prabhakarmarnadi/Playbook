"""
Embedding wrapper — replaces legacy AIFlow gRPC embedding service.

Legacy pain point: AIFlowClient.compute_embeddings() requires a running gRPC server,
network connectivity, and authentication. Embeddings are stored remotely in AIDB.

V2 approach: Local sentence-transformers model (all-MiniLM-L6-v2).
Zero network dependency. Same model family as production (MiniLM).
Embeddings stored in DuckDB as BLOB.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class Embedder:
    """Local embedding model wrapper. Singleton-friendly via module-level caching."""

    _instance: "Embedder | None" = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "Embedder":
        if cls._instance is None or (model_name and cls._instance.model_name != model_name):
            cls._instance = cls(model_name)
        return cls._instance

    def warmup(self):
        """Run a dummy encode to trigger any lazy initialization."""
        _ = self.embed("warmup", show_progress=False)

    def embed(self, texts: str | list[str], normalize: bool = True,
              show_progress: bool = False) -> np.ndarray:
        """
        Embed one or more texts. Returns (n, dim) array.

        Legacy equivalent: AIFlowClient.compute_embeddings() — 1 gRPC call.
        V2: Local inference, no network hop.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            batch_size=64,
        )

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed a single text, return 1D vector."""
        return self.embed(text, normalize=normalize)[0]
