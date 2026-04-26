"""
Nomic GPU embedder — replaces MiniLM CPU embedder.

Uses nomic-ai/nomic-embed-text-v1.5 (768-dim, Matryoshka-capable).
Runs on GPU via device='cuda' for ~10x throughput vs CPU MiniLM.

Key advantages over MiniLM-L6-v2:
  - 768-dim (vs 384) — richer representation for legal text
  - Matryoshka support — can truncate to 256/512-dim for speed
  - Task-prefixed prompts ("search_document:", "search_query:")
  - Trained on 235M pairs including legal/financial text
"""
import logging
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model — can be overridden via NOMIC_MODEL env var
DEFAULT_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class NomicEmbedder:
    """
    GPU-accelerated Nomic embedder with task prefixes and optional dim truncation.

    Nomic separates document vs query embeddings via prefixes:
      - "search_document: <text>" for indexing/clustering
      - "search_query: <text>" for retrieval queries
      - "clustering: <text>" for clustering tasks
      - "classification: <text>" for classification

    This matters for asymmetric tasks (query vs doc), but for clustering
    all docs get the same prefix ("search_document:" or "clustering:").
    """

    _instance: "NomicEmbedder | None" = None

    def __init__(
        self,
        model_name: str = DEFAULT_NOMIC_MODEL,
        device: str = "cuda",
        matryoshka_dim: int | None = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.matryoshka_dim = matryoshka_dim

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        # Full model dim (768 for nomic-embed-text-v1.5)
        self._full_dim = self.model.get_sentence_embedding_dimension()
        self.dim = matryoshka_dim or self._full_dim

        logger.info(
            f"NomicEmbedder initialized: {model_name} on {device}, "
            f"dim={self.dim} (full={self._full_dim})"
        )

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_NOMIC_MODEL,
        device: str = "cuda",
        matryoshka_dim: int | None = None,
    ) -> "NomicEmbedder":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name, device=device, matryoshka_dim=matryoshka_dim)
        return cls._instance

    def warmup(self):
        """Run a dummy encode to trigger any lazy initialization (CUDA kernels, etc.)."""
        logger.info("NomicEmbedder warmup: running dummy encode...")
        _ = self.embed("warmup", show_progress=False)
        logger.info("NomicEmbedder warmup complete.")

    def embed(
        self,
        texts: str | list[str],
        prefix: Literal[
            "search_document", "search_query", "clustering", "classification"
        ] = "search_document",
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Embed texts with a Nomic task prefix.

        Args:
            texts: Single text or list of texts.
            prefix: Nomic task prefix (determines embedding space).
            normalize: L2-normalize the embeddings.
            show_progress: Show progress bar.
            batch_size: Inference batch size (higher = faster on GPU, more VRAM).

        Returns:
            (n, dim) float32 array. If matryoshka_dim is set, truncated + re-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Prepend Nomic task prefix
        prefixed = [f"{prefix}: {t}" for t in texts]

        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )

        # Matryoshka truncation — slice to desired dim and re-normalize
        if self.matryoshka_dim and self.matryoshka_dim < self._full_dim:
            embeddings = embeddings[:, : self.matryoshka_dim]
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def embed_single(
        self,
        text: str,
        prefix: Literal[
            "search_document", "search_query", "clustering", "classification"
        ] = "search_document",
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a single text, return 1D vector."""
        return self.embed(text, prefix=prefix, normalize=normalize)[0]

    def embed_for_clustering(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Shorthand: embed with 'clustering' prefix and larger batches."""
        return self.embed(
            texts,
            prefix="clustering",
            normalize=normalize,
            show_progress=show_progress,
            batch_size=batch_size,
        )

    def embed_for_search(
        self,
        texts: list[str],
        is_query: bool = False,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Shorthand: embed docs or queries for retrieval."""
        prefix = "search_query" if is_query else "search_document"
        return self.embed(
            texts,
            prefix=prefix,
            normalize=normalize,
            show_progress=show_progress,
        )
