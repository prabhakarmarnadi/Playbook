"""
KeyBERT clause-type scoring for clustering context.

Loads pre-computed clause fingerprints and provides:
1. classify(text) -> ranked clause-type similarities
2. augment_text(text) -> text with clause-type prefix for embedding
3. get_label(text) -> integer label for semi-supervised UMAP (-1 if uncertain)

Degrades gracefully if fingerprints file is missing.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    KEYBERT_AUGMENT_TOP_K,
    KEYBERT_CLASSIFICATION_THRESHOLD,
    KEYBERT_FINGERPRINTS_PATH,
)

logger = logging.getLogger(__name__)

_FINGERPRINTS_LOADED = False


def keybert_available() -> bool:
    return _FINGERPRINTS_LOADED


class KeyBERTScorer:
    _instance: "KeyBERTScorer | None" = None

    def __init__(self, fingerprints_path: Path | str | None = None):
        self.fingerprints_path = Path(fingerprints_path or KEYBERT_FINGERPRINTS_PATH)
        self.clause_types: list[str] = []
        self.fingerprint_embeddings: np.ndarray | None = None
        self.keyphrases: dict[str, list[tuple[str, float]]] = {}
        self._model: SentenceTransformer | None = None
        self._loaded = False
        self._load_fingerprints()

    @classmethod
    def get_instance(cls, fingerprints_path: Path | str | None = None) -> "KeyBERTScorer":
        if cls._instance is None:
            cls._instance = cls(fingerprints_path)
        return cls._instance

    def _load_fingerprints(self) -> None:
        global _FINGERPRINTS_LOADED
        if not self.fingerprints_path.exists():
            logger.warning(f"KeyBERT fingerprints not found at {self.fingerprints_path}")
            return
        try:
            with open(self.fingerprints_path) as f:
                data = json.load(f)
            fingerprints = data.get("fingerprints", {})
            model_name = data.get("model_name", EMBEDDING_MODEL)
            self.clause_types = sorted(fingerprints.keys())
            embeddings = []
            for ct in self.clause_types:
                fp = fingerprints[ct]
                if "embedding" in fp:
                    embeddings.append(np.array(fp["embedding"], dtype=np.float32))
                self.keyphrases[ct] = fp.get("keyphrases", [])
            if embeddings:
                self.fingerprint_embeddings = np.stack(embeddings)
                norms = np.linalg.norm(self.fingerprint_embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                self.fingerprint_embeddings = self.fingerprint_embeddings / norms
            self._model = SentenceTransformer(model_name)
            self._loaded = True
            _FINGERPRINTS_LOADED = True
            logger.info(f"Loaded {len(self.clause_types)} clause fingerprints")
        except Exception as e:
            logger.warning(f"Failed to load KeyBERT fingerprints: {e}")

    def classify(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._loaded or self.fingerprint_embeddings is None:
            return []
        text_embedding = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = self.fingerprint_embeddings @ text_embedding
        indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if sim >= KEYBERT_CLASSIFICATION_THRESHOLD:
                results.append((self.clause_types[idx], sim))
        return results

    def score_all(self, text: str) -> dict[str, float]:
        if not self._loaded or self.fingerprint_embeddings is None:
            return {}
        text_embedding = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = self.fingerprint_embeddings @ text_embedding
        return {ct: float(similarities[i]) for i, ct in enumerate(self.clause_types)}

    def augment_text(self, text: str) -> str:
        results = self.classify(text, top_k=1)
        if not results:
            return text
        clause_type, _ = results[0]
        kps = self.keyphrases.get(clause_type, [])
        top_kps = sorted([kp for kp, _ in kps[:KEYBERT_AUGMENT_TOP_K]])
        kp_str = ", ".join(top_kps)
        return f"[{clause_type} | {kp_str}] {text}"

    def get_label(self, text: str) -> int:
        results = self.classify(text, top_k=1)
        if not results:
            return -1
        clause_type, _ = results[0]
        return self.clause_types.index(clause_type)

    def batch_classify(self, texts: list[str], top_k: int = 5) -> list[list[tuple[str, float]]]:
        if not self._loaded or self.fingerprint_embeddings is None:
            return [[] for _ in texts]
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        all_similarities = embeddings @ self.fingerprint_embeddings.T
        results = []
        for row in all_similarities:
            indices = np.argsort(row)[::-1][:top_k]
            matches = []
            for idx in indices:
                sim = float(row[idx])
                if sim >= KEYBERT_CLASSIFICATION_THRESHOLD:
                    matches.append((self.clause_types[idx], sim))
            results.append(matches)
        return results

    def batch_augment(self, texts: list[str]) -> list[str]:
        if not self._loaded:
            return list(texts)
        classified = self.batch_classify(texts, top_k=1)
        augmented = []
        for text, results in zip(texts, classified):
            if results:
                clause_type, _ = results[0]
                kps = self.keyphrases.get(clause_type, [])
                top_kps = sorted([kp for kp, _ in kps[:KEYBERT_AUGMENT_TOP_K]])
                kp_str = ", ".join(top_kps)
                augmented.append(f"[{clause_type} | {kp_str}] {text}")
            else:
                augmented.append(text)
        return augmented

    def batch_labels(self, texts: list[str]) -> np.ndarray:
        if not self._loaded:
            return np.full(len(texts), -1, dtype=np.int32)
        classified = self.batch_classify(texts, top_k=1)
        labels = np.full(len(texts), -1, dtype=np.int32)
        for i, results in enumerate(classified):
            if results:
                clause_type, _ = results[0]
                labels[i] = self.clause_types.index(clause_type)
        return labels
