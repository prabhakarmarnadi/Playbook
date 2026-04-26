"""
cuML GPU-accelerated UMAP + HDBSCAN wrappers.

Replaces sklearn/hdbscan CPU implementations with RAPIDS cuML equivalents.
Falls back to CPU gracefully when CUDA is unavailable.

Performance: cuML UMAP is ~50-100x faster on 10k+ point datasets.
cuML HDBSCAN is ~10-20x faster than the CPU hdbscan library.
"""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Detect GPU availability ────────────────────────────────────────────────────
_CUML_AVAILABLE = False
try:
    import cupy as cp
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    _CUML_AVAILABLE = True
    logger.info("cuML RAPIDS detected — using GPU-accelerated UMAP/HDBSCAN")
except ImportError:
    logger.warning("cuML not available — falling back to CPU UMAP/HDBSCAN")


def cuml_available() -> bool:
    """Check whether cuML GPU acceleration is available."""
    return _CUML_AVAILABLE


class GPUUmap:
    """
    GPU-accelerated UMAP via cuML, with CPU fallback.

    Mirrors the scikit-learn-compatible API that BERTopic expects:
    .fit(X), .transform(X), .fit_transform(X).
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.params = dict(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )
        if _CUML_AVAILABLE:
            # cuML UMAP supports cosine via 'euclidean' on L2-normalized data
            # or directly via metric='cosine' (cuML >= 23.04)
            self._model = cuUMAP(**self.params)
        else:
            from umap import UMAP as cpuUMAP
            self._model = cpuUMAP(**self.params)

    def fit(self, X: np.ndarray, y=None):
        self._model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = self._model.transform(X)
        if _CUML_AVAILABLE:
            return cp.asnumpy(result) if hasattr(result, "get") else np.asarray(result)
        return np.asarray(result)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        result = self._model.fit_transform(X, y)
        if _CUML_AVAILABLE:
            return cp.asnumpy(result) if hasattr(result, "get") else np.asarray(result)
        return np.asarray(result)


class GPUHdbscan:
    """
    GPU-accelerated HDBSCAN via cuML, with CPU fallback.

    Mirrors the hdbscan library API that BERTopic expects:
    .fit(X), .labels_, .probabilities_.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 2,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        prediction_data: bool = True,
        **kwargs: Any,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        if _CUML_AVAILABLE:
            # cuML HDBSCAN supports most of the same params
            self._model = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                prediction_data=prediction_data,
                **kwargs,
            )
        else:
            from hdbscan import HDBSCAN as cpuHDBSCAN
            self._model = cpuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                prediction_data=prediction_data,
                **kwargs,
            )

    def fit(self, X: np.ndarray, y=None):
        self._model.fit(X, y)
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        self._model.fit(X, y)
        labels = self._model.labels_
        if _CUML_AVAILABLE and hasattr(labels, "get"):
            return cp.asnumpy(labels)
        return np.asarray(labels)

    @property
    def labels_(self) -> np.ndarray:
        labels = self._model.labels_
        if _CUML_AVAILABLE and hasattr(labels, "get"):
            return cp.asnumpy(labels)
        return np.asarray(labels)

    @property
    def probabilities_(self) -> np.ndarray:
        probs = self._model.probabilities_
        if _CUML_AVAILABLE and hasattr(probs, "get"):
            return cp.asnumpy(probs)
        return np.asarray(probs)


def make_umap(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    target: np.ndarray | None = None,
    target_weight: float = 0.0,
    **kwargs: Any,
):
    """Factory: create a GPU or CPU UMAP model.

    When cuML is available, returns the native cuML UMAP directly
    so BERTopic's isinstance checks pass correctly.

    Semi-supervised support: when target_weight > 0, UMAP uses labels
    to guide dimensionality reduction. Labels of -1 are unlabeled.
    """
    ss_kwargs = {}
    if target is not None and target_weight > 0.0:
        ss_kwargs["target_metric"] = "categorical"
        ss_kwargs["target_weight"] = target_weight

    if _CUML_AVAILABLE:
        return cuUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **ss_kwargs,
            **kwargs,
        )
    from umap import UMAP as cpuUMAP
    return cpuUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **ss_kwargs,
        **kwargs,
    )


def make_hdbscan(
    min_cluster_size: int = 5,
    min_samples: int = 2,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    prediction_data: bool = True,
    **kwargs: Any,
):
    """Factory: create a GPU or CPU HDBSCAN model.

    When cuML is available, returns the native cuML HDBSCAN directly
    so BERTopic's isinstance checks (e.g. all_points_membership_vectors) pass.
    """
    if _CUML_AVAILABLE:
        return cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=prediction_data,
            **kwargs,
        )
    from hdbscan import HDBSCAN as cpuHDBSCAN
    return cpuHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data,
        **kwargs,
    )
