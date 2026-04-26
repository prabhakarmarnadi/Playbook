"""
Ray-distributed Bayesian optimizer for clustering hyperparameters.

Uses @ray.remote tasks for parallel trial execution — lightweight, no
framework overhead. Optuna TPE drives parameter suggestions; Ray tasks
execute batches of evaluations concurrently.

Architecture:
  1. Optuna TPE sampler proposes a batch of `n_parallel` param configs
  2. Each config is dispatched as a @ray.remote task (zero-copy shared data)
  3. Completed results are reported back to Optuna to update the TPE model
  4. Repeat until `n_trials` trials are done

Why Ray Tasks over Ray Tune:
  - No trial-runner actor overhead (~2s startup per trial eliminated)
  - No log directory / TensorBoard / checkpoint machinery
  - Direct control over batch scheduling and early-abort
  - Simpler code: just futures + ray.get()

Legacy: Optuna TPE sampler, 30 trials, sequential, single-process.
V2: Optuna TPE + Ray @ray.remote tasks, 50 trials, parallel batches.

Parameter space ported from legacy bayesian_optimizer.py DEFAULT_PARAM_RANGES.
"""
import logging
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from config import LEGAL_STOPWORDS
from core.cuml_wrappers import make_umap, make_hdbscan
from core.scoring import (
    calculate_scores,
    apply_penalties,
    ClusterScores,
    DEFAULT_MIN_CLUSTERS,
    DEFAULT_MAX_DOMINANCE,
    DEFAULT_MAX_OUTLIER_RATIO,
    DEFAULT_KEYWORD_DIVERSITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Ray cluster setup
# ═══════════════════════════════════════════════════════════════════════════════

def init_ray(
    address: str | None = None,
    num_gpus: int | None = None,
) -> dict:
    """
    Initialize Ray with GPU detection and resource logging.

    Call this once before optimize_with_ray(). Safe to call multiple times
    (idempotent via ignore_reinit_error).

    Args:
        address: Ray cluster address. None = start local single-node cluster.
                 Use "auto" to connect to an existing cluster.
        num_gpus: Override GPU count. None = auto-detect via torch.cuda.

    Returns:
        Dict with cluster resource summary:
        {"cpus": int, "gpus": int, "gpu_names": list[str], "address": str}
    """
    import ray

    # Auto-detect GPUs via torch (more reliable than Ray's default detection
    # which may miss GPUs if CUDA_VISIBLE_DEVICES isn't set)
    detected_gpus = 0
    gpu_names: list[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            detected_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(detected_gpus)]
    except ImportError:
        pass

    actual_gpus = num_gpus if num_gpus is not None else detected_gpus

    if not ray.is_initialized():
        ray.init(
            address=address,
            num_gpus=actual_gpus,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )

    resources = ray.cluster_resources()
    info = {
        "cpus": int(resources.get("CPU", 0)),
        "gpus": int(resources.get("GPU", 0)),
        "gpu_names": gpu_names,
        "address": ray.get_runtime_context().gcs_address if hasattr(ray.get_runtime_context(), "gcs_address") else "local",
    }

    logger.info(
        f"Ray initialized: {info['cpus']} CPUs, {info['gpus']} GPUs "
        f"({', '.join(gpu_names) if gpu_names else 'none'})"
    )
    return info


def shutdown_ray():
    """Shutdown Ray cluster gracefully."""
    import ray
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")


# Combined stopword set (English + legal domain)
COMBINED_STOPWORDS = set(ENGLISH_STOP_WORDS) | set(LEGAL_STOPWORDS)


class StopwordFilteringAnalyzer:
    """Picklable custom analyzer that filters n-grams where ALL tokens are stopwords.

    Ported from legacy bertopic_clusterer.py.
    """

    def __init__(self, ngram_range: tuple = (1, 4), stopwords: set | None = None):
        self.ngram_range = ngram_range
        self.stopwords = stopwords or COMBINED_STOPWORDS
        self._analyzer = None

    def _build_analyzer(self):
        if self._analyzer is None:
            base_vectorizer = CountVectorizer(ngram_range=self.ngram_range)
            self._analyzer = base_vectorizer.build_analyzer()
        return self._analyzer

    def __call__(self, doc: str) -> list:
        analyzer = self._build_analyzer()
        tokens = analyzer(doc)
        return [t for t in tokens if any(w not in self.stopwords for w in t.split())]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_analyzer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ── Default parameter ranges (from legacy bayesian_optimizer.py) ───────────────
DEFAULT_PARAM_RANGES = {
    "umap_n_neighbors": (10, 30),
    "umap_n_components": (12, 25),
    "umap_min_dist": (0.0, 0.05),
    "hdbscan_min_cluster_size": (5, 15),
    "hdbscan_min_samples": (2, 5),
    "hdbscan_cluster_selection_epsilon": (0.05, 0.15),
}


@dataclass
class OptimizationResult:
    """Result of a completed optimization run."""
    best_params: dict
    best_score: float
    best_scores: ClusterScores
    all_trials: list[dict]
    n_trials: int
    elapsed_seconds: float


def _build_and_evaluate(
    params: dict[str, Any],
    docs: list[str],
    embeddings: np.ndarray,
    stopwords: list[str] | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_dominance: float = DEFAULT_MAX_DOMINANCE,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
) -> ClusterScores:
    """
    Build a BERTopic model with given UMAP/HDBSCAN params, fit it, and score.

    This is the objective function executed by each Ray task.
    Uses cuML wrappers (GPU if available, CPU fallback).
    """
    n = len(docs)

    # Clamp params to data size
    n_neighbors = min(int(params["umap_n_neighbors"]), n - 1)
    n_components = min(int(params["umap_n_components"]), n - 2, n_neighbors - 1)
    n_components = max(2, n_components)

    umap_model = make_umap(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=float(params["umap_min_dist"]),
        metric="cosine",
    )
    hdbscan_model = make_hdbscan(
        min_cluster_size=max(2, int(params["hdbscan_min_cluster_size"])),
        min_samples=max(1, int(params["hdbscan_min_samples"])),
        metric="euclidean",
        cluster_selection_method="eom",
    )

    sw = set(stopwords) if stopwords else COMBINED_STOPWORDS
    analyzer = StopwordFilteringAnalyzer(
        ngram_range=(1, 4), stopwords=sw,
    )
    # BERTopic aggregates docs-per-topic, so effective doc count for the
    # vectorizer equals n_topics (usually small). Keep min_df low.
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        min_df=1,
        max_features=100000,
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True, reduce_frequent_words=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _probs = topic_model.fit_transform(docs, embeddings)
    topics_arr = np.array(topics)

    # Get UMAP embeddings for distance-based metrics
    umap_embs = umap_model.fit_transform(embeddings)

    scores = calculate_scores(topics_arr, embeddings, umap_embeddings=umap_embs)
    scores.params = params
    scores = apply_penalties(
        scores,
        min_clusters=min_clusters,
        max_dominance=max_dominance,
        max_outlier_ratio=max_outlier_ratio,
    )

    return scores


def _sample_params(
    study: Any,
    ranges: dict[str, tuple],
) -> tuple[Any, dict[str, Any]]:
    """Ask Optuna to suggest one parameter config. Returns (trial, params)."""
    trial = study.ask()
    params = {}
    for key, (lo, hi) in ranges.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[key] = trial.suggest_int(key, lo, hi)
        else:
            params[key] = trial.suggest_float(key, lo, hi)
    return trial, params


def optimize_with_ray(
    docs: list[str],
    embeddings: np.ndarray,
    n_trials: int = 50,
    n_parallel: int = 4,
    param_ranges: dict[str, tuple] | None = None,
    stopwords: list[str] | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_dominance: float = DEFAULT_MAX_DOMINANCE,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
    ray_address: str | None = None,
) -> OptimizationResult:
    """
    Run Bayesian hyperparameter optimization via Ray @ray.remote tasks.

    Flow:
      1. Optuna TPE proposes up to `n_parallel` configs at once
      2. Each is dispatched as a lightweight Ray task (shared-memory data)
      3. As tasks complete, results feed back into Optuna for the next batch
      4. Continues until `n_trials` total completions

    Args:
        docs: Document texts for clustering.
        embeddings: Pre-computed embeddings (N, dim).
        n_trials: Total number of trials (default 50, legacy was 30).
        n_parallel: Max concurrent Ray tasks.
        param_ranges: Override default parameter ranges.
        stopwords: Stopword list for CountVectorizer.
        min_clusters: Minimum cluster count threshold for penalty.
        max_dominance: Maximum cluster dominance threshold.
        max_outlier_ratio: Maximum outlier ratio threshold.
        ray_address: Ray cluster address (None = local).

    Returns:
        OptimizationResult with best params and all trial data.
    """
    import optuna
    import ray

    ranges = param_ranges or DEFAULT_PARAM_RANGES
    t0 = time.time()

    # ── Initialize Ray with GPU awareness ──────────────────────────────
    cluster_info = init_ray(address=ray_address)
    n_gpus_available = cluster_info["gpus"]

    # Fractional GPU per task: share the GPU across parallel trials.
    # cuML UMAP/HDBSCAN on a single trial uses ~200-500 MB VRAM, so
    # 4 trials can easily share one 16GB T4/A10.
    # If no GPUs, tasks run CPU-only (cuml_wrappers falls back).
    if n_gpus_available > 0:
        gpu_per_task = n_gpus_available / n_parallel
    else:
        gpu_per_task = 0

    logger.info(
        f"Task GPU allocation: {gpu_per_task:.2f} GPU/task "
        f"({n_parallel} parallel, {n_gpus_available} total GPUs)"
    )

    # Put large data in Ray object store (shared memory, zero-copy reads)
    docs_ref = ray.put(docs)
    embs_ref = ray.put(embeddings)
    sw_ref = ray.put(stopwords)

    # Define the remote task with GPU resource request.
    # num_gpus tells Ray's scheduler to reserve fractional GPU for this task,
    # which also sets CUDA_VISIBLE_DEVICES so cuML/PyTorch see the GPU.
    @ray.remote(num_gpus=gpu_per_task)
    def _evaluate_remote(
        params: dict,
        docs_ref_inner,
        embs_ref_inner,
        sw_ref_inner,
        _min_clusters: int,
        _max_dominance: float,
        _max_outlier_ratio: float,
    ) -> dict:
        """Ray task: evaluate one param config and return scores dict."""
        _docs = ray.get(docs_ref_inner)
        _embs = ray.get(embs_ref_inner)
        _sw = ray.get(sw_ref_inner)

        try:
            scores = _build_and_evaluate(
                params=params,
                docs=_docs,
                embeddings=_embs,
                stopwords=_sw,
                min_clusters=_min_clusters,
                max_dominance=_max_dominance,
                max_outlier_ratio=_max_outlier_ratio,
            )
            return {
                "params": params,
                "adjusted_score": scores.adjusted_score,
                "z_score": scores.z_score,
                "n_clusters": scores.n_clusters,
                "outlier_ratio": scores.outlier_ratio,
                "avg_cosine": scores.average_cosine_similarity,
                "silhouette": scores.silhouette_score,
                "dbcv": scores.dbcv_score,
                "error": None,
            }
        except Exception as e:
            return {"params": params, "adjusted_score": -1.0, "error": str(e)}

    # Optuna study with TPE sampler (same algorithm as legacy)
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    all_trials: list[dict] = []
    dispatched = 0  # total trials asked from Optuna

    # ── Batch dispatch loop ────────────────────────────────────────────────
    # Dispatch up to n_parallel tasks, wait for all, report to Optuna, repeat.
    while dispatched < n_trials:
        batch_size = min(n_parallel, n_trials - dispatched)

        # Ask Optuna for a batch of configs
        batch: list[tuple[Any, dict]] = []
        for _ in range(batch_size):
            trial, params = _sample_params(study, ranges)
            batch.append((trial, params))

        # Launch Ray tasks
        futures = []
        for _, params in batch:
            fut = _evaluate_remote.remote(
                params, docs_ref, embs_ref, sw_ref,
                min_clusters, max_dominance, max_outlier_ratio,
            )
            futures.append(fut)

        # Wait for the batch to complete
        results = ray.get(futures)

        # Report results back to Optuna and collect
        for (optuna_trial, _), result in zip(batch, results):
            score = result.get("adjusted_score", -1.0)
            study.tell(optuna_trial, score)
            all_trials.append(result)

        dispatched += batch_size
        logger.info(
            f"  Batch complete: {dispatched}/{n_trials} trials, "
            f"best so far={study.best_value:.4f}"
        )

    elapsed = time.time() - t0

    # Re-evaluate best params for full ClusterScores object
    best_params = study.best_params
    best_scores = _build_and_evaluate(
        params=best_params,
        docs=docs,
        embeddings=embeddings,
        stopwords=stopwords,
        min_clusters=min_clusters,
        max_dominance=max_dominance,
        max_outlier_ratio=max_outlier_ratio,
    )

    logger.info(
        f"Optimization complete: {n_trials} trials in {elapsed:.1f}s. "
        f"Best adjusted_score={study.best_value:.4f}, "
        f"n_clusters={best_scores.n_clusters}, "
        f"outlier_ratio={best_scores.outlier_ratio:.3f}"
    )

    return OptimizationResult(
        best_params=best_params,
        best_score=study.best_value,
        best_scores=best_scores,
        all_trials=all_trials,
        n_trials=n_trials,
        elapsed_seconds=elapsed,
    )


def optimize_sequential(
    docs: list[str],
    embeddings: np.ndarray,
    n_trials: int = 30,
    param_ranges: dict[str, tuple] | None = None,
    stopwords: list[str] | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_dominance: float = DEFAULT_MAX_DOMINANCE,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
    callback: Callable[[int, int, float], None] | None = None,
) -> OptimizationResult:
    """
    Fallback sequential optimizer (no Ray dependency).

    Uses Optuna directly for environments without Ray.
    Same algorithm as legacy BayesianOptimizer.optimize().
    """
    import optuna

    ranges = param_ranges or DEFAULT_PARAM_RANGES
    t0 = time.time()
    all_trials: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for key, (lo, hi) in ranges.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[key] = trial.suggest_int(key, lo, hi)
            else:
                params[key] = trial.suggest_float(key, lo, hi)

        try:
            scores = _build_and_evaluate(
                params=params,
                docs=docs,
                embeddings=embeddings,
                stopwords=stopwords,
                min_clusters=min_clusters,
                max_dominance=max_dominance,
                max_outlier_ratio=max_outlier_ratio,
            )
            all_trials.append({
                "params": params,
                "adjusted_score": scores.adjusted_score,
                "z_score": scores.z_score,
                "n_clusters": scores.n_clusters,
                "outlier_ratio": scores.outlier_ratio,
            })
            if callback:
                callback(trial.number + 1, n_trials, scores.adjusted_score)
            return scores.adjusted_score
        except Exception:
            return -1.0

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5, seed=42, multivariate=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed = time.time() - t0
    best_params = study.best_params
    best_scores = _build_and_evaluate(
        params=best_params,
        docs=docs,
        embeddings=embeddings,
        stopwords=stopwords,
        min_clusters=min_clusters,
        max_dominance=max_dominance,
        max_outlier_ratio=max_outlier_ratio,
    )

    logger.info(
        f"Sequential optimization: {n_trials} trials in {elapsed:.1f}s. "
        f"Best score={study.best_value:.4f}"
    )

    return OptimizationResult(
        best_params=best_params,
        best_score=study.best_value,
        best_scores=best_scores,
        all_trials=all_trials,
        n_trials=n_trials,
        elapsed_seconds=elapsed,
    )
