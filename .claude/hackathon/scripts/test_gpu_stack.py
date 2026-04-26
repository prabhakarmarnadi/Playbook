"""
GPU stack diagnostic test.

Verifies that all GPU-accelerated components are available and functional:
  1. CUDA availability
  2. cuML UMAP/HDBSCAN (RAPIDS)
  3. Nomic embedder on GPU
  4. LanceDB vector storage
  5. Ray cluster health
  6. Scoring engine
  7. End-to-end mini pipeline

Run: poetry run python scripts/test_gpu_stack.py
"""
import sys
import time
import logging
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
results = []


def report(name: str, passed: bool, detail: str = "", warn: bool = False):
    tag = PASS if passed else (WARN if warn else FAIL)
    results.append((name, passed, warn))
    logger.info(f"[{tag}] {name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CUDA
# ═══════════════════════════════════════════════════════════════════════════════
def test_cuda():
    try:
        import torch
        avail = torch.cuda.is_available()
        if avail:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            report("CUDA", True, f"{name}, {mem:.1f} GB")
        else:
            report("CUDA", False, "torch.cuda.is_available() = False", warn=True)
    except ImportError:
        report("CUDA", False, "PyTorch not installed", warn=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. cuML UMAP/HDBSCAN
# ═══════════════════════════════════════════════════════════════════════════════
def test_cuml():
    from core.cuml_wrappers import cuml_available, make_umap, make_hdbscan

    if cuml_available():
        # Smoke test with random data
        rng = np.random.default_rng(42)
        X = rng.random((100, 50)).astype(np.float32)
        umap = make_umap(n_neighbors=10, n_components=5)
        reduced = umap.fit_transform(X)
        assert reduced.shape == (100, 5), f"UMAP output shape wrong: {reduced.shape}"

        hdb = make_hdbscan(min_cluster_size=5)
        labels = hdb.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        report("cuML UMAP/HDBSCAN", True, f"GPU — {n_clusters} clusters from 100 points")
    else:
        # Test CPU fallback
        rng = np.random.default_rng(42)
        X = rng.random((50, 20)).astype(np.float32)
        umap = make_umap(n_neighbors=10, n_components=5)
        reduced = umap.fit_transform(X)
        hdb = make_hdbscan(min_cluster_size=5)
        labels = hdb.fit_predict(reduced)
        report("cuML UMAP/HDBSCAN", True, "CPU fallback works", warn=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Nomic Embedder
# ═══════════════════════════════════════════════════════════════════════════════
def test_nomic_embedder():
    try:
        from core.nomic_embedder import NomicEmbedder
        t0 = time.time()
        embedder = NomicEmbedder(device="cuda")
        texts = [
            "This agreement shall be governed by the laws of California.",
            "The term of this NDA shall be for a period of three years.",
            "Payment shall be made within 30 days of invoice.",
        ]
        embs = embedder.embed(texts, prefix="clustering")
        elapsed = time.time() - t0
        assert embs.shape[0] == 3
        assert embs.shape[1] >= 256  # At least Matryoshka-min
        report("Nomic Embedder", True, f"dim={embs.shape[1]}, {elapsed:.2f}s on GPU")
    except Exception as e:
        # Try CPU fallback
        try:
            from core.nomic_embedder import NomicEmbedder
            embedder = NomicEmbedder(device="cpu")
            embs = embedder.embed(["test"], prefix="clustering")
            report("Nomic Embedder", True, f"CPU fallback, dim={embs.shape[1]}", warn=True)
        except Exception as e2:
            report("Nomic Embedder", False, str(e2))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LanceDB
# ═══════════════════════════════════════════════════════════════════════════════
def test_lancedb():
    try:
        from core.lancedb_store import LanceVectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceVectorStore(Path(tmpdir) / "test.lance")
            rng = np.random.default_rng(42)
            vecs = rng.random((10, 64)).astype(np.float32)
            store.upsert_chunks(
                chunk_ids=[f"c{i}" for i in range(10)],
                vectors=vecs,
                texts=[f"Chunk {i}" for i in range(10)],
                agreement_ids=[f"a{i%3}" for i in range(10)],
            )
            assert store.count("chunks") == 10
            results = store.search_chunks(vecs[0], top_k=3)
            assert len(results) == 3
            report("LanceDB", True, f"10 vectors stored, 3 retrieved")
    except Exception as e:
        report("LanceDB", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Ray
# ═══════════════════════════════════════════════════════════════════════════════
def test_ray():
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        info = ray.cluster_resources()
        cpus = info.get("CPU", 0)
        gpus = info.get("GPU", 0)
        report("Ray", True, f"cluster: {cpus:.0f} CPUs, {gpus:.0f} GPUs")
        ray.shutdown()
    except ImportError:
        report("Ray", False, "ray not installed", warn=True)
    except Exception as e:
        report("Ray", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Scoring Engine
# ═══════════════════════════════════════════════════════════════════════════════
def test_scoring():
    from core.scoring import calculate_scores, apply_penalties

    rng = np.random.default_rng(42)
    embeddings = rng.random((50, 20)).astype(np.float32)
    topics = np.array([0]*15 + [1]*15 + [2]*10 + [-1]*10)

    scores = calculate_scores(topics, embeddings)
    assert scores.n_clusters == 3
    assert scores.n_outliers == 10
    assert 0.0 <= scores.z_score <= 2.0

    scores = apply_penalties(scores, min_clusters=2)
    assert scores.adjusted_score > 0
    report("Scoring Engine", True, f"z={scores.z_score:.3f}, adj={scores.adjusted_score:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Mini E2E pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def test_mini_pipeline():
    """Quick smoke test: embed → cluster → score (no LLM calls)."""
    from core.cuml_wrappers import make_umap, make_hdbscan
    from core.scoring import calculate_scores

    try:
        from core.nomic_embedder import NomicEmbedder
        embedder = NomicEmbedder(device="cuda")
    except Exception:
        from core.nomic_embedder import NomicEmbedder
        embedder = NomicEmbedder(device="cpu")

    docs = [
        "Confidentiality. Each party agrees to keep information confidential.",
        "Non-disclosure. Neither party shall disclose proprietary information.",
        "Trade secrets shall be protected for a period of five years.",
        "Payment terms. Invoices are due within 30 days of receipt.",
        "Compensation shall be paid monthly at the agreed rate.",
        "Service fees are non-refundable after 60 days.",
        "Termination. Either party may terminate with 30 days notice.",
        "This agreement may be terminated for cause upon written notice.",
        "Early termination requires payment of the remaining balance.",
    ]

    t0 = time.time()
    embs = embedder.embed_for_clustering(docs, show_progress=False)
    n = len(docs)
    umap = make_umap(n_neighbors=min(5, n-1), n_components=min(2, n-2))
    reduced = umap.fit_transform(embs)
    hdb = make_hdbscan(min_cluster_size=2, min_samples=1)
    labels = hdb.fit_predict(reduced)

    scores = calculate_scores(np.array(labels), embs, umap_embeddings=reduced)
    elapsed = time.time() - t0

    n_clusters = scores.n_clusters
    report(
        "Mini E2E Pipeline", n_clusters >= 1,
        f"{n_clusters} clusters from {len(docs)} docs in {elapsed:.2f}s"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("GPU Stack Diagnostic")
    logger.info("=" * 60)

    test_cuda()
    test_cuml()
    test_nomic_embedder()
    test_lancedb()
    test_ray()
    test_scoring()
    test_mini_pipeline()

    logger.info("=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    warned = sum(1 for _, p, w in results if p and w)
    failed = sum(1 for _, p, _ in results if not p)
    total = len(results)
    logger.info(f"Results: {passed}/{total} passed ({warned} with warnings), {failed} failed")

    if failed > 0:
        logger.info("Some components are unavailable. GPU pipeline will use CPU fallbacks.")
        sys.exit(1)
    else:
        logger.info("All GPU stack components operational.")
        sys.exit(0)
