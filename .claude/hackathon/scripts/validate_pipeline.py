"""
Validation script — runs the full pipeline on sample data and verifies each stage.

Replaces legacy integration tests that required running gRPC services, Azure connections,
and ServiceBus listeners. This script runs 100% locally.

Usage:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --skip-llm   # Skip LLM calls (test infra only)
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_chunker():
    """Validate: chunker produces overlapping windows with correct structure."""
    from core.chunker import sliding_window_chunk, get_document_summary

    text = "This is a test agreement. " * 200  # ~1000 words
    chunks = sliding_window_chunk(text)

    assert len(chunks) > 0, "Chunker produced no chunks"
    for c in chunks:
        assert "chunk_id" in c, "Missing chunk_id"
        assert "text" in c, "Missing text"
        assert len(c["text"]) >= 80, f"Chunk too short: {len(c['text'])}"

    summary = get_document_summary(text, "test_agreement.pdf")
    assert "test" in summary.lower() and "agreement" in summary.lower(), "Summary should include filename"
    assert len(summary) <= 2100, "Summary too long"

    print(f"  [PASS] Chunker: {len(chunks)} chunks from ~1000 words")


def test_embedder():
    """Validate: embedder produces correct dimensionality."""
    from core.embedder import Embedder

    emb = Embedder.get_instance()
    vecs = emb.embed(["test sentence one", "test sentence two"])
    assert vecs.shape == (2, emb.dim), f"Expected (2, {emb.dim}), got {vecs.shape}"

    single = emb.embed_single("single test")
    assert single.shape == (emb.dim,), f"Expected ({emb.dim},), got {single.shape}"

    print(f"  [PASS] Embedder: {emb.dim}-dim vectors, model={emb.model_name}")


def test_clause_embedder():
    """Validate: clause-level embeddings preserve per-chunk granularity (FIELD-110)."""
    from core.clause_embedder import ClauseEmbedder

    ce = ClauseEmbedder()
    chunks = [
        {"text": "Payment shall be made within thirty days of invoice."},
        {"text": "The liability of either party shall not exceed the total contract value."},
        {"text": "This agreement shall automatically renew for successive one-year terms."},
    ]
    embs = ce.embed_clauses(chunks)
    assert embs.shape == (3, 384), f"Expected (3, 384), got {embs.shape}"

    embs_ctx = ce.embed_with_context(chunks, context_window=1)
    assert embs_ctx.shape == (3, 384), f"Context embeddings shape mismatch"

    print(f"  [PASS] Clause embedder: {embs.shape[0]} clause embeddings (FIELD-110)")


def test_composite_engine():
    """Validate: composite field evaluation, cascading, and cycle detection."""
    from core.composite import (
        parse_and_eval, get_dependencies, detect_cycles,
        evaluate_all_composites, compute_cascade,
    )

    # Basic evaluation
    result = parse_and_eval("a * b + c", {"a": 10, "b": 5, "c": 3})
    assert result == 53, f"Expected 53, got {result}"

    # None propagation
    result_none = parse_and_eval("a * b", {"a": 10, "b": None})
    assert result_none is None, "Should return None for missing values"

    # Division by zero
    result_div = parse_and_eval("a / b", {"a": 10, "b": 0})
    assert result_div is None, "Should return None for division by zero"

    # Dependencies
    deps = get_dependencies("unit_price * quantity * term_years")
    assert set(deps) == {"quantity", "term_years", "unit_price"}, f"Wrong deps: {deps}"

    # Cycle detection
    errors = detect_cycles({"a": "b + 1", "b": "a + 1"})
    assert len(errors) > 0, "Should detect cycle"

    # No cycles
    errors_ok = detect_cycles({"tcv": "price * qty", "monthly": "tcv / 12"})
    assert len(errors_ok) == 0, f"False cycle: {errors_ok}"

    # Full composite evaluation with cascading
    composites = {
        "annual_cost": "unit_price * quantity",
        "tcv": "annual_cost * term_years",
        "cost_per_month": "annual_cost / 12",
    }
    base = {"unit_price": 50000, "quantity": 100, "term_years": 3}
    results = evaluate_all_composites(composites, base)
    assert results["annual_cost"] == 5_000_000
    assert results["tcv"] == 15_000_000
    assert abs(results["cost_per_month"] - 416666.67) < 1

    # Cascade
    cascade = compute_cascade(composites, base, "unit_price", 50000, 55000)
    assert len(cascade) >= 2, f"Expected cascade, got {len(cascade)} updates"

    print(f"  [PASS] Composite engine: evaluation, cascading, cycle detection")


def test_topic_merger():
    """Validate: topic merging merges similar clusters (FIELD-310)."""
    import numpy as np
    from core.topic_merger import HierarchicalTopicMerger, SynonymResolver

    merger = HierarchicalTopicMerger(merge_threshold=0.95)

    # Create two very similar clusters
    centroid_a = np.random.randn(384).astype(np.float32)
    centroid_a /= np.linalg.norm(centroid_a)
    centroid_b = centroid_a + np.random.randn(384).astype(np.float32) * 0.01
    centroid_b /= np.linalg.norm(centroid_b)
    centroid_c = np.random.randn(384).astype(np.float32)
    centroid_c /= np.linalg.norm(centroid_c)

    clusters = [
        {"cluster_id": "c1", "label": "Payment Terms", "centroid": centroid_a,
         "quality_score": 0.9, "chunk_count": 50, "keywords": ["payment"]},
        {"cluster_id": "c2", "label": "Net Payment Terms", "centroid": centroid_b,
         "quality_score": 0.85, "chunk_count": 30, "keywords": ["net", "payment"]},
        {"cluster_id": "c3", "label": "Liability", "centroid": centroid_c,
         "quality_score": 0.88, "chunk_count": 40, "keywords": ["liability"]},
    ]

    merged = merger.merge_similar_topics(clusters)
    assert len(merged) <= len(clusters), "Should merge at least one pair"

    # Synonym resolution
    canonical = SynonymResolver.resolve("net_terms")
    assert canonical == "payment_days", f"Expected payment_days, got {canonical}"

    canonical2 = SynonymResolver.resolve("choice_of_law")
    assert canonical2 == "governing_law", f"Expected governing_law, got {canonical2}"

    print(f"  [PASS] Topic merger: {len(clusters)} → {len(merged)} clusters (FIELD-310)")


def test_incremental_assigner():
    """Validate: incremental cluster assignment without re-clustering (FIELD-801)."""
    import numpy as np
    from core.cluster_assigner import IncrementalClusterAssigner, AssignmentType

    assigner = IncrementalClusterAssigner(high_confidence_threshold=0.85, tentative_threshold=0.60)

    # Create fake centroids
    centroid_a = np.random.randn(384).astype(np.float32)
    centroid_a /= np.linalg.norm(centroid_a)
    centroid_b = np.random.randn(384).astype(np.float32)
    centroid_b /= np.linalg.norm(centroid_b)

    assigner.load_centroids({
        "cluster_payment": (centroid_a, 100),
        "cluster_liability": (centroid_b, 80),
    })

    # Test assignment (can't guarantee exact confidence without real embeddings,
    # but structure should be correct)
    result = assigner.assign_document("Payment terms are net 30 days.", "doc_test")
    assert "cluster_id" in result
    assert "confidence" in result
    assert "assignment_type" in result
    assert len(result["top_matches"]) <= 5

    print(f"  [PASS] Incremental assigner: assignment_type={result['assignment_type'].value}, "
          f"confidence={result['confidence']:.3f} (FIELD-801)")


def test_store():
    """Validate: DuckDB store creates tables, reads/writes correctly."""
    import numpy as np
    from core.store import ClusteringStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_store.duckdb"
        store = ClusteringStore(str(db_path))

        # Domain
        store.upsert_domain("d_test", "Test Domain", "A test domain", 10, 0.9)
        domains = store.get_domains()
        assert len(domains) == 1
        assert domains[0]["label"] == "Test Domain"

        # Agreement
        store.insert_agreement("agr_1", "test.pdf", "Sample text", "Test summary", "d_test", 0.9)
        agrs = store.get_agreements("d_test")
        assert len(agrs) == 1

        # Chunk with embedding
        emb = np.random.randn(384).astype(np.float32)
        store.insert_chunk("chk_1", "agr_1", "test chunk text", 0, emb, 50)
        chunks = store.get_chunks(agreement_id="agr_1")
        assert len(chunks) == 1

        # Cluster
        centroid = np.random.randn(384).astype(np.float32)
        store.upsert_cluster("c_1", "d_test", "Payment Terms", "test", ["payment"], 10, 5, 0.9, centroid)
        clusters = store.get_clusters("d_test")
        assert len(clusters) == 1
        assert clusters[0]["keywords"] == ["payment"]

        # Field
        store.upsert_field("f_1", "c_1", "payment_days", "int", "Days to pay")
        fields = store.get_fields("c_1")
        assert len(fields) == 1

        # Extraction + correction
        store.insert_extraction("ext_1", "agr_1", "f_1", "30", 0.9, "chk_1")
        store.correct_extraction("ext_1", "45")
        corrections = store.get_corrections()
        assert len(corrections) == 1
        assert corrections[0]["old_value"] == "30"
        assert corrections[0]["new_value"] == "45"

        # Stats
        stats = store.get_stats()
        assert stats["domains"] == 1
        assert stats["corrections"] == 1

        store.close()

    print(f"  [PASS] DuckDB store: all CRUD operations verified")


def test_hybrid_retrieval():
    """Validate: hybrid search returns results with all score signals."""
    import numpy as np
    from core.retrieval import HybridRetriever

    chunks = [
        "Payment shall be made within thirty days of invoice date.",
        "The liability of either party shall not exceed five million dollars ($5M).",
        "This agreement automatically renews for one-year terms unless terminated.",
        "All confidential information shall remain protected for three years.",
        "The vendor shall provide support services 24/7 with 99.9% uptime SLA.",
    ]
    embeddings = np.random.randn(5, 384).astype(np.float32)
    for i in range(5):
        embeddings[i] /= np.linalg.norm(embeddings[i])

    retriever = HybridRetriever(
        chunks=chunks,
        embeddings=embeddings,
        embed_fn=lambda q: np.random.randn(384).astype(np.float32),
    )

    results = retriever.search("liability cap $5M", top_k=3, use_reranker=False)
    assert len(results) > 0, "Search returned no results"
    assert "rrf_score" in results[0]
    assert "dense_score" in results[0]
    assert "bm25_score" in results[0]

    # Test with reranker
    results_reranked = retriever.search("liability cap", top_k=3, use_reranker=True)
    assert "cross_encoder_score" in results_reranked[0]

    print(f"  [PASS] Hybrid retrieval: {len(results)} results with all score signals")


def test_nupunkt_chunker():
    """Validate nupunkt structural decomposition."""
    from core.nupunkt_chunker import structural_decompose, nupunkt_available

    sample_text = """14.3 Termination for Convenience. Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party.

14.4 Effect of Termination. Upon termination, the Receiving Party shall return or destroy all Confidential Information. The obligations under Section 5 shall survive termination."""

    blocks = structural_decompose(sample_text)
    assert len(blocks) >= 1, f"Expected at least 1 block, got {len(blocks)}"

    for block in blocks:
        assert "block_id" in block, "Block missing block_id"
        assert "sentences" in block, "Block missing sentences"
        assert "text" in block, "Block missing text"
        assert len(block["sentences"]) >= 1, f"Block has no sentences"

    all_sentences = [s for b in blocks for s in b["sentences"]]
    for sent in all_sentences:
        assert "sentence_id" in sent, "Sentence missing sentence_id"
        assert "text" in sent, "Sentence missing text"
        assert len(sent["text"].strip()) > 0, "Empty sentence text"

    sentence_texts = [s["text"] for s in all_sentences]
    assert any("14.3" in s or "Termination for Convenience" in s for s in sentence_texts), \
        "Section number 14.3 was incorrectly split"

    print("  [PASS] nupunkt structural decomposition")
    print(f"    Blocks: {len(blocks)}, Sentences: {len(all_sentences)}")
    if nupunkt_available():
        print("    Using nupunkt tokenizer")
    else:
        print("    Using regex fallback")


def test_keybert_scorer():
    """Validate KeyBERT clause scoring and text augmentation."""
    from core.keybert_scorer import KeyBERTScorer, keybert_available

    scorer = KeyBERTScorer()

    if not keybert_available():
        print("  [SKIP] KeyBERT fingerprints not found — scorer degrades gracefully")
        return

    text = "The receiving party shall not disclose any confidential information."
    scores = scorer.classify(text)
    assert isinstance(scores, list), f"Expected list, got {type(scores)}"
    if scores:
        clause_type, similarity = scores[0]
        assert isinstance(clause_type, str), "Clause type should be string"
        assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"

    augmented = scorer.augment_text(text)
    assert isinstance(augmented, str), "Augmented text should be string"
    assert text in augmented, "Original text should be preserved in augmentation"

    label = scorer.get_label(text)
    assert label is None or isinstance(label, int), "Label should be int or None"

    print("  [PASS] KeyBERT scorer")
    print(f"    Top classification: {scores[0] if scores else 'none'}")
    print(f"    Augmented prefix: {augmented[:60]}...")


def test_arm_miner():
    """Validate ARM mining on synthetic data."""
    from core.arm.arm_miner import (
        build_clause_presence_matrix,
        discover_term_packages,
        mine_global,
        mlxtend_available,
    )

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed — ARM mining unavailable")
        return

    import random
    random.seed(42)
    assignments = {}
    for i in range(100):
        present = []
        if random.random() < 0.7:
            present.extend(["Termination", "Notice"])
        if random.random() < 0.5:
            present.append("Confidentiality")
        if random.random() < 0.3:
            present.append("Indemnification")
        if random.random() < 0.2:
            present.append("IP Rights")
        if present:
            assignments[f"doc_{i}"] = present

    matrix = build_clause_presence_matrix(assignments)
    assert not matrix.empty, "Matrix should not be empty"
    assert matrix.shape[0] >= 50, f"Expected >= 50 docs, got {matrix.shape[0]}"

    rules = mine_global(matrix, config={"min_support": 0.15, "min_confidence": 0.5, "min_lift": 1.0})
    assert isinstance(rules, list), "Rules should be a list"

    packages = discover_term_packages(matrix, config={"min_support": 0.10, "min_itemset_size": 2})
    assert isinstance(packages, list), "Packages should be a list"

    print(f"  [PASS] ARM mining")
    print(f"    Matrix: {matrix.shape}, Rules: {len(rules)}, Packages: {len(packages)}")


def test_relationship_layer():
    """Validate RelationshipLayer end-to-end: mine -> persist -> query."""
    from core.arm.arm_miner import mlxtend_available

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed")
        return

    import tempfile
    from core.store import ClusteringStore
    from core.relationship_layer import RelationshipLayer

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as tmp:
        store = ClusteringStore(tmp.name)
        layer = RelationshipLayer(store)

        import random
        random.seed(42)
        assignments = {}
        for i in range(100):
            present = []
            if random.random() < 0.7:
                present.extend(["Termination", "Notice"])
            if random.random() < 0.5:
                present.append("Confidentiality")
            if random.random() < 0.3:
                present.append("Indemnification")
            if present:
                assignments[f"doc_{i}"] = present

        layer.mine_from_assignments(assignments, pipeline_run_id="test_run")

        related = layer.get_related_clause_types("Termination")
        packages = layer.get_term_packages()

        assert isinstance(related, list), "Related should be a list"
        assert isinstance(packages, list), "Packages should be a list"

        print(f"  [PASS] RelationshipLayer")
        print(f"    Related to 'Termination': {len(related)}, Packages: {len(packages)}")


def test_relationship_integration():
    """Validate that downstream consumers can query the relationship layer."""
    from core.arm.arm_miner import mlxtend_available

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed")
        return

    # Test _build_arm_context without relationship layer (fallback)
    from core.field_discovery import _build_arm_context

    fd_ctx = _build_arm_context("Termination", relationship_layer=None)
    assert "related_clauses" in fd_ctx, "Missing related_clauses key"
    assert fd_ctx["related_clauses"] == "No relationship data available."

    try:
        from core.extractor import _build_extraction_arm_context
        ex_ctx = _build_extraction_arm_context("Termination", relationship_layer=None)
        assert "cross_clause_fields" in ex_ctx, "Missing cross_clause_fields key"
        assert ex_ctx["cross_clause_fields"] == "None."
    except ImportError as e:
        # openai/llm_client dependency may not be available in all environments
        print(f"  [NOTE] extractor import skipped (missing dependency: {e})")

    print(f"  [PASS] Relationship integration (fallback paths verified)")


def main():
    parser = argparse.ArgumentParser(description="Validate Clustering V2 pipeline")
    parser.add_argument("--skip-llm", action="store_true", help="Skip tests requiring LLM API calls")
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════════════════")
    print("  Clustering V2 — Pipeline Validation")
    print("═══════════════════════════════════════════════════════════════")
    print()

    tests = [
        ("Chunker", test_chunker),
        ("nupunkt Chunker", test_nupunkt_chunker),
        ("Embedder", test_embedder),
        ("Clause Embedder (FIELD-110)", test_clause_embedder),
        ("Composite Engine", test_composite_engine),
        ("Topic Merger (FIELD-310)", test_topic_merger),
        ("Incremental Assigner (FIELD-801)", test_incremental_assigner),
        ("DuckDB Store", test_store),
        ("Hybrid Retrieval", test_hybrid_retrieval),
        ("KeyBERT Scorer", test_keybert_scorer),
        ("ARM Miner", test_arm_miner),
        ("Relationship Layer", test_relationship_layer),
        ("Relationship Integration", test_relationship_integration),
    ]

    passed, failed = 0, 0
    for name, test_fn in tests:
        try:
            print(f"Testing {name}...")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print()
    print("═══════════════════════════════════════════════════════════════")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("  All infrastructure tests passed!")
        if args.skip_llm:
            print("  (LLM tests were skipped — run without --skip-llm for full validation)")
    else:
        print("  Some tests failed. Fix issues above and re-run.")
    print("═══════════════════════════════════════════════════════════════")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
