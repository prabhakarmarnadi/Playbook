#!/usr/bin/env python3
"""
Re-run EVoC clustering from cached embeddings and persist all native layers.

This does NOT re-embed, re-extract fields, or re-do intents.
It only:
  1) Reads chunk/clause/summary embeddings from the existing DB
  2) Runs EVoC clause clustering → stores all layers
  3) Runs EVoC macro clustering → stores all layers
  4) Writes layer data to cluster_layers / cluster_layer_meta tables

Usage:
    python scripts/recluster_with_layers.py data/cuad_510_demo.duckdb
"""
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.store import ClusteringStore
from core.evoc_pipeline import evoc_cluster, _select_best_layer


def recluster(db_path: str):
    store = ClusteringStore(db_path)

    # ── 1. Load clause embeddings (mean of chunk embeddings per clause) ──
    print("Loading clauses and chunk embeddings...")
    clauses = store.conn.execute("""
        SELECT c.clause_id, c.agreement_id
        FROM clauses c
        WHERE EXISTS (SELECT 1 FROM chunks ch WHERE ch.clause_id = c.clause_id AND ch.embedding IS NOT NULL)
        ORDER BY c.clause_id
    """).fetchdf().to_dict("records")
    print(f"  {len(clauses)} clauses with embeddings")

    # Build clause → chunk embedding mapping
    chunk_rows = store.conn.execute("""
        SELECT ch.clause_id, ch.embedding
        FROM chunks ch
        WHERE ch.clause_id IS NOT NULL AND ch.embedding IS NOT NULL
    """).fetchall()

    from collections import defaultdict
    clause_chunk_embs = defaultdict(list)
    for clause_id, emb_blob in chunk_rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        clause_chunk_embs[clause_id].append(emb)

    valid_clauses = [c for c in clauses if c["clause_id"] in clause_chunk_embs]
    print(f"  {len(valid_clauses)} valid clauses (with chunk embeddings)")

    dim = len(next(iter(clause_chunk_embs.values()))[0])
    clause_embeddings = np.zeros((len(valid_clauses), dim), dtype=np.float32)
    for i, cl in enumerate(valid_clauses):
        embs = np.array(clause_chunk_embs[cl["clause_id"]])
        mean_emb = embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        clause_embeddings[i] = mean_emb / norm if norm > 0 else mean_emb

    # ── 2. EVoC clause clustering ────────────────────────────────────────
    n_clauses = len(valid_clauses)
    print(f"\nRunning EVoC clause clustering on {n_clauses} clauses...")
    clause_topics, clause_layers, clause_persistence = evoc_cluster(
        clause_embeddings,
        base_min_cluster_size=max(3, n_clauses // 100),
        noise_level=0.5,
        n_neighbors=min(15, n_clauses - 1),
        min_samples=3,
    )

    clause_topics, best_clause_idx, clause_lscores = _select_best_layer(
        clause_layers, clause_embeddings, clause_persistence,
    )

    print(f"  {len(clause_layers)} layers produced")
    for li, layer in enumerate(clause_layers):
        n_t = len(set(layer) - {-1})
        n_o = sum(1 for t in layer if t == -1)
        sel = " ← SELECTED" if li == best_clause_idx else ""
        sc = clause_lscores[li] if li < len(clause_lscores) else {}
        comp = sc.get("composite", "N/A")
        print(f"    Layer {li}: {n_t} clusters, {n_o} outliers, "
              f"persistence={clause_persistence[li]:.4f}, composite={comp}{sel}")

    # Persist clause layers
    clause_item_ids = [cl["clause_id"] for cl in valid_clauses]
    store.save_cluster_layers(
        "clause", clause_layers, clause_item_ids,
        clause_persistence, best_clause_idx, clause_lscores,
    )
    print(f"  Saved {len(clause_layers)} clause layers to DB")

    # ── 3. EVoC macro domain clustering ──────────────────────────────────
    print("\nLoading summary embeddings for macro clustering...")
    agreements = store.get_agreements()

    # Re-embed summaries using the same model
    summary_texts = [a.get("summary_text", "") for a in agreements]
    n_with_summary = sum(1 for s in summary_texts if s)
    print(f"  {len(agreements)} agreements, {n_with_summary} with summaries")

    if n_with_summary < 10:
        print("  Skipping macro clustering (too few summaries)")
        store.close()
        return

    # Use nomic model for summary embeddings (same as pipeline)
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        print("  Encoding summaries...")
        # Nomic requires "search_document: " prefix
        prefixed = [f"search_document: {s[:2000]}" for s in summary_texts]
        summary_embeddings = embed_model.encode(
            prefixed, show_progress_bar=True, batch_size=64,
            normalize_embeddings=True,
        ).astype(np.float32)
    except Exception as e:
        print(f"  Could not load embedding model: {e}")
        print("  Trying to reuse existing embeddings from chunks per agreement...")
        # Fallback: use mean of chunk embeddings per agreement
        agr_embs = {}
        for cl in valid_clauses:
            aid = cl["agreement_id"]
            if aid not in agr_embs:
                agr_embs[aid] = []
        for i, cl in enumerate(valid_clauses):
            agr_embs[cl["agreement_id"]].append(clause_embeddings[i])

        summary_embeddings = np.zeros((len(agreements), dim), dtype=np.float32)
        for i, a in enumerate(agreements):
            embs = agr_embs.get(a["agreement_id"], [])
            if embs:
                mean = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean)
                summary_embeddings[i] = mean / norm if norm > 0 else mean

    n_docs = len(agreements)
    print(f"\nRunning EVoC macro clustering on {n_docs} documents...")
    macro_labels, macro_layers, macro_persistence = evoc_cluster(
        summary_embeddings,
        base_min_cluster_size=max(3, n_docs // 50),
        noise_level=0.5,
        n_neighbors=min(15, n_docs - 1),
        min_samples=2,
    )

    macro_labels, best_macro_idx, macro_lscores = _select_best_layer(
        macro_layers, summary_embeddings, macro_persistence,
        prefer_granularity=True,
    )

    print(f"  {len(macro_layers)} layers produced")
    for li, layer in enumerate(macro_layers):
        n_t = len(set(layer) - {-1})
        n_o = sum(1 for t in layer if t == -1)
        sel = " ← SELECTED" if li == best_macro_idx else ""
        sc = macro_lscores[li] if li < len(macro_lscores) else {}
        comp = sc.get("composite", "N/A")
        print(f"    Layer {li}: {n_t} domains, {n_o} outliers, "
              f"persistence={macro_persistence[li]:.4f}, composite={comp}{sel}")

    # Persist macro layers
    agr_ids = [a["agreement_id"] for a in agreements]
    store.save_cluster_layers(
        "macro", macro_layers, agr_ids,
        macro_persistence, best_macro_idx, macro_lscores,
    )
    print(f"  Saved {len(macro_layers)} macro layers to DB")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone! Layers stored in {db_path}")
    print(f"  Clause: {len(clause_layers)} layers, "
          f"selected layer {best_clause_idx} with "
          f"{len(set(clause_topics) - {-1})} clusters")
    print(f"  Macro: {len(macro_layers)} layers, "
          f"selected layer {best_macro_idx} with "
          f"{len(set(macro_labels) - {-1})} domains")

    store.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Re-cluster with layer persistence")
    p.add_argument("db_path", help="Path to DuckDB file")
    args = p.parse_args()
    recluster(args.db_path)
