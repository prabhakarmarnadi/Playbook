"""
Clause-type clustering pipeline.

Runs the full: parse → chunk → embed → clause-type cluster → field discovery flow.
Groups clauses across all documents by semantic similarity, producing clause types
with associated fields.

Artifacts stored per account_id under data/artifacts/<account_id>/:
  - clause_model/         BERTopic safetensors + config
  - embeddings.npz        chunk + summary embeddings
  - clause_types.json     hierarchical clause type → clause → field output
  - run_manifest.json     run metadata + stats

Usage:
    # CLI
    python -m core.clause_pipeline /path/to/pdfs --account-id acct_001

    # Python
    from core.clause_pipeline import run_clause_pipeline
    result = run_clause_pipeline("/path/to/pdfs", account_id="acct_001")
"""
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from config import (
    DB_PATH, DATA_DIR, LEGAL_STOPWORDS,
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES,
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST,
    OPTIMIZE_TRIALS,
)
from core.pdf_parser import parse_pdf_directory, load_parsed_agreements, PARSED_TEXT_DIR
from core.chunker import semantic_chunk, sliding_window_chunk, get_document_summary, SemanticChunkerConfig, extract_clause_header
from core.nomic_embedder import NomicEmbedder
from core.cuml_wrappers import make_umap, make_hdbscan, cuml_available
from core.lancedb_store import LanceVectorStore
from core.store import ClusteringStore
from core.topic_merger import HierarchicalTopicMerger
from core.micro_clusterer import compute_cluster_quality

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = DATA_DIR / "artifacts"

# Combined stopword set (English + legal domain) — matches legacy
COMBINED_STOPWORDS = set(ENGLISH_STOP_WORDS) | set(LEGAL_STOPWORDS)


class StopwordFilteringAnalyzer:
    """Picklable custom analyzer that filters n-grams where ALL tokens are stopwords.

    Ported from legacy bertopic_clusterer.py — ensures n-grams like
    'pursuant to' or 'in accordance with' are filtered out while keeping
    meaningful phrases like 'termination clause'.
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
        filtered = []
        for token in tokens:
            words = token.split()
            if any(word not in self.stopwords for word in words):
                filtered.append(token)
        return filtered

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_analyzer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def _auto_label_topic(topic_model: BERTopic, topic_id: int, prefix: str = "") -> dict:
    if topic_id == -1:
        return {"label": "Uncategorized", "description": "Not assigned to any cluster"}
    keywords = [w for w, _ in topic_model.get_topic(topic_id)]
    top = keywords[:5]
    label = f"{prefix}{' / '.join(top[:3]).title()}" if top else f"{prefix}Topic_{topic_id}"
    desc = f"Keywords: {', '.join(top)}" if top else ""
    return {"label": label, "description": desc}


# ── LLM Cluster Labeling ──────────────────────────────────────────────────────

def _get_labeling_client():
    """Create Azure OpenAI client for labeling (reuses field discovery creds)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        return None
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception:
        return None


def _llm_label_call(client, deployment: str, prompt: str) -> dict | None:
    """Single Azure OpenAI call for labeling. Returns parsed JSON or None."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
                timeout=30,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"LLM label attempt {attempt+1} failed: {e}")
            if attempt < 1:
                time.sleep(1)
    return None


def _llm_label_domains(
    macro_model: BERTopic,
    domain_map: dict,
    summaries: list[str],
    agreements: list[dict],
) -> dict[int, dict]:
    """Label all macro domains via LLM. Returns {topic_id: {label, description}}."""
    client = _get_labeling_client()
    if not client:
        logger.info("LLM labeling: Azure OpenAI not configured, using keyword labels")
        return {}

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    labels = {}

    for topic_id, dr in domain_map.items():
        if topic_id == -1:
            continue

        # Gather keywords + representative doc snippets
        keywords = []
        if macro_model:
            try:
                keywords = [w for w, _ in macro_model.get_topic(topic_id)][:10]
            except Exception:
                pass

        # Sample 3 doc summaries from this domain
        sample_indices = dr["agreement_indices"][:3]
        sample_summaries = [summaries[i][:500] for i in sample_indices]
        sample_filenames = [agreements[i].get("filename", "") for i in sample_indices]

        prompt = f"""You are categorizing clusters of legal agreements/documents.

Keywords for this cluster: {', '.join(keywords[:10])}

Sample documents in this cluster:
{chr(10).join(f'- [{fn}]: {s}' for fn, s in zip(sample_filenames, sample_summaries))}

What type of agreements/documents are in this cluster? 

Respond in JSON:
{{"label": "A clear 2-5 word domain name like 'SaaS Agreements', 'Employment Contracts', 'Real Estate Leases', 'Government Procurement', 'Financial Services'", "description": "One sentence describing what types of documents are in this domain"}}"""

        result = _llm_label_call(client, deployment, prompt)
        if result and result.get("label"):
            labels[topic_id] = result
            logger.info(f"  Domain {topic_id}: '{result['label']}'")
        else:
            # Fallback to keyword-based
            labels[topic_id] = _auto_label_topic(macro_model, topic_id) if macro_model else {
                "label": f"Domain_{topic_id}", "description": ""
            }

    return labels


def _llm_label_micro_clusters(
    micro_model: BERTopic,
    domain_label: str,
    chunk_texts: list[str],
    micro_topics: list[int],
) -> dict[int, dict]:
    """Label all micro clusters within a domain via LLM. Returns {topic_id: {label, description}}."""
    client = _get_labeling_client()
    if not client:
        return {}

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    labels = {}
    unique_topics = sorted(set(t for t in micro_topics if t != -1))

    for tid in unique_topics:
        keywords = [w for w, _ in micro_model.get_topic(tid)][:10]

        # Sample 3 representative chunks
        mask = [i for i, t in enumerate(micro_topics) if t == tid]
        sample_chunks = [chunk_texts[i][:400] for i in mask[:3]]

        prompt = f"""You are labeling clause clusters within "{domain_label}" agreements.

Keywords for this cluster: {', '.join(keywords)}

Representative clause excerpts:
{chr(10).join(f'- {c}' for c in sample_chunks)}

What type of legal clause or document section is this?

Respond in JSON:
{{"label": "A clear clause type name like 'Payment Terms', 'Liability Limitation', 'Confidentiality', 'Termination Rights', 'Scope of Work', 'Insurance Requirements'", "description": "One sentence describing what this clause covers"}}"""

        result = _llm_label_call(client, deployment, prompt)
        if result and result.get("label"):
            labels[tid] = result
        else:
            labels[tid] = _auto_label_topic(micro_model, tid)

    return labels


def _save_model(model: BERTopic, path: Path, name: str = "model"):
    """Save BERTopic model + embedding_model=None (we embed separately)."""
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / name
    try:
        model.save(
            str(save_path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        logger.info(f"Saved BERTopic model: {save_path}")
    except (TypeError, Exception) as e:
        # BERTopic save can fail with numpy int64 JSON serialization;
        # fall back to pickle serialization in a separate path
        import shutil
        if save_path.exists():
            shutil.rmtree(save_path, ignore_errors=True)
        model.save(
            str(save_path),
            serialization="pickle",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        logger.info(f"Saved BERTopic model (pickle fallback): {save_path}")


def _save_embeddings(path: Path, chunk_embeddings: np.ndarray, summary_embeddings: np.ndarray):
    """Save embeddings as compressed npz for re-use."""
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "embeddings.npz",
        chunk_embeddings=chunk_embeddings,
        summary_embeddings=summary_embeddings,
    )
    logger.info(f"Saved embeddings: {path / 'embeddings.npz'} "
                f"(chunks={chunk_embeddings.shape}, summaries={summary_embeddings.shape})")


def _load_cached_embeddings(account_dir: Path, expected_chunks: int, expected_summaries: int):
    """Try to load cached embeddings from the most recent run for this account.
    
    Returns (chunk_embeddings, summary_embeddings) if cache hit, else (None, None).
    """
    if not account_dir.exists():
        return None, None
    # Find most recent run dir with embeddings
    run_dirs = sorted(
        [d for d in account_dir.iterdir() if d.is_dir() and (d / "embeddings.npz").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        return None, None
    cache_path = run_dirs[0] / "embeddings.npz"
    try:
        data = np.load(cache_path)
        chunk_emb = data["chunk_embeddings"]
        summary_emb = data["summary_embeddings"]
        if chunk_emb.shape[0] == expected_chunks and summary_emb.shape[0] == expected_summaries:
            logger.info(f"Embedding cache HIT: {cache_path} "
                        f"(chunks={chunk_emb.shape}, summaries={summary_emb.shape})")
            return chunk_emb, summary_emb
        logger.info(f"Embedding cache MISS (shape mismatch): cached chunks={chunk_emb.shape[0]} "
                     f"vs expected={expected_chunks}, cached summaries={summary_emb.shape[0]} "
                     f"vs expected={expected_summaries}")
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings from {cache_path}: {e}")
    return None, None


def _save_manifest(path: Path, manifest: dict):
    """Save run manifest JSON."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Saved manifest: {path / 'run_manifest.json'}")


def _save_quality_report(path: Path, report: dict):
    """Save cluster quality evaluation report."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cluster_quality.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved quality report: {path / 'cluster_quality.json'}")


def _save_clustering_results(
    path: Path,
    account_id: str,
    agreements: list[dict],
    summaries: list[str],
    summary_embeddings: np.ndarray,
    domain_map: dict,
    macro_topics: list[int],
    macro_model: BERTopic | None,
    all_chunks: list[dict],
    chunk_embeddings: np.ndarray,
    cluster_records_by_domain: dict[str, list[dict]],
    micro_topics_by_domain: dict[str, list[int]],
    quality_report: dict,
):
    """Save clustering_results.json in legacy-compatible format."""
    from datetime import datetime
    from sklearn.metrics import silhouette_score as sk_silhouette

    path.mkdir(parents=True, exist_ok=True)

    # Per-domain cosine similarities
    cosine_sims = {}
    for topic_id, dr in domain_map.items():
        if topic_id == -1:
            continue
        mask = dr["agreement_indices"]
        if len(mask) < 2:
            continue
        embs = summary_embeddings[mask]
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = [float(np.dot(embs[i], centroid)) for i in range(len(mask))]
        cosine_sims[str(topic_id)] = float(np.mean(sims))

    avg_cosine = float(np.mean(list(cosine_sims.values()))) if cosine_sims else 0.0

    # Silhouette score (macro level)
    non_outlier = [i for i, t in enumerate(macro_topics) if t != -1]
    silhouette = 0.0
    if len(set(macro_topics[i] for i in non_outlier)) >= 2 and len(non_outlier) >= 3:
        try:
            silhouette = float(sk_silhouette(
                summary_embeddings[non_outlier],
                [macro_topics[i] for i in non_outlier],
                metric="cosine",
            ))
        except Exception:
            pass

    # Build cluster entries in legacy format
    clusters = []
    for topic_id, dr in sorted(domain_map.items(), key=lambda x: x[0]):
        if topic_id == -1:
            continue
        domain_id = dr["domain_id"]
        label = dr.get("label", f"Topic_{topic_id}")
        mask = dr["agreement_indices"]

        # Representative words from macro model
        rep_words = []
        if macro_model and topic_id != -1:
            try:
                rep_words = [w for w, _ in macro_model.get_topic(topic_id)][:10]
            except Exception:
                pass

        # Representative docs (up to 3 per domain)
        rep_docs = []
        for idx in mask[:3]:
            agr = agreements[idx]
            rep_docs.append({
                "agreement_id": agr["agreement_id"],
                "document_text": summaries[idx][:2000],
                "snippets": [],
                "reasoning": "",
                "cluster_id": topic_id,
                "cluster_name": label,
                "agreement_type": None,
                "probability": 1.0,
                "outlier_score": 0.0,
            })

        # Micro-level sub-clusters for this domain
        sub_clusters = []
        domain_clusters = cluster_records_by_domain.get(domain_id, [])
        for cr in domain_clusters:
            sub_clusters.append({
                "cluster_id": cr["cluster_id"],
                "label": cr["label"],
                "keywords": cr.get("keywords", [])[:10],
                "chunk_count": cr["chunk_count"],
                "quality_score": round(cr.get("quality_score", 0), 4),
            })

        clusters.append({
            "topic_id": topic_id,
            "topic_name": label,
            "domain_id": domain_id,
            "document_count": len(mask),
            "unique_document_count": len(mask),
            "representative_words": rep_words,
            "representative_docs": rep_docs,
            "micro_clusters": sub_clusters,
        })

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "account_id": account_id,
            "version": "v2.0",
            "umap_params": {},
            "hdbscan_params": {},
        },
        "total_topics": len([t for t in domain_map if t != -1]),
        "total_documents": len(agreements),
        "scores": {
            "cosine_similarities": cosine_sims,
            "average_cosine_similarity": avg_cosine,
            "silhouette_score": silhouette,
            "dbcv_score": 0.0,
            "ari_score": 0.0,
            "nmi_score": 0.0,
        },
        "clusters": clusters,
    }

    out_path = path / "clustering_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved clustering_results.json: {out_path}")
    return result


def _dedup_universal_fields(store, progress):
    """Identify fields that appear across multiple clusters and tag them as universal.

    Common fields like 'party_name', 'effective_date', 'document_title' that appear
    in 2+ clusters with the same name and type are marked with source='universal'.
    This metadata informs the ontology tree and extraction dedup.
    """
    from collections import Counter

    all_fields = store.get_fields()
    # Group by (name, field_type)
    field_key_counts = Counter()
    field_key_ids = {}
    for f in all_fields:
        key = (f["name"], f.get("field_type", "text"))
        field_key_counts[key] += 1
        if key not in field_key_ids:
            field_key_ids[key] = []
        field_key_ids[key].append(f["field_id"])

    universal_fields = [(name, ftype, ids) for (name, ftype), ids
                        in field_key_ids.items() if field_key_counts[(name, ftype)] >= 2]

    if universal_fields:
        for name, ftype, ids in universal_fields:
            # Mark all instances as universal
            for fid in ids:
                try:
                    store.conn.execute(
                        "UPDATE field_definitions SET source = 'universal' WHERE field_id = ?",
                        [fid]
                    )
                except Exception:
                    pass
        progress("field_dedup", f"Tagged {len(universal_fields)} universal fields: "
                 f"{[n for n, _, _ in universal_fields]}")
    else:
        progress("field_dedup", "No cross-cluster duplicates found")


def _export_fields_json(store, path: Path):
    """Export all discovered fields with cluster/domain context to fields.json."""
    path.mkdir(parents=True, exist_ok=True)

    domains = {d["domain_id"]: d for d in store.get_domains()}
    clusters = {c["cluster_id"]: c for c in store.get_clusters()}

    fields = []
    for fd in store.get_fields():
        cluster = clusters.get(fd.get("cluster_id", ""), {})
        domain = domains.get(cluster.get("domain_id", ""), {})
        examples = fd.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except Exception:
                examples = []
        fields.append({
            "field_id": fd.get("field_id"),
            "name": fd.get("name"),
            "field_type": fd.get("field_type"),
            "description": fd.get("description"),
            "source": fd.get("source"),
            "examples": examples,
            "cluster_id": fd.get("cluster_id"),
            "cluster_label": cluster.get("label", ""),
            "domain_id": cluster.get("domain_id", ""),
            "domain_label": domain.get("label", ""),
        })

    out = path / "fields.json"
    with open(out, "w") as f:
        json.dump({"total_fields": len(fields), "fields": fields}, f, indent=2, default=str)
    logger.info(f"Saved fields.json: {out} ({len(fields)} fields)")


def _export_chunks_json(store, path: Path):
    """Export all chunks with cluster/domain assignments to chunks.json."""
    path.mkdir(parents=True, exist_ok=True)

    domains = {d["domain_id"]: d for d in store.get_domains()}
    clusters = {c["cluster_id"]: c for c in store.get_clusters()}
    agreements = {a["agreement_id"]: a for a in store.get_agreements()}

    # Single query: all chunks with their cluster assignment
    rows = store.conn.execute("""
        SELECT ch.chunk_id, ch.agreement_id, ch.chunk_text, ch.chunk_index, ch.token_count,
               ca.cluster_id
        FROM chunks ch
        LEFT JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        ORDER BY ch.agreement_id, ch.chunk_index
    """).fetchall()

    chunks = []
    for chunk_id, agr_id, text, cidx, tcnt, cid in rows:
        cluster = clusters.get(cid or "", {})
        domain = domains.get(cluster.get("domain_id", ""), {})
        agr = agreements.get(agr_id or "", {})
        chunks.append({
            "chunk_id": chunk_id,
            "agreement_id": agr_id,
            "filename": agr.get("filename", ""),
            "chunk_index": cidx,
            "token_count": tcnt,
            "text": text,
            "cluster_id": cid,
            "cluster_label": cluster.get("label", ""),
            "domain_id": cluster.get("domain_id", "") or None,
            "domain_label": domain.get("label", ""),
        })

    out = path / "chunks.json"
    with open(out, "w") as f:
        json.dump({"total_chunks": len(chunks), "chunks": chunks}, f, indent=2, default=str)
    logger.info(f"Saved chunks.json: {out} ({len(chunks)} chunks)")


def _export_clause_types_json(store, path: Path, progress_fn=None):
    """Export the full clause-type ontology: clause_types → clauses → fields.

    Produces clause_types.json with structure:
    {
        "total_clause_types": N,
        "total_clauses": M,
        "total_fields": F,
        "clause_types": [
            {
                "clause_type_id": "ct_...",
                "label": "Indemnification",
                "description": "...",
                "keywords": [...],
                "clause_count": 47,
                "agreement_count": 35,
                "fields": [
                    {"field_id": "fld_...", "name": "indemnifying_party", "type": "entity_name", ...},
                    ...
                ],
                "clauses": [
                    {"clause_id": "cls_...", "agreement_id": "agr_...", "filename": "...",
                     "clause_number": "12.3", "clause_title": "Indemnification",
                     "heading_level": 2, "clause_index": 14},
                    ...
                ]
            },
            ...
        ]
    }
    """
    path.mkdir(parents=True, exist_ok=True)
    progress = progress_fn or (lambda s, m: None)

    clusters = store.get_clusters()
    agreements_map = {a["agreement_id"]: a for a in store.get_agreements()}

    # Get all clauses with their clause_type_id
    all_clauses_rows = store.conn.execute("""
        SELECT clause_id, agreement_id, clause_number, clause_title,
               heading_level, clause_index, clause_type_id, token_count
        FROM clauses ORDER BY agreement_id, clause_index
    """).fetchdf().to_dict("records")

    # Group clauses by clause_type_id
    clauses_by_type: dict[str, list[dict]] = {}
    for cl in all_clauses_rows:
        ctid = cl.get("clause_type_id") or ""
        clauses_by_type.setdefault(ctid, []).append(cl)

    # Get all fields grouped by cluster_id
    all_fields = store.get_fields()
    fields_by_cluster: dict[str, list[dict]] = {}
    for f in all_fields:
        cid = f.get("cluster_id", "")
        fields_by_cluster.setdefault(cid, []).append(f)

    clause_types = []
    total_clauses = 0
    total_fields = 0

    for cluster in clusters:
        cid = cluster["cluster_id"]
        type_clauses = clauses_by_type.get(cid, [])
        type_fields = fields_by_cluster.get(cid, [])

        clause_entries = []
        for cl in type_clauses:
            agr = agreements_map.get(cl.get("agreement_id", ""), {})
            clause_entries.append({
                "clause_id": cl["clause_id"],
                "agreement_id": cl["agreement_id"],
                "filename": agr.get("filename", ""),
                "clause_number": cl.get("clause_number", ""),
                "clause_title": cl.get("clause_title", ""),
                "heading_level": cl.get("heading_level", 0),
                "clause_index": cl.get("clause_index", 0),
            })

        field_entries = []
        for f in type_fields:
            examples = f.get("examples", [])
            if isinstance(examples, str):
                try:
                    examples = json.loads(examples)
                except Exception:
                    examples = []
            field_entries.append({
                "field_id": f.get("field_id"),
                "name": f.get("name"),
                "type": f.get("field_type"),
                "description": f.get("description"),
                "source": f.get("source"),
                "examples": examples,
            })

        clause_types.append({
            "clause_type_id": cid,
            "label": cluster.get("label", ""),
            "description": cluster.get("description", ""),
            "keywords": cluster.get("keywords", [])[:10],
            "clause_count": len(clause_entries),
            "chunk_count": cluster.get("chunk_count", 0),
            "agreement_count": cluster.get("agreement_count", 0),
            "quality_score": round(cluster.get("quality_score", 0) or 0, 4),
            "fields": field_entries,
            "clauses": clause_entries,
        })

        total_clauses += len(clause_entries)
        total_fields += len(field_entries)

    # Sort by clause count descending
    clause_types.sort(key=lambda x: x["clause_count"], reverse=True)

    out = path / "clause_types.json"
    with open(out, "w") as f:
        json.dump({
            "total_clause_types": len(clause_types),
            "total_clauses": total_clauses,
            "total_fields": total_fields,
            "clause_types": clause_types,
        }, f, indent=2, default=str)

    progress("export", f"Saved clause_types.json: {out} "
             f"({len(clause_types)} types, {total_clauses} clauses, {total_fields} fields)")


def run_clause_pipeline(
    pdf_dir: str | Path,
    account_id: str = "default",
    db_path: str | Path | None = None,
    lance_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str = "cuda",
    max_docs: int | None = None,
    use_semantic_chunking: bool = True,
    use_llm_summaries: bool = False,
    callback=None,
) -> dict:
    """
    Parse PDFs → chunk → embed → clause-type cluster → field discovery → save artifacts.

    All BERTopic models and embeddings are persisted locally under
    <artifacts_dir>/<account_id>/<run_id>/ for later incremental assignment.

    Args:
        pdf_dir: Directory containing PDF files (searched recursively).
        account_id: Tenant ID for artifact isolation.
        db_path: DuckDB path (default: data/<account_id>.duckdb).
        lance_path: LanceDB path (default: data/<account_id>.lance).
        artifacts_dir: Root for model artifacts (default: data/artifacts/).
        device: "cuda" or "cpu".
        max_docs: Limit number of documents (for testing).
        use_semantic_chunking: If False, use fast sliding window (for large batches).
        callback: Progress callback(stage, message).

    Returns:
        Dict with store, lance_store, run_id, stats, artifact paths.
    """
    t0 = time.time()
    stage_timings: dict[str, float] = {}  # stage_name -> seconds
    _stage_t0: dict[str, float] = {}  # for tracking in-progress stages

    def progress(stage: str, msg: str):
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:6.1f}s] [{stage}] {msg}")
        if callback:
            callback(stage, msg)

    def stage_start(name: str):
        _stage_t0[name] = time.time()

    def stage_end(name: str):
        if name in _stage_t0:
            stage_timings[name] = round(time.time() - _stage_t0.pop(name), 2)
            progress(name, f"completed in {stage_timings[name]:.1f}s")

    # ── Paths ──────────────────────────────────────────────────────────────
    artifacts_root = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    account_dir = artifacts_root / account_id

    db = Path(db_path) if db_path else DATA_DIR / f"{account_id}.duckdb"
    lance = Path(lance_path) if lance_path else DATA_DIR / f"{account_id}.lance"

    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = account_dir / run_id

    store.start_run(run_id, "init")
    progress("init", f"account={account_id} cuML={cuml_available()} device={device}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Parse PDFs (with caching)
    # ══════════════════════════════════════════════════════════════════════
    stage_start("parse")

    # Check for previously parsed results first
    cached = load_parsed_agreements()
    pdf_dir_resolved = str(Path(pdf_dir).resolve())

    # Filter cached results to those from the same source directory
    manifest_path = PARSED_TEXT_DIR / "manifest.json"
    cache_hit = False
    if cached and manifest_path.exists():
        import json as _json
        with open(manifest_path) as _f:
            _manifest = _json.load(_f)
        if str(Path(_manifest.get("source_dir", "")).resolve()) == pdf_dir_resolved:
            cache_hit = True

    if cache_hit:
        progress("parse", f"Using {len(cached)} cached parsed PDFs from {PARSED_TEXT_DIR}")
        parsed_docs = cached
    else:
        progress("parse", f"Parsing PDFs from {pdf_dir} (no cache found)...")
        parsed_docs = parse_pdf_directory(pdf_dir)

    if max_docs and len(parsed_docs) > max_docs:
        parsed_docs = parsed_docs[:max_docs]
        progress("parse", f"Limited to {max_docs} documents")

    if not parsed_docs:
        progress("error", "No PDFs parsed")
        return {"store": store, "lance_store": lance_store, "run_id": run_id,
                "stats": {}, "n_documents": 0}

    agreements = [
        {"agreement_id": d["agreement_id"], "filename": d["filename"], "raw_text": d["raw_text"]}
        for d in parsed_docs
        if len(d.get("raw_text", "")) > 100  # skip near-empty PDFs
    ]
    total_chars = sum(len(a["raw_text"]) for a in agreements)
    progress("parse", f"{len(agreements)} valid PDFs ({total_chars:,} chars)")
    stage_end("parse")
    store.update_run(run_id, "parse_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Chunk
    # ══════════════════════════════════════════════════════════════════════
    stage_start("chunk")
    embedder = NomicEmbedder.get_instance(device=device)
    embedder.warmup()  # Pre-warm CUDA kernels on first run

    if use_semantic_chunking:
        progress("chunk", "Semantic chunking (clause-aware)...")
        chunker_config = SemanticChunkerConfig(
            buffer_size=1, breakpoint_percentile=90.0,
            min_chunk_chars=100, max_chunk_chars=2000,
            clause_split=True,
        )
        MAX_CHUNKS_PER_DOC = 80  # Truncate to avoid embedding explosion on huge docs

        from core.chunker import _structural_clause_split, _split_sentences, _combine_sentence_buffers, _compute_breakpoints, _assemble_chunks, _approx_token_count
        import uuid as _uuid

        # ── Phase 1: structural clause split (CPU, fast) ───────────────────
        # For each doc, split into structural sections (clauses).
        # Sections that fit within max_chunk_chars become chunks directly.
        # Oversized sections need semantic sub-splitting (requires embeddings).
        doc_clauses = []        # list of (agr, clauses: list[str])
        oversized_segments = [] # (doc_idx, clause_idx, text) for segments needing semantic split
        os_sentence_data = []   # (sentences, buffers) per oversized segment
        all_buffers = []        # flat buffer list for single batched embed
        buffer_offsets = []     # (start, end) per oversized segment into all_buffers

        for agr in agreements:
            raw_text = agr["raw_text"]
            sections = _structural_clause_split(raw_text, min_clause_chars=chunker_config.min_chunk_chars)
            if not sections:
                sections = [raw_text.strip()] if raw_text.strip() else []
            doc_clauses.append((agr, sections))

            for ci, section in enumerate(sections):
                if len(section) > chunker_config.max_chunk_chars:
                    # Needs semantic sub-splitting — prepare buffers
                    sentences = _split_sentences(section)
                    if sentences and len(sentences) > 2:
                        buffers = _combine_sentence_buffers(sentences, chunker_config.buffer_size)
                        os_idx = len(oversized_segments)
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences, buffers))
                        buf_start = len(all_buffers)
                        all_buffers.extend(buffers)
                        buffer_offsets.append((buf_start, len(all_buffers)))
                    else:
                        # Too few sentences to sub-split, will be force-split later
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences or [], []))
                        buffer_offsets.append((len(all_buffers), len(all_buffers)))

        n_direct = sum(
            sum(1 for s in sections if len(s) <= chunker_config.max_chunk_chars)
            for _, sections in doc_clauses
        )
        progress("chunk", f"Structural split: {sum(len(s) for _, s in doc_clauses)} sections "
                 f"({n_direct} direct, {len(oversized_segments)} need sub-splitting, "
                 f"{len(all_buffers)} buffers to embed)")

        # ── Phase 2: single batched embed for oversized segments ───────────
        # Truncate buffers to 512 chars — only need rough similarity for breakpoints
        truncated_buffers = [b[:512] for b in all_buffers]
        if truncated_buffers:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_buffer_embeddings = embedder.embed_for_clustering(
                truncated_buffers, show_progress=True, batch_size=8,
            )
        else:
            all_buffer_embeddings = np.empty((0, embedder.dim), dtype=np.float32)

        # Pre-compute semantic sub-chunks for oversized segments
        os_sub_chunks: dict[tuple[int, int], list[str]] = {}  # (doc_idx, clause_idx) -> chunk_texts
        for os_i, (doc_idx, clause_idx, section_text) in enumerate(oversized_segments):
            sentences, buffers = os_sentence_data[os_i]
            buf_start, buf_end = buffer_offsets[os_i]

            if buf_start < buf_end and sentences and len(sentences) > 2:
                doc_embeddings = all_buffer_embeddings[buf_start:buf_end]
                # Use a sub-config without clause_split to avoid re-splitting
                sub_cfg = SemanticChunkerConfig(
                    buffer_size=chunker_config.buffer_size,
                    breakpoint_percentile=chunker_config.breakpoint_percentile,
                    min_chunk_chars=chunker_config.min_chunk_chars,
                    max_chunk_chars=chunker_config.max_chunk_chars,
                    neighbor_weight=chunker_config.neighbor_weight,
                )
                breakpoints = _compute_breakpoints(doc_embeddings, sentences, sub_cfg)
                sub_texts = _assemble_chunks(sentences, breakpoints, sub_cfg)
            else:
                # Force-split by character limit
                from core.chunker import _force_split
                sub_texts = _force_split(sentences, chunker_config.max_chunk_chars) if sentences else [section_text]

            os_sub_chunks[(doc_idx, clause_idx)] = sub_texts

        # ── Phase 3: assemble final chunks per doc WITH clause metadata ────
        all_chunks = []
        all_clauses = []   # {clause_id, agreement_id, clause_number, clause_title, heading_level, clause_index, full_text, chunk_ids}
        for doc_idx, (agr, sections) in enumerate(doc_clauses):
            if (doc_idx + 1) % 200 == 0:
                progress("chunk", f"  {doc_idx+1}/{len(agreements)} documents assembled")
            chunk_idx = 0
            for ci, section in enumerate(sections):
                if (doc_idx, ci) in os_sub_chunks:
                    # Oversized: use pre-computed sub-chunks
                    sub_texts = os_sub_chunks[(doc_idx, ci)]
                else:
                    # Fits directly as one chunk
                    sub_texts = [section]

                # Extract clause metadata from the section header
                header = extract_clause_header(section)
                clause_id = f"cls_{uuid.uuid4().hex[:10]}"
                clause_rec = {
                    "clause_id": clause_id,
                    "agreement_id": agr["agreement_id"],
                    "clause_number": header["clause_number"] if header else "",
                    "clause_title": header["clause_title"] if header else "",
                    "heading_level": header["heading_level"] if header else 0,
                    "clause_index": ci,
                    "full_text": section,
                    "token_count": _approx_token_count(section),
                    "chunk_ids": [],
                }

                for ct in sub_texts:
                    if chunk_idx >= MAX_CHUNKS_PER_DOC:
                        break
                    chk_id = f"chk_{_uuid.uuid4().hex[:10]}"
                    all_chunks.append({
                        "chunk_id": chk_id,
                        "text": ct, "char_start": 0, "char_end": len(ct),
                        "token_count": _approx_token_count(ct), "chunk_index": chunk_idx,
                        "agreement_id": agr["agreement_id"],
                        "clause_id": clause_id,
                    })
                    clause_rec["chunk_ids"].append(chk_id)
                    chunk_idx += 1

                all_clauses.append(clause_rec)
    else:
        progress("chunk", "Sliding window chunking (fast mode)...")
        all_chunks = []
        all_clauses = []
        for agr in agreements:
            chunks = sliding_window_chunk(agr["raw_text"])
            for c in chunks:
                c["agreement_id"] = agr["agreement_id"]
            all_chunks.extend(chunks)

    progress("chunk", f"{len(all_chunks)} chunks from {len(agreements)} docs")
    stage_end("chunk")

    # Store agreements in DuckDB
    for agr in agreements:
        summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
        store.insert_agreement(agr["agreement_id"], agr.get("filename", ""), agr["raw_text"], summary)

    store.update_run(run_id, "chunk_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Embed
    # ══════════════════════════════════════════════════════════════════════
    stage_start("embed")
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    chunk_agr_ids = [c["agreement_id"] for c in all_chunks]

    # Build summaries first (cheap) so we can check cache with correct counts
    from core.macro_clusterer import get_field_summaries

    # Try loading cached embeddings from a previous run
    cached_chunk_emb, cached_summary_emb = _load_cached_embeddings(
        account_dir, expected_chunks=len(chunk_texts), expected_summaries=len(agreements),
    )

    _embed_fn = lambda texts, **kw: embedder.embed_for_clustering(texts, batch_size=8, **kw)

    if cached_chunk_emb is not None and not use_llm_summaries:
        # Can reuse cached chunk embeddings; but if LLM summaries are requested,
        # we need fresh summaries (and therefore fresh summary embeddings).
        if cached_summary_emb is not None:
            progress("embed", f"Using cached embeddings ({cached_chunk_emb.shape[0]} chunks, {cached_summary_emb.shape[0]} summaries)")
            chunk_embeddings = cached_chunk_emb
            summary_embeddings = cached_summary_emb
            summaries = get_field_summaries(
                agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn,
            )
        else:
            chunk_embeddings = cached_chunk_emb
            summaries = get_field_summaries(
                agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn,
            )
            summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)
    else:
        progress("embed", f"Embedding {len(all_chunks)} chunks on {device}...")
        chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True, batch_size=8)

        # Free VRAM before summary embedding
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_llm_summaries:
            # LLM scenario summaries: use field context → Azure OpenAI
            progress("embed", "Generating LLM scenario summaries...")
            from core.llm_summarizer import generate_llm_summaries
            summaries = generate_llm_summaries(
                agreements, all_chunks, chunk_embeddings,
                embed_fn=_embed_fn,
                progress_fn=progress,
            )
        else:
            progress("embed", "Building field-aware summaries...")
            summaries = get_field_summaries(
                agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn,
            )
        summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)

    progress("embed", f"Embedded: {chunk_embeddings.shape[0]} chunks + {summary_embeddings.shape[0]} summaries ({embedder.dim}-dim)")
    stage_end("embed")

    # Persist chunks to DuckDB (batch)
    stage_start("persist_db")
    progress("persist", "Writing chunks to DuckDB...")
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
            clause_id=chunk.get("clause_id"),
        )

    # Persist clauses
    for clause in all_clauses:
        store.insert_clause(
            clause["clause_id"], clause["agreement_id"],
            clause.get("clause_number", ""), clause.get("clause_title", ""),
            clause.get("heading_level", 0), clause.get("clause_index", 0),
            clause.get("full_text", ""), clause.get("token_count", 0),
        )

    stage_end("persist_db")

    # Persist to LanceDB
    stage_start("persist_lance")
    progress("persist", "Writing vectors to LanceDB...")
    lance_store.upsert_chunks(
        chunk_ids=chunk_ids, vectors=chunk_embeddings,
        texts=chunk_texts, agreement_ids=chunk_agr_ids,
    )
    lance_store.upsert_documents(
        agreement_ids=[a["agreement_id"] for a in agreements],
        vectors=summary_embeddings,
        filenames=[a.get("filename", "") for a in agreements],
        summaries=summaries,
    )
    stage_end("persist_lance")
    store.update_run(run_id, "embed_complete")

    # Save embeddings
    _save_embeddings(run_dir, chunk_embeddings, summary_embeddings)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Clause-level embedding + Clause-type clustering
    # ══════════════════════════════════════════════════════════════════════
    #
    # Architecture: skip the macro → micro two-stage approach entirely.
    # Instead, compute one embedding per clause (mean of its chunk embeddings),
    # then cluster ALL clauses in a single flat pass.  Each resulting cluster
    # is a "clause type" (e.g. Indemnification, Termination, Payment Terms).
    #
    # This finds common clauses *across* document types — an "Indemnification"
    # clause in a lease and one in a service agreement end up in the same
    # clause type, which is what users actually want.
    # ══════════════════════════════════════════════════════════════════════
    stage_start("clause_cluster")
    _calc_probs = cuml_available()

    # ── 4a: Compute clause-level embeddings (mean of constituent chunks) ──
    # Build a mapping from clause_id → indices into chunk_embeddings
    clause_id_to_chunk_indices: dict[str, list[int]] = {}
    for i, c in enumerate(all_chunks):
        cid = c.get("clause_id")
        if cid:
            clause_id_to_chunk_indices.setdefault(cid, []).append(i)

    # Filter to clauses that actually have chunks
    valid_clauses = [cl for cl in all_clauses if cl["clause_id"] in clause_id_to_chunk_indices]
    n_clauses = len(valid_clauses)
    progress("clause_cluster", f"Computing clause embeddings for {n_clauses} clauses...")

    clause_ids_ordered = [cl["clause_id"] for cl in valid_clauses]
    clause_embeddings = np.zeros((n_clauses, chunk_embeddings.shape[1]), dtype=np.float32)
    for ci, cl in enumerate(valid_clauses):
        idx = clause_id_to_chunk_indices[cl["clause_id"]]
        emb = chunk_embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(emb)
        clause_embeddings[ci] = emb / norm if norm > 0 else emb

    # Also build clause text representations (for BERTopic's c-TF-IDF)
    clause_texts = []
    for cl in valid_clauses:
        # Use title + first 1500 chars of full_text for representation
        title = cl.get("clause_title", "")
        body = cl.get("full_text", "")[:1500]
        clause_texts.append(f"{title}\n{body}" if title else body)

    progress("clause_cluster", f"Clause embeddings: {clause_embeddings.shape}")

    # ── 4b: Single-pass HDBSCAN clustering of all clauses ──
    if n_clauses < 10:
        # Too few clauses — put all in one cluster
        progress("clause_cluster", f"Small corpus ({n_clauses} clauses) — single clause type")
        clause_topics = [0] * n_clauses
        clause_model = None
    else:
        # Adaptive params based on corpus size
        _n_neighbors = min(15, n_clauses - 1)
        _n_components = max(2, min(10, n_clauses - 2))
        _min_cluster = max(3, n_clauses // 50)  # ~2% of clauses minimum
        _min_samples = 2

        progress("clause_cluster", f"Clustering {n_clauses} clauses "
                 f"(n_neighbors={_n_neighbors}, min_cluster={_min_cluster})...")

        clause_umap = make_umap(
            n_neighbors=_n_neighbors, n_components=_n_components,
            min_dist=0.0, metric="cosine",
        )
        clause_hdbscan = make_hdbscan(
            min_cluster_size=_min_cluster, min_samples=_min_samples,
            metric="euclidean", cluster_selection_method="eom",
        )
        clause_analyzer = StopwordFilteringAnalyzer(
            ngram_range=(1, 4), stopwords=COMBINED_STOPWORDS,
        )
        clause_vectorizer = CountVectorizer(
            analyzer=clause_analyzer, max_features=100000, min_df=1,
        )
        clause_ctfidf = ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True,
        )
        clause_model = BERTopic(
            umap_model=clause_umap, hdbscan_model=clause_hdbscan,
            vectorizer_model=clause_vectorizer, ctfidf_model=clause_ctfidf,
            calculate_probabilities=_calc_probs, verbose=False,
        )
        clause_topics_raw, _ = clause_model.fit_transform(clause_texts, clause_embeddings)

        # Outlier reduction — assign outlier clauses to nearest cluster
        if -1 in clause_topics_raw:
            try:
                clause_topics_reduced = clause_model.reduce_outliers(
                    clause_texts, clause_topics_raw,
                    strategy="c-tf-idf", threshold=0.2,
                )
                clause_model.update_topics(
                    clause_texts, topics=clause_topics_reduced,
                    vectorizer_model=clause_model.vectorizer_model,
                    representation_model=None,
                )
                clause_topics = clause_topics_reduced
            except Exception:
                clause_topics = clause_topics_raw
        else:
            clause_topics = clause_topics_raw

        # Save clause model
        _save_model(clause_model, run_dir, "clause_model")

    # ── 4c: Build clause-type records ──
    unique_clause_types = sorted(set(t for t in clause_topics if t != -1))
    quality = compute_cluster_quality(clause_embeddings, clause_topics)
    progress("clause_cluster", f"Found {len(unique_clause_types)} clause types "
             f"({sum(1 for t in clause_topics if t == -1)} outlier clauses)")

    # LLM-label clause types
    llm_clause_labels = {}
    if clause_model:
        llm_clause_labels = _llm_label_micro_clusters(
            clause_model, "all documents", clause_texts, clause_topics,
        )

    cluster_records = []
    all_domain_id = f"d_{uuid.uuid4().hex[:8]}"  # single domain for all clauses
    for tid in unique_clause_types:
        mask = [i for i, t in enumerate(clause_topics) if t == tid]
        c_embs = clause_embeddings[mask]
        centroid = c_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else centroid

        keywords = []
        if clause_model:
            try:
                keywords = [w for w, _ in clause_model.get_topic(tid)][:15]
            except Exception:
                pass

        if tid in llm_clause_labels and llm_clause_labels[tid].get("label"):
            info = llm_clause_labels[tid]
        else:
            info = _auto_label_topic(clause_model, tid) if clause_model else {
                "label": f"Clause_Type_{tid}", "description": ""
            }

        cluster_id = f"ct_{uuid.uuid4().hex[:8]}"

        # Collect chunk_ids for all clauses in this type
        type_chunk_ids = []
        type_clause_ids = []
        type_agr_ids = set()
        for ci in mask:
            cl = valid_clauses[ci]
            type_clause_ids.append(cl["clause_id"])
            type_chunk_ids.extend(cl.get("chunk_ids", []))
            type_agr_ids.add(cl["agreement_id"])

        cluster_records.append({
            "cluster_id": cluster_id, "topic_id": tid,
            "domain_id": all_domain_id, "label": info["label"],
            "description": info.get("description", ""),
            "keywords": keywords,
            "chunk_count": len(type_chunk_ids),
            "clause_count": len(mask),
            "agreement_count": len(type_agr_ids),
            "quality_score": quality.get(tid, 0.0),
            "centroid": centroid,
            "chunk_ids": type_chunk_ids,
            "clause_ids": type_clause_ids,
        })

    # ── Merge near-duplicate clause types (complete-linkage, conservative) ──
    if len(cluster_records) > 1:
        merger = HierarchicalTopicMerger(merge_threshold=0.93)
        pre_merge = len(cluster_records)
        cluster_records = merger.merge_similar_topics(cluster_records)
        if len(cluster_records) < pre_merge:
            progress("clause_cluster", f"Merged {pre_merge} → {len(cluster_records)} clause types")

    total_clusters = len(cluster_records)
    n_domains = 1  # single domain in clause-first mode
    n_outliers = 0

    # ── 4d: Persist clause types (as clusters) + link clauses ──
    # Create one "domain" to hold all clause types
    store.upsert_domain(all_domain_id, "All Documents", "Clause-type clustered corpus",
                        len(agreements), 1.0)
    for agr in agreements:
        store.update_agreement_domain(agr["agreement_id"], all_domain_id, 1.0)

    clause_id_to_cluster: dict[str, str] = {}
    for cr in cluster_records:
        store.upsert_cluster(
            cr["cluster_id"], all_domain_id, cr["label"], cr.get("description", ""),
            cr.get("keywords", []), cr["chunk_count"], cr.get("agreement_count", 0),
            cr.get("quality_score"), cr.get("centroid"),
            cr.get("aliases", []), cr.get("merged_from", []),
        )
        if cr.get("centroid") is not None:
            store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])
        # Assign chunks to cluster
        for cid in cr.get("chunk_ids", []):
            store.insert_cluster_assignment(cid, cr["cluster_id"])
        # Link clauses to clause type
        for cls_id in cr.get("clause_ids", []):
            store.update_clause_type(cls_id, cr["cluster_id"])
            clause_id_to_cluster[cls_id] = cr["cluster_id"]

    # Also assign outlier clauses to nearest clause type
    outlier_clause_indices = [i for i, t in enumerate(clause_topics) if t == -1]
    if outlier_clause_indices and cluster_records:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        type_centroids = np.array([cr["centroid"] for cr in cluster_records])
        for oi in outlier_clause_indices:
            cl = valid_clauses[oi]
            sims = cos_sim(clause_embeddings[oi:oi+1], type_centroids)[0]
            best = int(np.argmax(sims))
            best_cr = cluster_records[best]
            for cid in cl.get("chunk_ids", []):
                store.insert_cluster_assignment(cid, best_cr["cluster_id"])
            store.update_clause_type(cl["clause_id"], best_cr["cluster_id"])
            clause_id_to_cluster[cl["clause_id"]] = best_cr["cluster_id"]
        progress("clause_cluster", f"Assigned {len(outlier_clause_indices)} outlier clauses to nearest type")

    # Update LanceDB
    lance_cids_all, lance_dids_all, lance_clids_all = [], [], []
    for cr in cluster_records:
        for cid in cr.get("chunk_ids", []):
            lance_cids_all.append(cid)
            lance_dids_all.append(all_domain_id)
            lance_clids_all.append(cr.get("topic_id", -1))
    if lance_cids_all:
        lance_store.update_cluster_assignments(lance_cids_all, lance_dids_all, lance_clids_all)

    progress("clause_cluster", f"{total_clusters} clause types from {n_clauses} clauses across {len(agreements)} docs")

    # ── Quality report ──
    quality_report: dict = {"domains": {}, "global": {}}
    all_quality_scores = [q for q in quality.values() if q > 0]
    n_outlier_clauses = sum(1 for t in clause_topics if t == -1)
    quality_report["global"] = {
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_clauses": n_clauses,
        "n_clause_types": total_clusters,
        "n_outlier_clauses": n_outlier_clauses,
        "clause_outlier_ratio": round(n_outlier_clauses / n_clauses, 4) if n_clauses > 0 else 0,
        "avg_quality": round(float(np.mean(all_quality_scores)), 4) if all_quality_scores else 0,
        "use_semantic_chunking": use_semantic_chunking,
        "device": device,
        "cuml_available": cuml_available(),
    }
    quality_report["clause_types"] = [
        {
            "cluster_id": cr["cluster_id"],
            "label": cr["label"],
            "clause_count": cr.get("clause_count", 0),
            "chunk_count": cr["chunk_count"],
            "agreement_count": cr.get("agreement_count", 0),
            "quality_score": round(cr.get("quality_score", 0), 4),
            "keywords": cr.get("keywords", [])[:10],
        }
        for cr in cluster_records
    ]
    _save_quality_report(run_dir, quality_report)
    progress("clause_cluster", f"Quality: avg={quality_report['global']['avg_quality']:.4f}")

    stage_end("clause_cluster")
    store.update_run(run_id, "clause_cluster_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 6: Field Discovery (Azure OpenAI)
    # ══════════════════════════════════════════════════════════════════════
    field_discovery_result = {"total_fields": 0, "total_extractions": 0}
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        stage_start("field_discovery")
        progress("field_discovery", "Discovering fields via Azure OpenAI...")
        try:
            from core.field_discovery import run_field_discovery, FieldDiscoveryConfig
            fd_config = FieldDiscoveryConfig(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
            )
            field_discovery_result = run_field_discovery(
                store=store, config=fd_config,
                progress_fn=progress, extract_values=True,
                min_cluster_chunks=5,
                max_extraction_workers=5,
                min_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Field discovery failed: {e}")
            progress("field_discovery", f"Failed: {e}")
        stage_end("field_discovery")

        # ── Universal field dedup: identify cross-cluster common fields ──
        progress("field_dedup", "Identifying universal fields...")
        try:
            _dedup_universal_fields(store, progress)
        except Exception as e:
            logger.warning(f"Field dedup failed (non-fatal): {e}")
    else:
        progress("field_discovery", "Skipped (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set)")

    store.update_run(run_id, "completed", "completed")

    # ══════════════════════════════════════════════════════════════════════
    # Export fields.json, chunks.json, and clause_types.json
    # ══════════════════════════════════════════════════════════════════════
    progress("export", "Exporting fields, chunks, and clause types JSON...")
    _export_fields_json(store, run_dir)
    _export_chunks_json(store, run_dir)
    _export_clause_types_json(store, run_dir, progress)

    # ══════════════════════════════════════════════════════════════════════
    # Save manifest
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    stats = store.get_stats()
    manifest = {
        "run_id": run_id,
        "account_id": account_id,
        "pdf_dir": str(pdf_dir),
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_clauses": len(all_clauses) if use_semantic_chunking else 0,
        "n_clause_types": total_clusters,
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "n_outlier_docs": n_outliers,
        "embedding_dim": int(embedder.dim),
        "device": device,
        "cuml_available": cuml_available(),
        "use_semantic_chunking": use_semantic_chunking,
        "elapsed_seconds": round(elapsed, 1),
        "stage_timings": stage_timings,
        "db_path": str(db),
        "lance_path": str(lance),
        "artifacts_dir": str(run_dir),
        "clause_model_path": str(run_dir / "clause_model"),
        "embeddings_path": str(run_dir / "embeddings.npz"),
        "quality_report_path": str(run_dir / "cluster_quality.json"),
        "clustering_results_path": str(run_dir / "clustering_results.json"),
        "fields_path": str(run_dir / "fields.json"),
        "chunks_path": str(run_dir / "chunks.json"),
        "clause_types_path": str(run_dir / "clause_types.json"),
        "field_discovery": field_discovery_result,
        "stats": stats,
    }
    _save_manifest(run_dir, manifest)

    progress("done", f"Complete in {elapsed:.0f}s — {len(agreements)} docs, "
             f"{len(all_chunks)} chunks, {len(all_clauses) if use_semantic_chunking else 0} clauses, "
             f"{total_clusters} clause types")
    progress("done", f"Artifacts: {run_dir}")

    return {
        "store": store,
        "lance_store": lance_store,
        "run_id": run_id,
        "account_id": account_id,
        "stats": stats,
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_clauses": len(all_clauses) if use_semantic_chunking else 0,
        "n_clause_types": total_clusters,
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "elapsed_seconds": round(elapsed, 1),
        "artifacts_dir": str(run_dir),
        "manifest": manifest,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Run clause-type clustering pipeline on PDFs")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--account-id", default="default", help="Account/tenant ID")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--fast-chunk", action="store_true", help="Use sliding window instead of semantic chunking")
    parser.add_argument("--llm-summaries", action="store_true",
                        help="Use Azure OpenAI to generate scenario summaries for macro clustering")
    args = parser.parse_args()

    result = run_clause_pipeline(
        pdf_dir=args.pdf_dir,
        account_id=args.account_id,
        max_docs=args.max_docs,
        device=args.device,
        use_semantic_chunking=not args.fast_chunk,
        use_llm_summaries=args.llm_summaries,
    )

    print(f"\n{'='*60}")
    print(f"  Documents:    {result['n_documents']}")
    print(f"  Chunks:       {result['n_chunks']}")
    print(f"  Clauses:      {result.get('n_clauses', 0)}")
    print(f"  Clause Types: {result.get('n_clause_types', 0)}")
    print(f"  Time:         {result['elapsed_seconds']}s")
    print(f"  Artifacts:    {result['artifacts_dir']}")
    print(f"{'='*60}")
