"""Cluster-hyperparameter advisor.

Asks an LLM (Azure OpenAI if configured, else whatever LLM_BACKEND points at)
to recommend hyperparameters for EVoC + downstream thresholds based on a
compact summary of the corpus. The output is purely advisory — the pipeline
can use it as Optuna seeds, as floor/ceiling constraints, or simply log it
for transparency.

Schema returned by `advise()`:

    {
      "n_clusters_target": 12,              # ideal cluster count for this size
      "min_cluster_size": 3,                # EVoC base_min_cluster_size
      "min_samples": 1,                     # HDBSCAN/EVoC min_samples
      "noise_level": 0.30,                  # EVoC noise_level
      "min_cluster_chunks": 3,              # field-discovery floor
      "coverage_threshold": 0.70,           # playbook miner coverage floor
      "rationale": "...",                   # one-paragraph why
      "source": "azure_openai" | "gemini" | "deterministic_fallback"
    }

The function is safe to call without any LLM creds: it falls through to a
deterministic heuristic when no backend responds.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You are a clustering hyperparameter advisor for an embedding-based legal-clause
discovery pipeline. The pipeline uses EVoC (a density-aware embedding clustering
algorithm similar to HDBSCAN) to cluster clause-level chunks from legal contracts.

Given the corpus profile below, recommend hyperparameters that will produce a
SMALL number of WELL-FORMED clusters (one per canonical legal clause type) rather
than many sparse ones. Tiny corpora (<10 docs) should produce roughly 6-12
clusters; large corpora (>100 docs) can produce 20-40.

Corpus profile:
- n_documents:       {n_documents}
- n_clauses:         {n_clauses}
- n_chunks:          {n_chunks}
- avg_chunks_per_doc: {avg_chunks_per_doc:.1f}
- avg_clauses_per_doc: {avg_clauses_per_doc:.1f}
- known clause types in source samples: {clause_hints}

Return ONLY a JSON object with these exact keys:
  n_clusters_target  (int)          : ideal final cluster count
  min_cluster_size   (int, >=2)     : EVoC base_min_cluster_size
  min_samples        (int, >=1)     : HDBSCAN/EVoC min_samples
  noise_level        (float 0-1)    : EVoC noise_level (0 = cluster aggressively,
                                       1 = pure clusters only)
  min_cluster_chunks (int, >=2)     : minimum chunks a cluster needs before
                                       field discovery is attempted
  coverage_threshold (float 0-1)    : fraction of agreements a cluster must
                                       appear in before the miner emits a
                                       coverage rule
  rationale          (string)       : one paragraph (<=350 chars) explaining
                                       the rec.

Use these heuristic anchors as guidance (NOT hard rules):
  N=5    docs → target 8-12 clusters, min_cluster_chunks 2
  N=20   docs → target 12-18 clusters, min_cluster_chunks 3
  N=100+ docs → target 20-30 clusters, min_cluster_chunks 5
"""


def advise(corpus_profile: dict) -> dict:
    """Compute or fetch hyperparameter recommendations for this corpus.

    Args:
      corpus_profile: dict with keys n_documents, n_clauses, n_chunks,
        avg_chunks_per_doc, avg_clauses_per_doc, clause_hints (str, optional).

    Returns:
      dict matching the schema in this module's docstring. Always succeeds —
      LLM failure falls through to a deterministic heuristic.
    """
    # Try the LLM path first.
    via_llm = _llm_advise(corpus_profile)
    if via_llm is not None:
        return via_llm
    return _deterministic_advise(corpus_profile)


def _deterministic_advise(p: dict) -> dict:
    """Honest heuristic fallback when no LLM responds. Mirrors the anchors
    in the LLM prompt so behavior is consistent across both paths."""
    n = max(1, int(p.get("n_documents", 1)))
    if n <= 5:
        target, min_size, min_chunks, cov = 10, 2, 2, 0.60
    elif n <= 10:
        target, min_size, min_chunks, cov = 12, 2, 2, 0.65
    elif n <= 30:
        target, min_size, min_chunks, cov = 16, 3, 3, 0.70
    elif n <= 100:
        target, min_size, min_chunks, cov = 22, 4, 4, 0.80
    else:
        target, min_size, min_chunks, cov = 30, 5, 5, 0.90
    return {
        "n_clusters_target":  target,
        "min_cluster_size":   min_size,
        "min_samples":        1,
        "noise_level":        0.30,
        "min_cluster_chunks": min_chunks,
        "coverage_threshold": cov,
        "rationale": (
            f"Deterministic heuristic for n_documents={n}. "
            f"Targets ~{target} canonical clusters; "
            f"field discovery enabled at chunk_count>={min_chunks}; "
            f"coverage threshold {int(cov * 100)}% scales with corpus size."
        ),
        "source": "deterministic_fallback",
    }


def _llm_advise(p: dict) -> Optional[dict]:
    """Single LLM call with the corpus profile. Returns None on any failure
    so the caller falls through to the deterministic path."""
    backend = (os.getenv("LLM_BACKEND") or "").lower()
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    # Determine which path: Azure if available, then whatever LLMClient picks.
    if not (az_endpoint and az_key) and backend not in {"openai", "ollama", "gemini", "litellm"}:
        logger.info("cluster_advisor: no LLM backend configured; using deterministic")
        return None

    prompt = _PROMPT_TEMPLATE.format(
        n_documents=p.get("n_documents", 0),
        n_clauses=p.get("n_clauses", 0),
        n_chunks=p.get("n_chunks", 0),
        avg_chunks_per_doc=float(p.get("avg_chunks_per_doc", 0.0)),
        avg_clauses_per_doc=float(p.get("avg_clauses_per_doc", 0.0)),
        clause_hints=p.get("clause_hints", "(none)"),
    )
    try:
        from core.llm_client import LLMClient
        client = LLMClient()
        raw = client.complete(
            prompt,
            system="You are a precise legal-tech clustering advisor. "
                    "Return only valid JSON matching the requested schema.",
            temperature=0.1,
            max_tokens=400,
            json_mode=True,
        )
    except Exception as e:
        logger.warning(f"cluster_advisor LLM call failed: {e}")
        return None

    try:
        data = json.loads(raw)
    except Exception:
        # Handle stray markdown fences
        s = raw.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1]
            if s.startswith("json\n"):
                s = s[5:]
            s = s.split("```", 1)[0].strip()
        try:
            data = json.loads(s)
        except Exception as e2:
            logger.warning(f"cluster_advisor JSON parse failed: {e2}; raw={raw[:200]!r}")
            return None

    # Coerce + bounds-check
    out = {
        "n_clusters_target":  _clamp_int(data.get("n_clusters_target"),  4, 60),
        "min_cluster_size":   _clamp_int(data.get("min_cluster_size"),    2, 50),
        "min_samples":        _clamp_int(data.get("min_samples"),         1, 10),
        "noise_level":        _clamp_float(data.get("noise_level"),       0.0, 1.0),
        "min_cluster_chunks": _clamp_int(data.get("min_cluster_chunks"),  2, 20),
        "coverage_threshold": _clamp_float(data.get("coverage_threshold"), 0.30, 1.0),
        "rationale":          (str(data.get("rationale", "")) or "")[:500],
        "source":             "azure_openai" if (az_endpoint and az_key) else f"llm:{backend or 'auto'}",
    }
    return out


def _clamp_int(v, lo, hi) -> int:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        x = lo
    return max(lo, min(hi, x))


def _clamp_float(v, lo, hi) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = lo
    return max(lo, min(hi, x))


def summarize_corpus_from_store(clustering_store) -> dict:
    """Build a corpus-profile dict from the existing DuckDB schema."""
    conn = clustering_store.conn
    try:
        n_documents = conn.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
        n_chunks    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except Exception:
        n_documents, n_chunks = 0, 0
    try:
        n_clauses = conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0]
    except Exception:
        n_clauses = 0
    avg_chunks = (n_chunks / n_documents) if n_documents else 0.0
    avg_clauses = (n_clauses / n_documents) if n_documents else 0.0
    return {
        "n_documents":         int(n_documents),
        "n_chunks":            int(n_chunks),
        "n_clauses":           int(n_clauses),
        "avg_chunks_per_doc":  avg_chunks,
        "avg_clauses_per_doc": avg_clauses,
        "clause_hints":        "Consulting agreements; expect Confidentiality, "
                                "Indemnification, Governing Law, Compensation, "
                                "Termination, Limitation of Liability, "
                                "Non-Compete, Insurance.",
    }
