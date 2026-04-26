"""
RAG Extractor — Retrieval-Augmented field extraction with agentic retry.
=========================================================================

Post-discovery extraction pipeline that replaces (or supplements) the naive
per-chunk LLM extraction with a multi-round RAG loop:

  Stage Architecture:
  ─────────────────────────────────────────────────────────────────────
  1. Cochran Sampling   — statistically sample documents per cluster
  2. Hybrid Retrieval   — semantic + lexical + RRF per (doc, field)
  3. Agentic Retry      — LLM-guided re-query on "insufficient context"
  4. Reviewer Agent      — cheap model accept/reject + re-extract
  5. Model-Routed LLM   — GPT-5.4 complex / GPT-5.4-nano simple
  6. Normalization       — DATE→YYYY/MM/DD, NUMBER preserve symbols
  7. AIDB Writes         — batched persistence via ClusteringStore
  ─────────────────────────────────────────────────────────────────────

Gains: +3% F1 from agentic retry (eval), reviewer catches ~8% of false
positives at < 5% of extraction cost.

Usage:
    from core.rag_extractor import RAGExtractor, RAGConfig

    extractor = RAGExtractor(store=store, llm=llm, embed_fn=embed_fn, config=cfg)
    results = await extractor.extract_cluster(
        cluster_id="cls_001",
        fields=[{"name": "termination_notice_period", "type": "duration", ...}],
        cluster_label="Termination Clauses",
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Fields that require GPT-5.4 (complex reasoning, cross-references)
COMPLEX_FIELD_TYPES = frozenset({
    "duration", "currency", "percentage", "enum",
})
COMPLEX_FIELD_KEYWORDS = frozenset({
    "termination", "renewal", "indemnification", "liability", "penalty",
    "force_majeure", "limitation", "warranty", "non_compete", "exclusivity",
    "assignment", "change_of_control", "liquidated_damages",
})

# Normalization patterns
_DATE_PATTERN = re.compile(
    r"(?P<month>\w+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    r"|(?P<d2>\d{1,2})[/\-.](?P<m2>\d{1,2})[/\-.](?P<y2>\d{2,4})"
    r"|(?P<y3>\d{4})[/\-.](?P<m3>\d{1,2})[/\-.](?P<d3>\d{1,2})"
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class RAGConfig:
    """Tunables for RAG extraction pipeline."""

    # Sampling
    cochran_z: float = 1.96            # 95% confidence
    cochran_p: float = 0.5             # max-variance proportion
    cochran_e: float = 0.15            # 15% margin of error
    max_sample_docs: int = 10          # hard cap on sampled documents
    min_sample_docs: int = 3           # minimum even for tiny clusters

    # Retrieval
    semantic_top_k: int = 10           # pre-RRF semantic candidates
    lexical_top_k: int = 10            # pre-RRF lexical candidates
    rrf_k: int = 60                    # RRF denominator constant
    final_top_k: int = 3              # snippets per (doc, field) after RRF

    # Agentic retry
    max_retry_rounds: int = 3          # max agentic re-query rounds
    insufficient_context_threshold: float = 0.3  # confidence below which we retry

    # NL Index (cross-doc heading lookup)
    enable_nl_index: bool = True
    nl_index_top_k: int = 5            # heading-level candidates

    # Reviewer
    enable_reviewer: bool = True
    reviewer_model: str | None = None  # None → use cheap model

    # Model routing
    complex_model: str = "gpt-5.4"
    simple_model: str = "gpt-5.4-nano"
    model_routing: bool = True         # False → use complex_model for all

    # Extraction
    max_chunk_chars: int = 3000
    min_extraction_confidence: float = 0.4

    # Concurrency
    max_concurrent_fields: int = 8
    max_concurrent_docs: int = 5


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class RetrievedSnippet:
    """A chunk retrieved for a specific (doc, field) pair."""
    chunk_id: str
    chunk_text: str
    agreement_id: str
    score: float
    source: str           # "semantic", "lexical", "rrf", "agentic_retry", "nl_index"
    rank: int


@dataclass
class ExtractionResult:
    """Single field extraction result with provenance."""
    field_name: str
    field_type: str
    value: str | None
    raw_value: str | None  # pre-normalization
    confidence: float
    evidence_span: str
    is_default: bool
    agreement_id: str
    chunk_id: str
    model_used: str
    retrieval_round: int
    reviewer_accepted: bool | None  # None = no reviewer pass
    snippets_used: int


@dataclass
class ExtractionBatch:
    """Batch of extraction results for a cluster."""
    cluster_id: str
    cluster_label: str
    fields: list[dict]
    results: list[ExtractionResult]
    meta: dict


# ── Cochran Sampling ──────────────────────────────────────────────────────────


def cochran_sample_size(
    population: int,
    z: float = 1.96,
    p: float = 0.5,
    e: float = 0.15,
    cap: int = 10,
    floor: int = 3,
) -> int:
    """
    Cochran's formula for sample size with finite population correction.

    n₀ = z²·p·(1-p) / e²
    n  = n₀ / (1 + (n₀-1)/N)
    """
    if population <= floor:
        return population
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / population)
    return max(floor, min(cap, math.ceil(n)))


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize_value(value: str, field_type: str) -> str:
    """Normalize extracted values by field type."""
    if not value or not value.strip():
        return value

    value = value.strip()

    if field_type == "date":
        return _normalize_date(value)
    elif field_type == "boolean":
        return _normalize_boolean(value)
    elif field_type in ("currency", "percentage", "integer"):
        # Preserve symbols — just clean whitespace
        return re.sub(r'\s+', ' ', value).strip()
    elif field_type == "duration":
        return _normalize_duration(value)
    return value


def _normalize_date(value: str) -> str:
    """Attempt to normalize date to YYYY/MM/DD."""
    m = _DATE_PATTERN.search(value)
    if not m:
        return value
    groups = m.groupdict()

    if groups.get("month"):
        month_name = groups["month"].lower()
        month = _MONTH_MAP.get(month_name)
        if not month:
            return value
        day = int(groups["day"])
        year = int(groups["year"])
    elif groups.get("m2"):
        month = int(groups["m2"])
        day = int(groups["d2"])
        year = int(groups["y2"])
        if year < 100:
            year += 2000 if year < 50 else 1900
    elif groups.get("m3"):
        month = int(groups["m3"])
        day = int(groups["d3"])
        year = int(groups["y3"])
    else:
        return value

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return value

    return f"{year:04d}/{month:02d}/{day:02d}"


def _normalize_boolean(value: str) -> str:
    """Normalize boolean values."""
    lower = value.lower().strip()
    if lower in ("yes", "true", "1", "y", "affirmative"):
        return "true"
    if lower in ("no", "false", "0", "n", "negative"):
        return "false"
    return value


def _normalize_duration(value: str) -> str:
    """Normalize duration strings — preserve original but clean whitespace."""
    return re.sub(r'\s+', ' ', value).strip()


# ── Field Complexity Routing ──────────────────────────────────────────────────


def is_complex_field(field_def: dict) -> bool:
    """Determine if a field requires the complex model."""
    ftype = field_def.get("type", "text")
    fname = field_def.get("name", "").lower()

    if ftype in COMPLEX_FIELD_TYPES:
        return True
    for keyword in COMPLEX_FIELD_KEYWORDS:
        if keyword in fname:
            return True

    desc = field_def.get("description", "").lower()
    complexity_signals = ["cross-reference", "conditional", "unless", "subject to",
                          "notwithstanding", "aggregate", "cumulative"]
    if any(sig in desc for sig in complexity_signals):
        return True

    return False


# ── Core RAG Extractor ────────────────────────────────────────────────────────


class RAGExtractor:
    """
    Multi-round RAG extraction pipeline.

    Architecture per (document, field) pair:
      Round 1  → Hybrid retrieval (semantic + lexical + RRF)
      Round 2+ → Agentic retry: LLM reformulates query, re-retrieves
      Round N  → Reviewer agent: cheap model accept/reject
    """

    def __init__(
        self,
        store,                          # ClusteringStore
        llm,                            # AsyncLLMClient
        embed_fn: Callable,             # text → (n, dim) ndarray
        config: RAGConfig | None = None,
        lance_store=None,               # LanceVectorStore (optional)
    ):
        self.store = store
        self.llm = llm
        self.embed_fn = embed_fn
        self.config = config or RAGConfig()
        self.lance = lance_store

        # Pre-built indexes per agreement_id (lazy)
        self._doc_retrievers: dict[str, Any] = {}
        self._nl_index: dict[str, list[dict]] | None = None

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    async def extract_cluster(
        self,
        cluster_id: str,
        fields: list[dict],
        cluster_label: str,
        cluster_keywords: list[str] | None = None,
        context_section: str = "",
    ) -> ExtractionBatch:
        """
        Run full RAG extraction for one cluster.

        1. Sample documents via Cochran formula
        2. For each (doc, field): hybrid retrieve → agentic retry → extract
        3. Reviewer agent pass (optional)
        4. Normalize + persist
        """
        t0 = time.time()
        meta: dict[str, Any] = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "n_fields": len(fields),
            "timings": {},
        }

        # ── 1. Get docs in this cluster and sample ──
        t_sample = time.time()
        chunk_rows = self.store.get_chunks(cluster_id=cluster_id)
        agreement_ids = list({r["agreement_id"] for r in chunk_rows})
        n_sample = cochran_sample_size(
            len(agreement_ids),
            z=self.config.cochran_z,
            p=self.config.cochran_p,
            e=self.config.cochran_e,
            cap=self.config.max_sample_docs,
            floor=self.config.min_sample_docs,
        )
        # Deterministic sample: pick first n (already sorted by store)
        # In production, could stratify by domain_confidence or agreement size
        sampled_ids = agreement_ids[:n_sample]
        meta["n_agreements_total"] = len(agreement_ids)
        meta["n_agreements_sampled"] = len(sampled_ids)
        meta["timings"]["sampling_s"] = round(time.time() - t_sample, 3)

        # Group chunks by agreement
        chunks_by_doc: dict[str, list[dict]] = {}
        for row in chunk_rows:
            if row["agreement_id"] in sampled_ids:
                chunks_by_doc.setdefault(row["agreement_id"], []).append(row)

        # ── 2. Build per-doc retrievers ──
        t_index = time.time()
        await self._build_doc_indexes(chunks_by_doc)
        meta["timings"]["indexing_s"] = round(time.time() - t_index, 3)

        # ── 3. Build NL index (cross-doc heading index) ──
        if self.config.enable_nl_index:
            t_nl = time.time()
            self._build_nl_index(chunks_by_doc)
            meta["timings"]["nl_index_s"] = round(time.time() - t_nl, 3)

        # ── 4. Extract all (doc, field) pairs ──
        t_extract = time.time()
        all_results: list[ExtractionResult] = []

        # Process docs with bounded concurrency
        doc_sem = asyncio.Semaphore(self.config.max_concurrent_docs)

        async def _process_doc(agreement_id: str):
            async with doc_sem:
                return await self._extract_document(
                    agreement_id=agreement_id,
                    fields=fields,
                    cluster_label=cluster_label,
                    context_section=context_section,
                    chunks=chunks_by_doc.get(agreement_id, []),
                )

        doc_tasks = [_process_doc(aid) for aid in sampled_ids]
        doc_results = await asyncio.gather(*doc_tasks, return_exceptions=True)

        total_retries = 0
        total_reviewer_rejections = 0
        for dr in doc_results:
            if isinstance(dr, Exception):
                logger.warning(f"Doc extraction failed: {dr}")
                continue
            results, doc_meta = dr
            all_results.extend(results)
            total_retries += doc_meta.get("retries", 0)
            total_reviewer_rejections += doc_meta.get("reviewer_rejections", 0)

        meta["timings"]["extraction_s"] = round(time.time() - t_extract, 3)
        meta["n_extractions"] = len(all_results)
        meta["total_retries"] = total_retries
        meta["total_reviewer_rejections"] = total_reviewer_rejections

        # ── 5. Persist to store ──
        t_persist = time.time()
        field_id_map = self._get_field_id_map(cluster_id, fields)
        n_persisted = self._persist_results(all_results, field_id_map)
        meta["n_persisted"] = n_persisted
        meta["timings"]["persist_s"] = round(time.time() - t_persist, 3)

        meta["total_time_s"] = round(time.time() - t0, 2)

        return ExtractionBatch(
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            fields=fields,
            results=all_results,
            meta=meta,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PER-DOCUMENT EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    async def _extract_document(
        self,
        agreement_id: str,
        fields: list[dict],
        cluster_label: str,
        context_section: str,
        chunks: list[dict],
    ) -> tuple[list[ExtractionResult], dict]:
        """Extract all fields from one document using multi-round RAG."""
        doc_meta = {"retries": 0, "reviewer_rejections": 0}
        results: list[ExtractionResult] = []

        # Process fields with bounded concurrency
        field_sem = asyncio.Semaphore(self.config.max_concurrent_fields)

        async def _process_field(field_def: dict):
            async with field_sem:
                return await self._extract_field_rag(
                    agreement_id=agreement_id,
                    field_def=field_def,
                    cluster_label=cluster_label,
                    context_section=context_section,
                    chunks=chunks,
                )

        field_tasks = [_process_field(f) for f in fields]
        field_results = await asyncio.gather(*field_tasks, return_exceptions=True)

        for fr in field_results:
            if isinstance(fr, Exception):
                logger.warning(f"Field extraction failed: {fr}")
                continue
            result, retries, reviewer_rejected = fr
            if result is not None:
                results.append(result)
            doc_meta["retries"] += retries
            if reviewer_rejected:
                doc_meta["reviewer_rejections"] += 1

        return results, doc_meta

    # ══════════════════════════════════════════════════════════════════════
    # PER-FIELD RAG EXTRACTION (the core loop)
    # ══════════════════════════════════════════════════════════════════════

    async def _extract_field_rag(
        self,
        agreement_id: str,
        field_def: dict,
        cluster_label: str,
        context_section: str,
        chunks: list[dict],
    ) -> tuple[ExtractionResult | None, int, bool]:
        """
        Multi-round RAG for a single (document, field) pair.

        Round 1: Hybrid retrieval → LLM extract
        Round 2+: If insufficient → agentic re-query
        Final: Reviewer agent accept/reject

        Returns: (result, n_retries, reviewer_rejected)
        """
        field_name = field_def["name"]
        field_type = field_def.get("type", "text")
        retries = 0
        reviewer_rejected = False

        # ── Round 1: Hybrid Retrieval ──
        snippets = await self._hybrid_retrieve(
            agreement_id=agreement_id,
            field_def=field_def,
            chunks=chunks,
        )

        # ── LLM Extraction ──
        result = await self._llm_extract(
            field_def=field_def,
            snippets=snippets,
            cluster_label=cluster_label,
            context_section=context_section,
            agreement_id=agreement_id,
            retrieval_round=1,
        )

        # ── Agentic Retry Loop (Rounds 2+) ──
        round_num = 1
        while (
            round_num < self.config.max_retry_rounds
            and result is not None
            and result.confidence < self.config.insufficient_context_threshold
        ):
            round_num += 1
            retries += 1

            # Ask LLM to generate better search terms
            new_terms = await self._generate_search_terms(
                field_def=field_def,
                cluster_label=cluster_label,
                previous_snippets=snippets,
                previous_result=result,
            )

            if not new_terms:
                break

            # Re-retrieve with new terms
            new_snippets = await self._agentic_retrieve(
                agreement_id=agreement_id,
                field_def=field_def,
                search_terms=new_terms,
                chunks=chunks,
            )

            # Cross-doc NL index lookup
            if self.config.enable_nl_index and self._nl_index:
                nl_snippets = self._nl_index_lookup(
                    field_def=field_def,
                    search_terms=new_terms,
                    agreement_id=agreement_id,
                    chunks=chunks,
                )
                new_snippets.extend(nl_snippets)

            if not new_snippets:
                break

            # Deduplicate snippets
            seen_ids = {s.chunk_id for s in snippets}
            for ns in new_snippets:
                if ns.chunk_id not in seen_ids:
                    snippets.append(ns)
                    seen_ids.add(ns.chunk_id)

            # Re-sort by score
            snippets.sort(key=lambda s: s.score, reverse=True)
            snippets = snippets[:self.config.final_top_k + 2]  # allow a little extra

            # Re-extract with enriched snippets
            result = await self._llm_extract(
                field_def=field_def,
                snippets=snippets,
                cluster_label=cluster_label,
                context_section=context_section,
                agreement_id=agreement_id,
                retrieval_round=round_num,
            )

        # ── Reviewer Agent Pass ──
        if self.config.enable_reviewer and result is not None and result.value is not None:
            accepted, revised_result = await self._reviewer_pass(
                result=result,
                snippets=snippets,
                field_def=field_def,
                cluster_label=cluster_label,
            )
            if not accepted:
                reviewer_rejected = True
                if revised_result is not None:
                    result = revised_result
                    result.reviewer_accepted = True
                else:
                    result.reviewer_accepted = False
                    result.confidence *= 0.5  # halve confidence for rejected
            else:
                result.reviewer_accepted = True

        # ── Normalize value ──
        if result is not None and result.value is not None:
            result.raw_value = result.value
            result.value = normalize_value(result.value, field_type)

        # Filter below threshold
        if result is not None and result.confidence < self.config.min_extraction_confidence:
            return None, retries, reviewer_rejected

        return result, retries, reviewer_rejected

    # ══════════════════════════════════════════════════════════════════════
    # RETRIEVAL METHODS
    # ══════════════════════════════════════════════════════════════════════

    async def _hybrid_retrieve(
        self,
        agreement_id: str,
        field_def: dict,
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """
        Round 1: Hybrid retrieval (semantic + lexical + RRF).

        Build query from field definition → fetch top-k from both signals
        → fuse with RRF → return top-3 snippets.
        """
        query = self._build_field_query(field_def)
        retriever = self._doc_retrievers.get(agreement_id)

        if retriever is not None:
            # Use the pre-built HybridRetriever
            raw_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retriever.search(
                    query,
                    top_k=self.config.final_top_k,
                    use_reranker=False,
                    rrf_k=self.config.rrf_k,
                ),
            )
            snippets = []
            for rank, r in enumerate(raw_results):
                idx = r["idx"]
                chunk_row = chunks[idx] if idx < len(chunks) else None
                snippets.append(RetrievedSnippet(
                    chunk_id=chunk_row["chunk_id"] if chunk_row else f"idx_{idx}",
                    chunk_text=r["chunk"],
                    agreement_id=agreement_id,
                    score=r["rrf_score"],
                    source="rrf",
                    rank=rank,
                ))
            return snippets

        # Fallback: simple semantic search via LanceDB
        if self.lance:
            query_vec = self.embed_fn(query).reshape(-1)
            results = self.lance.search_chunks(
                query_vector=query_vec,
                top_k=self.config.final_top_k,
                where=f"agreement_id = '{agreement_id}'",
            )
            return [
                RetrievedSnippet(
                    chunk_id=r.get("chunk_id", ""),
                    chunk_text=r.get("text", ""),
                    agreement_id=agreement_id,
                    score=1.0 - r.get("_distance", 0),
                    source="semantic",
                    rank=i,
                )
                for i, r in enumerate(results)
            ]

        # Last resort: return all chunks sorted by position
        return [
            RetrievedSnippet(
                chunk_id=c.get("chunk_id", f"idx_{i}"),
                chunk_text=c.get("chunk_text", ""),
                agreement_id=agreement_id,
                score=1.0 / (i + 1),
                source="positional",
                rank=i,
            )
            for i, c in enumerate(chunks[:self.config.final_top_k])
        ]

    async def _agentic_retrieve(
        self,
        agreement_id: str,
        field_def: dict,
        search_terms: list[str],
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """
        Agentic retry: re-query with LLM-generated search terms.

        Uses each search term as a separate query, then merges results
        via RRF across all sub-queries.
        """
        retriever = self._doc_retrievers.get(agreement_id)
        if retriever is None:
            return []

        all_rrf: dict[int, float] = {}
        per_idx_text: dict[int, str] = {}

        for term in search_terms[:3]:  # cap at 3 sub-queries
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda t=term: retriever.search(
                    t,
                    top_k=self.config.semantic_top_k,
                    use_reranker=False,
                    rrf_k=self.config.rrf_k,
                ),
            )
            for rank, r in enumerate(raw):
                idx = r["idx"]
                all_rrf[idx] = all_rrf.get(idx, 0) + 1.0 / (self.config.rrf_k + rank)
                per_idx_text[idx] = r["chunk"]

        # Sort by fused score
        sorted_results = sorted(all_rrf.items(), key=lambda x: x[1], reverse=True)
        snippets = []
        for rank, (idx, score) in enumerate(sorted_results[:self.config.final_top_k]):
            chunk_row = chunks[idx] if idx < len(chunks) else None
            snippets.append(RetrievedSnippet(
                chunk_id=chunk_row["chunk_id"] if chunk_row else f"idx_{idx}",
                chunk_text=per_idx_text.get(idx, ""),
                agreement_id=agreement_id,
                score=score,
                source="agentic_retry",
                rank=rank,
            ))
        return snippets

    # ══════════════════════════════════════════════════════════════════════
    # NL INDEX (cross-doc heading lookup)
    # ══════════════════════════════════════════════════════════════════════

    def _build_nl_index(self, chunks_by_doc: dict[str, list[dict]]):
        """
        Build a lightweight heading/section index from chunk texts.

        Looks for markdown headings, numbered sections, and clause titles
        to enable heading-targeted retrieval.
        """
        heading_pattern = re.compile(
            r'^(?:#{1,4}\s+|(?:\d+\.)+\s+|Article\s+\d+|Section\s+\d+|ARTICLE\s+[IVXLC]+)',
            re.MULTILINE,
        )
        self._nl_index = {}

        for agreement_id, chunks in chunks_by_doc.items():
            headings = []
            for chunk in chunks:
                text = chunk.get("chunk_text", "")
                matches = heading_pattern.findall(text)
                if matches:
                    for m in matches[:3]:
                        # Extract the full heading line
                        for line in text.split("\n"):
                            if m.strip() in line:
                                headings.append({
                                    "heading": line.strip()[:200],
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "chunk_text": text,
                                    "agreement_id": agreement_id,
                                })
                                break
            self._nl_index[agreement_id] = headings

    def _nl_index_lookup(
        self,
        field_def: dict,
        search_terms: list[str],
        agreement_id: str,
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """Heading-targeted retrieval from NL index."""
        if not self._nl_index:
            return []

        headings = self._nl_index.get(agreement_id, [])
        if not headings:
            return []

        # Keyword match on headings
        all_terms = search_terms + [field_def["name"].replace("_", " ")]
        scored: list[tuple[dict, float]] = []
        for h in headings:
            heading_lower = h["heading"].lower()
            score = 0.0
            for term in all_terms:
                for word in term.lower().split():
                    if len(word) > 2 and word in heading_lower:
                        score += 1.0
            if score > 0:
                scored.append((h, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        snippets = []
        seen = set()
        for h, score in scored[:self.config.nl_index_top_k]:
            cid = h["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            snippets.append(RetrievedSnippet(
                chunk_id=cid,
                chunk_text=h["chunk_text"],
                agreement_id=agreement_id,
                score=score / max(len(all_terms), 1),
                source="nl_index",
                rank=len(snippets),
            ))
        return snippets

    # ══════════════════════════════════════════════════════════════════════
    # LLM EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    async def _llm_extract(
        self,
        field_def: dict,
        snippets: list[RetrievedSnippet],
        cluster_label: str,
        context_section: str,
        agreement_id: str,
        retrieval_round: int,
    ) -> ExtractionResult | None:
        """
        Run LLM extraction over retrieved snippets for one field.

        Model routing: complex fields → GPT-5.4, simple → GPT-5.4-nano.
        """
        if not snippets:
            return None

        field_name = field_def["name"]
        field_type = field_def.get("type", "text")
        description = field_def.get("description", "")
        examples = field_def.get("examples", [])

        # Choose model
        model = self.config.complex_model
        if self.config.model_routing and not is_complex_field(field_def):
            model = self.config.simple_model

        # Build snippet context
        snippet_texts = []
        for i, s in enumerate(snippets[:self.config.final_top_k]):
            snippet_texts.append(
                f"[Snippet {i+1}] (score={s.score:.3f}, source={s.source}):\n"
                f"{s.chunk_text[:self.config.max_chunk_chars]}"
            )
        snippet_section = "\n\n".join(snippet_texts)

        # Build examples section
        examples_section = ""
        if examples:
            ex_str = ", ".join(f'"{e}"' for e in examples[:3] if e)
            if ex_str:
                examples_section = f"\nExamples of valid values: {ex_str}"

        prompt = _EXTRACTION_PROMPT.format(
            cluster_label=cluster_label,
            context_section=context_section,
            field_name=field_name,
            field_type=field_type,
            field_description=description,
            examples_section=examples_section,
            snippet_section=snippet_section,
        )

        parsed = await self.llm.complete_json(
            prompt, temperature=0.0, max_tokens=500, model=model,
        )

        if not parsed or not isinstance(parsed, dict):
            return None

        value = parsed.get("value")
        if value is None or str(value).lower() in ("null", "n/a", "none", ""):
            return ExtractionResult(
                field_name=field_name,
                field_type=field_type,
                value=None,
                raw_value=None,
                confidence=0.0,
                evidence_span="",
                is_default=True,
                agreement_id=agreement_id,
                chunk_id=snippets[0].chunk_id if snippets else "",
                model_used=model,
                retrieval_round=retrieval_round,
                reviewer_accepted=None,
                snippets_used=len(snippets),
            )

        value_str = str(value)
        evidence = str(parsed.get("evidence", ""))[:500]
        is_default = bool(parsed.get("is_default", False))
        confidence = float(parsed.get("confidence", 0.5))

        # Grounding boost: verify value substring in snippets
        combined = " ".join(s.chunk_text for s in snippets)
        if value_str.lower() in combined.lower():
            confidence = min(1.0, confidence + 0.1)

        return ExtractionResult(
            field_name=field_name,
            field_type=field_type,
            value=value_str,
            raw_value=value_str,
            confidence=round(confidence, 3),
            evidence_span=evidence,
            is_default=is_default,
            agreement_id=agreement_id,
            chunk_id=snippets[0].chunk_id if snippets else "",
            model_used=model,
            retrieval_round=retrieval_round,
            reviewer_accepted=None,
            snippets_used=len(snippets),
        )

    # ══════════════════════════════════════════════════════════════════════
    # AGENTIC SEARCH TERM GENERATION
    # ══════════════════════════════════════════════════════════════════════

    async def _generate_search_terms(
        self,
        field_def: dict,
        cluster_label: str,
        previous_snippets: list[RetrievedSnippet],
        previous_result: ExtractionResult | None,
    ) -> list[str]:
        """
        Ask LLM to generate alternative search terms when extraction
        confidence is low ("Insufficient Context" scenario).
        """
        prev_texts = "\n".join(
            f"- {s.chunk_text[:200]}" for s in previous_snippets[:3]
        )
        prev_value = previous_result.value if previous_result else "None"

        prompt = _SEARCH_TERM_PROMPT.format(
            field_name=field_def["name"],
            field_type=field_def.get("type", "text"),
            field_description=field_def.get("description", ""),
            cluster_label=cluster_label,
            previous_snippets=prev_texts,
            previous_value=prev_value,
        )

        parsed = await self.llm.complete_json(prompt, temperature=0.3, max_tokens=300)
        if not parsed or not isinstance(parsed, dict):
            return []

        terms = parsed.get("search_terms", [])
        if not isinstance(terms, list):
            return []
        return [str(t).strip() for t in terms if t and str(t).strip()][:5]

    # ══════════════════════════════════════════════════════════════════════
    # REVIEWER AGENT
    # ══════════════════════════════════════════════════════════════════════

    async def _reviewer_pass(
        self,
        result: ExtractionResult,
        snippets: list[RetrievedSnippet],
        field_def: dict,
        cluster_label: str,
    ) -> tuple[bool, ExtractionResult | None]:
        """
        Cheap reviewer model: accept/reject extraction + optional re-extract.

        Returns: (accepted, revised_result_or_None)
        """
        snippet_evidence = "\n".join(
            f"[{i+1}] {s.chunk_text[:500]}" for i, s in enumerate(snippets[:3])
        )

        model = self.config.reviewer_model or self.config.simple_model
        prompt = _REVIEWER_PROMPT.format(
            field_name=field_def["name"],
            field_type=field_def.get("type", "text"),
            field_description=field_def.get("description", ""),
            cluster_label=cluster_label,
            extracted_value=result.value,
            evidence_span=result.evidence_span,
            snippet_evidence=snippet_evidence,
        )

        parsed = await self.llm.complete_json(
            prompt, temperature=0.0, max_tokens=400, model=model,
        )
        if not parsed or not isinstance(parsed, dict):
            return True, None  # fail-open: accept if reviewer fails

        verdict = str(parsed.get("verdict", "accept")).lower()
        if verdict == "accept":
            return True, None

        # Rejected — check if reviewer provided a corrected value
        corrected = parsed.get("corrected_value")
        if corrected and str(corrected).lower() not in ("null", "none", "n/a", ""):
            revised = ExtractionResult(
                field_name=result.field_name,
                field_type=result.field_type,
                value=str(corrected),
                raw_value=str(corrected),
                confidence=float(parsed.get("confidence", 0.4)),
                evidence_span=str(parsed.get("evidence", result.evidence_span))[:500],
                is_default=False,
                agreement_id=result.agreement_id,
                chunk_id=result.chunk_id,
                model_used=model,
                retrieval_round=result.retrieval_round,
                reviewer_accepted=None,
                snippets_used=result.snippets_used,
            )
            return False, revised

        return False, None

    # ══════════════════════════════════════════════════════════════════════
    # INDEX BUILDING
    # ══════════════════════════════════════════════════════════════════════

    async def _build_doc_indexes(self, chunks_by_doc: dict[str, list[dict]]):
        """
        Build per-document HybridRetriever instances.

        Each document gets its own FAISS + BM25 index for fast in-doc search.
        """
        from core.retrieval import HybridRetriever

        loop = asyncio.get_event_loop()

        def _build_one(agreement_id: str, chunks: list[dict]):
            texts = [c.get("chunk_text", "") for c in chunks]
            if not texts:
                return agreement_id, None

            embeddings = self.embed_fn(texts)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            metadata = [
                {"chunk_id": c.get("chunk_id", ""), "agreement_id": agreement_id}
                for c in chunks
            ]
            retriever = HybridRetriever(
                chunks=texts,
                embeddings=embeddings,
                embed_fn=lambda t: self.embed_fn(t) if isinstance(t, str) else self.embed_fn([t])[0],
                chunk_metadata=metadata,
            )
            return agreement_id, retriever

        # Build in parallel via thread pool (embedding is CPU/GPU bound)
        tasks = []
        for aid, chunks in chunks_by_doc.items():
            if aid not in self._doc_retrievers:
                tasks.append(loop.run_in_executor(None, _build_one, aid, chunks))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Failed to build retriever: {r}")
                    continue
                aid, retriever = r
                if retriever is not None:
                    self._doc_retrievers[aid] = retriever

    # ══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════

    def _get_field_id_map(self, cluster_id: str, fields: list[dict]) -> dict[str, str]:
        """Map field names to field_ids from the store."""
        existing = self.store.get_fields(cluster_id=cluster_id)
        fmap = {f["name"]: f["field_id"] for f in existing}

        # Ensure all fields exist
        for f in fields:
            if f["name"] not in fmap:
                fid = f"fld_{uuid.uuid4().hex[:10]}"
                self.store.upsert_field(
                    field_id=fid,
                    cluster_id=cluster_id,
                    name=f["name"],
                    field_type=f.get("type", "text"),
                    description=f.get("description", ""),
                    source="rag_extractor",
                    examples=f.get("examples", []),
                )
                fmap[f["name"]] = fid
        return fmap

    def _persist_results(
        self,
        results: list[ExtractionResult],
        field_id_map: dict[str, str],
    ) -> int:
        """Batch-persist extraction results to store."""
        count = 0
        for r in results:
            if r.value is None:
                continue
            field_id = field_id_map.get(r.field_name)
            if not field_id:
                continue
            self.store.insert_extraction(
                extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
                agreement_id=r.agreement_id,
                field_id=field_id,
                value=r.value,
                confidence=r.confidence,
                source_chunk_id=r.chunk_id,
            )
            count += 1
        return count

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_field_query(field_def: dict) -> str:
        """Build a retrieval query from a field definition."""
        parts = [field_def["name"].replace("_", " ")]
        desc = field_def.get("description", "")
        if desc:
            parts.append(desc)
        examples = field_def.get("examples", [])
        if examples:
            parts.append("examples: " + ", ".join(str(e) for e in examples[:3]))
        return " ".join(parts)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are extracting a specific field from a legal agreement clause.

Clause Type: {cluster_label}
{context_section}

FIELD TO EXTRACT:
  Name: {field_name}
  Type: {field_type}
  Description: {field_description}{examples_section}

RETRIEVED SNIPPETS (ranked by relevance):
{snippet_section}

INSTRUCTIONS:
1. Find the value of "{field_name}" in the snippets above.
2. If the value is explicitly stated, extract it verbatim.
3. If the value must be inferred from context, note that in evidence.
4. If not found in any snippet, set value to null.

Return JSON:
{{
  "value": "<extracted value or null>",
  "evidence": "<verbatim quote from snippet supporting the extraction>",
  "confidence": <0.0-1.0>,
  "is_default": <true if this is a standard/boilerplate value>
}}"""

_SEARCH_TERM_PROMPT = """\
You are helping find better search terms for field extraction.

Field: {field_name} ({field_type})
Description: {field_description}
Clause Type: {cluster_label}

Previous search returned these snippets (which were insufficient):
{previous_snippets}

Previous extraction attempt returned: {previous_value}

Generate 3-5 alternative search terms that might locate the relevant clause.
Think about:
- Synonyms and legal terminology
- Section headings where this info appears
- Related concepts that co-occur with this field

Return JSON:
{{
  "search_terms": ["term1", "term2", "term3"]
}}"""

_REVIEWER_PROMPT = """\
You are a quality reviewer for contract field extraction. Your job is to
verify whether an extraction is correct and well-supported by evidence.

Field: {field_name} ({field_type})
Description: {field_description}
Clause Type: {cluster_label}

EXTRACTED VALUE: {extracted_value}
EVIDENCE: {evidence_span}

SOURCE SNIPPETS:
{snippet_evidence}

Review criteria:
1. Is the extracted value actually present in or supported by the snippets?
2. Does the value match the expected field type?
3. Is this the correct/complete value (not a partial extraction)?

Return JSON:
{{
  "verdict": "accept" or "reject",
  "reason": "<brief reason>",
  "corrected_value": "<if rejected, provide corrected value or null>",
  "confidence": <0.0-1.0>,
  "evidence": "<supporting quote if corrected>"
}}"""


# ── Module-level Entry Points ─────────────────────────────────────────────────


async def extract_cluster_rag(
    store,
    llm,
    embed_fn: Callable,
    cluster_id: str,
    fields: list[dict],
    cluster_label: str,
    cluster_keywords: list[str] | None = None,
    context_section: str = "",
    config: RAGConfig | None = None,
    lance_store=None,
) -> ExtractionBatch:
    """
    Module-level convenience — run RAG extraction for one cluster.

    Returns ExtractionBatch with results + meta.
    """
    extractor = RAGExtractor(
        store=store, llm=llm, embed_fn=embed_fn,
        config=config, lance_store=lance_store,
    )
    return await extractor.extract_cluster(
        cluster_id=cluster_id,
        fields=fields,
        cluster_label=cluster_label,
        cluster_keywords=cluster_keywords,
        context_section=context_section,
    )
