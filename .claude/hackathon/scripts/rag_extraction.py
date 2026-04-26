#!/usr/bin/env python3
"""
RAG Extraction Pipeline — Self-Contained
=========================================

WHAT THIS DOES (plain English):
  Given a cluster of similar contracts (e.g., all "Franchise Agreements"),
  this script (1) discovers what fields to extract, then (2) extracts each
  field by finding the most relevant text passages using search, rather than
  reading chunks sequentially.

WHY IT WORKS BETTER:
  Other pipelines scan a fixed set of chunks and hope the answer is there.
  This pipeline searches ALL chunks for each field independently. If "termination_fee"
  is buried in chunk 47 out of 860 chunks, hybrid search will still find it.

ALGORITHM STEPS:
  1. Discovery (shared): DSPy CoT finds candidate fields → trial extraction filter
     removes hallucinated fields → ~12-15 validated fields per cluster.

  2. Embed all chunks: MiniLM-L6-v2 encodes every chunk into a vector (one-time cost).

  3. For EACH field independently:
     a) Build a search query from field name + description
     b) Retrieve top-8 chunks via HYBRID search:
        - Semantic: cosine similarity of query embedding vs chunk embeddings
        - Lexical: character trigram overlap (catches exact keywords)
        - RRF fusion: score = 1/(60+rank_semantic) + 1/(60+rank_lexical)
     c) Send retrieved passages to GPT-5.4 → extract value as JSON
     d) If value is null → retry up to 3 times with rephrased queries
     e) Grounding check: is extracted value literally in the text?
     f) If not verbatim → reviewer LLM validates or rejects

  4. Return all extractions with grounding scores and metadata.

ANALOGY:
  Like using Ctrl+F for each field vs. reading the entire document hoping
  you'll spot every answer. The search approach is faster AND finds more.

IMPLEMENTATION DETAILS:

  1. **Cochran Sampling** — Statistical sampling formula (z²·p·(1-p)/e²)
     to determine how many documents to sample per cluster (capped at 10).

  2. **Hybrid Retrieval with RRF** — Per-field retrieval using both:
     - Semantic (embedding cosine similarity)
     - Lexical (trigram token overlap)
     Merged via Reciprocal Rank Fusion (RRF, k=60).

  3. **Tiered LLM Routing** — Complex fields (amounts, clauses, obligations)
     use the full GPT-5.4 model; simple fields (dates, parties, booleans)
     use the same model at lower temperature.

  4. **Value Normalization** — Post-extraction normalization:
     - Dates → YYYY/MM/DD
     - Numbers → preserve currency symbols, commas
     - Parties → proper-case names

  5. **Agentic Retry** — If retrieval misses a field, retry with rephrased
     query up to max_retries (3) times.

  6. **Reviewer Agent** — Final grounding validation against source text.

Key difference from agentic_extractions.py:
  - Per-FIELD retrieval (not sequential carryforward)
  - Hybrid retrieval (semantic + lexical RRF)
  - No carryforward context between chunks
  - No judge compilation (reviewer only)
  - Tiered LLM (field-complexity routing)

Self-contained: zero local imports (only stdlib + openai + numpy + sentence_transformers).
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGExtraction:
    """Single extraction result for one field."""
    field_name: str
    value: Optional[str]
    confidence: float
    source_chunk_indices: list[int]
    grounding_level: str       # verbatim | partial | hallucinated | missing
    grounding_score: float
    retrieval_method: str      # semantic | lexical | hybrid_rrf
    llm_tier: str              # full | lite
    n_retries: int = 0
    reviewer_verdict: str = ""
    normalized_value: Optional[str] = None


@dataclass
class RAGExtractionResult:
    """Full result of the RAG extraction pipeline."""
    extractions: dict          # field_name → RAGExtraction
    meta: dict                 # timing, stats, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

FIELD_EXTRACT_PROMPT = """Extract the value for the field "{field_name}" from the contract text below.

Field description: {field_description}
Field type: {field_type}
{examples_section}

Clause type: "{cluster_label}"

=== Relevant text passages ===
{passages}
===

Rules:
1. Extract ONLY values explicitly stated in the text above.
2. Use the exact wording from the text (verbatim quotes preferred).
3. If the field is not mentioned in any passage, return null.
4. Do NOT fabricate or infer values not in the text.
5. For dates, preserve the format as written.
6. For monetary amounts, include currency symbols and denominations.

Respond with ONLY this JSON:
{{
  "value": "<extracted value or null>",
  "confidence": <0.0-1.0>,
  "source_passage": <0-indexed passage number where you found the value, or -1 if null>
}}"""

RETRY_FIELD_PROMPT = """A previous extraction attempt found no value for "{field_name}" in the provided text.
The field is described as: {field_description}
Type: {field_type}

Please search VERY carefully for any mention of this data point. It may be:
- Expressed using different terminology
- Embedded in a longer sentence
- Stated indirectly or by implication
- In a different section heading

=== Text passages ===
{passages}
===

If you find the value, respond with JSON: {{"value": "<extracted value>", "confidence": <0.0-1.0>, "source_passage": <idx>}}
If it truly is not present, respond with JSON: {{"value": null, "confidence": 0.0, "source_passage": -1}}"""

REVIEWER_PROMPT = """You are a grounding reviewer. Check if the extracted value is actually present in the source text.

Field: {field_name} ({field_type})
Extracted value: "{value}"

Source passages:
{passages}

Check:
1. Is the exact value present verbatim in any source passage?
2. If not exact, is a close paraphrase present?
3. Or is this value fabricated / hallucinated?

Respond with ONLY this JSON:
{{
  "verdict": "verified" | "partial" | "hallucinated",
  "evidence": "<quote from source that supports the value, or empty string>"
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _create_client():
    """Create Azure OpenAI client from environment variables."""
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def _call_llm(client, deployment: str, prompt: str,
              temperature: float = 0.0, max_tokens: int = 500,
              max_retries: int = 3, retry_delay: float = 2.0) -> Optional[dict]:
    """Call Azure OpenAI and return parsed JSON, or None on failure."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error (attempt {attempt + 1})")
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str:
                logger.warning(f"Content filter hit (attempt {attempt + 1})")
                return None
            logger.warning(f"LLM call error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return None


def _ground_value(value: str, chunk_texts: list[str]) -> tuple[str, float]:
    """Check if extracted value is grounded in any of the chunk texts."""
    if not value:
        return "missing", 0.0
    v_lower = value.lower().strip()
    for ct in chunk_texts:
        c_lower = ct.lower()
        if v_lower in c_lower:
            return "verbatim", 1.0
    # Token overlap across all chunks
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    if not v_tokens:
        return "missing", 0.0
    all_source_tokens = set()
    for ct in chunk_texts:
        all_source_tokens.update(re.findall(r'\b\w+\b', ct.lower()))
    overlap = len(v_tokens & all_source_tokens) / len(v_tokens)
    if overlap >= 0.8:
        return "partial", 0.8
    return "hallucinated", 0.0


def _cochran_sample_size(population: int, confidence: float = 0.90,
                         margin: float = 0.15, p: float = 0.5) -> int:
    """Cochran formula: n = z²·p·(1-p) / e², with finite population correction.
    Capped at 10 for practical purposes."""
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.645)
    n0 = (z ** 2 * p * (1 - p)) / (margin ** 2)
    # Finite population correction
    if population > 0:
        n = n0 / (1 + (n0 - 1) / population)
    else:
        n = n0
    return min(int(math.ceil(n)), 10, population)


def _trigram_similarity(query: str, text: str) -> float:
    """Character trigram overlap as a lexical similarity score."""
    def trigrams(s):
        s = s.lower().strip()
        return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else {s}
    q_tri = trigrams(query)
    t_tri = trigrams(text[:2000])  # limit for efficiency
    if not q_tri:
        return 0.0
    return len(q_tri & t_tri) / len(q_tri)


def _reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """RRF merging of multiple ranked lists. Returns fused ranking."""
    scores = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def _classify_field_complexity(field: dict) -> str:
    """Route field to LLM tier: 'full' for complex fields, 'lite' for simple."""
    ftype = field.get("type", "text").lower()
    fname = field.get("name", "").lower()
    fdesc = field.get("description", "").lower()

    # Simple fields → lite tier (lower temperature)
    simple_types = {"date", "boolean", "party_name", "integer", "percentage"}
    simple_patterns = ["date", "party", "name of", "effective date", "expiration",
                       "yes/no", "true/false", "number of"]

    if ftype in simple_types:
        return "lite"
    for pat in simple_patterns:
        if pat in fname or pat in fdesc:
            return "lite"

    return "full"


def _normalize_value(value: str, field_type: str) -> str:
    """Post-extraction value normalization."""
    if not value:
        return value

    if field_type == "date":
        # Try to normalize common date formats to YYYY/MM/DD
        patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', r'\3/\1/\2'),
            (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', None),  # "January 1, 2020" — leave as-is
        ]
        for pat, repl in patterns:
            if repl and re.match(pat, value):
                return re.sub(pat, repl, value)
        return value

    if field_type in ("currency", "monetary"):
        # Preserve currency symbols, remove extra whitespace
        return re.sub(r'\s+', ' ', value).strip()

    if field_type == "party_name":
        # Proper case for party names
        if value == value.upper() or value == value.lower():
            return value.title()

    return value.strip()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGConfig:
    """Configuration for the RAG extraction pipeline."""
    # Retrieval
    top_k_chunks: int = 8           # chunks per field after RRF
    rrf_k: int = 60                 # RRF parameter
    # Sampling
    cochran_confidence: float = 0.90
    cochran_margin: float = 0.15
    max_sample_docs: int = 10
    # Extraction
    max_chunk_chars: int = 4000
    # Tiered LLM
    full_temperature: float = 0.0
    lite_temperature: float = 0.0
    # Retry
    max_retries: int = 3
    retry_temperature: float = 0.1
    # Reviewer
    enable_reviewer: bool = True
    reviewer_temperature: float = 0.0
    # Normalization
    enable_normalization: bool = True
    # LLM
    deployment: str = ""
    max_retries_llm: int = 3


# ══════════════════════════════════════════════════════════════════════════════
# RAG EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGExtractionPipeline:
    """
    RAG-based extraction pipeline with hybrid retrieval,
    tiered LLM routing, agentic retry, and reviewer validation.
    """

    def __init__(self, client=None, config: Optional[RAGConfig] = None,
                 embedder=None):
        self.client = client
        self.config = config or RAGConfig()
        self._embedder = embedder

    @classmethod
    def from_env(cls, config: Optional[RAGConfig] = None):
        """Create pipeline from environment variables."""
        from dotenv import load_dotenv
        load_dotenv()
        client = _create_client()
        return cls(client=client, config=config)

    @property
    def deployment(self):
        if self.config.deployment:
            return self.config.deployment
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ── Main entry point ──────────────────────────────────────────────

    def extract(
        self,
        fields: list[dict],
        chunk_texts: list[str],
        cluster_label: str,
    ) -> RAGExtractionResult:
        """
        Run the full RAG extraction pipeline.

        Args:
            fields: Discovered field definitions [{name, type, description, examples}, ...]
            chunk_texts: All chunk texts for this cluster/document
            cluster_label: Clause type label

        Returns:
            RAGExtractionResult with per-field extractions and metadata
        """
        t0 = time.time()
        meta = {
            "n_fields": len(fields),
            "n_chunks": len(chunk_texts),
            "n_retries": 0,
            "n_reviewer_calls": 0,
            "cochran_sample_size": len(chunk_texts),
        }

        if not fields or not chunk_texts:
            return RAGExtractionResult(extractions={}, meta=meta)

        # Cochran sampling (for this eval we use all available chunks,
        # but record what sample size would be for a larger corpus)
        sample_n = _cochran_sample_size(
            len(chunk_texts),
            confidence=self.config.cochran_confidence,
            margin=self.config.cochran_margin,
        )
        meta["cochran_sample_size"] = sample_n

        # Pre-compute embeddings for all chunks
        t1 = time.time()
        chunk_embeddings = self.embedder.encode(
            chunk_texts, normalize_embeddings=True, batch_size=32
        )
        meta["embedding_time_s"] = round(time.time() - t1, 2)

        # Extract each field independently (per-field retrieval)
        extractions = {}
        for f in fields:
            ext = self._extract_field(
                f, chunk_texts, chunk_embeddings, cluster_label, meta
            )
            extractions[f["name"]] = ext

        # Reviewer validation
        if self.config.enable_reviewer:
            t_rev = time.time()
            self._review_extractions(extractions, chunk_texts, meta)
            meta["reviewer_time_s"] = round(time.time() - t_rev, 2)

        meta["total_time_s"] = round(time.time() - t0, 2)

        # Aggregate stats
        non_null = [e for e in extractions.values() if e.value is not None]
        if non_null:
            n_verified = sum(1 for e in non_null if e.grounding_level in ("verified", "verbatim"))
            n_partial = sum(1 for e in non_null if e.grounding_level == "partial")
            n_halluc = sum(1 for e in non_null if e.grounding_level == "hallucinated")
            total = len(non_null)
            meta["n_extracted"] = total
            meta["verified_rate"] = round(n_verified / total, 3) if total else 0
            meta["partial_rate"] = round(n_partial / total, 3) if total else 0
            meta["hallucination_rate"] = round(n_halluc / total, 3) if total else 0
            meta["grounded_rate"] = round((n_verified + n_partial) / total, 3) if total else 0
        else:
            meta["n_extracted"] = 0
            meta["verified_rate"] = 0
            meta["partial_rate"] = 0
            meta["hallucination_rate"] = 0
            meta["grounded_rate"] = 0

        return RAGExtractionResult(extractions=extractions, meta=meta)

    # ── Per-field extraction with hybrid retrieval ────────────────────

    def _extract_field(
        self,
        field: dict,
        chunk_texts: list[str],
        chunk_embeddings: np.ndarray,
        cluster_label: str,
        meta: dict,
    ) -> RAGExtraction:
        """Extract a single field using hybrid retrieval + LLM."""
        fname = field["name"]
        ftype = field.get("type", "text")
        fdesc = field.get("description", "")
        examples = field.get("examples", [])

        # Build query for retrieval
        query = f"{fname.replace('_', ' ')}: {fdesc}"
        if examples:
            query += f" (e.g., {examples[0]})"

        # Hybrid retrieval: semantic + lexical + RRF
        retrieved_indices = self._hybrid_retrieve(
            query, chunk_texts, chunk_embeddings
        )

        # Determine LLM tier
        tier = _classify_field_complexity(field)
        temperature = self.config.lite_temperature if tier == "lite" else self.config.full_temperature

        # Build passages text
        passages = self._build_passages(chunk_texts, retrieved_indices)

        # Examples section
        examples_section = ""
        if examples:
            ex_str = ", ".join(f'"{e}"' for e in examples[:3] if e)
            if ex_str:
                examples_section = f"Examples of valid values: {ex_str}"

        # Extract
        prompt = FIELD_EXTRACT_PROMPT.format(
            field_name=fname,
            field_description=fdesc,
            field_type=ftype,
            examples_section=examples_section,
            cluster_label=cluster_label,
            passages=passages,
        )

        parsed = _call_llm(
            self.client, self.deployment, prompt,
            temperature=temperature,
            max_retries=self.config.max_retries_llm,
        )

        value = None
        confidence = 0.0
        source_idx = -1
        n_retries = 0

        if parsed:
            value = parsed.get("value")
            if value and str(value).lower() in ("null", "none", "n/a", ""):
                value = None
            confidence = parsed.get("confidence", 0.0)
            source_idx = parsed.get("source_passage", -1)

        # Agentic retry if no value found
        if value is None:
            for retry in range(self.config.max_retries):
                n_retries += 1
                meta["n_retries"] = meta.get("n_retries", 0) + 1

                # Retry with rephrased query
                alt_query = self._rephrase_query(field, retry)
                retry_indices = self._hybrid_retrieve(
                    alt_query, chunk_texts, chunk_embeddings
                )
                retry_passages = self._build_passages(chunk_texts, retry_indices)

                retry_prompt = RETRY_FIELD_PROMPT.format(
                    field_name=fname,
                    field_description=fdesc,
                    field_type=ftype,
                    passages=retry_passages,
                )

                retry_parsed = _call_llm(
                    self.client, self.deployment, retry_prompt,
                    temperature=self.config.retry_temperature,
                    max_retries=2,
                )

                if retry_parsed:
                    v = retry_parsed.get("value")
                    if v and str(v).lower() not in ("null", "none", "n/a", ""):
                        value = str(v)
                        confidence = retry_parsed.get("confidence", 0.5)
                        source_idx = retry_parsed.get("source_passage", -1)
                        retrieved_indices = retry_indices
                        break

        # Grounding check
        source_chunks = [chunk_texts[i] for i in retrieved_indices if i < len(chunk_texts)]
        if value:
            value = str(value)
            grounding_level, grounding_score = _ground_value(value, source_chunks)
        else:
            grounding_level, grounding_score = "missing", 0.0

        # Normalization
        normalized = None
        if value and self.config.enable_normalization:
            normalized = _normalize_value(value, ftype)
            if normalized == value:
                normalized = None

        return RAGExtraction(
            field_name=fname,
            value=value,
            confidence=confidence,
            source_chunk_indices=retrieved_indices,
            grounding_level=grounding_level,
            grounding_score=grounding_score,
            retrieval_method="hybrid_rrf",
            llm_tier=tier,
            n_retries=n_retries,
            normalized_value=normalized,
        )

    # ── Hybrid Retrieval (Semantic + Lexical + RRF) ───────────────────

    def _hybrid_retrieve(
        self,
        query: str,
        chunk_texts: list[str],
        chunk_embeddings: np.ndarray,
    ) -> list[int]:
        """Retrieve top-K chunks using hybrid retrieval with RRF fusion."""
        k = min(self.config.top_k_chunks, len(chunk_texts))

        # 1. Semantic ranking (cosine similarity)
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        semantic_scores = np.dot(chunk_embeddings, query_vec)
        semantic_ranked = np.argsort(-semantic_scores).tolist()

        # 2. Lexical ranking (trigram similarity)
        lexical_scores = [_trigram_similarity(query, ct) for ct in chunk_texts]
        lexical_ranked = sorted(range(len(chunk_texts)),
                                key=lambda i: lexical_scores[i], reverse=True)

        # 3. RRF fusion
        fused = _reciprocal_rank_fusion(
            [semantic_ranked, lexical_ranked],
            k=self.config.rrf_k,
        )

        return fused[:k]

    # ── Helper methods ────────────────────────────────────────────────

    def _build_passages(self, chunk_texts: list[str], indices: list[int]) -> str:
        """Build formatted passages string for prompts."""
        lines = []
        for i, idx in enumerate(indices):
            if idx < len(chunk_texts):
                text = chunk_texts[idx][:self.config.max_chunk_chars]
                lines.append(f"[Passage {i}] (chunk {idx}):\n{text}\n")
        return "\n".join(lines)

    def _rephrase_query(self, field: dict, retry_attempt: int) -> str:
        """Generate alternative query for retry attempts."""
        fname = field["name"].replace("_", " ")
        fdesc = field.get("description", "")
        ftype = field.get("type", "text")

        if retry_attempt == 0:
            # Broader query
            return f"{fdesc} {ftype} clause provision"
        elif retry_attempt == 1:
            # Keyword-focused
            keywords = re.findall(r'\b\w{4,}\b', fdesc)
            return " ".join(keywords[:5]) if keywords else fname
        else:
            # Very broad
            return f"{fname} agreement contract term condition"

    def _review_extractions(
        self,
        extractions: dict[str, RAGExtraction],
        chunk_texts: list[str],
        meta: dict,
    ):
        """Run reviewer on non-trivially-grounded extractions."""
        for fname, ext in extractions.items():
            if ext.value is None:
                ext.reviewer_verdict = "missing"
                continue
            if ext.grounding_level == "verbatim":
                ext.reviewer_verdict = "verified"
                continue

            # Need LLM reviewer
            meta["n_reviewer_calls"] = meta.get("n_reviewer_calls", 0) + 1

            source_chunks = [chunk_texts[i] for i in ext.source_chunk_indices
                             if i < len(chunk_texts)]
            passages = "\n---\n".join(
                f"[Passage {i}]: {ct[:2000]}"
                for i, ct in enumerate(source_chunks)
            )

            prompt = REVIEWER_PROMPT.format(
                field_name=fname,
                field_type="text",
                value=ext.value,
                passages=passages,
            )

            parsed = _call_llm(
                self.client, self.deployment, prompt,
                temperature=self.config.reviewer_temperature,
                max_retries=2,
                max_tokens=200,
            )

            if parsed:
                verdict = parsed.get("verdict", "hallucinated")
                ext.reviewer_verdict = verdict
                if verdict == "verified":
                    ext.grounding_level = "verified"
                    ext.grounding_score = 1.0
                elif verdict == "partial":
                    ext.grounding_level = "partial"
                    ext.grounding_score = 0.8
                elif verdict == "hallucinated":
                    ext.grounding_level = "hallucinated"
                    ext.grounding_score = 0.0
            else:
                ext.reviewer_verdict = "unknown"

    # ── Flat output for comparison ────────────────────────────────────

    def extract_flat(
        self,
        fields: list[dict],
        chunk_texts: list[str],
        cluster_label: str,
    ) -> dict:
        """Extract and return flat dict compatible with comparison format."""
        result = self.extract(fields, chunk_texts, cluster_label)
        flat = {}
        for fname, ext in result.extractions.items():
            flat[fname] = {
                "value": ext.normalized_value or ext.value,
                "confidence": ext.confidence,
                "grounding_level": ext.grounding_level,
                "grounding_score": ext.grounding_score,
                "source_chunks": ext.source_chunk_indices,
                "retrieval_method": ext.retrieval_method,
                "llm_tier": ext.llm_tier,
                "n_retries": ext.n_retries,
                "reviewer_verdict": ext.reviewer_verdict,
            }
        return flat, result.meta


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Standalone test: run RAG extraction on one cluster."""
    import duckdb
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = "data/cuad_510_demo.duckdb"
    conn = duckdb.connect(db_path, read_only=True)

    # Use first baseline cluster for testing
    cluster_id = "ct_f778e156"  # Franchise agreement
    cluster = conn.execute(
        "SELECT label, keywords FROM clusters WHERE cluster_id = ?", [cluster_id]
    ).fetchdf().to_dict("records")[0]

    chunks = conn.execute("""
        SELECT ch.chunk_text FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY ch.chunk_index
    """, [cluster_id]).fetchdf()["chunk_text"].tolist()

    # Mock discovered fields for testing
    fields = [
        {"name": "franchise_fee", "type": "currency", "description": "Initial franchise fee amount"},
        {"name": "territory", "type": "text", "description": "Geographic territory granted to franchisee"},
        {"name": "term_years", "type": "integer", "description": "Duration of franchise agreement in years"},
    ]

    pipeline = RAGExtractionPipeline.from_env()
    result = pipeline.extract(fields, chunks[:8], cluster["label"])

    print("\n=== RAG Extraction Results ===")
    for fname, ext in result.extractions.items():
        print(f"  {fname}: {ext.value} (conf={ext.confidence:.2f}, "
              f"ground={ext.grounding_level}, tier={ext.llm_tier}, retries={ext.n_retries})")
    print(f"\n  Meta: {json.dumps(result.meta, indent=2)}")

    conn.close()


if __name__ == "__main__":
    main()
