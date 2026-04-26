#!/usr/bin/env python3
"""
Agentic Extraction Pipeline — Self-Contained
=============================================

A production-grade extraction backend that implements:

  1. **Embedding-based Retrieval** — For each field, retrieve the top-K most
     relevant chunks using cosine similarity (replaces linear scanning).

  2. **Sequential Extraction with Carryforward** — Process chunks in order
     per document. Each chunk extraction sees what was already found in prior
     chunks, preventing duplicates and enabling cross-reference resolution.

  3. **Agentic Retry** — If an extraction returns null for a field, rephrase
     the query and re-retrieve up to max_retries times.

  4. **Judge LLM Compilation** — After all chunk-level extractions, a judge
     LLM reviews all values for each field across chunks, resolves conflicts,
     and produces one canonical value per (document, field).

  5. **Reviewer Validation** — Post-compilation grounding check. The reviewer
     verifies each compiled value is actually present in the source text.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  AgenticExtractionPipeline                                      │
  │                                                                 │
  │  Input: fields (from discovery), chunks (per document)          │
  │                                                                 │
  │  Phase 1: Retrieval                                             │
  │    For each field → embed field desc → cosine top-K chunks     │
  │                                                                 │
  │  Phase 2: Sequential Extraction w/ Carryforward                 │
  │    For each chunk (in doc order):                               │
  │      ├── Carryforward: summary of prior extractions            │
  │      ├── Extractor LLM: extract + grounding check inline       │
  │      └── Agentic retry: if field missed, re-query (max 2x)    │
  │                                                                 │
  │  Phase 3: Judge Compilation                                     │
  │    For each field:                                              │
  │      ├── Collect all (chunk, value, confidence) triples        │
  │      ├── Judge LLM: pick best value, resolve conflicts         │
  │      └── Output: one canonical value + confidence + source     │
  │                                                                 │
  │  Phase 4: Reviewer Validation                                   │
  │    For each compiled value:                                     │
  │      ├── Verbatim check against source chunk text              │
  │      └── Mark: verified / partial / hallucinated               │
  └─────────────────────────────────────────────────────────────────┘

Self-contained: zero local imports (only stdlib + openai + numpy + sentence_transformers).

Usage:
    from scripts.agentic_extractions import AgenticExtractionPipeline

    pipeline = AgenticExtractionPipeline.from_env()
    result = pipeline.extract(
        fields=discovered_fields,       # list[dict] from discovery
        chunk_texts=chunk_texts,         # list[str] per document
        cluster_label="Franchise agreement",
        rlm_context={"clause_types": [...], ...},  # optional
    )
    # result.compiled  → dict[field_name, CompiledValue]
    # result.meta      → timing, grounding stats, retry counts
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkExtraction:
    """Single extraction from one chunk."""
    chunk_index: int
    field_name: str
    value: Optional[str]
    confidence: float
    grounding_level: str       # verbatim | partial | hallucinated | missing
    grounding_score: float
    was_retry: bool = False


@dataclass
class CompiledValue:
    """Final compiled value for a (document, field) pair after judge review."""
    field_name: str
    value: Optional[str]
    confidence: float
    source_chunk_indices: list[int]
    grounding_level: str       # verified | partial | hallucinated | missing
    judge_reasoning: str = ""
    reviewer_verdict: str = ""
    conflict_resolved: bool = False
    raw_extractions: list[ChunkExtraction] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Full result of the agentic extraction pipeline."""
    compiled: dict              # field_name → CompiledValue
    chunk_extractions: list     # all ChunkExtraction objects
    meta: dict                  # timing, stats, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

EXTRACT_WITH_CARRYFORWARD_PROMPT = """Extract field values from this contract clause.
IMPORTANT: Only extract values EXPLICITLY stated in the text. Do NOT hallucinate.

Clause type: "{cluster_label}"
{rlm_context_section}

=== Previously extracted values from earlier chunks of this document ===
{carryforward_context}
===

Clause text (chunk {chunk_index}):
{chunk_text}

Fields to extract:
{fields_schema}

Rules:
1. Extract ONLY values explicitly stated in the text above.
2. If a value was already found in a previous chunk (shown above), only extract
   it again if THIS chunk contains a DIFFERENT or MORE SPECIFIC value.
3. Use the exact wording from the text (verbatim quotes preferred).
4. Return null if the field is not present in this chunk.
5. Do not fabricate values not in the text.

Respond with ONLY a JSON object mapping field names to extracted values.
Example: {{"notice_period": "30 days", "auto_renewal": null, "termination_fee": "$5,000"}}"""

RETRY_EXTRACT_PROMPT = """A previous extraction attempt found no value for the field "{field_name}" in this text.
The field is described as: {field_description}

Please look very carefully for any mention of this data point in the text below.
It may be expressed indirectly, using different terminology, or embedded in a longer sentence.

Text:
{chunk_text}

If you find the value, respond with JSON: {{"{field_name}": "<extracted value>"}}
If it truly is not present, respond with JSON: {{"{field_name}": null}}"""

JUDGE_COMPILATION_PROMPT = """You are a legal document expert reviewing extracted values for the field "{field_name}".
Field description: {field_description}
Field type: {field_type}

Multiple chunks from the same document produced these extractions:
{extractions_summary}

Your task:
1. If all chunks agree, confirm the value.
2. If chunks disagree, determine which value is most likely correct based on:
   - Specificity (more specific > vague)
   - Grounding confidence (higher confidence > lower)
   - Context (later chunks may update/override earlier ones)
3. If no chunk had a value, return null.

Respond with ONLY this JSON:
{{
  "value": "<best value or null>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining your choice>",
  "source_chunk": <chunk index of the chosen value, or -1 if null>
}}"""

REVIEWER_PROMPT = """You are a grounding reviewer. Check if the extracted value is actually present in the source text.

Field: {field_name} ({field_type})
Extracted value: "{value}"
Source text from chunk {chunk_index}:
{chunk_text}

Check:
1. Is the exact value present verbatim in the source text?
2. If not exact, is a close paraphrase present?
3. Or is this value fabricated / hallucinated?

Respond with ONLY this JSON:
{{
  "verdict": "verified" | "partial" | "hallucinated",
  "evidence": "<quote from source text that supports the value, or empty string>"
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLING UTILITIES
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
              temperature: float = 0.0, max_tokens: int = 1000,
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
                return None  # don't retry content filters
            logger.warning(f"LLM call error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GROUNDING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _ground_value(value: str, chunk_text: str) -> tuple[str, float]:
    """Check if extracted value is grounded in source text.
    Returns (level, score) where level is verbatim|partial|hallucinated."""
    if not value or not chunk_text:
        return "missing", 0.0
    v_lower = value.lower().strip()
    c_lower = chunk_text.lower()
    if v_lower in c_lower:
        return "verbatim", 1.0
    # Token overlap check
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    c_tokens = set(re.findall(r'\b\w+\b', c_lower))
    if not v_tokens:
        return "missing", 0.0
    overlap = len(v_tokens & c_tokens) / len(v_tokens)
    if overlap >= 0.8:
        return "partial", 0.8
    return "hallucinated", 0.0


def _build_fields_schema(fields: list[dict]) -> str:
    """Build the fields schema string for extraction prompts."""
    lines = []
    for f in fields:
        line = f"- {f['name']} ({f.get('type', 'text')}): {f.get('description', '')}"
        examples = f.get('examples', [])
        if examples:
            ex_str = ', '.join(f'"{e}"' for e in examples[:2] if e)
            if ex_str:
                line += f"  [examples: {ex_str}]"
        lines.append(line)
    return "\n".join(lines)


def _build_rlm_context_section(rlm_context: Optional[dict]) -> str:
    """Build RLM context section for extraction prompts."""
    if not rlm_context:
        return ""
    parts = []
    for ct in rlm_context.get("clause_types", [])[:3]:
        parts.append(f"Clause sub-type: {ct.get('name', '')}: {ct.get('emergent_description', '')}")
    for ob in rlm_context.get("obligations", [])[:3]:
        trigger = f" if {ob['trigger_condition']}" if ob.get("trigger_condition") else ""
        parts.append(f"Obligation: {ob.get('obligor', '')} must {ob.get('action', '')}{trigger}")
    for intent in rlm_context.get("intents", [])[:3]:
        parts.append(f"Intent: {intent.get('intent_description', '')}")
    return "\n".join(parts) + "\n" if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgenticConfig:
    """Configuration for the agentic extraction pipeline."""
    # Retrieval
    top_k_chunks: int = 8                # chunks per field for retrieval
    # Extraction
    max_chunk_chars: int = 4000          # max chars per chunk sent to LLM
    temperature: float = 0.0
    # Agentic retry
    max_retries: int = 2                 # retry attempts for missing fields
    retry_temperature: float = 0.1       # slightly higher temp for retries
    # Judge
    enable_judge: bool = True            # compile values across chunks
    judge_temperature: float = 0.0
    # Reviewer
    enable_reviewer: bool = True         # post-compilation grounding check
    reviewer_temperature: float = 0.0
    # LLM
    deployment: str = ""                 # Azure deployment (from env if empty)
    max_retries_llm: int = 3


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class AgenticExtractionPipeline:
    """
    Full agentic extraction pipeline with retrieval, carryforward,
    retry, judge compilation, and reviewer validation.
    """

    def __init__(self, client=None, config: Optional[AgenticConfig] = None,
                 embedder=None):
        self.client = client
        self.config = config or AgenticConfig()
        self._embedder = embedder

    @classmethod
    def from_env(cls, config: Optional[AgenticConfig] = None):
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
        rlm_context: Optional[dict] = None,
    ) -> ExtractionResult:
        """
        Run the full agentic extraction pipeline.

        Args:
            fields: Discovered field definitions [{name, type, description, examples}, ...]
            chunk_texts: All chunk texts for this cluster/document
            cluster_label: Clause type label
            rlm_context: Optional discovery context (clause_types, intents, obligations)

        Returns:
            ExtractionResult with compiled values, chunk extractions, and metadata
        """
        t0 = time.time()
        meta = {
            "n_fields": len(fields),
            "n_chunks": len(chunk_texts),
            "n_retries": 0,
            "n_judge_calls": 0,
            "n_reviewer_calls": 0,
            "n_conflicts_resolved": 0,
        }

        if not fields or not chunk_texts:
            return ExtractionResult(compiled={}, chunk_extractions=[], meta=meta)

        # Phase 1: Retrieve top-K chunks per field
        t1 = time.time()
        field_chunk_map = self._retrieve_chunks_per_field(fields, chunk_texts)
        meta["retrieval_time_s"] = round(time.time() - t1, 2)

        # Phase 2: Sequential extraction with carryforward
        t2 = time.time()
        all_extractions = self._extract_with_carryforward(
            fields, chunk_texts, field_chunk_map, cluster_label, rlm_context, meta,
        )
        meta["extraction_time_s"] = round(time.time() - t2, 2)

        # Phase 3: Judge compilation
        t3 = time.time()
        compiled = self._judge_compile(fields, all_extractions, chunk_texts, meta)
        meta["judge_time_s"] = round(time.time() - t3, 2)

        # Phase 4: Reviewer validation
        t4 = time.time()
        compiled = self._reviewer_validate(compiled, chunk_texts, meta)
        meta["reviewer_time_s"] = round(time.time() - t4, 2)

        meta["total_time_s"] = round(time.time() - t0, 2)

        # Compute aggregate grounding stats
        compiled_values = [cv for cv in compiled.values() if cv.value is not None]
        if compiled_values:
            n_verified = sum(1 for cv in compiled_values if cv.grounding_level == "verified")
            n_partial = sum(1 for cv in compiled_values if cv.grounding_level == "partial")
            n_halluc = sum(1 for cv in compiled_values if cv.grounding_level == "hallucinated")
            total = len(compiled_values)
            meta["n_compiled_values"] = total
            meta["verified_rate"] = round(n_verified / total, 3) if total else 0
            meta["partial_rate"] = round(n_partial / total, 3) if total else 0
            meta["hallucination_rate"] = round(n_halluc / total, 3) if total else 0
            meta["grounded_rate"] = round((n_verified + n_partial) / total, 3) if total else 0
        else:
            meta["n_compiled_values"] = 0
            meta["verified_rate"] = 0
            meta["partial_rate"] = 0
            meta["hallucination_rate"] = 0
            meta["grounded_rate"] = 0

        return ExtractionResult(
            compiled=compiled,
            chunk_extractions=all_extractions,
            meta=meta,
        )

    # ── Phase 1: Embedding-based Retrieval ────────────────────────────

    def _retrieve_chunks_per_field(
        self, fields: list[dict], chunk_texts: list[str],
    ) -> dict[str, list[int]]:
        """For each field, find top-K most relevant chunk indices."""
        if not chunk_texts:
            return {}

        # Build query per field: name + description + examples
        field_queries = []
        for f in fields:
            query = f"{f['name'].replace('_', ' ')}: {f.get('description', '')}"
            examples = f.get("examples", [])
            if examples:
                query += f" (e.g., {examples[0]})"
            field_queries.append(query)

        # Encode everything
        all_texts = field_queries + chunk_texts
        all_vecs = self.embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)
        field_vecs = all_vecs[:len(field_queries)]
        chunk_vecs = all_vecs[len(field_queries):]

        # Cosine similarity: fields × chunks
        sim_matrix = np.dot(field_vecs, chunk_vecs.T)

        # Top-K per field
        k = min(self.config.top_k_chunks, len(chunk_texts))
        field_chunk_map = {}
        for i, f in enumerate(fields):
            top_indices = np.argsort(-sim_matrix[i])[:k].tolist()
            # Sort by document order for sequential processing
            top_indices.sort()
            field_chunk_map[f["name"]] = top_indices

        return field_chunk_map

    # ── Phase 2: Sequential Extraction with Carryforward ──────────────

    def _extract_with_carryforward(
        self,
        fields: list[dict],
        chunk_texts: list[str],
        field_chunk_map: dict[str, list[int]],
        cluster_label: str,
        rlm_context: Optional[dict],
        meta: dict,
    ) -> list[ChunkExtraction]:
        """Extract from chunks sequentially, passing carryforward context."""

        # Determine which chunks to process (union of all field retrievals)
        all_chunk_indices = sorted(set(
            idx for indices in field_chunk_map.values() for idx in indices
        ))

        fields_schema = _build_fields_schema(fields)
        rlm_section = _build_rlm_context_section(rlm_context)

        # Carryforward state: best extraction per field so far
        carryforward: dict[str, tuple[str, float, int]] = {}  # field_name → (value, conf, chunk_idx)
        all_extractions: list[ChunkExtraction] = []

        for chunk_idx in all_chunk_indices:
            chunk_text = chunk_texts[chunk_idx]

            # Build carryforward context string
            carryforward_lines = []
            if carryforward:
                for fname, (val, conf, src_idx) in sorted(carryforward.items()):
                    carryforward_lines.append(
                        f"  {fname}: \"{val}\" (from chunk {src_idx}, confidence={conf:.2f})"
                    )
            carryforward_str = "\n".join(carryforward_lines) if carryforward_lines else "(none — this is the first chunk)"

            # Determine which fields are relevant to this chunk
            relevant_fields = [
                f for f in fields
                if chunk_idx in field_chunk_map.get(f["name"], [])
            ]
            if not relevant_fields:
                continue

            relevant_schema = _build_fields_schema(relevant_fields)

            prompt = EXTRACT_WITH_CARRYFORWARD_PROMPT.format(
                cluster_label=cluster_label,
                rlm_context_section=rlm_section,
                carryforward_context=carryforward_str,
                chunk_index=chunk_idx,
                chunk_text=chunk_text[:self.config.max_chunk_chars],
                fields_schema=relevant_schema,
            )

            parsed = _call_llm(
                self.client, self.deployment, prompt,
                temperature=self.config.temperature,
                max_retries=self.config.max_retries_llm,
            )

            if not parsed:
                continue

            valid_names = {f["name"] for f in relevant_fields}
            extracted_in_chunk = set()

            for fname, value in parsed.items():
                if fname not in valid_names:
                    continue
                if value is None or str(value).strip() in ("", "null", "N/A", "n/a"):
                    continue

                value_str = str(value)
                grounding_level, grounding_score = _ground_value(value_str, chunk_text)

                ext = ChunkExtraction(
                    chunk_index=chunk_idx,
                    field_name=fname,
                    value=value_str,
                    confidence=grounding_score,
                    grounding_level=grounding_level,
                    grounding_score=grounding_score,
                )
                all_extractions.append(ext)
                extracted_in_chunk.add(fname)

                # Update carryforward if this is better than what we have
                if fname not in carryforward or grounding_score > carryforward[fname][1]:
                    carryforward[fname] = (value_str, grounding_score, chunk_idx)

            # ── Agentic retry for missing fields ──────────────────────
            missing_fields = [
                f for f in relevant_fields
                if f["name"] not in extracted_in_chunk
                and f["name"] not in carryforward
            ]

            for retry_attempt in range(self.config.max_retries):
                if not missing_fields:
                    break

                still_missing = []
                for f in missing_fields:
                    retry_prompt = RETRY_EXTRACT_PROMPT.format(
                        field_name=f["name"],
                        field_description=f.get("description", ""),
                        chunk_text=chunk_text[:self.config.max_chunk_chars],
                    )

                    retry_parsed = _call_llm(
                        self.client, self.deployment, retry_prompt,
                        temperature=self.config.retry_temperature,
                        max_retries=2,
                        max_tokens=200,
                    )
                    meta["n_retries"] += 1

                    if retry_parsed:
                        value = retry_parsed.get(f["name"])
                        if value and str(value).strip() not in ("", "null", "N/A"):
                            value_str = str(value)
                            g_level, g_score = _ground_value(value_str, chunk_text)
                            ext = ChunkExtraction(
                                chunk_index=chunk_idx,
                                field_name=f["name"],
                                value=value_str,
                                confidence=g_score,
                                grounding_level=g_level,
                                grounding_score=g_score,
                                was_retry=True,
                            )
                            all_extractions.append(ext)

                            if f["name"] not in carryforward or g_score > carryforward[f["name"]][1]:
                                carryforward[f["name"]] = (value_str, g_score, chunk_idx)
                        else:
                            still_missing.append(f)
                    else:
                        still_missing.append(f)

                missing_fields = still_missing

        return all_extractions

    # ── Phase 3: Judge LLM Compilation ────────────────────────────────

    def _judge_compile(
        self,
        fields: list[dict],
        all_extractions: list[ChunkExtraction],
        chunk_texts: list[str],
        meta: dict,
    ) -> dict[str, CompiledValue]:
        """Compile chunk-level extractions into one canonical value per field."""

        # Group extractions by field
        by_field: dict[str, list[ChunkExtraction]] = {}
        for ext in all_extractions:
            by_field.setdefault(ext.field_name, []).append(ext)

        compiled = {}

        for f in fields:
            fname = f["name"]
            exts = by_field.get(fname, [])

            if not exts:
                compiled[fname] = CompiledValue(
                    field_name=fname,
                    value=None,
                    confidence=0.0,
                    source_chunk_indices=[],
                    grounding_level="missing",
                )
                continue

            # Get unique values
            unique_values = {}
            for ext in exts:
                if ext.value not in unique_values or ext.confidence > unique_values[ext.value].confidence:
                    unique_values[ext.value] = ext

            if len(unique_values) == 1:
                # All agree — no need for judge
                best = next(iter(unique_values.values()))
                compiled[fname] = CompiledValue(
                    field_name=fname,
                    value=best.value,
                    confidence=best.confidence,
                    source_chunk_indices=[ext.chunk_index for ext in exts],
                    grounding_level=best.grounding_level,
                    raw_extractions=exts,
                )
                continue

            if not self.config.enable_judge:
                # No judge — pick highest confidence
                best = max(exts, key=lambda e: e.confidence)
                compiled[fname] = CompiledValue(
                    field_name=fname,
                    value=best.value,
                    confidence=best.confidence,
                    source_chunk_indices=[best.chunk_index],
                    grounding_level=best.grounding_level,
                    raw_extractions=exts,
                )
                continue

            # Multiple distinct values → call judge
            summary_lines = []
            for ext in sorted(exts, key=lambda e: e.chunk_index):
                summary_lines.append(
                    f"  Chunk {ext.chunk_index}: \"{ext.value}\" "
                    f"(confidence={ext.confidence:.2f}, grounding={ext.grounding_level}"
                    f"{', retry' if ext.was_retry else ''})"
                )

            judge_prompt = JUDGE_COMPILATION_PROMPT.format(
                field_name=fname,
                field_description=f.get("description", ""),
                field_type=f.get("type", "text"),
                extractions_summary="\n".join(summary_lines),
            )

            judge_result = _call_llm(
                self.client, self.deployment, judge_prompt,
                temperature=self.config.judge_temperature,
                max_retries=2,
                max_tokens=300,
            )
            meta["n_judge_calls"] += 1

            if judge_result:
                j_value = judge_result.get("value")
                j_conf = judge_result.get("confidence", 0.5)
                j_reason = judge_result.get("reasoning", "")
                j_source = judge_result.get("source_chunk", -1)

                if j_value and str(j_value).strip() not in ("", "null"):
                    # Ground the judge's chosen value against source
                    if j_source >= 0 and j_source < len(chunk_texts):
                        g_level, g_score = _ground_value(str(j_value), chunk_texts[j_source])
                    else:
                        # Judge picked a value but didn't cite source — check all
                        g_level = "hallucinated"
                        g_score = 0.0
                        for ext in exts:
                            gl, gs = _ground_value(str(j_value), chunk_texts[ext.chunk_index])
                            if gs > g_score:
                                g_level, g_score = gl, gs
                                j_source = ext.chunk_index

                    compiled[fname] = CompiledValue(
                        field_name=fname,
                        value=str(j_value),
                        confidence=float(j_conf),
                        source_chunk_indices=[j_source] if j_source >= 0 else [e.chunk_index for e in exts],
                        grounding_level=g_level,
                        judge_reasoning=j_reason,
                        conflict_resolved=True,
                        raw_extractions=exts,
                    )
                    meta["n_conflicts_resolved"] += 1
                    continue

            # Judge failed or returned null — fall back to best confidence
            best = max(exts, key=lambda e: e.confidence)
            compiled[fname] = CompiledValue(
                field_name=fname,
                value=best.value,
                confidence=best.confidence,
                source_chunk_indices=[best.chunk_index],
                grounding_level=best.grounding_level,
                raw_extractions=exts,
            )

        return compiled

    # ── Phase 4: Reviewer Validation ──────────────────────────────────

    def _reviewer_validate(
        self,
        compiled: dict[str, CompiledValue],
        chunk_texts: list[str],
        meta: dict,
    ) -> dict[str, CompiledValue]:
        """Post-compilation grounding check via reviewer LLM."""

        if not self.config.enable_reviewer:
            return compiled

        for fname, cv in compiled.items():
            if cv.value is None:
                continue

            # Skip if already verified by string matching
            if cv.grounding_level == "verbatim":
                cv.reviewer_verdict = "verified"
                continue

            # Find the best source chunk for review
            source_idx = cv.source_chunk_indices[0] if cv.source_chunk_indices else 0
            if source_idx >= len(chunk_texts):
                source_idx = 0

            reviewer_prompt = REVIEWER_PROMPT.format(
                field_name=fname,
                field_type="text",
                value=cv.value,
                chunk_index=source_idx,
                chunk_text=chunk_texts[source_idx][:self.config.max_chunk_chars],
            )

            result = _call_llm(
                self.client, self.deployment, reviewer_prompt,
                temperature=self.config.reviewer_temperature,
                max_retries=2,
                max_tokens=200,
            )
            meta["n_reviewer_calls"] += 1

            if result:
                verdict = result.get("verdict", "hallucinated")
                evidence = result.get("evidence", "")
                cv.reviewer_verdict = verdict
                # Update grounding level based on reviewer
                if verdict == "verified":
                    cv.grounding_level = "verified"
                    cv.confidence = max(cv.confidence, 0.85)
                elif verdict == "partial":
                    cv.grounding_level = "partial"
                elif verdict == "hallucinated":
                    cv.grounding_level = "hallucinated"
                    cv.confidence = min(cv.confidence, 0.2)

        return compiled

    # ── Convenience: extract to flat dict ─────────────────────────────

    def extract_flat(
        self,
        fields: list[dict],
        chunk_texts: list[str],
        cluster_label: str,
        rlm_context: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict]:
        """
        Extract and return a flat result dict compatible with comparison scripts.

        Returns:
            (field_results, meta)
            field_results: {field_name: {value, confidence, grounding: {level, score}}}
        """
        result = self.extract(fields, chunk_texts, cluster_label, rlm_context)

        field_results = {}
        for fname, cv in result.compiled.items():
            field_results[fname] = {
                "value": cv.value,
                "confidence": cv.confidence,
                "grounding": {
                    "level": cv.grounding_level,
                    "score": 1.0 if cv.grounding_level == "verified" else
                             0.8 if cv.grounding_level == "partial" else
                             0.0,
                },
                "source_chunks": cv.source_chunk_indices,
                "judge_reasoning": cv.judge_reasoning,
                "reviewer_verdict": cv.reviewer_verdict,
                "conflict_resolved": cv.conflict_resolved,
            }

        return field_results, result.meta


# ══════════════════════════════════════════════════════════════════════════════
# CLI — Standalone test
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run agentic extraction on a single cluster for testing."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Agentic extraction pipeline")
    parser.add_argument("--db", default="data/cuad_510_demo.duckdb",
                        help="Path to DuckDB file")
    parser.add_argument("--cluster-id", default=None,
                        help="Specific cluster ID (default: first with >10 chunks)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable judge compilation")
    parser.add_argument("--no-reviewer", action="store_true",
                        help="Disable reviewer validation")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max agentic retries for missing fields")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-K chunks per field for retrieval")
    args = parser.parse_args()

    import duckdb

    config = AgenticConfig(
        enable_judge=not args.no_judge,
        enable_reviewer=not args.no_reviewer,
        max_retries=args.max_retries,
        top_k_chunks=args.top_k,
    )
    pipeline = AgenticExtractionPipeline.from_env(config=config)

    conn = duckdb.connect(args.db, read_only=True)

    # Pick a cluster
    if args.cluster_id:
        cluster = conn.execute(
            "SELECT cluster_id, label, keywords FROM clusters WHERE cluster_id = ?",
            [args.cluster_id],
        ).fetchdf().to_dict("records")[0]
    else:
        cluster = conn.execute(
            "SELECT cluster_id, label, keywords FROM clusters WHERE chunk_count > 10 ORDER BY chunk_count DESC LIMIT 1",
        ).fetchdf().to_dict("records")[0]

    cid = cluster["cluster_id"]
    label = cluster["label"]
    keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else []

    chunks = conn.execute("""
        SELECT ch.chunk_text FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY ch.chunk_index
    """, [cid]).fetchdf()["chunk_text"].tolist()

    conn.close()

    logger.info(f"Cluster: {label} ({len(chunks)} chunks)")

    # Discover fields using DSPy pipeline
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.rlm_v2_dspy import ClusterDiscoveryPipeline, _configure_dspy_lm

    _configure_dspy_lm()
    disc_pipeline = ClusterDiscoveryPipeline()

    logger.info("Running field discovery...")
    t0 = time.time()
    disc_result = disc_pipeline(
        cluster_label=label,
        keywords=", ".join(keywords[:10]),
        chunk_texts=chunks,
    )
    fields = disc_result.fields
    disc_meta = disc_result.meta
    disc_time = round(time.time() - t0, 2)
    logger.info(f"Discovered {len(fields)} fields in {disc_time}s")

    # Build RLM context
    rlm_context = {
        "clause_types": disc_meta.get("clause_types", []),
        "intents": disc_meta.get("intents", []),
        "obligations": disc_meta.get("obligations", []),
    }

    # Run agentic extraction
    logger.info("Running agentic extraction...")
    result = pipeline.extract(
        fields=fields,
        chunk_texts=chunks[:20],  # cap for testing
        cluster_label=label,
        rlm_context=rlm_context,
    )

    # Print results
    print(f"\n{'='*80}")
    print(f" Agentic Extraction Results: {label}")
    print(f"{'='*80}")
    print(f" Fields: {len(fields)} discovered, {result.meta.get('n_compiled_values', 0)} extracted")
    print(f" Retries: {result.meta['n_retries']}")
    print(f" Judge calls: {result.meta['n_judge_calls']}")
    print(f" Reviewer calls: {result.meta['n_reviewer_calls']}")
    print(f" Conflicts resolved: {result.meta['n_conflicts_resolved']}")
    print(f" Grounded rate: {result.meta.get('grounded_rate', 0):.1%}")
    print(f" Hallucination rate: {result.meta.get('hallucination_rate', 0):.1%}")
    print(f" Total time: {result.meta['total_time_s']}s")
    print(f"{'─'*80}")

    for fname, cv in result.compiled.items():
        status = "✓" if cv.grounding_level in ("verified", "partial", "verbatim") else "✗" if cv.value else "·"
        val_display = f'"{cv.value[:60]}"' if cv.value else "null"
        print(f" {status} {fname}: {val_display}")
        if cv.judge_reasoning:
            print(f"   Judge: {cv.judge_reasoning}")
        if cv.reviewer_verdict:
            print(f"   Reviewer: {cv.reviewer_verdict}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
