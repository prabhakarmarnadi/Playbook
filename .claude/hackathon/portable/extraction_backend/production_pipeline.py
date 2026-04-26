"""
Production Pipeline — Consolidated async clustering → discovery → extraction.
=============================================================================

Single-entry-point, high-performance pipeline combining:
  - EVoC clustering (macro + micro) with Optuna tuning
  - Field discovery (Hybrid: Standard + RLM+LB with V2 rolling state)
  - Cluster-conditioned extraction with RLM context enrichment
  - Intent discovery with dynamic registry
  - Ontology refinement (outlier assignment, field dedup, label synthesis)
  - Knowledge graph construction

Async architecture:
  - asyncio.Semaphore-gated concurrent LLM calls
  - Batch embedding with GPU
  - Parallel cluster processing with configurable concurrency
  - Graceful error handling with per-cluster fallback

Usage:
    # CLI
    python -m core.production_pipeline /path/to/pdfs --account-id acct_001

    # Python
    import asyncio
    from core.production_pipeline import run_production_pipeline
    result = asyncio.run(run_production_pipeline("/path/to/pdfs"))
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """All tunables for the production pipeline."""

    # Clustering
    mode: str = "hybrid"                      # hybrid | clause | macro-micro
    device: str = "cuda"
    use_optuna: bool = True
    optuna_trials: int = 30
    evoc_noise_level: float = 0.3

    # Chunking
    use_semantic_chunking: bool = True
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2000
    extraction_chunk_chars: int = 16000   # ~4000 tok for extraction chunks
    max_chunks_per_doc: int = 80

    # LLM concurrency
    max_llm_concurrency: int = 8              # async semaphore limit
    max_extraction_workers: int = 5           # per-cluster extraction parallelism
    max_intent_workers: int = 5

    # Field discovery
    discovery_mode: str = "hybrid"            # standard | rlm | hybrid | rlm_v2 | rlm_v3 | rlm_v3_unified
    max_sample_chunks: int = 30
    max_fields_per_cluster: int = 15
    min_cluster_chunks: int = 5
    discovery_temperature: float = 0.2

    # Extraction
    extraction_backend: str = "sequential"    # sequential | rag
    min_extraction_confidence: float = 0.5
    extract_values: bool = True

    # Intent
    run_intents: bool = True
    min_intent_clause_chars: int = 80
    intent_match_threshold: float = 0.85

    # Post-pipeline
    run_refinement: bool = True
    run_knowledge_graph: bool = True
    outlier_min_similarity: float = 0.35

    # Paths (resolved at runtime)
    db_path: str | None = None
    lance_path: str | None = None
    artifacts_dir: str | None = None
    max_docs: int | None = None


# ── Async LLM Client ─────────────────────────────────────────────────────────


class AsyncLLMClient:
    """Rate-limited async wrapper around Azure OpenAI for production throughput."""

    def __init__(self, max_concurrency: int = 8):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self._client = None
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        self._call_count = 0
        self._total_latency = 0.0

    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        return self._client

    async def complete_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        retries: int = 3,
        model: str | None = None,
    ) -> dict | None:
        """Async JSON completion with semaphore rate-limiting and retries."""
        deployment = model or self._deployment
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    result = await loop.run_in_executor(
                        self._executor,
                        partial(self._sync_call, prompt, temperature, max_tokens, deployment),
                    )
                    self._call_count += 1
                    self._total_latency += time.time() - t0
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"JSON parse error (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"LLM call error (attempt {attempt + 1}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))
        return None

    def _sync_call(self, prompt: str, temperature: float, max_tokens: int, deployment: str | None = None) -> dict:
        model = deployment or self._deployment
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency_s": round(self._total_latency, 1),
            "avg_latency_s": round(
                self._total_latency / max(1, self._call_count), 2
            ),
        }


# ── Field Discovery Prompts ──────────────────────────────────────────────────

STANDARD_DISCOVERY_PROMPT = """You are a legal contract analyst. Analyze these clause excerpts that all belong to the same clause type: "{cluster_label}".

Keywords: {keywords}

CLAUSE EXCERPTS:
{chunks_text}

Identify the recurring **fields** (structured data points) across these clauses. For each field provide:
1. A snake_case field name (max 40 chars)
2. Data type: text, date, duration, currency, percentage, integer, boolean, entity_name, or enum
3. One-sentence description
4. 2-3 example values from the excerpts

Rules:
- Only include fields appearing in at least 2 excerpts
- Return between 2 and {max_fields} fields
- Focus on extractable structured data, not free text
- Prefer specific fields (e.g., "renewal_term" over "terms")

Respond with ONLY JSON: {{"fields": [...]}}
Each field: {{"name": "...", "type": "...", "description": "...", "examples": ["...", "..."]}}"""

EXTRACTION_PROMPT = """Extract field values from this contract clause.
IMPORTANT: Only extract values that appear explicitly in the text. Do NOT hallucinate or infer values not stated.

Clause type: "{cluster_label}"
{context_section}
Clause text:
{chunk_text}

Fields to extract (with descriptions and example values from similar clauses):
{fields_schema}

Rules:
1. Extract ONLY values explicitly stated in the text above.
2. If a field's value is not clearly stated in the text, return null for that field.
3. Use the exact wording from the text when possible (prefer verbatim quotes).
4. Do not copy the field name or clause type as a value.
5. Do not fabricate URLs, dates, or names not in the text.

Respond with ONLY a JSON object mapping field names to extracted values.
Use null if the field is not present. For booleans use true/false.

Example: {{"notice_period": "30 days", "auto_renewal": true, "termination_fee": null}}"""

EXTRACTION_WITH_CARRYFORWARD_PROMPT = """Extract field values from this contract chunk.
IMPORTANT: Only extract values that appear explicitly in the text. Do NOT hallucinate or infer values not stated.

Clause type: "{cluster_label}"
{context_section}
--- Knowledge from previous chunks ---
{carryforward_notes}
--- End previous knowledge ---

Chunk text (chunk {chunk_number} of {total_chunks}):
{chunk_text}

Fields to extract (with descriptions and example values from similar clauses):
{fields_schema}

Rules:
1. Extract ONLY values explicitly stated in the text above.
2. If a field's value is not clearly stated, return null for that field.
3. Use exact wording from the text when possible (prefer verbatim quotes).
4. Do not copy the field name or clause type as a value.
5. Do not fabricate URLs, dates, or names not in the text.
6. Use knowledge from previous chunks to resolve cross-references (e.g., party names defined earlier).

Respond with ONLY a JSON object with two keys:
- "extractions": object mapping field names to extracted values (null if not present)
- "carryforward": a short string of notes for future chunks — key definitions, party names,
  cross-references, or context that will help extract fields from later chunks.
  Keep under 300 words. Only note genuinely useful context.

Example:
{{
  "extractions": {{"notice_period": "30 days", "auto_renewal": true, "termination_fee": null}},
  "carryforward": "Party A = Acme Corp. Effective Date = Jan 1 2024. Agreement governed by Delaware law."
}}"""

JUDGE_COMPILATION_PROMPT = """You are a legal extraction judge. Your job is to compile per-chunk extractions into final field values.

Clause type: "{cluster_label}"
{context_section}

Field definitions:
{fields_schema}

Per-chunk extractions (from {n_chunks} chunks):
{chunk_extractions_json}

For each field, review ALL extracted values across chunks and:
1. Pick the best value with the strongest textual evidence.
2. If multiple chunks extracted different values for the same field, prefer the most specific and complete value.
3. If values conflict, pick the one with the most context/evidence.
4. Validate formatting against the field type (date fields should be dates, currency should include amounts, etc.).
5. If no chunk extracted a value for a field, return null.

Respond with ONLY a JSON object mapping field names to final values.
Use null for fields with no valid extraction.

Example: {{"notice_period": "30 days written notice", "governing_law": "State of Delaware", "termination_fee": null}}"""

LLM_LABEL_PROMPT = """You are labeling clusters of text from {context}.

This cluster has {count} items. Here are representative excerpts:
{samples}

What type of content is in this cluster?

Respond in JSON:
{{"label": "A clear 2-5 word descriptive name", "description": "One sentence describing what this cluster covers"}}"""

INTENT_PROMPT = """You are a legal contract analyst. Analyze this clause and identify every distinct legal intent it contains.

A "legal intent" is a single obligation, right, prohibition, condition, or declaration that has legal effect.

Clause type: "{clause_type}"
Clause text:
{clause_text}

For each intent found, provide:
- label: a short snake_case name
- summary: one sentence describing what this intent does
- party_from: who bears the obligation/restriction (or "none" if declarative)
- party_to: who benefits (or "none" if declarative)
- attributes: a dict of intent-specific properties stated in the text

If the clause has NO legal intent, return an empty array.

Respond with ONLY a JSON object:
{{"intents": [
  {{"label": "...", "summary": "...", "party_from": "...", "party_to": "...", "attributes": {{...}}}},
]}}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

import re

VALID_FIELD_TYPES = {"text", "date", "duration", "currency", "percentage",
                     "integer", "boolean", "entity_name", "enum"}


def _normalize_field(f: dict, max_name_len: int = 40) -> dict | None:
    """Normalize a field dict from LLM output. Returns None if invalid."""
    name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
    if not name:
        return None
    name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:max_name_len]
    ftype = f.get("type", f.get("value_type", "text")).strip().lower()
    type_map = {"string": "text", "money": "currency", "ref": "text"}
    ftype = type_map.get(ftype, ftype)
    if ftype not in VALID_FIELD_TYPES:
        ftype = "text"
    return {
        "name": name,
        "type": ftype,
        "description": f.get("description", ""),
        "examples": f.get("examples") or ([f.get("span_text")] if f.get("span_text") else []),
        "source": f.get("source", "standard"),
    }


def _confidence_score(value: str, source_text: str) -> float:
    """Compute grounded confidence score for an extraction."""
    v_str = str(value)
    chunk_lower = source_text.lower()
    v_lower = v_str.lower().strip()
    conf = 0.5
    if v_lower in chunk_lower:
        conf += 0.3
    elif len(v_lower) > 5 and any(w in chunk_lower for w in v_lower.split()[:3]):
        conf += 0.15
    if len(v_str) > 3:
        conf += 0.05
    if any(c.isdigit() for c in v_str):
        conf += 0.05
    if len(v_str) > 50:
        conf -= 0.1
    if v_str.startswith("{") or v_str.startswith("["):
        conf -= 0.15
    return max(0.1, min(1.0, round(conf, 3)))


def _merge_field_lists(
    fields_a: list[dict], fields_b: list[dict], max_fields: int = 15
) -> list[dict]:
    """Merge two field lists by exact + fuzzy name matching."""
    map_a = {f["name"]: f for f in fields_a}
    map_b = {f["name"]: f for f in fields_b}
    common = set(map_a) & set(map_b)
    remaining_a = {n for n in map_a if n not in common}
    remaining_b = {n for n in map_b if n not in common}

    fuzzy_pairs, matched_b = [], set()
    for na in sorted(remaining_a):
        for nb in sorted(remaining_b):
            if nb in matched_b:
                continue
            if na in nb or nb in na:
                fuzzy_pairs.append((na, nb))
                matched_b.add(nb)
                break

    def _pick(fa, fb):
        best = fa if len(fa.get("description", "")) >= len(fb.get("description", "")) else fb
        examples = list(dict.fromkeys((fa.get("examples") or []) + (fb.get("examples") or [])))[:5]
        return {**best, "examples": examples, "source": "both"}

    merged = []
    for n in sorted(common):
        merged.append(_pick(map_a[n], map_b[n]))
    fuzzy_a_used = set()
    for na, nb in fuzzy_pairs:
        entry = _pick(map_a[na], map_b[nb])
        entry["name"] = na if len(na) <= len(nb) else nb
        entry["source"] = "both_fuzzy"
        merged.append(entry)
        fuzzy_a_used.add(na)
    for na in sorted(remaining_a - fuzzy_a_used):
        merged.append({**map_a[na], "source": "standard"})
    for nb in sorted(remaining_b - matched_b):
        merged.append({**map_b[nb], "source": "rlm"})
    return merged[:max_fields]


# ── Core Async Operations ─────────────────────────────────────────────────────


async def _discover_fields_standard(
    llm: AsyncLLMClient,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: PipelineConfig,
) -> list[dict]:
    """Standard single-shot field discovery for a cluster."""
    sample = chunk_texts[:config.max_sample_chunks]
    chunks_text = "\n\n".join(
        f"[Excerpt {i+1}]\n{text[:1500]}" for i, text in enumerate(sample)
    )
    prompt = STANDARD_DISCOVERY_PROMPT.format(
        cluster_label=cluster_label,
        keywords=", ".join(keywords[:10]),
        chunks_text=chunks_text,
        max_fields=config.max_fields_per_cluster,
    )
    parsed = await llm.complete_json(prompt, temperature=config.discovery_temperature)
    if not parsed:
        return []
    raw_fields = parsed.get("fields", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(raw_fields, list):
        return []
    return [f for f in (_normalize_field(f) for f in raw_fields[:config.max_fields_per_cluster]) if f]


async def _discover_fields_rlm_v2(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: PipelineConfig,
) -> tuple[list[dict], dict]:
    """RLM+LB with V2 rolling state discovery (runs in executor)."""
    loop = asyncio.get_event_loop()

    def _run_sync():
        from core.rlm_with_memory import RLMWithLatentBriefing
        import dspy

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        sub_model = os.getenv("RLM_SUB_MODEL", "")

        retry_kwargs = dict(num_retries=5)
        if endpoint and api_key:
            lm = dspy.LM(f"azure/{deployment}", api_key=api_key,
                         api_base=endpoint, api_version=api_version, **retry_kwargs)
        else:
            lm = dspy.LM(f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}",
                         api_key=os.getenv("OPENAI_API_KEY", ""), **retry_kwargs)
        dspy.configure(lm=lm)
        sub_lm = dspy.LM(sub_model, num_retries=5) if sub_model else lm

        class FieldDiscoverySig(dspy.Signature):
            """Discover recurring structured fields from legal clause excerpts.
            Return fields_json as a JSON list of {name, type, description, examples}."""
            chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
            cluster_label: str = dspy.InputField(desc="The clause type label")
            keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
            max_fields: str = dspy.InputField(desc="Maximum fields to return")
            fields_json: str = dspy.OutputField(desc="JSON array of field objects")

        rlm = RLMWithLatentBriefing(
            signature=FieldDiscoverySig,
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "25")),
            max_llm_calls=int(os.getenv("RLM_MAX_LLM_CALLS", "50")),
            verbose=False,
            sub_lm=sub_lm,
            briefing_tau=1.5,
            briefing_max_tokens=2000,
            adaptive_tau=True,
        )

        # Use the V2 rolling state path
        fields, meta = rlm.discover_with_rolling_state(
            cluster_label=cluster_label,
            keywords=keywords,
            chunk_texts=chunk_texts,
            max_sample_chunks=config.max_sample_chunks,
            max_fields=config.max_fields_per_cluster,
        )
        return fields, meta

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.warning(f"RLM+LB V2 discovery failed for '{cluster_label}': {e}")
        return [], {"error": str(e)}


async def _discover_fields_rlm_v3(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: PipelineConfig,
) -> tuple[list[dict], dict]:
    """RLM V3 — agentic REPL + structured state + dual-mode workers."""
    loop = asyncio.get_event_loop()

    def _run_sync():
        from core.rlm_v3 import RLMV3, RLMV3Config

        v3_config = RLMV3Config(
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "25")),
            max_llm_calls=int(os.getenv("RLM_MAX_LLM_CALLS", "50")),
            max_sample_chunks=config.max_sample_chunks,
            max_fields=config.max_fields_per_cluster,
        )
        v3 = RLMV3(config=v3_config)
        return v3.discover(cluster_label, keywords, chunk_texts)

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.warning(f"RLM V3 discovery failed for '{cluster_label}': {e}")
        return [], {"error": str(e)}


async def _discover_and_extract_rlm_v3(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    chunk_ids: list[str],
    agreement_ids: list[str],
    config: PipelineConfig,
) -> tuple[list[dict], list[dict], dict]:
    """RLM V3 unified — discover fields + extract values in a single REPL pass."""
    loop = asyncio.get_event_loop()

    def _run_sync():
        from core.rlm_v3 import RLMV3, RLMV3Config

        v3_config = RLMV3Config(
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "25")),
            max_llm_calls=int(os.getenv("RLM_MAX_LLM_CALLS", "50")),
            max_sample_chunks=config.max_sample_chunks,
            max_fields=config.max_fields_per_cluster,
            extract_inline=True,
            min_extraction_confidence=config.min_extraction_confidence,
        )
        v3 = RLMV3(config=v3_config)
        return v3.discover_and_extract(
            cluster_label, keywords, chunk_texts,
            chunk_ids=chunk_ids, agreement_ids=agreement_ids,
        )

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.warning(f"RLM V3 unified failed for '{cluster_label}': {e}")
        return [], [], {"error": str(e)}


async def _discover_fields_hybrid(
    llm: AsyncLLMClient,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: PipelineConfig,
) -> tuple[list[dict], dict]:
    """Hybrid discovery: Standard + RLM+LB V2, merged."""
    meta: dict = {"standard_ok": False, "rlm_ok": False}

    # Run standard (fast) and RLM (deep) concurrently
    std_task = asyncio.create_task(
        _discover_fields_standard(llm, cluster_label, keywords, chunk_texts, config)
    )
    rlm_task = asyncio.create_task(
        _discover_fields_rlm_v2(cluster_label, keywords, chunk_texts, config)
    )

    fields_std = await std_task
    meta["standard_ok"] = bool(fields_std)
    meta["standard_count"] = len(fields_std)

    fields_rlm, rlm_meta = await rlm_task
    meta["rlm_ok"] = bool(fields_rlm)
    meta["rlm_count"] = len(fields_rlm)
    meta["rlm_meta"] = rlm_meta

    if fields_std and fields_rlm:
        merged = _merge_field_lists(fields_std, fields_rlm, config.max_fields_per_cluster)
        meta["strategy"] = "merged"
    elif fields_std:
        merged = [{**f, "source": "standard"} for f in fields_std]
        meta["strategy"] = "standard_only"
    elif fields_rlm:
        merged = [{**f, "source": "rlm"} for f in fields_rlm]
        meta["strategy"] = "rlm_only"
    else:
        merged = []
        meta["strategy"] = "none"

    meta["merged_count"] = len(merged)
    return merged, meta


def _get_embedder(config: PipelineConfig):
    """Get or create the singleton embedder for RAG retrieval."""
    from core.embedder import Embedder
    return Embedder.get_instance()


async def _extract_chunk(
    llm: AsyncLLMClient,
    cluster_label: str,
    chunk_text: str,
    fields: list[dict],
    context_section: str = "",
) -> dict[str, tuple[str | None, float]]:
    """Extract fields from a single chunk with confidence scoring."""
    schema_lines = []
    for f in fields:
        line = f"- {f['name']} ({f['type']}): {f['description']}"
        examples = f.get("examples", [])
        if examples:
            ex_str = ", ".join(f'"{e}"' for e in examples[:2] if e)
            if ex_str:
                line += f"  [examples: {ex_str}]"
        schema_lines.append(line)
    fields_schema = "\n".join(schema_lines)

    prompt = EXTRACTION_PROMPT.format(
        cluster_label=cluster_label,
        context_section=context_section,
        chunk_text=chunk_text[:4000],
        fields_schema=fields_schema,
    )
    parsed = await llm.complete_json(prompt, temperature=0.0, max_tokens=1000)
    if not parsed or not isinstance(parsed, dict):
        return {}

    valid_names = {f["name"] for f in fields}
    results = {}
    for k, v in parsed.items():
        if k not in valid_names:
            continue
        if v is None or v == "" or str(v).lower() in ("null", "n/a", "none"):
            continue
        v_str = str(v)
        conf = _confidence_score(v_str, chunk_text)
        results[k] = (v_str, conf)
    return results


def _build_fields_schema(fields: list[dict]) -> str:
    """Build field schema string for extraction prompts."""
    schema_lines = []
    for f in fields:
        line = f"- {f['name']} ({f['type']}): {f['description']}"
        examples = f.get("examples", [])
        if examples:
            ex_str = ", ".join(f'"{e}"' for e in examples[:2] if e)
            if ex_str:
                line += f"  [examples: {ex_str}]"
        schema_lines.append(line)
    return "\n".join(schema_lines)


async def _extract_chunk_with_carryforward(
    llm: AsyncLLMClient,
    cluster_label: str,
    chunk_text: str,
    fields: list[dict],
    context_section: str = "",
    carryforward_notes: str = "",
    chunk_number: int = 1,
    total_chunks: int = 1,
) -> tuple[dict[str, tuple[str | None, float]], str]:
    """
    Extract fields from a chunk with knowledge carryforward.

    Returns:
        (extractions dict, updated carryforward notes string)
    """
    fields_schema = _build_fields_schema(fields)

    prompt = EXTRACTION_WITH_CARRYFORWARD_PROMPT.format(
        cluster_label=cluster_label,
        context_section=context_section,
        carryforward_notes=carryforward_notes or "(first chunk — no prior context)",
        chunk_number=chunk_number,
        total_chunks=total_chunks,
        chunk_text=chunk_text[:8000],  # extraction chunks are larger
        fields_schema=fields_schema,
    )
    parsed = await llm.complete_json(prompt, temperature=0.0, max_tokens=1500)
    if not parsed or not isinstance(parsed, dict):
        return {}, carryforward_notes

    # Extract carryforward notes for next chunk
    new_carryforward = str(parsed.get("carryforward", "")).strip()
    if new_carryforward:
        # Append to existing notes, keeping total under ~1000 chars
        if carryforward_notes:
            combined = carryforward_notes + "\n" + new_carryforward
            if len(combined) > 1000:
                # Keep the newest notes, trim oldest
                combined = combined[-1000:]
            carryforward_notes = combined
        else:
            carryforward_notes = new_carryforward[:1000]
    # Parse extractions
    raw_extractions = parsed.get("extractions", parsed)
    if not isinstance(raw_extractions, dict):
        return {}, carryforward_notes

    valid_names = {f["name"] for f in fields}
    results = {}
    for k, v in raw_extractions.items():
        if k not in valid_names or k == "carryforward":
            continue
        if v is None or v == "" or str(v).lower() in ("null", "n/a", "none"):
            continue
        v_str = str(v)
        conf = _confidence_score(v_str, chunk_text)
        results[k] = (v_str, conf)
    return results, carryforward_notes


async def _judge_compile_extractions(
    llm: AsyncLLMClient,
    cluster_label: str,
    fields: list[dict],
    per_chunk_extractions: list[dict[str, tuple[str | None, float]]],
    context_section: str = "",
) -> dict[str, tuple[str | None, float]]:
    """
    Judge LLM compiles all chunk-level extractions into final field values.

    Resolves conflicts when multiple chunks extract different values for the same field.
    Validates formatting against field type definitions.
    """
    if not per_chunk_extractions:
        return {}

    fields_schema = _build_fields_schema(fields)

    # Build per-chunk extraction summary for the judge
    chunk_summaries = []
    for i, chunk_ext in enumerate(per_chunk_extractions):
        if not chunk_ext:
            continue
        entries = {}
        for field_name, (value, confidence) in chunk_ext.items():
            entries[field_name] = {"value": value, "confidence": round(confidence, 2)}
        if entries:
            chunk_summaries.append({"chunk": i + 1, "extractions": entries})

    if not chunk_summaries:
        return {}

    prompt = JUDGE_COMPILATION_PROMPT.format(
        cluster_label=cluster_label,
        context_section=context_section,
        fields_schema=fields_schema,
        n_chunks=len(per_chunk_extractions),
        chunk_extractions_json=json.dumps(chunk_summaries, indent=2)[:6000],
    )
    parsed = await llm.complete_json(prompt, temperature=0.0, max_tokens=1500)
    if not parsed or not isinstance(parsed, dict):
        # Fallback: merge by highest confidence
        return _merge_extractions_by_confidence(per_chunk_extractions, fields)

    valid_names = {f["name"] for f in fields}
    results = {}
    for k, v in parsed.items():
        if k not in valid_names:
            continue
        if v is None or v == "" or str(v).lower() in ("null", "n/a", "none"):
            continue
        v_str = str(v)
        # Judge-compiled values get a confidence boost
        conf = min(0.95, 0.7 + 0.1)  # base 0.7 for judge-validated
        # Check if any chunk had this value for grounding
        for chunk_ext in per_chunk_extractions:
            if k in chunk_ext:
                orig_val, orig_conf = chunk_ext[k]
                if orig_val and v_str.lower() in orig_val.lower() or orig_val.lower() in v_str.lower():
                    conf = min(0.95, max(conf, orig_conf + 0.1))
                    break
        results[k] = (v_str, conf)
    return results


def _merge_extractions_by_confidence(
    per_chunk_extractions: list[dict[str, tuple[str | None, float]]],
    fields: list[dict],
) -> dict[str, tuple[str | None, float]]:
    """Fallback: merge extractions by picking the highest-confidence value per field."""
    merged: dict[str, tuple[str | None, float]] = {}
    for chunk_ext in per_chunk_extractions:
        for field_name, (value, confidence) in chunk_ext.items():
            if field_name not in merged or confidence > merged[field_name][1]:
                merged[field_name] = (value, confidence)
    return merged


async def _extract_intents_for_clause(
    llm: AsyncLLMClient,
    clause_text: str,
    clause_type: str,
    max_chars: int = 3000,
) -> list[dict]:
    """Discover intents for a single clause."""
    prompt = INTENT_PROMPT.format(
        clause_type=clause_type,
        clause_text=clause_text[:max_chars],
    )
    parsed = await llm.complete_json(prompt, temperature=0.0, max_tokens=1500)
    if not parsed or not isinstance(parsed, dict):
        return []
    raw_intents = parsed.get("intents", [])
    if not isinstance(raw_intents, list):
        return []

    results = []
    for raw in raw_intents:
        if not isinstance(raw, dict):
            continue
        label = raw.get("label", "").strip()
        if not label:
            continue
        label = label.lower().replace(" ", "_").replace("-", "_")
        attrs = raw.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        clean_attrs = {
            str(k).strip().lower().replace(" ", "_"): v
            for k, v in attrs.items()
            if k and v is not None and str(v).strip().lower() not in ("", "none", "null", "n/a")
        }
        results.append({
            "label": label,
            "summary": str(raw.get("summary", "")).strip(),
            "party_from": str(raw.get("party_from", "none")).strip(),
            "party_to": str(raw.get("party_to", "none")).strip(),
            "attributes": clean_attrs,
        })
    return results


# ── Cluster Processing ────────────────────────────────────────────────────────


async def _process_cluster(
    cluster: dict,
    chunk_rows: list[dict],
    store,
    llm: AsyncLLMClient,
    config: PipelineConfig,
    cluster_idx: int,
    total_clusters: int,
    progress,
    extraction_chunks_by_agreement: dict[str, list[dict]] | None = None,
) -> dict:
    """Process a single cluster: discover fields → extract values."""
    cluster_id = cluster["cluster_id"]
    cluster_label = cluster["label"] or "Unknown"
    try:
        keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(
            cluster.get("keywords"), str) else (cluster.get("keywords") or [])
    except json.JSONDecodeError:
        keywords = []

    chunk_texts = [r["chunk_text"] for r in chunk_rows]
    stats = {"cluster_id": cluster_id, "label": cluster_label, "fields": 0, "extractions": 0}

    progress("discovery", f"  [{cluster_idx+1}/{total_clusters}] {cluster_label} ({len(chunk_rows)} chunks)")

    # ── Step 1: Discover fields (+ inline extractions for unified mode) ──
    rlm_meta = None
    inline_extractions: list[dict] = []
    already_extracted_indices: set[int] = set()

    if config.discovery_mode == "rlm_v3_unified":
        chunk_ids = [r["chunk_id"] for r in chunk_rows]
        agr_ids = [r["agreement_id"] for r in chunk_rows]
        fields, inline_extractions, rlm_meta = await _discover_and_extract_rlm_v3(
            cluster_label, keywords, chunk_texts,
            chunk_ids, agr_ids, config)
        if rlm_meta:
            already_extracted_indices = set(
                rlm_meta.get("remaining_chunk_indices", [])
            )
            # Invert: remaining_chunk_indices are NOT extracted yet
            all_indices = set(range(len(chunk_rows)))
            already_extracted_indices = all_indices - already_extracted_indices
        if not fields:
            fields = await _discover_fields_standard(
                llm, cluster_label, keywords, chunk_texts, config)
    elif config.discovery_mode == "hybrid":
        fields, rlm_meta = await _discover_fields_hybrid(
            llm, cluster_label, keywords, chunk_texts, config)
    elif config.discovery_mode == "rlm_v3":
        fields, rlm_meta = await _discover_fields_rlm_v3(
            cluster_label, keywords, chunk_texts, config)
        if not fields:
            fields = await _discover_fields_standard(
                llm, cluster_label, keywords, chunk_texts, config)
    elif config.discovery_mode in ("rlm", "rlm_v2"):
        fields, rlm_meta = await _discover_fields_rlm_v2(
            cluster_label, keywords, chunk_texts, config)
        if not fields:
            fields = await _discover_fields_standard(
                llm, cluster_label, keywords, chunk_texts, config)
    else:
        fields = await _discover_fields_standard(
            llm, cluster_label, keywords, chunk_texts, config)

    if not fields:
        progress("discovery", f"    No fields discovered")
        return stats

    # Persist field definitions
    field_id_map = {}
    for f in fields:
        field_id = f"fld_{uuid.uuid4().hex[:10]}"
        field_id_map[f["name"]] = field_id
        store.upsert_field(
            field_id=field_id, cluster_id=cluster_id,
            name=f["name"], field_type=f["type"],
            description=f["description"], source=f.get("source", "auto_discovered"),
            examples=f.get("examples", []),
        )
    stats["fields"] = len(fields)

    field_names = [f["name"] for f in fields]
    strategy = rlm_meta.get("strategy", config.discovery_mode) if rlm_meta else config.discovery_mode
    progress("discovery", f"    {len(fields)} fields ({strategy}): {field_names}")

    # ── Step 2: Extract values ──
    if not config.extract_values:
        return stats

    cluster_extractions = 0

    # 2a: Persist inline extractions from unified discovery
    for ext in inline_extractions:
        field_id = field_id_map.get(ext["field_name"])
        if not field_id:
            continue
        store.insert_extraction(
            extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
            agreement_id=ext.get("agreement_id", ""),
            field_id=field_id,
            value=ext["value"],
            confidence=ext.get("confidence", 0.5),
            source_chunk_id=ext.get("chunk_id", ""),
        )
        cluster_extractions += 1

    if inline_extractions:
        progress("extraction", f"    {cluster_extractions} inline extractions from discovery")

    # 2b: Extract remaining chunks that weren't covered by inline extraction
    remaining_rows = [
        r for i, r in enumerate(chunk_rows)
        if i not in already_extracted_indices
    ]
    if not remaining_rows:
        stats["extractions"] = cluster_extractions
        progress("extraction", f"    {cluster_extractions} total extractions (all inline)")
        return stats

    # Build context section from RLM V2 meta if available
    context_section = ""
    if rlm_meta:
        parts = []
        clause_types = rlm_meta.get("clause_types", [])
        if clause_types:
            ct_str = "; ".join(
                f"{ct.get('name', '')}: {ct.get('emergent_description', '')}"
                for ct in clause_types[:3]
            )
            parts.append(f"Clause sub-types: {ct_str}")
        obligations = rlm_meta.get("obligations", [])
        if obligations:
            ob_str = "; ".join(
                f"{ob.get('obligor', '')} must {ob.get('action', '')}"
                for ob in obligations[:3]
            )
            parts.append(f"Key obligations: {ob_str}")
        if parts:
            context_section = "\n".join(parts) + "\n"

    # ── RAG backend: multi-round retrieval + agentic retry ──
    if config.extraction_backend == "rag":
        from core.rag_extractor import RAGExtractor, RAGConfig

        embedder = _get_embedder(config)
        rag_config = RAGConfig(
            max_sample_docs=int(os.getenv("RAG_MAX_SAMPLE_DOCS", "10")),
            max_retry_rounds=int(os.getenv("RAG_MAX_RETRY_ROUNDS", "3")),
            enable_reviewer=os.getenv("RAG_ENABLE_REVIEWER", "1") == "1",
            min_extraction_confidence=config.min_extraction_confidence,
        )
        rag_extractor = RAGExtractor(
            store=store, llm=llm, embed_fn=embedder.embed, config=rag_config,
        )
        try:
            batch = await rag_extractor.extract_cluster(
                cluster_id=cluster_id,
                fields=fields,
                cluster_label=cluster_label,
                cluster_keywords=keywords,
                context_section=context_section,
            )
            cluster_extractions += batch.meta.get("n_persisted", 0)
            n_retries = batch.meta.get("total_retries", 0)
            n_rejected = batch.meta.get("total_reviewer_rejections", 0)
            stats["extractions"] = cluster_extractions
            stats["rag_meta"] = batch.meta
            n_inline = len(inline_extractions)
            progress(
                "extraction",
                f"    {cluster_extractions} extractions (RAG: "
                f"{batch.meta.get('n_agreements_sampled', 0)} docs sampled, "
                f"{n_retries} retries, {n_rejected} reviewer rejections"
                f"{f', {n_inline} inline' if n_inline else ''})"
            )
            return stats
        except Exception as e:
            logger.warning(f"RAG extraction failed for '{cluster_label}', falling back to sequential: {e}")
            # Fall through to sequential extraction

    # ── Sequential backend: per-chunk extraction with carryforward + judge ──
    # Group remaining rows by agreement for per-document sequential processing
    agreement_rows: dict[str, list[dict]] = {}
    for row in remaining_rows:
        agr_id = row["agreement_id"]
        if agr_id not in agreement_rows:
            agreement_rows[agr_id] = []
        agreement_rows[agr_id].append(row)

    for agr_id, agr_rows in agreement_rows.items():
        # Use extraction chunks (larger, ~4000 tok) if available for this agreement
        if extraction_chunks_by_agreement and agr_id in extraction_chunks_by_agreement:
            ext_chunks = extraction_chunks_by_agreement[agr_id]
            # Sort by chunk_index to process in document order
            ext_chunks = sorted(ext_chunks, key=lambda c: c.get("chunk_index", 0))
        else:
            # Fallback: use the embedding chunks in document order
            ext_chunks = sorted(agr_rows, key=lambda r: r.get("chunk_index", 0))

        # Sequential extraction with knowledge carryforward
        carryforward_notes = ""
        per_chunk_extractions: list[dict[str, tuple[str | None, float]]] = []
        chunk_source_ids: list[str] = []

        for chunk_num, chunk_data in enumerate(ext_chunks):
            chunk_text = chunk_data.get("text", chunk_data.get("chunk_text", ""))
            chunk_id = chunk_data.get("chunk_id", "")
            try:
                result, carryforward_notes = await _extract_chunk_with_carryforward(
                    llm, cluster_label, chunk_text, fields, context_section,
                    carryforward_notes=carryforward_notes,
                    chunk_number=chunk_num + 1,
                    total_chunks=len(ext_chunks),
                )
                per_chunk_extractions.append(result)
                chunk_source_ids.append(chunk_id)
            except Exception as e:
                logger.warning(f"Carryforward extraction failed for chunk {chunk_num}: {e}")
                per_chunk_extractions.append({})
                chunk_source_ids.append(chunk_id)

        # Judge compilation: resolve conflicts across chunks
        if len(per_chunk_extractions) > 1:
            try:
                judged = await _judge_compile_extractions(
                    llm, cluster_label, fields, per_chunk_extractions, context_section,
                )
            except Exception as e:
                logger.warning(f"Judge compilation failed for '{cluster_label}' agr={agr_id}: {e}")
                judged = _merge_extractions_by_confidence(per_chunk_extractions, fields)
        elif per_chunk_extractions:
            judged = per_chunk_extractions[0]
        else:
            judged = {}

        # Persist judge-compiled extractions
        # Find the source chunk with the best evidence for each field
        for field_name, (value, confidence) in judged.items():
            if confidence < config.min_extraction_confidence:
                continue
            field_id = field_id_map.get(field_name)
            if not field_id:
                continue
            # Find which chunk originally extracted this value for source attribution
            source_chunk_id = ""
            for ci, chunk_ext in enumerate(per_chunk_extractions):
                if field_name in chunk_ext:
                    source_chunk_id = chunk_source_ids[ci] if ci < len(chunk_source_ids) else ""
                    break
            store.insert_extraction(
                extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
                agreement_id=agr_id,
                field_id=field_id,
                value=value,
                confidence=confidence,
                source_chunk_id=source_chunk_id,
            )
            cluster_extractions += 1

    stats["extractions"] = cluster_extractions
    n_inline = len(inline_extractions)
    n_downstream = cluster_extractions - n_inline
    if n_inline > 0:
        progress("extraction", f"    {cluster_extractions} extractions ({n_inline} inline + {n_downstream} sequential+judge, {len(remaining_rows)} chunks)")
    else:
        progress("extraction", f"    {cluster_extractions} extractions (sequential+judge)")
    return stats


# ── Intent Processing ─────────────────────────────────────────────────────────


async def _process_intents(
    store,
    llm: AsyncLLMClient,
    config: PipelineConfig,
    progress,
) -> dict:
    """Discover intents for all clustered clauses."""
    from core.intent_extractor import IntentRegistry, _normalize_intent

    registry = IntentRegistry(store)
    clauses = store.conn.execute("""
        SELECT cl.clause_id, cl.agreement_id, cl.clause_type_id,
               cl.full_text, c.label as clause_type_label
        FROM clauses cl
        JOIN clusters c ON cl.clause_type_id = c.cluster_id
        WHERE cl.clause_type_id IS NOT NULL
          AND cl.clause_id NOT IN (SELECT DISTINCT clause_id FROM clause_intents)
          AND LENGTH(cl.full_text) >= ?
        ORDER BY c.label, cl.clause_id
    """, [config.min_intent_clause_chars]).fetchdf().to_dict("records")

    if not clauses:
        progress("intent", "No new clauses for intent extraction")
        return {"total_clauses": 0, "total_intents": 0}

    progress("intent", f"Discovering intents for {len(clauses)} clauses...")

    total_intents = 0
    total_failed = 0
    batch_size = 50

    for batch_start in range(0, len(clauses), batch_size):
        batch = clauses[batch_start:batch_start + batch_size]
        batch_intents = 0

        # Launch all intent extractions for the batch
        tasks = [
            _extract_intents_for_clause(
                llm, cr["full_text"], cr["clause_type_label"],
                config.min_intent_clause_chars,
            )
            for cr in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                total_failed += 1
                continue
            cr = batch[i]
            for intent in result:
                normalized = _normalize_intent(intent) if hasattr(intent, 'get') else None
                if not normalized:
                    # Minimal normalization for our format
                    label = intent.get("label", "")
                    if not label:
                        continue
                    normalized = intent
                    normalized.setdefault("intent_hash", "")

                intent_type_id = registry.match_or_register(normalized)
                store.insert_clause_intent(
                    intent_id=f"ci_{uuid.uuid4().hex[:10]}",
                    clause_id=cr["clause_id"],
                    agreement_id=cr["agreement_id"],
                    clause_type_id=cr["clause_type_id"],
                    intent_type_id=intent_type_id,
                    intent_label=normalized.get("label", ""),
                    summary=normalized.get("summary", ""),
                    party_from=normalized.get("party_from", "none"),
                    party_to=normalized.get("party_to", "none"),
                    attributes=normalized.get("attributes", {}),
                    intent_hash=normalized.get("intent_hash", ""),
                )
                batch_intents += 1

        total_intents += batch_intents
        progress("intent", f"  Batch {batch_start//batch_size + 1}: "
                 f"{batch_intents} intents from {len(batch)} clauses")

    progress("intent", f"Done: {total_intents} intents, {total_failed} failed")
    return {"total_clauses": len(clauses), "total_intents": total_intents,
            "total_failed": total_failed}


# ── Post-Pipeline ─────────────────────────────────────────────────────────────


async def _run_refinement(store, embedder, llm, config, progress):
    """Ontology refinement: outlier assignment + field dedup + label synthesis."""
    from core.ontology_refiner import OntologyRefiner
    loop = asyncio.get_event_loop()

    refiner = OntologyRefiner(store, embedder)

    progress("refine", "Assigning outlier clauses...")
    outlier_stats = await loop.run_in_executor(
        None, partial(refiner.assign_outliers, min_similarity=config.outlier_min_similarity)
    )
    progress("refine", f"  Assigned {outlier_stats.get('assigned', 0)} outliers")

    progress("refine", "Deduplicating fields...")
    dedup_stats = await loop.run_in_executor(None, refiner.deduplicate_fields)
    progress("refine", f"  Removed {dedup_stats.get('removed', 0)} duplicate fields")

    progress("refine", "Synthesizing cluster labels...")
    label_stats = await loop.run_in_executor(None, refiner.synthesize_labels)
    progress("refine", f"  Labeled {label_stats.get('labeled', 0)} clusters")

    return {"outlier": outlier_stats, "dedup": dedup_stats, "labels": label_stats}


async def _run_knowledge_graph(store, progress):
    """Build the contract knowledge graph."""
    from core.knowledge_graph import ContractKnowledgeGraph
    loop = asyncio.get_event_loop()

    progress("kg", "Building knowledge graph...")
    kg = ContractKnowledgeGraph(store)
    stats = await loop.run_in_executor(None, kg.build)
    progress("kg", f"  {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
    return {"nodes": kg.G.number_of_nodes(), "edges": kg.G.number_of_edges()}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


async def run_production_pipeline(
    pdf_dir: str | Path,
    account_id: str = "default",
    config: PipelineConfig | None = None,
    callback=None,
) -> dict:
    """
    Full production pipeline: parse → chunk → embed → cluster → discover → extract → refine.

    Returns dict with all stats and timing information.
    """
    config = config or PipelineConfig()
    t0 = time.time()
    stage_timings: dict[str, float] = {}

    def progress(stage: str, msg: str):
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:6.1f}s] [{stage}] {msg}")
        if callback:
            callback(stage, msg)

    def _time(name):
        return _StageTimer(name, stage_timings)

    # ── Paths ──
    from config import DATA_DIR
    artifacts_root = Path(config.artifacts_dir) if config.artifacts_dir else DATA_DIR / "artifacts"
    db = Path(config.db_path) if config.db_path else DATA_DIR / f"{account_id}.duckdb"
    lance = Path(config.lance_path) if config.lance_path else DATA_DIR / f"{account_id}.lance"

    from core.store import ClusteringStore
    from core.lancedb_store import LanceVectorStore
    from core.nomic_embedder import NomicEmbedder

    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    llm = AsyncLLMClient(max_concurrency=config.max_llm_concurrency)

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    store.start_run(run_id, "init")
    progress("init", f"Production pipeline: mode={config.mode}, discovery={config.discovery_mode}, "
             f"account={account_id}, device={config.device}")

    pipeline_stats: dict[str, Any] = {
        "run_id": run_id, "account_id": account_id,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
    }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Parse PDFs
    # ══════════════════════════════════════════════════════════════════════
    with _time("parse"):
        from core.pdf_parser import parse_pdf_directory, load_parsed_agreements, PARSED_TEXT_DIR

        cached = load_parsed_agreements()
        pdf_dir_resolved = str(Path(pdf_dir).resolve())
        manifest_path = PARSED_TEXT_DIR / "manifest.json"
        cache_hit = False
        if cached and manifest_path.exists():
            with open(manifest_path) as _f:
                _manifest = json.load(_f)
            if str(Path(_manifest.get("source_dir", "")).resolve()) == pdf_dir_resolved:
                cache_hit = True

        if cache_hit:
            progress("parse", f"Using {len(cached)} cached PDFs")
            parsed_docs = cached
        else:
            progress("parse", f"Parsing PDFs from {pdf_dir}...")
            parsed_docs = parse_pdf_directory(pdf_dir)

        if config.max_docs and len(parsed_docs) > config.max_docs:
            parsed_docs = parsed_docs[:config.max_docs]

        agreements = [
            {"agreement_id": d["agreement_id"], "filename": d["filename"], "raw_text": d["raw_text"]}
            for d in parsed_docs if len(d.get("raw_text", "")) > 100
        ]
        progress("parse", f"{len(agreements)} valid PDFs")

    if not agreements:
        progress("error", "No documents found")
        return {"error": "no_documents", **pipeline_stats}

    pipeline_stats["n_documents"] = len(agreements)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Chunk (3 chunk types: embedding, extraction, clause)
    # ══════════════════════════════════════════════════════════════════════
    with _time("chunk"):
        embedder = NomicEmbedder.get_instance(device=config.device)
        embedder.warmup()

        from core.chunker import (
            SemanticChunkerConfig, _structural_clause_split, _split_sentences,
            _combine_sentence_buffers, _compute_breakpoints, _assemble_chunks,
            _approx_token_count, extract_clause_header, sliding_window_chunk,
            cohesion_subsplit_clauses, generate_typed_chunks,
        )

        if config.use_semantic_chunking:
            progress("chunk", "Semantic chunking (3 chunk types: embedding, extraction, clause)...")
            chunker_config = SemanticChunkerConfig(
                buffer_size=1, breakpoint_percentile=90.0,
                min_chunk_chars=config.min_chunk_chars,
                max_chunk_chars=config.max_chunk_chars,
                extraction_max_chunk_chars=config.extraction_chunk_chars,
                clause_split=True,
            )
            _embed_fn_chunk = lambda texts: embedder.embed_for_clustering(texts, batch_size=8)

            all_chunks = []          # embedding chunks (used for clustering, RAG, dedup)
            all_extraction_chunks = []  # extraction chunks (used by sequential extraction)
            all_clauses = []         # clause records

            for agr in agreements:
                raw = agr["raw_text"]
                typed = generate_typed_chunks(raw, embed_fn=_embed_fn_chunk, config=chunker_config)

                # ── Embedding chunks ──
                for c in typed["embedding"]:
                    c["agreement_id"] = agr["agreement_id"]
                    c["clause_id"] = None  # linked below if clause data available
                all_chunks.extend(typed["embedding"])

                # ── Extraction chunks ──
                for c in typed["extraction"]:
                    c["agreement_id"] = agr["agreement_id"]
                all_extraction_chunks.extend(typed["extraction"])

                # ── Clause chunks → clause records ──
                clause_chunk_data = typed["clause"]
                for ci, cc in enumerate(clause_chunk_data):
                    cc["agreement_id"] = agr["agreement_id"]
                    header = extract_clause_header(cc["text"])
                    clause_id = f"cls_{uuid.uuid4().hex[:10]}"
                    cc["clause_id"] = clause_id

                    # Find embedding chunks that overlap this clause
                    linked_chunk_ids = []
                    for ec in typed["embedding"]:
                        if (ec["char_start"] < cc["char_end"] and ec["char_end"] > cc["char_start"]):
                            linked_chunk_ids.append(ec["chunk_id"])
                            if ec.get("clause_id") is None:
                                ec["clause_id"] = clause_id

                    all_clauses.append({
                        "clause_id": clause_id,
                        "agreement_id": agr["agreement_id"],
                        "clause_number": header["clause_number"] if header else "",
                        "clause_title": header["clause_title"] if header else "",
                        "heading_level": header["heading_level"] if header else 0,
                        "clause_index": ci,
                        "full_text": cc["text"],
                        "token_count": cc["token_count"],
                        "chunk_ids": linked_chunk_ids,
                    })

                # Cap embedding chunks per doc
                doc_embed_count = len(typed["embedding"])
                if doc_embed_count > config.max_chunks_per_doc:
                    excess = doc_embed_count - config.max_chunks_per_doc
                    all_chunks = all_chunks[:-excess]

            progress("chunk", f"{len(all_chunks)} embedding chunks, {len(all_extraction_chunks)} extraction chunks, {len(all_clauses)} clauses from {len(agreements)} docs")
        else:
            progress("chunk", "Sliding window chunking (fast mode)...")
            all_chunks = []
            all_extraction_chunks = []
            all_clauses = []
            for agr in agreements:
                chunks = sliding_window_chunk(agr["raw_text"])
                for c in chunks:
                    c["agreement_id"] = agr["agreement_id"]
                    c["chunk_type"] = "embedding"
                all_chunks.extend(chunks)

        progress("chunk", f"{len(all_chunks)} chunks, {len(all_clauses)} clauses from {len(agreements)} docs")

    pipeline_stats["n_chunks"] = len(all_chunks)
    pipeline_stats["n_extraction_chunks"] = len(all_extraction_chunks)
    pipeline_stats["n_clauses"] = len(all_clauses)

    # Store agreements
    for agr in agreements:
        from core.chunker import get_document_summary
        summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
        store.insert_agreement(agr["agreement_id"], agr.get("filename", ""), agr["raw_text"], summary)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Embed
    # ══════════════════════════════════════════════════════════════════════
    with _time("embed"):
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_ids = [c["chunk_id"] for c in all_chunks]

        progress("embed", f"Embedding {len(all_chunks)} chunks on {config.device}...")
        chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True, batch_size=8)

        _embed_fn = lambda texts, **kw: embedder.embed_for_clustering(texts, batch_size=8, **kw)
        from core.macro_clusterer import get_field_summaries
        summaries = get_field_summaries(agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn)
        summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)

        progress("embed", f"Embedded: {chunk_embeddings.shape[0]} chunks + {summary_embeddings.shape[0]} summaries")

    # Persist to DB
    with _time("persist"):
        for i, chunk in enumerate(all_chunks):
            store.insert_chunk(
                chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
                chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
                clause_id=chunk.get("clause_id"),
            )
        for clause in all_clauses:
            store.insert_clause(
                clause["clause_id"], clause["agreement_id"],
                clause.get("clause_number", ""), clause.get("clause_title", ""),
                clause.get("heading_level", 0), clause.get("clause_index", 0),
                clause.get("full_text", ""), clause.get("token_count", 0),
            )
        # LanceDB
        lance_store.upsert_chunks(
            chunk_ids=chunk_ids, vectors=chunk_embeddings,
            texts=chunk_texts,
            agreement_ids=[c["agreement_id"] for c in all_chunks],
        )
        lance_store.upsert_documents(
            agreement_ids=[a["agreement_id"] for a in agreements],
            vectors=summary_embeddings,
            filenames=[a.get("filename", "") for a in agreements],
            summaries=summaries,
        )
        progress("persist", f"Stored {len(all_chunks)} chunks + {len(all_clauses)} clauses")

    store.update_run(run_id, "embed_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: EVoC Clustering
    # ══════════════════════════════════════════════════════════════════════
    with _time("cluster"):
        from core.evoc_pipeline import (
            evoc_cluster, evoc_optimize, evoc_cluster_with_target,
            _select_best_layer, _llm_label_clusters,
        )
        from core.topic_merger import HierarchicalTopicMerger

        n_docs = len(agreements)
        total_clusters = 0

        # ── Macro clustering (document-level) ──
        if config.mode in ("macro-micro", "hybrid"):
            progress("cluster", f"EVoC macro clustering {n_docs} documents...")
            if n_docs < 10:
                macro_labels = np.zeros(n_docs, dtype=int)
                domain_id = f"d_{uuid.uuid4().hex[:8]}"
                centroid = summary_embeddings.mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                store.upsert_domain(domain_id, "All Documents", "Single domain (few docs)",
                                    n_docs, 1.0, centroid)
                for i in range(n_docs):
                    store.update_agreement_domain(agreements[i]["agreement_id"], domain_id,
                                                  float(np.dot(summary_embeddings[i], centroid)))
                domain_map = {0: {"domain_id": domain_id, "agreement_indices": list(range(n_docs)),
                                  "label": "All Documents"}}
            else:
                macro_labels, macro_layers, macro_persist = evoc_cluster(
                    summary_embeddings,
                    base_min_cluster_size=max(3, n_docs // 50),
                    noise_level=config.evoc_noise_level,
                    n_neighbors=min(15, n_docs - 1),
                )
                macro_labels, _, macro_lscores = _select_best_layer(
                    macro_layers, summary_embeddings, macro_persist, prefer_granularity=True)

                llm_labels = _llm_label_clusters(macro_labels, summaries, "legal agreement summaries")
                domain_map = {}
                for tid in sorted(set(t for t in macro_labels if t != -1)):
                    mask = [i for i, t in enumerate(macro_labels) if t == tid]
                    centroid = summary_embeddings[mask].mean(axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-12
                    info = llm_labels.get(tid, {"label": f"Domain_{tid}", "description": ""})
                    did = f"d_{uuid.uuid4().hex[:8]}"
                    domain_map[tid] = {"domain_id": did, "agreement_indices": mask,
                                       "label": info["label"]}
                    store.upsert_domain(did, info["label"], info.get("description", ""),
                                        len(mask),
                                        float(np.mean([np.dot(summary_embeddings[i], centroid) for i in mask])),
                                        centroid)
                    for idx in mask:
                        store.update_agreement_domain(
                            agreements[idx]["agreement_id"], did,
                            float(np.dot(summary_embeddings[idx], centroid)))

                # Outlier domain
                outlier_mask = [i for i, t in enumerate(macro_labels) if t == -1]
                if outlier_mask:
                    odid = f"d_{uuid.uuid4().hex[:8]}"
                    store.upsert_domain(odid, "Uncategorized", "Unassigned", len(outlier_mask))
                    domain_map[-1] = {"domain_id": odid, "agreement_indices": outlier_mask,
                                      "label": "Uncategorized"}
                    for idx in outlier_mask:
                        store.update_agreement_domain(agreements[idx]["agreement_id"], odid, 0.0)

            progress("cluster", f"  {len([k for k in domain_map if k != -1])} macro domains")

        # ── Clause-level clustering (micro or flat) ──
        if config.mode in ("clause", "hybrid"):
            n_chunks = len(all_chunks)
            progress("cluster", f"EVoC clause clustering on {n_chunks} chunks...")

            if config.use_optuna:
                best_params, clause_labels, best_score = evoc_optimize(
                    chunk_embeddings, n_trials=config.optuna_trials,
                    min_clusters=3, progress_fn=progress,
                )
                progress("cluster", f"  Optuna best: score={best_score:.4f}, params={best_params}")
            else:
                clause_labels, clause_layers, clause_persist = evoc_cluster(
                    chunk_embeddings,
                    base_min_cluster_size=max(3, n_chunks // 100),
                    noise_level=config.evoc_noise_level,
                    n_neighbors=min(15, n_chunks - 1),
                )
                clause_labels, _, _ = _select_best_layer(
                    clause_layers, chunk_embeddings, clause_persist)

            # Save all layers
            store.save_cluster_layers(
                "clause", [clause_labels], chunk_ids, [1.0], 0, [{}])

            # LLM-label clause clusters
            clause_cluster_labels = _llm_label_clusters(
                clause_labels, chunk_texts, "legal contract clauses")

            # Merge similar clusters
            merger = HierarchicalTopicMerger()
            unique_ctopics = sorted(set(t for t in clause_labels if t != -1))
            for tid in unique_ctopics:
                mask = [i for i, t in enumerate(clause_labels) if t == tid]
                centroid = chunk_embeddings[mask].mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                info = clause_cluster_labels.get(tid, {"label": f"Clause_{tid}", "description": ""})
                cid = f"c_{uuid.uuid4().hex[:8]}"

                # Determine domain
                domain_id = ""
                if config.mode == "hybrid" and domain_map:
                    agr_ids_in_cluster = set(all_chunks[i]["agreement_id"] for i in mask)
                    for dtid, dr in domain_map.items():
                        if dtid == -1:
                            continue
                        dr_agr_ids = set(agreements[idx]["agreement_id"] for idx in dr["agreement_indices"])
                        if agr_ids_in_cluster & dr_agr_ids:
                            domain_id = dr["domain_id"]
                            break

                store.upsert_cluster(
                    cluster_id=cid, domain_id=domain_id or None,
                    label=info["label"], description=info.get("description", ""),
                    keywords=[],
                    chunk_count=len(mask),
                    agreement_count=len(set(all_chunks[i]["agreement_id"] for i in mask)),
                    quality_score=0.0,
                    centroid=centroid,
                )
                store.save_centroid(cid, centroid, len(mask))

                for idx in mask:
                    store.insert_cluster_assignment(
                        all_chunks[idx]["chunk_id"], cid,
                        float(np.dot(chunk_embeddings[idx], centroid)),
                    )

                # Assign clauses to cluster
                clause_ids_in_cluster = set(all_chunks[i].get("clause_id") for i in mask)
                for cls_id in clause_ids_in_cluster:
                    if cls_id:
                        try:
                            store.conn.execute(
                                "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ? AND clause_type_id IS NULL",
                                [cid, cls_id])
                        except Exception:
                            pass

                total_clusters += 1

            progress("cluster", f"  {total_clusters} clause clusters")

        elif config.mode == "macro-micro":
            progress("cluster", "Micro-clustering within domains...")
            for dtid, dr in domain_map.items():
                if dtid == -1:
                    continue
                domain_agr_ids = set(agreements[i]["agreement_id"] for i in dr["agreement_indices"])
                domain_chunk_idx = [i for i, c in enumerate(all_chunks) if c["agreement_id"] in domain_agr_ids]
                if len(domain_chunk_idx) < 5:
                    continue

                d_texts = [all_chunks[i]["text"] for i in domain_chunk_idx]
                d_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_idx]
                d_embs = chunk_embeddings[domain_chunk_idx]

                micro_labels, micro_layers = evoc_cluster_with_target(
                    d_embs, noise_level=config.evoc_noise_level,
                    n_neighbors=min(15, len(d_embs) - 1),
                )
                micro_labels, _, _ = _select_best_layer(micro_layers, d_embs, [1.0])
                micro_llm_labels = _llm_label_clusters(
                    micro_labels, d_texts, f"clauses from {dr['label']}")

                for tid in sorted(set(t for t in micro_labels if t != -1)):
                    m_mask = [i for i, t in enumerate(micro_labels) if t == tid]
                    centroid = d_embs[m_mask].mean(axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-12
                    info = micro_llm_labels.get(tid, {"label": f"Type_{tid}", "description": ""})
                    cid = f"c_{uuid.uuid4().hex[:8]}"

                    store.upsert_cluster(
                        cluster_id=cid, domain_id=dr["domain_id"],
                        label=info["label"], description=info.get("description", ""),
                        keywords=[],
                        chunk_count=len(m_mask),
                        agreement_count=len(set(all_chunks[domain_chunk_idx[i]]["agreement_id"] for i in m_mask)),
                        quality_score=0.0,
                        centroid=centroid,
                    )
                    store.save_centroid(cid, centroid, len(m_mask))
                    for mi in m_mask:
                        orig_idx = domain_chunk_idx[mi]
                        store.insert_cluster_assignment(
                            all_chunks[orig_idx]["chunk_id"], cid,
                            float(np.dot(chunk_embeddings[orig_idx], centroid)),
                        )
                        cls_id = all_chunks[orig_idx].get("clause_id")
                        if cls_id:
                            try:
                                store.conn.execute(
                                    "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ? AND clause_type_id IS NULL",
                                    [cid, cls_id])
                            except Exception:
                                pass
                    total_clusters += 1

            progress("cluster", f"  {total_clusters} micro clusters across domains")

    pipeline_stats["n_clusters"] = total_clusters
    store.update_run(run_id, "cluster_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Field Discovery + Extraction (async)
    # ══════════════════════════════════════════════════════════════════════
    with _time("discovery_extraction"):
        # Build extraction chunks index for sequential extraction with carryforward
        extraction_chunks_by_agreement: dict[str, list[dict]] = {}
        if config.extraction_backend == "sequential" and all_extraction_chunks:
            for ec in all_extraction_chunks:
                agr_id = ec.get("agreement_id", "")
                if agr_id:
                    if agr_id not in extraction_chunks_by_agreement:
                        extraction_chunks_by_agreement[agr_id] = []
                    extraction_chunks_by_agreement[agr_id].append(ec)

        clusters = store.conn.execute("""
            SELECT c.cluster_id, c.domain_id, c.label, c.keywords, c.chunk_count
            FROM clusters c WHERE c.chunk_count >= ?
            ORDER BY c.chunk_count DESC
        """, [config.min_cluster_chunks]).fetchdf().to_dict("records")

        progress("discovery", f"Processing {len(clusters)} clusters (discovery + extraction)...")

        total_fields = 0
        total_extractions = 0

        # Process clusters sequentially to avoid overloading (LLM concurrency is
        # managed by the semaphore within each cluster's extraction)
        for ci, cluster in enumerate(clusters):
            cid = cluster["cluster_id"]
            chunk_rows = store.conn.execute("""
                SELECT ch.chunk_id, ch.chunk_text, ch.agreement_id
                FROM chunks ch
                JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
                WHERE ca.cluster_id = ?
                ORDER BY ch.chunk_index
            """, [cid]).fetchdf().to_dict("records")

            if len(chunk_rows) < config.min_cluster_chunks:
                continue

            stats = await _process_cluster(
                cluster, chunk_rows, store, llm, config,
                ci, len(clusters), progress,
                extraction_chunks_by_agreement=extraction_chunks_by_agreement,
            )
            total_fields += stats["fields"]
            total_extractions += stats["extractions"]

        progress("discovery", f"Done: {total_fields} fields, {total_extractions} extractions "
                 f"across {len(clusters)} clusters")

    pipeline_stats["n_fields"] = total_fields
    pipeline_stats["n_extractions"] = total_extractions
    store.update_run(run_id, "extraction_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6: Intent Discovery (async)
    # ══════════════════════════════════════════════════════════════════════
    if config.run_intents:
        with _time("intents"):
            intent_stats = await _process_intents(store, llm, config, progress)
            pipeline_stats["intent_stats"] = intent_stats

    store.update_run(run_id, "intent_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7: Post-Pipeline Refinement
    # ══════════════════════════════════════════════════════════════════════
    if config.run_refinement:
        with _time("refinement"):
            try:
                refine_stats = await _run_refinement(store, embedder, llm, config, progress)
                pipeline_stats["refinement"] = refine_stats
            except Exception as e:
                logger.warning(f"Refinement failed (non-fatal): {e}")
                pipeline_stats["refinement"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 8: Knowledge Graph
    # ══════════════════════════════════════════════════════════════════════
    if config.run_knowledge_graph:
        with _time("knowledge_graph"):
            try:
                kg_stats = await _run_knowledge_graph(store, progress)
                pipeline_stats["knowledge_graph"] = kg_stats
            except Exception as e:
                logger.warning(f"KG construction failed (non-fatal): {e}")
                pipeline_stats["knowledge_graph"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - t0
    store.update_run(run_id, "complete", status="complete")

    pipeline_stats.update({
        "stage_timings": stage_timings,
        "total_elapsed_s": round(total_elapsed, 1),
        "llm_stats": llm.stats,
        "db_path": str(db),
    })

    progress("done", f"Pipeline complete in {total_elapsed:.0f}s — "
             f"{pipeline_stats['n_documents']} docs, {total_clusters} clusters, "
             f"{total_fields} fields, {total_extractions} extractions")

    # Save manifest
    manifest_dir = artifacts_root / account_id / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "pipeline_manifest.json", "w") as f:
        json.dump(pipeline_stats, f, indent=2, default=str)

    return pipeline_stats


# ── Helpers ───────────────────────────────────────────────────────────────────


class _StageTimer:
    """Context manager that records stage timing."""
    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings
    def __enter__(self):
        self._t0 = time.time()
        return self
    def __exit__(self, *args):
        self.timings[self.name] = round(time.time() - self._t0, 2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Production clustering → discovery → extraction pipeline")
    parser.add_argument("pdf_dir", help="Path to PDF directory")
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "clause", "macro-micro"])
    parser.add_argument("--discovery", default="hybrid", choices=["standard", "rlm", "hybrid", "rlm_v2", "rlm_v3", "rlm_v3_unified"])
    parser.add_argument("--extraction-backend", default="sequential", choices=["sequential", "rag"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--no-intents", action="store_true")
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--no-kg", action="store_true")
    parser.add_argument("--no-extraction", action="store_true")
    parser.add_argument("--db", default=None, help="DuckDB path")
    parser.add_argument("--lance", default=None, help="LanceDB path")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)-20s %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        mode=args.mode,
        device=args.device,
        discovery_mode=args.discovery,
        extraction_backend=args.extraction_backend,
        max_docs=args.max_docs,
        max_llm_concurrency=args.max_concurrency,
        use_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        run_intents=not args.no_intents,
        run_refinement=not args.no_refinement,
        run_knowledge_graph=not args.no_kg,
        extract_values=not args.no_extraction,
        db_path=args.db,
        lance_path=args.lance,
    )

    result = asyncio.run(run_production_pipeline(
        pdf_dir=args.pdf_dir,
        account_id=args.account_id,
        config=config,
    ))

    print(json.dumps(result, indent=2, default=str))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
