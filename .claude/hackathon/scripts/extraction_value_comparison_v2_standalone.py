#!/usr/bin/env python3
"""
4-Way Extraction Value Comparison (Standalone)
===============================================

Self-contained script with zero local imports. All code from:
  - config.py (RLM constants)
  - core/latent_briefing.py (TextLatentBriefing, CompactionConfig, CompactionStats)
  - core/rlm_with_memory.py (RLMWithLatentBriefing)
  - core/field_discovery.py (FieldDiscoveryConfig, discovery/extraction functions)
  - core/rlm_v2.py (LatentRLMv2, Pydantic models, prompts)

is inlined into this single file.

Required pip packages:
    pip install openai dspy-ai duckdb numpy pydantic python-dotenv sentence-transformers torch

Usage:
    python scripts/extraction_value_comparison_v2_standalone.py data/cuad_510_demo.duckdb --n-clusters 5
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import ast
import json
import logging
import os
import random
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG (from config.py)
# ══════════════════════════════════════════════════════════════════════════════

RLM_SUB_MODEL = os.getenv("RLM_SUB_MODEL", "")
RLM_MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "25"))
RLM_MAX_LLM_CALLS = int(os.getenv("RLM_MAX_LLM_CALLS", "50"))


# ══════════════════════════════════════════════════════════════════════════════
# LATENT BRIEFING (from core/latent_briefing.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompactionConfig:
    """Configuration for KV cache compaction."""
    tau: float = 1.5
    min_retention: float = 0.1
    max_retention: float = 0.9
    task_guided: bool = True
    layer_sample_frac: float = 0.25
    device: str = "cuda"


@dataclass
class CompactionStats:
    """Stats from a single compaction operation."""
    original_length: int = 0
    compacted_length: int = 0
    retention_rate: float = 1.0
    compaction_time_s: float = 0.0
    tau_used: float = 0.0
    tokens_saved: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return 1.0 - (self.compacted_length / self.original_length)


class TextLatentBriefing:
    """
    Text-level Latent Briefing for API-based models (no KV cache access).

    Emulates the core Latent Briefing insights at the text layer:
    1. Task-guided extraction: Score trajectory segments by relevance to
       the current subtask
    2. Attention-inspired compression: Use embedding similarity as a proxy
       for attention scores
    3. Adaptive thresholding: Keep more context for easy tasks, less for hard
    """

    def __init__(self, embed_model=None, config: CompactionConfig | None = None):
        self.config = config or CompactionConfig(tau=1.5)
        self.embed_model = embed_model
        self._trajectory_segments: list[dict] = []
        self._total_tokens_saved = 0

    def _get_embedder(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embed_model

    def add_trajectory_segment(
        self,
        text: str,
        metadata: dict | None = None,
        segment_type: str = "observation",
    ):
        embedder = self._get_embedder()
        embedding = embedder.encode(text, normalize_embeddings=True)

        self._trajectory_segments.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "type": segment_type,
            "token_count": len(text.split()),
        })

    def compact_for_task(
        self,
        task_prompt: str,
        tau: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, CompactionStats]:
        start_time = time.time()
        tau = tau if tau is not None else self.config.tau

        if not self._trajectory_segments:
            return "", CompactionStats()

        embedder = self._get_embedder()

        task_embedding = embedder.encode(task_prompt, normalize_embeddings=True)
        task_tensor = torch.tensor(task_embedding, dtype=torch.float32)

        scores = []
        for seg in self._trajectory_segments:
            seg_tensor = torch.tensor(seg["embedding"], dtype=torch.float32)
            score = F.cosine_similarity(task_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)).item()

            type_boost = {
                "worker_reply": 0.1,
                "tool_output": 0.05,
                "reasoning": 0.0,
                "observation": -0.05,
            }
            score += type_boost.get(seg["type"], 0.0)
            scores.append(score)

        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        median = scores_tensor.median()
        mad = (scores_tensor - median).abs().median()

        if mad < 1e-8:
            keep_mask = torch.ones(len(scores), dtype=torch.bool)
        else:
            threshold = median + tau * mad
            keep_mask = scores_tensor >= threshold

        n_total = len(scores)
        n_kept = keep_mask.sum().item()
        min_keep = max(1, int(n_total * self.config.min_retention))
        max_keep = int(n_total * self.config.max_retention)

        if n_kept < min_keep:
            _, top_indices = scores_tensor.topk(min_keep)
            keep_mask = torch.zeros(n_total, dtype=torch.bool)
            keep_mask[top_indices] = True
        elif n_kept > max_keep:
            _, top_indices = scores_tensor.topk(max_keep)
            keep_mask = torch.zeros(n_total, dtype=torch.bool)
            keep_mask[top_indices] = True

        retained_segments = []
        total_original_tokens = 0
        total_kept_tokens = 0

        for i, seg in enumerate(self._trajectory_segments):
            total_original_tokens += seg["token_count"]
            if keep_mask[i]:
                retained_segments.append(seg)
                total_kept_tokens += seg["token_count"]

        if max_tokens and total_kept_tokens > max_tokens:
            retained_scores = [(scores[i], i, seg) for i, seg in enumerate(self._trajectory_segments) if keep_mask[i]]
            retained_scores.sort(key=lambda x: x[0], reverse=True)

            budget = max_tokens
            final_segments = []
            for score, orig_idx, seg in retained_scores:
                if budget <= 0:
                    break
                final_segments.append((orig_idx, seg))
                budget -= seg["token_count"]

            final_segments.sort(key=lambda x: x[0])
            retained_segments = [seg for _, seg in final_segments]
            total_kept_tokens = sum(seg["token_count"] for seg in retained_segments)

        briefing_parts = []
        for seg in retained_segments:
            prefix = f"[{seg['type'].upper()}]" if seg["type"] != "observation" else ""
            text = seg["text"].strip()
            if prefix:
                briefing_parts.append(f"{prefix} {text}")
            else:
                briefing_parts.append(text)

        briefing_text = "\n---\n".join(briefing_parts)

        elapsed = time.time() - start_time
        tokens_saved = total_original_tokens - total_kept_tokens
        self._total_tokens_saved += tokens_saved

        stats = CompactionStats(
            original_length=total_original_tokens,
            compacted_length=total_kept_tokens,
            retention_rate=total_kept_tokens / max(1, total_original_tokens),
            compaction_time_s=elapsed,
            tau_used=tau,
            tokens_saved=tokens_saved,
        )

        logger.info(
            f"Text Latent Briefing: {total_original_tokens} → {total_kept_tokens} tokens "
            f"({stats.compression_ratio:.1%} compressed, {elapsed:.3f}s, τ={tau}, "
            f"{keep_mask.sum().item()}/{n_total} segments retained)"
        )

        return briefing_text, stats

    def reset(self):
        self._trajectory_segments = []
        self._total_tokens_saved = 0

    @property
    def total_tokens_saved(self) -> int:
        return self._total_tokens_saved

    @property
    def trajectory_length(self) -> int:
        return len(self._trajectory_segments)

    def get_full_trajectory_text(self) -> str:
        return "\n---\n".join(seg["text"] for seg in self._trajectory_segments)


# ══════════════════════════════════════════════════════════════════════════════
# RLM WITH MEMORY (from core/rlm_with_memory.py)
# ══════════════════════════════════════════════════════════════════════════════

import dspy
from dspy import Prediction


class RLMWithLatentBriefing(dspy.RLM):
    """
    DSPy RLM subclass with Latent Briefing memory sharing.

    Overrides _make_llm_tools() to wrap llm_query/llm_query_batched with
    memory injection, and _execute_iteration() to capture trajectory
    segments after each REPL step.
    """

    def __init__(
        self,
        signature,
        max_iterations: int = 25,
        max_llm_calls: int = 50,
        verbose: bool = False,
        sub_lm=None,
        tools: list[Callable] | None = None,
        briefing_tau: float = 1.5,
        briefing_max_tokens: int = 2000,
        briefing_min_retention: float = 0.1,
        briefing_max_retention: float = 0.9,
        briefing_embed_model=None,
        adaptive_tau: bool = True,
    ):
        self.briefing_tau = briefing_tau
        self.briefing_max_tokens = briefing_max_tokens
        self.adaptive_tau = adaptive_tau

        self.briefing_config = CompactionConfig(
            tau=briefing_tau,
            min_retention=briefing_min_retention,
            max_retention=briefing_max_retention,
        )
        self.briefing_engine = TextLatentBriefing(
            embed_model=briefing_embed_model,
            config=self.briefing_config,
        )

        self._all_stats: list[CompactionStats] = []
        self._iteration_count = 0

        super().__init__(
            signature=signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=verbose,
            sub_lm=sub_lm,
            tools=tools,
        )

    def _compact_for_worker(self, task_prompt: str) -> str:
        """Compact trajectory for a worker task."""
        if not self.briefing_engine.trajectory_length:
            return ""

        tau = self.briefing_tau
        if self.adaptive_tau and self._iteration_count > 2:
            tau = min(tau * 1.3, 3.0)

        briefing, stats = self.briefing_engine.compact_for_task(
            task_prompt, tau=tau, max_tokens=self.briefing_max_tokens,
        )
        self._all_stats.append(stats)
        return briefing

    def _add_to_trajectory(self, text: str, segment_type: str = "observation"):
        """Add a segment to the trajectory memory."""
        self.briefing_engine.add_trajectory_segment(
            text, segment_type=segment_type,
        )

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        base_tools = super()._make_llm_tools(max_workers)
        base_llm_query = base_tools["llm_query"]
        base_llm_query_batched = base_tools["llm_query_batched"]

        engine = self

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            briefing = engine._compact_for_worker(prompt)
            if briefing:
                augmented = (
                    f"=== CONTEXT FROM PRIOR ANALYSIS ===\n"
                    f"{briefing}\n"
                    f"=== END CONTEXT ===\n\n"
                    f"{prompt}"
                )
            else:
                augmented = prompt

            response = base_llm_query(augmented)

            engine._add_to_trajectory(
                f"Worker query: {prompt[:200]}\nWorker response: {response[:500]}",
                segment_type="worker_reply",
            )
            return response

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query the LLM with multiple prompts concurrently."""
            if not prompts:
                return []

            briefing = engine._compact_for_worker(prompts[0])
            if briefing:
                augmented = [
                    f"=== CONTEXT FROM PRIOR ANALYSIS ===\n{briefing}\n=== END CONTEXT ===\n\n{p}"
                    for p in prompts
                ]
            else:
                augmented = prompts

            responses = base_llm_query_batched(augmented)

            engine._add_to_trajectory(
                f"Batch of {len(prompts)} worker queries completed",
                segment_type="worker_reply",
            )
            return responses

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    def _execute_iteration(self, repl, variables, history, iteration,
                           input_args, output_field_names):
        self._iteration_count = iteration

        result = super()._execute_iteration(
            repl, variables, history, iteration, input_args, output_field_names,
        )

        if isinstance(result, Prediction):
            reasoning = getattr(result, "final_reasoning", "")
            if reasoning:
                self._add_to_trajectory(
                    f"Final reasoning: {reasoning[:500]}",
                    segment_type="reasoning",
                )
        else:
            entries = list(result)
            if entries:
                last = entries[-1]
                code = getattr(last, "code", "")
                output = getattr(last, "output", "")
                reasoning = getattr(last, "reasoning", "")
                segment = (
                    f"[Iteration {iteration+1}]\n"
                    f"Reasoning: {reasoning[:300]}\n"
                    f"Code: {code[:300]}\n"
                    f"Output: {output[:500]}"
                )
                self._add_to_trajectory(segment, segment_type="observation")

        return result

    def forward(self, **input_args) -> Prediction:
        """Execute RLM with Latent Briefing memory augmentation."""
        self.briefing_engine.reset()
        self._all_stats = []
        self._iteration_count = 0

        result = super().forward(**input_args)

        result.briefing_stats = {
            "n_compactions": len(self._all_stats),
            "total_tokens_saved": self.briefing_engine.total_tokens_saved,
            "avg_compression_ratio": (
                sum(s.compression_ratio for s in self._all_stats) / len(self._all_stats)
                if self._all_stats else 0.0
            ),
            "avg_retention_rate": (
                sum(s.retention_rate for s in self._all_stats) / len(self._all_stats)
                if self._all_stats else 1.0
            ),
            "total_compaction_time_s": sum(s.compaction_time_s for s in self._all_stats),
            "per_compaction": [
                {
                    "original": s.original_length,
                    "compacted": s.compacted_length,
                    "ratio": round(s.compression_ratio, 3),
                    "tau": s.tau_used,
                    "time_s": round(s.compaction_time_s, 3),
                }
                for s in self._all_stats
            ],
        }

        return result


# ══════════════════════════════════════════════════════════════════════════════
# FIELD DISCOVERY (from core/field_discovery.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_dspy_lm() -> "dspy.LM":
    """Build a dspy.LM from the existing env-var configuration with retry logic."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

    retry_kwargs = dict(num_retries=5)

    if endpoint and api_key:
        return dspy.LM(
            f"azure/{deployment}",
            api_key=api_key,
            api_base=endpoint,
            api_version=api_version,
            **retry_kwargs,
        )
    openai_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return dspy.LM(f"openai/{model}", api_key=openai_key, **retry_kwargs)


@dataclass
class FieldDiscoveryConfig:
    max_sample_chunks: int = 30
    max_fields_per_cluster: int = 15
    temperature: float = 0.2
    model: str = ""
    max_retries: int = 3
    retry_delay: float = 2.0


def _get_azure_client():
    """Create Azure OpenAI client from env vars."""
    from openai import AzureOpenAI

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars. "
            "Example: export AZURE_OPENAI_ENDPOINT='https://myorg.openai.azure.com/'"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _get_deployment(config: FieldDiscoveryConfig) -> str:
    if config.model:
        return config.model
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")


FIELD_DISCOVERY_PROMPT = """You are a legal contract analyst. Analyze these clause excerpts that all belong to the same clause type: "{cluster_label}".

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

FIELD_EXTRACTION_PROMPT = """Extract field values from this contract clause.

Clause type: "{cluster_label}"
Clause text:
{chunk_text}

Fields to extract:
{fields_schema}

Respond with ONLY a JSON object mapping field names to extracted values.
Use null if the field is not present. For booleans use true/false.

Example: {{"notice_period": "30 days", "auto_renewal": true, "termination_fee": null}}"""

FIELD_EXTRACTION_PROMPT_ENHANCED = """Extract field values from this contract clause.
IMPORTANT: Only extract values that appear explicitly in the text. Do NOT hallucinate or infer values not stated.

Clause type: "{cluster_label}"
{rlm_context_section}
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


def _call_azure(client, deployment: str, prompt: str, config: FieldDiscoveryConfig,
                temperature: float | None = None, max_tokens: int = 2000) -> dict | None:
    """Call Azure OpenAI with retries, return parsed JSON or None."""
    temp = temperature if temperature is not None else config.temperature
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"Azure API error (attempt {attempt+1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    return None


def discover_fields_for_cluster(
    client, deployment: str,
    cluster_label: str, keywords: list[str],
    chunk_texts: list[str], config: FieldDiscoveryConfig,
) -> list[dict]:
    """Discover fields in a cluster's chunks. Returns list of {name, type, description, examples}."""
    sample = chunk_texts[:config.max_sample_chunks]
    chunks_text = "\n\n".join(
        f"[Excerpt {i+1}]\n{text[:1500]}" for i, text in enumerate(sample)
    )

    prompt = FIELD_DISCOVERY_PROMPT.format(
        cluster_label=cluster_label,
        keywords=", ".join(keywords[:10]),
        chunks_text=chunks_text,
        max_fields=config.max_fields_per_cluster,
    )

    parsed = _call_azure(client, deployment, prompt, config)
    if not parsed:
        return []

    if isinstance(parsed, dict):
        fields = parsed.get("fields", [])
    elif isinstance(parsed, list):
        fields = parsed
    else:
        return []

    valid_types = {"text", "date", "duration", "currency", "percentage",
                   "integer", "boolean", "entity_name", "enum"}
    result = []
    for f in fields[:config.max_fields_per_cluster]:
        name = f.get("name", "").strip()
        if not name:
            continue
        name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        ftype = f.get("type", "text").strip().lower()
        if ftype not in valid_types:
            ftype = "text"
        result.append({
            "name": name,
            "type": ftype,
            "description": f.get("description", ""),
            "examples": f.get("examples", [])[:5],
        })
    return result


def _normalize_rlm_fields(fields: list, config: FieldDiscoveryConfig) -> list[dict]:
    """Normalize RLM-discovered fields into standard format."""
    valid_types = {"text", "date", "duration", "currency", "percentage",
                   "integer", "boolean", "entity_name", "enum"}
    result_fields = []
    for f in fields[:config.max_fields_per_cluster]:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
        if not name:
            continue
        name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        ftype = f.get("type", "text").strip().lower()
        if ftype not in valid_types:
            ftype = "text"
        result_fields.append({
            "name": name,
            "type": ftype,
            "description": f.get("description", ""),
            "examples": f.get("examples") or ([f["field_value"]] if f.get("field_value") else []),
        })
    return result_fields


def _parse_rlm_output(raw, config: FieldDiscoveryConfig) -> list[dict]:
    """Parse RLM output with multiple fallback strategies."""
    if not raw or (isinstance(raw, str) and not raw.strip()):
        logger.warning("RLM returned empty output")
        return []

    # Try JSON parse first
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        # Try Python literal eval
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None

        if parsed is not None and isinstance(parsed, (list, dict)):
            if isinstance(parsed, dict):
                parsed = parsed.get("fields", [])
            fields = parsed if isinstance(parsed, list) else []
            return _normalize_rlm_fields(fields, config)

        # Try extracting JSON from markdown fences
        match = re.search(r'```(?:json)?\s*(.+?)```', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed = None

        if parsed is None:
            # Try finding any JSON array in the raw output
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                try:
                    parsed = json.loads(arr_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("RLM returned unparseable output")
                    return []
            else:
                logger.warning("RLM returned unparseable output")
                return []

    if isinstance(parsed, dict):
        fields = parsed.get("fields", [])
    elif isinstance(parsed, list):
        fields = parsed
    else:
        return []

    return _normalize_rlm_fields(fields, config)


def discover_fields_for_cluster_rlm(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: FieldDiscoveryConfig,
) -> list[dict]:
    """
    RLM-based field discovery — lets the LLM programmatically explore ALL
    chunks via a sandboxed REPL instead of seeing only a 30-chunk sample.
    """
    lm = _get_dspy_lm()
    dspy.configure(lm=lm)

    sub_lm = lm
    if RLM_SUB_MODEL:
        sub_lm = dspy.LM(RLM_SUB_MODEL, num_retries=5)

    class FieldDiscoverySig(dspy.Signature):
        """You are a legal contract analyst. Given clause excerpts (chunks_json)
        belonging to a clause type (cluster_label), discover ALL recurring structured
        fields. Return fields_json as a JSON list of objects with keys: name
        (snake_case), type (text|date|duration|currency|percentage|integer|boolean|
        entity_name|enum), description (one sentence), examples (2-3 values).
        Only include fields appearing in 2+ chunks. Return up to max_fields fields.

        STRATEGY — follow these steps:
        1. Parse chunks_json and print len, then print 5 diverse chunks in full.
        2. Use llm_query_batched to send 3-6 batches of chunks to the sub-LLM,
           each batch asking: 'List every extractable field (name, type, description,
           2 examples) from these contract clauses.' Cover ALL chunks across batches.
        3. Aggregate field lists from all batches. Count how many batches mention
           each field name. Keep fields mentioned in 2+ batches.
        4. Deduplicate near-synonyms (e.g. notice_period vs notice_timing).
        5. Build the final list with up to max_fields fields and SUBMIT.

        Be thorough — aim for 10-15 fields per cluster. Miss nothing."""

        chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
        cluster_label: str = dspy.InputField(desc="The clause type label")
        keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
        max_fields: str = dspy.InputField(desc="Maximum number of fields to return")
        fields_json: str = dspy.OutputField(desc="JSON array of {name, type, description, examples} objects")

    rlm = dspy.RLM(
        FieldDiscoverySig,
        max_iterations=RLM_MAX_ITERATIONS,
        max_llm_calls=RLM_MAX_LLM_CALLS,
        verbose=False,
        sub_lm=sub_lm,
    )

    # Cap chunks
    max_rlm_chunks = 60
    if len(chunk_texts) > max_rlm_chunks:
        rng = random.Random(42)
        head = chunk_texts[:20]
        tail_pool = chunk_texts[20:]
        sampled = rng.sample(tail_pool, min(max_rlm_chunks - 20, len(tail_pool)))
        capped_texts = head + sampled
    else:
        capped_texts = chunk_texts

    chunks_payload = json.dumps(capped_texts, ensure_ascii=False)

    try:
        result = rlm(
            chunks_json=chunks_payload,
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            max_fields=str(config.max_fields_per_cluster),
        )
        raw = result.fields_json

        trajectory = getattr(result, "trajectory", None)
        if trajectory:
            n_steps = len(trajectory)
            logger.info(f"RLM field discovery for '{cluster_label}': {n_steps} REPL steps")
            for i, step in enumerate(trajectory):
                code = step.get("code", "")[:200]
                output = step.get("output", "")[:200]
                logger.debug(f"  Step {i+1}: code={code!r} output={output!r}")
    except Exception as e:
        logger.warning(f"RLM field discovery failed, falling back to standard: {e}")
        return []

    return _parse_rlm_output(raw, config)


def discover_fields_for_cluster_rlm_lb(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: FieldDiscoveryConfig,
    briefing_tau: float = 1.5,
    briefing_max_tokens: int = 2000,
    adaptive_tau: bool = True,
) -> tuple[list[dict], dict]:
    """
    RLM field discovery with Latent Briefing memory sharing.

    Same RLM approach as discover_fields_for_cluster_rlm() but each
    llm_query() worker call receives a compressed trajectory briefing
    instead of operating in isolation.
    """
    lm = _get_dspy_lm()
    dspy.configure(lm=lm)

    sub_lm = lm
    if RLM_SUB_MODEL:
        sub_lm = dspy.LM(RLM_SUB_MODEL, num_retries=5)

    class FieldDiscoverySig(dspy.Signature):
        """You are a legal contract analyst. Given clause excerpts (chunks_json)
        belonging to a clause type (cluster_label), discover ALL recurring structured
        fields. Return fields_json as a JSON list of objects with keys: name
        (snake_case), type (text|date|duration|currency|percentage|integer|boolean|
        entity_name|enum), description (one sentence), examples (2-3 values).
        Only include fields appearing in 2+ chunks. Return up to max_fields fields.

        STRATEGY — follow these steps:
        1. Parse chunks_json and print len, then print 5 diverse chunks in full.
        2. Use llm_query_batched to send 3-6 batches of chunks to the sub-LLM,
           each batch asking: 'List every extractable field (name, type, description,
           2 examples) from these contract clauses.' Cover ALL chunks across batches.
        3. Aggregate field lists from all batches. Count how many batches mention
           each field name. Keep fields mentioned in 2+ batches.
        4. Deduplicate near-synonyms (e.g. notice_period vs notice_timing).
        5. Build the final list with up to max_fields fields and SUBMIT.

        NOTE: Your llm_query calls receive context from prior analysis automatically
        via Latent Briefing. Each worker sees relevant memory from prior steps,
        so later batches are aware of what was already discovered. Use this to
        focus on finding NEW fields, not re-discovering known ones."""

        chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
        cluster_label: str = dspy.InputField(desc="The clause type label")
        keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
        max_fields: str = dspy.InputField(desc="Maximum number of fields to return")
        fields_json: str = dspy.OutputField(desc="JSON array of {name, type, description, examples} objects")

    rlm = RLMWithLatentBriefing(
        signature=FieldDiscoverySig,
        max_iterations=RLM_MAX_ITERATIONS,
        max_llm_calls=RLM_MAX_LLM_CALLS,
        verbose=False,
        sub_lm=sub_lm,
        briefing_tau=briefing_tau,
        briefing_max_tokens=briefing_max_tokens,
        adaptive_tau=adaptive_tau,
    )

    # Cap chunks
    max_rlm_chunks = 60
    if len(chunk_texts) > max_rlm_chunks:
        rng = random.Random(42)
        head = chunk_texts[:20]
        tail_pool = chunk_texts[20:]
        sampled = rng.sample(tail_pool, min(max_rlm_chunks - 20, len(tail_pool)))
        capped_texts = head + sampled
    else:
        capped_texts = chunk_texts

    chunks_payload = json.dumps(capped_texts, ensure_ascii=False)

    briefing_meta = {"briefing_enabled": True, "tau": briefing_tau, "adaptive": adaptive_tau}

    try:
        result = rlm(
            chunks_json=chunks_payload,
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            max_fields=str(config.max_fields_per_cluster),
        )
        raw = result.fields_json

        if hasattr(result, "briefing_stats"):
            briefing_meta.update(result.briefing_stats)

    except Exception as e:
        logger.warning(f"RLM+LB field discovery failed: {e}")
        briefing_meta["error"] = str(e)
        return [], briefing_meta

    fields = _parse_rlm_output(raw, config)
    briefing_meta["fields_discovered"] = len(fields)
    return fields, briefing_meta


def extract_fields_from_chunk(
    client, deployment: str,
    cluster_label: str, chunk_text: str,
    fields: list[dict], config: FieldDiscoveryConfig,
    rlm_context: dict | None = None,
) -> dict[str, tuple[str | None, float]]:
    """Extract field values from a single chunk. Returns {field_name: (value, confidence)}."""
    schema_lines = []
    for f in fields:
        line = f"- {f['name']} ({f['type']}): {f['description']}"
        examples = f.get('examples', [])
        if examples:
            ex_str = ', '.join(f'"{e}"' for e in examples[:2] if e)
            if ex_str:
                line += f"  [examples from similar clauses: {ex_str}]"
        schema_lines.append(line)
    fields_schema = "\n".join(schema_lines)

    rlm_context_section = ""
    if rlm_context:
        parts = []
        clause_types = rlm_context.get('clause_types', [])
        if clause_types:
            ct_str = "; ".join(
                f"{ct.get('name', '')}: {ct.get('emergent_description', '')}"
                for ct in clause_types[:3]
            )
            parts.append(f"Clause sub-types found in similar clauses: {ct_str}")
        obligations = rlm_context.get('obligations', [])
        if obligations:
            ob_str = "; ".join(
                f"{ob.get('obligor', '')} must {ob.get('action', '')}" +
                (f" if {ob['trigger_condition']}" if ob.get('trigger_condition') else "")
                for ob in obligations[:3]
            )
            parts.append(f"Key obligations: {ob_str}")
        intents = rlm_context.get('intents', [])
        if intents:
            int_str = "; ".join(i.get('intent_description', '') for i in intents[:3])
            parts.append(f"Clause intents: {int_str}")
        if parts:
            rlm_context_section = "\n".join(parts) + "\n"

    if rlm_context or any(f.get('examples') for f in fields):
        prompt = FIELD_EXTRACTION_PROMPT_ENHANCED.format(
            cluster_label=cluster_label,
            rlm_context_section=rlm_context_section,
            chunk_text=chunk_text[:4000],
            fields_schema=fields_schema,
        )
    else:
        prompt = FIELD_EXTRACTION_PROMPT.format(
            cluster_label=cluster_label,
            chunk_text=chunk_text[:4000],
            fields_schema=fields_schema,
        )
    parsed = _call_azure(client, deployment, prompt, config, temperature=0.0, max_tokens=1000)
    if not parsed or not isinstance(parsed, dict):
        return {}
    valid_names = {f["name"] for f in fields}
    results = {}
    chunk_lower = chunk_text.lower()
    for k, v in parsed.items():
        if k not in valid_names:
            continue
        if v is None or v == "" or v == "null" or v == "N/A" or v == "n/a":
            continue
        v_str = str(v)
        conf = 0.5
        v_lower = v_str.lower().strip()
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
        conf = max(0.1, min(1.0, conf))
        results[k] = (v_str, round(conf, 3))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RLM V2 (from core/rlm_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

class FieldV2(BaseModel):
    name: str
    value_type: str = "string"
    description: str = ""
    span_text: str = ""
    confidence: float = 0.5
    source: str = "grounded"


class ClauseTypeV2(BaseModel):
    name: str
    emergent_description: str = ""
    span_text: str = ""
    source: str = "grounded"


class IntentV2(BaseModel):
    clause_ref: str = ""
    intent_description: str = ""


class ObligationV2(BaseModel):
    clause_ref: str = ""
    obligor: str = ""
    action: str = ""
    trigger_condition: Optional[str] = None
    deadline: Optional[str] = None


class RollingState(BaseModel):
    """Rolling Structured State — typed, cluster-scoped accumulator."""
    cluster_label: str = ""
    chunks_processed: int = 0
    fields: list[FieldV2] = []
    clause_types: list[ClauseTypeV2] = []
    intents: list[IntentV2] = []
    obligations: list[ObligationV2] = []
    open_questions: list[str] = []
    anchor_terms: list[tuple[str, int]] = []
    iteration_count: int = 0
    last_state_delta: float = 1.0


GROUNDED_PROMPT = """You are a legal contract analyst. Analyze these clause excerpts and discover structured elements.

{brief_section}CLAUSE TYPE: {cluster_label}
KEYWORDS: {keywords}

EXCERPTS:
{chunks_text}

Discover four types of structured elements from the text:

1. FIELDS: Named data points whose values vary across contracts. Use snake_case names.
   Types: string, date, money, enum, ref, boolean, duration, percentage, integer
2. CLAUSE TYPES: Recurring contractual patterns, named by what they DO.
3. INTENTS: WHY each clause type exists (the underlying purpose).
4. OBLIGATIONS: Actionable commitments — who MUST do what, under what conditions.

Rules:
- Only include items grounded in the actual text — include exact span_text quotes
- For fields: be specific (e.g., "auto_renewal_notice_window_days" not "terms")
- For obligations: only clear, enforceable duties
- Focus on NEW discoveries not already in prior context
- Return 5-15 fields, 2-8 clause types

Respond with ONLY JSON:
{{
  "fields": [{{"name": "...", "value_type": "string|date|money|enum|ref|boolean|duration|percentage|integer", "description": "...", "span_text": "exact quote from text", "confidence": 0.8}}],
  "clause_types": [{{"name": "...", "emergent_description": "...", "span_text": "..."}}],
  "intents": [{{"clause_ref": "clause_type_name", "intent_description": "..."}}],
  "obligations": [{{"clause_ref": "...", "obligor": "...", "action": "...", "trigger_condition": "...", "deadline": "..."}}]
}}"""

EXPLORATORY_PROMPT = """You are a legal contract analyst specializing in finding NOVEL and NICHE patterns that standard analysis overlooks.

CLAUSE TYPE: {cluster_label}
KEYWORDS: {keywords}

EXCERPTS:
{chunks_text}

With FRESH EYES — find unusual, domain-specific, or niche patterns:

1. FIELDS: Unusual or domain-specific data points highly specific to this contract type.
   Do NOT include generic fields like party_name, date, amount. Be very specific.
   Types: string, date, money, enum, ref, boolean, duration, percentage, integer
2. CLAUSE TYPES: Non-obvious contractual patterns hiding in the text.
3. INTENTS: Hidden purposes or implications beyond the surface reading.
4. OBLIGATIONS: Subtle duties or conditional requirements often overlooked.

Rules:
- Focus on what makes THESE clauses UNIQUE — not boilerplate
- Prefer highly specific, legally meaningful names
- Include exact span_text quotes from the excerpts
- Return 3-10 fields, 1-5 clause types

Respond with ONLY JSON:
{{
  "fields": [{{"name": "...", "value_type": "string|date|money|enum|ref|boolean|duration|percentage|integer", "description": "...", "span_text": "exact quote from text", "confidence": 0.7}}],
  "clause_types": [{{"name": "...", "emergent_description": "...", "span_text": "..."}}],
  "intents": [{{"clause_ref": "...", "intent_description": "..."}}],
  "obligations": [{{"clause_ref": "...", "obligor": "...", "action": "...", "trigger_condition": "...", "deadline": "..."}}]
}}"""


class LatentRLMv2:
    """
    Spec-aligned discovery pipeline with rolling state and dual-mode exploration.

    Architecture per chunk-batch:
      1. Build embedding-scored brief from rolling state (§4.2)
      2. Grounded pass: 4-output discovery with brief (temp 0.2)
      3. Exploratory pass: 4-output discovery without brief (temp 0.8)
      4. Overlap-penalty merge (cosine > 0.85 → grounded wins) (§4.5)
      5. Update rolling state (§4.1)
      6. Check adaptive stop (state delta < epsilon)
    Post-loop: two-tier canonicalization (§4.6)
    """

    def __init__(self, client, deployment, config, embedder=None):
        self.client = client
        self.deployment = deployment
        self.config = config
        self._embedder = embedder
        self.adaptive_stop_epsilon = 0.05

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def discover_for_cluster(
        self,
        cluster_label: str,
        keywords: list[str],
        chunk_texts: list[str],
    ) -> tuple[list[dict], dict]:
        sample = chunk_texts[:self.config.max_sample_chunks]
        state = RollingState(cluster_label=cluster_label)

        batch_size = 2
        n_batches = min(5, (len(sample) + batch_size - 1) // batch_size)

        meta = {
            "n_batches": 0,
            "n_grounded_calls": 0,
            "n_exploratory_calls": 0,
            "stopped_early": False,
            "exploratory_kept": 0,
            "exploratory_dropped": 0,
        }

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch = sample[start:start + batch_size]
            if not batch:
                break

            chunks_text = "\n\n".join(
                f"[Excerpt {start + i + 1}]\n{t[:1500]}" for i, t in enumerate(batch)
            )

            prev_field_count = len(state.fields)

            brief = self._build_brief(state, chunks_text)

            grounded = self._discover(
                chunks_text, cluster_label, keywords, brief,
                temperature=0.2, mode="grounded",
            )
            meta["n_grounded_calls"] += 1

            exploratory = self._discover(
                chunks_text, cluster_label, keywords, "",
                temperature=0.8, mode="exploratory",
            )
            meta["n_exploratory_calls"] += 1

            merged, merge_stats = self._overlap_merge(grounded, exploratory)
            meta["exploratory_kept"] += merge_stats["kept"]
            meta["exploratory_dropped"] += merge_stats["dropped"]

            self._update_state(state, merged, batch)

            new_field_count = len(state.fields)
            if prev_field_count > 0:
                delta = abs(new_field_count - prev_field_count) / max(new_field_count, 1)
            else:
                delta = 1.0
            state.last_state_delta = delta
            state.iteration_count += 1
            meta["n_batches"] += 1

            logger.info(
                f"    V2 batch {batch_idx + 1}/{n_batches}: "
                f"{len(state.fields)} fields, {len(state.clause_types)} clause types, "
                f"delta={delta:.3f}"
            )

            if delta < self.adaptive_stop_epsilon and state.iteration_count >= 3:
                meta["stopped_early"] = True
                logger.info(
                    f"    V2 adaptive stop after {state.iteration_count} batches "
                    f"(delta={delta:.3f} < {self.adaptive_stop_epsilon})"
                )
                break

        pre_canon = len(state.fields)
        state = self._canonicalize(state)
        post_canon = len(state.fields)

        fields = self._to_field_dicts(state)

        meta.update({
            "n_fields": len(fields),
            "n_clause_types": len(state.clause_types),
            "n_intents": len(state.intents),
            "n_obligations": len(state.obligations),
            "iterations": state.iteration_count,
            "final_delta": state.last_state_delta,
            "canonicalization_removed": pre_canon - post_canon,
            "exploration_ratio": (
                meta["exploratory_kept"] / max(1, meta["exploratory_kept"] + meta["exploratory_dropped"])
            ),
            "clause_types": [ct.model_dump() for ct in state.clause_types],
            "intents": [i.model_dump() for i in state.intents],
            "obligations": [o.model_dump() for o in state.obligations],
        })

        return fields, meta

    def _call_azure(self, prompt, temperature=0.2, max_tokens=2000):
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"V2 JSON parse error (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"V2 Azure API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        return None

    def _discover(self, chunks_text, cluster_label, keywords, brief,
                  temperature, mode):
        if mode == "grounded":
            brief_section = ""
            if brief:
                brief_section = (
                    f"PRIOR CONTEXT (from previous chunk analysis):\n"
                    f"{brief}\n\n"
                    f"Build on this context — focus on NEW discoveries not listed above.\n\n"
                )

            prompt = GROUNDED_PROMPT.format(
                brief_section=brief_section,
                cluster_label=cluster_label,
                keywords=", ".join(keywords[:10]),
                chunks_text=chunks_text,
            )
        else:
            prompt = EXPLORATORY_PROMPT.format(
                cluster_label=cluster_label,
                keywords=", ".join(keywords[:10]),
                chunks_text=chunks_text,
            )

        parsed = self._call_azure(prompt, temperature=temperature)
        if not parsed:
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}

        return {
            "fields": parsed.get("fields", []),
            "clause_types": parsed.get("clause_types", []),
            "intents": parsed.get("intents", []),
            "obligations": parsed.get("obligations", []),
        }

    def _build_brief(self, state, next_text, k=15, token_budget=1200):
        if not state.fields and not state.clause_types:
            return ""

        items = []
        for f in state.fields:
            items.append(("field", f.name, f"{f.name} ({f.value_type}): {f.description}"))
        for c in state.clause_types:
            items.append(("clause_type", c.name, f"{c.name}: {c.emergent_description}"))

        if not items:
            return ""

        query_text = next_text[:2000]
        all_texts = [query_text] + [text for _, _, text in items]
        all_vecs = self.embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)

        query_vec = all_vecs[0]
        item_vecs = all_vecs[1:]

        scores = np.dot(item_vecs, query_vec)

        for idx in range(len(scores)):
            age = max(1, state.iteration_count - idx // max(1, len(items) // max(1, state.iteration_count)))
            recency_boost = 1.0 + 0.2 * (age ** -0.5)
            scores[idx] *= recency_boost

        sorted_indices = np.argsort(-scores)[:k]

        kept_fields = []
        kept_clauses = []
        for idx in sorted_indices:
            typ, name, _ = items[idx]
            if typ == "field":
                f = next((f for f in state.fields if f.name == name), None)
                if f:
                    kept_fields.append({"name": f.name, "type": f.value_type, "desc": f.description[:60]})
            elif typ == "clause_type":
                c = next((c for c in state.clause_types if c.name == name), None)
                if c:
                    kept_clauses.append({"name": c.name, "desc": c.emergent_description[:60]})

        brief = {
            "already_found": {
                "fields": kept_fields,
                "clause_types": kept_clauses,
            },
            "still_looking_for": state.open_questions[:5],
            "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
        }

        text = json.dumps(brief, indent=None)
        words = text.split()
        if len(words) > token_budget:
            text = " ".join(words[:token_budget])

        return text

    def _overlap_merge(self, grounded, exploratory, threshold=0.85):
        result = {}
        total_kept = 0
        total_dropped = 0

        for key in ["fields", "clause_types", "intents", "obligations"]:
            g_items = grounded.get(key, [])
            e_items = exploratory.get(key, [])

            if not g_items:
                result[key] = e_items
                total_kept += len(e_items)
                continue
            if not e_items:
                result[key] = g_items
                continue

            def item_text(item, k=key):
                if k == "fields":
                    return f"{item.get('name', '')}: {item.get('description', '')}"
                elif k == "clause_types":
                    return f"{item.get('name', '')}: {item.get('emergent_description', '')}"
                elif k == "intents":
                    return f"{item.get('clause_ref', '')}: {item.get('intent_description', '')}"
                elif k == "obligations":
                    return f"{item.get('obligor', '')} must {item.get('action', '')}"
                return str(item)

            g_texts = [item_text(it) for it in g_items]
            e_texts = [item_text(it) for it in e_items]

            all_texts = g_texts + e_texts
            all_vecs = self.embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)

            g_vecs = all_vecs[:len(g_texts)]
            e_vecs = all_vecs[len(g_texts):]

            merged = list(g_items)
            for i, e_item in enumerate(e_items):
                sims = np.dot(g_vecs, e_vecs[i])
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                if max_sim < threshold:
                    e_item["source"] = "exploratory"
                    merged.append(e_item)
                    total_kept += 1
                else:
                    total_dropped += 1

            result[key] = merged

        return result, {"kept": total_kept, "dropped": total_dropped}

    def _update_state(self, state, merged, batch_chunks):
        existing_names = {f.name for f in state.fields}
        for f_dict in merged.get("fields", []):
            name = (f_dict.get("name") or "").strip()
            if not name:
                continue
            name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
            if name not in existing_names:
                state.fields.append(FieldV2(
                    name=name,
                    value_type=f_dict.get("value_type", "string"),
                    description=f_dict.get("description", ""),
                    span_text=f_dict.get("span_text", ""),
                    confidence=f_dict.get("confidence", 0.5),
                    source=f_dict.get("source", "grounded"),
                ))
                existing_names.add(name)

        existing_ct = {c.name.lower() for c in state.clause_types}
        for c_dict in merged.get("clause_types", []):
            name = (c_dict.get("name") or "").strip()
            if not name or name.lower() in existing_ct:
                continue
            state.clause_types.append(ClauseTypeV2(
                name=name,
                emergent_description=c_dict.get("emergent_description", ""),
                span_text=c_dict.get("span_text", ""),
                source=c_dict.get("source", "grounded"),
            ))
            existing_ct.add(name.lower())

        for i_dict in merged.get("intents", []):
            desc = i_dict.get("intent_description", "")
            if desc:
                state.intents.append(IntentV2(
                    clause_ref=i_dict.get("clause_ref", ""),
                    intent_description=desc,
                ))

        for o_dict in merged.get("obligations", []):
            action = o_dict.get("action", "")
            if action:
                state.obligations.append(ObligationV2(
                    clause_ref=o_dict.get("clause_ref", ""),
                    obligor=o_dict.get("obligor", ""),
                    action=action,
                    trigger_condition=o_dict.get("trigger_condition"),
                    deadline=o_dict.get("deadline"),
                ))

        existing_terms = dict(state.anchor_terms)
        for chunk in batch_chunks:
            words = re.findall(r'\b[a-z]{4,}\b', chunk.lower())
            for term, count in Counter(words).most_common(20):
                existing_terms[term] = existing_terms.get(term, 0) + count
        state.anchor_terms = sorted(existing_terms.items(), key=lambda x: -x[1])[:20]

        state.chunks_processed += len(batch_chunks)

    def _canonicalize(self, state):
        if len(state.fields) <= 1:
            return state

        to_remove = set()
        names = [f.name for f in state.fields]
        normalized = [re.sub(r'[_\s]', '', n.lower()) for n in names]

        for i in range(len(normalized)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(normalized)):
                if j in to_remove:
                    continue
                fi, fj = state.fields[i], state.fields[j]

                if fi.source == "exploratory" and fj.source == "exploratory":
                    continue

                if normalized[i] in normalized[j] or normalized[j] in normalized[i]:
                    if len(names[i]) <= len(names[j]):
                        if len(fj.description) > len(fi.description):
                            state.fields[i] = fi.model_copy(update={"description": fj.description})
                        to_remove.add(j)
                    else:
                        if len(fi.description) > len(fj.description):
                            state.fields[j] = fj.model_copy(update={"description": fi.description})
                        to_remove.add(i)
                elif (normalized[i] + 's' == normalized[j] or
                      normalized[j] + 's' == normalized[i]):
                    shorter = i if len(names[i]) <= len(names[j]) else j
                    longer = j if shorter == i else i
                    to_remove.add(longer)

        remaining = [i for i in range(len(state.fields)) if i not in to_remove]
        if len(remaining) > 2:
            texts = [
                f"{state.fields[i].name}: {state.fields[i].description}"
                for i in remaining
            ]
            vecs = self.embedder.encode(texts, normalize_embeddings=True, batch_size=32)
            sim = np.dot(vecs, vecs.T)

            for a in range(len(remaining)):
                if remaining[a] in to_remove:
                    continue
                for b in range(a + 1, len(remaining)):
                    if remaining[b] in to_remove:
                        continue
                    fa = state.fields[remaining[a]]
                    fb = state.fields[remaining[b]]
                    if fa.source == "exploratory" and fb.source == "exploratory":
                        continue
                    if sim[a, b] > 0.82:
                        if len(fa.description) >= len(fb.description):
                            to_remove.add(remaining[b])
                        else:
                            to_remove.add(remaining[a])

        state.fields = [f for i, f in enumerate(state.fields) if i not in to_remove]

        seen_ct = {}
        unique_cts = []
        for ct in state.clause_types:
            norm = re.sub(r'[_\s]', '', ct.name.lower())
            if norm not in seen_ct:
                seen_ct[norm] = True
                unique_cts.append(ct)
        state.clause_types = unique_cts

        return state

    def _to_field_dicts(self, state):
        type_map = {"string": "text", "money": "currency", "ref": "text"}
        valid_types = {
            "text", "date", "duration", "currency", "percentage",
            "integer", "boolean", "entity_name", "enum",
        }

        result = []
        for f in state.fields[:self.config.max_fields_per_cluster]:
            ftype = f.value_type.strip().lower()
            ftype = type_map.get(ftype, ftype)
            if ftype not in valid_types:
                ftype = "text"
            result.append({
                "name": f.name,
                "type": ftype,
                "description": f.description,
                "examples": [f.span_text] if f.span_text else [],
                "source": f.source,
            })
        return result


def discover_fields_for_cluster_rlm_v2(
    client,
    deployment: str,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config,
    embedder=None,
) -> tuple[list[dict], dict]:
    pipeline = LatentRLMv2(
        client=client,
        deployment=deployment,
        config=config,
        embedder=embedder,
    )
    return pipeline.discover_for_cluster(cluster_label, keywords, chunk_texts)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def grounding_detail(value: str, chunk_text: str) -> dict:
    if not value or not chunk_text:
        return {"level": "missing", "score": 0.0}
    v_lower = value.lower().strip()
    c_lower = chunk_text.lower()
    if v_lower in c_lower:
        return {"level": "verbatim", "score": 1.0}
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    c_tokens = set(re.findall(r'\b\w+\b', c_lower))
    if not v_tokens:
        return {"level": "empty", "score": 0.0}
    overlap = v_tokens & c_tokens
    token_ratio = len(overlap) / len(v_tokens)
    if token_ratio >= 0.8:
        return {"level": "high_overlap", "score": 0.8, "token_ratio": round(token_ratio, 3)}
    elif token_ratio >= 0.5:
        return {"level": "partial_overlap", "score": 0.5, "token_ratio": round(token_ratio, 3)}
    elif token_ratio >= 0.2:
        return {"level": "weak_overlap", "score": 0.2, "token_ratio": round(token_ratio, 3)}
    else:
        return {"level": "hallucinated", "score": 0.0, "token_ratio": round(token_ratio, 3)}


def classify_value(value: str) -> str:
    if not value:
        return "null"
    v = value.strip()
    if re.match(r'^\d+$', v):
        return "integer"
    if re.match(r'^\$?[\d,]+\.?\d*$', v):
        return "currency"
    if re.match(r'^\d+(\.\d+)?%$', v):
        return "percentage"
    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', v):
        return "date"
    if v.lower() in ("true", "false", "yes", "no"):
        return "boolean"
    if len(v) > 100:
        return "long_text"
    return "short_text"


def run_extraction_comparison(db_path: str, n_clusters: int = 5,
                              n_extract_chunks: int = 8):
    conn = duckdb.connect(db_path, read_only=True)
    config = FieldDiscoveryConfig()

    client = _get_azure_client()
    deployment = _get_deployment(config)
    logger.info(f"Azure deployment: {deployment}")

    all_clusters = conn.execute("""
        SELECT c.cluster_id, c.label, c.keywords, c.chunk_count
        FROM clusters c
        WHERE c.chunk_count >= 15
        ORDER BY c.chunk_count DESC
    """).fetchdf().to_dict("records")

    logger.info(f"Found {len(all_clusters)} clusters with ≥15 chunks")

    total = len(all_clusters)
    if total <= n_clusters:
        sample = all_clusters
    else:
        n_top = max(1, int(n_clusters * 0.4))
        n_mid = max(1, int(n_clusters * 0.3))
        n_bot = n_clusters - n_top - n_mid
        top_idx = list(range(n_top))
        mid_start = total // 3
        mid_idx = list(range(mid_start, mid_start + n_mid))
        bot_start = max(total - n_bot, mid_start + n_mid)
        bot_idx = list(range(bot_start, bot_start + n_bot))
        indices = sorted(set(top_idx + mid_idx + bot_idx))
        sample = [all_clusters[i] for i in indices if i < total]

    logger.info(f"Sampled {len(sample)} clusters for 4-way comparison")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded MiniLM-L6-v2 for v2 embedding scoring")

    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    all_results = []

    for ci, cluster in enumerate(sample):
        cluster_id = cluster["cluster_id"]
        cluster_label = cluster["label"] or "Unknown"
        chunk_count = cluster["chunk_count"]

        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunk_rows = conn.execute("""
            SELECT ch.chunk_id, ch.chunk_text
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cluster_id]).fetchdf().to_dict("records")

        chunk_texts = [r["chunk_text"] for r in chunk_rows]
        if len(chunk_texts) < 5:
            continue

        extract_chunks = chunk_texts[:n_extract_chunks]

        logger.info(f"\n{'='*80}")
        logger.info(f"[{ci+1}/{len(sample)}] {cluster_label} ({len(chunk_texts)} chunks, extracting from {len(extract_chunks)})")
        logger.info(f"{'='*80}")

        cluster_result = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "chunk_count": chunk_count,
            "n_extract_chunks": len(extract_chunks),
            "methods": {},
        }

        # ── Step 1: Discover fields with each method ──

        # Standard GPT
        logger.info(f"  [Standard GPT] Discovering fields...")
        t0 = time.time()
        try:
            fields_std = discover_fields_for_cluster(
                client, deployment, cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  Standard GPT failed: {e}")
            fields_std = []
        time_std_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_std)} fields in {time_std_disc}s")

        # DSPy RLM
        logger.info(f"  [DSPy RLM] Discovering fields...")
        t0 = time.time()
        try:
            fields_rlm = discover_fields_for_cluster_rlm(
                cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  DSPy RLM failed: {e}")
            fields_rlm = []
        time_rlm_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_rlm)} fields in {time_rlm_disc}s")

        # RLM + Latent Briefing
        logger.info(f"  [RLM+LB] Discovering fields...")
        t0 = time.time()
        try:
            fields_lb, briefing_meta = discover_fields_for_cluster_rlm_lb(
                cluster_label, keywords, chunk_texts, config,
                briefing_tau=1.5, briefing_max_tokens=2000, adaptive_tau=True,
            )
        except Exception as e:
            logger.warning(f"  RLM+LB failed: {e}")
            fields_lb = []
            briefing_meta = {"error": str(e)}
        time_lb_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_lb)} fields in {time_lb_disc}s")

        # RLM v2 (spec-aligned)
        logger.info(f"  [RLM v2] Discovering fields (rolling state + dual mode)...")
        t0 = time.time()
        try:
            fields_v2, v2_meta = discover_fields_for_cluster_rlm_v2(
                client, deployment, cluster_label, keywords, chunk_texts, config,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning(f"  RLM v2 failed: {e}")
            fields_v2 = []
            v2_meta = {"error": str(e)}
        time_v2_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_v2)} fields in {time_v2_disc}s")
        if v2_meta.get("n_clause_types"):
            logger.info(f"      + {v2_meta['n_clause_types']} clause types, "
                         f"{v2_meta.get('n_intents', 0)} intents, "
                         f"{v2_meta.get('n_obligations', 0)} obligations")

        method_fields = {
            "standard_gpt": fields_std,
            "dspy_rlm": fields_rlm,
            "rlm_lb": fields_lb,
            "rlm_v2": fields_v2,
        }
        disc_times = {
            "standard_gpt": time_std_disc,
            "dspy_rlm": time_rlm_disc,
            "rlm_lb": time_lb_disc,
            "rlm_v2": time_v2_disc,
        }

        # ── Step 2: Extract values from the SAME chunks with each method ──

        for method in METHODS:
            fields = method_fields[method]
            if not fields:
                cluster_result["methods"][method] = {
                    "n_fields": 0,
                    "fields": [],
                    "discovery_time_s": disc_times[method],
                    "extraction_time_s": 0,
                    "extractions": [],
                    "aggregate": _empty_aggregate(),
                }
                continue

            logger.info(f"  [{method}] Extracting from {len(extract_chunks)} chunks × {len(fields)} fields...")

            rlm_context = None
            if method == "rlm_v2" and v2_meta:
                rlm_context = {
                    "clause_types": v2_meta.get("clause_types", []),
                    "intents": v2_meta.get("intents", []),
                    "obligations": v2_meta.get("obligations", []),
                }

            t0 = time.time()
            chunk_extractions = []
            for chunk_idx, chunk_text in enumerate(extract_chunks):
                try:
                    ext = extract_fields_from_chunk(
                        client, deployment, cluster_label, chunk_text, fields, config,
                        rlm_context=rlm_context,
                    )
                except Exception as e:
                    logger.warning(f"    Chunk {chunk_idx} extraction failed: {e}")
                    ext = {}

                chunk_detail = {
                    "chunk_index": chunk_idx,
                    "chunk_preview": chunk_text[:200],
                    "fields": {},
                }
                for f in fields:
                    fname = f["name"]
                    if fname in ext:
                        value, confidence = ext[fname]
                        grounding = grounding_detail(value, chunk_text)
                        chunk_detail["fields"][fname] = {
                            "value": value,
                            "confidence": confidence,
                            "grounding": grounding,
                            "value_type": classify_value(value),
                        }
                    else:
                        chunk_detail["fields"][fname] = {
                            "value": None,
                            "confidence": 0.0,
                            "grounding": {"level": "missing", "score": 0.0},
                            "value_type": "null",
                        }

                chunk_extractions.append(chunk_detail)

            extraction_time = round(time.time() - t0, 2)
            agg = _compute_aggregate(fields, chunk_extractions)

            extra_meta = {}
            if method == "rlm_lb":
                extra_meta["briefing_meta"] = briefing_meta
            elif method == "rlm_v2":
                extra_meta["v2_meta"] = v2_meta

            cluster_result["methods"][method] = {
                "n_fields": len(fields),
                "fields": fields,
                "discovery_time_s": disc_times[method],
                "extraction_time_s": extraction_time,
                "extractions": chunk_extractions,
                "aggregate": agg,
                **extra_meta,
            }

            logger.info(f"    → fill={agg['fill_rate']:.3f}, conf={agg['avg_confidence']:.3f}, "
                         f"verbatim={agg['verbatim_rate']:.3f}, halluc={agg['hallucination_rate']:.3f}, "
                         f"time={extraction_time}s")

        all_results.append(cluster_result)

    conn.close()

    # ── Generate report ──
    report = _build_report(all_results)

    out_path = Path(db_path).parent / "extraction_value_comparison_v2.json"
    with open(out_path, "w") as f:
        json.dump({"report": report, "per_cluster": all_results}, f, indent=2, default=str)
    logger.info(f"\nJSON saved to {out_path}")

    _print_report(report, all_results)

    return report, all_results


def _empty_aggregate():
    return {
        "fill_rate": 0, "avg_confidence": 0, "median_confidence": 0,
        "verbatim_rate": 0, "high_overlap_rate": 0, "partial_rate": 0,
        "hallucination_rate": 0, "n_extractions": 0, "n_total_slots": 0,
    }


def _compute_aggregate(fields, chunk_extractions):
    n_fields = len(fields)
    n_chunks = len(chunk_extractions)
    total_slots = n_fields * n_chunks

    values, confs, groundings = [], [], []

    for ce in chunk_extractions:
        for f in fields:
            fname = f["name"]
            fd = ce["fields"].get(fname, {})
            val = fd.get("value")
            if val is not None:
                values.append(val)
                confs.append(fd.get("confidence", 0))
                groundings.append(fd.get("grounding", {}).get("level", "missing"))

    n_extracted = len(values)
    fill_rate = n_extracted / total_slots if total_slots else 0

    verbatim = sum(1 for g in groundings if g == "verbatim")
    high_overlap = sum(1 for g in groundings if g == "high_overlap")
    partial = sum(1 for g in groundings if g in ("partial_overlap", "weak_overlap"))
    halluc = sum(1 for g in groundings if g == "hallucinated")

    return {
        "fill_rate": round(fill_rate, 3),
        "avg_confidence": round(np.mean(confs), 3) if confs else 0,
        "median_confidence": round(np.median(confs), 3) if confs else 0,
        "verbatim_rate": round(verbatim / n_extracted, 3) if n_extracted else 0,
        "high_overlap_rate": round(high_overlap / n_extracted, 3) if n_extracted else 0,
        "partial_rate": round(partial / n_extracted, 3) if n_extracted else 0,
        "hallucination_rate": round(halluc / n_extracted, 3) if n_extracted else 0,
        "grounded_rate": round((verbatim + high_overlap) / n_extracted, 3) if n_extracted else 0,
        "n_extractions": n_extracted,
        "n_total_slots": total_slots,
    }


def _build_report(all_results):
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    report = {"n_clusters": len(all_results), "methods": {}}

    for method in METHODS:
        aggs = [r["methods"][method]["aggregate"] for r in all_results
                if r["methods"].get(method)]
        n_fields_list = [r["methods"][method]["n_fields"] for r in all_results
                         if r["methods"].get(method)]
        disc_times = [r["methods"][method]["discovery_time_s"] for r in all_results
                      if r["methods"].get(method)]
        ext_times = [r["methods"][method]["extraction_time_s"] for r in all_results
                     if r["methods"].get(method)]

        failures = sum(1 for nf in n_fields_list if nf == 0)
        valid_aggs = [a for a in aggs if a["n_extractions"] > 0]

        report["methods"][method] = {
            "avg_fields": round(np.mean(n_fields_list), 1) if n_fields_list else 0,
            "failures": failures,
            "avg_fill_rate": round(np.mean([a["fill_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_confidence": round(np.mean([a["avg_confidence"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_verbatim_rate": round(np.mean([a["verbatim_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_grounded_rate": round(np.mean([a["grounded_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_hallucination_rate": round(np.mean([a["hallucination_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "total_extractions": sum(a["n_extractions"] for a in aggs),
            "total_slots": sum(a["n_total_slots"] for a in aggs),
            "avg_discovery_time_s": round(np.mean(disc_times), 1) if disc_times else 0,
            "avg_extraction_time_s": round(np.mean(ext_times), 1) if ext_times else 0,
        }

    return report


def _print_report(report, all_results):
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    LABELS = {"standard_gpt": "Standard GPT", "dspy_rlm": "DSPy RLM",
              "rlm_lb": "RLM + LB", "rlm_v2": "RLM v2"}
    W = 115

    print(f"\n{'='*W}")
    print(f" 4-WAY EXTRACTION VALUE COMPARISON")
    print(f" Standard GPT vs DSPy RLM vs RLM + Latent Briefing vs RLM v2 (Spec-Aligned)")
    print(f" {report['n_clusters']} clusters, {8} chunks/cluster extraction sample")
    print(f"{'='*W}")

    rm = report["methods"]
    print(f"\n┌{'─'*40}┬{'─'*16}┬{'─'*16}┬{'─'*16}┬{'─'*16}┐")
    print(f"│ {'Metric':<38} │ {'Standard GPT':>14} │ {'DSPy RLM':>14} │ {'RLM + LB':>14} │ {'RLM v2':>14} │")
    print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")

    def _v(method, key, fmt=".1f"):
        val = rm.get(method, {}).get(key, 0)
        return f"{val:{fmt}}"

    rows = [
        ("FIELD DISCOVERY", "", "", "", ""),
        ("  Avg fields/cluster",
         _v("standard_gpt", "avg_fields"),
         _v("dspy_rlm", "avg_fields"),
         _v("rlm_lb", "avg_fields"),
         _v("rlm_v2", "avg_fields")),
        ("  Failures (0 fields)",
         str(rm.get("standard_gpt", {}).get("failures", 0)),
         str(rm.get("dspy_rlm", {}).get("failures", 0)),
         str(rm.get("rlm_lb", {}).get("failures", 0)),
         str(rm.get("rlm_v2", {}).get("failures", 0))),
        ("  Avg discovery time (s)",
         _v("standard_gpt", "avg_discovery_time_s"),
         _v("dspy_rlm", "avg_discovery_time_s"),
         _v("rlm_lb", "avg_discovery_time_s"),
         _v("rlm_v2", "avg_discovery_time_s")),
        ("", "", "", "", ""),
        ("EXTRACTION QUALITY", "", "", "", ""),
        ("  Fill rate",
         _v("standard_gpt", "avg_fill_rate", ".3f"),
         _v("dspy_rlm", "avg_fill_rate", ".3f"),
         _v("rlm_lb", "avg_fill_rate", ".3f"),
         _v("rlm_v2", "avg_fill_rate", ".3f")),
        ("  Avg confidence",
         _v("standard_gpt", "avg_confidence", ".3f"),
         _v("dspy_rlm", "avg_confidence", ".3f"),
         _v("rlm_lb", "avg_confidence", ".3f"),
         _v("rlm_v2", "avg_confidence", ".3f")),
        ("", "", "", "", ""),
        ("GROUNDING ANALYSIS", "", "", "", ""),
        ("  Verbatim match rate",
         _v("standard_gpt", "avg_verbatim_rate", ".3f"),
         _v("dspy_rlm", "avg_verbatim_rate", ".3f"),
         _v("rlm_lb", "avg_verbatim_rate", ".3f"),
         _v("rlm_v2", "avg_verbatim_rate", ".3f")),
        ("  Grounded rate (verbatim+high)",
         _v("standard_gpt", "avg_grounded_rate", ".3f"),
         _v("dspy_rlm", "avg_grounded_rate", ".3f"),
         _v("rlm_lb", "avg_grounded_rate", ".3f"),
         _v("rlm_v2", "avg_grounded_rate", ".3f")),
        ("  Hallucination rate",
         _v("standard_gpt", "avg_hallucination_rate", ".3f"),
         _v("dspy_rlm", "avg_hallucination_rate", ".3f"),
         _v("rlm_lb", "avg_hallucination_rate", ".3f"),
         _v("rlm_v2", "avg_hallucination_rate", ".3f")),
        ("", "", "", "", ""),
        ("TOTALS", "", "", "", ""),
        ("  Total extractions",
         str(rm.get("standard_gpt", {}).get("total_extractions", 0)),
         str(rm.get("dspy_rlm", {}).get("total_extractions", 0)),
         str(rm.get("rlm_lb", {}).get("total_extractions", 0)),
         str(rm.get("rlm_v2", {}).get("total_extractions", 0))),
        ("  Avg extraction time (s)",
         _v("standard_gpt", "avg_extraction_time_s"),
         _v("dspy_rlm", "avg_extraction_time_s"),
         _v("rlm_lb", "avg_extraction_time_s"),
         _v("rlm_v2", "avg_extraction_time_s")),
    ]

    for label, v1, v2, v3, v4 in rows:
        if not label:
            print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")
        else:
            print(f"│ {label:<38} │ {v1:>14} │ {v2:>14} │ {v3:>14} │ {v4:>14} │")

    print(f"└{'─'*40}┴{'─'*16}┴{'─'*16}┴{'─'*16}┴{'─'*16}┘")

    # ── Per-cluster summary ──
    print(f"\n{'─'*W}")
    print(f" Per-Cluster Extraction Summary")
    print(f"{'─'*W}")
    hdr = (f"{'Cluster':<25} │ "
           f"{'S':>3} {'R':>3} {'L':>3} {'V':>3} │ "
           f"{'S.Fil':>5} {'R.Fil':>5} {'L.Fil':>5} {'V.Fil':>5} │ "
           f"{'S.Vrb':>5} {'R.Vrb':>5} {'L.Vrb':>5} {'V.Vrb':>5} │ "
           f"{'S.Hal':>5} {'R.Hal':>5} {'L.Hal':>5} {'V.Hal':>5}")
    print(hdr)
    print(f"{'─'*W}")

    for r in all_results:
        label = r["cluster_label"][:23]
        s = r["methods"].get("standard_gpt", {})
        d = r["methods"].get("dspy_rlm", {})
        l = r["methods"].get("rlm_lb", {})
        v = r["methods"].get("rlm_v2", {})

        sa = s.get("aggregate", _empty_aggregate())
        da = d.get("aggregate", _empty_aggregate())
        la = l.get("aggregate", _empty_aggregate())
        va = v.get("aggregate", _empty_aggregate())

        print(f"{label:<25} │ "
              f"{s.get('n_fields',0):>3} {d.get('n_fields',0):>3} {l.get('n_fields',0):>3} {v.get('n_fields',0):>3} │ "
              f"{sa['fill_rate']:>5.3f} {da['fill_rate']:>5.3f} {la['fill_rate']:>5.3f} {va['fill_rate']:>5.3f} │ "
              f"{sa['verbatim_rate']:>5.3f} {da['verbatim_rate']:>5.3f} {la['verbatim_rate']:>5.3f} {va['verbatim_rate']:>5.3f} │ "
              f"{sa['hallucination_rate']:>5.3f} {da['hallucination_rate']:>5.3f} {la['hallucination_rate']:>5.3f} {va['hallucination_rate']:>5.3f}")

    # ── RLM v2 bonus outputs ──
    print(f"\n{'─'*W}")
    print(f" RLM v2 Bonus Outputs (beyond fields)")
    print(f"{'─'*W}")
    for r in all_results:
        v2m = r["methods"].get("rlm_v2", {})
        v2_meta = v2m.get("v2_meta", {})
        print(f"  {r['cluster_label'][:30]:<32} "
              f"clause_types={v2_meta.get('n_clause_types', 0):>2}, "
              f"intents={v2_meta.get('n_intents', 0):>2}, "
              f"obligations={v2_meta.get('n_obligations', 0):>2}, "
              f"batches={v2_meta.get('iterations', 0)}, "
              f"explore_ratio={v2_meta.get('exploration_ratio', 0):.2f}, "
              f"canon_removed={v2_meta.get('canonicalization_removed', 0)}")

    # ── Grounding distribution ──
    print(f"\n{'─'*W}")
    print(f" Grounding Distribution (all extractions)")
    print(f"{'─'*W}")
    print(f"  {'Level':<20} {'Standard GPT':>14} {'DSPy RLM':>14} {'RLM + LB':>14} {'RLM v2':>14}")
    print(f"  {'─'*78}")

    for level in ["verbatim", "high_overlap", "partial_overlap", "weak_overlap", "hallucinated"]:
        counts = []
        for method in METHODS:
            total = 0
            for r in all_results:
                m = r["methods"].get(method, {})
                for ext in m.get("extractions", []):
                    for fname, fdata in ext.get("fields", {}).items():
                        if fdata.get("value") is not None and fdata.get("grounding", {}).get("level") == level:
                            total += 1
            counts.append(total)
        print(f"  {level:<20} {counts[0]:>14} {counts[1]:>14} {counts[2]:>14} {counts[3]:>14}")

    # ── Winner determination ──
    print(f"\n{'='*W}")
    print(f" WINNER ANALYSIS")
    print(f"{'='*W}")

    best_fill = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_fill_rate", 0))
    best_conf = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_confidence", 0))
    best_grounded = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_grounded_rate", 0))
    best_verbatim = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_verbatim_rate", 0))
    least_halluc = min(METHODS, key=lambda m: rm.get(m, {}).get("avg_hallucination_rate", 1))
    most_fields = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_fields", 0))

    print(f"  Most fields discovered:     {LABELS[most_fields]} ({rm.get(most_fields, {}).get('avg_fields', 0):.1f} avg)")
    print(f"  Best fill rate:             {LABELS[best_fill]} ({rm.get(best_fill, {}).get('avg_fill_rate', 0):.3f})")
    print(f"  Highest confidence:         {LABELS[best_conf]} ({rm.get(best_conf, {}).get('avg_confidence', 0):.3f})")
    print(f"  Best verbatim match:        {LABELS[best_verbatim]} ({rm.get(best_verbatim, {}).get('avg_verbatim_rate', 0):.3f})")
    print(f"  Best grounded rate:         {LABELS[best_grounded]} ({rm.get(best_grounded, {}).get('avg_grounded_rate', 0):.3f})")
    print(f"  Lowest hallucination:       {LABELS[least_halluc]} ({rm.get(least_halluc, {}).get('avg_hallucination_rate', 0):.3f})")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-way extraction value comparison (with RLM v2)")
    parser.add_argument("db_path", help="Path to DuckDB file")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--n-chunks", type=int, default=8, help="Chunks to extract per cluster (default: 8)")
    args = parser.parse_args()

    run_extraction_comparison(args.db_path, n_clusters=args.n_clusters,
                              n_extract_chunks=args.n_chunks)
