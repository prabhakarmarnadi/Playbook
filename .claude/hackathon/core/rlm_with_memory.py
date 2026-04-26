"""
RLM with Latent Briefing — Memory-Augmented Recursive Language Model
=====================================================================

HOW RLMs WORK — THE FOUNDATION
================================

A Recursive Language Model (RLM) is a DSPy module (dspy.RLM) that turns
a single LLM call into a multi-turn agentic loop.  The core idea, from
Weitekamp et al. (Feb 2026, "Recursive Language Models as Memory Systems"):

  Normal LLM call:
    Input → LLM → Output           (one shot, no iteration)

  RLM (CodeAct REPL):
    Input → LLM writes Python code → REPL executes code → output captured
         → LLM sees output, writes more code → REPL → ...
         → LLM calls SUBMIT(final_answer) → done

The orchestrator model doesn't just answer — it writes code in a sandbox,
sees the execution results, and iterates.  This is the CodeAct pattern:
the LLM acts as a programmer running an interactive REPL.

Key primitive: llm_query(prompt)
  The orchestrator can delegate sub-tasks to a worker LLM.  The orchestrator
  decides WHAT to ask; the worker does the actual analysis.  This separation
  means the orchestrator can break a big task into sub-queries, inspect
  intermediate results, and adjust strategy — which is why RLMs beat
  single-shot prompting on long-context tasks.

DSPy's RLM lifecycle (what super().__init__ sets up):
  1. Signature: defines input fields (chunks_json, cluster_label) and
     output fields (fields_json).  The REPL loop runs until the agent
     calls SUBMIT with the output field values.
  2. max_iterations: hard cap on REPL turns (default 25).
  3. max_llm_calls: hard cap on total worker calls (default 50).
  4. tools: Python functions that the agent can call in its code.
  5. sub_lm: separate model for llm_query() worker calls (can be cheaper).
  6. _make_llm_tools(): creates the llm_query() and llm_query_batched()
     functions that the REPL code can call.
  7. _execute_iteration(): runs one REPL turn (LLM generates code →
     code is executed → output captured → history updated).
  8. forward(**inputs): the entry point.  Runs the REPL loop until
     SUBMIT or max_iterations, returns a dspy.Prediction.

HOW THIS FILE EXTENDS THE BASE RLM
====================================

This file subclasses dspy.RLM to add two capabilities:

  A) LATENT BRIEFING MEMORY INJECTION
  B) V2-STYLE ROLLING STATE + DUAL-MODE DISCOVERY

────────────────────────────────────────────────────────────────────────
A) LATENT BRIEFING — Trajectory Compression for Worker Calls
────────────────────────────────────────────────────────────────────────

Problem: As the RLM iterates, its trajectory (code + outputs from all
prior iterations) grows.  When the agent calls llm_query(task), the
worker needs context from this trajectory to give a good answer.  But
passing the full trajectory is expensive and noisy — most of it is
irrelevant to the specific sub-task.

Solution: Before each llm_query() call, compress the trajectory into
a task-specific "briefing" using embedding similarity scoring.

How it works step by step:

  1. TRAJECTORY CAPTURE (_execute_iteration override):
     After each REPL iteration, we extract the code, output, and
     reasoning from the iteration and store it as a "trajectory segment"
     in the TextLatentBriefing engine.  Worker replies are also captured.

       Segment example:
         "[Iteration 3]
          Reasoning: Let's look for date-related fields...
          Code: result = llm_query('Find date fields in: ...')
          Output: Found 3 date fields: effective_date, ..."

  2. TASK-GUIDED SCORING (_compact_for_worker):
     When llm_query(task) is called, we score every trajectory segment
     against the task prompt using cosine similarity:

       a. Encode the task prompt into a vector (MiniLM, 384-dim)
       b. Encode every trajectory segment into vectors
       c. Compute cosine similarity: score_i = cos(task_vec, segment_i_vec)
       d. Apply MAD thresholding: keep segments where
            score > median(scores) + tau * MAD(scores)
          (MAD = median absolute deviation)
       e. Concatenate surviving segments into a "briefing" text
       f. Truncate to briefing_max_tokens (default 2000)

     The key insight: the SAME trajectory produces DIFFERENT briefings
     for different tasks.  If the agent asks "find date fields", only
     segments mentioning dates/durations score high.  If the agent asks
     "find financial fields", different segments survive.

  3. WORKER AUGMENTATION (_make_llm_tools override):
     The briefing text is prepended to the worker prompt:

       "=== CONTEXT FROM PRIOR ANALYSIS ===
       [Iteration 1] Found fields: renewal_period, ...
       [Iteration 2] Clause types: auto_renewal, ...
       === END CONTEXT ===

       Find obligation-related fields in: ..."

     The worker sees exactly the context it needs — not 10 pages of
     trajectory noise.

  4. ADAPTIVE TAU (_compute_adaptive_tau):
     The compaction aggressiveness adapts over time:
       - Early iterations: light compression (tau stays low, preserve
         exploration context because we haven't built much yet)
       - Late iterations: aggressive compression (tau increases,
         filter dead-end noise that accumulated)
       - Very long trajectories: slightly lighter (tau drops,
         preserve dispersed evidence across many segments)

     Formula: effective_tau = base_tau * depth_factor * length_factor
       depth_factor = min(1.0 + iteration/15 * 0.5, 2.0)
       length_factor = 0.8 if >20 segments, 0.9 if >10, 1.0 otherwise

  5. STATISTICS (forward override):
     After the full RLM loop completes, we attach briefing_stats to
     the Prediction: n_compactions, total_tokens_saved, avg_compression_ratio,
     avg_retention_rate, per-compaction breakdown.

Architecture (2 API models + local embeddings):
  ┌────────────────────────────────────────────────────────────────┐
  │ Model 1 — Orchestrator (e.g. GPT-4o)                         │
  │ Runs the RLM CodeAct REPL loop, generates Python code         │
  │                                                                │
  │  Iteration 1: code → REPL output                              │
  │       ↓ captured as trajectory segment                        │
  │  Iteration 2: code calls llm_query(task)                      │
  │       ↓ TextLatentBriefing compacts trajectory (MiniLM)       │
  │       ↓ prepend briefing to worker prompt                     │
  │       ──→ Model 2 — Worker (e.g. GPT-4o-mini) answers        │
  │       ↓ worker reply captured as trajectory segment           │
  │  Iteration 3: code calls llm_query(task2)                     │
  │       ↓ DIFFERENT briefing for task2 (task-guided)            │
  │       ──→ Model 2 sees different context this time            │
  │  ...                                                           │
  │  SUBMIT: final aggregated answer                               │
  └────────────────────────────────────────────────────────────────┘

  MiniLM (~80MB, local, CPU) — sentence-transformer for scoring
  trajectory segments by cosine similarity. NOT an LLM.

────────────────────────────────────────────────────────────────────────
B) V2-STYLE ROLLING STATE + DUAL-MODE DISCOVERY
────────────────────────────────────────────────────────────────────────

The discover_with_rolling_state() method layers V2 features on top of
the RLM+LB base.  Instead of relying on the REPL agent to manage its
own state, we impose structured state management externally:

  Per chunk batch:
  1. Build embedding-scored brief from rolling state (§4.2):
     - Embed all state items (discovered fields, clause types)
     - Embed the next chunk batch
     - Score by cosine similarity with recency boost
     - Top-K items become the "brief" for this batch

  2. Grounded pass (temp 0.2, with brief):
     - Prompt: "You are a legal analyst. Here's what we've found so far.
       Find NEW patterns in these excerpts."
     - Worker gets trajectory briefing + state brief
     - Returns 4 output types: fields, clause_types, intents, obligations

  3. Exploratory pass (temp 0.8, NO brief):
     - Prompt: "With FRESH EYES — find unusual, niche patterns."
     - Worker gets NO prior context (blank slate)
     - Returns same 4 output types

  4. Overlap-penalty merge (§4.5):
     - For each exploratory item, compute cosine sim to every grounded item
     - If max_sim > 0.85 → drop exploratory (grounded wins, it's a duplicate)
     - If max_sim < 0.85 → keep exploratory (it's genuinely novel)
     - Purpose: grounded mode gives precision; exploratory gives recall.
       The merge keeps novel exploratory finds while deduplicating redundant ones.

  5. Update rolling state (§4.1):
     - Merge new fields/clause_types/intents/obligations into the accumulator
     - Dedup by normalized name (snake_case, lowercase)
     - Update anchor terms (frequency-counted keywords from chunks)

  6. Adaptive stopping (§4.3):
     - delta = |new_fields - old_fields| / max(new_fields, 1)
     - If delta < 0.05 for 3 consecutive batches → stop early
     - "We've stopped finding new things — no point processing more chunks"

  Post-loop:
  7. Two-tier canonicalization (§4.6):
     - Tier 1: Rule-based fuzzy merge:
       - Substring containment: "renewal" ⊂ "renewal_period" → merge
       - Plural/singular: "fee" vs "fees" → merge to shorter
       - NICHE PRESERVATION: never auto-merge two "exploratory" items
         (they're rare discoveries, worth keeping separate)
     - Tier 2: Embedding similarity > 0.82 → merge
       - "termination_notice_days" ≈ "notice_period_for_termination" → merge
       - Keeps the entry with the longer description

────────────────────────────────────────────────────────────────────────
C) MIPRO COMPILATION — Prompt Optimization
────────────────────────────────────────────────────────────────────────

compile_with_mipro() uses DSPy's MIPROv2 optimizer to tune the
orchestrator's instruction prefix and select optimal few-shot demos:

  1. Requires a training set of dspy.Example objects, each with
     chunks_json, cluster_label, and expected fields_json.
  2. MIPROv2 searches over instruction variants + demo selection
     across num_trials trials, maximizing field_discovery_metric().
  3. The metric scores:
     - Recall (35%): fraction of expected fields discovered
     - Precision (25%): fraction of predicted fields that are relevant
     - Type accuracy (20%): correct value_type assignments
     - Quality (20%): descriptions, span_text grounding
  4. The compiled state (optimized instructions + selected demos) is
     cached to disk and re-loaded on subsequent runs.

WARNING from empirical results (Weitekamp, §GEPA):
  Prompt optimization can OVERFIT.  $400 of GEPA optimization regressed
  accuracy from 89.8% to 87.8%.  compile_with_mipro() requires ≥20
  training examples.  For smaller datasets, the structured output
  approach (V2/V3 Pydantic types) gives more reliable gains.

Training set utilities:
  - build_trainset_from_gold(): loads from eval/gold_standard.json
  - build_trainset_from_cuad(): maps CUAD's 41 clause categories to
    expected field schemas (e.g. "Renewal Term" → renewal_period, auto_renewal)

References:
  - Ramp Labs Latent Briefing: x.com/RampLabs/status/2042660310851449223
  - Attention Matching: arxiv.org/abs/2602.16284
  - DSPy RLM: arxiv.org/abs/2512.24601
  - Sparse Signal Loop (stochi0, Apr 2026)
  - RLMs Are the New Reasoning Models (Weitekamp, Apr 2026)
  - Recursive Language Models as Memory Systems (Weitekamp, Feb 2026)

Usage:
    from core.rlm_with_memory import RLMWithLatentBriefing

    rlm = RLMWithLatentBriefing(
        signature=FieldDiscoverySig,
        max_iterations=25,
        briefing_tau=1.5,          # Compaction aggressiveness
        briefing_max_tokens=2000,  # Hard cap on briefing size
    )
    result = rlm(chunks_json=..., cluster_label=...)
    print(result.fields_json)
    print(result.briefing_stats)  # Compaction statistics
"""

import json
import logging
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import dspy
from dspy import Prediction
from pydantic import BaseModel

from core.latent_briefing import TextLatentBriefing, CompactionConfig, CompactionStats

logger = logging.getLogger(__name__)


# ── Rolling Structured State (from RLM V2 §4.1) ─────────────────────────────

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
    """Rolling Structured State — typed, cluster-scoped accumulator (V2 §4.1)."""
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


class RLMWithLatentBriefing(dspy.RLM):
    """
    DSPy RLM subclass with Latent Briefing memory sharing.

    Overrides _make_llm_tools() to wrap llm_query/llm_query_batched with
    memory injection, and _execute_iteration() to capture trajectory
    segments after each REPL step.

    Designed for API-based models (Azure OpenAI, etc.) — uses embedding
    cosine similarity as a proxy for attention scoring. No KV cache
    access or self-hosted model required.
    """

    def __init__(
        self,
        signature,
        max_iterations: int = 25,
        max_llm_calls: int = 50,
        verbose: bool = False,
        sub_lm=None,
        tools: list[Callable] | None = None,
        # Latent Briefing parameters
        briefing_tau: float = 1.5,
        briefing_max_tokens: int = 2000,
        briefing_min_retention: float = 0.1,
        briefing_max_retention: float = 0.9,
        briefing_embed_model=None,
        adaptive_tau: bool = True,
    ):
        # Initialize Latent Briefing BEFORE super().__init__ so it's
        # available when _make_llm_tools() is called during forward()
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

        # Now call RLM's __init__
        super().__init__(
            signature=signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=verbose,
            sub_lm=sub_lm,
            tools=tools,
        )

    # ------------------------------------------------------------------
    # Override: inject memory-augmented llm_query / llm_query_batched
    # ------------------------------------------------------------------
    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        """Create memory-augmented llm_query tools.

        Gets the standard tools from RLM (with call counting, threading)
        then wraps them so every worker call receives a task-specific
        compacted briefing from the trajectory history.
        """
        base_tools = super()._make_llm_tools(max_workers)
        base_llm_query = base_tools["llm_query"]
        base_llm_query_batched = base_tools["llm_query_batched"]

        engine = self  # capture for closures

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

            # Record the exchange as a trajectory segment
            engine._add_to_trajectory(
                f"Worker query: {prompt[:200]}\nWorker response: {response[:500]}",
                segment_type="worker_reply",
            )
            return response

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query the LLM with multiple prompts concurrently."""
            if not prompts:
                return []

            # Compact once per batch using the first prompt as the
            # representative query (all prompts in a batch are typically
            # about the same subtask)
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

    # ------------------------------------------------------------------
    # Override: capture REPL iterations as trajectory segments
    # ------------------------------------------------------------------
    def _execute_iteration(self, repl, variables, history, iteration,
                           input_args, output_field_names):
        """Execute one iteration, then record it in the trajectory."""
        self._iteration_count = iteration

        result = super()._execute_iteration(
            repl, variables, history, iteration, input_args, output_field_names,
        )

        # result is either a Prediction (FINAL) or updated REPLHistory
        if isinstance(result, Prediction):
            # Final iteration — record the concluding reasoning
            reasoning = getattr(result, "final_reasoning", "")
            if reasoning:
                self._add_to_trajectory(
                    f"Final reasoning: {reasoning[:500]}",
                    segment_type="reasoning",
                )
        else:
            # Normal iteration — capture the latest history entry
            # REPLHistory is a list of entries with reasoning/code/output
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

    # ------------------------------------------------------------------
    # Override: reset briefing state at the start of each execution
    # ------------------------------------------------------------------
    def forward(self, **input_args) -> Prediction:
        """Execute RLM with Latent Briefing memory augmentation."""
        self.briefing_engine.reset()
        self._all_stats = []
        self._iteration_count = 0

        result = super().forward(**input_args)

        # Attach briefing statistics to the result
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compact_for_worker(self, task_prompt: str) -> str:
        """Produce a compacted briefing for a worker llm_query() call."""
        if self.briefing_engine.trajectory_length == 0:
            return ""

        tau = self._compute_adaptive_tau(
            self._iteration_count,
            self.briefing_engine.trajectory_length,
        )

        briefing_text, stats = self.briefing_engine.compact_for_task(
            task_prompt=task_prompt,
            tau=tau,
            max_tokens=self.briefing_max_tokens,
        )

        self._all_stats.append(stats)
        return briefing_text

    def _add_to_trajectory(self, text: str, segment_type: str = "observation"):
        """Add a segment to the trajectory memory."""
        self.briefing_engine.add_trajectory_segment(
            text=text,
            metadata={"iteration": self._iteration_count},
            segment_type=segment_type,
        )

    def _compute_adaptive_tau(self, iteration: int, n_segments: int) -> float:
        """Adapt compaction aggressiveness based on trajectory state.

        - Early iterations: light compression (preserve exploration context)
        - Later iterations: more aggressive (filter dead-end noise)
        - Very long trajectories: lighter (keep dispersed evidence)
        """
        base_tau = self.briefing_tau
        if not self.adaptive_tau:
            return base_tau

        depth_factor = min(1.0 + (iteration / 15) * 0.5, 2.0)

        if n_segments > 20:
            length_factor = 0.8
        elif n_segments > 10:
            length_factor = 0.9
        else:
            length_factor = 1.0

        return round(base_tau * depth_factor * length_factor, 2)

    # ------------------------------------------------------------------
    # V2 FEATURES: Rolling State + Dual-Mode + Canonicalization
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load MiniLM embedder for V2 features (overlap merge, canonicalization)."""
        if not hasattr(self, "_embedder") or self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def discover_with_rolling_state(
        self,
        cluster_label: str,
        keywords: list[str],
        chunk_texts: list[str],
        max_sample_chunks: int = 30,
        max_fields: int = 15,
        adaptive_stop_epsilon: float = 0.05,
    ) -> tuple[list[dict], dict]:
        """
        RLM+LB discovery enhanced with V2 rolling state + dual-mode exploration.

        Runs the RLM agentic loop per batch, but wraps it with:
        1. Rolling state accumulation across batches
        2. Dual-mode: each batch runs grounded + exploratory workers
        3. Overlap-penalty merge between grounded/exploratory results
        4. Post-loop two-tier canonicalization
        5. Adaptive early stopping

        Returns:
            (fields, meta) — field dicts compatible with extract_fields_from_chunk(),
            meta with rolling state stats + briefing stats.
        """
        sample = chunk_texts[:max_sample_chunks]
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
            "briefing_enabled": True,
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

            # Build embedding-scored brief from rolling state (V2 §4.2)
            brief = self._build_state_brief(state, chunks_text)

            # Grounded pass: run RLM+LB with brief context (low temp strategy)
            grounded = self._run_discovery_pass(
                chunks_text, cluster_label, keywords, brief,
                mode="grounded",
            )
            meta["n_grounded_calls"] += 1

            # Exploratory pass: fresh eyes, no brief (high temp strategy)
            exploratory = self._run_discovery_pass(
                chunks_text, cluster_label, keywords, "",
                mode="exploratory",
            )
            meta["n_exploratory_calls"] += 1

            # Overlap-penalty merge (V2 §4.5)
            merged, merge_stats = self._overlap_merge(grounded, exploratory)
            meta["exploratory_kept"] += merge_stats["kept"]
            meta["exploratory_dropped"] += merge_stats["dropped"]

            # Update rolling state (V2 §4.1)
            self._update_rolling_state(state, merged, batch)

            # Adaptive stopping (V2 §4.3)
            new_field_count = len(state.fields)
            if prev_field_count > 0:
                delta = abs(new_field_count - prev_field_count) / max(new_field_count, 1)
            else:
                delta = 1.0
            state.last_state_delta = delta
            state.iteration_count += 1
            meta["n_batches"] += 1

            logger.info(
                f"  RLM+LB+V2 batch {batch_idx + 1}/{n_batches}: "
                f"{len(state.fields)} fields, {len(state.clause_types)} clause types, "
                f"delta={delta:.3f}"
            )

            if delta < adaptive_stop_epsilon and state.iteration_count >= 3:
                meta["stopped_early"] = True
                logger.info(
                    f"  Adaptive stop after {state.iteration_count} batches "
                    f"(delta={delta:.3f} < {adaptive_stop_epsilon})"
                )
                break

        # Two-tier canonicalization (V2 §4.6)
        pre_canon = len(state.fields)
        state = self._canonicalize(state)
        post_canon = len(state.fields)

        # Convert to standard field dicts
        fields = self._state_to_field_dicts(state, max_fields)

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

        # Attach briefing stats from the RLM if available
        if self._all_stats:
            meta["briefing_stats"] = {
                "n_compactions": len(self._all_stats),
                "total_tokens_saved": self.briefing_engine.total_tokens_saved,
                "avg_compression_ratio": (
                    sum(s.compression_ratio for s in self._all_stats) / len(self._all_stats)
                ),
            }

        return fields, meta

    def _run_discovery_pass(
        self,
        chunks_text: str,
        cluster_label: str,
        keywords: list[str],
        brief: str,
        mode: str = "grounded",
    ) -> dict:
        """Run a single discovery pass via the RLM+LB's worker llm_query.

        In grounded mode: low-temp prompt with prior context brief.
        In exploratory mode: high-temp prompt looking for niche patterns.
        """
        if mode == "grounded":
            brief_section = ""
            if brief:
                brief_section = (
                    f"PRIOR CONTEXT (from previous chunk analysis):\n"
                    f"{brief}\n\n"
                    f"Build on this context — focus on NEW discoveries.\n\n"
                )
            prompt = (
                f"You are a legal contract analyst. Analyze these clause excerpts.\n\n"
                f"{brief_section}"
                f"CLAUSE TYPE: {cluster_label}\n"
                f"KEYWORDS: {', '.join(keywords[:10])}\n\n"
                f"EXCERPTS:\n{chunks_text}\n\n"
                f"Discover structured elements from the text:\n"
                f"1. FIELDS: Named data points (snake_case). Types: string, date, money, enum, ref, boolean, duration, percentage, integer\n"
                f"2. CLAUSE TYPES: Recurring patterns\n"
                f"3. INTENTS: Why each clause exists\n"
                f"4. OBLIGATIONS: Who must do what\n\n"
                f"Include exact span_text quotes. Return 5-15 fields.\n\n"
                f'Respond with ONLY JSON: {{"fields": [...], "clause_types": [...], "intents": [...], "obligations": [...]}}'
            )
        else:
            prompt = (
                f"You are a legal analyst specializing in NOVEL and NICHE patterns.\n\n"
                f"CLAUSE TYPE: {cluster_label}\n"
                f"KEYWORDS: {', '.join(keywords[:10])}\n\n"
                f"EXCERPTS:\n{chunks_text}\n\n"
                f"With FRESH EYES — find unusual, domain-specific patterns:\n"
                f"1. FIELDS: Unusual data points specific to this contract type (NOT generic like party_name)\n"
                f"2. CLAUSE TYPES: Non-obvious contractual patterns\n"
                f"3. INTENTS: Hidden purposes beyond surface reading\n"
                f"4. OBLIGATIONS: Subtle duties often overlooked\n\n"
                f"Include exact span_text quotes. Return 3-10 fields.\n\n"
                f'Respond with ONLY JSON: {{"fields": [...], "clause_types": [...], "intents": [...], "obligations": [...]}}'
            )

        # Use the briefing engine's worker path if we have trajectory history
        briefing = self._compact_for_worker(prompt) if self.briefing_engine.trajectory_length > 0 else ""
        if briefing:
            prompt = f"=== CONTEXT FROM PRIOR ANALYSIS ===\n{briefing}\n=== END CONTEXT ===\n\n{prompt}"

        # Direct LLM call (not through RLM REPL — we already have the REPL trajectory
        # being captured; this is a structured multi-output call)
        try:
            import os
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
            temp = 0.2 if mode == "grounded" else 0.8
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_completion_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)

            # Record in trajectory
            self._add_to_trajectory(
                f"[{mode.upper()} PASS] Discovered: "
                f"{len(parsed.get('fields', []))} fields, "
                f"{len(parsed.get('clause_types', []))} clause types",
                segment_type="worker_reply",
            )

            return {
                "fields": parsed.get("fields", []),
                "clause_types": parsed.get("clause_types", []),
                "intents": parsed.get("intents", []),
                "obligations": parsed.get("obligations", []),
            }
        except Exception as e:
            logger.warning(f"Discovery pass ({mode}) failed: {e}")
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}

    def _build_state_brief(self, state: RollingState, next_text: str,
                           k: int = 15, token_budget: int = 1200) -> str:
        """Build embedding-scored brief from rolling state (V2 §4.2)."""
        if not state.fields and not state.clause_types:
            return ""

        embedder = self._get_embedder()
        items = []
        for f in state.fields:
            items.append(("field", f.name, f"{f.name} ({f.value_type}): {f.description}"))
        for c in state.clause_types:
            items.append(("clause_type", c.name, f"{c.name}: {c.emergent_description}"))

        if not items:
            return ""

        query_text = next_text[:2000]
        all_texts = [query_text] + [text for _, _, text in items]
        all_vecs = embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)

        query_vec = all_vecs[0]
        item_vecs = all_vecs[1:]
        scores = np.dot(item_vecs, query_vec)

        # Recency boost (V2 §4.2)
        for idx in range(len(scores)):
            age = max(1, state.iteration_count - idx // max(1, len(items) // max(1, state.iteration_count)))
            recency_boost = 1.0 + 0.2 * (age ** -0.5)
            scores[idx] *= recency_boost

        sorted_indices = np.argsort(-scores)[:k]

        kept_fields, kept_clauses = [], []
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
            "already_found": {"fields": kept_fields, "clause_types": kept_clauses},
            "still_looking_for": state.open_questions[:5],
            "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
        }
        text = json.dumps(brief, indent=None)
        words = text.split()
        if len(words) > token_budget:
            text = " ".join(words[:token_budget])
        return text

    def _overlap_merge(self, grounded: dict, exploratory: dict,
                       threshold: float = 0.85) -> tuple[dict, dict]:
        """Merge grounded + exploratory with cosine overlap penalty (V2 §4.5)."""
        embedder = self._get_embedder()
        result = {}
        total_kept, total_dropped = 0, 0

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
            all_vecs = embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)
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

    def _update_rolling_state(self, state: RollingState, merged: dict,
                              batch_chunks: list[str]):
        """Update rolling state with merged results (V2 §4.1)."""
        existing_names = {f.name for f in state.fields}
        for f_dict in merged.get("fields", []):
            name = (f_dict.get("name") or "").strip()
            if not name:
                continue
            name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
            if name not in existing_names:
                state.fields.append(FieldV2(
                    name=name,
                    value_type=f_dict.get("value_type", f_dict.get("type", "string")),
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

        # Anchor terms
        existing_terms = dict(state.anchor_terms)
        for chunk in batch_chunks:
            words = re.findall(r'\b[a-z]{4,}\b', chunk.lower())
            for term, count in Counter(words).most_common(20):
                existing_terms[term] = existing_terms.get(term, 0) + count
        state.anchor_terms = sorted(existing_terms.items(), key=lambda x: -x[1])[:20]
        state.chunks_processed += len(batch_chunks)

    def _canonicalize(self, state: RollingState) -> RollingState:
        """Two-tier canonicalization (V2 §4.6): fuzzy name + embedding sim."""
        if len(state.fields) <= 1:
            return state

        # Tier 1: Rule-based fuzzy match
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
                # Niche preservation: skip if both exploratory
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

        # Tier 2: Embedding similarity
        remaining = [i for i in range(len(state.fields)) if i not in to_remove]
        if len(remaining) > 2:
            embedder = self._get_embedder()
            texts = [
                f"{state.fields[i].name}: {state.fields[i].description}"
                for i in remaining
            ]
            vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
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

        # Canonicalize clause types
        seen_ct = {}
        unique_cts = []
        for ct in state.clause_types:
            norm = re.sub(r'[_\s]', '', ct.name.lower())
            if norm not in seen_ct:
                seen_ct[norm] = True
                unique_cts.append(ct)
        state.clause_types = unique_cts

        return state

    def _state_to_field_dicts(self, state: RollingState, max_fields: int = 15) -> list[dict]:
        """Convert RollingState fields to extraction-compatible dicts."""
        type_map = {"string": "text", "money": "currency", "ref": "text"}
        valid_types = {"text", "date", "duration", "currency", "percentage",
                       "integer", "boolean", "entity_name", "enum"}
        result = []
        for f in state.fields[:max_fields]:
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

    # ------------------------------------------------------------------
    # MIPRO COMPILATION — Prompt Optimization via DSPy MIPROv2
    # ------------------------------------------------------------------

    @classmethod
    def compile_with_mipro(
        cls,
        signature,
        trainset: list[dspy.Example],
        metric: Optional[Callable] = None,
        auto: str = "medium",
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 4,
        num_trials: int = 15,
        cache_dir: str = ".cache/mipro",
        cache_name: str = "rlm_compiled",
        **rlm_kwargs,
    ) -> "RLMWithLatentBriefing":
        """
        Compile this RLM with MIPROv2 prompt optimization.

        MIPROv2 optimizes the orchestrator's instruction prefix and selects
        the best few-shot demos from the training set, maximizing the metric
        on held-out examples.

        Args:
            signature: DSPy Signature class (e.g., FieldDiscoverySig)
            trainset: List of dspy.Example with inputs + expected outputs
            metric: Scoring function(example, prediction, trace=None) -> float.
                    Defaults to field_discovery_metric if None.
            auto: MIPROv2 auto setting ("light", "medium", "heavy")
            max_bootstrapped_demos: Max bootstrapped few-shot demos
            max_labeled_demos: Max labeled few-shot demos
            num_trials: Number of optimization trials
            cache_dir: Directory to cache compiled program
            cache_name: Name for the cached compiled state
            **rlm_kwargs: Additional args passed to RLMWithLatentBriefing()

        Returns:
            Compiled RLMWithLatentBriefing instance with optimized prompts.
        """
        if metric is None:
            metric = field_discovery_metric

        # Check cache first
        cache_path = Path(cache_dir) / f"{cache_name}.json"
        if cache_path.exists():
            logger.info(f"Loading cached MIPRO-compiled program from {cache_path}")
            instance = cls(signature=signature, **rlm_kwargs)
            instance.load(str(cache_path))
            instance._mipro_compiled = True
            return instance

        logger.info(
            f"Compiling RLM+LB with MIPROv2 (auto={auto}, "
            f"trials={num_trials}, trainset={len(trainset)} examples)"
        )

        # Create the student program
        student = cls(signature=signature, **rlm_kwargs)

        # Run MIPROv2 compilation
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=auto,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            verbose=True,
        )

        compiled = optimizer.compile(
            student,
            trainset=trainset,
            num_trials=num_trials,
        )

        # Save compiled state
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(cache_path))
        logger.info(f"Saved MIPRO-compiled program to {cache_path}")

        compiled._mipro_compiled = True
        return compiled

    @classmethod
    def load_compiled(
        cls,
        signature,
        cache_dir: str = ".cache/mipro",
        cache_name: str = "rlm_compiled",
        **rlm_kwargs,
    ) -> Optional["RLMWithLatentBriefing"]:
        """Load a previously MIPRO-compiled program from cache.

        Returns None if no cached version exists.
        """
        cache_path = Path(cache_dir) / f"{cache_name}.json"
        if not cache_path.exists():
            logger.info(f"No compiled cache found at {cache_path}")
            return None

        instance = cls(signature=signature, **rlm_kwargs)
        instance.load(str(cache_path))
        instance._mipro_compiled = True
        logger.info(f"Loaded MIPRO-compiled RLM from {cache_path}")
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# MIPRO METRIC + TRAINING SET UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def field_discovery_metric(
    example: dspy.Example,
    prediction: Prediction,
    trace=None,
) -> float:
    """
    Metric for MIPRO optimization of the field discovery RLM.

    Scores based on:
      - Field coverage: how many expected fields were discovered (recall)
      - Field precision: how many discovered fields are relevant
      - Type accuracy: correct value_type assignments
      - Description quality: non-empty descriptions with reasonable length

    Returns a float in [0, 1].
    """
    # Parse expected fields from the example
    expected_raw = getattr(example, "fields_json", "[]")
    if isinstance(expected_raw, str):
        try:
            expected_fields = json.loads(expected_raw)
        except json.JSONDecodeError:
            expected_fields = []
    else:
        expected_fields = expected_raw

    # Parse predicted fields
    predicted_raw = getattr(prediction, "fields_json", "[]")
    if isinstance(predicted_raw, str):
        try:
            predicted_fields = json.loads(predicted_raw)
        except json.JSONDecodeError:
            return 0.0
    else:
        predicted_fields = predicted_raw if predicted_raw else []

    if not expected_fields:
        # If no ground truth, score on basic quality signals
        if not predicted_fields:
            return 0.0
        quality = _quality_score(predicted_fields)
        return min(quality, 0.7)  # Cap without ground truth

    if not predicted_fields:
        return 0.0

    # Normalize names for matching
    expected_names = {_normalize_field_name(f.get("name", "")) for f in expected_fields}
    predicted_names = {_normalize_field_name(f.get("name", "")) for f in predicted_fields}

    # Recall: what fraction of expected fields were found
    if expected_names:
        matched = expected_names & predicted_names
        # Also check fuzzy matches (substring containment)
        for en in expected_names - matched:
            for pn in predicted_names:
                if en in pn or pn in en:
                    matched.add(en)
                    break
        recall = len(matched) / len(expected_names)
    else:
        recall = 0.5

    # Precision: predicted fields that overlap with expected
    if predicted_names:
        pred_matched = predicted_names & expected_names
        for pn in predicted_names - pred_matched:
            for en in expected_names:
                if pn in en or en in pn:
                    pred_matched.add(pn)
                    break
        precision = len(pred_matched) / len(predicted_names)
    else:
        precision = 0.0

    # Type accuracy
    type_score = _type_accuracy(expected_fields, predicted_fields)

    # Description quality
    quality = _quality_score(predicted_fields)

    # Weighted combination
    score = (
        0.35 * recall
        + 0.25 * precision
        + 0.20 * type_score
        + 0.20 * quality
    )

    return round(min(max(score, 0.0), 1.0), 4)


def _normalize_field_name(name: str) -> str:
    """Normalize field name for comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _type_accuracy(expected: list[dict], predicted: list[dict]) -> float:
    """Score how well predicted types match expected types."""
    exp_map = {_normalize_field_name(f.get("name", "")): f.get("type", "text") for f in expected}
    matches = 0
    comparisons = 0
    for pf in predicted:
        pname = _normalize_field_name(pf.get("name", ""))
        if pname in exp_map:
            comparisons += 1
            if pf.get("type", "text") == exp_map[pname]:
                matches += 1
    return matches / max(comparisons, 1)


def _quality_score(fields: list[dict]) -> float:
    """Score intrinsic quality of discovered fields."""
    if not fields:
        return 0.0

    scores = []
    for f in fields:
        s = 0.0
        # Has a name
        if f.get("name"):
            s += 0.3
        # Has a non-trivial description
        desc = f.get("description", "")
        if len(desc) > 10:
            s += 0.3
        elif len(desc) > 0:
            s += 0.15
        # Has a valid type
        valid_types = {"text", "date", "duration", "currency", "percentage",
                       "integer", "boolean", "entity_name", "enum", "string",
                       "money", "ref"}
        if f.get("type", "") in valid_types or f.get("value_type", "") in valid_types:
            s += 0.2
        # Has examples/span
        if f.get("examples") or f.get("span_text"):
            s += 0.2
        scores.append(s)

    return sum(scores) / len(scores)


def build_trainset_from_gold(
    gold_path: str = "eval/gold_standard.json",
    max_examples: int = 50,
) -> list[dspy.Example]:
    """
    Build a MIPRO-compatible training set from the gold standard eval file.

    Each example represents one cluster/clause type with:
      - Input: chunks_json, cluster_label
      - Output: fields_json (expected discovered fields)

    Returns list of dspy.Example objects ready for MIPROv2.compile().
    """
    gold_file = Path(gold_path)
    if not gold_file.exists():
        logger.warning(f"Gold standard not found at {gold_path}")
        return []

    with open(gold_file) as f:
        gold = json.load(f)

    examples = []
    entries = gold if isinstance(gold, list) else gold.get("clusters", gold.get("examples", []))

    for entry in entries[:max_examples]:
        # Support multiple gold standard formats
        cluster_label = entry.get("cluster_label", entry.get("clause_type", ""))
        chunks = entry.get("chunks", entry.get("chunk_texts", []))
        expected_fields = entry.get("fields", entry.get("expected_fields", []))

        if not cluster_label or not chunks:
            continue

        chunks_json = json.dumps(chunks) if isinstance(chunks, list) else chunks
        fields_json = json.dumps(expected_fields) if isinstance(expected_fields, list) else expected_fields

        ex = dspy.Example(
            chunks_json=chunks_json,
            cluster_label=cluster_label,
            fields_json=fields_json,
        ).with_inputs("chunks_json", "cluster_label")

        examples.append(ex)

    logger.info(f"Built {len(examples)} training examples from {gold_path}")
    return examples


def build_trainset_from_cuad(
    cuad_json_path: str = "data/cuad/CUADv1.json",
    max_examples: int = 40,
) -> list[dspy.Example]:
    """
    Build a MIPRO-compatible training set from CUAD annotations.

    Maps CUAD's 41 clause categories directly to expected field types,
    producing training examples with clause text + expected field schemas.
    """
    cuad_path = Path(cuad_json_path)
    if not cuad_path.exists():
        logger.warning(f"CUAD JSON not found at {cuad_json_path}")
        return []

    with open(cuad_path) as f:
        cuad = json.load(f)

    # CUAD category → expected field schema mapping
    category_fields = {
        "Parties": [
            {"name": "party_name", "type": "entity_name", "description": "Name of contracting party"},
            {"name": "party_role", "type": "enum", "description": "Role (buyer, seller, licensor, etc.)"},
        ],
        "Agreement Date": [
            {"name": "agreement_date", "type": "date", "description": "Date agreement was executed"},
        ],
        "Effective Date": [
            {"name": "effective_date", "type": "date", "description": "Date agreement becomes effective"},
        ],
        "Expiration Date": [
            {"name": "expiration_date", "type": "date", "description": "Date agreement expires"},
            {"name": "term_duration", "type": "duration", "description": "Duration of agreement term"},
        ],
        "Renewal Term": [
            {"name": "renewal_period", "type": "duration", "description": "Length of renewal term"},
            {"name": "auto_renewal", "type": "boolean", "description": "Whether renewal is automatic"},
        ],
        "Termination For Convenience": [
            {"name": "notice_period", "type": "duration", "description": "Required notice for termination"},
            {"name": "terminating_party", "type": "entity_name", "description": "Who can terminate"},
        ],
        "Governing Law": [
            {"name": "governing_jurisdiction", "type": "text", "description": "Jurisdiction whose laws govern"},
        ],
        "Most Favored Nation": [
            {"name": "mfn_beneficiary", "type": "entity_name", "description": "Party receiving MFN protection"},
            {"name": "mfn_trigger", "type": "text", "description": "Condition triggering MFN clause"},
        ],
        "Non-Compete": [
            {"name": "restricted_period", "type": "duration", "description": "Duration of non-compete"},
            {"name": "geographic_scope", "type": "text", "description": "Geographic limitation"},
            {"name": "restricted_activities", "type": "text", "description": "Activities restricted"},
        ],
        "Exclusivity": [
            {"name": "exclusivity_scope", "type": "text", "description": "Scope of exclusive rights"},
            {"name": "exclusivity_territory", "type": "text", "description": "Geographic territory"},
        ],
        "Change Of Control": [
            {"name": "trigger_event", "type": "text", "description": "What constitutes change of control"},
            {"name": "consent_required", "type": "boolean", "description": "Whether consent is needed"},
        ],
        "Liquidated Damages": [
            {"name": "damage_amount", "type": "currency", "description": "Liquidated damage amount"},
            {"name": "damage_trigger", "type": "text", "description": "Event triggering damages"},
        ],
        "Cap On Liability": [
            {"name": "liability_cap", "type": "currency", "description": "Maximum liability amount"},
            {"name": "cap_basis", "type": "text", "description": "Basis for cap calculation"},
        ],
        "Insurance": [
            {"name": "coverage_amount", "type": "currency", "description": "Required insurance coverage"},
            {"name": "insurance_type", "type": "text", "description": "Type of insurance required"},
        ],
        "Price Restrictions": [
            {"name": "price_cap", "type": "currency", "description": "Maximum price allowed"},
            {"name": "price_floor", "type": "currency", "description": "Minimum price required"},
        ],
        "Minimum Commitment": [
            {"name": "minimum_amount", "type": "currency", "description": "Minimum purchase/commitment"},
            {"name": "commitment_period", "type": "duration", "description": "Period for minimum"},
        ],
        "Revenue/Profit Sharing": [
            {"name": "share_percentage", "type": "percentage", "description": "Revenue/profit share %"},
            {"name": "calculation_basis", "type": "text", "description": "How share is calculated"},
        ],
        "Ip Ownership Assignment": [
            {"name": "assigned_ip", "type": "text", "description": "IP being assigned"},
            {"name": "assignee", "type": "entity_name", "description": "Party receiving IP rights"},
        ],
        "License Grant": [
            {"name": "license_scope", "type": "text", "description": "Scope of license"},
            {"name": "license_type", "type": "enum", "description": "Exclusive, non-exclusive, etc."},
            {"name": "licensed_territory", "type": "text", "description": "Territory of license"},
        ],
        "Warranty Duration": [
            {"name": "warranty_period", "type": "duration", "description": "Length of warranty"},
        ],
        "Post-Termination Services": [
            {"name": "service_period", "type": "duration", "description": "Post-termination service period"},
            {"name": "service_scope", "type": "text", "description": "What services continue"},
        ],
        "Audit Rights": [
            {"name": "audit_notice_period", "type": "duration", "description": "Notice before audit"},
            {"name": "audit_frequency", "type": "text", "description": "How often audits can occur"},
        ],
    }

    # Group CUAD annotations by category
    paragraphs = cuad.get("data", [])
    category_chunks: dict[str, list[str]] = {}

    for para in paragraphs[:100]:  # Cap for speed
        for qa in para.get("qas", []):
            question = qa.get("question", "")
            answers = qa.get("answers", [])
            if not answers:
                continue
            # Extract category from question (CUAD format: "Highlight the parts...")
            for cat_name in category_fields:
                if cat_name.lower().replace(" ", "") in question.lower().replace(" ", ""):
                    texts = [a["text"] for a in answers if a.get("text")]
                    if texts:
                        category_chunks.setdefault(cat_name, []).extend(texts)
                    break

    examples = []
    for cat_name, chunks in list(category_chunks.items())[:max_examples]:
        if not chunks:
            continue
        fields = category_fields.get(cat_name, [])
        if not fields:
            continue

        ex = dspy.Example(
            chunks_json=json.dumps(chunks[:10]),
            cluster_label=cat_name,
            fields_json=json.dumps(fields),
        ).with_inputs("chunks_json", "cluster_label")
        examples.append(ex)

    logger.info(f"Built {len(examples)} training examples from CUAD categories")
    return examples
