"""
RLM V3 — Agentic Discovery with Structured State & Dual-Mode Workers
=====================================================================

HOW V3 WORKS UNDER THE HOOD — THE FULL PICTURE
=================================================

V3 is the hybrid: it combines the RLM agent's autonomous exploration
(from rlm_with_memory.py) with V2's structured state pipeline (from
rlm_v2.py).  The agent DRIVES exploration strategy, but deposits all
findings into typed Pydantic stores via tool calls.

Think of it as: "V2's structured pipeline running inside an RLM's
agentic REPL, with Latent Briefing memory compression."

LIFECYCLE OF A V3 CALL
=======================

1. CALLER invokes:
     v3 = RLMV3(config=RLMV3Config())
     fields, meta = v3.discover(cluster_label, keywords, chunk_texts)

   Or the unified discover+extract variant:
     fields, extractions, meta = v3.discover_and_extract(
         cluster_label, keywords, chunk_texts, chunk_ids, agreement_ids)

2. discover() orchestrates three phases:

   ┌────────────────────────────────────────────────────────────────┐
   │              V3 THREE-PHASE ARCHITECTURE                       │
   │                                                                │
   │  Phase 1: AGENTIC REPL DISCOVERY                              │
   │    DSPy RLM loop with state tools + dual workers              │
   │    Agent writes code → calls tools → deposits findings        │
   │    Budget controls: token meter, wall-clock, field cap         │
   │                                                                │
   │  Phase 2: POST-PROCESSING                                     │
   │    Cross-source overlap dedup (grounded vs exploratory)        │
   │    Two-tier canonicalization (fuzzy + embedding > 0.82)        │
   │    Single shared embedding pass for efficiency                 │
   │                                                                │
   │  Phase 3: OUTPUT CONVERSION                                   │
   │    Convert Pydantic models to extraction-compatible dicts      │
   │    Type normalization (string→text, money→currency)            │
   │    Attach metadata, briefing stats, extraction coverage        │
   └────────────────────────────────────────────────────────────────┘

PHASE 1 DEEP DIVE: THE AGENTIC REPL
======================================

Phase 1 (_phase1_agentic_discovery) builds a custom dspy.RLM subclass
(_V3RLM) at runtime with state-bound tools and dual-mode workers.

Here's what happens step by step:

  A) TOOL CREATION (_make_state_tools)
  ─────────────────────────────────────
  Nine Python functions are created, all bound to the live RollingStateV3
  instance.  The agent calls these in its generated code:

    get_state()            → Returns current state as JSON snapshot:
                             n_fields, n_clause_types, field details,
                             anchor_terms, open_questions.  The agent
                             checks this to decide what to do next.

    deposit_fields(json)   → Parses JSON array of field objects, validates
                             names (snake_case, 40 char max), deduplicates
                             against existing fields, appends to state.
                             Returns: "Added 5 new fields (total: 12)"

    deposit_clauses(json)  → Same pattern for clause types.

    deposit_intents(json)  → Same pattern for intents.

    deposit_obligations(json) → Same for obligations (obligor, action,
                                trigger, deadline).

    deposit_extractions(json) → For unified discover+extract mode.
                                Validates field_name against known fields,
                                resolves chunk_id from index, applies
                                grounding boost (+0.15 confidence if value
                                appears in evidence_span).

    coverage_report()      → Returns convergence + efficiency metrics:
                             fields_discovered, recent deltas, converging
                             (bool), token budget, fields_per_1k_tokens,
                             extraction coverage, recommendation
                             ("CONTINUE" / "STOP" / "EXTRACT ONLY").

    critique_fields()      → Dense rubric self-critique (Concern 2):
                             Scores across 5 dimensions weighted for
                             TASK ALIGNMENT (not judge alignment):
                               type_coverage    (20%)  — diverse value types
                               description_quality (25%) — desc > 10 chars
                               grounding_score  (30%)  — has span_text evidence
                               avg_confidence   (15%)  — self-reported scores
                               source_diversity (10%)  — grounded + exploratory

                             Returns overall grade (A-F), specific_gaps
                             ("Missing temporal fields", "Weak grounding on:
                             fee_amount"), and a tight 3-field micro-state
                             for reinjection:
                               last_failure: the most urgent gap
                               next_check: what to fix next
                               strongest_signal: best field found so far

    budget()               → Resource dashboard:
                             elapsed_s, time_remaining_s, tokens_consumed,
                             token_budget_remaining, extraction_only_mode,
                             warnings (LOW TIME, LOW TOKENS, LOW ITERATIONS).
                             Recommendation: "WRAP UP" / "CONTINUE".

  B) DUAL-MODE WORKERS (_V3RLM._make_llm_tools)
  ───────────────────────────────────────────────
  The agent gets TWO worker functions (unlike base RLM which has one):

    llm_query(prompt)   — GROUNDED mode:
      1. Trajectory is compressed via Latent Briefing (MiniLM cosine
         scoring, MAD thresholding, adaptive tau)
      2. Briefing prepended: "=== CONTEXT FROM PRIOR ANALYSIS ===\n..."
      3. Worker called with temp 0.2 (high precision)
      4. Response captured as trajectory segment for future briefings
      5. state.grounded_calls incremented

    llm_explore(prompt) — EXPLORATORY mode:
      1. NO briefing (worker gets a blank slate)
      2. Worker called with temp 0.8 (high creativity/recall)
      3. Response captured as trajectory segment
      4. state.exploratory_calls incremented

    Why two channels: The agent actively decides WHEN to use each.
    A typical strategy:
      Iteration 1: llm_query("Analyze these chunks: ...") — baseline
      Iteration 2: llm_explore("Find unusual patterns: ...") — niche
      Iteration 3: get_state() + critique_fields() — assess quality
      Iteration 4: llm_query("Find date/duration fields: ...") — fill gaps
      Iteration 5: coverage_report() → "converging" → SUBMIT

  C) PER-ITERATION LIFECYCLE (_V3RLM._execute_iteration)
  ──────────────────────────────────────────────────────────
  After each REPL iteration, the override runs these controls:

    1. STATE DELTA tracking:
         delta = |new_fields - old_fields| / max(new, 1)
         Appended to state.state_deltas for convergence detection.

    2. TOKEN METERING (Concern 4: efficiency early-exit):
         Rough token count: (code_len + output_len) / 4
         Accumulated to state.tokens_consumed.
         If fields_per_1k_tokens < 0.5 for 2 consecutive iterations:
           → state.extraction_only_mode = True
           "We're spending tokens but not finding new fields."

         RECOVERY: if extraction_only_mode is on but a burst of ≥2
         new fields appears in the last window → re-enable discovery.
         "The agent hit a new vein — let it keep exploring."

    3. STATE PRUNING (Concern 1: state budget):
         If len(fields) > state_field_budget (25):
           Sort by confidence descending → keep top state_prune_keep (20)
           Evicted fields are gone — this enforces "constrained carry-over"
           which empirically beats "unconstrained notebooks" (stochi0, §Phase 2).

    4. TOKEN BUDGET HARD STOP:
         If tokens_consumed > token_budget (500k): log warning.
         Next budget() call will recommend "WRAP UP".

    5. WALL-CLOCK TIMEOUT (PR #9295):
         If elapsed > max_time_s (300s): log warning.
         Prevents runaway clusters from blocking the pipeline.

    6. TRAJECTORY CAPTURE:
         Iteration's code + output recorded via briefing engine.
         This feeds future Latent Briefing compressions.

  D) RLM CONSTRUCTION AND EXECUTION
  ──────────────────────────────────
  The _V3RLM is instantiated with:
    - Signature: V3DiscoverySig (discovery only) or V3DiscoverExtractSig
      (unified discover+extract with chunk_id/agreement_id metadata)
    - state_tools: the 9 tools bound to the live RollingStateV3
    - Interpreter: LocalInterpreter (host-process exec(), PR #9295)
      for direct access to numpy/embeddings in agent code
    - max_depth=2: llm_query() can spawn a child RLM with its own REPL
      (PR #9295 feature 6).  Depth > 2 "overthinks" (Concern 4).
    - max_time: wall-clock budget forwarded to DSPy's REPL loop

  Chunks are serialized as JSON records:
    [{"idx": 0, "text": "...", "chunk_id": "abc", "agreement_id": "xyz"}, ...]

  After the REPL completes, the agent's SUBMIT output (fields_json) is
  parsed and any fields not already in state are deposited as source="repl_output".

  E) FALLBACK (_fallback_structured_discovery)
  ─────────────────────────────────────────────
  If the REPL fails (DSPy error, timeout, bad code generation), Phase 1
  degrades gracefully to V2-style batch discovery:
    - Same dual-mode grounded + exploratory passes
    - Same overlap merge, state update, adaptive stopping
    - Just no agent autonomy — fixed batch order
  This ensures V3 never returns empty-handed.

PHASE 2 DEEP DIVE: POST-PROCESSING
======================================

Phase 2 (_phase2_postprocess) deduplicates the accumulated state.

  Single shared embedding pass:
    All field text representations are embedded once (MiniLM).
    The same vectors are reused for both overlap dedup AND canonicalization.

  Step 1 — Cross-source overlap dedup:
    For each exploratory field, compute max cosine sim to all grounded fields.
    If max_sim >= overlap_threshold (0.85): remove the exploratory field.
    "This exploratory find is just a rephrasing of a grounded one."

  Step 2 — Two-tier canonicalization:
    Same as V2: fuzzy name match + embedding sim > 0.82.
    Niche preservation rule: never merge two exploratory items.

  Step 3 — Clause type dedup:
    Simple normalized-name dedup for clause types.

UNIFIED DISCOVER + EXTRACT (discover_and_extract)
===================================================

When config.extract_inline=True, the agent discovers fields AND extracts
values in a single REPL pass.  The Signature changes to V3DiscoverExtractSig,
which instructs the agent to deposit_extractions() alongside deposit_fields().

  Per chunk, the agent:
    1. Calls llm_query("Find fields AND extract values from: ...")
    2. The worker returns field schemas + concrete values
    3. Agent calls deposit_fields() for schemas + deposit_extractions() for values

  Each extraction is an ExtractionV3:
    field_name, value, confidence, evidence_span, chunk_idx,
    chunk_id, agreement_id, source, iteration

  Grounding boost: if value appears verbatim in evidence_span → +0.15 confidence.

  After Phase 2, extractions are filtered to:
    - Only fields that survived canonicalization
    - Only extractions above min_extraction_confidence (0.4)

  Meta includes remaining_chunk_indices: chunks the agent didn't read.
  These are flagged for a lightweight downstream extraction pass.

CONFIGURATION (RLMV3Config)
==============================

  Category          Parameter                   Default   Purpose
  ──────────────────────────────────────────────────────────────────
  REPL limits       max_iterations              25        REPL turns
                    max_llm_calls               50        Worker invocations
  Discovery         max_sample_chunks           30        Chunks to read
                    max_fields                  15        Output cap
  Latent Briefing   briefing_tau                1.5       Compaction aggressiveness
                    briefing_max_tokens         2000      Max briefing length
                    adaptive_tau                True      Dynamic tau adjustment
  Dual-mode         grounded_temp               0.2       Precision channel
                    exploratory_temp            0.8       Recall channel
  Overlap           overlap_threshold           0.85      Cosine dedup cutoff
  Canonicalization  canon_embedding_threshold   0.82      Embedding merge cutoff
  Stopping          adaptive_stop_epsilon       0.05      Convergence threshold
                    adaptive_stop_patience      3         Consecutive stable batches
  State budget      state_field_budget          25        Hard cap on fields
                    state_prune_keep            20        Fields kept after prune
  Efficiency        token_budget                500k      Soft token cap
                    efficiency_window           2         Consecutive low-yield iters
                    min_fields_per_1k_tokens    0.5       Below = extraction-only
  Timeout           max_time_s                  300       Wall-clock seconds
  Recursion         max_sub_depth               2         Child RLM depth
  Interpreter       use_local_interpreter       True      exec() vs sandbox
  Extraction        extract_inline              False     Unified discover+extract
                    min_extraction_confidence   0.4       Minimum to keep

RLM Architecture Concerns — Informed by Empirical Literature
=============================================================

These design decisions are grounded in three empirical studies:

  [1] "Sparse Signal Loop" (stochi0, Apr 2026)
      — 2×2 chat vs RLM × dense vs sparse feedback on LBP + Mini SWE
  [2] "RLMs Are the New Reasoning Models" (Weitekamp, Apr 2026)
      — Survey of RLM results across Oolong, LongMemEval, LongCoT
  [3] "Recursive Language Models as Memory Systems" (Weitekamp, Feb 2026)
      — DSPy.RLM + Pydantic structure on LongMemEval

Concern 1: STATE SHAPE > STATE SIZE  [1, §Phase 2]
  The sparse signal paper's strongest finding: constrained carry-over
  (system reinjection of small tagged blocks) beat unconstrained skill
  files and free-form diaries. A bad notebook is still bad management.

  → Implementation: RollingStateV3 enforces a hard field budget
    (state_field_budget). Lowest-confidence fields are pruned each
    iteration. The state never bloats into a "free-form diary."

  → Implementation: critique_fields() tool returns a tight 3-field
    micro-state (last_failure, next_check, strongest_signal) rather
    than verbose commentary. This is the "system reinjection" pattern.

Concern 2: DENSE FEEDBACK FOR RLM  [1, §Phase 0 Mini SWE]
  RLMs preferred dense rubric feedback while chat preferred sparse.
  The extra rubric structure gives the RLM loop more to grab onto
  because it has more degrees of freedom.

  → Implementation: critique_fields() provides a dense rubric:
    type_coverage (%), description_quality (%), grounding_score (%),
    redundancy_flags, and specific_gaps. This is the RLM-native dense
    feedback that stochi0 found outperformed sparse single-criterion.

Concern 3: JUDGE ≠ TASK SUCCESS  [1, §Phase 2 Mini SWE]
  Skill-file arms hit 29/30 judge-yes but only 17/30 actual solves.
  Persisted procedure can stabilize overfitting to the wrong critic.

  → Implementation: critique_fields() scores against extraction-ready
    quality signals (has span_text grounding, has valid type, desc > 10
    chars) rather than stylistic coherence. The critic is task-aligned.

Concern 4: EFFICIENCY AND EARLY EXIT  [1, §Efficiency; 2, §Challenges]
  RLMs front-load exploration and often burn tokens on diminishing
  returns in later iterations. Depth-1 recursion helps; deeper can
  "overthink." Cost/time unpredictable.

  → Implementation: Per-iteration token metering via _tokens_consumed
    counter. When new_fields_per_1k_tokens drops below a threshold
    for 2 consecutive iterations, the loop auto-downgrades to
    extraction-only mode (no new discovery, just extract from remaining
    chunks). This prevents the "token bonfire" failure mode.

Concern 5: STRUCTURED OUTPUT IS THE REAL WIN  [3, §Results]
  DSPy.RLM + Pydantic-typed outputs boosted Gemini Flash from 58%
  to 87.2% on LongMemEval. Structure keeps models on rails.

  → Already implemented: All deposit_*() tools enforce Pydantic type
    validation. FieldV3/ClauseTypeV3/etc. are typed models. This is
    our strongest aligned feature with the literature.

Concern 6: DELEGATION > DIRECT PROCESSING  [3, §Results]
  Telling the model to delegate sub-queries works better than having
  it read everything itself.

  → Already implemented: llm_query() and llm_explore() are delegation
    channels — the orchestrator never processes chunks directly, it
    delegates to worker models via these tools.

Concern 7: PROMPT OPTIMIZATION CAN OVERFIT  [3, §GEPA Results]
  Weitekamp spent ~$400 on GEPA optimization and it regressed from
  89.8% to 87.8%. Structure helps more than optimized instructions.

  → Guard: compile_with_mipro() in rlm_with_memory.py requires a
    minimum trainset of 20 examples. For smaller datasets, the
    structured tools + Pydantic types provide more reliable gains.

COMPARISON OF ALL THREE RLM VERSIONS
========================================

  Feature                rlm_with_memory     rlm_v2          rlm_v3
  ──────────────────────────────────────────────────────────────────
  Agent autonomy         YES (REPL)          NO (batch)      YES (REPL)
  Structured state       V2-style addon      Native          Native
  Dual-mode workers      V2-style addon      Native          Native
  Trajectory memory      LB compaction       None            LB compaction
  State budget/pruning   No                  No              Yes (25 cap)
  Token metering         No                  No              Yes
  Extraction inline      No                  No              Yes
  Fallback on failure    None                N/A (no REPL)   V2-style batch
  MIPRO compilation      Yes                 No              No (planned)
  Self-critique          No                  No              Yes (rubric)
  Wall-clock timeout     No                  No              Yes (300s)
  Child RLM recursion    No                  No              Yes (depth=2)

Usage:
    from core.rlm_v3 import RLMV3, RLMV3Config
    v3 = RLMV3(config=RLMV3Config())
    fields, meta = v3.discover(cluster_label, keywords, chunk_texts)

    # Unified discover + extract:
    fields, extractions, meta = v3.discover_and_extract(
        cluster_label, keywords, chunk_texts, chunk_ids, agreement_ids)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

import dspy
import numpy as np
from dspy import Prediction
from pydantic import BaseModel

from core.latent_briefing import TextLatentBriefing, CompactionConfig, CompactionStats

logger = logging.getLogger(__name__)

# ── Module-level embedder singleton (avoids ~2-3s reload per RLMV3 instance) ─
_EMBEDDER_INSTANCE = None
_EMBEDDER_LOCK = threading.Lock()


def _get_shared_embedder():
    """Return a singleton SentenceTransformer. Thread-safe, loaded once."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER_INSTANCE is None:
                from sentence_transformers import SentenceTransformer
                _EMBEDDER_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded shared embedder: all-MiniLM-L6-v2")
    return _EMBEDDER_INSTANCE


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models (shared with V2)
# ══════════════════════════════════════════════════════════════════════════════


class FieldV3(BaseModel):
    name: str
    value_type: str = "string"
    description: str = ""
    span_text: str = ""
    span_texts: list[str] = []       # multiple evidence spans for derived/composite fields
    derivation: str = ""             # how the field is computed (e.g. "price - discount")
    confidence: float = 0.5
    source: str = "grounded"        # grounded | exploratory | agent
    iteration: int = 0


class ClauseTypeV3(BaseModel):
    name: str
    emergent_description: str = ""
    span_text: str = ""
    span_texts: list[str] = []       # multiple evidence spans for composite clauses
    confidence: float = 0.5
    source: str = "grounded"
    iteration: int = 0


class IntentV3(BaseModel):
    clause_ref: str = ""
    clause_refs: list[str] = []       # multiple clause references for composite intents
    intent_description: str = ""
    label: str = ""                   # snake_case intent label
    summary: str = ""                 # one-sentence description
    party_from: str = "none"          # obligor / right-holder
    party_to: str = "none"            # beneficiary
    attributes: dict = {}             # intent-specific KV pairs
    source: str = "grounded"
    chunk_idx: int = -1               # index into chunk_texts for provenance
    chunk_ids: list[int] = []         # multiple chunk indices for cross-chunk intents
    chunk_id: str = ""
    agreement_id: str = ""


class ObligationV3(BaseModel):
    clause_ref: str = ""
    clause_refs: list[str] = []       # multiple clause refs for cross-section obligations
    obligor: str = ""
    action: str = ""
    trigger_condition: Optional[str] = None
    deadline: Optional[str] = None
    confidence: float = 0.5
    source: str = "grounded"


class ExtractionV3(BaseModel):
    """A single field extraction produced inline during discovery."""
    field_name: str
    value: str
    confidence: float = 0.5
    evidence_span: str = ""          # exact text span that grounds the value
    chunk_idx: int = -1               # index into the chunk_texts array
    chunk_id: str = ""                # opaque ID for DB persistence
    agreement_id: str = ""
    source: str = "agent"             # agent | grounded | exploratory
    iteration: int = 0


class RollingStateV3(BaseModel):
    """Typed, cluster-scoped accumulator that persists across REPL iterations."""
    cluster_label: str = ""
    chunks_seen: int = 0
    total_chunks: int = 0
    fields: list[FieldV3] = []
    clause_types: list[ClauseTypeV3] = []
    intents: list[IntentV3] = []
    obligations: list[ObligationV3] = []
    extractions: list[ExtractionV3] = []
    open_questions: list[str] = []
    anchor_terms: list[tuple[str, int]] = []
    iteration: int = 0
    state_deltas: list[float] = []
    grounded_calls: int = 0
    exploratory_calls: int = 0
    # Token efficiency tracking (Concern 4: prevent token bonfire)
    tokens_consumed: int = 0
    fields_per_iteration: list[int] = []       # n_new_fields per iteration
    efficiency_warnings: int = 0               # consecutive low-yield iterations
    extraction_only_mode: bool = False          # downgraded after efficiency drop
    # Pruning log (Concern 1: state shape > state size)
    fields_pruned_total: int = 0
    # Wall-clock tracking (PR #9295: budget awareness)
    start_time: float = 0.0                   # set at discovery start
    # Chunk metadata (populated when extract_inline=True)
    _chunk_ids: list[str] = []
    _agreement_ids: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


class RLMV3Config(BaseModel):
    """All V3 tunables."""
    # REPL limits
    max_iterations: int = 15
    max_llm_calls: int = 30

    # Discovery scope
    max_sample_chunks: int = 60
    max_fields: int = 25

    # Latent Briefing
    briefing_tau: float = 1.5
    briefing_max_tokens: int = 2000
    briefing_min_retention: float = 0.1
    briefing_max_retention: float = 0.9
    adaptive_tau: bool = True

    # Dual-mode
    grounded_temp: float = 0.2
    exploratory_temp: float = 0.8

    # Overlap merge
    overlap_threshold: float = 0.85

    # Canonicalization
    canon_embedding_threshold: float = 0.82

    # Adaptive stopping
    adaptive_stop_epsilon: float = 0.05
    adaptive_stop_patience: int = 3

    # LLM
    max_retries: int = 3
    retry_delay: float = 1.0

    # State budget (Concern 1: constrained carry-over > unconstrained diaries)
    state_field_budget: int = 30              # hard cap; lowest-confidence evicted
    state_prune_keep: int = 25                # after pruning, keep top-K by confidence

    # Token efficiency (Concern 4: prevent overthinking / token bonfire)
    token_budget: int = 300_000               # soft cap on total tokens consumed
    efficiency_window: int = 2                # consecutive low-yield iters before downgrade
    min_fields_per_1k_tokens: float = 0.5     # below this → extraction-only mode

    # Wall-clock timeout (PR #9295: budget awareness — prevents runaway clusters)
    max_time_s: int = 300                     # wall-clock seconds; 0 = unlimited

    # Recursive subcalls (PR #9295: depth > 1 for llm_query workers)
    max_sub_depth: int = 2                    # 1 = flat LM call (original), 2 = child gets own REPL

    # LocalInterpreter (PR #9295: host-process access for numpy/embeddings)
    use_local_interpreter: bool = True        # True = exec()-based, False = Deno/Pyodide sandbox

    # Unified discover + extract
    extract_inline: bool = False      # when True, agent extracts values as it discovers
    min_extraction_confidence: float = 0.4

    # Domain-level seed fields (two-tier architecture):
    # When provided, these universal fields are pre-loaded into the REPL state
    # so the agent focuses on discovering cluster-specific delta fields rather
    # than re-discovering effective_date/governing_law/party_names in every cluster.
    seed_fields: list[dict] | None = None     # list of {name, type, description, examples}


# ══════════════════════════════════════════════════════════════════════════════
# DSPy Signature for V3 Discovery
# ══════════════════════════════════════════════════════════════════════════════


class V3DiscoverySig(dspy.Signature):
    """Discover structured fields, clause types, intents, and obligations from
    legal clause excerpts using iterative analysis.

    You have access to special tools:
    - llm_query(prompt): For precise, grounded analysis with prior context
    - llm_explore(prompt): For creative, niche pattern discovery
    - get_state(): See what you've found so far
    - deposit_fields(json_str): Save discovered fields
    - deposit_clauses(json_str): Save discovered clause types
    - deposit_intents(json_str): Save discovered intents
    - deposit_obligations(json_str): Save discovered obligations
    - coverage_report(): Check convergence and gaps
    - critique_fields(): Dense quality rubric — call after depositing fields
    - budget(): Check remaining time/token/iteration budget

    Strategy:
    1. Start with llm_query to get grounded baseline fields
    2. Deposit fields first, then clauses, intents, obligations
    3. Call critique_fields() to identify quality gaps
    4. Use llm_explore to find niche/unusual patterns
    5. Check coverage_report() to see if you're converging
    6. Check budget() to decide whether to continue or wrap up
    7. Stop when coverage_report shows convergence or budget says WRAP UP

    Precision rules for ALL deposit_* calls:
    - Names and labels must reflect the ACTUAL legal effect. Distinguish rights
      from restrictions from obligations — do not mislabel one as another.
    - For multi-provision clauses, use names/labels that cover the FULL scope,
      not just one sub-provision.
    - Descriptions must be consistent with names. If the description says
      "right", the name must not say "restriction", and vice versa.
    - Field names should be specific to what is extracted (e.g.
      "termination_notice_period" not "termination_info").
    - Clause type names should capture the legal function, not just the topic.
    - Obligation actions should be concrete and verifiable.

    GROUNDING RULE: Only deposit fields you can POINT TO in the text. Every
    field must have span_text — an exact quote from a chunk that proves the
    field exists. If you cannot quote text evidence, do NOT deposit the field.
    Fields inferred from domain knowledge but absent from the actual text are
    hallucinations and will be rejected.

    DERIVED FIELDS: For fields whose value is computed from 2+ text locations
    (e.g. net_value = total - discount), provide span_texts (a list of quotes)
    and derivation (the formula). These are grounded differently — they need
    evidence from each component, not a single span.

    Return fields_json as the final consolidated JSON array of discovered fields."""

    chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
    cluster_label: str = dspy.InputField(desc="The clause type label for this cluster")
    keywords: str = dspy.InputField(desc="Comma-separated keywords for this cluster")
    max_fields: str = dspy.InputField(desc="Maximum number of fields to discover")
    fields_json: str = dspy.OutputField(desc="Final JSON array of {name, value_type, description, span_text, confidence} field objects")


class V3DiscoverExtractSig(dspy.Signature):
    """Discover structured fields AND extract their values from legal clause
    excerpts in a single unified pass.

    You have access to special tools:
    - llm_query(prompt): For precise, grounded analysis with prior context
    - llm_explore(prompt): For creative, niche pattern discovery
    - get_state(): See what you've found so far (fields + extractions)
    - deposit_fields(json_str): Save discovered field definitions
    - deposit_extractions(json_str): Save extracted field values from chunks
    - deposit_clauses(json_str): Save discovered clause types
    - deposit_intents(json_str): Save discovered intents
    - deposit_obligations(json_str): Save discovered obligations
    - coverage_report(): Check convergence and extraction coverage
    - critique_fields(): Dense quality rubric — call after depositing fields
    - budget(): Check remaining time/token/iteration budget

    Strategy:
    1. For each chunk batch, use llm_query to discover fields AND extract values
       in the same call. Ask the worker to return both the field schema and any
       concrete values it sees in the text.
    2. Deposit field definitions via deposit_fields() and concrete values via
       deposit_extractions().
    3. Also deposit clauses, intents, and obligations as you discover them.
    4. Call critique_fields() to identify quality gaps.
    5. Use llm_explore for niche patterns, then extract values for those too.
    6. Check coverage_report() — it shows both field convergence AND extraction
       counts. Any chunks not yet extracted will be handled by a downstream pass.
    7. Check budget() to decide whether to continue or wrap up.
    8. Stop when fields converge or budget says WRAP UP.

    Precision rules for ALL deposit_* calls:
    - Names and labels must reflect the ACTUAL legal effect. Distinguish rights
      from restrictions from obligations — do not mislabel one as another.
    - For multi-provision clauses, use names/labels that cover the FULL scope,
      not just one sub-provision.
    - Descriptions must be consistent with names. If the description says
      "right", the name must not say "restriction", and vice versa.
    - Field names should be specific to what is extracted (e.g.
      "termination_notice_period" not "termination_info").
    - Clause type names should capture the legal function, not just the topic.
    - Obligation actions should be concrete and verifiable.

    GROUNDING RULE: Only deposit fields you can POINT TO in the text. Every
    field must have span_text — an exact quote from a chunk that proves the
    field exists. If you cannot quote text evidence, do NOT deposit the field.
    Fields inferred from domain knowledge but absent from the actual text are
    hallucinations and will be rejected.

    DERIVED FIELDS: For fields whose value is computed from 2+ text locations
    (e.g. net_value = total - discount), provide span_texts (a list of quotes)
    and derivation (the formula). These are grounded differently — they need
    evidence from each component, not a single span.

    Return fields_json as the final consolidated JSON array of discovered fields."""

    chunks_json: str = dspy.InputField(desc="JSON array of {idx, text, chunk_id, agreement_id} objects")
    cluster_label: str = dspy.InputField(desc="The clause type label for this cluster")
    keywords: str = dspy.InputField(desc="Comma-separated keywords for this cluster")
    max_fields: str = dspy.InputField(desc="Maximum number of fields to discover")
    fields_json: str = dspy.OutputField(desc="Final JSON array of {name, value_type, description, span_text, confidence} field objects")


# ══════════════════════════════════════════════════════════════════════════════
# REPL Tool Functions (created per-invocation, bound to state)
# ══════════════════════════════════════════════════════════════════════════════


def _make_state_tools(
    state: RollingStateV3,
    briefing_engine: TextLatentBriefing,
    config: RLMV3Config,
    meta: dict,
    embedder_fn: Callable,
) -> list[Callable]:
    """Create REPL-accessible tool functions bound to the live RollingState.

    These are real Python functions the DSPy REPL agent can call in its
    generated code. They mutate `state` in-place and track statistics in `meta`.
    """
    _lock = threading.Lock()

    def get_state() -> str:
        """Get the current discovery state as JSON — what has been found so far."""
        with _lock:
            summary = {
                "cluster_label": state.cluster_label,
                "chunks_seen": state.chunks_seen,
                "total_chunks": state.total_chunks,
                "n_fields": len(state.fields),
                "n_clause_types": len(state.clause_types),
                "n_intents": len(state.intents),
                "n_obligations": len(state.obligations),
                "fields": [
                    {"name": f.name, "type": f.value_type, "desc": f.description[:80],
                     "source": f.source}
                    for f in state.fields
                ],
                "clause_types": [
                    {"name": c.name, "desc": c.emergent_description[:80]}
                    for c in state.clause_types
                ],
                "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
                "open_questions": state.open_questions[:5],
            }
        return json.dumps(summary, indent=2)

    def deposit_fields(json_str: str) -> str:
        """Deposit discovered fields into the rolling state.
        Input: JSON string of array of field objects with keys:
        name, value_type, description, span_text, confidence.

        QUALITY RULES:
        - Field names must be specific to what is extracted, not vague topic
          labels. e.g. "termination_notice_period" not "termination_info".
        - The description must accurately match the name — if the name says
          "cap" the description must be about a limit, not a general amount.
        - value_type must match the actual data ("date" for dates, "currency"
          for monetary values, "string" for text, "number" for numeric values).
        - Avoid overly broad fields that conflate multiple distinct values into
          one. Split them into separate fields instead.

        GROUNDING REQUIREMENT: span_text is MANDATORY for direct fields. Fields
        without span_text (an exact quote from the chunk text) will be penalized.
        If you cannot find a verbatim span in the text, the field is likely
        hallucinated — do NOT deposit it.

        DERIVED/COMPOSITE FIELDS: Some fields are computed from multiple text
        locations (e.g. net_value = price - discount). For these, provide:
          - span_texts: a JSON array of 2+ exact quotes, one per component
          - derivation: brief formula (e.g. "total_price - discount_amount")
        Derived fields with 2+ span_texts are NOT penalized for missing span_text.

        GOOD field examples (grounded — has text evidence, from diverse contract types):
          {"name": "interest_rate_type", "value_type": "enum",
           "description": "Whether the loan bears fixed or variable interest",
           "span_text": "shall bear interest at a fixed rate of 4.5% per annum",
           "confidence": 0.9}
          {"name": "license_territory", "value_type": "string",
           "description": "Geographic scope of the IP license grant",
           "span_text": "an exclusive license in the United States and Canada",
           "confidence": 0.85}

        GOOD derived field example (composite — grounded from multiple spans):
          {"name": "net_purchase_price", "value_type": "currency",
           "description": "Purchase price after adjustments and credits",
           "span_texts": ["aggregate purchase price of $12,000,000",
                          "less the Adjustment Amount of $500,000"],
           "derivation": "purchase_price - adjustment_amount",
           "confidence": 0.85}

        BAD field examples (hallucinated — DO NOT deposit these):
          {"name": "force_majeure_events", "value_type": "string",
           "description": "Types of force majeure events covered",
           "span_text": "", "confidence": 0.6}
           ^ No span_text! Inferred from domain knowledge, not found in text.
          {"name": "general_obligations", "value_type": "string",
           "description": "General obligations of the parties",
           "span_text": "the parties agree to", "confidence": 0.5}
           ^ Too vague — what specific obligation? What specific value?
          {"name": "governing_law", "value_type": "string",
           "description": "Applicable governing law",
           "span_text": "", "confidence": 0.7}
           ^ Common field but NOT in this cluster's text. Don't assume.
          {"name": "total_value", "value_type": "currency",
           "description": "Total contract value",
           "span_text": "", "derivation": "price + fees", "confidence": 0.7}
           ^ Claims derivation but provides NO span_texts — still hallucinated.

        Returns count of new fields added (and count rejected for no grounding)."""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, dict):
                data = data.get("fields", [data])
            if not isinstance(data, list):
                return "Error: expected a JSON array of field objects"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        added = 0
        ungrounded = 0
        with _lock:
            existing = {f.name for f in state.fields}
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                name = (raw.get("name") or "").strip()
                if not name:
                    continue
                name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
                if name in existing:
                    continue
                span = raw.get("span_text", "").strip()
                span_texts = raw.get("span_texts", [])
                if isinstance(span_texts, str):
                    try:
                        span_texts = json.loads(span_texts)
                    except (json.JSONDecodeError, TypeError):
                        span_texts = [span_texts] if span_texts.strip() else []
                if not isinstance(span_texts, list):
                    span_texts = []
                span_texts = [s.strip() for s in span_texts if isinstance(s, str) and len(s.strip()) >= 5]
                derivation = raw.get("derivation", "").strip()
                conf = float(raw.get("confidence", 0.5))
                source = raw.get("source", "agent")
                # Grounding check — derived fields with 2+ span_texts are grounded
                has_single_span = span and len(span) >= 5
                has_multi_span = len(span_texts) >= 2
                if not has_single_span and not has_multi_span:
                    if source == "exploratory":
                        conf = min(conf, 0.45)  # lighter penalty — creative finds may lack exact spans
                    else:
                        conf = min(conf, 0.3)
                    ungrounded += 1
                state.fields.append(FieldV3(
                    name=name,
                    value_type=raw.get("value_type", raw.get("type", "string")),
                    description=raw.get("description", ""),
                    span_text=span,
                    span_texts=span_texts,
                    derivation=derivation,
                    confidence=conf,
                    source=source,
                    iteration=state.iteration,
                ))
                existing.add(name)
                added += 1
        parts = [f"Added {added} new fields (total: {len(state.fields)})"]
        grounded_count = added - ungrounded
        if grounded_count > 0 and ungrounded == 0:
            parts.append(f"All {added} fields grounded with span_text — high quality deposit.")
        elif grounded_count > 0 and ungrounded > 0:
            parts.append(f"{grounded_count} well-grounded, {ungrounded} missing span_text.")
        if ungrounded:
            parts.append(
                f"WARNING: {ungrounded} fields had no span_text evidence — "
                f"confidence capped. These will likely be pruned. "
                f"Always include an exact text quote in span_text."
            )
        return ". ".join(parts)

    def deposit_clauses(json_str: str) -> str:
        """Deposit discovered clause types into the rolling state.
        Input: JSON string of array with keys: name, emergent_description, span_text.

        QUALITY RULES:
        - Clause type names must capture the legal function, not just the topic.
          e.g. "mutual_termination_right" not just "termination".
        - The emergent_description must be consistent with the name. If the
          clause grants a right, do not name it a restriction.
        - For clauses covering multiple provisions, use a name that reflects
          the full scope rather than just one sub-provision.

        GROUNDING REQUIREMENT: span_text is MANDATORY for single-source clauses.
        Clause types without span_text will be flagged as ungrounded.
        Only deposit clause types you can point to in the actual text.

        COMPOSITE CLAUSES: Some clause types combine provisions from multiple
        text locations (e.g. indemnity + liability cap). For these, provide:
          - span_texts: a JSON array of 2+ exact quotes, one per provision
        Composite clauses with 2+ span_texts are NOT flagged as ungrounded.

        GOOD clause examples (grounded, from diverse contract types):
          {"name": "ip_ownership_work_for_hire",
           "emergent_description": "All deliverables constitute work-for-hire owned exclusively by Client",
           "span_text": "all Work Product shall be considered work made for hire and shall be the sole property of Client"}
          {"name": "cross_default_acceleration",
           "emergent_description": "Default under any related agreement triggers acceleration of this loan",
           "span_text": "a default under the Credit Agreement shall constitute an Event of Default hereunder"}

        GOOD composite clause example (multi-provision):
          {"name": "indemnity_with_liability_cap",
           "emergent_description": "Vendor indemnifies Client for IP claims, capped at total fees paid",
           "span_texts": ["Vendor shall indemnify and hold harmless Client from any third-party IP infringement claims",
                          "aggregate liability shall not exceed the total fees paid under this Agreement"]}

        BAD clause examples (DO NOT deposit):
          {"name": "force_majeure",
           "emergent_description": "Force majeure provisions",
           "span_text": ""}
           ^ No span_text! Assumed from domain knowledge, not found in text.
          {"name": "miscellaneous",
           "emergent_description": "Various general provisions",
           "span_text": "the parties agree"}
           ^ Too vague — what specific legal function?

        Returns count of new clause types added (and count flagged as ungrounded)."""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, dict):
                data = data.get("clause_types", [data])
            if not isinstance(data, list):
                return "Error: expected a JSON array"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        added = 0
        ungrounded = 0
        with _lock:
            existing = {c.name.lower() for c in state.clause_types}
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                name = (raw.get("name") or "").strip()
                if not name or name.lower() in existing:
                    continue
                span = raw.get("span_text", "").strip()
                span_texts = raw.get("span_texts", [])
                if isinstance(span_texts, str):
                    try:
                        span_texts = json.loads(span_texts)
                    except (json.JSONDecodeError, TypeError):
                        span_texts = [span_texts] if span_texts.strip() else []
                if not isinstance(span_texts, list):
                    span_texts = []
                span_texts = [s.strip() for s in span_texts if isinstance(s, str) and len(s.strip()) >= 5]
                has_single = span and len(span) >= 5
                has_multi = len(span_texts) >= 2
                if not has_single and not has_multi:
                    ungrounded += 1
                state.clause_types.append(ClauseTypeV3(
                    name=name,
                    emergent_description=raw.get("emergent_description", ""),
                    span_text=span,
                    span_texts=span_texts,
                    source=raw.get("source", "agent"),
                    iteration=state.iteration,
                ))
                existing.add(name.lower())
                added += 1
        parts = [f"Added {added} new clause types (total: {len(state.clause_types)})"]
        grounded_count = added - ungrounded
        if grounded_count > 0 and ungrounded == 0:
            parts.append(f"All {added} clause types grounded with span_text — high quality.")
        if ungrounded:
            parts.append(
                f"WARNING: {ungrounded} clause types had no span_text evidence. "
                f"These are likely hallucinated. Always include an exact text quote."
            )
        return ". ".join(parts)

    def deposit_intents(json_str: str) -> str:
        """Deposit discovered intents into the rolling state.
        Input: JSON string of array with keys: label, summary, party_from,
        party_to, attributes, clause_ref, chunk_idx.

        LABEL RULES (critical — labels must accurately characterize the intent):
        - The label must reflect the ACTUAL legal effect, not just the topic.
          e.g. a clause that PERMITS early termination → "early_termination_right",
          NOT "early_termination_restriction".
        - Distinguish allowances from restrictions: if the clause grants a right,
          the label must say so (e.g. "assignment_right"); if it prohibits, say so
          (e.g. "assignment_prohibition").
        - For multi-provision clauses, use a label that covers the FULL scope —
          do NOT reduce to just one sub-provision. e.g. a clause covering both
          indemnity cap and liability exclusion → "liability_and_indemnity_limits",
          NOT just "indemnity_cap".
        - Keep labels specific to the clause content. Avoid vague labels like
          "general_provision" or "miscellaneous".
        - The summary must be consistent with the label. If the summary describes
          a right, the label must not say restriction, and vice versa.

        GROUNDING REQUIREMENT: clause_ref OR chunk_idx is MANDATORY. Intents must
        be traceable to a specific clause or chunk in the text. If you cannot point
        to where in the text this intent comes from, do NOT deposit it.

        COMPOSITE INTENTS: Some intents span multiple clauses or chunks (e.g. a
        conditional right defined in Section 5.1 with conditions in Section 5.2).
        For these, provide:
          - clause_refs: a JSON array of section references
          - chunk_ids: a JSON array of chunk indices
        Composite intents with clause_refs or chunk_ids arrays are grounded.

        GOOD intent examples (grounded, from diverse contract types):
          {"label": "non_compete_geographic_scope",
           "summary": "Employee may not engage in competing business within 50 miles of any company office for 2 years post-termination",
           "party_from": "Employee", "party_to": "Employer",
           "clause_ref": "Section 7.1", "chunk_idx": 3}
          {"label": "royalty_audit_right",
           "summary": "Licensor may audit Licensee's sales records once per year to verify royalty calculations",
           "party_from": "Licensor", "party_to": "Licensee",
           "clause_ref": "Section 4.5", "chunk_idx": 8}

        GOOD composite intent example (multi-section):
          {"label": "conditional_assignment_with_consent",
           "summary": "Licensee may assign the agreement only with prior written consent from Licensor and payment of a $10,000 transfer fee",
           "party_from": "Licensee", "party_to": "Licensor",
           "clause_refs": ["Section 12.1", "Section 12.3"],
           "chunk_ids": [5, 7]}

        BAD intent examples (DO NOT deposit):
          {"label": "general_compliance",
           "summary": "Parties must comply with applicable laws",
           "clause_ref": "", "chunk_idx": -1}
           ^ No provenance! Too vague — what specific compliance obligation?
          {"label": "termination_restriction",
           "summary": "Landlord grants tenant the right to terminate early",
           "clause_ref": "Section 12", "chunk_idx": 4}
           ^ Label says restriction but summary says right — contradictory!

        Returns count added (and count flagged as ungrounded)."""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, dict):
                data = data.get("intents", [data])
            if not isinstance(data, list):
                return "Error: expected a JSON array"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        added = 0
        ungrounded = 0
        with _lock:
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                label = raw.get("label", "").strip()
                desc = raw.get("summary", raw.get("intent_description", "")).strip()
                if not label and not desc:
                    continue
                if not label:
                    label = re.sub(r'[^a-z0-9_]', '_', desc[:50].lower())
                attrs = raw.get("attributes", {})
                if not isinstance(attrs, dict):
                    attrs = {}
                # Resolve chunk provenance if available
                chunk_idx = int(raw.get("chunk_idx", -1))
                chunk_ids_list = raw.get("chunk_ids", [])
                if isinstance(chunk_ids_list, str):
                    try:
                        chunk_ids_list = json.loads(chunk_ids_list)
                    except (json.JSONDecodeError, TypeError):
                        chunk_ids_list = []
                if not isinstance(chunk_ids_list, list):
                    chunk_ids_list = []
                chunk_ids_list = [int(c) for c in chunk_ids_list if isinstance(c, (int, float))]
                chunk_id = ""
                agreement_id = ""
                if chunk_idx >= 0:
                    if state._chunk_ids and chunk_idx < len(state._chunk_ids):
                        chunk_id = state._chunk_ids[chunk_idx]
                    if state._agreement_ids and chunk_idx < len(state._agreement_ids):
                        agreement_id = state._agreement_ids[chunk_idx]
                clause_ref = raw.get("clause_ref", "").strip()
                clause_refs_list = raw.get("clause_refs", [])
                if isinstance(clause_refs_list, str):
                    try:
                        clause_refs_list = json.loads(clause_refs_list)
                    except (json.JSONDecodeError, TypeError):
                        clause_refs_list = [clause_refs_list] if clause_refs_list.strip() else []
                if not isinstance(clause_refs_list, list):
                    clause_refs_list = []
                clause_refs_list = [str(r).strip() for r in clause_refs_list if str(r).strip()]
                # Grounding: need at least one of: clause_ref, chunk_idx, clause_refs, chunk_ids
                has_provenance = (
                    chunk_idx >= 0
                    or clause_ref
                    or len(clause_refs_list) >= 1
                    or len(chunk_ids_list) >= 1
                )
                if not has_provenance:
                    ungrounded += 1
                state.intents.append(IntentV3(
                    clause_ref=clause_ref,
                    clause_refs=clause_refs_list,
                    intent_description=desc,
                    label=label,
                    summary=desc,
                    party_from=raw.get("party_from", "none"),
                    party_to=raw.get("party_to", "none"),
                    attributes=attrs,
                    source=raw.get("source", "agent"),
                    chunk_idx=chunk_idx,
                    chunk_ids=chunk_ids_list,
                    chunk_id=chunk_id,
                    agreement_id=agreement_id,
                ))
                added += 1
        parts = [f"Added {added} intents (total: {len(state.intents)})"]
        grounded_count = added - ungrounded
        if grounded_count > 0 and ungrounded == 0:
            parts.append(f"All {added} intents have provenance — high quality.")
        if ungrounded:
            parts.append(
                f"WARNING: {ungrounded} intents had no clause_ref or chunk_idx — "
                f"these are ungrounded and likely hallucinated. "
                f"Always specify which clause/chunk the intent comes from."
            )
        return ". ".join(parts)

    def deposit_obligations(json_str: str) -> str:
        """Deposit discovered obligations into the rolling state.
        Input: JSON string of array with keys: clause_ref, obligor, action,
        trigger_condition, deadline.

        QUALITY RULES:
        - The action must be concrete and verifiable — what exactly must the
          obligor do? e.g. "deliver written notice within 30 days" not
          "comply with terms".
        - obligor must identify WHO is bound (e.g. "Tenant", "Borrower",
          "Seller"), not vague references like "party" or "they".
        - trigger_condition should state the specific event or circumstance,
          not restate the action.
        - deadline must be the actual time constraint from the text, or null
          if none is specified. Do not invent deadlines.

        GROUNDING REQUIREMENT: clause_ref is MANDATORY. Obligations must be
        traceable to a specific clause. If you cannot point to where in the
        text this obligation comes from, do NOT deposit it.

        COMPOSITE OBLIGATIONS: Some obligations combine terms from multiple
        clauses (e.g. delivery action in Section 4, quality standard in Section 7).
        For these, provide:
          - clause_refs: a JSON array of section references
        Composite obligations with clause_refs are grounded.

        GOOD obligation examples (grounded, from diverse contract types):
          {"clause_ref": "Section 6.3", "obligor": "Licensee",
           "action": "pay quarterly royalties of 5% of net sales within 30 days after each calendar quarter",
           "trigger_condition": "end of each calendar quarter",
           "deadline": "30 days after quarter end"}
          {"clause_ref": "Section 11.2", "obligor": "Supplier",
           "action": "replace or refund any non-conforming goods within 15 business days of written notice",
           "trigger_condition": "receipt of written notice of non-conformance",
           "deadline": "15 business days"}

        GOOD composite obligation example (multi-clause):
          {"clause_refs": ["Section 4.1", "Section 7.2"], "obligor": "Contractor",
           "action": "deliver work product meeting acceptance criteria within 30 days",
           "trigger_condition": "milestone completion per Section 4.1",
           "deadline": "30 days per Section 7.2"}

        BAD obligation examples (DO NOT deposit):
          {"clause_ref": "", "obligor": "party",
           "action": "comply with all terms",
           "trigger_condition": null, "deadline": null}
           ^ No clause_ref, vague obligor, vague action — entirely hallucinated.
          {"clause_ref": "Section 3", "obligor": "Seller",
           "action": "act in good faith",
           "trigger_condition": null, "deadline": "30 days"}
           ^ Action not concrete/verifiable. Deadline invented (not in text).

        Returns count added (and count flagged as ungrounded)."""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, dict):
                data = data.get("obligations", [data])
            if not isinstance(data, list):
                return "Error: expected a JSON array"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        added = 0
        ungrounded = 0
        vague_obligor = 0
        with _lock:
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                action = raw.get("action", "").strip()
                if not action:
                    continue
                clause_ref = raw.get("clause_ref", "").strip()
                clause_refs_list = raw.get("clause_refs", [])
                if isinstance(clause_refs_list, str):
                    try:
                        clause_refs_list = json.loads(clause_refs_list)
                    except (json.JSONDecodeError, TypeError):
                        clause_refs_list = [clause_refs_list] if clause_refs_list.strip() else []
                if not isinstance(clause_refs_list, list):
                    clause_refs_list = []
                clause_refs_list = [str(r).strip() for r in clause_refs_list if str(r).strip()]
                obligor = raw.get("obligor", "").strip()
                if not clause_ref and len(clause_refs_list) < 1:
                    ungrounded += 1
                if obligor.lower() in ("", "party", "parties", "they", "it"):
                    vague_obligor += 1
                state.obligations.append(ObligationV3(
                    clause_ref=clause_ref,
                    clause_refs=clause_refs_list,
                    obligor=obligor,
                    action=action,
                    trigger_condition=raw.get("trigger_condition"),
                    deadline=raw.get("deadline"),
                    source=raw.get("source", "agent"),
                ))
                added += 1
        parts = [f"Added {added} obligations (total: {len(state.obligations)})"]
        clean = added - ungrounded - vague_obligor
        if clean > 0 and ungrounded == 0 and vague_obligor == 0:
            parts.append(f"All {added} obligations well-specified — high quality.")
        if ungrounded:
            parts.append(
                f"WARNING: {ungrounded} obligations had no clause_ref — "
                f"these are ungrounded. Always specify the source clause."
            )
        if vague_obligor:
            parts.append(
                f"WARNING: {vague_obligor} obligations had vague obligor "
                f"(e.g. 'party'). Use specific names like 'Tenant', 'Borrower'."
            )
        return ". ".join(parts)

    def deposit_extractions(json_str: str) -> str:
        """Deposit extracted field values from specific chunks.
        Input: JSON string of array of extraction objects with keys:
        field_name (must match a deposited field), value, chunk_idx (index into
        chunks_json array), evidence_span (exact quote from text), confidence (0-1).
        Returns count of extractions added."""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, dict):
                data = data.get("extractions", [data])
            if not isinstance(data, list):
                return "Error: expected a JSON array of extraction objects"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        added = 0
        skipped_no_field = 0
        skipped_empty = 0
        with _lock:
            known_fields = {f.name for f in state.fields}
            for raw in data:
                if not isinstance(raw, dict):
                    continue
                field_name = re.sub(r'[^a-z0-9_]', '_',
                                    (raw.get("field_name") or "").strip().lower())[:40]
                if not field_name or field_name not in known_fields:
                    skipped_no_field += 1
                    continue
                value = str(raw.get("value", "")).strip()
                if not value or value.lower() in ("null", "none", "n/a", ""):
                    skipped_empty += 1
                    continue
                chunk_idx = int(raw.get("chunk_idx", -1))
                # Resolve chunk_id and agreement_id from metadata
                chunk_id = raw.get("chunk_id", "")
                agreement_id = raw.get("agreement_id", "")
                if chunk_idx >= 0 and not chunk_id and state._chunk_ids:
                    if chunk_idx < len(state._chunk_ids):
                        chunk_id = state._chunk_ids[chunk_idx]
                if chunk_idx >= 0 and not agreement_id and state._agreement_ids:
                    if chunk_idx < len(state._agreement_ids):
                        agreement_id = state._agreement_ids[chunk_idx]

                evidence = raw.get("evidence_span", raw.get("span_text", ""))
                conf = float(raw.get("confidence", 0.5))

                # Grounding boost: if value appears in evidence span
                if evidence and value.lower() in evidence.lower():
                    conf = min(1.0, conf + 0.15)

                state.extractions.append(ExtractionV3(
                    field_name=field_name,
                    value=value,
                    confidence=conf,
                    evidence_span=str(evidence)[:500],
                    chunk_idx=chunk_idx,
                    chunk_id=chunk_id,
                    agreement_id=agreement_id,
                    source=raw.get("source", "agent"),
                    iteration=state.iteration,
                ))
                added += 1

        parts = [f"Added {added} extractions (total: {len(state.extractions)})"]
        if skipped_no_field:
            parts.append(f"{skipped_no_field} skipped (unknown field — deposit fields first)")
        if skipped_empty:
            parts.append(f"{skipped_empty} skipped (empty value)")
        return ". ".join(parts)

    def coverage_report() -> str:
        """Get convergence metrics, gap analysis, extraction coverage, and efficiency.
        Returns JSON with field count, extraction count, delta trend, token usage, and whether to continue."""
        with _lock:
            n = len(state.fields)
            n_ext = len(state.extractions)
            deltas = state.state_deltas[-5:]
            converging = (
                len(deltas) >= config.adaptive_stop_patience
                and all(d < config.adaptive_stop_epsilon for d in deltas[-config.adaptive_stop_patience:])
            )
            # Extraction coverage: how many fields have at least one extraction
            fields_with_extraction = len({e.field_name for e in state.extractions})
            chunks_with_extraction = len({e.chunk_idx for e in state.extractions if e.chunk_idx >= 0})

            # Token efficiency (Concern 4)
            tokens_k = state.tokens_consumed / 1000.0
            fields_per_1k = n / max(tokens_k, 0.1)
            over_budget = state.tokens_consumed > config.token_budget

            report = {
                "fields_discovered": n,
                "clause_types_discovered": len(state.clause_types),
                "intents_discovered": len(state.intents),
                "obligations_discovered": len(state.obligations),
                "extractions_deposited": n_ext,
                "fields_with_extractions": fields_with_extraction,
                "extraction_field_coverage": round(fields_with_extraction / max(1, n), 2),
                "chunks_with_extractions": chunks_with_extraction,
                "chunks_extraction_coverage": round(chunks_with_extraction / max(1, state.total_chunks), 2),
                "chunks_seen": state.chunks_seen,
                "total_chunks": state.total_chunks,
                "chunks_remaining": state.total_chunks - state.chunks_seen,
                "recent_deltas": [round(d, 3) for d in deltas],
                "converging": converging,
                "grounded_calls": state.grounded_calls,
                "exploratory_calls": state.exploratory_calls,
                # Efficiency metrics (Concern 4)
                "tokens_consumed": state.tokens_consumed,
                "token_budget_remaining": max(0, config.token_budget - state.tokens_consumed),
                "fields_per_1k_tokens": round(fields_per_1k, 3),
                "extraction_only_mode": state.extraction_only_mode,
                "efficiency_warnings": state.efficiency_warnings,
                "fields_pruned_total": state.fields_pruned_total,
                "recommendation": (
                    "STOP — token budget exceeded"
                    if over_budget else
                    "EXTRACT ONLY — low discovery yield, focus on extraction"
                    if state.extraction_only_mode else
                    "STOP — fields have converged, diminishing returns"
                    if converging else
                    "CONTINUE — still finding new patterns"
                    if n < config.max_fields else
                    "CONSIDER STOPPING — near max field limit"
                ),
            }
        return json.dumps(report, indent=2)

    def critique_fields() -> str:
        """Dense self-critique of the current field set (Concern 2: dense feedback for RLM).
        Scores against extraction-ready quality signals, not stylistic coherence.
        Returns a structured rubric with scores and specific actionable gaps.

        Use this after depositing fields to understand what to improve."""
        with _lock:
            fields = state.fields
            if not fields:
                return json.dumps({
                    "overall_grade": "F",
                    "message": "No fields deposited yet. Start with llm_query() to analyze chunks.",
                    "scores": {},
                })

            n = len(fields)

            # 1. Type coverage: how many distinct types are represented
            types_present = {f.value_type for f in fields}
            all_types = {"string", "date", "money", "duration", "enum",
                         "boolean", "percentage", "integer", "entity_name", "ref"}
            type_coverage = len(types_present) / min(len(all_types), max(n, 1))
            type_coverage = min(type_coverage, 1.0)

            # 2. Description quality: non-empty desc > 10 chars
            good_desc = sum(1 for f in fields if len(f.description) > 10)
            desc_quality = good_desc / n

            # 3. Grounding score: fields with span_text OR multi-span evidence
            grounded = sum(
                1 for f in fields
                if (f.span_text and len(f.span_text) > 5) or len(f.span_texts) >= 2
            )
            grounding_score = grounded / n

            # 4. Confidence distribution
            avg_confidence = sum(f.confidence for f in fields) / n
            low_conf = [f.name for f in fields if f.confidence < 0.4]

            # 5. Redundancy check: flag fields with very similar names
            redundancy_flags = []
            norm_names = [re.sub(r'[_\s]', '', f.name.lower()) for f in fields]
            for i in range(len(norm_names)):
                for j in range(i + 1, len(norm_names)):
                    if norm_names[i] in norm_names[j] or norm_names[j] in norm_names[i]:
                        redundancy_flags.append(f"{fields[i].name} ~ {fields[j].name}")

            # 6. Source diversity
            sources = {f.source for f in fields}
            source_diversity = len(sources) / 3.0  # grounded, exploratory, agent

            # Weighted overall (extraction-task-aligned, Concern 3)
            overall = (
                0.20 * type_coverage
                + 0.25 * desc_quality
                + 0.30 * grounding_score  # heaviest — task-aligned, not judge-aligned
                + 0.15 * avg_confidence
                + 0.10 * source_diversity
            )

            # Specific gaps (the dense feedback RLMs need, Concern 2)
            gaps = []
            if not types_present & {"date", "duration"}:
                gaps.append("Missing temporal fields (date/duration) — look for deadlines, terms, effective dates")
            if not types_present & {"money", "currency", "percentage"}:
                gaps.append("Missing financial fields — look for amounts, fees, percentages")
            if grounding_score < 0.5:
                ungrounded = [
                    f.name for f in fields
                    if (not f.span_text or len(f.span_text) <= 5) and len(f.span_texts) < 2
                ]
                gaps.append(f"Weak grounding on: {', '.join(ungrounded[:5])}. Re-run llm_query asking for exact span_text quotes (or span_texts for derived fields).")
            if desc_quality < 0.7:
                weak_desc = [f.name for f in fields if len(f.description) <= 10]
                gaps.append(f"Weak descriptions on: {', '.join(weak_desc[:5])}. Enrich with one-sentence explanations.")
            if redundancy_flags:
                gaps.append(f"Possible duplicates: {'; '.join(redundancy_flags[:3])}")
            if n < 5:
                gaps.append(f"Only {n} fields — try llm_explore() for more niche patterns.")
            if state.extraction_only_mode:
                gaps.append("EXTRACTION-ONLY MODE active — focus on extracting values, not new fields.")

            grade = (
                "A" if overall >= 0.85 else
                "B" if overall >= 0.70 else
                "C" if overall >= 0.55 else
                "D" if overall >= 0.40 else "F"
            )

            critique = {
                "overall_grade": grade,
                "overall_score": round(overall, 3),
                "scores": {
                    "type_coverage": round(type_coverage, 3),
                    "description_quality": round(desc_quality, 3),
                    "grounding_score": round(grounding_score, 3),
                    "avg_confidence": round(avg_confidence, 3),
                    "source_diversity": round(source_diversity, 3),
                },
                "n_fields": n,
                "low_confidence_fields": low_conf[:5],
                "redundancy_flags": redundancy_flags[:5],
                "specific_gaps": gaps,
                # Tight micro-state for reinjection (Concern 1: system reinjection pattern)
                "last_failure": gaps[0] if gaps else "none",
                "next_check": (
                    "ground ungrounded fields" if grounding_score < 0.5
                    else "add temporal/financial types" if type_coverage < 0.4
                    else "enrich descriptions" if desc_quality < 0.7
                    else "explore for niche patterns" if n < 8
                    else "extraction coverage"
                ),
                "strongest_signal": (
                    f"Best field: {max(fields, key=lambda f: f.confidence).name} "
                    f"(conf={max(f.confidence for f in fields):.2f})"
                    if fields else "none"
                ),
            }
        return json.dumps(critique, indent=2)

    def budget() -> str:
        """Check remaining resources: time, tokens, iterations.
        Call this to decide whether to continue exploring or wrap up.
        Inspired by DSPy PR #9295 budget awareness pattern."""
        with _lock:
            elapsed = time.time() - state.start_time if state.start_time else 0.0
            time_budget = config.max_time_s
            time_remaining = max(0.0, time_budget - elapsed) if time_budget > 0 else float("inf")
            token_remaining = max(0, config.token_budget - state.tokens_consumed)
            iter_count = state.iteration

            # Low-resource warnings
            warnings = []
            if time_budget > 0 and time_remaining < 30:
                warnings.append("LOW TIME — less than 30s remaining")
            if token_remaining < 50_000:
                warnings.append("LOW TOKENS — less than 50k tokens remaining")
            if iter_count >= config.max_iterations - 2:
                warnings.append("LOW ITERATIONS — near max iteration limit")
            if state.extraction_only_mode:
                warnings.append("EXTRACTION-ONLY MODE — discovery yield too low")

            report = {
                "elapsed_s": round(elapsed, 1),
                "time_remaining_s": round(time_remaining, 1) if time_budget > 0 else "unlimited",
                "tokens_consumed": state.tokens_consumed,
                "token_budget_remaining": token_remaining,
                "iteration": iter_count,
                "max_iterations": config.max_iterations,
                "extraction_only_mode": state.extraction_only_mode,
                "warnings": warnings,
                "recommendation": (
                    "WRAP UP — time almost out, call SUBMIT"
                    if time_budget > 0 and time_remaining < 15 else
                    "WRAP UP — token budget nearly exhausted"
                    if token_remaining < 20_000 else
                    "EXTRACT ONLY — focus on extraction, not new fields"
                    if state.extraction_only_mode else
                    "CONTINUE"
                ),
            }
        return json.dumps(report, indent=2)

    tools = [get_state, deposit_fields, deposit_clauses, deposit_intents,
             deposit_obligations, deposit_extractions, coverage_report,
             critique_fields, budget]
    return tools


# ══════════════════════════════════════════════════════════════════════════════
# V3 RLM Class
# ══════════════════════════════════════════════════════════════════════════════


class RLMV3:
    """
    RLM V3 — Agentic discovery with structured state accumulation.

    Combines:
    - DSPy RLM REPL loop (agent writes code, calls tools)
    - Latent Briefing trajectory compression (task-guided context)
    - RollingState typed accumulator (all findings are structured)
    - Dual-mode workers (grounded + exploratory)
    - Post-loop canonicalization and overlap dedup
    """

    def __init__(
        self,
        config: RLMV3Config | None = None,
        lm: dspy.LM | None = None,
        sub_lm: dspy.LM | None = None,
    ):
        self.config = config or RLMV3Config()
        self._lm = lm
        self._sub_lm = sub_lm
        self._embedder = None
        self._briefing_engine: TextLatentBriefing | None = None
        self._fallback_client = None  # cached AzureOpenAI client for structured calls

    # ── Lazy initialization ───────────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _get_shared_embedder()
        return self._embedder

    def _get_lm(self) -> dspy.LM:
        if self._lm is not None:
            return self._lm
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Main LM always uses gpt-5.4 (full model) for quality discovery
        deployment = os.getenv("AZURE_OPENAI_MAIN_MODEL", "gpt-5.4")
        if endpoint and api_key:
            lm = dspy.LM(f"azure/{deployment}", api_key=api_key,
                         api_base=endpoint, api_version=api_version, num_retries=5)
        else:
            lm = dspy.LM(f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}",
                         api_key=os.getenv("OPENAI_API_KEY", ""), num_retries=5)
        self._lm = lm
        return lm

    def _get_sub_lm(self) -> dspy.LM:
        if self._sub_lm is not None:
            return self._sub_lm
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Sub LM uses gpt-5.4-mini for cheaper/faster sub-calls inside the REPL
        sub_deployment = os.getenv("AZURE_OPENAI_SUB_MODEL", "gpt-5.4-mini")
        if endpoint and api_key:
            self._sub_lm = dspy.LM(f"azure/{sub_deployment}", api_key=api_key,
                                   api_base=endpoint, api_version=api_version, num_retries=5)
        else:
            self._sub_lm = self._get_lm()
        return self._sub_lm

    def _make_briefing_engine(self) -> TextLatentBriefing:
        return TextLatentBriefing(
            config=CompactionConfig(
                tau=self.config.briefing_tau,
                min_retention=self.config.briefing_min_retention,
                max_retention=self.config.briefing_max_retention,
            ),
        )

    # ══════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════

    def discover(
        self,
        cluster_label: str,
        keywords: list[str],
        chunk_texts: list[str],
    ) -> tuple[list[dict], dict]:
        """
        Full V3 hybrid discovery pipeline.

        Phase 1: Agentic REPL exploration with state tools + dual workers
        Phase 2: Post-loop overlap merge + canonicalization
        Phase 3: Quality scoring + output conversion

        Returns:
            (fields, meta) — fields compatible with extract_fields_from_chunk()
        """
        t0 = time.time()
        # Random sample (not first-N) for diverse coverage across the cluster
        import random as _rand
        n_sample = min(self.config.max_sample_chunks, len(chunk_texts))
        if n_sample < len(chunk_texts):
            sample = _rand.sample(chunk_texts, n_sample)
        else:
            sample = list(chunk_texts)

        # ── Initialize state + briefing ──
        state = RollingStateV3(
            cluster_label=cluster_label,
            total_chunks=len(sample),
            start_time=t0,
        )
        # Pre-seed domain-level universal fields so the agent focuses on
        # cluster-specific delta fields (two-tier field architecture).
        if self.config.seed_fields:
            for sf in self.config.seed_fields:
                state.fields.append(FieldV3(
                    name=sf["name"],
                    value_type=sf.get("type", "text"),
                    description=sf.get("description", ""),
                    span_text="",
                    confidence=0.8,
                    source="domain_seed",
                    iteration=0,
                ))
        briefing_engine = self._make_briefing_engine()
        self._briefing_engine = briefing_engine

        meta: dict[str, Any] = {
            "version": "v3",
            "n_chunks_sampled": len(sample),
            "n_seed_fields": len(self.config.seed_fields or []),
            "phase1_repl": {},
            "phase2_postprocess": {},
            "briefing_stats": {},
        }

        # ── Phase 1: Agentic REPL discovery ──
        phase1_meta = self._phase1_agentic_discovery(
            state, sample, cluster_label, keywords, briefing_engine,
        )
        meta["phase1_repl"] = phase1_meta

        # ── Phase 2: Post-loop refinement ──
        phase2_meta = self._phase2_postprocess(state)
        meta["phase2_postprocess"] = phase2_meta

        # ── Phase 3: Convert to output format ──
        fields = self._to_field_dicts(state)

        meta.update({
            "n_fields": len(fields),
            "n_clause_types": len(state.clause_types),
            "n_intents": len(state.intents),
            "n_obligations": len(state.obligations),
            "total_time_s": round(time.time() - t0, 2),
            "clause_types": [ct.model_dump() for ct in state.clause_types],
            "intents": [i.model_dump() for i in state.intents],
            "obligations": [o.model_dump() for o in state.obligations],
        })

        # Briefing stats
        if briefing_engine.total_tokens_saved > 0:
            meta["briefing_stats"] = {
                "total_tokens_saved": briefing_engine.total_tokens_saved,
                "trajectory_segments": briefing_engine.trajectory_length,
            }

        return fields, meta

    # ══════════════════════════════════════════════════════════════════════
    # UNIFIED DISCOVER + EXTRACT
    # ══════════════════════════════════════════════════════════════════════

    def discover_and_extract(
        self,
        cluster_label: str,
        keywords: list[str],
        chunk_texts: list[str],
        chunk_ids: list[str] | None = None,
        agreement_ids: list[str] | None = None,
    ) -> tuple[list[dict], list[dict], dict]:
        """
        Discover fields AND extract values in a single agentic pass.

        The REPL agent discovers field schemas and immediately extracts values
        from every chunk it reads. Chunks seen during discovery get "free"
        extractions with full trajectory context. Remaining chunks are flagged
        in meta for a lightweight downstream extraction pass.

        Args:
            cluster_label: Clause type label for this cluster
            keywords: Cluster keywords
            chunk_texts: All chunk texts in this cluster
            chunk_ids: Optional DB chunk IDs (parallel to chunk_texts)
            agreement_ids: Optional agreement IDs (parallel to chunk_texts)

        Returns:
            (fields, extractions, meta)
            - fields: list[dict] compatible with extract_fields_from_chunk()
            - extractions: list[dict] with keys {field_name, value, confidence,
              evidence_span, chunk_id, agreement_id}
            - meta: pipeline statistics including chunks_extracted and
              chunks_remaining for downstream pass
        """
        # Enable inline extraction
        orig_extract = self.config.extract_inline
        self.config.extract_inline = True

        t0 = time.time()
        sample = chunk_texts[:self.config.max_sample_chunks]
        ids = (chunk_ids or [""] * len(chunk_texts))[:self.config.max_sample_chunks]
        agr_ids = (agreement_ids or [""] * len(chunk_texts))[:self.config.max_sample_chunks]

        state = RollingStateV3(
            cluster_label=cluster_label,
            total_chunks=len(sample),
            start_time=t0,
            _chunk_ids=ids,
            _agreement_ids=agr_ids,
        )
        # Pre-seed domain-level universal fields (two-tier field architecture).
        if self.config.seed_fields:
            for sf in self.config.seed_fields:
                state.fields.append(FieldV3(
                    name=sf["name"],
                    value_type=sf.get("type", "text"),
                    description=sf.get("description", ""),
                    span_text="",
                    confidence=0.8,
                    source="domain_seed",
                    iteration=0,
                ))
        briefing_engine = self._make_briefing_engine()
        self._briefing_engine = briefing_engine

        meta: dict[str, Any] = {
            "version": "v3_unified",
            "n_chunks_sampled": len(sample),
            "n_seed_fields": len(self.config.seed_fields or []),
            "phase1_repl": {},
            "phase2_postprocess": {},
            "briefing_stats": {},
        }

        # Phase 1: Agentic REPL with extraction
        phase1_meta = self._phase1_agentic_discovery(
            state, sample, cluster_label, keywords, briefing_engine,
        )
        meta["phase1_repl"] = phase1_meta

        # Phase 2: Post-loop refinement (fields only; extractions are kept as-is)
        phase2_meta = self._phase2_postprocess(state)
        meta["phase2_postprocess"] = phase2_meta

        # Phase 3: Convert outputs
        fields = self._to_field_dicts(state)

        # Filter extractions to only keep fields that survived canonicalization
        surviving_field_names = {f["name"] for f in fields}
        valid_extractions = [
            e for e in state.extractions
            if e.field_name in surviving_field_names
            and e.confidence >= self.config.min_extraction_confidence
        ]

        extractions = [
            {
                "field_name": e.field_name,
                "value": e.value,
                "confidence": round(e.confidence, 3),
                "evidence_span": e.evidence_span,
                "chunk_idx": e.chunk_idx,
                "chunk_id": e.chunk_id,
                "agreement_id": e.agreement_id,
                "source": e.source,
            }
            for e in valid_extractions
        ]

        # Identify chunks NOT yet extracted (for downstream lightweight pass)
        extracted_chunk_indices = {e.chunk_idx for e in valid_extractions if e.chunk_idx >= 0}
        remaining_indices = [i for i in range(len(chunk_texts)) if i not in extracted_chunk_indices]

        meta.update({
            "n_fields": len(fields),
            "n_extractions_inline": len(extractions),
            "n_chunks_extracted": len(extracted_chunk_indices),
            "n_chunks_remaining": len(remaining_indices),
            "remaining_chunk_indices": remaining_indices,
            "n_clause_types": len(state.clause_types),
            "n_intents": len(state.intents),
            "n_obligations": len(state.obligations),
            "total_time_s": round(time.time() - t0, 2),
            "clause_types": [ct.model_dump() for ct in state.clause_types],
            "intents": [i.model_dump() for i in state.intents],
            "obligations": [o.model_dump() for o in state.obligations],
        })

        if briefing_engine.total_tokens_saved > 0:
            meta["briefing_stats"] = {
                "total_tokens_saved": briefing_engine.total_tokens_saved,
                "trajectory_segments": briefing_engine.trajectory_length,
            }

        self.config.extract_inline = orig_extract
        return fields, extractions, meta

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Agentic REPL with structured tools
    # ══════════════════════════════════════════════════════════════════════

    def _prune_state(self, state: RollingStateV3) -> int:
        """Prune lowest-confidence fields when over budget (Concern 1: state budget).

        Keeps the top `state_prune_keep` fields by confidence.
        Returns the number of fields pruned.
        """
        budget = self.config.state_field_budget
        if len(state.fields) <= budget:
            return 0
        # Sort descending by confidence, keep top N
        state.fields.sort(key=lambda f: f.confidence, reverse=True)
        pruned = state.fields[self.config.state_prune_keep:]
        state.fields = state.fields[:self.config.state_prune_keep]
        n_pruned = len(pruned)
        state.fields_pruned_total += n_pruned
        logger.info(
            "State pruned: removed %d lowest-confidence fields (kept %d, total pruned: %d)",
            n_pruned, len(state.fields), state.fields_pruned_total,
        )
        return n_pruned

    def _phase1_agentic_discovery(
        self,
        state: RollingStateV3,
        chunk_texts: list[str],
        cluster_label: str,
        keywords: list[str],
        briefing_engine: TextLatentBriefing,
    ) -> dict:
        """Run DSPy RLM with custom tools that deposit into RollingState."""
        phase_meta: dict[str, Any] = {
            "repl_iterations": 0,
            "grounded_calls": 0,
            "exploratory_calls": 0,
            "fields_from_repl": 0,
            "repl_error": None,
        }

        lm = self._get_lm()
        sub_lm = self._get_sub_lm()

        # Use dspy.context() instead of dspy.configure() for thread safety.
        # dspy.configure() uses thread-local storage and fails when called from
        # multiple concurrent thread-pool workers. dspy.context() is a context
        # manager that's safe to use from any thread.
        with dspy.context(lm=lm):
            return self._phase1_inner(
                state, chunk_texts, cluster_label, keywords,
                briefing_engine, phase_meta, lm, sub_lm,
            )

    def _phase1_inner(self, state, chunk_texts, cluster_label, keywords,
                      briefing_engine, phase_meta, lm, sub_lm):
        """Inner body of phase1 discovery, runs inside dspy.context(lm=...)."""
        # Create state-bound tools
        state_tools = _make_state_tools(
            state, briefing_engine, self.config, phase_meta,
            embedder_fn=lambda texts: self.embedder.encode(texts, normalize_embeddings=True),
        )

        # ── Build the RLM with memory-augmented + exploratory workers ──
        class _V3RLM(dspy.RLM):
            """Inner RLM subclass with LB memory injection + dual-mode workers."""

            def __init__(inner_self, **kwargs):
                inner_self._briefing_engine = briefing_engine
                inner_self._config = self.config
                inner_self._state = state
                inner_self._iteration_count = 0
                inner_self._all_stats: list[CompactionStats] = []
                super().__init__(**kwargs)

            def _make_llm_tools(inner_self, max_workers: int = 8) -> dict[str, Callable]:
                """Create dual-mode LLM tools with trajectory memory."""
                base_tools = super()._make_llm_tools(max_workers)
                base_query = base_tools["llm_query"]
                base_batched = base_tools["llm_query_batched"]
                engine = inner_self

                def llm_query(prompt: str) -> str:
                    """Query the LLM with context from prior analysis (grounded mode, precise).
                    Use this for thorough, accurate analysis of specific chunks."""
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

                    response = base_query(augmented)
                    engine._add_trajectory(
                        f"[GROUNDED] Query: {prompt[:200]}\nResponse: {response[:500]}",
                    )
                    with threading.Lock():
                        state.grounded_calls += 1
                        phase_meta["grounded_calls"] += 1
                    return response

                def llm_explore(prompt: str) -> str:
                    """Query the LLM WITHOUT prior context (exploratory mode, creative).
                    Use this to find novel, niche patterns with fresh eyes.
                    Good for finding unusual fields that grounded analysis might miss."""
                    # No briefing — fresh perspective
                    response = base_query(prompt)
                    engine._add_trajectory(
                        f"[EXPLORATORY] Query: {prompt[:200]}\nResponse: {response[:500]}",
                    )
                    with threading.Lock():
                        state.exploratory_calls += 1
                        phase_meta["exploratory_calls"] += 1
                    return response

                def llm_query_batched(prompts: list[str]) -> list[str]:
                    """Query the LLM with multiple prompts concurrently."""
                    if not prompts:
                        return []
                    briefing = engine._compact_for_worker(prompts[0])
                    if briefing:
                        augmented = [
                            f"=== CONTEXT ===\n{briefing}\n=== END ===\n\n{p}"
                            for p in prompts
                        ]
                    else:
                        augmented = prompts
                    responses = base_batched(augmented)
                    engine._add_trajectory(
                        f"Batch of {len(prompts)} worker queries completed",
                    )
                    return responses

                return {
                    "llm_query": llm_query,
                    "llm_explore": llm_explore,
                    "llm_query_batched": llm_query_batched,
                }

            def _execute_iteration(inner_self, repl, variables, history,
                                   iteration, input_args, output_field_names):
                """Capture iteration in trajectory + track state delta + token metering."""
                inner_self._iteration_count = iteration
                prev_fields = len(state.fields)

                result = super()._execute_iteration(
                    repl, variables, history, iteration,
                    input_args, output_field_names,
                )

                state.iteration = iteration + 1
                # Track state delta for convergence
                new_fields = len(state.fields)
                if prev_fields > 0:
                    delta = abs(new_fields - prev_fields) / max(new_fields, 1)
                else:
                    delta = 1.0 if new_fields == 0 else 0.5
                state.state_deltas.append(delta)

                # ── Token metering (Concern 4: efficiency early-exit) ──
                iter_tokens = 0
                if not isinstance(result, Prediction):
                    entries = list(result)
                    for entry in entries:
                        code_len = len(getattr(entry, 'code', '') or '')
                        output_len = len(getattr(entry, 'output', '') or '')
                        # Rough token estimate: ~4 chars per token
                        iter_tokens += (code_len + output_len) // 4
                state.tokens_consumed += iter_tokens

                # Track fields discovered this iteration
                fields_this_iter = new_fields - prev_fields
                state.fields_per_iteration.append(fields_this_iter)

                # Check efficiency: if recent iterations produce too few fields
                # per token, switch to extraction-only mode (Concern 4)
                # Recovery: if a burst of new fields appears, exit extraction-only
                cfg = inner_self._config
                window = cfg.efficiency_window
                if len(state.fields_per_iteration) >= window:
                    recent_fields = sum(state.fields_per_iteration[-window:])
                    # Sum tokens across recent iterations (rough estimate)
                    recent_tokens_k = max(iter_tokens * window / 1000.0, 0.1)
                    efficiency = recent_fields / recent_tokens_k

                    if state.extraction_only_mode:
                        # Recovery: if we found >=2 new fields in the last window,
                        # re-enable discovery (the agent hit a new vein)
                        if recent_fields >= 2:
                            state.extraction_only_mode = False
                            state.efficiency_warnings = 0
                            logger.info(
                                "Efficiency recovery: extraction_only_mode=False "
                                "(%d new fields in last %d iters)",
                                recent_fields, window,
                            )
                    elif efficiency < cfg.min_fields_per_1k_tokens and iteration >= 2:
                        state.efficiency_warnings += 1
                        if state.efficiency_warnings >= window:
                            state.extraction_only_mode = True
                            logger.info(
                                "Efficiency early-exit: extraction_only_mode=True "
                                "(%.2f fields/1k tokens over %d iters)",
                                efficiency, window,
                            )

                # ── State pruning (Concern 1: capped state budget) ──
                self._prune_state(state)

                # ── Token budget hard stop — return Prediction to force REPL exit ──
                if state.tokens_consumed > cfg.token_budget:
                    logger.warning(
                        "Token budget exceeded (%d > %d), forcing REPL exit",
                        state.tokens_consumed, cfg.token_budget,
                    )
                    return Prediction(fields_json=json.dumps([
                        {"name": f.name, "type": f.value_type, "description": f.description}
                        for f in state.fields
                    ]))

                # ── Wall-clock timeout — return Prediction to force REPL exit ──
                if cfg.max_time_s > 0 and state.start_time > 0:
                    elapsed = time.time() - state.start_time
                    if elapsed > cfg.max_time_s:
                        logger.warning(
                            "Wall-clock timeout: %.1fs > %ds, forcing REPL exit",
                            elapsed, cfg.max_time_s,
                        )
                        return Prediction(fields_json=json.dumps([
                            {"name": f.name, "type": f.value_type, "description": f.description}
                            for f in state.fields
                        ]))

                # Record in trajectory
                if isinstance(result, Prediction):
                    reasoning = getattr(result, "final_reasoning", "")
                    if reasoning:
                        inner_self._add_trajectory(f"Final: {reasoning[:500]}")
                else:
                    if not isinstance(result, list):
                        entries = list(result)
                    if entries:
                        last = entries[-1]
                        inner_self._add_trajectory(
                            f"[Iter {iteration+1}] "
                            f"Code: {getattr(last, 'code', '')[:300]}\n"
                            f"Output: {getattr(last, 'output', '')[:500]}"
                        )

                phase_meta["repl_iterations"] = iteration + 1
                return result

            def forward(inner_self, **input_args) -> Prediction:
                inner_self._briefing_engine.reset()
                inner_self._all_stats = []
                inner_self._iteration_count = 0
                result = super().forward(**input_args)
                result.briefing_stats = {
                    "n_compactions": len(inner_self._all_stats),
                    "total_tokens_saved": inner_self._briefing_engine.total_tokens_saved,
                }
                return result

            # ── Briefing helpers ──
            def _compact_for_worker(inner_self, task_prompt: str) -> str:
                if inner_self._briefing_engine.trajectory_length == 0:
                    return ""
                tau = inner_self._compute_tau()
                briefing_text, stats = inner_self._briefing_engine.compact_for_task(
                    task_prompt=task_prompt,
                    tau=tau,
                    max_tokens=inner_self._config.briefing_max_tokens,
                )
                inner_self._all_stats.append(stats)
                return briefing_text

            def _add_trajectory(inner_self, text: str):
                inner_self._briefing_engine.add_trajectory_segment(
                    text=text,
                    metadata={"iteration": inner_self._iteration_count},
                    segment_type="observation",
                )

            def _compute_tau(inner_self) -> float:
                base = inner_self._config.briefing_tau
                if not inner_self._config.adaptive_tau:
                    return base
                it = inner_self._iteration_count
                n_seg = inner_self._briefing_engine.trajectory_length
                depth = min(1.0 + (it / 15) * 0.5, 2.0)
                length = 0.8 if n_seg > 20 else (0.9 if n_seg > 10 else 1.0)
                return round(base * depth * length, 2)

        # ── Build the RLM and run ──
        try:
            sig = V3DiscoverExtractSig if self.config.extract_inline else V3DiscoverySig

            # Build RLM kwargs — DSPy 3.2 API
            rlm_kwargs: dict[str, Any] = {
                "signature": sig,
                "max_iterations": self.config.max_iterations,
                "max_llm_calls": self.config.max_llm_calls,
                "verbose": False,
                "sub_lm": sub_lm,
                "tools": state_tools,
            }

            rlm = _V3RLM(**rlm_kwargs)
            logger.info(
                f"RLM interpreter: "
                f"{type(rlm._interpreter).__name__ if rlm._interpreter else 'PythonInterpreter (Deno/Pyodide sandbox, default)'}"
            )

            # Enrich chunks with IDs and clause metadata
            chunk_records = []
            for i, t in enumerate(chunk_texts):
                rec: dict[str, Any] = {"idx": i, "text": t[:2500]}
                if self.config.extract_inline:
                    if state._chunk_ids and i < len(state._chunk_ids):
                        rec["chunk_id"] = state._chunk_ids[i]
                    if state._agreement_ids and i < len(state._agreement_ids):
                        rec["agreement_id"] = state._agreement_ids[i]
                chunk_records.append(rec)
            chunks_json = json.dumps(chunk_records)

            # Run RLM with hard wall-clock timeout via concurrent.futures
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            def _run_rlm():
                # dspy.context() is thread-local and doesn't propagate to
                # ThreadPoolExecutor threads, so we must re-enter it here.
                with dspy.context(lm=lm):
                    return rlm(
                        chunks_json=chunks_json,
                        cluster_label=cluster_label,
                        keywords=", ".join(keywords[:10]),
                        max_fields=str(self.config.max_fields),
                    )

            timeout = self.config.max_time_s if self.config.max_time_s > 0 else 300
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_rlm)
                try:
                    result = future.result(timeout=timeout)
                except (FuturesTimeout, TimeoutError):
                    logger.warning(
                        f"REPL hard timeout after {timeout}s — "
                        f"state has {len(state.fields)} fields, falling back"
                    )
                    future.cancel()
                    if len(state.fields) == 0:
                        raise TimeoutError(f"REPL timed out after {timeout}s with 0 fields")
                    phase_meta["repl_timeout"] = True
                    phase_meta["fields_from_repl"] = len(state.fields)
                    return phase_meta

            # The agent may also have returned fields in the output field —
            # parse those and deposit into state too
            raw_output = getattr(result, "fields_json", "")
            if raw_output:
                try:
                    parsed = json.loads(raw_output)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("fields", [])
                    if isinstance(parsed, list):
                        existing = {f.name for f in state.fields}
                        for raw in parsed:
                            if not isinstance(raw, dict):
                                continue
                            name = (raw.get("name") or "").strip()
                            if not name:
                                continue
                            name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
                            if name in existing:
                                continue
                            state.fields.append(FieldV3(
                                name=name,
                                value_type=raw.get("value_type", raw.get("type", "string")),
                                description=raw.get("description", ""),
                                span_text=raw.get("span_text", ""),
                                confidence=float(raw.get("confidence", 0.5)),
                                source="repl_output",
                                iteration=state.iteration,
                            ))
                            existing.add(name)
                except (json.JSONDecodeError, TypeError):
                    pass

            phase_meta["fields_from_repl"] = len(state.fields)

            if hasattr(result, "briefing_stats"):
                phase_meta["briefing_stats"] = result.briefing_stats

        except Exception as e:
            logger.warning(f"V3 REPL phase failed: {e}")
            phase_meta["repl_error"] = str(e)
            # Fallback: run structured dual-mode passes (V2-style)
            self._fallback_structured_discovery(
                state, chunk_texts, cluster_label, keywords, briefing_engine, phase_meta,
            )

        return phase_meta

    def _fallback_structured_discovery(
        self,
        state: RollingStateV3,
        chunk_texts: list[str],
        cluster_label: str,
        keywords: list[str],
        briefing_engine: TextLatentBriefing,
        meta: dict,
    ):
        """V2-style batch discovery as fallback when REPL fails."""
        logger.info("V3 fallback: running V2-style structured discovery")
        meta["fallback_used"] = True

        # Use larger batches for better context per LLM call
        batch_size = 4
        n_batches = min(8, (len(chunk_texts) + batch_size - 1) // batch_size)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch = chunk_texts[start:start + batch_size]
            if not batch:
                break

            chunks_text = "\n\n".join(
                f"[Excerpt {start + i + 1}]\n{t[:2000]}" for i, t in enumerate(batch)
            )
            prev_count = len(state.fields)

            # Grounded pass
            grounded = self._structured_call(
                chunks_text, cluster_label, keywords, state,
                temperature=self.config.grounded_temp, mode="grounded",
            )
            logger.info(
                f"  Fallback batch {batch_idx+1}/{n_batches} grounded: "
                f"{len(grounded.get('fields', []))} fields"
            )
            # Exploratory pass
            exploratory = self._structured_call(
                chunks_text, cluster_label, keywords, state,
                temperature=self.config.exploratory_temp, mode="exploratory",
            )
            logger.info(
                f"  Fallback batch {batch_idx+1}/{n_batches} exploratory: "
                f"{len(exploratory.get('fields', []))} fields"
            )

            # Overlap merge
            merged, merge_stats = self._overlap_merge(grounded, exploratory)

            # Update state
            self._update_state_from_merged(state, merged, batch)

            # Check adaptive stop — only after we have at least some fields
            new_count = len(state.fields)
            delta = (
                abs(new_count - prev_count) / max(new_count, 1) if prev_count > 0 else 1.0
            )
            state.state_deltas.append(delta)
            state.iteration += 1

            logger.info(
                f"  Fallback batch {batch_idx+1} state: {new_count} fields total, "
                f"delta={delta:.3f}"
            )

            # Only stop early if we already have fields (avoid stopping at 0)
            if (
                new_count > 0
                and len(state.state_deltas) >= self.config.adaptive_stop_patience
                and all(
                    d < self.config.adaptive_stop_epsilon
                    for d in state.state_deltas[-self.config.adaptive_stop_patience:]
                )
            ):
                meta["fallback_stopped_early"] = True
                logger.info(f"  Fallback adaptive stop at batch {batch_idx+1}")
                break

    def _structured_call(
        self,
        chunks_text: str,
        cluster_label: str,
        keywords: list[str],
        state: RollingStateV3,
        temperature: float,
        mode: str,
    ) -> dict:
        """Direct Azure OpenAI call for structured 4-output discovery."""
        if mode == "grounded":
            brief = self._build_state_brief(state, chunks_text)
            brief_section = ""
            if brief:
                brief_section = (
                    f"PRIOR CONTEXT:\n{brief}\n\n"
                    f"Focus on NEW discoveries not listed above.\n\n"
                )
            prompt = (
                f"You are a legal contract analyst extracting structured metadata.\n\n"
                f"{brief_section}"
                f"CLAUSE TYPE: {cluster_label}\n"
                f"KEYWORDS: {', '.join(keywords[:10])}\n\n"
                f"EXCERPTS:\n{chunks_text}\n\n"
                f"Analyze these excerpts and discover EXTRACTABLE fields — meaning "
                f"fields whose values appear literally in the text (e.g. party names, "
                f"dates, dollar amounts, percentages, durations, specific terms).\n\n"
                f"For each field, provide:\n"
                f'- "name": snake_case identifier (e.g. "effective_date", "royalty_rate")\n'
                f'- "value_type": one of string/date/money/enum/boolean/duration/percentage/integer\n'
                f'- "description": what this field represents\n'
                f'- "span_text": exact quote from the text where a value appears\n'
                f'- "confidence": 0.0-1.0\n\n'
                f"Return 5-15 fields. Focus on concrete, extractable data points.\n\n"
                f'Respond ONLY with valid JSON: {{"fields": [{{...}}], "clause_types": [{{...}}], "intents": [{{...}}], "obligations": [{{...}}]}}'
            )
        else:
            prompt = (
                f"You are a legal analyst finding NOVEL, NICHE patterns in contracts.\n\n"
                f"CLAUSE TYPE: {cluster_label}\n"
                f"KEYWORDS: {', '.join(keywords[:10])}\n\n"
                f"EXCERPTS:\n{chunks_text}\n\n"
                f"With FRESH EYES, discover unusual or domain-specific EXTRACTABLE fields "
                f"— meaning values that appear literally in the text.\n\n"
                f"For each field, provide:\n"
                f'- "name": snake_case identifier\n'
                f'- "value_type": one of string/date/money/enum/boolean/duration/percentage/integer\n'
                f'- "description": what this field represents\n'
                f'- "span_text": exact quote from the text where a value appears\n'
                f'- "confidence": 0.0-1.0\n\n'
                f"Avoid generic fields like party_name or effective_date. "
                f"Return 3-10 niche fields.\n\n"
                f'Respond ONLY with valid JSON: {{"fields": [{{...}}], "clause_types": [{{...}}], "intents": [{{...}}], "obligations": [{{...}}]}}'
            )

        try:
            # Reuse cached client (avoids TCP/TLS handshake per call)
            if not hasattr(self, '_fallback_client') or self._fallback_client is None:
                from openai import AzureOpenAI
                self._fallback_client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                )
            client = self._fallback_client
            # Always use gpt-5.4 for discovery quality
            deployment = os.getenv("AZURE_OPENAI_MAIN_MODEL", "gpt-5.4")
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=4000,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            # Handle nested structure or flat structure
            fields = parsed.get("fields", [])
            if not fields and isinstance(parsed, list):
                fields = parsed
            return {
                "fields": fields,
                "clause_types": parsed.get("clause_types", []),
                "intents": parsed.get("intents", []),
                "obligations": parsed.get("obligations", []),
            }
        except json.JSONDecodeError as je:
            logger.warning(f"V3 structured call ({mode}) JSON parse failed: {je}")
            # Try to extract JSON from response
            try:
                import json_repair
                repaired = json_repair.loads(raw_content)
                if isinstance(repaired, dict):
                    return {
                        "fields": repaired.get("fields", []),
                        "clause_types": repaired.get("clause_types", []),
                        "intents": repaired.get("intents", []),
                        "obligations": repaired.get("obligations", []),
                    }
            except Exception:
                pass
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}
        except Exception as e:
            logger.warning(f"V3 structured call ({mode}) failed: {e}")
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Post-processing (overlap merge + canonicalization)
    # ══════════════════════════════════════════════════════════════════════

    def _phase2_postprocess(self, state: RollingStateV3) -> dict:
        """Overlap dedup + canonicalization with a single shared embedding pass."""
        pre_fields = len(state.fields)
        pre_clauses = len(state.clause_types)

        if len(state.fields) <= 1:
            # Clause type dedup only
            state = self._dedup_clause_types(state)
            return {
                "fields_before": pre_fields, "fields_after": len(state.fields),
                "fields_removed": 0, "clauses_before": pre_clauses,
                "clauses_after": len(state.clause_types),
            }

        # ── Single encode for ALL fields (used by both overlap dedup and canon) ──
        all_texts = [f"{f.name}: {f.description}" for f in state.fields]
        all_vecs = self.embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)

        # ── Step 1: Cross-source overlap dedup ──
        grounded_idx = [i for i, f in enumerate(state.fields)
                        if f.source in ("grounded", "agent", "repl_output")]
        exploratory_idx = [i for i, f in enumerate(state.fields)
                          if f.source == "exploratory"]

        overlap_remove = set()
        if grounded_idx and exploratory_idx:
            g_vecs = all_vecs[grounded_idx]
            for ei in exploratory_idx:
                sims = np.dot(g_vecs, all_vecs[ei])
                if float(sims.max()) >= self.config.overlap_threshold:
                    overlap_remove.add(ei)

        # Rebuild fields and vectors with overlap-removed items excluded
        surviving_idx = [i for i in range(len(state.fields)) if i not in overlap_remove]
        state.fields = [state.fields[i] for i in surviving_idx]
        all_vecs = all_vecs[surviving_idx]

        # ── Step 2: Two-tier canonicalization (reuses same vectors) ──
        state, all_vecs = self._canonicalize(state, all_vecs)

        # ── Step 3: Clause type dedup ──
        state = self._dedup_clause_types(state)

        # ── Step 4: Intent dedup (by normalized label) ──
        pre_intents = len(state.intents)
        state = self._dedup_intents(state)

        # ── Step 5: Obligation dedup (by normalized action + obligor) ──
        pre_obligations = len(state.obligations)
        state = self._dedup_obligations(state)

        post_fields = len(state.fields)
        post_clauses = len(state.clause_types)

        return {
            "fields_before": pre_fields,
            "fields_after": post_fields,
            "fields_removed": pre_fields - post_fields,
            "clauses_before": pre_clauses,
            "clauses_after": post_clauses,
            "intents_before": pre_intents,
            "intents_after": len(state.intents),
            "obligations_before": pre_obligations,
            "obligations_after": len(state.obligations),
        }

    def _canonicalize(self, state: RollingStateV3,
                      precomputed_vecs: np.ndarray | None = None,
                      ) -> tuple[RollingStateV3, np.ndarray]:
        """Two-tier canonicalization: fuzzy name + embedding sim > threshold.
        Accepts pre-computed embeddings to avoid redundant encode calls."""
        if len(state.fields) <= 1:
            vecs = precomputed_vecs if precomputed_vecs is not None else np.empty((0, 384))
            return state, vecs

        to_remove = set()
        names = [f.name for f in state.fields]
        normalized = [re.sub(r'[_\s]', '', n.lower()) for n in names]

        # Tier 1: Rule-based fuzzy
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
                    keep, drop = (i, j) if len(names[i]) <= len(names[j]) else (j, i)
                    if len(state.fields[drop].description) > len(state.fields[keep].description):
                        state.fields[keep] = state.fields[keep].model_copy(
                            update={"description": state.fields[drop].description}
                        )
                    to_remove.add(drop)
                elif (normalized[i] + 's' == normalized[j] or
                      normalized[j] + 's' == normalized[i]):
                    shorter = i if len(names[i]) <= len(names[j]) else j
                    longer = j if shorter == i else i
                    to_remove.add(longer)

        # Tier 2: Embedding similarity (reuse precomputed vectors)
        remaining = [i for i in range(len(state.fields)) if i not in to_remove]
        if len(remaining) > 2:
            if precomputed_vecs is not None:
                vecs = precomputed_vecs[remaining]
            else:
                texts = [f"{state.fields[i].name}: {state.fields[i].description}"
                         for i in remaining]
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
                    if sim[a, b] > self.config.canon_embedding_threshold:
                        if len(fa.description) >= len(fb.description):
                            to_remove.add(remaining[b])
                        else:
                            to_remove.add(remaining[a])

        surviving = [i for i in range(len(state.fields)) if i not in to_remove]
        state.fields = [state.fields[i] for i in surviving]
        out_vecs = precomputed_vecs[surviving] if precomputed_vecs is not None else np.empty((0, 384))
        return state, out_vecs

    @staticmethod
    def _dedup_clause_types(state: RollingStateV3) -> RollingStateV3:
        """Deduplicate clause types by normalized name."""
        seen = {}
        unique = []
        for ct in state.clause_types:
            norm = re.sub(r'[_\s]', '', ct.name.lower())
            if norm not in seen:
                seen[norm] = True
                unique.append(ct)
        state.clause_types = unique
        return state

    @staticmethod
    def _dedup_intents(state: RollingStateV3) -> RollingStateV3:
        """Deduplicate intents by normalized label."""
        seen = {}
        unique = []
        for intent in state.intents:
            norm = re.sub(r'[_\s]', '', intent.label.lower())
            if norm not in seen:
                seen[norm] = True
                unique.append(intent)
            else:
                # Keep the one with longer summary (more information)
                for i, u in enumerate(unique):
                    if re.sub(r'[_\s]', '', u.label.lower()) == norm:
                        if len(intent.summary) > len(u.summary):
                            unique[i] = intent
                        break
        state.intents = unique
        return state

    @staticmethod
    def _dedup_obligations(state: RollingStateV3) -> RollingStateV3:
        """Deduplicate obligations by normalized obligor + action key."""
        seen = {}
        unique = []
        for ob in state.obligations:
            key = (
                re.sub(r'[_\s]', '', ob.obligor.lower()),
                re.sub(r'[_\s]', '', ob.action.lower()[:60]),
            )
            if key not in seen:
                seen[key] = True
                unique.append(ob)
            else:
                # Keep the one with more detail (longer action)
                for i, u in enumerate(unique):
                    u_key = (
                        re.sub(r'[_\s]', '', u.obligor.lower()),
                        re.sub(r'[_\s]', '', u.action.lower()[:60]),
                    )
                    if u_key == key and len(ob.action) > len(u.action):
                        unique[i] = ob
                        break
        state.obligations = unique
        return state

    # ── Shared helpers ────────────────────────────────────────────────────

    def _build_state_brief(self, state: RollingStateV3, next_text: str,
                           k: int = 15, token_budget: int = 1200) -> str:
        """Embedding-scored brief from rolling state for next batch context.

        Includes extraction coverage summary so the agent knows its progress
        without needing to call coverage_report() explicitly.
        """
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
            age = max(1, state.iteration - idx // max(1, len(items) // max(1, state.iteration)))
            scores[idx] *= 1.0 + 0.2 * (age ** -0.5)

        sorted_idx = np.argsort(-scores)[:k]

        # Dict lookup instead of O(n) linear scan per result
        field_map = {f.name: f for f in state.fields}
        clause_map = {c.name: c for c in state.clause_types}

        kept_fields, kept_clauses = [], []
        for idx in sorted_idx:
            typ, name, _ = items[idx]
            if typ == "field":
                f = field_map.get(name)
                if f:
                    kept_fields.append({"name": f.name, "type": f.value_type, "desc": f.description[:60]})
            elif typ == "clause_type":
                c = clause_map.get(name)
                if c:
                    kept_clauses.append({"name": c.name, "desc": c.emergent_description[:60]})

        # Extraction coverage summary (avoids agent needing to call coverage_report())
        n_fields = len(state.fields)
        fields_with_ext = len({e.field_name for e in state.extractions})
        chunks_extracted = len({e.chunk_idx for e in state.extractions if e.chunk_idx >= 0})
        coverage_summary = {
            "fields_discovered": n_fields,
            "fields_with_extractions": fields_with_ext,
            "chunks_extracted": chunks_extracted,
            "chunks_total": state.total_chunks,
            "extraction_only_mode": state.extraction_only_mode,
        }

        brief = {
            "already_found": {"fields": kept_fields, "clause_types": kept_clauses},
            "coverage": coverage_summary,
            "still_looking_for": state.open_questions[:5],
            "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
        }
        text = json.dumps(brief, indent=None)
        words = text.split()
        if len(words) > token_budget:
            text = " ".join(words[:token_budget])
        return text

    def _overlap_merge(self, grounded: dict, exploratory: dict,
                       threshold: float | None = None) -> tuple[dict, dict]:
        """Cosine overlap-penalty merge (V2 §4.5)."""
        threshold = threshold or self.config.overlap_threshold
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
                if isinstance(item, str):
                    return item
                if not isinstance(item, dict):
                    return str(item)
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
            all_vecs = self.embedder.encode(
                g_texts + e_texts, normalize_embeddings=True, batch_size=32
            )
            g_vecs = all_vecs[:len(g_texts)]
            e_vecs = all_vecs[len(g_texts):]

            merged = list(g_items)
            for i, e_item in enumerate(e_items):
                sims = np.dot(g_vecs, e_vecs[i])
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                if max_sim < threshold:
                    if isinstance(e_item, dict):
                        e_item["source"] = "exploratory"
                    merged.append(e_item)
                    total_kept += 1
                else:
                    total_dropped += 1

            result[key] = merged

        return result, {"kept": total_kept, "dropped": total_dropped}

    def _update_state_from_merged(self, state: RollingStateV3, merged: dict,
                                  batch_chunks: list[str]):
        """Update rolling state from a merged 4-output dict."""
        existing_names = {f.name for f in state.fields}
        for f_dict in merged.get("fields", []):
            if isinstance(f_dict, str):
                f_dict = {"name": f_dict}
            if not isinstance(f_dict, dict):
                continue
            name = (f_dict.get("name") or "").strip()
            if not name:
                continue
            name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
            if name in existing_names:
                continue
            state.fields.append(FieldV3(
                name=name,
                value_type=f_dict.get("value_type", f_dict.get("type", "string")),
                description=f_dict.get("description", ""),
                span_text=f_dict.get("span_text", ""),
                confidence=float(f_dict.get("confidence", 0.5)),
                source=f_dict.get("source", "grounded"),
                iteration=state.iteration,
            ))
            existing_names.add(name)

        existing_ct = {c.name.lower() for c in state.clause_types}
        for c_dict in merged.get("clause_types", []):
            if isinstance(c_dict, str):
                c_dict = {"name": c_dict}
            if not isinstance(c_dict, dict):
                continue
            name = (c_dict.get("name") or "").strip()
            if not name or name.lower() in existing_ct:
                continue
            state.clause_types.append(ClauseTypeV3(
                name=name,
                emergent_description=c_dict.get("emergent_description", ""),
                span_text=c_dict.get("span_text", ""),
                source=c_dict.get("source", "grounded"),
                iteration=state.iteration,
            ))
            existing_ct.add(name.lower())

        for i_dict in merged.get("intents", []):
            if isinstance(i_dict, str):
                i_dict = {"intent_description": i_dict}
            if not isinstance(i_dict, dict):
                continue
            desc = i_dict.get("intent_description", "").strip()
            if desc:
                state.intents.append(IntentV3(
                    clause_ref=i_dict.get("clause_ref", ""),
                    intent_description=desc,
                    source=i_dict.get("source", "grounded"),
                ))

        for o_dict in merged.get("obligations", []):
            if isinstance(o_dict, str):
                o_dict = {"action": o_dict}
            if not isinstance(o_dict, dict):
                continue
            action = o_dict.get("action", "").strip()
            if action:
                state.obligations.append(ObligationV3(
                    clause_ref=o_dict.get("clause_ref", ""),
                    obligor=o_dict.get("obligor", ""),
                    action=action,
                    trigger_condition=o_dict.get("trigger_condition"),
                    deadline=o_dict.get("deadline"),
                    source=o_dict.get("source", "grounded"),
                ))

        # Anchor terms
        existing_terms = dict(state.anchor_terms)
        for chunk in batch_chunks:
            for term, count in Counter(re.findall(r'\b[a-z]{4,}\b', chunk.lower())).most_common(20):
                existing_terms[term] = existing_terms.get(term, 0) + count
        state.anchor_terms = sorted(existing_terms.items(), key=lambda x: -x[1])[:20]
        state.chunks_seen += len(batch_chunks)

    def _to_field_dicts(self, state: RollingStateV3) -> list[dict]:
        """Convert to extraction-compatible field dicts.

        Passes through ALL metadata so downstream persistence can store it.
        """
        type_map = {"string": "text", "money": "currency", "ref": "text"}
        valid_types = {"text", "date", "duration", "currency", "percentage",
                       "integer", "boolean", "entity_name", "enum"}
        result = []
        for f in state.fields[:self.config.max_fields]:
            ftype = f.value_type.strip().lower()
            ftype = type_map.get(ftype, ftype)
            if ftype not in valid_types:
                ftype = "text"
            # Build examples from span_text + span_texts (deduped)
            examples = []
            if f.span_text:
                examples.append(f.span_text)
            for st in f.span_texts:
                if st and st not in examples:
                    examples.append(st)
            result.append({
                "name": f.name,
                "type": ftype,
                "description": f.description,
                "examples": examples,
                "source": f.source,
                "confidence": f.confidence,
                "span_text": f.span_text,
                "span_texts": f.span_texts,
                "derivation": f.derivation,
                "iteration": f.iteration,
            })
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Module-level entry point
# ══════════════════════════════════════════════════════════════════════════════


def discover_fields_rlm_v3(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: RLMV3Config | None = None,
    lm: dspy.LM | None = None,
    sub_lm: dspy.LM | None = None,
) -> tuple[list[dict], dict]:
    """
    V3 field discovery — drop-in replacement for pipeline integration.

    Returns:
        (fields, meta) — fields compatible with extract_fields_from_chunk()
    """
    v3 = RLMV3(config=config, lm=lm, sub_lm=sub_lm)
    return v3.discover(cluster_label, keywords, chunk_texts)


def discover_and_extract_rlm_v3(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    chunk_ids: list[str] | None = None,
    agreement_ids: list[str] | None = None,
    config: RLMV3Config | None = None,
    lm: dspy.LM | None = None,
    sub_lm: dspy.LM | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    V3 unified discover+extract — single REPL pass discovers fields AND
    extracts values from sampled chunks simultaneously.

    Returns:
        (fields, extractions, meta)
        - fields: list[dict] compatible with extract_fields_from_chunk()
        - extractions: list[dict] with {field_name, value, confidence,
          evidence_span, chunk_id, agreement_id, chunk_idx, source}
        - meta: includes remaining_chunk_indices for lightweight downstream pass
    """
    cfg = config or RLMV3Config()
    cfg.extract_inline = True
    v3 = RLMV3(config=cfg, lm=lm, sub_lm=sub_lm)
    return v3.discover_and_extract(
        cluster_label, keywords, chunk_texts,
        chunk_ids=chunk_ids, agreement_ids=agreement_ids,
    )
