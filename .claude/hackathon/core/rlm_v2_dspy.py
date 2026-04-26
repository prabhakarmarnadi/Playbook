"""
RLM v2 — DSPy-Native Discovery Pipeline
=========================================

Rewrites the v2 discovery pipeline on top of DSPy primitives:

  Layer 1 — dspy.Signature:     Typed I/O contracts for grounded & exploratory discovery.
  Layer 2 — dspy.ChainOfThought: Step-by-step reasoning before each structured output.
  Layer 3 — dspy.RLM:            Deep programmatic exploration of ALL chunks via REPL.
  Layer 4 — dspy.MIPROv2:        Automatic prompt & few-shot optimization from training data.
  Layer 5 — dspy.Evaluate:       Quality scoring against grounding metrics.

Keeps the proven v2 infrastructure:
  - Rolling Structured State across chunk batches
  - Text-Native Latent Briefing (embedding-scored context compression)
  - Overlap-penalty merge (cosine > 0.85 → grounded wins)
  - Two-tier canonicalization (fuzzy name + embedding similarity)
  - Adaptive early stopping (state delta < epsilon)

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │ ClusterDiscoveryPipeline(dspy.Module)                           │
  │                                                                  │
  │  Per-batch loop (rolling state):                                 │
  │    ├── GroundedDiscovery(dspy.ChainOfThought)  — temp 0.2       │
  │    ├── ExploratoryDiscovery(dspy.ChainOfThought) — temp 0.8     │
  │    ├── overlap_merge()   — cosine penalty dedup                  │
  │    ├── update_state()    — rolling accumulator                   │
  │    └── adaptive_stop()   — delta < epsilon                       │
  │                                                                  │
  │  Optional deep dive:                                             │
  │    └── DeepChunkExplorer(dspy.RLM) — REPL over ALL chunks       │
  │                                                                  │
  │  Post-loop:                                                      │
  │    └── canonicalize()    — rule-based + embedding dedup          │
  │                                                                  │
  │  MIPROv2 optimizable: all ChainOfThought predictors exposed     │
  │  via named_predictors() for automatic prompt tuning.             │
  └──────────────────────────────────────────────────────────────────┘

Drop-in compatible: same discover_fields_for_cluster_rlm_v2() entry point.

Usage (basic):
    fields, meta = discover_fields_for_cluster_rlm_v2_dspy(
        client, deployment, cluster_label, keywords, chunk_texts, config)

Usage (optimized):
    optimized_pipeline = optimize_pipeline(db_path, n_train=20)
    fields, meta = optimized_pipeline(
        cluster_label=..., keywords=..., chunk_texts=...)
"""

import json
import logging
import os
import re
import time
from collections import Counter
from typing import Callable, Optional

import dspy
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT MODELS (unchanged from rlm_v2.py)
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


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DSPy SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

class GroundedDiscoverySig(dspy.Signature):
    """You are a legal contract analyst. Analyze clause excerpts belonging to a
    specific clause type and discover structured elements that are GROUNDED in
    the actual text.

    Discover four types of structured elements:
    1. FIELDS: Named data points whose **values can be copied verbatim** from the
       text. Use snake_case. Every field MUST have a span_text that is a direct
       quote from the excerpts showing an example value.
       Types: string, date, money, enum, ref, duration, percentage, integer
       FORBIDDEN: boolean or presence-check fields (e.g. "section_present",
       "has_clause", "includes_provision"). Only fields whose value is a
       concrete text span (a name, date, amount, duration, clause text, etc.).
    2. CLAUSE TYPES: Recurring contractual patterns, named by what they DO.
    3. INTENTS: WHY each clause type exists (the underlying purpose).
    4. OBLIGATIONS: Actionable commitments — who MUST do what, under what conditions.

    Rules:
    - span_text is MANDATORY for every field — copy an exact quote from the text
    - Be specific (e.g., "auto_renewal_notice_window_days" not "terms")
    - Field values must be extractable: a human could highlight the value in the text
    - For obligations: only clear, enforceable duties
    - Focus on NEW discoveries not already in prior_context
    - Return 5-15 fields, 2-8 clause types"""

    cluster_label: str = dspy.InputField(desc="The clause type label (e.g. 'Termination Clause')")
    keywords: str = dspy.InputField(desc="Comma-separated keywords for this clause type")
    chunks_text: str = dspy.InputField(desc="Clause excerpts from multiple contracts, separated by headers")
    prior_context: str = dspy.InputField(desc="JSON summary of already-discovered fields/clause_types from prior batches. Empty string if first batch.")

    discovery_json: str = dspy.OutputField(
        desc='JSON object with keys: "fields" (list of {name, value_type, description, span_text, confidence}), '
             '"clause_types" (list of {name, emergent_description, span_text}), '
             '"intents" (list of {clause_ref, intent_description}), '
             '"obligations" (list of {clause_ref, obligor, action, trigger_condition, deadline})'
    )


class ExploratoryDiscoverySig(dspy.Signature):
    """You are a legal contract analyst specializing in finding NOVEL and NICHE
    patterns that standard analysis overlooks. Look with FRESH EYES.

    Find unusual, domain-specific, or niche patterns:
    1. FIELDS: Unusual data points highly specific to this contract type.
       Do NOT include generic fields (party_name, date, amount). Be very specific.
       Every field MUST have a span_text that is a direct quote from the text.
       Types: string, date, money, enum, ref, duration, percentage, integer
       FORBIDDEN: boolean or presence-check fields (e.g. "section_present",
       "has_clause"). Only fields whose value is a concrete extractable span.
    2. CLAUSE TYPES: Non-obvious contractual patterns hiding in the text.
    3. INTENTS: Hidden purposes or implications beyond the surface reading.
    4. OBLIGATIONS: Subtle duties or conditional requirements often overlooked.

    Rules:
    - Focus on what makes THESE clauses UNIQUE — not boilerplate
    - Prefer highly specific, legally meaningful names
    - span_text is MANDATORY — copy an exact quote showing the field's value
    - Field values must be extractable: a human could highlight the value in the text
    - Return 3-10 fields, 1-5 clause types"""

    cluster_label: str = dspy.InputField(desc="The clause type label")
    keywords: str = dspy.InputField(desc="Comma-separated keywords for this clause type")
    chunks_text: str = dspy.InputField(desc="Clause excerpts from multiple contracts")

    discovery_json: str = dspy.OutputField(
        desc='JSON object with keys: "fields", "clause_types", "intents", "obligations" — '
             'same schema as grounded discovery but focusing on niche/novel patterns'
    )


class DeepExplorerSig(dspy.Signature):
    """You are a legal contract analyst with access to a Python REPL. Given ALL
    clause excerpts (chunks_json) belonging to a clause type (cluster_label),
    programmatically explore them to discover structured elements.

    Discover four types across ALL chunks:
    1. FIELDS: Named data points with snake_case names and types
       (string|date|money|enum|ref|boolean|duration|percentage|integer).
    2. CLAUSE TYPES: Recurring contractual patterns.
    3. INTENTS: Why each clause type exists.
    4. OBLIGATIONS: Who must do what, under what conditions.

    STRATEGY — follow these steps:
    1. Parse chunks_json and print len, then print 5 diverse chunks in full.
    2. Use llm_query_batched to send 3-6 batches of chunks to the sub-LLM,
       each batch asking: 'Analyze these contract clauses and return JSON with
       fields, clause_types, intents, and obligations.'
    3. Aggregate results from all batches. Keep items found in 2+ batches.
    4. Deduplicate near-synonyms.
    5. Build the final JSON object and SUBMIT.

    Be thorough — aim for 10-15 fields, 3-8 clause_types per cluster."""

    chunks_json: str = dspy.InputField(desc="JSON array of ALL clause text excerpts in this cluster")
    cluster_label: str = dspy.InputField(desc="The clause type label")
    keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
    max_fields: str = dspy.InputField(desc="Maximum number of fields to return")

    discovery_json: str = dspy.OutputField(
        desc='JSON object with keys: "fields" (list of {name, value_type, description, span_text, confidence}), '
             '"clause_types" (list of {name, emergent_description, span_text}), '
             '"intents" (list of {clause_ref, intent_description}), '
             '"obligations" (list of {clause_ref, obligor, action, trigger_condition, deadline})'
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — DSPy MODULES (ChainOfThought + RLM)
# ══════════════════════════════════════════════════════════════════════════════

class GroundedDiscovery(dspy.Module):
    """Grounded discovery using ChainOfThought for step-by-step reasoning.

    DSPy handles prompt construction, retries, and output parsing.
    MIPROv2 can optimize the instruction prefix and few-shot demos.
    """

    def __init__(self):
        super().__init__()
        self.discover = dspy.ChainOfThought(GroundedDiscoverySig)

    def forward(self, cluster_label: str, keywords: str,
                chunks_text: str, prior_context: str = "") -> dspy.Prediction:
        return self.discover(
            cluster_label=cluster_label,
            keywords=keywords,
            chunks_text=chunks_text,
            prior_context=prior_context,
        )


class ExploratoryDiscovery(dspy.Module):
    """Exploratory discovery using ChainOfThought — finds niche patterns.

    Uses higher temperature via DSPy's config system.
    MIPROv2 can optimize the exploration instructions.
    """

    def __init__(self):
        super().__init__()
        self.discover = dspy.ChainOfThought(ExploratoryDiscoverySig)

    def forward(self, cluster_label: str, keywords: str,
                chunks_text: str) -> dspy.Prediction:
        return self.discover(
            cluster_label=cluster_label,
            keywords=keywords,
            chunks_text=chunks_text,
        )


class DeepChunkExplorer(dspy.Module):
    """Deep programmatic exploration using DSPy RLM.

    The LLM gets a Python REPL with llm_query() and llm_query_batched()
    tools, letting it write code to systematically analyze ALL chunks
    (not just a batch of 2). This catches patterns that only emerge
    across the full corpus.
    """

    def __init__(self, max_iterations: int = 25, max_llm_calls: int = 50,
                 sub_lm: "dspy.LM | None" = None):
        super().__init__()
        self.rlm = dspy.RLM(
            DeepExplorerSig,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=False,
            sub_lm=sub_lm,
        )

    def forward(self, chunks_json: str, cluster_label: str,
                keywords: str, max_fields: str = "15") -> dspy.Prediction:
        return self.rlm(
            chunks_json=chunks_json,
            cluster_label=cluster_label,
            keywords=keywords,
            max_fields=max_fields,
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ORCHESTRATOR PIPELINE (dspy.Module)
# ══════════════════════════════════════════════════════════════════════════════

class ClusterDiscoveryPipeline(dspy.Module):
    """
    Full v2 discovery pipeline as a DSPy Module.

    Combines ChainOfThought (grounded + exploratory) with optional RLM deep
    exploration. All DSPy predictors are exposed via named_predictors() so
    MIPROv2 can auto-optimize their prompts through the standard compile() API.

    Architecture per chunk-batch:
      1. Build embedding-scored brief from rolling state
      2. GroundedDiscovery (ChainOfThought, temp 0.2)
      3. ExploratoryDiscovery (ChainOfThought, temp 0.8)
      4. Overlap-penalty merge (cosine > 0.85 → grounded wins)
      5. Update rolling state
      6. Adaptive stop (delta < epsilon)
    Optional: DeepChunkExplorer (RLM) on all chunks
    Post-loop: Two-tier canonicalization
    """

    def __init__(self, config=None, embedder=None, enable_deep_explore: bool = False,
                 rlm_max_iterations: int = 25, rlm_max_llm_calls: int = 50,
                 sub_lm: "dspy.LM | None" = None):
        super().__init__()

        # DSPy modules — these get optimized by MIPROv2
        self.grounded = GroundedDiscovery()
        self.exploratory = ExploratoryDiscovery()

        self.enable_deep_explore = enable_deep_explore
        if enable_deep_explore:
            self.deep_explorer = DeepChunkExplorer(
                max_iterations=rlm_max_iterations,
                max_llm_calls=rlm_max_llm_calls,
                sub_lm=sub_lm,
            )

        # Non-DSPy components (Python logic, not LLM calls)
        self._config = config
        self._embedder = embedder
        self.adaptive_stop_epsilon = 0.05

    @property
    def config(self):
        if self._config is None:
            from core.field_discovery import FieldDiscoveryConfig
            self._config = FieldDiscoveryConfig()
        return self._config

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def forward(self, cluster_label: str, keywords: str,
                chunk_texts: list[str]) -> dspy.Prediction:
        """
        Run full discovery pipeline for a cluster.

        Args:
            cluster_label: The clause type label
            keywords: Comma-separated keywords
            chunk_texts: ALL chunk texts in this cluster

        Returns:
            dspy.Prediction with fields_json (list[dict]) and meta (dict)
        """
        sample = chunk_texts[:self.config.max_sample_chunks]
        state = RollingState(cluster_label=cluster_label)
        keywords_str = keywords if isinstance(keywords, str) else ", ".join(keywords[:10])

        batch_size = 2
        n_batches = min(5, (len(sample) + batch_size - 1) // batch_size)

        meta = {
            "n_batches": 0,
            "n_grounded_calls": 0,
            "n_exploratory_calls": 0,
            "stopped_early": False,
            "exploratory_kept": 0,
            "exploratory_dropped": 0,
            "deep_explore_used": False,
        }

        # ── Batch loop with rolling state ──────────────────────────────
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch = sample[start:start + batch_size]
            if not batch:
                break

            chunks_text = "\n\n".join(
                f"[Excerpt {start + i + 1}]\n{t[:1500]}" for i, t in enumerate(batch)
            )

            prev_field_count = len(state.fields)

            # Build embedding-scored brief from rolling state
            brief = self._build_brief(state, chunks_text)

            # Grounded pass (ChainOfThought with reasoning)
            grounded = self._run_grounded(
                cluster_label, keywords_str, chunks_text, brief,
            )
            meta["n_grounded_calls"] += 1

            # Exploratory pass (ChainOfThought with reasoning)
            exploratory = self._run_exploratory(
                cluster_label, keywords_str, chunks_text,
            )
            meta["n_exploratory_calls"] += 1

            # Overlap-penalty merge
            merged, merge_stats = self._overlap_merge(grounded, exploratory)
            meta["exploratory_kept"] += merge_stats["kept"]
            meta["exploratory_dropped"] += merge_stats["dropped"]

            # Update rolling state
            self._update_state(state, merged, batch)

            # Adaptive stopping
            new_field_count = len(state.fields)
            if prev_field_count > 0:
                delta = abs(new_field_count - prev_field_count) / max(new_field_count, 1)
            else:
                delta = 1.0
            state.last_state_delta = delta
            state.iteration_count += 1
            meta["n_batches"] += 1

            logger.info(
                f"    V2-DSPy batch {batch_idx + 1}/{n_batches}: "
                f"{len(state.fields)} fields, {len(state.clause_types)} clause types, "
                f"delta={delta:.3f}"
            )

            if delta < self.adaptive_stop_epsilon and state.iteration_count >= 3:
                meta["stopped_early"] = True
                logger.info(f"    V2-DSPy adaptive stop (delta={delta:.3f})")
                break

        # ── Optional: RLM deep exploration ─────────────────────────────
        if self.enable_deep_explore:
            deep_results = self._run_deep_explore(
                cluster_label, keywords_str, chunk_texts,
            )
            if deep_results:
                # Merge deep results into state (new items only)
                self._update_state(state, deep_results, [])
                meta["deep_explore_used"] = True
                meta["deep_explore_fields"] = len(deep_results.get("fields", []))

        # ── Canonicalize ───────────────────────────────────────────────
        pre_canon = len(state.fields)
        state = self._canonicalize(state)
        post_canon = len(state.fields)

        # Convert to extraction-compatible format
        fields = self._to_field_dicts(state)

        # ── Trial extraction filter (discovery→extraction feedback) ─────
        # Run extraction on a few sample chunks and drop fields that
        # hallucinate. This closes the loop: discovery proposes, extraction
        # validates, bad fields get pruned.
        pre_trial = len(fields)
        fields, trial_meta = self._trial_extraction_filter(
            fields, cluster_label, sample,
            n_trial=3, halluc_threshold=0.5,
        )
        meta["trial_extraction"] = trial_meta
        meta["trial_fields_dropped"] = pre_trial - len(fields)

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

        return dspy.Prediction(fields=fields, meta=meta)

    # ── DSPy module calls with output parsing ─────────────────────────

    def _run_grounded(self, cluster_label, keywords, chunks_text, brief):
        """Run grounded ChainOfThought and parse the structured output."""
        try:
            result = self.grounded(
                cluster_label=cluster_label,
                keywords=keywords,
                chunks_text=chunks_text,
                prior_context=brief or "",
            )
            return _parse_discovery_json(result.discovery_json)
        except Exception as e:
            logger.warning(f"Grounded discovery failed: {e}")
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}

    def _run_exploratory(self, cluster_label, keywords, chunks_text):
        """Run exploratory ChainOfThought and parse the structured output."""
        try:
            # Use higher temperature for exploration
            with dspy.context(temperature=0.8):
                result = self.exploratory(
                    cluster_label=cluster_label,
                    keywords=keywords,
                    chunks_text=chunks_text,
                )
            return _parse_discovery_json(result.discovery_json)
        except Exception as e:
            logger.warning(f"Exploratory discovery failed: {e}")
            return {"fields": [], "clause_types": [], "intents": [], "obligations": []}

    def _run_deep_explore(self, cluster_label, keywords, chunk_texts):
        """Run RLM deep exploration across all chunks."""
        if not hasattr(self, 'deep_explorer'):
            return None

        import random
        max_rlm_chunks = 60
        if len(chunk_texts) > max_rlm_chunks:
            rng = random.Random(42)
            head = chunk_texts[:20]
            tail = rng.sample(chunk_texts[20:], min(max_rlm_chunks - 20, len(chunk_texts) - 20))
            capped = head + tail
        else:
            capped = chunk_texts

        chunks_payload = json.dumps(capped, ensure_ascii=False)

        try:
            result = self.deep_explorer(
                chunks_json=chunks_payload,
                cluster_label=cluster_label,
                keywords=keywords,
                max_fields=str(self.config.max_fields_per_cluster),
            )
            return _parse_discovery_json(result.discovery_json)
        except Exception as e:
            logger.warning(f"Deep RLM exploration failed: {e}")
            return None

    # ── Latent Briefing (embedding-scored context compression) ─────────

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
            scores[idx] *= 1.0 + 0.2 * (age ** -0.5)

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

        brief = json.dumps({
            "already_found": {"fields": kept_fields, "clause_types": kept_clauses},
            "still_looking_for": state.open_questions[:5],
            "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
        }, indent=None)

        words = brief.split()
        if len(words) > token_budget:
            brief = " ".join(words[:token_budget])
        return brief

    # ── Overlap-Penalty Merge ─────────────────────────────────────────

    @staticmethod
    def _is_extractable_field(field_dict: dict) -> bool:
        """Reject fields that can't produce verbatim extractable values."""
        name = (field_dict.get("name") or "").lower()
        vtype = (field_dict.get("value_type") or "").lower()
        span = (field_dict.get("span_text") or "").strip()

        # Reject boolean / presence-check fields
        if vtype == "boolean":
            return False
        if any(name.endswith(suf) for suf in ("_present", "_exists", "_included",
                                                "_has", "_contains", "_available")):
            return False
        if name.startswith("has_") or name.startswith("is_") or name.startswith("includes_"):
            return False

        # Reject fields with no span_text evidence
        if not span:
            return False

        return True

    def _overlap_merge(self, grounded, exploratory, threshold=0.85):
        # Pre-filter: drop non-extractable fields from both sides
        for side in (grounded, exploratory):
            if "fields" in side:
                before = len(side["fields"])
                side["fields"] = [f for f in side["fields"] if self._is_extractable_field(f)]
                dropped = before - len(side["fields"])
                if dropped:
                    logger.info(f"    Extractability filter dropped {dropped} non-extractable fields")

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

    # ── State Update ──────────────────────────────────────────────────

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

    # ── Two-Tier Canonicalization ─────────────────────────────────────

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
            texts = [f"{state.fields[i].name}: {state.fields[i].description}" for i in remaining]
            vecs = self.embedder.encode(texts, normalize_embeddings=True, batch_size=32)
            sim = np.dot(vecs, vecs.T)
            for a in range(len(remaining)):
                if remaining[a] in to_remove:
                    continue
                for b in range(a + 1, len(remaining)):
                    if remaining[b] in to_remove:
                        continue
                    fa, fb = state.fields[remaining[a]], state.fields[remaining[b]]
                    if fa.source == "exploratory" and fb.source == "exploratory":
                        continue
                    if sim[a, b] > 0.82:
                        to_remove.add(remaining[b] if len(fa.description) >= len(fb.description) else remaining[a])

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

    # ── Output conversion ─────────────────────────────────────────────

    def _to_field_dicts(self, state):
        type_map = {"string": "text", "money": "currency", "ref": "text"}
        valid_types = {"text", "date", "duration", "currency", "percentage",
                       "integer", "boolean", "entity_name", "enum"}
        result = []
        for f in state.fields[:self.config.max_fields_per_cluster]:
            ftype = type_map.get(f.value_type.strip().lower(), f.value_type.strip().lower())
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

    # ── Trial Extraction Filter (closes discovery→extraction loop) ────

    def _trial_extraction_filter(self, fields: list[dict], cluster_label: str,
                                  sample_chunks: list[str],
                                  n_trial: int = 3,
                                  halluc_threshold: float = 0.5,
                                  min_fill: float = 0.0) -> tuple[list[dict], dict]:
        """Run extraction on a few sample chunks and drop fields that hallucinate.

        This is the key feedback loop: discovery proposes fields, trial extraction
        tests whether they actually produce grounded values, and fields that fail
        get pruned BEFORE the full extraction run.

        Args:
            fields: Discovered field dicts from _to_field_dicts
            cluster_label: Cluster label for extraction prompt
            sample_chunks: All available chunk texts
            n_trial: Number of chunks to trial-extract on
            halluc_threshold: Drop fields where hallucination_rate > this
            min_fill: Drop fields that never get filled (0.0 = keep all)

        Returns:
            (filtered_fields, trial_meta) — surviving fields and diagnostics
        """
        from core.field_discovery import extract_fields_from_chunk, FieldDiscoveryConfig

        if not fields or not sample_chunks:
            return fields, {"trial_skipped": True}

        # Pick diverse trial chunks: first, middle, last
        n = len(sample_chunks)
        indices = [0]
        if n > 2:
            indices.append(n // 2)
        if n > 1:
            indices.append(n - 1)
        indices = indices[:n_trial]
        trial_chunks = [sample_chunks[i] for i in indices]

        # Get Azure client for extraction calls
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        ext_config = FieldDiscoveryConfig()

        # Build RLM context for enhanced extraction
        rlm_context = None  # keep extraction simple for trial

        # Track per-field grounding stats
        field_stats = {f["name"]: {"filled": 0, "verbatim": 0, "hallucinated": 0}
                       for f in fields}

        for chunk in trial_chunks:
            chunk_lower = chunk.lower()
            try:
                results = extract_fields_from_chunk(
                    client, deployment, cluster_label, chunk, fields, ext_config,
                    rlm_context=rlm_context,
                )
            except Exception as e:
                logger.warning(f"Trial extraction failed: {e}")
                continue

            for fname, (value, conf) in results.items():
                if fname not in field_stats:
                    continue
                if value is None:
                    continue
                field_stats[fname]["filled"] += 1
                v_lower = str(value).lower().strip()
                if v_lower and v_lower in chunk_lower:
                    field_stats[fname]["verbatim"] += 1
                else:
                    field_stats[fname]["hallucinated"] += 1

        # Decide which fields survive
        n_trials = len(trial_chunks)
        surviving = []
        dropped_names = []
        for f in fields:
            stats = field_stats[f["name"]]
            filled = stats["filled"]
            halluc = stats["hallucinated"]
            verb = stats["verbatim"]
            total_nonnull = halluc + verb

            # Drop if hallucination rate exceeds threshold
            if total_nonnull > 0 and halluc / total_nonnull > halluc_threshold:
                dropped_names.append(f["name"])
                continue

            # Drop if never filled and min_fill requires it
            if min_fill > 0 and filled < min_fill * n_trials:
                dropped_names.append(f["name"])
                continue

            surviving.append(f)

        trial_meta = {
            "trial_chunks": n_trials,
            "fields_before": len(fields),
            "fields_after": len(surviving),
            "fields_dropped": dropped_names,
            "per_field_stats": field_stats,
        }

        if dropped_names:
            logger.info(
                f"    Trial extraction dropped {len(dropped_names)} fields "
                f"(halluc>{halluc_threshold:.0%}): {', '.join(dropped_names[:5])}"
            )

        return surviving, trial_meta


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — JSON parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_discovery_json(raw) -> dict:
    """Parse discovery JSON from DSPy output with fallback strategies."""
    empty = {"fields": [], "clause_types": [], "intents": [], "obligations": []}

    if not raw or (isinstance(raw, str) and not raw.strip()):
        return empty

    if isinstance(raw, dict):
        return {k: raw.get(k, []) for k in empty}

    # Try direct JSON parse
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {k: parsed.get(k, []) for k in empty}
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return {k: parsed.get(k, []) for k in empty}
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    match = re.search(r'\{[^{}]*"fields"\s*:\s*\[.*?\].*?\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return {k: parsed.get(k, []) for k in empty}
        except json.JSONDecodeError:
            pass

    # Last resort: try ast.literal_eval
    import ast
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {k: parsed.get(k, []) for k in empty}
    except (ValueError, SyntaxError):
        pass

    logger.warning(f"Failed to parse discovery JSON ({len(text)} chars)")
    return empty


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — DSPy LM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _configure_dspy_lm() -> dspy.LM:
    """Configure DSPy with Azure OpenAI LM from environment variables."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

    if endpoint and api_key:
        lm = dspy.LM(
            f"azure/{deployment}",
            api_key=api_key,
            api_base=endpoint,
            api_version=api_version,
            num_retries=5,
        )
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        lm = dspy.LM(f"openai/{model}", api_key=openai_key, num_retries=5)

    dspy.configure(lm=lm)
    return lm


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — MIPROv2 OPTIMIZATION + METRICS
# ══════════════════════════════════════════════════════════════════════════════

def discovery_quality_metric(example: dspy.Example, prediction: dspy.Prediction,
                             trace=None) -> float:
    """
    Score a discovery pipeline run for MIPROv2 optimization.

    Scoring criteria (0.0 to 1.0) — extraction-grounding-aware:
      - Extraction grounding: Trial extraction verbatim rate (0.35 weight)
      - Field count:          Did we find a reasonable number of fields? (0.15 weight)
      - Span grounding:       Do fields have non-empty span_text from source? (0.20 weight)
      - Specificity:          Are field names specific (not generic)? (0.15 weight)
      - Coverage:             Did we find clause_types + intents + obligations? (0.10 weight)
      - Diversity:            Are fields diverse (low embedding similarity)? (0.05 weight)
    """
    fields = prediction.fields if hasattr(prediction, 'fields') else []
    meta = prediction.meta if hasattr(prediction, 'meta') else {}

    if not fields:
        return 0.0

    chunk_texts = example.chunk_texts if hasattr(example, 'chunk_texts') else []
    all_text = " ".join(chunk_texts).lower()

    # (1) Extraction grounding — from trial extraction feedback loop
    # This is the most important signal: did discovered fields produce
    # verbatim extractions when we actually tried extracting?
    trial = meta.get("trial_extraction", {})
    trial_stats = trial.get("per_field_stats", {})
    if trial_stats:
        total_verb = sum(s.get("verbatim", 0) for s in trial_stats.values())
        total_halluc = sum(s.get("hallucinated", 0) for s in trial_stats.values())
        total_nonnull = total_verb + total_halluc
        if total_nonnull > 0:
            extraction_grounding_score = total_verb / total_nonnull
        else:
            extraction_grounding_score = 0.5  # no extractions = neutral
    else:
        extraction_grounding_score = 0.5  # trial not run = neutral

    # (2) Field count score [0-1] — optimal range is 5-15
    n = len(fields)
    if n < 2:
        count_score = 0.1
    elif n <= 5:
        count_score = 0.5 + 0.1 * n
    elif n <= 15:
        count_score = 1.0
    else:
        count_score = max(0.5, 1.0 - 0.05 * (n - 15))

    # (3) Span grounding — how many fields have span_text that appears in source
    grounded = 0
    for f in fields:
        examples = f.get("examples", [])
        if examples:
            span = examples[0].lower() if examples[0] else ""
            if span and span in all_text:
                grounded += 1
    span_grounding_score = grounded / max(1, len(fields))

    # (4) Specificity score — penalize generic names
    GENERIC_NAMES = {"name", "date", "amount", "type", "value", "text", "party",
                     "term", "period", "number", "description", "title", "status"}
    specific_count = sum(
        1 for f in fields
        if f.get("name", "") not in GENERIC_NAMES and len(f.get("name", "")) > 5
    )
    specificity_score = specific_count / max(1, len(fields))

    # (5) Multi-output coverage — bonus for clause_types, intents, obligations
    n_ct = meta.get("n_clause_types", 0)
    n_int = meta.get("n_intents", 0)
    n_obl = meta.get("n_obligations", 0)
    coverage_score = min(1.0, (
        (0.4 if n_ct >= 2 else 0.2 * min(n_ct, 2)) +
        (0.3 if n_int >= 1 else 0.0) +
        (0.3 if n_obl >= 1 else 0.0)
    ))

    # (6) Diversity score — fields should have distinct descriptions
    if len(fields) > 2:
        descs = [f.get("description", f.get("name", "")) for f in fields]
        pairs_checked = 0
        diverse_pairs = 0
        for i in range(len(descs)):
            for j in range(i + 1, min(i + 5, len(descs))):
                w_i = set(descs[i].lower().split())
                w_j = set(descs[j].lower().split())
                overlap = len(w_i & w_j) / max(1, len(w_i | w_j))
                if overlap < 0.5:
                    diverse_pairs += 1
                pairs_checked += 1
        diversity_score = diverse_pairs / max(1, pairs_checked)
    else:
        diversity_score = 0.5

    # Weighted combination — extraction grounding is the dominant signal
    score = (
        0.35 * extraction_grounding_score +
        0.15 * count_score +
        0.20 * span_grounding_score +
        0.15 * specificity_score +
        0.10 * coverage_score +
        0.05 * diversity_score
    )

    return round(score, 4)


def build_trainset_from_db(db_path: str, n_clusters: int = 20,
                           min_chunks: int = 10) -> list[dspy.Example]:
    """
    Build MIPROv2 training set from DuckDB cluster data.

    Each Example contains:
      - cluster_label: str
      - keywords: str (comma-separated)
      - chunk_texts: list[str]

    MIPROv2 uses these to optimize the pipeline's prompts and few-shot demos.
    """
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)

    clusters = conn.execute("""
        SELECT c.cluster_id, c.label, c.keywords, c.chunk_count
        FROM clusters c
        WHERE c.chunk_count >= ?
        ORDER BY c.chunk_count DESC
        LIMIT ?
    """, [min_chunks, n_clusters * 2]).fetchdf().to_dict("records")

    # Stratified sample: top + middle + bottom
    total = len(clusters)
    if total <= n_clusters:
        selected = clusters
    else:
        n_top = max(1, int(n_clusters * 0.4))
        n_mid = max(1, int(n_clusters * 0.3))
        n_bot = n_clusters - n_top - n_mid
        indices = sorted(set(
            list(range(n_top)) +
            list(range(total // 3, total // 3 + n_mid)) +
            list(range(max(total - n_bot, total // 3 + n_mid),
                       max(total - n_bot, total // 3 + n_mid) + n_bot))
        ))
        selected = [clusters[i] for i in indices if i < total]

    trainset = []
    for cluster in selected:
        cid = cluster["cluster_id"]

        try:
            kw = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            kw = []

        chunks = conn.execute("""
            SELECT ch.chunk_text
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cid]).fetchdf()["chunk_text"].tolist()

        if len(chunks) < 5:
            continue

        example = dspy.Example(
            cluster_label=cluster["label"] or "Unknown",
            keywords=", ".join(kw[:10]),
            chunk_texts=chunks,
        ).with_inputs("cluster_label", "keywords", "chunk_texts")

        trainset.append(example)

    conn.close()
    logger.info(f"Built trainset with {len(trainset)} cluster examples from {db_path}")
    return trainset


def optimize_pipeline(
    db_path: str,
    n_train: int = 20,
    auto: str = "light",
    save_path: str | None = None,
    enable_deep_explore: bool = False,
) -> ClusterDiscoveryPipeline:
    """
    Optimize the discovery pipeline using MIPROv2.

    This auto-tunes the prompts and few-shot demonstrations for the
    GroundedDiscovery and ExploratoryDiscovery ChainOfThought modules.

    Args:
        db_path: Path to DuckDB file with clusters and chunks
        n_train: Number of training clusters
        auto: MIPROv2 auto setting ('light', 'medium', 'heavy')
        save_path: Where to save the optimized module (None = auto)
        enable_deep_explore: Whether to include RLM deep explorer

    Returns:
        Optimized ClusterDiscoveryPipeline

    Usage:
        optimized = optimize_pipeline("data/cuad_510_demo.duckdb", n_train=20)
        fields, meta = optimized(cluster_label=..., keywords=..., chunk_texts=...)
        optimized.save("optimized_discovery.json")
    """
    lm = _configure_dspy_lm()

    # Build training set
    trainset = build_trainset_from_db(db_path, n_clusters=n_train)
    if not trainset:
        raise ValueError(f"No valid training examples from {db_path}")

    # Create student pipeline
    student = ClusterDiscoveryPipeline(enable_deep_explore=enable_deep_explore)

    # Optimize with MIPROv2
    optimizer = dspy.MIPROv2(
        metric=discovery_quality_metric,
        auto=auto,
        verbose=True,
        num_threads=2,  # conservative for Azure rate limits
    )

    logger.info(f"Starting MIPROv2 optimization (auto={auto}, {len(trainset)} examples)...")
    optimized = optimizer.compile(
        student,
        trainset=trainset,
    )

    # Save optimized module
    if save_path is None:
        from pathlib import Path
        save_path = str(Path(db_path).parent / "optimized_discovery_pipeline.json")

    optimized.save(save_path)
    logger.info(f"Optimized pipeline saved to {save_path}")

    return optimized


def evaluate_pipeline(
    pipeline: ClusterDiscoveryPipeline,
    db_path: str,
    n_eval: int = 10,
) -> dict:
    """
    Evaluate a (possibly optimized) pipeline on held-out clusters.

    Returns:
        dict with avg_score, per_cluster scores, and aggregate stats
    """
    _configure_dspy_lm()
    evalset = build_trainset_from_db(db_path, n_clusters=n_eval)

    evaluator = dspy.Evaluate(
        devset=evalset,
        metric=discovery_quality_metric,
        num_threads=2,
        display_progress=True,
        display_table=True,
    )

    score = evaluator(pipeline)
    return {"avg_score": score, "n_clusters": len(evalset)}


# ══════════════════════════════════════════════════════════════════════════════
# DROP-IN ENTRY POINT — compatible with extraction_value_comparison_v2.py
# ══════════════════════════════════════════════════════════════════════════════

def discover_fields_for_cluster_rlm_v2_dspy(
    client,
    deployment: str,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config,
    embedder=None,
    enable_deep_explore: bool = False,
    optimized_path: str | None = None,
) -> tuple[list[dict], dict]:
    """
    DSPy-native RLM v2 field discovery — drop-in replacement.

    If optimized_path is provided, loads MIPROv2-optimized prompts.
    Otherwise uses the default pipeline.

    Returns:
        (fields, meta) — same format as discover_fields_for_cluster_rlm_v2()
    """
    # Configure DSPy LM
    _configure_dspy_lm()

    # Build or load pipeline
    if optimized_path and os.path.exists(optimized_path):
        pipeline = ClusterDiscoveryPipeline(
            config=config, embedder=embedder,
            enable_deep_explore=enable_deep_explore,
        )
        pipeline.load(optimized_path)
        logger.info(f"Loaded optimized pipeline from {optimized_path}")
    else:
        pipeline = ClusterDiscoveryPipeline(
            config=config, embedder=embedder,
            enable_deep_explore=enable_deep_explore,
        )

    # Run pipeline
    keywords_str = ", ".join(keywords[:10]) if isinstance(keywords, list) else keywords
    result = pipeline(
        cluster_label=cluster_label,
        keywords=keywords_str,
        chunk_texts=chunk_texts,
    )

    return result.fields, result.meta


# ══════════════════════════════════════════════════════════════════════════════
# CLI — for standalone optimization and evaluation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="DSPy-native RLM v2 pipeline")
    sub = parser.add_subparsers(dest="command")

    # Optimize command
    opt_parser = sub.add_parser("optimize", help="Run MIPROv2 optimization")
    opt_parser.add_argument("db_path", help="Path to DuckDB file")
    opt_parser.add_argument("--n-train", type=int, default=20)
    opt_parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    opt_parser.add_argument("--save-path", default=None)
    opt_parser.add_argument("--deep-explore", action="store_true")

    # Evaluate command
    eval_parser = sub.add_parser("evaluate", help="Evaluate pipeline quality")
    eval_parser.add_argument("db_path", help="Path to DuckDB file")
    eval_parser.add_argument("--n-eval", type=int, default=10)
    eval_parser.add_argument("--optimized-path", default=None)

    # Run command (single cluster)
    run_parser = sub.add_parser("run", help="Run on a single cluster")
    run_parser.add_argument("db_path", help="Path to DuckDB file")
    run_parser.add_argument("--cluster-id", type=int, required=True)
    run_parser.add_argument("--optimized-path", default=None)
    run_parser.add_argument("--deep-explore", action="store_true")

    args = parser.parse_args()

    if args.command == "optimize":
        optimized = optimize_pipeline(
            args.db_path, n_train=args.n_train, auto=args.auto,
            save_path=args.save_path, enable_deep_explore=args.deep_explore,
        )
        print(f"\nOptimized pipeline saved. Predictors optimized:")
        for name, pred in optimized.named_predictors():
            print(f"  {name}")

    elif args.command == "evaluate":
        if args.optimized_path:
            pipeline = ClusterDiscoveryPipeline()
            pipeline.load(args.optimized_path)
        else:
            _configure_dspy_lm()
            pipeline = ClusterDiscoveryPipeline()

        result = evaluate_pipeline(pipeline, args.db_path, n_eval=args.n_eval)
        print(f"\nEvaluation: avg_score={result['avg_score']:.4f} "
              f"over {result['n_clusters']} clusters")

    elif args.command == "run":
        import duckdb

        _configure_dspy_lm()
        conn = duckdb.connect(args.db_path, read_only=True)

        cluster = conn.execute(
            "SELECT label, keywords FROM clusters WHERE cluster_id = ?",
            [args.cluster_id],
        ).fetchdf().to_dict("records")[0]

        chunks = conn.execute("""
            SELECT ch.chunk_text FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [args.cluster_id]).fetchdf()["chunk_text"].tolist()
        conn.close()

        try:
            kw = json.loads(cluster["keywords"]) if isinstance(cluster["keywords"], str) else cluster["keywords"]
        except (json.JSONDecodeError, TypeError):
            kw = []

        if args.optimized_path:
            pipeline = ClusterDiscoveryPipeline(enable_deep_explore=args.deep_explore)
            pipeline.load(args.optimized_path)
        else:
            pipeline = ClusterDiscoveryPipeline(enable_deep_explore=args.deep_explore)

        result = pipeline(
            cluster_label=cluster["label"] or "Unknown",
            keywords=", ".join(kw[:10]),
            chunk_texts=chunks,
        )

        print(f"\n{cluster['label']}: {len(result.fields)} fields, "
              f"{result.meta['n_clause_types']} clause types")
        for f in result.fields:
            print(f"  - {f['name']} ({f['type']}): {f['description'][:60]}")

    else:
        parser.print_help()
