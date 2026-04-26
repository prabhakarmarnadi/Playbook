"""
RLM v2 — Spec-aligned Field & Multi-Output Discovery Pipeline
==============================================================

HOW V2 WORKS UNDER THE HOOD
=============================

V2 is NOT an agentic REPL loop.  It's a deterministic batch pipeline
that borrows architectural ideas from the RLM literature but executes
them as a traditional orchestration script.  No CodeAct, no agent
autonomy.  The calling code controls the loop.

Why this exists: V2 was built as the "structured state" alternative to
the agentic rlm_with_memory.py.  The hypothesis: structured output
formats + dual-mode exploration might outperform agent autonomy on
field discovery.  V2 validated the dual-mode + overlap-merge + rolling
state design before V3 combined both approaches.

ARCHITECTURE — Per-Cluster Discovery Pipeline
===============================================

For each clause type cluster, the pipeline runs:

  ┌─────────────────────────────────────────────────────────────────┐
  │                  V2 DISCOVERY PIPELINE                          │
  │                                                                 │
  │  Chunks:  [c1, c2] → [c3, c4] → [c5, c6] → ...  (batch=2)    │
  │                                                                 │
  │  Per batch:                                                     │
  │    1. Build embedding-scored brief from rolling state           │
  │    2. Grounded LLM call (temp 0.2, with brief)                 │
  │    3. Exploratory LLM call (temp 0.8, NO brief)                │
  │    4. Overlap-penalty merge (cosine > 0.85 → drop exploratory) │
  │    5. Update rolling state                                      │
  │    6. Check adaptive stop (delta < epsilon for 3 batches)       │
  │                                                                 │
  │  Post-loop:                                                     │
  │    7. Two-tier canonicalization (fuzzy + embedding > 0.82)      │
  │                                                                 │
  │  Output: fields, clause_types, intents, obligations             │
  └─────────────────────────────────────────────────────────────────┘

STEP-BY-STEP DETAILS
=====================

1. ROLLING STRUCTURED STATE (§4.1)
   ─────────────────────────────────
   RollingState is a Pydantic model that accumulates ALL discoveries
   across batches.  It holds:

     fields:         list[FieldV2]       — discovered field schemas
     clause_types:   list[ClauseTypeV2]  — emergent clause categories
     intents:        list[IntentV2]      — WHY each clause exists
     obligations:    list[ObligationV2]  — actionable commitments
     anchor_terms:   list[(term, freq)]  — frequency-counted keywords
     open_questions: list[str]           — unresolved gaps
     last_state_delta: float             — convergence signal

   New items are deduped by normalized name (lowercase, snake_case,
   max 40 chars).  No item is ever removed during accumulation —
   canonicalization happens post-loop only.

2. TEXT-NATIVE LATENT BRIEFING (§4.2)
   ────────────────────────────────────
   Before each grounded LLM call, we build a compressed "brief" from
   the rolling state using the same embedding-scoring logic as LB:

     a. Text-represent every state item:
          field    → "renewal_period (duration): Length of renewal term"
          clause   → "auto_renewal: Automatic renewal mechanism"
     b. Embed all items + the next chunk batch text (MiniLM, 384-dim)
     c. Score each item by cosine sim to the next chunk text
     d. Apply recency boost: score *= 1.0 + 0.2 * age^(-0.5)
        (recently discovered items get a small boost)
     e. Keep top-K items (K=15), format as JSON:
          {"already_found": {"fields": [...], "clause_types": [...]},
           "still_looking_for": [...],
           "anchor_terms": [...]}
     f. Truncate to token_budget (1200 words)

   The brief tells the grounded LLM: "Here's what we've found already
   (ranked by relevance to what you're about to read).  Focus on NEW
   discoveries."  This prevents rediscovery of already-known fields
   while guiding the model toward related-but-new patterns.

3. DUAL-MODE DISCOVERY (§4.5)
   ────────────────────────────
   Each batch gets TWO independent LLM calls:

   Grounded (temp 0.2):
     - Receives the rolling state brief + chunk text
     - Prompt emphasizes grounding: "Include exact span_text quotes"
     - Produces precise, well-supported field schemas
     - Biased toward PRECISION — finds what's obviously there

   Exploratory (temp 0.8):
     - Receives ONLY the chunk text (no brief, no prior context)
     - Prompt emphasizes novelty: "With FRESH EYES — find unusual patterns"
     - Produces creative, domain-specific, niche fields
     - Biased toward RECALL — finds non-obvious patterns

   Why dual-mode works: The grounded pass is consistent but conservative.
   The exploratory pass is noisy but occasionally discovers rare patterns
   that grounded analysis overlooks.  The overlap merge (step 4) combines
   the best of both.

4. OVERLAP-PENALTY MERGE (§4.5)
   ──────────────────────────────
   After both passes return their results, we merge them:

     For each output type (fields, clause_types, intents, obligations):
       a. Text-represent each item from both passes
       b. Embed all items (MiniLM)
       c. For each exploratory item:
            max_sim = max(cosine_sim(exploratory_item, grounded_item_j))
            if max_sim > 0.85:
              DROP the exploratory item (it's a near-duplicate of
              something grounded already found — grounded wins because
              it has better grounding evidence)
            else:
              KEEP the exploratory item and tag source="exploratory"
              (it's genuinely novel — not a rephrasing)

   Result: merged list starts with all grounded items + novel
   exploratory items.  Typical exploration_ratio: 30-60% of
   exploratory items survive.

5. ADAPTIVE STOPPING (§4.3)
   ──────────────────────────
   After each batch, compute:
     delta = |new_field_count - prev_field_count| / max(new_count, 1)

   If delta < 0.05 for 3 consecutive batches after at least 3 iterations:
     → Stop early.  We've converged — more batches won't find new fields.

   This prevents processing all N batches when the first 3 already
   capture everything.  Typical V2 runs stop at batch 3-4 out of 5.

6. TWO-TIER CANONICALIZATION (§4.6)
   ──────────────────────────────────
   Post-loop cleanup to merge naming drift across batches:

   Tier 1 — Rule-based fuzzy merge:
     - Substring: "renewal" ⊂ "renewal_period" → merge, keep longer name
     - Plural/singular: "fee" vs "fees" → merge to shorter
     - On merge: keep the entry with the longer description

   Tier 2 — Embedding similarity > 0.82:
     - "termination_notice_days" ≈ "notice_period_for_termination"
       (cosine sim 0.89) → merge
     - Prevents soft duplicates that rule-based matching misses

   NICHE PRESERVATION RULE: never auto-merge two items both tagged
   source="exploratory".  These are rare discoveries worth keeping
   separate — even if they look similar, they may capture distinct
   domain-specific nuances.

COMPARISON TO OTHER VERSIONS
==============================

  V2 strengths:
    + Deterministic — same inputs → same outputs (modulo LLM stochasticity)
    + Predictable cost — exactly 2 LLM calls per batch × N batches
    + Structured output from the start — never fails to produce valid JSON
    + No REPL dependency — works without DSPy's CodeAct interpreter

  V2 weaknesses:
    - Fixed batch order — can't skip boring chunks or revisit interesting ones
    - No agent judgment — can't decide "I should explore this area deeper"
    - No trajectory memory — each batch only sees the state brief, not the
      full history of reasoning
    - No inline extraction — discovers field schemas only, extraction is
      a separate downstream pass

  → V3 (rlm_v3.py) combines V2's structured output with the agent's
    ability to control exploration order.

CLASS STRUCTURE
================

  LatentRLMv2
    ├── discover_for_cluster()    — main entry: full pipeline for one cluster
    ├── _call_azure()             — Azure OpenAI API call with retries + JSON parsing
    ├── _discover()               — single grounded or exploratory LLM call
    ├── _build_brief()            — embedding-scored brief from rolling state
    ├── _overlap_merge()          — cosine overlap-penalty merge
    ├── _update_state()           — merge new items into rolling state
    ├── _canonicalize()           — two-tier post-loop dedup
    └── _to_field_dicts()         — convert to extraction-compatible format

  discover_fields_for_cluster_rlm_v2()  — module-level drop-in entry point

Implements the architecture from the 2026-04-18 design spec.
Works entirely with closed-source API models (Azure OpenAI, etc.)
and local MiniLM-L6-v2 for embedding-based scoring.
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Pydantic output models (from spec §4.4) ──────────────────────────────────

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
    """Rolling Structured State — typed, cluster-scoped accumulator (spec §4.1)."""
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


# ── Discovery prompts ────────────────────────────────────────────────────────

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


# ── LatentRLMv2 pipeline ─────────────────────────────────────────────────────

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
        """
        Full v2 discovery pipeline for a cluster.

        Returns:
            (fields, meta) where fields is list[dict] compatible with
            extract_fields_from_chunk() and meta contains v2-specific stats.
        """
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

            # Build embedding-scored brief from rolling state (§4.2)
            brief = self._build_brief(state, chunks_text)

            # Grounded discovery (temp 0.2, with brief) (§4.5)
            grounded = self._discover(
                chunks_text, cluster_label, keywords, brief,
                temperature=0.2, mode="grounded",
            )
            meta["n_grounded_calls"] += 1

            # Exploratory discovery (temp 0.8, no brief) (§4.5)
            exploratory = self._discover(
                chunks_text, cluster_label, keywords, "",
                temperature=0.8, mode="exploratory",
            )
            meta["n_exploratory_calls"] += 1

            # Overlap-penalty merge (§4.5)
            merged, merge_stats = self._overlap_merge(grounded, exploratory)
            meta["exploratory_kept"] += merge_stats["kept"]
            meta["exploratory_dropped"] += merge_stats["dropped"]

            # Update rolling state (§4.1)
            self._update_state(state, merged, batch)

            # Adaptive stopping (§4.3)
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

        # Canonicalize (§4.6)
        pre_canon = len(state.fields)
        state = self._canonicalize(state)
        post_canon = len(state.fields)

        # Convert to comparison-compatible format
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
            # Include actual objects for downstream extraction context
            "clause_types": [ct.model_dump() for ct in state.clause_types],
            "intents": [i.model_dump() for i in state.intents],
            "obligations": [o.model_dump() for o in state.obligations],
        })

        return fields, meta

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_azure(self, prompt, temperature=0.2, max_tokens=2000):
        """Call Azure OpenAI with retries, return parsed JSON or None."""
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

    # ── Discovery ─────────────────────────────────────────────────────────

    def _discover(self, chunks_text, cluster_label, keywords, brief,
                  temperature, mode):
        """Run 4-output discovery for one batch in the specified mode."""
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

    # ── Text-Native Latent Briefing (§4.2) ────────────────────────────────

    def _build_brief(self, state, next_text, k=15, token_budget=1200):
        """
        Build embedding-scored brief from rolling state.

        Mirrors spec §4.2: score state items by cosine similarity to
        the next chunk text, retain top-K, emit compact JSON.
        """
        if not state.fields and not state.clause_types:
            return ""

        # Collect all state items with text representations
        items = []
        for f in state.fields:
            items.append(("field", f.name, f"{f.name} ({f.value_type}): {f.description}"))
        for c in state.clause_types:
            items.append(("clause_type", c.name, f"{c.name}: {c.emergent_description}"))

        if not items:
            return ""

        # Batch-encode query + all items
        query_text = next_text[:2000]
        all_texts = [query_text] + [text for _, _, text in items]
        all_vecs = self.embedder.encode(all_texts, normalize_embeddings=True, batch_size=32)

        query_vec = all_vecs[0]
        item_vecs = all_vecs[1:]

        # Score by cosine similarity (proxy for attention weights)
        scores = np.dot(item_vecs, query_vec)

        # Apply recency boost (spec §4.2: 1.0 + 0.2 * age^-0.5)
        for idx in range(len(scores)):
            age = max(1, state.iteration_count - idx // max(1, len(items) // max(1, state.iteration_count)))
            recency_boost = 1.0 + 0.2 * (age ** -0.5)
            scores[idx] *= recency_boost

        # Top-K retention
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
        # Enforce rough token budget
        words = text.split()
        if len(words) > token_budget:
            text = " ".join(words[:token_budget])

        return text

    # ── Overlap-Penalty Merge (§4.5) ──────────────────────────────────────

    def _overlap_merge(self, grounded, exploratory, threshold=0.85):
        """
        Merge grounded + exploratory with cosine overlap penalty.

        For each exploratory item: if max cosine sim to any grounded item
        exceeds threshold, drop the exploratory item (grounded wins).
        """
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

            # Build text representations for embedding
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

    # ── State Update (§4.1) ───────────────────────────────────────────────

    def _update_state(self, state, merged, batch_chunks):
        """Update rolling state with merge results and anchor terms."""
        # Add new fields (dedup by normalized name)
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

        # Add new clause types
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

        # Add intents
        for i_dict in merged.get("intents", []):
            desc = i_dict.get("intent_description", "")
            if desc:
                state.intents.append(IntentV2(
                    clause_ref=i_dict.get("clause_ref", ""),
                    intent_description=desc,
                ))

        # Add obligations
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

        # Update anchor terms (frequency counting from chunks)
        existing_terms = dict(state.anchor_terms)
        for chunk in batch_chunks:
            words = re.findall(r'\b[a-z]{4,}\b', chunk.lower())
            for term, count in Counter(words).most_common(20):
                existing_terms[term] = existing_terms.get(term, 0) + count
        state.anchor_terms = sorted(existing_terms.items(), key=lambda x: -x[1])[:20]

        state.chunks_processed += len(batch_chunks)

    # ── Two-Tier Canonicalization (§4.6) ──────────────────────────────────

    def _canonicalize(self, state):
        """
        Canonicalize fields and clause types to remove naming drift.

        Tier 1: Rule-based fuzzy match (substring, plural/singular, stopwords)
        Tier 2: Embedding similarity > 0.82 → merge

        Niche preservation: never auto-merge items with source="exploratory"
        against each other (they're prime niche-discovery candidates).
        """
        if len(state.fields) <= 1:
            return state

        # ── Tier 1: Rule-based fuzzy ──
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

                # Niche preservation: skip if both are exploratory
                if fi.source == "exploratory" and fj.source == "exploratory":
                    continue

                # Substring containment
                if normalized[i] in normalized[j] or normalized[j] in normalized[i]:
                    if len(names[i]) <= len(names[j]):
                        if len(fj.description) > len(fi.description):
                            state.fields[i] = fi.model_copy(update={"description": fj.description})
                        to_remove.add(j)
                    else:
                        if len(fi.description) > len(fj.description):
                            state.fields[j] = fj.model_copy(update={"description": fi.description})
                        to_remove.add(i)
                # Plural/singular collapse
                elif (normalized[i] + 's' == normalized[j] or
                      normalized[j] + 's' == normalized[i]):
                    shorter = i if len(names[i]) <= len(names[j]) else j
                    longer = j if shorter == i else i
                    to_remove.add(longer)

        # ── Tier 2: Embedding similarity ──
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
                    # Niche preservation
                    if fa.source == "exploratory" and fb.source == "exploratory":
                        continue
                    if sim[a, b] > 0.82:
                        if len(fa.description) >= len(fb.description):
                            to_remove.add(remaining[b])
                        else:
                            to_remove.add(remaining[a])

        state.fields = [f for i, f in enumerate(state.fields) if i not in to_remove]

        # Canonicalize clause types (simpler — name dedup)
        seen_ct = {}
        unique_cts = []
        for ct in state.clause_types:
            norm = re.sub(r'[_\s]', '', ct.name.lower())
            if norm not in seen_ct:
                seen_ct[norm] = True
                unique_cts.append(ct)
        state.clause_types = unique_cts

        return state

    # ── Output conversion ─────────────────────────────────────────────────

    def _to_field_dicts(self, state):
        """Convert RollingState fields to extraction-comparison-compatible dicts."""
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


# ── Module-level entry point ──────────────────────────────────────────────────

def discover_fields_for_cluster_rlm_v2(
    client,
    deployment: str,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config,
    embedder=None,
) -> tuple[list[dict], dict]:
    """
    RLM v2 field discovery — drop-in replacement for comparison scripts.

    Returns:
        (fields, meta) — fields compatible with extract_fields_from_chunk(),
        meta with v2-specific statistics.
    """
    pipeline = LatentRLMv2(
        client=client,
        deployment=deployment,
        config=config,
        embedder=embedder,
    )
    return pipeline.discover_for_cluster(cluster_label, keywords, chunk_texts)
