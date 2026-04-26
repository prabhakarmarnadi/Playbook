#!/usr/bin/env python3
"""
context_graph.py — Temporal Context Graph for Intent-Driven Contract Analysis
=============================================================================

HOW CONTEXT GRAPHS WORK (and why they matter for systems of intent)
-------------------------------------------------------------------

A **knowledge graph** is a static snapshot: nodes are entities, edges are
relationships.  "Agreement A CONTAINS Termination" — period.  This is fine
for records, but a system of intent needs to answer questions like:

  - "What did this agreement look like BEFORE the amendment?"
  - "Which facts were invalidated by the user's feedback?"
  - "How confident are we in this relationship given it was extracted 6 months ago?"
  - "What's the most relevant view of this graph RIGHT NOW for this tenant?"

A **context graph** extends a knowledge graph with five dimensions:

  1. TEMPORAL VALIDITY — every fact (edge) has [valid_from, valid_until].
     Facts are never deleted; they are invalidated.  This gives full history.

  2. EPISODE PROVENANCE — every fact traces to an Episode: the exact event
     that created it (a pipeline run, a user feedback click, an amendment).
     Episodes are immutable.  You can always answer "why does this fact exist?"

  3. ACTIVATION CONTEXT — every fact carries a `context` dict with metadata
     that controls its relevance at query time.  The graph can be "activated"
     with a reference time + decay function to produce a weighted view where
     recent, high-confidence, feedback-validated facts dominate.

  4. ACTIVITY SIGNALS — fine-grained, timestamped user/system actions below
     the Episode level. "User viewed clause X", "user compared agreements A
     and B", "user exported field Y".  These signals capture HOW work is
     actually done — the observable digital trail that, over many cycles,
     approximates the WHY (intent) behind user behavior.
     (Inspired by Glean's context data platform: "You can't reliably capture
     the why; you can capture the how.")

  5. PROCESS TRACES — higher-level constructs inferred from sequences of
     activity signals.  A burst of view → compare → accept actions across
     3 agreements = a "contract review task".  Repeated patterns across
     users reveal tribal knowledge that was never explicitly documented.
     Process traces let agents learn and automate real work.

ARCHITECTURE
------------

```
  ┌────────────────────────────────────────────────────────┐
  │                 ContractContextGraph                    │
  │                                                        │
  │  ┌─────────────────────┐   ┌────────────────────────┐  │
  │  │   NetworkX DiGraph   │   │     Fact Store          │  │
  │  │   (active view)      │   │   ALL TemporalFact      │  │
  │  │   Only valid facts   │   │   (incl. invalidated)   │  │
  │  │   For fast analysis  │   │   For history/audit     │  │
  │  └──────────┬──────────┘   └────────────┬───────────┘  │
  │             │                            │              │
  │  ┌──────────┴──────────┐   ┌────────────┴───────────┐  │
  │  │   Episode Store      │   │  Entity State History   │  │
  │  │   Immutable events   │   │  Per-node summaries     │  │
  │  │   (provenance chain) │   │  that evolve over time  │  │
  │  └─────────────────────┘   └────────────────────────┘  │
  └────────────────────────────────────────────────────────┘
```

There are TWO parallel representations:

  - **Active View** (NetworkX DiGraph): contains ONLY currently-valid facts.
    This is what graph algorithms operate on — recommendations, anomaly
    detection, archetype clustering, strategy discovery.  Fast, no filtering
    needed at query time.

  - **Fact Store** (dict[str, TemporalFact]): contains ALL facts ever created,
    including invalidated ones.  This is the audit trail.  Point-in-time
    queries (`as_of="2024-06-01"`) scan this store and reconstruct the
    graph as it was at that moment.

When a new fact is added:
  1. A TemporalFact is created with valid_from = now, valid_until = None
  2. It's stored in the fact store
  3. A corresponding edge is added to the NetworkX active view
  4. An index entry maps (source, target, relation) → fact_id for O(1) lookup

When a fact is invalidated:
  1. valid_until is set to now
  2. invalidated_by is set to the episode that caused it
  3. The edge is REMOVED from the NetworkX active view
  4. The fact STAYS in the fact store (never deleted)

TEMPORAL PRIMITIVES
-------------------

**Episode** — an atomic ingestion event.  The ground truth provenance unit.

  Types:
    pipeline_run   — bulk pipeline execution (initial build)
    feedback       — human feedback (field_accept, field_reject, cluster_merge)
    amendment      — contract amendment that changes terms
    incremental    — single document ingested via assign_new_documents
    reclassify     — ontology evolution (cluster merge/split/relabel)

  Every fact traces to exactly one Episode.  Episodes form a totally-ordered
  stream — the immutable event log of the system.

**TemporalFact** — a single relationship with temporal validity.

  Fields:
    source_node, target_node, relation  — the triple
    weight, confidence                  — strength and extraction certainty
    valid_from, valid_until             — temporal window (None = still valid)
    episode_id                          — provenance (which episode created this)
    invalidated_by                      — which episode killed this (None = active)
    context                             — activation metadata dict

  The `activated_weight()` method computes a time-decayed, feedback-adjusted
  weight at query time:

    activated_weight = base_weight × temporal_decay × feedback_mult × confidence

  where temporal_decay = exp(-0.693 × age_days / halflife_days)
  (exponential decay with configurable half-life, default 90 days)

**EntityState** — a snapshot of an entity's evolving summary.

  Entities (agreements, clause types, intent types, parties) have summaries
  that change over time.  Each change is recorded as an EntityState with the
  episode that caused it.  You can reconstruct the full history of any entity.

ACTIVATED SUBGRAPH
------------------

The `activated_subgraph()` method builds a view of the graph weighted by
temporal relevance:

  1. Iterate all active facts
  2. Compute activated_weight for each
  3. Prune facts below min_activated_weight
  4. Return a new DiGraph with activated weights

This gives a "what matters now" view.  Recent, high-confidence facts from
validated episodes dominate.  Stale facts from months ago fade.  Facts
boosted by user feedback shine brighter.

FEEDBACK INTEGRATION
--------------------

The `ingest_feedback_episode()` method maps FeedbackStore events to graph
mutations:

  field_reject  → invalidate HOSTS facts for that field's cluster
  field_accept  → boost confidence + feedback_weight on matching facts
  field_rename  → record entity state change (old name → new name)
  cluster_merge → record state changes on merged clusters
  threshold_adj → update entity properties

This closes the loop: user feedback doesn't just update extraction —
it reshapes the graph's truth, weakening bad facts and strengthening good ones.

ACTIVITY SIGNALS & PROCESS TRACES  (inspired by Glean's context data platform)
-------------------------------------------------------------------------------

Glean's key insight: "You can't reliably capture the why; you can capture the
how."  The WHY is a thinking step that usually resides in someone's head.  The
HOW leaves a rich digital trail: recurring steps, data updates, approvals,
collaboration patterns, changing fields, cross-system behavior over time.

Over many cycles, process traces APPROXIMATE the why: you can infer rationales
from patterns in how work repeatedly gets done.

**ActivitySignal** — a discrete, timestamped action taken by a user or system.
Lower-level than Episodes.  Examples:
    view_agreement, view_clause, compare_agreements, export_field,
    edit_field, accept_recommendation, reject_recommendation,
    search_query, run_pipeline, agent_action

Signals are captured in chronological order.  State changes between them
provide the raw activity data needed to infer process patterns.

**ProcessTrace** — a higher-level construct derived from activity signal
sequences.  Patterns across many low-level actions are aggregated:
    contract_review    — view + compare + accept/reject across agreements
    taxonomy_cleanup   — field_reject + cluster_merge + relabel sequence
    onboarding_audit   — bulk view + export + anomaly review
    risk_assessment    — anomaly_review + recommendation_accept + compare

`derive_process_traces()` scans activity signals for known patterns,
grouped by user session (time windows + shared entities).  Each trace
becomes an Episode of type 'process_trace' and can generate facts like:
    USER_PERFORMED  user:alice → process:contract_review
    INVOLVED        process:contract_review → agr:a_001

**Enterprise Memory** — when the system (or an agent) performs an action,
the execution trace is captured as an Episode of type 'agent_execution'.
This captures WHAT the agent did, in WHAT order, with WHAT outcome.
Process understanding emerges from structural understanding + learned behavior.

Ref: https://www.glean.com/blog/context-data-platform
Ref: Jaya Gupta & Ashu Garg — "AI's next trillion-dollar opportunity"

POINT-IN-TIME QUERIES
---------------------

  `query_facts(as_of="2024-06-01")` returns facts that were valid on June 1st:
    - valid_from ≤ 2024-06-01
    - valid_until is None OR valid_until > 2024-06-01

  `get_entity_timeline("ct:Termination")` returns chronological state snapshots:
    [{timestamp, summary, episode_id, episode_type, properties}, ...]

  `get_fact_provenance(fact_id)` returns the creating episode + invalidating
  episode (if any) for any fact in the store.

GRAPH SCHEMA
------------

Node types (prefixed):
    agr:<id>    Agreement documents
    ct:<label>  Clause type clusters (from clustering pipeline)
    it:<label>  Intent types (from intent extraction)
    pty:<name>  Party roles (normalized names)
    ep:<id>     Episodes (provenance nodes)
    usr:<id>    Users / actors (for activity tracking)
    proc:<id>   Process traces (inferred higher-level tasks)

Fact types (temporal edges):
    CONTAINS        agr → ct    Agreement contains clause type
    HOSTS           ct  → it    Clause type carries intent type
    CO_OCCURS       it  ↔ it    Intents co-occur in same agreement
    CT_CO_OCCURS    ct  ↔ ct    Clause types co-occur in same agreement
    BEARS           pty → it    Party bears obligation
    BENEFITS_FROM   pty → it    Party benefits from obligation
    SIMILAR_INTENT  it  ↔ it    Embedding cosine similarity above threshold
    SOURCED_BY      *   → ep    Provenance link fact → episode
    PERFORMED       usr → proc  User performed a process trace
    INVOLVED        proc → *    Process trace involved an entity
    INTERACTED      usr → *     User interacted with an entity (from signals)

ANALYSIS METHODS
----------------

All analysis methods operate on the NetworkX active view for speed.
They preserve backward compatibility with the old static KnowledgeGraph API.

  recommend()              — Association rule mining: missing types/intents
  detect_anomalies()       — Missing correlated types, rare orphan intents,
                             recently invalidated facts
  discover_archetypes()    — Jaccard + Louvain community detection on agreements
  get_implication_rules()  — Directed A→B confidence/support/lift rules
  discover_strategies()    — Co-occurrence communities = legal strategies
  add_intent_similarity()  — Embedding-based SIMILAR_INTENT edge creation
  compare_agreements()     — Deep diff with similarity-aware matching
  find_most_similar()      — Top-N agreement similarity ranking

REFERENCES
----------
  - Graphiti (Zep): temporal knowledge graph with episode provenance
    https://github.com/getzep/graphiti
  - Glean: context data platform — activity signals, process traces,
    enterprise memory.  "You can't reliably capture the why; you can
    capture the how."
    https://www.glean.com/blog/context-data-platform
  - Jaya Gupta & Ashu Garg: "AI's next trillion-dollar opportunity" —
    the original context graph framing for enterprise AI
  - Temporal Knowledge Graphs: "Towards Temporal Knowledge Graph Embeddings
    with Arbitrary Time Precision" (Lacroix et al., 2020)
  - Context-Aware KGs: "Context-Aware Knowledge Graph Embedding for
    Completing Incomplete KGs" (Wang et al., 2019)
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ── Node type prefixes ───────────────────────────────────────────────────────

AGR  = "agr:"       # agreement nodes
CT   = "ct:"        # clause type nodes
IT   = "it:"        # intent type nodes
PTY  = "pty:"       # party nodes
EP   = "ep:"        # episode nodes (provenance)
USR  = "usr:"       # user / actor nodes (activity tracking)
PROC = "proc:"      # process trace nodes (inferred tasks)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════


def _now() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp, return None if missing."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


@dataclass
class Episode:
    """An atomic ingestion event — the ground truth provenance unit.

    Every fact in the context graph traces back to one or more episodes.
    Episodes are immutable once created.

    Types:
      pipeline_run     — a full pipeline execution (bulk ingestion)
      feedback         — a human feedback event (field_accept, field_reject, etc.)
      amendment        — an agreement amendment/renewal that changes terms
      incremental      — a single document ingested via assign_new_documents
      reclassify       — ontology evolution (cluster merge/split/relabel)
      process_trace    — inferred higher-level task derived from activity signals
      agent_execution  — system/agent performed an action (enterprise memory)
    """
    episode_id: str = ""
    episode_type: str = "pipeline_run"
    source: str = ""
    timestamp: str = ""
    account_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = _now()


@dataclass
class TemporalFact:
    """A single fact (edge) in the context graph with temporal validity.

    Facts are never deleted — they are invalidated by setting valid_until.
    This preserves full history for point-in-time queries.

    The activated_weight() method computes:
      base_weight × temporal_decay × feedback_mult × confidence
    where temporal_decay = exp(-0.693 × age_days / halflife_days)
    """
    fact_id: str = ""
    source_node: str = ""
    target_node: str = ""
    relation: str = ""
    weight: float = 1.0
    confidence: float = 1.0
    valid_from: str = ""
    valid_until: str | None = None
    episode_id: str = ""
    invalidated_by: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.fact_id:
            self.fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        if not self.valid_from:
            self.valid_from = _now()

    @property
    def is_active(self) -> bool:
        """Whether this fact is currently valid (not superseded)."""
        return self.valid_until is None

    def is_active_at(self, as_of: str | datetime) -> bool:
        """Whether this fact was valid at a specific point in time."""
        if isinstance(as_of, str):
            as_of = _parse_ts(as_of)
        if as_of is None:
            return self.is_active
        vf = _parse_ts(self.valid_from)
        vu = _parse_ts(self.valid_until)
        if vf and as_of < vf:
            return False
        if vu and as_of >= vu:
            return False
        return True

    def activated_weight(self, reference_time: str | None = None,
                         decay_halflife_days: float = 90.0) -> float:
        """Compute time-decayed, feedback-adjusted weight."""
        ref = _parse_ts(reference_time) if reference_time else datetime.now(timezone.utc)
        vf = _parse_ts(self.valid_from)

        decay = 1.0
        if ref and vf:
            age_days = (ref - vf).total_seconds() / 86400.0
            if age_days > 0 and decay_halflife_days > 0:
                decay = math.exp(-0.693 * age_days / decay_halflife_days)

        feedback_mult = self.context.get("feedback_weight", 1.0)
        return self.weight * decay * feedback_mult * self.confidence


@dataclass
class EntityState:
    """A snapshot of an entity's evolving summary at a point in time."""
    entity_id: str = ""
    summary: str = ""
    timestamp: str = ""
    episode_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivitySignal:
    """A discrete, timestamped user/system action — the atomic unit of 'how'.

    Activity signals capture the fine-grained digital trail of how work happens.
    They are lower-level than Episodes: an Episode might be "pipeline_run",
    while signals within it are "view_agreement", "compare_agreements", etc.

    Over many cycles, patterns in these signals approximate intent — you infer
    rationales from how work repeatedly gets done, not from trying to store
    every human thought.  (Glean: "You can't capture the why; you can capture
    the how.")

    Action types:
      view_agreement, view_clause, view_field, compare_agreements,
      export_field, edit_field, accept_recommendation, reject_recommendation,
      search_query, run_report, agent_action, api_call
    """
    signal_id: str = ""
    action: str = ""               # e.g. "view_agreement", "compare_agreements"
    actor_id: str = ""             # user or system identifier
    entity_ids: list[str] = field(default_factory=list)  # entities involved
    timestamp: str = ""
    account_id: str = ""
    session_id: str = ""           # groups signals into user sessions
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = f"sig_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = _now()


@dataclass
class ProcessTrace:
    """A higher-level construct inferred from sequences of activity signals.

    Process traces represent the tribal knowledge of how work gets done:
    a "contract review" is never labeled as such in any single system, but
    the pattern of view → compare → accept across several agreements reveals
    it.  Traces become Episodes of type 'process_trace' and generate facts
    in the graph (PERFORMED, INVOLVED edges).

    Trace types:
      contract_review   — view + compare + accept/reject across agreements
      taxonomy_cleanup  — field_reject + cluster_merge + relabel sequence
      onboarding_audit  — bulk view + export + anomaly review
      risk_assessment   — anomaly_review + recommendation_accept + compare
      bulk_export       — multiple export actions in short window
      custom            — user-defined or ML-inferred trace
    """
    trace_id: str = ""
    trace_type: str = "custom"     # contract_review | taxonomy_cleanup | etc.
    actor_id: str = ""
    signal_ids: list[str] = field(default_factory=list)  # signals that compose this trace
    entity_ids: list[str] = field(default_factory=list)  # entities involved
    started_at: str = ""
    ended_at: str = ""
    episode_id: str = ""           # the process_trace Episode created for this
    confidence: float = 1.0        # inference confidence
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = f"proc_{uuid.uuid4().hex[:12]}"

    @property
    def duration_seconds(self) -> float:
        """Duration of this process trace in seconds."""
        s = _parse_ts(self.started_at)
        e = _parse_ts(self.ended_at)
        if s and e:
            return (e - s).total_seconds()
        return 0.0


@dataclass
class GraphStats:
    """Statistics for the context graph — both current and historical."""
    agreements: int = 0
    clause_types: int = 0
    intent_types: int = 0
    parties: int = 0
    episodes: int = 0
    total_facts: int = 0
    active_facts: int = 0
    invalidated_facts: int = 0
    co_occurrence_edges: int = 0
    activity_signals: int = 0
    process_traces: int = 0

    @property
    def edges(self):
        """Legacy compat: total active edges in the NetworkX active view."""
        return self.active_facts


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT GRAPH
# ══════════════════════════════════════════════════════════════════════════════


class ContractContextGraph:
    """Temporal context graph over legal contract structure.

    Architecture:
      - NetworkX DiGraph (self.G): the "active view" — only currently-valid facts.
      - Fact store (self._facts): ALL TemporalFact objects, including invalidated.
      - Episode store (self._episodes): every ingestion event.
      - Entity state history (self._entity_states): per-entity summary evolution.
      - Activity signal log (self._signals): fine-grained timestamped user actions.
      - Process trace store (self._traces): inferred higher-level tasks.

    Node prefixes:
        agr:<id>   — agreements       ct:<label> — clause types
        it:<label> — intent types     pty:<name> — party roles
        ep:<id>    — episodes         usr:<id>   — users/actors
        proc:<id>  — process traces

    Fact types (temporal edges):
        CONTAINS, HOSTS, CO_OCCURS, CT_CO_OCCURS, BEARS, BENEFITS_FROM,
        SIMILAR_INTENT, SOURCED_BY, PERFORMED, INVOLVED, INTERACTED
    """

    def __init__(self, store):
        self.store = store
        self.G = nx.DiGraph()
        self.stats = GraphStats()

        # ── Temporal stores ──
        self._facts: dict[str, TemporalFact] = {}
        self._episodes: dict[str, Episode] = {}
        self._entity_states: dict[str, list[EntityState]] = defaultdict(list)

        # ── Index: edge key → fact_id (for active view sync) ──
        self._edge_fact_index: dict[tuple[str, str, str], str] = {}

        # ── Caches for fast lookup ──
        self._agr_intents: dict[str, set[str]] = defaultdict(set)
        self._agr_clause_types: dict[str, set[str]] = defaultdict(set)
        self._intent_agreements: dict[str, set[str]] = defaultdict(set)

        # ── Activity & Process stores (Glean-inspired) ──
        self._signals: list[ActivitySignal] = []             # chronological activity log
        self._signals_by_session: dict[str, list[ActivitySignal]] = defaultdict(list)
        self._signals_by_actor: dict[str, list[ActivitySignal]] = defaultdict(list)
        self._traces: dict[str, ProcessTrace] = {}           # trace_id → ProcessTrace

    # ══════════════════════════════════════════════════════════════════════
    # TEMPORAL CORE — fact lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def _add_fact(self, fact: TemporalFact) -> TemporalFact:
        """Register a new temporal fact and sync it to the active view."""
        self._facts[fact.fact_id] = fact
        self.stats.total_facts += 1
        if fact.is_active:
            self.stats.active_facts += 1
            self.G.add_edge(
                fact.source_node, fact.target_node,
                key=fact.relation,
                rel=fact.relation,
                weight=fact.weight,
                fact_id=fact.fact_id,
                confidence=fact.confidence,
                episode_id=fact.episode_id,
            )
            self._edge_fact_index[(fact.source_node, fact.target_node, fact.relation)] = fact.fact_id
        return fact

    def _add_or_increment_fact(
        self,
        source: str, target: str, relation: str,
        episode: Episode,
        weight: float = 1.0,
        confidence: float = 1.0,
        context: dict | None = None,
        increment: bool = True,
    ) -> TemporalFact:
        """Add a new fact or increment the weight of an existing active fact."""
        key = (source, target, relation)
        existing_fid = self._edge_fact_index.get(key)
        if existing_fid and increment:
            existing = self._facts.get(existing_fid)
            if existing and existing.is_active:
                existing.weight += weight
                if self.G.has_edge(source, target):
                    self.G[source][target]["weight"] = existing.weight
                return existing

        fact = TemporalFact(
            source_node=source, target_node=target, relation=relation,
            weight=weight, confidence=confidence,
            valid_from=episode.timestamp, episode_id=episode.episode_id,
            context=context or {},
        )
        return self._add_fact(fact)

    def invalidate_fact(self, fact_id: str, reason: str = "",
                        episode_id: str | None = None) -> bool:
        """Invalidate a fact by setting its valid_until timestamp.

        The fact is never deleted — it remains in the fact store for history.
        It IS removed from the NetworkX active view.
        """
        fact = self._facts.get(fact_id)
        if not fact or not fact.is_active:
            return False

        fact.valid_until = _now()
        fact.invalidated_by = episode_id
        fact.context["invalidation_reason"] = reason
        self.stats.active_facts -= 1
        self.stats.invalidated_facts += 1

        if self.G.has_edge(fact.source_node, fact.target_node):
            self.G.remove_edge(fact.source_node, fact.target_node)
        key = (fact.source_node, fact.target_node, fact.relation)
        self._edge_fact_index.pop(key, None)

        logger.debug(f"Invalidated fact {fact_id}: {fact.source_node} "
                     f"→{fact.relation}→ {fact.target_node} ({reason})")
        return True

    def _update_entity_state(self, entity_id: str, summary: str,
                             episode: Episode,
                             properties: dict | None = None):
        """Record a new state snapshot for an entity."""
        state = EntityState(
            entity_id=entity_id, summary=summary,
            timestamp=episode.timestamp, episode_id=episode.episode_id,
            properties=properties or {},
        )
        self._entity_states[entity_id].append(state)
        if self.G.has_node(entity_id):
            self.G.nodes[entity_id]["summary"] = summary
            self.G.nodes[entity_id]["last_episode"] = episode.episode_id

    def _register_episode(self, episode: Episode) -> Episode:
        """Store an episode and add it as a node in the graph."""
        self._episodes[episode.episode_id] = episode
        self.stats.episodes += 1
        self.G.add_node(
            f"{EP}{episode.episode_id}", kind="episode",
            episode_type=episode.episode_type,
            timestamp=episode.timestamp, source=episode.source,
        )
        return episode

    # ══════════════════════════════════════════════════════════════════════
    # TEMPORAL QUERIES
    # ══════════════════════════════════════════════════════════════════════

    def query_facts(
        self,
        entity: str | None = None,
        relation: str | None = None,
        as_of: str | None = None,
        active_only: bool = True,
    ) -> list[TemporalFact]:
        """Query facts with optional temporal and structural filters."""
        results = []
        for fact in self._facts.values():
            if entity and entity != fact.source_node and entity != fact.target_node:
                continue
            if relation and fact.relation != relation:
                continue
            if as_of:
                if not fact.is_active_at(as_of):
                    continue
            elif active_only and not fact.is_active:
                continue
            results.append(fact)
        results.sort(key=lambda f: f.valid_from, reverse=True)
        return results

    def get_entity_timeline(self, entity_id: str) -> list[dict]:
        """Get the full history of an entity's state changes."""
        states = self._entity_states.get(entity_id, [])
        timeline = []
        for s in states:
            ep = self._episodes.get(s.episode_id)
            timeline.append({
                "timestamp": s.timestamp,
                "summary": s.summary,
                "episode_id": s.episode_id,
                "episode_type": ep.episode_type if ep else "unknown",
                "properties": s.properties,
            })
        timeline.sort(key=lambda e: e["timestamp"])
        return timeline

    def get_fact_provenance(self, fact_id: str) -> dict | None:
        """Trace a fact back to its source episode."""
        fact = self._facts.get(fact_id)
        if not fact:
            return None
        creator = self._episodes.get(fact.episode_id)
        invalidator = self._episodes.get(fact.invalidated_by) if fact.invalidated_by else None
        return {
            "fact": {
                "fact_id": fact.fact_id,
                "source": fact.source_node,
                "target": fact.target_node,
                "relation": fact.relation,
                "weight": fact.weight,
                "confidence": fact.confidence,
                "valid_from": fact.valid_from,
                "valid_until": fact.valid_until,
                "is_active": fact.is_active,
                "context": fact.context,
            },
            "created_by": {
                "episode_id": creator.episode_id,
                "type": creator.episode_type,
                "source": creator.source,
                "timestamp": creator.timestamp,
            } if creator else None,
            "invalidated_by": {
                "episode_id": invalidator.episode_id,
                "type": invalidator.episode_type,
                "source": invalidator.source,
                "timestamp": invalidator.timestamp,
                "reason": fact.context.get("invalidation_reason", ""),
            } if invalidator else None,
        }

    def activated_subgraph(
        self,
        reference_time: str | None = None,
        min_activated_weight: float = 0.1,
        decay_halflife_days: float = 90.0,
    ) -> nx.DiGraph:
        """Build a context-activated subgraph where edge weights reflect temporal decay.

        This gives a "what matters now" view of the graph.  Recent, high-confidence,
        feedback-validated facts dominate.  Old facts fade via exponential decay.
        """
        activated = nx.DiGraph()
        for fact in self._facts.values():
            if not fact.is_active:
                continue
            aw = fact.activated_weight(reference_time, decay_halflife_days)
            if aw >= min_activated_weight:
                activated.add_edge(
                    fact.source_node, fact.target_node,
                    rel=fact.relation,
                    weight=round(aw, 4),
                    fact_id=fact.fact_id,
                    confidence=fact.confidence,
                )
        for node in activated.nodes():
            if node in self.G.nodes:
                activated.nodes[node].update(self.G.nodes[node])
        return activated

    # ══════════════════════════════════════════════════════════════════════
    # EPISODE INGESTION — incremental graph updates
    # ══════════════════════════════════════════════════════════════════════

    def ingest_episode(self, episode: Episode,
                       facts: list[TemporalFact] | None = None,
                       invalidations: list[str] | None = None):
        """Incrementally update the graph from a new episode.

        This is the primary mutation entry point for the context graph.
        """
        self._register_episode(episode)

        for fid in (invalidations or []):
            self.invalidate_fact(fid, reason=f"superseded by {episode.episode_type}",
                                episode_id=episode.episode_id)

        for fact in (facts or []):
            fact.episode_id = episode.episode_id
            if not fact.valid_from:
                fact.valid_from = episode.timestamp
            self._add_fact(fact)

        logger.info(f"Ingested episode {episode.episode_id} ({episode.episode_type}): "
                     f"+{len(facts or [])} facts, "
                     f"-{len(invalidations or [])} invalidated")

    def ingest_feedback_episode(self, feedback_event: dict):
        """Ingest a FeedbackStore event as a graph episode.

        Maps feedback actions to graph mutations:
          field_reject  → invalidate HOSTS facts for that field's cluster
          field_accept  → boost confidence on matching facts
          field_rename  → record entity state change
          cluster_merge → record state changes on merged clusters
          threshold_adj → update entity state properties
        """
        action = feedback_event.get("action", "")
        entity_id = feedback_event.get("entity_id", "")
        episode = Episode(
            episode_type="feedback",
            source=feedback_event.get("event_id", ""),
            account_id=feedback_event.get("tenant_id", ""),
            metadata=feedback_event,
        )
        self._register_episode(episode)

        if action == "field_reject":
            cluster_id = feedback_event.get("cluster_id", "")
            for fact in list(self._facts.values()):
                if (fact.is_active and fact.relation == "HOSTS"
                        and fact.context.get("cluster_id") == cluster_id
                        and fact.context.get("field_name") == entity_id):
                    self.invalidate_fact(fact.fact_id, reason="field_reject feedback",
                                        episode_id=episode.episode_id)

        elif action == "field_accept":
            for fact in self._facts.values():
                if (fact.is_active
                        and fact.context.get("field_name") == entity_id):
                    fact.context["feedback_weight"] = fact.context.get("feedback_weight", 1.0) + 0.2
                    fact.confidence = min(1.0, fact.confidence + 0.1)

        elif action == "field_rename":
            old_val = feedback_event.get("old_value", "")
            new_val = feedback_event.get("new_value", "")
            if old_val and new_val:
                self._update_entity_state(
                    f"{IT}{old_val}", f"Renamed to {new_val}", episode,
                    {"renamed_from": old_val, "renamed_to": new_val})

        elif action == "cluster_merge":
            meta = feedback_event.get("metadata", {})
            merged_ids = meta.get("merged_cluster_ids", [])
            new_id = meta.get("new_cluster_id", "")
            for cid in merged_ids:
                label = self._ct_id_to_label(cid) or cid
                self._update_entity_state(
                    f"{CT}{label}", f"Merged into {new_id}", episode,
                    {"merged_into": new_id})

        elif action == "threshold_adj":
            self._update_entity_state(
                f"agr:{feedback_event.get('tenant_id', 'global')}",
                f"Threshold adjusted: {feedback_event.get('new_value')}",
                episode,
                {"threshold": feedback_event.get("new_value")},
            )

    # ══════════════════════════════════════════════════════════════════
    # ACTIVITY SIGNALS & PROCESS TRACES (Glean-inspired)
    # "You can't capture the why; you can capture the how."
    # ══════════════════════════════════════════════════════════════════

    def ingest_activity_signal(self, signal: ActivitySignal):
        """Record a fine-grained user/system action.

        Activity signals are lower-level than Episodes — they capture the
        discrete timestamped actions that form the digital trail of HOW work
        happens.  Over many cycles, patterns in these signals approximate
        the WHY (intent) behind user behavior.

        Signals are indexed by session and actor for process trace derivation.
        High-frequency entity interactions also boost fact confidence
        (more attention = more relevant).

        Args:
            signal: An ActivitySignal with action, actor_id, entity_ids, etc.
        """
        self._signals.append(signal)
        self.stats.activity_signals += 1

        if signal.session_id:
            self._signals_by_session[signal.session_id].append(signal)
        if signal.actor_id:
            self._signals_by_actor[signal.actor_id].append(signal)

        # Ensure actor node exists
        actor_node = f"{USR}{signal.actor_id}"
        if signal.actor_id and not self.G.has_node(actor_node):
            self.G.add_node(actor_node, kind="user", id=signal.actor_id)

        # Boost engagement weight on facts involving these entities
        for eid in signal.entity_ids:
            for fact in self._facts.values():
                if not fact.is_active:
                    continue
                if fact.source_node == eid or fact.target_node == eid:
                    bump = 0.05 if signal.action.startswith("view") else 0.1
                    fact.context["engagement_weight"] = (
                        fact.context.get("engagement_weight", 1.0) + bump
                    )

        logger.debug(f"Signal {signal.signal_id}: {signal.action} by "
                     f"{signal.actor_id} on {signal.entity_ids}")

    # ── Process Trace Pattern Definitions ────────────────────────────────

    PROCESS_PATTERNS: dict[str, dict] = {
        "contract_review": {
            "sequence": ["view_agreement", "compare_agreements"],
            "must_contain_one": ["accept_recommendation", "reject_recommendation",
                                "edit_field", "export_field"],
            "min_signals": 3,
            "max_gap_seconds": 3600,  # signals within 1 hour
            "description": "User reviewing and acting on contract content",
        },
        "taxonomy_cleanup": {
            "sequence": ["reject_recommendation"],
            "must_contain_one": ["edit_field"],
            "min_signals": 2,
            "max_gap_seconds": 7200,
            "description": "User correcting field labels or cluster assignments",
        },
        "onboarding_audit": {
            "sequence": ["view_agreement"],
            "must_contain_one": ["export_field", "run_report"],
            "min_signals": 4,
            "max_gap_seconds": 3600,
            "description": "Bulk review and export during onboarding",
        },
        "risk_assessment": {
            "sequence": ["view_agreement"],
            "must_contain_one": ["accept_recommendation", "compare_agreements"],
            "min_signals": 3,
            "max_gap_seconds": 7200,
            "description": "Anomaly-driven review across agreements",
        },
    }

    def derive_process_traces(self,
                              session_id: str | None = None,
                              actor_id: str | None = None,
                              ) -> list[ProcessTrace]:
        """Infer higher-level process traces from activity signal sequences.

        Scans signals grouped by session (or actor if no session) for known
        patterns.  Each detected pattern becomes a ProcessTrace with an
        associated Episode of type 'process_trace', and generates PERFORMED
        and INVOLVED facts in the graph.

        Args:
            session_id: Derive traces for a specific session only.
            actor_id:   Derive traces for a specific actor only.

        Returns:
            List of newly created ProcessTrace objects.
        """
        new_traces = []

        # Gather signal groups to scan
        if session_id:
            groups = {session_id: self._signals_by_session.get(session_id, [])}
        elif actor_id:
            groups = {actor_id: self._signals_by_actor.get(actor_id, [])}
        else:
            groups = dict(self._signals_by_session)
            # Also group by actor for signals without sessions
            for aid, sigs in self._signals_by_actor.items():
                sessionless = [s for s in sigs if not s.session_id]
                if sessionless:
                    groups[f"actor:{aid}"] = sessionless

        for group_key, signals in groups.items():
            if len(signals) < 2:
                continue

            # Sort by timestamp
            signals = sorted(signals, key=lambda s: s.timestamp)

            for trace_type, pattern in self.PROCESS_PATTERNS.items():
                trace = self._match_pattern(signals, trace_type, pattern)
                if trace:
                    self._register_process_trace(trace)
                    new_traces.append(trace)

        logger.info(f"Derived {len(new_traces)} process traces")
        return new_traces

    def _match_pattern(self, signals: list[ActivitySignal],
                       trace_type: str,
                       pattern: dict) -> ProcessTrace | None:
        """Check if a list of signals matches a process pattern."""
        actions = [s.action for s in signals]
        min_signals = pattern.get("min_signals", 2)
        max_gap = pattern.get("max_gap_seconds", 3600)

        if len(signals) < min_signals:
            return None

        # Check sequence requirement: all sequence actions must appear in order
        seq_actions = pattern.get("sequence", [])
        seq_idx = 0
        for action in actions:
            if seq_idx < len(seq_actions) and action == seq_actions[seq_idx]:
                seq_idx += 1
        if seq_idx < len(seq_actions):
            return None

        # Check must_contain_one
        must_one = pattern.get("must_contain_one", [])
        if must_one and not any(a in actions for a in must_one):
            return None

        # Check time gap: all signals within max_gap of first
        first_ts = _parse_ts(signals[0].timestamp)
        last_ts = _parse_ts(signals[-1].timestamp)
        if first_ts and last_ts:
            gap = (last_ts - first_ts).total_seconds()
            if gap > max_gap:
                return None

        # Build trace
        all_entities = []
        for s in signals:
            all_entities.extend(s.entity_ids)
        unique_entities = list(dict.fromkeys(all_entities))  # dedupe, preserve order

        actor = signals[0].actor_id
        return ProcessTrace(
            trace_type=trace_type,
            actor_id=actor,
            signal_ids=[s.signal_id for s in signals],
            entity_ids=unique_entities,
            started_at=signals[0].timestamp,
            ended_at=signals[-1].timestamp,
            confidence=min(1.0, len(signals) / (min_signals * 2)),
            metadata={"pattern": trace_type, "n_signals": len(signals)},
        )

    def _register_process_trace(self, trace: ProcessTrace):
        """Register a process trace: store it, create Episode + graph facts."""
        self._traces[trace.trace_id] = trace
        self.stats.process_traces += 1

        # Create process_trace Episode
        episode = Episode(
            episode_type="process_trace",
            source=trace.trace_id,
            account_id=trace.actor_id,
            metadata={
                "trace_type": trace.trace_type,
                "n_signals": len(trace.signal_ids),
                "entity_ids": trace.entity_ids,
                "duration_seconds": trace.duration_seconds,
            },
        )
        self._register_episode(episode)
        trace.episode_id = episode.episode_id

        # Add process node
        proc_node = f"{PROC}{trace.trace_id}"
        self.G.add_node(proc_node, kind="process_trace",
                        trace_type=trace.trace_type,
                        actor_id=trace.actor_id)

        # PERFORMED: user → process
        actor_node = f"{USR}{trace.actor_id}"
        if not self.G.has_node(actor_node):
            self.G.add_node(actor_node, kind="user", id=trace.actor_id)
        self._add_or_increment_fact(
            actor_node, proc_node, "PERFORMED", episode,
            weight=1.0, confidence=trace.confidence,
            context={"trace_type": trace.trace_type},
        )

        # INVOLVED: process → entity (for each entity touched)
        for eid in trace.entity_ids:
            if self.G.has_node(eid):
                self._add_or_increment_fact(
                    proc_node, eid, "INVOLVED", episode,
                    weight=1.0, confidence=trace.confidence,
                    context={"trace_type": trace.trace_type},
                )

    def ingest_agent_execution(self, agent_id: str, actions: list[dict],
                                outcome: str = "success",
                                metadata: dict | None = None) -> Episode:
        """Record an agent execution trace as enterprise memory.

        When the system or an agent performs work (e.g., auto-classifying,
        generating recommendations, running a pipeline), the execution trace
        is captured so the system can learn from its own behavior over time.

        Process understanding emerges from structural understanding +
        learned behavior. (Glean: "execution traces capture what actually
        works for agents over time.")

        Args:
            agent_id:  Identifier for the agent/system component.
            actions:   Ordered list of action dicts, each with:
                       {"action": str, "entity_ids": list, "result": str,
                        "timestamp": str (optional)}
            outcome:   Overall outcome: "success", "partial", "failure".
            metadata:  Additional context.

        Returns:
            The created Episode for this execution trace.
        """
        episode = Episode(
            episode_type="agent_execution",
            source=agent_id,
            metadata={
                "agent_id": agent_id,
                "outcome": outcome,
                "n_actions": len(actions),
                "actions": actions,
                **(metadata or {}),
            },
        )
        self._register_episode(episode)

        # Record each action as an activity signal for pattern learning
        for action_dict in actions:
            signal = ActivitySignal(
                action=action_dict.get("action", "agent_action"),
                actor_id=agent_id,
                entity_ids=action_dict.get("entity_ids", []),
                timestamp=action_dict.get("timestamp", episode.timestamp),
                session_id=episode.episode_id,
                metadata={"outcome": outcome, "agent": True},
            )
            self.ingest_activity_signal(signal)

        # Create INTERACTED facts: agent → entities touched
        all_entities = []
        for a in actions:
            all_entities.extend(a.get("entity_ids", []))
        unique_entities = list(dict.fromkeys(all_entities))

        agent_node = f"{USR}{agent_id}"
        if not self.G.has_node(agent_node):
            self.G.add_node(agent_node, kind="agent", id=agent_id)

        for eid in unique_entities:
            if self.G.has_node(eid):
                self._add_or_increment_fact(
                    agent_node, eid, "INTERACTED", episode,
                    weight=1.0 if outcome == "success" else 0.5,
                    confidence=1.0 if outcome == "success" else 0.6,
                    context={"outcome": outcome, "agent_id": agent_id},
                )

        logger.info(f"Agent execution trace: {agent_id}, {len(actions)} actions, "
                     f"outcome={outcome}")
        return episode

    def get_actor_activity_summary(self, actor_id: str) -> dict:
        """Summarize an actor's activity: signals, traces, entities touched."""
        signals = self._signals_by_actor.get(actor_id, [])
        traces = [t for t in self._traces.values() if t.actor_id == actor_id]

        action_counts: Counter = Counter(s.action for s in signals)
        entity_counts: Counter = Counter()
        for s in signals:
            for eid in s.entity_ids:
                entity_counts[eid] += 1

        return {
            "actor_id": actor_id,
            "total_signals": len(signals),
            "action_breakdown": dict(action_counts.most_common(20)),
            "top_entities": dict(entity_counts.most_common(20)),
            "process_traces": [
                {"trace_id": t.trace_id, "type": t.trace_type,
                 "started": t.started_at, "n_signals": len(t.signal_ids)}
                for t in traces
            ],
            "n_traces": len(traces),
        }

    def get_entity_engagement(self, entity_id: str) -> dict:
        """How much attention has an entity received from users/agents?"""
        signals = [s for s in self._signals if entity_id in s.entity_ids]
        actors = set(s.actor_id for s in signals if s.actor_id)
        action_counts: Counter = Counter(s.action for s in signals)

        return {
            "entity_id": entity_id,
            "total_signals": len(signals),
            "unique_actors": len(actors),
            "action_breakdown": dict(action_counts.most_common(20)),
            "actors": sorted(actors),
        }

    # ══════════════════════════════════════════════════════════════════════
    # GRAPH CONSTRUCTION — initial build from DuckDB
    # ══════════════════════════════════════════════════════════════════════

    def build(self) -> GraphStats:
        """Build the full context graph from DuckDB data.

        Creates a single 'pipeline_run' episode as the provenance root.
        """
        self.G.clear()
        self._facts.clear()
        self._edge_fact_index.clear()
        self._agr_intents.clear()
        self._agr_clause_types.clear()
        self._intent_agreements.clear()
        self.stats = GraphStats()

        episode = Episode(
            episode_type="pipeline_run",
            source="initial_build",
            metadata={"method": "build"},
        )
        self._register_episode(episode)

        self._build_clause_type_layer(episode)
        self._build_intent_layer(episode)
        self._build_co_occurrence_edges(episode)

        logger.info(
            f"Context graph built: {self.G.number_of_nodes()} nodes, "
            f"{self.stats.active_facts} active facts "
            f"({self.stats.total_facts} total), "
            f"{self.stats.episodes} episodes"
        )
        return self.stats

    def _build_clause_type_layer(self, episode: Episode):
        """Add agreement and clause-type nodes + CONTAINS facts."""
        rows = self.store.conn.execute("""
            SELECT cl.agreement_id, c.label, c.cluster_id,
                   COUNT(*) as clause_count
            FROM clauses cl
            JOIN clusters c ON cl.clause_type_id = c.cluster_id
            WHERE cl.clause_type_id IS NOT NULL
            GROUP BY cl.agreement_id, c.label, c.cluster_id
        """).fetchall()

        agreements = set()
        clause_types = set()

        for agr_id, ct_label, ct_id, count in rows:
            agr_node = f"{AGR}{agr_id}"
            ct_node = f"{CT}{ct_label}"

            if agr_id not in agreements:
                self.G.add_node(agr_node, kind="agreement", id=agr_id)
                self._update_entity_state(agr_node, f"Agreement {agr_id}", episode)
                agreements.add(agr_id)

            if ct_label not in clause_types:
                self.G.add_node(ct_node, kind="clause_type", label=ct_label,
                                cluster_id=ct_id)
                self._update_entity_state(ct_node, f"Clause type: {ct_label}", episode,
                                          {"cluster_id": ct_id})
                clause_types.add(ct_label)

            self._add_or_increment_fact(
                agr_node, ct_node, "CONTAINS", episode,
                weight=count, confidence=1.0,
                context={"cluster_id": ct_id},
            )
            self._agr_clause_types[agr_id].add(ct_node)

        for agr_id, ct_set in self._agr_clause_types.items():
            for a, b in combinations(sorted(ct_set), 2):
                self._add_or_increment_fact(
                    a, b, "CT_CO_OCCURS", episode,
                    weight=1, confidence=1.0,
                    context={"from_agreement": agr_id},
                )
                self.stats.co_occurrence_edges += 1

        self.stats.agreements = len(agreements)
        self.stats.clause_types = len(clause_types)

    def _build_intent_layer(self, episode: Episode):
        """Add intent-type and party nodes with temporal facts."""
        try:
            cols = [r[0] for r in self.store.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='clause_intents'").fetchall()]
        except Exception:
            return

        if "intent_type_id" in cols:
            self._build_intent_layer_new(episode, cols)
        elif "primary_obligation" in cols:
            self._build_intent_layer_old(episode)

    def _build_intent_layer_new(self, episode: Episode, cols: list[str]):
        """Build from new dynamic intent schema."""
        rows = self.store.conn.execute("""
            SELECT ci.agreement_id, ci.clause_type_id, ci.intent_type_id,
                   ci.intent_label, ci.party_from, ci.party_to
            FROM clause_intents ci
            WHERE ci.intent_label IS NOT NULL AND ci.intent_label != ''
        """).fetchall()

        intent_types = set()
        parties = set()

        for agr_id, ct_id, it_id, it_label, party_from, party_to in rows:
            it_node = f"{IT}{it_label}"

            if it_label not in intent_types:
                self.G.add_node(it_node, kind="intent_type",
                                label=it_label, intent_type_id=it_id)
                self._update_entity_state(it_node, f"Intent: {it_label}", episode,
                                          {"intent_type_id": it_id})
                intent_types.add(it_label)

            ct_label = self._ct_id_to_label(ct_id)
            if ct_label:
                ct_node = f"{CT}{ct_label}"
                self._add_or_increment_fact(
                    ct_node, it_node, "HOSTS", episode,
                    weight=1, confidence=0.9,
                    context={"cluster_id": ct_id, "intent_type_id": it_id},
                )

            self._agr_intents[agr_id].add(it_node)
            self._intent_agreements[it_node].add(agr_id)

            for party_name, edge_rel in [(party_from, "BEARS"),
                                          (party_to, "BENEFITS_FROM")]:
                pname = _normalize_party(party_name)
                if pname:
                    pty_node = f"{PTY}{pname}"
                    if pname not in parties:
                        self.G.add_node(pty_node, kind="party", name=pname)
                        self._update_entity_state(pty_node, f"Party: {pname}", episode)
                        parties.add(pname)
                    self._add_or_increment_fact(
                        pty_node, it_node, edge_rel, episode,
                        weight=1, confidence=0.8,
                        context={"agreement_id": agr_id},
                    )

        self.stats.intent_types = len(intent_types)
        self.stats.parties = len(parties)

    def _build_intent_layer_old(self, episode: Episode):
        """Build from old 7-field intent schema."""
        rows = self.store.conn.execute("""
            SELECT ci.agreement_id, ci.clause_type_id,
                   ci.primary_obligation, ci.obligated_party, ci.benefiting_party
            FROM clause_intents ci
            WHERE ci.primary_obligation IS NOT NULL AND ci.primary_obligation != ''
        """).fetchall()

        intent_types = set()
        parties = set()

        for agr_id, ct_id, prim_obl, obl_party, ben_party in rows:
            short_label = _shorten_obligation(prim_obl)
            it_node = f"{IT}{short_label}"

            if short_label not in intent_types:
                self.G.add_node(it_node, kind="intent_type", label=short_label,
                                full_text=prim_obl[:200])
                self._update_entity_state(it_node, f"Obligation: {prim_obl[:100]}", episode)
                intent_types.add(short_label)

            ct_label = self._ct_id_to_label(ct_id)
            if ct_label:
                ct_node = f"{CT}{ct_label}"
                self._add_or_increment_fact(
                    ct_node, it_node, "HOSTS", episode,
                    weight=1, confidence=0.7,
                    context={"cluster_id": ct_id, "original_text": prim_obl[:200]},
                )

            self._agr_intents[agr_id].add(it_node)
            self._intent_agreements[it_node].add(agr_id)

            for party_name, edge_rel in [(obl_party, "BEARS"),
                                          (ben_party, "BENEFITS_FROM")]:
                pname = _normalize_party(party_name)
                if pname:
                    pty_node = f"{PTY}{pname}"
                    if pname not in parties:
                        self.G.add_node(pty_node, kind="party", name=pname)
                        self._update_entity_state(pty_node, f"Party: {pname}", episode)
                        parties.add(pname)
                    self._add_or_increment_fact(
                        pty_node, it_node, edge_rel, episode,
                        weight=1, confidence=0.7,
                        context={"agreement_id": agr_id},
                    )

        self.stats.intent_types = len(intent_types)
        self.stats.parties = len(parties)

    def _build_co_occurrence_edges(self, episode: Episode):
        """Add CO_OCCURS facts between intents in the same agreement."""
        co_count: Counter = Counter()
        for agr_id, intents in self._agr_intents.items():
            for a, b in combinations(sorted(intents), 2):
                co_count[(a, b)] += 1

        for (a, b), count in co_count.items():
            if count >= 2:
                self._add_or_increment_fact(
                    a, b, "CO_OCCURS", episode,
                    weight=count, confidence=1.0,
                    context={"co_occurrence_count": count},
                    increment=False,
                )

    def _ct_id_to_label(self, ct_id: str | None) -> str | None:
        """Resolve cluster_id to label via graph or DB."""
        if not ct_id:
            return None
        for node, data in self.G.nodes(data=True):
            if data.get("kind") == "clause_type" and data.get("cluster_id") == ct_id:
                return data["label"]
        try:
            row = self.store.conn.execute(
                "SELECT label FROM clusters WHERE cluster_id = ?", [ct_id]
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS — backward-compatible API, operating on active view
    # ══════════════════════════════════════════════════════════════════════

    def recommend(self, agreement_id: str, top_n: int = 10,
                  as_of: str | None = None) -> list[dict]:
        """Recommend missing clause types and intent types for an agreement."""
        recs = []
        recs.extend(self._recommend_clause_types(agreement_id, top_n))
        recs.extend(self._recommend_intent_types(agreement_id, top_n))
        recs.sort(key=lambda r: -r["score"])
        return recs[:top_n]

    def _recommend_clause_types(self, agreement_id: str, top_n: int) -> list[dict]:
        present = self._agr_clause_types.get(agreement_id, set())
        if not present:
            return []

        all_ct = {n for n, d in self.G.nodes(data=True) if d.get("kind") == "clause_type"}
        missing = all_ct - present
        n_agreements = self.stats.agreements

        recs = []
        for ct_node in missing:
            ct_label = self.G.nodes[ct_node].get("label", ct_node)
            agr_with = sum(1 for a, cts in self._agr_clause_types.items()
                           if ct_node in cts)
            if agr_with < 3:
                continue

            overlap_count = 0
            overlap_total = 0
            for other_agr, other_cts in self._agr_clause_types.items():
                if other_agr == agreement_id:
                    continue
                shared = len(present & other_cts)
                if shared >= len(present) * 0.4:
                    overlap_total += 1
                    if ct_node in other_cts:
                        overlap_count += 1

            if overlap_total < 2:
                continue
            score = overlap_count / overlap_total
            if score < 0.3:
                continue

            best_pair = ""
            best_w = 0
            for p in present:
                a_, b_ = tuple(sorted([p, ct_node]))
                if self.G.has_edge(a_, b_):
                    w = self.G[a_][b_].get("weight", 0)
                    if w > best_w:
                        best_w = w
                        best_pair = self.G.nodes[p].get("label", p)

            recs.append({
                "type": "clause_type",
                "label": ct_label,
                "score": round(score, 3),
                "present_in_pct": round(100 * agr_with / max(n_agreements, 1), 1),
                "reason": f"appears in {score:.0%} of similar agreements"
                          + (f" (strongly paired with '{best_pair}')" if best_pair else ""),
            })

        recs.sort(key=lambda r: -r["score"])
        return recs[:top_n]

    def _recommend_intent_types(self, agreement_id: str, top_n: int) -> list[dict]:
        present = self._agr_intents.get(agreement_id, set())
        if not present:
            return []

        all_it = {n for n, d in self.G.nodes(data=True) if d.get("kind") == "intent_type"}
        missing = all_it - present

        recs = []
        for it_node in missing:
            it_agrs = self._intent_agreements.get(it_node, set())
            if len(it_agrs) < 2:
                continue

            overlap_total = 0
            overlap_hit = 0
            candidate_agrs = it_agrs | {a for p in present
                                         for a in self._intent_agreements.get(p, set())}
            for other_agr in candidate_agrs:
                if other_agr == agreement_id:
                    continue
                other_intents = self._agr_intents.get(other_agr, set())
                if present & other_intents:
                    overlap_total += 1
                    if it_node in other_intents:
                        overlap_hit += 1

            if overlap_total < 2:
                continue
            score = overlap_hit / overlap_total
            if score < 0.25:
                continue

            it_label = self.G.nodes[it_node].get("label", it_node)
            recs.append({
                "type": "intent_type",
                "label": it_label,
                "score": round(score, 3),
                "present_in_pct": round(100 * len(it_agrs) / max(self.stats.agreements, 1), 1),
                "reason": f"appears in {score:.0%} of agreements sharing your intents",
            })

        recs.sort(key=lambda r: -r["score"])
        return recs[:top_n]

    # ── Anomaly Detection ────────────────────────────────────────────────

    def detect_anomalies(self, agreement_id: str) -> list[dict]:
        """Flag unusual patterns in an agreement."""
        anomalies = []
        present_ct = self._agr_clause_types.get(agreement_id, set())
        present_it = self._agr_intents.get(agreement_id, set())

        for ct_node in present_ct:
            for _, neighbor, edata in self.G.edges(ct_node, data=True):
                if edata.get("rel") != "CT_CO_OCCURS":
                    continue
                if neighbor not in present_ct and edata.get("weight", 0) >= 5:
                    anomalies.append({
                        "kind": "missing_correlated_type",
                        "severity": "medium",
                        "message": (
                            f"'{self.G.nodes[ct_node].get('label')}' usually appears with "
                            f"'{self.G.nodes[neighbor].get('label')}' "
                            f"({edata['weight']} agreements) but it's missing here"
                        ),
                    })

        for it_node in present_it:
            agr_count = len(self._intent_agreements.get(it_node, set()))
            if 0 < agr_count <= 2:
                anomalies.append({
                    "kind": "rare_intent",
                    "severity": "low",
                    "message": (
                        f"Intent '{self.G.nodes[it_node].get('label')}' "
                        f"appears in only {agr_count} agreement(s) — "
                        "may be unique or a labeling artifact"
                    ),
                })

        agr_node = f"{AGR}{agreement_id}"
        invalidated_recently = [
            f for f in self._facts.values()
            if not f.is_active and (f.source_node == agr_node or f.target_node == agr_node)
        ]
        if invalidated_recently:
            anomalies.append({
                "kind": "invalidated_facts",
                "severity": "info",
                "message": (
                    f"{len(invalidated_recently)} fact(s) involving this agreement "
                    "have been invalidated — may indicate contract amendments or corrections"
                ),
                "fact_ids": [f.fact_id for f in invalidated_recently[:10]],
            })

        return anomalies

    # ── Archetypes ───────────────────────────────────────────────────────

    def discover_archetypes(self, min_agreements: int = 3) -> list[dict]:
        """Cluster agreements by clause-type fingerprints (Jaccard + Louvain)."""
        if not self._agr_clause_types:
            return []

        sim_G = nx.Graph()
        agr_ids = list(self._agr_clause_types.keys())
        for i, a1 in enumerate(agr_ids):
            sim_G.add_node(a1)
            s1 = self._agr_clause_types[a1]
            for a2 in agr_ids[i + 1:]:
                s2 = self._agr_clause_types[a2]
                jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
                if jaccard >= 0.3:
                    sim_G.add_edge(a1, a2, weight=jaccard)

        try:
            communities = nx.community.louvain_communities(
                sim_G, weight="weight", resolution=1.0, seed=42)
        except Exception:
            communities = list(nx.connected_components(sim_G))

        archetypes = []
        for idx, community in enumerate(communities):
            if len(community) < min_agreements:
                continue
            type_counts: Counter = Counter()
            for agr_id in community:
                for ct in self._agr_clause_types.get(agr_id, set()):
                    type_counts[ct] += 1
            defining = [
                self.G.nodes[ct].get("label", ct)
                for ct, cnt in type_counts.most_common()
                if cnt >= len(community) * 0.6
            ]
            label_parts = defining[:2] if defining else ["Mixed"]
            label = " + ".join(label_parts)

            archetypes.append({
                "archetype_id": idx,
                "label": label,
                "agreements": sorted(community),
                "defining_types": defining,
                "size": len(community),
            })

        archetypes.sort(key=lambda a: -a["size"])
        return archetypes

    # ── Implication Rules ────────────────────────────────────────────────

    def get_implication_rules(self, min_confidence: float = 0.7,
                               min_support: int = 3) -> list[dict]:
        """Find directed implication rules: "if type A → type B with P%"."""
        rules = []
        n = self.stats.agreements
        if n < 5:
            return rules

        ct_agr_count = {}
        for ct_node in (nd for nd, d in self.G.nodes(data=True)
                        if d.get("kind") == "clause_type"):
            ct_agr_count[ct_node] = sum(
                1 for cts in self._agr_clause_types.values() if ct_node in cts)

        for a, b in combinations(ct_agr_count.keys(), 2):
            both = sum(1 for cts in self._agr_clause_types.values()
                       if a in cts and b in cts)
            if both < min_support:
                continue

            expected = ct_agr_count[a] * ct_agr_count[b] / n
            lift = both / expected if expected else 0

            conf_ab = both / ct_agr_count[a] if ct_agr_count[a] else 0
            if conf_ab >= min_confidence:
                rules.append({
                    "antecedent": self.G.nodes[a].get("label", a),
                    "consequent": self.G.nodes[b].get("label", b),
                    "confidence": round(conf_ab, 3),
                    "support": both,
                    "lift": round(lift, 2),
                    "level": "clause_type",
                })

            conf_ba = both / ct_agr_count[b] if ct_agr_count[b] else 0
            if conf_ba >= min_confidence:
                rules.append({
                    "antecedent": self.G.nodes[b].get("label", b),
                    "consequent": self.G.nodes[a].get("label", a),
                    "confidence": round(conf_ba, 3),
                    "support": both,
                    "lift": round(lift, 2),
                    "level": "clause_type",
                })

        rules.sort(key=lambda r: (-r["confidence"], -r["support"]))
        return rules

    # ── Strategy Detection ───────────────────────────────────────────────

    def discover_strategies(self, min_cluster_size: int = 3) -> list[dict]:
        """Discover legal strategies — tightly co-occurring groups of types/intents."""
        co_edges = [(u, v, d) for u, v, d in self.G.edges(data=True)
                    if d.get("rel") in ("CT_CO_OCCURS", "CO_OCCURS")
                    and d.get("weight", 0) >= 2]
        if len(co_edges) < 3:
            return []

        co_G = nx.Graph()
        for u, v, d in co_edges:
            co_G.add_edge(u, v, weight=d["weight"])

        try:
            communities = nx.community.louvain_communities(
                co_G, weight="weight", resolution=0.8, seed=42)
        except Exception:
            communities = list(nx.connected_components(co_G))

        strategies = []
        for idx, community in enumerate(communities):
            if len(community) < min_cluster_size:
                continue

            members = sorted(community)
            labels = [self.G.nodes[n].get("label", n.split(":", 1)[-1])
                      for n in members]

            internal_weight = sum(
                co_G[u][v]["weight"]
                for u, v in combinations(members, 2)
                if co_G.has_edge(u, v)
            )
            max_edges = len(members) * (len(members) - 1) / 2
            cohesion = internal_weight / max_edges if max_edges else 0

            agr_using = 0
            ct_members = {m for m in members if m.startswith(CT)}
            if ct_members:
                for agr_id, cts in self._agr_clause_types.items():
                    if ct_members <= cts:
                        agr_using += 1

            sub = co_G.subgraph(members)
            centrality = nx.degree_centrality(sub)
            top_labels = [self.G.nodes[n].get("label", n.split(":", 1)[-1])
                          for n, _ in sorted(centrality.items(),
                                              key=lambda x: -x[1])[:3]]

            strategies.append({
                "strategy_id": idx,
                "label": " / ".join(top_labels),
                "members": labels,
                "cohesion": round(cohesion, 3),
                "agreements_using": agr_using,
            })

        strategies.sort(key=lambda s: (-s["cohesion"], -len(s["members"])))
        return strategies

    # ══════════════════════════════════════════════════════════════════════
    # SIMILARITY — embedding-based intent similarity
    # ══════════════════════════════════════════════════════════════════════

    def add_intent_similarity(self, embedder, similarity_threshold: float = 0.82):
        """Embed intent types and add SIMILAR_INTENT facts."""
        it_nodes = [(n, d) for n, d in self.G.nodes(data=True)
                    if d.get("kind") == "intent_type"]
        if len(it_nodes) < 2:
            return 0

        episode = Episode(
            episode_type="pipeline_run",
            source="intent_similarity",
            metadata={"threshold": similarity_threshold},
        )
        self._register_episode(episode)

        texts = []
        node_ids = []
        for node_id, data in it_nodes:
            label = data.get("label", "").replace("_", " ")
            full_text = data.get("full_text", "")
            summary = data.get("summary", "")
            text = f"{label}: {summary or full_text}".strip()
            texts.append(text)
            node_ids.append(node_id)

        embeddings = embedder.embed(texts, prefix="search_document",
                                     normalize=True, batch_size=64)
        sim_matrix = embeddings @ embeddings.T

        n_added = 0
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                sim = float(sim_matrix[i, j])
                if sim >= similarity_threshold:
                    self._add_or_increment_fact(
                        node_ids[i], node_ids[j], "SIMILAR_INTENT", episode,
                        weight=round(sim, 4), confidence=sim,
                        increment=False,
                    )
                    n_added += 1

        logger.info(f"Added {n_added} SIMILAR_INTENT facts "
                     f"(threshold={similarity_threshold})")
        self._intent_similarity_clusters = self._cluster_similar_intents()
        return n_added

    def _cluster_similar_intents(self) -> list[set[str]]:
        """Connected components in the SIMILAR_INTENT subgraph."""
        sim_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                     if d.get("rel") == "SIMILAR_INTENT"]
        if not sim_edges:
            return []
        sim_G = nx.Graph()
        sim_G.add_edges_from(sim_edges)
        return [c for c in nx.connected_components(sim_G) if len(c) >= 2]

    def get_intent_similarity_clusters(self) -> list[dict]:
        """Return similarity clusters with labels and statistics."""
        clusters = getattr(self, "_intent_similarity_clusters", None)
        if not clusters:
            clusters = self._cluster_similar_intents()

        results = []
        for cluster in clusters:
            labels = sorted(
                self.G.nodes[n].get("label", n.split(":", 1)[-1])
                for n in cluster
            )
            best_label = labels[0]
            best_count = 0
            for n in cluster:
                count = len(self._intent_agreements.get(n, set()))
                if count > best_count:
                    best_count = count
                    best_label = self.G.nodes[n].get("label", n.split(":", 1)[-1])

            all_agrs = set()
            for n in cluster:
                all_agrs |= self._intent_agreements.get(n, set())

            results.append({
                "canonical": best_label,
                "members": labels,
                "size": len(cluster),
                "agreements_covered": len(all_agrs),
            })

        results.sort(key=lambda r: (-r["size"], -r["agreements_covered"]))
        return results

    # ══════════════════════════════════════════════════════════════════════
    # AGREEMENT COMPARISON
    # ══════════════════════════════════════════════════════════════════════

    def compare_agreements(self, agr_a: str, agr_b: str) -> dict:
        """Deep comparison of two agreements using intent + similarity edges."""
        intents_a = self._agr_intents.get(agr_a, set())
        intents_b = self._agr_intents.get(agr_b, set())

        shared = intents_a & intents_b
        only_a = intents_a - intents_b
        only_b = intents_b - intents_a

        similar_pairs = []
        matched_a, matched_b = set(), set()
        for a_node in only_a:
            for b_node in only_b:
                if self.G.has_edge(a_node, b_node):
                    edge = self.G[a_node][b_node]
                    if edge.get("rel") == "SIMILAR_INTENT":
                        similar_pairs.append({
                            "a": self.G.nodes[a_node].get("label", a_node),
                            "b": self.G.nodes[b_node].get("label", b_node),
                            "similarity": edge.get("weight", 0),
                        })
                        matched_a.add(a_node)
                        matched_b.add(b_node)

        truly_only_a = only_a - matched_a
        truly_only_b = only_b - matched_b

        matched_count = len(shared) + len(similar_pairs)
        union_count = len(intents_a | intents_b)
        sim_score = matched_count / union_count if union_count else 0

        _label = lambda n: self.G.nodes[n].get("label", n.split(":", 1)[-1])

        return {
            "shared_intents": sorted(_label(n) for n in shared),
            "similar_pairs": sorted(similar_pairs, key=lambda p: -p["similarity"]),
            "only_in_a": sorted(_label(n) for n in truly_only_a),
            "only_in_b": sorted(_label(n) for n in truly_only_b),
            "similarity_score": round(sim_score, 3),
            "n_shared": len(shared),
            "n_similar": len(similar_pairs),
            "n_only_a": len(truly_only_a),
            "n_only_b": len(truly_only_b),
        }

    def find_most_similar_agreements(self, agreement_id: str,
                                      top_n: int = 5) -> list[dict]:
        """Find top-N most similar agreements by intent fingerprint."""
        my_intents = self._agr_intents.get(agreement_id, set())
        if not my_intents:
            return []

        scores = []
        for other_id in self._agr_intents:
            if other_id == agreement_id:
                continue
            comparison = self.compare_agreements(agreement_id, other_id)
            scores.append({
                "agreement_id": other_id,
                "similarity_score": comparison["similarity_score"],
                "n_shared": comparison["n_shared"],
                "n_similar": comparison["n_similar"],
                "n_only_other": comparison["n_only_b"],
            })

        scores.sort(key=lambda s: -s["similarity_score"])
        return scores[:top_n]

    # ══════════════════════════════════════════════════════════════════════
    # ARM INTEGRATION — association rule mining relationships as graph edges
    # ══════════════════════════════════════════════════════════════════════

    def add_arm_relationships(self, relationship_layer) -> int:
        """Add ARM-discovered relationships as graph edges.
        Returns the number of edges added.
        """
        episode = Episode(
            episode_type="pipeline_run",
            source="arm_relationships",
            metadata={"method": "add_arm_relationships"},
        )
        self._register_episode(episode)

        edges_added = 0
        rules = relationship_layer._load_rules()
        for rule in rules:
            if rule["rule_type"] == "clause_dependency":
                for ant in rule["antecedent"]:
                    for con in rule["consequent"]:
                        ant_node = f"{CT}{ant}"
                        con_node = f"{CT}{con}"
                        if not self.G.has_node(ant_node):
                            self.G.add_node(ant_node, kind="clause_type", label=ant)
                        if not self.G.has_node(con_node):
                            self.G.add_node(con_node, kind="clause_type", label=con)
                        self._add_or_increment_fact(
                            source=ant_node,
                            target=con_node,
                            relation="DEPENDS_ON",
                            episode=episode,
                            weight=rule["confidence"],
                            confidence=rule["confidence"],
                            context={
                                "confidence": rule["confidence"],
                                "lift": rule["lift"],
                                "support": rule["support"],
                                "rule_id": rule["rule_id"],
                            },
                        )
                        edges_added += 1
            elif rule["rule_type"] in ("field_correlation", "cross_clause_field"):
                for ant in rule["antecedent"]:
                    for con in rule["consequent"]:
                        self._add_or_increment_fact(
                            source=ant,
                            target=con,
                            relation="FIELD_CORRELATES",
                            episode=episode,
                            weight=rule["confidence"],
                            confidence=rule["confidence"],
                            context={
                                "lift": rule["lift"],
                                "confidence": rule["confidence"],
                                "rule_id": rule["rule_id"],
                            },
                        )
                        edges_added += 1

        packages = relationship_layer.get_term_packages()
        for pkg in packages:
            pkg_node = f"TP:{','.join(pkg['clause_types'][:3])}"
            if not self.G.has_node(pkg_node):
                self.G.add_node(pkg_node, kind="term_package",
                                package_id=pkg["package_id"],
                                support=pkg["support"])
            for ct in pkg["clause_types"]:
                ct_node = f"{CT}{ct}"
                if not self.G.has_node(ct_node):
                    self.G.add_node(ct_node, kind="clause_type", label=ct)
                self._add_or_increment_fact(
                    source=ct_node,
                    target=pkg_node,
                    relation="BUNDLED_WITH",
                    episode=episode,
                    weight=pkg["support"],
                    confidence=1.0,
                    context={
                        "support": pkg["support"],
                        "package_size": pkg["size"],
                        "package_id": pkg["package_id"],
                    },
                )
                edges_added += 1

        logger.info(f"Added {edges_added} ARM relationship edges to context graph")
        return edges_added

    # ══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    def get_agreement_fingerprint(self, agreement_id: str) -> dict:
        """Return the full fingerprint: types, intents, parties, and provenance."""
        agr_node = f"{AGR}{agreement_id}"
        related_episodes = [
            f.episode_id for f in self._facts.values()
            if f.source_node == agr_node or f.target_node == agr_node
        ]

        return {
            "agreement_id": agreement_id,
            "clause_types": sorted(
                self.G.nodes[ct].get("label", ct)
                for ct in self._agr_clause_types.get(agreement_id, set())
            ),
            "intents": sorted(
                self.G.nodes[it].get("label", it)
                for it in self._agr_intents.get(agreement_id, set())
            ),
            "n_clause_types": len(self._agr_clause_types.get(agreement_id, set())),
            "n_intents": len(self._agr_intents.get(agreement_id, set())),
            "episode_count": len(set(related_episodes)),
            "timeline": self.get_entity_timeline(agr_node),
        }

    def export_temporal_facts(self, active_only: bool = False) -> list[dict]:
        """Export all facts as serializable dicts."""
        facts = []
        for f in self._facts.values():
            if active_only and not f.is_active:
                continue
            facts.append({
                "fact_id": f.fact_id,
                "source": f.source_node,
                "target": f.target_node,
                "relation": f.relation,
                "weight": f.weight,
                "confidence": f.confidence,
                "valid_from": f.valid_from,
                "valid_until": f.valid_until,
                "is_active": f.is_active,
                "episode_id": f.episode_id,
                "invalidated_by": f.invalidated_by,
                "context": f.context,
            })
        facts.sort(key=lambda f: f["valid_from"])
        return facts

    def summary(self) -> str:
        """Human-readable summary of the context graph."""
        lines = [
            "Context Graph Summary",
            f"  Agreements:        {self.stats.agreements}",
            f"  Clause types:      {self.stats.clause_types}",
            f"  Intent types:      {self.stats.intent_types}",
            f"  Party nodes:       {self.stats.parties}",
            f"  Episodes:          {self.stats.episodes}",
            f"  Total facts:       {self.stats.total_facts}",
            f"  Active facts:      {self.stats.active_facts}",
            f"  Invalidated facts: {self.stats.invalidated_facts}",
            f"  Co-occur edges:    {self.stats.co_occurrence_edges}",
            f"  Entity timelines:  {len(self._entity_states)}",
            f"  Activity signals:  {self.stats.activity_signals}",
            f"  Process traces:    {self.stats.process_traces}",
        ]
        return "\n".join(lines)


# ── Backward compatibility ───────────────────────────────────────────────────
ContractKnowledgeGraph = ContractContextGraph


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_party(name: str | None) -> str | None:
    """Normalize party name for node identity."""
    if not name:
        return None
    name = name.strip().lower()
    if name in ("none", "n/a", "null", "", "both", "both parties",
                "not specified", "unspecified"):
        return None
    name = name.replace("the ", "").replace("  ", " ").strip()
    if len(name) > 50:
        name = name[:50]
    return name


def _shorten_obligation(text: str) -> str:
    """Create a rough label from free-text primary_obligation (old schema)."""
    text = text.strip()[:120].lower()
    text = re.sub(r'\b(shall|must|will|agrees? to|is required to|hereby)\b', '', text)
    words = text.split()[:6]
    label = "_".join(w for w in words if len(w) > 2)
    return label[:60] if label else "unnamed_obligation"
