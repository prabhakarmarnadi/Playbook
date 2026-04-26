"""
knowledge_graph.py — backward-compatible re-export shim.
=========================================================

The full implementation now lives in ``context_graph.py``.
This file re-exports every public name so that existing imports like

    from core.knowledge_graph import ContractKnowledgeGraph

continue to work without changes.
"""
from core.context_graph import (  # noqa: F401
    # Primitives
    Episode,
    TemporalFact,
    EntityState,
    GraphStats,
    ActivitySignal,
    ProcessTrace,
    # Main class
    ContractContextGraph,
    ContractKnowledgeGraph,
    # Node prefixes
    AGR,
    CT,
    IT,
    PTY,
    EP,
    USR,
    PROC,
    # Helpers (private but used externally in some scripts)
    _normalize_party,
    _shorten_obligation,
    _now,
    _parse_ts,
)

__all__ = [
    "Episode",
    "TemporalFact",
    "EntityState",
    "GraphStats",
    "ActivitySignal",
    "ProcessTrace",
    "ContractContextGraph",
    "ContractKnowledgeGraph",
    "AGR",
    "CT",
    "IT",
    "PTY",
    "EP",
    "USR",
    "PROC",
]
