"""JSON s-expression evaluator. Each node = {op, args}.

ctx schema (loose):
  {
    "fields":   {field_name: value, ...}     # for field.* ops
    "clauses":  [{"label": str, "text": str, "embedding": list[float]?}, ...]
    "domain":   {"id": str, "name": str}
    "doc":      {"value": float, "party_role": str, ...}
  }
"""
from __future__ import annotations
from typing import Any

# Side-effect imports register operators
from . import field_ops, clause_ops, logical_ops, context_ops  # noqa: F401
from . import get


def evaluate(node: Any, ctx: dict) -> Any:
    if not isinstance(node, dict) or "op" not in node:
        return node  # literal pass-through (numbers, strings, lists)
    op = node["op"]
    args = node.get("args", [])
    return get(op)(args, ctx, evaluate)
