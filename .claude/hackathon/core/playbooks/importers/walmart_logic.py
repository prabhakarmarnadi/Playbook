"""Parse Walmart-style 'Deployment Logic' strings into predicate JSON.

Grammar covered (per Walmart Custom Extractions xlsx):
  '<n>) IF <expr> IS SET --> Answer = <ans>'
  '<n>) IF (<a> OR <b>) IS SET --> Answer = <ans>'
  '<n>) IF <a> AND <b> IS SET --> Answer = <ans>'
  '<n>) Else --> Answer = <ans>'

The :* suffix on field names is stripped (it's a wildcard for sub-properties).
"""
from __future__ import annotations
import re

CLAUSE_RE = re.compile(
    r"(?:\d+\)\s*)?(?:IF\s+(?P<expr>.+?)\s+IS\s+SET|Else)\s*-->?\s*Answer\s*=\s*(?P<ans>[^\n]+)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?::\*)?")


def _to_predicate(expr: str) -> dict:
    expr = expr.strip().strip("()").strip()
    upper = expr.upper()
    # OR has lower precedence than AND
    if " OR " in upper:
        parts = re.split(r"\s+OR\s+", expr, flags=re.IGNORECASE)
        return {"op": "or", "args": [_to_predicate(p) for p in parts]}
    if " AND " in upper:
        parts = re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)
        return {"op": "and", "args": [_to_predicate(p) for p in parts]}
    m = TOKEN_RE.match(expr)
    if not m:
        return {"op": "field.exists", "args": [expr]}
    name = m.group(0).rstrip(":*")
    return {"op": "field.exists", "args": [name]}


def parse(text: str) -> list[dict]:
    out = []
    for m in CLAUSE_RE.finditer(text or ""):
        ans = (m.group("ans") or "").strip()
        expr = m.group("expr")
        pred = _to_predicate(expr) if expr else {"op": "and", "args": []}  # else
        out.append({"predicate": pred, "answer": ans, "is_else": expr is None})
    return out
