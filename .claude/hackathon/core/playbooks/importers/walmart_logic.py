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


def _strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    return expr  # the leading ( closes before the end → not a wrapping pair
        return expr[1:-1].strip()
    return expr


def _split_top_level(expr: str, sep: str) -> list:
    """Split `expr` on `sep` (case-insensitive), but only at paren depth 0."""
    parts = []
    depth = 0
    i = 0
    sep_lower = sep.lower()
    sep_len = len(sep)
    last = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
            i += 1
        elif ch == ")":
            depth -= 1
            i += 1
        elif depth == 0 and expr[i: i + sep_len].lower() == sep_lower and (
            i == 0 or not expr[i - 1].isalnum()
        ) and (i + sep_len >= len(expr) or not expr[i + sep_len].isalnum()):
            parts.append(expr[last:i])
            i += sep_len
            last = i
        else:
            i += 1
    parts.append(expr[last:])
    return [p.strip() for p in parts if p.strip()]


def _to_predicate(expr: str) -> dict:
    expr = _strip_outer_parens(expr)
    or_parts = _split_top_level(expr, "OR")
    if len(or_parts) > 1:
        return {"op": "or", "args": [_to_predicate(p) for p in or_parts]}
    and_parts = _split_top_level(expr, "AND")
    if len(and_parts) > 1:
        return {"op": "and", "args": [_to_predicate(p) for p in and_parts]}
    # leaf: a token (possibly with `:*` wildcard suffix)
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
