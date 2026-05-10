"""Playbook alignment engine.

Inputs:  PlaybookStore, playbook_id, ctx
Output:  list of Findings dicts (also persisted to rule_evaluations)

ctx is the same shape consumed by the predicate evaluator, plus:
  agreement_id, run_id (auto-generated if absent), embed (callable, optional),
  nl_judge (callable[(assertion:str, evidence:str) -> ("pass"|"fail"|"n/a", rationale)],
            optional).

Combiner policy:
  - If any evaluator says fail   → outcome = fail
  - If all set evaluators pass   → outcome = pass
  - Otherwise                    → outcome = needs_human

Scoping (§8 step 2):
  applies_to=document    — full ctx, no filter
  applies_to=domain      — fire only if ctx["domain"]["id"] matches a binding entity_id
  applies_to=cluster     — restrict ctx["clauses"] to those with cluster_id in bindings
  applies_to=field       — restrict ctx["fields"] to keys in bindings
  applies_to=composite   — restrict ctx["composites"] to keys in bindings
  applies_to=cross_field — full ctx (predicate references multiple fields directly)

  Bindings with entity_id starting "label:" are unresolved soft bindings and
  never match real ontology IDs. A rule whose only bindings are unresolved
  reports outcome="n/a".

  Legacy fallback: a rule with ZERO bindings and a non-document applies_to scope
  is treated as applies_to=document so that tests written before binding support
  was added continue to pass unchanged.
"""
from __future__ import annotations
import uuid
from typing import Any, Callable, Optional

import numpy as np

from .predicates.evaluator import evaluate
from .store import PlaybookStore


def _scope_ctx(rule: dict, bindings: list[dict], ctx: dict) -> Optional[dict]:
    """Return a NEW ctx dict scoped to the rule's applies_to + bindings.

    Returns None when:
    - All bindings are unresolved (entity_id starts with "label:"), OR
    - applies_to=domain and ctx's domain id doesn't match any resolved binding.

    Never mutates the input ctx.
    """
    applies_to = rule.get("applies_to") or "document"

    # Partition bindings into resolved vs. unresolved (soft).
    resolved = [b for b in bindings if not b["entity_id"].startswith("label:")]
    unresolved = [b for b in bindings if b["entity_id"].startswith("label:")]

    # If every binding is unresolved, this rule has no real ontology anchor → n/a.
    if bindings and not resolved:
        return None

    # No-op scopes: document and cross_field pass ctx through unchanged.
    if applies_to in ("document", "cross_field"):
        return ctx

    resolved_ids = {b["entity_id"] for b in resolved}

    if applies_to == "domain":
        domain = ctx.get("domain") or {}
        if domain.get("id") in resolved_ids:
            return ctx
        # Domain doesn't match any resolved binding → out of scope.
        return None

    if applies_to == "cluster":
        all_clauses = ctx.get("clauses") or []
        filtered = [c for c in all_clauses if c.get("cluster_id") in resolved_ids]
        # Return a shallow copy with filtered clauses; even if empty, let the
        # predicate decide — the predicate may legitimately report "no clauses found".
        return {**ctx, "clauses": filtered}

    if applies_to in ("field", "composite"):
        src_key = "fields" if applies_to == "field" else "composites"
        src = ctx.get(src_key) or {}
        filtered = {k: v for k, v in src.items() if k in resolved_ids}
        return {**ctx, src_key: filtered}

    # Unknown applies_to value — pass through unchanged rather than hard-failing.
    return ctx


def _evaluator_used(rule: dict) -> dict:
    return {
        "predicate": rule.get("predicate") is not None,
        "similarity": bool(rule.get("reference_text")),
        "nl": bool(rule.get("nl_assertion")),
    }


def _eval_predicate(rule: dict, ctx: dict) -> Optional[bool]:
    pred = rule.get("predicate")
    if not pred:
        return None
    try:
        return bool(evaluate(pred, ctx))
    except Exception:
        return None


def _eval_similarity(rule: dict, ctx: dict) -> Optional[float]:
    """Returns deviation in [0,1] (0=aligned to reference, 1=most deviating).
    None if the rule has no reference or ctx has no embed callable."""
    ref = rule.get("reference_text")
    embed: Optional[Callable] = ctx.get("embed")
    if not ref or embed is None:
        return None
    target = ctx.get("clause_text") or " ".join(
        c.get("text", "") for c in ctx.get("clauses", [])[:3]
    )
    if not target:
        return None
    try:
        a = np.asarray(embed(target))
        b = np.asarray(embed(ref))
        if a.size == 0 or b.size == 0:
            return None
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return max(0.0, 1.0 - cos)
    except Exception:
        return None


def _eval_nl(rule: dict, ctx: dict) -> Optional[tuple[str, str]]:
    assertion = rule.get("nl_assertion")
    judge: Optional[Callable] = ctx.get("nl_judge")
    if not assertion or judge is None:
        return None
    evidence = ctx.get("clause_text") or "\n".join(
        c.get("text", "") for c in ctx.get("clauses", [])[:5]
    )
    try:
        return judge(assertion, evidence)
    except Exception:
        return None


def align(store: PlaybookStore, playbook_id: str, ctx: dict) -> list[dict]:
    run_id = ctx.get("run_id") or str(uuid.uuid4())
    agreement_id = ctx["agreement_id"]
    out: list[dict] = []

    rules = [r for r in store.list_rules(playbook_id) if r["status"] == "active"]
    for rule in rules:
        bindings = store.bindings_for(rule["rule_id"])

        # Legacy fallback: a rule with no bindings AND a non-document applies_to scope
        # behaves like applies_to=document so tests written before binding support was
        # added continue to pass without constructing bindings in their setup.
        if not bindings and rule.get("applies_to") not in (None, "document"):
            scoped_ctx = ctx
        else:
            scoped_ctx = _scope_ctx(rule, bindings, ctx)

        if scoped_ctx is None:
            # All bindings unresolved, or domain/scope filter found no match → n/a.
            eid = store.record_eval(
                rule_id=rule["rule_id"], agreement_id=agreement_id, run_id=run_id,
                outcome="n/a", severity=rule["severity"], deviation=0.0,
                rationale="rule out of scope for this agreement",
                evaluator_used={"predicate": False, "similarity": False, "nl": False},
            )
            out.append({"eval_id": eid, "rule_id": rule["rule_id"], "outcome": "n/a",
                        "severity": rule["severity"], "deviation": 0.0, "evidence": []})
            continue

        evaluator_used = _evaluator_used(rule)
        outcomes: list[str] = []
        rationale_parts: list[str] = []
        deviation = 0.0
        answer_value: dict[str, Any] = {}

        # 1. predicate
        if evaluator_used["predicate"]:
            ok = _eval_predicate(rule, scoped_ctx)
            if ok is True:
                outcomes.append("pass")
            elif ok is False:
                outcomes.append("fail")
            else:
                outcomes.append("needs_human")
                rationale_parts.append("predicate inconclusive")

        # 2. similarity
        if evaluator_used["similarity"]:
            d = _eval_similarity(rule, scoped_ctx)
            if d is None:
                pass
            else:
                deviation = max(deviation, d)
                thr = rule.get("similarity_threshold") or 0.15
                if d > (1.0 - thr):
                    outcomes.append("fail")
                else:
                    outcomes.append("pass")

        # 3. NL judge — only if predicate inconclusive or absent
        run_nl = evaluator_used["nl"] and (
            (not evaluator_used["predicate"]) or "needs_human" in outcomes
        )
        if run_nl:
            res = _eval_nl(rule, scoped_ctx)
            if res is not None:
                verdict, why = res
                outcomes.append("fail" if verdict == "fail"
                                 else "pass" if verdict == "pass" else "needs_human")
                rationale_parts.append(why)

        # combiner
        if "fail" in outcomes:
            outcome = "fail"
        elif outcomes and all(o == "pass" for o in outcomes):
            outcome = "pass"
        else:
            outcome = "needs_human"

        evidence = [{"clause_id": c.get("id"),
                     "text": (c.get("text") or "")[:300]}
                    for c in scoped_ctx.get("clauses", [])[:3]]

        eid = store.record_eval(
            rule_id=rule["rule_id"], agreement_id=agreement_id, run_id=run_id,
            outcome=outcome, severity=rule["severity"], deviation=deviation,
            answer_value=answer_value or None, evidence=evidence,
            rationale=" / ".join(rationale_parts) or None,
            evaluator_used=evaluator_used,
        )
        out.append({"eval_id": eid, "rule_id": rule["rule_id"],
                    "outcome": outcome, "severity": rule["severity"],
                    "deviation": deviation, "evidence": evidence})

    return out
