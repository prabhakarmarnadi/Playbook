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
"""
from __future__ import annotations
import uuid
from typing import Any, Callable, Optional

import numpy as np

from .predicates.evaluator import evaluate
from .store import PlaybookStore


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
    a = np.asarray(embed(target))
    b = np.asarray(embed(ref))
    if a.size == 0 or b.size == 0:
        return None
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return max(0.0, 1.0 - cos)


def _eval_nl(rule: dict, ctx: dict) -> Optional[tuple[str, str]]:
    assertion = rule.get("nl_assertion")
    judge: Optional[Callable] = ctx.get("nl_judge")
    if not assertion or judge is None:
        return None
    evidence = ctx.get("clause_text") or "\n".join(
        c.get("text", "") for c in ctx.get("clauses", [])[:5]
    )
    return judge(assertion, evidence)


def align(store: PlaybookStore, playbook_id: str, ctx: dict) -> list[dict]:
    run_id = ctx.get("run_id") or str(uuid.uuid4())
    agreement_id = ctx["agreement_id"]
    out: list[dict] = []

    rules = [r for r in store.list_rules(playbook_id) if r["status"] == "active"]
    for rule in rules:
        evaluator_used = _evaluator_used(rule)
        outcomes: list[str] = []
        rationale_parts: list[str] = []
        deviation = 0.0
        answer_value: dict[str, Any] = {}

        # 1. predicate
        if evaluator_used["predicate"]:
            ok = _eval_predicate(rule, ctx)
            if ok is True:
                outcomes.append("pass")
            elif ok is False:
                outcomes.append("fail")
            else:
                outcomes.append("needs_human")
                rationale_parts.append("predicate inconclusive")

        # 2. similarity
        if evaluator_used["similarity"]:
            d = _eval_similarity(rule, ctx)
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
            res = _eval_nl(rule, ctx)
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
                    for c in ctx.get("clauses", [])[:3]]

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
