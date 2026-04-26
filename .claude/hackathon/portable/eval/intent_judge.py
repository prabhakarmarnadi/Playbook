"""
LLM-as-a-Judge for Intent Discovery evaluation.

Scores each intent on two axes:
  - Faithfulness: Is the intent actually present in the source clause text?
  - Actionability: Would a legal ops team act on this insight?

Also produces a composite "Healthy Insight" score per agreement.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_JUDGE_SYSTEM = """\
You are a senior legal contract analyst acting as an evaluation judge.
You evaluate machine-generated "intents" extracted from legal clauses.
You MUST respond with valid JSON only. No commentary outside the JSON."""

INTENT_JUDGE_PROMPT = """\
## Task
Evaluate the following machine-extracted intent against its source clause text.
Score the intent on two dimensions, each from 1-5.

## Scoring Rubric

### Faithfulness (Is this intent actually in the clause?)
- **5**: Intent is explicitly stated in the clause text word-for-word or with trivial paraphrase.
- **4**: Intent is clearly implied by the clause language. A lawyer would agree it's there.
- **3**: Intent is a reasonable inference but requires domain knowledge to connect.
- **2**: Intent is loosely related to the clause but overstates or distorts the meaning.
- **1**: Intent is hallucinated — not supported by the clause text at all.

### Actionability (Would a legal ops team use this?)
- **5**: Directly triggers a workflow (e.g., "30-day termination notice required" → set calendar reminder).
- **4**: Identifies a concrete obligation, right, or constraint that needs tracking.
- **3**: Provides useful context but doesn't map to a specific action.
- **2**: Too vague or generic to be useful (e.g., "parties have obligations").
- **1**: Noise — restates the clause title or provides no new information.

## Source Clause Text
```
{clause_text}
```

## Intent to Evaluate
- **Label**: {intent_label}
- **Summary**: {intent_summary}
- **Party From**: {party_from}
- **Party To**: {party_to}

## Response Format
Return ONLY this JSON:
{{
  "faithfulness_score": <1-5>,
  "faithfulness_reason": "<one sentence explaining the score>",
  "actionability_score": <1-5>,
  "actionability_reason": "<one sentence explaining the score>",
  "is_hallucinated": <true/false>,
  "suggested_fix": "<empty string if fine, else a corrected intent label>"
}}"""

# Batch variant — judge multiple intents for one clause in a single call
INTENT_JUDGE_BATCH_PROMPT = """\
## Task
Evaluate each of the following machine-extracted intents against the source clause.
Score each intent on Faithfulness (1-5) and Actionability (1-5) using the rubric below.

### Faithfulness (Is this intent actually in the clause?)
5=explicitly stated, 4=clearly implied, 3=reasonable inference, 2=loosely related/distorted, 1=hallucinated

### Actionability (Would a legal ops team use this?)
5=triggers workflow, 4=concrete obligation/right, 3=useful context, 2=too vague, 1=noise

## Source Clause Text
```
{clause_text}
```

## Intents to Evaluate
{intents_json}

## Response Format
Return ONLY a JSON array with one object per intent, in the same order:
[
  {{
    "intent_label": "<echo the label>",
    "faithfulness_score": <1-5>,
    "actionability_score": <1-5>,
    "is_hallucinated": <true/false>
  }}
]"""


@dataclass
class IntentJudge:
    """
    LLM-as-a-Judge for intent quality evaluation.

    Uses the pipeline's own LLM client to evaluate intents
    against their source clause text.
    """

    db_path: str
    llm_client: Any = None  # LLMClient instance
    sample_size: int = 50   # number of clauses to judge (cost control)
    batch_mode: bool = True  # batch intents per clause in one LLM call
    _results: list[dict] = field(default_factory=list, init=False)

    def evaluate(self) -> dict[str, Any]:
        """Run the judge on a sample of clauses with intents."""
        if self.llm_client is None:
            logger.warning("No LLM client provided — skipping intent judge")
            return self._empty_metrics()

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._run_judge(conn)
        finally:
            conn.close()

    def _run_judge(self, conn: duckdb.DuckDBPyConnection) -> dict:
        # Sample clauses that have intents
        clause_rows = conn.execute(f"""
            SELECT DISTINCT ci.clause_id, cl.full_text
            FROM clause_intents ci
            JOIN clauses cl ON ci.clause_id = cl.clause_id
            WHERE LENGTH(cl.full_text) >= 50
            ORDER BY RANDOM()
            LIMIT {self.sample_size}
        """).fetchall()

        if not clause_rows:
            return self._empty_metrics()

        clause_ids = [r[0] for r in clause_rows]
        clause_texts = {r[0]: r[1] for r in clause_rows}

        # Get intents for these clauses
        placeholders = ",".join(["?"] * len(clause_ids))
        intent_rows = conn.execute(f"""
            SELECT clause_id, intent_label, summary, party_from, party_to
            FROM clause_intents
            WHERE clause_id IN ({placeholders})
            ORDER BY clause_id
        """, clause_ids).fetchall()

        # Group intents by clause
        clause_intents: dict[str, list[dict]] = {}
        for cid, label, summary, pf, pt in intent_rows:
            clause_intents.setdefault(cid, []).append({
                "intent_label": label,
                "summary": summary or "",
                "party_from": pf or "unspecified",
                "party_to": pt or "unspecified",
            })

        self._results = []
        n_judged = 0
        n_errors = 0

        for cid in clause_ids:
            if cid not in clause_intents:
                continue
            intents = clause_intents[cid]
            clause_text = clause_texts[cid][:3000]  # cap context

            if self.batch_mode:
                scores = self._judge_batch(clause_text, intents)
            else:
                scores = [
                    self._judge_single(clause_text, intent)
                    for intent in intents
                ]

            for s in scores:
                if s is not None:
                    self._results.append(s)
                    n_judged += 1
                else:
                    n_errors += 1

        logger.info(
            f"Intent judge: {n_judged} intents judged, {n_errors} errors, "
            f"across {len(clause_ids)} clauses"
        )
        return self._compute_metrics()

    def _judge_single(self, clause_text: str, intent: dict) -> dict | None:
        prompt = INTENT_JUDGE_PROMPT.format(
            clause_text=clause_text,
            intent_label=intent["intent_label"],
            intent_summary=intent["summary"],
            party_from=intent["party_from"],
            party_to=intent["party_to"],
        )
        try:
            result = self.llm_client.complete_json(
                prompt, system=INTENT_JUDGE_SYSTEM, temperature=0.0
            )
            if "parse_error" in result:
                return None
            result["intent_label"] = intent["intent_label"]
            return result
        except Exception as e:
            logger.warning(f"Judge call failed: {e}")
            return None

    def _judge_batch(self, clause_text: str, intents: list[dict]) -> list[dict | None]:
        intents_json = json.dumps(
            [
                {"intent_label": i["intent_label"], "summary": i["summary"],
                 "party_from": i["party_from"], "party_to": i["party_to"]}
                for i in intents
            ],
            indent=2,
        )
        prompt = INTENT_JUDGE_BATCH_PROMPT.format(
            clause_text=clause_text,
            intents_json=intents_json,
        )
        try:
            raw = self.llm_client.complete(
                prompt, system=INTENT_JUDGE_SYSTEM,
                temperature=0.0, json_mode=True,
            )
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Some models wrap array in {"results": [...]}
                for key in ("results", "intents", "evaluations"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    parsed = [parsed]
            return parsed if isinstance(parsed, list) else [None] * len(intents)
        except Exception as e:
            logger.warning(f"Batch judge call failed: {e}")
            return [None] * len(intents)

    def _compute_metrics(self) -> dict:
        if not self._results:
            return self._empty_metrics()

        faithfulness = []
        actionability = []
        hallucinated = 0

        for r in self._results:
            f = r.get("faithfulness_score")
            a = r.get("actionability_score")
            if isinstance(f, (int, float)):
                faithfulness.append(f)
            if isinstance(a, (int, float)):
                actionability.append(a)
            if r.get("is_hallucinated") is True:
                hallucinated += 1

        n = len(self._results)
        avg_f = sum(faithfulness) / len(faithfulness) if faithfulness else 0.0
        avg_a = sum(actionability) / len(actionability) if actionability else 0.0

        # Healthy Insight Score = (Faithfulness + Actionability) / 10
        # Ranges 0.2 (both=1) to 1.0 (both=5)
        healthy_insight = (avg_f + avg_a) / 10.0

        return {
            "judge.intents_evaluated": n,
            "judge.avg_faithfulness": round(avg_f, 2),
            "judge.avg_actionability": round(avg_a, 2),
            "judge.healthy_insight_score": round(healthy_insight, 4),
            "judge.hallucination_rate": round(hallucinated / n * 100, 2) if n > 0 else 0.0,
            "judge.faithfulness_below_3_pct": round(
                sum(1 for f in faithfulness if f < 3) / len(faithfulness) * 100, 2
            ) if faithfulness else 0.0,
            "judge.actionability_below_3_pct": round(
                sum(1 for a in actionability if a < 3) / len(actionability) * 100, 2
            ) if actionability else 0.0,
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "judge.intents_evaluated": 0,
            "judge.avg_faithfulness": 0.0,
            "judge.avg_actionability": 0.0,
            "judge.healthy_insight_score": 0.0,
            "judge.hallucination_rate": 0.0,
            "judge.faithfulness_below_3_pct": 0.0,
            "judge.actionability_below_3_pct": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GOLD SET INTENT COMPARISON (non-LLM, uses embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentGoldEval:
    """
    Compare pipeline-discovered intents against gold annotations
    using embedding similarity (no LLM call required).
    """

    db_path: str
    gold_intents: dict[str, list[dict]] = field(default_factory=dict)
    # gold_intents: {clause_id: [{label, summary, ...}, ...]}
    embedder: Any = None

    def evaluate(self) -> dict[str, Any]:
        if not self.gold_intents or self.embedder is None:
            return {
                "intent_gold.avg_best_match_sim": 0.0,
                "intent_gold.recall_at_0_8": 0.0,
                "intent_gold.gold_count": len(self.gold_intents),
            }

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._compare(conn)
        finally:
            conn.close()

    def _compare(self, conn: duckdb.DuckDBPyConnection) -> dict:
        import numpy as np

        gold_clause_ids = list(self.gold_intents.keys())
        placeholders = ",".join(["?"] * len(gold_clause_ids))
        pred_rows = conn.execute(f"""
            SELECT clause_id, intent_label, summary
            FROM clause_intents
            WHERE clause_id IN ({placeholders})
        """, gold_clause_ids).fetchall()

        # Group predicted intents by clause
        pred_by_clause: dict[str, list[str]] = {}
        for cid, label, summary in pred_rows:
            text = f"{label}: {summary}" if summary else label
            pred_by_clause.setdefault(cid, []).append(text)

        best_sims = []
        total_gold = 0
        recalled_08 = 0

        for cid, gold_list in self.gold_intents.items():
            pred_list = pred_by_clause.get(cid, [])
            if not pred_list:
                total_gold += len(gold_list)
                continue

            # Embed gold and predicted
            gold_texts = [
                f"{g['label']}: {g.get('summary', '')}" for g in gold_list
            ]
            gold_embs = self.embedder.embed(gold_texts, prefix="search_document")
            pred_embs = self.embedder.embed(pred_list, prefix="search_document")

            # For each gold intent, find best matching predicted intent
            g_norm = gold_embs / np.clip(
                np.linalg.norm(gold_embs, axis=1, keepdims=True), 1e-12, None
            )
            p_norm = pred_embs / np.clip(
                np.linalg.norm(pred_embs, axis=1, keepdims=True), 1e-12, None
            )
            sim_matrix = g_norm @ p_norm.T  # (n_gold, n_pred)

            for i in range(len(gold_list)):
                best = float(sim_matrix[i].max())
                best_sims.append(best)
                total_gold += 1
                if best >= 0.8:
                    recalled_08 += 1

        if not best_sims:
            return {
                "intent_gold.avg_best_match_sim": 0.0,
                "intent_gold.recall_at_0_8": 0.0,
                "intent_gold.gold_count": total_gold,
            }

        return {
            "intent_gold.avg_best_match_sim": round(
                float(np.mean(best_sims)), 4
            ),
            "intent_gold.recall_at_0_8": round(
                recalled_08 / total_gold * 100, 2
            ) if total_gold > 0 else 0.0,
            "intent_gold.gold_count": total_gold,
            "intent_gold.matched_count": len(best_sims),
        }
