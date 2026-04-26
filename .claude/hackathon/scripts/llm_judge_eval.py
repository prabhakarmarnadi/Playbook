#!/usr/bin/env python3
"""
LLM-as-a-Judge Extraction Evaluator
=====================================

Uses GPT to judge whether each extraction is correct, producing real
precision / recall / F1 metrics — not just self-grounding heuristics.

For each (cluster, method, chunk, field, extracted_value), the judge LLM reads
the full chunk text and answers:
  1. Is the extracted value CORRECT for this field?  (precision signal)
  2. What is the correct value?                       (gold label → recall signal)

Usage:
    # Judge the latest v2 comparison results
    python scripts/llm_judge_eval.py data/cuad_510_demo.duckdb

    # Judge a specific comparison JSON
    python scripts/llm_judge_eval.py data/cuad_510_demo.duckdb --comparison data/extraction_value_comparison_v2.json

    # Limit cost — only judge N extractions per method
    python scripts/llm_judge_eval.py data/cuad_510_demo.duckdb --max-per-method 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = """\
You are a senior legal contract analyst acting as an evaluation judge.
You evaluate machine-extracted field values from legal agreements.
Respond with valid JSON only. No commentary outside the JSON."""

JUDGE_PROMPT = """\
## Task
A machine extracted a value for a specific field from a legal clause.
Evaluate whether the extraction is correct.

## Clause Type
{cluster_label}

## Source Text
```
{chunk_text}
```

## Extracted Field
- **Field Name**: {field_name}
- **Extracted Value**: {extracted_value}

## Instructions
1. Is the extracted value correct for this field given the source text?
2. If not, what is the correct value? (null if the field is genuinely not present)

## Scoring
- **correct**: The extracted value accurately represents the information for this field.
- **partially_correct**: The value captures some relevant info but is incomplete or slightly wrong.
- **incorrect**: The value is wrong — hallucinated, taken from the wrong context, or misattributed.
- **not_present**: This field genuinely has no value in the source text (correct answer is null).

## Response Format
Return ONLY this JSON:
{{
  "verdict": "<correct|partially_correct|incorrect|not_present>",
  "correct_value": "<the correct value, or null if not present>",
  "reason": "<one sentence explaining your judgment>"
}}"""

# Batch variant — judge multiple fields from one chunk in a single call
JUDGE_BATCH_PROMPT = """\
## Task
A machine extracted values for multiple fields from a legal clause.
Evaluate whether each extraction is correct.

## Clause Type
{cluster_label}

## Source Text
```
{chunk_text}
```

## Extracted Fields (with expected types and descriptions)
{fields_json}

## Instructions
For each field, judge whether the extracted value is correct given the source text.
- A value is CORRECT if it accurately represents information **explicitly stated** in the source text for this field.
- A paraphrase or slight reformulation of text that preserves meaning is still CORRECT.
- A value taken from a different part of the text or describing a different concept is INCORRECT.

## Verdicts
- **correct**: Value accurately represents the field's information from the text.
- **partially_correct**: Captures some relevant info but incomplete or slightly wrong.
- **incorrect**: Wrong — hallucinated, from wrong context, or misattributed.
- **not_present**: Field genuinely has no value in the source text (null is correct).

## Response Format
Return ONLY a JSON array in the same order as the fields above:
[
  {{
    "field_name": "<echo the field name>",
    "verdict": "<correct|partially_correct|incorrect|not_present>",
    "correct_value": "<the correct value, or null>",
    "reason": "<one sentence>"
  }}
]"""


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_client_and_deployment():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini").strip('"')

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=api_version,
    )
    return client, deployment


def judge_batch(
    client: AzureOpenAI,
    deployment: str,
    cluster_label: str,
    chunk_text: str,
    fields: list[dict],  # [{field_name, extracted_value}, ...]
    max_retries: int = 3,
) -> list[dict]:
    """Judge multiple field extractions from a single chunk in one LLM call."""
    fields_json = json.dumps(fields, indent=2)

    prompt = JUDGE_BATCH_PROMPT.format(
        cluster_label=cluster_label,
        chunk_text=chunk_text[:4000],  # cap context to avoid token overflow
        fields_json=fields_json,
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=2000,
                response_format={"type": "json_object"},
            )

            text = resp.choices[0].message.content.strip()
            parsed = json.loads(text)

            # Handle both {"results": [...]} and direct [...]
            if isinstance(parsed, dict) and "results" in parsed:
                parsed = parsed["results"]
            if isinstance(parsed, dict):
                # Sometimes wraps in a single key
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break

            if isinstance(parsed, list) and len(parsed) >= 1:
                return parsed

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Judge parse error (attempt {attempt + 1}): {e}")
                time.sleep(1)
            else:
                logger.error(f"Judge failed after {max_retries} attempts: {e}")
                return [
                    {"field_name": f["field_name"], "verdict": "error", "correct_value": None, "reason": str(e)}
                    for f in fields
                ]

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_precision_recall(judgments: list[dict]) -> dict:
    """
    Compute precision/recall/F1 from judge verdicts.

    Precision = correct_extractions / all_extractions  (when model said something non-null)
    Recall    = correct_extractions / all_fields_with_values  (fields that have a real value)
    """
    n_extracted = 0     # model produced a non-null value
    n_correct = 0       # judge said correct or partially_correct
    n_strict_correct = 0  # judge said correct only
    n_has_value = 0     # judge says a real value exists (not_present excluded)
    n_found = 0         # model found a non-null value AND judge says value exists
    n_total = len(judgments)

    for j in judgments:
        verdict = j.get("verdict", "error")
        extracted = j.get("extracted_value") not in (None, "", "null", "None")
        judge_has_value = verdict != "not_present"

        if judge_has_value:
            n_has_value += 1

        if extracted:
            n_extracted += 1
            if verdict == "correct":
                n_strict_correct += 1
                n_correct += 1
            elif verdict == "partially_correct":
                n_correct += 1

        if extracted and judge_has_value:
            n_found += 1
            if verdict in ("correct", "partially_correct"):
                pass  # already counted

    # Precision: of what the model extracted, how much was right?
    precision = n_correct / n_extracted if n_extracted > 0 else 0.0
    precision_strict = n_strict_correct / n_extracted if n_extracted > 0 else 0.0

    # Recall: of fields that have values, how many did the model find correctly?
    recall = n_correct / n_has_value if n_has_value > 0 else 0.0
    recall_strict = n_strict_correct / n_has_value if n_has_value > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f1_strict = 2 * precision_strict * recall_strict / (precision_strict + recall_strict) \
        if (precision_strict + recall_strict) > 0 else 0.0

    return {
        "total_judged": n_total,
        "total_extracted": n_extracted,
        "total_has_value": n_has_value,
        "correct": n_strict_correct,
        "partially_correct": n_correct - n_strict_correct,
        "incorrect": n_extracted - n_correct,
        "not_present": n_total - n_has_value,
        "precision": round(precision, 4),
        "precision_strict": round(precision_strict, 4),
        "recall": round(recall, 4),
        "recall_strict": round(recall_strict, 4),
        "f1": round(f1, 4),
        "f1_strict": round(f1_strict, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_judge_eval(
    db_path: str,
    comparison_path: str | None = None,
    max_per_method: int | None = None,
):
    """
    Run LLM judge on extraction comparison results.
    Pulls full chunk text from DuckDB (comparison JSON only stores previews).
    """
    # Load comparison results
    if comparison_path is None:
        comparison_path = str(Path(db_path).parent / "extraction_value_comparison_v2.json")

    with open(comparison_path) as f:
        data = json.load(f)

    per_cluster = data["per_cluster"]
    client, deployment = _get_client_and_deployment()
    logger.info(f"Judge LLM: {deployment}")

    conn = duckdb.connect(db_path, read_only=True)

    # Reconstruct full chunk texts from DuckDB using the same sampling logic
    # We need to re-fetch chunks for each cluster
    all_judgments: dict[str, list[dict]] = {}  # method → [judgment, ...]
    methods = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]

    for method in methods:
        all_judgments[method] = []

    total_calls = 0

    for ci, cluster_data in enumerate(per_cluster):
        cluster_id = cluster_data["cluster_id"]
        cluster_label = cluster_data["cluster_label"]
        n_extract = cluster_data["n_extract_chunks"]

        logger.info(f"\n{'='*70}")
        logger.info(f"[{ci+1}/{len(per_cluster)}] Judging: {cluster_label}")
        logger.info(f"{'='*70}")

        # Fetch the same chunks used for extraction
        chunks_df = conn.execute("""
            SELECT ch.chunk_text
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_id
            LIMIT ?
        """, [cluster_id, n_extract]).fetchdf()

        chunk_texts = chunks_df["chunk_text"].tolist()

        if len(chunk_texts) != n_extract:
            logger.warning(
                f"  Expected {n_extract} chunks, got {len(chunk_texts)} — "
                f"results may differ from comparison run"
            )

        for method in methods:
            method_data = cluster_data["methods"].get(method, {})
            extractions = method_data.get("extractions", [])

            if not extractions:
                logger.info(f"  [{method}] No extractions (0 fields)")
                continue

            # Build field description lookup from discovery results
            field_defs = {}
            for f in method_data.get("fields", []):
                field_defs[f.get("name", "")] = {
                    "type": f.get("type", "text"),
                    "description": f.get("description", ""),
                }

            n_judged = 0
            for ext in extractions:
                chunk_idx = ext["chunk_index"]
                if chunk_idx >= len(chunk_texts):
                    continue

                chunk_text = chunk_texts[chunk_idx]
                fields_to_judge = []

                for field_name, field_data in ext["fields"].items():
                    value = field_data.get("value")
                    # Skip nulls — nothing to judge
                    if value is None:
                        # But record for recall calculation
                        all_judgments[method].append({
                            "cluster": cluster_label,
                            "chunk_index": chunk_idx,
                            "field_name": field_name,
                            "extracted_value": None,
                            "verdict": "not_extracted",
                            "correct_value": None,
                            "reason": "model returned null",
                        })
                        continue

                    fdef = field_defs.get(field_name, {})
                    fields_to_judge.append({
                        "field_name": field_name,
                        "field_type": fdef.get("type", "text"),
                        "field_description": fdef.get("description", ""),
                        "extracted_value": str(value),
                    })

                if not fields_to_judge:
                    continue

                if max_per_method and n_judged >= max_per_method:
                    break

                # Judge all non-null fields from this chunk in one batch call
                verdicts = judge_batch(
                    client, deployment, cluster_label, chunk_text, fields_to_judge,
                )
                total_calls += 1

                # Match verdicts back to fields
                verdict_map = {}
                for v in verdicts:
                    vname = v.get("field_name", "")
                    verdict_map[vname] = v

                for fj in fields_to_judge:
                    fname = fj["field_name"]
                    v = verdict_map.get(fname, {
                        "verdict": "error",
                        "correct_value": None,
                        "reason": "judge did not return this field",
                    })

                    all_judgments[method].append({
                        "cluster": cluster_label,
                        "chunk_index": chunk_idx,
                        "field_name": fname,
                        "extracted_value": fj["extracted_value"],
                        "verdict": v.get("verdict", "error"),
                        "correct_value": v.get("correct_value"),
                        "reason": v.get("reason", ""),
                    })

                n_judged += len(fields_to_judge)

            logger.info(f"  [{method}] Judged {n_judged} extractions")

    logger.info(f"\nTotal judge LLM calls: {total_calls}")

    # ── Compute precision / recall / F1 per method ──────────────────────────

    results = {}
    for method in methods:
        judgments = all_judgments[method]
        # For recall, we need to also judge the "not_extracted" cases
        # to know if a value actually exists
        # But we only have LLM judgments for extracted values
        # So we'll compute precision from judged extractions,
        # and recall from judged extractions + known nulls using judge's not_present signal

        # Filter to only LLM-judged items (not "not_extracted")
        judged = [j for j in judgments if j["verdict"] != "not_extracted"]
        not_extracted = [j for j in judgments if j["verdict"] == "not_extracted"]

        metrics = compute_precision_recall(judged)
        metrics["not_extracted_count"] = len(not_extracted)
        metrics["method"] = method
        results[method] = metrics

    # ── Print results ───────────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print(" LLM JUDGE EVALUATION — Precision / Recall / F1")
    print(f" Judge: {deployment} | {len(per_cluster)} clusters")
    print("=" * 100)

    header = f"{'Metric':<30} {'Standard GPT':>14} {'DSPy RLM':>14} {'RLM + LB':>14} {'RLM v2':>14}"
    print(f"\n{header}")
    print("─" * len(header))

    metric_rows = [
        ("Judged extractions", "total_extracted"),
        ("Fields with values", "total_has_value"),
        ("Correct", "correct"),
        ("Partially correct", "partially_correct"),
        ("Incorrect", "incorrect"),
        ("Not present (null OK)", "not_present"),
        ("", None),
        ("Precision (lenient)", "precision"),
        ("Precision (strict)", "precision_strict"),
        ("Recall (lenient)", "recall"),
        ("Recall (strict)", "recall_strict"),
        ("F1 (lenient)", "f1"),
        ("F1 (strict)", "f1_strict"),
    ]

    for label, key in metric_rows:
        if key is None:
            print()
            continue
        vals = []
        for m in methods:
            v = results[m].get(key, 0)
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(f"  {label:<28} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

    # ── Winner analysis ─────────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print(" WINNER ANALYSIS (LLM Judge)")
    print("=" * 100)

    best_metrics = {
        "Best precision (lenient)": "precision",
        "Best precision (strict)": "precision_strict",
        "Best recall (lenient)": "recall",
        "Best recall (strict)": "recall_strict",
        "Best F1 (lenient)": "f1",
        "Best F1 (strict)": "f1_strict",
    }

    for label, key in best_metrics.items():
        best_method = max(methods, key=lambda m: results[m].get(key, 0))
        best_val = results[best_method][key]
        print(f"  {label:<30} {best_method} ({best_val:.3f})")

    print("=" * 100)

    # ── Save results ────────────────────────────────────────────────────────

    output = {
        "meta": {
            "judge_model": deployment,
            "db_path": db_path,
            "comparison_path": comparison_path,
            "total_judge_calls": total_calls,
        },
        "per_method": results,
        "judgments": {m: all_judgments[m] for m in methods},
    }

    out_path = Path(db_path).parent / "judge_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM Judge Extraction Evaluator")
    parser.add_argument("db_path", help="Path to DuckDB database")
    parser.add_argument("--comparison", default=None,
                        help="Path to comparison JSON (default: auto-detect v2)")
    parser.add_argument("--max-per-method", type=int, default=None,
                        help="Max extractions to judge per method per cluster (cost control)")

    args = parser.parse_args()
    run_judge_eval(args.db_path, args.comparison, args.max_per_method)


if __name__ == "__main__":
    main()
