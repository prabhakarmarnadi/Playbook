"""Evaluation harness: measure findings precision/recall vs a small gold set.

Gold set JSON shape (eval/playbook_gold.json):
  [
    {"agreement_id": "a1", "playbook_id": "p1",
     "expected": [{"rule_id": "r1", "outcome": "fail"},
                   {"rule_id": "r2", "outcome": "pass"}]},
    ...
  ]

Usage:
  poetry run python -m eval.playbook_eval data/evoc_200_refined.duckdb p1 \\
      --gold eval/playbook_gold.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.aligner import align
from core.playbooks.store import PlaybookStore


def evaluate(db_path: str, playbook_id: str, gold_path: str) -> dict:
    gold = json.loads(Path(gold_path).read_text())
    pb = PlaybookStore(db_path)
    tp = fp = fn = 0
    for entry in gold:
        ctx = {"agreement_id": entry["agreement_id"],
                "fields": entry.get("fields", {}),
                "clauses": entry.get("clauses", [])}
        actual = {f["rule_id"]: f["outcome"]
                   for f in align(pb, playbook_id, ctx)}
        for exp in entry["expected"]:
            rid, want = exp["rule_id"], exp["outcome"]
            got = actual.get(rid)
            if got == want and want == "fail":
                tp += 1
            elif got == "fail" and want != "fail":
                fp += 1
            elif got != "fail" and want == "fail":
                fn += 1
    pb.close()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db_path")
    p.add_argument("playbook_id")
    p.add_argument("--gold", default="eval/playbook_gold.json")
    args = p.parse_args()
    result = evaluate(args.db_path, args.playbook_id, args.gold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
