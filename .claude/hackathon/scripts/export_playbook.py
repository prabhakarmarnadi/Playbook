"""Dump one or all playbooks from a DuckDB to JSON (stdout or --out file).

Schema:
  {
    "exported_at": "...",
    "source_db": "...",
    "playbooks": [
      {
        "playbook_id": "...",
        "name": "...",
        "owner_org": "...",
        "description": "...",
        "status": "draft|published|archived",
        "version": 1,
        "created_at": "...",
        "updated_at": "...",
        "rules": [
          {
            "rule_id": "...", "title": "...", "description": "...",
            "applies_to": "domain|cluster|field|composite|cross_field|document",
            "predicate": {...} | null,
            "nl_assertion": "..." | null,
            "reference_text": "..." | null,
            "similarity_threshold": 0.85 | null,
            "answer_type": "yes_no|standard_nonstandard|...",
            "answer_options": [...] | null,
            "severity": "info|warn|approval_required|blocker",
            "escalation_owner": "..." | null,
            "preferred_language": "..." | null,
            "fallback_language": [...] | null,
            "walkaway_language": "..." | null,
            "rationale": "..." | null,
            "tags": [...] | null,
            "source_provenance": {...} | null,
            "status": "draft|active|retired",
            "bindings": [
              {"entity_kind": "cluster", "entity_id": "...",
               "label_text": "...", "confidence": 1.0}
            ]
          }
        ]
      }
    ]
  }

Usage:
  python scripts/export_playbook.py data/demo.duckdb
  python scripts/export_playbook.py data/demo.duckdb --playbook-id <uuid>
  python scripts/export_playbook.py data/demo.duckdb --out /tmp/playbooks.json
  python scripts/export_playbook.py data/demo.duckdb --status draft
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.store import PlaybookStore


_JSON_COLS = {"predicate", "answer_options", "fallback_language",
              "tags", "source_provenance"}


def _coerce_json(row: dict) -> dict:
    """Parse JSON-typed columns back to Python objects so the export stays
    pure JSON (no nested escape strings)."""
    out = {}
    for k, v in row.items():
        if k in _JSON_COLS and isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        elif isinstance(v, _dt.datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def export(db_path: str, *,
            playbook_id: str | None = None,
            status: str | None = None) -> dict:
    """Return a JSON-serializable dict containing the requested playbooks."""
    s = PlaybookStore(db_path)
    try:
        playbooks = s.list_playbooks(status=status)
        if playbook_id:
            playbooks = [p for p in playbooks if p["playbook_id"] == playbook_id]

        out_playbooks: list[dict] = []
        for pb in playbooks:
            pb = _coerce_json(pb)
            rules = s.list_rules(pb["playbook_id"])
            rules_out: list[dict] = []
            for r in rules:
                r = _coerce_json(r)
                # Drop the playbook_id duplicate (already in parent)
                r.pop("playbook_id", None)
                r["bindings"] = [
                    {
                        "entity_kind": b["entity_kind"],
                        "entity_id": b["entity_id"],
                        "label_text": b.get("label_text"),
                        "confidence": b.get("confidence"),
                    }
                    for b in s.bindings_for(r["rule_id"])
                ]
                rules_out.append(r)
            pb["rules"] = rules_out
            out_playbooks.append(pb)

        return {
            "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "source_db": str(Path(db_path).resolve()),
            "playbook_count": len(out_playbooks),
            "rule_count": sum(len(p["rules"]) for p in out_playbooks),
            "playbooks": out_playbooks,
        }
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Dump playbooks from a DuckDB as JSON")
    p.add_argument("db_path", help="Path to the DuckDB file")
    p.add_argument("--playbook-id", help="Export only this playbook (default: all)")
    p.add_argument("--status", choices=["draft", "published", "archived"],
                    help="Filter by playbook status")
    p.add_argument("--out", help="Write JSON here (default: stdout)")
    p.add_argument("--pretty", action="store_true", default=True,
                    help="Pretty-print (default true)")
    p.add_argument("--compact", action="store_false", dest="pretty",
                    help="Single-line JSON")
    args = p.parse_args()

    payload = export(args.db_path,
                      playbook_id=args.playbook_id,
                      status=args.status)
    body = json.dumps(payload, indent=2 if args.pretty else None, default=str)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"Wrote {len(body):,} bytes to {args.out} "
              f"({payload['playbook_count']} playbooks, {payload['rule_count']} rules)",
              file=sys.stderr)
    else:
        print(body)


if __name__ == "__main__":
    main()
