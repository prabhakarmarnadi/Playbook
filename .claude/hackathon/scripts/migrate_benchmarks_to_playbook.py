"""One-off migration: promote core/playbook.py's playbook_benchmarks rows
into the new playbook subsystem as a default 'Auto-mined baseline' playbook.

Each row → one rule with applies_to=cluster, similarity_threshold derived from
mean_deviation, reference_text=benchmark_text. Idempotent: re-running creates a
new playbook each time (caller should retire the old one).

Usage:
    python scripts/migrate_benchmarks_to_playbook.py data/my.duckdb
    python scripts/migrate_benchmarks_to_playbook.py data/my.duckdb --name "Q2 Baseline"
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.store import PlaybookStore


def migrate(db_path: str, *, name: str = "Auto-mined baseline") -> str:
    """Lift playbook_benchmarks rows into the new schema and return the new playbook_id."""
    s = PlaybookStore(db_path)
    rows = s.conn.execute(
        "SELECT cluster_id, cluster_label, benchmark_text, source, mean_deviation "
        "FROM playbook_benchmarks"
    ).fetchall()
    pid = s.create_playbook(
        name=name,
        owner_org="auto",
        description=f"Migrated {len(rows)} cluster benchmarks from playbook_benchmarks",
    )
    for cluster_id, cluster_label, text, source, mean_dev in rows:
        rid = s.create_rule(
            playbook_id=pid,
            title=f"{cluster_label} similarity to benchmark",
            applies_to="cluster",
            severity="info",
            reference_text=text,
            similarity_threshold=max(0.5, 1.0 - 2.0 * (mean_dev or 0.0)),
            tags=[cluster_label, source or "unknown", "auto"],
            source_provenance={"migrated_from": "playbook_benchmarks", "cluster_id": cluster_id},
            status="draft",
        )
        s.add_binding(
            rule_id=rid,
            entity_kind="cluster",
            entity_id=cluster_id,
            label_text=cluster_label,
            confidence=1.0,
        )
    s.close()
    return pid


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Migrate playbook_benchmarks → new playbook schema")
    p.add_argument("db_path", help="Path to the DuckDB file")
    p.add_argument("--name", default="Auto-mined baseline", help="Name for the created playbook")
    args = p.parse_args()
    pid = migrate(args.db_path, name=args.name)
    print(f"Created playbook: {pid}")
