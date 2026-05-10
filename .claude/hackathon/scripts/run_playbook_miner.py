"""CLI: poetry run python scripts/run_playbook_miner.py <duckdb_path> [--name "..."]

Reads the existing ClusteringStore + PlaybookStore (same DB), builds the corpus,
and inserts draft mined rules into a new playbook.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.playbooks.miner_runner import run_miner
from core.playbooks.store import PlaybookStore
from core.store import ClusteringStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db_path")
    p.add_argument("--name", default="Auto-mined draft")
    args = p.parse_args()

    cs = ClusteringStore(args.db_path)
    pb = PlaybookStore(args.db_path)
    try:
        pid, cands = run_miner(cs, pb, playbook_name=args.name)
        print(f"playbook_id={pid}")
        print(f"candidates={len(cands)}")
        by_kind = {}
        for c in cands:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        for kind, count in sorted(by_kind.items()):
            print(f"  {kind}: {count}")
    finally:
        pb.close()
        if hasattr(cs, "close"):
            cs.close()


if __name__ == "__main__":
    main()
