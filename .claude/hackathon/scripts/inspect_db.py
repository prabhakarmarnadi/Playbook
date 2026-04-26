#!/usr/bin/env python
"""Inspect what's stored in a clustering DuckDB — table counts, samples, relationships.

Usage:
    poetry run python scripts/inspect_db.py                           # auto-find latest
    poetry run python scripts/inspect_db.py data/public_docs_semantic_test.duckdb
    poetry run python scripts/inspect_db.py --account-id public_docs_semantic_test
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_db(account_id: str | None = None) -> Path:
    if account_id:
        p = DATA_DIR / f"{account_id}.duckdb"
        if p.exists():
            return p
    # Find largest .duckdb file
    dbs = sorted(DATA_DIR.glob("*.duckdb"), key=lambda f: f.stat().st_size, reverse=True)
    if not dbs:
        print("No .duckdb files found in data/", file=sys.stderr)
        sys.exit(1)
    return dbs[0]


def section(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def inspect(db_path: str):
    conn = duckdb.connect(db_path, read_only=True)
    print(f"Database: {db_path}  ({Path(db_path).stat().st_size / 1024:.0f} KB)")

    # ── Table row counts ───────────────────────────────────────────────
    section("TABLE ROW COUNTS")
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    for (t,) in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<30} {cnt:>8} rows")

    # ── Pipeline runs ──────────────────────────────────────────────────
    section("PIPELINE RUNS")
    runs = conn.execute(
        "SELECT run_id, status, stage, started_at, ended_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 5"
    ).fetchall()
    if runs:
        print(f"  {'run_id':<20} {'status':<12} {'stage':<18} {'started':<22} {'ended'}")
        print(f"  {'─'*20} {'─'*12} {'─'*18} {'─'*22} {'─'*20}")
        for r in runs:
            ended = str(r[4])[:19] if r[4] else "—"
            print(f"  {r[0]:<20} {r[1]:<12} {str(r[2] or ''):<18} {str(r[3])[:19]:<22} {ended}")
    else:
        print("  (no runs)")

    # ── Domains (macro clusters) ───────────────────────────────────────
    section("DOMAINS (MACRO CLUSTERS)")
    domains = conn.execute(
        "SELECT domain_id, label, agreement_count, confidence FROM domains ORDER BY agreement_count DESC"
    ).fetchall()
    if domains:
        print(f"  {'domain_id':<18} {'label':<40} {'docs':>5} {'conf':>6}")
        print(f"  {'─'*18} {'─'*40} {'─'*5} {'─'*6}")
        for d in domains:
            print(f"  {d[0]:<18} {str(d[1])[:40]:<40} {d[2]:>5} {(d[3] or 0):>6.3f}")
    else:
        print("  (no domains)")

    # ── Agreements per domain ──────────────────────────────────────────
    section("AGREEMENTS BY DOMAIN")
    agr_by_domain = conn.execute("""
        SELECT d.label, COUNT(a.agreement_id) as n,
               AVG(LENGTH(a.raw_text)) as avg_chars,
               AVG(a.domain_confidence) as avg_conf
        FROM agreements a
        LEFT JOIN domains d ON a.domain_id = d.domain_id
        GROUP BY d.label
        ORDER BY n DESC
    """).fetchall()
    if agr_by_domain:
        print(f"  {'domain':<40} {'docs':>5} {'avg_chars':>10} {'avg_conf':>9}")
        print(f"  {'─'*40} {'─'*5} {'─'*10} {'─'*9}")
        for row in agr_by_domain:
            label = str(row[0] or "Unassigned")[:40]
            print(f"  {label:<40} {row[1]:>5} {int(row[2] or 0):>10} {(row[3] or 0):>9.3f}")

    # ── Clusters (micro clusters) per domain ───────────────────────────
    section("CLUSTERS (MICRO) BY DOMAIN")
    clusters = conn.execute("""
        SELECT d.label as domain_label, c.cluster_id, c.label, c.chunk_count,
               c.agreement_count, c.quality_score,
               c.keywords
        FROM clusters c
        LEFT JOIN domains d ON c.domain_id = d.domain_id
        ORDER BY d.label, c.chunk_count DESC
    """).fetchall()
    if clusters:
        current_domain = None
        for cl in clusters:
            domain = str(cl[0] or "Unknown")
            if domain != current_domain:
                current_domain = domain
                print(f"\n  ┌─ Domain: {domain}")
                print(f"  │ {'cluster_id':<18} {'label':<35} {'chunks':>6} {'docs':>5} {'quality':>8}")
                print(f"  │ {'─'*18} {'─'*35} {'─'*6} {'─'*5} {'─'*8}")
            kw = ""
            if cl[6]:
                try:
                    kw_list = json.loads(cl[6])
                    kw = ", ".join(kw_list[:5])
                except Exception:
                    kw = str(cl[6])[:50]
            print(f"  │ {cl[1]:<18} {str(cl[2])[:35]:<35} {cl[3]:>6} {cl[4]:>5} {(cl[5] or 0):>8.4f}")
            if kw:
                print(f"  │   keywords: {kw[:70]}")
    else:
        print("  (no clusters)")

    # ── Chunk distribution ─────────────────────────────────────────────
    section("CHUNK STATISTICS")
    chunk_stats = conn.execute("""
        SELECT COUNT(*) as n_chunks,
               AVG(LENGTH(chunk_text)) as avg_len,
               MIN(LENGTH(chunk_text)) as min_len,
               MAX(LENGTH(chunk_text)) as max_len,
               AVG(token_count) as avg_tokens
        FROM chunks
    """).fetchone()
    if chunk_stats and chunk_stats[0]:
        print(f"  Total chunks:    {chunk_stats[0]:>8}")
        print(f"  Avg char length: {int(chunk_stats[1] or 0):>8}")
        print(f"  Min char length: {int(chunk_stats[2] or 0):>8}")
        print(f"  Max char length: {int(chunk_stats[3] or 0):>8}")
        print(f"  Avg tokens:      {int(chunk_stats[4] or 0):>8}")

    chunks_per_doc = conn.execute("""
        SELECT agreement_id, COUNT(*) as n
        FROM chunks GROUP BY agreement_id
        ORDER BY n DESC LIMIT 10
    """).fetchall()
    if chunks_per_doc:
        print(f"\n  Top 10 docs by chunk count:")
        for agr_id, n in chunks_per_doc:
            print(f"    {agr_id:<40} {n:>4} chunks")

    # ── Cluster assignments ────────────────────────────────────────────
    section("CLUSTER ASSIGNMENT COVERAGE")
    assigned = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM cluster_assignments").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total:
        pct = assigned / total * 100
        print(f"  Assigned chunks: {assigned:>8} / {total} ({pct:.1f}%)")
        unassigned = total - assigned
        print(f"  Unassigned:      {unassigned:>8} ({100-pct:.1f}%)")

    # ── Sample agreements ──────────────────────────────────────────────
    section("SAMPLE AGREEMENTS (first 5)")
    samples = conn.execute("""
        SELECT a.agreement_id, a.filename, d.label as domain,
               a.domain_confidence, LENGTH(a.raw_text) as chars
        FROM agreements a
        LEFT JOIN domains d ON a.domain_id = d.domain_id
        LIMIT 5
    """).fetchall()
    for s in samples:
        print(f"  {s[1][:50]:<50} domain={str(s[2])[:25]:<25} conf={s[3] or 0:.3f} chars={s[4]}")

    # ── Sample chunks from first cluster ───────────────────────────────
    section("SAMPLE CHUNKS (from first cluster)")
    first_cluster = conn.execute(
        "SELECT cluster_id, label FROM clusters LIMIT 1"
    ).fetchone()
    if first_cluster:
        print(f"  Cluster: {first_cluster[0]} — {first_cluster[1]}")
        sample_chunks = conn.execute("""
            SELECT c.chunk_text
            FROM cluster_assignments ca
            JOIN chunks c ON ca.chunk_id = c.chunk_id
            WHERE ca.cluster_id = ?
            LIMIT 3
        """, [first_cluster[0]]).fetchall()
        for i, (text,) in enumerate(sample_chunks):
            preview = text[:200].replace("\n", " ") + ("..." if len(text) > 200 else "")
            print(f"  [{i+1}] {preview}")

    conn.close()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect clustering DuckDB")
    parser.add_argument("db_path", nargs="?", help="Path to .duckdb file")
    parser.add_argument("--account-id", help="Account ID (auto-resolves to data/<id>.duckdb)")
    args = parser.parse_args()

    if args.db_path:
        path = args.db_path
    else:
        db = find_db(args.account_id)
        path = str(db)

    inspect(path)
