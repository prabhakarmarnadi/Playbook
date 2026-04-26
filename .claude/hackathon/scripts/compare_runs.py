#!/usr/bin/env python
"""Compare cluster quality reports across pipeline runs.

Usage:
    # Compare all runs for an account
    poetry run python scripts/compare_runs.py --account-id public_docs_semantic

    # Compare specific run directories
    poetry run python scripts/compare_runs.py \
        data/artifacts/public_docs_fast/run_abc12345 \
        data/artifacts/public_docs_semantic/run_def67890

    # Output as JSON
    poetry run python scripts/compare_runs.py --account-id public_docs_semantic --json
"""
import argparse
import json
import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"


def load_run(run_dir: Path) -> dict | None:
    quality_path = run_dir / "cluster_quality.json"
    manifest_path = run_dir / "run_manifest.json"
    if not quality_path.exists():
        return None
    data = {"run_dir": str(run_dir), "run_id": run_dir.name}
    with open(quality_path) as f:
        data["quality"] = json.load(f)
    if manifest_path.exists():
        with open(manifest_path) as f:
            data["manifest"] = json.load(f)
    return data


def collect_runs(account_id: str) -> list[dict]:
    account_dir = ARTIFACTS_DIR / account_id
    if not account_dir.exists():
        print(f"No artifacts found for account: {account_id}", file=sys.stderr)
        return []
    runs = []
    for run_dir in sorted(account_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            data = load_run(run_dir)
            if data:
                runs.append(data)
    return runs


def print_comparison(runs: list[dict], as_json: bool = False):
    if not runs:
        print("No runs with quality reports found.")
        return

    if as_json:
        summary = []
        for r in runs:
            g = r["quality"].get("global", {})
            m = r.get("manifest", {})
            summary.append({
                "run_id": r["run_id"],
                "n_documents": g.get("n_documents", 0),
                "n_chunks": g.get("n_chunks", 0),
                "n_domains": g.get("n_domains", 0),
                "n_clusters": g.get("n_clusters", 0),
                "avg_quality": g.get("avg_quality_across_domains", 0),
                "chunk_outlier_ratio": g.get("chunk_outlier_ratio", 0),
                "semantic_chunking": g.get("use_semantic_chunking"),
                "elapsed_seconds": m.get("elapsed_seconds"),
                "stage_timings": m.get("stage_timings", {}),
            })
        print(json.dumps(summary, indent=2))
        return

    # ── Table header ────────────────────────────────────
    cols = ["Run ID", "Docs", "Chunks", "Domains", "Clusters", "Avg Qual", "Outlier%", "Chunking", "Time(s)"]
    widths = [16, 6, 8, 7, 8, 9, 9, 10, 8]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{header}")
    print(sep)

    for r in runs:
        g = r["quality"].get("global", {})
        m = r.get("manifest", {})
        chunking = "semantic" if g.get("use_semantic_chunking") else "sliding"
        row = [
            r["run_id"][:16],
            str(g.get("n_documents", "")),
            str(g.get("n_chunks", "")),
            str(g.get("n_domains", "")),
            str(g.get("n_clusters", "")),
            f"{g.get('avg_quality_across_domains', 0):.4f}",
            f"{g.get('chunk_outlier_ratio', 0):.2%}",
            chunking,
            str(m.get("elapsed_seconds", "")),
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))

    # ── Per-stage timing comparison ────────────────────
    has_timings = any(r.get("manifest", {}).get("stage_timings") for r in runs)
    if has_timings:
        print(f"\n{'Stage Timings (seconds)':}")
        stages = set()
        for r in runs:
            stages.update(r.get("manifest", {}).get("stage_timings", {}).keys())
        stages = sorted(stages)

        stage_widths = [16] + [max(len(s), 10) for s in stages]
        stage_header = " | ".join(["Run ID".ljust(16)] + [s.ljust(w) for s, w in zip(stages, stage_widths[1:])])
        print(stage_header)
        print("-+-".join("-" * w for w in stage_widths))

        for r in runs:
            timings = r.get("manifest", {}).get("stage_timings", {})
            vals = [r["run_id"][:16]] + [f"{timings.get(s, 0):.1f}".ljust(w) for s, w in zip(stages, stage_widths[1:])]
            print(" | ".join(vals))

    # ── Per-domain breakdown ───────────────────────────
    if len(runs) > 0:
        print(f"\n{'Per-Domain Details (latest run)':}")
        latest = runs[-1]
        domains = latest["quality"].get("domains", {})
        if domains:
            d_cols = ["Domain", "Chunks", "Clusters", "Avg Qual", "Outlier%"]
            d_w = [35, 8, 8, 9, 9]
            print(" | ".join(c.ljust(w) for c, w in zip(d_cols, d_w)))
            print("-+-".join("-" * w for w in d_w))
            for did, d in domains.items():
                row = [
                    d.get("label", did)[:35],
                    str(d.get("n_chunks", "")),
                    str(d.get("n_clusters", "")),
                    f"{d.get('avg_quality', 0):.4f}",
                    f"{d.get('outlier_ratio', 0):.2%}",
                ]
                print(" | ".join(v.ljust(w) for v, w in zip(row, d_w)))

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare cluster quality across pipeline runs")
    parser.add_argument("run_dirs", nargs="*", help="Specific run directories to compare")
    parser.add_argument("--account-id", help="Compare all runs for this account")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    runs = []
    if args.account_id:
        runs = collect_runs(args.account_id)
    elif args.run_dirs:
        for d in args.run_dirs:
            data = load_run(Path(d))
            if data:
                runs.append(data)
            else:
                print(f"Warning: no cluster_quality.json in {d}", file=sys.stderr)
    else:
        # List all accounts and their runs
        if ARTIFACTS_DIR.exists():
            for acct_dir in sorted(ARTIFACTS_DIR.iterdir()):
                if acct_dir.is_dir():
                    acct_runs = collect_runs(acct_dir.name)
                    if acct_runs:
                        print(f"\n{'='*60}")
                        print(f"  Account: {acct_dir.name}")
                        print(f"{'='*60}")
                        print_comparison(acct_runs, args.json)
            return
        else:
            print("No artifacts directory found.", file=sys.stderr)
            return

    print_comparison(runs, args.json)


if __name__ == "__main__":
    main()
