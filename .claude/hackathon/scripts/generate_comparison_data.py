#!/usr/bin/env python3
"""
Generate comparison viz data from field_discovery_comparison.json
for the HTML comparison dashboard.

Usage:
    python scripts/generate_comparison_data.py data/field_discovery_comparison.json --output-dir cuad_html/data
"""
import json
import sys
from pathlib import Path


def generate(comparison_path: str, output_dir: str | None = None):
    with open(comparison_path) as f:
        data = json.load(f)

    out = Path(output_dir) if output_dir else Path(comparison_path).parent
    out.mkdir(parents=True, exist_ok=True)

    summary = data["summary"]
    per_cluster = data["per_cluster"]

    # Build comparison viz data
    result = {
        "summary": summary,
        "clusters": [],
    }

    for r in per_cluster:
        std = r["standard"]
        rlm = r["rlm"]
        overlap = r["overlap"]

        # Build detailed field comparison per cluster
        std_field_map = {f["name"]: f for f in std.get("fields", [])}
        rlm_field_map = {f["name"]: f for f in rlm.get("fields", [])}
        all_field_names = sorted(set(std_field_map.keys()) | set(rlm_field_map.keys()))

        fields_detail = []
        for fn in all_field_names:
            in_std = fn in std_field_map
            in_rlm = fn in rlm_field_map
            std_f = std_field_map.get(fn, {})
            rlm_f = rlm_field_map.get(fn, {})
            fields_detail.append({
                "name": fn,
                "in_standard": in_std,
                "in_rlm": in_rlm,
                "in_both": in_std and in_rlm,
                "std_type": std_f.get("type", ""),
                "rlm_type": rlm_f.get("type", ""),
                "type_match": in_std and in_rlm and std_f.get("type") == rlm_f.get("type"),
                "std_description": std_f.get("description", ""),
                "rlm_description": rlm_f.get("description", ""),
            })

        cluster_entry = {
            "cluster_id": r["cluster_id"],
            "label": r["cluster_label"],
            "chunk_count": r["chunk_count"],
            "n_chunks_used": r.get("n_chunks_used", 0),
            "std_fields": std["n_fields"],
            "rlm_fields": rlm["n_fields"],
            "std_time": std["time_s"],
            "rlm_time": rlm["time_s"],
            "jaccard": overlap["jaccard"],
            "type_agreement": r["type_agreement"],
            "exact_overlap": overlap["exact_overlap"],
            "only_std": overlap.get("only_in_a", []),
            "only_rlm": overlap.get("only_in_b", []),
            "common": overlap.get("common", []),
            "std_ext_fill": std.get("extraction", {}).get("fill_rate") if std.get("extraction") else None,
            "rlm_ext_fill": rlm.get("extraction", {}).get("fill_rate") if rlm.get("extraction") else None,
            "std_ext_conf": std.get("extraction", {}).get("avg_confidence") if std.get("extraction") else None,
            "rlm_ext_conf": rlm.get("extraction", {}).get("avg_confidence") if rlm.get("extraction") else None,
            "fields": fields_detail,
        }
        result["clusters"].append(cluster_entry)

    dest = out / "comparison_viz.json"
    dest.write_text(json.dumps(result, indent=2, default=str))
    print(f"Written {dest} ({len(result['clusters'])} clusters)")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("comparison_path", help="Path to field_discovery_comparison.json")
    p.add_argument("--output-dir", help="Output directory")
    args = p.parse_args()
    generate(args.comparison_path, args.output_dir)
