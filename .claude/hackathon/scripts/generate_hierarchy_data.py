#!/usr/bin/env python3
"""
Generate hierarchy_viz.json from native EVoC layers stored in
cluster_layers / cluster_layer_meta tables.

Produces EXPLAINABLE hierarchy data:
  - Semantic cluster names derived from pipeline labels
  - Merge narratives between consecutive layers
  - Legal-domain context (fields, intents) per cluster
  - Connected to ontology explorer data

Usage:
    python scripts/generate_hierarchy_data.py data/cuad_510_demo.duckdb
    python scripts/generate_hierarchy_data.py data/cuad_510_demo.duckdb --output-dir cuad_html/data
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.store import ClusteringStore


# ── helpers ───────────────────────────────────────────────────────
def _derive_cluster_name(top_labels: list[tuple[str, int]], max_parts=2) -> str:
    """Derive a short semantic name from top pipeline labels in a cluster."""
    if not top_labels:
        return "Miscellaneous"
    if len(top_labels) == 1:
        return top_labels[0][0]
    # Use top 1-2 labels, abbreviating if needed
    names = [n for n, _ in top_labels[:max_parts]]
    return " & ".join(names)


def _derive_theme(cluster_names: list[str]) -> str:
    """Derive a layer-level theme from its cluster names."""
    # Group by common legal keywords
    keywords = Counter()
    legal_terms = [
        "confidential", "ip", "intellectual property", "license", "trademark",
        "patent", "termination", "boilerplate", "dispute", "arbitration",
        "warranty", "indemnif", "agreement", "contract", "franchise",
        "compliance", "regulatory", "governance", "financial", "service",
    ]
    for name in cluster_names:
        nl = name.lower()
        for t in legal_terms:
            if t in nl:
                keywords[t] += 1
    if not keywords:
        return "Mixed legal provisions"
    top = [k for k, _ in keywords.most_common(3)]
    return ", ".join(t.title() for t in top) + " provisions"


def generate(db_path: str, output_dir: str | None = None):
    store = ClusteringStore(db_path)
    out = Path(output_dir) if output_dir else Path(db_path).parent
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating hierarchy data from {db_path} -> {out}/")

    runs = store.get_cluster_layer_runs()
    if not runs:
        print("  No cluster layers found. Run recluster_with_layers.py first.")
        store.close()
        return

    print(f"  Found clustering runs: {runs}")

    all_layer_meta = {}
    for run in runs:
        meta = store.get_cluster_layer_meta(run)
        all_layer_meta[run] = meta
        print(f"    {run}: {len(meta)} layers")
        for m in meta:
            sel = " <- SELECTED" if m.get("is_selected") else ""
            print(f"      L{m['layer_index']}: {m['n_clusters']} clusters, "
                  f"{m['n_outliers']} outliers, "
                  f"composite={m.get('composite_score', 'N/A')}{sel}")

    clusters = store.get_clusters()
    domains = store.get_domains()
    agreements = store.get_agreements()
    stats = store.get_stats()
    fields = store.get_fields()
    domain_label_map = {d["domain_id"]: d["label"] for d in domains}
    agr_domain = {a["agreement_id"]: a.get("domain_id", "") for a in agreements}
    cluster_label_map = {c["cluster_id"]: c.get("label", c["cluster_id"]) for c in clusters}
    cluster_fields_map = defaultdict(list)
    for f in fields:
        cluster_fields_map[f.get("cluster_id", "")].append(f.get("field_name", ""))

    # ── Build item_id -> pipeline label mappings ──────────────────

    # Clause: item_id = clause_id -> chunk -> cluster_assignment -> cluster label
    clause_to_pipeline = {}
    try:
        rows = store.conn.execute("""
            SELECT ch.clause_id, c.label
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            JOIN clusters c ON ca.cluster_id = c.cluster_id
        """).fetchall()
        for clause_id, label in rows:
            clause_to_pipeline[clause_id] = label
    except Exception:
        pass

    # Macro: item_id = agreement_id -> domain label
    agr_to_pipeline = {}
    for a in agreements:
        did = a.get("domain_id", "")
        agr_to_pipeline[a["agreement_id"]] = domain_label_map.get(did, "Unknown")

    # ── Derive semantic names for every EVoC cluster in every layer ──
    # evoc_names[run][layer_index][evoc_label] = {"name": str, "top_labels": [...]}
    evoc_names = {}
    all_assignments_cache = {}  # run -> layer_idx -> {item_index: label}
    all_item_ids_cache = {}     # run -> layer_idx -> {item_index: item_id}

    for run in runs:
        pipeline_map = clause_to_pipeline if "clause" in run else agr_to_pipeline
        evoc_names[run] = {}
        all_assignments_cache[run] = {}
        all_item_ids_cache[run] = {}

        for m in all_layer_meta[run]:
            li = m["layer_index"]
            assigns = store.get_cluster_layer(run, li)
            all_assignments_cache[run][li] = {a["item_index"]: a["cluster_label"] for a in assigns}
            all_item_ids_cache[run][li] = {a["item_index"]: a["item_id"] for a in assigns}

            # Group items by EVoC label, collect pipeline labels
            label_to_pipeline_labels = defaultdict(list)
            for a in assigns:
                if a["cluster_label"] == -1:
                    continue
                pl = pipeline_map.get(a["item_id"])
                if pl:
                    label_to_pipeline_labels[a["cluster_label"]].append(pl)

            evoc_names[run][li] = {}
            for evoc_label, pls in label_to_pipeline_labels.items():
                top = Counter(pls).most_common(5)
                name = _derive_cluster_name(top)
                evoc_names[run][li][evoc_label] = {
                    "name": name,
                    "top_labels": [{"label": n, "count": c} for n, c in top],
                    "total": sum(c for _, c in top),
                }

    print("\n  Semantic names derived for all layers")

    # ── Build enriched layer stacks ──────────────────────────────
    layer_stacks = {}
    for run in runs:
        meta_list = all_layer_meta[run]
        layers_out = []
        for m in meta_list:
            li = m["layer_index"]
            assigns_dict = all_assignments_cache[run][li]
            label_counts = Counter(v for v in assigns_dict.values() if v != -1)
            names_for_layer = evoc_names[run].get(li, {})

            cluster_details = []
            for cl, cnt in label_counts.most_common():
                info = names_for_layer.get(cl, {})
                cluster_details.append({
                    "evoc_label": cl,
                    "name": info.get("name", f"C{cl}"),
                    "count": cnt,
                    "top_labels": info.get("top_labels", [])[:3],
                    "fields": cluster_fields_map.get(cl, [])[:5],  # won't match for non-selected
                })

            cluster_sizes = sorted(label_counts.values(), reverse=True)
            # Derive a layer-level description
            cnames = [cd["name"] for cd in cluster_details]
            n_cl = m["n_clusters"]
            if n_cl >= 15:
                granularity = "fine-grained"
            elif n_cl >= 7:
                granularity = "mid-level"
            else:
                granularity = "broad"
            run_type = "clause types" if "clause" in run else "contract categories"
            description = f"{n_cl} {granularity} {run_type}"

            layers_out.append({
                "layer_index": li,
                "n_clusters": n_cl,
                "n_outliers": m["n_outliers"],
                "n_items": len(assigns_dict),
                "description": description,
                "persistence_score": m.get("persistence_score"),
                "silhouette_score": m.get("silhouette_score"),
                "cosine_score": m.get("cosine_score"),
                "composite_score": m.get("composite_score"),
                "is_selected": bool(m.get("is_selected")),
                "cluster_sizes": cluster_sizes[:30],
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "clusters": cluster_details,
            })
        layer_stacks[run] = layers_out

    # ── Build merge narratives between consecutive layers ─────────
    merges = {}
    for run in runs:
        meta_list = all_layer_meta[run]
        layer_indices = sorted(all_assignments_cache[run].keys())
        run_merges = []

        for i in range(len(layer_indices) - 1):
            li_from = layer_indices[i]
            li_to = layer_indices[i + 1]
            a_from = all_assignments_cache[run][li_from]
            a_to = all_assignments_cache[run][li_to]
            names_from = evoc_names[run].get(li_from, {})
            names_to = evoc_names[run].get(li_to, {})

            # Track which from-clusters map to which to-clusters
            flow_map = defaultdict(lambda: Counter())  # to_label -> from_label -> count
            for idx in set(a_from.keys()) & set(a_to.keys()):
                cf = a_from[idx]
                ct = a_to[idx]
                if cf == -1 or ct == -1:
                    continue
                flow_map[ct][cf] += 1

            n_from = len(set(v for v in a_from.values() if v != -1))
            n_to = len(set(v for v in a_to.values() if v != -1))
            notable = []

            for to_label, from_counts in flow_map.items():
                sources = from_counts.most_common()
                if len(sources) > 1:
                    # This is a true merge
                    source_names = []
                    for fl, fc in sources:
                        fn = names_from.get(fl, {}).get("name", f"C{fl}")
                        source_names.append({"label": fl, "name": fn, "count": fc})
                    result_name = names_to.get(to_label, {}).get("name", f"C{to_label}")
                    notable.append({
                        "result_label": to_label,
                        "result_name": result_name,
                        "sources": source_names,
                        "explanation": f"{' + '.join(s['name'] for s in source_names[:3])} → {result_name}",
                    })

            # Sort by number of sources (most interesting merges first)
            notable.sort(key=lambda x: len(x["sources"]), reverse=True)

            run_type = "clause types" if "clause" in run else "contract categories"
            run_merges.append({
                "from_layer": li_from,
                "to_layer": li_to,
                "from_count": n_from,
                "to_count": n_to,
                "description": f"{n_from} {run_type} consolidate into {n_to}",
                "notable_merges": notable[:10],
            })

        merges[run] = run_merges

    print("  Merge narratives generated")

    # ── Build alluvial flows (with names) ─────────────────────────
    alluvial = {}
    for run in runs:
        meta_list = all_layer_meta[run]
        if len(meta_list) < 2:
            alluvial[run] = {"nodes": [], "links": []}
            continue

        nodes = []
        links = []
        node_id_map = {}

        for m in meta_list:
            li = m["layer_index"]
            assigns = all_assignments_cache[run][li]
            names_map = evoc_names[run].get(li, {})
            label_counts = Counter(v for v in assigns.values() if v != -1)
            n_outliers = sum(1 for v in assigns.values() if v == -1)

            if n_outliers > 0:
                nid = len(nodes)
                node_id_map[(li, -1)] = nid
                nodes.append({
                    "id": f"{run}_L{li}_outlier", "layer": li,
                    "cluster_label": -1, "name": "Outliers",
                    "semantic_name": "Outliers",
                    "count": n_outliers,
                    "is_selected": bool(m.get("is_selected")), "is_outlier": True,
                })

            for cl, cnt in label_counts.most_common():
                nid = len(nodes)
                node_id_map[(li, cl)] = nid
                info = names_map.get(cl, {})
                sem_name = info.get("name", f"C{cl}")
                nodes.append({
                    "id": f"{run}_L{li}_C{cl}", "layer": li,
                    "cluster_label": cl,
                    "name": sem_name,
                    "semantic_name": sem_name,
                    "top_labels": info.get("top_labels", [])[:3],
                    "count": cnt,
                    "is_selected": bool(m.get("is_selected")), "is_outlier": False,
                })

        layer_indices = sorted(all_assignments_cache[run].keys())
        for i in range(len(layer_indices) - 1):
            li_from = layer_indices[i]
            li_to = layer_indices[i + 1]
            a_from = all_assignments_cache[run][li_from]
            a_to = all_assignments_cache[run][li_to]
            flows = Counter()
            for idx in set(a_from.keys()) & set(a_to.keys()):
                flows[(a_from[idx], a_to[idx])] += 1
            for (cf, ct), count in flows.items():
                src = node_id_map.get((li_from, cf))
                tgt = node_id_map.get((li_to, ct))
                if src is not None and tgt is not None and count > 0:
                    links.append({"source": src, "target": tgt, "value": count})

        alluvial[run] = {"nodes": nodes, "links": links}

    # ── Dual track ────────────────────────────────────────────────
    clause_sel = macro_sel = None
    for run in runs:
        for m in all_layer_meta[run]:
            if m.get("is_selected"):
                info = {"run": run, "layer_index": m["layer_index"],
                        "n_clusters": m["n_clusters"], "n_outliers": m["n_outliers"],
                        "composite_score": m.get("composite_score")}
                if "clause" in run:
                    clause_sel = info
                elif "macro" in run:
                    macro_sel = info

    domain_nodes = [{"id": d["domain_id"], "name": d["label"],
                     "agreement_count": d.get("agreement_count", 0)}
                    for d in sorted(domains, key=lambda x: x.get("agreement_count", 0), reverse=True)
                    if d["label"].lower() != "all documents"]

    cluster_agr_map = defaultdict(set)
    try:
        rows = store.conn.execute("""
            SELECT ca.cluster_id, ch.agreement_id
            FROM cluster_assignments ca JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            WHERE ca.is_outlier = false""").fetchall()
        for cid, aid in rows:
            cluster_agr_map[cid].add(aid)
    except Exception:
        pass

    cluster_domain_flows = []
    for c in sorted(clusters, key=lambda x: x.get("chunk_count", 0), reverse=True)[:50]:
        agrs = cluster_agr_map.get(c["cluster_id"], set())
        dd = Counter(agr_domain.get(aid, "") for aid in agrs)
        cf_list = cluster_fields_map.get(c["cluster_id"], [])[:5]
        for did, cnt in dd.most_common(3):
            if did and domain_label_map.get(did, "").lower() != "all documents":
                cluster_domain_flows.append({
                    "cluster_id": c["cluster_id"],
                    "cluster_name": c.get("label", c["cluster_id"]),
                    "cluster_chunks": c.get("chunk_count", 0),
                    "cluster_fields": cf_list,
                    "domain_id": did,
                    "domain_name": domain_label_map.get(did, did),
                    "agreement_count": cnt,
                })

    dual_track = {
        "clause_selected": clause_sel,
        "macro_selected": macro_sel,
        "domains": domain_nodes,
        "top_clusters": [
            {"id": c["cluster_id"], "name": c.get("label", c["cluster_id"]),
             "chunk_count": c.get("chunk_count", 0),
             "agreement_count": c.get("agreement_count", 0),
             "quality_score": round(c.get("quality_score", 0), 3),
             "fields": cluster_fields_map.get(c["cluster_id"], [])[:5]}
            for c in sorted(clusters, key=lambda x: x.get("chunk_count", 0), reverse=True)[:40]
        ],
        "cluster_domain_flows": cluster_domain_flows,
    }

    # ── Ontology connection: cluster intents ──────────────────────
    cluster_intents = defaultdict(list)
    try:
        rows = store.conn.execute("""
            SELECT ca.cluster_id, ci.intent_label, COUNT(*) as cnt
            FROM cluster_assignments ca
            JOIN chunks ch ON ca.chunk_id = ch.chunk_id
            JOIN clause_intents ci ON ch.clause_id = ci.clause_id
            GROUP BY ca.cluster_id, ci.intent_label
            ORDER BY ca.cluster_id, cnt DESC
        """).fetchall()
        for cid, intent, cnt in rows:
            if len(cluster_intents[cid]) < 5:
                cluster_intents[cid].append({"intent": intent, "count": cnt})
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────────────
    ftd = Counter(f.get("field_type", "text") for f in fields)
    try:
        n_clauses = store.conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0]
    except Exception:
        n_clauses = 0

    summary = {
        "total_agreements": len(agreements),
        "total_clusters": len(clusters),
        "total_domains": len([d for d in domains if d["label"].lower() != "all documents"]),
        "total_fields": len(fields),
        "total_extractions": stats.get("extractions", 0),
        "field_type_distribution": dict(ftd),
        "layers": [
            {"name": "Documents", "icon": "\U0001f4c4", "count": len(agreements), "color": "#58a6ff"},
            {"name": "Clauses", "icon": "\U0001f4dc", "count": n_clauses, "color": "#79c0ff"},
            {"name": "Clusters", "icon": "\U0001f52c", "count": len(clusters), "color": "#3fb950"},
            {"name": "Domains", "icon": "\U0001f3f7\ufe0f",
             "count": len([d for d in domains if d["label"].lower() != "all documents"]), "color": "#d29922"},
            {"name": "Fields", "icon": "\U0001f4cb", "count": len(fields), "color": "#bc8cff"},
            {"name": "Extractions", "icon": "\u26cf\ufe0f",
             "count": stats.get("extractions", 0), "color": "#f778ba"},
        ],
        "clustering_runs": {
            run: {
                "n_layers": len(all_layer_meta[run]),
                "selected_layer": next(
                    (m["layer_index"] for m in all_layer_meta[run] if m.get("is_selected")), None),
                "layer_range": (
                    f"{min(m['n_clusters'] for m in all_layer_meta[run])} - "
                    f"{max(m['n_clusters'] for m in all_layer_meta[run])} clusters"
                ) if all_layer_meta[run] else "N/A",
            }
            for run in runs
        },
    }

    result = {
        "layer_stacks": layer_stacks,
        "alluvial": alluvial,
        "dual_track": dual_track,
        "merges": merges,
        "cluster_intents": {k: v for k, v in cluster_intents.items()},
        "summary": summary,
    }

    dest = out / "hierarchy_viz.json"
    dest.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Written to {dest}")
    for run in runs:
        ls = layer_stacks[run]
        al = alluvial[run]
        mg = merges.get(run, [])
        print(f"    {run}: {len(ls)} layers, "
              f"{len(al['nodes'])} alluvial nodes, {len(al['links'])} links, "
              f"{sum(len(m['notable_merges']) for m in mg)} merge narratives")
    print("Done!")
    store.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate hierarchy viz JSON from EVoC layers")
    p.add_argument("db_path", help="Path to DuckDB file")
    p.add_argument("--output-dir", help="Output directory")
    args = p.parse_args()
    generate(args.db_path, args.output_dir)
