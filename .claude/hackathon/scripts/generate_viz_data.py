#!/usr/bin/env python3
"""
Generate all 3 viz JSON files from a DuckDB pipeline database.

Usage:
    python scripts/generate_viz_data.py data/cuad_510.duckdb
    python scripts/generate_viz_data.py data/cuad_510.duckdb --output-dir data/cuad_viz/
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.store import ClusteringStore


def generate_all(db_path: str, output_dir: str | None = None):
    import duckdb as _ddb
    # Try writable store first, fall back to read-only duckdb connection
    try:
        store = ClusteringStore(db_path)
    except Exception:
        class _ROStore:
            """Minimal read-only wrapper matching ClusteringStore interface."""
            def __init__(self, path):
                self.conn = _ddb.connect(path, read_only=True)
            def get_domains(self):
                return self.conn.execute("SELECT * FROM domains").fetchdf().to_dict("records")
            def get_clusters(self):
                return self.conn.execute("SELECT * FROM clusters").fetchdf().to_dict("records")
            def get_agreements(self):
                return self.conn.execute("SELECT * FROM agreements").fetchdf().to_dict("records")
            def get_intent_types(self):
                try: return self.conn.execute("SELECT * FROM intent_types").fetchdf().to_dict("records")
                except: return []
            def get_stats(self):
                return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        for t in ['domains','agreements','clusters','field_definitions','extractions']}
            def get_fields(self, cluster_id=None):
                if cluster_id:
                    return self.conn.execute("SELECT * FROM field_definitions WHERE cluster_id = ?", [cluster_id]).fetchdf().to_dict("records")
                return self.conn.execute("SELECT * FROM field_definitions").fetchdf().to_dict("records")
            def get_chunks(self, cluster_id=None, agreement_id=None):
                if cluster_id:
                    return self.conn.execute("SELECT c.* FROM chunks c JOIN cluster_assignments ca ON c.chunk_id=ca.chunk_id WHERE ca.cluster_id=?", [cluster_id]).fetchdf().to_dict("records")
                if agreement_id:
                    return self.conn.execute("SELECT * FROM chunks WHERE agreement_id=?", [agreement_id]).fetchdf().to_dict("records")
                return self.conn.execute("SELECT * FROM chunks").fetchdf().to_dict("records")
            def get_clause_intents(self, clause_type_id=None):
                try:
                    if clause_type_id:
                        return self.conn.execute("""
                            SELECT ci.* FROM clause_intents ci
                            JOIN chunks ch ON ci.chunk_id=ch.chunk_id
                            JOIN cluster_assignments ca ON ch.chunk_id=ca.chunk_id
                            WHERE ca.cluster_id=?""", [clause_type_id]).fetchdf().to_dict("records")
                    return self.conn.execute("SELECT * FROM clause_intents").fetchdf().to_dict("records")
                except: return []
            def close(self):
                self.conn.close()
        store = _ROStore(db_path)
    out = Path(output_dir) if output_dir else Path(db_path).parent
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating viz data from {db_path} → {out}/")

    domains = store.get_domains()
    clusters = store.get_clusters()
    agreements = store.get_agreements()
    intent_types = store.get_intent_types()
    stats = store.get_stats()

    domain_map = {d["domain_id"]: d.get("label", d["domain_id"]) for d in domains}

    # ─── 1. ontology_viz_data.json ─────────────────────────────────────
    print("  [1/3] ontology_viz_data.json ...")

    # ── Compute cluster→domain via majority vote from member agreements ──
    # agreements already have real domain_id; clusters.domain_id may be stale
    agr_domain = {a["agreement_id"]: a.get("domain_id", "") for a in agreements}
    # Exclude the catch-all "All Documents" domain when a real one exists
    all_doc_domains = {d["domain_id"] for d in domains
                       if (d.get("label", "") or "").lower() in ("all documents", "")}

    cluster_domain_override = {}
    for c in clusters:
        cid = c["cluster_id"]
        chunk_rows = store.get_chunks(cluster_id=cid)
        dom_votes = Counter()
        for ch in chunk_rows:
            aid = ch.get("agreement_id", "")
            d = agr_domain.get(aid, "")
            if d and d not in all_doc_domains:
                dom_votes[d] += 1
        if dom_votes:
            best_dom = dom_votes.most_common(1)[0][0]
        else:
            best_dom = c.get("domain_id", "")
        cluster_domain_override[cid] = best_dom

    cluster_nodes = []
    cluster_intents = {}
    cluster_intent_sets = {}  # cid -> set of intent labels (for edge computation)
    for c in clusters:
        cid = c["cluster_id"]
        fields = store.get_fields(cluster_id=cid)
        field_names = [f["name"] for f in fields]
        chunks = store.get_chunks(cluster_id=cid)
        intents = store.get_clause_intents(clause_type_id=cid)

        real_domain = cluster_domain_override.get(cid, c.get("domain_id", ""))

        # Cluster importance score for ranking:
        #   coverage * field_richness * intent_diversity
        n_agreements = len({ch.get("agreement_id") for ch in chunks})
        coverage = n_agreements / max(len(agreements), 1)
        field_richness = min(len(fields) / 15.0, 1.0)
        intent_div = min(len(intents) / 10.0, 1.0) if intents else 0
        importance = round(0.4 * coverage + 0.35 * field_richness + 0.25 * intent_div, 4)

        cluster_nodes.append({
            "id": cid,
            "label": c.get("label", cid),
            "description": c.get("description", ""),
            "domain_id": real_domain,
            "domain_label": domain_map.get(real_domain, "Unknown"),
            "clauses": len(chunks),
            "agreements": n_agreements,
            "fields": len(fields),
            "field_names": field_names,
            "importance": importance,
            "domain_sim": round(c.get("quality_score", 0), 3),
        })

        if intents:
            _ic = Counter()
            for it in intents:
                _ic[it.get("intent_label", it.get("intent_type_id", ""))] += 1
            cluster_intents[cid] = [
                {"intent": label, "count": cnt}
                for label, cnt in _ic.most_common(20)
            ]
            cluster_intent_sets[cid] = set(_ic.keys())
        else:
            cluster_intent_sets[cid] = set()

    # ── Cluster edges: field_overlap + shared_intents ──
    edges = []
    # 1) field_overlap edges (shared field names between clusters)
    for i, c1 in enumerate(cluster_nodes):
        s1 = set(c1["field_names"])
        for c2 in cluster_nodes[i+1:]:
            s2 = set(c2["field_names"])
            shared = s1 & s2
            if len(shared) >= 2:
                jaccard = len(shared) / len(s1 | s2) if (s1 | s2) else 0
                edges.append({
                    "source": c1["id"], "target": c2["id"],
                    "shared_fields": sorted(shared),
                    "jaccard": round(jaccard, 3),
                    "shared_count": len(shared),
                    "type": "field_overlap",
                })

    # 2) shared_intents edges (clusters sharing intent labels)
    cids = [c["id"] for c in cluster_nodes]
    for i, cid1 in enumerate(cids):
        s1 = cluster_intent_sets.get(cid1, set())
        if not s1:
            continue
        for cid2 in cids[i+1:]:
            s2 = cluster_intent_sets.get(cid2, set())
            shared_int = s1 & s2
            if len(shared_int) >= 2:
                jaccard = len(shared_int) / len(s1 | s2) if (s1 | s2) else 0
                edges.append({
                    "source": cid1, "target": cid2,
                    "shared_intents": sorted(shared_int),
                    "jaccard": round(jaccard, 3),
                    "shared_count": len(shared_int),
                    "type": "shared_intents",
                })

    # Top intents
    intent_counter = Counter()
    for it in intent_types:
        intent_counter[it.get("label", it.get("intent_type_id", ""))] += it.get("frequency", 1)
    top_intents = [{"label": l, "count": c} for l, c in intent_counter.most_common(50)]

    # Build stats in the format the HTML viewer expects
    total_chunks = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_clustered = store.conn.execute("SELECT COUNT(*) FROM cluster_assignments").fetchone()[0]
    try:
        total_intents = store.conn.execute("SELECT COUNT(DISTINCT intent_type_id) FROM intent_types").fetchone()[0]
        total_clause_intents = store.conn.execute("SELECT COUNT(*) FROM clause_intents").fetchone()[0]
    except Exception:
        total_intents = 0
        total_clause_intents = 0

    viz_stats = {
        "agreements": len(agreements),
        "clauses": total_chunks,
        "clustered": total_clustered,
        "clusters": len(clusters),
        "fields": stats.get("fields", len(store.get_fields())),
        "extractions": stats.get("extractions", 0),
        "intent_types": total_intents,
        "clause_intents": total_clause_intents,
        "domains": len(domains),
    }

    ontology_data = {
        "stats": viz_stats,
        "domains": [{"id": d["domain_id"], "label": d.get("label", d["domain_id"])} for d in domains],
        "clusters": cluster_nodes,
        "cluster_intents": cluster_intents,
        "edges": edges,
        "top_intents": top_intents,
    }
    (out / "ontology_viz_data.json").write_text(json.dumps(ontology_data, indent=2))
    print(f"    {len(cluster_nodes)} clusters, {len(edges)} edges, {len(top_intents)} intents")

    # ─── 2. agreement_network_viz.json ─────────────────────────────────
    print("  [2/3] agreement_network_viz.json ...")

    n_agreements = len(agreements)

    # Build agreement → clause_types mapping (via cluster_assignments join)
    agr_clause_types = defaultdict(set)
    rows = store.conn.execute("""
        SELECT ch.agreement_id, ca.cluster_id
        FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id IS NOT NULL
    """).fetchall()
    for agr_id, cluster_id in rows:
        agr_clause_types[agr_id].add(cluster_id)
    # Ensure all agreements have entries
    for agr in agreements:
        agr_clause_types.setdefault(agr["agreement_id"], set())

    # IDF per clause type
    doc_freq = Counter()
    for ct_set in agr_clause_types.values():
        for ct in ct_set:
            doc_freq[ct] += 1

    ct_label_map = {c["cluster_id"]: c.get("label", c["cluster_id"]) for c in clusters}
    ct_chunk_count = {}
    for c in clusters:
        ct_chunk_count[c["cluster_id"]] = c.get("chunk_count", 0)

    def rarity_info(ct_id):
        freq = doc_freq.get(ct_id, 0)
        pct = (freq / n_agreements * 100) if n_agreements else 0
        if pct >= 15:
            return "boilerplate", "Common across docs", 0.2
        elif pct >= 5:
            return "common", "Moderately common", 0.5
        elif pct >= 2:
            return "specialized", "Specialized", 1.0
        else:
            return "unique", "Highly specific", 1.5

    # Clause type ranking
    ct_ranking = []
    for ct_id, freq in doc_freq.most_common():
        rarity, rarity_label, weight = rarity_info(ct_id)
        importance = round(weight * 100 / 1.5)  # normalize to 0-100
        ct_ranking.append({
            "clause_type": ct_label_map.get(ct_id, ct_id),
            "importance": importance,
            "rarity": rarity,
            "rarity_label": rarity_label,
            "doc_freq": freq,
            "doc_freq_pct": round(freq / n_agreements * 100, 1) if n_agreements else 0,
            "total_clauses": ct_chunk_count.get(ct_id, 0),
        })
    ct_ranking.sort(key=lambda x: -x["importance"])

    # Build agreement nodes
    agr_nodes = []
    for agr in agreements:
        aid = agr["agreement_id"]
        ct_set = agr_clause_types.get(aid, set())
        ct_details = []
        bp_count, sp_count, common_count, unique_count = 0, 0, 0, 0
        avg_imp = 0
        for ct_id in ct_set:
            rarity, rarity_label, weight = rarity_info(ct_id)
            imp = round(weight * 100 / 1.5)
            avg_imp += imp
            freq = doc_freq.get(ct_id, 0)
            pct = round(freq / n_agreements * 100, 1) if n_agreements else 0
            ct_details.append({
                "clause_type": ct_label_map.get(ct_id, ct_id),
                "importance": imp,
                "rarity": rarity,
                "rarity_label": rarity_label,
                "doc_freq_pct": pct,
            })
            if rarity == "boilerplate":
                bp_count += 1
            elif rarity == "common":
                common_count += 1
            elif rarity == "specialized":
                sp_count += 1
            elif rarity == "unique":
                unique_count += 1
        avg_imp = round(avg_imp / len(ct_set)) if ct_set else 0

        # Compute archetype from clause composition
        n_ct = len(ct_set)
        if n_ct == 0:
            archetype = "Empty"
        elif (sp_count + unique_count) / n_ct >= 0.5:
            archetype = "Distinctive"
        elif bp_count / n_ct >= 0.6:
            archetype = "Standardized"
        elif unique_count >= 2:
            archetype = "Niche"
        else:
            archetype = "Mixed"

        agr_nodes.append({
            "id": aid,
            "label": agr.get("filename", aid),
            "domain": domain_map.get(agr.get("domain_id", ""), "Unknown"),
            "archetype": archetype,
            "n_clause_types": n_ct,
            "n_connections": 0,  # filled below
            "best_connection_strength": 0,
            "avg_type_importance": avg_imp,
            "clause_types": ct_details,
            "boilerplate_count": bp_count,
            "specialized_count": sp_count + unique_count,
            "isolation_reason": None,
        })

    agr_node_map = {n["id"]: n for n in agr_nodes}

    # Build agreement edges (weighted Jaccard)
    rarity_weights = {"boilerplate": 0.2, "common": 0.5, "specialized": 1.0, "unique": 1.5}
    agr_edges = []
    agr_list = list(agr_clause_types.keys())
    conn_count = Counter()
    total_edges = 0

    for i, a1 in enumerate(agr_list):
        s1 = agr_clause_types[a1]
        if not s1:
            continue
        for a2 in agr_list[i+1:]:
            s2 = agr_clause_types[a2]
            if not s2:
                continue
            shared = s1 & s2
            if not shared:
                continue
            union = s1 | s2

            # Weighted Jaccard
            w_shared = sum(rarity_weights.get(rarity_info(ct)[0], 0.5) for ct in shared)
            w_union = sum(rarity_weights.get(rarity_info(ct)[0], 0.5) for ct in union)
            jaccard = w_shared / w_union if w_union else 0

            if jaccard < 0.15:
                continue

            total_edges += 1

            # Strength
            strength = round(jaccard * 100)
            if strength >= 60:
                strength_label = "Strong"
            elif strength >= 30:
                strength_label = "Moderate"
            else:
                strength_label = "Weak"

            # Breakdown
            breakdown = {"boilerplate": 0, "common": 0, "specialized": 0, "unique": 0}
            positive_reasons = []
            avg_imp = 0
            for ct in shared:
                rarity, rarity_label, weight = rarity_info(ct)
                breakdown[rarity] += 1
                imp = round(weight * 100 / 1.5)
                avg_imp += imp
                if rarity in ("specialized", "unique"):
                    positive_reasons.append({
                        "clause_type": ct_label_map.get(ct, ct),
                        "importance": imp,
                        "rarity": rarity,
                        "reason": f"Both have '{ct_label_map.get(ct, ct)}' ({rarity_label})",
                    })
            avg_imp = round(avg_imp / len(shared)) if shared else 0

            str_reason = f"Share {len(shared)} types"
            distinctive = breakdown["specialized"] + breakdown["unique"]
            if distinctive:
                str_reason += f" including {distinctive} distinctive"

            edge = {
                "source": a1, "target": a2,
                "jaccard": round(jaccard, 3),
                "shared_types": len(shared),
                "connection_strength": strength,
                "strength_label": strength_label,
                "strength_reason": str_reason,
                "avg_importance": avg_imp,
                "breakdown": breakdown,
                "positive_reasons": positive_reasons[:5],
                "negative_reasons": [],
                "shared_labels": sorted(ct_label_map.get(ct, ct) for ct in shared),
            }

            conn_count[a1] += 1
            conn_count[a2] += 1

            # Only keep top edges for viz (limit to avoid massive JSON)
            if strength >= 25 and len(agr_edges) < 5000:
                agr_edges.append(edge)

    # Update connection counts
    strong = sum(1 for e in agr_edges if e["strength_label"] == "Strong")
    moderate = sum(1 for e in agr_edges if e["strength_label"] == "Moderate")
    weak = sum(1 for e in agr_edges if e["strength_label"] == "Weak")
    isolated = 0
    for n in agr_nodes:
        n["n_connections"] = conn_count.get(n["id"], 0)
        best = 0
        for e in agr_edges:
            if e["source"] == n["id"] or e["target"] == n["id"]:
                best = max(best, e["connection_strength"])
        n["best_connection_strength"] = best
        if n["n_connections"] == 0:
            n["isolation_reason"] = "No shared clause types with other agreements"
            isolated += 1

    agr_edges.sort(key=lambda e: -e["connection_strength"])

    agr_network = {
        "nodes": agr_nodes,
        "edges": agr_edges[:3000],  # cap for viz performance
        "all_edges_count": total_edges,
        "clause_type_ranking": ct_ranking,
        "stats": {
            "total_agreements": n_agreements,
            "total_connections": total_edges,
            "filtered_connections": len(agr_edges),
            "strong_connections": strong,
            "moderate_connections": moderate,
            "weak_connections": weak,
            "isolated_agreements": isolated,
            "avg_connections": round(sum(conn_count.values()) / n_agreements, 1) if n_agreements else 0,
        },
    }
    (out / "agreement_network_viz.json").write_text(json.dumps(agr_network, indent=2))
    print(f"    {n_agreements} agreements, {total_edges} total edges, "
          f"{strong} strong, {moderate} moderate, {weak} weak, {isolated} isolated")

    # ─── 3. dashboard_data.json ────────────────────────────────────────
    print("  [3/3] dashboard_data.json ...")

    # Enrich clusters
    dash_clusters = []
    for c in clusters:
        cid = c["cluster_id"]
        fields = store.get_fields(cluster_id=cid)
        freq = doc_freq.get(cid, 0)
        rarity, rarity_label, weight = rarity_info(cid)
        importance = round(weight * 100 / 1.5)

        dash_clusters.append({
            "id": cid,
            "label": c.get("label", cid),
            "domain_id": c.get("domain_id", ""),
            "doc_freq": freq,
            "doc_freq_pct": round(freq / n_agreements * 100, 1) if n_agreements else 0,
            "clause_count": c.get("chunk_count", 0),
            "importance": importance,
            "rarity": rarity,
            "field_count": len(fields),
            "fields": [
                {"id": f["field_id"], "name": f["name"], "type": f.get("field_type", "string")}
                for f in fields
            ],
            "domain_label": domain_map.get(c.get("domain_id", ""), "Unknown"),
        })

    # Enrich agreements
    dash_agreements = []
    for agr in agreements:
        aid = agr["agreement_id"]
        chunks = store.get_chunks(agreement_id=aid)
        clustered = [ch for ch in chunks if ch.get("cluster_id")]
        unclustered = len(chunks) - len(clustered)
        ct_set = agr_clause_types.get(aid, set())
        extractions = store.get_extractions(agreement_id=aid)

        dash_agreements.append({
            "id": aid,
            "filename": agr.get("filename", aid),
            "domain_id": agr.get("domain_id", ""),
            "domain_confidence": agr.get("domain_confidence", 0),
            "n_clause_types": len(ct_set),
            "n_clauses": len(chunks),
            "unclustered_clauses": unclustered,
            "clause_types": sorted(ct_label_map.get(ct, ct) for ct in ct_set),
            "extractions": len(extractions),
            "domain_label": domain_map.get(agr.get("domain_id", ""), "Unknown"),
        })

    # Enrich domains
    dash_domains = []
    for d in domains:
        did = d["domain_id"]
        agr_count = sum(1 for a in agreements if a.get("domain_id") == did)
        confs = [a.get("domain_confidence", 0) for a in agreements if a.get("domain_id") == did]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0
        dash_domains.append({
            "id": did,
            "label": d.get("label", did),
            "agreement_count": agr_count,
            "avg_confidence": avg_conf,
        })

    # Stats
    all_extractions = store.get_extractions()
    high_conf = sum(1 for e in all_extractions if e.get("confidence", 0) >= 0.7)
    low_conf = sum(1 for e in all_extractions if e.get("confidence", 0) < 0.5)
    all_chunks = store.get_chunks()
    unclustered_total = sum(1 for ch in all_chunks if not ch.get("cluster_id"))

    dash_stats = {
        "total_agreements": n_agreements,
        "total_clauses": len(all_chunks),
        "total_clusters": len(clusters),
        "total_fields": sum(c["field_count"] for c in dash_clusters),
        "total_extractions": len(all_extractions),
        "unclustered_clauses": unclustered_total,
        "unclustered_pct": round(unclustered_total / len(all_chunks) * 100, 1) if all_chunks else 0,
        "high_conf_extractions": high_conf,
        "low_conf_extractions": low_conf,
        "high_conf_pct": round(high_conf / len(all_extractions) * 100, 1) if all_extractions else 0,
    }

    dashboard_data = {
        "domains": dash_domains,
        "clusters": dash_clusters,
        "agreements": dash_agreements,
        "stats": dash_stats,
    }
    (out / "dashboard_data.json").write_text(json.dumps(dashboard_data, indent=2))
    print(f"    {len(dash_domains)} domains, {len(dash_clusters)} clusters, "
          f"{len(dash_agreements)} agreements, {len(all_extractions)} extractions")

    store.close()
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate viz JSON files from pipeline DB")
    parser.add_argument("db_path", help="Path to DuckDB file")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as DB)")
    args = parser.parse_args()
    generate_all(args.db_path, args.output_dir)
