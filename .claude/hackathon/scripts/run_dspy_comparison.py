#!/usr/bin/env python3
"""
Run DSPy v2 pipeline on the same 5 clusters as the last comparison run,
then produce a side-by-side report against the stored baseline.
"""
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

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rlm_v2_dspy import (
    ClusterDiscoveryPipeline,
    _configure_dspy_lm,
    discovery_quality_metric,
)
from core.field_discovery import (
    FieldDiscoveryConfig,
    extract_fields_from_chunk,
    _get_azure_client,
    _get_deployment,
)
import dspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def grounding_detail(value, chunk_text):
    if not value or not chunk_text:
        return {"level": "missing", "score": 0.0}
    v_lower = value.lower().strip()
    c_lower = chunk_text.lower()
    if v_lower in c_lower:
        return {"level": "verbatim", "score": 1.0}
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    c_tokens = set(re.findall(r'\b\w+\b', c_lower))
    if not v_tokens:
        return {"level": "empty", "score": 0.0}
    overlap = v_tokens & c_tokens
    ratio = len(overlap) / len(v_tokens)
    if ratio >= 0.8:
        return {"level": "high_overlap", "score": 0.8, "token_ratio": round(ratio, 3)}
    elif ratio >= 0.5:
        return {"level": "partial_overlap", "score": 0.5, "token_ratio": round(ratio, 3)}
    elif ratio >= 0.2:
        return {"level": "weak_overlap", "score": 0.2, "token_ratio": round(ratio, 3)}
    return {"level": "hallucinated", "score": 0.0, "token_ratio": round(ratio, 3)}


def classify_value(value):
    if not value:
        return "null"
    v = value.strip()
    if re.match(r'^\d+$', v):
        return "integer"
    if re.match(r'^\$?[\d,]+\.?\d*$', v):
        return "currency"
    if re.match(r'^\d+(\.\d+)?%$', v):
        return "percentage"
    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', v):
        return "date"
    if v.lower() in ("true", "false", "yes", "no"):
        return "boolean"
    if len(v) > 100:
        return "long_text"
    return "short_text"


def main():
    db_path = "data/cuad_510_demo.duckdb"
    n_extract_chunks = 8

    # Load baseline
    baseline_path = Path("data/extraction_value_comparison_v2.json")
    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_clusters = {r["cluster_id"]: r for r in baseline["per_cluster"]}
    cluster_ids = [r["cluster_id"] for r in baseline["per_cluster"]]

    logger.info(f"Running DSPy pipeline on {len(cluster_ids)} clusters (same as baseline)")

    # Setup
    lm = _configure_dspy_lm()
    config = FieldDiscoveryConfig()
    client = _get_azure_client()
    deployment = _get_deployment(config)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create pipeline (no deep explore for fair comparison — same LLM budget)
    pipeline = ClusterDiscoveryPipeline(config=config, embedder=embedder)

    conn = duckdb.connect(db_path, read_only=True)
    all_results = []

    for ci, cid in enumerate(cluster_ids):
        cluster = conn.execute(
            "SELECT cluster_id, label, keywords, chunk_count FROM clusters WHERE cluster_id = ?",
            [cid],
        ).fetchdf().to_dict("records")[0]

        cluster_label = cluster["label"] or "Unknown"
        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunks = conn.execute("""
            SELECT ch.chunk_text FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cid]).fetchdf()["chunk_text"].tolist()

        extract_chunks = chunks[:n_extract_chunks]

        logger.info(f"\n{'='*80}")
        logger.info(f"[{ci+1}/{len(cluster_ids)}] {cluster_label} ({len(chunks)} chunks)")
        logger.info(f"{'='*80}")

        # ── Discovery ──
        t0 = time.time()
        try:
            result = pipeline(
                cluster_label=cluster_label,
                keywords=", ".join(keywords[:10]),
                chunk_texts=chunks,
            )
            fields = result.fields
            meta = result.meta
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            fields = []
            meta = {"error": str(e)}
        disc_time = round(time.time() - t0, 2)
        logger.info(f"  Discovery: {len(fields)} fields in {disc_time}s")

        # ── Extraction (same as baseline: extract_fields_from_chunk) ──
        rlm_context = None
        if meta.get("clause_types"):
            rlm_context = {
                "clause_types": meta.get("clause_types", []),
                "intents": meta.get("intents", []),
                "obligations": meta.get("obligations", []),
            }

        t0 = time.time()
        chunk_extractions = []
        for chunk_idx, chunk_text in enumerate(extract_chunks):
            try:
                ext = extract_fields_from_chunk(
                    client, deployment, cluster_label, chunk_text, fields, config,
                    rlm_context=rlm_context,
                )
            except Exception as e:
                logger.warning(f"  Chunk {chunk_idx} extraction failed: {e}")
                ext = {}

            chunk_detail = {"chunk_index": chunk_idx, "fields": {}}
            for f in fields:
                fname = f["name"]
                if fname in ext:
                    value, confidence = ext[fname]
                    grounding = grounding_detail(value, chunk_text)
                    chunk_detail["fields"][fname] = {
                        "value": value,
                        "confidence": confidence,
                        "grounding": grounding,
                        "value_type": classify_value(value),
                    }
                else:
                    chunk_detail["fields"][fname] = {
                        "value": None,
                        "confidence": 0.0,
                        "grounding": {"level": "missing", "score": 0.0},
                        "value_type": "null",
                    }
            chunk_extractions.append(chunk_detail)

        ext_time = round(time.time() - t0, 2)

        # ── Aggregate ──
        n_fields = len(fields)
        n_chunks = len(chunk_extractions)
        total_slots = n_fields * n_chunks
        values, confs, groundings = [], [], []
        for ce in chunk_extractions:
            for f in fields:
                fd = ce["fields"].get(f["name"], {})
                val = fd.get("value")
                if val is not None:
                    values.append(val)
                    confs.append(fd.get("confidence", 0))
                    groundings.append(fd.get("grounding", {}).get("level", "missing"))

        n_ext = len(values)
        verbatim = sum(1 for g in groundings if g == "verbatim")
        high_ov = sum(1 for g in groundings if g == "high_overlap")
        halluc = sum(1 for g in groundings if g == "hallucinated")

        agg = {
            "fill_rate": round(n_ext / total_slots, 3) if total_slots else 0,
            "avg_confidence": round(np.mean(confs), 3) if confs else 0,
            "verbatim_rate": round(verbatim / n_ext, 3) if n_ext else 0,
            "grounded_rate": round((verbatim + high_ov) / n_ext, 3) if n_ext else 0,
            "hallucination_rate": round(halluc / n_ext, 3) if n_ext else 0,
            "n_extractions": n_ext,
            "n_total_slots": total_slots,
        }

        # DSPy quality metric score
        example = dspy.Example(
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            chunk_texts=chunks,
        ).with_inputs("cluster_label", "keywords", "chunk_texts")
        prediction = dspy.Prediction(fields=fields, meta=meta)
        quality_score = discovery_quality_metric(example, prediction)

        cluster_result = {
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "chunk_count": cluster["chunk_count"],
            "n_fields": n_fields,
            "discovery_time_s": disc_time,
            "extraction_time_s": ext_time,
            "aggregate": agg,
            "meta": meta,
            "quality_score": quality_score,
            "extractions": chunk_extractions,
            "fields": fields,
        }
        all_results.append(cluster_result)

        logger.info(f"  fill={agg['fill_rate']:.3f}  verb={agg['verbatim_rate']:.3f}  "
                     f"halluc={agg['hallucination_rate']:.3f}  quality={quality_score:.4f}  "
                     f"disc={disc_time}s  ext={ext_time}s")

    conn.close()

    # ── Save results ──
    out_path = Path("data/dspy_v2_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"dspy_results": all_results}, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # ── Print comparison table ──
    print(f"\n{'='*120}")
    print(f" DSPy v2 vs Baseline Comparison ({len(all_results)} clusters)")
    print(f"{'='*120}")

    # Aggregate DSPy results
    dspy_aggs = [r["aggregate"] for r in all_results]
    dspy_valid = [a for a in dspy_aggs if a["n_extractions"] > 0]

    # Per-cluster detail
    print(f"\n{'Cluster':<25} │ {'Method':<12} │ {'Fields':>6} │ {'Fill':>6} │ {'Verb':>6} │ {'Ground':>6} │ {'Halluc':>6} │ {'DiscT':>6} │ {'ExtT':>5}")
    print(f"{'─'*120}")

    for r in all_results:
        cid = r["cluster_id"]
        bl = baseline_clusters.get(cid, {})
        label = r["cluster_label"][:23]

        # DSPy row
        a = r["aggregate"]
        print(f"{label:<25} │ {'DSPy v2':<12} │ {r['n_fields']:>6} │ {a['fill_rate']:>6.3f} │ "
              f"{a['verbatim_rate']:>6.3f} │ {a['grounded_rate']:>6.3f} │ {a['hallucination_rate']:>6.3f} │ "
              f"{r['discovery_time_s']:>5.1f}s │ {r['extraction_time_s']:>4.1f}s")

        # Baseline rlm_v2 row
        bl_v2 = bl.get("methods", {}).get("rlm_v2", {})
        ba = bl_v2.get("aggregate", {})
        if ba:
            print(f"{'':25} │ {'OG rlm_v2':<12} │ {bl_v2.get('n_fields', 0):>6} │ {ba.get('fill_rate', 0):>6.3f} │ "
                  f"{ba.get('verbatim_rate', 0):>6.3f} │ {ba.get('grounded_rate', 0):>6.3f} │ {ba.get('hallucination_rate', 0):>6.3f} │ "
                  f"{bl_v2.get('discovery_time_s', 0):>5.1f}s │ {bl_v2.get('extraction_time_s', 0):>4.1f}s")

        # Baseline best (dspy_rlm for grounding)
        bl_rlm = bl.get("methods", {}).get("dspy_rlm", {})
        bra = bl_rlm.get("aggregate", {})
        if bra:
            print(f"{'':25} │ {'OG dspy_rlm':<12} │ {bl_rlm.get('n_fields', 0):>6} │ {bra.get('fill_rate', 0):>6.3f} │ "
                  f"{bra.get('verbatim_rate', 0):>6.3f} │ {bra.get('grounded_rate', 0):>6.3f} │ {bra.get('hallucination_rate', 0):>6.3f} │ "
                  f"{bl_rlm.get('discovery_time_s', 0):>5.1f}s │ {bl_rlm.get('extraction_time_s', 0):>4.1f}s")

        # Baseline standard_gpt
        bl_std = bl.get("methods", {}).get("standard_gpt", {})
        bsa = bl_std.get("aggregate", {})
        if bsa:
            print(f"{'':25} │ {'OG std_gpt':<12} │ {bl_std.get('n_fields', 0):>6} │ {bsa.get('fill_rate', 0):>6.3f} │ "
                  f"{bsa.get('verbatim_rate', 0):>6.3f} │ {bsa.get('grounded_rate', 0):>6.3f} │ {bsa.get('hallucination_rate', 0):>6.3f} │ "
                  f"{bl_std.get('discovery_time_s', 0):>5.1f}s │ {bl_std.get('extraction_time_s', 0):>4.1f}s")
        print(f"{'─'*120}")

    # Aggregate summary
    bl_report = baseline["report"]["methods"]
    print(f"\n{'='*120}")
    print(f" AGGREGATE SUMMARY")
    print(f"{'='*120}")
    print(f"{'Metric':<30} │ {'DSPy v2':>12} │ {'OG rlm_v2':>12} │ {'OG dspy_rlm':>12} │ {'OG std_gpt':>12} │ {'OG rlm_lb':>12}")
    print(f"{'─'*100}")

    dspy_avg = lambda k: round(np.mean([a[k] for a in dspy_valid]), 3) if dspy_valid else 0
    dspy_n_fields = round(np.mean([r["n_fields"] for r in all_results]), 1)
    dspy_disc_t = round(np.mean([r["discovery_time_s"] for r in all_results]), 1)
    dspy_ext_t = round(np.mean([r["extraction_time_s"] for r in all_results]), 1)
    dspy_quality = round(np.mean([r["quality_score"] for r in all_results]), 4)

    rows = [
        ("Avg fields/cluster", f"{dspy_n_fields}", f"{bl_report['rlm_v2']['avg_fields']}", f"{bl_report['dspy_rlm']['avg_fields']}", f"{bl_report['standard_gpt']['avg_fields']}", f"{bl_report['rlm_lb']['avg_fields']}"),
        ("Avg fill rate", f"{dspy_avg('fill_rate'):.3f}", f"{bl_report['rlm_v2']['avg_fill_rate']:.3f}", f"{bl_report['dspy_rlm']['avg_fill_rate']:.3f}", f"{bl_report['standard_gpt']['avg_fill_rate']:.3f}", f"{bl_report['rlm_lb']['avg_fill_rate']:.3f}"),
        ("Avg verbatim rate", f"{dspy_avg('verbatim_rate'):.3f}", f"{bl_report['rlm_v2']['avg_verbatim_rate']:.3f}", f"{bl_report['dspy_rlm']['avg_verbatim_rate']:.3f}", f"{bl_report['standard_gpt']['avg_verbatim_rate']:.3f}", f"{bl_report['rlm_lb']['avg_verbatim_rate']:.3f}"),
        ("Avg grounded rate", f"{dspy_avg('grounded_rate'):.3f}", f"{bl_report['rlm_v2']['avg_grounded_rate']:.3f}", f"{bl_report['dspy_rlm']['avg_grounded_rate']:.3f}", f"{bl_report['standard_gpt']['avg_grounded_rate']:.3f}", f"{bl_report['rlm_lb']['avg_grounded_rate']:.3f}"),
        ("Avg hallucination rate", f"{dspy_avg('hallucination_rate'):.3f}", f"{bl_report['rlm_v2']['avg_hallucination_rate']:.3f}", f"{bl_report['dspy_rlm']['avg_hallucination_rate']:.3f}", f"{bl_report['standard_gpt']['avg_hallucination_rate']:.3f}", f"{bl_report['rlm_lb']['avg_hallucination_rate']:.3f}"),
        ("Avg discovery time (s)", f"{dspy_disc_t}", f"{bl_report['rlm_v2']['avg_discovery_time_s']}", f"{bl_report['dspy_rlm']['avg_discovery_time_s']}", f"{bl_report['standard_gpt']['avg_discovery_time_s']}", f"{bl_report['rlm_lb']['avg_discovery_time_s']}"),
        ("Avg extraction time (s)", f"{dspy_ext_t}", f"{bl_report['rlm_v2']['avg_extraction_time_s']}", f"{bl_report['dspy_rlm']['avg_extraction_time_s']}", f"{bl_report['standard_gpt']['avg_extraction_time_s']}", f"{bl_report['rlm_lb']['avg_extraction_time_s']}"),
        ("DSPy quality score", f"{dspy_quality}", "n/a", "n/a", "n/a", "n/a"),
    ]

    for label, *vals in rows:
        print(f"{label:<30} │ {vals[0]:>12} │ {vals[1]:>12} │ {vals[2]:>12} │ {vals[3]:>12} │ {vals[4]:>12}")

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
