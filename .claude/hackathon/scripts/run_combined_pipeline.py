#!/usr/bin/env python3
"""
Combined Pipeline: DSPy Discovery + Agentic Extraction
=======================================================

Chains two systems:
  1. DSPy v2 field discovery (with trial extraction filter)
  2. Agentic extraction backend (retrieval, carryforward, retry, judge, reviewer)

Runs on the same 5 clusters as the baseline comparison and produces
a side-by-side report showing the combined pipeline vs all baselines.
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
sys.path.insert(0, str(Path(__file__).parent))

from core.rlm_v2_dspy import (
    ClusterDiscoveryPipeline,
    _configure_dspy_lm,
    discovery_quality_metric,
)
from agentic_extractions import (
    AgenticExtractionPipeline,
    AgenticConfig,
    _ground_value,
)
import dspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def grounding_detail(value, chunk_text):
    """Compute grounding detail for comparison compatibility."""
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
        return {"level": "high_overlap", "score": 0.8}
    return {"level": "hallucinated", "score": 0.0}


def main():
    db_path = "data/cuad_510_demo.duckdb"
    n_extract_chunks = 8

    # Load baseline for comparison
    baseline_path = Path("data/extraction_value_comparison_v2.json")
    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_clusters = {r["cluster_id"]: r for r in baseline["per_cluster"]}
    cluster_ids = [r["cluster_id"] for r in baseline["per_cluster"]]

    logger.info(f"Running combined pipeline on {len(cluster_ids)} clusters")

    # ── Setup ──
    lm = _configure_dspy_lm()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Discovery pipeline (DSPy v2 with trial extraction)
    from core.field_discovery import FieldDiscoveryConfig
    discovery_config = FieldDiscoveryConfig()
    disc_pipeline = ClusterDiscoveryPipeline(config=discovery_config, embedder=embedder)

    # Agentic extraction pipeline
    from agentic_extractions import _create_client
    agentic_config = AgenticConfig(
        top_k_chunks=n_extract_chunks,
        max_retries=2,
        enable_judge=True,
        enable_reviewer=True,
    )
    agentic_pipeline = AgenticExtractionPipeline(
        client=_create_client(),
        config=agentic_config,
        embedder=embedder,
    )

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

        # ═══ Phase 1: DSPy Discovery ═══
        t0 = time.time()
        try:
            disc_result = disc_pipeline(
                cluster_label=cluster_label,
                keywords=", ".join(keywords[:10]),
                chunk_texts=chunks,
            )
            fields = disc_result.fields
            disc_meta = disc_result.meta
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            fields = []
            disc_meta = {"error": str(e)}
        disc_time = round(time.time() - t0, 2)
        logger.info(f"  Discovery: {len(fields)} fields in {disc_time}s")

        # Build RLM context from discovery metadata
        rlm_context = None
        if disc_meta.get("clause_types"):
            rlm_context = {
                "clause_types": disc_meta.get("clause_types", []),
                "intents": disc_meta.get("intents", []),
                "obligations": disc_meta.get("obligations", []),
            }

        # ═══ Phase 2: Agentic Extraction ═══
        t1 = time.time()
        try:
            ext_result = agentic_pipeline.extract(
                fields=fields,
                chunk_texts=extract_chunks,
                cluster_label=cluster_label,
                rlm_context=rlm_context,
            )
            ext_meta = ext_result.meta
        except Exception as e:
            logger.error(f"Agentic extraction failed: {e}")
            ext_result = None
            ext_meta = {"error": str(e)}
        ext_time = round(time.time() - t1, 2)

        # ═══ Compute comparison-compatible metrics ═══
        # Build chunk_extractions in the same format as the baseline comparison
        chunk_extractions = []
        for chunk_idx, chunk_text in enumerate(extract_chunks):
            chunk_detail = {"chunk_index": chunk_idx, "fields": {}}
            for f in fields:
                fname = f["name"]
                cv = ext_result.compiled.get(fname) if ext_result else None
                if cv and cv.value is not None and chunk_idx in cv.source_chunk_indices:
                    grounding = grounding_detail(cv.value, chunk_text)
                    chunk_detail["fields"][fname] = {
                        "value": cv.value,
                        "confidence": cv.confidence,
                        "grounding": grounding,
                        "value_type": "short_text",
                        "reviewer_verdict": cv.reviewer_verdict,
                    }
                else:
                    chunk_detail["fields"][fname] = {
                        "value": None,
                        "confidence": 0.0,
                        "grounding": {"level": "missing", "score": 0.0},
                        "value_type": "null",
                    }
            chunk_extractions.append(chunk_detail)

        # Aggregate metrics (using compiled values for grounding)
        n_fields = len(fields)
        compiled_values = []
        if ext_result:
            for fname, cv in ext_result.compiled.items():
                if cv.value is not None:
                    compiled_values.append(cv)

        # For fill rate: check each (field, chunk) like the baseline does
        total_slots = n_fields * len(extract_chunks)
        filled_slots = 0
        verbatim_count = 0
        halluc_count = 0
        grounded_count = 0

        for ce in chunk_extractions:
            for f in fields:
                fd = ce["fields"].get(f["name"], {})
                if fd.get("value") is not None:
                    filled_slots += 1
                    level = fd.get("grounding", {}).get("level", "missing")
                    if level == "verbatim":
                        verbatim_count += 1
                        grounded_count += 1
                    elif level == "high_overlap":
                        grounded_count += 1
                    elif level == "hallucinated":
                        halluc_count += 1

        # Also compute from compiled values (document-level, better for this pipeline)
        n_compiled = len(compiled_values)
        n_compiled_verified = sum(1 for cv in compiled_values if cv.grounding_level in ("verified", "verbatim"))
        n_compiled_partial = sum(1 for cv in compiled_values if cv.grounding_level == "partial")
        n_compiled_halluc = sum(1 for cv in compiled_values if cv.grounding_level == "hallucinated")

        agg = {
            "fill_rate": round(filled_slots / total_slots, 3) if total_slots else 0,
            "verbatim_rate": round(verbatim_count / filled_slots, 3) if filled_slots else 0,
            "grounded_rate": round(grounded_count / filled_slots, 3) if filled_slots else 0,
            "hallucination_rate": round(halluc_count / filled_slots, 3) if filled_slots else 0,
            "n_extractions": filled_slots,
            "n_total_slots": total_slots,
            # Document-level compiled metrics (the real story)
            "compiled_count": n_compiled,
            "compiled_verified_rate": round(n_compiled_verified / n_compiled, 3) if n_compiled else 0,
            "compiled_grounded_rate": round((n_compiled_verified + n_compiled_partial) / n_compiled, 3) if n_compiled else 0,
            "compiled_halluc_rate": round(n_compiled_halluc / n_compiled, 3) if n_compiled else 0,
        }

        # Quality score
        example = dspy.Example(
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            chunk_texts=chunks,
        ).with_inputs("cluster_label", "keywords", "chunk_texts")
        prediction = dspy.Prediction(fields=fields, meta=disc_meta)
        quality_score = discovery_quality_metric(example, prediction)

        cluster_result = {
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "chunk_count": cluster["chunk_count"],
            "n_fields": n_fields,
            "discovery_time_s": disc_time,
            "extraction_time_s": ext_time,
            "aggregate": agg,
            "disc_meta": disc_meta,
            "ext_meta": ext_meta,
            "quality_score": quality_score,
            "extractions": chunk_extractions,
            "fields": fields,
        }
        all_results.append(cluster_result)

        logger.info(
            f"  fill={agg['fill_rate']:.3f}  verb={agg['verbatim_rate']:.3f}  "
            f"halluc={agg['hallucination_rate']:.3f}  compiled_ground={agg['compiled_grounded_rate']:.3f}  "
            f"retries={ext_meta.get('n_retries', 0)}  judge={ext_meta.get('n_judge_calls', 0)}  "
            f"reviewer={ext_meta.get('n_reviewer_calls', 0)}  "
            f"disc={disc_time}s  ext={ext_time}s"
        )

    conn.close()

    # ── Save ──
    out_path = Path("data/combined_pipeline_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"combined_results": all_results}, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # ── Print comparison table ──
    print(f"\n{'='*130}")
    print(f" Combined Pipeline (Discovery + Agentic) vs Baselines ({len(all_results)} clusters)")
    print(f"{'='*130}")

    hdr = (f"{'Cluster':<25} │ {'Method':<16} │ {'Fields':>6} │ {'Fill':>6} │ "
           f"{'Verb':>6} │ {'Ground':>6} │ {'Halluc':>6} │ {'DiscT':>6} │ {'ExtT':>5}")
    print(f"\n{hdr}")
    print(f"{'─'*130}")

    for r in all_results:
        cid = r["cluster_id"]
        bl = baseline_clusters.get(cid, {})
        label = r["cluster_label"][:23]
        a = r["aggregate"]

        # Combined pipeline row
        print(f"{label:<25} │ {'Combined':<16} │ {r['n_fields']:>6} │ {a['fill_rate']:>6.3f} │ "
              f"{a['verbatim_rate']:>6.3f} │ {a['grounded_rate']:>6.3f} │ {a['hallucination_rate']:>6.3f} │ "
              f"{r['discovery_time_s']:>5.1f}s │ {r['extraction_time_s']:>4.1f}s")

        # DSPy v2-only row (from last benchmark)
        dspy_path = Path("data/dspy_v2_comparison.json")
        if dspy_path.exists():
            dspy_data = json.load(open(dspy_path))
            for dr in dspy_data.get("dspy_results", []):
                if dr["cluster_id"] == cid:
                    da = dr["aggregate"]
                    print(f"{'':25} │ {'DSPy v2 only':<16} │ {dr['n_fields']:>6} │ {da['fill_rate']:>6.3f} │ "
                          f"{da['verbatim_rate']:>6.3f} │ {da['grounded_rate']:>6.3f} │ {da['hallucination_rate']:>6.3f} │ "
                          f"{dr['discovery_time_s']:>5.1f}s │ {dr['extraction_time_s']:>4.1f}s")
                    break

        # OG baselines
        for method_name, display_name in [("rlm_v2", "OG rlm_v2"), ("standard_gpt", "OG std_gpt"), ("dspy_rlm", "OG dspy_rlm")]:
            bm = bl.get("methods", {}).get(method_name, {})
            ba = bm.get("aggregate", {})
            if ba:
                print(f"{'':25} │ {display_name:<16} │ {bm.get('n_fields', 0):>6} │ {ba.get('fill_rate', 0):>6.3f} │ "
                      f"{ba.get('verbatim_rate', 0):>6.3f} │ {ba.get('grounded_rate', 0):>6.3f} │ {ba.get('hallucination_rate', 0):>6.3f} │ "
                      f"{bm.get('discovery_time_s', 0):>5.1f}s │ {bm.get('extraction_time_s', 0):>4.1f}s")

        print(f"{'─'*130}")

    # ── Aggregate summary ──
    bl_report = baseline["report"]["methods"]

    print(f"\n{'='*130}")
    print(f" AGGREGATE SUMMARY")
    print(f"{'='*130}")

    header = (f"{'Metric':<32} │ {'Combined':>12} │ {'DSPy v2':>12} │ "
              f"{'OG rlm_v2':>12} │ {'OG dspy_rlm':>12} │ {'OG std_gpt':>12}")
    print(header)
    print(f"{'─'*110}")

    valid = [r["aggregate"] for r in all_results if r["aggregate"]["n_extractions"] > 0]
    avg = lambda k: round(np.mean([a[k] for a in valid]), 3) if valid else 0
    n_fields = round(np.mean([r["n_fields"] for r in all_results]), 1)
    disc_t = round(np.mean([r["discovery_time_s"] for r in all_results]), 1)
    ext_t = round(np.mean([r["extraction_time_s"] for r in all_results]), 1)

    # Load DSPy v2-only aggregates
    dspy_v2_str = "n/a"
    dspy_path = Path("data/dspy_v2_comparison.json")
    if dspy_path.exists():
        dspy_data = json.load(open(dspy_path))
        dr_valid = [r["aggregate"] for r in dspy_data["dspy_results"] if r["aggregate"]["n_extractions"] > 0]
        d_avg = lambda k: round(np.mean([a[k] for a in dr_valid]), 3) if dr_valid else 0

    rows = [
        ("Avg fields/cluster",
         f"{n_fields}",
         f"{round(np.mean([r['n_fields'] for r in dspy_data['dspy_results']]), 1)}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_fields']}",
         f"{bl_report['dspy_rlm']['avg_fields']}",
         f"{bl_report['standard_gpt']['avg_fields']}"),
        ("Avg fill rate",
         f"{avg('fill_rate'):.3f}",
         f"{d_avg('fill_rate'):.3f}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_fill_rate']:.3f}",
         f"{bl_report['dspy_rlm']['avg_fill_rate']:.3f}",
         f"{bl_report['standard_gpt']['avg_fill_rate']:.3f}"),
        ("Avg verbatim rate",
         f"{avg('verbatim_rate'):.3f}",
         f"{d_avg('verbatim_rate'):.3f}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_verbatim_rate']:.3f}",
         f"{bl_report['dspy_rlm']['avg_verbatim_rate']:.3f}",
         f"{bl_report['standard_gpt']['avg_verbatim_rate']:.3f}"),
        ("Avg grounded rate",
         f"{avg('grounded_rate'):.3f}",
         f"{d_avg('grounded_rate'):.3f}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_grounded_rate']:.3f}",
         f"{bl_report['dspy_rlm']['avg_grounded_rate']:.3f}",
         f"{bl_report['standard_gpt']['avg_grounded_rate']:.3f}"),
        ("Avg hallucination rate",
         f"{avg('hallucination_rate'):.3f}",
         f"{d_avg('hallucination_rate'):.3f}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_hallucination_rate']:.3f}",
         f"{bl_report['dspy_rlm']['avg_hallucination_rate']:.3f}",
         f"{bl_report['standard_gpt']['avg_hallucination_rate']:.3f}"),
        ("Avg discovery time (s)",
         f"{disc_t}",
         f"{round(np.mean([r['discovery_time_s'] for r in dspy_data['dspy_results']]), 1)}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_discovery_time_s']}",
         f"{bl_report['dspy_rlm']['avg_discovery_time_s']}",
         f"{bl_report['standard_gpt']['avg_discovery_time_s']}"),
        ("Avg extraction time (s)",
         f"{ext_t}",
         f"{round(np.mean([r['extraction_time_s'] for r in dspy_data['dspy_results']]), 1)}" if dspy_path.exists() else "n/a",
         f"{bl_report['rlm_v2']['avg_extraction_time_s']}",
         f"{bl_report['dspy_rlm']['avg_extraction_time_s']}",
         f"{bl_report['standard_gpt']['avg_extraction_time_s']}"),
    ]

    # Agentic-specific rows
    all_ext_meta = [r.get("ext_meta", {}) for r in all_results]
    avg_retries = round(np.mean([m.get("n_retries", 0) for m in all_ext_meta]), 1)
    avg_judge = round(np.mean([m.get("n_judge_calls", 0) for m in all_ext_meta]), 1)
    avg_reviewer = round(np.mean([m.get("n_reviewer_calls", 0) for m in all_ext_meta]), 1)
    avg_conflicts = round(np.mean([m.get("n_conflicts_resolved", 0) for m in all_ext_meta]), 1)
    avg_compiled_ground = round(np.mean([a.get("compiled_grounded_rate", 0) for a in valid]), 3) if valid else 0

    rows.extend([
        ("Compiled grounded rate",
         f"{avg_compiled_ground:.3f}", "n/a", "n/a", "n/a", "n/a"),
        ("Avg retries/cluster",
         f"{avg_retries}", "n/a", "n/a", "n/a", "n/a"),
        ("Avg judge calls/cluster",
         f"{avg_judge}", "n/a", "n/a", "n/a", "n/a"),
        ("Avg reviewer calls/cluster",
         f"{avg_reviewer}", "n/a", "n/a", "n/a", "n/a"),
        ("Avg conflicts resolved",
         f"{avg_conflicts}", "n/a", "n/a", "n/a", "n/a"),
    ])

    for label, *vals in rows:
        print(f"{label:<32} │ {vals[0]:>12} │ {vals[1]:>12} │ {vals[2]:>12} │ {vals[3]:>12} │ {vals[4]:>12}")

    print(f"{'='*130}")


if __name__ == "__main__":
    main()
