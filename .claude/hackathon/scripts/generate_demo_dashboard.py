#!/usr/bin/env python3
"""Generate a self-contained HTML demo dashboard from the mini top-30 run.

Reads from the DuckDB and produces a single interactive HTML file with:
  - Pipeline overview stats
  - Domain → Cluster hierarchy (sunburst)
  - Per-cluster field breakdown (bar chart)
  - Extraction confidence distribution (histogram)
  - Top extracted fields table
  - Sample extractions per cluster (searchable table)
  - Intent coverage treemap
  - Domain-level universal fields vs cluster-level delta fields comparison

Usage:
    python scripts/generate_demo_dashboard.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = str(PROJECT_ROOT / "data" / "public_contracts_perf.duckdb")
OUTPUT_HTML = PROJECT_ROOT / "data" / "demo_dashboard.html"


def connect():
    return duckdb.connect(DB_PATH, read_only=True)


def get_overview(db):
    """Pipeline-level summary stats."""
    stats = {}
    for table, col in [
        ("agreements", "agreement_id"),
        ("chunks", "chunk_id"),
        ("clusters", "cluster_id"),
        ("domains", "domain_id"),
        ("field_definitions", "field_id"),
        ("extractions", "extraction_id"),
        ("clause_intents", "intent_id"),
    ]:
        try:
            stats[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            stats[table] = 0

    try:
        stats["domain_fields"] = db.execute(
            "SELECT COUNT(*) FROM domain_field_definitions"
        ).fetchone()[0]
    except Exception:
        stats["domain_fields"] = 0

    try:
        stats["intent_extractions"] = db.execute(
            "SELECT COUNT(*) FROM intent_extractions"
        ).fetchone()[0]
    except Exception:
        stats["intent_extractions"] = 0

    return stats


def get_domain_cluster_hierarchy(db):
    """Domain → cluster hierarchy data for sunburst."""
    rows = db.execute("""
        SELECT COALESCE(d.label, 'Uncategorized') AS domain_label,
               c.label AS cluster_label,
               c.chunk_count, c.cluster_id,
               COALESCE(d.domain_id, '__uncategorized__') AS domain_id
        FROM clusters c
        LEFT JOIN domains d ON c.domain_id = d.domain_id
        WHERE c.cluster_id IN (SELECT DISTINCT cluster_id FROM field_definitions)
        ORDER BY c.chunk_count DESC
    """).fetchall()
    return rows


def get_fields_per_cluster(db):
    """Count of fields per cluster."""
    return db.execute("""
        SELECT c.label, COUNT(fd.field_id) AS n_fields, c.chunk_count
        FROM field_definitions fd
        JOIN clusters c ON fd.cluster_id = c.cluster_id
        GROUP BY c.cluster_id, c.label, c.chunk_count
        ORDER BY n_fields DESC
        LIMIT 30
    """).fetchall()


def get_extraction_confidence(db):
    """All confidence values for histogram."""
    result = db.execute(
        "SELECT confidence FROM extractions WHERE confidence IS NOT NULL"
    ).fetchnumpy()
    if result and "confidence" in result:
        return result["confidence"].astype(float)
    return np.array([])


def get_top_fields(db):
    """Top fields by extraction count."""
    return db.execute("""
        SELECT fd.name, fd.field_type, fd.description,
               COUNT(e.extraction_id) AS n_extractions,
               AVG(e.confidence) AS avg_conf
        FROM field_definitions fd
        JOIN extractions e ON e.field_id = fd.field_id
        GROUP BY fd.field_id, fd.name, fd.field_type, fd.description
        ORDER BY n_extractions DESC
        LIMIT 25
    """).fetchall()


def get_sample_extractions(db):
    """Sample extractions per cluster (up to 5 per cluster, top 30 clusters)."""
    return db.execute("""
        WITH ranked AS (
            SELECT c.label AS cluster_label, fd.name AS field_name,
                   e.value, e.confidence, a.filename,
                   ROW_NUMBER() OVER (PARTITION BY c.cluster_id ORDER BY e.confidence DESC) AS rn
            FROM extractions e
            JOIN field_definitions fd ON e.field_id = fd.field_id
            JOIN clusters c ON fd.cluster_id = c.cluster_id
            JOIN agreements a ON e.agreement_id = a.agreement_id
        )
        SELECT cluster_label, field_name, value, confidence, filename
        FROM ranked WHERE rn <= 5
        ORDER BY cluster_label, confidence DESC
    """).fetchall()


def get_intents(db):
    """Intent distribution."""
    return db.execute("""
        SELECT intent_label, COUNT(*) AS n,
               c.label AS cluster_label
        FROM clause_intents ci
        JOIN clusters c ON ci.clause_type_id = c.cluster_id
        GROUP BY intent_label, c.label
        ORDER BY n DESC
        LIMIT 60
    """).fetchall()


def get_domain_vs_cluster_fields(db):
    """Domain-level universal fields vs cluster-level delta fields."""
    domain_fields = []
    try:
        domain_fields = db.execute("""
            SELECT d.label AS domain_label, dfd.name, dfd.field_type, dfd.description
            FROM domain_field_definitions dfd
            JOIN domains d ON dfd.domain_id = d.domain_id
            ORDER BY d.label, dfd.name
        """).fetchall()
    except Exception:
        pass

    cluster_fields = db.execute("""
        SELECT c.label AS cluster_label, fd.name, fd.field_type, fd.source
        FROM field_definitions fd
        JOIN clusters c ON fd.cluster_id = c.cluster_id
        ORDER BY c.label, fd.name
    """).fetchall()

    return domain_fields, cluster_fields


def get_clause_types(db):
    """Clause type summary: cluster label = clause type, with field / intent counts."""
    return db.execute("""
        SELECT c.label AS clause_type,
               COALESCE(d.label, 'Uncategorized') AS domain_label,
               c.chunk_count,
               COALESCE(f_cnt.n_fields, 0) AS n_fields,
               COALESCE(i_cnt.n_intents, 0) AS n_intents,
               COALESCE(e_cnt.n_extractions, 0) AS n_extractions
        FROM clusters c
        LEFT JOIN domains d ON c.domain_id = d.domain_id
        LEFT JOIN (
            SELECT cluster_id, COUNT(*) AS n_fields FROM field_definitions GROUP BY cluster_id
        ) f_cnt ON f_cnt.cluster_id = c.cluster_id
        LEFT JOIN (
            SELECT clause_type_id, COUNT(*) AS n_intents FROM clause_intents GROUP BY clause_type_id
        ) i_cnt ON i_cnt.clause_type_id = c.cluster_id
        LEFT JOIN (
            SELECT fd.cluster_id, COUNT(e.extraction_id) AS n_extractions
            FROM extractions e JOIN field_definitions fd ON e.field_id = fd.field_id
            GROUP BY fd.cluster_id
        ) e_cnt ON e_cnt.cluster_id = c.cluster_id
        WHERE COALESCE(f_cnt.n_fields, 0) > 0
        ORDER BY n_extractions DESC
    """).fetchall()


def get_intent_extractions(db):
    """Intent extractions with values."""
    try:
        return db.execute("""
            SELECT ci.intent_label,
                   c.label AS cluster_label,
                   ie.value,
                   ie.confidence,
                   a.filename
            FROM intent_extractions ie
            JOIN clause_intents ci ON ie.intent_id = ci.intent_id
            JOIN clusters c ON ci.clause_type_id = c.cluster_id
            LEFT JOIN agreements a ON ie.agreement_id = a.agreement_id
            ORDER BY ie.confidence DESC
            LIMIT 150
        """).fetchall()
    except Exception:
        return []


def build_html(db):
    overview = get_overview(db)
    hierarchy = get_domain_cluster_hierarchy(db)
    fields_per_cluster = get_fields_per_cluster(db)
    conf_data = get_extraction_confidence(db)
    top_fields = get_top_fields(db)
    samples = get_sample_extractions(db)
    intents = get_intents(db)
    domain_fields, cluster_fields = get_domain_vs_cluster_fields(db)
    clause_types = get_clause_types(db)
    intent_extractions = get_intent_extractions(db)

    # ── 1. Overview cards ──
    cards_html = f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">{overview['agreements']:,}</div><div class="stat-label">Agreements</div></div>
        <div class="stat-card"><div class="stat-value">{overview['chunks']:,}</div><div class="stat-label">Chunks</div></div>
        <div class="stat-card"><div class="stat-value">{overview['clusters']:,}</div><div class="stat-label">Total Clusters</div></div>
        <div class="stat-card"><div class="stat-value">{overview['domains']:,}</div><div class="stat-label">Macro Domains</div></div>
        <div class="stat-card highlight"><div class="stat-value">{overview['domain_fields']:,}</div><div class="stat-label">Domain Fields (Tier 1)</div></div>
        <div class="stat-card highlight"><div class="stat-value">{overview['field_definitions']:,}</div><div class="stat-label">Cluster Fields (Tier 2)</div></div>
        <div class="stat-card accent"><div class="stat-value">{overview['extractions']:,}</div><div class="stat-label">Extractions</div></div>
        <div class="stat-card accent"><div class="stat-value">{overview['clause_intents']:,}</div><div class="stat-label">Intents</div></div>
        <div class="stat-card accent"><div class="stat-value">{overview['intent_extractions']:,}</div><div class="stat-label">Intent Extractions</div></div>
    </div>
    """

    # ── 2. Sunburst: Domain → Cluster hierarchy ──
    if hierarchy:
        parents, labels, values, ids, colors = [], [], [], [], []
        domain_set = set()
        # Colour palette for domains
        palette = ['#4f46e5','#10b981','#f59e0b','#ef4444','#8b5cf6',
                   '#06b6d4','#ec4899','#14b8a6','#f97316','#6366f1',
                   '#84cc16','#a855f7','#22d3ee','#fb923c','#e879f9']
        domain_color = {}
        ci = 0
        for domain_label, cluster_label, chunk_count, cid, did in hierarchy:
            if did not in domain_set:
                parents.append("")
                labels.append(domain_label)
                values.append(0)
                ids.append(did)
                col = palette[ci % len(palette)]
                colors.append(col)
                domain_color[did] = col
                domain_set.add(did)
                ci += 1
            parents.append(did)
            labels.append(cluster_label[:40])
            values.append(chunk_count)
            ids.append(cid)
            colors.append(domain_color.get(did, '#475569'))

        fig_sunburst = go.Figure(go.Sunburst(
            ids=ids, labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=colors),
            hovertemplate="<b>%{label}</b><br>Chunks: %{value}<extra></extra>",
            maxdepth=2,
        ))
        fig_sunburst.update_layout(
            margin=dict(t=30, l=0, r=0, b=0), height=550,
            font=dict(size=11),
        )
        sunburst_html = fig_sunburst.to_html(full_html=False, include_plotlyjs=False)
    else:
        sunburst_html = "<p>No hierarchy data</p>"

    # ── 3. Bar chart: fields per cluster ──
    if fields_per_cluster:
        labels_fc = [f"{r[0][:35]} ({r[2]} chunks)" for r in fields_per_cluster]
        values_fc = [r[1] for r in fields_per_cluster]
        fig_fields = go.Figure(go.Bar(
            x=values_fc, y=labels_fc, orientation="h",
            marker_color="#4f46e5",
            hovertemplate="%{y}: %{x} fields<extra></extra>",
        ))
        fig_fields.update_layout(
            margin=dict(t=20, l=250, r=20, b=40), height=max(400, len(labels_fc) * 28),
            yaxis=dict(autorange="reversed"), xaxis_title="Number of Fields",
            font=dict(size=11),
        )
        fields_bar_html = fig_fields.to_html(full_html=False, include_plotlyjs=False)
    else:
        fields_bar_html = "<p>No field data</p>"

    # ── 4. Confidence histogram ──
    if len(conf_data) > 0:
        fig_hist = go.Figure(go.Histogram(
            x=conf_data, nbinsx=30,
            marker_color="#10b981",
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        mean_c = float(np.mean(conf_data))
        median_c = float(np.median(conf_data))
        fig_hist.add_vline(x=mean_c, line_dash="dash", line_color="red",
                           annotation_text=f"Mean: {mean_c:.2f}")
        fig_hist.add_vline(x=median_c, line_dash="dot", line_color="blue",
                           annotation_text=f"Median: {median_c:.2f}")
        fig_hist.update_layout(
            margin=dict(t=40, l=40, r=20, b=40), height=350,
            xaxis_title="Confidence", yaxis_title="Count",
            font=dict(size=11),
        )
        hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False)
        conf_summary = (f"<b>{len(conf_data):,}</b> extractions — "
                        f"mean: <b>{mean_c:.3f}</b>, median: <b>{median_c:.3f}</b>, "
                        f"≥0.7: <b>{int((conf_data >= 0.7).sum()):,}</b> "
                        f"({(conf_data >= 0.7).sum() / len(conf_data) * 100:.0f}%)")
    else:
        hist_html = "<p>No extraction data</p>"
        conf_summary = ""

    # ── 5. Top fields table ──
    if top_fields:
        tf_rows = "".join(
            f"<tr><td>{name}</td><td><code>{ftype}</code></td>"
            f"<td class='desc'>{desc[:80] if desc else ''}</td>"
            f"<td class='num'>{n_ext:,}</td>"
            f"<td class='num'>{avg_c:.2f}</td></tr>"
            for name, ftype, desc, n_ext, avg_c in top_fields
        )
        top_fields_table = f"""
        <table class="data-table">
            <thead><tr><th>Field Name</th><th>Type</th><th>Description</th>
                       <th>Extractions</th><th>Avg Conf</th></tr></thead>
            <tbody>{tf_rows}</tbody>
        </table>"""
    else:
        top_fields_table = "<p>No fields</p>"

    # ── 6. Sample extractions table ──
    if samples:
        se_rows = "".join(
            f"<tr><td>{cl[:30]}</td><td>{fn}</td>"
            f"<td class='val'>{(v[:120] + '…') if v and len(v) > 120 else (v or '')}</td>"
            f"<td class='num'>{c:.2f}</td>"
            f"<td class='file'>{Path(f).stem[:25] if f else ''}</td></tr>"
            for cl, fn, v, c, f in samples[:150]
        )
        samples_table = f"""
        <div class="table-scroll">
        <table class="data-table">
            <thead><tr><th>Cluster</th><th>Field</th><th>Extracted Value</th>
                       <th>Conf</th><th>Source</th></tr></thead>
            <tbody>{se_rows}</tbody>
        </table></div>"""
    else:
        samples_table = "<p>No extractions</p>"

    # ── 7. Intent visualization ──
    if intents:
        # Build treemap with proper root → cluster → intent hierarchy
        tm_ids, tm_labels, tm_parents, tm_values, tm_colors = [], [], [], [], []
        # Root node
        tm_ids.append("__root__")
        tm_labels.append("All Intents")
        tm_parents.append("")
        tm_values.append(0)
        tm_colors.append("#1e293b")
        # Cluster parent nodes
        cluster_set = set()
        palette = ['#4f46e5','#10b981','#f59e0b','#ef4444','#8b5cf6',
                   '#06b6d4','#ec4899','#14b8a6','#f97316','#6366f1']
        cluster_color = {}
        ci = 0
        for _, n, cluster_label in intents:
            clbl = cluster_label[:35]
            if clbl not in cluster_set:
                tm_ids.append(f"cl_{clbl}")
                tm_labels.append(clbl)
                tm_parents.append("__root__")
                tm_values.append(0)
                col = palette[ci % len(palette)]
                tm_colors.append(col)
                cluster_color[clbl] = col
                cluster_set.add(clbl)
                ci += 1
        # Intent leaf nodes
        seen_intents = set()
        for intent_label, n, cluster_label in intents:
            clbl = cluster_label[:35]
            # Deduplicate intent labels within same cluster
            key = f"{clbl}_{intent_label}"
            if key in seen_intents:
                continue
            seen_intents.add(key)
            tm_ids.append(key)
            tm_labels.append(intent_label.replace('_', ' '))
            tm_parents.append(f"cl_{clbl}")
            tm_values.append(n)
            tm_colors.append(cluster_color.get(clbl, '#475569'))

        fig_intent = go.Figure(go.Treemap(
            ids=tm_ids, labels=tm_labels, parents=tm_parents,
            values=tm_values,
            branchvalues="total",
            marker=dict(colors=tm_colors),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
            maxdepth=2,
        ))
        fig_intent.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=500)
        intent_html = fig_intent.to_html(full_html=False, include_plotlyjs=False)

        # Also build an intent summary table
        intent_table_rows = "".join(
            f"<tr><td>{cl[:30]}</td><td>{il.replace('_',' ')}</td><td class='num'>{n}</td></tr>"
            for il, n, cl in intents[:40]
        )
        intent_html += f"""
        <h3 style="margin-top:16px;color:var(--muted);font-size:14px;">Intent Details</h3>
        <div class="table-scroll">
        <table class="data-table">
            <thead><tr><th>Cluster</th><th>Intent</th><th>Count</th></tr></thead>
            <tbody>{intent_table_rows}</tbody>
        </table></div>"""
    else:
        intent_html = "<p>No intent data</p>"

    # ── 8. Domain fields vs Cluster fields comparison table ──
    if domain_fields:
        # Group by domain for a nicer display
        from collections import defaultdict
        by_domain = defaultdict(list)
        for dl, name, ft, desc in domain_fields:
            by_domain[dl].append((name, ft, desc))

        df_sections = []
        for domain_name, fields in by_domain.items():
            df_sections.append(
                f"<tr><td colspan='4' style='background:var(--accent);color:white;"
                f"font-weight:600;padding:8px 10px;'>"
                f"{domain_name} — {len(fields)} universal fields</td></tr>"
            )
            for name, ft, desc in fields:
                safe_desc = (desc[:100] if desc else '').replace('<', '&lt;').replace('>', '&gt;')
                df_sections.append(
                    f"<tr><td></td><td>{name}</td><td><code>{ft}</code></td>"
                    f"<td class='desc'>{safe_desc}</td></tr>"
                )
        domain_fields_table = f"""
        <p style="color:var(--muted);font-size:13px;margin-bottom:8px;">
          {len(domain_fields)} universal fields across {len(by_domain)} domain(s)
          — these are seeded into every cluster-level RLM session within that domain.
        </p>
        <table class="data-table">
            <thead><tr><th>Domain</th><th>Universal Field</th><th>Type</th>
                       <th>Description</th></tr></thead>
            <tbody>{''.join(df_sections)}</tbody>
        </table>"""
    else:
        domain_fields_table = "<p>No domain-level fields discovered (Stage 3.5 may not have run)</p>"

    # ── 9. Clause Types table ──
    if clause_types:
        ct_rows = "".join(
            f"<tr><td>{ct[:45]}</td><td>{dl[:30]}</td>"
            f"<td class='num'>{cc:,}</td>"
            f"<td class='num'>{nf}</td>"
            f"<td class='num'>{ni}</td>"
            f"<td class='num'>{ne:,}</td></tr>"
            for ct, dl, cc, nf, ni, ne in clause_types
        )
        clause_types_html = f"""
        <p style="color:var(--muted);font-size:13px;margin-bottom:8px;">
          {len(clause_types)} clause types discovered — each cluster represents a distinct clause type.
        </p>
        <div class="table-scroll">
        <table class="data-table">
            <thead><tr><th>Clause Type</th><th>Domain</th><th>Chunks</th>
                       <th>Fields</th><th>Intents</th><th>Extractions</th></tr></thead>
            <tbody>{ct_rows}</tbody>
        </table></div>"""
    else:
        clause_types_html = "<p>No clause types data</p>"

    # ── 10. Intent Extractions table ──
    if intent_extractions:
        ie_rows = "".join(
            f"<tr><td>{il.replace('_', ' ')}</td><td>{cl[:30]}</td>"
            f"<td class='val'>{(v[:150] + '…') if v and len(v) > 150 else (v or '')}</td>"
            f"<td class='num'>{c:.2f}</td>"
            f"<td class='file'>{Path(f).stem[:25] if f else ''}</td></tr>"
            for il, cl, v, c, f in intent_extractions
        )
        intent_extractions_html = f"""
        <p style="color:var(--muted);font-size:13px;margin-bottom:8px;">
          {len(intent_extractions)} intent extraction samples — concrete clause text expressing each intent.
        </p>
        <div class="table-scroll">
        <table class="data-table">
            <thead><tr><th>Intent</th><th>Cluster</th><th>Extracted Value</th>
                       <th>Conf</th><th>Source</th></tr></thead>
            <tbody>{ie_rows}</tbody>
        </table></div>"""
    else:
        intent_extractions_html = "<p>No intent extractions yet (rerun pipeline with intent RAG extraction)</p>"

    # ── Assemble HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EVoC Pipeline — Demo Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --bg: #0f172a; --card: #1e293b; --card2: #334155; --accent: #4f46e5;
         --green: #10b981; --text: #e2e8f0; --muted: #94a3b8; --border: #475569; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, sans-serif;
        padding: 24px; max-width: 1400px; margin: auto; }}
h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 32px 0 12px; color: var(--muted);
      letter-spacing: 1px; text-transform: uppercase; border-bottom: 1px solid var(--border);
      padding-bottom: 6px; }}
.subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 24px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
               gap: 12px; margin-bottom: 24px; }}
.stat-card {{ background: var(--card); border-radius: 10px; padding: 16px; text-align: center;
              border: 1px solid var(--border); }}
.stat-card.highlight {{ border-color: var(--accent); background: rgba(79, 70, 229, 0.1); }}
.stat-card.accent {{ border-color: var(--green); background: rgba(16, 185, 129, 0.08); }}
.stat-value {{ font-size: 28px; font-weight: 700; }}
.stat-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.chart-card {{ background: var(--card); border-radius: 10px; padding: 16px;
               margin-bottom: 16px; border: 1px solid var(--border); }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.data-table th {{ background: var(--card2); padding: 8px 10px; text-align: left;
                  font-weight: 600; position: sticky; top: 0; }}
.data-table td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
.data-table tr:hover {{ background: rgba(255,255,255,0.03); }}
.data-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.data-table .val {{ max-width: 400px; word-break: break-word; font-size: 12px; }}
.data-table .desc {{ max-width: 300px; font-size: 12px; color: var(--muted); }}
.data-table .file {{ font-size: 11px; color: var(--muted); }}
.table-scroll {{ max-height: 500px; overflow-y: auto; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.badge {{ display: inline-block; background: var(--accent); color: white; padding: 2px 8px;
          border-radius: 4px; font-size: 11px; font-weight: 600; }}
.conf-summary {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
code {{ background: var(--card2); padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>

<h1>EVoC Clustering Pipeline <span class="badge">Two-Tier Fields</span></h1>
<p class="subtitle">Mini Top-30 Run — {datetime.now().strftime('%B %d, %Y %H:%M')} —
   {overview['agreements']:,} agreements → {overview['extractions']:,} extractions</p>

{cards_html}

<div class="two-col">
    <div>
        <h2>Domain → Cluster Hierarchy</h2>
        <div class="chart-card">{sunburst_html}</div>
    </div>
    <div>
        <h2>Fields per Cluster</h2>
        <div class="chart-card">{fields_bar_html}</div>
    </div>
</div>

<h2>Extraction Confidence Distribution</h2>
<p class="conf-summary">{conf_summary}</p>
<div class="chart-card">{hist_html}</div>

<h2>Domain-Level Universal Fields <span class="badge">Tier 1 — Stage 3.5</span></h2>
<div class="chart-card">{domain_fields_table}</div>

<h2>Top Extracted Fields <span class="badge">Tier 2 — Cluster Level</span></h2>
<div class="chart-card">{top_fields_table}</div>

<h2>Sample Extracted Values</h2>
<div class="chart-card">{samples_table}</div>

<h2>Clause Types <span class="badge">Cluster = Clause Type</span></h2>
<div class="chart-card">{clause_types_html}</div>

<h2>Intent Coverage</h2>
<div class="chart-card">{intent_html}</div>

<h2>Intent Extracted Values <span class="badge">RAG</span></h2>
<div class="chart-card">{intent_extractions_html}</div>

</body>
</html>"""
    return html


def main():
    print(f"Connecting to {DB_PATH}...")
    db = connect()
    print("Building dashboard...")
    html = build_html(db)
    db.close()

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html)
    print(f"Dashboard → {OUTPUT_HTML}")
    print(f"Size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
