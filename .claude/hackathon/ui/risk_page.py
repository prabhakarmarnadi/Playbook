"""Risk & Portfolio Analysis page.

Risk heatmap, per-agreement risk profiles, portfolio-level analytics.
Wires up core/risk_scorer.py which was previously CLI-only.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.store import ClusteringStore

RISK_COLORS = {
    "liability": "#f85149", "ip": "#bc8cff", "termination": "#d29922",
    "compliance": "#58a6ff", "financial": "#3fb950",
    "confidentiality": "#f778ba", "operational": "#79c0ff",
}
RISK_ICONS = {
    "liability": "⚠", "ip": "🔒", "termination": "⏹",
    "compliance": "📋", "financial": "💰",
    "confidentiality": "🔐", "operational": "⚙",
}
PROFILE_COLORS = {"high-risk": "#f85149", "moderate": "#d29922", "low-risk": "#3fb950"}


def _get_risk_scorer(db_path: str, with_llm: bool = False):
    from core.risk_scorer import RiskScorer
    llm = None
    if with_llm:
        from core.llm_client import LLMClient
        llm = LLMClient()
    return RiskScorer(db_path, llm=llm)


def render(store: ClusteringStore):
    st.header("🛡️ Risk & Portfolio Analysis")
    st.caption("Risk classification across 7 categories × 4 severity levels. Portfolio-level heatmaps.")

    scorer = _get_risk_scorer(store.db_path)

    # Check if risk scores exist
    scored = scorer.db.execute("SELECT COUNT(*) FROM cluster_risks").fetchone()[0]

    if scored == 0:
        st.warning("Risk scores have not been computed yet.")
        col_a, col_b = st.columns(2)
        if col_a.button("⚡ Quick Score (heuristic — no LLM)", key="run_risk_fast"):
            with st.spinner("Scoring all clause types (heuristic)..."):
                scorer_fast = _get_risk_scorer(store.db_path, with_llm=False)
                cluster_results = scorer_fast.score_all_clusters(use_llm=False)
                agr_results = scorer_fast.score_all_agreements()
            st.success(f"Scored {len(cluster_results)} clause types and {len(agr_results)} agreements.")
            st.rerun()
        if col_b.button("🧠 Deep Score (LLM — with reasoning)", key="run_risk_llm"):
            with st.spinner("Scoring with LLM reasoning (this takes ~2 min)..."):
                scorer_llm = _get_risk_scorer(store.db_path, with_llm=True)
                cluster_results = scorer_llm.score_all_clusters(use_llm=True)
                agr_results = scorer_llm.score_all_agreements()
            st.success(f"Scored {len(cluster_results)} clause types with LLM reasoning.")
            st.rerun()
        return

    # Check if existing scores have real rationale or just heuristic
    sample_cats = scorer.db.execute(
        "SELECT categories FROM cluster_risks LIMIT 1"
    ).fetchone()
    has_llm_rationale = False
    if sample_cats:
        cats = json.loads(sample_cats[0]) if isinstance(sample_cats[0], str) else sample_cats[0]
        has_llm_rationale = cats and not cats[0].get("rationale", "").startswith("Matched ")

    if not has_llm_rationale:
        st.info("💡 Current scores use heuristic keywords — reasoning is generic. "
                "Click below to re-score with LLM for detailed rationale.")
        if st.button("🧠 Re-score with LLM Reasoning", key="rescore_llm"):
            with st.spinner("Re-scoring with LLM reasoning..."):
                scorer_llm = _get_risk_scorer(store.db_path, with_llm=True)
                cluster_results = scorer_llm.score_all_clusters(use_llm=True)
                agr_results = scorer_llm.score_all_agreements()
            st.success(f"Re-scored {len(cluster_results)} clause types with LLM reasoning.")
            st.rerun()

    heatmap = scorer.get_portfolio_heatmap()

    # ── Top metrics ─────────────────────────────────────────────────
    profiles = heatmap["agreement_profiles"]
    total = heatmap["total_agreements"]
    cols = st.columns(5)
    cols[0].metric("📄 Agreements Scored", total)
    cols[1].metric("🔴 High Risk", profiles.get("high-risk", 0),
                   delta=f"{profiles.get('high-risk', 0)/max(total,1):.0%}",
                   delta_color="inverse")
    cols[2].metric("🟡 Moderate", profiles.get("moderate", 0))
    cols[3].metric("🟢 Low Risk", profiles.get("low-risk", 0))
    cat_summary = heatmap["category_summary"]
    max_cat = max(cat_summary.items(), key=lambda x: x[1]["max_severity"])
    cols[4].metric("⚠️ Top Risk Category",
                   f"{RISK_ICONS.get(max_cat[0], '')} {max_cat[0].title()}")

    st.divider()

    tab_heatmap, tab_agreements, tab_clauses = st.tabs([
        "🗺️ Risk Heatmap", "📄 Agreement Profiles", "💼 Clause Risk Details",
    ])

    # ── TAB 1: HEATMAP ─────────────────────────────────────────────
    with tab_heatmap:
        st.subheader("Portfolio Risk Heatmap")

        # Category overview bars
        cat_data = []
        for cat, info in cat_summary.items():
            cat_data.append({
                "Category": f"{RISK_ICONS.get(cat, '')} {cat.title()}",
                "Clause Types": info["count"],
                "Max Severity": info["max_severity"],
            })
        cat_df = pd.DataFrame(cat_data)

        fig = px.bar(
            cat_df, x="Category", y="Clause Types",
            color="Max Severity",
            color_continuous_scale=["#3fb950", "#d29922", "#f85149"],
            range_color=[0, 1],
            title="Risk Categories — Clause Type Count & Max Severity",
        )
        fig.update_layout(coloraxis_colorbar_title="Severity")
        st.plotly_chart(fig, use_container_width=True)

        # Profile pie
        profile_df = pd.DataFrame([
            {"Profile": k, "Count": v}
            for k, v in profiles.items() if v > 0
        ])
        if not profile_df.empty:
            fig2 = px.pie(
                profile_df, values="Count", names="Profile",
                title="Agreement Risk Profile Distribution",
                color="Profile",
                color_discrete_map=PROFILE_COLORS,
                hole=0.4,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: AGREEMENT PROFILES ──────────────────────────────────
    with tab_agreements:
        st.subheader("Per-Agreement Risk Profiles")
        top_risks = heatmap["top_risk_agreements"]
        if top_risks:
            agr_data = []
            for a in top_risks:
                row = {
                    "Document": a["filename"][:50],
                    "Risk Score": a["score"],
                    "Profile": a["profile"],
                }
                cats = a["categories"] if isinstance(a["categories"], dict) else {}
                for cat in RISK_COLORS:
                    row[f"{RISK_ICONS.get(cat, '')} {cat[:5]}"] = cats.get(cat, 0.0)
                agr_data.append(row)

            agr_df = pd.DataFrame(agr_data)
            st.dataframe(agr_df, use_container_width=True, hide_index=True)

            # Risk score bar chart
            fig = px.bar(
                agr_df.sort_values("Risk Score"),
                x="Document", y="Risk Score", color="Profile",
                color_discrete_map=PROFILE_COLORS,
                title="Agreement Risk Scores (sorted)",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: CLAUSE RISK DETAILS ─────────────────────────────────
    with tab_clauses:
        st.subheader("Clause Type Risk Classification")
        clause_risks = scorer.db.execute("""
            SELECT cr.cluster_id, c.label, cr.categories, cr.max_severity, cr.primary_category,
                   (SELECT ch.chunk_text FROM chunks ch
                    JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
                    WHERE ca.cluster_id = c.cluster_id LIMIT 1) as sample_text
            FROM cluster_risks cr
            JOIN clusters c ON cr.cluster_id = c.cluster_id
            ORDER BY cr.max_severity DESC
        """).fetchall()

        if clause_risks:
            # Summary table
            clause_data = []
            for cid, label, cats_json, max_sev, primary, _ in clause_risks:
                cats = json.loads(cats_json) if isinstance(cats_json, str) else cats_json
                clause_data.append({
                    "Clause Type": label,
                    "Primary Category": f"{RISK_ICONS.get(primary, '')} {primary}",
                    "Max Severity": max_sev,
                    "Risk Areas": len(cats),
                })
            st.dataframe(pd.DataFrame(clause_data), use_container_width=True,
                         hide_index=True)

            st.divider()

            # Severity filter
            severity_filter = st.selectbox(
                "Filter by minimum severity",
                ["All", "critical (1.0)", "high (0.75)", "medium (0.5)", "low (0.25)"],
                key="risk_sev_filter",
            )
            min_sev = {"All": 0, "critical (1.0)": 1.0, "high (0.75)": 0.75,
                       "medium (0.5)": 0.5, "low (0.25)": 0.25}.get(severity_filter, 0)

            # Detailed risk cards with reasoning
            st.subheader("Risk Reasoning Details")
            for cid, label, cats_json, max_sev, primary, sample in clause_risks:
                if max_sev < min_sev:
                    continue
                cats = json.loads(cats_json) if isinstance(cats_json, str) else cats_json

                sev_label = "critical" if max_sev >= 1.0 else "high" if max_sev >= 0.75 else "medium" if max_sev >= 0.5 else "low"
                sev_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[sev_label]

                with st.expander(
                    f"{sev_color} **{label}** — {RISK_ICONS.get(primary, '')} {primary} "
                    f"({sev_label})",
                    expanded=(max_sev >= 0.75),
                ):
                    # Risk category breakdown
                    for cat in cats:
                        cat_name = cat["category"]
                        cat_sev = cat["severity"]
                        rationale = cat.get("rationale", "")
                        icon = RISK_ICONS.get(cat_name, "•")
                        sev_badge = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(cat_sev, "⚪")

                        st.markdown(
                            f"{sev_badge} **{icon} {cat_name.title()}** — *{cat_sev}*"
                        )
                        if rationale:
                            st.markdown(f"> {rationale}")
                        st.markdown("")

                    # Sample clause text
                    if sample:
                        st.markdown("---")
                        st.caption("📄 Sample clause text from corpus:")
                        st.text_area(
                            "Evidence",
                            (sample or "")[:800],
                            height=100,
                            disabled=True,
                            key=f"risk_sample_{cid}",
                        )
