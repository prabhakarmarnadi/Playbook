"""Pipeline Health & Evaluation page.

Live eval dashboard: health score gauge, 48-metric breakdown, regression tracking.
Wires up eval/runner.py which was previously CLI-only.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.store import ClusteringStore


def _run_eval(db_path: str) -> dict:
    """Run unsupervised eval (no gold set, no LLM judge) — fast."""
    from eval.runner import run_evaluation
    return run_evaluation(db_path=db_path)


def _load_baseline(path: str) -> dict | None:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def render(store: ClusteringStore):
    st.header("🏥 Pipeline Health & Evaluation")
    st.caption("48 metrics across clustering, KG, fields, naming, intents, and system health.")

    eval_dir = Path(store.db_path).parent.parent / "eval"
    baseline_path = eval_dir / "baseline_results.json"

    # Check for cached results
    if "eval_results" not in st.session_state:
        st.info("Run the evaluation suite to see metrics.")
        if st.button("⚡ Run Evaluation (unsupervised — ~2s)", key="run_eval"):
            with st.spinner("Evaluating pipeline..."):
                results = _run_eval(store.db_path)
            st.session_state["eval_results"] = results
            st.rerun()
        return

    results = st.session_state["eval_results"]

    # ── Health Score Gauge ──────────────────────────────────────────
    health = results.get("health.score", 0)
    grade = results.get("health.grade", "Unknown")

    col_gauge, col_info = st.columns([1, 2])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            number={"suffix": f"  {grade}", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.5, 0.7, 0.85, 1]},
                "bar": {"color": "#2ecc71" if health >= 0.85 else
                        "#f39c12" if health >= 0.7 else
                        "#e74c3c"},
                "steps": [
                    {"range": [0, 0.5], "color": "#fde2e2"},
                    {"range": [0.5, 0.7], "color": "#fef3cd"},
                    {"range": [0.7, 0.85], "color": "#d4edda"},
                    {"range": [0.85, 1], "color": "#c3e6cb"},
                ],
                "threshold": {"line": {"color": "black", "width": 2},
                              "value": health},
            },
            title={"text": "Pipeline Health Score"},
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        # Health components
        st.markdown("**Health Score Components**")
        components = {
            "Quality (40%)": results.get("health.quality", 0),
            "Coverage (30%)": results.get("health.coverage", 0),
            "Structure (20%)": results.get("health.structure", 0),
            "Consistency (10%)": results.get("health.consistency", 0),
        }
        for label, val in components.items():
            bar_pct = int(val * 100)
            color = "#2ecc71" if val >= 0.85 else "#f39c12" if val >= 0.7 else "#e74c3c"
            st.markdown(
                f"**{label}**: `{val:.3f}` "
                f'<div style="background:#eee;border-radius:4px;height:12px;width:100%">'
                f'<div style="background:{color};width:{bar_pct}%;height:12px;border-radius:4px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    tab_metrics, tab_regression = st.tabs(["📊 All Metrics", "📈 Regression Check"])

    # ── TAB 1: ALL METRICS BREAKDOWN ───────────────────────────────
    with tab_metrics:
        st.subheader("Full Metric Breakdown")

        # Group metrics by prefix
        groups: dict[str, list] = {}
        for key, val in sorted(results.items()):
            if key.startswith("_"):
                continue
            prefix = key.split(".")[0] if "." in key else "other"
            groups.setdefault(prefix, []).append((key, val))

        for group_name, metrics in groups.items():
            with st.expander(f"**{group_name.upper()}** ({len(metrics)} metrics)", expanded=False):
                rows = []
                for key, val in metrics:
                    if isinstance(val, float):
                        rows.append({"Metric": key, "Value": f"{val:.4f}"})
                    elif isinstance(val, (int, bool)):
                        rows.append({"Metric": key, "Value": str(val)})
                    else:
                        rows.append({"Metric": key, "Value": str(val)[:80]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── TAB 2: REGRESSION CHECK ────────────────────────────────────
    with tab_regression:
        st.subheader("Regression Detection")
        baseline = _load_baseline(str(baseline_path))

        if baseline is None:
            st.info("No baseline found. Save current results as baseline to enable regression tracking.")
            if st.button("💾 Save as Baseline", key="save_baseline"):
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                with open(baseline_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                st.success(f"Saved baseline to {baseline_path}")
                st.rerun()
            return

        from eval.regression import RegressionTracker
        tracker = RegressionTracker(baseline)
        report = tracker.check(results)

        # Summary
        cols = st.columns(3)
        cols[0].metric("✅ Pass", report["pass_count"],
                       delta_color="normal")
        cols[1].metric("❌ Regressed", report["fail_count"],
                       delta_color="inverse")
        cols[2].metric("Status", "PASS ✅" if report["fail_count"] == 0 else "REGRESSED ❌")

        # Details table
        if report.get("details"):
            rows = []
            for d in report["details"]:
                rows.append({
                    "Status": "✅" if d["status"] == "pass" else "❌",
                    "Metric": d["metric"],
                    "Baseline": f"{d['baseline']:.4f}" if isinstance(d['baseline'], float) else str(d['baseline']),
                    "Current": f"{d['current']:.4f}" if isinstance(d['current'], float) else str(d['current']),
                    "Delta": f"{d.get('delta', 0):+.4f}" if isinstance(d.get('delta'), float) else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
