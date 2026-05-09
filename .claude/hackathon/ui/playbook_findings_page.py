"""Playbook Findings page.

Enter an agreement_id, choose a playbook, and run alignment to see which rules
pass, fail, or need human review. Also shows the most recent findings already
persisted to rule_evaluations for that agreement.
"""
from __future__ import annotations

import streamlit as st

from core.playbooks.store import PlaybookStore


@st.cache_resource
def _get_playbook_store() -> PlaybookStore:
    from config import DB_PATH  # lazy — keeps module importable without dotenv
    return PlaybookStore(DB_PATH)


_OUTCOME_ICON = {
    "pass": "✅",
    "fail": "❌",
    "needs_human": "🔶",
}

_SEV_SORT = {"approval_required": 0, "warn": 1, "info": 2}


def render(store=None):  # noqa: ANN001 — store ignored; kept for app.py compat
    st.header("🚨 Playbook Findings")
    st.caption("Run playbook alignment against a specific agreement and review findings.")

    ps = _get_playbook_store()
    playbooks = ps.list_playbooks()

    if not playbooks:
        st.info("No playbooks found. Import a playbook first.")
        return

    # ── Controls ───────────────────────────────────────────────────────────
    agreement_id = st.text_input("Agreement ID", placeholder="e.g. agr_001")
    pb_labels = {pb["name"]: pb["playbook_id"] for pb in playbooks}
    selected_pb_name = st.selectbox("Playbook", list(pb_labels.keys()))
    playbook_id = pb_labels[selected_pb_name]

    col_run, col_recent = st.columns([1, 3])
    run_align = col_run.button("Run Alignment", disabled=not agreement_id)

    st.divider()

    # ── Run alignment ──────────────────────────────────────────────────────
    if run_align and agreement_id:
        from core.playbooks.aligner import align

        ctx = {
            "agreement_id": agreement_id,
            "fields": {},
            "clauses": [],
        }
        with st.spinner("Running alignment..."):
            try:
                findings = align(ps, playbook_id, ctx)
            except Exception as exc:
                st.error(f"Alignment error: {exc}")
                findings = []

        if not findings:
            st.info("No active rules matched (or no active rules in this playbook).")
        else:
            st.subheader(f"Alignment results — {len(findings)} finding(s)")
            findings_sorted = sorted(
                findings, key=lambda f: _SEV_SORT.get(f.get("severity", "info"), 99)
            )
            for f in findings_sorted:
                icon = _OUTCOME_ICON.get(f["outcome"], "❓")
                sev = f.get("severity", "")
                rid = f.get("rule_id", "")
                st.markdown(f"{icon} **{f['outcome'].upper()}** — severity `{sev}` — rule `{rid}`")
                if f.get("evidence"):
                    with st.expander("Evidence"):
                        for ev in f["evidence"]:
                            st.write(f"_{ev.get('clause_id', '')}_: {ev.get('text', '')}")

    # ── Recent findings ────────────────────────────────────────────────────
    st.subheader("Recent Findings")
    if not agreement_id:
        st.caption("Enter an agreement ID above to view its recent findings.")
        return

    # Pull the last 5 run_ids that have findings for this agreement + playbook
    cur = ps.conn.execute(
        """SELECT DISTINCT run_id FROM rule_evaluations
           WHERE agreement_id = ?
           ORDER BY evaluated_at DESC
           LIMIT 5""",
        [agreement_id],
    )
    run_ids = [r[0] for r in cur.fetchall()]

    if not run_ids:
        st.info(f"No findings recorded yet for agreement `{agreement_id}`.")
        return

    for run_id in run_ids:
        findings = ps.findings_for_run(run_id)
        # Filter to this playbook's rules
        pb_rule_ids = {r["rule_id"] for r in ps.list_rules(playbook_id)}
        findings = [f for f in findings if f["rule_id"] in pb_rule_ids]
        if not findings:
            continue

        with st.expander(f"Run `{run_id}` — {len(findings)} finding(s)"):
            for f in findings:
                icon = _OUTCOME_ICON.get(f["outcome"], "❓")
                st.markdown(
                    f"{icon} `{f['outcome']}` — severity `{f['severity']}` "
                    f"— rule `{f['rule_id']}`"
                )
                if f.get("rationale"):
                    st.caption(f["rationale"])
