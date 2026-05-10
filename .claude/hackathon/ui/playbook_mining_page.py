"""Playbook Mining page.

Lists draft rules that the miner has proposed. Users can accept (promote to
active) or retire each candidate.
"""
from __future__ import annotations

import streamlit as st

from core.playbooks.store import PlaybookStore


@st.cache_resource
def _get_playbook_store() -> PlaybookStore:
    from config import DB_PATH  # lazy — keeps module importable without dotenv
    return PlaybookStore(DB_PATH)


def render(store=None):  # noqa: ANN001 — store ignored; kept for app.py compat
    st.header("⛏️ Playbook Mining")
    st.caption("Review miner-proposed draft rules and promote or retire them.")

    ps = _get_playbook_store()

    # Mine from corpus (gap-close)
    with st.expander("⛏️ Mine from corpus"):
        st.caption("Build a fresh draft playbook from cluster coverage, "
                   "field distributions, categorical modes, and clause outliers in the current DuckDB.")
        new_name = st.text_input("New playbook name", value="Auto-mined draft")
        if st.button("Mine candidates now"):
            from core.store import ClusteringStore
            from core.playbooks.miner_runner import run_miner
            from config import DB_PATH
            cs = ClusteringStore(DB_PATH)
            new_pid, cands = run_miner(cs, ps, playbook_name=new_name)
            by_kind = {}
            for c in cands:
                by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
            st.success(f"Mined {len(cands)} candidates: {by_kind}")
            st.rerun()

    playbooks = ps.list_playbooks()

    if not playbooks:
        st.info("No playbooks found. Run the miner or import a playbook first.")
        return

    for pb in playbooks:
        rules = ps.list_rules(pb["playbook_id"])
        draft_rules = [r for r in rules if r["status"] == "draft"]
        if not draft_rules:
            continue

        st.subheader(f"📋 {pb['name']}")
        st.caption(f"{len(draft_rules)} draft rule(s) awaiting review")

        for rule in draft_rules:
            sev = rule.get("severity", "")
            with st.expander(f"**{rule['title']}** — `{sev}`", expanded=True):
                if rule.get("description"):
                    st.write(rule["description"])
                if rule.get("nl_assertion"):
                    st.markdown(f"_Assertion:_ {rule['nl_assertion']}")

                col_accept, col_retire, _ = st.columns([1, 1, 4])
                if col_accept.button("Accept", key=f"mine_accept_{rule['rule_id']}"):
                    ps.update_rule_status(rule["rule_id"], "active")
                    st.rerun()
                if col_retire.button("Retire", key=f"mine_retire_{rule['rule_id']}"):
                    ps.update_rule_status(rule["rule_id"], "retired")
                    st.rerun()

        st.divider()

    all_drafts = sum(
        1
        for pb in playbooks
        for r in ps.list_rules(pb["playbook_id"])
        if r["status"] == "draft"
    )
    if all_drafts == 0:
        st.success("All draft rules have been reviewed.")
