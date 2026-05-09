"""Playbook Authoring page.

Lists all playbooks and their rules with severity, predicate JSON, NL assertion,
and preferred language. Provides Activate/Retire buttons per rule and an import
expander that accepts an xlsx or docx file path.
"""
from __future__ import annotations
import json

import streamlit as st

from core.playbooks.store import PlaybookStore


@st.cache_resource
def _get_playbook_store() -> PlaybookStore:
    from config import DB_PATH  # lazy — keeps module importable without dotenv
    return PlaybookStore(DB_PATH)


def render(store=None):  # noqa: ANN001 — store ignored; kept for app.py compat
    st.header("📜 Playbook Authoring")
    st.caption("Review, activate, retire, and import playbook rules.")

    ps = _get_playbook_store()

    # ── Import expander ────────────────────────────────────────────────────
    with st.expander("Import a playbook (xlsx/docx)"):
        file_path = st.text_input("File path", placeholder="/path/to/playbook.xlsx")
        pb_name = st.text_input("Playbook name (optional)", placeholder="My Playbook")
        if st.button("Import", key="pb_import"):
            if not file_path:
                st.warning("Enter a file path first.")
            else:
                try:
                    from core.playbooks.importers import import_file
                    kwargs = {"name": pb_name} if pb_name else {}
                    pid = import_file(ps, file_path, **kwargs)
                    rules = ps.list_rules(pid)
                    st.success(f"Imported playbook `{pid}` with {len(rules)} rule(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    st.divider()

    # ── Playbook list ──────────────────────────────────────────────────────
    playbooks = ps.list_playbooks()
    if not playbooks:
        st.info("No playbooks found. Import one above or run the miner.")
        return

    for pb in playbooks:
        pb_label = f"{pb['name']} — {pb['status']} (v{pb['version']})"
        with st.expander(pb_label, expanded=False):
            st.caption(pb.get("description") or "")
            rules = ps.list_rules(pb["playbook_id"])
            if not rules:
                st.write("_No rules in this playbook._")
                continue

            for rule in rules:
                sev = rule.get("severity", "")
                sev_color = {
                    "approval_required": "🔴",
                    "warn": "🟡",
                    "info": "🔵",
                }.get(sev, "⚪")

                rule_label = f"{sev_color} **{rule['title']}** — `{sev}` — `{rule['status']}`"
                with st.expander(rule_label, expanded=False):
                    # Predicate JSON
                    if rule.get("predicate"):
                        st.markdown("**Predicate**")
                        st.code(json.dumps(rule["predicate"], indent=2), language="json")

                    # NL assertion
                    if rule.get("nl_assertion"):
                        st.markdown("**NL Assertion**")
                        st.write(rule["nl_assertion"])

                    # Preferred language
                    if rule.get("preferred_language"):
                        st.markdown("**Preferred Language**")
                        st.write(rule["preferred_language"])

                    # Walkaway language
                    if rule.get("walkaway_language"):
                        st.markdown("**Walkaway Language**")
                        st.write(rule["walkaway_language"])

                    # Activate / Retire buttons
                    col_a, col_r, _ = st.columns([1, 1, 4])
                    if col_a.button("Activate", key=f"act_{rule['rule_id']}"):
                        ps.update_rule_status(rule["rule_id"], "active")
                        st.rerun()
                    if col_r.button("Retire", key=f"ret_{rule['rule_id']}"):
                        ps.update_rule_status(rule["rule_id"], "retired")
                        st.rerun()
