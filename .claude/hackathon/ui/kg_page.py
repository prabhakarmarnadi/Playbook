"""Knowledge Graph Explorer page.

Interactive exploration of the multi-relational legal knowledge graph:
recommendations, anomalies, archetypes, implication rules.
Wires up core/knowledge_graph.py which was previously CLI-only.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from core.store import ClusteringStore


def _get_kg(store: ClusteringStore):
    from core.knowledge_graph import ContractKnowledgeGraph
    kg = ContractKnowledgeGraph(store)
    kg.build()
    return kg


def render(store: ClusteringStore):
    st.header("🕸️ Knowledge Graph Explorer")
    st.caption("Multi-relational graph: agreements, clause types, intents, parties.")

    if "kg_instance" not in st.session_state:
        with st.spinner("Building knowledge graph..."):
            st.session_state["kg_instance"] = _get_kg(store)

    kg = st.session_state["kg_instance"]
    stats = kg.stats

    # Top metrics
    cols = st.columns(5)
    cols[0].metric("📄 Agreements", stats.agreements)
    cols[1].metric("💼 Clause Types", stats.clause_types)
    cols[2].metric("🎯 Intent Types", stats.intent_types)
    cols[3].metric("👥 Parties", stats.parties)
    cols[4].metric("🔗 Edges", stats.edges)

    st.divider()

    tab_arch, tab_recs, tab_anom, tab_rules = st.tabs([
        "🏛️ Deal Archetypes",
        "💡 Recommendations",
        "🔍 Anomaly Detection",
        "📐 Implication Rules",
    ])

    # ── TAB 1: ARCHETYPES ──────────────────────────────────────────
    with tab_arch:
        st.subheader("Deal Family Archetypes")
        st.caption("Community detection over intent fingerprints — clusters agreements into deal families.")

        archetypes = kg.discover_archetypes()
        if archetypes:
            for i, arch in enumerate(archetypes):
                members = arch.get("members", [])
                label = arch.get("label", f"Archetype {i+1}")
                intents = arch.get("common_intents", [])

                with st.expander(
                    f"🏛️ {label} — {len(members)} agreements | "
                    f"{len(intents)} shared intents",
                    expanded=(i == 0),
                ):
                    if intents:
                        st.markdown("**Common Intents:** " + ", ".join(
                            f"`{it}`" for it in intents[:15]
                        ))
                    st.markdown(f"**Members:** {', '.join(str(m)[:40] for m in members[:10])}")
                    if len(members) > 10:
                        st.caption(f"... and {len(members) - 10} more")
        else:
            st.info("Not enough data for archetype discovery.")

    # ── TAB 2: RECOMMENDATIONS ─────────────────────────────────────
    with tab_recs:
        st.subheader("Missing Intent Recommendations")
        st.caption("Select an agreement to see statistically expected but missing intents.")

        agreements = store.get_agreements()
        if not agreements:
            st.info("No agreements found.")
        else:
            agr_options = {
                f"{a.get('filename', a['agreement_id'][:20])}": a["agreement_id"]
                for a in agreements[:100]
            }
            selected = st.selectbox("Select Agreement", list(agr_options.keys()),
                                    key="kg_rec_agr")
            agr_id = agr_options[selected]

            recs = kg.recommend(agr_id)
            if recs:
                rec_data = []
                for r in recs:
                    rec_data.append({
                        "Missing Intent": r.get("intent", r.get("label", "?")),
                        "Confidence": f"{r.get('confidence', r.get('score', 0)):.0%}",
                        "Reason": r.get("reason", r.get("rationale", "")),
                    })
                st.dataframe(pd.DataFrame(rec_data), use_container_width=True,
                             hide_index=True)
            else:
                st.success("This agreement has complete intent coverage — no recommendations.")

    # ── TAB 3: ANOMALIES ───────────────────────────────────────────
    with tab_anom:
        st.subheader("Anomaly Detection")
        st.caption("Flag unusual intent combinations or missing expected clauses.")

        agreements = store.get_agreements()
        if agreements:
            agr_options = {
                f"{a.get('filename', a['agreement_id'][:20])}": a["agreement_id"]
                for a in agreements[:100]
            }
            selected = st.selectbox("Select Agreement", list(agr_options.keys()),
                                    key="kg_anom_agr")
            agr_id = agr_options[selected]

            anomalies = kg.detect_anomalies(agr_id)
            if anomalies:
                for anom in anomalies:
                    severity = anom.get("severity", "info")
                    icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🔵"
                    st.markdown(
                        f"{icon} **{anom.get('type', 'Anomaly')}**: "
                        f"{anom.get('description', anom.get('message', str(anom)))}"
                    )
            else:
                st.success("No anomalies detected in this agreement.")

    # ── TAB 4: IMPLICATION RULES ───────────────────────────────────
    with tab_rules:
        st.subheader("Intent Implication Rules")
        st.caption("\"If intent A is present, intent B appears X% of the time.\"")

        rules = kg.get_implication_rules()
        if rules:
            rule_data = []
            for r in rules[:50]:
                rule_data.append({
                    "If Present": r.get("antecedent", r.get("if_intent", "?")),
                    "Then Expect": r.get("consequent", r.get("then_intent", "?")),
                    "Confidence": f"{r.get('confidence', r.get('support', 0)):.0%}",
                    "Support": r.get("support_count", r.get("count", 0)),
                })
            st.dataframe(pd.DataFrame(rule_data), use_container_width=True,
                         hide_index=True)
        else:
            st.info("Not enough co-occurrence data for implication rules.")
