"""Composite Fields page — Demo Act 5.

Define composite fields as formulas. Correct a base field → cascade animation.
"One correction. Three fields improved. Zero AI cost."
"""
import time
import uuid

import streamlit as st

from core.store import ClusteringStore
from core.composite import (
    evaluate_all_composites, compute_cascade,
    get_dependencies, detect_cycles,
)


def render(store: ClusteringStore):
    st.header("🧮 Composite Fields")
    st.caption("Define formulas over extracted fields. Corrections cascade automatically — zero LLM cost.")

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found. Run the pipeline first.")
        return

    # ── Cluster selector ───────────────────────────────────────────────
    cluster_options = {f"{c['label']}": c["cluster_id"] for c in clusters}
    selected = st.selectbox("Select Cluster", list(cluster_options.keys()), key="comp_cluster")
    cluster_id = cluster_options[selected]

    fields = store.get_fields(cluster_id)
    composites = store.get_composites(cluster_id)

    # ── Define new composite ───────────────────────────────────────────
    st.subheader("Define Composite Field")
    with st.form("new_composite"):
        col1, col2 = st.columns(2)
        with col1:
            comp_name = st.text_input("Name", placeholder="total_contract_value")
        with col2:
            comp_expr = st.text_input("Expression", placeholder="unit_price * quantity * term_years")

        st.caption(f"Available base fields: {', '.join(f['name'] for f in fields)}")
        if composites:
            st.caption(f"Available composites: {', '.join(c['name'] for c in composites)}")

        if st.form_submit_button("Add Composite"):
            if comp_name and comp_expr:
                # Validate
                deps = get_dependencies(comp_expr)
                all_composites = {c["name"]: c["expression"] for c in composites}
                all_composites[comp_name] = comp_expr
                errors = detect_cycles(all_composites)

                if errors:
                    st.error(f"Cycle detected: {errors}")
                else:
                    source_ids = [f["field_id"] for f in fields if f["name"] in deps]
                    store.upsert_composite(
                        f"comp_{uuid.uuid4().hex[:8]}", comp_name, comp_expr,
                        source_ids, "float", cluster_id,
                    )
                    st.success(f"Created composite: {comp_name} = {comp_expr}")
                    st.rerun()

    # ── Current composites ─────────────────────────────────────────────
    composites = store.get_composites(cluster_id)
    if composites:
        st.subheader("Defined Composites")
        for comp in composites:
            st.markdown(f"🧮 **{comp['name']}** = `{comp['expression']}`")

    # ── Evaluate composites with current values ────────────────────────
    if composites and fields:
        st.subheader("Computed Values")

        # Get base field values (from first agreement's extractions)
        extractions = store.get_extractions(cluster_id=cluster_id)
        base_values = {}
        for f in fields:
            match = [e for e in extractions if e["field_id"] == f["field_id"] and e.get("value")]
            if match:
                try:
                    base_values[f["name"]] = float(match[0]["value"])
                except (ValueError, TypeError):
                    base_values[f["name"]] = None

        if base_values:
            comp_dict = {c["name"]: c["expression"] for c in composites}
            results = evaluate_all_composites(comp_dict, base_values)

            cols = st.columns(max(len(results), 1))
            for i, (name, val) in enumerate(results.items()):
                with cols[i % len(cols)]:
                    display = f"${val:,.2f}" if val is not None else "—"
                    st.metric(name, display)

        # ── Cascade demo ───────────────────────────────────────────────
        st.subheader("🔄 Correction Cascade Demo")
        st.caption("Change a base field value and watch composites cascade.")

        numeric_fields = [f for f in fields if f["field_type"] in ("int", "float")]
        if numeric_fields and base_values:
            change_field = st.selectbox("Field to correct",
                                        [f["name"] for f in numeric_fields if f["name"] in base_values])
            current = base_values.get(change_field)
            if current is not None:
                new_val = st.number_input(f"New value for {change_field}",
                                          value=float(current) * 1.1,
                                          key="cascade_val")
                if st.button("⚡ Show Cascade"):
                    comp_dict = {c["name"]: c["expression"] for c in composites}
                    cascade = compute_cascade(comp_dict, base_values, change_field, current, new_val)

                    st.markdown(f"✏️ **{change_field}** corrected: "
                               f"${current:,.2f} → ${new_val:,.2f}")

                    for step in cascade:
                        time.sleep(0.4)  # Animation delay
                        old = step["old_value"]
                        new = step["new_value"]
                        if old is not None and new is not None and old != 0:
                            pct = ((new - old) / abs(old)) * 100
                            st.markdown(
                                f"&emsp; ↓ 🧮 **{step['name']}** = `{step['expression']}`\n\n"
                                f"&emsp;&emsp; = ${new:,.2f} (was ${old:,.2f}) "
                                f"{'↑' if pct > 0 else '↓'}{abs(pct):.1f}%"
                            )
                        else:
                            st.markdown(
                                f"&emsp; ↓ 🧮 **{step['name']}** = `{step['expression']}` → {new}"
                            )

                    st.success(f"✅ 1 correction → {len(cascade)} composites updated → 0 AI calls")


    # ── Delete composite ───────────────────────────────────────────────
    if composites:
        st.divider()
        del_comp = st.selectbox("Delete composite", [c["name"] for c in composites], key="del_comp")
        if st.button("🗑️ Delete"):
            comp_to_del = next(c for c in composites if c["name"] == del_comp)
            store.delete_composite(comp_to_del["field_id"])
            st.rerun()
