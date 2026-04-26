"""Field CRUD page — Demo Act 7.

Rename, retype, add examples, create/delete fields. Users own their schema.
"Every refinement makes the system smarter."
"""
import uuid
import streamlit as st

from core.store import ClusteringStore


def render(store: ClusteringStore):
    st.header("✏️ Field Definition CRUD")
    st.caption("Rename, retype, add examples, create new fields. Users own their schema.")

    clusters = store.get_clusters()
    if not clusters:
        st.info("No clusters found. Run the pipeline first.")
        return

    # ── Cluster selector ───────────────────────────────────────────────
    cluster_options = {f"{c['label']}": c["cluster_id"] for c in clusters}
    selected = st.selectbox("Select Cluster", list(cluster_options.keys()), key="crud_cluster")
    cluster_id = cluster_options[selected]

    fields = store.get_fields(cluster_id)

    # ── Existing fields ────────────────────────────────────────────────
    if fields:
        st.subheader("Current Fields")
        for f in fields:
            with st.expander(f"📋 {f['name']} ({f['field_type']}) — {f.get('source', '?')}", expanded=False):
                col1, col2 = st.columns(2)
                new_name = col1.text_input("Name", f["name"], key=f"name_{f['field_id']}")
                new_type = col2.selectbox(
                    "Type",
                    ["string", "int", "float", "date", "enum", "bool"],
                    index=["string", "int", "float", "date", "enum", "bool"].index(f["field_type"]),
                    key=f"type_{f['field_id']}",
                )
                new_desc = st.text_input("Description", f.get("description", ""), key=f"desc_{f['field_id']}")

                # Gold examples
                examples = f.get("examples", [])
                st.markdown("**Gold examples:**")
                if examples:
                    for i, ex in enumerate(examples):
                        st.text(f"  {i+1}. {ex}")
                else:
                    st.caption("No examples yet.")

                new_example_text = st.text_input("Example text", key=f"ext_{f['field_id']}",
                                                  placeholder='"net 30 days"')
                new_example_val = st.text_input("Example value", key=f"exv_{f['field_id']}",
                                                 placeholder="30")

                bcol1, bcol2, bcol3 = st.columns(3)
                if bcol1.button("💾 Update", key=f"upd_{f['field_id']}"):
                    updates = {}
                    if new_name != f["name"]:
                        updates["name"] = new_name
                        updates["source"] = "user_modified"
                    if new_type != f["field_type"]:
                        updates["field_type"] = new_type
                        updates["source"] = "user_modified"
                    if new_desc != f.get("description", ""):
                        updates["description"] = new_desc
                    if updates:
                        store.update_field(f["field_id"], **updates)
                        st.success(f"Updated {f['name']}")
                        st.rerun()

                if bcol2.button("➕ Add Example", key=f"addex_{f['field_id']}"):
                    if new_example_text and new_example_val:
                        current_examples = list(examples)
                        current_examples.append({"text": new_example_text, "value": new_example_val})
                        store.update_field(f["field_id"], examples=current_examples, source="user_modified")
                        st.success("Example added")
                        st.rerun()

                if bcol3.button("🗑️ Delete", key=f"del_{f['field_id']}"):
                    store.delete_field(f["field_id"])
                    st.rerun()

    # ── Create new field ───────────────────────────────────────────────
    st.divider()
    st.subheader("Create New Field")
    with st.form("new_field"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Field name", placeholder="early_payment_discount")
        ftype = col2.selectbox("Type", ["string", "int", "float", "date", "enum", "bool"])
        desc = st.text_input("Description", placeholder="Discount percentage for early payment")

        if st.form_submit_button("➕ Create Field"):
            if name:
                field_id = f"f_{uuid.uuid4().hex[:8]}"
                store.upsert_field(field_id, cluster_id, name, ftype, desc, "user_created")
                st.success(f"Created field: {name} ({ftype})")
                st.rerun()
            else:
                st.warning("Name is required.")
