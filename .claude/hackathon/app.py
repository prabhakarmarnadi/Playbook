"""
Streamlit main app — grouped sidebar navigation.

5 sections:  Discover → Explore → Author → Analyze → Configure
"""
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from core.store import ClusteringStore

st.set_page_config(
    page_title="Agreement Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_store():
    return ClusteringStore(DB_PATH)


# ── Page registry: section → [(label, module_path)] ──────────────────────
SECTIONS = {
    "🚀 Discover": [
        ("📤 Upload & Process",      "ui.upload_page"),
        ("📁 Domains",               "ui.domain_page"),
        ("💼 Clause Types",          "ui.cluster_page"),
        ("🌳 Ontology Tree",         "ui.ontology_page"),
    ],
    "🔍 Explore": [
        ("📚 Clause Library",        "ui.clause_library_page"),
        ("�️ Cluster Map",           "ui.cluster_map_page"),
        ("�🕸️ Knowledge Graph",       "ui.kg_page"),
        ("🔎 Hybrid Search",         "ui.search_page"),
    ],
    "✏️ Author": [
        ("📋 Example Set Review",    "ui.example_set_page"),
        ("⚖️ Extraction Compare",    "ui.extraction_page"),
        ("🧮 Composite Fields",      "ui.composite_page"),
        ("✏️ Field CRUD",            "ui.field_crud_page"),
    ],
    "📜 Playbooks": [
        ("📜 Authoring", "ui.playbook_authoring_page"),
        ("⛏️ Mining",   "ui.playbook_mining_page"),
        ("🚨 Findings",  "ui.playbook_findings_page"),
    ],
    "📊 Analyze": [
        ("📊 Dashboard",             "ui.dashboard_page"),
        ("🛡️ Risk & Portfolio",      "ui.risk_page"),
        ("🏥 Pipeline Health",       "ui.eval_page"),
    ],
    "⚙️ Configure": [
        ("🔄 Feedback & Personalization", "ui.feedback_page"),
    ],
}


def main():
    st.sidebar.title("🧠 Agreement Intelligence")
    st.sidebar.caption("Discover → Explore → Author → Analyze → Configure")

    # Build grouped radio options
    all_pages = []
    for section, pages in SECTIONS.items():
        all_pages.append(f"─── {section} ───")
        for label, _ in pages:
            all_pages.append(label)

    # Map labels to modules
    label_to_module = {}
    for _, pages in SECTIONS.items():
        for label, module in pages:
            label_to_module[label] = module

    selected = st.sidebar.radio(
        "Navigate",
        all_pages,
        index=1,  # default to first real page
        format_func=lambda x: x,
        label_visibility="collapsed",
    )

    # Section headers are not navigable
    if selected.startswith("───"):
        st.info("Select a page from the sidebar.")
        return

    store = get_store()

    # Dynamic import and render
    module_path = label_to_module[selected]
    import importlib
    mod = importlib.import_module(module_path)
    mod.render(store)


if __name__ == "__main__":
    main()
