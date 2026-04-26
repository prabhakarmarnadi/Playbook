"""Upload & Process page — Demo Act 1: Domain Intelligence.

Users upload agreements (or load sample data), then the pipeline runs
with progressive status updates.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, SAMPLE_DIR
from core.store import ClusteringStore
from core.pipeline import run_full_pipeline, load_sample_documents


def render(store: ClusteringStore):
    st.header("📤 Upload & Process Agreements")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load Sample Data")
        # Check for legacy example data
        legacy_path = Path(__file__).parent.parent.parent / "legacy_stack" / "apr-agreement-clustering" / "examples" / "data"
        data_source = SAMPLE_DIR
        if legacy_path.exists():
            use_legacy = st.checkbox("Use legacy CUAD sample data", value=True)
            if use_legacy:
                data_source = legacy_path

        if st.button("🚀 Run Full Pipeline", type="primary"):
            _run_pipeline(store, data_dir=data_source)

    with col2:
        st.subheader("Upload Your Own")
        uploaded = st.file_uploader(
            "Upload .txt or .pdf agreements",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("📄 Process Uploaded Files"):
            agreements = []
            for f in uploaded:
                content = f.read().decode("utf-8", errors="replace")
                agreements.append({
                    "agreement_id": f"agr_{hash(f.name) % 100000:05d}",
                    "filename": f.name,
                    "raw_text": content,
                })
            _run_pipeline(store, agreements=agreements)

    # Show current state
    st.divider()
    stats = store.get_stats()
    if stats["agreements"] > 0:
        st.success(f"Database contains {stats['agreements']} agreements, "
                   f"{stats['domains']} domains, {stats['clusters']} clusters, "
                   f"{stats['fields']} fields")
    else:
        st.info("No data loaded yet. Use the buttons above to get started.")


def _run_pipeline(store: ClusteringStore, agreements=None, data_dir=None):
    """Run the pipeline with Streamlit progress indicators."""
    progress_bar = st.progress(0, text="Initializing...")
    status_area = st.empty()

    stages = {
        "load": (0.05, "Loading documents..."),
        "chunk": (0.15, "Chunking documents..."),
        "embed": (0.30, "Computing clause-level embeddings..."),
        "macro": (0.50, "Discovering macro domains..."),
        "micro": (0.65, "Finding clause types within domains..."),
        "merge": (0.72, "Merging similar topics..."),
        "fields": (0.80, "Discovering extractable fields..."),
        "extract": (0.90, "Extracting field values..."),
        "done": (1.0, "Pipeline complete!"),
    }

    def callback(stage, msg):
        pct, default_msg = stages.get(stage, (0.5, msg))
        progress_bar.progress(pct, text=msg)
        status_area.info(msg)

    try:
        run_full_pipeline(
            agreements=agreements,
            data_dir=data_dir,
            db_path=DB_PATH,
            callback=callback,
        )
        progress_bar.progress(1.0, text="Done!")
        st.success("Pipeline completed successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        raise
