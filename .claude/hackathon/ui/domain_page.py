"""Domain Explorer page — Demo Act 1: Domain Intelligence.

Shows macro domain distribution bar chart, confidence scores, domain descriptions.
"V1 clusters everything together. V2 knows NDA != SaaS."
"""
import streamlit as st
import plotly.express as px
import pandas as pd

from core.store import ClusteringStore


def render(store: ClusteringStore):
    st.header("📁 Domain Explorer")
    st.caption("Macro domains discovered from document-level summary embeddings — "
               "no per-doc LLM calls, just smart clustering.")

    domains = store.get_domains()
    if not domains:
        st.info("No domains found. Run the pipeline first from the Upload page.")
        return

    # ── Bar chart: domain distribution ─────────────────────────────────────
    df = pd.DataFrame(domains)
    fig = px.bar(
        df, x="label", y="agreement_count",
        color="confidence",
        color_continuous_scale="Blues",
        title="Document Distribution Across Domains",
        labels={"agreement_count": "Documents", "label": "Domain", "confidence": "Cluster Confidence"},
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # ── Domain detail cards ────────────────────────────────────────────────
    st.subheader("Domain Details")
    cols = st.columns(min(len(domains), 3))
    for i, domain in enumerate(domains):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### 📁 {domain['label']}")
                st.metric("Agreements", domain["agreement_count"])
                if domain.get("confidence"):
                    st.metric("Confidence", f"{domain['confidence']:.2f}")
                if domain.get("description"):
                    st.caption(domain["description"])

                # Show agreements in this domain
                agreements = store.get_agreements(domain["domain_id"])
                if agreements:
                    with st.expander(f"View {len(agreements)} agreements"):
                        for agr in agreements[:10]:
                            conf = f" ({agr['domain_confidence']:.2f})" if agr.get("domain_confidence") else ""
                            st.text(f"  {agr['filename']}{conf}")
                        if len(agreements) > 10:
                            st.caption(f"... and {len(agreements) - 10} more")

    # ── Comparison callout ─────────────────────────────────────────────────
    st.divider()
    st.info(
        "**V1 vs V2:** V1 clusters everything into a flat topic structure. "
        "V2 first identifies document domains (SaaS, NDA, Employment...) then "
        "clusters within each domain. Indemnity in an NDA is treated differently "
        "from indemnity in a SaaS agreement."
    )
