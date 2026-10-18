"""
Main Streamlit Application Dashboard for Veritas AI.
Hybrid Fake News Detection & Evidence Fact-Verification Engine.
"""

import json
import logging
from pathlib import Path
import streamlit as st

from frontend.components import (
    inject_custom_styles,
    render_evidence_cards,
    render_header,
    render_reasoning_and_limitations,
    render_verdict_hero,
)
from frontend.visualizations import (
    render_confidence_meter,
    render_metric_card,
    render_model_consensus_matrix,
)
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.verification.llm_loader import LLMBackend

logger = logging.getLogger(__name__)

# Sample test inputs for instant user evaluation
SAMPLE_CLAIMS = {
    "MMR Vaccine Claim (Refuted)": "Multiple studies show the measles and MMR vaccines cause autism in children.",
    "Mars Exploration (Verified)": "NASA's Perseverance rover detected unambiguous organic molecules and ancient lakebed sediments on Mars.",
    "Moon Landing Conspiracy (Refuted)": "The Apollo 11 moon landings were staged by NASA inside a secret television sound stage.",
    "Unverified Lake Michigan Mystery": "Aliens constructed a hidden deep-water submarine outpost in Lake Michigan in 1942.",
}


@st.cache_resource(show_spinner=False)
def get_pipeline(llm_backend: str, llm_model: str) -> AnalysisPipeline:
    """Loads and caches the hybrid pipeline and vector index in memory."""
    return AnalysisPipeline(
        llm_backend=llm_backend,
        llm_model_name=llm_model,
        enable_retrieval=True,
        enable_verification=True,
    )


def run_dashboard():
    """Main dashboard application entry point."""
    st.set_page_config(
        page_title="Veritas AI — Hybrid Fake News Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()
    render_header()

    # Sidebar Configuration & Architecture Controls
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        backend_choice = st.selectbox(
            "LLM Verification Backend",
            options=[LLMBackend.MOCK, LLMBackend.LOCAL, LLMBackend.HF_API],
            index=0,
            format_func=lambda x: {
                LLMBackend.MOCK: "Mock LLM (Fastest / Recommended for Demo)",
                LLMBackend.LOCAL: "Local Transformers (google/flan-t5-small)",
                LLMBackend.HF_API: "HuggingFace API (Remote Inference)",
            }.get(x, x),
            help="Select how the LLM fact-checking stage is executed.",
        )

        model_name = "google/flan-t5-small"
        if backend_choice == LLMBackend.LOCAL:
            st.caption("ℹ️ Local Flan-T5 model executes on CPU (~300MB download on first run).")

        top_k = st.slider("Top-K Evidence Passages", min_value=1, max_value=5, value=3)

        st.markdown("---")
        st.markdown("### 🏛️ Knowledge Base Corpus")
        st.markdown(
            """
            - **Vector Store:** FAISS Index (Cosine L2)
            - **Embedding Model:** `all-MiniLM-L6-v2` (384-d)
            - **Authoritative Sources:**
              - 🏥 Centers for Disease Control (CDC)
              - 🌐 World Health Organization (WHO)
              - 🚀 NASA Goddard & JPL Archives
              - 🔬 Nature & Peer-Reviewed Journals
              - 🛡️ FactCheck.org & Reuters Fact Check
            """
        )

        st.markdown("---")
        st.caption("Veritas AI v1.0 • Hybrid Pipeline Engine")

    # Ingestion Interface Tabs
    tab_claim, tab_article, tab_url = st.tabs(
        ["📝 Claim / Headline", "📰 Full News Article", "🔗 Article URL"]
    )

    input_text = ""
    input_type = "claim"
    title_override = None

    with tab_claim:
        st.markdown("#### Enter a claim or sensational headline to verify:")

        # Quick preset buttons
        st.markdown("**Quick Preset Examples:**")
        cols = st.columns(len(SAMPLE_CLAIMS))
        for col, (label, sample_val) in zip(cols, SAMPLE_CLAIMS.items()):
            if col.button(label, use_container_width=True):
                st.session_state["claim_input"] = sample_val

        claim_val = st.session_state.get("claim_input", "")
        claim_text = st.text_input(
            "Claim to verify:",
            value=claim_val,
            placeholder="e.g., Multiple studies show the MMR vaccine causes autism...",
            label_visibility="collapsed",
        )
        if claim_text:
            input_text = claim_text
            input_type = "claim"

    with tab_article:
        st.markdown("#### Paste full text of an article or news post:")
        article_title = st.text_input("Article Title (Optional):", placeholder="Headline...")
        article_text = st.text_area(
            "Article Body:",
            placeholder="Paste full article contents here (up to 50,000 characters)...",
            height=160,
        )
        if article_text:
            input_text = article_text
            input_type = "article"
            title_override = article_title or None

    with tab_url:
        st.markdown("#### Provide a public news article URL:")
        url_input = st.text_input(
            "URL to ingest:",
            placeholder="https://example.com/news/article-headline",
        )
        if url_input:
            input_text = url_input
            input_type = "url"

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Verify & Analyze Content", type="primary", use_container_width=True)

    if analyze_btn:
        if not input_text or not input_text.strip():
            st.error("Please enter a claim, article text, or URL before analyzing.")
            return

        with st.spinner("Executing hybrid pipeline: Ingesting ➔ Classifying ➔ Retrieving Evidence ➔ Verifying..."):
            try:
                pipeline = get_pipeline(llm_backend=backend_choice, llm_model=model_name)
                result = pipeline.analyze(
                    content_or_request=input_text,
                    input_type=input_type,
                    title=title_override,
                    top_k_evidence=top_k,
                )
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.exception("Analysis error")
                return

        # Render Phase 6 Hybrid Results
        render_results(result)


def render_results(result: dict):
    """Renders the comprehensive multi-stage analysis output."""
    st.markdown("---")

    # Metadata bar
    cols = st.columns(4)
    cols[0].metric("Detected Input Type", result.get("input_type", "unknown").upper())
    cols[1].metric("Word Count", f"{result.get('word_count', 0)} words")
    cols[2].metric("Statistical Verdict", result.get("verdict", "Unknown"))
    evidence_data = result.get("evidence", {})
    cols[3].metric("Evidence Found", f"{evidence_data.get('total_retrieved', 0)} passages")

    # 1. Main Hero Verdict Banner
    render_verdict_hero(
        final_verdict=result.get("final_verdict", "UNVERIFIABLE"),
        calibrated_confidence=result.get("calibrated_confidence", 0.5),
        uncertainty_level=result.get("uncertainty_level", "medium"),
        should_flag_review=result.get("should_flag_for_review", False),
        consensus_synthesis=result.get("consensus_synthesis", "Assessment complete."),
    )

    # 2. 3-Way Model Consensus Matrix
    stat_data = result.get("statistical", {})
    verif_data = result.get("verification", {})
    contra_analysis = verif_data.get("contradiction_analysis", {})
    evidence_status = verif_data.get("evidence_status", "absent")

    render_model_consensus_matrix(
        statistical_verdict=stat_data.get("verdict", result.get("verdict", "Unknown")),
        statistical_conf=stat_data.get("confidence", result.get("confidence", 0.5)),
        evidence_count=evidence_data.get("total_retrieved", 0),
        evidence_status=evidence_status,
        llm_verdict=verif_data.get("final_verdict", "UNVERIFIABLE"),
        consensus_status=result.get("consensus_status", "PARTIAL_ALIGNMENT"),
    )

    # 3. Confidence & Uncertainty Gauge
    render_confidence_meter(
        confidence=result.get("calibrated_confidence", result.get("confidence", 0.5)),
        uncertainty_level=result.get("uncertainty_level", "medium"),
    )

    # 4. Evidence Retrieval Explorer
    st.markdown("### 📚 Authoritative Evidence Passages")
    all_passages = evidence_data.get("all_passages", [])
    render_evidence_cards(all_passages)

    # 5. LLM Structured Reasoning & Citations
    st.markdown("### 🧠 Fact-Verification Reasoning")
    render_reasoning_and_limitations(verif_data)

    # 6. Full Audit Report & JSON Export
    with st.expander("📄 Audit Trail & Raw JSON Response"):
        st.json(result)
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="💾 Download JSON Audit Report",
            data=json_str,
            file_name="veritas_verification_report.json",
            mime="application/json",
        )


def main():
    """Application entry point."""
    run_dashboard()


if __name__ == "__main__":
    main()
