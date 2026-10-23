# -*- coding: utf-8 -*-
"""
UI Components and Design System for Veritas AI Fake News Detection Dashboard.
Implements rich aesthetics, dark mode glassmorphism, responsive cards, and glowing badges.
"""

from typing import Any, Dict, List
import streamlit as st


def inject_custom_styles():
    """Injects modern, premium CSS styling with Google Fonts, glassmorphism, and responsive layouts."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Force dark background on the entire Streamlit app container */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0b0f19 45%, #030712 100%) !important;
            color: #f8fafc !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Glowing sidebar expand button when collapsed */
        [data-testid="collapsedControl"] {
            color: #ffffff !important;
            background: rgba(99, 102, 241, 0.25) !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
            border-radius: 8px !important;
            margin: 0.5rem !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
        }

        [data-testid="collapsedControl"]:hover {
            background: rgba(99, 102, 241, 0.45) !important;
            box-shadow: 0 0 25px rgba(99, 102, 241, 0.7) !important;
        }

        /* Headers and text colors */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
        }
        p, span, label, div {
            color: #cbd5e1;
        }

        /* Action buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.75rem 2rem !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 28px rgba(99, 102, 241, 0.6) !important;
        }

        /* Preset sample buttons in horizontal row */
        div[data-testid="stHorizontalBlock"] button {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #cbd5e1 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stHorizontalBlock"] button:hover {
            border-color: #818cf8 !important;
            color: #ffffff !important;
            background: rgba(99, 102, 241, 0.25) !important;
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2) !important;
        }

        /* Input fields and text areas */
        div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
            background-color: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
        }

        div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
        }

        input, textarea {
            color: #f8fafc !important;
        }

        /* Tab styling */
        button[data-baseweb="tab"] {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #a5b4fc !important;
            border-bottom-color: #6366f1 !important;
        }

        /* Glassmorphic main cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 14px 40px -10px rgba(99, 102, 241, 0.15);
        }

        /* Hero verdict styles */
        .verdict-hero {
            padding: 2.25rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .verdict-hero.verdict-false {
            background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(239, 68, 68, 0.5);
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.25);
        }

        .verdict-hero.verdict-true {
            background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(16, 185, 129, 0.5);
            box-shadow: 0 0 50px rgba(16, 185, 129, 0.25);
        }

        .verdict-hero.verdict-unverifiable {
            background: radial-gradient(circle at 50% 0%, rgba(100, 116, 139, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(100, 116, 139, 0.5);
            box-shadow: 0 0 50px rgba(100, 116, 139, 0.2);
        }

        .verdict-hero.verdict-needs_review {
            background: radial-gradient(circle at 50% 0%, rgba(245, 158, 11, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(245, 158, 11, 0.5);
            box-shadow: 0 0 50px rgba(245, 158, 11, 0.25);
        }

        .verdict-title {
            font-size: 3.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0.5rem 0;
            line-height: 1.1;
        }

        .pill-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .pill-supports {
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .pill-contradicts {
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .pill-neutral {
            background: rgba(100, 116, 139, 0.18);
            color: #cbd5e1;
            border: 1px solid rgba(100, 116, 139, 0.35);
        }

        .evidence-card {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 14px;
            padding: 1.35rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.2s ease;
        }

        .evidence-card:hover {
            border-color: rgba(99, 102, 241, 0.45);
            background: rgba(15, 23, 42, 0.95);
            transform: translateY(-1px);
        }

        .source-link {
            color: #818cf8;
            text-decoration: none;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .source-link:hover {
            color: #a5b4fc;
            text-decoration: underline;
        }

        /* Mono code citation block */
        .citation-tag {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.82rem;
            border: 1px solid rgba(99, 102, 241, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renders the dashboard top hero banner."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 1rem 1.25rem; margin-bottom: 1.25rem;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35); padding: 0.35rem 1rem; border-radius: 9999px; margin-bottom: 0.9rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #818cf8; box-shadow: 0 0 10px #818cf8;"></span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #c7d2fe; letter-spacing: 0.08em; text-transform: uppercase;">
                    Veritas AI • Evidence-Grounded Verification Engine
                </span>
            </div>
            <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 15%, #a5b4fc 60%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.25rem 0 0.75rem; letter-spacing: -0.02em;">
                Hybrid Fake News & Fact Verification
            </h1>
            <p style="color: #cbd5e1; font-size: 1.1rem; max-width: 780px; margin: 0 auto; line-height: 1.6;">
                Synthesizing <strong style="color: #ffffff;">Statistical TF-IDF Modeling</strong>, <strong style="color: #ffffff;">FAISS Semantic Evidence Retrieval</strong>, 
                and <strong style="color: #ffffff;">Open-Weight LLM Fact Verification</strong> with robust uncertainty bounds and human-in-the-loop review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_hero(
    final_verdict: str,
    calibrated_confidence: float,
    uncertainty_level: str,
    should_flag_review: bool,
    consensus_synthesis: str,
):
    """Renders the main animated verdict hero card."""
    v_clean = final_verdict.upper().replace(" ", "_")
    verdict_class = f"verdict-{v_clean.lower()}"

    verdict_colors = {
        "FALSE": "#ef4444",
        "FAKE": "#ef4444",
        "TRUE": "#10b981",
        "REAL": "#10b981",
        "UNVERIFIABLE": "#94a3b8",
        "NEEDS_REVIEW": "#f59e0b",
    }
    color = verdict_colors.get(v_clean, "#6366f1")

    icons = {
        "FALSE": "&#10060;",
        "FAKE": "&#10060;",
        "TRUE": "&#9989;",
        "REAL": "&#9989;",
        "UNVERIFIABLE": "&#10067;",
        "NEEDS_REVIEW": "&#9888;&#65039;",
    }
    icon = icons.get(v_clean, "&#128269;")

    descriptions = {
        "FALSE": "Contradicted by authoritative evidence and refuted by factual analysis.",
        "FAKE": "Content shows fabricated or disproven elements contradicted by authoritative records.",
        "TRUE": "Corroborated by trustworthy scientific or governmental evidence repositories.",
        "REAL": "Factual alignment verified across independent authoritative reference documents.",
        "UNVERIFIABLE": "No authoritative evidence found in the verified database to confirm or refute.",
        "NEEDS_REVIEW": "Conflicting signals or elevated uncertainty detected. Manual human review recommended.",
    }
    desc = descriptions.get(v_clean, "Analysis completed.")
    conf_display = f"{calibrated_confidence * 100:.1f}%"

    st.markdown(
        f"""
        <div class="verdict-hero {verdict_class}">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: {color}; font-weight: 700;">
                {icon} Hybrid Synthesis Verdict
            </div>
            <div class="verdict-title" style="color: {color};">
                {final_verdict}
            </div>
            <div style="color: #e2e8f0; font-size: 1.05rem; max-width: 650px; margin: 0 auto 1.25rem; line-height: 1.5;">
                {desc}
            </div>
            <div style="display: inline-flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; background: rgba(15, 23, 42, 0.6); padding: 0.75rem 1.5rem; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="font-size: 0.85rem; color: #cbd5e1;">Confidence: <strong style="color: #ffffff;">{conf_display}</strong></div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Uncertainty: <strong style="color: #ffffff; text-transform: uppercase;">{uncertainty_level}</strong></div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Review Flag: <strong style="color: {'#f59e0b' if should_flag_review else '#10b981'};">{'Triggered' if should_flag_review else 'Clear'}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if should_flag_review:
        st.warning(
            f"⚠️ **Human Review Flagged:** {consensus_synthesis}",
            icon="🚨",
        )
    else:
        st.info(
            f"💡 **Consensus Synthesis:** {consensus_synthesis}",
            icon="ℹ️",
        )


def render_evidence_cards(evidence_passages: List[Dict[str, Any]]):
    """Renders evidence passages with stances, source credibility, and clickable links."""
    if not evidence_passages:
        st.markdown(
            """
            <div style="padding: 1.5rem; background: rgba(30, 41, 59, 0.4); border-radius: 12px; text-align: center; color: #94a3b8; border: 1px dashed rgba(255, 255, 255, 0.1);">
                No matching reference evidence found in authoritative vector corpus.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for idx, item in enumerate(evidence_passages):
        passage = item.get("passage", {}) if isinstance(item, dict) else {}
        if not passage and isinstance(item, dict):
            passage = item  # Fallback if passage flattened

        stance = item.get("stance", "neutral") if isinstance(item, dict) else "neutral"
        if hasattr(stance, "value"):
            stance = stance.value
        stance = str(stance).lower()

        score = item.get("score", item.get("similarity_score", 0.0))
        text = passage.get("text", "")
        title = passage.get("title") or passage.get("source", "Reference Source")
        source = passage.get("source", "Authoritative Repository")
        source_url = passage.get("source_url") or "#"
        passage_id = passage.get("passage_id", f"PASSAGE-{idx+1}")
        tier = passage.get("metadata", {}).get("credibility_tier", 1) if isinstance(passage.get("metadata"), dict) else 1

        stance_pill_class = f"pill-{stance}" if stance in ("supports", "contradicts") else "pill-neutral"
        stance_label = stance.upper()
        score_display = f"{score * 100:.1f}%"

        st.markdown(
            f"""
            <div class="evidence-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span class="pill-badge {stance_pill_class}">{stance_label}</span>
                        <span class="citation-tag">{passage_id}</span>
                        <span style="font-size: 0.8rem; background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.25);">
                            Tier {tier} Verified Source
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8;">
                        Similarity: {score_display}
                    </div>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0; font-style: italic; margin-bottom: 0.8rem; border-left: 3px solid rgba(99, 102, 241, 0.4); padding-left: 0.75rem;">
                    "{text}"
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 0.5rem;">
                    <div>
                        <strong>Document:</strong> {title} ({source})
                    </div>
                    <div>
                        <a href="{source_url}" target="_blank" class="source-link">
                            &#128279; View Official Source &rarr;
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_reasoning_and_limitations(verification: Dict[str, Any]):
    """Renders structured LLM reasoning, citations, and caveats."""
    if not verification:
        return

    reasoning = verification.get("reasoning", "No detailed reasoning provided.")
    citations = verification.get("evidence_citations", [])
    limitations = verification.get("limitations", "None stated.")
    model_name = verification.get("model_name", "flan-t5-small")

    with st.expander("Fact-Verification Reasoning & Citations", expanded=True):
        st.markdown(f"**Verification Model:** `{model_name}`")
        st.markdown(f"**Step-by-Step Fact-Checking Reasoning:**\n\n{reasoning}")

        if citations:
            citations_html = " ".join([f'<span class="citation-tag">{c}</span>' for c in citations])
            st.markdown(f"**Cited Evidence Passages:** {citations_html}", unsafe_allow_html=True)
        else:
            st.markdown("**Cited Evidence Passages:** None cited.")

        st.markdown(f"**Recognized Limitations & Caveats:**\n\n_{limitations}_")
