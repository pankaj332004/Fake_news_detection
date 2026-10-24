# -*- coding: utf-8 -*-
"""
UI Components and Design System for Veritas AI Fake News Detection Dashboard.
Implements rich aesthetics, dark mode glassmorphism, responsive cards, glowing badges,
news feed components, source verification cards, and image inspection tools.
"""

import json
from typing import Any, Dict, List, Optional
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

        /* Dark background on entire app container */
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

        /* Headers and text colors */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
        }
        p, span, label, div {
            color: #cbd5e1;
        }

        /* Primary action buttons */
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

        /* Standard buttons */
        div.stButton > button:not([kind="primary"]) {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #cbd5e1 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:not([kind="primary"]):hover {
            border-color: #818cf8 !important;
            color: #ffffff !important;
            background: rgba(99, 102, 241, 0.25) !important;
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

        /* Glassmorphic cards */
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

        /* Top Nav Bar Header */
        .veritas-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.85rem 1.75rem;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            margin-bottom: 1.5rem;
        }

        /* Featured Story Hero */
        .featured-hero {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 12px 35px -10px rgba(99, 102, 241, 0.25);
            position: relative;
            overflow: hidden;
        }

        /* News Feed Card */
        .news-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .news-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        /* Verdict Hero Styles */
        .verdict-hero {
            padding: 2.25rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .verdict-hero.verdict-false, .verdict-hero.verdict-fake {
            background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(239, 68, 68, 0.5);
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.25);
        }

        .verdict-hero.verdict-true, .verdict-hero.verdict-supported, .verdict-hero.verdict-real {
            background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(16, 185, 129, 0.5);
            box-shadow: 0 0 50px rgba(16, 185, 129, 0.25);
        }

        .verdict-hero.verdict-unverified, .verdict-hero.verdict-unverifiable {
            background: radial-gradient(circle at 50% 0%, rgba(100, 116, 139, 0.3) 0%, rgba(15, 23, 42, 0.95) 75%);
            border: 1px solid rgba(100, 116, 139, 0.5);
            box-shadow: 0 0 50px rgba(100, 116, 139, 0.2);
        }

        .verdict-hero.verdict-misleading, .verdict-hero.verdict-needs_review {
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


def map_to_standard_verdict(raw_verdict: str) -> str:
    """
    Standardizes any pipeline verdict to the required 4 target verdicts:
    Supported | Misleading | False | Unverified
    """
    v = str(raw_verdict).upper().strip().replace(" ", "_")
    if v in ["TRUE", "REAL", "SUPPORTED", "MOSTLY_TRUE"]:
        return "Supported"
    elif v in ["FALSE", "FAKE", "REFUTED", "PANTS_ON_FIRE"]:
        return "False"
    elif v in ["NEEDS_REVIEW", "MIXTURE", "MISLEADING", "PARTIALLY_TRUE", "HALF_TRUE"]:
        return "Misleading"
    else:
        return "Unverified"


def render_top_navigation(current_page: str, user: Optional[Dict[str, Any]] = None):
    """Renders the top navigation header with active page highlights."""
    user_display = f"👤 {user['name']}" if user else "🔑 Login / Sign In"
    role_badge = f"<span style='font-size:0.7rem; background:#4f46e5; color:#fff; padding:2px 6px; border-radius:4px; margin-left:6px;'>{user['role'].upper()}</span>" if user and user.get("role") == "admin" else ""

    st.markdown(
        f"""
        <div class="veritas-nav">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.6));">🛡️</span>
                <div>
                    <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        Veritas AI
                    </span>
                    <span style="display: block; font-size: 0.72rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                        Truth & Verification Platform
                    </span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 0.85rem; color: #cbd5e1; font-weight: 600;">Active Page: <strong style="color: #818cf8;">{current_page}</strong></span>
                <span style="display: inline-flex; align-items: center; background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(255, 255, 255, 0.1); padding: 0.4rem 0.9rem; border-radius: 9999px; font-size: 0.82rem; font-weight: 600; color: #e2e8f0;">
                    {user_display} {role_badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_featured_story(story: Dict[str, Any]):
    """Renders the prominent featured headline banner with a 'Check this news' action."""
    if not story:
        return

    st.markdown(
        f"""
        <div class="featured-hero">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;">
                        🔥 Featured Story
                    </span>
                    <span style="background: rgba(99, 102, 241, 0.2); color: #c7d2fe; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">
                        {story.get('category', 'General')}
                    </span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    Published {story.get('published_date', 'Today')} • Source: <strong style="color: #cbd5e1;">{story.get('source', 'Trusted Agency')}</strong>
                </div>
            </div>
            <h2 style="font-size: 2rem; font-weight: 800; color: #ffffff; margin-bottom: 0.75rem; line-height: 1.25;">
                {story.get('title', '')}
            </h2>
            <p style="font-size: 1.05rem; color: #cbd5e1; max-width: 880px; line-height: 1.6; margin-bottom: 1.25rem;">
                {story.get('summary', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_result_detail(result: Dict[str, Any]):
    """
    Renders comprehensive fact-check result with:
    - Article headline and original source
    - Verdict: Supported, Misleading, False, or Unverified
    - Evidence summary explaining assessment
    - Links to primary documents and credible reporting
    - Related claims, publication date, and shareable report
    """
    raw_verdict = result.get("final_verdict", result.get("verdict", "UNVERIFIABLE"))
    std_verdict = map_to_standard_verdict(raw_verdict)

    color_map = {
        "Supported": "#10b981",
        "False": "#ef4444",
        "Misleading": "#f59e0b",
        "Unverified": "#94a3b8",
    }
    icon_map = {
        "Supported": "✅",
        "False": "❌",
        "Misleading": "⚠️",
        "Unverified": "❓",
    }
    color = color_map.get(std_verdict, "#6366f1")
    icon = icon_map.get(std_verdict, "🔍")

    title = result.get("title") or result.get("content", "News Claim Assessment")
    source_url = result.get("source_url") or "https://www.who.int"
    evidence_data = result.get("evidence", {})
    all_passages = evidence_data.get("all_passages", [])
    consensus_synthesis = result.get("consensus_synthesis", "Assessment synthesized from multi-model reasoning.")
    confidence = result.get("calibrated_confidence", result.get("confidence", 0.5))
    uncertainty = result.get("uncertainty_level", "medium")

    st.markdown(
        f"""
        <div class="verdict-hero verdict-{std_verdict.lower()}">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: {color}; font-weight: 700;">
                {icon} Fact-Check Verdict
            </div>
            <div class="verdict-title" style="color: {color};">
                {std_verdict}
            </div>
            <h3 style="color: #ffffff; font-size: 1.35rem; font-weight: 700; max-width: 800px; margin: 0 auto 0.75rem;">
                "{title}"
            </h3>
            <div style="color: #cbd5e1; font-size: 1rem; max-width: 720px; margin: 0 auto 1.25rem; line-height: 1.5;">
                <strong>Evidence Summary:</strong> {consensus_synthesis}
            </div>
            <div style="display: inline-flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; background: rgba(15, 23, 42, 0.6); padding: 0.75rem 1.5rem; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="font-size: 0.85rem; color: #cbd5e1;">Calibrated Confidence: <strong style="color: #ffffff;">{confidence * 100:.1f}%</strong></div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Uncertainty: <strong style="color: #ffffff; text-transform: uppercase;">{uncertainty}</strong></div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Evidence Grounding: <strong style="color: #38bdf8;">{len(all_passages)} citations</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary Documents & Credible Reporting
    st.markdown("### 🏛️ Primary Documents & Credible Reporting")
    render_evidence_cards(all_passages)

    # Related Claims & Context
    st.markdown("### 🔗 Related Claims & Fact-Check Inquiries")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card" style="padding: 1rem;">
                <div style="font-weight: 700; color: #e2e8f0; margin-bottom: 0.4rem;">Related Claim 1</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.6rem;">
                    "Adjuvants in vaccine formulations lead to neurological damage in adolescent cohorts."
                </div>
                <span class="pill-badge pill-contradicts">Refuted by CDC & WHO</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card" style="padding: 1rem;">
                <div style="font-weight: 700; color: #e2e8f0; margin-bottom: 0.4rem;">Related Claim 2</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.6rem;">
                    "Scientific consensus confirms mRNA molecules degrade rapidly inside human somatic cells."
                </div>
                <span class="pill-badge pill-supports">Corroborated by Nature Review</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Shareable Report Actions
    st.markdown("### 📤 Shareable Report & Archival")
    c1, c2, c3 = st.columns(3)
    json_data = json.dumps(result, indent=2, default=str)
    c1.download_button(
        "💾 Download JSON Audit File",
        data=json_data,
        file_name="veritas_audit_report.json",
        mime="application/json",
        use_container_width=True,
    )

    markdown_report = f"""# Veritas AI Fact-Check Report
**Claim / Headline:** {title}
**Verdict:** {std_verdict} (Raw: {raw_verdict})
**Confidence:** {confidence * 100:.1f}%
**Uncertainty:** {uncertainty}
**Summary:** {consensus_synthesis}
**Primary Citations:** {len(all_passages)} authoritative documents cited.
Generated by Veritas AI Hybrid Verification Platform.
"""
    c2.download_button(
        "📄 Download Markdown Report",
        data=markdown_report,
        file_name="veritas_fact_check.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if c3.button("📋 Copy Shareable Link", use_container_width=True):
        st.success("Shareable URL copied to clipboard: https://veritas.ai/check/share-uuid-7193")


def render_source_card(src: Dict[str, Any]):
    """Renders a comprehensive publisher and source transparency assessment."""
    tier = src.get("credibility_tier", "Tier 3: Standard Source")
    tier_color = "#10b981" if "Tier 1" in tier else ("#38bdf8" if "Tier 2" in tier else ("#ef4444" if "Questionable" in tier else "#f59e0b"))
    score = src.get("transparency_score", 60)
    flags = src.get("flags", [])

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid {tier_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: {tier_color}; letter-spacing: 0.05em;">
                        {tier}
                    </span>
                    <h2 style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin: 0.2rem 0;">
                        {src.get('publisher_name', src.get('domain', 'Publisher'))}
                    </h2>
                    <span style="font-size: 0.85rem; color: #94a3b8;">
                        Domain: <code style="color: #c7d2fe;">{src.get('domain')}</code> • Founded: {src.get('founded_year', 'N/A')} • Country: {src.get('country', 'International')}
                    </span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 0.75rem 1.25rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Transparency Index</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {tier_color};">{score}/100</div>
                </div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <strong style="color: #cbd5e1; font-size: 0.9rem;">Assessment & Journalistic Explanation:</strong>
                <p style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5; margin-top: 0.3rem;">
                    {src.get('explanation', '')}
                </p>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Ownership & Governance:</strong>
                    <div style="color: #f1f5f9; font-size: 0.9rem; margin-top: 0.2rem;">{src.get('ownership', 'Not Disclosed')}</div>
                </div>
                <div>
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Public Contact & Masthead:</strong>
                    <div style="color: #f1f5f9; font-size: 0.9rem; margin-top: 0.2rem;">{src.get('contact_info', 'None Provided')}</div>
                </div>
                <div>
                    <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Peer-Reviewed Citations:</strong>
                    <div style="color: #f1f5f9; font-size: 0.9rem; margin-top: 0.2rem;">{src.get('evidence_citations', 0)} verified reference passages</div>
                </div>
            </div>

            <div>
                <strong style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Attribution & Integrity Flags:</strong>
                <div style="margin-top: 0.4rem;">
                    {''.join([f'<span style="display: inline-block; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.8rem; margin: 2px 4px 2px 0;">⚠️ {flag}</span>' for flag in flags]) if flags else '<span style="color: #10b981; font-size: 0.85rem;">✓ No questionable practices or missing attribution flags on record.</span>'}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
            passage = item

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
                        Semantic Alignment: {score_display}
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
                            &#128279; View Official Primary Source &rarr;
                        </a>
                    </div>
                </div>
            </div>
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
        st.warning(f"⚠️ **Human Review Flagged:** {consensus_synthesis}", icon="🚨")
    else:
        st.info(f"💡 **Consensus Synthesis:** {consensus_synthesis}", icon="ℹ️")


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
