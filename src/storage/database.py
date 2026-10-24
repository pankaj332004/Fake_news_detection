# -*- coding: utf-8 -*-
"""
Database configuration and initialization module for Veritas AI.
Implements SQLite storage for users, fact-checks, review queue, sources, news feed, and audit logs.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "veritas.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with dictionary cursor."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    """Returns SHA-256 hash of password with a static salt for local demo security."""
    salt = "veritas_ai_salt_2026"
    return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()


def init_db():
    """Initializes SQLite tables and seeds initial data if database is new."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        avatar_url TEXT,
        provider TEXT DEFAULT 'local',
        created_at TEXT NOT NULL
    )
    """)

    # 2. Fact Check History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fact_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TEXT NOT NULL,
        input_type TEXT NOT NULL,
        title TEXT,
        query_text TEXT NOT NULL,
        verdict TEXT NOT NULL,
        confidence REAL NOT NULL,
        uncertainty TEXT,
        flagged_review INTEGER DEFAULT 0,
        consensus_status TEXT,
        consensus_synthesis TEXT,
        source_url TEXT,
        result_json TEXT NOT NULL,
        bookmarked INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    # 3. Review Queue Table (Admin moderation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        check_id INTEGER,
        claim TEXT NOT NULL,
        initial_verdict TEXT NOT NULL,
        flagged_reason TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        notes TEXT,
        moderator TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        FOREIGN KEY (check_id) REFERENCES fact_checks (id)
    )
    """)

    # 4. Source Directory Database
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        domain TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        credibility_tier TEXT NOT NULL,
        ownership TEXT,
        country TEXT,
        founded_year INTEGER,
        contact_info TEXT,
        history_summary TEXT,
        editorial_practices TEXT,
        flag_reasons TEXT,
        verified_evidence_count INTEGER DEFAULT 0
    )
    """)

    # 5. News Feed Articles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_feed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        category TEXT NOT NULL,
        source TEXT NOT NULL,
        source_url TEXT NOT NULL,
        published_date TEXT NOT NULL,
        trending_score INTEGER DEFAULT 0,
        is_featured INTEGER DEFAULT 0,
        preliminary_verdict TEXT DEFAULT 'UNVERIFIED',
        image_url TEXT
    )
    """)

    # 6. Audit Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_email TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        ip_address TEXT DEFAULT '127.0.0.1'
    )
    """)

    conn.commit()

    # Seed Initial Content if tables are empty
    _seed_initial_data(conn)
    conn.close()


def _seed_initial_data(conn: sqlite3.Connection):
    """Populates database with initial users, sources, news feed, and review items."""
    cursor = conn.cursor()

    # 1. Seed Default Admin & Regular Users
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        now = datetime.now(timezone.utc).isoformat()
        cursor.executemany("""
        INSERT INTO users (email, password_hash, name, role, avatar_url, provider, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                "admin@veritas.ai",
                hash_password("admin123"),
                "Veritas Lead Moderator",
                "admin",
                "🛡️",
                "local",
                now,
            ),
            (
                "researcher@veritas.ai",
                hash_password("veritas123"),
                "Sarah Jenkins (Journalist)",
                "user",
                "🔬",
                "local",
                now,
            ),
        ])

    # 2. Seed Authoritative and Questionable Sources
    cursor.execute("SELECT COUNT(*) FROM sources")
    if cursor.fetchone()[0] == 0:
        seed_sources = [
            (
                "who.int",
                "World Health Organization (WHO)",
                "Tier 1: Authoritative Institutional",
                "United Nations Specialized Health Agency",
                "International (Geneva, Switzerland)",
                1948,
                "mediainquiries@who.int",
                "Primary international public health coordinating body with strict peer-reviewed epidemiological research standards.",
                "Systematic literature synthesis, multi-member scientific advisory panels, declared conflict of interest disclosures.",
                "None",
                420
            ),
            (
                "cdc.gov",
                "Centers for Disease Control and Prevention",
                "Tier 1: Authoritative Institutional",
                "U.S. Department of Health and Human Services",
                "United States",
                1946,
                "cdcinfo@cdc.gov",
                "Leading national public health institute in the US conducting biomedical and epidemiological surveillance.",
                "Rigorous scientific review, public data release, adherence to federal peer-review bulletins.",
                "None",
                380
            ),
            (
                "nasa.gov",
                "National Aeronautics and Space Administration",
                "Tier 1: Authoritative Institutional",
                "U.S. Federal Government Executive Agency",
                "United States",
                1958,
                "public-inquiries@hq.nasa.gov",
                "Pioneering space exploration, aeronautics research, and planetary climate observation center.",
                "Open telemetry datasets, peer-reviewed astrophysics publications, raw sensor archival.",
                "None",
                510
            ),
            (
                "reuters.com",
                "Reuters News & Fact Check",
                "Tier 2: Major Accredited Fact-Checking",
                "Thomson Reuters Corporation",
                "United Kingdom / Global",
                1851,
                "contact@reuters.com",
                "Global international news agency and signatory to the International Fact-Checking Network (IFCN) code of principles.",
                "Strict double-sourcing policy, clear correction archival, explicit non-partisanship declaration.",
                "None",
                650
            ),
            (
                "apnews.com",
                "Associated Press News",
                "Tier 2: Major Accredited Fact-Checking",
                "Not-for-profit news cooperative owned by US newspaper and broadcast members",
                "United States / Global",
                1846,
                "info@ap.org",
                "Historic not-for-profit cooperative news agency providing primary coverage for thousands of outlets worldwide.",
                "AP Stylebook standards, rigorous factual verification, transparent correction history.",
                "None",
                710
            ),
            (
                "theonion.com",
                "The Onion",
                "Satire: Humor / Parody Publication",
                "Global Tetrahedron (Privately Owned)",
                "United States",
                1988,
                "public@theonion.com",
                "Widely recognized satirical digital media company publishing humorous and fictitious news parody.",
                "Intentionally fabricated articles written purely for social commentary, satire, and comedy.",
                "Fictional content frequently reshared out of context by users mistaking satire for literal reporting.",
                12
            ),
            (
                "naturalnews.com",
                "Natural News",
                "Questionable: Pseudoscience & Conspiracy",
                "Truth Publishing International / Mike Adams",
                "United States",
                2003,
                "support@naturalnews.com",
                "Commercial website promoting alternative wellness products alongside unsupported conspiracy theories and medical misinformation.",
                "Lacks independent scientific review; frequently cites non-indexed blogs and unverified testimonials.",
                "Repeatedly flagged by independent fact-checkers for false claims regarding vaccine efficacy and health remedies.",
                8
            ),
            (
                "bbc.com",
                "BBC News",
                "Tier 2: Major Accredited Fact-Checking",
                "Public Corporation / Royal Charter",
                "United Kingdom",
                1922,
                "press.office@bbc.co.uk",
                "Public service broadcaster with global bureau presence and dedicated Reality Check verification unit.",
                "Impartiality guidelines regulated by Ofcom, editorial accuracy standards, transparent public accountability.",
                "None",
                590
            ),
            (
                "nature.com",
                "Nature Publishing Group",
                "Tier 1: Authoritative Institutional",
                "Springer Nature",
                "United Kingdom / Global",
                1869,
                "nature@nature.com",
                "Top-tier international multidisciplinary weekly scientific journal with highest global citation impact.",
                "Double-blind and transparent peer review, data availability mandates, strict replicability checks.",
                "None",
                890
            ),
        ]
        cursor.executemany("""
        INSERT INTO sources (domain, name, credibility_tier, ownership, country, founded_year, 
                            contact_info, history_summary, editorial_practices, flag_reasons, verified_evidence_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_sources)

    # 3. Seed News Feed with Featured & Trending Stories
    cursor.execute("SELECT COUNT(*) FROM news_feed")
    if cursor.fetchone()[0] == 0:
        seed_news = [
            (
                "WHO Clarifies Guidance on Novel Respiratory Variant Monitoring",
                "The World Health Organization confirmed ongoing genomic surveillance for seasonal influenza and respiratory pathogens, emphasizing that routine immunizations remain effective.",
                "Health",
                "World Health Organization",
                "https://www.who.int/news/item/respiratory-update-2026",
                "2026-10-23",
                98,
                1,  # Featured
                "SUPPORTED",
                "🏥"
            ),
            (
                "Viral Social Media Posts Claim New Satellite Network Controls Global Rainfall",
                "Widespread TikTok and Facebook videos allege low-orbit communications satellites are altering regional monsoons through micro-radiation frequencies. Climate physicists refute the physics.",
                "Environment",
                "Associated Press Fact Check",
                "https://apnews.com/article/weather-modification-satellite-claims-debunked",
                "2026-10-22",
                92,
                0,
                "FALSE",
                "🛰️"
            ),
            (
                "Quantum Computing Breakthrough: Researchers Demonstrate Error-Corrected Logical Qubits",
                "Physicists have sustained stable coherence across 100 fault-tolerant logical qubits, demonstrating key benchmarks for scalable algorithmic cryptography.",
                "Technology",
                "Nature Quantum Information",
                "https://www.nature.com/articles/s41534-quantum-logical-2026",
                "2026-10-21",
                86,
                0,
                "SUPPORTED",
                "💻"
            ),
            (
                "Election Integrity Officials Issue Alert Over Synthetic Voice Robocalls",
                "State election commissions warn voters against automated phone messages using synthesized candidate voices delivering false polling station closure notices.",
                "Politics",
                "Reuters Integrity Bureau",
                "https://www.reuters.com/world/us/election-robocall-deepfakes-alert-2026",
                "2026-10-20",
                79,
                0,
                "MISLEADING",
                "🗳️"
            ),
            (
                "NASA Mars Perseverance Rover Discovers Organic Carbon Assemblages in Jezero Crater",
                "Spectroscopic data from the SHERLOC instrument on Mars Perseverance revealed recurring organic carbon compounds preserved in mudstone sediments.",
                "Science",
                "NASA Jet Propulsion Laboratory",
                "https://www.nasa.gov/news/perseverance-jezero-sediments-2026",
                "2026-10-19",
                74,
                0,
                "SUPPORTED",
                "🪐"
            ),
            (
                "Online Retailers Flooded with Counterfeit 'Miracle Toxin-Cleansing' Teas",
                "Consumer safety groups discovered high concentrations of unregulated prescription laxatives disguised as organic herbal detox mixtures.",
                "Health",
                "FDA Safety Alerts",
                "https://www.fda.gov/safety/recalls-market-withdrawals",
                "2026-10-18",
                67,
                0,
                "FALSE",
                "🍵"
            ),
        ]
        cursor.executemany("""
        INSERT INTO news_feed (title, summary, category, source, source_url, published_date, 
                               trending_score, is_featured, preliminary_verdict, image_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_news)

    # 4. Seed Review Queue for Admin Moderation
    cursor.execute("SELECT COUNT(*) FROM review_queue")
    if cursor.fetchone()[0] == 0:
        seed_queue = [
            (
                1,
                "Drinking distilled vinegar cures chronic respiratory viral infections in 48 hours.",
                "NEEDS_REVIEW",
                "Low Confidence / Potential Harmful Medical Claim",
                "pending",
                "Awaiting panel review; zero biomedical literature supports this therapeutic mechanism.",
                "Unassigned",
                "2026-10-23T14:20:00Z"
            ),
            (
                2,
                "Federal agency quietly mandated digital currencies to replace all paper banknotes by November 2026.",
                "NEEDS_REVIEW",
                "Contradictory Bureau Statements",
                "pending",
                "User requested priority re-check. Treasury department reports pilot studies only, no mandate.",
                "Unassigned",
                "2026-10-22T09:15:00Z"
            ),
            (
                3,
                "New solar panel coating doubles atmospheric oxygen generation in urban centers.",
                "FALSE",
                "User Dispute Filed",
                "pending",
                "Submitter claims novel photocatalysis patent exists; verifying patent office documentation.",
                "Unassigned",
                "2026-10-21T18:40:00Z"
            ),
        ]
        cursor.executemany("""
        INSERT INTO review_queue (check_id, claim, initial_verdict, flagged_reason, status, notes, moderator, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_queue)

    # 5. Seed Audit Logs
    cursor.execute("SELECT COUNT(*) FROM audit_logs")
    if cursor.fetchone()[0] == 0:
        seed_logs = [
            (datetime.now(timezone.utc).isoformat(), "system", "SYSTEM_INIT", "Database initialized with baseline seeds", "127.0.0.1"),
            (datetime.now(timezone.utc).isoformat(), "admin@veritas.ai", "MODEL_POLICY_UPDATE", "Decision policy sensitivity set to Standard", "127.0.0.1"),
            (datetime.now(timezone.utc).isoformat(), "researcher@veritas.ai", "VERIFICATION_RUN", "Verified claim: Mars Perseverance Rover findings", "127.0.0.1"),
        ]
        cursor.executemany("""
        INSERT INTO audit_logs (timestamp, user_email, action, details, ip_address)
        VALUES (?, ?, ?, ?, ?)
        """, seed_logs)

    conn.commit()


# Run initialization on import
init_db()
