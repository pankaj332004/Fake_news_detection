# -*- coding: utf-8 -*-
"""
History and Persistence Repository for Veritas AI.
Provides clean CRUD access to user history, bookmarked articles, source directory,
news feed, review queue, and audit logs.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.storage.database import get_db_connection, hash_password

logger = logging.getLogger(__name__)


class HistoryRepository:
    """Data access repository for Veritas AI storage layer."""

    @staticmethod
    def save_fact_check(
        user_id: Optional[int],
        input_type: str,
        title: Optional[str],
        query_text: str,
        verdict: str,
        confidence: float,
        uncertainty: str,
        flagged_review: bool,
        consensus_status: str,
        consensus_synthesis: str,
        source_url: Optional[str],
        result_dict: Dict[str, Any],
    ) -> int:
        """Saves a completed fact check run to SQLite."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        json_payload = json.dumps(result_dict, default=str)

        cursor.execute("""
        INSERT INTO fact_checks (
            user_id, timestamp, input_type, title, query_text, verdict, confidence,
            uncertainty, flagged_review, consensus_status, consensus_synthesis,
            source_url, result_json, bookmarked
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            user_id,
            now,
            input_type,
            title or (query_text[:60] + "..." if len(query_text) > 60 else query_text),
            query_text,
            verdict,
            float(confidence),
            uncertainty,
            1 if flagged_review else 0,
            consensus_status,
            consensus_synthesis,
            source_url,
            json_payload
        ))

        check_id = cursor.lastrowid

        # If flagged for review or high uncertainty, auto-enqueue in review queue
        if flagged_review or verdict in ["NEEDS_REVIEW", "UNVERIFIABLE"]:
            cursor.execute("""
            INSERT INTO review_queue (
                check_id, claim, initial_verdict, flagged_reason, status, notes, moderator, created_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, 'Unassigned', ?)
            """, (
                check_id,
                query_text[:200],
                verdict,
                f"Consensus: {consensus_status} • Uncertainty: {uncertainty}",
                consensus_synthesis[:200] if consensus_synthesis else "Automated pipeline review flag",
                now
            ))

        # Add audit log
        cursor.execute("""
        INSERT INTO audit_logs (timestamp, user_email, action, details)
        VALUES (?, ?, 'FACT_CHECK_EXECUTED', ?)
        """, (
            now,
            f"user_{user_id}" if user_id else "anonymous",
            f"Analyzed {input_type}: {verdict} ({confidence:.2f})"
        ))

        conn.commit()
        conn.close()
        return check_id

    @staticmethod
    def get_user_history(
        user_id: Optional[int] = None,
        search_query: Optional[str] = None,
        verdict_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves history records with optional search, verdict, and input_type filters."""
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "SELECT * FROM fact_checks WHERE 1=1"
        params: List[Any] = []

        if user_id is not None:
            sql += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)

        if search_query and search_query.strip():
            sql += " AND (query_text LIKE ? OR title LIKE ?)"
            wildcard = f"%{search_query.strip()}%"
            params.extend([wildcard, wildcard])

        if verdict_filter and verdict_filter != "All":
            sql += " AND verdict = ?"
            params.append(verdict_filter)

        if type_filter and type_filter != "All":
            sql += " AND input_type = ?"
            params.append(type_filter.lower())

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def toggle_bookmark(check_id: int) -> bool:
        """Toggles the bookmarked flag for a specific fact check record."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT bookmarked FROM fact_checks WHERE id = ?", (check_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        new_val = 0 if row["bookmarked"] == 1 else 1
        cursor.execute("UPDATE fact_checks SET bookmarked = ? WHERE id = ?", (new_val, check_id))
        conn.commit()
        conn.close()
        return bool(new_val)

    @staticmethod
    def get_bookmarked_checks(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns all bookmarked checks."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if user_id:
            cursor.execute("SELECT * FROM fact_checks WHERE bookmarked = 1 AND (user_id = ? OR user_id IS NULL) ORDER BY id DESC", (user_id,))
        else:
            cursor.execute("SELECT * FROM fact_checks WHERE bookmarked = 1 ORDER BY id DESC")
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def clear_user_history(user_id: Optional[int] = None) -> int:
        """Clears user history (Privacy Controls feature)."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if user_id:
            cursor.execute("DELETE FROM fact_checks WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("DELETE FROM fact_checks")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # --- News Feed Methods ---
    @staticmethod
    def get_news_feed(category: Optional[str] = None, limit: int = 12) -> List[Dict[str, Any]]:
        """Returns news feed articles, optionally filtered by category."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if category and category != "All":
            cursor.execute("""
            SELECT * FROM news_feed 
            WHERE category = ? 
            ORDER BY is_featured DESC, trending_score DESC, id DESC 
            LIMIT ?
            """, (category, limit))
        else:
            cursor.execute("""
            SELECT * FROM news_feed 
            ORDER BY is_featured DESC, trending_score DESC, id DESC 
            LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def get_featured_story() -> Optional[Dict[str, Any]]:
        """Returns the prominent featured story for the Home news feed."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM news_feed WHERE is_featured = 1 ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # --- Source Verification Methods ---
    @staticmethod
    def search_sources(query: str) -> List[Dict[str, Any]]:
        """Searches sources database by domain, name, or keywords."""
        conn = get_db_connection()
        cursor = conn.cursor()
        wildcard = f"%{query.strip().lower()}%"
        cursor.execute("""
        SELECT * FROM sources 
        WHERE LOWER(domain) LIKE ? OR LOWER(name) LIKE ? OR LOWER(country) LIKE ?
        ORDER BY verified_evidence_count DESC
        """, (wildcard, wildcard, wildcard))
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def get_all_sources() -> List[Dict[str, Any]]:
        """Returns all sources in the registry."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sources ORDER BY verified_evidence_count DESC")
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def add_or_update_source(source_data: Dict[str, Any]) -> bool:
        """Admin method to register or update a news source."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO sources (
            domain, name, credibility_tier, ownership, country, founded_year,
            contact_info, history_summary, editorial_practices, flag_reasons, verified_evidence_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_data["domain"].lower().strip(),
            source_data["name"].strip(),
            source_data.get("credibility_tier", "Tier 3: Standard Source"),
            source_data.get("ownership", "Not Disclosed"),
            source_data.get("country", "Unknown"),
            int(source_data.get("founded_year", 2000)),
            source_data.get("contact_info", "editorial@example.com"),
            source_data.get("history_summary", "Independent publisher."),
            source_data.get("editorial_practices", "Standard editorial review."),
            source_data.get("flag_reasons", "None"),
            int(source_data.get("verified_evidence_count", 0))
        ))
        conn.commit()
        conn.close()
        return True

    # --- Review Queue Methods (Admin) ---
    @staticmethod
    def get_review_queue(status_filter: str = "pending") -> List[Dict[str, Any]]:
        """Returns items in the review queue for moderation."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if status_filter != "all":
            cursor.execute("SELECT * FROM review_queue WHERE status = ? ORDER BY id DESC", (status_filter,))
        else:
            cursor.execute("SELECT * FROM review_queue ORDER BY id DESC")
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    @staticmethod
    def update_review_item(item_id: int, new_status: str, notes: str, moderator: str) -> bool:
        """Updates the status of a review item."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
        UPDATE review_queue 
        SET status = ?, notes = ?, moderator = ?, updated_at = ?
        WHERE id = ?
        """, (new_status, notes, moderator, now, item_id))
        conn.commit()
        conn.close()
        return True

    # --- Admin Analytics & Metrics ---
    @staticmethod
    def get_admin_metrics() -> Dict[str, Any]:
        """Calculates real-time verification metrics and activity volume."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM fact_checks")
        total_checks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'")
        pending_reviews = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE flagged_review = 1")
        flagged_reports = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sources")
        total_sources = cursor.fetchone()[0]

        # Breakdown by verdict
        cursor.execute("SELECT verdict, COUNT(*) as count FROM fact_checks GROUP BY verdict")
        verdict_rows = cursor.fetchall()
        verdicts = {row["verdict"]: row["count"] for row in verdict_rows}

        conn.close()
        return {
            "total_checks": total_checks,
            "pending_reviews": pending_reviews,
            "flagged_reports": flagged_reports,
            "total_users": total_users,
            "total_sources": total_sources,
            "verdict_distribution": verdicts,
        }

    @staticmethod
    def get_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent system and user audit logs."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    # --- Authentication Methods ---
    @staticmethod
    def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticates user via email and hashed password."""
        conn = get_db_connection()
        cursor = conn.cursor()
        pass_hash = hash_password(password)
        cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?) AND password_hash = ?", (email.strip(), pass_hash))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def register_user(email: str, password: str, name: str, role: str = "user") -> Tuple[bool, str]:
        """Registers a new user account."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
        if cursor.fetchone():
            conn.close()
            return False, "An account with this email address already exists."

        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
        INSERT INTO users (email, password_hash, name, role, avatar_url, provider, created_at)
        VALUES (?, ?, ?, ?, '👤', 'local', ?)
        """, (email.strip().lower(), hash_password(password), name.strip(), role, now))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
