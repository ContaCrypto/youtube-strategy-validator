"""SQLite persistence layer for the Strategy Validator."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("strategy_validator.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                youtube_url           TEXT NOT NULL,
                video_id              TEXT,
                strategy_name         TEXT,
                strategy_type         TEXT,
                market                TEXT,
                timeframe             TEXT,
                coding_readiness_score INTEGER,
                confidence_score      INTEGER,
                pine_script_ready     INTEGER,
                warning               TEXT,
                missing_information   TEXT,
                failure_reasons       TEXT,
                subjective_terms      TEXT,
                verdict               TEXT,
                automation_difficulty TEXT,
                created_at            TEXT NOT NULL,
                pine_script_missing   TEXT
            )
        """)
        # Migrate existing databases that pre-date pine_script_missing
        try:
            conn.execute("ALTER TABLE analyses ADD COLUMN pine_script_missing TEXT")
        except Exception:
            pass  # Column already exists


def _extract_text(value: Any) -> str:
    """Flatten a rule object (dict/str) to plain text."""
    if isinstance(value, dict):
        return value.get("rule", str(value))
    return str(value) if value is not None else ""


def _list_to_json(items: Any) -> str:
    if not isinstance(items, list):
        return "[]"
    return json.dumps([_extract_text(i) for i in items])


def save_analysis(
    youtube_url: str,
    video_id: str,
    result: Dict[str, Any],
    verdict: Dict[str, Any],
) -> int:
    warning_raw = result.get("scam_or_cherry_pick_warning", "")
    warning_text = _extract_text(warning_raw)

    pine_missing_raw = result.get("pine_script_missing", [])
    pine_missing_json = json.dumps(
        pine_missing_raw if isinstance(pine_missing_raw, list) else []
    )

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (
                youtube_url, video_id, strategy_name, strategy_type,
                market, timeframe,
                coding_readiness_score, confidence_score, pine_script_ready,
                warning, missing_information, failure_reasons, subjective_terms,
                verdict, automation_difficulty, created_at,
                pine_script_missing
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                youtube_url,
                video_id,
                result.get("strategy_name"),
                result.get("strategy_type"),
                result.get("market"),
                result.get("timeframe"),
                int(result.get("coding_readiness_score", 0)),
                int(result.get("confidence_score", 0)),
                1 if result.get("pine_script_ready") else 0,
                warning_text,
                _list_to_json(result.get("missing_information", [])),
                _list_to_json(result.get("failure_reasons", [])),
                _list_to_json(result.get("subjective_terms", [])),
                verdict.get("verdict"),
                verdict.get("difficulty"),
                datetime.utcnow().isoformat(timespec="seconds"),
                pine_missing_json,
            ),
        )
        return cur.lastrowid


def get_all_analyses() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(analysis_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    rec = dict(row)
    for field in ("missing_information", "failure_reasons", "subjective_terms",
                  "pine_script_missing"):
        try:
            rec[field] = json.loads(rec[field] or "[]")
        except Exception:
            rec[field] = []
    return rec
