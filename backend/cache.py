"""
cache.py

SQLite persistence for:
  - TBA API response cache (ETag + body)
  - Match audit records
  - Solver snapshots (latest result per season)

Thread-safety model:
  SQLite in WAL mode allows concurrent readers and one writer.
  We serialise all writes through a module-level threading.Lock so that
  the background refresh task and concurrent API request handlers don't
  interleave writes.  Reads use short-lived connections that are closed
  immediately after the query.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from config import CONFIG

# One write lock shared across all threads in the process
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _open() -> sqlite3.Connection:
    conn = sqlite3.connect(
        CONFIG.db_path,
        check_same_thread=False,
        timeout=30,  # wait up to 30 s for a locked DB before raising
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL, faster than FULL
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")   # 30 s busy timeout in SQLite itself
    return conn


@contextmanager
def _reader() -> Generator[sqlite3.Connection, None, None]:
    """Open a short-lived read-only connection and close it on exit."""
    conn = _open()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _writer() -> Generator[sqlite3.Connection, None, None]:
    """Open a write connection serialised through _write_lock."""
    with _write_lock:
        conn = _open()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _writer() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_cache (
                endpoint        TEXT PRIMARY KEY,
                etag            TEXT,
                fetch_timestamp REAL,
                response_body   TEXT,
                cache_status    TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                match_key       TEXT NOT NULL,
                event_key       TEXT NOT NULL,
                alliance_side   TEXT,
                expected_score  REAL,
                actual_score    REAL,
                residual        REAL,
                refund          REAL,
                defender_keys   TEXT,
                row_weight      REAL,
                is_breaker      INTEGER DEFAULT 0,
                is_excluded     INTEGER DEFAULT 0,
                created_at      REAL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_records(event_key);
            CREATE INDEX IF NOT EXISTS idx_audit_match ON audit_records(match_key);

            CREATE TABLE IF NOT EXISTS solver_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                solve_timestamp REAL,
                season_year     INTEGER,
                results_json    TEXT
            );
        """)


# ---------------------------------------------------------------------------
# API cache
# ---------------------------------------------------------------------------

def get_cached(endpoint: str) -> Optional[dict]:
    with _reader() as conn:
        row = conn.execute(
            "SELECT * FROM api_cache WHERE endpoint=?", (endpoint,)
        ).fetchone()
        return dict(row) if row else None


def set_cached(endpoint: str, etag: str, body: str, status: str = "fresh") -> None:
    with _writer() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO api_cache
               (endpoint, etag, fetch_timestamp, response_body, cache_status)
               VALUES (?, ?, ?, ?, ?)""",
            (endpoint, etag, datetime.now().timestamp(), body, status),
        )


def get_cache_age(endpoint: str) -> Optional[float]:
    cached = get_cached(endpoint)
    if not cached or not cached.get("fetch_timestamp"):
        return None
    return datetime.now().timestamp() - cached["fetch_timestamp"]


# ---------------------------------------------------------------------------
# Audit records
# ---------------------------------------------------------------------------

def save_audit_records(records: list) -> None:
    if not records:
        return
    now = datetime.now().timestamp()
    with _writer() as conn:
        conn.executemany(
            """INSERT INTO audit_records
               (match_key, event_key, alliance_side, expected_score, actual_score,
                residual, refund, defender_keys, row_weight, is_breaker, is_excluded, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r["match_key"],
                    r["event_key"],
                    r.get("alliance_side", ""),
                    r.get("expected_score", 0.0),
                    r.get("actual_score", 0.0),
                    r.get("residual", 0.0),
                    r.get("refund", 0.0),
                    json.dumps(r.get("defender_keys", [])),
                    r.get("row_weight", 1.0),
                    int(r.get("is_breaker", False)),
                    int(r.get("is_excluded", False)),
                    now,
                )
                for r in records
            ],
        )


def get_audit_for_event(event_key: str) -> list:
    with _reader() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_records WHERE event_key=? ORDER BY match_key",
            (event_key,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["defender_keys"] = json.loads(d.get("defender_keys") or "[]")
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# Solver snapshots
# ---------------------------------------------------------------------------

def save_solver_snapshot(season_year: int, results: dict) -> None:
    with _writer() as conn:
        conn.execute(
            """INSERT INTO solver_snapshots (solve_timestamp, season_year, results_json)
               VALUES (?, ?, ?)""",
            (datetime.now().timestamp(), season_year, json.dumps(results)),
        )
        # Keep only the 5 most recent snapshots to bound disk usage
        conn.execute(
            """DELETE FROM solver_snapshots
               WHERE id NOT IN (
                   SELECT id FROM solver_snapshots
                   WHERE season_year=?
                   ORDER BY solve_timestamp DESC
                   LIMIT 5
               ) AND season_year=?""",
            (season_year, season_year),
        )


def get_latest_snapshot(season_year: int) -> Optional[dict]:
    with _reader() as conn:
        row = conn.execute(
            """SELECT results_json, solve_timestamp FROM solver_snapshots
               WHERE season_year=? ORDER BY solve_timestamp DESC LIMIT 1""",
            (season_year,),
        ).fetchone()
        if not row:
            return None
        return {
            "results":   json.loads(row["results_json"]),
            "timestamp": row["solve_timestamp"],
        }
