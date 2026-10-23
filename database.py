"""Small SQLite/PostgreSQL adapter for FlareGuard's simple SQL operations.

Use DATABASE_URL=postgresql://... for a persistent hosted database. When it is
unset, local development continues to use SQLite.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

try:
    import psycopg
    from psycopg import IntegrityError as PostgresIntegrityError
except ImportError:  # SQLite-only local development remains possible.
    psycopg = None
    PostgresIntegrityError = None

INTEGRITY_ERRORS = (sqlite3.IntegrityError,) + ((PostgresIntegrityError,) if PostgresIntegrityError else ())


def is_postgres() -> bool:
    return DATABASE_URL.startswith(("postgres://", "postgresql://"))


def connect(db_path: str | None = None):
    if is_postgres():
        if psycopg is None:
            raise RuntimeError("Install psycopg[binary] to use DATABASE_URL.")
        # Neon requires TLS; its connection string normally includes sslmode.
        return psycopg.connect(DATABASE_URL, connect_timeout=10)
    conn = sqlite3.connect(db_path or os.getenv("FLAREGUARD_DB_PATH", "flareguard.db"), timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def execute(conn: Any, sql: str, params: tuple | list = ()):
    if is_postgres():
        sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        sql = sql.replace("?", "%s")
    return conn.execute(sql, params)
