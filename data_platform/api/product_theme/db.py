"""PostgreSQL helpers for the product theme API."""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

from fastapi import HTTPException

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


_pg_pool_lock = threading.Lock()
_pg_pool = None


def _get_pg_connect_kwargs() -> dict[str, Any]:
    return {
        "host": os.environ.get("PG_HOST", "localhost"),
        "port": int(os.environ.get("PG_PORT", "5432")),
        "dbname": os.environ.get("PG_DB", "xiamimate"),
        "user": os.environ.get("PG_USER", "xiamimate"),
        "password": os.environ.get("PG_PASSWORD", "xiamimate"),
    }


def _get_pg_pool():
    if psycopg2 is None:
        raise HTTPException(
            status_code=500,
            detail="psycopg2 is required for PostgreSQL-backed theme_api serving",
        )

    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    with _pg_pool_lock:
        if _pg_pool is not None:
            return _pg_pool
        minconn = max(1, int(os.environ.get("THEME_API_PG_POOL_MIN", "1")))
        maxconn = max(minconn, int(os.environ.get("THEME_API_PG_POOL_MAX", "8")))
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            **_get_pg_connect_kwargs(),
        )
        return _pg_pool


@contextlib.contextmanager
def _postgres_conn():
    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        yield conn
    finally:
        pool.putconn(conn)


def _run_pg_dict_query(conn, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(sql, params or [])
        return [dict(row) for row in cursor.fetchall()]
