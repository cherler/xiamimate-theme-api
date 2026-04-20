from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
import secrets
import string
import threading
from typing import Any, Iterator

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

API_KEY_ENV_VAR = "XIAMIMATE_THEME_API_KEY"
DEFAULT_API_KEY_PREFIX = "xia_theme_"
DEFAULT_API_KEY_LENGTH = 40
DEFAULT_DAILY_QUOTA = 1000

TIER_FREE = "free"
TIER_STANDARD = "standard"
TIER_PRO = "pro"
TIER_ADMIN = "admin"
TIER_CUSTOM = "custom"

TIER_QUOTAS: dict[str, int | None] = {
    TIER_FREE: 100,
    TIER_STANDARD: 1000,
    TIER_PRO: 10000,
    TIER_ADMIN: None,
    TIER_CUSTOM: None,
}

DEFAULT_CREATE_TIER = TIER_STANDARD

# PG auth tables live in the serving schema
_API_KEYS_TABLE = "serving.api_keys"
_API_USAGE_TABLE = "serving.api_usage"


@dataclass
class APIKeyRecord:
    key_id: str
    name: str
    tier: str
    key_prefix: str
    key_raw: str | None
    status: str
    daily_quota: int | None
    created_at: str
    revoked_at: str | None
    last_used_at: str | None


@dataclass
class DeletedAPIKeyResult:
    record: APIKeyRecord
    deleted_usage_rows: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_today_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def build_api_key(prefix: str = DEFAULT_API_KEY_PREFIX, length: int = DEFAULT_API_KEY_LENGTH) -> str:
    alphabet = string.ascii_letters + string.digits
    token = "".join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{token}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def normalize_tier(tier: str) -> str:
    normalized = tier.strip().lower()
    if normalized not in TIER_QUOTAS:
        raise ValueError(f"unsupported tier: {tier}")
    return normalized


def resolve_tier_quota(tier: str, daily_quota: int | None = None) -> int | None:
    normalized_tier = normalize_tier(tier)
    default_quota = TIER_QUOTAS[normalized_tier]
    if normalized_tier != TIER_CUSTOM:
        if daily_quota is not None and daily_quota != default_quota:
            raise ValueError(f"tier {normalized_tier} uses fixed daily quota {default_quota}")
        return default_quota
    if daily_quota is None:
        raise ValueError("custom tier requires daily_quota")
    if daily_quota < 1:
        raise ValueError("daily_quota must be >= 1")
    return daily_quota


# ---------------------------------------------------------------------------
# PostgreSQL connection pool (module-level, lazy-init)
# ---------------------------------------------------------------------------

_pg_pool_lock = threading.Lock()
_pg_pool: Any = None


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
        raise RuntimeError("psycopg2 is required for PG-backed auth")

    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    with _pg_pool_lock:
        if _pg_pool is not None:
            return _pg_pool
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            1,
            4,
            **_get_pg_connect_kwargs(),
        )
        return _pg_pool


def _is_pg_connection_closed(conn: Any) -> bool:
    return conn is None or bool(getattr(conn, "closed", 1))


def _ping_pg_connection(conn: Any) -> None:
    previous_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    finally:
        try:
            conn.autocommit = previous_autocommit
        except Exception:
            pass


def _acquire_healthy_pg_connection(pool: Any) -> Any:
    last_error: Exception | None = None
    for _ in range(2):
        conn = pool.getconn()
        if _is_pg_connection_closed(conn):
            pool.putconn(conn, close=True)
            continue

        try:
            _ping_pg_connection(conn)
            return conn
        except Exception as exc:
            last_error = exc
            pool.putconn(conn, close=True)

    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to acquire healthy PostgreSQL connection")


@contextmanager
def _pg_conn() -> Iterator[Any]:
    pool = _get_pg_pool()
    conn = _acquire_healthy_pg_connection(pool)
    discard_conn = False
    try:
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception as exc:
        discard_conn = _is_pg_connection_closed(conn)
        if not discard_conn:
            try:
                conn.rollback()
            except Exception:
                discard_conn = True

        if psycopg2 is not None and isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError)):
            discard_conn = True
        raise
    finally:
        pool.putconn(conn, close=discard_conn or _is_pg_connection_closed(conn))


def _dict_row(cursor) -> dict[str, Any] | None:
    row = cursor.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def _dict_rows(cursor) -> list[dict[str, Any]]:
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _row_to_api_key_record(row: dict[str, Any]) -> APIKeyRecord:
    daily_quota = row["daily_quota"]
    normalized_daily_quota = None
    if daily_quota is not None and int(daily_quota) > 0:
        normalized_daily_quota = int(daily_quota)
    return APIKeyRecord(
        key_id=str(row["key_id"]),
        name=str(row["name"]),
        tier=str(row["tier"]),
        key_prefix=str(row["key_prefix"]),
        key_raw=str(row["key_raw"]) if row.get("key_raw") is not None else None,
        status=str(row["status"]),
        daily_quota=normalized_daily_quota,
        created_at=str(row["created_at"]),
        revoked_at=str(row["revoked_at"]) if row.get("revoked_at") is not None else None,
        last_used_at=str(row["last_used_at"]) if row.get("last_used_at") is not None else None,
    )


def create_api_key(
    name: str,
    tier: str = DEFAULT_CREATE_TIER,
    daily_quota: int | None = None,
    prefix: str = DEFAULT_API_KEY_PREFIX,
    length: int = DEFAULT_API_KEY_LENGTH,
) -> tuple[APIKeyRecord, str]:
    if length < 16:
        raise ValueError("length must be >= 16")

    normalized_tier = normalize_tier(tier)
    resolved_daily_quota = resolve_tier_quota(normalized_tier, daily_quota)
    stored_daily_quota = 0 if resolved_daily_quota is None else resolved_daily_quota

    api_key = build_api_key(prefix=prefix, length=length)
    key_hash = hash_api_key(api_key)
    key_id = f"key_{secrets.token_hex(8)}"
    key_prefix = api_key[: min(len(api_key), 18)]
    created_at = utc_now_iso()

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_API_KEYS_TABLE} (
                    key_id, name, tier, key_prefix, key_hash, key_raw, status, daily_quota, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, 'active', %s, %s)
                """,
                (key_id, name, normalized_tier, key_prefix, key_hash, api_key, stored_daily_quota, created_at),
            )

    return (
        APIKeyRecord(
            key_id=key_id,
            name=name,
            tier=normalized_tier,
            key_prefix=key_prefix,
            key_raw=api_key,
            status="active",
            daily_quota=resolved_daily_quota,
            created_at=created_at,
            revoked_at=None,
            last_used_at=None,
        ),
        api_key,
    )


def resolve_api_key(api_key: str) -> APIKeyRecord | None:
    key_hash = hash_api_key(api_key)
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT key_id, name, tier, key_prefix, key_raw, status, daily_quota, created_at, revoked_at, last_used_at
                FROM {_API_KEYS_TABLE}
                WHERE key_hash = %s
                LIMIT 1
                """,
                (key_hash,),
            )
            row = _dict_row(cur)

    if row is None:
        return None
    return _row_to_api_key_record(row)


def get_daily_usage_count(key_id: str, usage_date: str | None = None) -> int:
    day = usage_date or utc_today_date()
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COALESCE(SUM(request_count), 0) AS total_requests
                FROM {_API_USAGE_TABLE}
                WHERE key_id = %s AND usage_date = %s
                """,
                (key_id, day),
            )
            row = _dict_row(cur)
    return int(row["total_requests"] if row is not None else 0)


def list_api_keys(include_inactive: bool = True) -> list[APIKeyRecord]:
    query = f"""
        SELECT key_id, name, tier, key_prefix, key_raw, status, daily_quota, created_at, revoked_at, last_used_at
        FROM {_API_KEYS_TABLE}
    """
    params: list[Any] = []
    if not include_inactive:
        query += " WHERE status = %s"
        params.append("active")
    query += " ORDER BY created_at DESC"

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = _dict_rows(cur)
    return [_row_to_api_key_record(row) for row in rows]


def deactivate_api_key(key_id: str) -> APIKeyRecord:
    revoked_at = utc_now_iso()
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT key_id, name, tier, key_prefix, key_raw, status, daily_quota, created_at, revoked_at, last_used_at
                FROM {_API_KEYS_TABLE}
                WHERE key_id = %s
                LIMIT 1
                """,
                (key_id,),
            )
            row = _dict_row(cur)
            if row is None:
                raise ValueError(f"api key not found: {key_id}")

            cur.execute(
                f"""
                UPDATE {_API_KEYS_TABLE}
                SET status = 'inactive', revoked_at = %s
                WHERE key_id = %s
                """,
                (revoked_at, key_id),
            )

            cur.execute(
                f"""
                SELECT key_id, name, tier, key_prefix, key_raw, status, daily_quota, created_at, revoked_at, last_used_at
                FROM {_API_KEYS_TABLE}
                WHERE key_id = %s
                LIMIT 1
                """,
                (key_id,),
            )
            updated_row = _dict_row(cur)
    return _row_to_api_key_record(updated_row)


def delete_api_key(key_id: str) -> DeletedAPIKeyResult:
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT key_id, name, tier, key_prefix, key_raw, status, daily_quota, created_at, revoked_at, last_used_at
                FROM {_API_KEYS_TABLE}
                WHERE key_id = %s
                LIMIT 1
                """,
                (key_id,),
            )
            row = _dict_row(cur)
            if row is None:
                raise ValueError(f"api key not found: {key_id}")

            record = _row_to_api_key_record(row)

            cur.execute(
                f"DELETE FROM {_API_USAGE_TABLE} WHERE key_id = %s",
                (key_id,),
            )
            deleted_usage_rows = cur.rowcount

            cur.execute(
                f"DELETE FROM {_API_KEYS_TABLE} WHERE key_id = %s",
                (key_id,),
            )

    return DeletedAPIKeyResult(record=record, deleted_usage_rows=deleted_usage_rows)


def get_usage_summary_by_key(usage_date: str | None = None) -> list[dict[str, Any]]:
    day = usage_date or utc_today_date()
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    k.key_id,
                    k.name,
                    k.tier,
                    k.status,
                    k.daily_quota,
                    COALESCE(SUM(u.request_count), 0) AS usage_count,
                    MAX(u.created_at) AS last_request_at
                FROM {_API_KEYS_TABLE} k
                LEFT JOIN {_API_USAGE_TABLE} u
                  ON k.key_id = u.key_id
                 AND u.usage_date = %s
                GROUP BY k.key_id, k.name, k.tier, k.status, k.daily_quota
                ORDER BY usage_count DESC, k.created_at DESC
                """,
                (day,),
            )
            rows = _dict_rows(cur)

    results: list[dict[str, Any]] = []
    for row in rows:
        daily_quota = row["daily_quota"]
        usage_count = int(row["usage_count"])
        normalized_daily_quota = None
        if daily_quota is not None and int(daily_quota) > 0:
            normalized_daily_quota = int(daily_quota)
        results.append(
            {
                "key_id": str(row["key_id"]),
                "name": str(row["name"]),
                "tier": str(row["tier"]),
                "status": str(row["status"]),
                "daily_quota": normalized_daily_quota,
                "usage_count": usage_count,
                "remaining_quota": None if normalized_daily_quota is None else max(normalized_daily_quota - usage_count, 0),
                "last_request_at": str(row["last_request_at"]) if row["last_request_at"] is not None else None,
                "usage_date": day,
            }
        )
    return results


def record_api_usage(
    key_id: str,
    endpoint: str,
    status_code: int,
    response_time_ms: int | None = None,
) -> None:
    created_at = utc_now_iso()
    usage_date = utc_today_date()
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_API_USAGE_TABLE} (key_id, endpoint, usage_date, status_code, response_time_ms, request_count, created_at)
                VALUES (%s, %s, %s, %s, %s, 1, %s)
                """,
                (key_id, endpoint, usage_date, status_code, response_time_ms, created_at),
            )
            cur.execute(
                f"""
                UPDATE {_API_KEYS_TABLE}
                SET last_used_at = %s
                WHERE key_id = %s
                """,
                (created_at, key_id),
            )


def get_active_key_count() -> int:
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS active_key_count
                FROM {_API_KEYS_TABLE}
                WHERE status = 'active'
                """
            )
            row = _dict_row(cur)
    return int(row["active_key_count"] if row is not None else 0)
