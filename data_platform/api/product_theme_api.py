from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from statistics import median
import threading
import time
from typing import Any

import requests as http_requests

import jieba
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

from data_platform.llm_client import ROOT_ENV_FILE, load_env_file_if_present
from data_platform.product_query_assistant import ProductRecallQueryAssistant

from data_platform.api.theme_api_auth import (
    API_KEY_ENV_VAR,
    API_KEY_NAME_ENV_VAR,
    APIKeyRecord,
    ensure_env_api_key_registered,
    get_active_key_count,
    record_api_usage,
    resolve_api_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data_platform" / "storage" / "features" / "training_sets" / "week1_foundation"
THEME_FEATURE_RETENTION_DAYS = max(30, int(os.environ.get("THEME_FEATURE_RETENTION_DAYS", "180")))
THEME_FEATURE_SERVING_TABLES = {
    "base": "serving.theme_base_daily",
    "trends": "serving.theme_trends_daily",
    "cross": "serving.theme_cross_daily",
}
THEME_FORECAST_SERVING_TABLES = {
    "release": "serving.sales_forecast_release",
    "coverage_current": "serving.sales_forecast_domain_coverage_current",
    "item_current": "serving.item_market_sales_forecast_current",
}
API_KEY_HEADER_NAME = "X-API-Key"
API_RESPONSE_SCHEMA = "xiamimate_theme_api_v1"
PROTECTED_API_PREFIX = "/api/product-theme/"
FORECAST_STATUS_READY = "ready"
FORECAST_STATUS_PARTIAL_COVERAGE = "partial_coverage"
FORECAST_STATUS_MISSING_DOMAIN_MODEL = "missing_domain_model"
FORECAST_STATUS_MISSING_ASIN_PREDICTION = "missing_asin_prediction"
FORECAST_STATUS_UNAVAILABLE = "unavailable"
FORECAST_HIGH_GROWTH_RATIO_THRESHOLD = 2.0
FORECAST_TOP_ASINS_LIMIT = 5

MARKETPLACE_TO_DOMAIN = {
    "US": 1,
    "UK": 2,
    "DE": 3,
    "FR": 4,
    "JP": 5,
    "CA": 6,
    "IT": 8,
    "ES": 9,
    "IN": 10,
    "MX": 11,
    "BR": 12,
    "AU": 13,
}
DOMAIN_TO_MARKETPLACE = {value: key for key, value in MARKETPLACE_TO_DOMAIN.items()}
ASIN_PATTERN = re.compile(r"^[A-Z0-9]{8,16}$")
ASCII_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MIN_TOKEN_LENGTH = 2

QUERY_MODIFIER_TOKENS = {
    "adjustable",
    "auto",
    "automatic",
    "battery",
    "best",
    "corded",
    "cordless",
    "digital",
    "electric",
    "for",
    "high",
    "large",
    "mini",
    "new",
    "portable",
    "pro",
    "professional",
    "rechargeable",
    "replacement",
    "small",
    "smart",
    "travel",
    "usb",
    "waterproof",
    "wireless",
    "with",
}

QUERY_ALIAS_EXPANSIONS = {
    "电子相框": ["digital photo frame", "digital picture frame", "wifi digital picture frame"],
    "数码相框": ["digital photo frame", "digital picture frame", "wifi digital picture frame"],
    "手持淋浴头": ["handheld shower head", "handheld showerhead", "high pressure handheld shower head"],
    "手持花洒": ["handheld shower head", "handheld showerhead", "high pressure handheld shower head"],
}

TOKEN_ALIAS_EXPANSIONS = {
    "电子": ["digital", "electronic"],
    "数码": ["digital"],
    "相框": ["photo frame", "picture frame"],
    "画框": ["picture frame", "photo frame"],
    "手持": ["handheld"],
    "淋浴": ["shower"],
    "淋浴头": ["shower head", "showerhead"],
    "花洒": ["shower head", "showerhead"],
}


load_env_file_if_present(ROOT_ENV_FILE)
QUERY_ASSISTANT = ProductRecallQueryAssistant(env_prefix="THEME_QUERY_NORMALIZER")
LOGGER = logging.getLogger(__name__)


class ResolveCandidatesRequest(BaseModel):
    product_query: str = Field(..., min_length=1)
    marketplace: str | int = "US"
    query_aliases: list[str] = Field(default_factory=list)
    category_hints: list[str] = Field(default_factory=list)
    price_min: float | None = None
    price_max: float | None = None
    max_candidates: int = Field(default=50, ge=1, le=500)
    active_only: bool = True

    @validator("query_aliases", "category_hints", pre=True, always=True)
    def _accept_csv_string(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v


class CandidatePoolRequest(BaseModel):
    candidate_asins: list[str] = Field(..., min_length=1)
    marketplace: str | int = "US"
    window_days: int = Field(default=30, ge=7, le=180)


class WeakForecastRequest(CandidatePoolRequest):
    top_n: int = Field(default=5, ge=1, le=20)


class DrilldownRequest(CandidatePoolRequest):
    top_n: int | None = Field(default=None, ge=1, le=20)


class AsinHistoryTimeseriesRequest(BaseModel):
    asins: list[str] = Field(..., min_length=1)
    marketplace: str | int = "US"
    window_days: int = Field(default=90, ge=7, le=90)
    interval: str = Field(default="day")
    metrics: list[str] = Field(default_factory=list)
    include_latest_snapshot: bool = True
    include_window_summary: bool = True
    source_preference: str = "local_first"
    fallback_keepa_snapshot: bool = True

    @validator("asins", "metrics", pre=True, always=True)
    def _accept_csv_string(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v

    @validator("interval")
    def _validate_interval(cls, v: str) -> str:  # noqa: N805
        allowed = {"day", "week"}
        value = (v or "day").strip().lower()
        if value not in allowed:
            raise ValueError("interval must be one of: day, week")
        return value


class KeepaAsinLookupRequest(BaseModel):
    asins: list[str] = Field(..., min_length=1)
    marketplace: str | int = "US"

    @validator("asins", pre=True, always=True)
    def _accept_csv_string(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _response_meta(endpoint: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "endpoint": endpoint,
        "api_version": "2026-04-10",
        "response_schema": API_RESPONSE_SCHEMA,
        "generated_at": _utc_now_iso(),
    }
    if extra:
        meta.update(extra)
    return meta


def _success_response(endpoint: str, data: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "meta": _response_meta(endpoint),
    }


def _error_response(endpoint: str, code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "code": code,
        "message": message,
        "data": {},
        "meta": _response_meta(endpoint),
    }


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix):].strip()
    return None


def _auth_error_response(endpoint: str, status_code: int, code: str, message: str, meta: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "code": code,
            "message": message,
            "data": {},
            "meta": _response_meta(endpoint=endpoint, extra=meta),
        },
    )


@dataclass
class CandidateRecord:
    asin: str
    domain: int
    marketplace: str
    product_title: str
    brand: str
    category: str
    category_path: str
    search_term: str
    business_priority: int
    business_tier: str
    is_active: bool
    current_price: float | None
    current_rating: float | None
    current_review_count: int | None
    current_bsr: int | None
    current_offer_count: int | None
    history_rows_30d: int
    has_sales_signal_30d: bool
    has_price_data_30d: bool
    latest_history_date: str | None
    keywords: list[str]


def _normalize_marketplace(value: str | int) -> tuple[int, str]:
    if isinstance(value, int):
        if value not in DOMAIN_TO_MARKETPLACE:
            raise HTTPException(status_code=400, detail=f"unsupported domain: {value}")
        return value, DOMAIN_TO_MARKETPLACE[value]

    text = str(value).strip().upper()
    if text.isdigit():
        domain = int(text)
        if domain not in DOMAIN_TO_MARKETPLACE:
            raise HTTPException(status_code=400, detail=f"unsupported domain: {domain}")
        return domain, DOMAIN_TO_MARKETPLACE[domain]
    if text not in MARKETPLACE_TO_DOMAIN:
        raise HTTPException(status_code=400, detail=f"unsupported marketplace: {value}")
    return MARKETPLACE_TO_DOMAIN[text], text


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _get_query_normalizer_config() -> dict[str, Any]:
    config = QUERY_ASSISTANT.provider_summary()
    return {
        "active_profile": config.get("active_profile"),
        "provider": config.get("provider"),
        "enabled": config.get("enabled"),
        "configured": config.get("configured"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "timeout_seconds": config.get("timeout_seconds"),
        "mode": config.get("mode"),
        "error": config.get("error"),
    }


def _tokenize_phrase(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    tokens: list[str] = []
    seen: set[str] = set()

    for token in ASCII_TOKEN_PATTERN.findall(text.lower()):
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    for token in jieba.lcut(text):
        normalized = _normalize_text(token)
        if not normalized:
            continue
        if ASCII_TOKEN_PATTERN.fullmatch(normalized):
            continue
        if len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)

    return tokens


def _category_path_parts(category_path: Any) -> list[str]:
    return [part.strip() for part in str(category_path or "").split(" > ") if part.strip()]


def _category_level_name(category_path: Any, index: int) -> str | None:
    parts = _category_path_parts(category_path)
    if 0 <= index < len(parts):
        return parts[index]
    return None


def _leaf_category_name(category_path: Any, fallback_category: Any = None) -> str | None:
    parts = _category_path_parts(category_path)
    if parts:
        return parts[-1]
    fallback = str(fallback_category or "").strip()
    return fallback or None


def _fine_category_name(category_path: Any, fallback_category: Any = None) -> str | None:
    return _leaf_category_name(category_path, fallback_category)


def _category_distribution(items: list[dict[str, Any]], field_name: str, limit: int = 12) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(field_name) or "").strip()
        if not value:
            value = "其他/未归类"
        counts[value] = counts.get(value, 0) + 1
    return [
        {"category": category, "candidate_count": count}
        for category, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def _candidate_field_matches_required_terms(item: dict[str, Any], field_names: list[str], required_terms: list[str]) -> bool:
    for field_name in field_names:
        text = _normalize_text(str(item.get(field_name) or ""))
        if any(_text_contains_token_variant(text, term) for term in required_terms):
            return True
    return False


def _is_query_modifier_token(token: str) -> bool:
    return _normalize_text(token) in QUERY_MODIFIER_TOKENS


def _token_variants(token: str) -> set[str]:
    normalized = _normalize_text(token)
    if not normalized:
        return set()
    variants = {normalized}
    if ASCII_TOKEN_PATTERN.fullmatch(normalized):
        if normalized.endswith("ies") and len(normalized) > 3:
            variants.add(normalized[:-3] + "y")
        elif normalized.endswith("y") and len(normalized) > 2:
            variants.add(normalized[:-1] + "ies")
        if normalized.endswith("es") and len(normalized) > 3:
            variants.add(normalized[:-2])
        if normalized.endswith("s") and len(normalized) > 3:
            variants.add(normalized[:-1])
        else:
            variants.add(normalized + "s")
    return {variant for variant in variants if len(variant) >= MIN_TOKEN_LENGTH}


def _text_contains_token_variant(text: str, token: str) -> bool:
    return any(_text_contains_token(text, variant) for variant in _token_variants(token))


def _build_required_product_terms(normalized_phrases: list[str], tokens: list[str], max_terms: int = 8) -> list[str]:
    required_terms: list[str] = []
    for phrase in normalized_phrases:
        phrase_tokens = [token for token in _tokenize_phrase(phrase) if ASCII_TOKEN_PATTERN.fullmatch(token)]
        if not phrase_tokens:
            continue
        product_tokens = [token for token in phrase_tokens if not _is_query_modifier_token(token)]
        required_terms.append((product_tokens or phrase_tokens)[-1])

    if not required_terms:
        ascii_tokens = [token for token in tokens if ASCII_TOKEN_PATTERN.fullmatch(token)]
        product_tokens = [token for token in ascii_tokens if not _is_query_modifier_token(token)]
        if product_tokens or ascii_tokens:
            required_terms.append((product_tokens or ascii_tokens)[-1])

    return _unique_nonempty(required_terms)[:max_terms]


def _expand_query_aliases(phrase_inputs: list[str], tokens: list[str]) -> list[str]:
    expansions: list[str] = []
    seen: set[str] = set()

    for phrase in phrase_inputs:
        normalized_phrase = _normalize_text(phrase)
        for expansion in QUERY_ALIAS_EXPANSIONS.get(normalized_phrase, []):
            normalized_expansion = _normalize_text(expansion)
            if not normalized_expansion or normalized_expansion in seen:
                continue
            seen.add(normalized_expansion)
            expansions.append(expansion)

    for token in tokens:
        normalized_token = _normalize_text(token)
        for expansion in TOKEN_ALIAS_EXPANSIONS.get(normalized_token, []):
            normalized_expansion = _normalize_text(expansion)
            if not normalized_expansion or normalized_expansion in seen:
                continue
            seen.add(normalized_expansion)
            expansions.append(expansion)

    return expansions


def _unique_nonempty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _sanitize_asins(asins: list[str]) -> list[str]:
    cleaned: list[str] = []
    for asin in asins:
        normalized = str(asin).strip().upper()
        if not ASIN_PATTERN.fullmatch(normalized):
            raise HTTPException(status_code=400, detail=f"invalid asin: {asin}")
        cleaned.append(normalized)
    unique: list[str] = []
    seen: set[str] = set()
    for asin in cleaned:
        if asin in seen:
            continue
        seen.add(asin)
        unique.append(asin)
    return unique


def _effective_feature_window_days(requested_days: int) -> int:
    return min(requested_days, THEME_FEATURE_RETENTION_DAYS)


# ---------------------------------------------------------------------------
# PostgreSQL online serving pool — theme_api online reads should use PG only
# ---------------------------------------------------------------------------

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


def _iso_date_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _get_theme_feature_serving_status(include_data_max_date: bool = True) -> dict[str, Any]:
    with _postgres_conn() as conn:
        registry_row = _run_pg_dict_query(
            conn,
            """
            SELECT
                to_regclass(%s) AS base_table,
                to_regclass(%s) AS trends_table,
                to_regclass(%s) AS cross_table
            """,
            [
                THEME_FEATURE_SERVING_TABLES["base"],
                THEME_FEATURE_SERVING_TABLES["trends"],
                THEME_FEATURE_SERVING_TABLES["cross"],
            ],
        )[0]

        missing_tables = [
            THEME_FEATURE_SERVING_TABLES[table_name]
            for table_name, registry_key in (
                ("base", "base_table"),
                ("trends", "trends_table"),
                ("cross", "cross_table"),
            )
            if registry_row.get(registry_key) is None
        ]
        if missing_tables:
            raise HTTPException(
                status_code=500,
                detail=(
                    "theme feature serving tables missing: "
                    f"{', '.join(missing_tables)}. "
                    "sync week1 foundation features into PostgreSQL before using the API."
                ),
            )

        max_date_row = {
            "base_max_date": None,
            "trends_max_date": None,
            "cross_max_date": None,
        }
        if include_data_max_date:
            max_date_row = _run_pg_dict_query(
                conn,
                """
                SELECT
                    (SELECT MAX(date) FROM serving.theme_base_daily) AS base_max_date,
                    (SELECT MAX(date) FROM serving.theme_trends_daily) AS trends_max_date,
                    (SELECT MAX(date) FROM serving.theme_cross_daily) AS cross_max_date
                """,
            )[0]

    max_dates = [value for value in max_date_row.values() if value is not None]
    return {
        "schema": "serving",
        "retention_days": THEME_FEATURE_RETENTION_DAYS,
        "data_max_date": max(max_dates).isoformat() if max_dates else None,
        "tables": {
            "theme_base_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["base"],
                "data_max_date": max_date_row["base_max_date"].isoformat() if max_date_row.get("base_max_date") else None,
            },
            "theme_trends_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["trends"],
                "data_max_date": max_date_row["trends_max_date"].isoformat() if max_date_row.get("trends_max_date") else None,
            },
            "theme_cross_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["cross"],
                "data_max_date": max_date_row["cross_max_date"].isoformat() if max_date_row.get("cross_max_date") else None,
            },
        },
    }


def _build_query_variants(product_query: str, query_aliases: list[str], category_hints: list[str]) -> tuple[list[str], list[str], list[str]]:
    phrase_inputs = _unique_nonempty([product_query] + query_aliases + category_hints)
    tokens: list[str] = []
    for phrase in phrase_inputs:
        tokens.extend(_tokenize_phrase(phrase))
    unique_tokens = _unique_nonempty(tokens)
    expansions = _expand_query_aliases(phrase_inputs, unique_tokens)
    all_phrase_inputs = _unique_nonempty(phrase_inputs + expansions)
    normalized_phrases = [_normalize_text(value) for value in all_phrase_inputs if _normalize_text(value)]
    expanded_tokens: list[str] = []
    for phrase in all_phrase_inputs:
        expanded_tokens.extend(_tokenize_phrase(phrase))
    return normalized_phrases, _unique_nonempty(expanded_tokens), expansions


def _match_list_contains(values: list[str], needle: str) -> bool:
    return any(needle in _normalize_text(value) for value in values if value)


def _text_contains_token(text: str, token: str) -> bool:
    if not text or not token:
        return False
    if ASCII_TOKEN_PATTERN.fullmatch(token):
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


def _escape_sql_like_term(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_sql_prefilter_terms(normalized_phrases: list[str], tokens: list[str], max_terms: int = 40) -> list[str]:
    terms: list[str] = []
    for value in normalized_phrases + tokens:
        normalized = _normalize_text(value)
        if not normalized or len(normalized) < MIN_TOKEN_LENGTH:
            continue
        terms.append(_escape_sql_like_term(normalized))
    return _unique_nonempty(terms)[:max_terms]


def _candidate_sql_prefilter_limit(max_candidates: int) -> int:
    requested = max(1, int(max_candidates or 1))
    return min(5000, max(500, requested * 25))


def _score_candidate(
    record: CandidateRecord,
    normalized_phrases: list[str],
    tokens: list[str],
) -> tuple[float, list[str], dict[str, Any]]:
    title = _normalize_text(record.product_title)
    category = _normalize_text(record.category)
    category_path = _normalize_text(record.category_path)
    search_term = _normalize_text(record.search_term)
    keywords = [_normalize_text(keyword) for keyword in record.keywords if keyword]
    required_product_terms = _build_required_product_terms(normalized_phrases, tokens)

    phrase_score = 0.0
    token_score = 0.0
    reasons: set[str] = set()

    for phrase in normalized_phrases:
        if phrase and phrase in title:
            phrase_score += 80
            reasons.add("title_phrase_match")
        if phrase and phrase in category:
            phrase_score += 50
            reasons.add("category_phrase_match")
        if phrase and phrase in category_path:
            phrase_score += 45
            reasons.add("category_path_phrase_match")
        if phrase and phrase in search_term:
            phrase_score += 35
            reasons.add("search_term_phrase_match")
        if phrase and _match_list_contains(keywords, phrase):
            phrase_score += 42
            reasons.add("keyword_phrase_match")

    combined_text = " ".join([title, category, category_path, search_term] + keywords)
    matched_tokens = 0
    for token in tokens:
        token_hit = False
        if _text_contains_token(title, token):
            token_score += 14
            reasons.add("title_token_match")
            token_hit = True
        if _text_contains_token(category, token):
            token_score += 10
            reasons.add("category_token_match")
            token_hit = True
        if _text_contains_token(category_path, token):
            token_score += 8
            reasons.add("category_path_token_match")
            token_hit = True
        if _text_contains_token(search_term, token):
            token_score += 7
            reasons.add("search_term_token_match")
            token_hit = True
        if any(_text_contains_token(keyword, token) for keyword in keywords):
            token_score += 12
            reasons.add("keyword_token_match")
            token_hit = True
        if token_hit:
            matched_tokens += 1

    high_signal_texts = [title, category, category_path] + keywords
    semantic_field_texts = [category, category_path] + keywords
    has_compound_phrase_match = any(
        phrase
        and len(_tokenize_phrase(phrase)) >= 2
        and any(phrase in text for text in high_signal_texts + [search_term])
        for phrase in normalized_phrases
    )
    matched_required_product_terms = [
        term
        for term in required_product_terms
        if any(_text_contains_token_variant(text, term) for text in high_signal_texts)
    ]
    semantic_field_required_product_terms = [
        term
        for term in required_product_terms
        if any(_text_contains_token_variant(text, term) for text in semantic_field_texts)
    ]
    if matched_required_product_terms:
        reasons.add("required_product_term_match")
    if semantic_field_required_product_terms:
        reasons.add("required_product_term_semantic_field_match")
    if has_compound_phrase_match:
        reasons.add("compound_phrase_match")

    if tokens and matched_tokens == len(tokens):
        token_score += 18
        reasons.add("all_tokens_covered")
    elif matched_tokens >= 2:
        token_score += 8
        reasons.add("multi_token_covered")

    business_score = min(max(record.business_priority, 0), 100) / 10.0
    freshness_score = 12.0 if record.has_sales_signal_30d else 0.0
    completeness_score = 8.0 if record.has_price_data_30d else 0.0

    total_score = phrase_score + token_score + business_score + freshness_score + completeness_score
    minimum_token_hits = 1 if len(tokens) <= 1 else min(2, len(tokens))
    has_phrase_match = phrase_score > 0
    required_product_term_coverage = (
        round(len(matched_required_product_terms) / len(required_product_terms), 4) if required_product_terms else None
    )
    matched_token_ratio = round(matched_tokens / len(tokens), 4) if tokens else None
    if required_product_terms and not matched_required_product_terms:
        total_score = 0.0
    if len(tokens) >= 2 and required_product_terms and not semantic_field_required_product_terms and not has_compound_phrase_match:
        total_score = 0.0
    if tokens and not has_phrase_match and matched_tokens < minimum_token_hits:
        total_score = 0.0
    if not normalized_phrases and not tokens:
        total_score = 0.0
    if combined_text == "":
        total_score = 0.0

    breakdown = {
        "phrase_score": round(phrase_score, 2),
        "token_score": round(token_score, 2),
        "business_score": round(business_score, 2),
        "freshness_score": round(freshness_score, 2),
        "completeness_score": round(completeness_score, 2),
        "query_token_count": len(tokens),
        "matched_query_token_count": matched_tokens,
        "matched_query_token_ratio": matched_token_ratio,
        "required_product_terms": required_product_terms,
        "matched_required_product_terms": matched_required_product_terms,
        "semantic_field_required_product_terms": semantic_field_required_product_terms,
        "required_product_term_coverage": required_product_term_coverage,
        "compound_phrase_match": has_compound_phrase_match,
    }
    return total_score, sorted(reasons), breakdown


def _keepa_latest_value(csv_2d: list, index: int) -> float | int | None:
    if not csv_2d or index >= len(csv_2d):
        return None
    arr = csv_2d[index]
    if not arr or len(arr) < 2:
        return None
    raw = arr[-1]
    if raw is None or raw == -1:
        return None
    return raw


def _keepa_latest_price(csv_2d: list, index: int, is_yen: bool) -> float | None:
    val = _keepa_latest_value(csv_2d, index)
    if val is None:
        return None
    if is_yen:
        return round(float(val), 2)
    return round(float(val) / 100, 2)


def _keepa_stats_value(stats_arr: list, index: int) -> float | int | None:
    if not stats_arr or index >= len(stats_arr):
        return None
    raw = stats_arr[index]
    if raw is None or raw == -1:
        return None
    return raw


def _keepa_stats_price(stats: dict, key: str, index: int, is_yen: bool) -> float | None:
    arr = stats.get(key) or []
    val = _keepa_stats_value(arr, index)
    if val is None:
        return None
    if is_yen:
        return round(float(val), 2)
    return round(float(val) / 100, 2)


class ProductThemeService:
    def _build_forecast_meta(self, coverage_row: dict[str, Any] | None = None) -> dict[str, Any]:
        coverage_row = coverage_row or {}
        return {
            "forecast_version": coverage_row.get("forecast_version"),
            "snapshot_date": _iso_date_or_none(coverage_row.get("snapshot_date")),
            "forecast_week_start": _iso_date_or_none(coverage_row.get("forecast_week_start")),
            "forecast_year_week": coverage_row.get("forecast_year_week"),
        }

    def _normalize_top_feature_contributions(self, raw_value: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_value, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "feature": item.get("feature"),
                    "label": item.get("label"),
                    "direction": item.get("direction"),
                    "contribution_share": round(float(item["contribution_share"]), 4)
                    if item.get("contribution_share") is not None
                    else None,
                }
            )
        return normalized

    def _build_item_sales_forecast_explainability(self, forecast_row: dict[str, Any] | None = None) -> dict[str, Any]:
        forecast_row = forecast_row or {}
        return {
            "primary_driver_feature": forecast_row.get("primary_driver_feature"),
            "primary_driver_label": forecast_row.get("primary_driver_label"),
            "primary_driver_direction": forecast_row.get("primary_driver_direction"),
            "primary_driver_contribution_share": round(float(forecast_row["primary_driver_contribution_share"]), 4)
            if forecast_row.get("primary_driver_contribution_share") is not None
            else None,
            "top_feature_contributions": self._normalize_top_feature_contributions(
                forecast_row.get("top_feature_contributions")
            ),
            "driver_summary_text": forecast_row.get("driver_summary_text"),
        }

    def _build_candidate_pool_driver_distribution(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []

        grouped: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
        for row in rows:
            driver_feature = row.get("primary_driver_feature")
            driver_label = row.get("primary_driver_label") or driver_feature
            driver_direction = row.get("primary_driver_direction")
            if not driver_feature and not driver_label:
                continue
            key = (driver_feature, driver_label, driver_direction)
            bucket = grouped.setdefault(
                key,
                {
                    "driver_feature": driver_feature,
                    "driver_label": driver_label,
                    "driver_direction": driver_direction,
                    "item_count": 0,
                },
            )
            bucket["item_count"] += 1

        denominator = sum(int(item["item_count"]) for item in grouped.values())
        if denominator <= 0:
            return []

        distribution = []
        for item in grouped.values():
            distribution.append(
                {
                    **item,
                    "item_ratio": round(int(item["item_count"]) / denominator, 4),
                }
            )

        distribution.sort(key=lambda item: (-int(item["item_count"]), str(item.get("driver_label") or "")))
        return distribution

    def _build_candidate_pool_sales_forecast_empty(
        self,
        *,
        status: str,
        candidate_asin_count: int,
        coverage_row: dict[str, Any] | None = None,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_forecast_meta(coverage_row)
        payload.update(
            {
                "status": status,
                "candidate_asin_count": candidate_asin_count,
                "covered_asin_count": 0,
                "missing_asin_count": candidate_asin_count,
                "coverage_ratio": 0.0,
                "high_growth_ratio_threshold": FORECAST_HIGH_GROWTH_RATIO_THRESHOLD,
                "predicted_sales_w4_total": None,
                "predicted_sales_w1_median": None,
                "high_growth_item_ratio": None,
                "top20_predicted_sales_w4_share": None,
                "driver_distribution": [],
                "predicted_top_asins_w4": [],
                "notes": notes or [],
            }
        )
        return payload

    def _build_item_sales_forecast_empty(
        self,
        *,
        status: str,
        coverage_row: dict[str, Any] | None = None,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_forecast_meta(coverage_row)
        payload.update(
            {
                "status": status,
                "predicted_weekly_sales_w1": None,
                "predicted_weekly_sales_w4": None,
                "predicted_growth_ratio_w4_over_w1": None,
                "predicted_growth_delta_w4_minus_w1": None,
                "predicted_rank_w1_within_domain": None,
                "predicted_rank_w4_within_domain": None,
                "model_config_name_w1": coverage_row.get("model_config_name_w1") if coverage_row else None,
                "model_config_name_w4": coverage_row.get("model_config_name_w4") if coverage_row else None,
                "primary_driver_feature": None,
                "primary_driver_label": None,
                "primary_driver_direction": None,
                "primary_driver_contribution_share": None,
                "top_feature_contributions": [],
                "driver_summary_text": None,
                "notes": notes or [],
            }
        )
        return payload

    def _fetch_sales_forecast_domain_coverage(self, conn, domain: int) -> dict[str, Any] | None:
        rows = _run_pg_dict_query(
            conn,
            f"""
            SELECT
                domain,
                forecast_version,
                snapshot_date,
                coverage_status,
                forecast_week_start,
                forecast_year_week,
                model_config_name_w1,
                model_config_name_w4,
                test_rmse_w1,
                test_rmse_w4,
                test_r2_w1,
                test_r2_w4
            FROM {THEME_FORECAST_SERVING_TABLES['coverage_current']}
            WHERE domain = %s
            LIMIT 1
            """,
            [domain],
        )
        return rows[0] if rows else None

    def _fetch_sales_forecast_items(self, conn, domain: int, candidate_asins: list[str]) -> dict[str, dict[str, Any]]:
        rows = _run_pg_dict_query(
            conn,
            f"""
            SELECT
                domain,
                asin,
                forecast_version,
                snapshot_date,
                forecast_week_start,
                forecast_year_week,
                predicted_weekly_sales_w1,
                predicted_weekly_sales_w4,
                predicted_growth_ratio_w4_over_w1,
                predicted_growth_delta_w4_minus_w1,
                predicted_rank_w1_within_domain,
                predicted_rank_w4_within_domain,
                model_config_name_w1,
                model_config_name_w4,
                test_rmse_w1,
                test_rmse_w4,
                test_mae_w1,
                test_mae_w4,
                test_mape_w1,
                test_mape_w4,
                test_smape_w1,
                test_smape_w4,
                test_r2_w1,
                                test_r2_w4,
                                primary_driver_feature,
                                primary_driver_label,
                                primary_driver_direction,
                                primary_driver_contribution_share,
                                top_feature_contributions,
                                driver_summary_text
            FROM {THEME_FORECAST_SERVING_TABLES['item_current']}
            WHERE domain = %s
              AND asin = ANY(%s)
            """,
            [domain, candidate_asins],
        )
        return {str(row["asin"]): row for row in rows}

    def _get_sales_forecast_context(self, conn, domain: int, candidate_asins: list[str]) -> dict[str, Any]:
        coverage_row = self._fetch_sales_forecast_domain_coverage(conn, domain)
        if not coverage_row:
            return {
                "status": FORECAST_STATUS_MISSING_DOMAIN_MODEL,
                "coverage_row": None,
                "items_by_asin": {},
            }

        coverage_status = str(coverage_row.get("coverage_status") or "").strip()
        if coverage_status == FORECAST_STATUS_MISSING_DOMAIN_MODEL:
            return {
                "status": FORECAST_STATUS_MISSING_DOMAIN_MODEL,
                "coverage_row": coverage_row,
                "items_by_asin": {},
            }

        if coverage_status != "covered":
            return {
                "status": FORECAST_STATUS_UNAVAILABLE,
                "coverage_row": coverage_row,
                "items_by_asin": {},
            }

        items_by_asin = self._fetch_sales_forecast_items(conn, domain, candidate_asins)
        return {
            "status": "covered",
            "coverage_row": coverage_row,
            "items_by_asin": items_by_asin,
        }

    def _build_candidate_pool_sales_forecast(self, conn, domain: int, candidate_asins: list[str]) -> dict[str, Any]:
        try:
            forecast_context = self._get_sales_forecast_context(conn, domain, candidate_asins)
        except Exception as exc:
            LOGGER.warning("sales forecast lookup unavailable for candidate_pool_stats domain=%s: %s", domain, exc)
            return self._build_candidate_pool_sales_forecast_empty(
                status=FORECAST_STATUS_UNAVAILABLE,
                candidate_asin_count=len(candidate_asins),
                notes=["sales forecast serving unavailable; base candidate pool stats returned without forecast merge"],
            )

        coverage_row = forecast_context["coverage_row"]
        if forecast_context["status"] == FORECAST_STATUS_MISSING_DOMAIN_MODEL:
            return self._build_candidate_pool_sales_forecast_empty(
                status=FORECAST_STATUS_MISSING_DOMAIN_MODEL,
                candidate_asin_count=len(candidate_asins),
                coverage_row=coverage_row,
                notes=["current domain does not have an active forecast model"],
            )

        if forecast_context["status"] == FORECAST_STATUS_UNAVAILABLE:
            return self._build_candidate_pool_sales_forecast_empty(
                status=FORECAST_STATUS_UNAVAILABLE,
                candidate_asin_count=len(candidate_asins),
                coverage_row=coverage_row,
                notes=["sales forecast serving returned an unsupported coverage status"],
            )

        items_by_asin = forecast_context["items_by_asin"]
        covered_rows = [items_by_asin[asin] for asin in candidate_asins if asin in items_by_asin]
        candidate_asin_count = len(candidate_asins)
        covered_asin_count = len(covered_rows)
        missing_asin_count = candidate_asin_count - covered_asin_count
        status = FORECAST_STATUS_READY if missing_asin_count == 0 else FORECAST_STATUS_PARTIAL_COVERAGE

        predicted_sales_w4_values = [
            float(row["predicted_weekly_sales_w4"])
            for row in covered_rows
            if row.get("predicted_weekly_sales_w4") is not None
        ]
        predicted_sales_w1_values = [
            float(row["predicted_weekly_sales_w1"])
            for row in covered_rows
            if row.get("predicted_weekly_sales_w1") is not None
        ]
        growth_ratio_values = [
            float(row["predicted_growth_ratio_w4_over_w1"])
            for row in covered_rows
            if row.get("predicted_growth_ratio_w4_over_w1") is not None
        ]
        high_growth_driver_rows = [
            row
            for row in covered_rows
            if row.get("predicted_growth_ratio_w4_over_w1") is not None
            and float(row["predicted_growth_ratio_w4_over_w1"]) >= FORECAST_HIGH_GROWTH_RATIO_THRESHOLD
        ]
        driver_rows = high_growth_driver_rows or covered_rows
        predicted_top_rows = sorted(
            covered_rows,
            key=lambda row: float(row.get("predicted_weekly_sales_w4") or float("-inf")),
            reverse=True,
        )

        total_w4 = sum(predicted_sales_w4_values) if predicted_sales_w4_values else None
        top20_w4 = sum(predicted_sales_w4_values[:20]) if predicted_sales_w4_values else None
        payload = self._build_forecast_meta(coverage_row)
        payload.update(
            {
                "status": status,
                "candidate_asin_count": candidate_asin_count,
                "covered_asin_count": covered_asin_count,
                "missing_asin_count": missing_asin_count,
                "coverage_ratio": round(covered_asin_count / candidate_asin_count, 4) if candidate_asin_count else 0.0,
                "high_growth_ratio_threshold": FORECAST_HIGH_GROWTH_RATIO_THRESHOLD,
                "predicted_sales_w4_total": round(total_w4, 2) if total_w4 is not None else None,
                "predicted_sales_w1_median": round(float(median(predicted_sales_w1_values)), 2) if predicted_sales_w1_values else None,
                "high_growth_item_ratio": (
                    round(
                        sum(1 for value in growth_ratio_values if value >= FORECAST_HIGH_GROWTH_RATIO_THRESHOLD)
                        / len(growth_ratio_values),
                        4,
                    )
                    if growth_ratio_values
                    else None
                ),
                "top20_predicted_sales_w4_share": (
                    round(top20_w4 / total_w4, 4)
                    if total_w4 not in (None, 0.0) and top20_w4 is not None
                    else None
                ),
                "driver_distribution": self._build_candidate_pool_driver_distribution(driver_rows),
                "predicted_top_asins_w4": [
                    {
                        "asin": str(row["asin"]),
                        "predicted_weekly_sales_w4": round(float(row["predicted_weekly_sales_w4"]), 2)
                        if row.get("predicted_weekly_sales_w4") is not None
                        else None,
                        "predicted_growth_delta_w4_minus_w1": round(float(row["predicted_growth_delta_w4_minus_w1"]), 2)
                        if row.get("predicted_growth_delta_w4_minus_w1") is not None
                        else None,
                        "predicted_rank_w4_within_domain": int(row["predicted_rank_w4_within_domain"])
                        if row.get("predicted_rank_w4_within_domain") is not None
                        else None,
                        "primary_driver_feature": row.get("primary_driver_feature"),
                        "primary_driver_label": row.get("primary_driver_label"),
                        "primary_driver_direction": row.get("primary_driver_direction"),
                    }
                    for row in predicted_top_rows[:FORECAST_TOP_ASINS_LIMIT]
                ],
                "notes": (
                    ["forecast summary covers only matched ASINs in the current release"]
                    if status == FORECAST_STATUS_PARTIAL_COVERAGE
                    else []
                ),
            }
        )
        return payload

    def _build_top_asin_sales_forecast_payload(self, conn, domain: int, candidate_asins: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        try:
            forecast_context = self._get_sales_forecast_context(conn, domain, candidate_asins)
        except Exception as exc:
            LOGGER.warning("sales forecast lookup unavailable for top_asin_drilldown domain=%s: %s", domain, exc)
            return (
                {
                    "status": FORECAST_STATUS_UNAVAILABLE,
                    "forecast_version": None,
                    "snapshot_date": None,
                    "forecast_week_start": None,
                    "forecast_year_week": None,
                    "notes": ["sales forecast serving unavailable; drilldown returned without forecast enrichment"],
                },
                {},
            )

        coverage_row = forecast_context["coverage_row"]
        if forecast_context["status"] == FORECAST_STATUS_MISSING_DOMAIN_MODEL:
            meta = self._build_forecast_meta(coverage_row)
            meta.update(
                {
                    "status": FORECAST_STATUS_MISSING_DOMAIN_MODEL,
                    "notes": ["current domain does not have an active forecast model"],
                }
            )
            return meta, {}

        if forecast_context["status"] == FORECAST_STATUS_UNAVAILABLE:
            meta = self._build_forecast_meta(coverage_row)
            meta.update(
                {
                    "status": FORECAST_STATUS_UNAVAILABLE,
                    "notes": ["sales forecast serving returned an unsupported coverage status"],
                }
            )
            return meta, {}

        items_by_asin = forecast_context["items_by_asin"]
        matched_count = sum(1 for asin in candidate_asins if asin in items_by_asin)
        meta = self._build_forecast_meta(coverage_row)
        meta.update(
            {
                "status": FORECAST_STATUS_READY if matched_count == len(candidate_asins) else FORECAST_STATUS_PARTIAL_COVERAGE,
                "notes": (
                    ["some drilldown ASINs are not included in the current forecast release"]
                    if matched_count != len(candidate_asins)
                    else []
                ),
            }
        )
        return meta, items_by_asin

    def _fetch_domain_candidates(
        self,
        domain: int,
        *,
        sql_prefilter_terms: list[str],
        sql_required_product_terms: list[str],
        active_only: bool,
        price_min: float | None,
        price_max: float | None,
        prefilter_limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch query-matching registry rows for a domain from PostgreSQL serving tables."""
        with _postgres_conn() as conn:
            return _run_pg_dict_query(
                conn,
                """
                WITH query_terms AS MATERIALIZED (
                    SELECT COALESCE(%s::TEXT[], ARRAY[]::TEXT[]) AS terms
                ),
                required_product_terms AS MATERIALIZED (
                    SELECT COALESCE(%s::TEXT[], ARRAY[]::TEXT[]) AS terms
                ),
                matched_registry AS MATERIALIZED (
                    SELECT
                        r.asin,
                        r.domain,
                        COALESCE(r.marketplace, '') AS marketplace,
                        COALESCE(r.product_title, '') AS product_title,
                        COALESCE(r.brand, '') AS brand,
                        COALESCE(r.category, '') AS category,
                        COALESCE(r.category_path, '') AS category_path,
                        COALESCE(r.search_term, '') AS search_term,
                        COALESCE(r.business_priority, r.priority, 0) AS business_priority,
                        COALESCE(r.business_tier, '') AS business_tier,
                        COALESCE(r.is_active, TRUE) AS is_active,
                        s.price AS current_price,
                        s.rating AS current_rating,
                        s.review_count AS current_review_count,
                        s.bsr AS current_bsr,
                        COALESCE(s.total_offer_count, s.seller_count, s.retrieved_offer_count) AS current_offer_count,
                        (
                            q.text_match_score
                            + q.keyword_match_score
                            + q.required_product_term_match_score
                            + (LEAST(GREATEST(COALESCE(r.business_priority, r.priority, 0), 0), 100) / 10.0)
                        ) AS sql_prefilter_score
                    FROM sync.keepa_asin_registry r
                    LEFT JOIN sync.keepa_product_snapshot s
                        ON r.asin = s.asin AND r.domain = s.domain
                    CROSS JOIN LATERAL (
                        SELECT
                            COALESCE(
                                SUM(
                                    CASE WHEN LOWER(COALESCE(r.product_title, '')) LIKE ('%%' || term || '%%') ESCAPE '\\' THEN 80 ELSE 0 END
                                    + CASE WHEN LOWER(COALESCE(r.category, '')) LIKE ('%%' || term || '%%') ESCAPE '\\' THEN 50 ELSE 0 END
                                    + CASE WHEN LOWER(COALESCE(r.category_path, '')) LIKE ('%%' || term || '%%') ESCAPE '\\' THEN 45 ELSE 0 END
                                    + CASE WHEN LOWER(COALESCE(r.search_term, '')) LIKE ('%%' || term || '%%') ESCAPE '\\' THEN 35 ELSE 0 END
                                ),
                                0
                            )::DOUBLE PRECISION AS text_match_score,
                            COALESCE(
                                SUM(
                                    CASE WHEN EXISTS (
                                        SELECT 1
                                        FROM sync.asin_keyword_mapping km
                                        WHERE km.domain = r.domain
                                          AND km.asin = r.asin
                                          AND LOWER(km.keyword) LIKE ('%%' || term || '%%') ESCAPE '\\'
                                    ) THEN 42 ELSE 0 END
                                ),
                                0
                            )::DOUBLE PRECISION AS keyword_match_score
                            ,
                            COALESCE(
                                (
                                    SELECT SUM(
                                        CASE WHEN LOWER(COALESCE(r.product_title, '')) LIKE ('%%' || required_term || '%%') ESCAPE '\\' THEN 90 ELSE 0 END
                                        + CASE WHEN LOWER(COALESCE(r.category, '')) LIKE ('%%' || required_term || '%%') ESCAPE '\\' THEN 55 ELSE 0 END
                                        + CASE WHEN LOWER(COALESCE(r.category_path, '')) LIKE ('%%' || required_term || '%%') ESCAPE '\\' THEN 60 ELSE 0 END
                                        + CASE WHEN EXISTS (
                                            SELECT 1
                                            FROM sync.asin_keyword_mapping km
                                            WHERE km.domain = r.domain
                                              AND km.asin = r.asin
                                              AND LOWER(km.keyword) LIKE ('%%' || required_term || '%%') ESCAPE '\\'
                                        ) THEN 48 ELSE 0 END
                                    )
                                    FROM UNNEST((SELECT terms FROM required_product_terms)) AS required_term
                                ),
                                0
                            )::DOUBLE PRECISION AS required_product_term_match_score
                        FROM UNNEST((SELECT terms FROM query_terms)) AS term
                    ) q
                    WHERE r.domain = %s
                      AND (%s IS FALSE OR COALESCE(r.is_active, TRUE) = TRUE)
                      AND (%s::DOUBLE PRECISION IS NULL OR s.price IS NULL OR s.price >= %s::DOUBLE PRECISION)
                      AND (%s::DOUBLE PRECISION IS NULL OR s.price IS NULL OR s.price <= %s::DOUBLE PRECISION)
                      AND (
                          CARDINALITY((SELECT terms FROM query_terms)) = 0
                          OR q.text_match_score > 0
                          OR q.keyword_match_score > 0
                      )
                      AND (
                          CARDINALITY((SELECT terms FROM required_product_terms)) = 0
                          OR q.required_product_term_match_score > 0
                      )
                ),
                ranked_registry AS MATERIALIZED (
                    SELECT
                        *,
                        COUNT(*) OVER() AS sql_prefilter_total_count
                    FROM matched_registry
                    ORDER BY
                        sql_prefilter_score DESC,
                        business_priority DESC,
                        current_review_count DESC NULLS LAST
                    LIMIT %s
                ),
                history_flags AS (
                    SELECT
                        h.asin,
                        h.domain,
                        MAX(h.date) AS latest_history_date,
                        COUNT(*) FILTER (WHERE h.date >= CURRENT_DATE - INTERVAL '30 days') AS history_rows_30d,
                        MAX(CASE WHEN h.date >= CURRENT_DATE - INTERVAL '30 days' AND (h.monthly_sold IS NOT NULL OR h.bsr IS NOT NULL) THEN 1 ELSE 0 END) AS has_sales_signal_30d,
                        MAX(CASE WHEN h.date >= CURRENT_DATE - INTERVAL '30 days' AND (
                            h.amazon_price IS NOT NULL OR h.new_price IS NOT NULL OR h.buy_box_price IS NOT NULL OR h.list_price IS NOT NULL
                        ) THEN 1 ELSE 0 END) AS has_price_data_30d
                    FROM sync.keepa_product_history h
                    JOIN ranked_registry r
                        ON h.asin = r.asin AND h.domain = r.domain
                    WHERE h.domain = %s
                    GROUP BY 1, 2
                ),
                keyword_agg AS (
                    SELECT m.asin, m.domain, ARRAY_AGG(DISTINCT m.keyword ORDER BY m.keyword) AS keywords
                    FROM sync.asin_keyword_mapping m
                    JOIN ranked_registry r
                        ON m.asin = r.asin AND m.domain = r.domain
                    WHERE m.domain = %s
                    GROUP BY m.asin, m.domain
                )
                SELECT
                    r.asin,
                    r.domain,
                    r.marketplace,
                    r.product_title,
                    r.brand,
                    r.category,
                    r.category_path,
                    r.search_term,
                    r.business_priority,
                    r.business_tier,
                    r.is_active,
                    r.current_price,
                    r.current_rating,
                    r.current_review_count,
                    r.current_bsr,
                    r.current_offer_count,
                    COALESCE(h.history_rows_30d, 0) AS history_rows_30d,
                    COALESCE(h.has_sales_signal_30d, 0) AS has_sales_signal_30d,
                    COALESCE(h.has_price_data_30d, 0) AS has_price_data_30d,
                    h.latest_history_date,
                    COALESCE(k.keywords, ARRAY[]::VARCHAR[]) AS keywords,
                    r.sql_prefilter_score,
                    r.sql_prefilter_total_count
                FROM ranked_registry r
                LEFT JOIN history_flags h
                    ON r.asin = h.asin AND r.domain = h.domain
                LEFT JOIN keyword_agg k
                    ON r.asin = k.asin AND r.domain = k.domain
                """,
                [
                    sql_prefilter_terms,
                    sql_required_product_terms,
                    domain,
                    active_only,
                    price_min,
                    price_min,
                    price_max,
                    price_max,
                    prefilter_limit,
                    domain,
                    domain,
                ],
            )

    async def resolve_candidates(self, request: ResolveCandidatesRequest) -> dict[str, Any]:
        request_started_at = time.perf_counter()
        domain, marketplace = _normalize_marketplace(request.marketplace)

        normalization_started_at = time.perf_counter()
        normalization = await asyncio.to_thread(
            QUERY_ASSISTANT.normalize,
            product_query=request.product_query,
            query_aliases=request.query_aliases,
            category_hints=request.category_hints,
            marketplace=marketplace,
        )
        normalization_ms = int((time.perf_counter() - normalization_started_at) * 1000)

        normalized_phrases, tokens, query_expansions = _build_query_variants(
            normalization.product_query,
            normalization.query_aliases,
            normalization.category_hints,
        )
        required_product_terms = _build_required_product_terms(normalized_phrases, tokens)
        sql_prefilter_terms = _build_sql_prefilter_terms(normalized_phrases, tokens)
        sql_required_product_terms = _build_sql_prefilter_terms(required_product_terms, [], max_terms=12)
        sql_prefilter_limit = _candidate_sql_prefilter_limit(request.max_candidates)

        def run_domain_fetch() -> tuple[list[dict[str, Any]], int]:
            started_at = time.perf_counter()
            result = self._fetch_domain_candidates(
                domain,
                sql_prefilter_terms=sql_prefilter_terms,
                sql_required_product_terms=sql_required_product_terms,
                active_only=request.active_only,
                price_min=request.price_min,
                price_max=request.price_max,
                prefilter_limit=sql_prefilter_limit,
            )
            return result, int((time.perf_counter() - started_at) * 1000)

        rows, pg_fetch_ms = await asyncio.to_thread(run_domain_fetch)

        scoring_started_at = time.perf_counter()
        sql_prefilter_total_count = max(
            [int(row.get("sql_prefilter_total_count") or 0) for row in rows] or [0]
        )

        candidates: list[dict[str, Any]] = []
        for row in rows:
            record = CandidateRecord(
                asin=str(row["asin"]),
                domain=int(row["domain"]),
                marketplace=str(row["marketplace"] or marketplace),
                product_title=str(row["product_title"] or ""),
                brand=str(row["brand"] or ""),
                category=str(row["category"] or ""),
                category_path=str(row["category_path"] or ""),
                search_term=str(row["search_term"] or ""),
                business_priority=int(row["business_priority"] or 0),
                business_tier=str(row["business_tier"] or ""),
                is_active=bool(row["is_active"]),
                current_price=float(row["current_price"]) if row["current_price"] is not None else None,
                current_rating=float(row["current_rating"]) if row["current_rating"] is not None else None,
                current_review_count=int(row["current_review_count"]) if row["current_review_count"] is not None else None,
                current_bsr=int(row["current_bsr"]) if row["current_bsr"] is not None else None,
                current_offer_count=int(row["current_offer_count"]) if row["current_offer_count"] is not None else None,
                history_rows_30d=int(row["history_rows_30d"] or 0),
                has_sales_signal_30d=bool(row["has_sales_signal_30d"]),
                has_price_data_30d=bool(row["has_price_data_30d"]),
                latest_history_date=str(row["latest_history_date"] or "") or None,
                keywords=list(row["keywords"] or []),
            )
            if request.active_only and not record.is_active:
                continue
            if request.price_min is not None and record.current_price is not None and record.current_price < request.price_min:
                continue
            if request.price_max is not None and record.current_price is not None and record.current_price > request.price_max:
                continue

            score, reasons, breakdown = _score_candidate(record, normalized_phrases, tokens)
            if score <= 0:
                continue
            root_category_name = _category_level_name(record.category_path, 0)
            category_l2_name = _category_level_name(record.category_path, 1)
            category_l3_name = _category_level_name(record.category_path, 2)
            leaf_category_name = _leaf_category_name(record.category_path, record.category)
            fine_category_name = _fine_category_name(record.category_path, record.category)
            candidates.append(
                {
                    "asin": record.asin,
                    "domain": record.domain,
                    "marketplace": marketplace,
                    "product_title": record.product_title,
                    "brand": record.brand,
                    "category": record.category,
                    "category_path": record.category_path,
                    "root_category_name": root_category_name,
                    "category_l2_name": category_l2_name,
                    "category_l3_name": category_l3_name,
                    "leaf_category_name": leaf_category_name,
                    "fine_category_name": fine_category_name,
                    "search_term": record.search_term,
                    "keywords": record.keywords,
                    "business_priority": record.business_priority,
                    "business_tier": record.business_tier,
                    "current_price": record.current_price,
                    "current_rating": record.current_rating,
                    "current_review_count": record.current_review_count,
                    "current_bsr": record.current_bsr,
                    "current_offer_count": record.current_offer_count,
                    "history_rows_30d": record.history_rows_30d,
                    "has_sales_signal_30d": record.has_sales_signal_30d,
                    "has_price_data_30d": record.has_price_data_30d,
                    "latest_history_date": record.latest_history_date,
                    "sql_prefilter_score": round(float(row.get("sql_prefilter_score") or 0), 2),
                    "match_score": round(score, 2),
                    "match_reasons": reasons,
                    "match_breakdown": breakdown,
                }
            )

        candidate_total_before_semantic_category_anchor = len(candidates)
        fine_category_anchored_candidates = [
            item
            for item in candidates
            if _candidate_field_matches_required_terms(
                item,
                ["leaf_category_name", "fine_category_name"],
                required_product_terms,
            )
        ]
        category_anchored_candidates = [
            item
            for item in candidates
            if (item.get("match_breakdown") or {}).get("semantic_field_required_product_terms")
        ]
        semantic_fine_category_anchor_applied = bool(required_product_terms and fine_category_anchored_candidates)
        semantic_category_anchor_applied = bool(required_product_terms and (fine_category_anchored_candidates or category_anchored_candidates))
        if semantic_fine_category_anchor_applied:
            candidates = fine_category_anchored_candidates
        elif semantic_category_anchor_applied:
            candidates = category_anchored_candidates

        candidates.sort(
            key=lambda item: (
                item["match_score"],
                item["business_priority"],
                1 if item["has_sales_signal_30d"] else 0,
                item["current_review_count"] or 0,
            ),
            reverse=True,
        )
        truncated = len(candidates) > request.max_candidates
        candidate_items = candidates[: request.max_candidates]
        theme_extraction = normalization.theme_extraction
        recall_normalization = normalization.recall_normalization
        scoring_ms = int((time.perf_counter() - scoring_started_at) * 1000)

        return {
            "marketplace": marketplace,
            "domain": domain,
            "raw_product_query": request.product_query,
            "normalized_query": normalized_phrases[0] if normalized_phrases else _normalize_text(normalization.product_query),
            "query_phrases": normalized_phrases,
            "query_tokens": tokens,
            "query_expansions": query_expansions,
            "required_product_terms": required_product_terms,
            "ranking_policy": {
                "primary_sort": ["match_score", "business_priority", "has_sales_signal_30d", "current_review_count"],
                "sql_prefilter_sort": ["sql_prefilter_score", "business_priority", "current_review_count"],
                "semantic_gate": "required_product_terms must match high-signal product fields before a candidate can enter the final pool",
                "semantic_category_anchor": "when leaf/fine category anchors exist they are preferred; otherwise category/keyword anchored candidates exclude title-only matches",
                "match_score_components": ["phrase_score", "token_score", "business_score", "freshness_score", "completeness_score"],
                "matched_fields": ["product_title", "category", "category_path", "leaf_category_name", "fine_category_name", "keywords"],
                "note": "candidate filtering uses SQL lexical prefilter/coarse sort plus required product-term gating before Python exact scoring; product-specific core-vs-adjacent cleanup should use returned candidate_items fields or a downstream configurable classifier, not hard-coded product names",
            },
            "timing_ms": {
                "query_normalization": normalization_ms,
                "domain_candidate_fetch": pg_fetch_ms,
                "scoring_and_sorting": scoring_ms,
                "total": int((time.perf_counter() - request_started_at) * 1000),
            },
            "query_normalization": {
                "mode": normalization.normalization_mode,
                "llm_used": normalization.llm_used,
                "pipeline_mode": normalization.pipeline_mode,
                "pipeline_llm_used": bool(
                    (theme_extraction.llm_used if theme_extraction is not None else False)
                    or (recall_normalization.llm_used if recall_normalization is not None else False)
                ),
                "llm_provider": normalization.llm_provider,
                "llm_model": normalization.llm_model,
                "llm_language": normalization.llm_language,
                "llm_confidence": normalization.llm_confidence,
                "llm_error": normalization.llm_error,
                "normalized_product_query": normalization.product_query,
                "normalized_query_aliases": normalization.query_aliases,
                "normalized_category_hints": normalization.category_hints,
                "theme_extraction": {
                    "mode": theme_extraction.extraction_mode if theme_extraction is not None else None,
                    "llm_used": theme_extraction.llm_used if theme_extraction is not None else False,
                    "llm_provider": theme_extraction.llm_provider if theme_extraction is not None else None,
                    "llm_model": theme_extraction.llm_model if theme_extraction is not None else None,
                    "llm_language": theme_extraction.llm_language if theme_extraction is not None else None,
                    "llm_confidence": theme_extraction.llm_confidence if theme_extraction is not None else None,
                    "llm_error": theme_extraction.llm_error if theme_extraction is not None else None,
                    "extracted_theme": theme_extraction.extracted_theme if theme_extraction is not None else None,
                    "extracted_query_aliases": theme_extraction.query_aliases if theme_extraction is not None else [],
                    "extracted_category_hints": theme_extraction.category_hints if theme_extraction is not None else [],
                },
                "recall_normalization": {
                    "mode": recall_normalization.normalization_mode if recall_normalization is not None else None,
                    "llm_used": recall_normalization.llm_used if recall_normalization is not None else False,
                    "llm_provider": recall_normalization.llm_provider if recall_normalization is not None else None,
                    "llm_model": recall_normalization.llm_model if recall_normalization is not None else None,
                    "llm_language": recall_normalization.llm_language if recall_normalization is not None else None,
                    "llm_confidence": recall_normalization.llm_confidence if recall_normalization is not None else None,
                    "llm_error": recall_normalization.llm_error if recall_normalization is not None else None,
                    "normalized_product_query": (
                        recall_normalization.normalized_product_query if recall_normalization is not None else None
                    ),
                    "normalized_query_aliases": recall_normalization.query_aliases if recall_normalization is not None else [],
                    "normalized_category_hints": (
                        recall_normalization.category_hints if recall_normalization is not None else []
                    ),
                },
            },
            "candidate_count": len(candidate_items),
            "candidate_total_before_truncate": len(candidates),
            "candidate_total_before_semantic_category_anchor": candidate_total_before_semantic_category_anchor,
            "semantic_fine_category_anchor_applied": semantic_fine_category_anchor_applied,
            "semantic_category_anchor_applied": semantic_category_anchor_applied,
            "candidate_sql_prefilter_count": sql_prefilter_total_count,
            "candidate_sql_prefilter_limit": sql_prefilter_limit,
            "candidate_sql_prefilter_truncated": sql_prefilter_total_count > sql_prefilter_limit,
            "truncated": truncated,
            "matched_categories": _unique_nonempty([item["category"] for item in candidate_items])[:10],
            "matched_leaf_categories": _category_distribution(candidate_items, "leaf_category_name"),
            "matched_fine_categories": _category_distribution(candidate_items, "fine_category_name"),
            "matched_root_categories": _category_distribution(candidate_items, "root_category_name"),
            "matched_keywords": _unique_nonempty([keyword for item in candidate_items for keyword in item["keywords"]])[:20],
            "candidate_asins": [item["asin"] for item in candidate_items],
            "candidate_items": candidate_items,
            "recall_notes": [
                "when configured, recall preparation now runs as two internal stages: theme extraction first, then recall normalization",
                "the external resolve_candidates tool surface stays merged; only the internal service layer is split into extraction and normalization",
                "candidate pool is resolved by multi-field recall over title/category/category_path/search_term/keyword with required product-term gating",
                "SQL prefilter narrows the domain candidate set before Python exact scoring, so downstream tools should rely on returned candidate_asins rather than re-scanning visible titles",
                "leaf_category_name and fine_category_name are preferred for product analysis; root/L2/L3 categories can be too coarse for candidate cleanup",
                "built-in Chinese-to-English query expansion remains as a fallback bridge for common product terms when source data is English-dominant",
                "business_priority and recent data completeness are used as ranking boosters, not hard filters",
            ],
        }

    def get_candidate_pool_stats(self, request: CandidatePoolRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins = _sanitize_asins(request.candidate_asins)
        effective_window_days = _effective_feature_window_days(request.window_days)

        with _postgres_conn() as conn:
            combined_rows = _run_pg_dict_query(
                conn,
                """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_base_daily
                WHERE domain = %s
            ),
            filtered AS (
                SELECT *
                FROM serving.theme_base_daily
                WHERE domain = %s
                  AND asin = ANY(%s)
                  AND date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            ),
            latest_ranked AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    brand,
                    category,
                    effective_price,
                    rating,
                    review_count,
                    COALESCE(new_offer_count, 0) + COALESCE(used_offer_count, 0) AS offer_count,
                    bsr,
                    ROW_NUMBER() OVER (PARTITION BY asin, domain ORDER BY date DESC) AS rn
                FROM filtered
            ),
            latest AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    brand,
                    category,
                    effective_price,
                    rating,
                    review_count,
                    offer_count,
                    bsr
                FROM latest_ranked
                WHERE rn = 1
            ),
            asin_window AS (
                SELECT
                    asin,
                    domain,
                    SUM(COALESCE(estimated_daily_sales, 0)) AS sales_window_sum,
                    AVG(estimated_daily_sales) AS sales_daily_avg,
                    AVG(rating) AS rating_avg_window,
                    MAX(review_count) - MIN(review_count) AS review_growth_window,
                    AVG(COALESCE(new_offer_count, 0) + COALESCE(used_offer_count, 0)) AS offer_count_avg_window
                FROM filtered
                GROUP BY 1, 2
            ),
            combined AS (
                SELECT
                    aw.*,
                    l.product_title,
                    l.brand,
                    l.category,
                    l.effective_price AS latest_effective_price,
                    l.rating AS latest_rating,
                    l.review_count AS latest_review_count,
                    l.offer_count AS latest_offer_count,
                    l.bsr AS latest_bsr
                FROM asin_window aw
                LEFT JOIN latest l USING (asin, domain)
            )
            SELECT
                COUNT(*) AS candidate_count,
                SUM(sales_window_sum) AS sales_window_sum,
                AVG(sales_window_sum) AS sales_window_avg,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sales_window_sum) AS sales_window_median,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY latest_effective_price) AS price_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latest_effective_price) AS price_p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY latest_effective_price) AS price_p75,
                AVG(latest_rating) AS rating_avg,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latest_rating) AS rating_median,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latest_review_count) AS review_count_median,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latest_offer_count) AS offer_count_median,
                AVG(review_growth_window) AS review_growth_avg,
                (SELECT max_date FROM max_date) AS max_date
            FROM combined
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )
            top_brand_rows = _run_pg_dict_query(
                conn,
                """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_base_daily
                WHERE domain = %s
            ),
            latest_ranked AS (
                SELECT
                    asin,
                    brand,
                    category,
                    ROW_NUMBER() OVER (PARTITION BY asin, domain ORDER BY date DESC) AS rn
                FROM serving.theme_base_daily
                WHERE domain = %s
                  AND asin = ANY(%s)
                  AND date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            ),
            latest AS (
                SELECT asin, brand, category
                FROM latest_ranked
                WHERE rn = 1
            ),
            brand_agg AS (
                SELECT COALESCE(brand, 'UNKNOWN') AS name, COUNT(*) AS count
                FROM latest
                GROUP BY 1
                ORDER BY count DESC, name ASC
                LIMIT 10
            ),
            category_agg AS (
                SELECT COALESCE(category, 'UNKNOWN') AS name, COUNT(*) AS count
                FROM latest
                GROUP BY 1
                ORDER BY count DESC, name ASC
                LIMIT 10
            )
            SELECT 'brand' AS agg_type, name, count FROM brand_agg
            UNION ALL
            SELECT 'category' AS agg_type, name, count FROM category_agg
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )
            sales_forecast = self._build_candidate_pool_sales_forecast(conn, domain, candidate_asins)

        top_brand_list = [{"name": r["name"], "count": r["count"]} for r in top_brand_rows if r["agg_type"] == "brand"]
        top_category_list = [{"name": r["name"], "count": r["count"]} for r in top_brand_rows if r["agg_type"] == "category"]

        stats = combined_rows[0] if combined_rows else {}
        tier_distribution = self._fetch_business_tier_distribution(domain, candidate_asins)

        return {
            "marketplace": marketplace,
            "domain": domain,
            "window_days": effective_window_days,
            "candidate_count": int(stats.get("candidate_count") or 0),
            "sales_window_sum": round(float(stats.get("sales_window_sum") or 0.0), 2),
            "sales_window_avg": round(float(stats.get("sales_window_avg") or 0.0), 2),
            "sales_window_median": round(float(stats.get("sales_window_median") or 0.0), 2),
            "price_distribution": {
                "p25": round(float(stats.get("price_p25") or 0.0), 2) if stats.get("price_p25") is not None else None,
                "p50": round(float(stats.get("price_p50") or 0.0), 2) if stats.get("price_p50") is not None else None,
                "p75": round(float(stats.get("price_p75") or 0.0), 2) if stats.get("price_p75") is not None else None,
            },
            "rating_distribution": {
                "avg": round(float(stats.get("rating_avg") or 0.0), 2) if stats.get("rating_avg") is not None else None,
                "median": round(float(stats.get("rating_median") or 0.0), 2) if stats.get("rating_median") is not None else None,
            },
            "review_count_median": int(stats.get("review_count_median") or 0),
            "offer_count_median": round(float(stats.get("offer_count_median") or 0.0), 2),
            "review_growth_avg": round(float(stats.get("review_growth_avg") or 0.0), 2),
            "business_tier_distribution": tier_distribution,
            "top_brands": top_brand_list,
            "top_categories": top_category_list,
            "data_max_date": stats.get("max_date"),
            "sales_forecast": sales_forecast,
        }

    def get_candidate_pool_trends(self, request: CandidatePoolRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins = _sanitize_asins(request.candidate_asins)
        effective_window_days = _effective_feature_window_days(request.window_days)

        with _postgres_conn() as conn:
            rows = _run_pg_dict_query(
                conn,
                """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_trends_daily
                WHERE domain = %s
            ),
            filtered AS (
                SELECT *
                FROM serving.theme_trends_daily
                WHERE domain = %s
                  AND asin = ANY(%s)
                  AND date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            )
            SELECT
                COUNT(*) AS row_count,
                COUNT(*) FILTER (WHERE trend_index_mean IS NOT NULL) AS trend_rows,
                AVG(trend_index_mean) FILTER (WHERE date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_7d_mean,
                AVG(trend_index_mean) AS trend_30d_mean,
                AVG(trend_index_wow) FILTER (WHERE date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_wow,
                AVG(trend_index_dod) FILTER (WHERE date >= (SELECT max_date - INTERVAL '2 day' FROM max_date)) AS trend_dod,
                AVG(trend_index_roll_std_7) AS trend_volatility,
                MAX(trend_index_roll_max_7) AS trend_peak_recent,
                AVG(trend_keyword_coverage_ratio) AS keyword_coverage_ratio,
                (SELECT max_date FROM max_date) AS max_date
            FROM filtered
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )

        stats = rows[0] if rows else {}
        trend_rows = int(stats.get("trend_rows") or 0)
        row_count = int(stats.get("row_count") or 0)
        coverage = round(trend_rows / row_count, 4) if row_count else 0.0
        trend_wow = float(stats.get("trend_wow") or 0.0)
        trend_volatility = float(stats.get("trend_volatility") or 0.0)

        if coverage == 0:
            trend_stage = "no_signal"
        elif trend_wow >= 3:
            trend_stage = "rising"
        elif trend_wow <= -3:
            trend_stage = "cooling"
        elif trend_volatility >= 8:
            trend_stage = "volatile"
        else:
            trend_stage = "flat"

        return {
            "marketplace": marketplace,
            "domain": domain,
            "data_source": "google_trends",
            "source_table": "serving.theme_trends_daily",
            "window_days": effective_window_days,
            "trend_7d_mean": round(float(stats.get("trend_7d_mean") or 0.0), 2) if stats.get("trend_7d_mean") is not None else None,
            "trend_30d_mean": round(float(stats.get("trend_30d_mean") or 0.0), 2) if stats.get("trend_30d_mean") is not None else None,
            "trend_wow": round(trend_wow, 2),
            "trend_dod": round(float(stats.get("trend_dod") or 0.0), 2),
            "trend_volatility": round(trend_volatility, 2),
            "trend_peak_recent": round(float(stats.get("trend_peak_recent") or 0.0), 2) if stats.get("trend_peak_recent") is not None else None,
            "google_trends_7d_mean": round(float(stats.get("trend_7d_mean") or 0.0), 2) if stats.get("trend_7d_mean") is not None else None,
            "google_trends_30d_mean": round(float(stats.get("trend_30d_mean") or 0.0), 2) if stats.get("trend_30d_mean") is not None else None,
            "google_trends_wow": round(trend_wow, 2),
            "google_trends_dod": round(float(stats.get("trend_dod") or 0.0), 2),
            "keyword_coverage_ratio": round(float(stats.get("keyword_coverage_ratio") or 0.0), 4) if stats.get("keyword_coverage_ratio") is not None else None,
            "trend_data_coverage": coverage,
            "trend_stage": trend_stage,
            "data_max_date": stats.get("max_date"),
        }

    def get_candidate_pool_weak_forecast(self, request: WeakForecastRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins = _sanitize_asins(request.candidate_asins)
        effective_window_days = max(30, _effective_feature_window_days(request.window_days))

        with _postgres_conn() as conn:
            rows = _run_pg_dict_query(
                conn,
                """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_cross_daily
                WHERE domain = %s
            ),
            filtered AS (
                SELECT *
                FROM serving.theme_cross_daily
                WHERE domain = %s
                  AND asin = ANY(%s)
                  AND date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            ),
            latest_ranked AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    effective_price,
                    bsr,
                    rating,
                    review_count,
                    COALESCE(new_offer_count, 0) + COALESCE(used_offer_count, 0) AS offer_count,
                    ROW_NUMBER() OVER (PARTITION BY asin, domain ORDER BY date DESC) AS rn
                FROM filtered
            ),
            latest AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    effective_price,
                    bsr,
                    rating,
                    review_count,
                    offer_count
                FROM latest_ranked
                WHERE rn = 1
            ),
            signal AS (
                SELECT
                    asin,
                    domain,
                    AVG(estimated_daily_sales) FILTER (WHERE date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS sales_mean_7,
                    AVG(estimated_daily_sales) FILTER (WHERE date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS sales_mean_prev,
                    AVG(trend_index_mean) FILTER (WHERE date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_mean_7,
                    AVG(trend_index_mean) FILTER (WHERE date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_mean_prev,
                    AVG(bsr) FILTER (WHERE date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS bsr_mean_7,
                    AVG(bsr) FILTER (WHERE date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS bsr_mean_prev,
                    MAX(review_count) - MIN(review_count) AS review_growth_30,
                    AVG(price_discount_pct) AS avg_discount_pct
                FROM filtered
                GROUP BY 1, 2
            )
            SELECT
                l.asin,
                l.domain,
                l.product_title,
                l.effective_price,
                l.bsr,
                l.rating,
                l.review_count,
                l.offer_count,
                s.sales_mean_7,
                s.sales_mean_prev,
                s.trend_mean_7,
                s.trend_mean_prev,
                s.bsr_mean_7,
                s.bsr_mean_prev,
                s.review_growth_30,
                s.avg_discount_pct
            FROM latest l
            LEFT JOIN signal s USING (asin, domain)
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )

        predictions: list[dict[str, Any]] = []
        for row in rows:
            sales_mean_7 = float(row.get("sales_mean_7") or 0.0)
            sales_mean_prev = float(row.get("sales_mean_prev") or 0.0)
            trend_mean_7 = float(row.get("trend_mean_7") or 0.0)
            trend_mean_prev = float(row.get("trend_mean_prev") or 0.0)
            bsr_mean_7 = float(row.get("bsr_mean_7") or 0.0)
            bsr_mean_prev = float(row.get("bsr_mean_prev") or 0.0)
            review_growth_30 = float(row.get("review_growth_30") or 0.0)
            offer_count = float(row.get("offer_count") or 0.0)

            sales_momentum = ((sales_mean_7 - sales_mean_prev) / sales_mean_prev * 100.0) if sales_mean_prev > 0 else 0.0
            trend_momentum = ((trend_mean_7 - trend_mean_prev) / trend_mean_prev * 100.0) if trend_mean_prev > 0 else 0.0
            bsr_improvement = ((bsr_mean_prev - bsr_mean_7) / bsr_mean_prev * 100.0) if bsr_mean_prev > 0 else 0.0

            heuristic_score = (
                min(max(sales_momentum, -100.0), 100.0) * 0.35
                + min(max(trend_momentum, -100.0), 100.0) * 0.25
                + min(max(bsr_improvement, -100.0), 100.0) * 0.20
                + min(max(review_growth_30, 0.0), 200.0) * 0.05
                - min(offer_count, 50.0) * 0.5
            )
            reasons: list[str] = []
            if sales_momentum > 5:
                reasons.append("sales_momentum_positive")
            if trend_momentum > 3:
                reasons.append("trend_momentum_positive")
            if bsr_improvement > 3:
                reasons.append("bsr_improving")
            if review_growth_30 > 0:
                reasons.append("review_growth_positive")
            if offer_count > 20:
                reasons.append("competition_high")

            predictions.append(
                {
                    "asin": row["asin"],
                    "product_title": row.get("product_title"),
                    "heuristic_score": round(heuristic_score, 2),
                    "sales_momentum_pct": round(sales_momentum, 2),
                    "trend_momentum_pct": round(trend_momentum, 2),
                    "bsr_improvement_pct": round(bsr_improvement, 2),
                    "review_growth_30": round(review_growth_30, 2),
                    "offer_count": int(offer_count),
                    "effective_price": float(row.get("effective_price") or 0.0) if row.get("effective_price") is not None else None,
                    "reasons": reasons,
                }
            )

        predictions.sort(key=lambda item: item["heuristic_score"], reverse=True)
        top_predictions = predictions[: request.top_n]
        bullish_count = sum(1 for item in predictions if item["heuristic_score"] >= 10)
        risk_count = sum(1 for item in predictions if item["heuristic_score"] <= -5)

        opportunity_flags: list[str] = []
        risk_flags: list[str] = []
        if any(item["sales_momentum_pct"] > 10 for item in predictions):
            opportunity_flags.append("some_candidates_show_positive_sales_momentum")
        if any(item["trend_momentum_pct"] > 8 for item in predictions):
            opportunity_flags.append("topic_has_rising_trend_signal")
        if any(item["offer_count"] > 20 for item in predictions):
            risk_flags.append("part_of_the_pool_has_high_offer_competition")
        if not predictions:
            risk_flags.append("no_candidate_signals_available")

        return {
            "marketplace": marketplace,
            "domain": domain,
            "forecast_type": "heuristic_v1",
            "window_days": effective_window_days,
            "bullish_asin_count": bullish_count,
            "risk_asin_count": risk_count,
            "predicted_top_asins": top_predictions,
            "opportunity_flags": opportunity_flags,
            "risk_flags": risk_flags,
            "notes": [
                "this is a weak-signal heuristic forecast, not a trained online prediction model",
                "score combines sales momentum, trend momentum, bsr improvement, review growth, and offer competition",
            ],
        }

    def get_top_asin_drilldown(self, request: DrilldownRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins = _sanitize_asins(request.candidate_asins)
        if request.top_n is not None:
            candidate_asins = candidate_asins[: request.top_n]
        effective_window_days = _effective_feature_window_days(request.window_days)

        with _postgres_conn() as conn:
            rows = _run_pg_dict_query(
                conn,
                """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_base_daily
                WHERE domain = %s
            ),
            filtered AS (
                SELECT *
                FROM serving.theme_base_daily
                WHERE domain = %s
                  AND asin = ANY(%s)
                  AND date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            ),
            latest_ranked AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    brand,
                    category,
                    effective_price,
                    rating,
                    review_count,
                    COALESCE(new_offer_count, 0) + COALESCE(used_offer_count, 0) AS offer_count,
                    bsr,
                    estimated_daily_sales,
                    date,
                    ROW_NUMBER() OVER (PARTITION BY asin, domain ORDER BY date DESC) AS rn
                FROM filtered
            ),
            latest AS (
                SELECT
                    asin,
                    domain,
                    product_title,
                    brand,
                    category,
                    effective_price,
                    rating,
                    review_count,
                    offer_count,
                    bsr,
                    estimated_daily_sales,
                    date
                FROM latest_ranked
                WHERE rn = 1
            ),
            summary AS (
                SELECT
                    asin,
                    domain,
                    SUM(COALESCE(estimated_daily_sales, 0)) AS sales_window_sum,
                    AVG(estimated_daily_sales) AS sales_daily_avg,
                    MIN(effective_price) AS price_min_window,
                    MAX(effective_price) AS price_max_window,
                    MAX(review_count) - MIN(review_count) AS review_growth_window,
                    AVG(COALESCE(new_offer_count, 0) + COALESCE(used_offer_count, 0)) AS offer_count_avg_window,
                    AVG(bsr) AS bsr_avg_window
                FROM filtered
                GROUP BY 1, 2
            )
            SELECT
                l.asin,
                l.product_title,
                l.brand,
                l.category,
                l.effective_price,
                l.rating,
                l.review_count,
                l.offer_count,
                l.bsr,
                l.estimated_daily_sales,
                l.date AS latest_date,
                s.sales_window_sum,
                s.sales_daily_avg,
                s.price_min_window,
                s.price_max_window,
                s.review_growth_window,
                s.offer_count_avg_window,
                s.bsr_avg_window
            FROM latest l
            LEFT JOIN summary s USING (asin, domain)
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )
            sales_forecast_meta, forecast_items_by_asin = self._build_top_asin_sales_forecast_payload(
                conn,
                domain,
                candidate_asins,
            )

        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched_row = dict(row)
            asin = str(enriched_row.get("asin") or "")
            forecast_row = forecast_items_by_asin.get(asin)
            if forecast_row:
                sales_forecast = self._build_forecast_meta(forecast_row)
                sales_forecast.update(
                    {
                        "status": FORECAST_STATUS_READY,
                        "predicted_weekly_sales_w1": round(float(forecast_row["predicted_weekly_sales_w1"]), 2)
                        if forecast_row.get("predicted_weekly_sales_w1") is not None
                        else None,
                        "predicted_weekly_sales_w4": round(float(forecast_row["predicted_weekly_sales_w4"]), 2)
                        if forecast_row.get("predicted_weekly_sales_w4") is not None
                        else None,
                        "predicted_growth_ratio_w4_over_w1": round(float(forecast_row["predicted_growth_ratio_w4_over_w1"]), 4)
                        if forecast_row.get("predicted_growth_ratio_w4_over_w1") is not None
                        else None,
                        "predicted_growth_delta_w4_minus_w1": round(float(forecast_row["predicted_growth_delta_w4_minus_w1"]), 2)
                        if forecast_row.get("predicted_growth_delta_w4_minus_w1") is not None
                        else None,
                        "predicted_rank_w1_within_domain": int(forecast_row["predicted_rank_w1_within_domain"])
                        if forecast_row.get("predicted_rank_w1_within_domain") is not None
                        else None,
                        "predicted_rank_w4_within_domain": int(forecast_row["predicted_rank_w4_within_domain"])
                        if forecast_row.get("predicted_rank_w4_within_domain") is not None
                        else None,
                        "model_config_name_w1": forecast_row.get("model_config_name_w1"),
                        "model_config_name_w4": forecast_row.get("model_config_name_w4"),
                        "notes": [],
                    }
                )
                sales_forecast.update(self._build_item_sales_forecast_explainability(forecast_row))
            elif sales_forecast_meta["status"] == FORECAST_STATUS_MISSING_DOMAIN_MODEL:
                sales_forecast = self._build_item_sales_forecast_empty(
                    status=FORECAST_STATUS_MISSING_DOMAIN_MODEL,
                    notes=sales_forecast_meta.get("notes"),
                )
            elif sales_forecast_meta["status"] == FORECAST_STATUS_UNAVAILABLE:
                sales_forecast = self._build_item_sales_forecast_empty(
                    status=FORECAST_STATUS_UNAVAILABLE,
                    notes=sales_forecast_meta.get("notes"),
                )
            else:
                sales_forecast = self._build_item_sales_forecast_empty(
                    status=FORECAST_STATUS_MISSING_ASIN_PREDICTION,
                    coverage_row=sales_forecast_meta,
                    notes=["current domain has forecast coverage, but this ASIN is not in the current release"],
                )

            enriched_row["sales_forecast"] = sales_forecast
            enriched_rows.append(enriched_row)

        return {
            "marketplace": marketplace,
            "domain": domain,
            "window_days": effective_window_days,
            "sales_forecast_meta": sales_forecast_meta,
            "items": enriched_rows,
        }

    def get_asin_history_timeseries(self, request: AsinHistoryTimeseriesRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        asins = _sanitize_asins(request.asins)
        if not asins:
            raise HTTPException(status_code=400, detail="no valid ASINs provided")

        effective_window_days = min(request.window_days, 90)
        metrics = self._normalize_asin_history_metrics(request.metrics)
        with _postgres_conn() as conn:
            rows = self._fetch_asin_history_daily_rows(conn, domain, asins, effective_window_days)

        rows_by_asin: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            asin = str(row.get("asin") or "").strip()
            if not asin:
                continue
            rows_by_asin.setdefault(asin, []).append(row)

        missing_asins = [asin for asin in asins if asin not in rows_by_asin]
        keepa_snapshot_map: dict[str, dict[str, Any]] = {}
        if request.fallback_keepa_snapshot and missing_asins:
            keepa_snapshot_map = self._fallback_keepa_snapshot_for_missing_asins(missing_asins, marketplace)

        truncated_keepa_fallback = False
        if request.fallback_keepa_snapshot and len(missing_asins) > 20:
            truncated_keepa_fallback = True

        items = self._build_asin_history_items(
            rows=rows,
            asins=asins,
            interval=request.interval,
            metrics=metrics,
            window_days=effective_window_days,
            include_latest_snapshot=request.include_latest_snapshot,
            include_window_summary=request.include_window_summary,
            keepa_snapshot_map=keepa_snapshot_map,
        )

        notes: list[str] = []
        if request.interval == "week":
            notes.append("week interval is aggregated in Python from local daily rows")
        if missing_asins and not request.fallback_keepa_snapshot:
            notes.append("some ASINs have no local history and keepa fallback is disabled")
        elif keepa_snapshot_map:
            notes.append("missing local history ASINs use Keepa latest snapshot as fallback")
        if truncated_keepa_fallback:
            notes.append("Keepa fallback currently caps missing ASIN lookups to the first 20 ASINs")

        return {
            "marketplace": marketplace,
            "domain": domain,
            "window_days": effective_window_days,
            "interval": request.interval,
            "metrics": metrics,
            "source_preference": request.source_preference,
            "requested_asin_count": len(asins),
            "local_history_hit_count": len(asins) - len(missing_asins),
            "missing_local_history_asin_count": len(missing_asins),
            "fallback_keepa_snapshot": request.fallback_keepa_snapshot,
            "items": items,
            "notes": notes,
        }

    def _normalize_asin_history_metrics(self, metrics: list[str]) -> list[str]:
        allowed = {
            "effective_price",
            "rating",
            "review_count",
            "offer_count",
            "bsr",
            "estimated_daily_sales",
        }
        normalized: list[str] = []
        for metric in metrics or []:
            value = str(metric or "").strip().lower()
            if value in allowed and value not in normalized:
                normalized.append(value)
        if normalized:
            return normalized
        return ["estimated_daily_sales", "effective_price", "bsr", "review_count"]

    def _fetch_asin_history_daily_rows(self, conn, domain: int, asins: list[str], window_days: int) -> list[dict[str, Any]]:
        if not asins:
            return []
        return _run_pg_dict_query(
            conn,
            """
            WITH max_date AS (
                SELECT MAX(date) AS max_date
                FROM serving.theme_base_daily
                WHERE domain = %s
            ),
            filtered AS (
                SELECT
                    d.asin,
                    d.domain,
                    d.product_title,
                    d.brand,
                    COALESCE(NULLIF(d.category, ''), NULLIF(r.category, '')) AS category,
                    COALESCE(r.category_path, '') AS category_path,
                    d.effective_price,
                    d.rating,
                    d.review_count,
                    COALESCE(d.new_offer_count, 0) + COALESCE(d.used_offer_count, 0) AS offer_count,
                    d.bsr,
                    d.estimated_daily_sales,
                    d.date
                FROM serving.theme_base_daily d
                LEFT JOIN sync.keepa_asin_registry r
                    ON d.asin = r.asin AND d.domain = r.domain
                WHERE d.domain = %s
                  AND d.asin = ANY(%s)
                  AND d.date >= (
                      SELECT max_date - (%s * INTERVAL '1 day')
                      FROM max_date
                  )
            )
            SELECT
                asin,
                product_title,
                brand,
                category,
                category_path,
                effective_price,
                rating,
                review_count,
                offer_count,
                bsr,
                estimated_daily_sales,
                date
            FROM filtered
            ORDER BY asin, date ASC
            """,
            [domain, domain, asins, window_days - 1],
        )

    def _build_asin_history_items(
        self,
        rows: list[dict[str, Any]],
        asins: list[str],
        interval: str,
        metrics: list[str],
        window_days: int,
        include_latest_snapshot: bool,
        include_window_summary: bool,
        keepa_snapshot_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows_by_asin: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            asin = str(row.get("asin") or "").strip()
            if not asin:
                continue
            rows_by_asin.setdefault(asin, []).append(row)

        items: list[dict[str, Any]] = []
        for asin in asins:
            asin_rows = rows_by_asin.get(asin) or []
            if not asin_rows:
                item: dict[str, Any] = {
                    "asin": asin,
                    "history_status": "no_local_history",
                    "series": [],
                }
                if include_latest_snapshot:
                    item["latest_snapshot"] = keepa_snapshot_map.get(asin)
                if include_window_summary:
                    item["window_summary"] = None
                if asin in keepa_snapshot_map:
                    item["notes"] = ["latest snapshot is from Keepa fallback because no local history was found"]
                items.append(item)
                continue

            series_rows = asin_rows
            if interval == "week":
                series_rows = self._group_asin_history_weekly(asin_rows)

            item = {
                "asin": asin,
                "history_status": "ready",
                "series": [self._build_asin_history_series_row(row, metrics, interval) for row in series_rows],
            }
            if include_latest_snapshot:
                item["latest_snapshot"] = self._build_asin_history_latest_snapshot(asin_rows[-1])
            if include_window_summary:
                item["window_summary"] = self._build_asin_history_window_summary(asin_rows, interval, window_days)
            items.append(item)
        return items

    def _group_asin_history_weekly(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for row in rows:
            asin = str(row.get("asin") or "").strip()
            row_date = row.get("date")
            if not asin or row_date is None or not hasattr(row_date, "isocalendar"):
                continue
            iso_year, iso_week, _iso_weekday = row_date.isocalendar()
            grouped.setdefault((asin, iso_year, iso_week), []).append(row)

        weekly_rows: list[dict[str, Any]] = []
        for (_asin, iso_year, iso_week), bucket in grouped.items():
            ordered_bucket = sorted(bucket, key=lambda item: item.get("date"))
            latest_row = ordered_bucket[-1]

            def _avg(field_name: str) -> float | None:
                values = [float(item[field_name]) for item in ordered_bucket if item.get(field_name) is not None]
                if not values:
                    return None
                return round(sum(values) / len(values), 2)

            weekly_rows.append(
                {
                    "asin": latest_row.get("asin"),
                    "product_title": latest_row.get("product_title"),
                    "brand": latest_row.get("brand"),
                    "category": latest_row.get("category"),
                    "category_path": latest_row.get("category_path"),
                    "effective_price": latest_row.get("effective_price"),
                    "rating": latest_row.get("rating"),
                    "review_count": latest_row.get("review_count"),
                    "offer_count": _avg("offer_count"),
                    "bsr": _avg("bsr"),
                    "estimated_daily_sales": _avg("estimated_daily_sales"),
                    "date": latest_row.get("date"),
                    "iso_year_week": "%04d-W%02d" % (iso_year, iso_week),
                }
            )

        weekly_rows.sort(key=lambda item: (str(item.get("asin") or ""), item.get("date") or datetime.min.date()))
        return weekly_rows

    def _fallback_keepa_snapshot_for_missing_asins(self, asins: list[str], marketplace: str | int) -> dict[str, dict[str, Any]]:
        missing = _sanitize_asins(asins)
        if not missing:
            return {}
        capped_asins = missing[:20]
        payload = self.keepa_asin_lookup(KeepaAsinLookupRequest(asins=capped_asins, marketplace=marketplace))
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return {}
        return {
            str(item.get("asin") or "").strip(): {
                "asin": item.get("asin"),
                "product_title": item.get("product_title"),
                "brand": item.get("brand"),
                "category": item.get("category"),
                "effective_price": item.get("effective_price"),
                "rating": item.get("rating"),
                "review_count": item.get("review_count"),
                "offer_count": item.get("offer_count"),
                "bsr": item.get("bsr"),
                "estimated_daily_sales": item.get("estimated_daily_sales"),
                "latest_date": item.get("latest_date"),
                "source": "keepa_api",
            }
            for item in items
            if str(item.get("asin") or "").strip()
        }

    def _build_asin_history_series_row(self, row: dict[str, Any], metrics: list[str], interval: str) -> dict[str, Any]:
        series_row = {
            "asin": row.get("asin"),
            "date": self._format_asin_history_date(row.get("date")),
        }
        if interval == "week" and row.get("iso_year_week"):
            series_row["iso_year_week"] = row.get("iso_year_week")
        for metric in metrics:
            series_row[metric] = row.get(metric)
        return series_row

    def _build_asin_history_latest_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "asin": row.get("asin"),
            "product_title": row.get("product_title"),
            "brand": row.get("brand"),
            "category": row.get("category"),
            "category_path": row.get("category_path"),
            "l3_category_name": self._extract_l3_category_name(row.get("category_path"), row.get("category")),
            "leaf_category_name": self._extract_leaf_category_name(row.get("category_path"), row.get("category")),
            "effective_price": row.get("effective_price"),
            "rating": row.get("rating"),
            "review_count": row.get("review_count"),
            "offer_count": row.get("offer_count"),
            "bsr": row.get("bsr"),
            "estimated_daily_sales": row.get("estimated_daily_sales"),
            "latest_date": self._format_asin_history_date(row.get("date")),
            "source": "local_theme_base_daily",
        }

    def _extract_l3_category_name(self, category_path: Any, fallback_category: Any = None) -> str | None:
        value = _category_level_name(category_path, 2)
        if value:
            return value
        leaf = _leaf_category_name(category_path)
        if leaf:
            return leaf
        fallback = str(fallback_category or "").strip()
        return fallback or None

    def _extract_leaf_category_name(self, category_path: Any, fallback_category: Any = None) -> str | None:
        return _leaf_category_name(category_path, fallback_category)

    def _build_asin_history_window_summary(self, rows: list[dict[str, Any]], interval: str, window_days: int) -> dict[str, Any] | None:
        if not rows:
            return None

        sales_values = [float(row["estimated_daily_sales"]) for row in rows if row.get("estimated_daily_sales") is not None]
        price_values = [float(row["effective_price"]) for row in rows if row.get("effective_price") is not None]
        offer_values = [float(row["offer_count"]) for row in rows if row.get("offer_count") is not None]
        bsr_values = [float(row["bsr"]) for row in rows if row.get("bsr") is not None]
        first_review = next((row.get("review_count") for row in rows if row.get("review_count") is not None), None)
        last_review = next((row.get("review_count") for row in reversed(rows) if row.get("review_count") is not None), None)
        expected_rows = window_days if interval == "day" else max(1, (window_days + 6) // 7)
        if interval == "week":
            grouped_rows = self._group_asin_history_weekly(rows)
            series_row_count = len(grouped_rows)
        else:
            series_row_count = len(rows)

        return {
            "sales_window_sum": round(sum(sales_values), 2) if sales_values else None,
            "sales_daily_avg": round(sum(sales_values) / len(sales_values), 2) if sales_values else None,
            "price_min_window": round(min(price_values), 2) if price_values else None,
            "price_max_window": round(max(price_values), 2) if price_values else None,
            "review_growth_window": int(last_review) - int(first_review) if last_review is not None and first_review is not None else None,
            "offer_count_avg_window": round(sum(offer_values) / len(offer_values), 2) if offer_values else None,
            "bsr_avg_window": round(sum(bsr_values) / len(bsr_values), 2) if bsr_values else None,
            "series_row_count": series_row_count,
            "coverage_ratio": round(min(len(rows) / float(max(expected_rows, 1)), 1.0), 4),
        }

    def _format_asin_history_date(self, value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)

    def get_category_benchmark(self, request: CandidatePoolRequest) -> dict[str, Any]:
        """Return L3-level category benchmark stats for comparison with candidate pool.

        Strategy:
        1. Get each candidate ASIN's leaf category_id from keepa_asin_registry
        2. Walk up parent_id chain in keepa_category_registry to find L3 ancestor (depth=3)
        3. Pick the dominant L3 category (mode) as the benchmark anchor
        4. Aggregate all ASINs whose category_id descends from that L3 node
        """
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins = _sanitize_asins(request.candidate_asins)
        effective_window_days = min(request.window_days, 90)

        with _postgres_conn() as conn:
            # ── Step 1+2: resolve each candidate ASIN's L3 ancestor category ──
            l3_rows = _run_pg_dict_query(
                conn,
                """
                WITH RECURSIVE
                candidate_leaf AS (
                    SELECT asin, category_id
                    FROM sync.keepa_asin_registry
                    WHERE domain = %s
                      AND asin = ANY(%s)
                      AND category_id IS NOT NULL
                ),
                -- walk up category tree to find L3 ancestor for each leaf
                ancestors AS (
                    SELECT
                        cl.asin,
                        c.category_id,
                        c.parent_id,
                        c.depth,
                        c.category_en,
                        c.category_cn
                    FROM candidate_leaf cl
                    JOIN sync.keepa_category_registry c
                        ON cl.category_id = c.category_id AND c.domain = %s

                    UNION ALL

                    SELECT
                        a.asin,
                        p.category_id,
                        p.parent_id,
                        p.depth,
                        p.category_en,
                        p.category_cn
                    FROM ancestors a
                    JOIN sync.keepa_category_registry p
                        ON a.parent_id = p.category_id AND p.domain = %s
                    WHERE a.depth > 3
                )
                SELECT
                    asin,
                    category_id AS l3_category_id,
                    COALESCE(category_cn, category_en, 'Unknown') AS l3_category_name,
                    category_en AS l3_category_en,
                    depth
                FROM ancestors
                WHERE depth = 3

                UNION ALL

                -- fallback: if leaf depth <= 3, use itself as the best available level
                SELECT
                    cl.asin,
                    c.category_id AS l3_category_id,
                    COALESCE(c.category_cn, c.category_en, 'Unknown') AS l3_category_name,
                    c.category_en AS l3_category_en,
                    c.depth
                FROM candidate_leaf cl
                JOIN sync.keepa_category_registry c
                    ON cl.category_id = c.category_id AND c.domain = %s
                WHERE c.depth <= 3
                  AND cl.asin NOT IN (
                      SELECT asin FROM ancestors WHERE depth = 3
                  )
                """,
                [domain, candidate_asins, domain, domain, domain],
            )

        if not l3_rows:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "window_days": request.window_days,
                "benchmark_category": None,
                "benchmark_category_level": None,
                "candidate_asin_count_in_category": 0,
                "category_total_asin_count": 0,
                "candidate_category_coverage_pct": 0,
                "all_candidate_l3_categories": [],
                "benchmark_stats": {},
                "notes": ["未能从候选池 ASIN 的类目信息中解析出 L3 类目"],
            }

        # ── Step 3: pick dominant L3 category (mode) ──
        from collections import Counter
        l3_counter: Counter[int] = Counter()
        l3_name_map: dict[int, str] = {}
        l3_en_map: dict[int, str] = {}
        l3_depth_map: dict[int, int] = {}
        for row in l3_rows:
            cat_id = int(row["l3_category_id"])
            l3_counter[cat_id] += 1
            l3_name_map[cat_id] = row["l3_category_name"]
            l3_en_map[cat_id] = row.get("l3_category_en") or ""
            l3_depth_map[cat_id] = int(row["depth"])

        dominant_l3_id, dominant_count = l3_counter.most_common(1)[0]
        dominant_l3_name = l3_name_map[dominant_l3_id]
        dominant_l3_en = l3_en_map[dominant_l3_id]
        dominant_depth = l3_depth_map[dominant_l3_id]

        # all L3 categories for transparency
        all_l3_cats = [
            {
                "l3_category_id": cat_id,
                "l3_category_name": l3_name_map[cat_id],
                "l3_category_en": l3_en_map[cat_id],
                "depth": l3_depth_map[cat_id],
                "candidate_asin_count": count,
            }
            for cat_id, count in l3_counter.most_common()
        ]

        # ── Step 4: find all ASINs that descend from the dominant L3 category ──
        # Use recursive CTE to get all descendant category_ids under dominant L3
        with _postgres_conn() as conn:
            bench_rows = _run_pg_dict_query(
                conn,
                """
                WITH RECURSIVE
                subtree AS (
                    SELECT category_id
                    FROM sync.keepa_category_registry
                    WHERE category_id = %s AND domain = %s

                    UNION ALL

                    SELECT c.category_id
                    FROM sync.keepa_category_registry c
                    JOIN subtree s ON c.parent_id = s.category_id
                    WHERE c.domain = %s
                ),
                category_asins AS (
                    SELECT r.asin
                    FROM sync.keepa_asin_registry r
                    WHERE r.domain = %s
                      AND r.category_id IN (SELECT category_id FROM subtree)
                ),
                latest_history AS (
                    SELECT * FROM (
                        SELECT
                            h.asin,
                            COALESCE(h.buy_box_price, h.amazon_price, h.new_price) AS effective_price,
                            h.bsr,
                            h.rating,
                            h.review_count,
                            h.monthly_sold,
                            COALESCE(h.new_offer_count, 0) + COALESCE(h.used_offer_count, 0) AS offer_count,
                            ROW_NUMBER() OVER (PARTITION BY h.asin ORDER BY h.date DESC) AS rn
                        FROM sync.keepa_product_history h
                        JOIN category_asins ca ON h.asin = ca.asin
                        WHERE h.domain = %s
                          AND h.date >= CURRENT_DATE - (%s * INTERVAL '1 day')
                    ) ranked
                    WHERE rn = 1
                )
                SELECT
                    COUNT(DISTINCT asin) AS category_total_asin_count,
                    AVG(effective_price) AS avg_price,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY effective_price) AS price_p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY effective_price) AS price_p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY effective_price) AS price_p75,
                    AVG(rating) AS avg_rating,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY rating) AS median_rating,
                    AVG(review_count) AS avg_review_count,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY review_count) AS median_review_count,
                    AVG(bsr) AS avg_bsr,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY bsr) AS median_bsr,
                    SUM(COALESCE(monthly_sold, 0)) AS sum_monthly_sold,
                    AVG(monthly_sold) AS avg_monthly_sold,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY offer_count) AS median_offer_count
                FROM latest_history
                """,
                [dominant_l3_id, domain, domain, domain, domain, effective_window_days],
            )

        def _safe_round(val: Any, decimals: int = 2) -> float | None:
            return round(float(val), decimals) if val is not None else None

        def _safe_int(val: Any) -> int | None:
            return int(val) if val is not None else None

        stats = bench_rows[0] if bench_rows else {}
        cat_total = int(stats.get("category_total_asin_count") or 0)

        benchmark_stats = {
            "avg_price": _safe_round(stats.get("avg_price")),
            "price_distribution": {
                "p25": _safe_round(stats.get("price_p25")),
                "p50": _safe_round(stats.get("price_p50")),
                "p75": _safe_round(stats.get("price_p75")),
            },
            "rating_distribution": {
                "avg": _safe_round(stats.get("avg_rating")),
                "median": _safe_round(stats.get("median_rating")),
            },
            "avg_review_count": _safe_round(stats.get("avg_review_count")),
            "median_review_count": _safe_int(stats.get("median_review_count")),
            "bsr_distribution": {
                "avg": _safe_round(stats.get("avg_bsr")),
                "median": _safe_int(stats.get("median_bsr")),
            },
            "sum_monthly_sold": _safe_round(stats.get("sum_monthly_sold")),
            "avg_monthly_sold": _safe_round(stats.get("avg_monthly_sold")),
            "median_offer_count": _safe_round(stats.get("median_offer_count")),
        }

        return {
            "marketplace": marketplace,
            "domain": domain,
            "window_days": effective_window_days,
            "benchmark_category": {
                "category_id": dominant_l3_id,
                "category_name": dominant_l3_name,
                "category_en": dominant_l3_en,
                "depth": dominant_depth,
                "level": f"L{dominant_depth}",
            },
            "benchmark_category_level": f"L{dominant_depth}",
            "candidate_asin_count_in_category": dominant_count,
            "category_total_asin_count": cat_total,
            "candidate_category_coverage_pct": round(
                dominant_count / cat_total * 100, 2
            ) if cat_total > 0 else 0,
            "all_candidate_l3_categories": all_l3_cats,
            "benchmark_stats": benchmark_stats,
            "notes": [
                f"对标类目由候选池 ASIN 的众数 L{dominant_depth} 类目自动选取",
                f"候选池中 {dominant_count}/{len(candidate_asins)} 个 ASIN 属于此类目",
                "聚合范围包含该 L3 类目及其所有子类目下的全部 ASIN",
                *( ["当前 benchmark 读取 PostgreSQL sync.keepa_product_history，在线窗口上限为近 90 天"] if request.window_days > effective_window_days else [] ),
            ],
        }

    def _fetch_business_tier_distribution(self, domain: int, candidate_asins: list[str]) -> dict[str, int]:
                with _postgres_conn() as conn:
                        rows = _run_pg_dict_query(
                conn,
                                """
                SELECT COALESCE(business_tier, 'UNKNOWN') AS business_tier, COUNT(*) AS count
                                FROM sync.keepa_asin_registry
                                WHERE domain = %s
                                    AND asin = ANY(%s)
                GROUP BY 1
                ORDER BY count DESC, business_tier ASC
                                """,
                                [domain, candidate_asins],
            )
                return {str(row["business_tier"]): int(row["count"]) for row in rows}

    def keepa_asin_lookup(self, request: KeepaAsinLookupRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        asins = _sanitize_asins(request.asins)
        if not asins:
            raise HTTPException(status_code=400, detail="no valid ASINs provided")
        if len(asins) > 20:
            asins = asins[:20]

        keepa_api_key = os.environ.get("KEEPA_API_KEY", "").strip()
        if not keepa_api_key:
            raise HTTPException(status_code=500, detail="KEEPA_API_KEY not configured")

        keepa_base_url = os.environ.get("KEEPA_BASE_URL", "https://api.keepa.com/product").strip()
        keepa_timeout = int(os.environ.get("KEEPA_TIMEOUT", "30"))
        stats_window = 30

        try:
            resp = http_requests.get(
                keepa_base_url,
                params={
                    "key": keepa_api_key,
                    "domain": domain,
                    "asin": ",".join(asins),
                    "history": 1,
                    "stats": stats_window,
                    "rating": 1,
                },
                timeout=keepa_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except http_requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Keepa API request failed: {exc}")

        products = payload.get("products") or []
        if not products:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "source": "keepa_api",
                "tokens_left": payload.get("tokensLeft"),
                "items": [],
                "notes": ["Keepa API 未返回任何商品数据，请检查 ASIN 是否正确"],
            }

        is_yen = domain in {5}
        items: list[dict[str, Any]] = []

        for product in products:
            csv = product.get("csv") or []
            stats = product.get("stats") or {}
            stats_current = stats.get("current") or []

            category = product.get("productGroup")
            cat_tree = product.get("categoryTree") or []
            if cat_tree:
                names = [n.get("name", "") for n in cat_tree if n.get("name")]
                if names:
                    category = names[-1]

            effective_price = _keepa_latest_price(csv, 18, is_yen)
            if effective_price is None:
                effective_price = _keepa_latest_price(csv, 0, is_yen)
            if effective_price is None:
                effective_price = _keepa_latest_price(csv, 1, is_yen)

            rating_raw = _keepa_latest_value(csv, 16)
            rating = round(rating_raw / 10, 1) if rating_raw is not None and rating_raw > 0 else None

            review_count = _keepa_latest_value(csv, 17)
            new_count = _keepa_latest_value(csv, 11) or 0
            used_count = _keepa_latest_value(csv, 12) or 0
            offer_count = (new_count or 0) + (used_count or 0)

            bsr = _keepa_latest_value(csv, 3)

            monthly_sold = product.get("monthlySold")
            est_daily_sales = round(monthly_sold / 30, 1) if monthly_sold and monthly_sold > 0 else None

            stats_avg = stats.get("avg") or []
            sales_window_sum = None
            sales_daily_avg = None
            if est_daily_sales is not None:
                sales_window_sum = round(est_daily_sales * stats_window, 1)
                sales_daily_avg = est_daily_sales

            price_min_window = _keepa_stats_price(stats, "min", 18, is_yen)
            price_max_window = _keepa_stats_price(stats, "max", 18, is_yen)

            bsr_avg_raw = _keepa_stats_value(stats_avg, 3)
            bsr_avg_window = round(float(bsr_avg_raw), 1) if bsr_avg_raw is not None else None

            items.append({
                "asin": product.get("asin"),
                "product_title": product.get("title"),
                "brand": product.get("brand"),
                "category": category,
                "effective_price": effective_price,
                "rating": rating,
                "review_count": int(review_count) if review_count is not None else None,
                "offer_count": offer_count if offer_count > 0 else None,
                "bsr": int(bsr) if bsr is not None else None,
                "estimated_daily_sales": est_daily_sales,
                "latest_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "sales_window_sum": sales_window_sum,
                "sales_daily_avg": sales_daily_avg,
                "price_min_window": price_min_window,
                "price_max_window": price_max_window,
                "review_growth_window": None,
                "offer_count_avg_window": None,
                "bsr_avg_window": bsr_avg_window,
            })

        return {
            "marketplace": marketplace,
            "domain": domain,
            "source": "keepa_api",
            "tokens_left": payload.get("tokensLeft"),
            "items": items,
            "notes": [
                "数据直接来自 Keepa API 实时查询，非本地数据库缓存",
                f"estimated_daily_sales 由 monthlySold / 30 估算",
                f"当本地数据库查不到 ASIN 时可用此工具作为补充数据源",
            ],
        }


app = FastAPI(title="xiamimate Product Theme API", version="2026-04-10")
service = ProductThemeService()


@app.on_event("startup")
def warmup_connection_pools() -> None:
    """Pre-load PostgreSQL pool and serving metadata to reduce cold-start latency."""
    import time as _time
    t0 = _time.monotonic()
    try:
        ensure_env_api_key_registered()
    except Exception as exc:
        print(f"[startup] api key bootstrap skipped: {exc}")
    try:
        with _postgres_conn() as conn:
            _run_pg_dict_query(conn, "SELECT 1 AS ok")
    except Exception:
        pass
    try:
        _get_theme_feature_serving_status(include_data_max_date=False)
    except Exception:
        pass
    elapsed = _time.monotonic() - t0
    print(f"[startup] connection warmup completed in {elapsed:.1f}s")


@app.middleware("http")
async def metered_api_key_middleware(request: Request, call_next):
    endpoint = request.url.path
    if not endpoint.startswith(PROTECTED_API_PREFIX):
        return await call_next(request)

    supplied_api_key = (request.headers.get(API_KEY_HEADER_NAME) or "").strip()
    if not supplied_api_key:
        supplied_api_key = (_extract_bearer_token(request.headers.get("Authorization")) or "").strip()
    if not supplied_api_key:
        return _auth_error_response(
            endpoint=endpoint,
            status_code=401,
            code="UNAUTHORIZED",
            message="missing api key",
        )

    api_key_record = resolve_api_key(supplied_api_key)
    if api_key_record is None:
        return _auth_error_response(
            endpoint=endpoint,
            status_code=401,
            code="UNAUTHORIZED",
            message="invalid api key",
        )
    if api_key_record.status != "active":
        return _auth_error_response(
            endpoint=endpoint,
            status_code=403,
            code="API_KEY_INACTIVE",
            message="api key is inactive",
        )

    request.state.api_key_record = api_key_record
    import time as _mw_time
    t0 = _mw_time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((_mw_time.monotonic() - t0) * 1000)
    record_api_usage(api_key_record.key_id, endpoint, response.status_code, response_time_ms=elapsed_ms)
    return response


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    endpoint = request.url.path
    code = "UNAUTHORIZED" if exc.status_code == 401 else "REQUEST_ERROR"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_response(endpoint=endpoint, code=code, message=str(exc.detail)),
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_error_response(endpoint=request.url.path, code="INTERNAL_ERROR", message=str(exc)),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    active_key_count = get_active_key_count()
    query_normalizer = _get_query_normalizer_config()
    feature_serving = _get_theme_feature_serving_status(include_data_max_date=False)
    return _success_response(
        endpoint="/health",
        message="service is healthy",
        data={
            "status": "ok",
            "online_store": {
                "type": "postgresql",
                "schemas": ["sync", "serving"],
                "host": os.environ.get("PG_HOST", "localhost"),
                "port": int(os.environ.get("PG_PORT", "5432")),
                "dbname": os.environ.get("PG_DB", "xiamimate"),
            },
            "theme_feature_serving": feature_serving,
            "offline_feature_artifacts": {
                "dir": str(DEFAULT_FEATURE_DIR),
                "role": "offline_training_and_feature_build_outputs",
            },
            "auth": {
                "mode": "pg_api_key_auth_with_usage_audit",
                "header_name": API_KEY_HEADER_NAME,
                "client_env_var": API_KEY_ENV_VAR,
                "bootstrap_env_var": API_KEY_NAME_ENV_VAR,
                "active_key_count": active_key_count,
                "configured": active_key_count > 0,
                "quota_model": "disabled_managed_by_chat_backend",
                "enforcement": "api_key_presence_and_status_only",
                "bootstrap": "startup_auto_register_from_env_when_present",
            },
            "query_normalizer": {
                "active_profile": query_normalizer["active_profile"],
                "mode": query_normalizer["mode"],
                "enabled": query_normalizer["enabled"],
                "configured": query_normalizer["configured"],
                "base_url": query_normalizer["base_url"],
                "model": query_normalizer["model"],
                "timeout_seconds": query_normalizer["timeout_seconds"],
                "env_file_autoload": str(ROOT_ENV_FILE),
            },
        },
    )


@app.post("/api/product-theme/resolve-candidates")
async def resolve_candidates(request: ResolveCandidatesRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/resolve-candidates",
        message="candidate pool resolved",
        data=await service.resolve_candidates(request),
    )


@app.post("/api/product-theme/candidate-pool-stats")
def candidate_pool_stats(request: CandidatePoolRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/candidate-pool-stats",
        message="candidate pool stats ready",
        data=service.get_candidate_pool_stats(request),
    )


@app.post("/api/product-theme/candidate-pool-trends")
def candidate_pool_trends(request: CandidatePoolRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/candidate-pool-trends",
        message="candidate pool trends ready",
        data=service.get_candidate_pool_trends(request),
    )


@app.post("/api/product-theme/candidate-pool-weak-forecast")
def candidate_pool_weak_forecast(request: WeakForecastRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/candidate-pool-weak-forecast",
        message="candidate pool weak forecast ready",
        data=service.get_candidate_pool_weak_forecast(request),
    )


@app.post("/api/product-theme/top-asin-drilldown")
def top_asin_drilldown(request: DrilldownRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/top-asin-drilldown",
        message="top asin drilldown ready",
        data=service.get_top_asin_drilldown(request),
    )


@app.post("/api/product-theme/asin-history-timeseries")
def asin_history_timeseries(request: AsinHistoryTimeseriesRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/asin-history-timeseries",
        message="asin history timeseries loaded",
        data=service.get_asin_history_timeseries(request),
    )


@app.post("/api/product-theme/category-benchmark")
def category_benchmark(request: CandidatePoolRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/category-benchmark",
        message="category benchmark ready",
        data=service.get_category_benchmark(request),
    )


@app.post("/api/product-theme/keepa-asin-lookup")
def keepa_asin_lookup(request: KeepaAsinLookupRequest) -> dict[str, Any]:
    return _success_response(
        endpoint="/api/product-theme/keepa-asin-lookup",
        message="keepa asin lookup ready",
        data=service.keepa_asin_lookup(request),
    )
