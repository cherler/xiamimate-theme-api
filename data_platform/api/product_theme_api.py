from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import re
from statistics import median
import time
from typing import Any
import uuid

import requests as http_requests

from fastapi import HTTPException

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

from data_platform.llm_client import ROOT_ENV_FILE, load_env_file_if_present
from data_platform.product_query_assistant import ProductRecallQueryAssistant

from data_platform.api.seller_scope import SellerScopeDecision, evaluate_seller_scope
from data_platform.api.product_theme.app import create_product_theme_app
from data_platform.api.product_theme.candidate_scoring import _bounded_score, _confidence_rank, _price_fit_score
from data_platform.api.product_theme.category_utils import (
    _build_candidate_pool_quality,
    _category_distribution,
    _category_level_name,
    _fine_category_name,
    _leaf_category_name,
    _opportunity_title_from_category_path,
)
from data_platform.api.product_theme.candidate_matching import (
    _build_query_variants,
    _build_required_product_terms,
    _build_sql_prefilter_terms,
    _candidate_field_matches_required_terms,
    _match_list_contains,
    _sanitize_asins,
    _score_category_match,
    _text_contains_token,
    _text_contains_token_variant,
    _unique_nonempty,
)
from data_platform.api.product_theme.constants import (
    CANDIDATE_EXPANSION_ACTIVE_STATUSES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_MIN_CANDIDATE_POOL_SIZE,
    DEFAULT_TARGET_CANDIDATE_POOL_SIZE,
    DOMAIN_TO_MARKETPLACE,
    FORECAST_HIGH_GROWTH_RATIO_THRESHOLD,
    FORECAST_STATUS_MISSING_ASIN_PREDICTION,
    FORECAST_STATUS_MISSING_DOMAIN_MODEL,
    FORECAST_STATUS_PARTIAL_COVERAGE,
    FORECAST_STATUS_READY,
    FORECAST_STATUS_UNAVAILABLE,
    FORECAST_TOP_ASINS_LIMIT,
    MARKETPLACE_TO_DOMAIN,
    OPPORTUNITY_SCORE_WEIGHTS,
    PROJECT_ROOT,
    THEME_FORECAST_SERVING_TABLES,
)
from data_platform.api.product_theme.db import _postgres_conn, _run_pg_dict_query
from data_platform.api.product_theme.feature_serving import (
    _effective_feature_window_days,
    _get_theme_feature_serving_status,
    _iso_date_or_none,
)
from data_platform.api.product_theme.keepa_utils import (
    _keepa_latest_price,
    _keepa_latest_value,
    _keepa_stats_price,
    _keepa_stats_value,
)
from data_platform.api.product_theme.query_utils import (
    _normalize_marketplace,
    _normalize_text,
    _safe_float,
    _safe_int,
    _tokenize_phrase,
)
from data_platform.api.product_theme.response_contract import _success_response
from data_platform.api.product_theme.services.launch_budget import calculate_launch_budget
from data_platform.api.product_theme.schemas import (
    AmazonKeywordDemandRequest,
    AsinHistoryTimeseriesRequest,
    AsinReviewInsightsRequest,
    CandidateExpansionJobRequest,
    CandidateExpansionJobStatusRequest,
    CandidatePoolRequest,
    CandidatePoolSliceRequest,
    CategoryBenchmarkRequest,
    CategoryResolveRequest,
    DrilldownRequest,
    KeepaAsinLookupRequest,
    LaunchBudgetCalculatorRequest,
    OpportunityDiscoveryJobStatusRequest,
    OpportunityDiscoveryRequest,
    ProductForecastExplainRequest,
    ResolveCandidatesRequest,
    WeakForecastRequest,
)


load_env_file_if_present(ROOT_ENV_FILE)
QUERY_ASSISTANT = ProductRecallQueryAssistant(env_prefix="THEME_QUERY_NORMALIZER")
LOGGER = logging.getLogger(__name__)


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


class ProductThemeService:
    def _build_candidate_pool_lineage(
        self,
        *,
        request: ResolveCandidatesRequest,
        marketplace: str,
        domain: int,
        normalized_query: str,
        normalized_phrases: list[str],
        tokens: list[str],
        query_expansions: list[str],
    ) -> dict[str, Any]:
        return {
            "source": "resolve_candidates",
            "ranking_version": "semantic_recall_v2",
            "query": {
                "raw_product_query": request.product_query,
                "normalized_query": normalized_query,
                "query_phrases": normalized_phrases,
                "query_tokens": tokens,
                "query_expansions": query_expansions,
            },
            "category": {
                "category_id": request.category_id,
                "category_path": request.category_path,
                "include_descendants": request.include_descendants,
            },
            "filters": {
                "marketplace": marketplace,
                "domain": domain,
                "recall_mode": request.recall_mode,
                "price_min": request.price_min,
                "price_max": request.price_max,
                "active_only": request.active_only,
                "min_pool_size": request.min_pool_size,
                "target_pool_size": request.target_pool_size,
                "max_candidates": request.max_candidates,
            },
            "sources": ["query", "category" if request.category_id is not None or request.category_path else "keyword"],
        }

    def _persist_candidate_pool(
        self,
        *,
        candidate_pool_id: str,
        domain: int,
        marketplace: str,
        request: ResolveCandidatesRequest,
        normalized_query: str,
        candidate_items: list[dict[str, Any]],
        candidate_total_before_truncate: int,
        pool_quality: dict[str, Any],
        lineage: dict[str, Any],
    ) -> dict[str, Any]:
        if psycopg2 is None:
            return {"persisted": False, "error": "psycopg2_unavailable"}

        try:
            pool_uuid = str(uuid.UUID(str(candidate_pool_id)))
        except ValueError:
            return {"persisted": False, "error": "invalid_candidate_pool_id"}

        try:
            with _postgres_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO serving.candidate_pools (
                            pool_id, version, domain, marketplace, product_query, normalized_query,
                            recall_mode, category_id, category_path, include_descendants, filters,
                            ranking_version, pool_quality, candidate_count,
                            candidate_total_before_truncate, source, lineage, updated_at
                        ) VALUES (
                            %s::uuid, 1, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s::jsonb,
                            %s, %s::jsonb, %s,
                            %s, %s, %s::jsonb, NOW()
                        )
                        ON CONFLICT (pool_id) DO UPDATE SET
                            updated_at = NOW(),
                            candidate_count = EXCLUDED.candidate_count,
                            candidate_total_before_truncate = EXCLUDED.candidate_total_before_truncate,
                            pool_quality = EXCLUDED.pool_quality,
                            lineage = EXCLUDED.lineage
                        """,
                        [
                            pool_uuid,
                            domain,
                            marketplace,
                            request.product_query,
                            normalized_query,
                            request.recall_mode,
                            request.category_id,
                            request.category_path,
                            request.include_descendants,
                            psycopg2.extras.Json(lineage.get("filters") or {}),
                            lineage.get("ranking_version") or "semantic_recall_v2",
                            psycopg2.extras.Json(pool_quality),
                            len(candidate_items),
                            candidate_total_before_truncate,
                            "resolve_candidates",
                            psycopg2.extras.Json(lineage),
                        ],
                    )
                    cursor.execute("DELETE FROM serving.candidate_pool_items WHERE pool_id = %s::uuid", [pool_uuid])
                    item_rows = []
                    for rank, item in enumerate(candidate_items, start=1):
                        asin = str(item.get("asin") or "").strip().upper()
                        if not asin:
                            continue
                        item_rows.append(
                            (
                                pool_uuid,
                                asin,
                                domain,
                                marketplace,
                                rank,
                                item.get("match_score"),
                                psycopg2.extras.Json(item.get("match_reasons") or []),
                                psycopg2.extras.Json(item),
                            )
                        )
                    if item_rows:
                        psycopg2.extras.execute_values(
                            cursor,
                            """
                            INSERT INTO serving.candidate_pool_items (
                                pool_id, asin, domain, marketplace, candidate_rank,
                                match_score, match_reasons, item_snapshot
                            ) VALUES %s
                            """,
                            item_rows,
                            template="(%s::uuid, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                        )
            return {"persisted": True, "item_count": len(candidate_items)}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("candidate pool persistence failed: %s", exc)
            return {"persisted": False, "error": str(exc)[:240]}

    def _resolve_candidate_asins_for_pool_request(
        self,
        request: CandidatePoolRequest,
        *,
        domain: int,
        marketplace: str,
    ) -> tuple[list[str], dict[str, Any]]:
        candidate_asins = _sanitize_asins(request.candidate_asins)
        candidate_pool_ref = {
            "candidate_pool_id": request.candidate_pool_id,
            "resolved_from_pool": False,
            "source": "candidate_asins" if candidate_asins else "candidate_pool_id",
        }
        if candidate_asins:
            candidate_pool_ref["candidate_count"] = len(candidate_asins)
            return candidate_asins, candidate_pool_ref

        if not request.candidate_pool_id:
            raise HTTPException(status_code=400, detail="candidate_asins or candidate_pool_id is required")
        try:
            pool_uuid = str(uuid.UUID(str(request.candidate_pool_id)))
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid candidate_pool_id")

        with _postgres_conn() as conn:
            rows = _run_pg_dict_query(
                conn,
                """
                SELECT p.pool_id::text AS pool_id, p.version, p.candidate_count,
                       i.asin, i.candidate_rank
                FROM serving.candidate_pools p
                JOIN serving.candidate_pool_items i ON i.pool_id = p.pool_id
                WHERE p.pool_id = %s::uuid
                  AND p.domain = %s
                ORDER BY i.candidate_rank ASC, i.asin ASC
                """,
                [pool_uuid, domain],
            )
        if not rows:
            raise HTTPException(status_code=404, detail="candidate_pool_id not found for marketplace")

        resolved_asins = _sanitize_asins([str(row["asin"]) for row in rows])
        candidate_pool_ref.update(
            {
                "candidate_pool_id": pool_uuid,
                "candidate_pool_version": int(rows[0].get("version") or 1),
                "resolved_from_pool": True,
                "candidate_count": len(resolved_asins),
                "marketplace": marketplace,
                "domain": domain,
            }
        )
        return resolved_asins, candidate_pool_ref

    def _build_forecast_meta(self, coverage_row: dict[str, Any] | None = None) -> dict[str, Any]:
        coverage_row = coverage_row or {}
        return {
            "forecast_version": coverage_row.get("forecast_version"),
            "snapshot_date": _iso_date_or_none(coverage_row.get("snapshot_date")),
            "forecast_week_start": _iso_date_or_none(coverage_row.get("forecast_week_start")),
            "forecast_year_week": coverage_row.get("forecast_year_week"),
        }

    def resolve_category(self, request: CategoryResolveRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        category_query = str(request.category_query or "").strip()
        category_path = str(request.category_path or "").strip()
        if not category_query and not category_path:
            raise HTTPException(status_code=400, detail="category_query or category_path is required")

        search_text = _normalize_text(category_path or category_query)
        search_like = f"%{search_text}%" if search_text else "%"

        with _postgres_conn() as conn:
            rows = _run_pg_dict_query(
                conn,
                """
                WITH RECURSIVE category_tree AS (
                    SELECT
                        c.category_id,
                        c.domain,
                        c.category_en,
                        c.category_cn,
                        c.parent_id,
                        c.depth,
                        c.product_count,
                        COALESCE(c.category_en, '')::text AS category_path
                    FROM sync.keepa_category_registry c
                    WHERE c.domain = %s
                      AND (c.parent_id IS NULL OR c.depth = 1)

                    UNION ALL

                    SELECT
                        child.category_id,
                        child.domain,
                        child.category_en,
                        child.category_cn,
                        child.parent_id,
                        child.depth,
                        child.product_count,
                        CONCAT_WS(' > ', NULLIF(parent.category_path, ''), child.category_en)::text AS category_path
                    FROM sync.keepa_category_registry child
                    JOIN category_tree parent
                      ON child.domain = parent.domain
                     AND child.parent_id = parent.category_id
                    WHERE child.category_id <> parent.category_id
                ),
                history_coverage AS (
                    SELECT DISTINCT asin, domain
                    FROM sync.keepa_product_history
                    WHERE domain = %s
                ),
                coverage AS (
                    SELECT
                        r.category_id,
                        COUNT(DISTINCT r.asin) FILTER (WHERE r.is_active) AS local_active_asin_count,
                        COUNT(DISTINCT h.asin) FILTER (WHERE r.is_active) AS local_history_coverage_count
                    FROM sync.keepa_asin_registry r
                    LEFT JOIN history_coverage h
                      ON r.asin = h.asin AND r.domain = h.domain
                    WHERE r.domain = %s
                    GROUP BY r.category_id
                )
                SELECT
                    t.category_id,
                    t.category_en,
                    t.category_cn,
                    t.parent_id,
                    t.depth,
                    t.product_count,
                    t.category_path,
                    COALESCE(c.local_active_asin_count, 0) AS local_active_asin_count,
                    COALESCE(c.local_history_coverage_count, 0) AS local_history_coverage_count
                FROM category_tree t
                LEFT JOIN coverage c ON t.category_id = c.category_id
                WHERE %s = ''
                   OR LOWER(COALESCE(t.category_path, '')) LIKE %s
                   OR LOWER(COALESCE(t.category_en, '')) LIKE %s
                   OR LOWER(COALESCE(t.category_cn, '')) LIKE %s
                LIMIT 300
                """,
                [domain, domain, domain, search_text, search_like, search_like, search_like],
            )

        matches: list[dict[str, Any]] = []
        for row in rows:
            match_confidence = _score_category_match(
                category_query=category_query,
                category_path=category_path,
                candidate_name=str(row.get("category_en") or ""),
                candidate_path=str(row.get("category_path") or ""),
            )
            if match_confidence <= 0:
                continue
            matches.append(
                {
                    "category_id": int(row["category_id"]) if row.get("category_id") is not None else None,
                    "category_name": row.get("category_en"),
                    "category_name_cn": row.get("category_cn"),
                    "category_path": row.get("category_path"),
                    "depth": int(row["depth"]) if row.get("depth") is not None else None,
                    "product_count": int(row["product_count"] or 0),
                    "parent_id": int(row["parent_id"]) if row.get("parent_id") is not None else None,
                    "local_active_asin_count": int(row["local_active_asin_count"] or 0),
                    "local_history_coverage_count": int(row["local_history_coverage_count"] or 0),
                    "match_confidence": match_confidence,
                }
            )

        matches.sort(
            key=lambda item: (
                item["match_confidence"],
                item["local_active_asin_count"],
                item["product_count"],
            ),
            reverse=True,
        )
        matches = matches[: request.max_matches]

        return {
            "marketplace": marketplace,
            "domain": domain,
            "category_query": category_query or None,
            "category_path": category_path or None,
            "match_count": len(matches),
            "matches": matches,
            "notes": [
                "category_resolve reads local sync.keepa_category_registry and sync.keepa_asin_registry only; it does not consume Keepa tokens",
                "local_active_asin_count and local_history_coverage_count indicate current local coverage before any online expansion",
                "use category_id as the stable execution key; category_path is a readable input and fallback",
            ],
        }

    def _estimate_candidate_expansion_tokens(self, request: CandidateExpansionJobRequest) -> int:
        discovery_tokens = 50 if request.category_id is not None or request.category_path else 12
        hydrate_tokens = min(request.target_asin_count, 100) * 2
        return discovery_tokens + hydrate_tokens

    def _validate_candidate_expansion_request(self, request: CandidateExpansionJobRequest) -> None:
        has_category_id = request.category_id is not None
        has_product_query = bool(request.product_query)
        has_category_path = bool(request.category_path)

        if not has_product_query and not has_category_id:
            detail = "product_query or category_id is required"
            if has_category_path:
                detail = "category_path is not executable by itself; call category_resolve first and pass category_id"
            raise HTTPException(status_code=400, detail=detail)

        if request.recall_mode == "category" and not has_category_id:
            raise HTTPException(
                status_code=400,
                detail="category recall requires category_id; call category_resolve first and pass the selected category_id",
            )

        if has_category_path and not has_category_id and request.recall_mode in {"category", "hybrid"}:
            raise HTTPException(
                status_code=400,
                detail="category_path cannot be used as an execution key; call category_resolve first and pass category_id",
            )

        scope_decision = evaluate_seller_scope(
            category_path=request.category_path,
            query=request.product_query,
        )
        if not scope_decision.allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "candidate expansion is outside the configured small cross-border seller product scope",
                    "seller_scope": scope_decision.as_dict(),
                },
            )

    def _candidate_expansion_category_lock_key(self, *, domain: int, category_id: int, include_descendants: bool) -> str:
        include_flag = 1 if include_descendants else 0
        return f"candidate-expansion:domain:{domain}:category:{category_id}:desc:{include_flag}"

    def _format_candidate_expansion_job(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": row.get("job_id"),
            "marketplace": row.get("marketplace") or DOMAIN_TO_MARKETPLACE.get(int(row.get("domain") or 1), "US"),
            "domain": int(row.get("domain") or 1),
            "source": row.get("source"),
            "priority": row.get("priority"),
            "product_query": row.get("product_query"),
            "recall_mode": row.get("recall_mode"),
            "category_id": int(row["category_id"]) if row.get("category_id") is not None else None,
            "category_path": row.get("category_path"),
            "include_descendants": bool(row.get("include_descendants")),
            "target_asin_count": int(row.get("target_asin_count") or 0),
            "min_pool_size": int(row.get("min_pool_size") or 0),
            "status": row.get("status"),
            "status_reason": row.get("status_reason"),
            "requested_by_session_id": row.get("requested_by_session_id"),
            "requested_by_user_id": row.get("requested_by_user_id"),
            "tokens_estimated": int(row.get("tokens_estimated") or 0),
            "tokens_reserved": int(row.get("tokens_reserved") or 0),
            "tokens_consumed": int(row.get("tokens_consumed") or 0),
            "token_wait_until": _iso_date_or_none(row.get("token_wait_until")),
            "result_candidate_asins": list(row.get("result_candidate_asins") or []),
            "result_new_asin_count": int(row.get("result_new_asin_count") or 0),
            "error_message": row.get("error_message"),
            "created_at": _iso_date_or_none(row.get("created_at")),
            "updated_at": _iso_date_or_none(row.get("updated_at")),
            "started_at": _iso_date_or_none(row.get("started_at")),
            "finished_at": _iso_date_or_none(row.get("finished_at")),
            "meta_json": row.get("meta_json") or {},
        }

    def _build_candidate_expansion_data_readiness(
        self,
        job: dict[str, Any],
        coverage_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        coverage_row = coverage_row or {}
        requested_asins = _sanitize_asins(job.get("result_candidate_asins") or [])
        requested_count = len(requested_asins)
        min_pool_size = int(job.get("min_pool_size") or DEFAULT_MIN_CANDIDATE_POOL_SIZE)
        ready_threshold = max(1, min(requested_count, min_pool_size)) if requested_count else 0

        registry_hit_count = int(coverage_row.get("registry_hit_count") or 0)
        snapshot_hit_count = int(coverage_row.get("snapshot_hit_count") or 0)
        history_hit_count = int(coverage_row.get("history_hit_count") or 0)
        history_row_count = int(coverage_row.get("history_row_count") or 0)
        serving_base_hit_count = int(coverage_row.get("serving_base_hit_count") or 0)
        serving_base_row_count = int(coverage_row.get("serving_base_row_count") or 0)

        def coverage_ratio(hit_count: int) -> float:
            return round(hit_count / requested_count, 4) if requested_count else 0.0

        registry_ready = requested_count == 0 or registry_hit_count >= requested_count
        hydration_hit_count = max(snapshot_hit_count, history_hit_count)
        hydration_ready = requested_count > 0 and hydration_hit_count >= ready_threshold
        serving_ready = requested_count > 0 and serving_base_hit_count >= ready_threshold

        if requested_count == 0:
            readiness_status = "no_candidates"
        elif serving_ready:
            readiness_status = "analysis_ready"
        elif hydration_ready:
            readiness_status = "serving_sync_pending"
        elif registry_ready:
            readiness_status = "history_hydration_pending"
        else:
            readiness_status = "registry_sync_pending"

        return {
            "requested_asin_count": requested_count,
            "ready_threshold": ready_threshold,
            "registry_hit_count": registry_hit_count,
            "registry_coverage_ratio": coverage_ratio(registry_hit_count),
            "snapshot_hit_count": snapshot_hit_count,
            "snapshot_coverage_ratio": coverage_ratio(snapshot_hit_count),
            "history_hit_count": history_hit_count,
            "history_coverage_ratio": coverage_ratio(history_hit_count),
            "history_row_count": history_row_count,
            "history_latest_date": _iso_date_or_none(coverage_row.get("history_latest_date")),
            "serving_base_hit_count": serving_base_hit_count,
            "serving_base_coverage_ratio": coverage_ratio(serving_base_hit_count),
            "serving_base_row_count": serving_base_row_count,
            "serving_base_latest_date": _iso_date_or_none(coverage_row.get("serving_base_latest_date")),
            "registry_ready": registry_ready,
            "hydration_ready": hydration_ready,
            "serving_ready": serving_ready,
            "analysis_ready": serving_ready,
            "readiness_status": readiness_status,
        }

    def _fetch_candidate_expansion_data_readiness(self, conn, job: dict[str, Any]) -> dict[str, Any]:
        requested_asins = _sanitize_asins(job.get("result_candidate_asins") or [])
        if not requested_asins:
            return self._build_candidate_expansion_data_readiness(job)

        domain = int(job.get("domain") or 1)
        rows = _run_pg_dict_query(
            conn,
            """
            WITH requested AS (
                SELECT UNNEST(%s::TEXT[]) AS asin
            ),
            registry AS (
                SELECT DISTINCT r.asin
                FROM sync.keepa_asin_registry r
                JOIN requested q ON q.asin = r.asin
                WHERE r.domain = %s
            ),
            snapshot AS (
                SELECT DISTINCT s.asin
                FROM sync.keepa_product_snapshot s
                JOIN requested q ON q.asin = s.asin
                WHERE s.domain = %s
            ),
            history AS (
                SELECT h.asin, COUNT(*) AS row_count, MAX(h.date) AS latest_date
                FROM sync.keepa_product_history h
                JOIN requested q ON q.asin = h.asin
                WHERE h.domain = %s
                GROUP BY h.asin
            ),
            serving_base AS (
                SELECT b.asin, COUNT(*) AS row_count, MAX(b.date) AS latest_date
                FROM serving.theme_base_daily b
                JOIN requested q ON q.asin = b.asin
                WHERE b.domain = %s
                GROUP BY b.asin
            )
            SELECT
                COUNT(DISTINCT registry.asin) AS registry_hit_count,
                COUNT(DISTINCT snapshot.asin) AS snapshot_hit_count,
                COUNT(DISTINCT history.asin) AS history_hit_count,
                COALESCE(SUM(history.row_count), 0) AS history_row_count,
                MAX(history.latest_date) AS history_latest_date,
                COUNT(DISTINCT serving_base.asin) AS serving_base_hit_count,
                COALESCE(SUM(serving_base.row_count), 0) AS serving_base_row_count,
                MAX(serving_base.latest_date) AS serving_base_latest_date
            FROM requested q
            LEFT JOIN registry ON registry.asin = q.asin
            LEFT JOIN snapshot ON snapshot.asin = q.asin
            LEFT JOIN history ON history.asin = q.asin
            LEFT JOIN serving_base ON serving_base.asin = q.asin
            """,
            [requested_asins, domain, domain, domain, domain],
        )
        return self._build_candidate_expansion_data_readiness(job, rows[0] if rows else {})

    def create_candidate_expansion_job(self, request: CandidateExpansionJobRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        self._validate_candidate_expansion_request(request)

        tokens_estimated = self._estimate_candidate_expansion_tokens(request)
        status_reason = "queued for collector; no Keepa token is consumed by theme-api"
        category_lock_key = (
            self._candidate_expansion_category_lock_key(
                domain=domain,
                category_id=int(request.category_id),
                include_descendants=request.include_descendants,
            )
            if request.category_id is not None
            else None
        )
        meta_json = {
            "notes": request.notes,
            "created_by": "theme-api",
            "token_budget_policy": "collector-owned",
            "category_execution_key": category_lock_key,
        }

        with _postgres_conn() as conn:
            category_lock_acquired = False
            try:
                if category_lock_key:
                    _run_pg_dict_query(conn, "SELECT pg_advisory_lock(hashtext(%s))", [category_lock_key])
                    category_lock_acquired = True

                if request.idempotency_key:
                    existing = _run_pg_dict_query(
                        conn,
                        """
                        SELECT *
                        FROM sync.keepa_candidate_expansion_jobs
                        WHERE idempotency_key = %s
                        LIMIT 1
                        """,
                        [request.idempotency_key],
                    )
                    if existing:
                        return {
                            "job": self._format_candidate_expansion_job(existing[0]),
                            "created": False,
                            "notes": ["existing job returned by idempotency_key"],
                        }

                if request.category_id is not None:
                    existing_category_job = _run_pg_dict_query(
                        conn,
                        """
                        SELECT *
                        FROM sync.keepa_candidate_expansion_jobs
                        WHERE domain = %s
                          AND category_id = %s
                          AND include_descendants IS NOT DISTINCT FROM %s
                          AND status = ANY(%s::TEXT[])
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        [
                            domain,
                            request.category_id,
                            request.include_descendants,
                            list(CANDIDATE_EXPANSION_ACTIVE_STATUSES),
                        ],
                    )
                    if existing_category_job:
                        return {
                            "job": self._format_candidate_expansion_job(existing_category_job[0]),
                            "created": False,
                            "notes": [
                                "existing active category expansion job returned",
                                "registry, hydration, and serving sync should be read from this job's data_readiness",
                            ],
                        }

                job_id = f"kexp_{uuid.uuid4().hex}"
                rows = _run_pg_dict_query(
                    conn,
                    """
                    INSERT INTO sync.keepa_candidate_expansion_jobs (
                        job_id,
                        domain,
                        marketplace,
                        source,
                        priority,
                        product_query,
                        recall_mode,
                        category_id,
                        category_path,
                        include_descendants,
                        target_asin_count,
                        min_pool_size,
                        status,
                        status_reason,
                        requested_by_session_id,
                        requested_by_user_id,
                        idempotency_key,
                        tokens_estimated,
                        meta_json,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'queued', %s, %s, %s, %s, %s, %s::JSONB, NOW(), NOW()
                    )
                    ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''
                    DO UPDATE SET updated_at = sync.keepa_candidate_expansion_jobs.updated_at
                    RETURNING *, (xmax = 0) AS inserted
                    """,
                    [
                        job_id,
                        domain,
                        marketplace,
                        request.source,
                        request.priority,
                        request.product_query,
                        request.recall_mode,
                        request.category_id,
                        request.category_path,
                        request.include_descendants,
                        request.target_asin_count,
                        request.min_pool_size,
                        status_reason,
                        request.requested_by_session_id,
                        request.requested_by_user_id,
                        request.idempotency_key,
                        tokens_estimated,
                        psycopg2.extras.Json(meta_json),
                    ],
                )
            finally:
                if category_lock_acquired and category_lock_key:
                    _run_pg_dict_query(conn, "SELECT pg_advisory_unlock(hashtext(%s))", [category_lock_key])

        created = bool(rows[0].pop("inserted", True))
        if not created:
            return {
                "job": self._format_candidate_expansion_job(rows[0]),
                "created": False,
                "notes": ["existing job returned by idempotency_key"],
            }
        return {
            "job": self._format_candidate_expansion_job(rows[0]),
            "created": True,
            "notes": [
                "candidate expansion is queued; collector owns Keepa token reservation and execution",
                "status will move through queued/waiting_token/discovering/hydrating/syncing/completed",
            ],
        }

    def _opportunity_data_confidence(self, *, candidate_count: int, trend_coverage: float, row_count: int) -> str:
        if candidate_count >= 12 and row_count >= 120 and trend_coverage >= 0.35:
            return "high"
        if candidate_count >= 5 and row_count >= 30:
            return "medium"
        return "low"

    def _build_opportunity_score(self, breakdown: dict[str, float]) -> float:
        return round(
            sum(OPPORTUNITY_SCORE_WEIGHTS[key] * breakdown.get(key, 0.0) for key in OPPORTUNITY_SCORE_WEIGHTS),
            2,
        )

    def _build_opportunity_score_explanation(self, breakdown: dict[str, float]) -> dict[str, Any]:
        weighted_components = {
            key: {
                "score": round(_safe_float(breakdown.get(key)), 2),
                "weight": weight,
                "weighted_points": round(_safe_float(breakdown.get(key)) * weight, 2),
            }
            for key, weight in OPPORTUNITY_SCORE_WEIGHTS.items()
        }
        return {
            "formula": "sum(component_score_0_to_100 * component_weight)",
            "weights": dict(OPPORTUNITY_SCORE_WEIGHTS),
            "components": weighted_components,
            "plain_language": (
                "机会得分是 0-100 的综合排序分，不是单一销量排名；它综合需求规模、趋势动能、"
                "竞争空间、价格适配、预测增长、补池空间和证据质量。"
            ),
        }

    def _opportunity_metric_definitions(self) -> dict[str, Any]:
        return {
            "opportunity_score": {
                "label": "机会得分",
                "meaning": "0-100 综合排序分，用于比较机会优先级，不代表确定收益。",
                "formula": "20%需求 + 20%趋势 + 15%竞争空间 + 15%价格适配 + 15%预测增长 + 10%覆盖差距 + 5%证据质量",
            },
            "sales_window_sum": {
                "label": "窗口销量",
                "meaning": "当前本地 serving 表 estimated_daily_sales 在窗口期内的合计。",
                "formula": "sum(estimated_daily_sales) over candidate ASIN daily rows in window_days",
                "display_guidance": "按销量估算/销售信号展示；单位是销量数量。只有金额字段明确为 GMV/销售额时才使用货币格式。",
                "sample_scope_fields": ["candidate_count", "row_count", "window_days"],
            },
            "sales_momentum_pct": {
                "label": "销量增长",
                "meaning": "最近 7 天日均 estimated_daily_sales 相比窗口前段日均值的变化率。",
                "formula": "(sales_mean_7 - sales_mean_prev) / sales_mean_prev * 100",
            },
            "trend_momentum_pct": {
                "label": "趋势增长",
                "meaning": "最近 7 天 Google Trends 均值相比窗口前段均值的变化率。",
                "formula": "(trend_mean_7 - trend_mean_prev) / trend_mean_prev * 100",
            },
            "offer_count_avg": {
                "label": "竞争强度",
                "meaning": "窗口内 ASIN-日粒度的 Amazon offer 数均值，来自 new_offer_count + used_offer_count；Offer 是前台可购买报价/卖家报价，不是供应商数量。",
                "formula": "avg(new_offer_count + used_offer_count) over candidate ASIN daily rows in window_days",
                "interpretation": "数值越低通常表示同一商品上的活跃报价更少，但仍需结合品牌、评论数、FBA/FBM 和真实搜索页竞争判断。",
            },
            "data_confidence": {
                "label": "数据置信度",
                "meaning": "按候选 ASIN 数、窗口日数据行数、趋势覆盖率评估证据充足度。",
                "formula": "high: candidate_count>=12 and row_count>=120 and trend_coverage>=0.35; medium: candidate_count>=5 and row_count>=30; otherwise low",
            },
        }

    def _build_category_metric_explanations(
        self,
        *,
        score_breakdown: dict[str, float],
        candidate_count: int,
        row_count: int,
        window_days: int,
        sales_window_sum: float,
        sales_mean_7: float,
        sales_mean_prev: float,
        sales_momentum_pct: float,
        trend_mean_7: float,
        trend_mean_prev: float,
        trend_momentum_pct: float,
        offer_count_avg: float,
        trend_coverage: float,
    ) -> dict[str, Any]:
        return {
            "opportunity_score": self._build_opportunity_score_explanation(score_breakdown),
            "sales_window_sum": {
                "formula": "sum(estimated_daily_sales) over candidate ASIN daily rows in window_days",
                "candidate_count": candidate_count,
                "row_count": row_count,
                "window_days": window_days,
                "value": round(sales_window_sum, 2),
                "display_guidance": "这是 estimated_daily_sales 的窗口合计，单位是销量数量，不是明确 GMV 字段。",
                "plain_language": f"该值来自 {candidate_count} 个候选 ASIN 在近 {window_days} 天内的 {row_count} 条日粒度数据。",
            },
            "sales_momentum_pct": {
                "formula": "(sales_mean_7 - sales_mean_prev) / sales_mean_prev * 100",
                "recent_7d_daily_avg": round(sales_mean_7, 4),
                "previous_window_daily_avg": round(sales_mean_prev, 4),
                "value_pct": round(sales_momentum_pct, 2),
            },
            "trend_momentum_pct": {
                "formula": "(trend_mean_7 - trend_mean_prev) / trend_mean_prev * 100",
                "recent_7d_trend_avg": round(trend_mean_7, 4),
                "previous_window_trend_avg": round(trend_mean_prev, 4),
                "value_pct": round(trend_momentum_pct, 2),
                "trend_coverage": trend_coverage,
            },
            "offer_count_avg": {
                "formula": "avg(new_offer_count + used_offer_count) over candidate ASIN daily rows in window_days",
                "value": round(offer_count_avg, 2),
                "plain_language": "Offer 是 Amazon 上同一 ASIN 的可购买报价/卖家报价，不是供应商数量；这里取窗口内 ASIN-日平均。",
            },
        }

    def _build_query_metric_explanations(
        self,
        *,
        score_breakdown: dict[str, float],
        candidate_count: int,
        window_days: int,
        sales_window_sum: float,
        sales_window_avg: float,
        trend_wow: float,
        trend_coverage: float,
        offer_count_median: float,
    ) -> dict[str, Any]:
        return {
            "opportunity_score": self._build_opportunity_score_explanation(score_breakdown),
            "sales_window_sum": {
                "formula": "sum(estimated_daily_sales) over resolved candidate ASIN daily rows in window_days",
                "candidate_count": candidate_count,
                "window_days": window_days,
                "value": round(sales_window_sum, 2),
                "daily_avg": round(sales_window_avg, 2),
                "display_guidance": "这是 estimated_daily_sales 的窗口合计，单位是销量数量，不是明确 GMV 字段。",
            },
            "trend_momentum_pct": {
                "formula": "trend_wow from candidate_pool_trends, expressed as percentage points in current response",
                "value_pct": round(trend_wow, 2),
                "trend_coverage": trend_coverage,
            },
            "offer_count_median": {
                "formula": "median offer count across resolved candidate pool",
                "value": round(offer_count_median, 2),
                "plain_language": "Offer 是 Amazon 上同一 ASIN 的可购买报价/卖家报价，不是供应商数量。",
            },
        }

    def _opportunity_title_key(self, card: dict[str, Any]) -> str:
        title = str(card.get("title") or _leaf_category_name(card.get("category_path")) or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", " ", title).strip()

    def _trend_momentum_signal(
        self,
        *,
        recent_mean: float,
        previous_mean: float,
        recent_rows: int,
        previous_rows: int,
        total_rows: int,
    ) -> dict[str, Any]:
        if total_rows <= 0 or (recent_rows <= 0 and previous_rows <= 0):
            return {
                "value_pct": None,
                "score_pct": 0.0,
                "display": "趋势数据缺失",
                "status": "missing_trend_data",
                "interpretation": "窗口内没有可用趋势数据，不应解读为趋势下跌。",
            }
        if recent_rows <= 0:
            return {
                "value_pct": None,
                "score_pct": 0.0,
                "display": "近期趋势缺失",
                "status": "recent_trend_missing",
                "interpretation": "最近 7 天缺少趋势观测，不应展示为 -100%。",
            }
        if previous_rows <= 0 or previous_mean <= 0:
            return {
                "value_pct": None,
                "score_pct": 0.0,
                "display": "趋势基线不足",
                "status": "trend_baseline_missing",
                "interpretation": "窗口前段趋势基线不足，不能计算相对增长率。",
            }

        value_pct = (recent_mean - previous_mean) / previous_mean * 100.0
        if recent_mean <= 0:
            return {
                "value_pct": round(value_pct, 2),
                "score_pct": value_pct,
                "display": "近期趋势为 0",
                "status": "recent_trend_zero",
                "interpretation": "最近 7 天有趋势观测但均值为 0；这是近期归零信号，不等同于数据缺失。",
            }
        if value_pct <= -95.0:
            return {
                "value_pct": round(value_pct, 2),
                "score_pct": value_pct,
                "display": f"{value_pct:+.2f}%（大幅下滑）",
                "status": "sharp_trend_decline",
                "interpretation": "最近 7 天趋势均值明显低于窗口前段，需要结合趋势覆盖率判断。",
            }
        return {
            "value_pct": round(value_pct, 2),
            "score_pct": value_pct,
            "display": f"{value_pct:+.2f}%",
            "status": "measured_pct",
            "interpretation": "趋势增长率可按公式解释。",
        }

    def _select_opportunities_by_confidence_and_title(
        self,
        cards: list[dict[str, Any]],
        *,
        min_data_confidence: str,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        min_rank = _confidence_rank(min_data_confidence)
        selected: list[dict[str, Any]] = []
        selected_by_title: dict[str, dict[str, Any]] = {}
        hidden_duplicates: list[dict[str, Any]] = []
        eligible_count = 0

        for card in cards:
            if _confidence_rank(str(card.get("data_confidence") or "low")) < min_rank:
                continue
            eligible_count += 1
            title_key = self._opportunity_title_key(card)
            if title_key and title_key in selected_by_title:
                kept = selected_by_title[title_key]
                hidden_duplicates.append(
                    {
                        "title": card.get("title"),
                        "kept_opportunity_id": kept.get("opportunity_id"),
                        "kept_category_path": kept.get("category_path"),
                        "hidden_opportunity_id": card.get("opportunity_id"),
                        "hidden_category_path": card.get("category_path"),
                        "hidden_opportunity_score": card.get("opportunity_score"),
                    }
                )
                continue

            selected.append(card)
            if title_key:
                selected_by_title[title_key] = card
            if len(selected) >= limit:
                break

        return selected, {
            "mode": "normalized_title_first_card_wins",
            "input_count": len(cards),
            "eligible_count": eligible_count,
            "kept_count": len(selected),
            "hidden_duplicate_count": len(hidden_duplicates),
            "hidden_duplicates": hidden_duplicates[:12],
        }

    def _opportunity_display_title(self, card: dict[str, Any]) -> str:
        title = str(card.get("title") or "").strip()
        category_path = str(card.get("category_path") or "").strip()
        category_parts = [part.strip() for part in category_path.split(" > ") if part.strip()]
        parent = category_parts[-2] if len(category_parts) >= 2 else None
        if title and parent and parent.lower() != title.lower():
            return f"{title} / {parent}"
        return title or _leaf_category_name(category_path) or "Amazon opportunity"

    def _opportunity_formula_details(self, metric_definitions: dict[str, Any], personalization_applied: bool) -> list[str]:
        details = [
            f"- 机会得分: {metric_definitions['opportunity_score']['formula']}。",
            f"- 窗口销量估算: {metric_definitions['sales_window_sum']['formula']}。",
            f"- 销量增长: {metric_definitions['sales_momentum_pct']['formula']}。",
            f"- 趋势增长: {metric_definitions['trend_momentum_pct']['formula']}；若近期趋势缺失、基线不足或近期均值为 0，会用文字状态展示，避免把缺失误读为 -100%。",
            f"- 竞争Offer: {metric_definitions['offer_count_avg']['formula']}。",
            f"- 数据置信度: {metric_definitions['data_confidence']['formula']}。",
        ]
        if personalization_applied:
            details.append("- 个性化分: 在机会得分上加入小幅偏好调整后的排序分，仅用于重排参考。")
        return details

    def _opportunity_llm_summary_guidance(
        self,
        *,
        duplicate_summary: dict[str, Any],
        personalization_applied: bool,
        opportunities: list[dict[str, Any]],
    ) -> list[str]:
        guidance = [
            "最终答复应把 opportunity_cards_text 中的总览表、字段解释和公式明细作为工具证据块展示；可在证据块前后用自己的语言做摘要和解读，但不要改写成平铺列表，不要丢列、改数值或补未返回的数值。",
            "展示趋势时优先使用 trend_momentum_display / trend_signal_status；不要把趋势缺失或近期为 0 简化成普通 -100%。",
        ]
        if _safe_int(duplicate_summary.get("hidden_duplicate_count")) > 0:
            guidance.append("必须明确提示 duplicate_title_filter.hidden_duplicate_count，说明同名主题已隐藏低优先级类目。")
        if personalization_applied:
            guidance.append("如果按个性化排序展示，表格必须保留个性化分；若另行排序，需要说明排序口径。")
        if any(isinstance(card, dict) and (card.get("next_action") or {}).get("requires_category_resolve") for card in opportunities):
            guidance.append("category_id 为空且 next_action.requires_category_resolve=true 的机会，必须提示先调用 category_resolve 再做类目召回分析。")
        return guidance

    def _build_opportunity_llm_presentation(self, result: dict[str, Any]) -> dict[str, Any]:
        opportunities = list(result.get("opportunities") or [])
        metric_definitions = result.get("metric_definitions") or self._opportunity_metric_definitions()
        personalization_applied = any(isinstance(card, dict) and card.get("memory_profile_rerank") for card in opportunities)
        formula_details = self._opportunity_formula_details(metric_definitions, personalization_applied)
        lines = [
            "## 机会发现结果",
            "",
            f"市场: {result.get('marketplace') or 'US'} | 平台: {result.get('platform') or 'Amazon'} | 实际返回机会数: {len(opportunities)}",
            "",
        ]
        duplicate_summary = ((result.get("diagnostics") or {}).get("duplicate_title_filter") or {})
        llm_summary_guidance = self._opportunity_llm_summary_guidance(
            duplicate_summary=duplicate_summary,
            personalization_applied=personalization_applied,
            opportunities=opportunities,
        )
        hidden_duplicate_count = _safe_int(duplicate_summary.get("hidden_duplicate_count"))
        if hidden_duplicate_count > 0:
            lines.extend(
                [
                    f"已隐藏 {hidden_duplicate_count} 个同名主题的低优先级类目，榜单默认保留每个主题当前最高排序机会。",
                    "",
                ]
            )
        if personalization_applied:
            lines.extend(
                [
                    "| 排名 | 机会 | 机会得分 | 个性化分 | 窗口销量估算 | 增长信号 | 竞争Offer | 样本 | 置信度 |",
                    "|---:|---|---:|---:|---:|---|---:|---:|---|",
                ]
            )
        else:
            lines.extend(
                [
                    "| 排名 | 机会 | 得分 | 窗口销量估算 | 增长信号 | 竞争Offer | 样本 | 置信度 |",
                    "|---:|---|---:|---:|---|---:|---:|---|",
                ]
            )
        compact_cards: list[dict[str, Any]] = []
        for index, card in enumerate(opportunities, start=1):
            evidence = card.get("evidence_summary") or {}
            explanations = card.get("metric_explanations") or {}
            sales_explanation = explanations.get("sales_window_sum") or {}
            offer_value = evidence.get("offer_count_avg")
            if offer_value is None:
                offer_value = evidence.get("offer_count_median")
            row_count = evidence.get("row_count", sales_explanation.get("row_count"))
            candidate_count = evidence.get("candidate_count", sales_explanation.get("candidate_count"))
            sales_window_sum = evidence.get("sales_window_sum")
            sales_momentum = evidence.get("sales_momentum_pct")
            trend_momentum = evidence.get("trend_momentum_pct", evidence.get("trend_wow"))
            trend_display = evidence.get("trend_momentum_display")
            display_title = self._opportunity_display_title(card)
            growth_signal = "销量 {sales} / 趋势 {trend}".format(
                sales="-" if sales_momentum is None else f"{_safe_float(sales_momentum):+.2f}%",
                trend=str(trend_display) if trend_display else ("-" if trend_momentum is None else f"{_safe_float(trend_momentum):+.2f}%"),
            )
            row_values = {
                "rank": index,
                "title": display_title,
                "score": _safe_float(card.get("opportunity_score")),
                "personalized_score": _safe_float(card.get("personalized_opportunity_score")),
                "path": str(card.get("category_path") or "-"),
                "sales": "-" if sales_window_sum is None else f"{_safe_float(sales_window_sum):,.2f}",
                "asins": "-" if candidate_count is None else str(candidate_count),
                "rows": "-" if row_count is None else str(row_count),
                "growth_signal": growth_signal,
                "offer": "-" if offer_value is None else f"{_safe_float(offer_value):.2f}",
                "confidence": str(card.get("data_confidence") or "-"),
            }
            if personalization_applied:
                lines.append(
                    "| {rank} | {title} | {score:.2f} | {personalized_score:.2f} | {sales} | {growth_signal} | {offer} | {asins}/{rows} | {confidence} |".format(**row_values)
                )
            else:
                lines.append(
                    "| {rank} | {title} | {score:.2f} | {sales} | {growth_signal} | {offer} | {asins}/{rows} | {confidence} |".format(**row_values)
                )
            compact_card = {
                "rank": index,
                "opportunity_id": card.get("opportunity_id"),
                "title": card.get("title"),
                "display_title": display_title,
                "source": card.get("source"),
                "category_id": card.get("category_id"),
                "category_path": card.get("category_path"),
                "candidate_pool_id": card.get("candidate_pool_id"),
                "opportunity_score": card.get("opportunity_score"),
                "base_opportunity_score": card.get("base_opportunity_score"),
                "personalized_opportunity_score": card.get("personalized_opportunity_score"),
                "memory_profile_rerank": card.get("memory_profile_rerank"),
                "candidate_count": candidate_count,
                "row_count": row_count,
                "sales_window_sum": sales_window_sum,
                "sales_momentum_pct": sales_momentum,
                "trend_momentum_pct": trend_momentum,
                "trend_momentum_display": trend_display,
                "trend_signal_status": evidence.get("trend_signal_status"),
                "offer_count": offer_value,
                "data_confidence": card.get("data_confidence"),
                "next_action": card.get("next_action"),
                "metric_explanations": card.get("metric_explanations"),
            }
            compact_cards.append({key: value for key, value in compact_card.items() if value is not None})

        lines.extend(
            [
                "",
                "### 字段解释",
                "- 机会得分: 0-100 排序分，越高越值得优先分析，不代表确定收益。",
                "- 窗口销量估算: estimated_daily_sales 的窗口合计，单位是销量数量，不是金额。",
                "- 样本: 写作 ASIN数/日数据行数，用来判断这条机会的证据厚度。",
                "- 增长与竞争: 销量/趋势为近 7 天相对前段变化；竞争Offer是同一 ASIN 的可购买报价均值。",
                *( ["- 个性化分: 只用于结合偏好重排，不能替代工具事实。"] if personalization_applied else [] ),
                "",
                "<details>",
                "<summary>公式明细</summary>",
                "",
                *formula_details,
                "",
                "</details>",
            ]
        )
        return {
            "opportunity_cards_text": "\n".join(lines),
            "opportunities_for_llm": compact_cards,
            "field_formula_details": formula_details,
            "llm_summary_guidance": llm_summary_guidance,
            "display_rules": [
                "must_include_metric_table",
                "must_render_opportunity_cards_text_as_evidence_block",
                "must_include_short_field_explanations",
                "formula_details_are_expandable",
                "preserve_llm_summary_guidance",
                *( ["must_include_duplicate_title_notice"] if hidden_duplicate_count > 0 else [] ),
                *( ["must_include_personalized_score"] if personalization_applied else [] ),
                *( ["must_note_category_resolve_required"] if any((card.get("next_action") or {}).get("requires_category_resolve") for card in opportunities if isinstance(card, dict)) else [] ),
                "use_trend_momentum_display",
                "do_not_pad_missing_opportunities",
                "do_not_format_estimated_daily_sales_as_currency",
            ],
        }

    def _with_opportunity_llm_presentation(self, result: dict[str, Any]) -> dict[str, Any]:
        result["metric_definitions"] = result.get("metric_definitions") or self._opportunity_metric_definitions()
        result["llm_presentation"] = self._build_opportunity_llm_presentation(result)
        result["opportunity_cards_text"] = result["llm_presentation"]["opportunity_cards_text"]
        result["opportunities_for_llm"] = result["llm_presentation"]["opportunities_for_llm"]
        result["field_formula_details"] = result["llm_presentation"]["field_formula_details"]
        result["llm_summary_guidance"] = result["llm_presentation"]["llm_summary_guidance"]
        result["display_rules"] = result["llm_presentation"]["display_rules"]
        return result

    def _finalize_opportunity_discovery_result(
        self,
        *,
        request: OpportunityDiscoveryRequest,
        domain: int,
        marketplace: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = self._with_opportunity_llm_presentation(result)
        return self._attach_opportunity_discovery_job(
            request=request,
            domain=domain,
            marketplace=marketplace,
            result=prepared,
        )

    def _opportunity_discovery_json(self, value: Any) -> Any:
        return psycopg2.extras.Json(value, dumps=lambda obj: json.dumps(obj, ensure_ascii=False, default=str))

    def _opportunity_discovery_job_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        opportunities = result.get("opportunities") or []
        compact_opportunities: list[dict[str, Any]] = []
        for item in opportunities[:30]:
            if not isinstance(item, dict):
                continue
            compact_opportunities.append(
                {
                    "opportunity_id": item.get("opportunity_id"),
                    "title": item.get("title"),
                    "category_id": item.get("category_id"),
                    "category_path": item.get("category_path"),
                    "opportunity_score": item.get("opportunity_score"),
                    "data_confidence": item.get("data_confidence"),
                    "next_action": item.get("next_action"),
                }
            )
        return {
            "opportunity_count": int(result.get("opportunity_count") or len(opportunities)),
            "opportunities": compact_opportunities,
            "opportunity_cards_text_preview": str(result.get("opportunity_cards_text") or "")[:6000],
        }

    def _attach_opportunity_discovery_job(
        self,
        *,
        request: OpportunityDiscoveryRequest,
        domain: int,
        marketplace: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        job_id = f"odisc_{uuid.uuid4().hex}"
        job_ref = {
            "job_id": job_id,
            "status": "completed",
            "marketplace": marketplace,
            "domain": domain,
            "result_endpoint": "/api/product-theme/opportunity-discovery-job",
        }
        result["opportunity_discovery_job_id"] = job_id
        result["opportunity_discovery_job"] = job_ref
        result["result_ref"] = {
            "type": "opportunity_discovery_job",
            "job_id": job_id,
            "status_endpoint": "/api/product-theme/opportunity-discovery-job",
        }

        request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        if isinstance(request_payload.get("memory_profile"), dict):
            profile_payload = request_payload.get("memory_profile") or {}
            profile = profile_payload.get("profile") if isinstance(profile_payload.get("profile"), dict) else profile_payload
            request_payload["memory_profile"] = {
                "present": True,
                "summary_version": profile_payload.get("summary_version") or profile.get("summary_version"),
                "profile_fields_present": sorted(key for key, value in profile.items() if value),
            }
        summary_payload = self._opportunity_discovery_job_summary(result)
        try:
            with _postgres_conn() as conn:
                _run_pg_dict_query(
                    conn,
                    """
                    INSERT INTO sync.keepa_opportunity_discovery_jobs (
                        job_id,
                        domain,
                        marketplace,
                        platform,
                        query,
                        category_id,
                        category_path,
                        include_descendants,
                        limit_count,
                        window_days,
                        min_data_confidence,
                        include_expandable,
                        status,
                        opportunity_count,
                        request_payload_json,
                        result_payload_json,
                        summary_payload_json,
                        created_at,
                        updated_at,
                        finished_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'completed', %s, %s::JSONB, %s::JSONB, %s::JSONB, NOW(), NOW(), NOW()
                    )
                    RETURNING job_id
                    """,
                    [
                        job_id,
                        domain,
                        marketplace,
                        request.platform,
                        request.query,
                        request.category_id,
                        request.category_path,
                        request.include_descendants,
                        request.limit,
                        request.window_days,
                        request.min_data_confidence,
                        request.include_expandable,
                        summary_payload["opportunity_count"],
                        self._opportunity_discovery_json(request_payload),
                        self._opportunity_discovery_json(result),
                        self._opportunity_discovery_json(summary_payload),
                    ],
                )
        except Exception as exc:  # pragma: no cover - defensive; discovery should still return facts if persistence is unavailable
            diagnostics = result.setdefault("diagnostics", {})
            diagnostics["opportunity_discovery_job_storage"] = {
                "status": "failed",
                "error": str(exc),
            }
            result["opportunity_discovery_job"]["status"] = "storage_failed"
            result["result_ref"]["status"] = "storage_failed"
        return result

    def _format_opportunity_discovery_job(self, row: dict[str, Any], *, include_result: bool) -> dict[str, Any]:
        job = {
            "job_id": row.get("job_id"),
            "marketplace": row.get("marketplace") or DOMAIN_TO_MARKETPLACE.get(int(row.get("domain") or 1), "US"),
            "domain": int(row.get("domain") or 1),
            "platform": row.get("platform"),
            "query": row.get("query"),
            "category_id": int(row["category_id"]) if row.get("category_id") is not None else None,
            "category_path": row.get("category_path"),
            "include_descendants": bool(row.get("include_descendants")),
            "limit": int(row.get("limit_count") or 0),
            "window_days": int(row.get("window_days") or 0),
            "min_data_confidence": row.get("min_data_confidence"),
            "include_expandable": bool(row.get("include_expandable")),
            "status": row.get("status"),
            "opportunity_count": int(row.get("opportunity_count") or 0),
            "summary_payload": row.get("summary_payload_json") or {},
            "created_at": _iso_date_or_none(row.get("created_at")),
            "updated_at": _iso_date_or_none(row.get("updated_at")),
            "finished_at": _iso_date_or_none(row.get("finished_at")),
        }
        if include_result:
            job["result_payload"] = row.get("result_payload_json") or {}
        return job

    def get_opportunity_discovery_job_status(self, request: OpportunityDiscoveryJobStatusRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        with _postgres_conn() as conn:
            if request.job_id:
                rows = _run_pg_dict_query(
                    conn,
                    """
                    SELECT *
                    FROM sync.keepa_opportunity_discovery_jobs
                    WHERE job_id = %s
                    LIMIT 1
                    """,
                    [request.job_id],
                )
            else:
                rows = _run_pg_dict_query(
                    conn,
                    """
                    SELECT *
                    FROM sync.keepa_opportunity_discovery_jobs
                    WHERE domain = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    [domain, request.limit],
                )

        return {
            "marketplace": marketplace,
            "domain": domain,
            "job_id": request.job_id,
            "job_count": len(rows),
            "jobs": [self._format_opportunity_discovery_job(row, include_result=request.include_result) for row in rows],
            "notes": [
                "opportunity discovery jobs preserve full tool evidence so agent context can pass compact references without losing facts",
                "result_payload.opportunity_cards_text is the canonical evidence block for this run",
            ],
        }

    def _make_opportunity_id(self, *parts: Any) -> str:
        key = ":".join(str(part or "") for part in parts)
        return f"opp_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:16]}"

    def _filter_opportunities_by_confidence(
        self,
        cards: list[dict[str, Any]],
        *,
        min_data_confidence: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        min_rank = _confidence_rank(min_data_confidence)
        filtered = [card for card in cards if _confidence_rank(str(card.get("data_confidence") or "low")) >= min_rank]
        return filtered[:limit]

    def _memory_profile_rerank_signals(self, memory_profile: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(memory_profile, dict) or not memory_profile:
            return {"has_signal": False, "reason": "memory_profile_missing"}

        profile = memory_profile.get("profile") if isinstance(memory_profile.get("profile"), dict) else memory_profile
        generic_topics = {
            "us",
            "usa",
            "u s",
            "amazon",
            "market",
            "marketplace",
            "seller",
            "sellers",
            "product",
            "products",
            "shopping",
            "ecommerce",
            "e commerce",
            "cross border",
        }
        raw_recent_topics = [str(item).strip().lower() for item in profile.get("recent_topics") or [] if str(item).strip()]
        recent_topics = [topic for topic in raw_recent_topics if topic not in generic_topics]
        ignored_recent_topics = [topic for topic in raw_recent_topics if topic in generic_topics]
        hard_constraints = [str(item).strip().lower() for item in profile.get("hard_constraints") or [] if str(item).strip()]
        market_focus = [str(item).strip().upper() for item in profile.get("market_focus") or [] if str(item).strip()]
        preferred_platforms = [str(item).strip().lower() for item in profile.get("preferred_platforms") or [] if str(item).strip()]
        risk_preference = str(profile.get("risk_preference") or "").strip().lower()
        decision_style = str(profile.get("decision_style") or "").strip().lower()
        preferred_price_band = profile.get("preferred_price_band") if isinstance(profile.get("preferred_price_band"), dict) else {}
        memory_confidence = profile.get("memory_confidence") if isinstance(profile.get("memory_confidence"), dict) else {}

        enough_signal = bool(
            recent_topics
            or hard_constraints
            or preferred_price_band
            or risk_preference in {"low", "medium", "high", "conservative", "evidence_first", "risk_averse"}
            or decision_style in {"evidence_first", "data_first", "fast_scan", "exploratory"}
        )
        if not enough_signal:
            return {
                "has_signal": False,
                "reason": "memory_profile_has_no_rerank_signal",
                "profile_fields_present": sorted(key for key, value in profile.items() if value),
            }

        return {
            "has_signal": True,
            "recent_topics": recent_topics[:12],
            "ignored_recent_topics": ignored_recent_topics[:12],
            "hard_constraints": hard_constraints[:12],
            "market_focus": market_focus[:8],
            "preferred_platforms": preferred_platforms[:8],
            "risk_preference": risk_preference,
            "decision_style": decision_style,
            "preferred_price_band": preferred_price_band,
            "memory_confidence": memory_confidence,
        }

    def _memory_topic_match(self, topic: str, searchable_text: str) -> tuple[bool, bool]:
        topic_tokens = [token for token in re.split(r"[^a-z0-9]+", topic.lower()) if len(token) >= 3]
        topic_tokens = [token for token in topic_tokens if token not in {"the", "and", "for", "with", "market", "amazon", "usa"}]
        if not topic_tokens:
            return False, False

        searchable_tokens = set(token for token in re.split(r"[^a-z0-9]+", searchable_text.lower()) if token)
        if len(topic_tokens) >= 2:
            return all(token in searchable_tokens for token in topic_tokens), True
        token = topic_tokens[0]
        if len(token) < 4:
            return False, False
        return token in searchable_tokens, False

    def _opportunity_memory_profile_adjustment(self, card: dict[str, Any], signals: dict[str, Any]) -> tuple[float, list[str]]:
        adjustment = 0.0
        reasons: list[str] = []
        searchable_text = " ".join(
            str(value or "").lower()
            for value in [
                card.get("title"),
                card.get("category_name"),
                card.get("category_path"),
                (card.get("seller_scope") or {}).get("reason") if isinstance(card.get("seller_scope"), dict) else None,
            ]
        )
        evidence_summary = card.get("evidence_summary") if isinstance(card.get("evidence_summary"), dict) else {}
        data_confidence = str(card.get("data_confidence") or "low").lower()
        price_p50 = evidence_summary.get("price_p50")
        has_strong_positive_signal = False

        for topic in signals.get("recent_topics") or []:
            matches_topic, strong_topic = self._memory_topic_match(topic, searchable_text)
            if topic and matches_topic:
                adjustment += 6.0
                reasons.append(f"recent_topic_match:{topic[:40]}")
                if strong_topic:
                    has_strong_positive_signal = True
                break

        price_band = signals.get("preferred_price_band") if isinstance(signals.get("preferred_price_band"), dict) else {}
        price_min = _safe_float(price_band.get("min"), -1.0)
        price_max = _safe_float(price_band.get("max"), -1.0)
        if price_p50 is not None and (price_min >= 0 or price_max >= 0):
            price_value = _safe_float(price_p50)
            min_ok = price_min < 0 or price_value >= price_min
            max_ok = price_max < 0 or price_value <= price_max
            if min_ok and max_ok:
                adjustment += 4.0
                reasons.append("preferred_price_band_match")
                has_strong_positive_signal = True
            else:
                adjustment -= 3.0
                reasons.append("preferred_price_band_mismatch")

        risk_preference = str(signals.get("risk_preference") or "")
        decision_style = str(signals.get("decision_style") or "")
        if risk_preference in {"conservative", "risk_averse", "evidence_first"} or decision_style in {"evidence_first", "data_first"}:
            if data_confidence == "high":
                adjustment += 4.0
                reasons.append("evidence_first_high_confidence")
            elif data_confidence == "low":
                adjustment -= 4.0
                reasons.append("evidence_first_low_confidence_penalty")

        if "amazon" in signals.get("preferred_platforms", []) and str(card.get("platform") or "").lower() == "amazon":
            adjustment += 1.0
            reasons.append("preferred_platform_match")
        if str(card.get("marketplace") or "").upper() in signals.get("market_focus", []):
            adjustment += 1.0
            reasons.append("market_focus_match")

        for constraint in signals.get("hard_constraints") or []:
            if constraint and constraint in searchable_text:
                adjustment -= 8.0
                reasons.append(f"hard_constraint_text_overlap:{constraint[:40]}")
                break

        if data_confidence == "low" and adjustment > 0 and not has_strong_positive_signal:
            adjustment = 0.0
            reasons.append("low_confidence_weak_preference_guard")

        return max(-12.0, min(12.0, adjustment)), reasons[:6]

    def _rerank_opportunities_with_memory_profile(
        self,
        cards: list[dict[str, Any]],
        memory_profile: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        signals = self._memory_profile_rerank_signals(memory_profile)
        if not signals.get("has_signal"):
            return cards, {
                "personalization_applied": False,
                "reason": signals.get("reason"),
                "profile_fields_present": signals.get("profile_fields_present", []),
            }

        reranked: list[dict[str, Any]] = []
        for original_rank, card in enumerate(cards, start=1):
            adjustment, reasons = self._opportunity_memory_profile_adjustment(card, signals)
            updated = dict(card)
            updated["base_opportunity_score"] = card.get("opportunity_score")
            updated["personalized_opportunity_score"] = round(_safe_float(card.get("opportunity_score")) + adjustment, 2)
            updated["memory_profile_rerank"] = {
                "original_rank": original_rank,
                "score_adjustment": round(adjustment, 2),
                "reasons": reasons,
            }
            reranked.append(updated)

        reranked.sort(
            key=lambda item: (
                _safe_float(item.get("personalized_opportunity_score")),
                _safe_float(item.get("opportunity_score")),
            ),
            reverse=True,
        )
        for personalized_rank, card in enumerate(reranked, start=1):
            card["memory_profile_rerank"]["personalized_rank"] = personalized_rank

        return reranked, {
            "personalization_applied": True,
            "rerank_basis": [
                "recent_topics",
                "preferred_price_band",
                "risk_preference",
                "decision_style",
                "market_focus",
                "preferred_platforms",
                "hard_constraints",
            ],
            "max_score_adjustment": 12.0,
            "profile_signal_counts": {
                "recent_topics": len(signals.get("recent_topics") or []),
                "ignored_recent_topics": len(signals.get("ignored_recent_topics") or []),
                "hard_constraints": len(signals.get("hard_constraints") or []),
                "market_focus": len(signals.get("market_focus") or []),
                "preferred_platforms": len(signals.get("preferred_platforms") or []),
                "has_preferred_price_band": bool(signals.get("preferred_price_band")),
                "has_risk_preference": bool(signals.get("risk_preference")),
                "has_decision_style": bool(signals.get("decision_style")),
            },
        }

    def _seller_scope_blocked_response(
        self,
        *,
        request: OpportunityDiscoveryRequest,
        marketplace: str,
        domain: int,
        decision: SellerScopeDecision,
    ) -> dict[str, Any]:
        result = {
            "marketplace": marketplace,
            "domain": domain,
            "platform": request.platform,
            "query": request.query,
            "category_id": request.category_id,
            "category_path": request.category_path,
            "opportunity_count": 0,
            "opportunities": [],
            "metric_definitions": self._opportunity_metric_definitions(),
            "diagnostics": {
                "seller_scope": decision.as_dict(),
                "seller_scope_filtered": True,
            },
            "notes": [
                "request is outside the configured small cross-border seller product scope",
                "digital/licensed/copyright media and restricted goods are filtered before opportunity analysis",
            ],
        }
        return self._finalize_opportunity_discovery_result(
            request=request,
            marketplace=marketplace,
            domain=domain,
            result=result,
        )

    def _filter_category_opportunity_rows_by_seller_scope(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        blocked_samples: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}
        for row in rows:
            decision = evaluate_seller_scope(
                category_path=row.get("category_path"),
                category_name=row.get("category_name"),
            )
            if decision.allowed:
                row["seller_scope"] = decision.as_dict()
                kept.append(row)
                continue
            reason_counts[decision.reason_code] = reason_counts.get(decision.reason_code, 0) + 1
            if len(blocked_samples) < 5:
                blocked_samples.append(
                    {
                        "category_id": row.get("category_id"),
                        "category_path": row.get("category_path"),
                        "reason_code": decision.reason_code,
                        "matched_terms": list(decision.matched_terms),
                    }
                )

        return kept, {
            "policy_version": evaluate_seller_scope().policy_version,
            "input_count": len(rows),
            "kept_count": len(kept),
            "filtered_count": len(rows) - len(kept),
            "reason_counts": reason_counts,
            "blocked_samples": blocked_samples,
        }

    def _filter_unclassified_category_opportunity_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        dropped_samples: list[dict[str, Any]] = []
        for row in rows:
            category_id = row.get("category_id")
            category_path = str(row.get("category_path") or "").strip()
            category_name = str(row.get("category_name") or "").strip()
            has_readable_category = any(
                value and value.upper() != "UNKNOWN"
                for value in (category_path, category_name)
            )
            if category_id is None and not has_readable_category:
                if len(dropped_samples) < 5:
                    dropped_samples.append(
                        {
                            "category_id": category_id,
                            "category_path": category_path or None,
                            "category_name": category_name or None,
                            "candidate_count": row.get("candidate_count"),
                            "row_count": row.get("row_count"),
                        }
                    )
                continue
            kept.append(row)

        return kept, {
            "input_count": len(rows),
            "kept_count": len(kept),
            "filtered_count": len(rows) - len(kept),
            "reason": "missing_category_id_path_and_name",
            "dropped_samples": dropped_samples,
        }

    def _fetch_category_opportunity_rows(
        self,
        *,
        domain: int,
        category_id: int | None,
        category_path: str | None,
        include_descendants: bool,
        window_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        effective_window_days = max(14, _effective_feature_window_days(window_days))
        with _postgres_conn() as conn:
            return _run_pg_dict_query(
                conn,
                """
                WITH RECURSIVE category_seed AS MATERIALIZED (
                    SELECT %s::BIGINT AS category_id
                ),
                category_tree(category_id, domain, parent_id, category_path, category_name, path_depth) AS MATERIALIZED (
                    SELECT
                        c.category_id,
                        c.domain,
                        c.parent_id,
                        c.category_en::TEXT AS category_path,
                        c.category_en::TEXT AS category_name,
                        1 AS path_depth
                    FROM sync.keepa_category_registry c
                    WHERE c.domain = %s
                      AND c.parent_id IS NULL

                    UNION ALL

                    SELECT
                        child.category_id,
                        child.domain,
                        child.parent_id,
                        CONCAT_WS(' > ', parent.category_path, child.category_en)::TEXT AS category_path,
                        child.category_en::TEXT AS category_name,
                        parent.path_depth + 1 AS path_depth
                    FROM sync.keepa_category_registry child
                    JOIN category_tree parent
                      ON child.domain = parent.domain
                     AND child.parent_id = parent.category_id
                    WHERE child.domain = %s
                      AND child.category_id <> parent.category_id
                      AND parent.path_depth < 12
                ),
                category_subtree(category_id) AS MATERIALIZED (
                    SELECT category_id
                    FROM category_seed
                    WHERE category_id IS NOT NULL

                    UNION

                    SELECT child.category_id
                    FROM sync.keepa_category_registry child
                    JOIN category_subtree parent
                      ON child.parent_id = parent.category_id
                    WHERE child.domain = %s
                      AND %s IS TRUE
                      AND child.category_id <> parent.category_id
                ),
                max_date AS (
                    SELECT MAX(date) AS max_date
                    FROM serving.theme_cross_daily
                    WHERE domain = %s
                ),
                scoped_registry AS MATERIALIZED (
                    SELECT
                        r.asin,
                        r.domain,
                        r.category_id,
                                                COALESCE(NULLIF(r.category_path, ''), ct.category_path) AS category_path,
                                                COALESCE(NULLIF(r.category, ''), ct.category_name) AS category_name,
                        COALESCE(r.is_active, TRUE) AS is_active
                    FROM sync.keepa_asin_registry r
                                        LEFT JOIN category_tree ct
                                            ON ct.domain = r.domain
                                         AND ct.category_id = r.category_id
                    WHERE r.domain = %s
                      AND COALESCE(r.is_active, TRUE) = TRUE
                                            AND (
                                                    r.category_id IS NOT NULL
                                                    OR NULLIF(r.category_path, '') IS NOT NULL
                                                    OR NULLIF(r.category, '') IS NOT NULL
                                            )
                      AND (
                          %s::BIGINT IS NULL
                          OR r.category_id IN (SELECT category_id FROM category_subtree)
                      )
                      AND (
                          %s::TEXT IS NULL
                          OR %s::TEXT = ''
                          OR LOWER(COALESCE(r.category_path, '')) = LOWER(%s::TEXT)
                          OR (%s IS TRUE AND LOWER(COALESCE(r.category_path, '')) LIKE (LOWER(%s::TEXT) || ' > %%'))
                      )
                ),
                filtered AS MATERIALIZED (
                    SELECT
                        d.asin,
                        d.domain,
                        d.date,
                        d.product_title,
                        d.effective_price,
                        d.bsr,
                        d.rating,
                        d.review_count,
                        COALESCE(d.new_offer_count, 0) + COALESCE(d.used_offer_count, 0) AS offer_count,
                        d.estimated_daily_sales,
                        d.trend_index_mean,
                        r.category_id,
                        r.category_path,
                        r.category_name
                    FROM serving.theme_cross_daily d
                    JOIN scoped_registry r
                      ON d.asin = r.asin AND d.domain = r.domain
                    WHERE d.domain = %s
                      AND d.date >= (
                          SELECT max_date - (%s * INTERVAL '1 day')
                          FROM max_date
                      )
                )
                SELECT
                    f.category_id,
                    COALESCE(f.category_path, f.category_name, 'UNKNOWN') AS category_path,
                    COALESCE(f.category_name, SPLIT_PART(COALESCE(f.category_path, 'UNKNOWN'), ' > ', ARRAY_LENGTH(STRING_TO_ARRAY(COALESCE(f.category_path, 'UNKNOWN'), ' > '), 1))) AS category_name,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT f.asin) AS candidate_count,
                    SUM(COALESCE(f.estimated_daily_sales, 0)) AS sales_window_sum,
                    AVG(f.estimated_daily_sales) AS sales_daily_avg,
                    AVG(f.estimated_daily_sales) FILTER (WHERE f.date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS sales_mean_7,
                    AVG(f.estimated_daily_sales) FILTER (WHERE f.date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS sales_mean_prev,
                    AVG(f.trend_index_mean) FILTER (WHERE f.date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_mean_7,
                    AVG(f.trend_index_mean) FILTER (WHERE f.date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_mean_prev,
                    COUNT(*) FILTER (WHERE f.trend_index_mean IS NOT NULL) AS trend_rows,
                    COUNT(*) FILTER (WHERE f.trend_index_mean IS NOT NULL AND f.date >= (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_rows_recent,
                    COUNT(*) FILTER (WHERE f.trend_index_mean IS NOT NULL AND f.date < (SELECT max_date - INTERVAL '6 day' FROM max_date)) AS trend_rows_prev,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY f.effective_price) AS price_p50,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY f.review_count) AS review_count_median,
                    AVG(f.offer_count) AS offer_count_avg,
                    MIN(f.date) AS min_date,
                    MAX(f.date) AS max_date
                FROM filtered f
                GROUP BY 1, 2, 3
                HAVING COUNT(DISTINCT f.asin) >= 3
                ORDER BY sales_window_sum DESC NULLS LAST, candidate_count DESC
                LIMIT %s
                """,
                [
                    category_id,
                    domain,
                    domain,
                    domain,
                    include_descendants,
                    domain,
                    domain,
                    category_id,
                    category_path,
                    category_path,
                    category_path,
                    include_descendants,
                    category_path,
                    domain,
                    effective_window_days - 1,
                    limit,
                ],
            )

    def _build_category_opportunity_card(
        self,
        *,
        row: dict[str, Any],
        marketplace: str,
        window_days: int,
        max_sales_window_sum: float,
        include_expandable: bool,
    ) -> dict[str, Any]:
        candidate_count = _safe_int(row.get("candidate_count"))
        row_count = _safe_int(row.get("row_count"))
        trend_rows = _safe_int(row.get("trend_rows"))
        trend_rows_recent = _safe_int(row.get("trend_rows_recent"))
        trend_rows_prev = _safe_int(row.get("trend_rows_prev"))
        trend_coverage = round(trend_rows / row_count, 4) if row_count else 0.0
        sales_window_sum = _safe_float(row.get("sales_window_sum"))
        sales_mean_7 = _safe_float(row.get("sales_mean_7"))
        sales_mean_prev = _safe_float(row.get("sales_mean_prev"))
        trend_mean_7 = _safe_float(row.get("trend_mean_7"))
        trend_mean_prev = _safe_float(row.get("trend_mean_prev"))
        sales_momentum_pct = ((sales_mean_7 - sales_mean_prev) / sales_mean_prev * 100.0) if sales_mean_prev > 0 else 0.0
        if trend_rows > 0 and trend_rows_recent <= 0 and row.get("trend_mean_7") is not None:
            trend_rows_recent = trend_rows
        if trend_rows > 0 and trend_rows_prev <= 0 and row.get("trend_mean_prev") is not None:
            trend_rows_prev = trend_rows
        trend_signal = self._trend_momentum_signal(
            recent_mean=trend_mean_7,
            previous_mean=trend_mean_prev,
            recent_rows=trend_rows_recent,
            previous_rows=trend_rows_prev,
            total_rows=trend_rows,
        )
        trend_momentum_pct = trend_signal.get("score_pct")
        offer_count_avg = _safe_float(row.get("offer_count_avg"))
        review_count_median = _safe_float(row.get("review_count_median"))
        price_p50 = row.get("price_p50")
        demand_score = _bounded_score((sales_window_sum / max(max_sales_window_sum, 1.0)) * 100.0)
        trend_score = _bounded_score(50.0 + trend_momentum_pct * 2.0 + sales_momentum_pct * 0.6)
        competition_headroom_score = _bounded_score(100.0 - min(60.0, offer_count_avg * 2.5) - min(25.0, review_count_median / 800.0))
        coverage_gap_score = _bounded_score(85.0 - min(55.0, candidate_count * 2.0)) if include_expandable else 40.0
        evidence_quality_score = _bounded_score(min(candidate_count * 6.0, 60.0) + trend_coverage * 40.0)
        score_breakdown = {
            "demand_score": demand_score,
            "trend_score": trend_score,
            "competition_headroom_score": competition_headroom_score,
            "price_fit_score": _price_fit_score(price_p50),
            "forecast_growth_score": trend_score,
            "coverage_gap_score": coverage_gap_score,
            "evidence_quality_score": evidence_quality_score,
        }
        data_confidence = self._opportunity_data_confidence(
            candidate_count=candidate_count,
            trend_coverage=trend_coverage,
            row_count=row_count,
        )
        category_path = str(row.get("category_path") or "UNKNOWN")
        category_name = _opportunity_title_from_category_path(category_path, row.get("category_name"))
        category_id_value = _safe_int(row.get("category_id")) if row.get("category_id") is not None else None
        next_action = {
            "type": "analyze_theme",
            "label": "进入商品主题分析" if category_id_value is not None else "先解析类目后进入商品主题分析",
            "requires_category_resolve": category_id_value is None,
            "request": {
                "product_query": category_name,
                "marketplace": marketplace,
                "category_id": category_id_value,
                "category_path": category_path,
                "recall_mode": "category",
                "include_descendants": True,
            },
        }
        if category_id_value is None:
            next_action["preflight_tool"] = "category_resolve"
            next_action["preflight_request"] = {
                "category_path": category_path,
                "marketplace": marketplace,
            }
            next_action["readiness_note"] = "category_id is missing; call category_resolve before category recall for a stable execution key."
        return {
            "opportunity_id": self._make_opportunity_id(marketplace, "category", row.get("category_id"), category_path),
            "title": category_name,
            "marketplace": marketplace,
            "platform": "Amazon",
            "source": "local_category_opportunity_scan",
            "category_id": category_id_value,
            "category_name": category_name,
            "raw_category_name": str(row.get("category_name") or "").strip() or None,
            "category_path": category_path,
            "candidate_pool_id": None,
            "seller_scope": row.get("seller_scope") or evaluate_seller_scope(
                category_path=category_path,
                category_name=category_name,
            ).as_dict(),
            "opportunity_score": self._build_opportunity_score(score_breakdown),
            "score_breakdown": score_breakdown,
            "evidence_summary": {
                "window_days": window_days,
                "candidate_count": candidate_count,
                "row_count": row_count,
                "sales_window_sum": round(sales_window_sum, 2),
                "sales_momentum_pct": round(sales_momentum_pct, 2),
                "trend_momentum_pct": trend_signal.get("value_pct"),
                "trend_momentum_display": trend_signal.get("display"),
                "trend_signal_status": trend_signal.get("status"),
                "trend_signal_interpretation": trend_signal.get("interpretation"),
                "trend_coverage": trend_coverage,
                "trend_rows_recent": trend_rows_recent,
                "trend_rows_previous": trend_rows_prev,
                "price_p50": round(_safe_float(price_p50), 2) if price_p50 is not None else None,
                "offer_count_avg": round(offer_count_avg, 2),
                "review_count_median": round(review_count_median, 2),
                "data_max_date": _iso_date_or_none(row.get("max_date")),
            },
            "metric_explanations": self._build_category_metric_explanations(
                score_breakdown=score_breakdown,
                candidate_count=candidate_count,
                row_count=row_count,
                window_days=window_days,
                sales_window_sum=sales_window_sum,
                sales_mean_7=sales_mean_7,
                sales_mean_prev=sales_mean_prev,
                sales_momentum_pct=sales_momentum_pct,
                trend_mean_7=trend_mean_7,
                trend_mean_prev=trend_mean_prev,
                trend_momentum_pct=trend_momentum_pct,
                offer_count_avg=offer_count_avg,
                trend_coverage=trend_coverage,
            ),
            "data_confidence": data_confidence,
            "next_action": next_action,
        }

    def _build_query_opportunity_card(
        self,
        *,
        request: OpportunityDiscoveryRequest,
        marketplace: str,
        resolved: dict[str, Any],
        stats: dict[str, Any],
        trends: dict[str, Any],
        forecast: dict[str, Any],
        benchmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        candidate_count = _safe_int(stats.get("candidate_count"))
        trend_coverage = _safe_float((trends.get("trend_data_coverage") if trends else None), 0.0)
        trend_wow = _safe_float(trends.get("trend_wow") if trends else None)
        bullish_count = _safe_int(forecast.get("bullish_asin_count"))
        risk_count = _safe_int(forecast.get("risk_asin_count"))
        forecast_total = bullish_count + risk_count
        bullish_ratio = bullish_count / forecast_total if forecast_total else 0.5
        sales_window_sum = _safe_float(stats.get("sales_window_sum"))
        sales_window_avg = _safe_float(stats.get("sales_window_avg"))
        offer_count_median = _safe_float(stats.get("offer_count_median"))
        review_count_median = _safe_float(stats.get("review_count_median"))
        score_breakdown = {
            "demand_score": _bounded_score(min(100.0, sales_window_avg * 1.8 + sales_window_sum / 400.0)),
            "trend_score": _bounded_score(50.0 + trend_wow * 4.0),
            "competition_headroom_score": _bounded_score(100.0 - min(60.0, offer_count_median * 2.5) - min(25.0, review_count_median / 800.0)),
            "price_fit_score": _price_fit_score((stats.get("price_distribution") or {}).get("p50")),
            "forecast_growth_score": _bounded_score(bullish_ratio * 100.0),
            "coverage_gap_score": _bounded_score(100.0 - min(100.0, candidate_count * 4.0)) if request.include_expandable else 40.0,
            "evidence_quality_score": _bounded_score(min(candidate_count * 6.0, 60.0) + trend_coverage * 40.0),
        }
        data_confidence = self._opportunity_data_confidence(
            candidate_count=candidate_count,
            trend_coverage=trend_coverage,
            row_count=max(candidate_count * request.window_days, 0),
        )
        category_constraint = resolved.get("category_constraint") or {}
        title = request.query or request.category_path or _leaf_category_name(category_constraint.get("category_path")) or "Amazon opportunity"
        return {
            "opportunity_id": self._make_opportunity_id(marketplace, "query", title, resolved.get("candidate_pool_id")),
            "title": title,
            "marketplace": marketplace,
            "platform": "Amazon",
            "source": "resolved_candidate_pool",
            "category_id": category_constraint.get("category_id") or request.category_id,
            "category_path": category_constraint.get("category_path") or request.category_path,
            "candidate_pool_id": resolved.get("candidate_pool_id"),
            "seller_scope": evaluate_seller_scope(
                category_path=category_constraint.get("category_path") or request.category_path,
                category_name=_leaf_category_name(category_constraint.get("category_path") or request.category_path),
                query=title,
            ).as_dict(),
            "opportunity_score": self._build_opportunity_score(score_breakdown),
            "score_breakdown": score_breakdown,
            "evidence_summary": {
                "window_days": request.window_days,
                "candidate_count": candidate_count,
                "candidate_asins": resolved.get("candidate_asins", [])[:20],
                "pool_quality": resolved.get("pool_quality") or {},
                "sales_window_sum": round(sales_window_sum, 2),
                "sales_window_avg": round(sales_window_avg, 2),
                "price_distribution": stats.get("price_distribution") or {},
                "offer_count_median": round(offer_count_median, 2),
                "review_count_median": round(review_count_median, 2),
                "trend_stage": trends.get("trend_stage") if trends else None,
                "trend_wow": trends.get("trend_wow") if trends else None,
                "trend_coverage": trend_coverage,
                "forecast_type": forecast.get("forecast_type"),
                "bullish_asin_count": bullish_count,
                "risk_asin_count": risk_count,
                "predicted_top_asins": forecast.get("predicted_top_asins", [])[:5],
                "benchmark_anchor": (benchmark or {}).get("benchmark_anchor"),
                "benchmark_is_precise": (benchmark or {}).get("benchmark_is_precise"),
                "data_max_date": _iso_date_or_none(stats.get("data_max_date")),
            },
            "metric_explanations": self._build_query_metric_explanations(
                score_breakdown=score_breakdown,
                candidate_count=candidate_count,
                window_days=request.window_days,
                sales_window_sum=sales_window_sum,
                sales_window_avg=sales_window_avg,
                trend_wow=trend_wow,
                trend_coverage=trend_coverage,
                offer_count_median=offer_count_median,
            ),
            "data_confidence": data_confidence,
            "next_action": {
                "type": "quick_report",
                "label": "生成快速选品报告",
                "request": {
                    "product_query": title,
                    "marketplace": marketplace,
                    "candidate_pool_id": resolved.get("candidate_pool_id"),
                    "candidate_asins": resolved.get("candidate_asins", [])[:50],
                    "category_id": category_constraint.get("category_id") or request.category_id,
                    "category_path": category_constraint.get("category_path") or request.category_path,
                },
            },
        }

    async def discover_opportunities(self, request: OpportunityDiscoveryRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        notes = [
            "opportunity discovery MVP is local and read-only; theme-api does not consume Keepa tokens here",
            "scores are directional and should route promising cards into product theme analysis before final selection",
            "seller_scope_policy=cross_border_sme_v1 filters non-physical, licensed, copyright-media, and restricted goods before ranking",
        ]

        request_scope_decision = evaluate_seller_scope(
            category_path=request.category_path,
            category_name=_leaf_category_name(request.category_path),
            query=request.query,
        )
        if not request_scope_decision.allowed:
            return self._seller_scope_blocked_response(
                request=request,
                marketplace=marketplace,
                domain=domain,
                decision=request_scope_decision,
            )

        if request.query:
            resolve_request = ResolveCandidatesRequest(
                product_query=request.query,
                marketplace=marketplace,
                recall_mode="hybrid" if request.category_id is not None or request.category_path else "keyword",
                category_id=request.category_id,
                category_path=request.category_path,
                include_descendants=request.include_descendants,
                min_pool_size=DEFAULT_MIN_CANDIDATE_POOL_SIZE,
                target_pool_size=max(DEFAULT_TARGET_CANDIDATE_POOL_SIZE, request.limit * 4),
                expand_if_small=request.include_expandable,
                max_candidates=max(50, request.limit * 10),
            )
            resolved = await self.resolve_candidates(resolve_request)
            category_constraint = resolved.get("category_constraint") or {}
            resolved_scope_decision = evaluate_seller_scope(
                category_path=category_constraint.get("category_path") or request.category_path,
                category_name=_leaf_category_name(category_constraint.get("category_path") or request.category_path),
                query=request.query,
            )
            if not resolved_scope_decision.allowed:
                return self._seller_scope_blocked_response(
                    request=request,
                    marketplace=marketplace,
                    domain=domain,
                    decision=resolved_scope_decision,
                )
            candidate_asins = _sanitize_asins(resolved.get("candidate_asins", []))
            if not candidate_asins:
                return self._finalize_opportunity_discovery_result(request=request, domain=domain, marketplace=marketplace, result={
                    "marketplace": marketplace,
                    "domain": domain,
                    "platform": request.platform,
                    "query": request.query,
                    "category_id": request.category_id,
                    "category_path": request.category_path,
                    "opportunity_count": 0,
                    "opportunities": [],
                    "metric_definitions": self._opportunity_metric_definitions(),
                    "notes": notes + ["no local candidate pool was resolved; queue candidate expansion before analysis"],
                })

            pool_request = CandidatePoolRequest(
                candidate_asins=candidate_asins,
                candidate_pool_id=resolved.get("candidate_pool_id"),
                marketplace=marketplace,
                window_days=request.window_days,
            )
            stats = self.get_candidate_pool_stats(pool_request)
            trends = self.get_candidate_pool_trends(pool_request)
            forecast = self.get_candidate_pool_weak_forecast(
                WeakForecastRequest(
                    candidate_asins=candidate_asins,
                    candidate_pool_id=resolved.get("candidate_pool_id"),
                    marketplace=marketplace,
                    window_days=request.window_days,
                    top_n=min(FORECAST_TOP_ASINS_LIMIT, request.limit),
                )
            )
            benchmark: dict[str, Any] | None = None
            with contextlib.suppress(Exception):
                benchmark = self.get_category_benchmark(
                    CategoryBenchmarkRequest(
                        candidate_asins=candidate_asins,
                        candidate_pool_id=resolved.get("candidate_pool_id"),
                        marketplace=marketplace,
                        window_days=request.window_days,
                        benchmark_category_id=request.category_id,
                        benchmark_category_path=request.category_path,
                        include_descendants=request.include_descendants,
                    )
                )
            card = self._build_query_opportunity_card(
                request=request,
                marketplace=marketplace,
                resolved=resolved,
                stats=stats,
                trends=trends,
                forecast=forecast,
                benchmark=benchmark,
            )
            opportunities = self._filter_opportunities_by_confidence(
                [card],
                min_data_confidence=request.min_data_confidence,
                limit=request.limit,
            )
            opportunities, personalization_summary = self._rerank_opportunities_with_memory_profile(opportunities, request.memory_profile)
            return self._finalize_opportunity_discovery_result(request=request, domain=domain, marketplace=marketplace, result={
                "marketplace": marketplace,
                "domain": domain,
                "platform": request.platform,
                "query": request.query,
                "category_id": request.category_id,
                "category_path": request.category_path,
                "opportunity_count": len(opportunities),
                "opportunities": opportunities,
                "metric_definitions": self._opportunity_metric_definitions(),
                "diagnostics": {
                    "resolved_candidate_count": len(candidate_asins),
                    "candidate_pool_id": resolved.get("candidate_pool_id"),
                    "recall_mode": resolved.get("recall_mode"),
                    "seller_scope": resolved_scope_decision.as_dict(),
                    "filtered_by_min_data_confidence": len(opportunities) == 0,
                    "memory_profile_rerank": personalization_summary,
                },
                "notes": notes,
            })

        rows = self._fetch_category_opportunity_rows(
            domain=domain,
            category_id=request.category_id,
            category_path=request.category_path,
            include_descendants=request.include_descendants,
            window_days=request.window_days,
            limit=max(request.limit * 12, 160),
        )
        rows, unclassified_summary = self._filter_unclassified_category_opportunity_rows(rows)
        rows, seller_scope_summary = self._filter_category_opportunity_rows_by_seller_scope(rows)
        max_sales_window_sum = max([_safe_float(row.get("sales_window_sum")) for row in rows] or [1.0])
        cards = [
            self._build_category_opportunity_card(
                row=row,
                marketplace=marketplace,
                window_days=request.window_days,
                max_sales_window_sum=max_sales_window_sum,
                include_expandable=request.include_expandable,
            )
            for row in rows
        ]
        cards.sort(key=lambda item: item["opportunity_score"], reverse=True)
        cards, personalization_summary = self._rerank_opportunities_with_memory_profile(cards, request.memory_profile)
        opportunities, duplicate_title_summary = self._select_opportunities_by_confidence_and_title(
            cards,
            min_data_confidence=request.min_data_confidence,
            limit=request.limit,
        )
        return self._finalize_opportunity_discovery_result(request=request, domain=domain, marketplace=marketplace, result={
            "marketplace": marketplace,
            "domain": domain,
            "platform": request.platform,
            "query": request.query,
            "category_id": request.category_id,
            "category_path": request.category_path,
            "opportunity_count": len(opportunities),
            "opportunities": opportunities,
            "metric_definitions": self._opportunity_metric_definitions(),
            "diagnostics": {
                "scanned_category_count": seller_scope_summary["input_count"],
                "unclassified_category_filter": unclassified_summary,
                "seller_scope": seller_scope_summary,
                "filtered_by_min_data_confidence": duplicate_title_summary["eligible_count"] < min(len(cards), request.limit),
                "duplicate_title_filter": duplicate_title_summary,
                "window_days": request.window_days,
                "memory_profile_rerank": personalization_summary,
            },
            "notes": notes,
        })

    def get_candidate_expansion_status(self, request: CandidateExpansionJobStatusRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        with _postgres_conn() as conn:
            if request.job_id:
                rows = _run_pg_dict_query(
                    conn,
                    """
                    SELECT *
                    FROM sync.keepa_candidate_expansion_jobs
                    WHERE job_id = %s
                    LIMIT 1
                    """,
                    [request.job_id],
                )
            else:
                rows = _run_pg_dict_query(
                    conn,
                    """
                    SELECT *
                    FROM sync.keepa_candidate_expansion_jobs
                    WHERE domain = %s
                      AND (%s::TEXT[] IS NULL OR status = ANY(%s::TEXT[]))
                    ORDER BY
                      CASE priority
                        WHEN 'interactive_high' THEN 1
                        WHEN 'interactive_normal' THEN 2
                        WHEN 'background_high' THEN 3
                        ELSE 4
                      END,
                      created_at ASC
                    LIMIT %s
                    """,
                    [domain, request.statuses or None, request.statuses or None, request.limit],
                )

            jobs = [self._format_candidate_expansion_job(row) for row in rows]
            for job in jobs:
                job["data_readiness"] = self._fetch_candidate_expansion_data_readiness(conn, job)

        return {
            "marketplace": marketplace,
            "domain": domain,
            "job_id": request.job_id,
            "job_count": len(jobs),
            "jobs": jobs,
            "notes": [
                "queued and waiting_token jobs do not consume Keepa tokens until collector reserves them",
                "completed means discovered ASINs are visible in PostgreSQL registry; check data_readiness.analysis_ready before running stats or benchmark analysis",
                "data_readiness distinguishes registry sync, product hydration, and serving feature sync readiness",
            ],
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
        category_id: int | None,
        category_path: str | None,
        include_descendants: bool,
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
                WITH RECURSIVE query_terms AS MATERIALIZED (
                    SELECT COALESCE(%s::TEXT[], ARRAY[]::TEXT[]) AS terms
                ),
                required_product_terms AS MATERIALIZED (
                    SELECT COALESCE(%s::TEXT[], ARRAY[]::TEXT[]) AS terms
                ),
                category_seed AS MATERIALIZED (
                    SELECT %s::BIGINT AS category_id
                ),
                category_subtree(category_id) AS MATERIALIZED (
                    SELECT category_id
                    FROM category_seed
                    WHERE category_id IS NOT NULL

                    UNION

                    SELECT child.category_id
                    FROM sync.keepa_category_registry child
                    JOIN category_subtree parent
                      ON child.parent_id = parent.category_id
                    WHERE child.domain = %s
                      AND %s IS TRUE
                      AND child.category_id <> parent.category_id
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
                          %s::BIGINT IS NULL
                          OR r.category_id IN (SELECT category_id FROM category_subtree)
                      )
                      AND (
                          %s::TEXT IS NULL
                          OR %s::TEXT = ''
                          OR LOWER(COALESCE(r.category_path, '')) = LOWER(%s::TEXT)
                          OR (%s IS TRUE AND LOWER(COALESCE(r.category_path, '')) LIKE (LOWER(%s::TEXT) || ' > %%'))
                      )
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
                    category_id,
                    domain,
                    include_descendants,
                    domain,
                    active_only,
                    price_min,
                    price_min,
                    price_max,
                    price_max,
                    category_id,
                    category_path,
                    category_path,
                    category_path,
                    include_descendants,
                    category_path,
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
        recall_mode = request.recall_mode
        category_scope_applied = bool(request.category_id is not None or request.category_path)
        category_only_recall = recall_mode == "category" and category_scope_applied
        effective_sql_prefilter_terms = [] if category_only_recall else _build_sql_prefilter_terms(normalized_phrases, tokens)
        effective_required_product_terms = [] if category_only_recall else required_product_terms
        sql_required_product_terms = _build_sql_prefilter_terms(effective_required_product_terms, [], max_terms=12)
        sql_prefilter_limit = _candidate_sql_prefilter_limit(max(request.max_candidates, request.target_pool_size))

        def run_domain_fetch() -> tuple[list[dict[str, Any]], int]:
            started_at = time.perf_counter()
            result = self._fetch_domain_candidates(
                domain,
                sql_prefilter_terms=effective_sql_prefilter_terms,
                sql_required_product_terms=sql_required_product_terms,
                category_id=request.category_id,
                category_path=request.category_path,
                include_descendants=request.include_descendants,
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
                if not category_only_recall:
                    continue
                score = 1.0
                reasons = ["category_scope_match"]
                breakdown = {"category_scope_match": 1.0}
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
                effective_required_product_terms,
            )
        ]
        category_anchored_candidates = [
            item
            for item in candidates
            if (item.get("match_breakdown") or {}).get("semantic_field_required_product_terms")
        ]
        semantic_fine_category_anchor_applied = bool(effective_required_product_terms and fine_category_anchored_candidates)
        semantic_category_anchor_applied = bool(effective_required_product_terms and (fine_category_anchored_candidates or category_anchored_candidates))
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

        matched_leaf_categories = _category_distribution(candidate_items, "leaf_category_name")
        matched_fine_categories = _category_distribution(candidate_items, "fine_category_name")
        matched_root_categories = _category_distribution(candidate_items, "root_category_name")
        pool_quality = _build_candidate_pool_quality(
            candidate_items,
            candidate_total_before_semantic_gate=len(rows),
            candidate_total_before_category_anchor=candidate_total_before_semantic_category_anchor,
            leaf_distribution=matched_leaf_categories,
            fine_distribution=matched_fine_categories,
            root_distribution=matched_root_categories,
            min_pool_size=request.min_pool_size,
            target_pool_size=request.target_pool_size,
        )
        normalized_query = normalized_phrases[0] if normalized_phrases else _normalize_text(normalization.product_query)
        candidate_pool_id = str(uuid.uuid4())
        candidate_pool_lineage = self._build_candidate_pool_lineage(
            request=request,
            marketplace=marketplace,
            domain=domain,
            normalized_query=normalized_query,
            normalized_phrases=normalized_phrases,
            tokens=tokens,
            query_expansions=query_expansions,
        )
        candidate_pool_persistence = await asyncio.to_thread(
            self._persist_candidate_pool,
            candidate_pool_id=candidate_pool_id,
            domain=domain,
            marketplace=marketplace,
            request=request,
            normalized_query=normalized_query,
            candidate_items=candidate_items,
            candidate_total_before_truncate=len(candidates),
            pool_quality=pool_quality,
            lineage=candidate_pool_lineage,
        )

        response_data = {
            "marketplace": marketplace,
            "domain": domain,
            "candidate_pool_id": candidate_pool_id,
            "candidate_pool_version": 1,
            "candidate_pool_lineage": candidate_pool_lineage,
            "candidate_pool_persistence": candidate_pool_persistence,
            "raw_product_query": request.product_query,
            "recall_mode": recall_mode,
            "category_constraint": {
                "applied": category_scope_applied,
                "category_id": request.category_id,
                "category_path": request.category_path,
                "include_descendants": request.include_descendants,
                "category_only_recall": category_only_recall,
            },
            "expand_if_small": request.expand_if_small,
            "normalized_query": normalized_query,
            "query_phrases": normalized_phrases,
            "query_tokens": tokens,
            "query_expansions": query_expansions,
            "required_product_terms": required_product_terms,
            "effective_required_product_terms": effective_required_product_terms,
            "ranking_policy": {
                "primary_sort": ["match_score", "business_priority", "has_sales_signal_30d", "current_review_count"],
                "sql_prefilter_sort": ["sql_prefilter_score", "business_priority", "current_review_count"],
                "semantic_gate": "required_product_terms must match high-signal product fields before a candidate can enter the final pool",
                "semantic_category_anchor": "when leaf/fine category anchors exist they are preferred; otherwise category/keyword anchored candidates exclude title-only matches",
                "recall_modes": {
                    "keyword": "default multi-field keyword recall over title/category/category_path/search_term/keyword",
                    "hybrid": "apply category_id/category_path scope first, then keep keyword semantic gating inside that scope",
                    "category": "apply category_id/category_path scope as the recall pool and use ranking boosters when keyword score is absent",
                },
                "match_score_components": ["phrase_score", "token_score", "business_score", "freshness_score", "completeness_score"],
                "matched_fields": ["product_title", "category", "category_path", "leaf_category_name", "fine_category_name", "keywords"],
                "note": "candidate filtering uses SQL lexical/category prefilter plus required product-term gating before Python exact scoring; product-specific core-vs-adjacent cleanup should use returned candidate_items fields or a downstream configurable classifier, not hard-coded product names",
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
            "pool_quality": pool_quality,
            "semantic_fine_category_anchor_applied": semantic_fine_category_anchor_applied,
            "semantic_category_anchor_applied": semantic_category_anchor_applied,
            "candidate_sql_prefilter_count": sql_prefilter_total_count,
            "candidate_sql_prefilter_limit": sql_prefilter_limit,
            "candidate_sql_prefilter_truncated": sql_prefilter_total_count > sql_prefilter_limit,
            "category_scope_applied": category_scope_applied,
            "truncated": truncated,
            "matched_categories": _unique_nonempty([item["category"] for item in candidate_items])[:10],
            "matched_leaf_categories": matched_leaf_categories,
            "matched_fine_categories": matched_fine_categories,
            "matched_root_categories": matched_root_categories,
            "matched_keywords": _unique_nonempty([keyword for item in candidate_items for keyword in item["keywords"]])[:20],
            "candidate_asins": [item["asin"] for item in candidate_items],
            "candidate_items": candidate_items,
            "recall_notes": [
                "when configured, recall preparation now runs as two internal stages: theme extraction first, then recall normalization",
                "the external resolve_candidates tool surface stays merged; only the internal service layer is split into extraction and normalization",
                "candidate pool is resolved by multi-field recall over title/category/category_path/search_term/keyword with required product-term gating",
                "category/hybrid recall can constrain the pool by category_id and/or category_path before scoring; category-only mode is suitable for补池 or broad category coverage checks",
                "expand_if_small is accepted as a planning signal; online Keepa token-consuming expansion is handled by the collection layer rather than this local resolver",
                "SQL prefilter narrows the domain candidate set before Python exact scoring, so downstream tools should rely on returned candidate_asins rather than re-scanning visible titles",
                "leaf_category_name and fine_category_name are preferred for product analysis; root/L2/L3 categories can be too coarse for candidate cleanup",
                "built-in Chinese-to-English query expansion remains as a fallback bridge for common product terms when source data is English-dominant",
                "business_priority and recent data completeness are used as ranking boosters, not hard filters",
            ],
            "response_profile": request.response_profile,
        }

        if request.response_profile == "compact" and not request.include_debug:
            query_normalization = response_data.get("query_normalization") or {}
            pipeline_llm_used = bool(query_normalization.get("pipeline_llm_used"))

            def compact_candidate_item(item: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: value
                    for key, value in {
                        "asin": item.get("asin"),
                        "product_title": item.get("product_title"),
                        "brand": item.get("brand"),
                        "category": item.get("category"),
                        "category_path": item.get("category_path"),
                        "leaf_category_name": item.get("leaf_category_name"),
                        "fine_category_name": item.get("fine_category_name"),
                        "current_price": item.get("current_price"),
                        "current_rating": item.get("current_rating"),
                        "current_review_count": item.get("current_review_count"),
                        "current_offer_count": item.get("current_offer_count"),
                    }.items()
                    if value is not None and value != ""
                }

            return {
                "marketplace": response_data["marketplace"],
                "domain": response_data["domain"],
                "candidate_pool_id": response_data["candidate_pool_id"],
                "candidate_pool_version": response_data["candidate_pool_version"],
                "candidate_pool_persistence": response_data["candidate_pool_persistence"],
                "raw_product_query": response_data["raw_product_query"],
                "recall_mode": response_data["recall_mode"],
                "category_constraint": response_data["category_constraint"],
                "expand_if_small": response_data["expand_if_small"],
                "normalized_query": response_data["normalized_query"],
                "query_phrases": response_data["query_phrases"][:5],
                "query_tokens": response_data["query_tokens"][:12],
                "query_expansions": response_data["query_expansions"][:8],
                "required_product_terms": response_data["required_product_terms"][:8],
                "effective_required_product_terms": response_data["effective_required_product_terms"][:8],
                "timing_ms": response_data["timing_ms"],
                "query_normalization_summary": {
                    "mode": query_normalization.get("mode"),
                    "pipeline_mode": query_normalization.get("pipeline_mode"),
                    "llm_used": query_normalization.get("llm_used"),
                    "pipeline_llm_used": pipeline_llm_used,
                    "llm_provider": query_normalization.get("llm_provider"),
                    "llm_model": query_normalization.get("llm_model"),
                    "llm_confidence": query_normalization.get("llm_confidence"),
                    "normalized_product_query": query_normalization.get("normalized_product_query"),
                    "normalized_query_aliases": query_normalization.get("normalized_query_aliases") or [],
                    "normalized_category_hints": query_normalization.get("normalized_category_hints") or [],
                },
                "candidate_count": response_data["candidate_count"],
                "candidate_total_before_truncate": response_data["candidate_total_before_truncate"],
                "candidate_total_before_semantic_category_anchor": response_data[
                    "candidate_total_before_semantic_category_anchor"
                ],
                "pool_quality": response_data["pool_quality"],
                "semantic_fine_category_anchor_applied": response_data["semantic_fine_category_anchor_applied"],
                "semantic_category_anchor_applied": response_data["semantic_category_anchor_applied"],
                "candidate_sql_prefilter_count": response_data["candidate_sql_prefilter_count"],
                "candidate_sql_prefilter_limit": response_data["candidate_sql_prefilter_limit"],
                "candidate_sql_prefilter_truncated": response_data["candidate_sql_prefilter_truncated"],
                "category_scope_applied": response_data["category_scope_applied"],
                "truncated": response_data["truncated"],
                "matched_categories": response_data["matched_categories"][:10],
                "matched_leaf_categories": response_data["matched_leaf_categories"],
                "matched_fine_categories": response_data["matched_fine_categories"],
                "matched_root_categories": response_data["matched_root_categories"],
                "matched_keywords": response_data["matched_keywords"][:12],
                "candidate_asins": response_data["candidate_asins"],
                "candidate_items": [compact_candidate_item(item) for item in candidate_items],
                "response_profile": "compact",
            }

        return response_data

    def get_candidate_pool_stats(self, request: CandidatePoolRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
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
            "candidate_pool": candidate_pool_ref,
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

    def get_candidate_pool_slice(self, request: CandidatePoolSliceRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
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
                SELECT *
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
                s.bsr_avg_window,
                (SELECT max_date FROM max_date) AS max_date
            FROM latest l
            LEFT JOIN summary s USING (asin, domain)
                """,
                [domain, domain, candidate_asins, effective_window_days - 1],
            )

        brand_terms = [term.strip().lower() for term in request.brand_include if term.strip()]
        title_terms = [term.strip().lower() for term in request.title_keywords if term.strip()]
        material_terms = [term.strip().lower() for term in request.material_keywords if term.strip()]

        def row_matches(row: dict[str, Any]) -> bool:
            brand_text = str(row.get("brand") or "").lower()
            searchable_text = " ".join(
                str(row.get(key) or "") for key in ("product_title", "category", "brand")
            ).lower()
            if brand_terms and not any(term in brand_text for term in brand_terms):
                return False
            if title_terms and not any(term in searchable_text for term in title_terms):
                return False
            if material_terms and not any(term in searchable_text for term in material_terms):
                return False
            return True

        matched_rows = [dict(row) for row in rows if row_matches(row)]

        sort_by = request.sort_by
        sort_field = {
            "sales_window_sum": "sales_window_sum",
            "sales_daily_avg": "sales_daily_avg",
            "review_count": "review_count",
            "rating": "rating",
            "bsr": "bsr",
            "price": "effective_price",
        }[sort_by]

        def numeric_value(row: dict[str, Any], field: str) -> float | None:
            value = row.get(field)
            if value is None:
                return None
            with contextlib.suppress(TypeError, ValueError):
                return float(value)
            return None

        if sort_by == "bsr":
            matched_rows.sort(key=lambda row: numeric_value(row, sort_field) if numeric_value(row, sort_field) is not None else float("inf"))
        else:
            matched_rows.sort(key=lambda row: numeric_value(row, sort_field) if numeric_value(row, sort_field) is not None else float("-inf"), reverse=True)

        top_rows = matched_rows[: request.top_n]

        def percentile(values: list[float], ratio: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            if len(ordered) == 1:
                return ordered[0]
            position = (len(ordered) - 1) * ratio
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        ratings = [value for value in (numeric_value(row, "rating") for row in matched_rows) if value is not None]
        review_counts = [value for value in (numeric_value(row, "review_count") for row in matched_rows) if value is not None]
        brands: dict[str, int] = {}
        for row in matched_rows:
            brand = str(row.get("brand") or "UNKNOWN").strip() or "UNKNOWN"
            brands[brand] = brands.get(brand, 0) + 1

        def rating_bucket(value: float | None) -> str:
            if value is None:
                return "missing"
            if value >= 4.7:
                return "4.7_plus"
            if value >= 4.5:
                return "4.5_to_4.69"
            if value >= 4.3:
                return "4.3_to_4.49"
            return "below_4.3"

        def review_bucket(value: float | None) -> str:
            if value is None:
                return "missing"
            if value < 100:
                return "lt_100"
            if value < 500:
                return "100_to_499"
            if value < 2000:
                return "500_to_1999"
            return "2000_plus"

        rating_buckets: dict[str, int] = {}
        review_buckets: dict[str, int] = {}
        for row in matched_rows:
            rb = rating_bucket(numeric_value(row, "rating"))
            cb = review_bucket(numeric_value(row, "review_count"))
            rating_buckets[rb] = rating_buckets.get(rb, 0) + 1
            review_buckets[cb] = review_buckets.get(cb, 0) + 1

        def compact_item(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "asin": row.get("asin"),
                "product_title": row.get("product_title"),
                "brand": row.get("brand"),
                "category": row.get("category"),
                "effective_price": round(float(row["effective_price"]), 2) if row.get("effective_price") is not None else None,
                "rating": round(float(row["rating"]), 2) if row.get("rating") is not None else None,
                "review_count": int(row.get("review_count") or 0) if row.get("review_count") is not None else None,
                "bsr": int(row.get("bsr") or 0) if row.get("bsr") is not None else None,
                "estimated_daily_sales": round(float(row["estimated_daily_sales"]), 2) if row.get("estimated_daily_sales") is not None else None,
                "sales_window_sum": round(float(row["sales_window_sum"]), 2) if row.get("sales_window_sum") is not None else None,
                "sales_daily_avg": round(float(row["sales_daily_avg"]), 2) if row.get("sales_daily_avg") is not None else None,
                "review_growth_window": round(float(row["review_growth_window"]), 2) if row.get("review_growth_window") is not None else None,
                "offer_count_avg_window": round(float(row["offer_count_avg_window"]), 2) if row.get("offer_count_avg_window") is not None else None,
                "latest_date": row.get("latest_date"),
            }

        return {
            "marketplace": marketplace,
            "domain": domain,
            "candidate_pool": candidate_pool_ref,
            "data_source": "local_postgres",
            "source_table": "serving.theme_base_daily",
            "window_days": effective_window_days,
            "filters": {
                "brand_include": request.brand_include,
                "title_keywords": request.title_keywords,
                "material_keywords": request.material_keywords,
            },
            "sort_by": sort_by,
            "top_n": request.top_n,
            "total_candidate_count": len(candidate_asins),
            "scanned_asin_count": len(rows),
            "slice_count": len(matched_rows),
            "items": [compact_item(row) for row in top_rows],
            "rating_distribution": {
                "avg": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "median": round(float(median(ratings)), 2) if ratings else None,
                "p25": round(float(percentile(ratings, 0.25)), 2) if ratings else None,
                "p75": round(float(percentile(ratings, 0.75)), 2) if ratings else None,
                "buckets": rating_buckets,
            },
            "review_count_distribution": {
                "median": int(median(review_counts)) if review_counts else None,
                "p25": int(percentile(review_counts, 0.25)) if review_counts else None,
                "p75": int(percentile(review_counts, 0.75)) if review_counts else None,
                "buckets": review_buckets,
            },
            "top_brands": [
                {"name": name, "count": count}
                for name, count in sorted(brands.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
            "data_max_date": rows[0].get("max_date") if rows else None,
            "limitations": [
                "slice filters use local catalog title/category/brand fields; they do not include review text semantics",
                "material_keywords are matched against product title/category text until a normalized material taxonomy is available",
            ],
        }

    def get_asin_review_insights(self, request: AsinReviewInsightsRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
        asins = candidate_asins[: request.max_asins]
        provider_url = (os.getenv("ASIN_REVIEW_INSIGHTS_PROVIDER_URL") or "").strip()
        if not provider_url:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "candidate_pool": candidate_pool_ref,
                "asins": asins,
                "provider_status": "provider_required",
                "provider_configured": False,
                "supported_now": False,
                "required_provider": "asin_review_insights",
                "missing_capability": "review_text_provider",
                "message": "当前本地数据只包含评分和评论数量，不包含评论正文；无法生成真实评论关键词、痛点聚类或低分原因。",
                "available_alternatives": ["candidate_pool_slice.rating_distribution", "candidate_pool_slice.review_count_distribution", "top_asin_drilldown.rating_and_review_counts"],
            }
        try:
            response = http_requests.post(
                provider_url,
                json={"marketplace": marketplace, "domain": domain, "asins": asins},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "candidate_pool": candidate_pool_ref,
                "asins": asins,
                "provider_status": "provider_error",
                "provider_configured": True,
                "supported_now": False,
                "error": str(exc)[:500],
            }
        return {
            "marketplace": marketplace,
            "domain": domain,
            "candidate_pool": candidate_pool_ref,
            "asins": asins,
            "provider_status": "ready",
            "provider_configured": True,
            "supported_now": True,
            "insights": payload,
        }

    def get_amazon_keyword_demand(self, request: AmazonKeywordDemandRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        keywords = [keyword.strip() for keyword in request.keywords if keyword.strip()]
        if not keywords and request.product_query:
            keywords = [request.product_query]
        provider_url = (os.getenv("AMAZON_KEYWORD_DEMAND_PROVIDER_URL") or "").strip()
        if not provider_url:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "keywords": keywords,
                "provider_status": "provider_required",
                "provider_configured": False,
                "supported_now": False,
                "required_provider": "amazon_keyword_demand",
                "missing_capability": "amazon_keyword_volume_provider",
                "message": "当前未配置 Amazon 关键词量 provider，不能输出真实月搜索量；可用 Google Trends 指数或 ASIN 销量/评论分布作为替代验证。",
                "available_alternatives": ["candidate_pool_trends.google_trends_index", "candidate_pool_slice.sales_window_sum", "top_asin_drilldown.sales_and_review_counts"],
            }
        try:
            response = http_requests.post(
                provider_url,
                json={"marketplace": marketplace, "domain": domain, "keywords": keywords, "product_query": request.product_query},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {
                "marketplace": marketplace,
                "domain": domain,
                "keywords": keywords,
                "provider_status": "provider_error",
                "provider_configured": True,
                "supported_now": False,
                "error": str(exc)[:500],
            }
        return {
            "marketplace": marketplace,
            "domain": domain,
            "keywords": keywords,
            "provider_status": "ready",
            "provider_configured": True,
            "supported_now": True,
            "demand": payload,
        }

    def get_candidate_pool_trends(self, request: CandidatePoolRequest) -> dict[str, Any]:
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
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
            "candidate_pool": candidate_pool_ref,
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
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
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
            "candidate_pool": candidate_pool_ref,
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
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
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
            "candidate_pool": candidate_pool_ref,
            "window_days": effective_window_days,
            "sales_forecast_meta": sales_forecast_meta,
            "items": enriched_rows,
        }

    def get_product_forecast_explain(self, request: ProductForecastExplainRequest) -> dict[str, Any]:
        drilldown = self.get_top_asin_drilldown(request)
        items = drilldown.get("items") if isinstance(drilldown.get("items"), list) else []
        top_n = request.top_n or 10
        status_counts: dict[str, int] = {}
        explanations: list[dict[str, Any]] = []
        model_hit_count = 0
        summary_lines: list[str] = []

        for item in items[:top_n]:
            if not isinstance(item, dict):
                continue
            forecast = item.get("sales_forecast") if isinstance(item.get("sales_forecast"), dict) else {}
            status = str(forecast.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            has_model = forecast.get("predicted_weekly_sales_w1") is not None or forecast.get("predicted_weekly_sales_w4") is not None
            if has_model:
                model_hit_count += 1
            driver_summary = str(forecast.get("driver_summary_text") or "").strip()
            if driver_summary:
                summary_lines.append(driver_summary)

            explanations.append(
                {
                    "asin": item.get("asin"),
                    "title": item.get("product_title"),
                    "brand": item.get("brand"),
                    "category": item.get("category"),
                    "forecast_status": status,
                    "predicted_weekly_sales_w1": forecast.get("predicted_weekly_sales_w1"),
                    "predicted_weekly_sales_w4": forecast.get("predicted_weekly_sales_w4"),
                    "predicted_growth_delta_w4_minus_w1": forecast.get("predicted_growth_delta_w4_minus_w1"),
                    "predicted_growth_ratio_w4_over_w1": forecast.get("predicted_growth_ratio_w4_over_w1"),
                    "predicted_rank_w1_within_domain": forecast.get("predicted_rank_w1_within_domain"),
                    "predicted_rank_w4_within_domain": forecast.get("predicted_rank_w4_within_domain"),
                    "model_config_name_w1": forecast.get("model_config_name_w1"),
                    "model_config_name_w4": forecast.get("model_config_name_w4"),
                    "primary_driver_feature": forecast.get("primary_driver_feature"),
                    "primary_driver_label": forecast.get("primary_driver_label"),
                    "primary_driver_direction": forecast.get("primary_driver_direction"),
                    "primary_driver_contribution_share": forecast.get("primary_driver_contribution_share"),
                    "top_feature_contributions": forecast.get("top_feature_contributions") or [],
                    "driver_summary_text": driver_summary or None,
                    "forecast_notes": forecast.get("notes") or [],
                }
            )

        forecast_meta = drilldown.get("sales_forecast_meta") if isinstance(drilldown.get("sales_forecast_meta"), dict) else {}
        return {
            "source_tool": "product_forecast_explain",
            "source_detail_tool": "top_asin_drilldown",
            "marketplace": drilldown.get("marketplace"),
            "domain": drilldown.get("domain"),
            "candidate_pool": drilldown.get("candidate_pool"),
            "window_days": drilldown.get("window_days"),
            "sales_forecast_meta": forecast_meta,
            "forecast_status": status_counts,
            "forecast_model_hit_count": model_hit_count,
            "forecast_model_summary_text": "Top ASIN 中有 %s 个包含训练模型预测字段。" % model_hit_count,
            "asin_forecast_explanations": explanations,
            "driver_summary_text": "\n".join(summary_lines),
            "notes": forecast_meta.get("notes") if isinstance(forecast_meta.get("notes"), list) else [],
        }

    def get_launch_budget_calculation(self, request: LaunchBudgetCalculatorRequest) -> dict[str, Any]:
        return calculate_launch_budget(request)

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

    def _benchmark_target_depth(self, benchmark_level: str) -> int | None:
        return {
            "root": 1,
            "l1": 1,
            "l2": 2,
            "l3": 3,
        }.get((benchmark_level or "auto").strip().lower())

    def _fetch_category_benchmark_anchor(
        self,
        conn,
        *,
        domain: int,
        benchmark_category_id: int | None,
        benchmark_category_path: str | None,
        benchmark_level: str,
    ) -> dict[str, Any] | None:
        if benchmark_category_id is None and not benchmark_category_path:
            return None

        target_depth = self._benchmark_target_depth(benchmark_level)
        rows = _run_pg_dict_query(
            conn,
            """
            WITH RECURSIVE category_tree AS (
                SELECT
                    c.category_id,
                    c.domain,
                    c.category_en,
                    c.category_cn,
                    c.parent_id,
                    c.depth,
                    c.product_count,
                    COALESCE(c.category_en, '')::text AS category_path
                FROM sync.keepa_category_registry c
                WHERE c.domain = %s
                  AND (c.parent_id IS NULL OR c.depth = 1)

                UNION ALL

                SELECT
                    child.category_id,
                    child.domain,
                    child.category_en,
                    child.category_cn,
                    child.parent_id,
                    child.depth,
                    child.product_count,
                    CONCAT_WS(' > ', NULLIF(parent.category_path, ''), child.category_en)::text AS category_path
                FROM sync.keepa_category_registry child
                JOIN category_tree parent
                  ON child.domain = parent.domain
                 AND child.parent_id = parent.category_id
                WHERE child.category_id <> parent.category_id
            ),
            selected AS (
                SELECT
                    t.*,
                    CASE
                        WHEN %s::BIGINT IS NOT NULL AND t.category_id = %s::BIGINT THEN 0
                        WHEN %s::TEXT IS NOT NULL AND %s::TEXT <> '' AND LOWER(t.category_path) = LOWER(%s::TEXT) THEN 1
                        WHEN %s::TEXT IS NOT NULL AND %s::TEXT <> '' AND LOWER(COALESCE(t.category_en, '')) = LOWER(%s::TEXT) THEN 2
                        WHEN %s::TEXT IS NOT NULL AND %s::TEXT <> '' AND LOWER(COALESCE(t.category_cn, '')) = LOWER(%s::TEXT) THEN 3
                        ELSE 4
                    END AS match_rank
                FROM category_tree t
                WHERE (%s::BIGINT IS NOT NULL AND t.category_id = %s::BIGINT)
                   OR (
                       %s::TEXT IS NOT NULL
                       AND %s::TEXT <> ''
                       AND (
                           LOWER(t.category_path) = LOWER(%s::TEXT)
                           OR LOWER(COALESCE(t.category_en, '')) = LOWER(%s::TEXT)
                           OR LOWER(COALESCE(t.category_cn, '')) = LOWER(%s::TEXT)
                           OR LOWER(t.category_path) LIKE ('%%' || LOWER(%s::TEXT) || '%%')
                       )
                   )
                ORDER BY match_rank ASC, t.depth DESC, t.product_count DESC NULLS LAST
                LIMIT 1
            ),
            ancestors AS (
                SELECT
                    s.category_id,
                    s.category_en,
                    s.category_cn,
                    s.parent_id,
                    s.depth,
                    s.product_count,
                    s.category_path
                FROM selected s

                UNION ALL

                SELECT
                    parent.category_id,
                    parent.category_en,
                    parent.category_cn,
                    parent.parent_id,
                    parent.depth,
                    parent.product_count,
                    parent.category_path
                FROM ancestors child
                JOIN category_tree parent
                  ON child.parent_id = parent.category_id
            )
            SELECT
                a.category_id,
                a.category_en,
                a.category_cn,
                a.parent_id,
                a.depth,
                a.product_count,
                a.category_path,
                (a.category_id = (SELECT category_id FROM selected)) AS is_selected_category
            FROM ancestors a
            ORDER BY
                CASE
                    WHEN %s::INTEGER IS NOT NULL AND a.depth = %s::INTEGER THEN 0
                    WHEN a.category_id = (SELECT category_id FROM selected) THEN 1
                    ELSE 2
                END ASC,
                a.depth DESC
            LIMIT 1
            """,
            [
                domain,
                benchmark_category_id,
                benchmark_category_id,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_id,
                benchmark_category_id,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                benchmark_category_path,
                target_depth,
                target_depth,
            ],
        )
        return rows[0] if rows else None

    def _count_candidate_asins_in_category(
        self,
        conn,
        *,
        domain: int,
        candidate_asins: list[str],
        category_id: int,
        include_descendants: bool,
    ) -> int:
        rows = _run_pg_dict_query(
            conn,
            """
            WITH RECURSIVE subtree(category_id) AS (
                SELECT %s::BIGINT AS category_id

                UNION

                SELECT child.category_id
                FROM sync.keepa_category_registry child
                JOIN subtree parent
                  ON child.parent_id = parent.category_id
                WHERE child.domain = %s
                  AND %s IS TRUE
                  AND child.category_id <> parent.category_id
            )
            SELECT COUNT(DISTINCT r.asin) AS candidate_asin_count
            FROM sync.keepa_asin_registry r
            WHERE r.domain = %s
              AND r.asin = ANY(%s)
              AND r.category_id IN (SELECT category_id FROM subtree)
            """,
            [category_id, domain, include_descendants, domain, candidate_asins],
        )
        return int(rows[0].get("candidate_asin_count") or 0) if rows else 0

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

    def get_category_benchmark(self, request: CategoryBenchmarkRequest) -> dict[str, Any]:
        """Return L3-level category benchmark stats for comparison with candidate pool.

        Strategy:
        1. Use explicit benchmark_category_id/path when provided.
        2. Otherwise get each candidate ASIN's leaf category_id from keepa_asin_registry.
        3. Walk up parent_id chain in keepa_category_registry to find L3 ancestor (depth=3).
        4. Pick the dominant L3 category (mode) as the benchmark anchor.
        5. Aggregate ASINs whose category_id descends from the anchor.
        """
        domain, marketplace = _normalize_marketplace(request.marketplace)
        candidate_asins, candidate_pool_ref = self._resolve_candidate_asins_for_pool_request(
            request,
            domain=domain,
            marketplace=marketplace,
        )
        effective_window_days = min(request.window_days, 90)
        explicit_anchor_requested = bool(request.benchmark_category_id is not None or request.benchmark_category_path)
        anchor_source = "auto_candidate_l3_mode"
        anchor_confidence = 0.0
        fallback_reason: str | None = None
        all_l3_cats: list[dict[str, Any]] = []

        with _postgres_conn() as conn:
            explicit_anchor = self._fetch_category_benchmark_anchor(
                conn,
                domain=domain,
                benchmark_category_id=request.benchmark_category_id,
                benchmark_category_path=request.benchmark_category_path,
                benchmark_level=request.benchmark_level,
            )
            if explicit_anchor:
                dominant_l3_id = int(explicit_anchor["category_id"])
                dominant_l3_name = str(
                    explicit_anchor.get("category_cn") or explicit_anchor.get("category_en") or "Unknown"
                )
                dominant_l3_en = str(explicit_anchor.get("category_en") or "")
                dominant_depth = int(explicit_anchor.get("depth") or 0)
                dominant_count = self._count_candidate_asins_in_category(
                    conn,
                    domain=domain,
                    candidate_asins=candidate_asins,
                    category_id=dominant_l3_id,
                    include_descendants=request.include_descendants,
                )
                anchor_source = "explicit_category_id" if request.benchmark_category_id is not None else "explicit_category_path"
                anchor_confidence = 0.98 if request.benchmark_category_id is not None else 0.9
                target_depth = self._benchmark_target_depth(request.benchmark_level)
                if target_depth is not None and dominant_depth != target_depth:
                    fallback_reason = f"requested benchmark_level={request.benchmark_level} but resolved anchor depth is L{dominant_depth}"
            else:
                if explicit_anchor_requested:
                    return {
                        "marketplace": marketplace,
                        "domain": domain,
                        "candidate_pool": candidate_pool_ref,
                        "window_days": request.window_days,
                        "benchmark_category": None,
                        "benchmark_category_level": None,
                        "benchmark_anchor": None,
                        "anchor_source": "explicit_anchor_not_found",
                        "anchor_confidence": 0.0,
                        "fallback_reason": "benchmark_category_id/path did not match local keepa_category_registry",
                        "benchmark_is_precise": False,
                        "local_category_coverage": {
                            "candidate_asin_count_in_category": 0,
                            "category_total_asin_count": 0,
                            "too_small": True,
                            "min_pool_size": DEFAULT_MIN_CANDIDATE_POOL_SIZE,
                        },
                        "candidate_asin_count_in_category": 0,
                        "category_total_asin_count": 0,
                        "candidate_category_coverage_pct": 0,
                        "all_candidate_l3_categories": [],
                        "benchmark_stats": {},
                        "notes": ["显式 benchmark 类目锚点未在本地类目表中命中"],
                    }

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
                        "benchmark_anchor": None,
                        "anchor_source": anchor_source,
                        "anchor_confidence": 0.0,
                        "fallback_reason": "candidate_asins do not have local category_id coverage",
                        "benchmark_is_precise": False,
                        "local_category_coverage": {
                            "candidate_asin_count_in_category": 0,
                            "category_total_asin_count": 0,
                            "too_small": True,
                            "min_pool_size": DEFAULT_MIN_CANDIDATE_POOL_SIZE,
                        },
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
                anchor_confidence = round(dominant_count / max(len(candidate_asins), 1), 4)

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
                                            AND %s IS TRUE
                                            AND c.category_id <> s.category_id
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
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY monthly_sold) AS median_monthly_sold,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY offer_count) AS median_offer_count
                FROM latest_history
                """,
                [dominant_l3_id, domain, domain, request.include_descendants, domain, domain, effective_window_days],
            )

        def _safe_round(val: Any, decimals: int = 2) -> float | None:
            return round(float(val), decimals) if val is not None else None

        def _safe_int(val: Any) -> int | None:
            return int(val) if val is not None else None

        stats = bench_rows[0] if bench_rows else {}
        cat_total = int(stats.get("category_total_asin_count") or 0)
        local_category_coverage = {
            "candidate_asin_count_in_category": dominant_count,
            "category_total_asin_count": cat_total,
            "candidate_category_coverage_pct": round(dominant_count / cat_total * 100, 2) if cat_total > 0 else 0,
            "too_small": cat_total < DEFAULT_MIN_CANDIDATE_POOL_SIZE,
            "min_pool_size": DEFAULT_MIN_CANDIDATE_POOL_SIZE,
            "include_descendants": request.include_descendants,
        }
        benchmark_is_precise = dominant_depth >= 3 and not local_category_coverage["too_small"]

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
            "median_monthly_sold": _safe_round(stats.get("median_monthly_sold")),
            "median_offer_count": _safe_round(stats.get("median_offer_count")),
        }

        return {
            "marketplace": marketplace,
            "domain": domain,
            "candidate_pool": candidate_pool_ref,
            "window_days": effective_window_days,
            "benchmark_category": {
                "category_id": dominant_l3_id,
                "category_name": dominant_l3_name,
                "category_en": dominant_l3_en,
                "depth": dominant_depth,
                "level": f"L{dominant_depth}",
            },
            "benchmark_category_level": f"L{dominant_depth}",
            "benchmark_anchor": {
                "category_id": dominant_l3_id,
                "category_name": dominant_l3_name,
                "category_en": dominant_l3_en,
                "depth": dominant_depth,
                "level": f"L{dominant_depth}",
                "requested_level": request.benchmark_level,
                "include_descendants": request.include_descendants,
            },
            "anchor_source": anchor_source,
            "anchor_confidence": anchor_confidence,
            "fallback_reason": fallback_reason,
            "benchmark_is_precise": benchmark_is_precise,
            "local_category_coverage": local_category_coverage,
            "candidate_asin_count_in_category": dominant_count,
            "category_total_asin_count": cat_total,
            "candidate_category_coverage_pct": local_category_coverage["candidate_category_coverage_pct"],
            "all_candidate_l3_categories": all_l3_cats,
            "benchmark_stats": benchmark_stats,
            "notes": [
                (
                    f"对标类目由显式 {anchor_source} 选取"
                    if explicit_anchor_requested
                    else f"对标类目由候选池 ASIN 的众数 L{dominant_depth} 类目自动选取"
                ),
                f"候选池中 {dominant_count}/{len(candidate_asins)} 个 ASIN 属于此类目",
                "聚合范围包含锚点类目及其所有子类目下的全部 ASIN" if request.include_descendants else "聚合范围仅包含锚点类目本身",
                *( ["本地类目覆盖不足，benchmark 不应作为强品类结论"] if local_category_coverage["too_small"] else [] ),
                *( [fallback_reason] if fallback_reason else [] ),
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


service = ProductThemeService()
app = create_product_theme_app(
    service=service,
    get_query_normalizer_config=_get_query_normalizer_config,
    get_theme_feature_serving_status=_get_theme_feature_serving_status,
    root_env_file=ROOT_ENV_FILE,
)
