"""FastAPI app assembly for Product Theme API."""
from __future__ import annotations

import os
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from data_platform.api.product_theme.constants import (
    API_KEY_HEADER_NAME,
    DEFAULT_FEATURE_DIR,
    PROTECTED_API_PREFIX,
)
from data_platform.api.product_theme.db import _postgres_conn, _run_pg_dict_query
from data_platform.api.product_theme.response_contract import _error_response, _response_meta, _success_response
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
from data_platform.api.theme_api_auth import (
    API_KEY_ENV_VAR,
    API_KEY_NAME_ENV_VAR,
    ensure_env_api_key_registered,
    get_active_key_count,
    record_api_usage,
    resolve_api_key,
)


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


def create_product_theme_app(
    *,
    service: Any,
    get_query_normalizer_config: Callable[[], dict[str, Any]],
    get_theme_feature_serving_status: Callable[..., dict[str, Any]],
    root_env_file: Any,
) -> FastAPI:
    app = FastAPI(title="xiamimate Product Theme API", version="2026-04-10")

    @app.on_event("startup")
    def warmup_connection_pools() -> None:
        """Pre-load PostgreSQL pool and serving metadata to reduce cold-start latency."""
        t0 = time.monotonic()
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
            get_theme_feature_serving_status(include_data_max_date=False)
        except Exception:
            pass
        elapsed = time.monotonic() - t0
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
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
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
        query_normalizer = get_query_normalizer_config()
        feature_serving = get_theme_feature_serving_status(include_data_max_date=False)
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
                    "env_file_autoload": str(root_env_file),
                },
            },
        )

    @app.post("/api/product-theme/resolve-candidates")
    async def resolve_candidates(request: ResolveCandidatesRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/resolve-candidates",
            message="candidate pool resolved",
            data=await service.resolve_candidates(request),
            include_response_contract=request.include_response_contract,
        )

    @app.post("/api/product-theme/category-resolve")
    def category_resolve(request: CategoryResolveRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/category-resolve",
            message="category matches resolved",
            data=service.resolve_category(request),
        )

    @app.post("/api/product-theme/expand-candidates")
    def expand_candidates(request: CandidateExpansionJobRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/expand-candidates",
            message="candidate expansion job queued",
            data=service.create_candidate_expansion_job(request),
        )

    @app.post("/api/product-theme/candidate-expansion-status")
    def candidate_expansion_status(request: CandidateExpansionJobStatusRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/candidate-expansion-status",
            message="candidate expansion job status ready",
            data=service.get_candidate_expansion_status(request),
        )

    @app.post("/api/product-theme/opportunity-discovery")
    async def opportunity_discovery(request: OpportunityDiscoveryRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/opportunity-discovery",
            message="opportunity discovery cards ready",
            data=await service.discover_opportunities(request),
        )

    @app.post("/api/product-theme/opportunity-discovery-job")
    def opportunity_discovery_job(request: OpportunityDiscoveryJobStatusRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/opportunity-discovery-job",
            message="opportunity discovery job result ready",
            data=service.get_opportunity_discovery_job_status(request),
        )

    @app.post("/api/product-theme/candidate-pool-stats")
    def candidate_pool_stats(request: CandidatePoolRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/candidate-pool-stats",
            message="candidate pool stats ready",
            data=service.get_candidate_pool_stats(request),
        )

    @app.post("/api/product-theme/candidate-pool-slice")
    def candidate_pool_slice(request: CandidatePoolSliceRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/candidate-pool-slice",
            message="candidate pool slice ready",
            data=service.get_candidate_pool_slice(request),
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

    @app.post("/api/product-theme/product-forecast-explain")
    def product_forecast_explain(request: ProductForecastExplainRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/product-forecast-explain",
            message="product forecast explanations ready",
            data=service.get_product_forecast_explain(request),
        )

    @app.post("/api/product-theme/launch-budget-calculator")
    def launch_budget_calculator(request: LaunchBudgetCalculatorRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/launch-budget-calculator",
            message="launch budget calculation ready",
            data=service.get_launch_budget_calculation(request),
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

    @app.post("/api/product-theme/asin-review-insights")
    def asin_review_insights(request: AsinReviewInsightsRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/asin-review-insights",
            message="asin review insights checked",
            data=service.get_asin_review_insights(request),
        )

    @app.post("/api/product-theme/amazon-keyword-demand")
    def amazon_keyword_demand(request: AmazonKeywordDemandRequest) -> dict[str, Any]:
        return _success_response(
            endpoint="/api/product-theme/amazon-keyword-demand",
            message="amazon keyword demand checked",
            data=service.get_amazon_keyword_demand(request),
        )

    @app.post("/api/product-theme/category-benchmark")
    def category_benchmark(request: CategoryBenchmarkRequest) -> dict[str, Any]:
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

    return app