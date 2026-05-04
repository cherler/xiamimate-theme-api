"""Request schemas for product theme API routes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, root_validator, validator

from data_platform.api.product_theme.constants import (
    DEFAULT_MIN_CANDIDATE_POOL_SIZE,
    DEFAULT_TARGET_CANDIDATE_POOL_SIZE,
)


class ResolveCandidatesRequest(BaseModel):
    product_query: str = Field(..., min_length=1)
    marketplace: str | int = "US"
    query_aliases: list[str] = Field(default_factory=list)
    category_hints: list[str] = Field(default_factory=list)
    recall_mode: str = "keyword"
    category_id: int | None = None
    category_path: str | None = None
    include_descendants: bool = True
    min_pool_size: int = Field(default=DEFAULT_MIN_CANDIDATE_POOL_SIZE, ge=1, le=500)
    target_pool_size: int = Field(default=DEFAULT_TARGET_CANDIDATE_POOL_SIZE, ge=1, le=500)
    expand_if_small: bool = False
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

    @validator("recall_mode")
    def _validate_recall_mode(cls, v: str) -> str:  # noqa: N805
        value = (v or "keyword").strip().lower()
        allowed = {"keyword", "category", "hybrid", "asin_seed_expand"}
        if value not in allowed:
            raise ValueError("recall_mode must be one of: keyword, category, hybrid, asin_seed_expand")
        return value

    @validator("category_path", pre=True, always=True)
    def _normalize_optional_category_path(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CategoryResolveRequest(BaseModel):
    marketplace: str | int = "US"
    category_query: str | None = None
    category_path: str | None = None
    max_matches: int = Field(default=10, ge=1, le=50)


class CandidatePoolRequest(BaseModel):
    candidate_asins: list[str] = Field(default_factory=list)
    candidate_pool_id: str | None = None
    marketplace: str | int = "US"
    window_days: int = Field(default=30, ge=7, le=180)

    @validator("candidate_asins", pre=True, always=True)
    def _accept_csv_asins(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v

    @validator("candidate_pool_id", pre=True, always=True)
    def _normalize_optional_pool_id(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @root_validator(skip_on_failure=True)
    def _require_pool_reference(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        candidate_asins = values.get("candidate_asins") or []
        candidate_pool_id = values.get("candidate_pool_id")
        if not candidate_asins and not candidate_pool_id:
            raise ValueError("candidate_asins or candidate_pool_id is required")
        return values


class CategoryBenchmarkRequest(CandidatePoolRequest):
    benchmark_category_id: int | None = None
    benchmark_category_path: str | None = None
    benchmark_level: str = "auto"
    include_descendants: bool = True

    @validator("benchmark_level")
    def _validate_benchmark_level(cls, v: str) -> str:  # noqa: N805
        value = (v or "auto").strip().lower()
        allowed = {"auto", "leaf", "fine", "l3", "l2", "l1", "root"}
        if value not in allowed:
            raise ValueError("benchmark_level must be one of: auto, leaf, fine, l3, l2, l1, root")
        return value

    @validator("benchmark_category_path", pre=True, always=True)
    def _normalize_optional_benchmark_category_path(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CandidateExpansionJobRequest(BaseModel):
    product_query: str | None = None
    marketplace: str | int = "US"
    recall_mode: str = "hybrid"
    category_id: int | None = None
    category_path: str | None = None
    include_descendants: bool = True
    target_asin_count: int = Field(default=20, ge=1, le=500)
    min_pool_size: int = Field(default=DEFAULT_MIN_CANDIDATE_POOL_SIZE, ge=1, le=500)
    source: str = "agent_interactive"
    priority: str = "interactive_normal"
    requested_by_session_id: str | None = None
    requested_by_user_id: str | None = None
    idempotency_key: str | None = None
    notes: str | None = None

    @validator("recall_mode")
    def _validate_expansion_recall_mode(cls, v: str) -> str:  # noqa: N805
        value = (v or "hybrid").strip().lower()
        allowed = {"keyword", "category", "hybrid", "asin_seed_expand"}
        if value not in allowed:
            raise ValueError("recall_mode must be one of: keyword, category, hybrid, asin_seed_expand")
        return value

    @validator("source")
    def _validate_expansion_source(cls, v: str) -> str:  # noqa: N805
        value = (v or "agent_interactive").strip().lower()
        allowed = {"agent_interactive", "auto_collect", "manual", "scheduled"}
        if value not in allowed:
            raise ValueError("source must be one of: agent_interactive, auto_collect, manual, scheduled")
        return value

    @validator("priority")
    def _validate_expansion_priority(cls, v: str) -> str:  # noqa: N805
        value = (v or "interactive_normal").strip().lower()
        allowed = {"interactive_high", "interactive_normal", "background_high", "background_low"}
        if value not in allowed:
            raise ValueError("priority must be one of: interactive_high, interactive_normal, background_high, background_low")
        return value

    @validator("product_query", "category_path", "requested_by_session_id", "requested_by_user_id", "idempotency_key", "notes", pre=True, always=True)
    def _normalize_optional_string(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CandidateExpansionJobStatusRequest(BaseModel):
    job_id: str | None = None
    marketplace: str | int = "US"
    statuses: list[str] = Field(default_factory=lambda: ["queued", "waiting_token", "discovering", "hydrating", "syncing"])
    limit: int = Field(default=20, ge=1, le=100)

    @validator("statuses", pre=True, always=True)
    def _accept_csv_statuses(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip().lower() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return [str(s).strip().lower() for s in v if str(s).strip()]

    @validator("job_id", pre=True, always=True)
    def _normalize_optional_job_id(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class OpportunityDiscoveryRequest(BaseModel):
    marketplace: str | int = "US"
    platform: str = "Amazon"
    query: str | None = None
    category_id: int | None = None
    category_path: str | None = None
    limit: int = Field(default=10, ge=1, le=30)
    window_days: int = Field(default=30, ge=7, le=180)
    min_data_confidence: str = "low"
    include_expandable: bool = True
    include_descendants: bool = True
    memory_profile: dict[str, Any] | None = None

    @validator("query", "category_path", pre=True, always=True)
    def _normalize_optional_string(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @validator("platform")
    def _validate_platform(cls, v: str) -> str:  # noqa: N805
        value = (v or "Amazon").strip()
        if value.lower() != "amazon":
            raise ValueError("opportunity discovery MVP currently supports Amazon only")
        return "Amazon"

    @validator("min_data_confidence")
    def _validate_min_data_confidence(cls, v: str) -> str:  # noqa: N805
        value = (v or "low").strip().lower()
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            raise ValueError("min_data_confidence must be one of: low, medium, high")
        return value


class OpportunityDiscoveryJobStatusRequest(BaseModel):
    job_id: str | None = None
    marketplace: str | int = "US"
    include_result: bool = True
    limit: int = Field(default=20, ge=1, le=100)

    @validator("job_id", pre=True, always=True)
    def _normalize_optional_job_id(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class WeakForecastRequest(CandidatePoolRequest):
    top_n: int = Field(default=5, ge=1, le=20)


class DrilldownRequest(CandidatePoolRequest):
    top_n: int | None = Field(default=None, ge=1, le=20)


class ProductForecastExplainRequest(DrilldownRequest):
    top_n: int | None = Field(default=10, ge=1, le=20)


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


class LaunchBudgetCalculatorRequest(BaseModel):
    marketplace: str | int = "US"
    product_theme: str | None = None
    selling_price: float | None = Field(default=None, gt=0)
    unit_product_cost: float | None = Field(default=None, ge=0)
    landed_cost_per_unit: float | None = Field(default=None, ge=0)
    packaging_cost: float | None = Field(default=None, ge=0)
    inbound_shipping_per_unit: float | None = Field(default=None, ge=0)
    duty_per_unit: float | None = Field(default=None, ge=0)
    fba_fee: float | None = Field(default=None, ge=0)
    referral_fee_rate: float | None = Field(default=None, ge=0, le=0.5)
    coupon_discount_rate: float | None = Field(default=None, ge=0, le=0.8)
    return_rate: float | None = Field(default=None, ge=0, le=0.8)
    fixed_startup_cost: float | None = Field(default=None, ge=0)
    monthly_fixed_cost: float | None = Field(default=None, ge=0)
    monthly_ad_budget: float | None = Field(default=None, ge=0)
    launch_units: int | None = Field(default=None, ge=1, le=100000)
    launch_months: int | None = Field(default=None, ge=1, le=24)

    @validator("product_theme", pre=True, always=True)
    def _normalize_optional_string(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None
