"""Request schemas for product theme API routes."""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

from data_platform.api.product_theme.constants import (
    DEFAULT_MIN_CANDIDATE_POOL_SIZE,
    DEFAULT_TARGET_CANDIDATE_POOL_SIZE,
)


class ResolveCandidatesRequest(BaseModel):
    product_query: str = Field(..., min_length=1)
    marketplace: Union[str, int] = "US"
    query_aliases: list[str] = Field(default_factory=list)
    category_hints: list[str] = Field(default_factory=list)
    recall_mode: str = "keyword"
    category_id: Optional[int] = None
    category_path: Optional[str] = None
    include_descendants: bool = True
    min_pool_size: int = Field(default=DEFAULT_MIN_CANDIDATE_POOL_SIZE, ge=1, le=500)
    target_pool_size: int = Field(default=DEFAULT_TARGET_CANDIDATE_POOL_SIZE, ge=1, le=500)
    expand_if_small: bool = False
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    max_candidates: int = Field(default=50, ge=1, le=500)
    active_only: bool = True
    response_profile: str = "standard"
    include_debug: bool = False
    include_response_contract: bool = True

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

    @validator("response_profile")
    def _validate_response_profile(cls, v: str) -> str:  # noqa: N805
        value = (v or "standard").strip().lower()
        allowed = {"standard", "compact", "debug"}
        if value not in allowed:
            raise ValueError("response_profile must be one of: standard, compact, debug")
        return value

    @validator("category_path", pre=True, always=True)
    def _normalize_optional_category_path(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CategoryResolveRequest(BaseModel):
    marketplace: Union[str, int] = "US"
    category_query: Optional[str] = None
    category_path: Optional[str] = None
    max_matches: int = Field(default=10, ge=1, le=50)


class CandidatePoolRequest(BaseModel):
    candidate_asins: list[str] = Field(default_factory=list)
    candidate_pool_id: Optional[str] = None
    marketplace: Union[str, int] = "US"
    window_days: int = Field(default=30, ge=7, le=180)

    @validator("candidate_asins", pre=True, always=True)
    def _accept_csv_asins(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v

    @validator("candidate_pool_id", pre=True, always=True)
    def _normalize_optional_pool_id(cls, v: Any) -> Optional[str]:  # noqa: N805
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


class CandidatePoolSliceRequest(CandidatePoolRequest):
    brand_include: list[str] = Field(default_factory=list)
    title_keywords: list[str] = Field(default_factory=list)
    material_keywords: list[str] = Field(default_factory=list)
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    sort_by: str = "sales_window_sum"
    top_n: int = Field(default=3, ge=1, le=20)

    @validator("brand_include", "title_keywords", "material_keywords", pre=True, always=True)
    def _accept_csv_terms(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return [str(s).strip() for s in v if str(s).strip()]

    @validator("price_min", "price_max", pre=True)
    def _normalize_optional_price(cls, v: Any) -> Optional[float]:  # noqa: N805
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        value = float(v)
        if value < 0:
            raise ValueError("price_min and price_max must be non-negative")
        return value

    @root_validator(skip_on_failure=True)
    def _validate_price_range(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        price_min = values.get("price_min")
        price_max = values.get("price_max")
        if price_min is not None and price_max is not None and price_min > price_max:
            raise ValueError("price_min must be less than or equal to price_max")
        return values

    @validator("sort_by")
    def _validate_sort_by(cls, v: str) -> str:  # noqa: N805
        value = (v or "sales_window_sum").strip().lower()
        allowed = {"sales_window_sum", "sales_daily_avg", "review_count", "rating", "bsr", "price"}
        if value not in allowed:
            raise ValueError("sort_by must be one of: sales_window_sum, sales_daily_avg, review_count, rating, bsr, price")
        return value


class AsinReviewInsightsRequest(CandidatePoolRequest):
    max_asins: int = Field(default=10, ge=1, le=20)


class AmazonKeywordDemandRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    product_query: Optional[str] = None
    marketplace: Union[str, int] = "US"

    @validator("keywords", pre=True, always=True)
    def _accept_csv_keywords(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return [str(s).strip() for s in v if str(s).strip()]

    @validator("product_query", pre=True, always=True)
    def _normalize_optional_product_query(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @root_validator(skip_on_failure=True)
    def _require_keywords_or_query(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        if not values.get("keywords") and not values.get("product_query"):
            raise ValueError("keywords or product_query is required")
        return values


class CategoryBenchmarkRequest(CandidatePoolRequest):
    benchmark_category_id: Optional[int] = None
    benchmark_category_path: Optional[str] = None
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
    def _normalize_optional_benchmark_category_path(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CandidateExpansionJobRequest(BaseModel):
    product_query: Optional[str] = None
    marketplace: Union[str, int] = "US"
    recall_mode: str = "hybrid"
    category_id: Optional[int] = None
    category_path: Optional[str] = None
    include_descendants: bool = True
    target_asin_count: int = Field(default=20, ge=1, le=500)
    min_pool_size: int = Field(default=DEFAULT_MIN_CANDIDATE_POOL_SIZE, ge=1, le=500)
    source: str = "agent_interactive"
    priority: str = "interactive_normal"
    requested_by_session_id: Optional[str] = None
    requested_by_user_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    notes: Optional[str] = None

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
    def _normalize_optional_string(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class CandidateExpansionJobStatusRequest(BaseModel):
    job_id: Optional[str] = None
    marketplace: Union[str, int] = "US"
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
    def _normalize_optional_job_id(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class OpportunityDiscoveryRequest(BaseModel):
    marketplace: Union[str, int] = "US"
    platform: str = "Amazon"
    query: Optional[str] = None
    category_id: Optional[int] = None
    category_path: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=30)
    window_days: int = Field(default=30, ge=7, le=180)
    min_data_confidence: str = "low"
    include_expandable: bool = True
    include_descendants: bool = True
    memory_profile: Optional[dict[str, Any]] = None

    @validator("query", "category_path", pre=True, always=True)
    def _normalize_optional_string(cls, v: Any) -> Optional[str]:  # noqa: N805
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
    job_id: Optional[str] = None
    marketplace: Union[str, int] = "US"
    include_result: bool = True
    limit: int = Field(default=20, ge=1, le=100)

    @validator("job_id", pre=True, always=True)
    def _normalize_optional_job_id(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None


class WeakForecastRequest(CandidatePoolRequest):
    top_n: int = Field(default=5, ge=1, le=20)


class DrilldownRequest(CandidatePoolRequest):
    top_n: Optional[int] = Field(default=None, ge=1, le=20)


class ProductForecastExplainRequest(DrilldownRequest):
    top_n: Optional[int] = Field(default=10, ge=1, le=20)


class AsinHistoryTimeseriesRequest(BaseModel):
    asins: list[str] = Field(..., min_length=1)
    marketplace: Union[str, int] = "US"
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
    marketplace: Union[str, int] = "US"

    @validator("asins", pre=True, always=True)
    def _accept_csv_string(cls, v: Any) -> list[str]:  # noqa: N805
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if v is None:
            return []
        return v


class LaunchBudgetCalculatorRequest(BaseModel):
    marketplace: Union[str, int] = "US"
    product_theme: Optional[str] = None
    selling_price: Optional[float] = Field(default=None, gt=0)
    unit_product_cost: Optional[float] = Field(default=None, ge=0)
    landed_cost_per_unit: Optional[float] = Field(default=None, ge=0)
    packaging_cost: Optional[float] = Field(default=None, ge=0)
    inbound_shipping_per_unit: Optional[float] = Field(default=None, ge=0)
    duty_per_unit: Optional[float] = Field(default=None, ge=0)
    fba_fee: Optional[float] = Field(default=None, ge=0)
    referral_fee_rate: Optional[float] = Field(default=None, ge=0, le=0.5)
    coupon_discount_rate: Optional[float] = Field(default=None, ge=0, le=0.8)
    return_rate: Optional[float] = Field(default=None, ge=0, le=0.8)
    fixed_startup_cost: Optional[float] = Field(default=None, ge=0)
    monthly_fixed_cost: Optional[float] = Field(default=None, ge=0)
    monthly_ad_budget: Optional[float] = Field(default=None, ge=0)
    launch_units: Optional[int] = Field(default=None, ge=1, le=100000)
    launch_months: Optional[int] = Field(default=None, ge=1, le=24)

    @validator("product_theme", pre=True, always=True)
    def _normalize_optional_string(cls, v: Any) -> Optional[str]:  # noqa: N805
        if v is None:
            return None
        value = str(v).strip()
        return value or None
