"""Constants for product theme API modules."""
from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
DEFAULT_MIN_CANDIDATE_POOL_SIZE = 8
DEFAULT_TARGET_CANDIDATE_POOL_SIZE = 20
DEFAULT_DOMINANT_CATEGORY_SHARE_THRESHOLD = 0.7
# Minimum share of the candidate pool that the benchmark anchor must represent
# before category_benchmark may claim benchmark_is_precise. A "dominant" L3 that
# only covers a sliver of the pool (e.g. 1/17) is a representativeness artifact,
# not a trustworthy benchmark, and must be flagged as imprecise.
BENCHMARK_ANCHOR_MIN_REPRESENTATIVENESS = 0.30
CANDIDATE_EXPANSION_ACTIVE_STATUSES = ("queued", "waiting_token", "discovering", "hydrating", "syncing")
OPPORTUNITY_SCORE_WEIGHTS = {
    "demand_score": 0.20,
    "trend_score": 0.20,
    "competition_headroom_score": 0.15,
    "price_fit_score": 0.15,
    "forecast_growth_score": 0.15,
    "coverage_gap_score": 0.10,
    "evidence_quality_score": 0.05,
}

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
