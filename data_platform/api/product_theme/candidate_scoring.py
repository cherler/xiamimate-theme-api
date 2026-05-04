"""Candidate and opportunity scoring helpers."""
from __future__ import annotations

from typing import Any

from data_platform.api.product_theme.query_utils import _safe_float


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 2)


def _price_fit_score(price: Any) -> float:
    value = _safe_float(price, 0.0)
    if value <= 0:
        return 50.0
    if 18 <= value <= 80:
        return 100.0
    if 10 <= value < 18 or 80 < value <= 140:
        return 75.0
    if 6 <= value < 10 or 140 < value <= 220:
        return 55.0
    return 35.0


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(value or "low").lower(), 1)
