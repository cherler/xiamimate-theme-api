"""Query and scalar normalization helpers for product theme APIs."""
from __future__ import annotations

from typing import Any

import jieba
from fastapi import HTTPException

from data_platform.api.product_theme.constants import (
    ASCII_TOKEN_PATTERN,
    DOMAIN_TO_MARKETPLACE,
    MARKETPLACE_TO_DOMAIN,
    MIN_TOKEN_LENGTH,
)


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
    return " ".join(value.strip().lower().split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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
