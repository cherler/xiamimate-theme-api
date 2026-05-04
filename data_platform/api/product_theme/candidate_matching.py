"""Candidate query matching and ASIN normalization helpers."""
from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException

from data_platform.api.product_theme.category_utils import _category_path_parts
from data_platform.api.product_theme.constants import (
    ASCII_TOKEN_PATTERN,
    ASIN_PATTERN,
    MIN_TOKEN_LENGTH,
    QUERY_ALIAS_EXPANSIONS,
    QUERY_MODIFIER_TOKENS,
    TOKEN_ALIAS_EXPANSIONS,
)
from data_platform.api.product_theme.query_utils import _normalize_text, _tokenize_phrase


def _normalize_category_path_for_match(value: Any) -> str:
    parts = _category_path_parts(value)
    if parts:
        return " > ".join(_normalize_text(part) for part in parts if _normalize_text(part))
    return _normalize_text(str(value or ""))


def _score_category_match(
    *,
    category_query: str | None,
    category_path: str | None,
    candidate_name: str | None,
    candidate_path: str | None,
) -> float:
    query_norm = _normalize_text(str(category_query or ""))
    requested_path_norm = _normalize_category_path_for_match(category_path)
    candidate_name_norm = _normalize_text(str(candidate_name or ""))
    candidate_path_norm = _normalize_category_path_for_match(candidate_path)

    score = 0.0
    if requested_path_norm:
        if candidate_path_norm == requested_path_norm:
            score = max(score, 0.98)
        elif candidate_path_norm.endswith(requested_path_norm) or requested_path_norm.endswith(candidate_path_norm):
            score = max(score, 0.88)
        elif requested_path_norm in candidate_path_norm:
            score = max(score, 0.8)
    if query_norm:
        if candidate_name_norm == query_norm:
            score = max(score, 0.92)
        elif query_norm in candidate_name_norm:
            score = max(score, 0.84)
        elif query_norm in candidate_path_norm:
            score = max(score, 0.74)
    return round(score, 4)


def _text_contains_token(text: str, token: str) -> bool:
    if not text or not token:
        return False
    if ASCII_TOKEN_PATTERN.fullmatch(token):
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


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
