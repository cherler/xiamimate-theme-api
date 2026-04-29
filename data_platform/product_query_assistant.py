from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import re

from data_platform.llm_client import LLMJSONParseError, LLMProvider, build_llm_provider


@dataclass
class ProductThemeExtractionResult:
    extracted_theme: str
    query_aliases: list[str]
    category_hints: list[str]
    extraction_mode: str
    llm_used: bool
    llm_provider: str
    llm_model: str | None
    llm_error: str | None = None
    llm_language: str | None = None
    llm_confidence: float | None = None


@dataclass
class ProductRecallNormalizationResult:
    normalized_product_query: str
    query_aliases: list[str]
    category_hints: list[str]
    normalization_mode: str
    llm_used: bool
    llm_provider: str
    llm_model: str | None
    llm_error: str | None = None
    llm_language: str | None = None
    llm_confidence: float | None = None


@dataclass
class ProductQueryAssistantResult:
    product_query: str
    query_aliases: list[str]
    category_hints: list[str]
    normalization_mode: str
    llm_used: bool
    llm_provider: str
    llm_model: str | None
    llm_error: str | None = None
    llm_language: str | None = None
    llm_confidence: float | None = None
    pipeline_mode: str | None = None
    theme_extraction: ProductThemeExtractionResult | None = None
    recall_normalization: ProductRecallNormalizationResult | None = None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _trim_phrase(value: str, *, max_length: int = 120) -> str:
    return value.strip()[:max_length].strip()


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


def _sanitize_phrase_list(values: list[str], *, max_items: int, max_length: int = 120) -> list[str]:
    sanitized = [_trim_phrase(str(value), max_length=max_length) for value in values if str(value).strip()]
    return _unique_nonempty(sanitized)[:max_items]


def _strip_reasoning_sections(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _clean_output_phrase(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^[-*\d.\s]+", "", cleaned)
    cleaned = cleaned.strip("` ")
    cleaned = re.sub(r"^\*+|\*+$", "", cleaned)
    cleaned = cleaned.strip(" ：:-")
    return _trim_phrase(cleaned)


def _extract_phrases_from_text_output(raw_text: str) -> list[str]:
    cleaned = _strip_reasoning_sections(raw_text)
    phrases: list[str] = []

    for match in re.findall(r"`([^`]+)`", cleaned):
        phrase = _clean_output_phrase(match)
        if phrase:
            phrases.append(phrase)

    for line in cleaned.splitlines():
        phrase = _clean_output_phrase(line)
        if not phrase:
            continue
        if any(marker in phrase.lower() for marker in ["英文召回词", "推荐第一个", "关键词顺序", "或拆分为关键词组合"]):
            continue
        if len(phrase.split()) > 12:
            continue
        if re.search(r"[a-zA-Z]", phrase):
            phrases.append(phrase)

    normalized_phrases = _sanitize_phrase_list(phrases, max_items=8)
    return [phrase for phrase in normalized_phrases if len(phrase) >= 3]


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_simple_english_catalog_query(
    product_query: str,
    query_aliases: list[str] | None = None,
    category_hints: list[str] | None = None,
) -> bool:
    if query_aliases or category_hints:
        return False
    text = _trim_phrase(product_query, max_length=240)
    if not text or len(text) > 80:
        return False
    if re.search(r"[^\x00-\x7F]", text):
        return False
    if re.search(r"[?？。；;!！\n\r]", text):
        return False
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not 1 <= len(tokens) <= 6:
        return False
    intent_words = {
        "analyze",
        "analysis",
        "compare",
        "competitor",
        "competitors",
        "find",
        "report",
        "research",
        "select",
        "trend",
        "trends",
    }
    return not any(token in intent_words for token in tokens)


def _should_skip_llm_for_simple_query(
    env_prefix: str,
    product_query: str,
    query_aliases: list[str] | None = None,
    category_hints: list[str] | None = None,
) -> bool:
    if _env_flag(f"{env_prefix}_FORCE_LLM", default=False):
        return False
    if not _env_flag(f"{env_prefix}_SKIP_SIMPLE_ENGLISH", default=True):
        return False
    return _looks_like_simple_english_catalog_query(product_query, query_aliases, category_hints)


def _parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
        if confidence > 1:
            confidence = confidence / 100.0
        return max(0.0, min(confidence, 1.0))

    text = str(value).strip().lower()
    if not text:
        return None

    confidence_map = {
        "very low": 0.1,
        "low": 0.25,
        "medium": 0.5,
        "moderate": 0.5,
        "medium high": 0.7,
        "high": 0.85,
        "very high": 0.95,
    }
    if text in confidence_map:
        return confidence_map[text]

    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent_match:
        confidence = float(percent_match.group(1)) / 100.0
        return max(0.0, min(confidence, 1.0))

    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        confidence = float(number_match.group(0))
        if confidence > 1:
            confidence = confidence / 100.0
        return max(0.0, min(confidence, 1.0))

    return None


def _build_theme_extraction_messages(
    product_query: str,
    query_aliases: tuple[str, ...],
    category_hints: tuple[str, ...],
    marketplace: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract the core ecommerce product theme from messy user input before catalog recall. "
                "Return only JSON with fields extracted_theme, query_aliases, category_hints, language, confidence. "
                "Focus on the product itself, not the user's analytical intent, reporting request, or conversational filler. "
                "Keep brand, model, material, size, pack-count, audience, and scenario only when they materially affect recall. "
                "If the input is multilingual, preserve useful original-language aliases when they improve downstream retrieval."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "marketplace": marketplace,
                    "raw_product_query": product_query,
                    "query_aliases": list(query_aliases),
                    "category_hints": list(category_hints),
                    "use_case": "theme_extraction_before_candidate_recall",
                    "output_rules": {
                        "extracted_theme_max_words": 10,
                        "query_aliases_max_items": 6,
                        "category_hints_max_items": 4,
                        "preserve_recall_critical_constraints": True,
                        "drop_conversational_or_analysis_intent": True,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_recall_normalization_messages(
    raw_product_query: str,
    extracted_theme: str,
    query_aliases: tuple[str, ...],
    category_hints: tuple[str, ...],
    marketplace: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You normalize extracted ecommerce product themes for sourcing and recall against multilingual Amazon catalogs. "
                "Return only JSON with fields normalized_product_query, query_aliases, category_hints, language, confidence. "
                "Always output one concise English normalized_product_query for internal standardization. "
                "For non-English marketplaces, keep the English canonical phrase but also add marketplace-local aliases and category hints when they materially improve recall. "
                "Preserve meaningful original-language aliases when the catalog may contain localized titles or categories. "
                "Keep brand, model, material, size, pack-count, and other recall-relevant constraints when they matter."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "marketplace": marketplace,
                    "raw_product_query": raw_product_query,
                    "extracted_theme": extracted_theme,
                    "query_aliases": list(query_aliases),
                    "category_hints": list(category_hints),
                    "use_case": "product_selection_and_candidate_recall",
                    "output_rules": {
                        "normalized_product_query_max_words": 8,
                        "query_aliases_max_items": 6,
                        "category_hints_max_items": 4,
                        "canonical_query_in_english": True,
                        "add_marketplace_local_aliases_when_helpful": True,
                        "preserve_original_language_aliases_when_useful": True,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def _build_theme_extraction_result(
    product_query: str,
    query_aliases: list[str],
    category_hints: list[str],
    *,
    mode: str,
    llm_used: bool,
    llm_provider: str,
    llm_model: str | None,
    llm_error: str | None = None,
    extracted_theme: str | None = None,
    extracted_query_aliases: list[str] | None = None,
    extracted_category_hints: list[str] | None = None,
    llm_language: str | None = None,
    llm_confidence: float | None = None,
) -> ProductThemeExtractionResult:
    final_theme = _trim_phrase(extracted_theme or product_query) or _trim_phrase(product_query)
    final_query_aliases = _sanitize_phrase_list((extracted_query_aliases or []) + query_aliases, max_items=6)
    if _normalize_text(final_theme) != _normalize_text(product_query):
        final_query_aliases = _sanitize_phrase_list([product_query] + final_query_aliases, max_items=6)
    final_category_hints = _sanitize_phrase_list((extracted_category_hints or []) + category_hints, max_items=4)
    return ProductThemeExtractionResult(
        extracted_theme=final_theme,
        query_aliases=final_query_aliases,
        category_hints=final_category_hints,
        extraction_mode=mode,
        llm_used=llm_used,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_error=llm_error,
        llm_language=llm_language,
        llm_confidence=llm_confidence,
    )


def _build_recall_normalization_result(
    raw_product_query: str,
    extracted_theme: str,
    query_aliases: list[str],
    category_hints: list[str],
    *,
    mode: str,
    llm_used: bool,
    llm_provider: str,
    llm_model: str | None,
    llm_error: str | None = None,
    normalized_product_query: str | None = None,
    normalized_query_aliases: list[str] | None = None,
    normalized_category_hints: list[str] | None = None,
    llm_language: str | None = None,
    llm_confidence: float | None = None,
) -> ProductRecallNormalizationResult:
    final_product_query = _trim_phrase(normalized_product_query or extracted_theme or raw_product_query)
    final_query_aliases = _sanitize_phrase_list((normalized_query_aliases or []) + query_aliases, max_items=6)
    if extracted_theme and _normalize_text(final_product_query) != _normalize_text(extracted_theme):
        final_query_aliases = _sanitize_phrase_list([extracted_theme] + final_query_aliases, max_items=6)
    final_category_hints = _sanitize_phrase_list((normalized_category_hints or []) + category_hints, max_items=4)
    return ProductRecallNormalizationResult(
        normalized_product_query=final_product_query,
        query_aliases=final_query_aliases,
        category_hints=final_category_hints,
        normalization_mode=mode,
        llm_used=llm_used,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_error=llm_error,
        llm_language=llm_language,
        llm_confidence=llm_confidence,
    )


def _build_salvaged_theme_extraction_result(
    product_query: str,
    query_aliases: list[str],
    category_hints: list[str],
    *,
    raw_text: str,
    llm_provider: str,
    llm_model: str | None,
) -> ProductThemeExtractionResult | None:
    phrases = _extract_phrases_from_text_output(raw_text)
    if not phrases:
        return None
    return _build_theme_extraction_result(
        product_query,
        query_aliases,
        category_hints,
        mode="llm_text_fallback",
        llm_used=True,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_error="provider ignored JSON mode; salvaged theme extraction from plain text output",
        extracted_theme=phrases[0],
        extracted_query_aliases=phrases[1:],
    )


def _build_salvaged_recall_normalization_result(
    raw_product_query: str,
    extracted_theme: str,
    query_aliases: list[str],
    category_hints: list[str],
    *,
    raw_text: str,
    llm_provider: str,
    llm_model: str | None,
) -> ProductRecallNormalizationResult | None:
    phrases = _extract_phrases_from_text_output(raw_text)
    if not phrases:
        return None
    return _build_recall_normalization_result(
        raw_product_query,
        extracted_theme,
        query_aliases,
        category_hints,
        mode="llm_text_fallback",
        llm_used=True,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_error="provider ignored JSON mode; salvaged recall normalization from plain text output",
        normalized_product_query=phrases[0],
        normalized_query_aliases=phrases[1:],
    )


@lru_cache(maxsize=512)
def _extract_product_theme_cached(
    env_prefix: str,
    product_query: str,
    query_aliases: tuple[str, ...],
    category_hints: tuple[str, ...],
    marketplace: str,
) -> ProductThemeExtractionResult:
    provider = build_llm_provider(env_prefix, provider_default="openai_compatible", enabled_default=False)
    if not provider.enabled:
        return _build_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            mode="disabled",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
        )
    if not provider.configured:
        return _build_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            mode="misconfigured",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=provider.error or f"{env_prefix} is enabled but provider config is incomplete",
        )

    try:
        data = provider.json(
            messages=_build_theme_extraction_messages(product_query, query_aliases, category_hints, marketplace),
            temperature=0,
        )
        llm_language = _trim_phrase(str(data.get("language") or ""), max_length=24) or None
        llm_confidence = _parse_confidence(data.get("confidence"))
        return _build_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            mode="llm",
            llm_used=True,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            extracted_theme=str(data.get("extracted_theme") or product_query),
            extracted_query_aliases=_sanitize_phrase_list(list(data.get("query_aliases") or []), max_items=6),
            extracted_category_hints=_sanitize_phrase_list(list(data.get("category_hints") or []), max_items=4),
            llm_language=llm_language,
            llm_confidence=llm_confidence,
        )
    except LLMJSONParseError as exc:
        salvaged = _build_salvaged_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            raw_text=exc.raw_text,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
        )
        if salvaged is not None:
            return salvaged
        return _build_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            mode="llm_failed_fallback_rules",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=f"{exc}; raw_text={exc.raw_text}",
        )
    except Exception as exc:
        return _build_theme_extraction_result(
            product_query,
            list(query_aliases),
            list(category_hints),
            mode="llm_failed_fallback_rules",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=str(exc),
        )


@lru_cache(maxsize=512)
def _normalize_product_recall_query_cached(
    env_prefix: str,
    raw_product_query: str,
    extracted_theme: str,
    query_aliases: tuple[str, ...],
    category_hints: tuple[str, ...],
    marketplace: str,
) -> ProductRecallNormalizationResult:
    provider = build_llm_provider(env_prefix, provider_default="openai_compatible", enabled_default=False)
    if not provider.enabled:
        return _build_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            mode="disabled",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
        )
    if not provider.configured:
        return _build_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            mode="misconfigured",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=provider.error or f"{env_prefix} is enabled but provider config is incomplete",
        )

    try:
        data = provider.json(
            messages=_build_recall_normalization_messages(
                raw_product_query,
                extracted_theme,
                query_aliases,
                category_hints,
                marketplace,
            ),
            temperature=0,
        )
        llm_language = _trim_phrase(str(data.get("language") or ""), max_length=24) or None
        llm_confidence = _parse_confidence(data.get("confidence"))
        return _build_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            mode="llm",
            llm_used=True,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            normalized_product_query=str(data.get("normalized_product_query") or extracted_theme or raw_product_query),
            normalized_query_aliases=_sanitize_phrase_list(list(data.get("query_aliases") or []), max_items=6),
            normalized_category_hints=_sanitize_phrase_list(list(data.get("category_hints") or []), max_items=4),
            llm_language=llm_language,
            llm_confidence=llm_confidence,
        )
    except LLMJSONParseError as exc:
        salvaged = _build_salvaged_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            raw_text=exc.raw_text,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
        )
        if salvaged is not None:
            return salvaged
        return _build_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            mode="llm_failed_fallback_rules",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=f"{exc}; raw_text={exc.raw_text}",
        )
    except Exception as exc:
        return _build_recall_normalization_result(
            raw_product_query,
            extracted_theme,
            list(query_aliases),
            list(category_hints),
            mode="llm_failed_fallback_rules",
            llm_used=False,
            llm_provider=provider.provider_name,
            llm_model=provider.model or None,
            llm_error=str(exc),
        )


def _compose_query_assistant_result(
    raw_product_query: str,
    raw_query_aliases: list[str],
    raw_category_hints: list[str],
    extraction: ProductThemeExtractionResult,
    normalization: ProductRecallNormalizationResult,
) -> ProductQueryAssistantResult:
    final_product_query = (
        _trim_phrase(normalization.normalized_product_query or extraction.extracted_theme or raw_product_query)
        or _trim_phrase(raw_product_query)
    )
    preserved_aliases = extraction.query_aliases + raw_query_aliases
    if _normalize_text(final_product_query) != _normalize_text(raw_product_query):
        preserved_aliases = [raw_product_query] + preserved_aliases
    final_query_aliases = _sanitize_phrase_list(normalization.query_aliases + preserved_aliases, max_items=8)
    final_category_hints = _sanitize_phrase_list(
        normalization.category_hints + extraction.category_hints + raw_category_hints,
        max_items=6,
    )
    return ProductQueryAssistantResult(
        product_query=final_product_query,
        query_aliases=final_query_aliases,
        category_hints=final_category_hints,
        normalization_mode=normalization.normalization_mode,
        llm_used=normalization.llm_used,
        llm_provider=normalization.llm_provider,
        llm_model=normalization.llm_model,
        llm_error=normalization.llm_error,
        llm_language=normalization.llm_language,
        llm_confidence=normalization.llm_confidence,
        pipeline_mode=f"{extraction.extraction_mode}->{normalization.normalization_mode}",
        theme_extraction=extraction,
        recall_normalization=normalization,
    )


class ProductRecallQueryAssistant:
    def __init__(self, env_prefix: str = "THEME_QUERY_NORMALIZER") -> None:
        self.env_prefix = env_prefix

    def provider(self) -> LLMProvider:
        return build_llm_provider(self.env_prefix, provider_default="openai_compatible", enabled_default=False)

    def provider_summary(self) -> dict[str, object]:
        return self.provider().summary()

    def extract_theme_intent(
        self,
        *,
        product_query: str,
        query_aliases: list[str] | None = None,
        category_hints: list[str] | None = None,
        marketplace: str = "US",
    ) -> ProductThemeExtractionResult:
        return _extract_product_theme_cached(
            self.env_prefix,
            product_query,
            tuple(query_aliases or []),
            tuple(category_hints or []),
            marketplace,
        )

    def normalize_recall_query(
        self,
        *,
        raw_product_query: str,
        extracted_theme: str,
        query_aliases: list[str] | None = None,
        category_hints: list[str] | None = None,
        marketplace: str = "US",
    ) -> ProductRecallNormalizationResult:
        return _normalize_product_recall_query_cached(
            self.env_prefix,
            raw_product_query,
            extracted_theme,
            tuple(query_aliases or []),
            tuple(category_hints or []),
            marketplace,
        )

    def normalize(
        self,
        *,
        product_query: str,
        query_aliases: list[str] | None = None,
        category_hints: list[str] | None = None,
        marketplace: str = "US",
    ) -> ProductQueryAssistantResult:
        if _should_skip_llm_for_simple_query(self.env_prefix, product_query, query_aliases, category_hints):
            provider = self.provider()
            extraction = _build_theme_extraction_result(
                product_query,
                list(query_aliases or []),
                list(category_hints or []),
                mode="rules_simple_english",
                llm_used=False,
                llm_provider=provider.provider_name,
                llm_model=provider.model or None,
            )
            normalization = _build_recall_normalization_result(
                product_query,
                extraction.extracted_theme,
                extraction.query_aliases,
                extraction.category_hints,
                mode="rules_simple_english",
                llm_used=False,
                llm_provider=provider.provider_name,
                llm_model=provider.model or None,
                normalized_product_query=extraction.extracted_theme,
            )
            return _compose_query_assistant_result(
                raw_product_query=product_query,
                raw_query_aliases=list(query_aliases or []),
                raw_category_hints=list(category_hints or []),
                extraction=extraction,
                normalization=normalization,
            )

        extraction = self.extract_theme_intent(
            product_query=product_query,
            query_aliases=query_aliases,
            category_hints=category_hints,
            marketplace=marketplace,
        )
        normalization = self.normalize_recall_query(
            raw_product_query=product_query,
            extracted_theme=extraction.extracted_theme,
            query_aliases=extraction.query_aliases,
            category_hints=extraction.category_hints,
            marketplace=marketplace,
        )
        return _compose_query_assistant_result(
            raw_product_query=product_query,
            raw_query_aliases=list(query_aliases or []),
            raw_category_hints=list(category_hints or []),
            extraction=extraction,
            normalization=normalization,
        )


def normalize_product_recall_query(
    *,
    product_query: str,
    query_aliases: list[str] | None = None,
    category_hints: list[str] | None = None,
    marketplace: str = "US",
    env_prefix: str = "THEME_QUERY_NORMALIZER",
) -> ProductQueryAssistantResult:
    assistant = ProductRecallQueryAssistant(env_prefix=env_prefix)
    return assistant.normalize(
        product_query=product_query,
        query_aliases=query_aliases,
        category_hints=category_hints,
        marketplace=marketplace,
    )