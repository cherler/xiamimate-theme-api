"""Category path and candidate-pool quality helpers."""
from __future__ import annotations

from typing import Any

from data_platform.api.product_theme.constants import (
    DEFAULT_DOMINANT_CATEGORY_SHARE_THRESHOLD,
    DEFAULT_MIN_CANDIDATE_POOL_SIZE,
    DEFAULT_TARGET_CANDIDATE_POOL_SIZE,
)


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


def _opportunity_title_from_category_path(category_path: Any, fallback_category: Any = None) -> str:
    parts = _category_path_parts(category_path)
    fallback = str(fallback_category or "").strip()
    if not parts:
        return fallback or "Amazon opportunity"

    leaf = parts[-1]
    audience_labels = {
        "women": "Women's",
        "men": "Men's",
        "girls": "Girls'",
        "boys": "Boys'",
    }
    for part in parts[:-1]:
        audience = audience_labels.get(part.strip().lower())
        if audience and not leaf.lower().startswith(audience.lower()):
            return f"{audience} {leaf}"

    return leaf or fallback or "Amazon opportunity"


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


def _dominant_category_summary(
    distribution: list[dict[str, Any]],
    candidate_count: int,
) -> tuple[str | None, int, float]:
    if not distribution or candidate_count <= 0:
        return None, 0, 0.0
    top = distribution[0]
    category = str(top.get("category") or "").strip() or None
    count = int(top.get("candidate_count") or 0)
    share = round(count / candidate_count, 4) if candidate_count > 0 else 0.0
    return category, count, share


def _build_candidate_pool_quality(
    candidate_items: list[dict[str, Any]],
    *,
    candidate_total_before_semantic_gate: int,
    candidate_total_before_category_anchor: int,
    leaf_distribution: list[dict[str, Any]],
    fine_distribution: list[dict[str, Any]],
    root_distribution: list[dict[str, Any]],
    min_pool_size: int = DEFAULT_MIN_CANDIDATE_POOL_SIZE,
    target_pool_size: int = DEFAULT_TARGET_CANDIDATE_POOL_SIZE,
    dominant_category_share_threshold: float = DEFAULT_DOMINANT_CATEGORY_SHARE_THRESHOLD,
) -> dict[str, Any]:
    candidate_count = len(candidate_items)
    dominant_leaf_category, dominant_leaf_count, dominant_leaf_share = _dominant_category_summary(
        leaf_distribution,
        candidate_count,
    )
    dominant_fine_category, dominant_fine_count, dominant_fine_share = _dominant_category_summary(
        fine_distribution,
        candidate_count,
    )
    dominant_root_category, dominant_root_count, dominant_root_share = _dominant_category_summary(
        root_distribution,
        candidate_count,
    )
    has_category_anchor = bool(
        dominant_leaf_category
        and dominant_leaf_category != "其他/未归类"
        and dominant_leaf_share >= dominant_category_share_threshold
    ) or bool(
        dominant_fine_category
        and dominant_fine_category != "其他/未归类"
        and dominant_fine_share >= dominant_category_share_threshold
    )

    insufficient_reasons: list[str] = []
    if candidate_count == 0:
        insufficient_reasons.append("no_candidates_after_recall")
    elif candidate_count < min_pool_size:
        insufficient_reasons.append("pure_candidate_count_below_min_pool_size")
    if candidate_count > 0 and not has_category_anchor:
        insufficient_reasons.append("dominant_category_share_below_threshold")

    is_sufficient_for_analysis = candidate_count >= min_pool_size and has_category_anchor
    should_expand_pool = (not is_sufficient_for_analysis) or candidate_count < target_pool_size
    if candidate_count < target_pool_size and "candidate_count_below_target_pool_size" not in insufficient_reasons:
        insufficient_reasons.append("candidate_count_below_target_pool_size")

    if is_sufficient_for_analysis and candidate_count >= target_pool_size:
        confidence = "high"
    elif is_sufficient_for_analysis:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "candidate_count": candidate_count,
        "candidate_total_before_semantic_gate": candidate_total_before_semantic_gate,
        "candidate_total_before_category_anchor": candidate_total_before_category_anchor,
        "min_pool_size": min_pool_size,
        "target_pool_size": target_pool_size,
        "dominant_category_share_threshold": dominant_category_share_threshold,
        "dominant_leaf_category": dominant_leaf_category,
        "dominant_leaf_count": dominant_leaf_count,
        "dominant_leaf_share": dominant_leaf_share,
        "dominant_fine_category": dominant_fine_category,
        "dominant_fine_count": dominant_fine_count,
        "dominant_fine_share": dominant_fine_share,
        "dominant_root_category": dominant_root_category,
        "dominant_root_count": dominant_root_count,
        "dominant_root_share": dominant_root_share,
        "category_anchor_confidence": confidence,
        "is_sufficient_for_analysis": is_sufficient_for_analysis,
        "should_expand_pool": should_expand_pool,
        "insufficient_coverage_reason": None if is_sufficient_for_analysis else (insufficient_reasons[0] if insufficient_reasons else None),
        "insufficient_coverage_reasons": insufficient_reasons,
    }
