"""Seller-scope policy for small cross-border ecommerce operators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


SELLER_SCOPE_POLICY_VERSION = "cross_border_sme_v1"

_BLOCKED_CATEGORY_SEGMENTS: dict[str, tuple[str, ...]] = {
    "security_surveillance_subcategory": (
        "security surveillance",
        "home security systems",
        "surveillance systems",
        "video surveillance",
        "安防",
        "监控",
    ),
}


@dataclass(frozen=True)
class SellerScopeDecision:
    allowed: bool
    reason_code: str
    matched_terms: tuple[str, ...] = ()
    policy_version: str = SELLER_SCOPE_POLICY_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "matched_terms": list(self.matched_terms),
            "policy_version": self.policy_version,
        }


_BLOCKED_TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "digital_or_licensed_goods": (
        "digital software",
        "software",
        "software download",
        "software license",
        "activation key",
        "license key",
        "product key",
        "antivirus",
        "anti virus",
        "malware",
        "vpn subscription",
        "subscription",
        "download code",
        "digital code",
        "kindle",
        "ebook",
        "e book",
        "audible",
        "audiobook",
        "apps games",
        "appstore",
        "gift card",
        "gift cards",
        "软件",
        "杀毒软件",
        "授权码",
        "激活码",
        "下载码",
        "订阅",
        "电子书",
        "有声书",
        "礼品卡",
    ),
    "copyright_media": (
        "movies tv",
        "movies",
        "prime video",
        "streaming",
        "dvd",
        "blu ray",
        "bluray",
        "cds vinyl",
        "vinyl record",
        "digital music",
        "music download",
        "video games",
        "console game",
        "电影",
        "影视",
        "影视节目",
        "流媒体",
        "音乐",
        "唱片",
        "电子游戏",
    ),
    "regulated_or_restricted_goods": (
        "firearm",
        "firearms",
        "ammunition",
        "gun parts",
        "tactical knife",
        "switchblade",
        "tobacco",
        "vape",
        "e cigarette",
        "alcohol",
        "prescription",
        "controlled substance",
        "medical device",
        "dietary supplement",
        "sex toy",
        "adult toy",
        "hazmat",
        "hazardous chemical",
        "live plant",
        "live animal",
        "perishable food",
        "枪支",
        "弹药",
        "烟草",
        "电子烟",
        "酒精",
        "处方",
        "医疗器械",
        "保健品",
        "成人用品",
        "危险化学品",
        "活体植物",
        "活体动物",
        "易腐食品",
    ),
}


def _normalize_scope_text(value: object) -> str:
    text = str(value or "").lower()
    return " ".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text))


def _iter_text_values(
    *,
    category_path: object | None = None,
    category_name: object | None = None,
    query: object | None = None,
    keywords: Iterable[object] | None = None,
    title: object | None = None,
) -> list[str]:
    values = [category_path, category_name, query, title]
    if keywords:
        values.extend(keywords)
    return [_normalize_scope_text(value) for value in values if _normalize_scope_text(value)]


def _matches_term(text: str, term: str) -> bool:
    normalized_term = _normalize_scope_text(term)
    if not normalized_term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", text) is not None


def _category_segment_reason(category_path: object | None = None, category_name: object | None = None) -> str | None:
    scope_text = " ".join(
        part
        for part in (
            _normalize_scope_text(category_path),
            _normalize_scope_text(category_name),
        )
        if part
    )
    if not scope_text:
        return None
    for reason_code, segments in _BLOCKED_CATEGORY_SEGMENTS.items():
        if any(_matches_term(scope_text, segment) for segment in segments):
            return reason_code
    return None


def evaluate_seller_scope(
    *,
    category_path: object | None = None,
    category_name: object | None = None,
    query: object | None = None,
    keywords: Iterable[object] | None = None,
    title: object | None = None,
) -> SellerScopeDecision:
    """Evaluate whether a category/query fits SME physical-goods cross-border selling."""
    category_reason = _category_segment_reason(category_path=category_path, category_name=category_name)
    if category_reason:
        return SellerScopeDecision(
            allowed=False,
            reason_code=category_reason,
            matched_terms=(str(category_path or category_name or "Security & Surveillance"),),
        )

    texts = _iter_text_values(
        category_path=category_path,
        category_name=category_name,
        query=query,
        keywords=keywords,
        title=title,
    )
    if not texts:
        return SellerScopeDecision(allowed=True, reason_code="insufficient_scope_signal")

    matched_by_code: dict[str, list[str]] = {}
    for reason_code, terms in _BLOCKED_TERM_GROUPS.items():
        for term in terms:
            if any(_matches_term(text, term) for text in texts):
                matched_by_code.setdefault(reason_code, []).append(term)

    if matched_by_code:
        reason_code = next(iter(matched_by_code))
        return SellerScopeDecision(
            allowed=False,
            reason_code=reason_code,
            matched_terms=tuple(dict.fromkeys(matched_by_code[reason_code])),
        )
    return SellerScopeDecision(allowed=True, reason_code="physical_goods_scope")


def is_seller_scope_allowed(**kwargs: object) -> bool:
    return evaluate_seller_scope(**kwargs).allowed


def filter_seller_scope_keywords(keywords: Iterable[str]) -> tuple[list[str], list[SellerScopeDecision]]:
    kept: list[str] = []
    blocked: list[SellerScopeDecision] = []
    for keyword in keywords:
        decision = evaluate_seller_scope(query=keyword, keywords=[keyword])
        if decision.allowed:
            kept.append(keyword)
        else:
            blocked.append(decision)
    return kept, blocked
