"""商品工作台数据契约（theme-api 侧，纯函数、可选附加）。

把一次主题分析结果（response data dict）映射为 chat-backend 工作台所需的
``brief`` / ``evidence`` 载荷。该模块是**附加**能力：

- 受 ``THEME_API_WORKSPACE_CONTRACT_ENABLED`` 开关控制是否在响应中挂载 ``workspace_payload``。
- 纯函数、零副作用，任何缺字段都安全降级为空结构，绝不影响既有响应。
- brief 作为「产品简报数据总线」：详情页生成、内容生产复用同一份。
- evidence 作为「证据数据」：证据图（SVG）、报告画布（ECharts）复用同一份。
"""
from __future__ import annotations

import os
from typing import Any


WORKSPACE_CONTRACT_ENABLED = os.environ.get(
    "THEME_API_WORKSPACE_CONTRACT_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

WORKSPACE_CONTRACT_SCHEMA_VERSION = "xiamimate_workspace_contract_v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _build_brief(data: dict[str, Any]) -> dict[str, Any]:
    """从分析结果抽取产品简报（详情页/内容生产的数据总线）。"""
    theme = _first_nonempty(data.get("product_theme"), data.get("theme"), data.get("query"))
    price_band = _as_dict(data.get("price_band") or data.get("price_range"))
    return {
        "product_theme": theme,
        "marketplace": _first_nonempty(data.get("marketplace"), data.get("target_market")),
        "category": _first_nonempty(data.get("category"), data.get("category_path")),
        "audience": data.get("audience"),
        "selling_points": _as_list(data.get("selling_points")),
        "price_band": {
            "low": price_band.get("low"),
            "high": price_band.get("high"),
            "current": price_band.get("current") or data.get("current_price"),
        },
    }


def _build_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """从分析结果抽取证据数据（证据图/报告画布复用同一份）。"""
    summary = _as_dict(data.get("window_summary"))
    return {
        "schema_version": WORKSPACE_CONTRACT_SCHEMA_VERSION,
        "trend_series": _as_list(data.get("trend_series") or summary.get("series")),
        "price_band": _as_dict(data.get("price_band") or data.get("price_range")),
        "competition_score": _first_nonempty(
            data.get("competition_score"), data.get("competition_index")
        ),
        "forecast_band": _as_dict(data.get("forecast_band")),
        "risk_lights": _as_list(data.get("risk_lights")),
        "coverage_status": data.get("coverage_status"),
    }


def build_workspace_payload(analysis_result: dict[str, Any]) -> dict[str, Any]:
    """把分析结果转成工作台载荷：``{theme_key, title, brief, evidence}``。

    纯函数：输入任意 dict，输出稳定结构；非 dict 输入安全降级。
    """
    data = _as_dict(analysis_result)
    theme = _first_nonempty(
        data.get("product_theme"), data.get("theme"), data.get("query"), "unknown"
    )
    theme_key = str(theme).strip().lower() or "unknown"
    title = str(_first_nonempty(data.get("title"), theme, "未命名工作台"))
    return {
        "schema_version": WORKSPACE_CONTRACT_SCHEMA_VERSION,
        "theme_key": theme_key,
        "title": title,
        "brief": _build_brief(data),
        "evidence": _build_evidence(data),
    }


def maybe_attach_workspace_payload(
    data: dict[str, Any], *, enabled: bool | None = None
) -> dict[str, Any]:
    """可选地把 ``workspace_payload`` 挂到响应 data 上（附加、幂等、可回滚）。

    开关关闭时原样返回，不改变任何既有字段。
    """
    if not isinstance(data, dict):
        return data
    is_enabled = WORKSPACE_CONTRACT_ENABLED if enabled is None else enabled
    if not is_enabled:
        return data
    if "workspace_payload" in data:
        return data
    enriched = dict(data)
    enriched["workspace_payload"] = build_workspace_payload(data)
    return enriched
