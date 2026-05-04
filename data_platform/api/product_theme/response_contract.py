"""Response envelope and evidence contract helpers for product theme tools."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data_platform.api.product_theme.constants import API_RESPONSE_SCHEMA


TOOL_CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "/api/product-theme/opportunity-discovery": {
        "tool_name": "opportunity_discovery",
        "capability": "opportunity_discovery",
        "answers": ["机会发现", "机会排序", "可下钻机会入口"],
        "fact_boundary": "Use opportunity rows and metric definitions as facts; strategic interpretations are model-authored hypotheses.",
    },
    "/api/product-theme/resolve-candidates": {
        "tool_name": "resolve_candidates",
        "capability": "candidate_pool_resolution",
        "answers": ["候选 ASIN 池", "召回依据", "候选池质量"],
        "fact_boundary": "Candidate identities, category paths, and pool_quality are facts; market conclusions require downstream metric tools.",
    },
    "/api/product-theme/candidate-pool-stats": {
        "tool_name": "candidate_pool_stats",
        "capability": "candidate_pool_descriptive_stats",
        "answers": ["销量/价格/评论描述统计", "样本规模"],
        "fact_boundary": "Returned aggregate metrics are tool facts for the requested ASIN pool and window.",
    },
    "/api/product-theme/candidate-pool-trends": {
        "tool_name": "candidate_pool_trends",
        "capability": "candidate_pool_trend_diagnostics",
        "answers": ["趋势覆盖", "搜索趋势", "趋势阶段"],
        "fact_boundary": "Trend metrics are directional signals unless coverage is high and the window is complete.",
    },
    "/api/product-theme/product-forecast-explain": {
        "tool_name": "product_forecast_explain",
        "capability": "trained_forecast_explainability",
        "answers": ["模型预测", "预测驱动因素", "预测覆盖率"],
        "fact_boundary": "Forecast fields are model outputs, not guaranteed future sales; explainability describes model drivers.",
    },
    "/api/product-theme/asin-history-timeseries": {
        "tool_name": "asin_history_timeseries",
        "capability": "asin_history_analysis",
        "answers": ["历史销量", "价格/BSR/评论趋势", "数据覆盖"],
        "fact_boundary": "Timeseries rows and summaries are facts; growth narratives must respect coverage and missing-history notes.",
    },
    "/api/product-theme/category-benchmark": {
        "tool_name": "category_benchmark",
        "capability": "category_benchmarking",
        "answers": ["类目基准", "候选池对比", "本地覆盖"],
        "fact_boundary": "Benchmark claims are bounded by benchmark_is_precise and local_category_coverage.",
    },
    "/api/product-theme/launch-budget-calculator": {
        "tool_name": "launch_budget_calculator",
        "capability": "unit_economics_and_launch_budget",
        "answers": ["启动资金", "单件经济模型", "盈亏平衡", "多场景预算"],
        "fact_boundary": "Arithmetic outputs are deterministic calculations from explicit assumptions; market feasibility remains a hypothesis.",
    },
}

CLAIM_STRENGTH_GUIDE = {
    "tool_fact": "Direct field returned by a tool for the requested entity/window.",
    "derived_metric": "Deterministic calculation from tool facts or explicit assumptions.",
    "directional_signal": "Reasonable trend or benchmark reading with limited coverage or sample size.",
    "hypothesis": "Business interpretation that should be phrased as a possibility, not a confirmed outcome.",
}


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _response_meta(endpoint: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "endpoint": endpoint,
        "api_version": "2026-04-10",
        "response_schema": API_RESPONSE_SCHEMA,
        "generated_at": _utc_now_iso(),
    }
    if extra:
        meta.update(extra)
    return meta


def _tool_name_from_endpoint(endpoint: str) -> str:
    return endpoint.rsplit("/", 1)[-1].replace("-", "_")


def _coverage_claim_strength(coverage_ratio: float | None) -> str:
    if coverage_ratio is None:
        return "directional_signal"
    if coverage_ratio >= 0.8:
        return "tool_fact"
    if coverage_ratio >= 0.5:
        return "directional_signal"
    return "hypothesis"


def _build_evidence_ledger(endpoint: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    tool_name = _tool_name_from_endpoint(endpoint)
    if endpoint == "/api/product-theme/asin-history-timeseries":
        ledger: list[dict[str, Any]] = []
        for item in data.get("items") if isinstance(data.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            summary = item.get("window_summary") if isinstance(item.get("window_summary"), dict) else {}
            latest = item.get("latest_snapshot") if isinstance(item.get("latest_snapshot"), dict) else {}
            coverage_ratio = _safe_float(summary.get("coverage_ratio"), -1.0)
            coverage = coverage_ratio if coverage_ratio >= 0 else None
            ledger.append(
                {
                    "evidence_id": f"asin_history:{item.get('asin') or latest.get('asin') or 'unknown'}:{data.get('window_days')}d",
                    "entity": {"type": "asin", "asin": item.get("asin") or latest.get("asin")},
                    "source_tool": tool_name,
                    "source_labels": ["tool_fact", "derived_metric"],
                    "coverage": {
                        "requested_days": data.get("window_days"),
                        "observed_rows": summary.get("series_row_count"),
                        "coverage_ratio": coverage,
                    },
                    "allowed_claim_strength": _coverage_claim_strength(coverage),
                    "notes": ["Low coverage should be described as directional, not confirmed growth."],
                }
            )
        return ledger

    if endpoint == "/api/product-theme/product-forecast-explain":
        explanations = data.get("asin_forecast_explanations") if isinstance(data.get("asin_forecast_explanations"), list) else []
        total = len(explanations)
        model_hits = _safe_int(data.get("forecast_model_hit_count"))
        coverage_ratio = round(model_hits / total, 4) if total else None
        return [
            {
                "evidence_id": f"forecast_explain:{data.get('candidate_pool', {}).get('candidate_pool_id') if isinstance(data.get('candidate_pool'), dict) else 'pool'}",
                "entity": {"type": "candidate_pool", "candidate_pool": data.get("candidate_pool")},
                "source_tool": tool_name,
                "source_labels": ["model_output", "derived_metric"],
                "coverage": {"model_hit_count": model_hits, "item_count": total, "coverage_ratio": coverage_ratio},
                "allowed_claim_strength": _coverage_claim_strength(coverage_ratio),
                "notes": ["Forecast values are model outputs and should not be presented as guaranteed future sales."],
            }
        ]

    if endpoint == "/api/product-theme/launch-budget-calculator":
        return [
            {
                "evidence_id": "launch_budget:deterministic_calculation",
                "entity": {"type": "scenario_set", "product_theme": data.get("product_theme")},
                "source_tool": tool_name,
                "source_labels": ["derived_metric", "explicit_assumption", "default_assumption"],
                "coverage": {"scenario_count": len(data.get("scenarios") or [])},
                "allowed_claim_strength": "derived_metric",
                "notes": ["Budget and break-even numbers are arithmetic outputs from assumptions, not market guarantees."],
            }
        ]

    if endpoint == "/api/product-theme/opportunity-discovery":
        return [
            {
                "evidence_id": f"opportunity_discovery:{data.get('marketplace') or 'market'}:{data.get('opportunity_count') or 0}",
                "entity": {"type": "opportunity_list", "marketplace": data.get("marketplace")},
                "source_tool": tool_name,
                "source_labels": ["tool_fact", "derived_metric"],
                "coverage": {"opportunity_count": data.get("opportunity_count")},
                "allowed_claim_strength": "directional_signal",
                "notes": ["Opportunity cards are prioritization signals and entry points for further analysis."],
            }
        ]

    return [
        {
            "evidence_id": f"{tool_name}:response",
            "entity": {"type": "tool_response"},
            "source_tool": tool_name,
            "source_labels": ["tool_fact"],
            "allowed_claim_strength": "tool_fact",
        }
    ]


def _with_response_contract(endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return data
    contract = TOOL_CAPABILITY_CONTRACTS.get(
        endpoint,
        {
            "tool_name": _tool_name_from_endpoint(endpoint),
            "capability": _tool_name_from_endpoint(endpoint),
            "answers": [],
            "fact_boundary": "Use returned fields as facts; strategic interpretation remains model-authored.",
        },
    )
    enriched = dict(data)
    enriched.setdefault(
        "tool_contract",
        {
            "schema_version": "xiamimate_tool_contract_v1",
            **contract,
            "usage_policy": "Do not hard-code user workflows. Select tools by capability, inputs, and returned evidence.",
        },
    )
    enriched.setdefault(
        "evidence_contract",
        {
            "schema_version": "xiamimate_evidence_contract_v1",
            "evidence_ledger": _build_evidence_ledger(endpoint, data),
            "claim_strength_guide": CLAIM_STRENGTH_GUIDE,
            "response_policy": "Separate tool facts, deterministic calculations, assumptions, and hypotheses in final answers.",
        },
    )
    return enriched


def _success_response(endpoint: str, data: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": _with_response_contract(endpoint, data),
        "meta": _response_meta(endpoint),
    }


def _error_response(endpoint: str, code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "code": code,
        "message": message,
        "data": {},
        "meta": _response_meta(endpoint),
    }
