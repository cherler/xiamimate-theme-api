"""Launch budget calculation service."""
from __future__ import annotations

from typing import Any

from data_platform.api.product_theme.query_utils import _normalize_marketplace
from data_platform.api.product_theme.schemas import LaunchBudgetCalculatorRequest


def calculate_launch_budget(request: LaunchBudgetCalculatorRequest) -> dict[str, Any]:
    _domain, marketplace = _normalize_marketplace(request.marketplace)

    def assumption(name: str, value: Any, default: Any) -> dict[str, Any]:
        explicit = value is not None
        return {
            "value": value if explicit else default,
            "source": "user_input" if explicit else "default_assumption",
        }

    assumptions = {
        "selling_price": assumption("selling_price", request.selling_price, 22.99),
        "unit_product_cost": assumption("unit_product_cost", request.unit_product_cost, 6.0),
        "landed_cost_per_unit": assumption("landed_cost_per_unit", request.landed_cost_per_unit, None),
        "packaging_cost": assumption("packaging_cost", request.packaging_cost, 0.5),
        "inbound_shipping_per_unit": assumption("inbound_shipping_per_unit", request.inbound_shipping_per_unit, 1.5),
        "duty_per_unit": assumption("duty_per_unit", request.duty_per_unit, 0.3),
        "fba_fee": assumption("fba_fee", request.fba_fee, 3.5),
        "referral_fee_rate": assumption("referral_fee_rate", request.referral_fee_rate, 0.15),
        "coupon_discount_rate": assumption("coupon_discount_rate", request.coupon_discount_rate, 0.10),
        "return_rate": assumption("return_rate", request.return_rate, 0.08),
        "fixed_startup_cost": assumption("fixed_startup_cost", request.fixed_startup_cost, 750.0),
        "monthly_fixed_cost": assumption("monthly_fixed_cost", request.monthly_fixed_cost, 80.0),
        "monthly_ad_budget": assumption("monthly_ad_budget", request.monthly_ad_budget, 800.0),
        "launch_units": assumption("launch_units", request.launch_units, 300),
        "launch_months": assumption("launch_months", request.launch_months, 3),
    }

    def av(name: str) -> float:
        return float(assumptions[name]["value"])

    selling_price = av("selling_price")
    landed_cost_is_explicit = assumptions["landed_cost_per_unit"]["source"] == "user_input"
    landed_cost_per_unit = av("landed_cost_per_unit") if landed_cost_is_explicit else av("unit_product_cost") + av("packaging_cost") + av("inbound_shipping_per_unit") + av("duty_per_unit")
    referral_fee_per_unit = selling_price * av("referral_fee_rate")
    coupon_reserve_per_unit = selling_price * av("coupon_discount_rate")
    return_reserve_per_unit = selling_price * av("return_rate")
    contribution_margin_per_unit = selling_price - landed_cost_per_unit - av("fba_fee") - referral_fee_per_unit - coupon_reserve_per_unit - return_reserve_per_unit
    contribution_margin_rate = contribution_margin_per_unit / selling_price if selling_price else 0.0
    monthly_operating_cost = av("monthly_fixed_cost") + av("monthly_ad_budget")
    break_even_units_per_month = monthly_operating_cost / contribution_margin_per_unit if contribution_margin_per_unit > 0 else None
    break_even_units_per_day = break_even_units_per_month / 30.0 if break_even_units_per_month is not None else None

    scenario_specs = [
        ("lean", 0.75, 0.75, 0.10),
        ("standard", 1.0, 1.0, 0.25),
        ("buffered", 1.25, 1.25, 0.35),
    ]
    scenarios: list[dict[str, Any]] = []
    for name, unit_multiplier, ad_multiplier, buffer_rate in scenario_specs:
        units = max(1, int(round(av("launch_units") * unit_multiplier)))
        months = int(round(av("launch_months")))
        inventory_cash = landed_cost_per_unit * units
        operating_cash = (av("monthly_fixed_cost") + av("monthly_ad_budget") * ad_multiplier) * months
        promo_reserve = (coupon_reserve_per_unit + return_reserve_per_unit) * units
        subtotal = av("fixed_startup_cost") + inventory_cash + operating_cash + promo_reserve
        buffer = subtotal * buffer_rate
        scenarios.append(
            {
                "scenario": name,
                "launch_units": units,
                "launch_months": months,
                "inventory_cash": round(inventory_cash, 2),
                "operating_cash": round(operating_cash, 2),
                "promo_and_return_reserve": round(promo_reserve, 2),
                "fixed_startup_cost": round(av("fixed_startup_cost"), 2),
                "buffer_rate": buffer_rate,
                "buffer_cash": round(buffer, 2),
                "startup_cash_required": round(subtotal + buffer, 2),
            }
        )

    return {
        "source_tool": "launch_budget_calculator",
        "marketplace": marketplace,
        "product_theme": request.product_theme,
        "currency": "USD",
        "assumptions": assumptions,
        "unit_economics": {
            "selling_price": round(selling_price, 2),
            "landed_cost_per_unit": round(landed_cost_per_unit, 2),
            "landed_cost_source": "user_input" if landed_cost_is_explicit else "component_sum",
            "referral_fee_per_unit": round(referral_fee_per_unit, 2),
            "fba_fee_per_unit": round(av("fba_fee"), 2),
            "coupon_reserve_per_unit": round(coupon_reserve_per_unit, 2),
            "return_reserve_per_unit": round(return_reserve_per_unit, 2),
            "contribution_margin_per_unit": round(contribution_margin_per_unit, 2),
            "contribution_margin_rate": round(contribution_margin_rate, 4),
        },
        "break_even": {
            "monthly_operating_cost": round(monthly_operating_cost, 2),
            "break_even_units_per_month": round(break_even_units_per_month, 2) if break_even_units_per_month is not None else None,
            "break_even_units_per_day": round(break_even_units_per_day, 2) if break_even_units_per_day is not None else None,
            "formula": "(monthly_fixed_cost + monthly_ad_budget) / contribution_margin_per_unit",
        },
        "scenarios": scenarios,
        "assumption_policy": {
            "user_input": "Value provided by the caller or inherited from prior tool facts.",
            "default_assumption": "Generic planning default; replace with supplier, ad, fee, or category-specific inputs when available.",
        },
        "warnings": [
            "Use explicit supplier quotes, FBA fee estimates, return rate, and ad budget when available.",
            "Break-even is a deterministic planning calculation, not a demand forecast.",
        ],
    }