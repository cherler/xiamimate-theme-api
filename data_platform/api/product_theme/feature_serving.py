"""Feature-serving status helpers for Product Theme API."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from data_platform.api.product_theme.constants import (
    THEME_FEATURE_RETENTION_DAYS,
    THEME_FEATURE_SERVING_TABLES,
)
from data_platform.api.product_theme.db import _postgres_conn, _run_pg_dict_query


def _effective_feature_window_days(requested_days: int) -> int:
    return min(requested_days, THEME_FEATURE_RETENTION_DAYS)


def _iso_date_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _get_theme_feature_serving_status(include_data_max_date: bool = True) -> dict[str, Any]:
    with _postgres_conn() as conn:
        registry_row = _run_pg_dict_query(
            conn,
            """
            SELECT
                to_regclass(%s) AS base_table,
                to_regclass(%s) AS trends_table,
                to_regclass(%s) AS cross_table
            """,
            [
                THEME_FEATURE_SERVING_TABLES["base"],
                THEME_FEATURE_SERVING_TABLES["trends"],
                THEME_FEATURE_SERVING_TABLES["cross"],
            ],
        )[0]

        missing_tables = [
            THEME_FEATURE_SERVING_TABLES[table_name]
            for table_name, registry_key in (
                ("base", "base_table"),
                ("trends", "trends_table"),
                ("cross", "cross_table"),
            )
            if registry_row.get(registry_key) is None
        ]
        if missing_tables:
            raise HTTPException(
                status_code=500,
                detail=(
                    "theme feature serving tables missing: "
                    f"{', '.join(missing_tables)}. "
                    "sync week1 foundation features into PostgreSQL before using the API."
                ),
            )

        max_date_row = {
            "base_max_date": None,
            "trends_max_date": None,
            "cross_max_date": None,
        }
        if include_data_max_date:
            max_date_row = _run_pg_dict_query(
                conn,
                """
                SELECT
                    (SELECT MAX(date) FROM serving.theme_base_daily) AS base_max_date,
                    (SELECT MAX(date) FROM serving.theme_trends_daily) AS trends_max_date,
                    (SELECT MAX(date) FROM serving.theme_cross_daily) AS cross_max_date
                """,
            )[0]

    max_dates = [value for value in max_date_row.values() if value is not None]
    return {
        "schema": "serving",
        "retention_days": THEME_FEATURE_RETENTION_DAYS,
        "data_max_date": max(max_dates).isoformat() if max_dates else None,
        "tables": {
            "theme_base_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["base"],
                "data_max_date": max_date_row["base_max_date"].isoformat() if max_date_row.get("base_max_date") else None,
            },
            "theme_trends_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["trends"],
                "data_max_date": max_date_row["trends_max_date"].isoformat() if max_date_row.get("trends_max_date") else None,
            },
            "theme_cross_daily": {
                "name": THEME_FEATURE_SERVING_TABLES["cross"],
                "data_max_date": max_date_row["cross_max_date"].isoformat() if max_date_row.get("cross_max_date") else None,
            },
        },
    }