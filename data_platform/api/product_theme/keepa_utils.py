"""Keepa CSV and stats extraction helpers."""
from __future__ import annotations


def _keepa_latest_value(csv_2d: list, index: int) -> float | int | None:
    if not csv_2d or index >= len(csv_2d):
        return None
    arr = csv_2d[index]
    if not arr or len(arr) < 2:
        return None
    raw = arr[-1]
    if raw is None or raw == -1:
        return None
    return raw


def _keepa_latest_price(csv_2d: list, index: int, is_yen: bool) -> float | None:
    val = _keepa_latest_value(csv_2d, index)
    if val is None:
        return None
    if is_yen:
        return round(float(val), 2)
    return round(float(val) / 100, 2)


def _keepa_stats_value(stats_arr: list, index: int) -> float | int | None:
    if not stats_arr or index >= len(stats_arr):
        return None
    raw = stats_arr[index]
    if raw is None or raw == -1:
        return None
    return raw


def _keepa_stats_price(stats: dict, key: str, index: int, is_yen: bool) -> float | None:
    arr = stats.get(key) or []
    val = _keepa_stats_value(arr, index)
    if val is None:
        return None
    if is_yen:
        return round(float(val), 2)
    return round(float(val) / 100, 2)