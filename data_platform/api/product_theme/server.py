"""Runtime ASGI entrypoint for Product Theme API.

Service startup scripts should load this module.  The legacy
data_platform.api.product_theme_api module remains available for import
compatibility while ProductThemeService is migrated endpoint by endpoint.
"""
from __future__ import annotations

from data_platform.api.product_theme_api import app


__all__ = ["app"]