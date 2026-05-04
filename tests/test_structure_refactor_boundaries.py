from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class ProductThemeStructureBoundaryTests(unittest.TestCase):
    def test_compat_entrypoint_keeps_app_factory_boundary(self) -> None:
        source = read_repo_file("data_platform/api/product_theme_api.py")

        self.assertIn("create_product_theme_app", source)
        self.assertIn("app = create_product_theme_app(", source)
        self.assertNotIn("@app.get", source)
        self.assertNotIn("@app.post", source)
        self.assertNotIn("FastAPI(", source)

    def test_compat_entrypoint_reexports_legacy_helpers(self) -> None:
        from data_platform.api import product_theme_api

        self.assertEqual(product_theme_api._success_response.__module__, "data_platform.api.product_theme.response_contract")
        self.assertEqual(product_theme_api._build_query_variants.__module__, "data_platform.api.product_theme.candidate_matching")
        self.assertEqual(product_theme_api._build_candidate_pool_quality.__module__, "data_platform.api.product_theme.category_utils")
        self.assertEqual(product_theme_api._keepa_latest_price.__module__, "data_platform.api.product_theme.keepa_utils")

    def test_product_theme_routes_remain_registered(self) -> None:
        from data_platform.api.product_theme.server import app

        paths = [route.path for route in app.routes if hasattr(route, "methods")]
        expected = [
            "/health",
            "/api/product-theme/resolve-candidates",
            "/api/product-theme/opportunity-discovery",
            "/api/product-theme/product-forecast-explain",
            "/api/product-theme/launch-budget-calculator",
            "/api/product-theme/keepa-asin-lookup",
        ]

        self.assertEqual([path for path in expected if path not in paths], [])

    def test_service_script_uses_new_product_theme_entrypoint(self) -> None:
        source = read_repo_file("scripts/manage_theme_api.sh")

        self.assertIn('APP_ENTRYPOINT="data_platform.api.product_theme.server:app"', source)
        self.assertIn("data_platform.api.product_theme.server:app", source)
        self.assertIn('"$PYTHON_BIN" -m uvicorn "$APP_ENTRYPOINT"', source)
        self.assertNotIn("uvicorn data_platform.api.product_theme_api:app", source)

    def test_tikhub_code_must_not_land_in_theme_api_entrypoint(self) -> None:
        source = read_repo_file("data_platform/api/product_theme_api.py").lower()

        self.assertNotIn("tikhub", source)
        self.assertNotIn("tiktok_shop_opportunity", source)


if __name__ == "__main__":
    unittest.main()