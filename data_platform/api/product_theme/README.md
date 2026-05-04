# Product Theme API modularization boundary

The runtime entrypoint is still `data_platform.api.product_theme_api`.

This package is reserved for gradual extraction from the monolithic module:

1. `constants.py` for marketplace/domain mappings, table names, and thresholds.
2. `response_contract.py` for success/error wrappers and evidence contracts.
3. `db.py` for PostgreSQL pool and query helpers.
4. `schemas.py` for Pydantic request models.
5. `query_utils.py` for marketplace, text, scalar, and token normalization helpers.
6. `candidate_scoring.py` for reusable score helpers.
7. `category_utils.py` for category path, display name, distribution, and pool-quality helpers.
8. `candidate_matching.py` for query variants, required terms, category match scoring, ASIN cleanup, and SQL prefilter terms.
9. `app.py` for FastAPI app assembly, middleware, exception handlers, and route wrappers.
10. `feature_serving.py` for feature window, serving table status, and date serialization helpers.
11. `keepa_utils.py` for Keepa CSV/latest value and stats extraction helpers.
12. `services/` for endpoint-by-endpoint service migration; `services/launch_budget.py` now owns the launch budget calculation, while candidate, opportunity, forecast, and ASIN history boundaries are reserved for later migration.
13. `routes/` for later route-module migration.

Do not add TikTok Shop/TikHub provider code here in the first phase; that native
tool belongs in chat-backend internal provider modules.
