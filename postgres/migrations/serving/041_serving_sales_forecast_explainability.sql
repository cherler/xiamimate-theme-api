-- explainability fields for item-level sales forecast payloads consumed by theme-api.

ALTER TABLE serving.item_market_sales_forecast_current
    ADD COLUMN IF NOT EXISTS primary_driver_feature VARCHAR,
    ADD COLUMN IF NOT EXISTS primary_driver_label VARCHAR,
    ADD COLUMN IF NOT EXISTS primary_driver_direction VARCHAR CHECK (primary_driver_direction IN ('positive', 'negative')),
    ADD COLUMN IF NOT EXISTS primary_driver_contribution_share DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS top_feature_contributions JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS driver_summary_text TEXT;

CREATE INDEX IF NOT EXISTS idx_item_market_sales_forecast_current_primary_driver_feature
    ON serving.item_market_sales_forecast_current (primary_driver_feature);
