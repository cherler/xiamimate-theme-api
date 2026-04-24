-- serving.* current forecast tables consumed by theme-api via second-query merge.

CREATE TABLE IF NOT EXISTS serving.sales_forecast_release (
    forecast_version   VARCHAR PRIMARY KEY,
    snapshot_date      DATE NOT NULL,
    forecast_week_start DATE NOT NULL,
    forecast_year_week VARCHAR NOT NULL,
    source_manifest    JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_item_rows   BIGINT,
    source_domain_rows INTEGER,
    is_active          BOOLEAN NOT NULL DEFAULT FALSE,
    published_by       VARCHAR,
    published_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    synced_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sales_forecast_release_single_active
    ON serving.sales_forecast_release ((is_active))
    WHERE is_active;

CREATE TABLE IF NOT EXISTS serving.sales_forecast_domain_coverage_current (
    domain               INTEGER PRIMARY KEY,
    forecast_version     VARCHAR NOT NULL REFERENCES serving.sales_forecast_release(forecast_version),
    snapshot_date        DATE NOT NULL,
    coverage_status      VARCHAR NOT NULL CHECK (coverage_status IN ('covered', 'missing_domain_model')),
    forecast_week_start  DATE,
    forecast_year_week   VARCHAR,
    model_config_name_w1 VARCHAR,
    model_config_name_w4 VARCHAR,
    test_rmse_w1         DOUBLE PRECISION,
    test_rmse_w4         DOUBLE PRECISION,
    test_r2_w1           DOUBLE PRECISION,
    test_r2_w4           DOUBLE PRECISION,
    synced_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_forecast_domain_coverage_status
    ON serving.sales_forecast_domain_coverage_current (coverage_status);

CREATE TABLE IF NOT EXISTS serving.item_market_sales_forecast_current (
    domain                              INTEGER NOT NULL,
    asin                                VARCHAR NOT NULL,
    forecast_version                    VARCHAR NOT NULL REFERENCES serving.sales_forecast_release(forecast_version),
    snapshot_date                       DATE NOT NULL,
    forecast_week_start                 DATE NOT NULL,
    forecast_year_week                  VARCHAR NOT NULL,
    predicted_weekly_sales_w1           DOUBLE PRECISION,
    predicted_weekly_sales_w4           DOUBLE PRECISION,
    predicted_growth_ratio_w4_over_w1   DOUBLE PRECISION,
    predicted_growth_delta_w4_minus_w1  DOUBLE PRECISION,
    predicted_rank_w1_within_domain     INTEGER,
    predicted_rank_w4_within_domain     INTEGER,
    model_config_name_w1                VARCHAR,
    model_config_name_w4                VARCHAR,
    test_rmse_w1                        DOUBLE PRECISION,
    test_rmse_w4                        DOUBLE PRECISION,
    test_mae_w1                         DOUBLE PRECISION,
    test_mae_w4                         DOUBLE PRECISION,
    test_mape_w1                        DOUBLE PRECISION,
    test_mape_w4                        DOUBLE PRECISION,
    test_smape_w1                       DOUBLE PRECISION,
    test_smape_w4                       DOUBLE PRECISION,
    test_r2_w1                          DOUBLE PRECISION,
    test_r2_w4                          DOUBLE PRECISION,
    synced_at                           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (domain, asin)
);

CREATE INDEX IF NOT EXISTS idx_item_market_sales_forecast_current_snapshot_date
    ON serving.item_market_sales_forecast_current (snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_item_market_sales_forecast_current_version
    ON serving.item_market_sales_forecast_current (forecast_version);