-- ============================================================

-- serving compatibility bootstrap: rebuild from postgres/migrations/serving/*
-- do not hand-edit this file; edit fragments then rerun rebuild
-- ============================================================

-- >>> BEGIN migrations/serving/010_serving_theme_feature_tables.sql
-- serving.* feature tables consumed by theme-api, rebuilt by collector.

CREATE SCHEMA IF NOT EXISTS serving;

CREATE TABLE IF NOT EXISTS serving.theme_base_daily (
    domain                INTEGER NOT NULL,
    asin                  VARCHAR NOT NULL,
    date                  DATE NOT NULL,
    product_title         VARCHAR,
    brand                 VARCHAR,
    category              VARCHAR,
    effective_price       DOUBLE PRECISION,
    rating                DOUBLE PRECISION,
    review_count          BIGINT,
    new_offer_count       INTEGER,
    used_offer_count      INTEGER,
    bsr                   BIGINT,
    estimated_daily_sales DOUBLE PRECISION,
    synced_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (domain, asin, date)
);

CREATE TABLE IF NOT EXISTS serving.theme_trends_daily (
    domain                       INTEGER NOT NULL,
    asin                         VARCHAR NOT NULL,
    date                         DATE NOT NULL,
    trend_index_mean             DOUBLE PRECISION,
    trend_index_wow              DOUBLE PRECISION,
    trend_index_dod              DOUBLE PRECISION,
    trend_index_roll_std_7       DOUBLE PRECISION,
    trend_index_roll_max_7       DOUBLE PRECISION,
    trend_keyword_coverage_ratio DOUBLE PRECISION,
    synced_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (domain, asin, date)
);

CREATE TABLE IF NOT EXISTS serving.theme_cross_daily (
    domain                INTEGER NOT NULL,
    asin                  VARCHAR NOT NULL,
    date                  DATE NOT NULL,
    product_title         VARCHAR,
    effective_price       DOUBLE PRECISION,
    bsr                   BIGINT,
    rating                DOUBLE PRECISION,
    review_count          BIGINT,
    new_offer_count       INTEGER,
    used_offer_count      INTEGER,
    estimated_daily_sales DOUBLE PRECISION,
    trend_index_mean      DOUBLE PRECISION,
    price_discount_pct    DOUBLE PRECISION,
    synced_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (domain, asin, date)
);

-- <<< END migrations/serving/010_serving_theme_feature_tables.sql

-- >>> BEGIN migrations/serving/020_serving_theme_api_auth_tables.sql
-- serving.* auth/audit tables owned by theme-api.

CREATE TABLE IF NOT EXISTS serving.api_keys (
    key_id          VARCHAR PRIMARY KEY,
    name            VARCHAR NOT NULL,
    tier            VARCHAR NOT NULL DEFAULT 'standard',
    key_prefix      VARCHAR NOT NULL,
    key_hash        VARCHAR NOT NULL UNIQUE,
    key_raw         VARCHAR,
    status          VARCHAR NOT NULL DEFAULT 'active',
    daily_quota     INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ,
    last_used_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS serving.api_usage (
    id              BIGSERIAL PRIMARY KEY,
    key_id          VARCHAR NOT NULL REFERENCES serving.api_keys(key_id),
    endpoint        VARCHAR NOT NULL,
    usage_date      DATE NOT NULL,
    status_code     INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_count   INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- <<< END migrations/serving/020_serving_theme_api_auth_tables.sql

-- >>> BEGIN migrations/serving/030_serving_indexes.sql
-- serving.* indexes aligned with theme-api query paths and auth audit lookups.

CREATE INDEX IF NOT EXISTS idx_theme_base_daily_domain_date ON serving.theme_base_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_theme_trends_daily_domain_date ON serving.theme_trends_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_theme_cross_daily_domain_date ON serving.theme_cross_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_key_date ON serving.api_usage(key_id, usage_date);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON serving.api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON serving.api_usage(endpoint, usage_date);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON serving.api_keys(status);

-- <<< END migrations/serving/030_serving_indexes.sql

-- >>> BEGIN migrations/serving/040_serving_sales_forecast_tables.sql
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
-- <<< END migrations/serving/040_serving_sales_forecast_tables.sql

-- >>> BEGIN migrations/serving/041_serving_sales_forecast_explainability.sql
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

-- <<< END migrations/serving/041_serving_sales_forecast_explainability.sql

