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

