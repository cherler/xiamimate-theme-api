-- serving.* feature tables consumed by theme-api, rebuilt by collector.

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
