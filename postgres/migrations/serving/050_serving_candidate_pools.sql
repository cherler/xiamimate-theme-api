-- serving.* candidate pool state created by theme-api resolve_candidates.

CREATE TABLE IF NOT EXISTS serving.candidate_pools (
    pool_id                         UUID PRIMARY KEY,
    version                         INTEGER NOT NULL DEFAULT 1,
    parent_pool_id                  UUID REFERENCES serving.candidate_pools(pool_id),
    domain                          INTEGER NOT NULL,
    marketplace                     VARCHAR NOT NULL,
    product_query                   TEXT NOT NULL,
    normalized_query                TEXT,
    recall_mode                     VARCHAR NOT NULL DEFAULT 'keyword',
    category_id                     BIGINT,
    category_path                   TEXT,
    include_descendants             BOOLEAN NOT NULL DEFAULT TRUE,
    filters                         JSONB NOT NULL DEFAULT '{}'::jsonb,
    ranking_version                 VARCHAR NOT NULL DEFAULT 'semantic_recall_v2',
    pool_quality                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidate_count                 INTEGER NOT NULL DEFAULT 0,
    candidate_total_before_truncate INTEGER NOT NULL DEFAULT 0,
    source                          VARCHAR NOT NULL DEFAULT 'resolve_candidates',
    lineage                         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS serving.candidate_pool_items (
    pool_id        UUID NOT NULL REFERENCES serving.candidate_pools(pool_id) ON DELETE CASCADE,
    asin           VARCHAR NOT NULL,
    domain         INTEGER NOT NULL,
    marketplace    VARCHAR NOT NULL,
    candidate_rank INTEGER NOT NULL,
    match_score    DOUBLE PRECISION,
    match_reasons  JSONB NOT NULL DEFAULT '[]'::jsonb,
    item_snapshot  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (pool_id, asin)
);

CREATE INDEX IF NOT EXISTS idx_candidate_pools_created_at
    ON serving.candidate_pools (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_candidate_pools_domain_query
    ON serving.candidate_pools (domain, normalized_query, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_candidate_pool_items_rank
    ON serving.candidate_pool_items (pool_id, candidate_rank ASC);

CREATE INDEX IF NOT EXISTS idx_candidate_pool_items_domain_asin
    ON serving.candidate_pool_items (domain, asin);
