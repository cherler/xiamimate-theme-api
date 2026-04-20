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
