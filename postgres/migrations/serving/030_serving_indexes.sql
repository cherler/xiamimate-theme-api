-- serving.* indexes aligned with theme-api query paths and auth audit lookups.

CREATE INDEX IF NOT EXISTS idx_theme_base_daily_domain_date ON serving.theme_base_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_theme_trends_daily_domain_date ON serving.theme_trends_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_theme_cross_daily_domain_date ON serving.theme_cross_daily(domain, date DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_key_date ON serving.api_usage(key_id, usage_date);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON serving.api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON serving.api_usage(endpoint, usage_date);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON serving.api_keys(status);
