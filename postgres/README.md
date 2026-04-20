# Theme API PostgreSQL DDL

本目录只保留 `theme_api` 自身负责的 `serving.*` migration。

当前结构：

- `migrations/serving/010_serving_theme_feature_tables.sql`
- `migrations/serving/020_serving_theme_api_auth_tables.sql`
- `migrations/serving/030_serving_indexes.sql`
- `init_serving_tables.sql`

说明：

1. `init_serving_tables.sql` 是兼容性入口，适合 phase 3 shadow 初始化时直接执行。
2. 入口 SQL 会先自动执行 `CREATE SCHEMA IF NOT EXISTS serving;`，因此可以直接对空库初始化。
3. 真正的编辑入口是 `migrations/serving/*.sql`。
4. 修改碎片 SQL 后，执行 `bash postgres/scripts/rebuild_init_serving_tables.sh` 重建兼容入口。
