# XiaMimate Theme API

这个仓库是 XiaMimate 拆分迁移 Phase 3 的 `theme_api` 子项目骨架。

当前状态：

- 创建时间：2026-04-15
- 来源：从旧基线 `/path/to/xiamimate` 复制最小运行集
- 当前用途：正式承接 Theme API 运行
- 默认正式端口：`8100`

当前已迁入内容：

- `data_platform/api/product_theme_api.py`
- `data_platform/api/theme_api_auth.py`
- `data_platform/product_query_assistant.py`
- `data_platform/llm_client.py`
- `scripts/manage_theme_api.sh`
- `scripts/smoke_test_product_theme_api.sh`
- `postgres/migrations/serving/`
- `postgres/init_serving_tables.sql`

边界说明：

1. 本仓拥有 `serving.*` 表和 Theme API 自身的 API key 审计结构。
2. `sync.*` 仍由 collector 负责写入，当前 phase 3 继续复用同一套 PostgreSQL。
3. 当前推荐复用共享运行时根目录 `/path/to/xiamimate-runtime` 的 Python 环境；不要在本仓复制 `.venv`。

推荐启动方式：

1. 复制 `.env.example` 为本地 `.env`。
2. 至少填写：
   - `XIAMIMATE_RUNTIME_ROOT`
   - `XIAMIMATE_PYTHON_BIN`
   - `PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD`
   - `XIAMIMATE_THEME_API_KEY`
3. 启动时会自动把 `XIAMIMATE_THEME_API_KEY` 注册到 `serving.api_keys`；如果 health 里仍显示 `active_key_count=0`，通常说明服务还没按新代码重启。
4. 用正式端口启动：
   - `bash scripts/manage_theme_api.sh start`
5. 跑 smoke test：
   - `bash scripts/smoke_test_product_theme_api.sh`

PostgreSQL DDL：

- `postgres/migrations/serving/` 是当前 serving 层拆分后的 source-of-truth。
- `postgres/init_serving_tables.sql` 是兼容入口，由 `bash postgres/scripts/rebuild_init_serving_tables.sh` 生成。

当前补充说明：

1. 在线查询当前为 PostgreSQL-only；health 不再暴露陈旧 DuckDB `source_db` 字段。
2. 旧仓路径目前仅保留兼容 symlink，正式运行应以 shared runtime 路径为准。
3. `resolve_candidates` 对外仍保持单一工具入口，但内部查询准备已拆为“主题抽取”与“召回归一化”两个阶段，便于后续独立观测和调优 canonical query、aliases、category hints 的生成质量。
