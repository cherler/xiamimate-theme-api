#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$ROOT_DIR/scripts/load_theme_api_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/load_theme_api_env.sh"
fi

ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

FORECAST_WORKSPACE_ROOT="${FORECAST_WORKSPACE_ROOT:-/Volumes/pytorch-work/xiamimate-forcast-model}"
SERVING_ROOT="${SERVING_ROOT:-$FORECAST_WORKSPACE_ROOT/reports/p0_t1_offline_serving/serving}"
MANIFEST_PATH="${MANIFEST_PATH:-$SERVING_ROOT/run_manifest.json}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$SERVING_ROOT/item_market_latest_predictions.parquet}"
TRAINING_SUMMARY_PATH="${TRAINING_SUMMARY_PATH:-$SERVING_ROOT/domain_training_summary.parquet}"
MIGRATION_SQL="${MIGRATION_SQL:-$ROOT_DIR/postgres/migrations/serving/041_serving_sales_forecast_explainability.sql}"
FORECAST_PYTHON_BIN="${FORECAST_PYTHON_BIN:-$FORECAST_WORKSPACE_ROOT/.venv/bin/python}"
PGHOST_VALUE="${PG_HOST:-localhost}"
PGPORT_VALUE="${PG_PORT:-5432}"
PGDATABASE_VALUE="${PG_DB:-xiamimate}"
PGUSER_VALUE="${PG_USER:-xiamimate}"
PGPASSWORD_VALUE="${PG_PASSWORD:-xiamimate}"
PUBLISHED_BY="${PUBLISHED_BY:-local-explainability-backfill}"
SCHEMA_NAME="${SCHEMA_NAME:-serving}"

if [[ ! -x "$FORECAST_PYTHON_BIN" && -x "$FORECAST_WORKSPACE_ROOT/venv/bin/python" ]]; then
  FORECAST_PYTHON_BIN="$FORECAST_WORKSPACE_ROOT/venv/bin/python"
fi

SYNC_COMMAND=(
  "$FORECAST_PYTHON_BIN"
  -m
  xiamimate_forecast_model.sync_to_rds
  --serving-root
  "$SERVING_ROOT"
  --manifest
  "$MANIFEST_PATH"
  --predictions
  "$PREDICTIONS_PATH"
  --training-summary
  "$TRAINING_SUMMARY_PATH"
  --rds-host
  "$PGHOST_VALUE"
  --rds-port
  "$PGPORT_VALUE"
  --pg-db
  "$PGDATABASE_VALUE"
  --pg-user
  "$PGUSER_VALUE"
  --pg-password
  "$PGPASSWORD_VALUE"
  --schema
  "$SCHEMA_NAME"
  --published-by
  "$PUBLISHED_BY"
)

VERIFY_SQL="SELECT COUNT(*) FILTER (WHERE primary_driver_feature IS NOT NULL), COUNT(*) FILTER (WHERE driver_summary_text IS NOT NULL), COUNT(*), COUNT(DISTINCT domain) FROM ${SCHEMA_NAME}.item_market_sales_forecast_current; SELECT '---'; SELECT forecast_version, is_active, source_item_rows, source_domain_rows FROM ${SCHEMA_NAME}.sales_forecast_release WHERE is_active = TRUE;"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/backfill_local_sales_forecast_explainability.sh [run|preview|verify] [extra sync_to_rds args...]

Actions:
  run      Apply 041 migration, republish local current tables, then verify explainability counts.
  preview  Print resolved paths and the exact commands without executing them.
  verify   Only run the final PostgreSQL verification query.

Environment overrides:
  FORECAST_WORKSPACE_ROOT  Forecast workspace root. Default: /Volumes/pytorch-work/xiamimate-forcast-model
  SERVING_ROOT             Serving output directory. Default: $FORECAST_WORKSPACE_ROOT/reports/p0_t1_offline_serving/serving
  MANIFEST_PATH            Serving manifest path. Default: $SERVING_ROOT/run_manifest.json
  PREDICTIONS_PATH         Predictions parquet path. Default: $SERVING_ROOT/item_market_latest_predictions.parquet
  TRAINING_SUMMARY_PATH    Domain training summary parquet path. Default: $SERVING_ROOT/domain_training_summary.parquet
  FORECAST_PYTHON_BIN      Python executable for sync_to_rds. Default: $FORECAST_WORKSPACE_ROOT/.venv/bin/python
  MIGRATION_SQL            PostgreSQL migration SQL path.
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
  PUBLISHED_BY             Publish marker. Default: local-explainability-backfill
  SCHEMA_NAME              PostgreSQL schema. Default: serving

Examples:
  bash scripts/backfill_local_sales_forecast_explainability.sh preview
  bash scripts/backfill_local_sales_forecast_explainability.sh run
  bash scripts/backfill_local_sales_forecast_explainability.sh run --domain-universe 13
EOF
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

print_config() {
  cat <<EOF
action=$ACTION
forecast_workspace_root=$FORECAST_WORKSPACE_ROOT
forecast_python_bin=$FORECAST_PYTHON_BIN
serving_root=$SERVING_ROOT
manifest_path=$MANIFEST_PATH
predictions_path=$PREDICTIONS_PATH
training_summary_path=$TRAINING_SUMMARY_PATH
migration_sql=$MIGRATION_SQL
pg_host=$PGHOST_VALUE
pg_port=$PGPORT_VALUE
pg_db=$PGDATABASE_VALUE
pg_user=$PGUSER_VALUE
schema_name=$SCHEMA_NAME
published_by=$PUBLISHED_BY
extra_sync_args=$*
EOF
}

validate_inputs() {
  require_dir "$FORECAST_WORKSPACE_ROOT" "forecast workspace root"
  require_dir "$SERVING_ROOT" "serving root"
  require_file "$MIGRATION_SQL" "migration sql"
  require_file "$MANIFEST_PATH" "serving manifest"
  require_file "$PREDICTIONS_PATH" "predictions parquet"
  require_file "$TRAINING_SUMMARY_PATH" "training summary parquet"

  if [[ ! -x "$FORECAST_PYTHON_BIN" ]]; then
    echo "forecast python not found: $FORECAST_PYTHON_BIN" >&2
    exit 1
  fi
}

run_migration() {
  echo "[1/3] apply explainability migration"
  PGPASSWORD="$PGPASSWORD_VALUE" psql \
    -h "$PGHOST_VALUE" \
    -p "$PGPORT_VALUE" \
    -U "$PGUSER_VALUE" \
    -d "$PGDATABASE_VALUE" \
    -f "$MIGRATION_SQL"
}

run_sync() {
  echo "[2/3] republish local forecast current tables"
  (
    cd "$FORECAST_WORKSPACE_ROOT"
    PYTHONPATH=src "${SYNC_COMMAND[@]}" "$@"
  )
}

run_verify() {
  echo "[3/3] verify explainability data in local PostgreSQL"
  PGPASSWORD="$PGPASSWORD_VALUE" psql \
    -h "$PGHOST_VALUE" \
    -p "$PGPORT_VALUE" \
    -U "$PGUSER_VALUE" \
    -d "$PGDATABASE_VALUE" \
    -Atc "$VERIFY_SQL"
}

case "$ACTION" in
  run)
    validate_inputs
    print_config "$@"
    run_migration
    run_sync "$@"
    run_verify
    ;;
  preview)
    validate_inputs
    print_config "$@"
    echo
    echo "migration_command=PGPASSWORD=*** psql -h $PGHOST_VALUE -p $PGPORT_VALUE -U $PGUSER_VALUE -d $PGDATABASE_VALUE -f $MIGRATION_SQL"
    printf 'sync_command=cd %q && PYTHONPATH=src ' "$FORECAST_WORKSPACE_ROOT"
    printf '%q ' "${SYNC_COMMAND[@]}" "$@"
    printf '\n'
    echo "verify_command=PGPASSWORD=*** psql -h $PGHOST_VALUE -p $PGPORT_VALUE -U $PGUSER_VALUE -d $PGDATABASE_VALUE -Atc \"$VERIFY_SQL\""
    ;;
  verify)
    run_verify
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unsupported action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac