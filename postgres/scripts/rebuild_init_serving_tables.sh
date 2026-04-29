#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="$ROOT_DIR/init_serving_tables.sql"
FRAGMENTS=(
  "$ROOT_DIR/migrations/serving/010_serving_theme_feature_tables.sql"
  "$ROOT_DIR/migrations/serving/020_serving_theme_api_auth_tables.sql"
  "$ROOT_DIR/migrations/serving/030_serving_indexes.sql"
  "$ROOT_DIR/migrations/serving/040_serving_sales_forecast_tables.sql"
  "$ROOT_DIR/migrations/serving/041_serving_sales_forecast_explainability.sql"
  "$ROOT_DIR/migrations/serving/050_serving_candidate_pools.sql"
)

for fragment in "${FRAGMENTS[@]}"; do
  if [[ ! -f "$fragment" ]]; then
    echo "missing fragment: $fragment" >&2
    exit 1
  fi
done

{
  printf '%s\n\n' '-- ============================================================'
  printf '%s\n' '-- serving compatibility bootstrap: rebuild from postgres/migrations/serving/*'
  printf '%s\n' '-- do not hand-edit this file; edit fragments then rerun rebuild'
  printf '%s\n' '-- ============================================================'
  printf '\n'

  for fragment in "${FRAGMENTS[@]}"; do
    relative_fragment="${fragment#"$ROOT_DIR/"}"
    printf '%s\n' "-- >>> BEGIN ${relative_fragment}"
    cat "$fragment"
    printf '\n%s\n\n' "-- <<< END ${relative_fragment}"
  done
} > "$OUTPUT_FILE"

echo "rebuilt $OUTPUT_FILE"
