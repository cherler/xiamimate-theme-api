#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/load_theme_api_env.sh
source "$PROJECT_ROOT/scripts/load_theme_api_env.sh"
ROOT_ENV_FILE="$PROJECT_ROOT/.env"
BASELINE_ENV_FILE=""
PROJECT_PYTHON="${XIAMIMATE_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ -n "${XIAMIMATE_BASELINE_ROOT:-}" ]]; then
  BASELINE_ENV_FILE="$XIAMIMATE_BASELINE_ROOT/.env"
fi

load_api_key_from_env_file() {
  local env_file="$1"
  "$PROJECT_PYTHON" - "$env_file" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
for raw_line in env_path.read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('export '):
        line = line[len('export '):].strip()
    if '=' not in line:
        continue
    key, value = line.split('=', 1)
    if key.strip() != 'XIAMIMATE_THEME_API_KEY':
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    print(value)
    break
PY
}

if [[ -z "${XIAMIMATE_THEME_API_KEY:-}" ]]; then
  for env_file in "$ROOT_ENV_FILE" "$BASELINE_ENV_FILE"; do
    if [[ -n "$env_file" && -f "$env_file" ]]; then
      XIAMIMATE_THEME_API_KEY="$(load_api_key_from_env_file "$env_file")"
      if [[ -n "${XIAMIMATE_THEME_API_KEY:-}" ]]; then
        export XIAMIMATE_THEME_API_KEY
        break
      fi
    fi
  done
fi

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:${THEME_API_PORT:-18100}}"
API_KEY="${XIAMIMATE_THEME_API_KEY:-}"
PRODUCT_QUERY="${PRODUCT_QUERY:-handheld shower head}"
MARKETPLACE="${MARKETPLACE:-US}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"
TOP_N="${TOP_N:-3}"
MAX_CANDIDATES="${MAX_CANDIDATES:-5}"
RESPONSE_SCHEMA="xiamimate_theme_api_v1"

if [[ -z "$API_KEY" ]]; then
  echo "XIAMIMATE_THEME_API_KEY is required (set it in the shell, $ROOT_ENV_FILE, or the baseline repo .env)"
  exit 1
fi

if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "python not found: $PROJECT_PYTHON"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl_json() {
  local method="$1"
  local url="$2"
  local payload_file="$3"
  local output_file="$4"

  if [[ -n "$payload_file" ]]; then
    curl -sS -X "$method" "$url" \
      -H 'Content-Type: application/json' \
      -H "X-API-Key: $API_KEY" \
      --data @"$payload_file" \
      > "$output_file"
  else
    curl -sS -X "$method" "$url" > "$output_file"
  fi
}

pretty_print() {
  local label="$1"
  local json_file="$2"
  echo "===== $label ====="
  "$PROJECT_PYTHON" -m json.tool "$json_file"
  echo
}

validate_response() {
  local label="$1"
  local json_file="$2"
  local endpoint="$3"

  "$PROJECT_PYTHON" - "$label" "$json_file" "$endpoint" "$RESPONSE_SCHEMA" <<'PY'
import json
import sys

label, path, endpoint, schema = sys.argv[1:5]
with open(path, 'r', encoding='utf-8') as handle:
  payload = json.load(handle)

if payload.get('success') is not True:
  raise SystemExit(f'{label}: success=false -> {payload}')
if payload.get('code') != 'OK':
  raise SystemExit(f'{label}: unexpected code -> {payload.get("code")}')

meta = payload.get('meta') or {}
if meta.get('endpoint') != endpoint:
  raise SystemExit(f'{label}: unexpected endpoint -> {meta.get("endpoint")}')
if meta.get('response_schema') != schema:
  raise SystemExit(f'{label}: unexpected response_schema -> {meta.get("response_schema")}')
PY
}

cat > "$TMP_DIR/resolve.json" <<EOF
{
  "product_query": "$PRODUCT_QUERY",
  "marketplace": "$MARKETPLACE",
  "max_candidates": $MAX_CANDIDATES,
  "active_only": true
}
EOF

curl_json GET "$API_BASE_URL/health" "" "$TMP_DIR/health.out.json"
curl_json POST "$API_BASE_URL/api/product-theme/resolve-candidates" "$TMP_DIR/resolve.json" "$TMP_DIR/resolve.out.json"

CANDIDATE_ASINS="$($PROJECT_PYTHON - "$TMP_DIR/resolve.out.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

items = payload.get('data', {}).get('candidate_asins', [])[:5]
print(json.dumps(items, ensure_ascii=False))
PY
)"

if [[ "$CANDIDATE_ASINS" == "[]" ]]; then
  echo "resolve-candidates returned no candidate_asins"
  pretty_print "health" "$TMP_DIR/health.out.json"
  pretty_print "resolve-candidates" "$TMP_DIR/resolve.out.json"
  exit 1
fi

cat > "$TMP_DIR/stats.json" <<EOF
{
  "candidate_asins": $CANDIDATE_ASINS,
  "marketplace": "$MARKETPLACE",
  "window_days": $WINDOW_DAYS
}
EOF

cat > "$TMP_DIR/forecast.json" <<EOF
{
  "candidate_asins": $CANDIDATE_ASINS,
  "marketplace": "$MARKETPLACE",
  "window_days": $WINDOW_DAYS,
  "top_n": $TOP_N
}
EOF

curl_json POST "$API_BASE_URL/api/product-theme/candidate-pool-stats" "$TMP_DIR/stats.json" "$TMP_DIR/stats.out.json"
curl_json POST "$API_BASE_URL/api/product-theme/candidate-pool-trends" "$TMP_DIR/stats.json" "$TMP_DIR/trends.out.json"
curl_json POST "$API_BASE_URL/api/product-theme/candidate-pool-weak-forecast" "$TMP_DIR/forecast.json" "$TMP_DIR/forecast.out.json"
curl_json POST "$API_BASE_URL/api/product-theme/category-benchmark" "$TMP_DIR/stats.json" "$TMP_DIR/benchmark.out.json"

TOP_ASINS="$($PROJECT_PYTHON - "$TMP_DIR/forecast.out.json" "$TOP_N" "$CANDIDATE_ASINS" <<'PY'
import json
import sys

path = sys.argv[1]
top_n = int(sys.argv[2])
fallback = json.loads(sys.argv[3])
with open(path, 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

predicted = payload.get('data', {}).get('predicted_top_asins', [])
asins = [item.get('asin') for item in predicted if item.get('asin')][:top_n]
if not asins:
    asins = fallback[:top_n]
print(json.dumps(asins, ensure_ascii=False))
PY
)"

cat > "$TMP_DIR/drilldown.json" <<EOF
{
  "candidate_asins": $TOP_ASINS,
  "marketplace": "$MARKETPLACE",
  "window_days": $WINDOW_DAYS,
  "top_n": $TOP_N
}
EOF

curl_json POST "$API_BASE_URL/api/product-theme/top-asin-drilldown" "$TMP_DIR/drilldown.json" "$TMP_DIR/drilldown.out.json"

validate_response "health" "$TMP_DIR/health.out.json" "/health"
validate_response "resolve-candidates" "$TMP_DIR/resolve.out.json" "/api/product-theme/resolve-candidates"
validate_response "candidate-pool-stats" "$TMP_DIR/stats.out.json" "/api/product-theme/candidate-pool-stats"
validate_response "candidate-pool-trends" "$TMP_DIR/trends.out.json" "/api/product-theme/candidate-pool-trends"
validate_response "candidate-pool-weak-forecast" "$TMP_DIR/forecast.out.json" "/api/product-theme/candidate-pool-weak-forecast"
validate_response "category-benchmark" "$TMP_DIR/benchmark.out.json" "/api/product-theme/category-benchmark"
validate_response "top-asin-drilldown" "$TMP_DIR/drilldown.out.json" "/api/product-theme/top-asin-drilldown"

pretty_print "health" "$TMP_DIR/health.out.json"
pretty_print "resolve-candidates" "$TMP_DIR/resolve.out.json"
pretty_print "candidate-pool-stats" "$TMP_DIR/stats.out.json"
pretty_print "candidate-pool-trends" "$TMP_DIR/trends.out.json"
pretty_print "candidate-pool-weak-forecast" "$TMP_DIR/forecast.out.json"
pretty_print "category-benchmark" "$TMP_DIR/benchmark.out.json"
pretty_print "top-asin-drilldown" "$TMP_DIR/drilldown.out.json"
