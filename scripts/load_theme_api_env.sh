#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THEME_API_ENV_FILE="${XIAMIMATE_THEME_API_ENV_FILE:-$ROOT_DIR/.env}"

THEME_API_ENV_OVERRIDE_NAMES=(
    XIAMIMATE_PYTHON_BIN
    PG_HOST
    PG_PORT
    PG_DB
    PG_USER
    PG_PASSWORD
    PGPASSWORD
    THEME_API_HOST
    THEME_API_PORT
)
THEME_API_ENV_OVERRIDE_PRESENT=()
THEME_API_ENV_OVERRIDE_VALUES=()

for theme_api_env_var_name in "${THEME_API_ENV_OVERRIDE_NAMES[@]}"; do
    if [[ -n "${!theme_api_env_var_name+x}" ]]; then
        THEME_API_ENV_OVERRIDE_PRESENT+=(1)
        THEME_API_ENV_OVERRIDE_VALUES+=("${!theme_api_env_var_name}")
    else
        THEME_API_ENV_OVERRIDE_PRESENT+=(0)
        THEME_API_ENV_OVERRIDE_VALUES+=("")
    fi
done

set_default_if_missing() {
    local var_name="$1"
    local candidate="$2"

    if [[ -n "${!var_name:-}" ]]; then
        return 0
    fi
    if [[ -z "$candidate" ]]; then
        return 0
    fi

    printf -v "$var_name" '%s' "$candidate"
    export "$var_name"
}

if [[ -f "$THEME_API_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$THEME_API_ENV_FILE"
    set +a
fi

for theme_api_env_var_index in "${!THEME_API_ENV_OVERRIDE_NAMES[@]}"; do
    if [[ "${THEME_API_ENV_OVERRIDE_PRESENT[$theme_api_env_var_index]}" == "1" ]]; then
        printf -v "${THEME_API_ENV_OVERRIDE_NAMES[$theme_api_env_var_index]}" '%s' "${THEME_API_ENV_OVERRIDE_VALUES[$theme_api_env_var_index]}"
        export "${THEME_API_ENV_OVERRIDE_NAMES[$theme_api_env_var_index]}"
    fi
done

unset theme_api_env_var_index theme_api_env_var_name
unset THEME_API_ENV_OVERRIDE_NAMES THEME_API_ENV_OVERRIDE_PRESENT THEME_API_ENV_OVERRIDE_VALUES

if [[ -z "${PG_PASSWORD:-}" && -n "${PGPASSWORD:-}" ]]; then
    PG_PASSWORD="$PGPASSWORD"
fi

if [[ -z "${PGPASSWORD:-}" && -n "${PG_PASSWORD:-}" ]]; then
    PGPASSWORD="$PG_PASSWORD"
fi

if [[ -z "${XIAMIMATE_RUNTIME_ROOT:-}" ]]; then
    default_runtime_root="$(cd "$ROOT_DIR/../xiamimate-runtime" 2>/dev/null && pwd || true)"
    if [[ -n "$default_runtime_root" && -d "$default_runtime_root" ]]; then
        XIAMIMATE_RUNTIME_ROOT="$default_runtime_root"
    fi
fi

if [[ -n "${XIAMIMATE_RUNTIME_ROOT:-}" ]]; then
    set_default_if_missing "XIAMIMATE_PYTHON_BIN" "$XIAMIMATE_RUNTIME_ROOT/python/.venv/bin/python"
fi

if [[ -z "${XIAMIMATE_BASELINE_ROOT:-}" ]]; then
    default_baseline_root="$(cd "$ROOT_DIR/../xiamimate" 2>/dev/null && pwd || true)"
    if [[ -n "$default_baseline_root" && -d "$default_baseline_root" ]]; then
        XIAMIMATE_BASELINE_ROOT="$default_baseline_root"
    fi
fi

if [[ -n "${XIAMIMATE_BASELINE_ROOT:-}" ]]; then
    set_default_if_missing "XIAMIMATE_PYTHON_BIN" "$XIAMIMATE_BASELINE_ROOT/.venv/bin/python"
fi

export XIAMIMATE_RUNTIME_ROOT
export XIAMIMATE_BASELINE_ROOT
export PG_HOST
export PG_PORT
export PG_DB
export PG_USER
export PG_PASSWORD
export PGPASSWORD
