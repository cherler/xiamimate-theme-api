#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/load_theme_api_env.sh
source "$ROOT_DIR/scripts/load_theme_api_env.sh"

PYTHON_BIN="${XIAMIMATE_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$LOG_DIR/theme_api.pid"
LOG_FILE="$LOG_DIR/theme_api.log"
HOST="${THEME_API_HOST:-0.0.0.0}"
PORT="${THEME_API_PORT:-18100}"
APP_ENTRYPOINT="data_platform.api.product_theme.server:app"

mkdir -p "$LOG_DIR"

cleanup_metadata() {
    rm -f "$PID_FILE"
}

wait_for_shutdown() {
    local pid="$1"
    local attempts="${2:-50}"
    local interval_seconds="${3:-0.2}"
    local attempt=0

    while kill -0 "$pid" 2>/dev/null; do
        if (( attempt >= attempts )); then
            return 1
        fi
        sleep "$interval_seconds"
        attempt=$((attempt + 1))
    done

    return 0
}

resolve_pid() {
    local pid

    # 1. PID 文件
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi

    # 2. 兜底：按端口查
    pid="$(lsof -ti:"$PORT" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "$pid" > "$PID_FILE"
        echo "$pid"
        return 0
    fi

    return 1
}

is_running() {
    resolve_pid >/dev/null 2>&1
}

preview_api() {
    echo "python_bin=$PYTHON_BIN"
    echo "host=$HOST"
    echo "port=$PORT"
    echo "log_file=$LOG_FILE"
    echo "command=$PYTHON_BIN -m uvicorn $APP_ENTRYPOINT --host $HOST --port $PORT"
}

start_api() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "python not found: $PYTHON_BIN"
        return 1
    fi

    if is_running; then
        echo "theme API already running: PID $(resolve_pid)  http://${HOST}:${PORT}"
        return 0
    fi

    cleanup_metadata
    nohup "$PYTHON_BIN" -m uvicorn "$APP_ENTRYPOINT" \
        --app-dir "$ROOT_DIR" \
        --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # 等待 uvicorn 启动
    local attempt=0
    while (( attempt < 30 )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "theme API started: PID $(cat "$PID_FILE")  http://${HOST}:${PORT}"
            echo "log file: $LOG_FILE"
            return 0
        fi
        sleep 0.5
        attempt=$((attempt + 1))
    done

    echo "theme API may have failed to start; check log: $LOG_FILE"
    return 1
}

stop_api() {
    if ! is_running; then
        cleanup_metadata
        echo "theme API is not running"
        return 0
    fi

    local pid
    pid="$(resolve_pid)"

    kill "$pid" 2>/dev/null || true
    if ! wait_for_shutdown "$pid"; then
        echo "theme API did not stop gracefully; forcing kill: PID $pid"
        kill -9 "$pid" 2>/dev/null || true
        if ! wait_for_shutdown "$pid" 25 0.2; then
            echo "failed to stop theme API: PID $pid"
            return 1
        fi
    fi

    cleanup_metadata
    echo "theme API stopped: PID $pid"
}

status_api() {
    if is_running; then
        local pid
        pid="$(resolve_pid)"
        echo "theme API running: PID $pid  http://${HOST}:${PORT}"
        echo "log file: $LOG_FILE"
        # 快速健康检查
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "health check: OK"
        else
            echo "health check: FAILED (process alive but /health not responding)"
        fi
    else
        echo "theme API is not running"
        return 1
    fi
}

restart_api() {
    stop_api || true
    start_api
}

show_logs() {
    local lines="${2:-50}"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "log file does not exist yet: $LOG_FILE"
        return 1
    fi
    tail -n "$lines" "$LOG_FILE"
}

case "${1:-}" in
    start)
        start_api
        ;;
    stop)
        stop_api
        ;;
    restart)
        restart_api
        ;;
    status)
        status_api
        ;;
    logs)
        show_logs "$@"
        ;;
    preview)
        preview_api
        ;;
    *)
        cat <<EOF
Usage: bash scripts/manage_theme_api.sh {start|stop|restart|status|logs|preview}

Commands:
    start    启动 API 服务（当前解析为 ${HOST}:${PORT}，可用 THEME_API_HOST / THEME_API_PORT 覆盖）
  stop     停止 API 服务
  restart  重启 API 服务（停旧 + 启新，代码更新后用这个）
  status   查看运行状态 + 健康检查
  logs     查看最近 50 行日志（可选：logs 100）
    preview  仅打印解析后的启动命令与端口

Environment:
  THEME_API_HOST   监听地址，默认 0.0.0.0
    THEME_API_PORT   监听端口，默认 18100
EOF
        exit 1
        ;;
esac
