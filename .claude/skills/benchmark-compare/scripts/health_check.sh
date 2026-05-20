#!/usr/bin/env bash
# health_check.sh — 服务健康检查脚本
# 轮询 /v1/models 接口直到服务就绪或超时
set -euo pipefail

HOST="127.0.0.1"
PORT=""
TIMEOUT=300
INTERVAL=30
INITIAL_WAIT=90
LOG_FILE=""

usage() {
    cat <<'EOF'
用法: bash health_check.sh [OPTIONS]

必需参数:
  --port <PORT>               服务端口

可选参数:
  --host <HOST>               服务地址 (默认: 127.0.0.1)
  --timeout <SECONDS>         超时时间 (默认: 300)
  --interval <SECONDS>        轮询间隔 (默认: 30)
  --initial-wait <SECONDS>    初始等待时间，让模型加载 (默认: 90)
  --log-file <PATH>           服务日志路径，超时时打印最后几行辅助排查

返回值:
  0 = 服务就绪
  1 = 超时未就绪

示例:
  bash health_check.sh --port 8180 --timeout 300 --log-file /tmp/fd_server.log
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)          HOST="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --interval)      INTERVAL="$2"; shift 2 ;;
        --initial-wait)  INITIAL_WAIT="$2"; shift 2 ;;
        --log-file)      LOG_FILE="$2"; shift 2 ;;
        --help|-h)       usage 0 ;;
        *)               echo "未知参数: $1"; usage 1 ;;
    esac
done

if [[ -z "$PORT" ]]; then
    echo "错误: --port 为必需参数"
    usage 1
fi

URL="http://${HOST}:${PORT}/v1/models"

echo "[INFO] 等待服务就绪: $URL"
echo "[INFO] 初始等待 ${INITIAL_WAIT}s (模型加载时间)..."
sleep "$INITIAL_WAIT"

elapsed=$INITIAL_WAIT
while [[ $elapsed -lt $TIMEOUT ]]; do
    if curl -s --max-time 5 "$URL" | grep -q '"id"'; then
        echo "[INFO] 服务已就绪! (耗时 ${elapsed}s)"
        exit 0
    fi
    echo "[INFO] 服务未就绪，${INTERVAL}s 后重试... (已等待 ${elapsed}s / ${TIMEOUT}s)"
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done

echo "[ERROR] 服务启动超时 (${TIMEOUT}s)"

if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    echo "[ERROR] 服务日志最后 30 行:"
    echo "========================================="
    tail -30 "$LOG_FILE"
    echo "========================================="
fi

exit 1
