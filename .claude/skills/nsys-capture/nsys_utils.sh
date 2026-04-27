#!/bin/bash
# ==============================================================================
# nsys_utils.sh
# 保留：timeit 单条请求耗时测试
# nsys 抓取主流程已迁移到 SKILL.md + nsys_capture.sh
# ==============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# 功能：测试单条流式文本请求耗时
# 用法：bash nsys_utils.sh timeit <host> <port> [client_script]
# ==============================================================================
timeit_request() {
    local host="${1:-127.0.0.1}"
    local port="${2:-8080}"
    local client_script="${3:-$(dirname "$0")/nsys_default_client.py}"

    if [ ! -f "$client_script" ]; then
        log_error "请求脚本不存在: $client_script"
        exit 1
    fi

    # 先检查服务是否就绪
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/v1/models" 2>/dev/null)
    if [ "$http_code" != "200" ]; then
        log_error "服务不可用 http://${host}:${port} (HTTP ${http_code})"
        exit 1
    fi

    log_info "=========================================="
    log_info "测试单条请求耗时"
    log_info "服务: http://${host}:${port}"
    log_info "脚本: ${client_script}"
    log_info "=========================================="

    { time python3 "$client_script" "$host" "$port"; } 2>&1 | tee /tmp/nsys_timeit_result.txt

    local real_time user_time sys_time
    real_time=$(grep "^real" /tmp/nsys_timeit_result.txt | awk '{print $2}')
    user_time=$(grep "^user" /tmp/nsys_timeit_result.txt | awk '{print $2}')
    sys_time=$(grep  "^sys"  /tmp/nsys_timeit_result.txt | awk '{print $2}')

    echo ""
    log_success "=========================================="
    log_success "单条请求耗时"
    log_success "=========================================="
    echo -e "  ${GREEN}real${NC}: $real_time"
    echo -e "  ${GREEN}user${NC}: $user_time"
    echo -e "  ${GREEN}sys ${NC}: $sys_time"
}

# ==============================================================================
show_help() {
    cat << EOF
用法:
  bash $0 timeit <host> <port> [client_script]
      测试单条流式请求的端到端耗时（real/user/sys）

参数:
  host           服务 IP（默认 127.0.0.1）
  port           服务端口（默认 8080）
  client_script  请求脚本路径（默认使用 nsys_default_client.py）

注意：nsys 抓取主流程请参考 SKILL.md 和 nsys_capture.sh
EOF
}

main() {
    case "${1:-help}" in
        timeit) timeit_request "$2" "$3" "$4" ;;
        help|--help|-h|*) show_help ;;
    esac
}

main "$@"
