#!/bin/bash
# ==============================================================================
# nsys 抓取工具脚本
# 由 nsys-capture skill 调用，提供服务健康检查、文件等待/重命名等工具函数
# ==============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[nsys]${NC} $1"; }
log_success() { echo -e "${GREEN}[nsys]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[nsys]${NC} $1"; }
log_error()   { echo -e "${RED}[nsys]${NC} $1"; }

# ==============================================================================
# 功能：轮询等待服务就绪（HTTP 200）
# 用法：bash nsys_capture.sh wait-service <host> <port> [timeout_minutes]
# ==============================================================================
wait_service() {
    local host="${1:-127.0.0.1}"
    local port="${2:-8080}"
    local timeout_min="${3:-20}"
    local max_seconds=$((timeout_min * 60))
    local elapsed=0
    local interval=15

    log_info "等待服务就绪: http://${host}:${port} (超时 ${timeout_min} 分钟)"

    while [ $elapsed -lt $max_seconds ]; do
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/v1/models" 2>/dev/null)
        if [ "$http_code" = "200" ]; then
            log_success "服务已就绪 (http://${host}:${port})"
            return 0
        fi

        # 检查日志是否有致命错误（只看最新 50 行，避免旧日志干扰）
        if [ -f /tmp/nsys_serve.log ]; then
            local recent_log
            recent_log=$(tail -50 /tmp/nsys_serve.log 2>/dev/null)
            if echo "$recent_log" | grep -qE "Traceback|AssertionError|OOM|Killed|CUDA out of memory"; then
                log_error "检测到致命错误，停止等待。最后日志："
                tail -30 /tmp/nsys_serve.log
                return 1
            fi
        fi

        log_info "  HTTP=${http_code}，已等待 ${elapsed}s，继续..."
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "等待超时（${timeout_min} 分钟），服务未就绪"
    tail -30 /tmp/nsys_serve.log 2>/dev/null
    return 1
}

# ==============================================================================
# 功能：等待 nsys 文件生成并大小稳定，然后重命名
# 用法：bash nsys_capture.sh wait-file <output_dir> <type> <level>
#   type:  text / mm / custom
#   level: low / high
# ==============================================================================
wait_and_rename_file() {
    local output_dir="${1:-/tmp/nsys_record}"
    local type="${2:-text}"
    local level="${3:-low}"
    local max_wait=120
    local waited=0

    # 读取服务启动时写入的 nsys 输出路径
    if [ ! -f /tmp/nsys_capture_output_path ]; then
        log_error "未找到 /tmp/nsys_capture_output_path，无法定位 nsys 文件"
        return 1
    fi

    local nsys_output_base
    nsys_output_base=$(cat /tmp/nsys_capture_output_path)
    local expected_file="${nsys_output_base}.nsys-rep"

    log_info "等待 nsys 文件生成: ${expected_file}"

    while [ $waited -lt $max_wait ]; do
        if [ -f "$expected_file" ]; then
            local size1 size2
            size1=$(stat -c%s "$expected_file" 2>/dev/null || echo "0")
            sleep 3
            size2=$(stat -c%s "$expected_file" 2>/dev/null || echo "0")

            if [ "$size1" = "$size2" ] && [ "$size1" -gt 0 ]; then
                local timestamp
                timestamp=$(date +%Y%m%d_%H%M%S)
                mkdir -p "$output_dir"
                local final_file="${output_dir}/${timestamp}_nsys_${type}_${level}.nsys-rep"
                mv "$expected_file" "$final_file"
                local file_size
                file_size=$(du -h "$final_file" | cut -f1)
                log_success "====================================="
                log_success "nsys 抓取完成！"
                log_success "文件: ${final_file}"
                log_success "大小: ${file_size}"
                log_success "====================================="
                return 0
            fi

            log_info "  文件大小 ${size1} → ${size2}，仍在写入..."
        else
            log_info "  文件尚未出现，已等待 ${waited}s..."
        fi

        sleep 5
        waited=$((waited + 5))
    done

    log_error "等待 nsys 文件超时（${max_wait}s）: ${expected_file}"
    return 1
}

# ==============================================================================
# 主函数
# ==============================================================================
main() {
    local command="${1:-help}"

    case "$command" in
        wait-service)
            wait_service "$2" "$3" "$4"
            ;;
        wait-file)
            wait_and_rename_file "$2" "$3" "$4"
            ;;
        help|--help|-h|*)
            cat << EOF
nsys 抓取工具脚本

用法:
  bash nsys_capture.sh wait-service <host> <port> [timeout_minutes]
      轮询等待服务就绪（HTTP 200），超时默认 20 分钟

  bash nsys_capture.sh wait-file <output_dir> <type> <level>
      等待 nsys 文件生成并大小稳定，重命名到 output_dir
      type:  text / mm / custom
      level: low / high

环境变量:
  无。所有参数通过命令行传入。

依赖:
  /tmp/nsys_capture_output_path  由 start_nsys.sh 写入的 nsys 输出路径
  /tmp/nsys_serve.log            服务启动日志
EOF
            ;;
    esac
}

main "$@"
