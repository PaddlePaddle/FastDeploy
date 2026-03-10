#!/opt/homebrew/bin/bash
#
# FD Test Runner - 静默执行，终端只显示统计
# Usage: ./run.sh --host <ip> --port <port> --cases <类别或文件> [--unexclude <排除的类别或文件>]
#

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CASE_DIR="${SCRIPT_DIR}"
readonly LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "$LOG_DIR"

readonly REPORT_FILE="${LOG_DIR}/report_$(date +%Y%m%d_%H%M%S).md"
readonly DETAIL_LOG="${LOG_DIR}/detail_$(date +%Y%m%d_%H%M%S).log"
readonly RAW_LOG="${LOG_DIR}/raw_$(date +%Y%m%d_%H%M%S).log"

# 超时设置（秒）
readonly PYTEST_TIMEOUT=3000  # 5分钟
readonly SCRIPT_TIMEOUT=3600 # 1小时总超时

# ============ 类别定义 ============

declare -A CASE_CATEGORIES=(
    ["smoke"]="test_chat.py, test_completions.py"
    ["base"]="test_fd_effect.py, test_fd_parameters_chat.py, test_fd_parameters_comp.py"
    ["vl"]="test_chat_image.py,test_chat_video.py"
    ["default_function"]="test_logprobs.py"
    ["special_function"]="test_prompt_logprobs.py"
    ["full"]="all"
)

HOST=""
PORT=""
CASES=""
EXCLUDE=""

# 清理函数
cleanup() {
    echo ""
    echo "正在清理..."
    [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null
    exit 130
}

trap cleanup INT TERM

usage() {
    cat << 'EOF_USAGE'
Usage: $0 --host <ip> --port <port> --cases <类别或文件> [--unexclude <排除项>]

预定义类别：
    smoke                 基于open ai基础case
    base                  fd基础case(fd特有字段，模型服务基础能力)
    vl                    多模case
    default_function      FD默认能力case
    special_function      FD非默认开启的能力case
    full                  全部文件

Example:
    ./run.sh -h 127.0.0.1 -p 8000 -c smoke
    ./run.sh -h 10.0.0.5 -p 8080 -c full -un smoke
    ./run.sh -h localhost -p 8000 -c all -un "test_chat.py,test_completions.py"
    ./run.sh -h 127.0.0.1 -p 8000 -c base,vl -un test_fd_field.py
EOF_USAGE
    exit 1
}

preflight_check() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "错误: 未找到 python3"
        exit 1
    fi

    if ! python3 -m pytest --version >/dev/null 2>&1; then
        echo "错误: pytest 未安装"
        exit 1
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--host) HOST="$2"; shift 2 ;;
            -p|--port) PORT="$2"; shift 2 ;;
            -c|--cases) CASES="$2"; shift 2 ;;
            -un|--unexclude) EXCLUDE="$2"; shift 2 ;;
            --help) usage ;;
            *) echo "未知参数: $1"; usage ;;
        esac
    done

    [[ -z "$HOST" ]] && { echo "错误: 缺少 --host"; usage; }
    [[ -z "$PORT" ]] && { echo "错误: 缺少 --port"; usage; }
    [[ -z "$CASES" ]] && { echo "错误: 缺少 --cases"; usage; }
}

resolve_to_files() {
    local input="$1"
    local result_files=()
    local IFS=','

    for item in $input; do
        item=$(echo "$item" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$item" ]] && continue

        if [[ -v CASE_CATEGORIES[$item] ]] && [[ -n "${CASE_CATEGORIES[$item]}" ]]; then
            local category_content="${CASE_CATEGORIES[$item]}"
            if [[ "$category_content" == "all" ]]; then
                for f in "${CASE_DIR}"/test_*.py; do
                    [[ -f "$f" ]] && result_files+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
                done
            else
                local IFS=','
                for name in $category_content; do
                    name=$(echo "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                    [[ -z "$name" ]] && continue
                    local file="${CASE_DIR}/${name}"
                    [[ -f "$file" ]] && result_files+=("$(cd "$(dirname "$file")" && pwd)/$(basename "$file")")
                done
            fi
        else
            local file="${CASE_DIR}/${item}"
            [[ -f "$file" ]] && result_files+=("$(cd "$(dirname "$file")" && pwd)/$(basename "$file")")
        fi
    done

    printf "%s\n" "${result_files[@]}" | sort -u
}

resolve_cases() {
    local cases_input="$1"
    local exclude_input="$2"

    local all_files=()
    while IFS= read -r file; do
        [[ -n "$file" ]] && all_files+=("$file")
    done < <(resolve_to_files "$cases_input")

    [[ ${#all_files[@]} -eq 0 ]] && return 0

    if [[ -n "$exclude_input" ]]; then
        local exclude_files=()
        while IFS= read -r file; do
            [[ -n "$file" ]] && exclude_files+=("$file")
        done < <(resolve_to_files "$exclude_input")

        local filtered_files=()
        for file in "${all_files[@]}"; do
            local should_exclude=false
            for ex_file in "${exclude_files[@]}"; do
                [[ "$file" == "$ex_file" ]] && { should_exclude=true; break; }
            done
            [[ "$should_exclude" == false ]] && filtered_files+=("$file")
        done
        printf "%s\n" "${filtered_files[@]}"
    else
        printf "%s\n" "${all_files[@]}"
    fi
}

# ============ 使用 Python 实现跨平台超时执行 ============
run_pytest_with_timeout() {
    local py_file="$1"
    local output_file="$2"
    local timeout_sec="$3"

    # 使用 Python 执行 pytest 并控制超时
    python3 << EOF
import subprocess
import sys
import os

env = os.environ.copy()
env['URL_HOST'] = '${HOST}'
env['URL_PORT'] = '${PORT}'

try:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-s','${py_file}', '-v', '--tb=short'],
        capture_output=True,
        text=True,
        timeout=${timeout_sec},
        env=env
    )
    with open('${output_file}', 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    with open('${output_file}', 'w') as f:
        f.write(f"TIMEOUT: Execution exceeded ${timeout_sec} seconds\\n")
    sys.exit(124)
except Exception as e:
    with open('${output_file}', 'w') as f:
        f.write(f"ERROR: {str(e)}\\n")
    sys.exit(1)
EOF
    return $?
}

# ============ 执行单个 case ============
run_case() {
    local py_file="$1"
    local case_name
    case_name=$(basename "$py_file")
    local result_file
    result_file=$(mktemp)

    printf "执行: %s ... " "$case_name" >&2

    local temp_output
    temp_output=$(mktemp)

    local exit_code=0
    local timed_out=false

    # 使用 Python 实现超时控制
    run_pytest_with_timeout "$py_file" "$temp_output" "$PYTEST_TIMEOUT"
    exit_code=$?

    # 检查是否超时
    if [[ $exit_code -eq 124 ]] || grep -q "^TIMEOUT:" "$temp_output" 2>/dev/null; then
        timed_out=true
        echo "TIMEOUT" >&2
    fi

    # 追加到原始日志
    {
        echo "=== $case_name ==="
        cat "$temp_output"
        echo ""
    } >> "$RAW_LOG"

    # 如果超时，返回超时结果
    if [[ "$timed_out" == true ]]; then
        {
            echo ">>> 执行: $case_name"
            echo "Case统计: collected=0, passed=0, failed=0, error=1, skipped=0"
            echo "ERROR: $case_name (执行超时，超过 ${PYTEST_TIMEOUT} 秒)"
            echo "---"
        } >> "$DETAIL_LOG"

        echo "0|0|0|1|0|124|" > "$result_file"
        echo "$result_file"
        rm -f "$temp_output"
        return
    fi

    # 解析统计
    local collected=0 passed=0 failed=0 error=0 skipped=0

    if grep -q "collected" "$temp_output" 2>/dev/null; then
        local col_line
        col_line=$(grep "collected" "$temp_output" | head -1)
        collected=$(echo "$col_line" | grep -oE '[0-9]+' | head -1)
        [[ -z "$collected" ]] && collected=0
    fi

    local summary_line
    summary_line=$(grep -E "(passed|failed|error|skipped)" "$temp_output" 2>/dev/null | grep -E "in [0-9]+\.[0-9]+s" | tail -1)

    if [[ -n "$summary_line" ]]; then
        local p_num f_num e_num s_num
        p_num=$(echo "$summary_line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
        f_num=$(echo "$summary_line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
        e_num=$(echo "$summary_line" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | head -1)
        s_num=$(echo "$summary_line" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' | head -1)

        [[ -n "$p_num" ]] && passed=$p_num
        [[ -n "$f_num" ]] && failed=$f_num
        [[ -n "$e_num" ]] && error=$e_num
        [[ -n "$s_num" ]] && skipped=$s_num
    fi

    # 解析失败的 case 名称
    local failed_cases=()
    if [[ $exit_code -ne 0 ]] && [[ $((failed + error)) -gt 0 ]]; then
        while IFS= read -r line; do
            line=$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g')
            if [[ "$line" =~ FAILED[[:space:]]+([^[:space:]]+) ]]; then
                local test_fullname="${BASH_REMATCH[1]}"
                if [[ "$test_fullname" =~ ::(.+)$ ]]; then
                    local test_name="${BASH_REMATCH[1]}"
                    if [[ -n "$test_name" ]] && [[ "$test_name" != "["* ]]; then
                        failed_cases+=("$test_name")
                    fi
                fi
            fi
        done < <(grep -E "^FAILED[[:space:]]+" "$temp_output" 2>/dev/null | head -20)

        if [[ ${#failed_cases[@]} -gt 0 ]]; then
            local unique_cases=()
            while IFS= read -r unique; do
                [[ -n "$unique" ]] && unique_cases+=("$unique")
            done < <(printf "%s\n" "${failed_cases[@]}" | sort -u)
            failed_cases=("${unique_cases[@]}")
        fi
    fi

    # 记录到详细日志
    {
        echo ">>> 执行: $case_name"
        echo "Case统计: collected=$collected, passed=$passed, failed=$failed, error=$error, skipped=$skipped"
        if [[ ${#failed_cases[@]} -gt 0 ]]; then
            echo "失败Cases:"
            printf "  - %s\n" "${failed_cases[@]}"
        fi
    } >> "$DETAIL_LOG"

    if [[ $exit_code -eq 0 ]]; then
        printf "PASS (%s cases)\n" "$collected" >&2
        echo "PASS: $case_name" >> "$DETAIL_LOG"
    else
        printf "FAIL (%s failed, %s error)\n" "$failed" "$error" >&2
        echo "FAIL: $case_name" >> "$DETAIL_LOG"
    fi

    echo "---" >> "$DETAIL_LOG"

    # 将 failed_cases 数组转换为逗号分隔的字符串
    local failed_cases_str=""
    if [[ ${#failed_cases[@]} -gt 0 ]]; then
        local IFS=','
        failed_cases_str="${failed_cases[*]}"
    fi

    echo "${collected}|${passed}|${failed}|${error}|${skipped}|${exit_code}|${failed_cases_str}" > "$result_file"
    echo "$result_file"
    rm -f "$temp_output"
}

# ============ 主流程 ============
main() {
    preflight_check
    parse_args "$@"

    echo "========== FD Test Runner =========="
    echo "脚本位置: ${SCRIPT_DIR}"
    echo "Case目录: ${CASE_DIR}"
    echo "Log目录:  ${LOG_DIR}"
    echo "目标服务: ${HOST}:${PORT}"
    echo "执行输入: ${CASES}"
    [[ -n "$EXCLUDE" ]] && echo "排除项:   ${EXCLUDE}"
    echo "Pytest超时: ${PYTEST_TIMEOUT}秒"
    echo ""

    local case_files=()
    while IFS= read -r file; do
        [[ -n "$file" ]] && case_files+=("$file")
    done < <(resolve_cases "$CASES" "$EXCLUDE")

    local total_files=${#case_files[@]}
    [[ $total_files -eq 0 ]] && { echo "错误: 没有找到可执行的case"; exit 1; }

    echo "将执行 ${total_files} 个文件..."
    echo ""

    local file_passed=0
    local file_failed=0
    local total_collected=0
    local total_passed=0
    local total_failed=0
    local total_error=0
    local total_skipped=0
    local results=()
    local failed_files=()
    local failed_cases_map=()
    local temp_result_files=()

    # 设置整体超时看门狗（使用 Python）
    (
        python3 -c "import time; time.sleep($SCRIPT_TIMEOUT); print(''); print('错误: 脚本整体执行超过 ${SCRIPT_TIMEOUT} 秒，强制退出'); import os, signal; os.kill($$, signal.SIGTERM)" 2>/dev/null
    ) &
    WATCHDOG_PID=$!

    for file in "${case_files[@]}"; do
        local case_name
        case_name=$(basename "$file")
        local result_file
        result_file=$(run_case "$file")
        temp_result_files+=("$result_file")

        local stats
        stats=$(cat "$result_file")
        local c p f e s exit_code failed_cases_str
        IFS='|' read -r c p f e s exit_code failed_cases_str <<< "$stats"

        c=${c:-0}; p=${p:-0}; f=${f:-0}; e=${e:-0}; s=${s:-0}

        total_collected=$((total_collected + c))
        total_passed=$((total_passed + p))
        total_failed=$((total_failed + f))
        total_error=$((total_error + e))
        total_skipped=$((total_skipped + s))

        if [[ "$exit_code" == "0" ]]; then
            file_passed=$((file_passed + 1))
            results+=("| ${case_name} | PASS | ${c} |")
        else
            file_failed=$((file_failed + 1))
            results+=("| ${case_name} | FAIL | ${c} |")
            failed_files+=("$case_name")
            failed_cases_map+=("${case_name}:${failed_cases_str}")
        fi
    done

    # 取消看门狗
    kill "$WATCHDOG_PID" 2>/dev/null
    wait "$WATCHDOG_PID" 2>/dev/null
    unset WATCHDOG_PID

    # 清理临时文件
    for tmp_file in "${temp_result_files[@]}"; do
        rm -f "$tmp_file"
    done

    echo ""

    local pass_rate=0
    [[ $total_collected -gt 0 ]] && pass_rate=$(awk "BEGIN {printf \"%.0f\", ($total_passed/$total_collected)*100}")

    # 生成失败文件详细列表
    local failed_details=""
    if [[ ${#failed_cases_map[@]} -gt 0 ]]; then
        for entry in "${failed_cases_map[@]}"; do
            local fname="${entry%%:*}"
            local cases_str="${entry#*:}"
            failed_details+="- **${fname}**"$'\n'
            if [[ -n "$cases_str" ]]; then
                local IFS=','
                for case_name in $cases_str; do
                    [[ -n "$case_name" ]] && failed_details+="  - ${case_name}"$'\n'
                done
            else
                failed_details+="  - (无法解析具体case名称)"$'\n'
            fi
            failed_details+=$'\n'
        done
    else
        failed_details="无"
    fi

    # 生成报告
    {
        echo "# FD Test Report"
        echo ""
        echo "**执行时间**: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**目标服务**: ${HOST}:${PORT}"
        echo "**执行输入**: ${CASES}"
        [[ -n "$EXCLUDE" ]] && echo "**排除项**: ${EXCLUDE}"
        echo ""
        echo "## 汇总"
        echo ""
        echo "| 指标 | 数值 |"
        echo "|-----|------|"
        echo "| 总Case数 | ${total_collected} |"
        echo "| 通过 | ${total_passed} |"
        echo "| 失败 | ${total_failed} |"
        echo "| 错误 | ${total_error} |"
        echo "| 跳过 | ${total_skipped} |"
        echo "| 超时 | $(grep -c "TIMEOUT" "$DETAIL_LOG" 2>/dev/null || echo 0) |"
        echo "| 通过率 | ${pass_rate}% |"
        echo ""
        echo "## 文件结果"
        echo ""
        echo "| 文件 | 结果 | Cases |"
        echo "|-----|------|-------|"
        printf "%s\n" "${results[@]}"
        echo ""
        echo "## 失败文件"
        echo ""
        printf "%s" "$failed_details"
        echo ""
        echo "## 日志位置"
        echo ""
        echo "- 详细日志: ${DETAIL_LOG}"
        echo "- 原始输出: ${RAW_LOG}"
    } > "$REPORT_FILE"

    echo "========== 执行完成 =========="
    echo ""
    echo "Case统计:"
    echo "   总Case数: ${total_collected}"
    echo "   通过:     ${total_passed}"
    echo "   失败:     ${total_failed}"
    echo "   错误:     ${total_error}"
    echo "   跳过:     ${total_skipped}"
    echo "   通过率:   ${pass_rate}%"
    echo ""
    echo "文件统计: ${file_passed}/${total_files} 通过, ${file_failed}/${total_files} 失败"
    echo ""
    echo "报告: ${REPORT_FILE}"
    echo "日志: ${DETAIL_LOG}"
    echo "原始: ${RAW_LOG}"

    [[ $file_failed -eq 0 ]] && return 0 || return 1
}

main "$@"
