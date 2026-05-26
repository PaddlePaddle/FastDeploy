#!/usr/bin/env bash
# run_benchmark.sh — Benchmark 执行封装脚本
# 调用 FastDeploy 的 benchmark_serving.py 对指定服务进行性能测试
set -euo pipefail

LABEL=""
MODEL=""
PORT=""
HOST="127.0.0.1"
DATASET=""
HYPERPARAMS=""
CONCURRENCY=32
NUM_PROMPTS=1024
OUTPUT=""
VENV=""
BENCHMARK_DIR=""
IP_LIST=""
BACKEND="openai-chat"
ENDPOINT="/v1/chat/completions"
EXTRA_ARGS=""

usage() {
    cat <<'EOF'
用法: bash run_benchmark.sh [OPTIONS]

必需参数:
  --label <fd|sg>             标识 (用于输出提示)
  --model <PATH>              模型路径
  --port <PORT>               服务端口
  --dataset <PATH>            数据集路径 (JSONL)
  --hyperparams <PATH>        Hyperparameter YAML 路径
  --output <PATH>             结果输出文件路径
  --venv <PATH>               FastDeploy 虚拟环境路径
  --benchmark-dir <PATH>      FastDeploy/benchmarks 目录路径

可选参数:
  --host <HOST>               服务地址 (默认: 127.0.0.1)
  --concurrency <N>           最大并发 (默认: 32)
  --num-prompts <N>           请求数 (默认: 1024)
  --backend <TYPE>            后端类型 (默认: openai-chat)
  --endpoint <PATH>           API 路径 (默认: /v1/chat/completions)
  --ip-list <IP:PORT,...>     多实例地址列表（多机模式）
  --extra-args <ARGS>         额外传递给 benchmark_serving.py 的参数

示例:
  bash run_benchmark.sh --label fd --model /path/to/model --port 8180 \
    --dataset /path/to/data.jsonl --hyperparams /path/to/GLM-32k.yaml \
    --output /tmp/result_fd.txt --venv /path/to/FastDeploy/.venv \
    --benchmark-dir /path/to/FastDeploy/benchmarks
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)         LABEL="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --host)          HOST="$2"; shift 2 ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --hyperparams)   HYPERPARAMS="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --num-prompts)   NUM_PROMPTS="$2"; shift 2 ;;
        --output)        OUTPUT="$2"; shift 2 ;;
        --venv)          VENV="$2"; shift 2 ;;
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --backend)       BACKEND="$2"; shift 2 ;;
        --endpoint)      ENDPOINT="$2"; shift 2 ;;
        --ip-list)       IP_LIST="$2"; shift 2 ;;
        --extra-args)    EXTRA_ARGS="$2"; shift 2 ;;
        --help|-h)       usage 0 ;;
        *)               echo "未知参数: $1"; usage 1 ;;
    esac
done

# 参数校验
for param in LABEL MODEL PORT DATASET HYPERPARAMS OUTPUT VENV BENCHMARK_DIR; do
    if [[ -z "${!param}" ]]; then
        echo "错误: --$(echo $param | tr '[:upper:]' '[:lower:]' | tr '_' '-') 为必需参数"
        usage 1
    fi
done

echo "[INFO] 开始 Benchmark 测试 [$LABEL]"
echo "  模型: $MODEL"
echo "  端口: $HOST:$PORT"
echo "  并发: $CONCURRENCY"
echo "  请求数: $NUM_PROMPTS"
echo "  输出: $OUTPUT"

# 激活虚拟环境
source "$VENV/bin/activate"

# 构建命令
CMD="python $BENCHMARK_DIR/benchmark_serving.py"
CMD+=" --backend $BACKEND"
CMD+=" --model $MODEL"
CMD+=" --endpoint $ENDPOINT"
CMD+=" --host $HOST"
CMD+=" --port $PORT"
CMD+=" --dataset-name EBChat"
CMD+=" --dataset-path $DATASET"
CMD+=" --hyperparameter-path $HYPERPARAMS"
CMD+=" --num-prompts $NUM_PROMPTS"
CMD+=" --max-concurrency $CONCURRENCY"
CMD+=" --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len"
CMD+=" --metric-percentiles 80,95,99,99.9,99.95,99.99"
CMD+=" --save-result"

# 多实例模式
if [[ -n "$IP_LIST" ]]; then
    CMD+=" --ip-list $IP_LIST"
fi

# 额外参数
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD+=" $EXTRA_ARGS"
fi

echo "[INFO] 执行: $CMD"
echo "[INFO] 输出重定向到: $OUTPUT"

eval "$CMD" > "$OUTPUT" 2>&1
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[INFO] Benchmark [$LABEL] 完成"
else
    echo "[ERROR] Benchmark [$LABEL] 失败 (exit code: $EXIT_CODE)"
    echo "[ERROR] 最后 20 行输出:"
    tail -20 "$OUTPUT" 2>/dev/null || true
fi

exit $EXIT_CODE
