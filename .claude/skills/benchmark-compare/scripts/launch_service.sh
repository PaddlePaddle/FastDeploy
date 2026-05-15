#!/usr/bin/env bash
# launch_service.sh — 通用推理框架服务启动脚本
# 支持 FastDeploy / SGLang，支持单卡/多卡 TP/DP/EP/PD 分离模式
set -euo pipefail

# ============================================================
# 参数解析
# ============================================================
FRAMEWORK=""
MODEL=""
PORT=""
GPUS=""
TP=1
DP=1
EP=0
CONCURRENCY=32
MAX_MODEL_LEN=65536
QUANTIZATION="none"
LOG_FILE=""
VENV=""
GPU_MEM_UTIL="0.97"
ATTENTION_BACKEND=""
PD_ROLE=""
EXTRA_ARGS=""

usage() {
    cat <<'EOF'
用法: bash launch_service.sh [OPTIONS]

必需参数:
  --framework <fd|sg>         推理框架 (fd=FastDeploy, sg=SGLang)
  --model <PATH>              模型权重路径
  --port <PORT>               服务端口
  --gpus <DEVICES>            CUDA_VISIBLE_DEVICES (如 "0" 或 "0,1,2,3,4,5,6,7")
  --venv <PATH>               虚拟环境路径 (.venv 目录)

可选参数:
  --tp <N>                    tensor-parallel-size (默认: 1)
  --dp <N>                    data-parallel-size (默认: 1)
  --ep <N>                    expert-parallel-size, MoE 模型专用 (默认: 0, 不启用)
                              FD: 映射为 --enable-expert-parallel (EP=TP×DP 隐式)
                              SG: 映射为 --ep-size N
  --concurrency <N>           max-num-seqs / max-running-requests (默认: 32)
  --max-model-len <N>         最大序列长度 (默认: 65536)
  --quantization <TYPE>       量化方式: none|block_wise_fp8|fp8|wint4|wint8 (默认: none)
  --log-file <PATH>           日志输出路径 (默认: /tmp/<framework>_server.log)
  --gpu-memory-utilization <F> GPU 显存利用率 (默认: 0.97)
  --attention-backend <TYPE>  注意力后端 (FD: MLA_ATTN; SG: flashmla)
  --pd-role <prefill|decode>  PD 分离角色, 仅 FD
  --extra-args <ARGS>         额外传递给服务的参数

示例:
  # 单卡启动 FastDeploy
  bash launch_service.sh --framework fd --model /path/to/model --port 8180 \
    --gpus 0 --venv /path/to/FastDeploy/.venv

  # TP=4 + DP=2 + EP=8 启动 FastDeploy (MoE, 8卡)
  bash launch_service.sh --framework fd --model /path/to/model --port 8180 \
    --gpus 0,1,2,3,4,5,6,7 --tp 4 --dp 2 --ep 8 --venv /path/to/FastDeploy/.venv

  # TP=4 + DP=2 + EP=8 启动 SGLang (MoE, 8卡)
  bash launch_service.sh --framework sg --model /path/to/model --port 8280 \
    --gpus 0,1,2,3,4,5,6,7 --tp 4 --dp 2 --ep 8 --venv /path/to/sglang_env/.venv
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework)       FRAMEWORK="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --gpus)            GPUS="$2"; shift 2 ;;
        --tp)              TP="$2"; shift 2 ;;
        --dp)              DP="$2"; shift 2 ;;
        --ep)              EP="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --quantization)    QUANTIZATION="$2"; shift 2 ;;
        --log-file)        LOG_FILE="$2"; shift 2 ;;
        --venv)            VENV="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
        --pd-role)         PD_ROLE="$2"; shift 2 ;;
        --extra-args)      EXTRA_ARGS="$2"; shift 2 ;;
        --help|-h)         usage 0 ;;
        *)                 echo "未知参数: $1"; usage 1 ;;
    esac
done

# 参数校验
if [[ -z "$FRAMEWORK" || -z "$MODEL" || -z "$PORT" || -z "$GPUS" || -z "$VENV" ]]; then
    echo "错误: --framework, --model, --port, --gpus, --venv 均为必需参数"
    usage 1
fi

if [[ "$FRAMEWORK" != "fd" && "$FRAMEWORK" != "sg" ]]; then
    echo "错误: --framework 必须为 fd 或 sg"
    exit 1
fi

# 默认日志路径
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="/tmp/${FRAMEWORK}_server_${PORT}.log"
fi

# ============================================================
# 端口清理
# ============================================================
if lsof -i :"$PORT" &>/dev/null; then
    echo "[INFO] 端口 $PORT 被占用，正在清理..."
    kill $(lsof -t -i :"$PORT") 2>/dev/null || true
    sleep 2
fi

# ============================================================
# 启动 FastDeploy
# ============================================================
launch_fastdeploy() {
    echo "[INFO] 启动 FastDeploy 服务..."
    echo "  模型: $MODEL"
    echo "  端口: $PORT"
    echo "  GPU: $GPUS (TP=$TP, DP=$DP, EP=$EP)"
    echo "  并发: $CONCURRENCY"
    echo "  量化: $QUANTIZATION"
    echo "  日志: $LOG_FILE"

    source "$VENV/bin/activate"

    # 设置 LD_LIBRARY_PATH (NVIDIA 库)
    export LD_LIBRARY_PATH=$(python3 -c "
import site, os
sp = site.getsitepackages()[0]
nvidia_dir = os.path.join(sp, 'nvidia')
if os.path.isdir(nvidia_dir):
    libs = [os.path.join(nvidia_dir, d, 'lib') for d in os.listdir(nvidia_dir) if os.path.isdir(os.path.join(nvidia_dir, d, 'lib'))]
    print(':'.join(libs))
else:
    print('')
"):${LD_LIBRARY_PATH:-}

    # 环境变量
    export CUDA_VISIBLE_DEVICES="$GPUS"

    # 注意力后端配置
    if [[ -z "$ATTENTION_BACKEND" ]]; then
        ATTENTION_BACKEND="MLA_ATTN"
    fi
    export FD_ATTENTION_BACKEND="$ATTENTION_BACKEND"
    export USE_FLASH_MLA=1
    export FLAGS_flash_attn_version=3
    export FD_SAMPLING_CLASS=rejection

    # 构建命令 (优先使用 fastdeploy CLI，如不可用则回退到 python -m)
    local CMD
    if command -v fastdeploy &>/dev/null; then
        CMD="fastdeploy serve"
    else
        CMD="python -m fastdeploy.entrypoints.openai.api_server"
    fi
    CMD+=" --model $MODEL"
    CMD+=" --port $PORT"
    CMD+=" --tensor-parallel-size $TP"
    CMD+=" --max-model-len $MAX_MODEL_LEN"
    CMD+=" --max-num-seqs $CONCURRENCY"
    CMD+=" --gpu-memory-utilization $GPU_MEM_UTIL"
    CMD+=" --trust-remote-code"

    # DP (data parallelism)
    if [[ "$DP" -gt 1 ]]; then
        CMD+=" --data-parallel-size $DP"
    fi

    # EP (expert parallelism) — FD 只有 flag，EP size 隐式 = TP×DP
    if [[ "$EP" -gt 0 ]]; then
        CMD+=" --enable-expert-parallel"
    fi

    # 量化
    if [[ "$QUANTIZATION" != "none" ]]; then
        CMD+=" --quantization $QUANTIZATION"
    fi

    # PD 分离
    if [[ -n "$PD_ROLE" ]]; then
        CMD+=" --splitwise-role $PD_ROLE"
    fi

    # 额外参数
    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD+=" $EXTRA_ARGS"
    fi

    echo "[INFO] 执行: $CMD"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo $! > "/tmp/fd_pid_${PORT}"
    echo "[INFO] FastDeploy PID: $! (已写入 /tmp/fd_pid_${PORT})"
}

# ============================================================
# 启动 SGLang
# ============================================================
launch_sglang() {
    echo "[INFO] 启动 SGLang 服务..."
    echo "  模型: $MODEL"
    echo "  端口: $PORT"
    echo "  GPU: $GPUS (TP=$TP, DP=$DP, EP=$EP)"
    echo "  并发: $CONCURRENCY"
    echo "  量化: $QUANTIZATION"
    echo "  日志: $LOG_FILE"

    source "$VENV/bin/activate"

    export CUDA_VISIBLE_DEVICES="$GPUS"

    # DP 模式下，设置 MASTER_PORT 避免 torch.distributed 端口冲突
    # 默认使用 45000+ 范围，避免与系统服务（18xxx）冲突
    if [[ "$DP" -gt 1 ]]; then
        export MASTER_PORT=${MASTER_PORT:-45000}
        echo "[INFO] DP=$DP, 设置 MASTER_PORT=$MASTER_PORT 避免端口冲突"
    fi

    # 注意力后端
    if [[ -z "$ATTENTION_BACKEND" ]]; then
        ATTENTION_BACKEND="flashmla"
    fi

    # 构建命令
    local CMD="python3 -m sglang.launch_server"
    CMD+=" --model-path $MODEL"
    CMD+=" --host 0.0.0.0"
    CMD+=" --port $PORT"
    CMD+=" --tp $TP"
    CMD+=" --context-length $MAX_MODEL_LEN"
    CMD+=" --max-running-requests $CONCURRENCY"
    CMD+=" --attention-backend $ATTENTION_BACKEND"
    CMD+=" --trust-remote-code"

    # DP (data parallelism)
    if [[ "$DP" -gt 1 ]]; then
        CMD+=" --dp-size $DP"
    fi

    # EP (expert parallelism) — SG 使用显式 --ep-size
    if [[ "$EP" -gt 0 ]]; then
        CMD+=" --ep-size $EP"
    fi

    # 量化
    if [[ "$QUANTIZATION" != "none" ]]; then
        local SG_QUANT="$QUANTIZATION"
        # 映射 FD 量化名到 SG 名
        if [[ "$SG_QUANT" == "block_wise_fp8" ]]; then
            SG_QUANT="fp8"
        fi
        CMD+=" --quantization $SG_QUANT"
    fi

    # 额外参数
    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD+=" $EXTRA_ARGS"
    fi

    echo "[INFO] 执行: $CMD"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo $! > "/tmp/sg_pid_${PORT}"
    echo "[INFO] SGLang PID: $! (已写入 /tmp/sg_pid_${PORT})"
}

# ============================================================
# 主入口
# ============================================================
case "$FRAMEWORK" in
    fd) launch_fastdeploy ;;
    sg) launch_sglang ;;
esac

echo "[INFO] 服务已在后台启动，请使用 health_check.sh 等待就绪"
