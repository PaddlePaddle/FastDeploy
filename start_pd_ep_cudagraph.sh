#!/bin/bash
# FastDeploy 服务启动脚本
# 支持两种模式：集中式EP服务 / PD分离服务

set -e

#============================================================
# 公共配置
#============================================================
export PYTHONPATH=/opt/output/work_dir/ssd1/yinwei06/FastDeploy
# export FD_DEBUG=1

# 取消代理设置
unset http_proxy
unset https_proxy

# 模型路径
MODEL_PATH="/opt/output/work_dir/ssd1/yinwei06/FD_slice_merge/10.63.230.221:8891//ERNIE-4.5-21B-A3B-Paddle"

#============================================================
# 第一部分：清理残余进程和日志
#============================================================
cleanup() {
    echo "[1/3] 清理残余进程和日志..."

    # 清理日志目录
    rm -rf log
    rm -rf log_router
    mkdir -p log_router

    # 杀死残余 Python 进程
    local patterns=("python -m" "python -u" "python -c" "splitwise_role")
    for pattern in "${patterns[@]}"; do
        pids=$(ps aux | grep "$pattern" | grep -v grep | awk '{print $2}' || true)
        if [ -n "$pids" ]; then
            echo "  - 终止进程: $pattern (PIDs: $pids)"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    done

    # 杀掉占用的端口
    local ports=(8188 8189 8199 8209 8210 8211 8212)
    for port in "${ports[@]}"; do
        pids=$(lsof -t -i :${port} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  - 杀掉端口 ${port} 占用进程 (PIDs: $pids)"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    done
    sleep 3
    xpu-smi -r -i "0,1,2,3,4,5,6,7"
    echo "  清理完成"
}

#============================================================
# 第二部分：集中式 EP 服务配置
#============================================================
start_centralized_ep() {
    echo "[2/3] 启动集中式 EP 服务..."

    # XPU 设备配置
    export XPU_VISIBLE_DEVICES="0,1,2,3"

    # BKCL 通信配置
    export BKCL_ENABLE_XDR=1
    export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
    export BKCL_TRACE_TOPO=1
    export BKCL_PCIE_RING=1
    export BKCL_RDMA_VERBS=1

    # 共享内存配置
    export XSHMEM_MODE=1
    export XSHMEM_QP_NUM_PER_RANK=32

    # 启动 Router
    local router_port=8188
    nohup python -m fastdeploy.router.launch \
        --port ${router_port} \
        2>&1 >./log_router/nohup &
    sleep 1

    # 启动 API Server
    python -m fastdeploy.entrypoints.openai.multi_api_server \
        --ports 8189 \
        --num-servers 1 \
        --metrics-ports 8199 \
        --args \
        --model ${MODEL_PATH} \
        --engine-worker-queue-port 8209 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --data-parallel-size 1 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --quantization wint4 \
        --enable-prefix-caching \
        --router 0.0.0.0:${router_port} \
        --graph-optimization-config '{"use_cudagraph":true}' \
        --disable-sequence-parallel-moe

    echo "  集中式 EP 服务启动完成"
}

#============================================================
# 第三部分：PD 分离服务配置
#============================================================
start_pd_disaggregated() {
    echo "[3/3] 启动 PD 分离服务..."

    # KVCache 传输配置
    export KVCACHE_GDRCOPY_FLUSH_ENABLE=1
    export CUDA_ENABLE_P2P_NO_UVA=1
    export KVCACHE_RDMA_NICS=mlx5_1,mlx5_1,mlx5_2,mlx5_2,mlx5_3,mlx5_3,mlx5_4,mlx5_4

    # BKCL 通信配置
    export BKCL_ENABLE_XDR=1
    export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2
    export BKCL_TRACE_TOPO=1
    export BKCL_PCIE_RING=1
    export BKCL_RDMA_VERBS=1

    # 共享内存配置
    export XSHMEM_MODE=1
    export XSHMEM_QP_NUM_PER_RANK=32

    # Router 配置
    local router_port=8188
    local prefill_port=8199
    local decode_port=8209

    # 启动 Router (splitwise 模式)
    python -m fastdeploy.router.launch \
        --port ${router_port} \
        --splitwise \
        2>&1 >./log_router/nohup &
    sleep 1

    # # 启动 Prefill 服务 (GPU 0-3)
    export XPU_VISIBLE_DEVICES="0,1,2,3"

    nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
        --port ${prefill_port} \
        --num-servers 1 \
        --args \
        --model ${MODEL_PATH} \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --quantization wint4 \
        --splitwise-role prefill \
        --cache-transfer-protocol rdma \
        --enable-expert-parallel \
        --graph-optimization-config '{"use_cudagraph":false}' \
        --router 0.0.0.0:${router_port} \
        --disable-sequence-parallel-moe \
        2>&1 >./log_router/nohup_prefill &

    # echo "  Prefill 服务启动中，等待 10 秒..."
    # sleep 10

    # 启动 Decode 服务 (GPU 4-7)
    export XPU_VISIBLE_DEVICES="4,5,6,7"
    nohup python -m fastdeploy.entrypoints.openai.multi_api_server \
        --port ${decode_port} \
        --num-servers 1 \
        --args \
        --model ${MODEL_PATH} \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --quantization wint4 \
        --splitwise-role decode \
        --cache-transfer-protocol rdma \
        --enable-expert-parallel \
        --graph-optimization-config '{"use_cudagraph":true}' \
        --router 0.0.0.0:${router_port} \
        --disable-sequence-parallel-moe \
        2>&1 >./log_router/nohup_decode &

    echo "  PD 分离服务启动完成"
    echo "  - Router:  port ${router_port}"
    echo "  - Prefill: port ${prefill_port}"
    echo "  - Decode:  port ${decode_port}"
}

#============================================================
# 主入口：选择启动模式
#============================================================
# 设置启动模式: "centralized" 或 "pd_disaggregated"
MODE="${1:-pd_disaggregated}"

echo "============================================"
echo "FastDeploy 服务启动脚本"
echo "模式: ${MODE}"
echo "============================================"

# 执行清理
cleanup

# 根据模式启动服务
case "${MODE}" in
    centralized)
        start_centralized_ep
        ;;
    pd_disaggregated)
        start_pd_disaggregated
        ;;
    *)
        echo "错误: 未知模式 '${MODE}'"
        echo "用法: $0 [centralized|pd_disaggregated]"
        exit 1
        ;;
esac

echo "============================================"
echo "服务启动完成！"
echo "============================================"
