#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#安装lsof工具
apt install -y lsof

#先kill一遍
function stop_processes() {
    ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E "$((8188 + XPU_ID * 100))" | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    lsof -t -i :$((8188 + XPU_ID * 100)) | xargs kill -9 || true
    for port in $(seq $((8188 + XPU_ID * 100 + 10)) $((8188 + XPU_ID * 100 + 40))); do
        lsof -t -i :${port} | xargs kill -9 || true
    done
    netstat -tunlp 2>/dev/null | grep $((8190 + XPU_ID * 100)) | awk '{print $NF}' | awk -F'/' '{print $1}' | xargs -r kill -9
    netstat -tunlp 2>/dev/null | grep $((8190 + XPU_ID * 100)) | awk '{print $(NF-1)}' | cut -d/ -f1 | grep -E '^[0-9]+$' | xargs -r kill -9
}

stop_processes >kill.log 2>&1

# 由于机器原因，需重启使用的卡，以保障没有问题
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi

mkdir -p /workspace/deps
cd /workspace/deps
wget -q https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.21/xre-Linux-x86_64-5.0.21.21.tar.gz
tar -zxf xre-Linux-x86_64-5.0.21.21.tar.gz && mv xre-Linux-x86_64-5.0.21.21 xre
cd -
export PATH=/workspace/deps/xre/bin:$PATH

xpu-smi -r -i $XPU_VISIBLE_DEVICES
xpu-smi

echo "pip requirements"
python -m pip install -r requirements.txt

echo "uninstall org"
python -m pip uninstall paddlepaddle-xpu -y
python -m pip uninstall fastdeploy-xpu -y

# python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
# 由于ep并行报错暂时锁死paddle版本
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/xpu-p800/paddlepaddle-xpu/paddlepaddle_xpu-3.3.0.dev20251123-cp310-cp310-linux_x86_64.whl
echo "build whl"
bash custom_ops/xpu_ops/download_dependencies.sh develop
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
bash build.sh || exit 1

echo "pip others"
python -m pip install openai -U
python -m pip uninstall -y triton
python -m pip install triton==3.3.0
python -m pip install pytest
python -m pip install pytest-timeout
unset http_proxy
unset https_proxy
unset no_proxy

stop_processes >kill.log 2>&1

# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
echo "============================开始V1模式测试!============================"
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi
export port_num=$((8188 + XPU_ID * 100))
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle \
    --port $port_num \
    --engine-worker-queue-port $((port_num + 1)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization wint4 \
    --enable-prefix-caching \
    --enable-chunked-prefill > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10            # 检查间隔（秒）
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s) # 记录开始时间戳
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    # 计算已耗时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 超时判断
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        echo "server.log"
        cat server.log
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行服务化推理
python -m pytest -s tests/ci_use/XPU_45T/run_45T.py
kv_block_test_exit_code=$?
echo kv_block_test_exit_code is ${kv_block_test_exit_code}

stop_processes >kill.log 2>&1

if [ ${kv_block_test_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "kv block相关测试失败，请检查pr代码"
    exit 1
fi

sleep 5
# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
echo "============================开始W4A8测试!============================"
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi
export port_num=$((8188 + XPU_ID * 100))
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \
    --port $port_num \
    --engine-worker-queue-port $((port_num + 1)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "W4A8"   > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10            # 检查间隔（秒）
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s) # 记录开始时间戳
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    # 计算已耗时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 超时判断
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        echo "server.log"
        cat server.log
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行服务化推理
python -m pytest -s tests/ci_use/XPU_45T/run_w4a8.py
w4a8_test_exit_code=$?
echo w4a8_test_exit_code is ${w4a8_test_exit_code}

stop_processes >kill.log 2>&1

if [ ${w4a8_test_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "w4a8 测试失败，请检查pr代码"
    exit 1
fi

sleep 5
# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
echo "============================开始vl模型测试!============================"
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi
export port_num=$((8188 + XPU_ID * 100))
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-VL-28B-A3B-Thinking \
    --port $port_num \
    --engine-worker-queue-port $((port_num + 1)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --quantization wint8 \
    --reasoning-parser ernie-45-vl-thinking \
    --tool-call-parser ernie-45-vl-thinking \
    --mm-processor-kwargs '{"image_max_pixels": 12845056 }' \
    --enable-chunked-prefill > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10            # 检查间隔（秒）
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s) # 记录开始时间戳
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    # 计算已耗时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 超时判断
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        echo "server.log"
        cat server.log
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行服务化推理
python -m pytest -s tests/ci_use/XPU_45T/run_45vl.py
vl_test_exit_code=$?
echo vl_test_exit_code is ${vl_test_exit_code}

stop_processes >kill.log 2>&1

if [ ${vl_test_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo " vl模型 测试失败，请检查pr代码"
    exit 1
fi


echo "============================开始 EP4TP4 在线服务测试!============================"
sleep 5
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
ipcrm --all=msg
xpu-smi
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi

export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=xgbe1,xgbe2,xgbe3,xgbe4
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1

wget -q https://paddle-qa.bj.bcebos.com/xpu_third_party/xDeepEP.tar.gz
tar -xzf xDeepEP.tar.gz
cd xDeepEP
bash build.sh
cd -

export port_num=$((8188 + XPU_ID * 100))
# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle \
    --port $port_num \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --engine-worker-queue-port $((port_num + 10)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --disable-sequence-parallel-moe \
    --gpu-memory-utilization 0.9 \
    --load-choices "default" > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s)
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        cat server.log
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行在线推理验证脚本
python -m pytest -s tests/ci_use/XPU_45T/run_ep_online.py
ep_online_exit_code=$?
echo ep_online_exit_code is ${ep_online_exit_code}

unset BKCL_ENABLE_XDR
unset BKCL_RDMA_NICS
unset BKCL_TRACE_TOPO
unset BKCL_PCIE_RING
unset XSHMEM_MODE
unset XSHMEM_QP_NUM_PER_RANK
unset BKCL_RDMA_VERBS
stop_processes >kill.log 2>&1

if [ ${ep_online_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    cat log/workerlog.0
    echo "EP4TP4 在线服务相关测试失败，请检查pr代码"
    exit 1
fi

echo "============================开始 EP4TP1 在线服务测试!============================"
sleep 5
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
ipcrm --all=msg
xpu-smi
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=xgbe1,xgbe2,xgbe3,xgbe4
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1

export port_num=$((8188 + XPU_ID * 100))
# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle \
    --port $port_num \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --engine-worker-queue-port "$((port_num + 10)),$((port_num + 20)),$((port_num + 30)),$((port_num + 40))" \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --gpu-memory-utilization 0.9 \
    --load-choices "default" > server.log 2>&1 &

sleep 60
# 探活（同上）
TIMEOUT=$((15 * 60))
INTERVAL=10
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s)
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        cat server.log
        cat log/workerlog.0
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行在线推理验证脚本
python -m pytest -s tests/ci_use/XPU_45T/run_ep_online.py
ep_online_exit_code=$?
echo ep_online_exit_code is ${ep_online_exit_code}

unset BKCL_ENABLE_XDR
unset BKCL_RDMA_NICS
unset BKCL_TRACE_TOPO
unset BKCL_PCIE_RING
unset XSHMEM_MODE
unset XSHMEM_QP_NUM_PER_RANK
unset BKCL_RDMA_VERBS
stop_processes >kill.log 2>&1

if [ ${ep_online_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    cat log/workerlog.0
    echo "EP4TP1 在线服务相关测试失败，请检查pr代码"
    exit 1
fi

echo "============================开始 EP4TP4 all2all 测试!============================"
sleep 5
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
ipcrm --all=msg
xpu-smi
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi

export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=xgbe1,xgbe2,xgbe3,xgbe4
export BKCL_TRACE_TOPO=1
export BKCL_PCIE_RING=1
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=32
export BKCL_RDMA_VERBS=1

export port_num=$((8188 + XPU_ID * 100))
# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle \
    --port $port_num \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "wint4" \
    --engine-worker-queue-port $((port_num + 10)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --gpu-memory-utilization 0.9 \
    --load-choices "default" > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10
ENDPOINT="http://0.0.0.0:${port_num}/health"
START_TIME=$(date +%s)
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        stop_processes
        cat server.log
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)
    echo -e "\r服务健康检查中... 已等待 ${ELAPSED} 秒，当前状态码：${HTTP_CODE}"
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done


# 执行在线推理验证脚本
python -m pytest -s tests/ci_use/XPU_45T/run_ep_online.py
ep_online_exit_code=$?
echo ep_online_exit_code is ${ep_online_exit_code}

unset BKCL_ENABLE_XDR
unset BKCL_RDMA_NICS
unset BKCL_TRACE_TOPO
unset BKCL_PCIE_RING
unset XSHMEM_MODE
unset XSHMEM_QP_NUM_PER_RANK
unset BKCL_RDMA_VERBS
stop_processes >kill.log 2>&1

if [ ${ep_online_exit_code} -ne 0 ]; then
    echo "server.log"
    cat server.log
    cat log/workerlog.0
    echo "EP4TP4 all2all 在线服务相关测试失败，请检查pr代码"
    exit 1
fi
