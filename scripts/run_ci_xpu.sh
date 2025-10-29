#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#安装lsof工具
apt install -y lsof

#先kill一遍
ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true
#设置模型路径
export model_path=${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle

echo "pip requirements"
python -m pip install -r requirements.txt

echo "uninstall org"
python -m pip uninstall paddlepaddle-xpu -y
python -m pip uninstall fastdeploy-xpu -y

python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/

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

# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
echo "============================开始V1模式测试!============================"
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --tensor-parallel-size 8 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization wint4   > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10            # 检查间隔（秒）
ENDPOINT="http://0.0.0.0:8188/health"
START_TIME=$(date +%s) # 记录开始时间戳
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    # 计算已耗时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 超时判断
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        cat server.log
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done

cat server.log

# 执行服务化推理
python -m pytest tests/ci_use/XPU_45T/run_45T.py
kv_block_test_exit_code=$?
echo kv_block_test_exit_code is ${kv_block_test_exit_code}

ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${kv_block_test_exit_code} -ne 0 ]; then
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
export XPU_VISIBLE_DEVICES="0,1,2,3"
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_PATH}/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \
    --port 8188 \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "W4A8"   > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((15 * 60))
INTERVAL=10            # 检查间隔（秒）
ENDPOINT="http://0.0.0.0:8188/health"
START_TIME=$(date +%s) # 记录开始时间戳
echo "开始服务健康检查，最长等待时间：${TIMEOUT}秒"
while true; do
    # 计算已耗时
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 超时判断
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n服务启动超时：经过 $((TIMEOUT/60)) 分钟服务仍未启动！"
        cat server.log
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\n服务启动成功！耗时 ${ELAPSED} 秒"
        break
    else
        sleep $INTERVAL
    fi
done

cat server.log

# 执行服务化推理
python -m pytest tests/ci_use/XPU_45T/run_w4a8.py
w4a8_test_exit_code=$?
echo w4a8_test_exit_code is ${w4a8_test_exit_code}

ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${w4a8_test_exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "w4a8 测试失败，请检查pr代码"
    exit 1
fi

echo "============================开始EP并行测试!============================"
sleep 5
rm -rf log/*
rm -f core*
xpu-smi
export XPU_VISIBLE_DEVICES="0,1,2,3"
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

python -m pytest -s --timeout=300 tests/ci_use/XPU_45T/run_ep.py
ep_exit_code=$?

unset BKCL_ENABLE_XDR
unset BKCL_RDMA_NICS
unset BKCL_TRACE_TOPO
unset BKCL_PCIE_RING
unset XSHMEM_MODE
unset XSHMEM_QP_NUM_PER_RANK
unset BKCL_RDMA_VERBS
ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${ep_exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "EP并行 相关测试失败，请检查pr代码"
    exit 1
fi
