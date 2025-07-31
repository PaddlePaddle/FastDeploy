#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#先kill一遍
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

export model_path=${MODEL_PATH}/data/eb45t_4_layer
export CLANG_PATH=${MODEL_PATH}/data/xtdk
export XVLLM_PATH=${MODEL_PATH}/data/xvllm

echo "pip requirements"
python -m pip install -r requirements.txt
echo "uninstall org"
python -m pip uninstall paddlepaddle-xpu -y
python -m pip uninstall fastdeploy-xpu -y
python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
echo "build whl"
bash build.sh || exit 1
echo "pip others"
python -m pip install openai -U
python -m pip uninstall -y triton
python -m pip install triton==3.3.0

unset http_proxy
unset https_proxy
unset no_proxy

# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
export XPU_VISIBLE_DEVICES="0,1,2,3"
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization wint4   > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((5 * 60))
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
python test/ci_use/XPU_45T/run_45T.py
exit_code=$?
echo exit_code is ${exit_code}

ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "模型起服务失败，请检查pr代码"
    exit 1
fi

#0731新增kv block集中式管理相关测试，在起服务时启用对应环境变量 export ENABLE_V1_KVCACHE_SCHEDULER=True
# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
export ENABLE_V1_KVCACHE_SCHEDULER=1
export XPU_VISIBLE_DEVICES="0,1,2,3"
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --quantization wint4   > server.log 2>&1 &

sleep 60
# 探活
TIMEOUT=$((5 * 60))
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
python test/ci_use/XPU_45T/run_45T.py
kv_block_test_exit_code=$?
echo kv_block_test_exit_code is ${kv_block_test_exit_code}

unset ENABLE_V1_KVCACHE_SCHEDULER
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "kv block相关测试失败，请检查pr代码"
    exit 1
fi
