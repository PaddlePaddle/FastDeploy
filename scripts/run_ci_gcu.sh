#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#先kill一遍
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

export model_path=${MODEL_PATH}/paddle/ERNIE-4.5-21B-A3B-Paddle

echo "pip install requirements"
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "uninstall org"
python -m pip uninstall paddlepaddle -y
python -m pip uninstall paddle-custom-gcu -y
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
echo "build whl"
bash build.sh 1 || exit 1

unset http_proxy
unset https_proxy
unset no_proxy

# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --metrics-port 8200 \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 4096 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --quantization wint4 > server.log 2>&1 &

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
python test/ci_use/GCU/run_ernie.py
exit_code=$?
echo exit_code is ${exit_code}

ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8188 | xargs kill -9 || true

if [ ${exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    exit 1
fi
