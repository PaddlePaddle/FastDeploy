

export http_proxy=http://agent.baidu.com:8891/
export https_proxy=http://agent.baidu.com:8891/

export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com,.cdn.bcebos.com,.baidu.com,.bcebos.com,.gz.bcebos.com,cdn.bcebos.com"
export no_proxy="${no_proxy},bcebos.com,apiin.im.baidu.com,gitee.com,aliyun.com,.baidu.com,.tuna.tsinghua.edu.cn,gitlab.com,gitlab.mpcdf.mpg.de"
export no_proxy="${no_proxy},paddle-ci.gz.bcebos.com,paddle-ci.cdn.bcebos.com,baidu-kunlun-product.su.bcebos.com,opencdncloud.game.eastecloud.com"

export FD_MODEL_SOURCE=HUGGINGFACE
export FD_MODEL_CACHE=./models

export CUDA_VISIBLE_DEVICES=0
export ENABLE_V1_KVCACHE_SCHEDULER=1
# FD_DETERMINISTIC_MODE: Toggle deterministic mode
#   0: Disable deterministic mode (non-deterministic)
#   1: Enable deterministic mode (default)
# Usage: bash start_fd.sh [0|1]
export FD_DETERMINISTIC_MODE=${1:-1}


source /root/paddlejob/workspace/env_run/gongweibao/fdenv/bin/activate

python -m fastdeploy.entrypoints.openai.api_server \
       --model ./models/Qwen/Qwen2.5-7B \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --enable-logprob \
       --graph-optimization-config '{"use_cudagraph":true}' \
       --no-enable-prefix-caching \
       --no-enable-output-caching
