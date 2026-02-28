export FD_MODEL_SOURCE=HUGGINGFACE
export FD_MODEL_CACHE=./models

export CUDA_VISIBLE_DEVICES=0
export ENABLE_V1_KVCACHE_SCHEDULER=1

# FD_DETERMINISTIC_MODE: Toggle deterministic mode
#   0: Disable deterministic mode (non-deterministic)
#   1: Enable deterministic mode (default)
# FD_DETERMINISTIC_LOG_MODE: Toggle determinism logging
#   0: Disable logging (high performance, recommended for production)
#   1: Enable logging with MD5 hashes (debug mode)
# Usage: bash start_fd.sh [deterministic_mode] [log_mode]
# Example:
#   bash start_fd.sh 1 0  # Deterministic mode without logging (fast)
#   bash start_fd.sh 1 1  # Deterministic mode with logging (debug)
export FD_DETERMINISTIC_MODE=${1:-1}
export FD_DETERMINISTIC_LOG_MODE=${2:-0}


python -m fastdeploy.entrypoints.openai.api_server \
       --model ./models/Qwen/Qwen2.5-7B \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --enable-logprob \
       --graph-optimization-config '{"use_cudagraph":true}' \
       --no-enable-prefix-caching \
       --no-enable-output-caching
