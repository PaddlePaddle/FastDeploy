port=${1:-8080}
model_path=/host/EB4.5T-VL-4layer

timestamp=$(date +%Y%m%d%H%M%S)
mv ./log ./log_history/log_$timestamp
export PYTHONPATH=$(pwd)
python -m fastdeploy.entrypoints.openai.api_server \
       --model $model_path \
       --port $port --engine-worker-queue-port $((port+1)) \
       --cache-queue-port $((port+2)) --metrics-port $((port+2)) \
       --tensor-parallel-size 8 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 10 \
       --enable-mm \
       --mm-processor-kwargs '{"video_max_frames": 30}' \
       --limit-mm-per-prompt '{"image": 10, "video": 3}' \
       --reasoning-parser ernie-45-vl
