pkill -f "fastdeploy.entrypoints.openai.api_server"
rm -rf log
# python -m fastdeploy.entrypoints.openai.api_server --model baidu/ERNIE-4.5-0.3B-Paddle --port 8180 --metrics-port 8181 --engine-worker-queue-port 8183 --quantization wint4 --max-model-len 32768 --max-num-seqs 32

python -m fastdeploy.entrypoints.openai.api_server \
       --model /root/.paddlenlp/models/Qwen/Qwen2.5-VL-7B-Instruct \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8183 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --reasoning-parser qwen-25-vl \
       --enable-mm