
export FLAGS_use_pd_disaggregation=1


export INFERENCE_MSG_QUEUE_ID=1
export FD_LOG_DIR="log_decode"
CUDA_VISIBLE_DEVICES=4,5,6,7 python fastdeploy.entrypoints.openai.api_server.py --config test.yaml --port 9812 --max-num-seqs 256 --kv-cache-ratio 0.8 --splitwise-role "decode"  --engine-worker-queue-port 6678 --innode-prefill-ports 6677 --cache-queue-port 55667 --enable-prefix-caching --enable-chunked-prefill &


export FD_LOG_DIR="log_prefill"
export INFERENCE_MSG_QUEUE_ID=3
export FLAGS_fmt_write_cache_completed_signal=1
export PREFILL_NODE_ONE_STEP_STOP=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python fastdeploy.entrypoints.openai.api_server.py --config test.yaml --port 9811 --cpu-offload-gb 5 --max-num-seqs 16 --kv-cache-ratio 0.9 --splitwise-role "prefill"  --engine-worker-queue-port 6677 --enable-prefix-caching --cache-queue-port 55663 &

