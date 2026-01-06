# MooncakeStore for FastDeploy

This document describes how to use MooncakeStore as the backend of FastDeploy.

## Preparation

### Install FastDeploy

Refer to [NVIDIA CUDA GPU Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/) for Fastdeploy installation.

### Install MooncakeStore

```bash
pip install mooncake-transfer-engine
```

## Run Examples

The example script is provided in `run.sh`. You can run it directly:
```
bash run.sh
```

In the example script, we will start a Mooncake master server and two FastDeploy server.

Launch Mooncake master server:
```bash
mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true  \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002 \
    --metrics_port=15003 \
```

More parameter can be found in the [official guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/python-api-reference/transfer-engine.md).

Launch the Fastdeploy with Mooncake enabled.

```bash
export MOONCAKE_CONFIG_PATH="./mooncake_config.json"

python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${PORT} \
       --metrics-port $((PORT + 1)) \
       --engine-worker-queue-port $((PORT + 2)) \
       --cache-queue-port $((PORT + 3)) \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake
```

## Troubleshooting

For more details, please refer to:
https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md
