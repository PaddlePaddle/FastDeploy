# MooncakeStore for FastDeploy

This document describes how to use MooncakeStore as the backend of FastDeploy for L3 Cache.

## Installation

### Install MooncakeStore with pip

```bash
pip install mooncake-transfer-engine
```

### Install MooncakeStore from source

```bash
git clone https://github.com/kvcache-ai/Mooncake --recursive
cd Mooncake
```

Install dependencies

```bash
cd Mooncake
bash dependencies.sh
```

Build the project. For additional build options, please refer to [the official guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html).

```bash
mkdir build
cd build
cmake ..
make -j
sudo make install
```

## Use Mooncake

Launch Mooncake master server:

```bash
mooncake_master \
    --enable_http_metadata_server=true  \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=7882 \
    --metrics_port=7883 \
    --port=7721
```

### Command line options
```
-metrics_port (Port for HTTP metrics server to listen on) type: int32
    default: 9003
-enable_http_metadata_server (Enable HTTP metadata server instead of etcd)
    type: bool default: false
-http_metadata_server_host (Host for HTTP metadata server to bind to)
    type: string default: "0.0.0.0"
-http_metadata_server_port (Port for HTTP metadata server to listen on)
    type: int32 default: 8080
-eviction_high_watermark_ratio (Ratio of high watermark trigger eviction)
    type: double default: 0.94999999999999996
```

more parameter can be found in the [official guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/python-api-reference/transfer-engine.md).

Start the Fastdeploy with Mooncake enabled. Mooncake configuration can be provided via environment variables:

```bash
MOONCAKE_CONFIG_PATH="./mooncake_config.json" \
python -m fastdeploy.entrypoints.openai.api_server \
    --enable-hierarchical-kvcache \
    --kvcache-storage-backend mooncake \
    --model-path [model_path]
```

## Troubleshooting

For more details, please refer to:
https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/troubleshooting.md
