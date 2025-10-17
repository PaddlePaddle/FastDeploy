[English](../../quantization/online_quantization.md)

# 在线量化

在线量化是指推理引擎在加载 BF16 权重后对权重做量化，而不是加载离线量化好的低精度权重。FastDeploy 支持将 BF16 在线量化到多种精度，包括：INT4, INT8 和 FP8.

## 1. WINT8 & WINT4

仅将权重在线量化为 INT8 或 INT4，推理时即时地将权重反量化为 BF16 后与激活进行计算。
- **量化粒度**：仅支持 channel-wise 粒度的量化；
- **支持硬件**：GPU，XPU
- **支持结构**：MoE 结构，Dense Linear

### 启动WINT8或WINT4推理服务

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8183 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

- 通过指定 `--model baidu/ERNIE-4.5-300B-A47B-Paddle` 可自动从AIStudio下载模型。FastDeploy依赖Paddle格式的模型，更多说明参考[支持模型列表](../supported_models.md)。
- 通过设置 `--quantization` 为 `wint8` 或 `wint4` 选择在线 INT8/INT4 量化。
- 部署 ERNIE-4.5-300B-A47B-Paddle WINT8 最少需要 80G *8卡, WINT4 则需要 80GB* 4卡。
- 更多部署教程请参考[get_started](../get_started/ernie-4.5.md).

## 2. Block-wise FP8

加载 BF16 模型，将权重以 128X128 block-wise 的粒度在线量化为 FP8 数值类型。推理时，激活会动态、即时地做 token-wise FP8 量化。

- **FP8规格**：float8_e4m3fn
- **支持硬件**：Hopper GPU 架构
- **支持结构**：MoE 结构，Dense Linear

### 启动Block-wise FP8推理服务

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8183 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization block_wise_fp8 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

- 通过指定 `--model baidu/ERNIE-4.5-300B-A47B-Paddle` 可自动从AIStudio下载模型。FastDeploy依赖Paddle格式的模型，更多说明参考[支持模型列表](../supported_models.md)。
- 通过设置 `--quantization` 为 `block_wise_fp8` 选择在线 Block-wise FP8 量化。
- 部署 ERNIE-4.5-300B-A47B-Paddle Block-wise FP8 最少需要 80G * 8卡。
- 更多部署教程请参考[get_started](../get_started/ernie-4.5.md)
