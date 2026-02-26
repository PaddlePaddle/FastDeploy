[English](../../best_practices/DeepSeek-V3-V3.1.md)

# DeepSeek-V3/V3.1 模型

## 一、环境准备
### 1.1 支持情况
DeepSeek-V3/V3.1 各量化精度，在下列硬件上部署所需要的最小卡数如下：

|     | WINT4 |
|-----|-----|
|H800 80GB| 8 |

### 1.2 安装fastdeploy

安装流程参考文档 [FastDeploy GPU 安装](../get_started/installation/nvidia_gpu.md)

## 二、如何使用
### 2.1 基础：启动服务
 **示例1：** H800上八卡部署wint4模型16K上下文的服务
```shell
MODEL_PATH=/models/DeepSeek-V3.2-Exp-BF16

export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3

python -m fastdeploy.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --cache-queue-port 8183 \
  --tensor-parallel-size 8 \
  --max-model-len  16384 \
  --max-num-seq 100 \
  --no-enable-prefix-caching \
  --quantization wint4

```
