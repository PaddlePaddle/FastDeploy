[English](../../best_practices/PaddleOCR-VL-0.9B.md)

# PaddleOCR-VL-0.9B

## 一、环境准备
### 1.1 支持情况
推荐硬件配置：
- 显存：8GB显存及以上
- 共享内存：4G及以上

### 1.2 安装fastdeploy

安装流程参考文档 [FastDeploy GPU 安装](../get_started/installation/nvidia_gpu.md)

## 二、如何使用
### 2.1 基础：启动服务
 **示例1：** 3060上单卡部署16K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 128
```

 **示例2：** 4090上单卡部署16K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 256
```

 **示例3：** A100上单卡部署16K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8185 \
    --metrics-port 8186 \
    --engine-worker-queue-port 8187 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 256
```

示例是可以稳定运行的一组配置，同时也能得到比较好的性能。
如果对精度、性能有进一步的要求，请继续阅读下面的内容。
### 2.2 进阶：如何获取更优性能

#### 2.2.1 评估应用场景，正确设置参数
> **上下文长度**
- **参数：** `--max-model-len`
- **描述：** 控制模型可处理的最大上下文长度。
- **推荐：** 更长的上下文会导致吞吐降低，根据实际情况设置，`PaddleOCR-VL-0.9B`最长支持**16k**（16,384）长度的上下文。

   ⚠️ 注：更长的上下文会显著增加GPU显存需求，设置更长的上下文之前确保硬件资源是满足的。
> **最大序列数量**
- **参数：** `--max-num-seqs`
- **描述：** 控制服务可以处理的最大序列数量，支持1～256。
- **推荐：** 如果您不知道实际应用场景中请求的平均序列数量是多少，并且显存充足，我们建议设置为**256**。如果您的应用场景中请求的平均序列数量明显少于256，或者显存资源紧张，我们建议设置为一个略大于平均值的较小值，以进一步降低显存占用，优化服务性能。

> **初始化时可用的显存比例**
- **参数：** `--gpu-memory-utilization`
- **用处：** 用于控制 FastDeploy 初始化服务的可用显存，默认0.9，即预留10%的显存备用。
- **推荐：** 推荐使用0.7。如果服务压测时提示显存不足，可以尝试调低该值。

#### 2.2.2 Chunked Prefill
- **参数：** `--max-num-batched-tokens`
- **用处：**  `chunked prefill`中限制每个chunk的最大token数量。
- **推荐：** 我们推荐设置为16384，即关闭`chunked prefill`。

#### 2.2.3  **可调整的环境变量**
> **Flash Attention 3：**`FLAGS_flash_attn_version=3`
- **描述**：开启 Flash Attention 3 算法。该功能仅支持H卡（Hopper架构，如H800）和B卡（Blackwell架构，如B200）等最新一代 NVIDIA GPU。
- **推荐**：在支持该功能的硬件上，Flash Attention 3 能带来显著的性能提升且通常不影响模型精度，强烈推荐启用。

> **拒绝采样：**`FD_SAMPLING_CLASS=rejection`
- **描述**：拒绝采样即从一个易于采样的提议分布（proposal distribution）中生成样本，避免显式排序从而达到提升采样速度的效果，可以提升推理性能。
- **推荐**：这是一种影响效果的较为激进的优化策略，我们还在全面验证影响。如果对性能有较高要求，也可以接受对效果的影响时可以尝试开启。

## 三、常见问题FAQ

### 3.1 显存不足(OOM)
如果服务启动时提示显存不足，请尝试以下方法：
1. 确保无其他进程占用显卡显存；
2. 酌情降低上下文长度和最大序列数量。

如果可以服务可以正常启动，运行时提示显存不足，请尝试以下方法：
1. 酌情降低初始化时可用的显存比例，即调整参数 `--gpu-memory-utilization` 的值。
