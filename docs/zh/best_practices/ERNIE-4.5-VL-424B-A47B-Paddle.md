[English](../../best_practices/ERNIE-4.5-VL-424B-A47B-Paddle.md)

# ERNIE-4.5-VL-424B-A47B-Paddle

## 一、环境准备
### 1.1 支持情况
在下列硬件上部署所需要的最小卡数如下：

| 设备[显存] | WINT4 | WINT8 | BFLOAT16 |
|:----------:|:----------:|:------:| :------:|
| H20 [144G] | 8 | 8 |  8 |
| A100 [80G] | 8 | 8 |  - |
| H800 [80G] | 8 | 8 |  - |

### 1.2 安装fastdeploy

安装流程参考文档 [FastDeploy GPU 安装](../get_started/installation/nvidia_gpu.md)

## 二、如何使用
### 2.1 基础：启动服务
 **示例1：** H800上8卡部署128K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-VL-424B-A47B-Paddle \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image": 100, "video": 100}' \
    --reasoning-parser ernie-45-vl \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 384 \
    --quantization wint4
```

示例是可以稳定运行的一组配置，同时也能得到比较好的性能。
如果对精度、性能有进一步的要求，请继续阅读下面的内容。
### 2.2 进阶：如何获取更优性能

#### 2.2.1 评估应用场景，正确设置参数
> **上下文长度**
- **参数：** `--max-model-len`
- **描述：** 控制模型可处理的最大上下文长度。
- **推荐：** 更长的上下文会导致吞吐降低，根据实际情况设置，`ERNIE-4.5-VL-424B-A47B-Paddle` 最长支持**128k**（131072）长度的上下文。

> **最大序列数量**
- **参数：** `--max-num-seqs`
- **描述：** 控制服务可以处理的最大序列数量，支持1～256。
- **推荐：** 128k场景下，80G显存的单机我们建议设置为**16**。

> **多图、多视频输入**
- **参数**：`--limit-mm-per-prompt`
- **描述**：我们的模型支持单次提示词（prompt）中输入多张图片和视频。请使用此参数限制每次请求的图片/视频数量，以确保资源高效利用。
- **推荐**：我们建议将单次提示词（prompt）中的图片和视频数量均设置为100个，以平衡性能与内存占用。

> **初始化时可用的显存比例**
- **参数：** `--gpu-memory-utilization`
- **用处：** 用于控制 FastDeploy 初始化服务的可用显存，默认0.9，即预留10%的显存备用。
- **推荐：** 128k长度的上下文时推荐使用0.8。如果服务压测时提示显存不足，可以尝试调低该值。

#### 2.2.2 Chunked Prefill
- **参数：** `--enable-chunked-prefill`
- **用处：** 开启 `chunked prefill` 可降低显存峰值并提升服务吞吐。2.2版本已经**默认开启**，2.2之前需要手动开启，参考2.1的最佳实践文档。

- **相关配置**:

    `--max-num-batched-tokens`：限制每个chunk的最大token数量。多模场景下每个chunk会向上取整保持图片的完整性，因此实际每次推理的总token数会大于该值。推荐设置为384。

#### 2.2.3  **量化精度**
- **参数：** `--quantization`

- **已支持的精度类型：**
  - WINT4 (适合大多数用户)
  - WINT8
  - BFLOAT16 (未设置 `--quantization` 参数时，默认使用BFLOAT16)

- **推荐：**
  - 除非您有极其严格的精度要求，否则我们建议使用WINT4量化。这将显著降低内存占用并提升吞吐量。
  - 若需要稍高的精度，可尝试WINT8。
  - 仅当您的应用场景对精度有极致要求时候才尝试使用BFLOAT16，因为它需要更多显存。

#### 2.2.4  **可调整的环境变量**
> **拒绝采样：**`FD_SAMPLING_CLASS=rejection`
- **描述**：拒绝采样即从一个易于采样的提议分布（proposal distribution）中生成样本，避免显式排序从而达到提升采样速度的效果，可以提升推理性能。
- **推荐**：这是一种影响效果的较为激进的优化策略，我们还在全面验证影响。如果对性能有较高要求，也可以接受对效果的影响时可以尝试开启。

## 三、常见问题FAQ
**注意：** 使用多模服务部署需要在配置中添加参数 `--enable-mm`。

### 3.1 显存不足(OOM)
如果服务启动时提示显存不足，请尝试以下方法：
1. 确保无其他进程占用显卡显存；
2. 使用WINT4/WINT8量化，开启chunked prefill；
3. 酌情降低上下文长度和最大序列数量。

如果可以服务可以正常启动，运行时提示显存不足，请尝试以下方法：
1. 酌情降低初始化时可用的显存比例，即调整参数 `--gpu-memory-utilization` 的值。
