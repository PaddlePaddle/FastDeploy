[English](../../best_practices/ERNIE-4.5-21B-A3B-Paddle.md)

# ERNIE-4.5-21B-A3B
## 一、环境准备
### 1.1 支持情况
ERNIE-4.5-21B-A3B 各量化精度，在下列硬件上部署所需要的最小卡数如下：

|  | WINT8 | WINT4 | FP8 |
|-----|-----|-----|-----|
|H800 80GB| 1 | 1 | 1 |
|A800 80GB| 1 | 1 | / |
|H20 96GB| 1 | 1 | 1 |
|L20 48GB| 1 | 1 | 1 |
|A30 40GB| 2 | 1 | / |
|A10 24GB| 2 | 1 | / |

**注：**
1. 在启动命令后指定`--tensor-parallel-size 2` 即可修改部署卡数
2. 表格中未列出的硬件，可根据显存大小进行预估是否可以部署

### 1.2 安装fastdeploy
- 安装，请参考[Fastdeploy Installation](../get_started/installation/README.md)完成安装。

- 模型下载，请参考[支持模型列表](../supported_models.md)。

## 二、如何使用
### 2.1 基础：启动服务
通过下列命令启动服务
```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Paddle \
       --tensor-parallel-size 1 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 128
```
其中：
- `--quantization`: 表示模型采用的量化策略。不同量化策略，模型的性能和精度也会不同。可选值包括：`wint8` / `wint4` / `block_wise_fp8`(需要Hopper架构)。
- `--max-model-len`：表示当前部署的服务所支持的最长Token数量。设置得越大，模型可支持的上下文长度也越大，但相应占用的显存也越多，可能影响并发数。

更多的参数含义与默认设置，请参见[FastDeploy参数说明](../parameters.md)。

### 2.2 进阶：如何获取更优性能
#### 2.2.1 评估应用场景，正确设置参数
结合应用场景，评估平均输入长度、平均输出长度、最大上下文长度。例如，平均输入长度为1000，输出长度为30000，那么建议设置为 32768
- 根据最大上下文长度，设置`max-model-len`

#### 2.2.2 Prefix Caching
**原理：** Prefix Caching的核心思想是通过缓存输入序列的中间计算结果（KV Cache），避免重复计算，从而加速具有相同前缀的多个请求的响应速度。具体参考[prefix-cache](../features/prefix_caching.md)

**启用方式：**
自2.2版本开始（包括develop分支），Prefix Caching已经默认开启。

对于2.1及更早的版本，需要手动开启。其中`--enable-prefix-caching`表示启用前缀缓存，`--swap-space`表示在GPU缓存的基础上，额外开启CPU缓存，大小为GB，应根据机器实际情况调整。建议取值为`(机器总内存 - 模型大小) * 20%`。如果因为其他程序占用内存等原因导致服务启动失败，可以尝试减小`--swap-space`的值。
```
--enable-prefix-caching
--swap-space 50
```

#### 2.2.3 Chunked Prefill
**原理：** 采用分块策略，将预填充（Prefill）阶段请求拆解为小规模子任务，与解码（Decode）请求混合批处理执行。可以更好地平衡计算密集型（Prefill）和访存密集型（Decode）操作，优化GPU资源利用率，减少单次Prefill的计算量和显存占用，从而降低显存峰值，避免显存不足的问题。 具体请参考[Chunked Prefill](../features/chunked_prefill.md)

**启用方式：**
自2.2版本开始（包括develop分支），Chunked Prefill已经默认开启。

对于2.1及更早的版本，需要手动开启。
```
--enable-chunked-prefill
```

#### 2.2.4 MTP (Multi-Token Prediction)
**原理：**
通过一次性预测多个Token，减少解码步数，以显著加快生成速度，同时通过一定策略保持生成质量。具体请参考[投机解码](../features/speculative_decoding.md)。

**启用方式：**
在启动参数下增加即可
```
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}'
```

注：
1. MTP当前暂不支持与Prefix Caching 、Chunked Prefill 、CUDAGraph同时使用。
   - 需要通过指定`export FD_DISABLE_CHUNKED_PREFILL=1` 关闭Chunked Prefill。
   - 指定`speculative-config`时，会自动关闭Prefix Caching功能。
2. MTP当前暂不支持服务管理全局 Block， 指定`speculative-config`时，会自动关闭全局Block调度器。
3. MTP当前暂不支持和拒绝采样同时使用，即不要开启`export FD_SAMPLING_CLASS=rejection`

#### 2.2.5 CUDAGraph
**原理：**
CUDAGraph 是 NVIDIA 提供的一项 GPU 计算加速技术，通过将 CUDA 操作序列捕获（capture）为图结构（graph），实现 GPU 任务的高效执行和优化。CUDAGraph 的核心思想是将一系列 GPU 计算和内存操作封装为一个可重复执行的图，从而减少 CPU-GPU 通信开销、降低内核启动延迟，并提升整体计算性能。

**启用方式：**
在2.3版本之前需要通过`--use-cudagraph`启用。

2.3版本开始部分场景已默认开启 CUDAGraph，对于暂时不能兼容 CUDAGraph 的功能（投机解码、强化学习训练、多模模型推理）CUDAGraph 会自动关闭。
注：
- 通常情况下不需要额外设置其他参数，但CUDAGraph会产生一些额外的显存开销，在一些显存受限的场景下可能需要调整。详细的参数调整请参考[GraphOptimizationBackend](../features/graph_optimization.md) 相关配置参数说明

#### 2.2.6 拒绝采样
**原理：**
拒绝采样即从一个易于采样的提议分布（proposal distribution）中生成样本，避免显式排序从而达到提升采样速度的效果，对小尺寸的模型有较明显的提升。

**启用方式：**
启动前增加下列环境变量
```
export FD_SAMPLING_CLASS=rejection
```

#### 2.2.7 分离式部署
**原理：** 分离式部署的核心思想是将Prefill 和 Decode 分开部署，在一定场景下可以提高硬件利用率，有效提高吞吐，降低整句时延。具体请参考分离式部署

**启用方式：** 以单机8GPU，1P1D（各4GPU）部署为例，与默认的混合式部署方式相比， 需要`--splitwise-role`指定节点的角色。并通过环境变量`FD_LOG_DIR`和`CUDA_VISIBLE_DEVICES`将两个节点的GPU 和日志隔离开
```
# prefill
export CUDA_VISIBLE_DEVICES=0,1,2,3
export INFERENCE_MSG_QUEUE_ID=1315
export FD_ATTENTION_BACKEND=FLASH_ATTN
export FD_LOG_DIR="prefill_log"

quant_type=block_wise_fp8

python -m fastdeploy.entrypoints.openai.api_server --model baidu/ERNIE-4.5-21B-A3B-Paddle \
    --max-model-len 131072 \
    --max-num-seqs 20 \
    --num-gpu-blocks-override 40000 \
    --quantization ${quant_type} \
    --gpu-memory-utilization 0.9 --kv-cache-ratio 0.9 \
    --port 7012 --engine-worker-queue-port 7013 --metrics-port 7014 --tensor-parallel-size 4 \
    --cache-queue-port 7015 \
    --splitwise-role "prefill" \
```
```
# decode
export CUDA_VISIBLE_DEVICES=4,5,6,7
export INFERENCE_MSG_QUEUE_ID=1215
export FD_LOG_DIR="decode_log"

quant_type=block_wise_fp8

python -m fastdeploy.entrypoints.openai.api_server --model baidu/ERNIE-4.5-21B-A3B-Paddle \
    --max-model-len 131072 \
    --max-num-seqs 20 \
    --quantization ${quant_type} \
    --gpu-memory-utilization 0.85 --kv-cache-ratio 0.1 \
    --port 9012 --engine-worker-queue-port 8013 --metrics-port 8014 --tensor-parallel-size 4 \
    --cache-queue-port 8015 \
    --innode-prefill-ports 7013 \
    --splitwise-role "decode"
```

## 三、常见问题FAQ
如果您在使用过程中遇到问题，可以在[FAQ](./FAQ.md)中查阅。
