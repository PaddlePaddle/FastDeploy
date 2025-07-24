# ERNIE-4.5-21B-A3B 最佳实践
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

### 1.2 安装fastdeploy与准备模型
- 安装：在开始部署前，请确保你的硬件环境满足如下条件：
```
GPU驱动 >= 535
CUDA >= 12.3
CUDNN >= 9.5
Linux X86_64
Python >= 3.10
```
针对SM 80/90的GPU（A30/A100/H100/）
```
# Install stable release
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install latest Nightly build
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
针对SM 86/89的GPU（A10/4090/L20/L40)
```
# Install stable release
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install latest Nightly build
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
安装详情，请参考[Fastdeploy Installation](../get_started/installation/README.md)完成安装。

- 模型下载，**请注意使用Fastdeploy部署需要Paddle后缀的模型**：
  - 执行时直接指定模型名（如`baidu/ERNIE-4.5-21B-A3B-Paddle`）即可自动下载，默认下载路径为 `~/`(即用户主目录)，也可以通过配置环境变量 `FD_MODEL_CACHE`修改默认下载的路径
  - 如受到网络或其他因素影响，也可以通过[huggingface](https://huggingface.co/)、[modelscope](https://www.modelscope.cn/home)等下载模型，并在启动时指定模型路径

## 二、启动服务

通过下列命令启动服务
```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Paddle \
       --tensor-parallel-size 1 \
       --quantization wint4 \
       --max-model-len 32768 \
       --kv-cache-ratio 0.75 \
       --max-num-seqs 128
```
其中：
- `--quantization`: 表示模型采用的量化策略。不同量化策略，模型的性能和精度也会不同。
- `--max-model-len`：表示当前部署的服务所支持的最长Token数量。设置得越大，模型可支持的上下文长度也越大，但相应占用的显存也越多，可能影响并发数。
- `--kv-cache-ratio`: 表示KVCache块按kv_cache_ratio比例分给Prefill阶段和Decode阶段。设置不合理会导致某个阶段的KVCache块不足，从而影响性能。如果开启服务管理全局Block功能，可以不用设置。

更多的参数含义与默认设置，请参见[FastDeploy参数说明](../parameters.md)。

### 2.2 进阶：如何获取更优性能
#### 2.2.1 评估应用场景，正确设置参数
结合应用场景，评估平均输入长度、平均输出长度、最大上下文长度。例如，平均输入长度为1000，输出长度为30000，那么建议设置为 32768
- 根据最大上下文长度，设置`max-model-len`
- **启用服务管理全局 Block**
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

#### 2.2.2 Prefix Caching
**原理：** Prefix Caching的核心思想是通过缓存输入序列的中间计算结果（KV Cache），避免重复计算，从而加速具有相同前缀的多个请求的响应速度。具体参考[prefix-cache](../features/prefix_caching.md)

**启用方式：**
在启动参数下增加下列两行，其中`--enable-prefix-caching`表示启用前缀缓存，`--swap-space`表示在GPU缓存的基础上，额外开启CPU缓存，大小为GB，应根据机器实际情况调整。
```
--enable-prefix-caching
--swap-space 50
```

#### 2.2.3 Chunked Prefill
**原理：** 采用分块策略，将预填充（Prefill）阶段请求拆解为小规模子任务，与解码（Decode）请求混合批处理执行。可以更好地平衡计算密集型（Prefill）和访存密集型（Decode）操作，优化GPU资源利用率，减少单次Prefill的计算量和显存占用，从而降低显存峰值，避免显存不足的问题。 具体请参考[Chunked Prefill](../features/chunked_prefill.md)

**启用方式：** 在启动参数下增加即可
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

#### 2.2.5 CUDAGraph
**原理：**
CUDAGraph 是 NVIDIA 提供的一项 GPU 计算加速技术，通过将 CUDA 操作序列捕获（capture）为图结构（graph），实现 GPU 任务的高效执行和优化。CUDAGraph 的核心思想是将一系列 GPU 计算和内存操作封装为一个可重复执行的图，从而减少 CPU-GPU 通信开销、降低内核启动延迟，并提升整体计算性能。

**启用方式：**
在启动命令中增加
```
--use-cudagraph
```
注：
1. 通常情况下不需要额外设置其他参数，但CUDAGraph会产生一些额外的显存开销，在一些显存受限的场景下可能需要调整。详细的参数调整请参考[GraphOptimizationBackend](../parameters.md) 相关配置参数说明
2. 开启CUDAGraph时，暂时只支持单卡推理，即`--tensor-parallel-size 1`
3. 开启CUDAGraph时，暂时不支持同时开启`Chunked Prefill`和`Prefix Caching`

#### 2.2.6 拒绝采样
**原理：**
拒绝采样即从一个易于采样的提议分布（proposal distribution）中生成样本，避免显式排序从而达到提升采样速度的效果，对小尺寸的模型有较明显的提升。

**启用方式：**
启动前增加下列环境变量
```
export FD_SAMPLING_CLASS=rejection
```

## 三、常见问题FAQ
如果您在使用过程中遇到问题，可以在[FAQ](./FAQ.md)中查阅。
