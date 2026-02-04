[English](../../best_practices/ERNIE-4.5-0.3B-Paddle.md)

# Deepseek-V3.1-Terminus
## 一、环境准备
### 1.1 支持情况
DeepSeek-V3.1 为大规模 MoE 架构模型，对显存与多卡并行有较高要求。不同量化精度在下列硬件上的最小部署卡数如下（参考值）：

|  | WINT8 | WINT4 | FP8 | W4AFP8 |
|-----|-----|-----|-----|-----|
|H800 80GB| 16 | 8 | 16 | TODO |
|A800 80GB| 8 | 8 | / | TODO |

**注：**
1. 在启动命令后指定`--tensor-parallel-size 1` 即可修改部署卡数
2. 表格中未列出的硬件，可根据显存大小进行预估是否可以部署

### 1.2 安装fastdeploy
- 安装请参考[Fastdeploy Installation](../get_started/installation/README.md)完成安装。

- 模型下载，请参考[支持模型列表](../supported_models.md)。

## 二、如何使用
### 2.1 基础：启动服务
通过下列命令启动服务
```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model deepseek-ai/DeepSeek-V3.1-Terminus \
       --port 8180 \
       --tensor-parallel-size 8 \
       --max-model-len  32768 \
       --max-num-seq 40 \
       --max-num-batched-tokens 32768 \
       --quantization wint4 &
```
其中：
- `--quantization`: 量化策略，可选：
       - WINT4 (适合大多数用户)
       - WINT8
       - BFLOAT16 (未设置 `--quantization` 参数时，默认使用BFLOAT16)
- **推荐：**
       - 除非您有极其严格的精度要求，否则我们建议使用WINT4量化。这将显著降低内存占用并提升吞吐量。
       - 若需要稍高的精度，可尝试WINT8。
       - 仅当您的应用场景对精度有极致要求时候才尝试使用BFLOAT16，因为它需要更多显存。
- `--max-model-len`：表示当前部署的服务所支持的最长Token数量。设置得越大，模型可支持的上下文长度也越大，但相应占用的显存也越多，可能影响并发数。

更多的参数含义与默认设置，请参见[FastDeploy参数说明](../parameters.md)。

### 2.2 进阶：如何获取更优性能
#### 2.2.1 评估应用场景，正确设置参数
> **最大上下文长度**
- 在部署前应结合实际业务场景评估以下指标：
       - 平均输入长度
       - 平均输出长度
       - 最大上下文长度
- 根据最大上下文长度，设置`max-model-len`
- 例如：平均输入长度为 1000 tokens，平均输出长度为 30000 tokens，建议将 `max-model-len` 设置为 32768

> **最大序列数量**
- **参数：** `--max-num-seqs`
- **描述：** 控制服务可以处理的最大序列数量，支持1～256。
- **推荐：** 128k场景下，80G显存的单机我们建议设置为**16**。

> **初始化时可用的显存比例**
- **参数：** `--gpu-memory-utilization`
- **用处：** 用于控制 FastDeploy 初始化服务的可用显存，默认0.9，即预留10%的显存备用。
- **推荐：** 128k长度的上下文时推荐使用0.8。如果服务压测时提示显存不足，可以尝试调低该值。

#### 2.2.2 Prefix Caching
**原理：** Prefix Caching 通过缓存具有相同前缀输入的 KV Cache 计算结果，避免重复 Prefill 计算，从而显著提升具有固定 system prompt 或模板化输入场景下的推理性能。具体参考[prefix-cache](../features/prefix_caching.md)

**启用方式：**
自2.2版本开始（包括develop分支），Prefix Caching已经默认开启。

对于2.1及更早的版本，需要手动开启。其中`--enable-prefix-caching`表示启用前缀缓存，`--swap-space`表示在GPU缓存的基础上，额外开启CPU缓存，大小为GB，应根据机器实际情况调整。建议取值为`(机器总内存 - 模型大小) * 20%`。如果因为其他程序占用内存等原因导致服务启动失败，可以尝试减小`--swap-space`的值。
```
--enable-prefix-caching
--swap-space 50
```

#### 2.2.3 Chunked Prefill
**原理：** 采用分块策略，将预填充（Prefill）阶段请求拆解为小规模子任务，与解码（Decode）请求混合批处理执行。可以更好地平衡计算密集型（Prefill）和访存密集型（Decode）操作，优化GPU资源利用率，减少单次Prefill的计算量和显存占用，从而降低显存峰值，避免显存不足的问题。 具体请参考[Chunked Prefill](../features/chunked_prefill.md)

- **参数：** `--enable-chunked-prefill`
- **用处：** 开启 `chunked prefill` 可降低显存峰值并提升服务吞吐。2.2版本已经**默认开启**，2.2之前需要手动开启，参考2.1的最佳实践文档。

- **相关配置**:

    `--max-num-batched-tokens`：限制每个chunk的最大token数量。多模场景下每个chunk会向上取整保持图片的完整性，因此实际每次推理的总token数会大于该值。推荐设置为384。

## 三、常见问题FAQ
如果您在使用过程中遇到问题，可以在[FAQ](./FAQ.md)中查阅。
