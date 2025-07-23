
# ERNIE-4.5-VL-28B-A3B-Paddle

## 一、环境准备
###  1.1 支持情况

| 设备[显存] | 可运行的量化精度 | 支持卡数 |
|:----------:|:----------:|:------:|
| A30 [24G] | wint4<br>wint8<br>bfloat16 | 2, 4<br>2, 4<br>4  |
| L20 [48G] | wint4<br>wint8<br>bfloat16 | 1, 2, 4<br>1, 2, 4<br>2, 4  |
| H20 [144G] | wint4<br>wint8<br>bfloat16 | 1, 2, 4<br>1, 2, 4<br>1, 2, 4 |
| A100 [80G] | wint4<br>wint8<br>bfloat16 |  1, 2, 4<br>1, 2, 4<br>1, 2, 4 |
| H800 [80G] | wint4<br>wint8<br>bfloat16 | 1, 2, 4<br>1, 2, 4<br>1, 2, 4 |

###  1.2 安装fastdeploy

安装流程参考文档 [FastDeploy GPU 安装](../get_started/installation/nvidia_gpu.md)

> ⚠️ 注意事项
> - FastDeploy只支持Paddle格式的模型，注意下载Paddle后缀的模型
> - 使用模型名称会自动下载模型，如果已经下载过模型，可以直接使用模型下载位置的绝对路径

## 二、如何使用
###  2.1 基础：启动服务
 **示例1：** 4090上单卡部署32K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 256 --limit-mm-per-prompt '{"image": 100, "video": 100}' --reasoning-parser ernie-45-vl --gpu-memory-utilization 0.9 --kv-cache-ratio 0.75 --enable-chunked-prefill --max-num-batched-tokens 384 --quantization wint4 --enable-mm
```
 **示例2：** H800上双卡部署128K上下文的服务
```shell
python -m fastdeploy.entrypoints.openai.api_server --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle --port 8180 --metrics-port 8181 --engine-worker-queue-port 8182 --tensor-parallel-size 2 --max-model-len 131072 --max-num-seqs 256 --limit-mm-per-prompt '{"image": 100, "video": 100}' --reasoning-parser ernie-45-vl --gpu-memory-utilization 0.9 --kv-cache-ratio 0.75 --enable-chunked-prefill --max-num-batched-tokens 384 --quantization wint4 --enable-mm
```
示例是可以稳定运行的一组配置，同时也能得到比较好的性能。
如果对精度、性能有进一步的要求，请继续阅读下面的内容。
###  2.2 进阶：如何获取更优性能

#### 2.2.1 评估应用场景，正确设置参数
> **上下文长度**  
- **参数：** `--max-model-len`  
- **描述：** 控制模型可处理的最大上下文长度。
- **推荐：** 更长的上下文会导致吞吐降低，根据实际情况设置，最长支持**128k**（131072）长度的上下文。

   ⚠️ 注：更长的上下文会显著增加GPU显存需求，设置更长的上下文之前确保硬件资源是满足的。
>  **最大序列数量**  
- **参数：** `--max-num-seqs`  
- **描述：** 控制服务可以处理的最大序列数量，支持1～256。
- **推荐：** 如果您不知道实际应用场景中请求的平均序列数量是多少，我们建议设置为**256**。如果您的应用场景中请求的平均序列数量明显少于256，我们建议设置为一个略大于平均值的较小值，以进一步降低显存占用，优化服务性能。

> **多图、多视频输入**  
- **参数**：`--limit-mm-per-prompt`
- **描述**：我们的模型支持单次提示词（prompt）中输入多张图片和视频。请使用此参数限制每次请求的图片/视频数量，以确保资源高效利用。
- **推荐**：我们建议将单次提示词（prompt）中的图片和视频数量均设置为100个，以平衡性能与内存占用。

> **初始化时可用的显存比例**  
- **参数：** `--gpu-memory-utilization`
- **用处：** 用于控制 FastDeploy 初始化服务的可用显存，默认0.9，即预留10%的显存备用。
- **推荐：** 推荐使用默认值0.9。如果服务压测时提示显存不足，可以尝试调低该值。

> **输入cache_kv比例**
- **参数：**`--kv-cache-ratio`
- **用处：** 用于控制 kv cache 显存的分配比例，默认0.75，即75%的 kv cache 显存给输入。
- **推荐：** 理论最佳值应设置为应用场景的 平均输入长度/(平均输入长度+平均输出长度)，如果您无法确定，保持默认值即可。

#### 2.2.2 Chunked Prefill
- **参数：** `--enable-chunked-prefill`
- **用处：** 开启 `chunked prefill` 可**降低显存峰值**并**提升服务吞吐**。

- **其他相关配置**:
   
    `--max-num-batched-tokens`：限制每个chunk的最大token数量，推荐设置为384。

#### 2.2.3  **量化精度**
- **参数：** `--quantization`

- **已支持的精度类型：**
  - wint4 (适合大多数用户)
  - wint8
  - bfloat16 (未设置 `--quantization` 参数时，默认使用bfloat16)

- **推荐：** 
    - 除非您有极其严格的精度要求，否则我们强烈建议使用wint4量化。这将显著降低内存占用并提升吞吐量。
    - 若需要稍高的精度，可尝试wint8。
    - 仅当您的应用场景对精度有极致要求时候才尝试使用bfloat16，因为它需要更多显存。

## 三、常见问题FAQ
**注意：** 使用多模服务部署需要在配置中添加参数 `--enable-mm`。

###  3.1 显存不足(OOM)
如果服务启动时提示显存不足，请尝试以下方法：
1. 确保无其他进程占用显卡显存；
2. 使用wint4/wint8量化，开启chunked prefill；
3. 酌情降低上下文长度和最大序列数量；
4. 增加部署卡数，使用2卡或4卡部署，即修改参数 `--tensor-parallel-size 2` 或 `--tensor-parallel-size 4`。

如果可以服务可以正常启动，运行时提示显存不足，请尝试以下方法：
1. 酌情降低初始化时可用的显存比例，即调整参数 `--gpu-memory-utilization` 的值；
2. 增加部署卡数，参数修改同上。

###  3.2 部分输出不完整或为空
检查日志中是否有以下提示：
> Message: 'recovery stop signal found at task chatcmpl-xx' \n Arguments: ('token_ids: [-3]',)

如果出现了上述日志，说明有请求触发recover stop返回，通过降低kv-cache-ratio，可以缓解该问题。
如果不要求极致性能，为了保证结果都能完整返回，可以使用以下方法避免该问题：
    
    配置--kv-cache-ratio=1.0, --static-decode-blocks=最大输出长度/64。
    例如128K的输出下，配置--static-decode-blocks=2048。