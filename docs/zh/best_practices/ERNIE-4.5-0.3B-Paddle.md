# ERNIE-4.5-0.3B
## 一、环境准备
### 1.1 支持情况
ERNIE-4.5-0.3B 各量化精度，在下列硬件上部署所需要的最小卡数如下：

|  | WINT8 | WINT4 | FP8 |
|-----|-----|-----|-----|
|H800 80GB| 1 | 1 | 1 |
|A800 80GB| 1 | 1 | / |
|H20 96GB| 1 | 1 | 1 |
|L20 48GB| 1 | 1 | 1 |
|A30 40GB| 1 | 1 | / |
|A10 24GB| 1 | 1 | / |

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
       --model baidu/ERNIE-4.5-0.3B-Paddle \
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

#### 2.2.2 CUDAGraph
**原理：**
CUDAGraph 是 NVIDIA 提供的一项 GPU 计算加速技术，通过将 CUDA 操作序列捕获（capture）为图结构（graph），实现 GPU 任务的高效执行和优化。CUDAGraph 的核心思想是将一系列 GPU 计算和内存操作封装为一个可重复执行的图，从而减少 CPU-GPU 通信开销、降低内核启动延迟，并提升整体计算性能。

**启用方式：**
在启动命令中增加
```
--use-cudagraph
```
注：
- 通常情况下不需要额外设置其他参数，但CUDAGraph会产生一些额外的显存开销，在一些显存受限的场景下可能需要调整。详细的参数调整请参考[GraphOptimizationBackend](../features/graph_optimization.md) 相关配置参数说明

#### 2.2.3 拒绝采样
**原理：**
拒绝采样即从一个易于采样的提议分布（proposal distribution）中生成样本，避免显式排序从而达到提升采样速度的效果，对小尺寸的模型有较明显的提升。

**启用方式：**
启动前增加下列环境变量
```
export FD_SAMPLING_CLASS=rejection
```

## 三、常见问题FAQ
如果您在使用过程中遇到问题，可以在[FAQ](./FAQ.md)中查阅。
