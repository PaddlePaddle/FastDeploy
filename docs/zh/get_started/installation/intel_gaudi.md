[English](../../../get_started/installation/intel_gaudi.md)

# 使用 Intel Gaudi 运行ERNIE 4.5 系列模型

在环境满足如下条件前提下

- Python 3.10
- Intel Gaudi 2
- Intel Gaudi software version 1.22.0
- Linux X86_64

## 1. 运行Docker容器

使用下面命令运行Docker容器. 确保更新的版本在如下列表中 [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html):

```{.console}
$ docker pull vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
```

### 2. 安装 PaddlePaddle

```bash
python -m pip install paddlepaddle==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### 3. 安装 PaddleCustomDevice
```shell
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice/backends/intel_hpu/
mkdir -p build
cd build
cmake ..
make -j
pip install --force-reinstall dist/paddle_intel_hpu*.whl
cd PaddleCustomDevice/backends/intel_hpu/custom_ops
python setup.py install
```

### 4. 安装 FastDeploy

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
bash build.sh
```

## 准备推理示例

### 1. 启动推理服务
```shell
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
export HABANA_PROFILE=0
export HPU_VISIBLE_DEVICES=0

#WARMUP Enabled
HPU_WARMUP_BUCKET=1 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=HPU_ATTN python -m fastdeploy.entrypoints.openai.api_server --model ERNIE-4.5-21B-A3B-Paddle --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128

#WARMUP Disabled
HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=HPU_ATTN python -m fastdeploy.entrypoints.openai.api_server --model ERNIE-4.5-21B-A3B-Paddle --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 --graph-optimization-config '{"use_cudagraph":false}'
```

### 2. 发送请求
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "What is AI?"}
  ], "max_tokens": 24
}'
```

### 3. 成功返回结果
```json
{"id":"chatcmpl-3bd98ae2-fafe-46ae-a552-d653a8526503","object":"chat.completion","created":1757653575,"model":"ERNIE-4.5-21B-A3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"**AI (Artificial Intelligence)** refers to the development of computer systems that can perform tasks typically requiring human intelligence.","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":11,"total_tokens":35,"completion_tokens":24,"prompt_tokens_details":{"cached_tokens":0}}}
```

## Tensorwise FP8 量化模型

Intel® Gaudi® 支持 FP8 数据类型, 目前 intel_HPU 支持 FastDeploy 运行于 `tensor_wise_fp8` 量化模式. 使用此模式进行推理的整体流程如下:
- 将现有 BF16 模型转化为 FP8 量化模型
  - 测量 BF16 模型运行时 activation 的范围, 生成校准文件
  - 将相关权重静态量化为 FP8 格式, 与校准文件共同生成 FP8 量化模型.
- 执行 FP8 模型推理.

具体流程如下:

### 1. 生成校准文件

设置环境变量, 通过 offline_demo.py 脚本运行 BF16 模型.

``` bash
export FD_HPU_MEASUREMENT_MODE=1
```

该过程会自动触发模型运行于测量模式, 并将统计结果合并更新在 `model_measurement.txt` 文件中, 支持多卡和多次测量 (多卡模式下会生成多个测量文件, 但最终会合并生成统一量化模型, 模型文件本身不会根据卡数拆分).

### 2. 离线生成 FP8 模型

运行位于 PaddleCustomDevice 的 [Model_convert.py](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/intel_hpu/tools/Model_convert.py) 脚本, 将模型相关模块 `weights` 静态量化为 FP8 数据类型, 同时写入对应的 `weight_scale`. 将测量校准的 `activation_scale` 同时记录到 FP8 模型文件当中.

``` bash
python Model_convert.py [bf16_model_path] [fp8_model_path] [model_measurement_name_or_path] <ranks_total_number>
```

- `bf16_model_path`: BF16 模型输入路径
- `fp8_model_path`: FP8 模型输出路径
- `model_measurement_name_or_path`: 测量生成文件名或文件夹.
  - 可以是单一测量文件
  - 可以是包含所有测量文件 (如多次测量或多卡) 的文件夹
  - 可以是多个测量文件的文件名前缀, 会根据 `ranks_total_number` 自动添加 `_{i}` 序号后缀
- `ranks_total_number`: 可选, 配合 `model_measurement_name_or_path` 为文件名前缀时使用

### 3. 运行 FP8 模型

所有 `tensor_wise_fp8` 相关配置会通过 Model_convert.py 脚本自动配置在 config.json 中. 运行时只要指定 `model_name_or_path` 为离线生成的 FP8 模型 (`fp8_model_path`) 即可, 不需要额外参数配置.
