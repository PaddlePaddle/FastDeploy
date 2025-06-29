# 如何在天数机器上运行 ERNIE-4.5-300B-A47B-BF16
当前版本软件只是作为天数芯片 + Fastdeploy 推理大模型的一个演示 demo，跑最新ERNIE4.5模型可能存在问题，后续进行修复和性能优化，给客户提供一个更稳定的版本。

## 准备机器
首先您需要准备以下配置的机器
| CPU | 内存 | 天数 | 硬盘|
|-----|------|-----|-----|
| x86 | 1TB| 8xBI150| 1TB|

目前需要将完整模型 load 到 host memory 中，需要需要大于 600GB 的 host memory，后续版本会优化。

## 镜像
从官网获取：

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

## 准备容器
1. 启动容器
```bash
docker run -itd --name paddle_infer -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle_infer bash
```
/home/paddle 为模型文件、whl包、脚本所在目录

2. 安装whl包

```bash
pip3 install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip3 install paddle-iluvatar-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
pip3 install fastdeploy -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
pip3 install aistudio-sdk==0.2.6
```

## 准备推理demo脚本
推理 demo 路径：/home/paddle/scripts
脚本内容如下

`run_demo.sh`:
```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export USE_WORKER_V1=1
export INFERENCE_MSG_QUEUE_ID=232132
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export FD_DEBUG=1
python3 run_demo.py
```

run_demo.py


```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]

# 采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# 加载模型
llm = LLM(model="/home/paddle/ernie-4_5-300b-a47b-bf16-paddle", tensor_parallel_size=16, max_model_len=8192)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)
# 注意将其中`/home/paddle/ernie-4_5-300b-a47b-bf16-paddle`替换为您下载的ERNIE模型的路径。
# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt, generated_text)
```

## 运行demo
执行
```bash
./run_demo.sh
```
会有如下 log 打印；load 模型耗时约470s，demo 运行约90s。
```
/usr/local/lib/python3.10/site-packages/paddle/utils/cpp_extension/extension_utils.py:715: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
/usr/local/lib/python3.10/site-packages/_distutils_hack/__init__.py:31: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-06-27 16:35:10,856] [    INFO] - Loading configuration file /home/paddle/ernie-45t/generation_config.json
/usr/local/lib/python3.10/site-packages/paddlenlp/generation/configuration_utils.py:250: UserWarning: using greedy search strategy. However, `temperature` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/paddlenlp/generation/configuration_utils.py:255: UserWarning: using greedy search strategy. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
INFO     2025-06-27 16:35:12,205 2717757 engine.py[line:134] Waitting worker processes ready...
Loading Weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 18.13it/s]
Loading Layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 199.50it/s]
[2025-06-27 16:35:24,030] [ WARNING] - import EventHandle and deep_ep Failed!
[2025-06-27 16:35:24,032] [ WARNING] - import EventHandle and deep_ep Failed!
INFO     2025-06-27 16:43:02,392 2717757 engine.py[line:700] Stop profile, num_gpu_blocks:  1820
INFO     2025-06-27 16:43:02,393 2717757 engine.py[line:175] Worker processes are launched with 471.5467264652252 seconds.
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:29<00:00, 89.98s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Hello, my name is Hello! It's nice to meet you. I'm here to help with questions, have conversations, or assist with whatever you need. What would you like to talk about today? 😊
```
