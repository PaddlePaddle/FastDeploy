# Run ERNIE-4.5-300B-A47B model on iluvatar machine
The current version of the software merely serves as a demonstration demo for the Iluvatar CoreX combined with the Fastdeploy inference framework for large models. There may be issues when running the latest ERNIE4.5 model, and we will conduct repairs and performance optimization in the future. Subsequent versions will provide customers with a more stable version.

##  Machine Preparation
First, you need to prepare a machine with the following configurations:

| CPU | Memory | Card | Hard Disk|
| :---: | :---: | :---: | :---: |
| x86 | 1TB| 8xBI150| 1TB|

Currently, the entire model needs to be loaded into the host memory, which requires more than 600GB of host memory. This issue will be optimized in subsequent versions.

## Image Preparation
Pull the Docker image

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

## Container Preparation
1. Start Container
```bash
docker run -itd --name paddle_infer -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle_infer bash
```
/home/paddle contains the model files, *.whl packages, and scripts.

2. Install packages

```bash
pip3 install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip3 install paddle-iluvatar-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
pip3 install fastdeploy -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
pip3 install aistudio-sdk==0.2.6
```

## Prepare the inference demo script

script list below:

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

`run_demo.py`:

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]

# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# load the model
llm = LLM(model="/home/paddle/ernie-4_5-300b-a47b-bf16-paddle", tensor_parallel_size=16, max_model_len=8192)

# Perform batch inference
outputs = llm.generate(prompts, sampling_params)
# Note：Rlace `/home/paddle/ernie-4_5-300b-a47b-bf16-paddle` in it with the path to the ERNIE model you have downloaded.。

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt, generated_text)
```

## run demo

```bash
./run_demo.sh
```
The following logs will be printed: Loading the model took approximately 470 seconds, and running the demo took approximately 90 seconds.
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
