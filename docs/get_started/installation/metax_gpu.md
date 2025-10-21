[简体中文](../../zh/get_started/installation/metax_gpu.md)

# Metax GPU Installation for running ERNIE 4.5 Series Models

The following installation methods are available when your environment meets these requirements:
- Python >= 3.10
- Linux X86_64

Before starting, prepare a machine equipped with Enflame S60 accelerator cards. Requirements:

| Chip Type | Driver Version | KMD Version |
| :---: | :---: | :---: |
| MetaX C550 | 3.0.0.1  | 2.14.6 |

## 1. Pre-built Docker Installation (Recommended)

```shell
docker login --username=cr_temp_user --password=eyJpbnN0YW5jZUlkIjoiY3JpLXpxYTIzejI2YTU5M3R3M2QiLCJ0aW1lIjoiMTc1NTUxODEwODAwMCIsInR5cGUiOiJzdWIiLCJ1c2VySWQiOiIyMDcwOTQwMTA1NjYzNDE3OTIifQ:8226ca50ce5476c42062e24d3c465545de1c1780 cr.metax-tech.com && docker pull cr.metax-tech.com/public-library/maca-native:3.0.0.4-ubuntu20.04-amd64
```

## 2. paddlepaddle and custom device installation

```shell
1）pip install paddlepaddle==3.0.0.dev20250825 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
2）pip install paddle-metax-gpu==3.0.0.dev20250826 -i https://www.paddlepaddle.org.cn/packages/nightly/maca/
```

## 3. Build Wheel from Source
Then clone the source code and build:
```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
bash build.sh
```
The built packages will be in the ```FastDeploy/dist``` directory.

## 4. Environment Verification

After installation, verify the environment with this Python code:
```python
import paddle
from paddle.jit.marker import unified
# Verify GPU availability
paddle.utils.run_check()
# Verify FastDeploy custom operators compilation
from fastdeploy.model_executor.ops.gpu import beam_search_softmax
```

If the above code executes successfully, the environment is ready.

## 5. Demo

```python
from fastdeploy import LLM, SamplingParams

prompts = [
    "Hello. My name is",
]

sampling_params = SamplingParams(top_p=0.95, max_tokens=32, temperature=0.6)

llm = LLM(model="/root/model/ERNIE-4.5-21B-A3B-Paddle", tensor_parallel_size=1, max_model_len=256, engine_worker_queue_port=9135, quantization='wint8', static_decode_blocks=0, gpu_memory_utilization=0.9)

outputs = llm.generate(prompts, sampling_params)

print(f"Generated {len(outputs)} outputs")
print("=" * 50 + "\n")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt)
    print(generated_text)
    print("-" * 50)
```

```
Output：
INFO     2025-08-18 10:54:18,455 416822 engine.py[line:202] Waiting worker processes ready...
Loading Weights: 100%|█████████████████████████████████████████████████████████████████████████| 100/100 [03:33<00:00,  2.14s/it]
Loading Layers: 100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:18<00:00,  5.54it/s]
INFO     2025-08-18 10:58:16,149 416822 engine.py[line:247] Worker processes are launched with 240.08204197883606 seconds.
Processed prompts: 100%|███████████████████████| 1/1 [00:21<00:00, 21.84s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Generated 1 outputs
==================================================

Hello. My name is
Alice and I'm here to help you. What can I do for you today?
Hello Alice! I'm trying to organize a small party
```
