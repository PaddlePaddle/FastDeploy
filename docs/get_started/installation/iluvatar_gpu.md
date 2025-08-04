# Run ERNIE-4.5-300B-A47B & ERNIE-4.5-21B-A3B model on iluvatar machine
The current version of the software merely serves as a demonstration demo for the Iluvatar CoreX combined with the Fastdeploy inference framework for large models. There may be issues when running the latest ERNIE4.5 model, and we will conduct repairs and performance optimization in the future. Subsequent versions will provide customers with a more stable version.

## Machine Preparation
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

1. Install packages

```bash
pip3 install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip3 install paddle-iluvatar-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
pip3 install fastdeploy_iluvatar_gpu -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
```

## Prepare the inference demo script

script list below:

`run_demo.sh`:

```bash
#!/bin/bash
export PADDLE_XCCL_BACKEND=iluvatar_gpu
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
    "The largest ocean is",
]

# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# load the model
llm = LLM(model="/home/paddle/ernie-4_5-21b-a3b-bf16-paddle", tensor_parallel_size=4, max_model_len=8192, static_decode_blocks=0, quantization='wint8')

# Perform batch inference
outputs = llm.generate(prompts, sampling_params)
# Note：Replace `/home/paddle/ernie-4_5-21b-a3b-bf16-paddle` in it with the path to the ERNIE model you have downloaded.

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(prompt, generated_text)
```

## run demo

```bash
./run_demo.sh
```

The following logs will be printed: Loading the model took approximately 74 seconds, and running the demo took approximately 240 seconds.

```
/usr/local/lib/python3.10/site-packages/paddle/utils/cpp_extension/extension_utils.py:715: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
/usr/local/lib/python3.10/site-packages/_distutils_hack/__init__.py:31: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-07-02 11:07:42,393] [    INFO] - Loading configuration file /home/paddle/ernie-4_5-21b-a3b-bf16-paddle/generation_config.json
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:250: UserWarning: using greedy search strategy. However, `temperature` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/paddleformers/generation/configuration_utils.py:255: UserWarning: using greedy search strategy. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `decode_strategy="greedy_search" ` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
INFO     2025-07-02 11:07:43,589 577964 engine.py[line:207] Waitting worker processes ready...
Loading Weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:57<00:00,  1.75it/s]
Loading Layers: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:08<00:00, 11.73it/s]
INFO     2025-07-02 11:08:55,261 577964 engine.py[line:277] Worker processes are launched with 73.76574492454529 seconds.
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:59<00:00, 119.96s/it, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Hello, my name is  Christopher. Today, I'm going to teach you how to draw a cute cartoon ghost. Let's get started!
 (1) First, draw a big circle for the ghost's head.
 (2) Then, add two small circles for the eyes, making sure they're not too big.
 (3) Next, draw a wide, open mouth that looks like a big "U".
 (4) After that, create the body by drawing a slightly smaller circle below the head.
 (5) Now, let's add some arms. Draw two short, curly lines on each side of the body.
 (6) Finally, give the ghost a wavy line at the bottom to represent its floating appearance.

Now, let's break down each step:

**Step 1: Drawing the Head**
- Start with a big circle to form the head of the ghost. This will be the foundation of your drawing.

**Step 2: Adding Eyes**
- On the head, place two small circles for the eyes. They should be centered and not too big, to give the ghost a cute and innocent look.

**Step 3: Drawing the
The largest ocean is  the Pacific Ocean, covering an area of approximately â¦ [3], The first scientific expeditions to determine the ocean's depth were the Challenger expedition (1872â1876) and the U.S. Navy Hydrographic Office survey (1877â1879). The oceanic crust is thin and irregular, consisting of upward moving magma from the mantle below, and cooling and solidifying on the surface. The shallowest parts of the ocean are called the continental shelves. Large tides are caused mainly by the alignment of the Sun, Moon, and Earth during new or full moons. The origin of the word "ocean" is not clear. The first global oceanic topography survey was completed by the Challenger expedition (1872â1876). [57] The sound speed in the ocean is primarily a function of water temperature and salinity, and varies with depth. The deep-ocean floor is mostly flat and devoid of life, with the exception of seamounts and various underwater volcanic features, including seamounts and hydrothermal vents. [73] Today, the five ocean
```
