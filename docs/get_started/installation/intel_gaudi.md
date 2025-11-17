[简体中文](../../zh/get_started/installation/intel_gaudi.md)

# Intel Gaudi Installation for running ERNIE 4.5 Series Models

The following installation methods are available when your environment meets these requirements:

- Python 3.10
- Intel Gaudi 2
- Intel Gaudi software version 1.22.0
- Linux X86_64

## 1. Run Docker Container

Use the following commands to run a Docker container. Make sure to update the versions below as listed in the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html):

```{.console}
$ docker pull vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
```

### 2. Install PaddlePaddle

```bash
python -m pip install paddlepaddle==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### 3. Install PaddleCustomDevice
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

### 4. Install FastDeploy

```shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy
bash build.sh
```

## Prepare the inference demo

### 1. Start inference service
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

### 2. Launch the request
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "What is AI?"}
  ], "max_tokens": 24
}'
```

### 3. Successfully returns the result
```json
{"id":"chatcmpl-3bd98ae2-fafe-46ae-a552-d653a8526503","object":"chat.completion","created":1757653575,"model":"ERNIE-4.5-21B-A3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"**AI (Artificial Intelligence)** refers to the development of computer systems that can perform tasks typically requiring human intelligence.","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":11,"total_tokens":35,"completion_tokens":24,"prompt_tokens_details":{"cached_tokens":0}}}
```
