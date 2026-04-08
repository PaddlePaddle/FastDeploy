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

## Tensorwise FP8 Quantized Model

Intel® Gaudi® supports the FP8 data type. Currently, FastDeploy in `tensor_wise_fp8` quantization mode is supported on intel_HPU. The overall workflow for inference using this mode is as follows:
- Convert the existing BF16 model to an FP8 quantized model:
  - Measure the activation ranges of the BF16 model to generate a calibration file.
  - Statically quantize the relevant weights to the FP8 format. Combine these with the calibration file to produce the final FP8 quantized model.
- Run inference using the FP8 model.

The detailed workflow is as follows:

### 1. Generate Calibration File

Set the environment variables and run the BF16 model via offline_demo.py script.

``` bash
export FD_HPU_MEASUREMENT_MODE=1
```

This process will automatically trigger the model to run in measurement mode. The statistical results are merged and updated in `model_measurement.txt` file. It supports multi-card and multiple measurement runs (in multi-card mode, multiple measurement files may be generated, but a unified quantized model will ultimately be produced; the model file itself is not split according to the number of cards).

### 2. Offline Generation of FP8 Model

Run [Model_convert.py](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/intel_hpu/tools/Model_convert.py) script located in PaddleCustomDevice to statically quantize the relevant module `weights` to the FP8 data type, while writing the corresponding `weight_scale`. The calibrated `activation_scale` from the measurement is also recorded into the final FP8 model file.

``` bash
python Model_convert.py [bf16_model_path] [fp8_model_path] [model_measurement_name_or_path] <ranks_total_number>
```

- `bf16_model_path`: BF16 model input path
- `fp8_model_path`: FP8 model output path
- `model_measurement_name_or_path`: calibrantion file name or path from measurement.
  - can be a single measurement file name.
  - can be a folder containing all measurement files (e.g., from multiple runs or multiple cards).
  - can be the filename prefix for multiple measurement files; the system will automatically append the `_{i}` suffix based on `ranks_total_number`.
- `ranks_total_number`: optional, used in conjunction with `model_measurement_name_or_path` when it serves as a filename prefix.

### 3. Run FP8 Model

All `tensor_wise_fp8` related configurations will be automatically set up in Model_convert.py file via config.json script. Simply specify the `model_name_or_path` as the offline-generated FP8 model path (`fp8_model_path`). No additional parameter configuration is required.
